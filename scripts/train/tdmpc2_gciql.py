"""GCIQL-guided TD-MPC2 for goal-conditioned offline visual control.

GCIQL is the offline-RL core: Q is trained only on dataset actions, V uses
expectile regression, and the policy uses advantage-weighted behavior cloning.
TD-MPC2 contributes the pixel encoder, latent dynamics, reward/termination
models, and a policy-centered MPPI planner whose terminal score is V.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.train.tdmpc2_official_gc import (
    GoalConditionedH5Replay,
    REWARD_SCHEME,
    build_official_config,
    load_official_agent,
)


class BalancedGoalReplay(GoalConditionedH5Replay):
    """Replay that explicitly samples a configurable share of goal sequences."""

    def __init__(self, *args, goal_sequence_fraction: float = 0.25, **kwargs):
        if not 0.0 <= goal_sequence_fraction <= 1.0:
            raise ValueError('goal_sequence_fraction must be in [0, 1]')
        self.goal_sequence_fraction = goal_sequence_fraction
        super().__init__(*args, **kwargs)

    def sample_indices(self) -> tuple[np.ndarray, np.ndarray, int]:
        episode_ids = self.rng.integers(
            0, len(self.offsets), size=self.batch_size
        )
        relative_starts = self.rng.integers(
            0, self.max_starts[episode_ids]
        )
        forced = int(round(self.batch_size * self.goal_sequence_fraction))
        # The latest valid H-step sequence contains the transition entering the
        # fixed hindsight goal. Keep forced examples first for testability.
        relative_starts[:forced] = self.max_starts[episode_ids[:forced]] - 1
        starts = self.offsets[episode_ids] + relative_starts
        indices = starts[:, None] + np.arange(self.horizon + 1)[None]
        return episode_ids, indices, forced

    def sample(self):
        episode_ids, indices, _ = self.sample_indices()
        current = torch.from_numpy(self.pixels[indices]).permute(1, 0, 4, 2, 3)
        goals = torch.from_numpy(self.pixels[self.goal_rows[episode_ids]])
        goals = goals.permute(0, 3, 1, 2)[None].expand(
            self.horizon + 1, -1, -1, -1, -1
        )
        obs = torch.cat([current, goals], dim=2).contiguous()

        transition_indices = indices[:, :-1]
        next_indices = indices[:, 1:]
        action = torch.from_numpy(self.actions[transition_indices]).transpose(
            0, 1
        )
        reward = torch.from_numpy(self.rewards[transition_indices]).transpose(
            0, 1
        )[..., None]
        terminated = torch.from_numpy(
            self.terminals[next_indices]
        ).transpose(0, 1)[..., None]
        return (
            obs.to(self.device, non_blocking=True),
            action.to(self.device, non_blocking=True),
            reward.to(self.device, non_blocking=True),
            terminated.to(
                self.device, dtype=torch.float32, non_blocking=True
            ),
            None,
        )


def build_gciql_config(args, action_dim, cfg_to_dataclass):
    base_args = SimpleNamespace(**vars(args), actor_bc_coef=0.0)
    base = build_official_config(base_args, action_dim, cfg_to_dataclass)
    config = asdict(base)
    config.pop('actor_bc_coef', None)
    config.update(
        {
            'expectile': args.expectile,
            'awr_beta': args.awr_beta,
            'awr_clip': args.awr_clip,
            'value_tau': args.value_tau,
            'goal_sequence_fraction': args.goal_sequence_fraction,
            'mppi_policy_center': True,
        }
    )
    return cfg_to_dataclass(OmegaConf.create(config))


def gciql_agent(base_cls):
    """Create a TD-MPC2 agent with IQL critics and AWR policy learning."""
    from common import init as td_init
    from common import layers as td_layers
    from common import math as td_math

    class GCIQLTDMPC2(base_cls):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.value = td_layers.mlp(
                cfg.latent_dim + cfg.task_dim,
                2 * [cfg.mlp_dim],
                1,
            ).to(self.device)
            self.value.apply(td_init.weight_init)
            self.target_value = deepcopy(self.value).requires_grad_(False)
            self.target_value.eval()
            self.value_optim = torch.optim.Adam(
                self.value.parameters(),
                lr=cfg.lr,
                eps=1e-5,
                capturable=True,
            )

        def _value(self, z, task, target=False):
            if self.cfg.multitask:
                z = self.model.task_emb(z, task)
            network = self.target_value if target else self.value
            return network(z)

        @torch.no_grad()
        def _estimate_value(self, z, actions, task):
            value, discount = 0, 1
            termination = torch.zeros(
                self.cfg.num_samples, 1, dtype=torch.float32, device=z.device
            )
            for t in range(self.cfg.horizon):
                reward = td_math.two_hot_inv(
                    self.model.reward(z, actions[t], task), self.cfg
                )
                z = self.model.next(z, actions[t], task)
                value = value + discount * (1 - termination) * reward
                discount_update = (
                    self.discount[torch.tensor(task)]
                    if self.cfg.multitask
                    else self.discount
                )
                discount = discount * discount_update
                if self.cfg.episodic:
                    termination = torch.clip(
                        termination
                        + (self.model.termination(z, task) > 0.5).float(),
                        max=1.0,
                    )
            return value + discount * (1 - termination) * self._value(
                z, task, target=True
            )

        @torch.no_grad()
        def _plan(self, obs, t0=False, eval_mode=False, task=None):
            z = self.model.encode(obs, task)
            policy_mean = torch.empty(
                self.cfg.horizon,
                self.cfg.action_dim,
                device=self.device,
            )
            mean_z = z
            for t in range(self.cfg.horizon):
                _, mean_info = self.model.pi(mean_z, task)
                policy_mean[t] = mean_info['mean'][0]
                mean_z = self.model.next(
                    mean_z, policy_mean[t].unsqueeze(0), task
                )

            if self.cfg.num_pi_trajs > 0:
                pi_actions = torch.empty(
                    self.cfg.horizon,
                    self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                    device=self.device,
                )
                pi_z = z.repeat(self.cfg.num_pi_trajs, 1)
                for t in range(self.cfg.horizon):
                    pi_actions[t], _ = self.model.pi(pi_z, task)
                    pi_z = self.model.next(pi_z, pi_actions[t], task)
                pi_actions[:, 0] = policy_mean

            z = z.repeat(self.cfg.num_samples, 1)
            mean = policy_mean.clone()
            std = torch.full(
                (self.cfg.horizon, self.cfg.action_dim),
                self.cfg.max_std,
                dtype=torch.float,
                device=self.device,
            )
            actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_samples,
                self.cfg.action_dim,
                device=self.device,
            )
            if self.cfg.num_pi_trajs > 0:
                actions[:, : self.cfg.num_pi_trajs] = pi_actions

            for _ in range(self.cfg.iterations):
                noise = torch.randn(
                    self.cfg.horizon,
                    self.cfg.num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                    device=std.device,
                )
                sampled = mean.unsqueeze(1) + std.unsqueeze(1) * noise
                actions[:, self.cfg.num_pi_trajs :] = sampled.clamp(-1, 1)
                if self.cfg.multitask:
                    actions = actions * self.model._action_masks[task]

                trajectory_value = self._estimate_value(
                    z, actions, task
                ).nan_to_num(0)
                elite_idxs = torch.topk(
                    trajectory_value.squeeze(1), self.cfg.num_elites, dim=0
                ).indices
                elite_value = trajectory_value[elite_idxs]
                elite_actions = actions[:, elite_idxs]
                max_value = elite_value.max(0).values
                score = torch.exp(
                    self.cfg.temperature * (elite_value - max_value)
                )
                score = score / score.sum(0)
                mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                    score.sum(0) + 1e-9
                )
                std = (
                    (
                        score.unsqueeze(0)
                        * (elite_actions - mean.unsqueeze(1)) ** 2
                    ).sum(dim=1)
                    / (score.sum(0) + 1e-9)
                ).sqrt()
                std = std.clamp(self.cfg.min_std, self.cfg.max_std)

            rand_idx = td_math.gumbel_softmax_sample(score.squeeze(1))
            selected = torch.index_select(
                elite_actions, 1, rand_idx
            ).squeeze(1)
            action, action_std = selected[0], std[0]
            if not eval_mode:
                action = action + action_std * torch.randn(
                    self.cfg.action_dim, device=action_std.device
                )
            self._prev_mean.copy_(mean)
            return action.clamp(-1, 1)

        def _policy_mean(self, z, task):
            if self.cfg.multitask:
                z = self.model.task_emb(z, task)
            raw_mean, _ = self.model._pi(z).chunk(2, dim=-1)
            return torch.tanh(raw_mean)

        def update_value_and_pi(self, zs, behavior_action, task):
            rho = torch.pow(
                self.cfg.rho, torch.arange(len(zs), device=self.device)
            )[:, None, None]
            with torch.no_grad():
                q_data = self.model.Q(
                    zs,
                    behavior_action,
                    task,
                    return_type='min',
                    target=True,
                )

            value = self._value(zs, task)
            advantage = q_data - value
            expectile_weight = torch.where(
                advantage > 0,
                self.cfg.expectile,
                1 - self.cfg.expectile,
            )
            value_loss = (
                rho * expectile_weight * advantage.square()
            ).mean()
            value_loss.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value.parameters(), self.cfg.grad_clip_norm
            )
            self.value_optim.step()
            self.value_optim.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Weight the actor with the same pre-update IQL advantage.
                # Recomputing V after its optimizer step can transiently turn a
                # small advantage into a large one during early joint training.
                detached_advantage = advantage.detach()
                awr_weight = torch.exp(
                    (self.cfg.awr_beta * detached_advantage).clamp(
                        max=float(np.log(self.cfg.awr_clip))
                    )
                )
            policy_mean = self._policy_mean(zs, task)
            # The play dataset contains genuinely clipped actions at +/-1.
            # A tanh-Gaussian likelihood maps those targets close to infinite
            # pre-tanh values and explodes when TD-MPC2 initializes std near
            # zero. Advantage-weighted action-space regression is the stable
            # AWR/BC mean objective and matches deterministic evaluation.
            bc_error = (policy_mean - behavior_action).square().sum(
                dim=-1, keepdim=True
            )
            pi_loss = (rho * awr_weight * bc_error).mean()
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model._pi.parameters(), self.cfg.grad_clip_norm
            )
            self.pi_optim.step()
            self.pi_optim.zero_grad(set_to_none=True)

            return {
                'value_loss': value_loss,
                'value_grad_norm': value_grad_norm,
                'value_mean': value.mean(),
                'value_std': value.std(),
                'advantage_mean': detached_advantage.mean(),
                'advantage_std': detached_advantage.std(),
                'awr_weight_mean': awr_weight.mean(),
                'awr_weight_max': awr_weight.max(),
                'pi_loss': pi_loss,
                'pi_bc_error': bc_error.mean(),
                'pi_grad_norm': pi_grad_norm,
                'pi_action_norm': policy_mean.norm(dim=-1).mean(),
                'behavior_action_norm': behavior_action.norm(dim=-1).mean(),
            }

        def _update(self, obs, action, reward, terminated, task=None):
            with torch.no_grad():
                next_z = self.model.encode(obs[1:], task)
                discount = (
                    self.discount[task].unsqueeze(-1)
                    if self.cfg.multitask
                    else self.discount
                )
                q_target = reward + discount * (1 - terminated) * self._value(
                    next_z, task, target=True
                )

            self.model.train()
            self.value.train()
            zs = torch.empty(
                self.cfg.horizon + 1,
                self.cfg.batch_size,
                self.cfg.latent_dim,
                device=self.device,
            )
            z = self.model.encode(obs[0], task)
            zs[0] = z
            consistency_loss = 0
            for t, (_action, _next_z) in enumerate(
                zip(action.unbind(0), next_z.unbind(0))
            ):
                z = self.model.next(z, _action, task)
                consistency_loss = (
                    consistency_loss
                    + F.mse_loss(z, _next_z) * self.cfg.rho**t
                )
                zs[t + 1] = z

            rollout_zs = zs[:-1]
            qs = self.model.Q(rollout_zs, action, task, return_type='all')
            reward_preds = self.model.reward(rollout_zs, action, task)
            if self.cfg.episodic:
                termination_pred = self.model.termination(
                    zs[1:], task, unnormalized=True
                )

            reward_loss, critic_loss = 0, 0
            for t, (reward_pred, reward_t, q_t, timestep_qs) in enumerate(
                zip(
                    reward_preds.unbind(0),
                    reward.unbind(0),
                    q_target.unbind(0),
                    qs.unbind(1),
                )
            ):
                reward_loss = reward_loss + td_math.soft_ce(
                    reward_pred, reward_t, self.cfg
                ).mean() * self.cfg.rho**t
                for q_pred in timestep_qs.unbind(0):
                    critic_loss = critic_loss + td_math.soft_ce(
                        q_pred, q_t, self.cfg
                    ).mean() * self.cfg.rho**t

            consistency_loss = consistency_loss / self.cfg.horizon
            reward_loss = reward_loss / self.cfg.horizon
            critic_loss = critic_loss / (
                self.cfg.horizon * self.cfg.num_q
            )
            if self.cfg.episodic:
                termination_loss = F.binary_cross_entropy_with_logits(
                    termination_pred, terminated
                )
            else:
                termination_loss = 0.0
            total_loss = (
                self.cfg.consistency_coef * consistency_loss
                + self.cfg.reward_coef * reward_loss
                + self.cfg.termination_coef * termination_loss
                + self.cfg.value_coef * critic_loss
            )
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm
            )
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

            iql_info = self.update_value_and_pi(
                rollout_zs.detach(), action.detach(), task
            )
            self.model.soft_update_target_Q()
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_value.parameters(), self.value.parameters()
                ):
                    target_param.lerp_(param, self.cfg.value_tau)
            self.model.eval()
            self.value.eval()
            info = {
                'consistency_loss': consistency_loss,
                'reward_loss': reward_loss,
                'critic_loss': critic_loss,
                'termination_loss': termination_loss,
                'total_loss': total_loss,
                'grad_norm': grad_norm,
                'reward_zero_rate': (reward == 0).float().mean(),
            }
            if self.cfg.episodic:
                info.update(
                    td_math.termination_statistics(
                        torch.sigmoid(termination_pred[-1]), terminated[-1]
                    )
                )
            info.update(iql_info)
            return {
                key: val.detach().mean()
                if isinstance(val, torch.Tensor)
                else torch.tensor(val)
                for key, val in info.items()
            }

        def load(self, fp):
            payload = (
                fp
                if isinstance(fp, dict)
                else torch.load(
                    fp, map_location=self.device, weights_only=False
                )
            )
            super().load(payload)
            if 'value' not in payload or 'target_value' not in payload:
                raise ValueError('G3 checkpoint is missing value networks')
            self.value.load_state_dict(payload['value'])
            self.target_value.load_state_dict(payload['target_value'])
            return self

    return GCIQLTDMPC2


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    output_dir: str
    task: str
    seed: int
    steps: int
    batch_size: int
    horizon: int
    segment_transitions: int
    model_size: int
    expectile: float
    awr_beta: float
    awr_clip: float
    value_tau: float
    goal_sequence_fraction: float
    episodic: bool
    compile: bool
    log_interval: int
    checkpoint_interval: int
    official_commit: str


def save_checkpoint(agent, output_dir: Path, step: int, run_config) -> Path:
    path = output_dir / f'weights_step_{step}.pt'
    temp = output_dir / f'.{path.name}.tmp-{os.getpid()}'
    torch.save(
        {
            'model': agent.model.state_dict(),
            'value': agent.value.state_dict(),
            'target_value': agent.target_value.state_dict(),
            'optim': agent.optim.state_dict(),
            'pi_optim': agent.pi_optim.state_dict(),
            'value_optim': agent.value_optim.state_dict(),
            'step': step,
            'run_config': asdict(run_config),
        },
        temp,
    )
    os.replace(temp, path)
    print(f'CHECKPOINT step={step} path={path}', flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--segment-transitions', type=int, default=50)
    parser.add_argument('--model-size', type=int, default=5)
    parser.add_argument('--expectile', type=float, default=0.9)
    parser.add_argument('--awr-beta', type=float, default=3.0)
    parser.add_argument('--awr-clip', type=float, default=100.0)
    parser.add_argument('--value-tau', type=float, default=0.005)
    parser.add_argument('--goal-sequence-fraction', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=10_000)
    parser.add_argument(
        '--episodic', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--compile', action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('G3 TD-MPC2 requires CUDA')
    if args.horizon < 1 or args.steps < 1 or args.batch_size < 1:
        raise ValueError('horizon, steps, and batch-size must be positive')
    if not 0.5 < args.expectile < 1.0:
        raise ValueError('expectile must be in (0.5, 1)')
    if args.awr_beta < 0 or args.awr_clip < 1 or args.value_tau <= 0:
        raise ValueError('Invalid AWR/value hyperparameters')

    repo_root = Path(__file__).resolve().parents[2]
    TDMPC2, cfg_to_dataclass, official_commit = load_official_agent(repo_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    replay = BalancedGoalReplay(
        args.dataset.expanduser().resolve(),
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
        expected_segment_transitions=args.segment_transitions,
        goal_sequence_fraction=args.goal_sequence_fraction,
    )
    cfg = build_gciql_config(args, replay.action_dim, cfg_to_dataclass)
    run_config = RunConfig(
        dataset=str(args.dataset.expanduser().resolve()),
        output_dir=str(output_dir),
        task=args.task,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        horizon=args.horizon,
        segment_transitions=args.segment_transitions,
        model_size=args.model_size,
        expectile=args.expectile,
        awr_beta=args.awr_beta,
        awr_clip=args.awr_clip,
        value_tau=args.value_tau,
        goal_sequence_fraction=args.goal_sequence_fraction,
        episodic=args.episodic,
        compile=args.compile,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        official_commit=official_commit,
    )
    (output_dir / 'config.json').write_text(
        json.dumps(asdict(run_config), indent=2) + '\n'
    )
    (output_dir / 'official_config.json').write_text(
        json.dumps(asdict(cfg), indent=2) + '\n'
    )

    Agent = gciql_agent(TDMPC2)
    agent = Agent(cfg)
    print(
        f'G3_TDMPC2_GCIQL commit={official_commit} '
        f'reward_scheme={REWARD_SCHEME} expectile={args.expectile} '
        f'awr_beta={args.awr_beta} '
        f'goal_sequence_fraction={args.goal_sequence_fraction} '
        'q_bootstrap=target_value policy=awr_bc '
        'mppi=policy_centered_terminal_iql_value',
        flush=True,
    )
    started = time.time()
    for step in range(1, args.steps + 1):
        metrics = agent.update(replay)
        if step == 1 or step % args.log_interval == 0:
            values = {
                key: float(value.float().mean().cpu())
                for key, value in metrics.items()
            }
            values.update(
                {
                    'step': step,
                    'elapsed_seconds': time.time() - started,
                    'gpu_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            print('TRAIN_METRICS=' + json.dumps(values, sort_keys=True), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps:
            save_checkpoint(agent, output_dir, step, run_config)

    print(
        'TRAINING_COMPLETE=' + json.dumps(
            {
                'task': args.task,
                'steps': args.steps,
                'elapsed_seconds': time.time() - started,
                'output_dir': str(output_dir),
                'official_commit': official_commit,
                'variant': 'g3_tdmpc2_gciql',
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
