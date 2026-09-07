"""Goal-conditioned offline RL built on the pinned official TD-MPC2 core.

The official source is pinned as ``third_party/tdmpc2``. This driver adapts
the data boundary and adds a behavior-cloning loss to the actor. The official
MPPI proposal, policy candidates, trajectory scores, and update remain intact.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


OFFICIAL_COMMIT = 'e9f59321933cbc8e11a002b842adc7d4ffae8ff1'
REWARD_SCHEME = 'negative_step_goal_zero_v2'
MODEL_SIZES = {
    1: {
        'enc_dim': 256,
        'mlp_dim': 384,
        'latent_dim': 128,
        'num_enc_layers': 2,
        'num_q': 2,
    },
    5: {
        'enc_dim': 256,
        'mlp_dim': 512,
        'latent_dim': 512,
        'num_enc_layers': 2,
        'num_q': 5,
    },
}


def load_official_agent(repo_root: Path):
    official_root = repo_root / 'third_party' / 'tdmpc2'
    source_root = official_root / 'tdmpc2'
    if not source_root.is_dir():
        raise FileNotFoundError(
            'Official TD-MPC2 submodule is missing. Run '
            '`git submodule update --init third_party/tdmpc2`.'
        )
    import subprocess

    commit = subprocess.check_output(
        ['git', '-C', str(official_root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f'Official TD-MPC2 must be {OFFICIAL_COMMIT}, found {commit}'
        )
    sys.path.insert(0, str(source_root))
    from common.parser import cfg_to_dataclass
    from tdmpc2 import TDMPC2

    return TDMPC2, cfg_to_dataclass, commit


def offline_constrained_agent(base_cls):
    """Add TD-M(PC)^2-style actor BC without modifying MPPI."""
    from common import math as td_math

    class OfflineConstrainedTDMPC2(base_cls):
        def update_pi(self, zs, behavior_action, task):
            policy_action, info = self.model.pi(zs, task)
            qs = self.model.Q(
                zs, policy_action, task, return_type='avg', detach=True
            )
            self.scale.update(qs[0])
            qs = self.scale(qs)

            rho = torch.pow(
                self.cfg.rho, torch.arange(len(qs), device=self.device)
            )
            q_loss = (
                -(
                    self.cfg.entropy_coef * info['scaled_entropy'] + qs
                ).mean(dim=(1, 2))
                * rho
            ).mean()
            bc_loss = (
                (policy_action - behavior_action)
                .square()
                .sum(dim=-1)
                .mean(dim=1)
                * rho
            ).mean()
            pi_loss = q_loss + self.cfg.actor_bc_coef * bc_loss
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model._pi.parameters(), self.cfg.grad_clip_norm
            )
            self.pi_optim.step()
            self.pi_optim.zero_grad(set_to_none=True)

            return {
                'pi_loss': pi_loss,
                'pi_q_loss': q_loss,
                'pi_bc_loss': bc_loss,
                'pi_grad_norm': pi_grad_norm,
                'pi_entropy': info['entropy'],
                'pi_scaled_entropy': info['scaled_entropy'],
                'pi_scale': self.scale.value,
                'pi_action_norm': policy_action.norm(dim=-1).mean(),
                'behavior_action_norm': behavior_action.norm(dim=-1).mean(),
            }

        def _update(self, obs, action, reward, terminated, task=None):
            with torch.no_grad():
                next_z = self.model.encode(obs[1:], task)
                td_targets = self._td_target(next_z, reward, terminated, task)

            self.model.train()
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

            reward_loss, value_loss = 0, 0
            for t, (
                reward_pred,
                reward_target,
                td_target,
                timestep_qs,
            ) in enumerate(
                zip(
                    reward_preds.unbind(0),
                    reward.unbind(0),
                    td_targets.unbind(0),
                    qs.unbind(1),
                )
            ):
                reward_loss = reward_loss + td_math.soft_ce(
                    reward_pred, reward_target, self.cfg
                ).mean() * self.cfg.rho**t
                for q_pred in timestep_qs.unbind(0):
                    value_loss = value_loss + td_math.soft_ce(
                        q_pred, td_target, self.cfg
                    ).mean() * self.cfg.rho**t

            consistency_loss = consistency_loss / self.cfg.horizon
            reward_loss = reward_loss / self.cfg.horizon
            if self.cfg.episodic:
                termination_loss = F.binary_cross_entropy_with_logits(
                    termination_pred, terminated
                )
            else:
                termination_loss = 0.0
            value_loss = value_loss / (
                self.cfg.horizon * self.cfg.num_q
            )
            total_loss = (
                self.cfg.consistency_coef * consistency_loss
                + self.cfg.reward_coef * reward_loss
                + self.cfg.termination_coef * termination_loss
                + self.cfg.value_coef * value_loss
            )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip_norm
            )
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)

            pi_info = self.update_pi(
                rollout_zs.detach(), action.detach(), task
            )
            self.model.soft_update_target_Q()
            self.model.eval()
            info = {
                'consistency_loss': consistency_loss,
                'reward_loss': reward_loss,
                'value_loss': value_loss,
                'termination_loss': termination_loss,
                'total_loss': total_loss,
                'grad_norm': grad_norm,
            }
            if self.cfg.episodic:
                info.update(
                    td_math.termination_statistics(
                        torch.sigmoid(termination_pred[-1]), terminated[-1]
                    )
                )
            info.update(pi_info)
            return {
                key: value.detach().mean()
                if isinstance(value, torch.Tensor)
                else torch.tensor(value)
                for key, value in info.items()
            }

    return OfflineConstrainedTDMPC2


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
    actor_bc_coef: float
    episodic: bool
    compile: bool
    log_interval: int
    checkpoint_interval: int
    official_commit: str


class GoalConditionedH5Replay:
    """In-memory, episode-safe sampler for converted OGBench HDF5 data."""

    def __init__(
        self,
        path: Path,
        horizon: int,
        batch_size: int,
        seed: int,
        expected_segment_transitions: int,
        device: str | torch.device = 'cuda:0',
    ) -> None:
        self.path = path
        self.horizon = horizon
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

        started = time.time()
        print(f'Loading replay dataset into RAM: {path}', flush=True)
        with h5py.File(path, 'r') as dataset:
            required = {
                'pixels',
                'action',
                'reward',
                'terminal',
                'ep_offset',
                'ep_len',
            }
            missing = sorted(required - set(dataset.keys()))
            if missing:
                raise ValueError(f'Dataset missing keys: {missing}')
            stored_horizon = int(dataset.attrs['segment_transitions'])
            if stored_horizon != expected_segment_transitions:
                raise ValueError(
                    'Dataset goal horizon mismatch: '
                    f'expected {expected_segment_transitions}, '
                    f'found {stored_horizon}'
                )
            reward_scheme = dataset.attrs.get('reward_scheme')
            if reward_scheme != REWARD_SCHEME:
                raise ValueError(
                    f'Expected reward_scheme={REWARD_SCHEME}, '
                    f'found {reward_scheme}'
                )
            if dataset['pixels'].dtype != np.uint8:
                raise ValueError('pixels must be uint8')
            if tuple(dataset['pixels'].shape[1:]) != (64, 64, 3):
                raise ValueError(
                    'Official pixel encoder requires HWC 64x64 RGB, got '
                    f'{dataset["pixels"].shape[1:]}'
                )

            self.pixels = dataset['pixels'][:]
            self.actions = dataset['action'][:].astype(np.float32, copy=False)
            self.rewards = dataset['reward'][:].astype(np.float32, copy=False)
            self.terminals = dataset['terminal'][:].astype(bool, copy=False)
            offsets = dataset['ep_offset'][:].astype(np.int64, copy=False)
            lengths = dataset['ep_len'][:].astype(np.int64, copy=False)

        valid = lengths >= horizon + 1
        self.offsets = offsets[valid]
        self.lengths = lengths[valid]
        if not len(self.offsets):
            raise ValueError(f'No episodes support horizon={horizon}')
        if not np.isfinite(self.actions).all():
            raise ValueError('actions contain NaN or Inf')
        action_min = float(self.actions.min())
        action_max = float(self.actions.max())
        if action_min < -1.0001 or action_max > 1.0001:
            raise ValueError(
                f'actions outside [-1, 1]: [{action_min}, {action_max}]'
            )

        self.goal_rows = self.offsets + self.lengths - 1
        self.max_starts = self.lengths - horizon
        self.action_dim = int(self.actions.shape[1])
        ram_gib = (
            self.pixels.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.terminals.nbytes
        ) / 2**30
        print(
            'REPLAY_READY '
            f'rows={len(self.pixels)} episodes={len(self.offsets)} '
            f'action_dim={self.action_dim} ram_gib={ram_gib:.2f} '
            f'load_seconds={time.time() - started:.1f}',
            flush=True,
        )

    def sample(self):
        episode_ids = self.rng.integers(
            0, len(self.offsets), size=self.batch_size
        )
        relative_starts = self.rng.integers(
            0, self.max_starts[episode_ids]
        )
        starts = self.offsets[episode_ids] + relative_starts
        indices = starts[:, None] + np.arange(self.horizon + 1)[None]

        # (B,T,H,W,C) -> (T,B,C,H,W), then append one hindsight goal image
        # to every current observation. ShiftAug consequently applies the
        # same spatial augmentation to the current/goal channel pair.
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
        # The converter stores terminal on the final observation. Associate
        # that flag with the transition entering the observation, matching
        # official TD-MPC2's (obs[t+1], reward[t], terminated[t]) target.
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


def build_official_config(args, action_dim, cfg_to_dataclass):
    if args.model_size not in MODEL_SIZES:
        raise ValueError(f'Supported model sizes: {sorted(MODEL_SIZES)}')
    size = MODEL_SIZES[args.model_size]
    config = {
        # environment/data contract
        'task': args.task,
        'obs': 'rgb',
        'episodic': args.episodic,
        'obs_shape': {'rgb': [6, 64, 64]},
        'action_dim': action_dim,
        'episode_length': args.segment_transitions,
        'multitask': False,
        'task_dim': 0,
        'tasks': [args.task],
        'episode_lengths': [args.segment_transitions],
        'obs_shapes': [[6, 64, 64]],
        'action_dims': [action_dim],
        # official training defaults
        'steps': args.steps,
        'batch_size': args.batch_size,
        'reward_coef': 0.1,
        'value_coef': 0.1,
        'termination_coef': 1.0,
        'consistency_coef': 20.0,
        'rho': 0.5,
        'lr': 3e-4,
        'enc_lr_scale': 0.3,
        'grad_clip_norm': 20.0,
        'tau': 0.01,
        'discount_denom': 5,
        'discount_min': 0.95,
        'discount_max': 0.995,
        # official planning defaults (also serialized for evaluation)
        'mpc': True,
        'iterations': 6,
        'num_samples': 512,
        'num_elites': 64,
        'num_pi_trajs': 24,
        'horizon': args.horizon,
        'min_std': 0.05,
        'max_std': 2.0,
        'temperature': 0.5,
        'actor_bc_coef': args.actor_bc_coef,
        # official actor/critic/architecture defaults
        'log_std_min': -10,
        'log_std_max': 2,
        'entropy_coef': 1e-4,
        'num_bins': 101,
        # Bounds are in symlog space, so the official [-10, 10] already covers
        # real returns of roughly [-22025, 22025]. Keep the symmetric support
        # to make an untrained categorical critic decode near zero.
        'vmin': -10,
        'vmax': 10,
        'bin_size': 0.2,
        'model_size': args.model_size,
        'num_channels': 32,
        'simnorm_dim': 8,
        'dropout': 0.01,
        'compile': args.compile,
        'seed': args.seed,
        **size,
    }
    return cfg_to_dataclass(OmegaConf.create(config))


def diagnostics(agent, replay) -> dict[str, float]:
    obs, action, _, _, _ = replay.sample()
    with torch.no_grad():
        z = agent.model.encode(obs[0], None)
        q = agent.model.Q(z, action[0], None, return_type='avg')
    return {
        'latent_feature_std': float(z.float().std(dim=0).mean().cpu()),
        'q_batch_std': float(q.float().std().cpu()),
        'q_mean': float(q.float().mean().cpu()),
        'running_scale': float(agent.scale.value.float().mean().cpu()),
    }


def save_checkpoint(agent, output_dir: Path, step: int, run_config) -> Path:
    path = output_dir / f'weights_step_{step}.pt'
    temp = output_dir / f'.{path.name}.tmp-{os.getpid()}'
    torch.save(
        {
            'model': agent.model.state_dict(),
            'scale': agent.scale.state_dict(),
            'optim': agent.optim.state_dict(),
            'pi_optim': agent.pi_optim.state_dict(),
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
    parser.add_argument('--actor-bc-coef', type=float, default=1.0)
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
        raise RuntimeError('Official TD-MPC2 requires CUDA')
    if args.horizon < 1 or args.steps < 1 or args.batch_size < 1:
        raise ValueError('horizon, steps, and batch-size must be positive')
    if args.actor_bc_coef < 0:
        raise ValueError('actor-bc-coef must be nonnegative')

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

    replay = GoalConditionedH5Replay(
        args.dataset.expanduser().resolve(),
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
        expected_segment_transitions=args.segment_transitions,
    )
    cfg = build_official_config(args, replay.action_dim, cfg_to_dataclass)
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
        actor_bc_coef=args.actor_bc_coef,
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

    Agent = offline_constrained_agent(TDMPC2)
    agent = Agent(cfg)
    print(
        f'OFFICIAL_TDMPC2 commit={official_commit} '
        f'model_size={args.model_size}M goal_conditioning=rgb_concat '
        f'offline_rl=true reward_scheme={REWARD_SCHEME} '
        f'actor_bc_coef={args.actor_bc_coef} '
        f'mppi=official_unmodified '
        f'terminal_masking={args.episodic}',
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
            values.update(diagnostics(agent, replay))
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
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
