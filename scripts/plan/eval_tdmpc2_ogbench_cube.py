"""Evaluate a pixel TD-MPC2 checkpoint on the OGBench cube task."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms import v2

import stable_worldmodel as swm
from stable_worldmodel.planning.solver import MPPISolver
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.wm.tdmpc2.module import two_hot_inv
from stable_worldmodel.wm.utils import load_pretrained


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TargetToGoalWrapper(gym.Wrapper):
    """Expose the CubeEnv target image under the policy's canonical key."""

    @staticmethod
    def _with_goal(info: dict) -> dict:
        if 'target' not in info:
            raise KeyError(
                f"Cube environment did not return 'target'; keys={sorted(info)}"
            )
        info = dict(info)
        info['goal'] = info['target']
        return info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return obs, self._with_goal(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs,
            reward,
            terminated,
            truncated,
            self._with_goal(info),
        )


class CachedTDMPC2PlanningAdapter(torch.nn.Module):
    """Adapt World/solver shapes and avoid re-encoding pixels per MPPI step."""

    def __init__(self, model: torch.nn.Module, num_pi_trajs: int):
        super().__init__()
        self.model = model
        self.num_pi_trajs = num_pi_trajs
        self._base_z: torch.Tensor | None = None

    @staticmethod
    def _latest(value: torch.Tensor) -> torch.Tensor:
        # World supplies (B, T, C, H, W); an expanded solver input would be
        # (B, N, T, C, H, W). TD-MPC2 is Markovian, so keep the latest frame.
        if value.ndim == 5:
            return value[:, -1]
        if value.ndim == 6:
            return value[:, :, -1]
        return value

    def _encode(self, info: dict) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoding_keys = list(self.model.cfg.wm.get('encoding', {}).keys())
        obs = {
            key: self._latest(info[key]).to(device) for key in encoding_keys
        }
        if self.model.use_pixels and 'goal' in info:
            obs['goal'] = self._latest(info['goal']).to(device)
        return self.model.encode(obs)

    @torch.inference_mode()
    def get_action(
        self,
        info: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self._encode(info)
        self._base_z = z
        if prefix_actions is not None:
            prefix_actions = prefix_actions.to(z.device).clamp(-1.0, 1.0)
            for t in range(prefix_actions.shape[1]):
                z = self.model.dynamics(
                    torch.cat([z, prefix_actions[:, t]], dim=-1)
                )
        return self.model.rollout(z, horizon, self.num_pi_trajs)

    @torch.inference_mode()
    def get_cost(
        self, info: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        del info
        if self._base_z is None:
            raise RuntimeError('MPPI requested costs before actor initialization.')

        candidates = action_candidates.clamp(-1.0, 1.0)
        batch_size, num_samples, horizon, action_dim = candidates.shape
        if self._base_z.shape[0] != batch_size:
            raise ValueError(
                'Cached latent batch does not match the solver batch: '
                f'{self._base_z.shape[0]} != {batch_size}'
            )

        z = (
            self._base_z[:, None]
            .expand(batch_size, num_samples, -1)
            .reshape(batch_size * num_samples, -1)
        )
        actions = candidates.reshape(
            batch_size * num_samples, horizon, action_dim
        )

        returns = torch.zeros(
            batch_size * num_samples, 1, device=z.device, dtype=z.dtype
        )
        discount = 1.0
        for t in range(horizon):
            z_action = torch.cat([z, actions[:, t]], dim=-1)
            returns += discount * two_hot_inv(
                self.model.reward(z_action), self.model.cfg
            )
            z = self.model.dynamics(z_action)
            discount *= float(self.model.cfg.wm.get('discount', 0.99))

        terminal_action = torch.tanh(self.model.pi(z).chunk(2, dim=-1)[0])
        terminal_za = torch.cat([z, terminal_action], dim=-1)
        q_values = torch.stack(
            [
                two_hot_inv(q(terminal_za), self.model.cfg)
                for q in self.model.qs
            ]
        )
        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)
        penalty_scale = float(
            self.model.cfg.wm.get('uncertainty_penalty', 0.5)
        )
        conservative_q = q_mean - penalty_scale * q_mean.abs() * q_std
        returns += discount * conservative_q
        return -returns.view(batch_size, num_samples)


class ClippedWorldModelPolicy(WorldModelPolicy):
    """Keep the executed action inside TD-MPC2's normalized action range."""

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        action = super().get_action(info_dict, **kwargs)
        return np.clip(action, -1.0, 1.0)


class ClippedMPPISolver(MPPISolver):
    """Clamp the optimized plan before execution and warm-start reuse."""

    def solve(self, info_dict: dict, init_action=None) -> dict:
        outputs = super().solve(info_dict, init_action=init_action)
        outputs['actions'] = outputs['actions'].clamp(-1.0, 1.0)
        return outputs


def image_transform(image_size: int):
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.Resize((image_size, image_size), antialias=True),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-episode-steps', type=int, default=50)
    parser.add_argument('--num-envs', type=int, default=10)
    parser.add_argument('--reward-task-id', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--num-samples', type=int, default=1024)
    parser.add_argument('--num-elites', type=int, default=64)
    parser.add_argument('--num-pi-trajs', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--no-video', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError('episodes and num-envs must be positive')
    if args.episodes % args.num_envs:
        raise ValueError('episodes must be divisible by num-envs')

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    videos = None if args.no_video else output_dir / 'videos'

    model = load_pretrained(str(checkpoint), cache_dir='/').cuda().eval()
    model.requires_grad_(False)
    image_size = int(model.cfg.get('image_size', 64))

    adapter = CachedTDMPC2PlanningAdapter(model, args.num_pi_trajs).cuda().eval()
    solver = ClippedMPPISolver(
        cost=adapter,
        batch_size=args.num_envs,
        num_samples=args.num_samples,
        var_scale=args.std,
        n_steps=args.iterations,
        topk=args.num_elites,
        temperature=args.temperature,
        device='cuda',
        seed=args.seed,
    )
    transform = image_transform(image_size)
    policy = ClippedWorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=args.horizon,
            receding_horizon=1,
            history_len=1,
            action_block=1,
            warm_start=True,
        ),
        transform={'pixels': transform, 'goal': transform},
    )

    world = swm.World(
        'swm/OGBCube-v0',
        num_envs=args.num_envs,
        image_shape=(image_size, image_size),
        max_episode_steps=args.max_episode_steps,
        pre_wrappers=[TargetToGoalWrapper],
        env_type='single',
        ob_type='pixels',
        multiview=False,
        height=image_size,
        width=image_size,
        reward_task_id=args.reward_task_id,
        terminate_at_goal=True,
        render_goal=True,
    )
    world.set_policy(policy)

    started = time.time()
    try:
        results = world.evaluate(
            episodes=args.episodes,
            seed=args.seed,
            video=videos,
            reset_mode='auto',
        )
    finally:
        world.close()

    serializable = {
        'label': args.label,
        'checkpoint': str(checkpoint),
        'episodes': args.episodes,
        'seed': args.seed,
        'success_rate': float(results['success_rate']),
        'episode_successes': [
            bool(x) for x in results['episode_successes'].tolist()
        ],
        'episode_seeds': [int(x) for x in results['seeds'].tolist()],
        'elapsed_seconds': time.time() - started,
        'environment': {
            'id': 'swm/OGBCube-v0',
            'env_type': 'single',
            'reward_task_id': args.reward_task_id,
            'max_episode_steps': args.max_episode_steps,
            'image_size': image_size,
        },
        'planner': {
            'name': 'MPPI',
            'horizon': args.horizon,
            'iterations': args.iterations,
            'num_samples': args.num_samples,
            'num_elites': args.num_elites,
            'num_pi_trajs': args.num_pi_trajs,
            'temperature': args.temperature,
            'std': args.std,
        },
    }
    result_path = output_dir / 'results.json'
    result_path.write_text(json.dumps(serializable, indent=2) + '\n')
    print('RESULT_JSON=' + json.dumps(serializable, sort_keys=True))
    print(f'WROTE_RESULTS={result_path}')


if __name__ == '__main__':
    main()
