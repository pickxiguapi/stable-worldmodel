"""Evaluate an official-core GC TD-MPC2 checkpoint on OGBench Cube."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
if extra_site := os.environ.get('OGBENCH_SITE_PACKAGES'):
    sys.path.append(extra_site)

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_worldmodel  # noqa: F401  # registers swm/* environments
from scripts.train.tdmpc2_official_gc import load_official_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-episode-steps', type=int, default=50)
    parser.add_argument('--reward-task-id', type=int, default=2)
    return parser.parse_args()


def gc_observation(observation: np.ndarray, goal: np.ndarray) -> torch.Tensor:
    observation = np.asarray(observation)
    goal = np.asarray(goal)
    expected = (64, 64, 3)
    if observation.shape != expected or goal.shape != expected:
        raise ValueError(
            f'Expected current and goal HWC RGB {expected}; '
            f'got {observation.shape} and {goal.shape}'
        )
    pair = np.concatenate([observation, goal], axis=-1)
    return torch.from_numpy(pair.transpose(2, 0, 1).copy()).cuda()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.max_episode_steps < 1:
        raise ValueError('episodes and max-episode-steps must be positive')

    repo_root = Path(__file__).resolve().parents[2]
    TDMPC2, cfg_to_dataclass, official_commit = load_official_agent(repo_root)
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads(config_path.read_text())
    config['compile'] = False
    cfg = cfg_to_dataclass(OmegaConf.create(config))
    if cfg.obs_shape != {'rgb': [6, 64, 64]}:
        raise ValueError(f'Not a 6-channel GC checkpoint: {cfg.obs_shape}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    agent = TDMPC2(cfg)
    payload = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    agent.load(payload)
    agent.eval()
    agent.requires_grad_(False)

    env = gym.make(
        'swm/OGBCube-v0',
        max_episode_steps=args.max_episode_steps,
        render_mode='rgb_array',
        env_type='single',
        ob_type='pixels',
        multiview=False,
        height=64,
        width=64,
        reward_task_id=args.reward_task_id,
        terminate_at_goal=True,
    )
    env.unwrapped._render_goal = False
    successes: list[bool] = []
    returns: list[float] = []
    lengths: list[int] = []
    started = time.time()
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            goal = info['target']
            episode_return = 0.0
            success = bool(info.get('success', False))
            length = 0
            for step in range(args.max_episode_steps):
                obs = gc_observation(observation, goal)
                with torch.inference_mode():
                    action = agent.act(
                        obs, t0=(step == 0), eval_mode=True
                    ).numpy()
                observation, reward, terminated, truncated, info = env.step(
                    np.clip(action, -1.0, 1.0)
                )
                episode_return += float(reward)
                success = success or bool(info.get('success', False))
                length = step + 1
                if terminated or truncated:
                    break
            successes.append(success)
            returns.append(episode_return)
            lengths.append(length)
            print(
                f'EPISODE episode={episode + 1} success={int(success)} '
                f'return={episode_return:.6f} length={length}',
                flush=True,
            )
    finally:
        env.close()

    result = {
        'label': args.label,
        'checkpoint': str(checkpoint),
        'official_commit': official_commit,
        'episodes': args.episodes,
        'seed': args.seed,
        'success_rate': float(np.mean(successes)),
        'episode_successes': successes,
        'episode_returns': returns,
        'episode_lengths': lengths,
        'elapsed_seconds': time.time() - started,
        'environment': {
            'id': 'swm/OGBCube-v0',
            'env_type': 'single',
            'reward_task_id': args.reward_task_id,
            'max_episode_steps': args.max_episode_steps,
        },
        'planner': {
            'name': 'official_tdmpc2_mppi',
            'horizon': cfg.horizon,
            'iterations': cfg.iterations,
            'num_samples': cfg.num_samples,
            'num_elites': cfg.num_elites,
            'num_pi_trajs': cfg.num_pi_trajs,
        },
    }
    result_path = output_dir / 'results.json'
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT_JSON=' + json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
