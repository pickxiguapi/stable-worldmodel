from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.data.convert_ogbench_npz_tdmpc2 import (
    convert,
    migrate_reward_scheme,
    validate,
)
from scripts.train.tdmpc2_official_gc import (
    OFFICIAL_COMMIT,
    REWARD_SCHEME,
    GoalConditionedH5Replay,
    build_official_config,
    load_official_agent,
)


def test_converter_negative_step_goal_zero_and_migration(tmp_path):
    source = tmp_path / 'source.npz'
    dest = tmp_path / 'replay.h5'
    rows = 7
    observations = np.empty((rows, 64, 64, 3), dtype=np.uint8)
    for row in range(rows):
        observations[row] = row * 20
    actions = np.linspace(-0.5, 0.5, rows * 2, dtype=np.float32).reshape(
        rows, 2
    )
    terminals = np.zeros(rows, dtype=bool)
    terminals[-1] = True
    np.savez(
        source,
        observations=observations,
        actions=actions,
        terminals=terminals,
    )

    convert(source, dest, segment_transitions=6, min_transitions=5)
    with h5py.File(dest, 'r') as dataset:
        np.testing.assert_array_equal(
            dataset['reward'][:], [-1, -1, -1, -1, -1, 0, 0]
        )
        assert dataset.attrs['reward_scheme'] == REWARD_SCHEME

    # Exercise the in-place migration used for the eight audited server files.
    with h5py.File(dest, 'r+') as dataset:
        dataset['reward'][:] = 0
        dataset['reward'][5] = 1
        del dataset.attrs['reward_scheme']
    migrate_reward_scheme(dest)
    summary = validate(dest, expected_segment_transitions=6)
    assert summary['negative_step_rewards'] == 5
    assert summary['zero_goal_transition_rewards'] == 1


def test_official_source_and_config_contract():
    repo_root = Path(__file__).resolve().parents[1]
    _, cfg_to_dataclass, commit = load_official_agent(repo_root)
    args = SimpleNamespace(
        task='visual-cube-single-play-v0',
        episodic=True,
        segment_transitions=50,
        steps=100_000,
        batch_size=256,
        horizon=3,
        model_size=5,
        actor_bc_coef=1.0,
        compile=True,
        seed=1,
    )
    cfg = build_official_config(
        args, action_dim=5, cfg_to_dataclass=cfg_to_dataclass
    )

    assert commit == OFFICIAL_COMMIT
    assert cfg.obs_shape == {'rgb': [6, 64, 64]}
    assert cfg.simnorm_dim == 8
    assert cfg.num_q == 5
    assert cfg.batch_size == 256
    assert cfg.actor_bc_coef == 1.0
    assert cfg.vmin == -10
    assert cfg.vmax == 10
    assert cfg.bin_size == 0.2


def test_goal_conditioned_replay_alignment(tmp_path):
    path = tmp_path / 'replay.h5'
    rows = 10
    pixels = np.zeros((rows, 64, 64, 3), dtype=np.uint8)
    for row in range(rows):
        pixels[row] = row
    action_values = np.arange(rows, dtype=np.float32) / 10.0
    actions = np.repeat(action_values[:, None], 2, axis=1)
    rewards = np.array(
        [-1, -1, -1, 0, 0, -1, -1, -1, 0, 0], dtype=np.float32
    )
    terminals = np.zeros(rows, dtype=bool)
    terminals[[4, 9]] = True

    with h5py.File(path, 'w') as dataset:
        dataset.create_dataset('pixels', data=pixels)
        dataset.create_dataset('action', data=actions)
        dataset.create_dataset('reward', data=rewards)
        dataset.create_dataset('terminal', data=terminals)
        dataset.create_dataset('ep_offset', data=np.array([0, 5]))
        dataset.create_dataset('ep_len', data=np.array([5, 5]))
        dataset.attrs['segment_transitions'] = 4
        dataset.attrs['reward_scheme'] = REWARD_SCHEME

    replay = GoalConditionedH5Replay(
        path,
        horizon=2,
        batch_size=64,
        seed=7,
        expected_segment_transitions=4,
        device='cpu',
    )
    obs, action, reward, terminated, task = replay.sample()

    assert obs.shape == (3, 64, 6, 64, 64)
    assert action.shape == (2, 64, 2)
    assert reward.shape == (2, 64, 1)
    assert terminated.shape == (2, 64, 1)
    assert task is None

    current_rows = obs[:, :, 0, 0, 0].numpy()
    goal_rows = obs[:, :, 3, 0, 0].numpy()
    np.testing.assert_array_equal(current_rows[1], current_rows[0] + 1)
    np.testing.assert_array_equal(current_rows[2], current_rows[1] + 1)
    assert set(np.unique(goal_rows)).issubset({4, 9})
    np.testing.assert_array_equal(goal_rows[0], goal_rows[1])
    np.testing.assert_array_equal(goal_rows[1], goal_rows[2])
    np.testing.assert_array_equal(
        action[:, :, 0].numpy(), current_rows[:-1].astype(np.float32) / 10.0
    )
    np.testing.assert_array_equal(
        reward[:, :, 0].numpy(), rewards[current_rows[:-1].astype(np.int64)]
    )
    np.testing.assert_array_equal(
        terminated[:, :, 0].numpy(),
        np.isin(current_rows[1:], [4, 9]).astype(np.float32),
    )
