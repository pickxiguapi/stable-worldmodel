from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.train.tdmpc2_official_gc import (
    OFFICIAL_COMMIT,
    GoalConditionedH5Replay,
    build_official_config,
    load_official_agent,
)


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


def test_goal_conditioned_replay_alignment(tmp_path):
    path = tmp_path / 'replay.h5'
    rows = 10
    pixels = np.zeros((rows, 64, 64, 3), dtype=np.uint8)
    for row in range(rows):
        pixels[row] = row
    action_values = np.arange(rows, dtype=np.float32) / 10.0
    actions = np.repeat(action_values[:, None], 2, axis=1)
    rewards = np.arange(rows, dtype=np.float32)
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
        reward[:, :, 0].numpy(), current_rows[:-1].astype(np.float32)
    )
    np.testing.assert_array_equal(
        terminated[:, :, 0].numpy(),
        np.isin(current_rows[1:], [4, 9]).astype(np.float32),
    )
