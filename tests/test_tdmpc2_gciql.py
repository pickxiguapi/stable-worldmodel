from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.train.tdmpc2_gciql import (  # noqa: E402
    BalancedGoalReplay,
    build_gciql_config,
    gciql_agent,
)
from scripts.train.tdmpc2_official_gc import (  # noqa: E402
    OFFICIAL_COMMIT,
    REWARD_SCHEME,
    load_official_agent,
)


def make_replay(path: Path) -> None:
    episode_len = 6
    episodes = 3
    rows = episode_len * episodes
    pixels = np.zeros((rows, 64, 64, 3), dtype=np.uint8)
    for row in range(rows):
        pixels[row] = row
    actions = np.linspace(-0.8, 0.8, rows * 2, dtype=np.float32).reshape(
        rows, 2
    )
    rewards = np.tile([-1, -1, -1, -1, 0, 0], episodes).astype(np.float32)
    terminals = np.zeros(rows, dtype=bool)
    terminals[np.arange(episode_len - 1, rows, episode_len)] = True
    with h5py.File(path, 'w') as dataset:
        dataset.create_dataset('pixels', data=pixels)
        dataset.create_dataset('action', data=actions)
        dataset.create_dataset('reward', data=rewards)
        dataset.create_dataset('terminal', data=terminals)
        dataset.create_dataset(
            'ep_offset', data=np.arange(0, rows, episode_len)
        )
        dataset.create_dataset(
            'ep_len', data=np.full(episodes, episode_len)
        )
        dataset.attrs['segment_transitions'] = episode_len - 1
        dataset.attrs['reward_scheme'] = REWARD_SCHEME


def test_balanced_replay_forces_goal_transition_sequences(tmp_path):
    path = tmp_path / 'replay.h5'
    make_replay(path)
    replay = BalancedGoalReplay(
        path,
        horizon=3,
        batch_size=40,
        seed=7,
        expected_segment_transitions=5,
        goal_sequence_fraction=0.25,
        device='cpu',
    )
    episode_ids, indices, forced = replay.sample_indices()

    assert forced == 10
    np.testing.assert_array_equal(
        indices[:forced, -1], replay.goal_rows[episode_ids[:forced]]
    )

    _, _, reward, terminated, _ = replay.sample()
    assert reward.shape == (3, 40, 1)
    assert int((reward == 0).sum()) >= forced
    assert int(terminated.sum()) >= forced


def test_g3_config_and_override_contract():
    repo_root = Path(__file__).resolve().parents[1]
    TDMPC2, cfg_to_dataclass, commit = load_official_agent(repo_root)
    args = SimpleNamespace(
        task='visual-cube-single-play-v0',
        episodic=True,
        segment_transitions=50,
        steps=100_000,
        batch_size=256,
        horizon=3,
        model_size=5,
        expectile=0.9,
        awr_beta=3.0,
        awr_clip=100.0,
        value_tau=0.005,
        goal_sequence_fraction=0.25,
        compile=True,
        seed=1,
    )
    cfg = build_gciql_config(args, action_dim=5, cfg_to_dataclass=cfg_to_dataclass)
    Agent = gciql_agent(TDMPC2)

    assert commit == OFFICIAL_COMMIT
    assert cfg.expectile == 0.9
    assert cfg.awr_beta == 3.0
    assert cfg.goal_sequence_fraction == 0.25
    assert cfg.mppi_policy_center is True
    assert cfg.vmin == -10 and cfg.vmax == 10
    assert Agent._update is not TDMPC2._update
    assert Agent._plan is not TDMPC2._plan
    assert Agent._estimate_value is not TDMPC2._estimate_value
    assert '_behavior_log_prob' not in Agent.__dict__
