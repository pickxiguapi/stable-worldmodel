from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np

SCRIPT = (
    Path(__file__).parents[2]
    / 'scripts'
    / 'data'
    / 'convert_ogbench_npz_tdmpc2.py'
)
SPEC = spec_from_file_location('convert_ogbench_npz_tdmpc2', SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_goal_segment_conversion(tmp_path):
    rows_per_episode = 11
    episodes = 2
    rows = rows_per_episode * episodes
    terminals = np.zeros(rows, dtype=bool)
    terminals[rows_per_episode - 1 :: rows_per_episode] = True
    observations = np.arange(rows * 4 * 4 * 3, dtype=np.uint8).reshape(
        rows, 4, 4, 3
    )
    actions = np.linspace(-1, 1, rows * 2, dtype=np.float32).reshape(rows, 2)
    source = tmp_path / 'tiny.npz'
    dest = tmp_path / 'tiny.h5'
    np.savez(
        source,
        observations=observations,
        actions=actions,
        terminals=terminals,
        # Privileged columns may exist upstream but must be ignored.
        qpos=np.zeros((rows, 3), dtype=np.float32),
        qvel=np.zeros((rows, 2), dtype=np.float32),
    )

    MODULE.convert(
        source,
        dest,
        segment_transitions=5,
        min_transitions=5,
        block_rows=7,
    )

    with h5py.File(dest, 'r') as dataset:
        assert dataset['pixels'].shape == (24, 4, 4, 3)
        assert not {'state', 'qpos', 'qvel', 'button_states'} & set(
            dataset.keys()
        )
        np.testing.assert_array_equal(dataset['ep_len'][:], [6, 6, 6, 6])
        np.testing.assert_array_equal(dataset['ep_offset'][:], [0, 6, 12, 18])
        np.testing.assert_array_equal(
            np.flatnonzero(dataset['reward'][:]), [4, 10, 16, 22]
        )
        np.testing.assert_array_equal(
            np.flatnonzero(dataset['terminal'][:]), [5, 11, 17, 23]
        )
        np.testing.assert_array_equal(
            dataset['source_episode'][:],
            np.repeat(np.arange(2, dtype=np.int32), 12),
        )
        # Each pair of five-transition segments shares its boundary state.
        np.testing.assert_array_equal(
            dataset['pixels'][5], dataset['pixels'][6]
        )
        np.testing.assert_array_equal(
            dataset['pixels'][17], dataset['pixels'][18]
        )

    summary = MODULE.validate(dest, expected_segment_transitions=5)
    assert summary['episodes'] == 4
    assert summary['positive_rewards'] == 4
