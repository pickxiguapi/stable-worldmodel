from pathlib import Path

import h5py
import numpy as np

from scripts.data.ldp_ogbench_data import (
    OGBenchLDPData,
    REWARD_SCHEME,
    h5_take,
)


def make_data(source_path: Path, latent_path: Path) -> None:
    lengths = np.array([6, 6, 6], dtype=np.int64)
    offsets = np.array([0, 6, 12], dtype=np.int64)
    rows = int(lengths.sum())
    pixels = np.zeros((rows, 64, 64, 3), dtype=np.uint8)
    pixels[:, 0, 0, 0] = np.arange(rows)
    actions = np.repeat(np.arange(rows, dtype=np.float32)[:, None], 2, axis=1)
    actions = actions / actions.max()
    actions[offsets + lengths - 1] = 0.0
    terminals = np.zeros(rows, dtype=bool)
    terminals[offsets + lengths - 1] = True
    with h5py.File(source_path, 'w') as source:
        source.create_dataset('pixels', data=pixels)
        source.create_dataset('action', data=actions)
        source.create_dataset('terminal', data=terminals)
        source.create_dataset('ep_offset', data=offsets)
        source.create_dataset('ep_len', data=lengths)
        source.attrs['segment_transitions'] = 5
        source.attrs['reward_scheme'] = REWARD_SCHEME
    latents = np.repeat(np.arange(rows, dtype=np.float32)[:, None], 3, axis=1)
    with h5py.File(latent_path, 'w') as latent:
        latent.create_dataset('latent', data=latents.astype(np.float16))
        latent.attrs['source_size_bytes'] = source_path.stat().st_size
        latent.attrs['latent_min'] = 0.0
        latent.attrs['latent_max'] = float(rows - 1)


def test_h5_take_preserves_unsorted_duplicates(tmp_path):
    path = tmp_path / 'rows.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('x', data=np.arange(10)[:, None])
    with h5py.File(path, 'r') as handle:
        result = h5_take(handle['x'], np.array([[7, 2, 7], [0, 9, 2]]))
    np.testing.assert_array_equal(result[..., 0], [[7, 2, 7], [0, 9, 2]])


def test_goal_windows_are_aligned_and_episode_safe(tmp_path):
    source = tmp_path / 'source.h5'
    latent = tmp_path / 'latent.h5'
    make_data(source, latent)
    data = OGBenchLDPData(source, latent, train_fraction=2 / 3)
    batch = data.sample_windows(
        np.random.default_rng(7), batch_size=128, pred_horizon=8
    )

    episode_offsets = data.offsets[batch['episode_ids']]
    episode_ends = episode_offsets + data.lengths[batch['episode_ids']] - 1
    np.testing.assert_array_equal(batch['goal_rows'], episode_ends)
    assert np.all(batch['current_rows'] >= episode_offsets)
    assert np.all(batch['future_rows'] <= episode_ends[:, None])
    assert np.all(batch['action_rows'] <= episode_ends[:, None])
    np.testing.assert_array_equal(batch['current'][..., 0], batch['current_rows'][:, None])
    np.testing.assert_array_equal(batch['future'][..., 0], batch['future_rows'])
    np.testing.assert_array_equal(batch['goal'][..., 0], batch['goal_rows'])
    assert batch['future'].shape == (128, 8, 3)
    assert batch['actions'].shape == (128, 8, 2)
    padded = batch['action_rows'] == episode_ends[:, None]
    assert padded.any()
    assert np.all(batch['actions'][padded] == 0.0)

    normalized = data.normalize_latent(np.array([0, 17], dtype=np.float32))
    np.testing.assert_allclose(normalized, [-1, 1])
    np.testing.assert_allclose(data.denormalize_latent(normalized), [0, 17])


def test_partial_latents_must_end_on_episode_boundary(tmp_path):
    source = tmp_path / 'source.h5'
    full_latent = tmp_path / 'full_latent.h5'
    partial_latent = tmp_path / 'partial_latent.h5'
    make_data(source, full_latent)
    with h5py.File(full_latent, 'r') as full, h5py.File(
        partial_latent, 'w'
    ) as partial:
        partial.create_dataset('latent', data=full['latent'][:12])
        for key, value in full.attrs.items():
            partial.attrs[key] = value
    data = OGBenchLDPData(source, partial_latent)
    assert data.summary.rows == 12
    assert data.summary.episodes == 2
    batch = data.sample_windows(
        np.random.default_rng(1), batch_size=64, pred_horizon=8
    )
    assert batch['future_rows'].max() < 12
    assert batch['goal_rows'].max() < 12

    with h5py.File(partial_latent, 'r+') as partial:
        values = partial['latent'][:11]
        del partial['latent']
        partial.create_dataset('latent', data=values)
    with np.testing.assert_raises_regex(ValueError, 'episode boundary'):
        OGBenchLDPData(source, partial_latent)
