from pathlib import Path

import h5py
import numpy as np

from scripts.audit_ldp_ogbench8 import TaskSpec, validate_eval_result
from scripts.data.ldp_ogbench_data import (
    OGBenchLDPData,
    REWARD_SCHEME,
    h5_take,
)
from scripts.train.ldp_ogbench import (
    NonstandardEvalHorizonError,
    parser,
    task_goal_residual,
    validate_cli_args,
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
    rewards = np.full(rows, -1.0, dtype=np.float32)
    goal_rows = offsets + lengths - 1
    rewards[goal_rows - 1] = 0.0
    rewards[goal_rows] = 0.0
    terminals = np.zeros(rows, dtype=bool)
    terminals[goal_rows] = True
    with h5py.File(source_path, 'w') as source:
        source.create_dataset('pixels', data=pixels)
        source.create_dataset('action', data=actions)
        source.create_dataset('reward', data=rewards)
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


def test_train_split_is_aligned_to_original_source_episode(tmp_path):
    source = tmp_path / 'source.h5'
    latent = tmp_path / 'latent.h5'
    make_data(source, latent)
    with h5py.File(source, 'r+') as handle:
        handle.create_dataset(
            'source_episode',
            data=np.repeat(np.array([10, 11, 12], dtype=np.int32), 6),
        )
    with h5py.File(latent, 'r+') as handle:
        handle.attrs['source_size_bytes'] = source.stat().st_size
    data = OGBenchLDPData(source, latent, train_fraction=0.8)
    assert data.train_episodes == 2
    assert data.training_episode_count(2, train_fraction=0.95) == 1


def test_terminal_rows_must_match_episode_metadata(tmp_path):
    source = tmp_path / 'source.h5'
    latent = tmp_path / 'latent.h5'
    make_data(source, latent)
    with h5py.File(source, 'r+') as handle:
        handle['terminal'][5] = False
        handle['terminal'][4] = True
    with np.testing.assert_raises_regex(ValueError, 'terminal rows disagree'):
        OGBenchLDPData(source, latent)


def test_task_goal_residual_uses_generic_official_reward():
    class FakeEnv:
        unwrapped = None

        def __init__(self, successes):
            self.successes = successes
            self.unwrapped = self

        def _compute_successes(self):
            return self.successes

    assert task_goal_residual(FakeEnv([False, True, False])) == 2.0
    assert task_goal_residual(FakeEnv(([True], [False, True], True, False))) == 2.0
    assert task_goal_residual(FakeEnv([True, True])) == 0.0


def test_formal_eval_rejects_nonstandard_horizon():
    base = [
        'eval',
        '--run-dir',
        'ldp',
        '--vae-dir',
        'vae',
        '--output-dir',
        'eval',
        '--dataset-id',
        'visual-cube-single-play-v0',
        '--max-episode-steps',
        '50',
    ]
    args = parser().parse_args(base)
    with np.testing.assert_raises_regex(
        NonstandardEvalHorizonError, 'forbidden for formal evaluation'
    ):
        validate_cli_args(args)

    smoke_args = parser().parse_args(base + ['--allow-nonstandard-horizon'])
    validate_cli_args(smoke_args)


def test_completion_audit_checks_raw_episode_results():
    spec = TaskSpec(
        'cube_double_play',
        'visual-cube-double-play-v0',
        'visual-cube-double-v0',
        0,
    )
    result = {
        'dataset_id': 'visual-cube-double-play-v0',
        'episodes_per_task': 2,
        'task_ids': [1, 2, 3, 4, 5],
        'total_episodes': 10,
        'seed': 42,
        'success_rate': 0.5,
        'tasks': [
            {
                'task_id': task_id,
                'task_name': f'task{task_id}',
                'episodes': 2,
                'success_rate': 0.5,
                'episode_successes': [True, False],
                'episode_returns': [1.0, 0.0],
                'episode_lengths': [1, 500],
                'initial_goal_residuals': [2.0, 2.0],
                'final_goal_residuals': [0.0, 1.0],
                'minimum_goal_residuals': [0.0, 1.0],
                'mean_action_norms': [0.5, 0.75],
            }
            for task_id in range(1, 6)
        ],
        'environment': {
            'id': 'visual-cube-double-v0',
            'creation_api': 'ogbench.make_env_and_datasets(env_only=True)',
            'max_episode_steps': 500,
            'uses_registered_horizon': True,
        },
        'ogbench': {
            'commit': '1d4140997f60c52c6fb0702ec100dc988b18c548',
            'origin': 'https://github.com/seohongpark/ogbench.git',
        },
        'planner': {
            'pred_horizon': 8,
            'action_horizon': 4,
            'diffusion_steps': 100,
            'goal_conditioning': 'planner_global_condition_current_plus_final_goal',
        },
    }
    assert validate_eval_result(result, spec, episodes=2, seed=42) == []
    result['tasks'][2]['success_rate'] = 1.0
    errors = validate_eval_result(result, spec, episodes=2, seed=42)
    assert 'task 3 success_rate is inconsistent' in errors
