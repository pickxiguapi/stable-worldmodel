"""Convert an OGBench visual NPZ into a goal-conditioned TD-MPC2 HDF5.

The visual play/noisy archives contain pixels, actions, episode terminals,
and privileged simulator state.  This converter deliberately stores only
RGB pixels, actions, and derived rewards; qpos/qvel/button states are neither
read nor written, so the resulting training input is vision-only.

Long OGBench episodes are split into fixed-length future-goal segments. The
last image in each segment is the goal used by ``scripts/train/tdmpc2.py``;
the transition entering it receives reward 0 and all earlier transitions
receive reward -1. This is a shortest-path reward for goal reaching.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

REQUIRED_KEYS = ('observations', 'actions', 'terminals')
REWARD_SCHEME = 'negative_step_goal_zero_v2'


def episode_bounds(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open episode bounds, accepting an unterminated final row."""
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    ends = np.flatnonzero(terminals) + 1
    if len(ends) == 0 or ends[-1] != len(terminals):
        ends = np.concatenate([ends, np.array([len(terminals)])])
    starts = np.concatenate([np.array([0]), ends[:-1]])
    lengths = ends - starts
    if np.any(lengths < 2):
        bad = int(np.flatnonzero(lengths < 2)[0])
        raise ValueError(f'Episode {bad} has only {int(lengths[bad])} row(s)')
    return starts.astype(np.int64), ends.astype(np.int64)


def segment_index(
    terminals: np.ndarray,
    segment_transitions: int,
    min_transitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build source-row and episode metadata for goal segments."""
    if segment_transitions < 1:
        raise ValueError('segment_transitions must be positive')
    if min_transitions < 1 or min_transitions > segment_transitions:
        raise ValueError('min_transitions must be in [1, segment_transitions]')

    original_starts, original_ends = episode_bounds(terminals)
    segments: list[tuple[int, int, int]] = []
    for episode, (start, end) in enumerate(
        zip(original_starts.tolist(), original_ends.tolist())
    ):
        # ``end - 1`` is the final observation; it has no useful outgoing
        # action.  A segment [seg_start, seg_goal] contains both endpoints.
        for seg_start in range(start, end - 1, segment_transitions):
            seg_goal = min(seg_start + segment_transitions, end - 1)
            if seg_goal - seg_start >= min_transitions:
                segments.append((episode, seg_start, seg_goal))

    if not segments:
        raise ValueError('No valid goal segments were produced')

    lengths = np.asarray(
        [goal - start + 1 for _, start, goal in segments], dtype=np.int32
    )
    offsets = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(lengths[:-1], dtype=np.int64),
        ]
    )
    total_rows = int(lengths.sum())
    source_rows = np.empty(total_rows, dtype=np.int64)
    source_episodes = np.empty(total_rows, dtype=np.int32)
    source_steps = np.empty(total_rows, dtype=np.int32)

    for out_start, length, (episode, start, goal) in zip(
        offsets.tolist(), lengths.tolist(), segments
    ):
        out_end = out_start + length
        source_rows[out_start:out_end] = np.arange(start, goal + 1)
        source_episodes[out_start:out_end] = episode
        source_steps[out_start:out_end] = np.arange(length)

    return source_rows, source_episodes, source_steps, lengths


def _dataset_kwargs(rows: int, width: int | None = None) -> dict:
    chunk_rows = min(8192, max(1, rows))
    chunks = (chunk_rows,) if width is None else (chunk_rows, width)
    return {'chunks': chunks, 'compression': 'lzf', 'shuffle': True}


def convert(
    source: Path,
    dest: Path,
    segment_transitions: int = 50,
    min_transitions: int = 5,
    block_rows: int = 2_048,
    overwrite: bool = False,
) -> None:
    """Convert one NPZ archive atomically."""
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f'{dest} already exists; pass --overwrite to replace it'
        )
    if block_rows < 1:
        raise ValueError('block_rows must be positive')

    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f'{source} is missing keys: {missing}')
        actions = np.asarray(archive['actions'], dtype=np.float32)
        terminals = np.asarray(archive['terminals'], dtype=bool)
        pixels = np.asarray(archive['observations'])

    num_rows = len(terminals)
    arrays = {'observations': pixels, 'actions': actions}
    wrong = {
        key: len(value)
        for key, value in arrays.items()
        if len(value) != num_rows
    }
    if wrong:
        raise ValueError(
            f'All arrays must have {num_rows} rows; mismatches: {wrong}'
        )
    if actions.ndim != 2:
        raise ValueError(f'actions must be 2D, got {actions.shape}')
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f'Expected HWC RGB observations, got {pixels.shape}')
    if pixels.dtype != np.uint8:
        raise ValueError(f'Expected uint8 observations, got {pixels.dtype}')
    if not np.isfinite(actions).all():
        raise ValueError('actions contain NaN or Inf')
    action_min, action_max = float(actions.min()), float(actions.max())
    if action_min < -1.0001 or action_max > 1.0001:
        raise ValueError(
            'TD-MPC2 requires actions in [-1, 1], got '
            f'[{action_min:.6f}, {action_max:.6f}]'
        )

    source_rows, source_episodes, source_steps, lengths = segment_index(
        terminals, segment_transitions, min_transitions
    )
    offsets = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(lengths[:-1], dtype=np.int64),
        ]
    )
    output_rows = len(source_rows)
    pixel_shape = tuple(pixels.shape[1:])

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(f'.{dest.name}.tmp-{os.getpid()}')
    if temp.exists():
        temp.unlink()

    try:
        with h5py.File(temp, 'w', libver='latest') as output:
            pixel_ds = output.create_dataset(
                'pixels',
                shape=(output_rows, *pixel_shape),
                dtype=np.uint8,
                chunks=(min(32, output_rows), *pixel_shape),
                compression='gzip',
                compression_opts=1,
                shuffle=True,
            )
            action_ds = output.create_dataset(
                'action',
                shape=(output_rows, actions.shape[1]),
                dtype=np.float32,
                **_dataset_kwargs(output_rows, actions.shape[1]),
            )
            reward_ds = output.create_dataset(
                'reward',
                shape=(output_rows,),
                dtype=np.float32,
                **_dataset_kwargs(output_rows),
            )
            terminal_ds = output.create_dataset(
                'terminal',
                shape=(output_rows,),
                dtype=np.bool_,
                **_dataset_kwargs(output_rows),
            )
            output.create_dataset('source_episode', data=source_episodes)
            output.create_dataset('source_step', data=source_steps)
            output.create_dataset('ep_len', data=lengths)
            output.create_dataset('ep_offset', data=offsets)

            reward_ds[:] = -1.0
            terminal_ds[:] = False
            goal_rows = offsets + lengths - 1
            reward_ds[goal_rows - 1] = 0.0
            reward_ds[goal_rows] = 0.0  # Dummy final-observation slots.
            terminal_ds[goal_rows] = True

            for out_start in range(0, output_rows, block_rows):
                out_end = min(out_start + block_rows, output_rows)
                idx = source_rows[out_start:out_end]
                pixel_ds[out_start:out_end] = pixels[idx]
                action_ds[out_start:out_end] = actions[idx]

            # The final observation in a segment has no outgoing transition.
            # Zeroing it avoids preserving a misleading action from the next
            # source transition; the loader never trains on this final action.
            action_ds[goal_rows] = np.zeros(
                (len(goal_rows), actions.shape[1]), dtype=np.float32
            )

            output.attrs['format'] = 'ogbench_goal_tdmpc2_pixels_v1'
            output.attrs['source'] = str(source)
            output.attrs['source_size_bytes'] = source.stat().st_size
            output.attrs['segment_transitions'] = segment_transitions
            output.attrs['min_transitions'] = min_transitions
            output.attrs['original_rows'] = num_rows
            output.attrs['original_episodes'] = len(
                episode_bounds(terminals)[0]
            )
            output.attrs['observation'] = 'pixels_only_rgb'
            output.attrs['reward_scheme'] = REWARD_SCHEME
            output.flush()
        os.replace(temp, dest)
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise

    validate(dest, expected_segment_transitions=segment_transitions)


def validate(
    path: Path,
    expected_segment_transitions: int | None = None,
    verify_source: bool = False,
) -> dict[str, int | float | str]:
    """Validate the training contract and print a compact summary."""
    path = path.expanduser().resolve()
    with h5py.File(path, 'r') as dataset:
        required = {
            'pixels',
            'action',
            'reward',
            'terminal',
            'ep_len',
            'ep_offset',
            'source_episode',
            'source_step',
        }
        missing = sorted(required - set(dataset.keys()))
        if missing:
            raise ValueError(f'{path} is missing datasets: {missing}')
        privileged = {'state', 'qpos', 'qvel', 'button_states'} & set(
            dataset.keys()
        )
        if privileged:
            raise ValueError(
                f'Vision-only dataset contains privileged keys: {privileged}'
            )
        lengths = dataset['ep_len'][:]
        offsets = dataset['ep_offset'][:]
        rows = int(dataset['pixels'].shape[0])
        row_datasets = (
            'action',
            'reward',
            'terminal',
            'source_episode',
            'source_step',
        )
        mismatched = {
            key: int(dataset[key].shape[0])
            for key in row_datasets
            if dataset[key].shape[0] != rows
        }
        if mismatched:
            raise ValueError(
                f'Row counts differ from pixels={rows}: {mismatched}'
            )
        if dataset['pixels'].ndim != 4 or dataset['pixels'].shape[-1] != 3:
            raise ValueError('pixels must be HWC RGB images')
        if dataset['pixels'].dtype != np.uint8:
            raise ValueError('pixels must use uint8 storage')
        if dataset['action'].ndim != 2:
            raise ValueError('action must have shape [rows, action_dim]')
        if dataset['action'].dtype != np.float32:
            raise ValueError('action must use float32 storage')
        if dataset['reward'].dtype != np.float32:
            raise ValueError('reward must use float32 storage')
        if dataset['terminal'].dtype != np.bool_:
            raise ValueError('terminal must use bool storage')
        if not np.issubdtype(dataset['source_episode'].dtype, np.integer):
            raise ValueError('source_episode must use an integer dtype')
        if not np.issubdtype(dataset['source_step'].dtype, np.integer):
            raise ValueError('source_step must use an integer dtype')
        for key in ('reward', 'terminal', 'source_episode', 'source_step'):
            if dataset[key].ndim != 1:
                raise ValueError(f'{key} must be one-dimensional')
        if dataset.attrs.get('format') != 'ogbench_goal_tdmpc2_pixels_v1':
            raise ValueError('Unexpected or missing TD-MPC2 dataset format')
        if dataset.attrs.get('observation') != 'pixels_only_rgb':
            raise ValueError('Dataset is not marked pixels-only RGB')
        if not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError('ep_len must use an integer dtype')
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError('ep_offset must use an integer dtype')
        if len(lengths) == 0 or len(offsets) != len(lengths):
            raise ValueError('ep_len/ep_offset must describe at least one segment')
        if int(lengths.sum()) != rows:
            raise ValueError('ep_len does not sum to the stored row count')
        expected_offsets = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.cumsum(lengths[:-1], dtype=np.int64),
            ]
        )
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError('ep_offset is inconsistent with ep_len')
        if int(lengths.min()) < 6:
            raise ValueError('A segment is too short for TD-MPC2 horizon=5')
        if expected_segment_transitions is not None:
            stored = int(dataset.attrs['segment_transitions'])
            if stored != expected_segment_transitions:
                raise ValueError(
                    f'Expected segment_transitions={expected_segment_transitions}, '
                    f'found {stored}'
                )

        goal_rows = offsets + lengths - 1
        rewards = dataset['reward'][:]
        if not np.isfinite(rewards).all():
            raise ValueError('Rewards contain NaN or Inf')
        if dataset.attrs.get('reward_scheme') != REWARD_SCHEME:
            raise ValueError(
                f'Expected reward_scheme={REWARD_SCHEME}, found '
                f'{dataset.attrs.get("reward_scheme")}'
            )
        expected_goal_reward_rows = goal_rows - 1
        expected_zero_rows = np.sort(
            np.concatenate((expected_goal_reward_rows, goal_rows))
        )
        zero_rows = np.flatnonzero(rewards == 0.0)
        if not np.array_equal(zero_rows, expected_zero_rows):
            raise ValueError(
                'Zero rewards are not exactly on goal transitions/dummy rows'
            )
        step_mask = np.ones(rows, dtype=bool)
        step_mask[expected_zero_rows] = False
        if not np.all(rewards[step_mask] == -1.0):
            raise ValueError('All non-goal transition rewards must be -1')
        terminals = dataset['terminal'][:].astype(bool, copy=False)
        terminal_rows = np.flatnonzero(terminals)
        if not np.array_equal(terminal_rows, goal_rows):
            raise ValueError('Terminals are not exactly on final goal observations')
        actions = dataset['action'][:]
        if not np.isfinite(actions).all():
            raise ValueError('Stored actions contain NaN or Inf')
        action_min, action_max = float(actions.min()), float(actions.max())
        if action_min < -1.0001 or action_max > 1.0001:
            raise ValueError('Stored actions are outside [-1, 1]')
        if not np.all(actions[goal_rows] == 0.0):
            raise ValueError('Final observation rows must have zero dummy actions')

        source_episode = dataset['source_episode'][:]
        source_step = dataset['source_step'][:]
        expected_source_step = np.concatenate(
            [np.arange(length, dtype=source_step.dtype) for length in lengths]
        )
        if not np.array_equal(source_step, expected_source_step):
            raise ValueError('source_step does not reset/increment per segment')
        segment_source_episode = source_episode[offsets]
        expected_source_episode = np.repeat(segment_source_episode, lengths)
        if not np.array_equal(source_episode, expected_source_episode):
            raise ValueError('source_episode changes inside a segment')
        if np.any(segment_source_episode < 0) or np.any(
            np.diff(segment_source_episode) < 0
        ):
            raise ValueError('source_episode IDs must be nonnegative and ordered')

        # Probe all structural boundary types across the full file. Reading these
        # rows also makes HDF5 decompress representative pixel chunks, catching
        # truncated/corrupt image payloads without loading tens of GiB at once.
        probe_episode_ids = np.unique(
            np.linspace(
                0,
                len(lengths) - 1,
                num=min(512, len(lengths)),
                dtype=np.int64,
            )
        )
        probe_rows = np.unique(
            np.concatenate(
                (
                    offsets[probe_episode_ids],
                    expected_goal_reward_rows[probe_episode_ids],
                    goal_rows[probe_episode_ids],
                )
            )
        )
        pixel_probe = dataset['pixels'][probe_rows]
        pixel_probe_min = int(pixel_probe.min())
        pixel_probe_max = int(pixel_probe.max())
        pixel_probe_std = float(pixel_probe.std())
        if pixel_probe_min == pixel_probe_max or pixel_probe_std == 0.0:
            raise ValueError('Representative pixel rows are constant/collapsed')

        source_verified = 0
        if verify_source:
            source = Path(str(dataset.attrs.get('source', '')))
            if not source.is_file():
                raise ValueError(f'Original source archive is missing: {source}')
            recorded_size = int(dataset.attrs.get('source_size_bytes', -1))
            if source.stat().st_size != recorded_size:
                raise ValueError(
                    f'Original source size changed: recorded={recorded_size}, '
                    f'current={source.stat().st_size}'
                )
            min_transitions = int(dataset.attrs['min_transitions'])
            with np.load(source, allow_pickle=False) as archive:
                source_terminals = np.asarray(archive['terminals'], dtype=bool)
                source_actions = np.asarray(
                    archive['actions'], dtype=np.float32
                )
            (
                source_rows,
                expected_source_episode,
                expected_source_step,
                expected_lengths,
            ) = segment_index(
                source_terminals,
                int(dataset.attrs['segment_transitions']),
                min_transitions,
            )
            if int(dataset.attrs['original_rows']) != len(source_terminals):
                raise ValueError('original_rows does not match source archive')
            original_episodes = len(episode_bounds(source_terminals)[0])
            if int(dataset.attrs['original_episodes']) != original_episodes:
                raise ValueError('original_episodes does not match source archive')
            if not np.array_equal(lengths, expected_lengths):
                raise ValueError('ep_len does not match source segmentation')
            if not np.array_equal(source_episode, expected_source_episode):
                raise ValueError('source_episode does not match source archive')
            if not np.array_equal(source_step, expected_source_step):
                raise ValueError('source_step does not match source archive')
            expected_actions = source_actions[source_rows]
            expected_actions[goal_rows] = 0.0
            if not np.array_equal(actions, expected_actions):
                raise ValueError('Stored actions do not match source archive')
            source_verified = 1

        summary: dict[str, int | float | str] = {
            'path': str(path),
            'rows': rows,
            'episodes': len(lengths),
            'pixel_shape': 'x'.join(map(str, dataset['pixels'].shape[1:])),
            'action_dim': int(dataset['action'].shape[1]),
            'min_episode_rows': int(lengths.min()),
            'max_episode_rows': int(lengths.max()),
            'negative_step_rewards': int(step_mask.sum()),
            'zero_goal_transition_rewards': len(expected_goal_reward_rows),
            'zero_dummy_rewards': len(goal_rows),
            'terminals': len(terminal_rows),
            'source_episodes': int(segment_source_episode.max()) + 1,
            'action_min': action_min,
            'action_max': action_max,
            'pixel_probe_min': pixel_probe_min,
            'pixel_probe_max': pixel_probe_max,
            'pixel_probe_std': round(pixel_probe_std, 4),
            'source_verified': source_verified,
        }
    print(
        'VALID ' + ' '.join(f'{key}={value}' for key, value in summary.items())
    )
    return summary


def migrate_reward_scheme(path: Path) -> None:
    """Rewrite only derived rewards in an existing v1 HDF5 file."""
    path = path.expanduser().resolve()
    with h5py.File(path, 'r+') as dataset:
        lengths = dataset['ep_len'][:].astype(np.int64, copy=False)
        offsets = dataset['ep_offset'][:].astype(np.int64, copy=False)
        rows = int(dataset['reward'].shape[0])
        goal_rows = offsets + lengths - 1
        goal_transition_rows = goal_rows - 1
        rewards = np.full(rows, -1.0, dtype=np.float32)
        rewards[goal_transition_rows] = 0.0
        rewards[goal_rows] = 0.0
        dataset['reward'][:] = rewards
        dataset.attrs['reward_scheme'] = REWARD_SCHEME
        dataset.flush()
    validate(path)
    print(f'MIGRATED_REWARD path={path} scheme={REWARD_SCHEME}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    convert_parser = subparsers.add_parser('convert')
    convert_parser.add_argument('--source', type=Path, required=True)
    convert_parser.add_argument('--dest', type=Path, required=True)
    convert_parser.add_argument('--segment-transitions', type=int, default=50)
    convert_parser.add_argument('--min-transitions', type=int, default=5)
    convert_parser.add_argument('--block-rows', type=int, default=2_048)
    convert_parser.add_argument('--overwrite', action='store_true')

    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('paths', nargs='+', type=Path)
    validate_parser.add_argument('--segment-transitions', type=int)
    validate_parser.add_argument('--verify-source', action='store_true')

    migrate_parser = subparsers.add_parser('migrate-reward')
    migrate_parser.add_argument('paths', nargs='+', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'convert':
        convert(
            source=args.source,
            dest=args.dest,
            segment_transitions=args.segment_transitions,
            min_transitions=args.min_transitions,
            block_rows=args.block_rows,
            overwrite=args.overwrite,
        )
    elif args.command == 'validate':
        for path in args.paths:
            validate(path, args.segment_transitions, args.verify_source)
    else:
        for path in args.paths:
            migrate_reward_scheme(path)


if __name__ == '__main__':
    main()
