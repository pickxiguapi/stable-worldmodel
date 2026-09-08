"""Episode-safe OGBench data access for goal-conditioned LDP.

The source is the audited TD-MPC2 conversion because it already contains
row-aligned RGB observations, normalized actions, and explicit episode bounds.
This module does not depend on Torch or JAX so its indexing contract can be
unit-tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


REWARD_SCHEME = 'negative_step_goal_zero_v2'


def h5_take(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary, possibly repeated HDF5 rows without ordering errors."""
    indices = np.asarray(indices, dtype=np.int64)
    shape = indices.shape
    flat = indices.reshape(-1)
    unique, inverse = np.unique(flat, return_inverse=True)
    values = dataset[unique]
    return values[inverse].reshape(*shape, *dataset.shape[1:])


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    episodes: int
    action_dim: int
    min_episode_rows: int
    max_episode_rows: int
    segment_transitions: int


class OGBenchLDPData:
    """Sampler that never crosses a converted OGBench segment boundary."""

    def __init__(
        self,
        source_path: str | Path,
        latent_path: str | Path | None = None,
        train_fraction: float = 0.95,
    ) -> None:
        if not 0 < train_fraction <= 1:
            raise ValueError('train_fraction must be in (0, 1]')
        self.source_path = Path(source_path).expanduser().resolve()
        self.latent_path = (
            Path(latent_path).expanduser().resolve()
            if latent_path is not None
            else None
        )
        with h5py.File(self.source_path, 'r') as source:
            required = {
                'pixels',
                'action',
                'reward',
                'terminal',
                'ep_offset',
                'ep_len',
            }
            missing = sorted(required - set(source.keys()))
            if missing:
                raise ValueError(f'Source dataset missing keys: {missing}')
            if source['pixels'].dtype != np.uint8:
                raise ValueError('pixels must be uint8')
            if tuple(source['pixels'].shape[1:]) != (64, 64, 3):
                raise ValueError('pixels must be HWC 64x64 RGB')
            reward_scheme = source.attrs.get('reward_scheme')
            if reward_scheme != REWARD_SCHEME:
                raise ValueError(
                    f'Expected reward_scheme={REWARD_SCHEME}, '
                    f'found {reward_scheme}'
                )
            self.offsets = source['ep_offset'][:].astype(np.int64, copy=False)
            self.lengths = source['ep_len'][:].astype(np.int64, copy=False)
            self.source_rows = int(source['pixels'].shape[0])
            self.action_dim = int(source['action'].shape[1])
            self.segment_transitions = int(
                source.attrs['segment_transitions']
            )
            terminals = source['terminal'][:].astype(bool, copy=False)
            rewards = source['reward'][:]
            source_episodes = (
                source['source_episode'][:]
                if 'source_episode' in source
                else None
            )
            actions = source['action'][:]
            if not np.isfinite(actions).all():
                raise ValueError('actions contain NaN or Inf')
            if actions.min() < -1.0001 or actions.max() > 1.0001:
                raise ValueError('actions must be normalized to [-1, 1]')
        if len(self.offsets) == 0 or np.any(self.lengths < 2):
            raise ValueError('Every episode must contain at least two rows')
        expected_offsets = np.concatenate(
            [np.array([0]), np.cumsum(self.lengths[:-1])]
        )
        if not np.array_equal(self.offsets, expected_offsets):
            raise ValueError('Episode offsets are not contiguous')
        if int(self.offsets[-1] + self.lengths[-1]) != self.source_rows:
            raise ValueError('Episode metadata does not cover all source rows')
        expected_terminals = np.zeros(self.source_rows, dtype=bool)
        expected_terminals[self.offsets + self.lengths - 1] = True
        if not np.array_equal(terminals, expected_terminals):
            raise ValueError('terminal rows disagree with episode metadata')
        goal_rows = self.offsets + self.lengths - 1
        expected_zero_rewards = np.zeros(self.source_rows, dtype=bool)
        expected_zero_rewards[goal_rows - 1] = True
        expected_zero_rewards[goal_rows] = True
        if not np.all(rewards[~expected_zero_rewards] == -1.0):
            raise ValueError('non-goal transition rewards must be -1')
        if not np.all(rewards[expected_zero_rewards] == 0.0):
            raise ValueError('goal-transition and dummy-row rewards must be 0')

        self.rows = self.source_rows
        self.latent_dim: int | None = None
        self.latent_min: float | None = None
        self.latent_max: float | None = None
        self._latents: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        if self.latent_path is not None:
            with h5py.File(self.latent_path, 'r') as latent_file:
                if 'latent' not in latent_file:
                    raise ValueError('Latent file is missing the latent dataset')
                latent_rows = int(latent_file['latent'].shape[0])
                if latent_rows > self.source_rows:
                    raise ValueError('Latent file has more rows than its source')
                if int(latent_file.attrs.get('source_size_bytes', -1)) != int(
                    self.source_path.stat().st_size
                ):
                    raise ValueError('Latent file was built from a different source')
                self.latent_dim = int(latent_file['latent'].shape[1])
                self.latent_min = float(latent_file.attrs['latent_min'])
                self.latent_max = float(latent_file.attrs['latent_max'])
                if not self.latent_max > self.latent_min:
                    raise ValueError('Invalid latent normalization bounds')
            available = self.offsets + self.lengths <= latent_rows
            if not np.array_equal(
                available, np.arange(len(available)) < int(available.sum())
            ):
                raise ValueError('Partial latent data must contain whole prefix episodes')
            self.offsets = self.offsets[available]
            self.lengths = self.lengths[available]
            if not len(self.offsets) or int(
                self.offsets[-1] + self.lengths[-1]
            ) != latent_rows:
                raise ValueError('Latent rows do not end on an episode boundary')
            self.rows = latent_rows
        self._segment_source_ids = (
            source_episodes[self.offsets] if source_episodes is not None else None
        )
        self.train_episodes = self.training_episode_count(
            len(self.offsets), train_fraction
        )
        if source_episodes is not None and self.train_episodes < len(self.offsets):
            train_end = int(
                self.offsets[self.train_episodes - 1]
                + self.lengths[self.train_episodes - 1]
            )
            validation_start = int(self.offsets[self.train_episodes])
            train_ids = np.unique(source_episodes[:train_end])
            validation_ids = np.unique(source_episodes[validation_start : self.rows])
            overlap = np.intersect1d(train_ids, validation_ids)
            if len(overlap):
                raise ValueError(
                    'train/validation segments share original source episodes: '
                    f'{overlap[:10].tolist()}'
                )

    def training_episode_count(
        self, total_episodes: int, train_fraction: float = 0.95
    ) -> int:
        """Choose a train boundary that does not split an original trajectory."""
        if not 1 <= total_episodes <= len(self.offsets):
            raise ValueError('total_episodes is outside the available prefix')
        desired = max(1, int(total_episodes * train_fraction))
        if self._segment_source_ids is not None and total_episodes > 1:
            segment_source_ids = self._segment_source_ids[:total_episodes]
            source_boundaries = np.flatnonzero(
                segment_source_ids[1:] != segment_source_ids[:-1]
            ) + 1
            eligible = source_boundaries[source_boundaries <= desired]
            if len(eligible):
                return int(eligible[-1])
            elif len(source_boundaries):
                return int(source_boundaries[0])
        return desired

    @property
    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            rows=self.rows,
            episodes=len(self.offsets),
            action_dim=self.action_dim,
            min_episode_rows=int(self.lengths.min()),
            max_episode_rows=int(self.lengths.max()),
            segment_transitions=self.segment_transitions,
        )

    def _episode_pool(self, split: str) -> np.ndarray:
        if split == 'train':
            return np.arange(self.train_episodes)
        if split == 'val':
            if self.train_episodes == len(self.offsets):
                return np.arange(len(self.offsets))
            return np.arange(self.train_episodes, len(self.offsets))
        raise ValueError("split must be 'train' or 'val'")

    def sample_image_rows(
        self, rng: np.random.Generator, batch_size: int, split: str = 'train'
    ) -> np.ndarray:
        pool = self._episode_pool(split)
        episode_ids = rng.choice(pool, size=batch_size, replace=True)
        relative = (
            rng.random(batch_size) * self.lengths[episode_ids]
        ).astype(np.int64)
        return self.offsets[episode_ids] + relative

    def sample_windows(
        self,
        rng: np.random.Generator,
        batch_size: int,
        pred_horizon: int,
        split: str = 'train',
    ) -> dict[str, np.ndarray]:
        if self.latent_path is None:
            raise ValueError('sample_windows requires a latent file')
        if batch_size < 1 or pred_horizon < 1:
            raise ValueError('batch_size and pred_horizon must be positive')
        pool = self._episode_pool(split)
        episode_ids = rng.choice(pool, size=batch_size, replace=True)
        lengths = self.lengths[episode_ids]
        relative_start = (rng.random(batch_size) * (lengths - 1)).astype(
            np.int64
        )
        offsets = self.offsets[episode_ids]
        current_rows = offsets + relative_start
        goal_rows = offsets + lengths - 1
        steps = np.arange(1, pred_horizon + 1, dtype=np.int64)[None]
        future_relative = np.minimum(
            relative_start[:, None] + steps, (lengths - 1)[:, None]
        )
        future_rows = offsets[:, None] + future_relative
        action_relative = np.minimum(
            relative_start[:, None] + steps - 1, (lengths - 1)[:, None]
        )
        action_rows = offsets[:, None] + action_relative

        if self._latents is None:
            with h5py.File(self.latent_path, 'r') as latent_file:
                latent = latent_file['latent']
                current = h5_take(latent, current_rows)[:, None]
                future = h5_take(latent, future_rows)
                goal = h5_take(latent, goal_rows)
        else:
            current = self._latents[current_rows, None]
            future = self._latents[future_rows]
            goal = self._latents[goal_rows]
        if self._actions is None:
            with h5py.File(self.source_path, 'r') as source:
                actions = h5_take(source['action'], action_rows).astype(
                    np.float32, copy=False
                )
        else:
            actions = self._actions[action_rows]
        return {
            'current': current.astype(np.float32),
            'future': future.astype(np.float32),
            'goal': goal.astype(np.float32),
            'actions': actions,
            'episode_ids': episode_ids,
            'current_rows': current_rows,
            'future_rows': future_rows,
            'goal_rows': goal_rows,
            'action_rows': action_rows,
        }

    def load_training_arrays(self) -> None:
        """Load compact latent/action arrays once for high-throughput training."""
        if self.latent_path is None:
            raise ValueError('load_training_arrays requires a latent file')
        with h5py.File(self.latent_path, 'r') as latent_file:
            self._latents = latent_file['latent'][:].astype(np.float32)
        with h5py.File(self.source_path, 'r') as source:
            self._actions = source['action'][: self.rows].astype(np.float32)

    def normalize_latent(self, value: np.ndarray) -> np.ndarray:
        if self.latent_min is None or self.latent_max is None:
            raise ValueError('normalize_latent requires a latent file')
        normalized = 2 * (value - self.latent_min) / (
            self.latent_max - self.latent_min
        ) - 1
        return np.clip(normalized, -1, 1).astype(np.float32)

    def denormalize_latent(self, value: np.ndarray) -> np.ndarray:
        if self.latent_min is None or self.latent_max is None:
            raise ValueError('denormalize_latent requires a latent file')
        return (
            (np.asarray(value) + 1) * 0.5
            * (self.latent_max - self.latent_min)
            + self.latent_min
        ).astype(np.float32)
