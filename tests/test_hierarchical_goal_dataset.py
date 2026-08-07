import numpy as np
import pytest
import torch

from stable_worldmodel.data.dataset import Dataset, HierarchicalGoalDataset


class IndexDataset(Dataset):
    def __init__(self, episode_length=10, **kwargs):
        super().__init__(
            lengths=np.array([episode_length]),
            offsets=np.array([0]),
            **kwargs,
        )

    @property
    def column_names(self):
        return ['pixels', 'proprio', 'action']

    def _load_slice(self, ep_idx, start, end):
        indices = torch.arange(start, end, self.frameskip)
        return {
            'pixels': indices[:, None],
            'proprio': indices[:, None].float(),
            'action': indices[:, None].float(),
        }


def test_hierarchical_goals_match_hiql_index_construction():
    dataset = IndexDataset(num_steps=3, frameskip=1)
    hierarchical = HierarchicalGoalDataset(
        dataset,
        goal_probabilities=(0.0, 0.0, 0.0, 1.0),
        actor_goal_probabilities=(0.0, 0.0, 0.0, 1.0),
        current_goal_offset=2,
        subgoal_steps=3,
        goal_keys={'pixels': 'goal_pixels'},
        low_goal_keys={'pixels': 'low_goal_pixels'},
        high_goal_keys={'pixels': 'high_goal_pixels'},
        high_target_keys={'pixels': 'high_target_pixels'},
        seed=0,
    )

    item = hierarchical[0]

    assert item['goal_pixels'].squeeze().item() == 1
    assert item['low_goal_pixels'].squeeze(-1).tolist() == [3, 4]
    assert item['high_goal_pixels'].squeeze().item() == 1
    assert item['high_target_pixels'].squeeze(-1).tolist() == [1, 1]


def test_random_high_goal_uses_reachable_k_step_targets():
    dataset = IndexDataset(num_steps=3, frameskip=1)
    hierarchical = HierarchicalGoalDataset(
        dataset,
        goal_probabilities=(0.0, 0.0, 0.0, 1.0),
        actor_goal_probabilities=(1.0, 0.0, 0.0, 0.0),
        current_goal_offset=2,
        subgoal_steps=3,
        goal_keys={'pixels': 'goal_pixels'},
        seed=1,
    )

    item = hierarchical[0]

    assert item['low_goal_pixels'].squeeze(-1).tolist() == [3, 4]
    assert item['high_target_pixels'].squeeze(-1).tolist() == [3, 4]


def test_hierarchical_goals_clip_to_episode_end_like_hiql():
    dataset = IndexDataset(episode_length=6, num_steps=3, frameskip=1)
    hierarchical = HierarchicalGoalDataset(
        dataset,
        goal_probabilities=(0.0, 0.0, 0.0, 1.0),
        actor_goal_probabilities=(1.0, 0.0, 0.0, 0.0),
        current_goal_offset=2,
        subgoal_steps=4,
        goal_keys={'pixels': 'goal_pixels'},
        seed=2,
    )

    item = hierarchical[len(hierarchical) - 1]

    assert item['pixels'].squeeze(-1).tolist() == [3, 4, 5]
    assert item['low_goal_pixels'].squeeze(-1).tolist() == [5, 5]
    assert item['high_target_pixels'].squeeze(-1).tolist() == [5, 5]


def test_hierarchical_dataset_requires_one_step_successors():
    dataset = IndexDataset(num_steps=2, frameskip=1)
    with pytest.raises(ValueError, match=r'current_goal_offset \+ 1'):
        HierarchicalGoalDataset(
            dataset,
            goal_probabilities=(0.0, 0.0, 0.0, 1.0),
            current_goal_offset=2,
            subgoal_steps=3,
        )
