from types import SimpleNamespace

import torch
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.wm.gcrl import (
    GCHIQL,
    GoalRepresentationPredictor,
    HierarchicalValuePredictor,
    RepresentationPredictor,
)

PREDICTOR_KWARGS = {
    'num_patches': 2,
    'num_frames': 3,
    'dim': 8,
    'depth': 2,
    'heads': 2,
    'mlp_dim': 16,
    'dim_head': 4,
}


class DummyEncoder(nn.Module):
    def forward(self, pixels):
        batch_size = pixels.shape[0]
        patches = torch.ones(batch_size, 3, 8, device=pixels.device)
        return SimpleNamespace(last_hidden_state=patches)


def test_gchiql_public_api_is_exported():
    assert swm.wm.gcrl.GCHIQL is GCHIQL
    assert swm.wm.GCHIQL is GCHIQL


def test_hierarchical_value_supports_shared_and_aligned_goals():
    model = HierarchicalValuePredictor(rep_dim=4, **PREDICTOR_KWARGS)
    states = torch.randn(2, 6, 8)
    shared_goal = torch.randn(2, 2, 8)
    aligned_goals = torch.randn(2, 6, 8)

    shared_representations = model.encode_goal(states, shared_goal)
    aligned_representations = model.encode_goal(states, aligned_goals)
    v1, v2 = model(states, shared_goal)

    assert shared_representations.shape == (2, 3, 4)
    assert aligned_representations.shape == (2, 3, 4)
    assert torch.allclose(
        shared_representations.norm(dim=-1),
        torch.full((2, 3), 2.0),
        atol=1e-5,
    )
    assert v1.shape == v2.shape == (2, 3, 1)


def test_gchiql_composes_high_and_low_policies():
    low_actor = RepresentationPredictor(
        out_dim=2,
        rep_dim=4,
        **PREDICTOR_KWARGS,
    )
    high_actor = GoalRepresentationPredictor(
        rep_dim=4,
        normalize=False,
        **PREDICTOR_KWARGS,
    )
    value = HierarchicalValuePredictor(rep_dim=4, **PREDICTOR_KWARGS)
    model = GCHIQL(
        encoder=DummyEncoder(),
        low_action_predictor=low_actor,
        high_action_predictor=high_actor,
        value_predictor=value,
        history_size=3,
        interpolate_pos_encoding=False,
    )

    assert not model.log_stds.requires_grad
    assert 'high_log_stds' in dict(model.named_buffers())
    assert 'high_log_stds' not in dict(model.named_parameters())

    info = {
        'pixels': torch.zeros(2, 3, 1, 1, 1),
        'goal': torch.zeros(2, 1, 1, 1, 1),
    }
    actions = model.get_action(info, sample=False)

    assert actions.shape == (2, 2)
    assert torch.isfinite(actions).all()
