from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stable_worldmodel.wm.gcrl import QCHIQLChunkNew
from stable_worldmodel.wm.gcrl.qchiql_chunk_new import MLP, ActorMLP


class DummyViT(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.mean(dim=(-1, -2))
        cls = self.proj(pooled).unsqueeze(1)
        return SimpleNamespace(last_hidden_state=cls)


def _small_model():
    return QCHIQLChunkNew(
        high_encoder=DummyViT(8),
        low_encoder=DummyViT(8),
        feature_dim=8,
        action_dim=6,
        rep_dim=4,
        value_hidden_dims=(16, 16),
        actor_hidden_dims=(16, 16),
        goal_rep_hidden_dims=(16,),
    )


def test_qchiql_control_heads_stay_float32_under_mixed_precision():
    model = MLP(8, 2, hidden_dims=(16,))
    inputs = torch.randn(4, 8)

    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        outputs = model(inputs)

    assert outputs.dtype == torch.float32


def test_qchiql_mlp_initialization_matches_ogbench():
    model = MLP(32, 4, hidden_dims=(64, 64), layer_norm=True)
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]

    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linears)
    assert all(norm.eps == 1e-6 for norm in norms)

    actor = ActorMLP(512, 10, hidden_dims=(512, 512, 512))
    expected_std = 0.1 * (2.0 / (512 + 10)) ** 0.5
    assert actor.output.weight.std().item() == pytest.approx(
        expected_std,
        rel=0.15,
    )
    assert torch.count_nonzero(actor.output.bias) == 0


def test_qchiql_initial_goal_representations_are_not_constant():
    torch.manual_seed(0)
    model = _small_model()
    states = model.encode_high(torch.randn(64, 1, 3, 16, 16))
    goals = model.encode_high(torch.randn(64, 1, 3, 16, 16))
    representations = model.represent_goal(states, goals)

    feature_std = representations.std(dim=0, unbiased=False).mean()
    pair_distance = (representations[1:] - representations[:-1]).norm(
        dim=-1
    ).mean()
    assert feature_std > 1e-3
    assert pair_distance > 1e-2


def test_qchiql_has_separate_high_low_encoders_and_frozen_targets():
    model = _small_model()

    assert model.high_encoder is not model.low_encoder
    assert not any(
        p is q
        for p in model.high_encoder.parameters()
        for q in model.low_encoder.parameters()
    )
    assert all(
        not p.requires_grad for p in model.target_high_encoder.parameters()
    )
    assert all(
        not p.requires_grad for p in model.target_high_value.parameters()
    )
    assert all(
        not p.requires_grad for p in model.target_low_encoder.parameters()
    )
    assert all(
        not p.requires_grad for p in model.target_low_critic.parameters()
    )
    assert all(
        not p.requires_grad
        for p in model.target_goal_representation.parameters()
    )


def test_qchiql_twin_value_q_and_actor_shapes():
    model = _small_model()
    batch_size, frames = 2, 3
    pixels = torch.randn(batch_size, frames, 3, 16, 16)
    goals = torch.randn(batch_size, 1, 3, 16, 16)

    high_states = model.encode_high(pixels)
    high_goals = model.encode_high(goals)
    low_states = model.encode_low(pixels)
    reps = model.represent_goal(high_states, high_goals)
    actions = torch.randn(batch_size, frames, 6)

    high_v1, high_v2 = model.predict_high_value(high_states, high_goals)
    low_q1, low_q2 = model.predict_low_q(low_states, reps, actions)
    low_values = model.predict_low_value(low_states, reps)
    high_actions, _ = model.predict_high_subgoals(high_states, high_goals)
    low_actions, _ = model.predict_low_actions(low_states, reps)

    assert high_v1.shape == high_v2.shape == (batch_size, frames, 1)
    assert low_q1.shape == low_q2.shape == (batch_size, frames, 1)
    assert low_values.shape == (batch_size, frames, 1)
    assert high_actions.shape == (batch_size, frames, 4)
    assert low_actions.shape == (batch_size, frames, 6)


def test_qchiql_polyak_updates_target_networks():
    model = _small_model()
    pairs = (
        (model.high_encoder, model.target_high_encoder),
        (model.high_value, model.target_high_value),
        (model.low_encoder, model.target_low_encoder),
        (model.goal_representation, model.target_goal_representation),
        (model.low_critic, model.target_low_critic),
    )
    snapshots = []
    for online_module, target_module in pairs:
        online = next(online_module.parameters())
        target = next(target_module.parameters())
        snapshots.append((target, target.detach().clone()))
        with torch.no_grad():
            online.add_(2.0)

    model.update_targets(tau=0.25)

    for target, before in snapshots:
        assert torch.allclose(target, before + 0.5)


def test_qchiql_updates_targets_after_optimizer_step(monkeypatch):
    import stable_pretraining as spt

    from scripts.train.qchiql_chunk_new import QCHIQLTrainingModule

    model = _small_model()
    module = QCHIQLTrainingModule(
        model=model,
        forward=lambda self, batch, stage: batch,
        optim={},
        target_tau=0.25,
    )
    online = next(model.high_encoder.parameters())
    target = next(model.target_high_encoder.parameters())
    target_before = target.detach().clone()
    events = []

    def fake_optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
    ):
        events.append('optimizer')
        with torch.no_grad():
            online.add_(2.0)

    original_update_targets = model.update_targets

    def tracked_update_targets(tau):
        events.append('target')
        original_update_targets(tau)

    monkeypatch.setattr(spt.Module, 'optimizer_step', fake_optimizer_step)
    monkeypatch.setattr(model, 'update_targets', tracked_update_targets)

    module.optimizer_step(0, 0, optimizer=None)

    assert events == ['optimizer', 'target']
    assert torch.allclose(target, target_before + 0.5)


def test_qchiql_awr_uses_stable_wm_alpha_multiplication():
    from scripts.train.qchiql_chunk_new import _awr_weights

    advantages = torch.tensor([-1.0, 0.0, 1.0, 10.0])
    weights = _awr_weights(advantages, alpha=3.0)

    assert torch.allclose(
        weights,
        torch.exp(3.0 * advantages).clamp(max=100.0),
    )


def test_qchiql_training_objective_is_finite():
    from scripts.train.qchiql_chunk_new import get_qchiql_chunk_new_model

    cfg = OmegaConf.create(
        {
            'frameskip': 2,
            'discount': 0.99,
            'expectile': 0.7,
            'low_alpha': 3.0,
            'high_alpha': 3.0,
            'low_actor_rep_grad': True,
            'gc_negative': True,
            'tau': 0.005,
            'image_size': 28,
            'patch_size': 14,
            'encoder': 'vit_tiny',
            'const_std': True,
            'discrete': False,
            'rep_dim': 4,
            'subgoal_steps': 4,
            'n_steps': 3,
            'lr': 3e-4,
            'dinowm': {
                'history_size': 2,
                'td_offset': 1,
                'use_proprio_encoder': False,
                'action_dim': 3,
            },
            'actor_hidden_dims': [32, 32],
            'value_hidden_dims': [32, 32],
            'layer_norm': True,
        }
    )
    module = get_qchiql_chunk_new_model(cfg)
    batch_size = 2
    batch = {
        'pixels': torch.rand(batch_size, 3, 3, 28, 28),
        # Deliberately distinct so the regression assertions below catch any
        # accidental mixing of value and high-actor goal fields.
        'goal_pixels': torch.zeros(batch_size, 1, 3, 28, 28),
        'low_goal_pixels': torch.rand(batch_size, 2, 3, 28, 28),
        'high_goal_pixels': torch.ones(batch_size, 1, 3, 28, 28),
        'high_target_pixels': torch.rand(batch_size, 2, 3, 28, 28),
        'action': torch.rand(batch_size, 3, 6),
    }

    captured_value_goals = []
    captured_actor_goal = {}
    captured_goal_rep_inputs = []
    original_predict_high_value = module.model.predict_high_value
    original_predict_high_subgoals = module.model.predict_high_subgoals
    original_represent_goal = module.model.represent_goal

    def capture_high_value(states, goals):
        captured_value_goals.append(goals.detach().clone())
        return original_predict_high_value(states, goals)

    def capture_high_subgoals(states, goals, temperature=1.0):
        captured_actor_goal['goal'] = goals.detach().clone()
        return original_predict_high_subgoals(states, goals, temperature)

    def capture_represent_goal(states, goals):
        captured_goal_rep_inputs.append(
            (states.detach().clone(), goals.detach().clone())
        )
        return original_represent_goal(states, goals)

    module.model.predict_high_value = capture_high_value
    module.model.predict_high_subgoals = capture_high_subgoals
    module.model.represent_goal = capture_represent_goal
    output = module(batch, 'train')

    with torch.no_grad():
        expected_high_states = module.model.encode_high(batch['pixels'])[
            :, : cfg.dinowm.history_size
        ]
        expected_low_states = module.model.encode_low(batch['pixels'])[
            :, : cfg.dinowm.history_size
        ]
        expected_value_goal = module.model.encode_high(batch['goal_pixels'])
        expected_actor_goal = module.model.encode_high(
            batch['high_goal_pixels']
        )
        expected_low_goal = module.model.encode_high(
            batch['low_goal_pixels']
        )
    assert len(captured_value_goals) == 3
    assert torch.allclose(captured_value_goals[0], expected_value_goal)
    assert torch.allclose(captured_value_goals[1], expected_actor_goal)
    assert torch.allclose(captured_value_goals[2], expected_actor_goal)
    assert torch.allclose(captured_actor_goal['goal'], expected_actor_goal)
    # V_H itself must learn through phi([s; g]), rather than concatenating a
    # raw goal feature outside the shared representation.
    assert torch.allclose(
        captured_goal_rep_inputs[0][0], expected_high_states
    )
    assert torch.allclose(
        captured_goal_rep_inputs[0][1], expected_value_goal
    )
    # The low branch then consumes phi([s; s_{t+c}]) from that same high-level
    # representation path, not features from its own visual encoder.
    assert torch.allclose(
        captured_goal_rep_inputs[1][0], expected_high_states
    )
    assert torch.allclose(captured_goal_rep_inputs[1][1], expected_low_goal)
    assert not torch.allclose(
        captured_goal_rep_inputs[1][0], expected_low_states
    )

    module.zero_grad(set_to_none=True)
    output['low_actor_loss'].backward(retain_graph=True)
    assert any(
        p.grad is not None
        for p in module.model.goal_representation.parameters()
    )
    assert any(
        p.grad is not None for p in module.model.high_encoder.parameters()
    )
    assert any(
        p.grad is not None for p in module.model.low_encoder.parameters()
    )

    module.zero_grad(set_to_none=True)
    output['loss'].backward()

    assert torch.isfinite(output['loss'])
    assert torch.isfinite(output['high_value_loss'])
    assert torch.isfinite(output['low_value_loss'])
    assert torch.isfinite(output['low_q_loss'])
    assert torch.isfinite(output['high_actor_loss'])
    assert torch.isfinite(output['low_actor_loss'])
    assert any(
        p.grad is not None for p in module.model.high_encoder.parameters()
    )
    assert any(
        p.grad is not None for p in module.model.low_encoder.parameters()
    )
    assert all(
        p.grad is None for p in module.model.target_high_encoder.parameters()
    )
    assert all(
        p.grad is None for p in module.model.target_low_encoder.parameters()
    )


def test_qchiql_rejects_non_chunk_aligned_actions():
    model = _small_model()
    low_states = torch.randn(2, 3, 8)
    reps = torch.randn(2, 3, 4)
    actions = torch.randn(2, 2, 6)

    with pytest.raises(ValueError, match='one action chunk per state frame'):
        model.predict_low_q(low_states, reps, actions)


def test_qchiql_rejects_misaligned_temporal_config():
    from scripts.train.qchiql_chunk_new import get_qchiql_chunk_new_model

    cfg = OmegaConf.create(
        {
            'frameskip': 5,
            'subgoal_steps': 12,
            'n_steps': 4,
            'gc_negative': True,
            'dinowm': {'history_size': 3, 'td_offset': 1},
        }
    )
    with pytest.raises(ValueError, match='positive multiple of frameskip'):
        get_qchiql_chunk_new_model(cfg)


def test_qchiql_uses_ogbench_goal_parameter_names():
    from scripts.train.qchiql_chunk_new import (
        _get_subgoal_steps,
        _goal_probabilities,
    )

    cfg = OmegaConf.create(
        {
            'frameskip': 5,
            'subgoal_steps': 10,
            'value_p_curgoal': 0.2,
            'value_p_trajgoal': 0.5,
            'value_p_randomgoal': 0.3,
            'value_geom_sample': True,
            'actor_p_curgoal': 0.0,
            'actor_p_trajgoal': 1.0,
            'actor_p_randomgoal': 0.0,
            'actor_geom_sample': False,
        }
    )

    assert _get_subgoal_steps(cfg) == (10, 2)
    assert _goal_probabilities(cfg, 'value') == (0.3, 0.5, 0.0, 0.2)
    assert _goal_probabilities(cfg, 'actor') == (0.0, 0.0, 1.0, 0.0)


def test_qchiql_keeps_legacy_checkpoint_config_compatibility():
    from scripts.train.qchiql_chunk_new import (
        _get_subgoal_steps,
        _goal_probabilities,
    )

    cfg = OmegaConf.create(
        {
            'frameskip': 5,
            'subgoal_horizon': 10,
            'subgoal_steps': 2,
            'goal_probabilities': {
                'random': 0.3,
                'geometric_future': 0.5,
                'uniform_future': 0.0,
                'current': 0.2,
            },
            'actor_goal_probabilities': {
                'random': 0.0,
                'geometric_future': 0.0,
                'uniform_future': 1.0,
                'current': 0.0,
            },
        }
    )

    assert _get_subgoal_steps(cfg) == (10, 2)
    assert _goal_probabilities(cfg, 'value') == (0.3, 0.5, 0.0, 0.2)
    assert _goal_probabilities(cfg, 'actor') == (0.0, 0.0, 1.0, 0.0)
