import pytest
import torch
from omegaconf import OmegaConf

from stable_worldmodel.wm.gcrl import RepresentationQPredictor


def _critic():
    return RepresentationQPredictor(
        num_patches=2,
        num_frames=3,
        dim=8,
        depth=1,
        heads=2,
        mlp_dim=16,
        action_dim=10,
        rep_dim=4,
        hidden_dim=12,
        dim_head=4,
    )


def test_representation_q_predictor_consumes_full_action_chunks():
    critic = _critic()
    states = torch.randn(2, 6, 8)
    action_chunks = torch.randn(2, 3, 10)
    latent_subgoals = torch.randn(2, 3, 4)

    q_values = critic(states, action_chunks, latent_subgoals)

    assert q_values.shape == (2, 3, 1)


def test_representation_q_predictor_broadcasts_shared_subgoal():
    critic = _critic()
    states = torch.randn(2, 6, 8)
    action_chunks = torch.randn(2, 3, 10)
    shared_subgoal = torch.randn(2, 1, 4)

    q_values = critic(states, action_chunks, shared_subgoal)

    assert q_values.shape == (2, 3, 1)


def test_representation_q_predictor_rejects_misaligned_chunks():
    critic = _critic()
    states = torch.randn(2, 6, 8)
    action_chunks = torch.randn(2, 2, 10)
    latent_subgoals = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match='one action chunk per state frame'):
        critic(states, action_chunks, latent_subgoals)


def test_gchiql_chunk_model_registers_critic_and_chunk_action_head():
    from scripts.train.gchiql import (
        get_chunk_td_constants,
        get_gchiql_chunk_model,
    )

    cfg = OmegaConf.create(
        {
            'frameskip': 2,
            'discount': 0.99,
            'expectile': 0.7,
            'image_size': 28,
            'patch_size': 14,
            'encoder_type': 'vit_tiny',
            'rep_dim': 4,
            'predictor_lr': 3e-4,
            'proprio_encoder_lr': 3e-4,
            'encoder_lr': 3e-4,
            'value_ema_tau': 0.995,
            'gc_negative': True,
            'dinowm': {
                'history_size': 2,
                'td_offset': 1,
                'use_proprio_encoder': False,
                'action_dim': 3,
            },
            'predictor': {
                'depth': 1,
                'heads': 2,
                'mlp_dim': 32,
                'dim_head': 8,
                'dropout': 0.0,
                'emb_dropout': 0.0,
            },
        }
    )

    gamma_span, reward_lump = get_chunk_td_constants(cfg)
    module = get_gchiql_chunk_model(cfg)

    assert gamma_span == pytest.approx(0.99**2)
    assert reward_lump == pytest.approx(1.0 + 0.99)
    assert module.model.critic_predictor is not None
    assert module.model.log_stds.shape == (6,)

    batch_size = 2
    batch = {
        'pixels': torch.rand(batch_size, 3, 3, 28, 28),
        'goal_pixels': torch.rand(batch_size, 1, 3, 28, 28),
        'low_goal_pixels': torch.rand(batch_size, 2, 3, 28, 28),
        'high_goal_pixels': torch.rand(batch_size, 1, 3, 28, 28),
        'high_target_pixels': torch.rand(batch_size, 2, 3, 28, 28),
        'action': torch.rand(batch_size, 3, 6),
    }

    output = module(batch, 'train')

    assert torch.isfinite(output['loss'])
    assert torch.isfinite(output['critic_loss'])
    assert torch.isfinite(output['low_value_loss'])
