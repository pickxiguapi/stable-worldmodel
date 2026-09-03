import torch
from omegaconf import OmegaConf

from stable_worldmodel.wm.tdmpc2 import TDMPC2


def test_tdmpc2_accepts_current_and_goal_rgb_channels():
    cfg = OmegaConf.create(
        {
            'image_size': 64,
            'image_channels': 6,
            'action_dim': 5,
            'extra_dims': {'action': 5},
            'wm': {
                'encoding': {'pixels': 128},
                'enc_dim': 256,
                'mlp_dim': 384,
                'simnorm_dim': 8,
                'num_q': 5,
                'num_bins': 101,
                'tau': 0.01,
            },
        }
    )
    model = TDMPC2(cfg)

    pixels = torch.randn(2, 6, 64, 64)
    encoded = model.encode({'pixels': pixels})

    assert model.cnn[0].in_channels == 6
    assert encoded.shape == (2, 128)

    current = torch.randn(2, 3, 64, 64)
    goal = torch.randn(2, 3, 64, 64)
    goal_encoded = model.encode({'pixels': current, 'goal': goal})
    assert goal_encoded.shape == (2, 128)
