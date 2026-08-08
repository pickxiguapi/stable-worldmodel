import math

import torch
from einops import rearrange
from torch import nn

from .gcrl import GCRL


class GCHIQL(GCRL):
    """Goal-conditioned hierarchical IQL policy and value container.

    The high-level policy predicts a normalized subgoal representation from
    the observation and final goal. The low-level policy maps the observation
    and that representation to an environment action.
    """

    def __init__(
        self,
        encoder,
        low_action_predictor,
        high_action_predictor,
        value_predictor,
        critic_predictor=None,
        extra_encoders=None,
        history_size=3,
        interpolate_pos_encoding=True,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__(
            encoder=encoder,
            action_predictor=low_action_predictor,
            value_predictor=value_predictor,
            critic_predictor=critic_predictor,
            extra_encoders=extra_encoders,
            history_size=history_size,
            interpolate_pos_encoding=interpolate_pos_encoding,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.high_action_predictor = high_action_predictor

        # OGBench HIQL uses const_std=True by default. GCRL exposes a
        # learnable low-level log-std for other algorithms, so freeze it here
        # and keep both hierarchical policies at unit standard deviation.
        self.log_stds.requires_grad_(False)

        out_proj = high_action_predictor.out_proj
        if isinstance(out_proj, nn.Sequential):
            rep_dim = out_proj[-1].out_features
        else:
            rep_dim = out_proj.out_features
        self.rep_dim = rep_dim
        self.register_buffer('high_log_stds', torch.zeros(rep_dim))

    @property
    def low_action_predictor(self):
        """Alias the base action predictor without registering it twice."""
        return self.action_predictor

    def predict_low_actions(
        self,
        embedding,
        goal_representations,
        temperature=1.0,
    ):
        """Return low-level Gaussian means and fixed unit stds."""
        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        means = self.low_action_predictor(embedding, goal_representations)
        log_stds = torch.clamp(
            self.log_stds, self.log_std_min, self.log_std_max
        )
        return means, torch.exp(log_stds) * temperature

    def predict_high_subgoals(
        self,
        embedding,
        embedding_goal,
        temperature=1.0,
    ):
        """Return high-level subgoal means and fixed unit stds."""
        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        embedding_goal = rearrange(embedding_goal, 'b t p d -> b (t p) d')
        means = self.high_action_predictor(embedding, embedding_goal)
        log_stds = torch.clamp(
            self.high_log_stds, self.log_std_min, self.log_std_max
        )
        return means, torch.exp(log_stds) * temperature

    def get_action(self, info, sample=False, temperature=1.0):
        """Compose the high- and low-level policies and return the last action."""
        info = self.encode(info, pixels_key='pixels', target='embed')
        info = self.encode(
            info,
            pixels_key='goal',
            prefix='goal_',
            target='goal_embed',
        )
        high_means, high_stds = self.predict_high_subgoals(
            info['embed'], info['goal_embed'], temperature=temperature
        )
        if sample:
            goal_representations = high_means + high_stds * torch.randn_like(
                high_means
            )
        else:
            goal_representations = high_means
        goal_representations = torch.nn.functional.normalize(
            goal_representations, dim=-1, eps=1e-8
        ) * math.sqrt(self.rep_dim)

        low_means, low_stds = self.predict_low_actions(
            info['embed'], goal_representations, temperature=temperature
        )
        if sample:
            actions = low_means + low_stds * torch.randn_like(low_means)
        else:
            actions = low_means
        return actions[:, -1, :]


__all__ = ['GCHIQL']
