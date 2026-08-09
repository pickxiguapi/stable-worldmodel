"""Compact hierarchical implicit Q-chunking networks.

The model deliberately keeps representation learning and control heads small:
the high- and low-level branches own separate ViT encoders, while all value,
critic, and actor heads are OGBench-style MLPs.  In particular, no control
head contains a Transformer; twin and target networks are explicit attributes.
"""

import copy
import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MLP(nn.Module):
    """MLP matching OGBench value/critic and representation heads."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(512, 512, 512),
        layer_norm=True,
    ):
        super().__init__()
        dims = (input_dim, *hidden_dims)
        layers = []
        for in_dim, out_dim in pairwise(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
        self.output = nn.Linear(dims[-1], output_dim)
        layers.append(self.output)
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        # Keep the small control heads in FP32 even when the visual encoders
        # run under bf16 mixed precision.  Goal-conditioned advantages are
        # differences between similarly scaled values; bf16 can round those
        # differences to exactly zero and silently reduce AWR to plain BC.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return self.net(inputs.float())


class TwinMLP(nn.Module):
    """Two independently initialized MLP estimates, as used by OGBench."""

    def __init__(self, input_dim, output_dim, hidden_dims=(512, 512, 512)):
        super().__init__()
        self.net1 = MLP(input_dim, output_dim, hidden_dims)
        self.net2 = MLP(input_dim, output_dim, hidden_dims)

    def forward(self, inputs):
        return self.net1(inputs), self.net2(inputs)


class ActorMLP(MLP):
    """OGBench Gaussian actor mean MLP with its small output initializer."""

    def __init__(self, input_dim, output_dim, hidden_dims=(512, 512, 512)):
        super().__init__(
            input_dim,
            output_dim,
            hidden_dims,
            layer_norm=False,
        )
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)


class QCHIQLChunkNew(nn.Module):
    """HiQC-style policy with independent high/low visual encoders.

    Conceptual components are one high-level value, one low-level value, one
    low-level chunk Q-function, and two independent actors.  Following the
    official OGBench implementations, high value and low Q use twin estimates
    and EMA target copies; low value is a single online network.
    """

    def __init__(
        self,
        high_encoder,
        low_encoder,
        feature_dim,
        action_dim,
        rep_dim=10,
        hidden_dims=(512, 512, 512),
        rep_hidden_dims=(512, 512, 512),
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.high_encoder = high_encoder
        self.low_encoder = low_encoder
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.rep_dim = rep_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # phi(s, w) is the HIQL latent subgoal space.  The high value learns
        # V_H(s, phi(s, g)); the high actor predicts phi(s, s_{t+c}); and the
        # low branch consumes the same latent representation.
        self.goal_representation = MLP(
            2 * feature_dim,
            rep_dim,
            rep_hidden_dims,
        )

        # High level follows OGBench HIQL: twin V_H plus target twin V_H.
        self.high_value = TwinMLP(feature_dim + rep_dim, 1, hidden_dims)
        self.high_actor = ActorMLP(2 * feature_dim, rep_dim, hidden_dims)

        # Low level: V_L(s, z), Q_L(s, z, a_{t:t+k}), and pi_L.
        low_condition_dim = feature_dim + rep_dim
        self.low_value = MLP(low_condition_dim, 1, hidden_dims)
        # Low level follows OGBench GCIQL: twin Q_L plus target twin Q_L.
        self.low_critic = TwinMLP(
            low_condition_dim + action_dim,
            1,
            hidden_dims,
        )
        self.low_actor = ActorMLP(
            low_condition_dim,
            action_dim,
            hidden_dims,
        )

        # OGBench HIQL uses constant policy standard deviations by default.
        self.register_buffer('high_log_stds', torch.zeros(rep_dim))
        self.register_buffer('low_log_stds', torch.zeros(action_dim))

        # Targets include their visual/goal encoders, just as the pixel-based
        # OGBench target networks do.  They never receive gradients.
        self.target_high_encoder = copy.deepcopy(high_encoder)
        self.target_high_value = copy.deepcopy(self.high_value)
        self.target_low_encoder = copy.deepcopy(low_encoder)
        self.target_goal_representation = copy.deepcopy(
            self.goal_representation
        )
        self.target_low_critic = copy.deepcopy(self.low_critic)
        for module in (
            self.target_high_encoder,
            self.target_high_value,
            self.target_low_encoder,
            self.target_goal_representation,
            self.target_low_critic,
        ):
            module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for module in (
            self.target_high_encoder,
            self.target_high_value,
            self.target_low_encoder,
            self.target_goal_representation,
            self.target_low_critic,
        ):
            module.eval()
        return self

    @staticmethod
    def _encode(encoder, pixels):
        """Encode ``(B,T,C,H,W)`` images to one CLS feature per frame."""
        batch_size = pixels.shape[0]
        flat_pixels = rearrange(pixels.float(), 'b t c h w -> (b t) c h w')
        output = encoder(flat_pixels, interpolate_pos_encoding=True)
        if hasattr(output, 'last_hidden_state'):
            features = output.last_hidden_state[:, 0]
        elif hasattr(output, 'logits'):
            features = output.logits
        else:
            raise TypeError(
                'ViT encoder must return last_hidden_state or logits'
            )
        return rearrange(features, '(b t) d -> b t d', b=batch_size)

    def encode_high(self, pixels):
        return self._encode(self.high_encoder, pixels)

    def encode_low(self, pixels):
        return self._encode(self.low_encoder, pixels)

    def encode_target_high(self, pixels):
        return self._encode(self.target_high_encoder, pixels)

    def encode_target_low(self, pixels):
        return self._encode(self.target_low_encoder, pixels)

    @staticmethod
    def _align(condition, num_frames):
        if condition.shape[1] == num_frames:
            return condition
        if condition.shape[1] == 1:
            return condition.expand(-1, num_frames, -1)
        raise ValueError(
            f'Expected one or {num_frames} conditioning frames, got '
            f'{condition.shape[1]}'
        )

    def represent_goal(self, reference_states, goal_states):
        """Compute the shared HIQL latent ``phi([state; goal])``."""
        goal_states = self._align(goal_states, reference_states.shape[1])
        representations = self.goal_representation(
            torch.cat([reference_states, goal_states], dim=-1)
        )
        return F.normalize(representations, dim=-1, eps=1e-8) * math.sqrt(
            self.rep_dim
        )

    def represent_target_goal(self, reference_states, goal_states):
        """Compute ``phi`` with the EMA representation parameters."""
        goal_states = self._align(goal_states, reference_states.shape[1])
        representations = self.target_goal_representation(
            torch.cat([reference_states, goal_states], dim=-1)
        )
        return F.normalize(representations, dim=-1, eps=1e-8) * math.sqrt(
            self.rep_dim
        )

    def predict_high_value(self, high_states, high_goals):
        goal_representations = self.represent_goal(high_states, high_goals)
        return self.high_value(
            torch.cat([high_states, goal_representations], dim=-1)
        )

    def predict_target_high_value(self, high_states, high_goals):
        goal_representations = self.represent_target_goal(
            high_states,
            high_goals,
        )
        return self.target_high_value(
            torch.cat([high_states, goal_representations], dim=-1)
        )

    def predict_low_value(self, low_states, goal_representations):
        goal_representations = self._align(
            goal_representations, low_states.shape[1]
        )
        return self.low_value(
            torch.cat([low_states, goal_representations], dim=-1)
        )

    def predict_low_q(self, low_states, goal_representations, action_chunks):
        goal_representations = self._align(
            goal_representations, low_states.shape[1]
        )
        if action_chunks.shape[:2] != low_states.shape[:2]:
            raise ValueError(
                'Expected one action chunk per state frame, got '
                f'{tuple(action_chunks.shape[:2])} for '
                f'{tuple(low_states.shape[:2])}'
            )
        return self.low_critic(
            torch.cat(
                [low_states, goal_representations, action_chunks], dim=-1
            )
        )

    def predict_target_low_q(
        self,
        low_states,
        goal_representations,
        action_chunks,
    ):
        goal_representations = self._align(
            goal_representations, low_states.shape[1]
        )
        if action_chunks.shape[:2] != low_states.shape[:2]:
            raise ValueError(
                'Expected one action chunk per state frame, got '
                f'{tuple(action_chunks.shape[:2])} for '
                f'{tuple(low_states.shape[:2])}'
            )
        return self.target_low_critic(
            torch.cat(
                [low_states, goal_representations, action_chunks], dim=-1
            )
        )

    def predict_high_subgoals(self, high_states, high_goals, temperature=1.0):
        high_goals = self._align(high_goals, high_states.shape[1])
        means = self.high_actor(torch.cat([high_states, high_goals], dim=-1))
        stds = (
            torch.exp(
                self.high_log_stds.clamp(self.log_std_min, self.log_std_max)
            )
            * temperature
        )
        return means, stds

    def predict_low_actions(
        self,
        low_states,
        goal_representations,
        temperature=1.0,
    ):
        goal_representations = self._align(
            goal_representations, low_states.shape[1]
        )
        means = self.low_actor(
            torch.cat([low_states, goal_representations], dim=-1)
        )
        stds = (
            torch.exp(
                self.low_log_stds.clamp(self.log_std_min, self.log_std_max)
            )
            * temperature
        )
        return means, stds

    def get_action(self, info, sample=False, temperature=1.0):
        """Plan a latent subgoal and return the latest full action chunk."""
        high_states = self.encode_high(info['pixels'])
        high_goals = self.encode_high(info['goal'])
        high_means, high_stds = self.predict_high_subgoals(
            high_states,
            high_goals,
            temperature=temperature,
        )
        if sample:
            subgoals = high_means + high_stds * torch.randn_like(high_means)
        else:
            subgoals = high_means
        subgoals = F.normalize(subgoals, dim=-1, eps=1e-8) * math.sqrt(
            self.rep_dim
        )

        low_states = self.encode_low(info['pixels'])
        low_means, low_stds = self.predict_low_actions(
            low_states,
            subgoals,
            temperature=temperature,
        )
        if sample:
            actions = low_means + low_stds * torch.randn_like(low_means)
        else:
            actions = low_means
        return actions[:, -1]

    @torch.no_grad()
    def update_targets(self, tau=0.005):
        """Polyak-update all OGBench-style target networks."""
        pairs = (
            (self.high_encoder, self.target_high_encoder),
            (self.high_value, self.target_high_value),
            (self.low_encoder, self.target_low_encoder),
            (self.goal_representation, self.target_goal_representation),
            (self.low_critic, self.target_low_critic),
        )
        for online, target in pairs:
            for online_param, target_param in zip(
                online.parameters(), target.parameters(), strict=True
            ):
                target_param.lerp_(online_param, tau)
            for online_buffer, target_buffer in zip(
                online.buffers(), target.buffers(), strict=True
            ):
                if torch.is_floating_point(target_buffer):
                    target_buffer.lerp_(online_buffer, tau)
                else:
                    target_buffer.copy_(online_buffer)

    def parameter_breakdown(self):
        """Return non-overlapping parameter counts for experiment logging."""
        names = (
            'high_encoder',
            'target_high_encoder',
            'low_encoder',
            'target_low_encoder',
            'goal_representation',
            'target_goal_representation',
            'high_value',
            'target_high_value',
            'low_value',
            'low_critic',
            'target_low_critic',
            'high_actor',
            'low_actor',
        )
        return {
            name: sum(p.numel() for p in getattr(self, name).parameters())
            for name in names
        }


__all__ = ['MLP', 'ActorMLP', 'QCHIQLChunkNew', 'TwinMLP']
