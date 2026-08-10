"""Train compact QCHIQL-Chunk with separate high/low ViT-Tiny encoders."""

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf

import stable_worldmodel as swm
from scripts.train.gchiql import (
    _gaussian_nll,
    _goal_reward_and_mask,
    get_chunk_td_constants,
)
from scripts.train.gchiql import get_data as get_hierarchical_data
from stable_worldmodel.wm.utils import save_pretrained


def _get_subgoal_steps(cfg):
    """Return ``(atomic_steps, sampled_steps)`` with legacy compatibility."""
    if 'subgoal_horizon' in cfg:
        # Legacy QCHIQL configs exposed raw steps as subgoal_horizon and used
        # subgoal_steps for the already-downsampled dataset distance.
        atomic_steps = int(cfg.subgoal_horizon)
        sampled_steps = int(cfg.subgoal_steps)
    else:
        # Match OGBench HIQL: subgoal_steps is measured in atomic environment
        # transitions.  Stable-WM samples one transition every frameskip.
        atomic_steps = int(cfg.subgoal_steps)
        if atomic_steps <= 0 or atomic_steps % cfg.frameskip != 0:
            raise ValueError(
                'subgoal_steps must be a positive multiple of frameskip, got '
                f'{atomic_steps} and {cfg.frameskip}'
            )
        sampled_steps = atomic_steps // cfg.frameskip
    if atomic_steps <= 0 or atomic_steps % cfg.frameskip != 0:
        raise ValueError(
            'subgoal_steps must be a positive multiple of frameskip, got '
            f'{atomic_steps} and {cfg.frameskip}'
        )
    if sampled_steps != atomic_steps // cfg.frameskip:
        raise ValueError(
            'Legacy sampled subgoal_steps must equal '
            'subgoal_horizon // frameskip, got '
            f'{sampled_steps} versus {atomic_steps // cfg.frameskip}'
        )
    return atomic_steps, sampled_steps


def _goal_probabilities(cfg, prefix):
    """Convert OGBench goal fields to Stable-WM's four-way sampler."""
    current_key = f'{prefix}_p_curgoal'
    if current_key in cfg:
        current = float(cfg[current_key])
        trajectory = float(cfg[f'{prefix}_p_trajgoal'])
        random = float(cfg[f'{prefix}_p_randomgoal'])
        geometric = bool(cfg[f'{prefix}_geom_sample'])
        return (
            random,
            trajectory if geometric else 0.0,
            0.0 if geometric else trajectory,
            current,
        )

    # Checkpoints produced before the OGBench naming alignment.
    legacy_key = (
        'goal_probabilities'
        if prefix == 'value'
        else 'actor_goal_probabilities'
    )
    legacy = cfg[legacy_key]
    return (
        float(legacy.random),
        float(legacy.geometric_future),
        float(legacy.uniform_future),
        float(legacy.current),
    )


def get_data(cfg):
    """Build hierarchical data using OGBench-compatible public names."""
    _, sampled_subgoal_steps = _get_subgoal_steps(cfg)
    value_probs = _goal_probabilities(cfg, 'value')
    actor_probs = _goal_probabilities(cfg, 'actor')
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.subgoal_steps = sampled_subgoal_steps
    data_cfg.goal_gamma = float(
        cfg.get('goal_gamma', cfg.discount**cfg.frameskip)
    )
    data_cfg.goal_probabilities = {
        'random': value_probs[0],
        'geometric_future': value_probs[1],
        'uniform_future': value_probs[2],
        'current': value_probs[3],
    }
    data_cfg.actor_goal_probabilities = {
        'random': actor_probs[0],
        'geometric_future': actor_probs[1],
        'uniform_future': actor_probs[2],
        'current': actor_probs[3],
    }
    return get_hierarchical_data(data_cfg)


def build_qchiql_chunk_new(cfg):
    """Build the five compact heads and their required OGBench targets."""
    encoder = cfg.get('encoder', cfg.get('encoder_type', 'vit_tiny'))
    if encoder != 'vit_tiny':
        raise ValueError('QCHIQL-Chunk-New requires encoder=vit_tiny')
    if not cfg.get('const_std', True):
        raise ValueError('QCHIQL-Chunk-New currently requires const_std=true')
    if cfg.get('discrete', False):
        raise ValueError('QCHIQL-Chunk-New currently requires discrete=false')
    if cfg.dinowm.get('use_proprio_encoder', False):
        raise ValueError(
            'QCHIQL-Chunk-New currently expects pixel-only inputs; set '
            'dinowm.use_proprio_encoder=false'
        )

    def make_encoder():
        return spt.backbone.utils.vit_hf(
            'tiny',
            patch_size=cfg.patch_size,
            image_size=cfg.image_size,
            pretrained=False,
            use_mask_token=False,
        )

    high_encoder = make_encoder()
    low_encoder = make_encoder()
    feature_dim = high_encoder.config.hidden_size
    if low_encoder.config.hidden_size != feature_dim:
        raise ValueError(
            'High- and low-level encoder feature sizes must match'
        )

    legacy_network = cfg.get('network', {})
    value_hidden_dims = tuple(
        cfg.get(
            'value_hidden_dims',
            legacy_network.get('hidden_dims', (512, 512, 512)),
        )
    )
    actor_hidden_dims = tuple(
        cfg.get(
            'actor_hidden_dims',
            legacy_network.get('hidden_dims', (512, 512, 512)),
        )
    )
    goal_rep_hidden_dims = tuple(
        cfg.get(
            'value_hidden_dims',
            legacy_network.get('rep_hidden_dims', value_hidden_dims),
        )
    )
    model = swm.wm.gcrl.QCHIQLChunkNew(
        high_encoder=high_encoder,
        low_encoder=low_encoder,
        feature_dim=feature_dim,
        action_dim=cfg.frameskip * cfg.dinowm.action_dim,
        rep_dim=cfg.rep_dim,
        value_hidden_dims=value_hidden_dims,
        actor_hidden_dims=actor_hidden_dims,
        goal_rep_hidden_dims=goal_rep_hidden_dims,
        layer_norm=cfg.get('layer_norm', True),
    )
    breakdown = model.parameter_breakdown()
    total = sum(p.numel() for p in model.parameters())
    logging.info(
        'QCHIQL-Chunk-New parameters: '
        + ', '.join(f'{name}={count:,}' for name, count in breakdown.items())
        + f', total={total:,}'
    )
    return model


def _awr_weights(advantages, alpha):
    """Stable-WM/OGBench AWR weights: exp(alpha * advantage), capped."""
    return torch.exp(alpha * advantages).clamp(max=100.0)


def get_qchiql_chunk_new_model(cfg):
    """Wrap the compact network in the Stable Pretraining trainer module."""
    history_size = cfg.dinowm.history_size
    td_offset = cfg.dinowm.td_offset
    high_horizon, _ = _get_subgoal_steps(cfg)
    if td_offset != 1:
        raise ValueError('QCHIQL-Chunk-New requires dinowm.td_offset=1')
    if not cfg.get('gc_negative', True):
        raise ValueError('QCHIQL-Chunk-New requires gc_negative=true')
    if cfg.n_steps != history_size + td_offset:
        raise ValueError(
            'n_steps must equal dinowm.history_size + dinowm.td_offset, got '
            f'{cfg.n_steps} versus {history_size + td_offset}'
        )
    model = build_qchiql_chunk_new(cfg)
    expectile_loss = swm.wm.gcrl.ExpectileLoss(tau=cfg.get('expectile', 0.7))
    low_gamma, _ = get_chunk_td_constants(cfg)
    high_gamma = cfg.get('discount', 0.99) ** high_horizon

    def forward(self, batch, stage):
        if 'action' in batch:
            batch['action'] = torch.nan_to_num(batch['action'], 0.0)

        # Encode each trainable branch in one batched call.  This preserves
        # high/low separation while avoiding repeated ViT launch overhead.
        obs_frames = batch['pixels'].shape[1]
        value_goal_frames = batch['goal_pixels'].shape[1]
        actor_goal_frames = batch['high_goal_pixels'].shape[1]
        low_goal_frames = batch['low_goal_pixels'].shape[1]
        high_encoded = self.model.encode_high(
            torch.cat(
                [
                    batch['pixels'],
                    batch['goal_pixels'],
                    batch['high_goal_pixels'],
                    batch['low_goal_pixels'],
                ],
                dim=1,
            )
        )
        high_all = high_encoded[:, :obs_frames]
        value_goal = high_encoded[
            :, obs_frames : obs_frames + value_goal_frames
        ]
        actor_goal = high_encoded[
            :,
            obs_frames + value_goal_frames : obs_frames
            + value_goal_frames
            + actor_goal_frames,
        ]
        high_low_goal = high_encoded[
            :,
            obs_frames + value_goal_frames + actor_goal_frames : obs_frames
            + value_goal_frames
            + actor_goal_frames
            + low_goal_frames,
        ]
        low_all = self.model.encode_low(batch['pixels'])

        with torch.no_grad():
            # Actor targets are labels; their visual paths must not retain
            # activation graphs or update either branch encoder.
            high_target = self.model.encode_high(batch['high_target_pixels'])
            target_high_encoded = self.model.encode_target_high(
                torch.cat(
                    [
                        batch['pixels'],
                        batch['goal_pixels'],
                        batch['low_goal_pixels'],
                    ],
                    dim=1,
                )
            )
            target_high_all = target_high_encoded[:, :obs_frames]
            target_value_goal = target_high_encoded[
                :, obs_frames : obs_frames + value_goal_frames
            ]
            # HierarchicalGoalDataset.low_goal is the uncluttered s_{t+c}
            # sequence. high_target is actor supervision and may be clipped
            # by a nearer sampled actor goal, so it must not enter Eq. 4.
            target_high_future = target_high_encoded[
                :,
                obs_frames + value_goal_frames : obs_frames
                + value_goal_frames
                + low_goal_frames,
            ]
            target_low_all = self.model.encode_target_low(batch['pixels'])

        high_states = high_all[:, :history_size]
        next_high_states = high_all[
            :, td_offset : td_offset + history_size
        ]
        low_states = low_all[:, :history_size]
        next_low_states = low_all[:, td_offset : td_offset + history_size]
        action_chunks = batch['action'][:, :history_size]
        target_high_states = target_high_all[:, :history_size]
        target_low_states = target_low_all[:, :history_size]

        rewards, masks = _goal_reward_and_mask(
            batch,
            history_size,
            gc_negative=True,
        )

        # OGBench HIQL high-level value loss: twin online V_H and an EMA
        # target twin.  HiQC Eq. 4 bootstraps directly from s_t to s_{t+c}.
        high_v1, high_v2 = self.model.predict_high_value(
            high_states,
            value_goal,
        )
        with torch.no_grad():
            next_high_v1_target, next_high_v2_target = (
                self.model.predict_target_high_value(
                    target_high_future,
                    target_value_goal,
                )
            )
            high_q = rewards + high_gamma * masks * torch.minimum(
                next_high_v1_target,
                next_high_v2_target,
            )
            high_v1_target, high_v2_target = (
                self.model.predict_target_high_value(
                    target_high_states,
                    target_value_goal,
                )
            )
            high_advantages = high_q - 0.5 * (high_v1_target + high_v2_target)
            high_q1_target = rewards + high_gamma * masks * next_high_v1_target
            high_q2_target = rewards + high_gamma * masks * next_high_v2_target
        high_value_loss = expectile_loss(
            high_v1,
            high_q1_target,
            high_advantages,
        ) + expectile_loss(
            high_v2,
            high_q2_target,
            high_advantages,
        )

        # Low-level chunked IQL: V_L is the expectile of Q_L and Q_L receives
        # a k-step TD target.  All three heads are compact MLPs.
        low_goal_reps = self.model.represent_goal(
            high_states,
            high_low_goal,
        )
        low_values = self.model.predict_low_value(low_states, low_goal_reps)
        low_q1, low_q2 = self.model.predict_low_q(
            low_states,
            low_goal_reps,
            action_chunks,
        )
        low_rewards, low_masks = _goal_reward_and_mask(
            batch,
            history_size,
            gc_negative=True,
            goal_prefix='low_',
        )
        with torch.no_grad():
            target_low_goal_reps = self.model.represent_target_goal(
                target_high_states,
                target_high_future,
            )
            target_low_q1, target_low_q2 = self.model.predict_target_low_q(
                target_low_states,
                target_low_goal_reps,
                action_chunks,
            )
            target_low_q = torch.minimum(target_low_q1, target_low_q2)
            low_value_advantages = target_low_q - low_values.detach()

            next_low_goal_reps = self.model.represent_goal(
                next_high_states,
                high_low_goal,
            )
            next_low_values = self.model.predict_low_value(
                next_low_states,
                next_low_goal_reps,
            )
            low_q_targets = (
                low_rewards + low_gamma * low_masks * next_low_values
            )
            low_actor_advantages = (
                torch.minimum(
                    low_q1.detach(),
                    low_q2.detach(),
                )
                - low_values.detach()
            )

        low_q_loss = (
            (low_q1 - low_q_targets).pow(2) + (low_q2 - low_q_targets).pow(2)
        ).mean()
        low_value_loss = expectile_loss(
            low_values,
            target_low_q,
            low_value_advantages,
        )

        # Low actor: advantage-weighted Gaussian regression, not flow matching.
        actor_goal_reps = (
            low_goal_reps
            if cfg.get('low_actor_rep_grad', True)
            else low_goal_reps.detach()
        )
        low_means, _ = self.model.predict_low_actions(
            low_states,
            actor_goal_reps,
        )
        low_nll = _gaussian_nll(
            action_chunks,
            low_means,
            self.model.low_log_stds,
        )
        low_weights = _awr_weights(
            low_actor_advantages,
            cfg.get('low_alpha', 3.0),
        )
        low_actor_loss = (low_weights * low_nll).mean()

        # High actor predicts phi(s, s_{t+c}) conditioned on its independently
        # sampled trajectory goal.  Value relabeling uses ``goal_pixels``;
        # actor supervision and its advantage must both use ``high_goal_pixels``.
        with torch.no_grad():
            target_subgoals = self.model.represent_goal(
                high_states,
                high_target,
            )
            target_high_v1, target_high_v2 = self.model.predict_high_value(
                high_target,
                actor_goal,
            )
            current_actor_v1, current_actor_v2 = (
                self.model.predict_high_value(
                    high_states,
                    actor_goal,
                )
            )
            high_actor_advantages = 0.5 * (
                target_high_v1 + target_high_v2
            ) - 0.5 * (current_actor_v1 + current_actor_v2)
        high_means, _ = self.model.predict_high_subgoals(
            high_states,
            actor_goal,
        )
        high_nll = _gaussian_nll(
            target_subgoals,
            high_means,
            self.model.high_log_stds,
        )
        high_weights = _awr_weights(
            high_actor_advantages,
            cfg.get('high_alpha', 3.0),
        )
        high_actor_loss = (high_weights * high_nll).mean()

        # Collapse diagnostics for the learned hierarchical latent.  A
        # near-zero feature std / pair distance together with unit AWR
        # weights indicates that the high-level policy has degenerated into
        # unconditional behavioral cloning even when the scalar losses stay
        # finite.
        flat_subgoals = target_subgoals.detach().reshape(
            -1, target_subgoals.shape[-1]
        )
        high_rep_feature_std = flat_subgoals.std(
            dim=0, unbiased=False
        ).mean()
        if flat_subgoals.shape[0] > 1:
            high_rep_pair_distance = (
                flat_subgoals[1:] - flat_subgoals[:-1]
            ).norm(dim=-1).mean()
        else:
            high_rep_pair_distance = flat_subgoals.new_zeros(())

        total_loss = (
            high_value_loss
            + low_value_loss
            + low_q_loss
            + high_actor_loss
            + low_actor_loss
        )
        batch['high_value_loss'] = high_value_loss
        batch['low_value_loss'] = low_value_loss
        batch['low_q_loss'] = low_q_loss
        batch['high_actor_loss'] = high_actor_loss
        batch['low_actor_loss'] = low_actor_loss
        batch['loss'] = total_loss

        prefix = 'train/' if self.training else 'val/'
        self.log_dict(
            {
                f'{prefix}loss': total_loss.detach(),
                f'{prefix}high_value_loss': high_value_loss.detach(),
                f'{prefix}low_value_loss': low_value_loss.detach(),
                f'{prefix}low_q_loss': low_q_loss.detach(),
                f'{prefix}high_actor_loss': high_actor_loss.detach(),
                f'{prefix}low_actor_loss': low_actor_loss.detach(),
                f'{prefix}high_value_mean': (0.5 * (high_v1 + high_v2))
                .mean()
                .detach(),
                f'{prefix}low_value_mean': low_values.mean().detach(),
                f'{prefix}low_q_mean': (
                    torch.minimum(low_q1, low_q2).mean().detach()
                ),
                f'{prefix}high_advantage_mean': (
                    high_actor_advantages.mean().detach()
                ),
                f'{prefix}low_advantage_mean': (
                    low_actor_advantages.mean().detach()
                ),
                f'{prefix}high_weight_mean': high_weights.mean().detach(),
                f'{prefix}low_weight_mean': low_weights.mean().detach(),
                f'{prefix}high_weight_max': high_weights.max().detach(),
                f'{prefix}low_weight_max': low_weights.max().detach(),
                f'{prefix}high_weight_saturation_rate': (
                    (high_weights >= 100.0).float().mean().detach()
                ),
                f'{prefix}low_weight_saturation_rate': (
                    (low_weights >= 100.0).float().mean().detach()
                ),
                f'{prefix}high_advantage_std': high_actor_advantages.std(
                    unbiased=False
                ).detach(),
                f'{prefix}low_advantage_std': low_actor_advantages.std(
                    unbiased=False
                ).detach(),
                f'{prefix}high_rep_feature_std': high_rep_feature_std,
                f'{prefix}high_rep_pair_distance': high_rep_pair_distance,
                f'{prefix}high_target_norm': target_subgoals.norm(
                    dim=-1
                ).mean().detach(),
                f'{prefix}high_prediction_norm': high_means.norm(
                    dim=-1
                ).mean().detach(),
                f'{prefix}goal_match_rate': (1.0 - masks).mean().detach(),
            },
            on_step=True,
            sync_dist=True,
            batch_size=batch['pixels'].shape[0],
        )
        return batch

    optim_config = {
        'model_opt': {
            'modules': 'model',
            'optimizer': {
                'type': 'Adam',
                'lr': cfg.lr,
            },
        }
    }
    return QCHIQLTrainingModule(
        model=model,
        forward=forward,
        optim=optim_config,
        target_tau=cfg.get('tau', 0.005),
    )


class QCHIQLTrainingModule(spt.Module):
    """Stable-pretraining module that updates EMA targets after Adam."""

    def __init__(self, target_tau=0.005, **kwargs):
        super().__init__(**kwargs)
        self.target_tau = target_tau

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
    ):
        """Step Adam first, then update targets exactly once per real step."""
        output = super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )
        self.model.update_targets(self.target_tau)
        return output


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if trainer.is_global_zero and (
            epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        ):
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.cfg,
                filename=f'weights_epoch_{epoch}.pt',
            )


def setup_pl_logger(cfg):
    if not cfg.wandb.enabled:
        return None
    logger = WandbLogger(**cfg.wandb.config)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return logger


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='qchiql_chunk_new',
)
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    data = get_data(cfg)
    model = get_qchiql_chunk_new_model(cfg)
    callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=cfg.get('checkpoint_interval', 3),
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[callback],
        num_sanity_val_steps=1,
        logger=setup_pl_logger(cfg),
        enable_checkpointing=True,
    )
    cache_dir = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    checkpoint_path = cache_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        ckpt_path=checkpoint_path if checkpoint_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
