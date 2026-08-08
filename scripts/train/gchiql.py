import os
from collections import OrderedDict

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from einops import rearrange, repeat
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Set up HIQL value, low-level, and high-level goal supervision."""

    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
        )

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    logging.info(
        f'Loading dataset "{cfg.dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
    )

    use_proprio = cfg.dinowm.get('use_proprio_encoder', True)
    keys_to_load = ['pixels', 'action'] + (['proprio'] if use_proprio else [])
    keys_to_cache = ['action'] + (['proprio'] if use_proprio else [])
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
    )

    transforms = [
        get_img_pipeline('pixels', 'pixels', cfg.image_size),
        get_column_normalizer(dataset, 'action', 'action'),
    ]
    if use_proprio:
        transforms.append(get_column_normalizer(dataset, 'proprio', 'proprio'))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    goal_keys = {'pixels': 'goal_pixels'}
    low_goal_keys = {'pixels': 'low_goal_pixels'}
    high_goal_keys = {'pixels': 'high_goal_pixels'}
    high_target_keys = {'pixels': 'high_target_pixels'}
    if use_proprio:
        goal_keys['proprio'] = 'goal_proprio'
        low_goal_keys['proprio'] = 'low_goal_proprio'
        high_goal_keys['proprio'] = 'high_goal_proprio'
        high_target_keys['proprio'] = 'high_target_proprio'

    value_goal_probabilities = (
        cfg.goal_probabilities.random,
        cfg.goal_probabilities.geometric_future,
        cfg.goal_probabilities.uniform_future,
        cfg.goal_probabilities.current,
    )
    actor_goal_probabilities = (
        cfg.actor_goal_probabilities.random,
        cfg.actor_goal_probabilities.geometric_future,
        cfg.actor_goal_probabilities.uniform_future,
        cfg.actor_goal_probabilities.current,
    )
    dataset = swm.data.HierarchicalGoalDataset(
        dataset=dataset,
        goal_probabilities=value_goal_probabilities,
        actor_goal_probabilities=actor_goal_probabilities,
        gamma=cfg.goal_gamma,
        current_goal_offset=cfg.dinowm.history_size,
        subgoal_steps=cfg.subgoal_steps,
        goal_keys=goal_keys,
        low_goal_keys=low_goal_keys,
        high_goal_keys=high_goal_keys,
        high_target_keys=high_target_keys,
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train_subset_fraction = cfg.get('train_subset_fraction', 1.0)
    if train_subset_fraction < 1.0:
        train_set, _ = spt.data.random_split(
            train_set,
            lengths=[train_subset_fraction, 1 - train_subset_fraction],
            generator=rnd_gen,
        )

    logging.info(f'Train: {len(train_set)}, Val: {len(val_set)}')
    train = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Loss Helpers
# ============================================================================
def _flatten_tokens(embedding):
    return rearrange(embedding, 'b t p d -> b (t p) d')


def _gaussian_nll(target, mean, log_stds):
    """Gaussian NLL summed over the event dimension, up to a constant."""
    variance = torch.exp(2 * log_stds)
    return (log_stds + 0.5 * (target - mean).pow(2) / variance).sum(
        dim=-1, keepdim=True
    )


def _goal_reward_and_mask(
    batch,
    history_size,
    gc_negative=True,
    goal_prefix='',
):
    """Compute the goal-reaching reward from exact raw-observation equality."""
    obs_pixels = batch['pixels'][:, :history_size]
    goal_pixels = batch[f'{goal_prefix}goal_pixels']
    if goal_pixels.shape[1] == 1:
        goal_pixels = repeat(
            goal_pixels,
            'b 1 c h w -> b t c h w',
            t=history_size,
        )
    elif goal_pixels.shape[1] != history_size:
        raise ValueError(
            f'{goal_prefix}goal_pixels must have one or {history_size} '
            f'frames, got {goal_pixels.shape[1]}'
        )
    matches = (obs_pixels == goal_pixels).all(dim=(2, 3, 4))
    if 'proprio' in batch:
        obs_proprio = batch['proprio'][:, :history_size]
        goal_proprio = batch[f'{goal_prefix}goal_proprio']
        if goal_proprio.shape[1] == 1:
            goal_proprio = repeat(
                goal_proprio,
                'b 1 d -> b t d',
                t=history_size,
            )
        matches = matches & (obs_proprio == goal_proprio).all(dim=2)
    successes = matches.float().unsqueeze(-1)
    masks = 1.0 - successes
    rewards = successes - (1.0 if gc_negative else 0.0)
    return rewards, masks


def get_chunk_td_constants(cfg):
    """Return the raw-frame discount and sparse-reward sum for one chunk."""
    gamma = cfg.get('discount', 0.99)
    span = cfg.frameskip * cfg.dinowm.td_offset
    gamma_span = gamma**span
    reward_lump = (1.0 - gamma_span) / (1.0 - gamma)
    return gamma_span, reward_lump


# ============================================================================
# Model Architecture and Objective
# ============================================================================
def get_gchiql_model(cfg, chunked=False):
    """Build the joint GCHIQL value, high-actor, and low-actor model."""
    expectile_loss = swm.wm.gcrl.ExpectileLoss(tau=cfg.get('expectile', 0.7))
    gamma = cfg.get('discount', 0.99)
    history_size = cfg.dinowm.history_size
    td_offset = cfg.dinowm.td_offset
    if chunked:
        gamma, reward_lump = get_chunk_td_constants(cfg)
        if not cfg.get('gc_negative', True):
            raise ValueError('GCHIQL-Chunk requires gc_negative=true')

    def encode_view(self, batch, pixels_key, target, prefix):
        return self.model.encode(
            batch,
            pixels_key=pixels_key,
            target=target,
            prefix=prefix,
        )

    def forward(self, batch, stage):
        """Compute the twin-value and two advantage-weighted actor losses."""
        tensor_keys = (
            'action',
            'proprio',
            'goal_proprio',
            'low_goal_proprio',
            'high_goal_proprio',
            'high_target_proprio',
        )
        for key in tensor_keys:
            if key in batch:
                batch[key] = torch.nan_to_num(batch[key], 0.0)

        batch = encode_view(self, batch, 'pixels', 'embed', '')
        batch = encode_view(self, batch, 'goal_pixels', 'goal_embed', 'goal_')
        batch = encode_view(
            self, batch, 'low_goal_pixels', 'low_goal_embed', 'low_goal_'
        )
        batch = encode_view(
            self, batch, 'high_goal_pixels', 'high_goal_embed', 'high_goal_'
        )
        batch = encode_view(
            self,
            batch,
            'high_target_pixels',
            'high_target_embed',
            'high_target_',
        )

        embedding_keys = (
            'embed',
            'goal_embed',
            'low_goal_embed',
            'high_goal_embed',
            'high_target_embed',
        )
        if not encoder_trainable:
            for key in embedding_keys:
                batch[key] = batch[key].detach()

        embedding = batch['embed'][:, :history_size]
        next_embedding = batch['embed'][
            :, td_offset : td_offset + history_size
        ]
        goal_embedding = batch['goal_embed']
        low_goal_embedding = batch['low_goal_embed']
        high_goal_embedding = batch['high_goal_embed']
        high_target_embedding = batch['high_target_embed']

        embedding_flat = _flatten_tokens(embedding)
        next_embedding_flat = _flatten_tokens(next_embedding)
        goal_embedding_flat = _flatten_tokens(goal_embedding)
        low_goal_embedding_flat = _flatten_tokens(low_goal_embedding)
        high_goal_embedding_flat = _flatten_tokens(high_goal_embedding)
        high_target_embedding_flat = _flatten_tokens(high_target_embedding)

        rewards, masks = _goal_reward_and_mask(
            batch,
            history_size,
            gc_negative=cfg.get('gc_negative', True),
        )
        if chunked:
            rewards = -masks * reward_lump

        # HIQL value objective. The teacher advantage selects the expectile
        # side, while each online head regresses to its own teacher TD target.
        with torch.no_grad():
            next_v1_target, next_v2_target = (
                self.model.value_predictor.forward_teacher(
                    next_embedding_flat, goal_embedding_flat
                )
            )
            conservative_target = rewards + gamma * masks * torch.minimum(
                next_v1_target, next_v2_target
            )
            v1_target, v2_target = self.model.value_predictor.forward_teacher(
                embedding_flat, goal_embedding_flat
            )
            value_advantage = conservative_target - 0.5 * (
                v1_target + v2_target
            )
            q1_target = rewards + gamma * masks * next_v1_target
            q2_target = rewards + gamma * masks * next_v2_target

        v1, v2 = self.model.value_predictor.forward_student(
            embedding_flat, goal_embedding_flat
        )
        value_loss1 = expectile_loss(v1, q1_target, value_advantage)
        value_loss2 = expectile_loss(v2, q2_target, value_advantage)
        value_loss = value_loss1 + value_loss2

        target_actions = batch['action'][:, :history_size]
        critic_loss = torch.zeros((), device=embedding.device)
        low_value_loss = torch.zeros((), device=embedding.device)

        if chunked:
            # HiQC Eq. 6-7.  The low-level value is the expectile of the
            # chunk-conditioned critic, whose TD target skips the full action
            # chunk.  The shared HIQL value representation keeps the critic
            # and low-level policy conditioned on the same latent subgoal.
            low_rewards, low_masks = _goal_reward_and_mask(
                batch,
                history_size,
                gc_negative=True,
                goal_prefix='low_',
            )
            low_rewards = -low_masks * reward_lump
            with torch.no_grad():
                low_goal_reps_target = (
                    self.model.value_predictor.teacher.encode_goal(
                        embedding_flat, low_goal_embedding_flat
                    )
                )
                low_q_target = self.model.critic_predictor.forward_teacher(
                    embedding_flat,
                    target_actions,
                    low_goal_reps_target,
                )
                low_v1_target, low_v2_target = (
                    self.model.value_predictor.forward_teacher(
                        embedding_flat, low_goal_embedding_flat
                    )
                )
                low_value_advantage = low_q_target - 0.5 * (
                    low_v1_target + low_v2_target
                )
                next_low_v1_target, next_low_v2_target = (
                    self.model.value_predictor.forward_teacher(
                        next_embedding_flat, low_goal_embedding_flat
                    )
                )
                low_q_td_target = (
                    low_rewards
                    + gamma
                    * low_masks
                    * torch.minimum(next_low_v1_target, next_low_v2_target)
                )

            low_v1, low_v2 = self.model.value_predictor.forward_student(
                embedding_flat, low_goal_embedding_flat
            )
            low_value_loss = expectile_loss(
                low_v1, low_q_target, low_value_advantage
            ) + expectile_loss(low_v2, low_q_target, low_value_advantage)
            low_goal_reps_for_q = (
                self.model.value_predictor.student.encode_goal(
                    embedding_flat, low_goal_embedding_flat
                )
            )
            low_q_pred = self.model.critic_predictor.forward_student(
                embedding_flat,
                target_actions,
                low_goal_reps_for_q.detach(),
            )
            critic_loss = (low_q_pred - low_q_td_target).pow(2).mean()
            value_loss = value_loss + low_value_loss

        # The high-level actor remains value-difference weighted (HiQC Eq. 5).
        # Only the low-level actor switches to the chunk Q-V advantage.
        with torch.no_grad():
            low_v1, low_v2 = self.model.value_predictor.forward_student(
                embedding_flat, low_goal_embedding_flat
            )
            if chunked:
                low_goal_reps = self.model.value_predictor.student.encode_goal(
                    embedding_flat, low_goal_embedding_flat
                )
                low_q = self.model.critic_predictor.forward_student(
                    embedding_flat, target_actions, low_goal_reps
                )
                low_advantage = low_q - 0.5 * (low_v1 + low_v2)
            else:
                low_next_v1, low_next_v2 = (
                    self.model.value_predictor.forward_student(
                        next_embedding_flat, low_goal_embedding_flat
                    )
                )
                low_advantage = 0.5 * (
                    low_next_v1 + low_next_v2 - low_v1 - low_v2
                )

            high_v1, high_v2 = self.model.value_predictor.forward_student(
                embedding_flat, high_goal_embedding_flat
            )
            high_next_v1, high_next_v2 = (
                self.model.value_predictor.forward_student(
                    high_target_embedding_flat, high_goal_embedding_flat
                )
            )
            high_advantage = 0.5 * (
                high_next_v1 + high_next_v2 - high_v1 - high_v2
            )
            high_targets = self.model.value_predictor.student.encode_goal(
                embedding_flat, high_target_embedding_flat
            )

        if cfg.get('low_actor_rep_grad', False):
            low_goal_representations = (
                self.model.value_predictor.student.encode_goal(
                    embedding_flat, low_goal_embedding_flat
                )
            )
        else:
            with torch.no_grad():
                low_goal_representations = (
                    self.model.value_predictor.student.encode_goal(
                        embedding_flat, low_goal_embedding_flat
                    )
                )

        low_means, _ = self.model.predict_low_actions(
            embedding, low_goal_representations
        )
        low_log_stds = torch.clamp(
            self.model.log_stds,
            self.model.log_std_min,
            self.model.log_std_max,
        )
        low_nll = _gaussian_nll(target_actions, low_means, low_log_stds)
        low_weights = torch.exp(
            cfg.get('low_alpha', 3.0) * low_advantage
        ).clamp(max=100.0)
        low_actor_loss = (low_weights * low_nll).mean()

        high_means, _ = self.model.predict_high_subgoals(
            embedding, high_goal_embedding
        )
        high_log_stds = torch.clamp(
            self.model.high_log_stds,
            self.model.log_std_min,
            self.model.log_std_max,
        )
        high_nll = _gaussian_nll(high_targets, high_means, high_log_stds)
        high_weights = torch.exp(
            cfg.get('high_alpha', 3.0) * high_advantage
        ).clamp(max=100.0)
        high_actor_loss = (high_weights * high_nll).mean()

        total_loss = (
            value_loss + critic_loss + low_actor_loss + high_actor_loss
        )
        batch['value_loss'] = value_loss
        if chunked:
            batch['low_value_loss'] = low_value_loss
            batch['critic_loss'] = critic_loss
        batch['low_actor_loss'] = low_actor_loss
        batch['high_actor_loss'] = high_actor_loss
        batch['loss'] = total_loss

        prefix = 'train/' if self.training else 'val/'
        diagnostics = {
            f'{prefix}loss': total_loss.detach(),
            f'{prefix}value_loss': value_loss.detach(),
            f'{prefix}low_value_loss': low_value_loss.detach(),
            f'{prefix}critic_loss': critic_loss.detach(),
            f'{prefix}low_actor_loss': low_actor_loss.detach(),
            f'{prefix}high_actor_loss': high_actor_loss.detach(),
            f'{prefix}value_mean': (0.5 * (v1 + v2)).mean().detach(),
            f'{prefix}value_advantage_mean': value_advantage.mean(),
            f'{prefix}low_advantage_mean': low_advantage.mean(),
            f'{prefix}high_advantage_mean': high_advantage.mean(),
            f'{prefix}low_weight_mean': low_weights.mean(),
            f'{prefix}high_weight_mean': high_weights.mean(),
            f'{prefix}goal_match_rate': (1.0 - masks).mean(),
            f'{prefix}goal_rep_norm': high_targets.norm(dim=-1).mean(),
        }
        self.log_dict(diagnostics, on_step=True, sync_dist=True)
        return batch

    encoder_type = cfg.get('encoder_type', 'dino')
    if encoder_type == 'dino':
        encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        embedding_dim = encoder.config.hidden_size
        encoder_trainable = False
    elif encoder_type == 'vit_tiny':
        encoder = spt.backbone.utils.vit_hf(
            'tiny',
            patch_size=cfg.patch_size,
            image_size=cfg.image_size,
            pretrained=False,
            use_mask_token=False,
        )
        embedding_dim = encoder.config.hidden_size
        encoder_trainable = True
    else:
        raise ValueError(f'Unknown encoder_type: {encoder_type}')

    if td_offset != 1:
        raise ValueError('GCHIQL requires dinowm.td_offset=1')
    if cfg.image_size % cfg.patch_size != 0:
        raise ValueError('image_size must be divisible by patch_size')
    num_patches = (cfg.image_size // cfg.patch_size) ** 2

    extra_encoders = None
    if cfg.dinowm.get('use_proprio_encoder', True):
        extra_encoders = OrderedDict()
        extra_encoders['proprio'] = swm.wm.gcrl.Embedder(
            in_chans=cfg.dinowm.proprio_dim,
            emb_dim=cfg.dinowm.proprio_embed_dim,
        )
        extra_encoders = torch.nn.ModuleDict(extra_encoders)
        embedding_dim += cfg.dinowm.proprio_embed_dim

    predictor_kwargs = dict(
        num_patches=num_patches,
        num_frames=history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )
    value_predictor = swm.wm.gcrl.HierarchicalValuePredictor(
        rep_dim=cfg.rep_dim,
        **predictor_kwargs,
    )
    wrapped_value_predictor = spt.TeacherStudentWrapper(
        value_predictor,
        warm_init=True,
        base_ema_coefficient=cfg.get('value_ema_tau', 0.995),
        final_ema_coefficient=cfg.get('value_ema_tau', 0.995),
    )
    effective_action_dim = cfg.frameskip * cfg.dinowm.action_dim
    low_action_predictor = swm.wm.gcrl.RepresentationPredictor(
        out_dim=effective_action_dim,
        rep_dim=cfg.rep_dim,
        **predictor_kwargs,
    )
    high_action_predictor = swm.wm.gcrl.GoalRepresentationPredictor(
        rep_dim=cfg.rep_dim,
        normalize=False,
        **predictor_kwargs,
    )
    wrapped_critic_predictor = None
    if chunked:
        critic_predictor = swm.wm.gcrl.RepresentationQPredictor(
            action_dim=effective_action_dim,
            rep_dim=cfg.rep_dim,
            **predictor_kwargs,
        )
        wrapped_critic_predictor = spt.TeacherStudentWrapper(
            critic_predictor,
            warm_init=True,
            base_ema_coefficient=cfg.get('value_ema_tau', 0.995),
            final_ema_coefficient=cfg.get('value_ema_tau', 0.995),
        )

    wrapped_encoder = (
        spt.backbone.EvalOnly(encoder) if not encoder_trainable else encoder
    )
    model = swm.wm.gcrl.GCHIQL(
        encoder=wrapped_encoder,
        low_action_predictor=low_action_predictor,
        high_action_predictor=high_action_predictor,
        value_predictor=wrapped_value_predictor,
        critic_predictor=wrapped_critic_predictor,
        extra_encoders=extra_encoders,
        history_size=history_size,
    )

    def add_opt(module_name, lr):
        return {
            'modules': str(module_name),
            'optimizer': {'type': 'AdamW', 'lr': lr},
        }

    optim_config = {
        'value_predictor_opt': add_opt(
            'model.value_predictor', cfg.predictor_lr
        ),
        'low_action_predictor_opt': add_opt(
            'model.action_predictor', cfg.predictor_lr
        ),
        'high_action_predictor_opt': add_opt(
            'model.high_action_predictor', cfg.predictor_lr
        ),
    }
    if chunked:
        optim_config['critic_predictor_opt'] = add_opt(
            'model.critic_predictor', cfg.predictor_lr
        )
    if extra_encoders is not None:
        optim_config['proprio_opt'] = add_opt(
            'model.extra_encoders.proprio', cfg.proprio_encoder_lr
        )
    if encoder_trainable:
        optim_config['encoder_opt'] = add_opt(
            'model.encoder', cfg.get('encoder_lr', 3e-4)
        )
    return spt.Module(model=model, forward=forward, optim=optim_config)


def get_gchiql_chunk_model(cfg):
    """Build GCHIQL with the HiQC low-level Q-chunking objective."""
    return get_gchiql_model(cfg, chunked=True)


# ============================================================================
# Training Setup
# ============================================================================
class SaveCkptCallback(Callback):
    """Save an inference-friendly model checkpoint at a fixed epoch interval."""

    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
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
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    return logger


@hydra.main(version_base=None, config_path='./config', config_name='gchiql')
def run(cfg):
    """Train GCHIQL jointly from offline goal-conditioned trajectories."""
    data = get_data(cfg)
    model = get_gchiql_model(cfg)
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
