from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

import stable_worldmodel as swm
from stable_worldmodel.wm.tdmpc2 import tdmpc2_forward
from stable_worldmodel.wm.utils import save_pretrained


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    Writes a state-dict ``weights_epoch_<n>.pt`` plus a ``config.json`` under
    ``$STABLEWM_HOME/checkpoints/<run_name>/`` (the format consumed by
    ``swm.wm.utils.load_pretrained``), replacing the old pickled
    ``*_object.ckpt`` dump. ``cfg`` is the instantiable model config
    (``cfg.model``), so ``config.json`` alone can rebuild the model.
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def get_column_normalizer(dataset, source, target):
    """Z-score normalization transform computed from the full dataset column."""
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = (
        data.mean(0, keepdim=True).clone(),
        data.std(0, keepdim=True).clone(),
    )
    mean, std = mean.squeeze(), std.squeeze() + 1e-2

    def norm_fn(x):
        return ((x - mean.to(x.device)) / std.to(x.device)).float()

    return spt.data.transforms.WrapTorchTransform(
        norm_fn, source=source, target=target
    )


def get_img_preprocessor(source, target, img_size=64, channels=3):
    """ImageNet-normalized + resized image preprocessing pipeline."""
    stats = dict(spt.data.dataset_stats.ImageNet)
    if channels % 3 != 0:
        raise ValueError(
            f'ImageNet normalization requires RGB groups, got {channels} channels'
        )
    repeats = channels // 3
    stats['mean'] = list(stats['mean']) * repeats
    stats['std'] = list(stats['std']) * repeats

    def channel_first_tensor(x):
        # Cached HDF5 columns are normally returned as tensors. Convert NumPy
        # inputs too so ToImage never has to guess whether a 6-channel array is
        # HWC or CHW.
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim >= 3 and x.shape[-1] == channels and x.shape[-3] != channels:
            x = x.movedim(-1, -3)
        return x

    return spt.data.transforms.Compose(
        spt.data.transforms.WrapTorchTransform(
            channel_first_tensor, source=source, target=target
        ),
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


def fill_pixel_episode_goals(
    augmented,
    raw_obs,
    episode_offsets,
    episode_lengths,
    goal_indices,
    source_channels,
    channel_last,
    episodes_per_chunk=256,
):
    """Fill the goal-channel half of a pixel cache in bounded chunks.

    A Python assignment per episode is prohibitively slow for multi-million
    frame datasets. Converted OGBench segments are contiguous, so repeat the
    one goal index per episode into modest row chunks and let NumPy perform
    each gather/copy in compiled code. The fallback preserves correctness for
    datasets whose episode rows are not contiguous.
    """
    contiguous = (
        len(episode_offsets) > 0
        and int(episode_offsets[0]) == 0
        and int(episode_offsets[-1] + episode_lengths[-1]) == len(raw_obs)
        and np.all(
            episode_offsets[1:]
            == episode_offsets[:-1] + episode_lengths[:-1]
        )
    )
    if not contiguous:
        for ep, (offset, length) in enumerate(
            zip(episode_offsets.tolist(), episode_lengths.tolist())
        ):
            goal = raw_obs[goal_indices[ep]]
            if channel_last:
                augmented[
                    offset : offset + length, ..., source_channels:
                ] = goal
            else:
                augmented[
                    offset : offset + length, source_channels:
                ] = goal
        return

    for ep_start in range(0, len(episode_offsets), episodes_per_chunk):
        ep_stop = min(ep_start + episodes_per_chunk, len(episode_offsets))
        row_start = int(episode_offsets[ep_start])
        row_stop = int(
            episode_offsets[ep_stop - 1] + episode_lengths[ep_stop - 1]
        )
        repeated_goal_indices = np.repeat(
            goal_indices[ep_start:ep_stop],
            episode_lengths[ep_start:ep_stop],
        )
        goals = raw_obs[repeated_goal_indices]
        if channel_last:
            augmented[row_start:row_stop, ..., source_channels:] = goals
        else:
            augmented[row_start:row_stop, source_channels:] = goals


@hydra.main(version_base=None, config_path='./config', config_name='tdmpc2')
def run(cfg):
    """
    Main training entry point for the TD-MPC2 model.

    Uses dataset rewards directly.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    torch.set_float32_matmul_precision('high')

    model_cfg = cfg.model.cfg  # the config TDMPC2 is instantiated from
    encoding_keys = list(model_cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        raise ValueError(
            'No encoding modalities defined in cfg.model.cfg.wm.encoding!'
        )

    use_pixels = 'pixels' in encoding_keys
    goal_obs_key = cfg.get(
        'goal_obs_key'
    )  # if set, concatenate episode goal into this key
    extra_keys = [k for k in encoding_keys if k != 'pixels']

    keys_to_load = list(encoding_keys) + ['action', 'reward']

    base_dataset = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.get('cache_dir'),
        num_steps=model_cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_load if cfg.get('cache_dataset', True) else [],
    )

    if goal_obs_key is not None:
        if goal_obs_key not in encoding_keys:
            raise ValueError(
                f'cfg.goal_obs_key="{goal_obs_key}" must be one of the encoding keys {encoding_keys}.'
            )
        _raw_obs = base_dataset.get_col_data(goal_obs_key)[:]
        _ep_off = (
            base_dataset.get_col_data('ep_offset')[:].flatten().astype(int)
        )
        _ep_len = base_dataset.get_col_data('ep_len')[:].flatten().astype(int)
        _goal_idx = np.clip(_ep_off + _ep_len - 1, 0, len(_raw_obs) - 1)
        if goal_obs_key == 'pixels':
            if _raw_obs.ndim != 4:
                raise ValueError(
                    'Pixel goal augmentation expects NCHW or NHWC images, '
                    f'got {_raw_obs.shape}'
                )
            if _raw_obs.shape[-1] in (1, 3, 4):
                source_channels = _raw_obs.shape[-1]
                augmented = np.empty(
                    (
                        len(_raw_obs),
                        _raw_obs.shape[1],
                        _raw_obs.shape[2],
                        2 * source_channels,
                    ),
                    dtype=_raw_obs.dtype,
                )
                augmented[..., :source_channels] = _raw_obs
                fill_pixel_episode_goals(
                    augmented,
                    _raw_obs,
                    _ep_off,
                    _ep_len,
                    _goal_idx,
                    source_channels,
                    channel_last=True,
                )
            elif _raw_obs.shape[1] in (1, 3, 4):
                source_channels = _raw_obs.shape[1]
                augmented = np.empty(
                    (
                        len(_raw_obs),
                        2 * source_channels,
                        _raw_obs.shape[2],
                        _raw_obs.shape[3],
                    ),
                    dtype=_raw_obs.dtype,
                )
                augmented[:, :source_channels] = _raw_obs
                fill_pixel_episode_goals(
                    augmented,
                    _raw_obs,
                    _ep_off,
                    _ep_len,
                    _goal_idx,
                    source_channels,
                    channel_last=False,
                )
            else:
                raise ValueError(
                    'Cannot identify the pixel channel axis in shape '
                    f'{_raw_obs.shape}'
                )
            augmented_channels = (
                augmented.shape[-1]
                if _raw_obs.shape[-1] in (1, 3, 4)
                else augmented.shape[1]
            )
            expected_channels = int(
                model_cfg.get('image_channels', augmented_channels)
            )
            if augmented_channels != expected_channels:
                raise ValueError(
                    f'Goal-concatenated pixels have {augmented_channels} '
                    f'channels, but model.cfg.image_channels={expected_channels}'
                )
            base_dataset._cache[goal_obs_key] = augmented
            dimension_summary = (
                f'channels {source_channels} → {augmented_channels}'
            )
        else:
            goals_by_step = np.empty_like(_raw_obs)
            for _ep, (_off, _len) in enumerate(
                zip(_ep_off.tolist(), _ep_len.tolist())
            ):
                goals_by_step[_off : _off + _len] = _raw_obs[_goal_idx[_ep]]
            base_dataset._cache[goal_obs_key] = np.concatenate(
                [_raw_obs, goals_by_step], axis=-1
            )
            dimension_summary = (
                f'dim {_raw_obs.shape[-1]} → '
                f'{base_dataset._cache[goal_obs_key].shape[-1]}'
            )
        logging.info(
            f'Goal augmentation: appended last obs of each episode to "{goal_obs_key}" '
            f'({dimension_summary})'
        )
        # The cache now owns the augmented array; release the raw source view
        # before training so large visual datasets do not stay resident twice.
        del _raw_obs
        if goal_obs_key != 'pixels':
            del goals_by_step

    raw_actions = base_dataset.get_col_data('action')[:]
    valid_actions = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    act_max = valid_actions.max()
    act_min = valid_actions.min()

    if act_max > 1.01 or act_min < -1.01:
        logging.error(
            f'Dataset actions fall outside the [-1, 1] range! (Min: {act_min:.2f}, Max: {act_max:.2f}).\n'
            'TD-MPC2 uses a Tanh actor and strictly requires actions to be bounded between [-1, 1].\n'
            'Please normalize your dataset actions.'
        )
        raise ValueError(
            'Unnormalized actions detected in the dataset. Training aborted.'
        )

    with open_dict(cfg):
        model_cfg.action_dim = base_dataset.get_dim('action')
        model_cfg.extra_dims = {'action': model_cfg.action_dim}

        for key in extra_keys:
            if goal_obs_key is not None and key == goal_obs_key:
                model_cfg.extra_dims[key] = base_dataset._cache[key].shape[-1]
            else:
                model_cfg.extra_dims[key] = base_dataset.get_dim(key)

    transforms = []
    if use_pixels:
        transforms.append(
            get_img_preprocessor(
                'pixels',
                'pixels',
                model_cfg.image_size,
                int(model_cfg.get('image_channels', 3)),
            )
        )

    for key in extra_keys:
        if goal_obs_key is not None and key == goal_obs_key:
            aug_data = torch.from_numpy(base_dataset._cache[key]).float()
            aug_clean = aug_data[~torch.isnan(aug_data).any(dim=1)]
            _mean = aug_clean.mean(0).clone()
            _std = aug_clean.std(0).clone() + 1e-2
            transforms.append(
                spt.data.transforms.WrapTorchTransform(
                    lambda x, m=_mean, s=_std: (
                        (x - m.to(x.device)) / s.to(x.device)
                    ).float(),
                    source=key,
                    target=key,
                )
            )
        else:
            transforms.append(get_column_normalizer(base_dataset, key, key))

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    train_set, val_set = spt.data.random_split(
        base_dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = hydra.utils.instantiate(cfg.model)

    def add_opt(module_regex, lr, eps=1e-8):
        opt_cfg = dict(cfg.optimizer)
        opt_cfg['lr'] = lr
        opt_cfg['eps'] = eps
        return {'modules': module_regex, 'optimizer': opt_cfg}

    module = spt.Module(
        model=model,
        forward=partial(tdmpc2_forward, cfg=model_cfg),
        hparams=OmegaConf.to_container(cfg, resolve=True),
        optim={
            'enc_opt': add_opt(
                r'model\.(cnn|pixel_encoder|extra_encoders|sim_norm).*',
                cfg.optimizer.lr * cfg.get('enc_lr_scale', 0.3),
            ),
            'wm_opt': add_opt(
                r'model\.(dynamics|reward|qs).*',
                cfg.optimizer.lr,
            ),
            'pi_opt': add_opt(
                r'model\.pi.*', cfg.optimizer.lr * 0.1, eps=1e-5
            ),
        },
    )
    subdir = cfg.subdir
    run_dir = Path(swm.data.utils.get_cache_dir(), subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name=f'{model_cfg.wm.name}_{cfg.dataset_name}_{subdir}',
            project=cfg.wandb.project,
            resume='allow' if subdir else None,
            id=subdir or None,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model)
        ],
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )()


if __name__ == '__main__':
    run()
