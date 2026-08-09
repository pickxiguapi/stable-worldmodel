"""Evaluate a feed-forward policy that predicts multi-step action chunks."""

import os


os.environ['MUJOCO_GL'] = 'egl'


import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


def load_chunk_model(name: str):
    """Rebuild and load a GCIQL-Chunk or GCHIQL-Chunk exported policy."""
    checkpoint_root = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    checkpoint_path = checkpoint_root / name
    config_path = checkpoint_path.parent / 'config.json'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Missing chunk policy: {checkpoint_path}')
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing chunk config: {config_path}')

    cfg = OmegaConf.load(config_path)
    run_name = checkpoint_path.parent.name
    if run_name.startswith('gciql_chunk_'):
        from scripts.train.gciql_chunk import (
            get_gciql_actor_model,
            get_gciql_critics_model,
        )

        critics_module = get_gciql_critics_model(cfg)
        module = get_gciql_actor_model(cfg, critics_module)
    elif run_name.startswith('gchiql_chunk_'):
        from scripts.train.gchiql import get_gchiql_chunk_model

        module = get_gchiql_chunk_model(cfg)
    else:
        raise ValueError(
            'eval_chunk.py only supports gciql_chunk_* and '
            f'gchiql_chunk_* exports, got {run_name!r}'
        )

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    module.model.load_state_dict(state_dict)
    return module.model, cfg


def img_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
        ]
    )


def episode_col(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def get_episode_lengths(dataset, episodes):
    column = episode_col(dataset)
    episode_ids = dataset.get_col_data(column)
    step_ids = dataset.get_col_data('step_idx')
    return np.asarray(
        [np.max(step_ids[episode_ids == episode]) + 1 for episode in episodes]
    )


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world, image_shape=(224, 224), render_mode='rgb_array'
    )
    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name,
        cache_dir=cfg.get('cache_dir', None),
    )

    actions = dataset.get_col_data('action')
    finite_actions = actions[~np.isnan(actions).any(axis=1)]
    action_process = preprocessing.StandardScaler().fit(finite_actions)
    process = {'action': action_process}

    model, checkpoint_cfg = load_chunk_model(cfg.policy)
    action_block = int(checkpoint_cfg.get('frameskip', 1))
    history_len = int(
        checkpoint_cfg.get('dinowm', {}).get('history_size', 1)
    )
    if action_block <= 1:
        raise ValueError(
            f'Checkpoint action block is {action_block}; use eval_ff.py '
            'for non-chunk policies.'
        )
    print(
        f'[eval_chunk] policy={cfg.policy} action_block={action_block} '
        f'history_len={history_len}'
    )

    model = model.to('cuda').eval()
    model.requires_grad_(False)
    policy = swm.policy.ChunkedFeedForwardPolicy(
        model=model,
        action_block=action_block,
        history_len=history_len,
        process=process,
        transform={'pixels': img_transform(), 'goal': img_transform()},
    )

    column = episode_col(dataset)
    episode_ids = dataset.get_col_data(column)
    episodes = np.unique(episode_ids)
    max_start = (
        get_episode_lengths(dataset, episodes)
        - cfg.eval.goal_offset_steps
        - 1
    )
    max_start_by_episode = dict(zip(episodes, max_start, strict=True))
    max_start_per_row = np.asarray(
        [max_start_by_episode[episode] for episode in episode_ids]
    )
    valid = np.nonzero(
        dataset.get_col_data('step_idx') <= max_start_per_row
    )[0]
    if len(valid) < cfg.eval.num_eval:
        raise ValueError(
            f'Only {len(valid)} valid starts for {cfg.eval.num_eval} evals.'
        )

    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(rng.choice(valid, size=cfg.eval.num_eval, replace=False))
    selected = dataset.get_row_data(rows)
    eval_episodes = selected[column]
    eval_starts = selected['step_idx']
    print(f'[eval_chunk] rows={rows.tolist()}')

    results_path = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
    ).parent
    results_path.mkdir(parents=True, exist_ok=True)
    world.set_policy(policy)
    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_starts.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True
        ),
        video=results_path,
    )
    elapsed = time.time() - started
    print(metrics)
    print(f'[eval_chunk] evaluation_time={elapsed:.1f}s')
    print(f'[eval_chunk] videos={results_path.resolve()}')

    output = results_path / cfg.output.filename
    with output.open('a') as file:
        file.write('\n==== CONFIG ====\n')
        file.write(OmegaConf.to_yaml(cfg))
        file.write('\n==== RESULTS ====\n')
        file.write(f'metrics: {metrics}\n')
        file.write(f'evaluation_time: {elapsed} seconds\n')
    world.close()


if __name__ == '__main__':
    run()
