"""Goal-conditioned Latent Diffusion Planning on OGBench pixels.

This is a deliberately small adaptation layer around the original LDP network
definitions.  The planner is conditioned on both the current and goal latent;
the inverse-dynamics model remains conditioned only on adjacent latent states.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if external_site := os.environ.get('OGBENCH_SITE_PACKAGES'):
    # Append so this venv's pinned numerical stack always wins.
    sys.path.append(external_site)

from scripts.data.ldp_ogbench_data import OGBenchLDPData, h5_take


UPSTREAM_COMMIT = 'a26cbf1d2c0aec7adc5d9746f47831b162a41c0c'
VAE_ARCH = {
    'act_fn': 'silu',
    'block_out_channels': [64, 128, 256, 256],
    'down_block_types': ['DownEncoderBlock2D'] * 4,
    'in_channels': 3,
    'latent_channels': 4,
    'layers_per_block': 2,
    'norm_num_groups': 32,
    'out_channels': 3,
    'sample_size': 64,
    'scaling_factor': 0.18215,
    'up_block_types': ['UpDecoderBlock2D'] * 4,
}


def repo_root() -> Path:
    return PROJECT_ROOT


def upstream_root() -> Path:
    return repo_root() / 'third_party' / 'latent_diffusion_planning'


def verify_upstream() -> str:
    root = upstream_root()
    if not (root / 'networks' / 'diffusion_nets_v2.py').is_file():
        raise RuntimeError(f'LDP submodule is not initialized: {root}')
    commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f'Expected upstream LDP {UPSTREAM_COMMIT}, found {commit}'
        )
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain'], text=True
    ).strip()
    if dirty:
        raise RuntimeError('The upstream LDP submodule must remain unmodified')
    return commit


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')


def create_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_msgpack(path: Path, tree: Any) -> None:
    from flax import serialization

    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(serialization.to_bytes(tree))
    temporary.replace(path)


def load_msgpack(path: Path) -> Any:
    from flax import serialization

    return serialization.msgpack_restore(path.read_bytes())


def make_vae():
    from diffusers import FlaxAutoencoderKL

    return FlaxAutoencoderKL(**VAE_ARCH)


def image_batch(source: h5py.File, rows: np.ndarray) -> np.ndarray:
    pixels = h5_take(source['pixels'], rows)
    return (pixels.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1.0)


def audit(args: argparse.Namespace) -> None:
    commit = verify_upstream()
    data = OGBenchLDPData(args.source, args.latents)
    result = {
        'upstream_commit': commit,
        'source': str(data.source_path),
        'source_size_bytes': data.source_path.stat().st_size,
        **data.summary.__dict__,
    }
    if args.latents:
        result.update(
            latent_path=str(data.latent_path),
            latent_dim=data.latent_dim,
            latent_min=data.latent_min,
            latent_max=data.latent_max,
        )
    print('AUDIT_JSON=' + json.dumps(result, sort_keys=True), flush=True)


def train_vae(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import optax

    commit = verify_upstream()
    data = OGBenchLDPData(args.source)
    output = create_output_dir(args.output_dir)
    config = {
        'kind': 'ogbench_ldp_vae',
        'upstream_commit': commit,
        'source': str(data.source_path),
        'source_size_bytes': data.source_path.stat().st_size,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'end_learning_rate': args.end_learning_rate,
        'warmup_steps': args.warmup_steps,
        'beta': args.beta,
        'ema_decay': args.ema_decay,
        'seed': args.seed,
        'vae_arch': VAE_ARCH,
    }
    write_json(output / 'config.json', config)
    print(f'JAX_DEVICES={jax.local_devices()}', flush=True)

    vae = make_vae()
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = vae.init(
        init_rng, jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)
    )['params']
    ema_params = params
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.end_learning_rate,
        peak_value=args.learning_rate,
        warmup_steps=min(args.warmup_steps, max(1, args.steps)),
        decay_steps=max(args.steps, args.warmup_steps + 1),
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, ema_params, opt_state, images, key):
        def loss_fn(candidate):
            posterior = vae.apply(
                {'params': candidate}, images, method=vae.encode
            ).latent_dist
            latent = posterior.sample(key)
            reconstruction = vae.apply(
                {'params': candidate}, latent, method=vae.decode
            ).sample
            mse = jnp.mean(jnp.square(images - reconstruction))
            kl = jnp.mean(posterior.kl())
            return mse + args.beta * kl, (mse, kl, jnp.std(latent))

        (loss, (mse, kl, latent_std)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = jax.tree_util.tree_map(
            lambda ema, new: args.ema_decay * ema
            + (1.0 - args.ema_decay) * new,
            ema_params,
            params,
        )
        return params, ema_params, opt_state, {
            'loss': loss,
            'mse': mse,
            'kl': kl,
            'latent_std': latent_std,
            'grad_norm': optax.global_norm(grads),
        }

    np_rng = np.random.default_rng(args.seed)
    started = time.time()
    metrics_path = output / 'metrics.jsonl'
    with h5py.File(data.source_path, 'r') as source:
        for step in range(1, args.steps + 1):
            rows = data.sample_image_rows(np_rng, args.batch_size)
            images = image_batch(source, rows)
            rng, update_rng = jax.random.split(rng)
            params, ema_params, opt_state, metrics = update(
                params, ema_params, opt_state, images, update_rng
            )
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {
                    'step': step,
                    'elapsed_seconds': time.time() - started,
                    'learning_rate': float(schedule(step - 1)),
                    **{key: float(value) for key, value in metrics.items()},
                }
                append_jsonl(metrics_path, record)
                print('VAE_METRICS=' + json.dumps(record, sort_keys=True), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_msgpack(
                    output / 'checkpoint.msgpack',
                    {'params': params, 'ema_params': ema_params, 'step': step},
                )
    print(f'VAE_COMPLETE={output}', flush=True)


def load_vae_run(run_dir: Path):
    run_dir = run_dir.expanduser().resolve()
    config = json.loads((run_dir / 'config.json').read_text())
    if config['upstream_commit'] != UPSTREAM_COMMIT:
        raise RuntimeError('VAE checkpoint upstream commit mismatch')
    if config['vae_arch'] != VAE_ARCH:
        raise RuntimeError('VAE architecture mismatch')
    checkpoint = load_msgpack(run_dir / 'checkpoint.msgpack')
    return make_vae(), checkpoint['ema_params'], config


def encode(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    verify_upstream()
    data = OGBenchLDPData(args.source)
    vae, params, vae_config = load_vae_run(args.vae_dir)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    episodes = len(data.offsets)
    if args.max_episodes is not None:
        episodes = min(episodes, args.max_episodes)
    rows = int(data.offsets[episodes - 1] + data.lengths[episodes - 1])

    @jax.jit
    def encode_images(images):
        return vae.apply(
            {'params': params}, images, method=vae.encode
        ).latent_dist.mean

    latent_min = np.inf
    latent_max = -np.inf
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with h5py.File(data.source_path, 'r') as source, h5py.File(
            temporary, 'w'
        ) as destination:
            latent_ds = destination.create_dataset(
                'latent',
                (rows, 64),
                dtype=np.float16,
                chunks=(min(4096, rows), 64),
            )
            for start in range(0, rows, args.batch_size):
                end = min(rows, start + args.batch_size)
                pixels = source['pixels'][start:end]
                images = pixels.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1
                if end - start < args.batch_size:
                    padding = np.repeat(images[-1:], args.batch_size - (end - start), axis=0)
                    images = np.concatenate([images, padding], axis=0)
                latent = np.asarray(encode_images(jnp.asarray(images)))[: end - start]
                flat = latent.reshape(end - start, -1)
                if flat.shape[1] != 64:
                    raise RuntimeError(f'Expected 64-D latent, got {flat.shape}')
                latent_ds[start:end] = flat.astype(np.float16)
                latent_min = min(latent_min, float(flat.min()))
                latent_max = max(latent_max, float(flat.max()))
                if start == 0 or end == rows or end % (args.batch_size * 100) == 0:
                    print(f'ENCODE_PROGRESS={end}/{rows}', flush=True)
            destination.attrs['source'] = str(data.source_path)
            destination.attrs['source_size_bytes'] = data.source_path.stat().st_size
            destination.attrs['source_rows'] = data.source_rows
            destination.attrs['rows'] = rows
            destination.attrs['episodes'] = episodes
            destination.attrs['latent_min'] = latent_min
            destination.attrs['latent_max'] = latent_max
            destination.attrs['vae_dir'] = str(args.vae_dir.expanduser().resolve())
            destination.attrs['vae_source_size_bytes'] = vae_config['source_size_bytes']
            destination.attrs['upstream_commit'] = UPSTREAM_COMMIT
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    OGBenchLDPData(data.source_path, output)
    print(
        'LATENT_COMPLETE='
        + json.dumps(
            {
                'path': str(output),
                'rows': rows,
                'episodes': episodes,
                'latent_min': latent_min,
                'latent_max': latent_max,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def make_ldp_models(latent_dim: int, action_dim: int):
    root = str(upstream_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from networks.diffusion import FourierFeatures
    from networks.diffusion_nets_v2 import ConditionalUnet1D
    from networks.mlp_diffusion_nets import MLPDiffusion, MLPResNet
    from networks.mlp_nets import MLP

    planner = ConditionalUnet1D(
        input_dim=latent_dim,
        global_cond_dim=latent_dim * 2,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        downsample=True,
    )
    idm = MLPDiffusion(
        cond_encoder_cls=partial(
            MLP,
            hidden_dims=(256, 256),
            activations='mish',
            activate_final=False,
        ),
        reverse_encoder_cls=partial(
            MLPResNet,
            n_blocks=3,
            out_dim=action_dim,
            use_layer_norm=True,
            hidden_dim=256,
        ),
        time_preprocess_cls=partial(
            FourierFeatures, output_size=256, learnable=False
        ),
    )
    return planner, idm


def make_schedulers(diffusion_steps: int):
    from diffusers.schedulers.scheduling_ddpm_flax import FlaxDDPMScheduler

    planner = FlaxDDPMScheduler(
        num_train_timesteps=diffusion_steps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    idm = FlaxDDPMScheduler(
        num_train_timesteps=diffusion_steps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    return planner, planner.create_state(), idm, idm.create_state()


def normalize(value, minimum: float, maximum: float):
    return np.clip(2 * (value - minimum) / (maximum - minimum) - 1, -1, 1).astype(
        np.float32
    )


def train_ldp(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import optax

    commit = verify_upstream()
    data = OGBenchLDPData(args.source, args.latents)
    data.load_training_arrays()
    output = create_output_dir(args.output_dir)
    planner, idm = make_ldp_models(data.latent_dim, data.action_dim)
    planner_scheduler, planner_scheduler_state, idm_scheduler, idm_scheduler_state = (
        make_schedulers(args.diffusion_steps)
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, planner_rng, idm_rng = jax.random.split(rng, 3)
    initial_time = jnp.zeros((1,), dtype=jnp.int32)
    planner_params = planner.init(
        planner_rng,
        jnp.zeros((1, args.pred_horizon, data.latent_dim), jnp.float32),
        initial_time,
        jnp.zeros((1, data.latent_dim * 2), jnp.float32),
    )['params']
    idm_params = idm.init(
        idm_rng,
        jnp.zeros((args.pred_horizon, data.latent_dim * 2), jnp.float32),
        jnp.zeros((args.pred_horizon, data.action_dim), jnp.float32),
        initial_time,
    )['params']
    params = {'planner': planner_params, 'idm': idm_params}
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.end_learning_rate,
        peak_value=args.learning_rate,
        warmup_steps=min(args.warmup_steps, max(1, args.steps)),
        decay_steps=max(args.steps, args.warmup_steps + 1),
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(schedule))
    opt_state = optimizer.init(params)
    config = {
        'kind': 'goal_conditioned_ogbench_ldp',
        'upstream_commit': commit,
        'goal_conditioning': 'planner_global_condition_current_plus_final_goal',
        'idm_conditioning': 'adjacent_latent_transition_only',
        'source': str(data.source_path),
        'source_size_bytes': data.source_path.stat().st_size,
        'latents': str(data.latent_path),
        'latent_size_bytes': data.latent_path.stat().st_size,
        'latent_dim': data.latent_dim,
        'latent_min': data.latent_min,
        'latent_max': data.latent_max,
        'action_dim': data.action_dim,
        'pred_horizon': args.pred_horizon,
        'action_horizon': args.action_horizon,
        'diffusion_steps': args.diffusion_steps,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'end_learning_rate': args.end_learning_rate,
        'warmup_steps': args.warmup_steps,
        'seed': args.seed,
        'planner_down_dims': [256, 512, 1024],
    }
    write_json(output / 'config.json', config)
    print(f'JAX_DEVICES={jax.local_devices()}', flush=True)
    print(
        f'PARAMETERS planner={sum(x.size for x in jax.tree_util.tree_leaves(planner_params))} '
        f'idm={sum(x.size for x in jax.tree_util.tree_leaves(idm_params))}',
        flush=True,
    )

    @jax.jit
    def update(params, opt_state, current, future, goal, actions, key):
        batch_size = current.shape[0]

        def loss_fn(candidate):
            planner_key, idm_key, planner_time_key, idm_time_key = jax.random.split(
                key, 4
            )
            planner_t = jax.random.randint(
                planner_time_key, (batch_size,), 0, args.diffusion_steps
            )
            planner_noise = jax.random.normal(planner_key, future.shape)
            noisy_future = planner_scheduler.add_noise(
                planner_scheduler_state, future, planner_noise, planner_t
            )
            condition = jnp.concatenate([current[:, 0], goal], axis=-1)
            predicted_planner_noise = planner.apply(
                {'params': candidate['planner']},
                noisy_future,
                planner_t,
                condition,
            )
            planner_loss = jnp.mean(
                jnp.square(predicted_planner_noise - planner_noise)
            )

            states = jnp.concatenate([current, future], axis=1)
            transitions = jnp.concatenate([states[:, :-1], states[:, 1:]], axis=-1)
            transitions = transitions.reshape(-1, data.latent_dim * 2)
            flat_actions = actions.reshape(-1, data.action_dim)
            idm_t = jax.random.randint(
                idm_time_key, (flat_actions.shape[0], 1), 0, args.diffusion_steps
            )
            idm_noise = jax.random.normal(idm_key, flat_actions.shape)
            noisy_actions = idm_scheduler.add_noise(
                idm_scheduler_state, flat_actions, idm_noise, idm_t
            )
            predicted_idm_noise = idm.apply(
                {'params': candidate['idm']},
                transitions,
                noisy_actions,
                idm_t,
            )
            idm_loss = jnp.mean(jnp.square(predicted_idm_noise - idm_noise))
            return planner_loss + idm_loss, (planner_loss, idm_loss)

        (loss, (planner_loss, idm_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {
            'loss': loss,
            'planner_loss': planner_loss,
            'idm_loss': idm_loss,
            'grad_norm': optax.global_norm(grads),
        }

    np_rng = np.random.default_rng(args.seed)
    started = time.time()
    metrics_path = output / 'metrics.jsonl'
    for step in range(1, args.steps + 1):
        batch = data.sample_windows(
            np_rng, args.batch_size, args.pred_horizon, split='train'
        )
        current = normalize(batch['current'], data.latent_min, data.latent_max)
        future = normalize(batch['future'], data.latent_min, data.latent_max)
        goal = normalize(batch['goal'], data.latent_min, data.latent_max)
        rng, update_rng = jax.random.split(rng)
        params, opt_state, metrics = update(
            params,
            opt_state,
            current,
            future,
            goal,
            batch['actions'],
            update_rng,
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                'step': step,
                'elapsed_seconds': time.time() - started,
                'learning_rate': float(schedule(step - 1)),
                **{key: float(value) for key, value in metrics.items()},
            }
            append_jsonl(metrics_path, record)
            print('LDP_METRICS=' + json.dumps(record, sort_keys=True), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_msgpack(
                output / 'checkpoint.msgpack',
                {'params': params, 'step': step},
            )
    print(f'LDP_COMPLETE={output}', flush=True)


def make_sampler(config: dict[str, Any], params: Any):
    import jax
    import jax.numpy as jnp

    latent_dim = int(config['latent_dim'])
    action_dim = int(config['action_dim'])
    pred_horizon = int(config['pred_horizon'])
    action_horizon = int(config['action_horizon'])
    diffusion_steps = int(config['diffusion_steps'])
    planner, idm = make_ldp_models(latent_dim, action_dim)
    planner_scheduler, planner_state, idm_scheduler, idm_state = make_schedulers(
        diffusion_steps
    )

    @jax.jit
    def sample(current, goal, key):
        planner_key, idm_key = jax.random.split(key)
        plan = jax.random.normal(
            planner_key, (current.shape[0], pred_horizon, latent_dim)
        )
        condition = jnp.concatenate([current, goal], axis=-1)

        def planner_loop(index, value):
            noisy_plan, loop_key = value
            loop_key, step_key = jax.random.split(loop_key)
            timestep = diffusion_steps - 1 - index
            noise = planner.apply(
                {'params': params['planner']},
                noisy_plan,
                timestep,
                condition,
                training=False,
            )
            noisy_plan = planner_scheduler.step(
                planner_state, noise, timestep, noisy_plan, step_key
            ).prev_sample
            return noisy_plan, loop_key

        plan, _ = jax.lax.fori_loop(
            0, diffusion_steps, planner_loop, (plan, planner_key)
        )
        short_plan = plan[:, :action_horizon]
        states = jnp.concatenate([current[:, None], short_plan], axis=1)
        transitions = jnp.concatenate([states[:, :-1], states[:, 1:]], axis=-1)
        flat_transitions = transitions.reshape(-1, latent_dim * 2)
        actions = jax.random.normal(
            idm_key, (flat_transitions.shape[0], action_dim)
        )

        def idm_loop(index, value):
            noisy_actions, loop_key = value
            loop_key, step_key = jax.random.split(loop_key)
            timestep = diffusion_steps - 1 - index
            noise = idm.apply(
                {'params': params['idm']},
                flat_transitions,
                noisy_actions,
                timestep,
                training=False,
            )
            noisy_actions = idm_scheduler.step(
                idm_state, noise, timestep, noisy_actions, step_key
            ).prev_sample
            return noisy_actions, loop_key

        actions, _ = jax.lax.fori_loop(
            0, diffusion_steps, idm_loop, (actions, idm_key)
        )
        return jnp.clip(actions.reshape(current.shape[0], action_horizon, action_dim), -1, 1)

    return sample


def cube_goal_distance(env) -> float:
    unwrapped = env.unwrapped
    cube_pos = unwrapped._data.joint('object_joint_0').qpos[:3]
    target_id = unwrapped._cube_target_mocap_ids[0]
    target_pos = unwrapped._data.mocap_pos[target_id]
    return float(np.linalg.norm(cube_pos - target_pos))


def evaluate(args: argparse.Namespace) -> None:
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import ogbench  # noqa: F401  # registers official environments

    verify_upstream()
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / 'config.json').read_text())
    if config['upstream_commit'] != UPSTREAM_COMMIT:
        raise RuntimeError('LDP checkpoint upstream commit mismatch')
    checkpoint = load_msgpack(run_dir / 'checkpoint.msgpack')
    sampler = make_sampler(config, checkpoint['params'])
    vae, vae_params, vae_config = load_vae_run(args.vae_dir)
    if vae_config['source_size_bytes'] != config['source_size_bytes']:
        raise RuntimeError('VAE and LDP were not trained from the same source data')

    @jax.jit
    def encode_pixels(pixels):
        images = pixels.transpose(0, 3, 1, 2).astype(jnp.float32) / 127.5 - 1
        latent = vae.apply(
            {'params': vae_params}, images, method=vae.encode
        ).latent_dist.mean
        flat = latent.reshape(latent.shape[0], -1)
        return jnp.clip(
            2 * (flat - config['latent_min'])
            / (config['latent_max'] - config['latent_min'])
            - 1,
            -1,
            1,
        )

    output = create_output_dir(args.output_dir)
    env = gym.make(
        args.env_id,
        max_episode_steps=args.max_episode_steps,
        render_mode='rgb_array',
        height=64,
        width=64,
        reward_task_id=args.reward_task_id,
        terminate_at_goal=True,
        visualize_info=False,
        permute_blocks=False,
    )
    successes: list[bool] = []
    returns: list[float] = []
    lengths: list[int] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []
    minimum_distances: list[float] = []
    action_norms: list[float] = []
    key = jax.random.PRNGKey(args.seed)
    started = time.time()
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            goal = info.get('goal', info.get('target'))
            if goal is None:
                raise RuntimeError('OGBench reset info has neither goal nor target')
            observation = np.asarray(observation)
            goal = np.asarray(goal)
            if observation.shape != (64, 64, 3) or goal.shape != (64, 64, 3):
                raise RuntimeError(
                    f'Expected 64x64 RGB current/goal, got {observation.shape}/{goal.shape}'
                )
            goal_latent = encode_pixels(jnp.asarray(goal[None]))
            initial_distance = cube_goal_distance(env)
            minimum_distance = initial_distance
            episode_return = 0.0
            episode_action_norms: list[float] = []
            success = bool(info.get('success', False))
            length = 0
            queued_actions = np.zeros((0, config['action_dim']), dtype=np.float32)
            for step in range(args.max_episode_steps):
                if len(queued_actions) == 0:
                    current_latent = encode_pixels(jnp.asarray(observation[None]))
                    key, sample_key = jax.random.split(key)
                    queued_actions = np.asarray(
                        sampler(current_latent, goal_latent, sample_key)[0]
                    )
                action, queued_actions = queued_actions[0], queued_actions[1:]
                episode_action_norms.append(float(np.linalg.norm(action)))
                observation, reward, terminated, truncated, info = env.step(action)
                distance = cube_goal_distance(env)
                minimum_distance = min(minimum_distance, distance)
                episode_return += float(reward)
                success = success or bool(info.get('success', False))
                length = step + 1
                if terminated or truncated:
                    break
            final_distance = cube_goal_distance(env)
            successes.append(success)
            returns.append(episode_return)
            lengths.append(length)
            initial_distances.append(initial_distance)
            final_distances.append(final_distance)
            minimum_distances.append(minimum_distance)
            action_norms.append(float(np.mean(episode_action_norms)))
            print(
                f'EPISODE episode={episode + 1} success={int(success)} '
                f'return={episode_return:.6f} length={length} '
                f'initial_distance={initial_distance:.6f} '
                f'final_distance={final_distance:.6f} '
                f'min_distance={minimum_distance:.6f} '
                f'mean_action_norm={action_norms[-1]:.6f}',
                flush=True,
            )
    finally:
        env.close()
    result = {
        'method': 'goal_conditioned_latent_diffusion_planning',
        'upstream_commit': UPSTREAM_COMMIT,
        'checkpoint': str(run_dir / 'checkpoint.msgpack'),
        'episodes': args.episodes,
        'seed': args.seed,
        'success_rate': float(np.mean(successes)),
        'episode_successes': successes,
        'episode_returns': returns,
        'episode_lengths': lengths,
        'initial_cube_goal_distances': initial_distances,
        'final_cube_goal_distances': final_distances,
        'minimum_cube_goal_distances': minimum_distances,
        'mean_action_norms': action_norms,
        'elapsed_seconds': time.time() - started,
        'environment': {
            'id': args.env_id,
            'reward_task_id': args.reward_task_id,
            'max_episode_steps': args.max_episode_steps,
        },
        'planner': {
            'pred_horizon': config['pred_horizon'],
            'action_horizon': config['action_horizon'],
            'diffusion_steps': config['diffusion_steps'],
            'goal_conditioning': config['goal_conditioning'],
        },
    }
    write_json(output / 'results.json', result)
    print('RESULT_JSON=' + json.dumps(result, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    audit_parser = commands.add_parser('audit')
    audit_parser.add_argument('--source', type=Path, required=True)
    audit_parser.add_argument('--latents', type=Path)
    audit_parser.set_defaults(func=audit)

    vae_parser = commands.add_parser('train-vae')
    vae_parser.add_argument('--source', type=Path, required=True)
    vae_parser.add_argument('--output-dir', type=Path, required=True)
    vae_parser.add_argument('--steps', type=int, default=100_000)
    vae_parser.add_argument('--batch-size', type=int, default=128)
    vae_parser.add_argument('--learning-rate', type=float, default=1e-4)
    vae_parser.add_argument('--end-learning-rate', type=float, default=1e-6)
    vae_parser.add_argument('--warmup-steps', type=int, default=1_000)
    vae_parser.add_argument('--beta', type=float, default=1e-5)
    vae_parser.add_argument('--ema-decay', type=float, default=0.99)
    vae_parser.add_argument('--seed', type=int, default=1)
    vae_parser.add_argument('--log-every', type=int, default=100)
    vae_parser.add_argument('--save-every', type=int, default=10_000)
    vae_parser.set_defaults(func=train_vae)

    encode_parser = commands.add_parser('encode')
    encode_parser.add_argument('--source', type=Path, required=True)
    encode_parser.add_argument('--vae-dir', type=Path, required=True)
    encode_parser.add_argument('--output', type=Path, required=True)
    encode_parser.add_argument('--batch-size', type=int, default=512)
    encode_parser.add_argument('--max-episodes', type=int)
    encode_parser.set_defaults(func=encode)

    ldp_parser = commands.add_parser('train-ldp')
    ldp_parser.add_argument('--source', type=Path, required=True)
    ldp_parser.add_argument('--latents', type=Path, required=True)
    ldp_parser.add_argument('--output-dir', type=Path, required=True)
    ldp_parser.add_argument('--steps', type=int, default=100_000)
    ldp_parser.add_argument('--batch-size', type=int, default=128)
    ldp_parser.add_argument('--pred-horizon', type=int, default=8)
    ldp_parser.add_argument('--action-horizon', type=int, default=4)
    ldp_parser.add_argument('--diffusion-steps', type=int, default=100)
    ldp_parser.add_argument('--learning-rate', type=float, default=1e-4)
    ldp_parser.add_argument('--end-learning-rate', type=float, default=1e-6)
    ldp_parser.add_argument('--warmup-steps', type=int, default=1_000)
    ldp_parser.add_argument('--seed', type=int, default=1)
    ldp_parser.add_argument('--log-every', type=int, default=100)
    ldp_parser.add_argument('--save-every', type=int, default=10_000)
    ldp_parser.set_defaults(func=train_ldp)

    eval_parser = commands.add_parser('eval')
    eval_parser.add_argument('--run-dir', type=Path, required=True)
    eval_parser.add_argument('--vae-dir', type=Path, required=True)
    eval_parser.add_argument('--output-dir', type=Path, required=True)
    eval_parser.add_argument('--env-id', default='visual-cube-single-v0')
    eval_parser.add_argument('--episodes', type=int, default=10)
    eval_parser.add_argument('--seed', type=int, default=42)
    eval_parser.add_argument('--max-episode-steps', type=int, default=50)
    eval_parser.add_argument('--reward-task-id', type=int, default=2)
    eval_parser.set_defaults(func=evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    positive = [
        name
        for name in ('steps', 'batch_size', 'pred_horizon', 'action_horizon', 'episodes')
        if hasattr(args, name) and getattr(args, name) < 1
    ]
    if positive:
        raise ValueError(f'These arguments must be positive: {positive}')
    if hasattr(args, 'action_horizon') and args.action_horizon > args.pred_horizon:
        raise ValueError('action_horizon cannot exceed pred_horizon')
    if hasattr(args, 'pred_horizon') and args.pred_horizon % 4:
        raise ValueError('pred_horizon must be divisible by 4 for the planner U-Net')
    args.func(args)


if __name__ == '__main__':
    main()
