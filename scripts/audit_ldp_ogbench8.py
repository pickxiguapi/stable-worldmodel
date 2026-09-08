#!/usr/bin/env python3
"""Deterministic completion audit for the eight OGBench LDP pipelines."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


UPSTREAM_COMMIT = 'a26cbf1d2c0aec7adc5d9746f47831b162a41c0c'
OGBENCH_COMMIT = '1d4140997f60c52c6fb0702ec100dc988b18c548'
GOAL_CONDITIONING = 'planner_global_condition_current_plus_final_goal'
IDM_CONDITIONING = 'adjacent_latent_transition_only'
ANOMALY_PATTERN = re.compile(
    r'Traceback \(most recent call last\):|CUDA out of memory|'
    r'\b(?:NaN|nan)\b|\bFATAL\b|\bERROR\b'
)

NATIVE_HORIZONS = {
    'visual-cube-single-v0': 200,
    'visual-cube-double-v0': 500,
    'visual-cube-triple-v0': 1000,
    'visual-scene-v0': 750,
}


@dataclass(frozen=True)
class TaskSpec:
    tag: str
    dataset_id: str
    env_id: str
    gpu: int


TASKS = (
    TaskSpec('cube_single', 'visual-cube-single-play-v0', 'visual-cube-single-v0', 6),
    TaskSpec('cube_double_play', 'visual-cube-double-play-v0', 'visual-cube-double-v0', 0),
    TaskSpec('cube_triple_play', 'visual-cube-triple-play-v0', 'visual-cube-triple-v0', 1),
    TaskSpec('scene_play', 'visual-scene-play-v0', 'visual-scene-v0', 2),
    TaskSpec('cube_single_noisy', 'visual-cube-single-noisy-v0', 'visual-cube-single-v0', 3),
    TaskSpec('cube_double_noisy', 'visual-cube-double-noisy-v0', 'visual-cube-double-v0', 4),
    TaskSpec('cube_triple_noisy', 'visual-cube-triple-noisy-v0', 'visual-cube-triple-v0', 5),
    TaskSpec('scene_noisy', 'visual-scene-noisy-v0', 'visual-scene-v0', 7),
)


class Audit:
    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.evidence: dict[str, Any] = {}

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def read_json(path: Path, audit: Audit, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        audit.errors.append(f'missing {label}: {path}')
        return None
    try:
        value = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - diagnostic path
        audit.errors.append(f'invalid {label} {path}: {exc}')
        return None
    if not isinstance(value, dict):
        audit.errors.append(f'{label} is not a JSON object: {path}')
        return None
    return value


def read_checkpoint(path: Path, audit: Audit, label: str) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        audit.errors.append(f'missing {label}: {path}')
        return None
    try:
        from flax import serialization

        value = serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:  # pragma: no cover - diagnostic path
        audit.errors.append(f'cannot load {label} {path}: {exc}')
        return None
    if not isinstance(value, dict):
        audit.errors.append(f'{label} is not a mapping: {path}')
        return None
    audit.evidence[f'{label}_bytes'] = path.stat().st_size
    return value


def last_metric(path: Path, audit: Audit, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        audit.errors.append(f'missing {label}: {path}')
        return None
    last = None
    try:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    last = json.loads(line)
    except Exception as exc:
        audit.errors.append(f'invalid {label} {path}: {exc}')
        return None
    if not isinstance(last, dict):
        audit.errors.append(f'empty {label}: {path}')
        return None
    return last


def finite_metrics(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    try:
        return all(math.isfinite(float(record[key])) for key in keys)
    except (KeyError, TypeError, ValueError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integer_field(
    record: dict[str, Any], key: str, audit: Audit, label: str
) -> int:
    try:
        return int(record[key])
    except (KeyError, TypeError, ValueError):
        audit.errors.append(f'invalid integer field {label}.{key}')
        return -1


def validate_eval_result(
    result: dict[str, Any], spec: TaskSpec, episodes: int, seed: int
) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(result.get('dataset_id') == spec.dataset_id, 'evaluation dataset mismatch')
    require(result.get('episodes_per_task') == episodes, 'episodes_per_task mismatch')
    require(result.get('task_ids') == [1, 2, 3, 4, 5], 'official task IDs mismatch')
    require(result.get('total_episodes') == episodes * 5, 'total episode count mismatch')
    require(result.get('seed') == seed, 'evaluation seed mismatch')
    environment = result.get('environment') or {}
    require(environment.get('id') == spec.env_id, 'evaluation environment mismatch')
    require(
        environment.get('creation_api')
        == 'ogbench.make_env_and_datasets(env_only=True)',
        'evaluation did not use the official OGBench factory',
    )
    require(environment.get('uses_registered_horizon') is True, 'registered horizon not used')
    require(
        environment.get('max_episode_steps') == NATIVE_HORIZONS[spec.env_id],
        'native max_episode_steps mismatch',
    )
    ogbench = result.get('ogbench') or {}
    require(ogbench.get('commit') == OGBENCH_COMMIT, 'OGBench commit mismatch')
    require(
        str(ogbench.get('origin', '')).rstrip('/').removesuffix('.git')
        == 'https://github.com/seohongpark/ogbench',
        'OGBench origin mismatch',
    )
    tasks = result.get('tasks')
    require(isinstance(tasks, list) and len(tasks) == 5, 'invalid task results')
    all_successes: list[bool] = []
    if isinstance(tasks, list) and len(tasks) == 5:
        for expected_id, task in enumerate(tasks, start=1):
            require(isinstance(task, dict), f'task {expected_id} is not an object')
            if not isinstance(task, dict):
                continue
            require(task.get('task_id') == expected_id, f'task {expected_id} ID mismatch')
            require(bool(task.get('task_name')), f'task {expected_id} name missing')
            require(task.get('episodes') == episodes, f'task {expected_id} episode mismatch')
            successes = task.get('episode_successes')
            require(
                isinstance(successes, list) and len(successes) == episodes,
                f'task {expected_id} invalid successes',
            )
            if isinstance(successes, list) and len(successes) == episodes:
                require(
                    all(isinstance(item, bool) for item in successes),
                    f'task {expected_id} successes must be bool',
                )
                expected_rate = float(np.mean(successes))
                all_successes.extend(successes)
                try:
                    actual_rate = float(task['success_rate'])
                except (KeyError, TypeError, ValueError):
                    errors.append(f'task {expected_id} invalid success_rate')
                else:
                    require(
                        math.isfinite(actual_rate)
                        and abs(actual_rate - expected_rate) <= 1e-12,
                        f'task {expected_id} success_rate is inconsistent',
                    )
            for key in (
                'episode_returns',
                'episode_lengths',
                'initial_goal_residuals',
                'final_goal_residuals',
                'minimum_goal_residuals',
                'mean_action_norms',
            ):
                values = task.get(key)
                require(
                    isinstance(values, list) and len(values) == episodes,
                    f'task {expected_id} invalid {key}',
                )
                if isinstance(values, list) and len(values) == episodes:
                    try:
                        require(
                            all(math.isfinite(float(item)) for item in values),
                            f'task {expected_id} non-finite {key}',
                        )
                    except (TypeError, ValueError):
                        errors.append(f'task {expected_id} non-numeric {key}')
    if len(all_successes) == episodes * 5:
        try:
            overall_rate = float(result['success_rate'])
        except (KeyError, TypeError, ValueError):
            errors.append('invalid overall success_rate')
        else:
            require(
                math.isfinite(overall_rate)
                and abs(overall_rate - float(np.mean(all_successes))) <= 1e-12,
                'overall success_rate is inconsistent',
            )
    planner = result.get('planner') or {}
    require(planner.get('pred_horizon') == 8, 'pred_horizon mismatch')
    require(planner.get('action_horizon') == 4, 'action_horizon mismatch')
    require(planner.get('diffusion_steps') == 100, 'diffusion_steps mismatch')
    require(planner.get('goal_conditioning') == GOAL_CONDITIONING, 'goal conditioning mismatch')
    return errors


def audit_task(
    spec: TaskSpec,
    dataset_root: Path,
    artifact_root: Path,
    label: str,
    seed: int,
    episodes: int,
    eval_seed: int,
) -> Audit:
    audit = Audit(spec)
    name = f'ldp_{spec.tag}_{label}_s{seed}'
    source_path = (dataset_root / f'{spec.dataset_id}.h5').resolve()
    vae_dir = artifact_root / 'runs' / f'{name}_vae'
    latent_path = artifact_root / 'data' / f'{name}_latents.h5'
    ldp_dir = artifact_root / 'runs' / f'{name}_ldp'
    eval_path = artifact_root / 'evals' / f'{name}_eval{episodes}_s{eval_seed}' / 'results.json'
    log_path = artifact_root / f'{name}.log'
    eval_log_path = artifact_root / f'{name}_official_eval.log'
    audit.evidence.update(
        run=name,
        dataset=str(source_path),
        environment=spec.env_id,
        gpu=spec.gpu,
    )

    source_rows = source_episodes = action_dim = source_size = None
    if not source_path.is_file():
        audit.errors.append(f'missing source dataset: {source_path}')
    else:
        source_size = source_path.stat().st_size
        try:
            with h5py.File(source_path, 'r') as source:
                source_rows = int(source['pixels'].shape[0])
                source_episodes = int(source['ep_len'].shape[0])
                action_dim = int(source['action'].shape[1])
                audit.require(tuple(source['pixels'].shape[1:]) == (64, 64, 3), 'source pixel shape mismatch')
                audit.require(action_dim == 5, 'source action_dim must be 5')
                audit.require(source.attrs.get('reward_scheme') == 'negative_step_goal_zero_v2', 'source reward scheme mismatch')
                audit.require(int(source.attrs.get('segment_transitions', -1)) == 50, 'segment length mismatch')
        except Exception as exc:
            audit.errors.append(f'invalid source dataset: {exc}')
    audit.evidence.update(source_rows=source_rows, source_episodes=source_episodes)

    vae_config = read_json(vae_dir / 'config.json', audit, 'VAE config')
    if vae_config:
        audit.require(vae_config.get('kind') == 'ogbench_ldp_vae', 'VAE kind mismatch')
        audit.require(vae_config.get('source') == str(source_path), 'VAE source mismatch')
        audit.require(vae_config.get('source_size_bytes') == source_size, 'VAE source size mismatch')
        audit.require(vae_config.get('steps') == 300_000, 'VAE steps config mismatch')
        audit.require(vae_config.get('batch_size') == 128, 'VAE batch size mismatch')
        audit.require(vae_config.get('seed') == seed, 'VAE seed mismatch')
        audit.require(vae_config.get('upstream_commit') == UPSTREAM_COMMIT, 'VAE upstream commit mismatch')
        if spec.tag == 'cube_single':
            audit.warn(bool(vae_config.get('adapter_commit')), 'legacy cube-single VAE config lacks adapter_commit')
        else:
            audit.require(bool(vae_config.get('adapter_commit')), 'VAE adapter_commit missing')

    vae_checkpoint = read_checkpoint(vae_dir / 'checkpoint.msgpack', audit, 'VAE checkpoint')
    if vae_checkpoint:
        vae_checkpoint_step = integer_field(
            vae_checkpoint, 'step', audit, 'VAE checkpoint'
        )
        audit.require(vae_checkpoint_step == 300_000, 'VAE checkpoint is not step 300000')
        audit.require({'params', 'ema_params'}.issubset(vae_checkpoint), 'VAE parameter trees missing')
        stateful = {'opt_state', 'rng'}.issubset(vae_checkpoint)
        if spec.tag == 'cube_single':
            audit.warn(stateful, 'legacy cube-single VAE checkpoint is not exact-resume capable')
        else:
            audit.require(stateful, 'VAE exact-resume state missing')
        audit.evidence['vae_checkpoint_step'] = vae_checkpoint_step
        del vae_checkpoint
        gc.collect()

    vae_metric = last_metric(vae_dir / 'metrics.jsonl', audit, 'VAE metrics')
    if vae_metric:
        audit.require(
            integer_field(vae_metric, 'step', audit, 'VAE metrics') == 300_000,
            'VAE metrics do not end at 300000',
        )
        audit.require(finite_metrics(vae_metric, ('loss', 'mse', 'kl', 'latent_std', 'grad_norm')), 'VAE final metrics invalid')
        audit.evidence['vae_final_metrics'] = vae_metric

    if not latent_path.is_file():
        audit.errors.append(f'missing latent file: {latent_path}')
    elif source_rows is not None:
        try:
            with h5py.File(latent_path, 'r') as latent_file:
                latent = latent_file['latent']
                audit.require(latent.shape == (source_rows, 64), 'latent shape mismatch')
                audit.require(latent.dtype == np.float16, 'latent dtype must be float16')
                audit.require(int(latent_file.attrs.get('rows', -1)) == source_rows, 'latent rows attr mismatch')
                audit.require(int(latent_file.attrs.get('episodes', -1)) == source_episodes, 'latent episodes attr mismatch')
                audit.require(int(latent_file.attrs.get('source_size_bytes', -1)) == source_size, 'latent source size mismatch')
                audit.require(latent_file.attrs.get('normalization_split') == 'train_episodes_only', 'latent normalization split mismatch')
                normalization_rows = int(latent_file.attrs.get('normalization_rows', -1))
                audit.require(0 < normalization_rows < source_rows, 'latent normalization rows invalid')
                validation_mse = float(latent_file.attrs.get('vae_validation_mse', np.nan))
                audit.require(math.isfinite(validation_mse) and validation_mse <= 0.1, 'latent VAE validation gate failed')
                latent_min = math.inf
                latent_max = -math.inf
                finite = True
                for start in range(0, source_rows, 65_536):
                    values = latent[start : start + 65_536]
                    finite = finite and bool(np.isfinite(values).all())
                    latent_min = min(latent_min, float(values.min()))
                    latent_max = max(latent_max, float(values.max()))
                audit.require(finite, 'latent file contains NaN or Inf')
                audit.require(latent_max > latent_min, 'latent range is degenerate')
                audit.evidence.update(
                    latent_rows=source_rows,
                    latent_min=latent_min,
                    latent_max=latent_max,
                    vae_validation_mse=validation_mse,
                )
        except Exception as exc:
            audit.errors.append(f'invalid latent file: {exc}')

    ldp_config = read_json(ldp_dir / 'config.json', audit, 'LDP config')
    if ldp_config:
        expected = {
            'kind': 'goal_conditioned_ogbench_ldp',
            'upstream_commit': UPSTREAM_COMMIT,
            'goal_conditioning': GOAL_CONDITIONING,
            'idm_conditioning': IDM_CONDITIONING,
            'source': str(source_path),
            'latents': str(latent_path.resolve()),
            'latent_dim': 64,
            'action_dim': action_dim,
            'pred_horizon': 8,
            'action_horizon': 4,
            'diffusion_steps': 100,
            'steps': 500_000,
            'batch_size': 128,
            'validation_batches': 4,
            'seed': seed,
            'validation_rng_isolated_from_training': True,
        }
        for key, value in expected.items():
            audit.require(ldp_config.get(key) == value, f'LDP config mismatch: {key}')
        audit.require(bool(ldp_config.get('adapter_commit')), 'LDP adapter_commit missing')

    ldp_checkpoint = read_checkpoint(ldp_dir / 'checkpoint.msgpack', audit, 'LDP checkpoint')
    if ldp_checkpoint:
        ldp_checkpoint_step = integer_field(
            ldp_checkpoint, 'step', audit, 'LDP checkpoint'
        )
        audit.require(ldp_checkpoint_step == 500_000, 'LDP checkpoint is not step 500000')
        audit.require({'params', 'opt_state', 'train_rng', 'validation_rng'}.issubset(ldp_checkpoint), 'LDP exact-resume state missing')
        audit.evidence['ldp_checkpoint_step'] = ldp_checkpoint_step
        del ldp_checkpoint
        gc.collect()

    ldp_metric = last_metric(ldp_dir / 'metrics.jsonl', audit, 'LDP metrics')
    if ldp_metric:
        audit.require(
            integer_field(ldp_metric, 'step', audit, 'LDP metrics') == 500_000,
            'LDP metrics do not end at 500000',
        )
        audit.require(
            finite_metrics(
                ldp_metric,
                ('loss', 'planner_loss', 'idm_loss', 'grad_norm', 'val_loss', 'val_planner_loss', 'val_idm_loss'),
            ),
            'LDP final train/validation metrics invalid',
        )
        audit.evidence['ldp_final_metrics'] = ldp_metric

    result = read_json(eval_path, audit, 'evaluation result')
    if result:
        audit.errors.extend(validate_eval_result(result, spec, episodes, eval_seed))
        audit.evidence['success_rate'] = result.get('success_rate')
        audit.evidence['task_success_rates'] = [
            task.get('success_rate') for task in result.get('tasks', [])
        ]

    if not log_path.is_file():
        audit.errors.append(f'missing formal log: {log_path}')
    else:
        matches = ANOMALY_PATTERN.findall(log_path.read_text(errors='replace'))
        audit.require(not matches, f'formal log contains {len(matches)} anomaly markers')
    if not eval_log_path.is_file():
        audit.errors.append(f'missing official evaluation log: {eval_log_path}')
    else:
        matches = ANOMALY_PATTERN.findall(
            eval_log_path.read_text(errors='replace')
        )
        audit.require(
            not matches,
            f'official evaluation log contains {len(matches)} anomaly markers',
        )

    artifact_paths = {
        'source': source_path,
        'vae_config': vae_dir / 'config.json',
        'vae_checkpoint': vae_dir / 'checkpoint.msgpack',
        'latents': latent_path,
        'ldp_config': ldp_dir / 'config.json',
        'ldp_checkpoint': ldp_dir / 'checkpoint.msgpack',
        'evaluation': eval_path,
    }
    audit.evidence['artifact_sha256'] = {
        key: sha256_file(path)
        for key, path in artifact_paths.items()
        if path.is_file()
    }

    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset-root', type=Path, required=True)
    result.add_argument('--artifact-root', type=Path, required=True)
    result.add_argument('--label', default='gc_finalgoal_h8_a4_ds100_v300k_p500k_b128')
    result.add_argument('--seed', type=int, default=1)
    result.add_argument('--episodes', type=int, default=10)
    result.add_argument('--eval-seed', type=int, default=42)
    result.add_argument('--output', type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    adapter_head = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True
    ).strip()
    adapter_main = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'origin/main'], text=True
    ).strip()
    adapter_dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain'], text=True
    ).strip()
    global_errors = []
    if adapter_head != adapter_main:
        global_errors.append('completion auditor checkout does not match origin/main')
    if adapter_dirty:
        global_errors.append('completion auditor checkout is dirty')
    audits = [
        audit_task(
            spec,
            args.dataset_root.resolve(),
            args.artifact_root.resolve(),
            args.label,
            args.seed,
            args.episodes,
            args.eval_seed,
        )
        for spec in TASKS
    ]
    report = {
        'kind': 'ldp_ogbench8_completion_audit',
        'complete': not global_errors and all(not audit.errors for audit in audits),
        'global_errors': global_errors,
        'adapter_provenance': {
            'head': adapter_head,
            'origin_main': adapter_main,
            'clean': not bool(adapter_dirty),
        },
        'task_count': len(audits),
        'error_count': len(global_errors) + sum(len(audit.errors) for audit in audits),
        'warning_count': sum(len(audit.warnings) for audit in audits),
        'tasks': [
            {
                'tag': audit.task.tag,
                'errors': audit.errors,
                'warnings': audit.warnings,
                'evidence': audit.evidence,
            }
            for audit in audits
        ],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(serialized + '\n')
        temporary.replace(args.output)
    print('COMPLETION_AUDIT=' + json.dumps(report, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
