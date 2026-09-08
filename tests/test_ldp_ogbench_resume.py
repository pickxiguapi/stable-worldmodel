from collections import namedtuple
import json

import pytest

from scripts.train.ldp_ogbench import (
    resolve_vae_adapter_commit,
    restore_optimizer_schedule_step,
)


ScaleByAdamState = namedtuple('ScaleByAdamState', ['count', 'mu', 'nu'])
ScaleByScheduleState = namedtuple('ScaleByScheduleState', ['count'])


def test_legacy_resume_advances_schedule_but_not_adam_count():
    state = (
        (),
        (
            ScaleByAdamState(count=0, mu={'weight': 0.0}, nu={'weight': 0.0}),
            ScaleByScheduleState(count=0),
        ),
    )

    restored = restore_optimizer_schedule_step(state, 50_000)

    assert restored[1][0].count == 0
    assert restored[1][1].count == 50_000
    assert restored[1][0].mu is state[1][0].mu
    assert restored[1][0].nu is state[1][0].nu


def test_legacy_resume_requires_exactly_one_schedule_state():
    with pytest.raises(RuntimeError, match='found 0'):
        restore_optimizer_schedule_step((ScaleByAdamState(0, {}, {}),), 10)

    with pytest.raises(RuntimeError, match='found 2'):
        restore_optimizer_schedule_step(
            (ScaleByScheduleState(0), ScaleByScheduleState(0)), 10
        )


def test_legacy_resume_rejects_negative_step():
    with pytest.raises(ValueError, match='non-negative'):
        restore_optimizer_schedule_step((ScaleByScheduleState(0),), -1)


def test_vae_adapter_commit_prefers_training_config(tmp_path):
    commit = 'a' * 40

    assert resolve_vae_adapter_commit(
        tmp_path, {'adapter_commit': commit}
    ) == (commit, 'config.json')


def test_legacy_vae_adapter_commit_uses_validated_provenance(tmp_path):
    commit = 'b' * 40
    source = tmp_path / 'source.h5'
    config = {
        'source': str(source),
        'source_size_bytes': 123,
        'upstream_commit': 'c' * 40,
    }
    (tmp_path / 'provenance.json').write_text(
        json.dumps(
            {
                'kind': 'ogbench_ldp_pipeline_provenance',
                'dataset': str(source.resolve()),
                'dataset_size_bytes': 123,
                'upstream_ldp_commit': 'c' * 40,
                'vae_adapter_commit': commit,
                'evidence': {'server_checkout_at_process_start': commit},
            }
        )
    )

    assert resolve_vae_adapter_commit(tmp_path, config) == (
        commit,
        'provenance.json',
    )


def test_legacy_vae_adapter_commit_rejects_mismatch(tmp_path):
    commit = 'd' * 40
    config = {
        'source': str(tmp_path / 'source.h5'),
        'source_size_bytes': 123,
        'upstream_commit': 'e' * 40,
    }
    (tmp_path / 'provenance.json').write_text(
        json.dumps(
            {
                'kind': 'ogbench_ldp_pipeline_provenance',
                'dataset': str(tmp_path / 'wrong.h5'),
                'dataset_size_bytes': 123,
                'upstream_ldp_commit': 'e' * 40,
                'vae_adapter_commit': commit,
                'evidence': {'server_checkout_at_process_start': commit},
            }
        )
    )

    with pytest.raises(RuntimeError, match='provenance mismatch'):
        resolve_vae_adapter_commit(tmp_path, config)
