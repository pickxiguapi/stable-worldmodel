from collections import namedtuple

import pytest

from scripts.train.ldp_ogbench import restore_optimizer_schedule_step


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
