from __future__ import annotations

from echochain.stats import (
    attractor_fit,
    degradation_step,
    kaplan_meier,
    log_rank,
    percentile,
    restricted_mean,
)


def test_percentile_and_survival() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    curve = kaplan_meier([(1, True), (2, False)])
    assert curve[0].survival == 0.5
    assert restricted_mean([(1, True), (2, False)], tau=2) > 0


def test_degradation_log_rank_and_attractor() -> None:
    assert degradation_step(((1, 1.0), (2, 1.0)), 2, 2) == 2
    assert degradation_step(((1, 5.0),), 2, 1) is None
    ranked = log_rank([(1, 1), (2, 1)], [(1, 0), (3, 1)])
    assert ranked is not None
    stat, probability = ranked
    assert stat >= 0
    assert 0 <= probability <= 1
    fitted = attractor_fit(((0.0, 0.2), (1.0, 0.6), (2.0, 1.0)))
    assert fitted is not None
    slope, strength, position = fitted
    assert strength == 1 - slope
    assert position > 0
