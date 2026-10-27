from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

_DRAWS = 1000
_LOW = 0.025
_HIGH = 0.975
_Z = 1.96
_HALF = 0.5
_BASELINE_METRICS = (
    "cosine_seed",
    "cosine_prev",
    "entail_seed_forward",
    "entail_seed_backward",
    "entail_prev_forward",
    "entail_prev_backward",
    "length_ratio",
    "technical_survival",
    "generic_share",
)
_METRICS = (
    "claim_fidelity",
    "claim_fidelity_vs_prev",
    "contradiction_rate",
    "fidelity_vs_seed",
    "fidelity_vs_prev",
    "anchored",
    "anchored_vs_prev",
    *_BASELINE_METRICS,
)
_PANELS = (
    ("claim_fidelity", "claim fidelity", False),
    ("cosine_seed", "cosine to the seed", False),
    ("entail_seed_backward", "backward entailment", False),
    ("technical_survival", "technical survival", False),
    ("generic_share", "generic share", False),
    ("length_ratio", "length ratio", True),
)
_ATTRACTOR_METRICS = ("generic_share", "length_ratio")
_ATTRACTOR_RUNS = 3
_ENDPOINTS = 2
_COLORS = ("#1f4e79", "#b85c38", "#2f6f4e", "#6b4c9a")
_CURVE_HEADER = (
    "condition",
    "step",
    "metric",
    "count",
    "mean",
    "ci_low",
    "ci_high",
)
_FIELD = {
    "claim_fidelity": "claim_fidelity",
    "claim_fidelity_vs_prev": "prev_claim_fidelity",
    "contradiction_rate": "contradiction_rate",
    "fidelity_vs_seed": "fidelity_vs_seed",
    "fidelity_vs_prev": "fidelity_vs_prev",
    "anchored": "anchored",
    "anchored_vs_prev": "anchored_vs_prev",
}


class AnalyzeError(RuntimeError):
    pass


@dataclass(frozen=True)
class KaplanPoint:
    time: int
    at_risk: int
    events: int
    survival: float
    variance: float


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linear percentile.

    Args:
        values: Sample values. They are sorted inside the function.
        fraction: Position from 0 to 1. The rank is that fraction times one
            less than the count, and the result blends the two nearest values.

    Returns:
        The interpolated value.

    Raises:
        AnalyzeError: The sample is empty.
    """
    ordered = sorted(values)
    if not ordered:
        raise AnalyzeError("percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def kaplan_meier(observations: Sequence[tuple[int, int]]) -> tuple[KaplanPoint, ...]:
    """Return the product-limit survival curve.

    Args:
        observations: Pairs of time and event. Event 1 is a failure. Event 0
            is censored. A time that is censored stays in the risk set at that
            time and then leaves.

    Returns:
        One point for each distinct time, in order.
    """
    if not observations:
        return ()
    survival = 1.0
    hazard = 0.0
    points: list[KaplanPoint] = []
    for time in sorted({item[0] for item in observations}):
        at_risk = sum(1 for observed, _event in observations if observed >= time)
        events = sum(
            1 for observed, event in observations if observed == time and event == 1
        )
        if at_risk > 0 and events > 0:
            if at_risk != events:
                hazard += events / (at_risk * (at_risk - events))
            survival *= 1.0 - events / at_risk
        variance = 0.0 if survival == 0.0 else survival * survival * hazard
        points.append(
            KaplanPoint(
                time=time,
                at_risk=at_risk,
                events=events,
                survival=survival,
                variance=variance,
            )
        )
    return tuple(points)


def degradation_step(
    scores: Sequence[tuple[int, float | None]],
    threshold: float,
    consecutive: int,
) -> int | None:
    """Return the step where the low-fidelity streak completes.

    Args:
        scores: Ordered pairs of step and fidelity. None skips that step and
            leaves the streak unchanged.
        threshold: Fidelity at or below this value counts toward the streak.
        consecutive: Scores required. Zero is read as one.

    Returns:
        The completing step, or None when the streak never completes.
    """
    needed = consecutive if consecutive > 0 else 1
    streak = 0
    for step, value in scores:
        if value is None:
            continue
        if value <= threshold:
            streak += 1
        else:
            streak = 0
        if streak >= needed:
            return step
    return None


def restricted_mean(observations: Sequence[tuple[int, int]], tau: int) -> float:
    """Return the area under the survival curve from zero to tau.

    Args:
        observations: Pairs of time and event. Event 1 is a failure. Event 0
            is censored. A time past tau is censored at tau.
        tau: Upper limit of the integral.

    Returns:
        The restricted mean. It equals tau when every run is censored at tau.

    Raises:
        AnalyzeError: tau is negative.
    """
    if tau < 0:
        raise AnalyzeError("tau is negative")
    capped = tuple(
        (time, event) if time <= tau else (tau, 0) for time, event in observations
    )
    area = 0.0
    previous_time = 0
    previous_survival = 1.0
    for point in kaplan_meier(capped):
        if point.time > tau:
            break
        area += previous_survival * (point.time - previous_time)
        previous_time = point.time
        previous_survival = point.survival
    if previous_time < tau:
        area += previous_survival * (tau - previous_time)
    return area


def median_time(points: Sequence[KaplanPoint]) -> int | None:
    """Return the first time survival is at or below one half.

    Args:
        points: Product-limit points in time order.

    Returns:
        That time, or None when survival stays above one half.
    """
    for point in points:
        if point.events > 0 and point.survival <= _HALF:
            return point.time
    return None


def median_bounds(points: Sequence[KaplanPoint]) -> tuple[int | None, int | None]:
    """Return the Greenwood bounds of the survival median.

    Args:
        points: Product-limit points in time order.

    Returns:
        The lower bound is the first time the upper band is at or below one
        half. The upper bound is the first time the lower band is at or below
        one half. Either side is None when that crossing never happens.
    """
    lower: int | None = None
    upper: int | None = None
    for point in points:
        if point.events == 0:
            continue
        low, high = _band(point)
        if lower is None and high <= _HALF:
            lower = point.time
        if upper is None and low <= _HALF:
            upper = point.time
    return lower, upper


def log_rank(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
) -> tuple[float, float] | None:
    """Return the Mantel-Haenszel log-rank statistic and its two-sided p-value.

    Args:
        left: Times and event flags for the first condition.
        right: Times and event flags for the second condition.

    Returns:
        Chi-square on one degree of freedom and the p-value from the
        complementary error function. None when the variance is zero.
    """
    times = sorted({time for time, event in (*left, *right) if event == 1})
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for time in times:
        at_risk_left = sum(1 for observed_time, _event in left if observed_time >= time)
        at_risk_right = sum(
            1 for observed_time, _event in right if observed_time >= time
        )
        events_left = sum(
            1 for observed_time, event in left if observed_time == time and event == 1
        )
        events_right = sum(
            1 for observed_time, event in right if observed_time == time and event == 1
        )
        at_risk = at_risk_left + at_risk_right
        events = events_left + events_right
        if at_risk <= 1 or events == 0:
            continue
        observed += events_left
        expected += at_risk_left * events / at_risk
        variance += (
            at_risk_left
            * at_risk_right
            * events
            * (at_risk - events)
            / (at_risk * at_risk * (at_risk - 1))
        )
    if variance == 0.0:
        return None
    stat = (observed - expected) ** 2 / variance
    return stat, math.erfc(math.sqrt(stat / 2.0))


def attractor_fit(
    pairs: Sequence[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """Fit the last value on the first when the absolute slope is below 1.

    Args:
        pairs: First and last values, one pair per run.

    Returns:
        Slope, strength, and position. None when the fit is not estimated.
    """
    if len(pairs) < _ATTRACTOR_RUNS:
        return None
    initials = [pair[0] for pair in pairs]
    finals = [pair[1] for pair in pairs]
    mean_initial = statistics.mean(initials)
    mean_final = statistics.mean(finals)
    variance = sum((value - mean_initial) ** 2 for value in initials)
    if variance == 0.0:
        return None
    slope = (
        sum((initial - mean_initial) * (final - mean_final) for initial, final in pairs)
        / variance
    )
    if not -1.0 < slope < 1.0:
        return None
    strength = 1.0 - slope
    intercept = mean_final - slope * mean_initial
    return slope, strength, intercept / strength


def _band(point: KaplanPoint) -> tuple[float, float]:
    margin = _Z * math.sqrt(point.variance)
    low = max(0.0, point.survival - margin)
    high = min(1.0, point.survival + margin)
    return low, high
