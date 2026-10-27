from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_DRAWS = 1000
_LOW = 0.025
_HIGH = 0.975
_Z = 1.96
_HALF = 0.5
_METRICS = (
    "claim_fidelity",
    "claim_fidelity_vs_prev",
    "contradiction_rate",
    "fidelity_vs_seed",
    "fidelity_vs_prev",
    "anchored",
    "anchored_vs_prev",
)
_CURVE_SERIES = ("claim_fidelity", "claim_fidelity_vs_prev")
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


def weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    """Return linear weighted Cohen's kappa.

    Args:
        left: First rater's scores.
        right: Second rater's scores, paired with `left`.

    Returns:
        Kappa. With two categories, linear weights match unweighted kappa.

    Raises:
        AnalyzeError: The two sequences have different lengths or are empty.
    """
    if len(left) != len(right) or not left:
        raise AnalyzeError("kappa ratings must be paired")
    categories = sorted({*left, *right})
    span = len(categories) - 1
    index = {category: position for position, category in enumerate(categories)}
    counts = [[0 for _category in categories] for _category in categories]
    for first, second in zip(left, right, strict=True):
        counts[index[first]][index[second]] += 1
    total = float(len(left))
    row_sums = [float(sum(row)) for row in counts]
    column_sums = [
        float(sum(counts[row][column] for row in range(len(categories))))
        for column in range(len(categories))
    ]
    observed = 0.0
    expected = 0.0
    for row, row_sum in enumerate(row_sums):
        for column, column_sum in enumerate(column_sums):
            weight = 1.0 if span == 0 else 1.0 - abs(row - column) / span
            observed += weight * counts[row][column] / total
            expected += weight * (row_sum / total) * (column_sum / total)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha(pairs: Sequence[tuple[str, str]]) -> float:
    """Return nominal Krippendorff's alpha for paired verdicts.

    Args:
        pairs: One pair of labels per unit. A matching pair adds two to the
            diagonal of the coincidence matrix. A mismatch adds one to each
            off-diagonal cell.

    Returns:
        Alpha. It is 1 when the expected disagreement is zero.

    Raises:
        AnalyzeError: No pairs were given.
    """
    if not pairs:
        raise AnalyzeError("alpha requires at least one pair")
    coincidence: dict[tuple[str, str], float] = defaultdict(float)
    for left, right in pairs:
        coincidence[(left, right)] += 1.0
        coincidence[(right, left)] += 1.0
    total = sum(coincidence.values())
    margins: dict[str, float] = defaultdict(float)
    for (left, _right), count in coincidence.items():
        margins[left] += count
    observed = sum(
        count for (left, right), count in coincidence.items() if left != right
    )
    if total <= 1.0:
        return 1.0
    expected = sum(margin * (total - margin) for margin in margins.values()) / (
        total * (total - 1.0)
    )
    if expected == 0.0:
        return 1.0
    return 1.0 - (observed / total) / expected


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman's rho from the Pearson correlation of average ranks.

    Args:
        left: First sequence of scores.
        right: Second sequence, paired with `left`.

    Returns:
        Rho. Tied values share the average of the ranks they occupy.

    Raises:
        AnalyzeError: The two sequences have different lengths or are empty.
    """
    if len(left) != len(right) or not left:
        raise AnalyzeError("spearman scores must be paired")
    return _pearson(_ranks(left), _ranks(right))


def report(
    directories: Sequence[Path],
    *,
    out: Path | None = None,
    rng: random.Random | None = None,
) -> str:
    """Summarize one or more finished runs.

    Args:
        directories: Run directories that contain a manifest and JSONL.
        out: Directory for the tables and the two SVG figures. None prints
            only the text report.
        rng: Generator for the clustered bootstrap. None uses `random.Random(0)`.

    Returns:
        The curves, survival, restricted mean, and comparison sections. Floats
        use six decimals.

    Raises:
        AnalyzeError: A directory is unreadable or the rubric versions differ.
    """
    if not directories:
        raise AnalyzeError("analyze needs at least one run")
    runs = tuple(_load(directory) for directory in directories)
    versions = {run.rubric for run in runs}
    if len(versions) != 1:
        raise AnalyzeError("rubric versions differ")
    generator = random.Random(0) if rng is None else rng
    survival_rng = random.Random(0) if rng is None else generator
    curves = _curve_lines(runs, generator)
    survival = _survival_lines(runs, survival_rng)
    comparisons = _comparison_lines(runs)
    text = "\n\n".join(
        (
            "curves\n" + "\n".join(curves),
            "survival\n" + "\n".join(survival),
            "comparisons\n" + "\n".join(comparisons),
        )
    )
    rendered = text + "\n"
    if out is not None:
        _write(out, curves, survival, comparisons, runs)
    return rendered


@dataclass(frozen=True)
class _Point:
    step: int
    values: Mapping[str, float]


@dataclass(frozen=True)
class _Run:
    run_id: str
    condition: str
    cluster: str
    rubric: str
    time: int
    event: int
    tau: int
    points: tuple[_Point, ...]


def _band(point: KaplanPoint) -> tuple[float, float]:
    margin = _Z * math.sqrt(point.variance)
    low = max(0.0, point.survival - margin)
    high = min(1.0, point.survival + margin)
    return low, high


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2.0
        for index in order[start : end + 1]:
            ranks[index] = average
        start = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum(
        (first - mean_left) * (second - mean_right)
        for first, second in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - mean_left) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - mean_right) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return numerator / (left_scale * right_scale)


def _load(directory: Path) -> _Run:
    manifest_path = directory / "manifest.json"
    rows_path = directory / "chain.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise AnalyzeError(f"{directory} is not a finished run")
    manifest = _mapping(_parsed(manifest_path), manifest_path)
    rows = [
        _mapping(_parsed_line(rows_path, number, line), f"{rows_path}:{number}")
        for number, line in enumerate(_lines(rows_path), start=1)
    ]
    if not rows:
        raise AnalyzeError(f"{rows_path} has no rows")
    run_id = _text(manifest.get("run_id"), "run_id")
    condition = _text(manifest.get("condition"), "condition")
    rubric = _text(manifest.get("rubric_version"), "rubric_version")
    seed_id = manifest.get("seed_id")
    cluster = seed_id if isinstance(seed_id, str) and seed_id != "" else run_id
    stop_rule = manifest.get("stop_rule")
    threshold = _threshold(stop_rule)
    consecutive = _consecutive(stop_rule)
    points = tuple(sorted((_point(row) for row in rows), key=lambda item: item.step))
    observed_time, observed_event = _event_time(points, threshold, consecutive)
    last = points[-1].step
    horizon = _horizon(manifest.get("max_cycles"))
    tau = last if horizon is None else horizon
    time, event = _cap(observed_time, observed_event, tau)
    return _Run(
        run_id=run_id,
        condition=condition,
        cluster=cluster,
        rubric=rubric,
        time=time,
        event=event,
        tau=tau,
        points=points,
    )


def _event_time(
    points: Sequence[_Point],
    threshold: float,
    consecutive: int,
) -> tuple[int, int]:
    last = points[-1].step if points else 0
    found = degradation_step(
        tuple((point.step, point.values.get("fidelity_vs_seed")) for point in points),
        threshold,
        consecutive,
    )
    if found is None:
        return last, 0
    return found, 1


def _cap(time: int, event: int, tau: int) -> tuple[int, int]:
    if time > tau:
        return tau, 0
    return time, event


def _point(row: Mapping[str, object]) -> _Point:
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise AnalyzeError("row step must be a positive integer")
    judge = row.get("judge")
    values: dict[str, float] = {}
    if isinstance(judge, dict):
        for metric, field in _FIELD.items():
            number = _optional_number(judge.get(field))
            if number is not None:
                values[metric] = number
    return _Point(step=step, values=values)


def _curve_lines(runs: Sequence[_Run], rng: random.Random) -> list[str]:
    grouped: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    for run in runs:
        for point in run.points:
            for metric in _METRICS:
                value = point.values.get(metric)
                if value is None:
                    continue
                key = (run.condition, point.step, metric)
                grouped.setdefault(key, {}).setdefault(run.cluster, []).append(value)
    lines = ["\t".join(_CURVE_HEADER)]
    for condition, step, metric in sorted(grouped):
        clusters = [
            grouped[(condition, step, metric)][name]
            for name in sorted(grouped[(condition, step, metric)])
        ]
        mean, low, high, count = _interval(clusters, rng)
        lines.append(
            f"{condition}\t{step}\t{metric}\t{count}\t"
            f"{_fmt(mean)}\t{_fmt(low)}\t{_fmt(high)}"
        )
    return lines


def _interval(
    clusters: Sequence[Sequence[float]], rng: random.Random
) -> tuple[float, float, float, int]:
    pooled = [value for cluster in clusters for value in cluster]
    mean = statistics.mean(pooled)
    width = len(clusters)
    draws = [
        statistics.mean(
            [
                value
                for cluster in (
                    clusters[rng.randrange(width)] for _draw in range(width)
                )
                for value in cluster
            ]
        )
        for _sample in range(_DRAWS)
    ]
    return mean, percentile(draws, _LOW), percentile(draws, _HIGH), len(pooled)


def _survival_lines(runs: Sequence[_Run], rng: random.Random) -> list[str]:
    lines = ["run_id\tcondition\ttime\tevent\toutcome"]
    for run in sorted(runs, key=lambda item: (item.condition, item.run_id)):
        lines.append(
            f"{run.run_id}\t{run.condition}\t{run.time}\t{run.event}\t{_outcome(run)}"
        )
    lines.append("kaplan_meier")
    lines.append("condition\ttime\tat_risk\tevents\tsurvival\tvariance")
    grouped = _by_condition(runs)
    medians = ["median", "condition\tmedian\tci_low\tci_high"]
    restricted = [
        "restricted_mean",
        "condition\tmean\tci_low\tci_high\ttau\tmedian\tdegraded",
    ]
    for condition in sorted(grouped):
        members = grouped[condition]
        observed = [(run.time, run.event) for run in members]
        points = kaplan_meier(observed)
        for point in points:
            lines.append(
                f"{condition}\t{point.time}\t{point.at_risk}\t{point.events}\t"
                f"{_fmt(point.survival)}\t{_fmt(point.variance)}"
            )
        low, high = median_bounds(points)
        reached = median_time(points)
        medians.append(f"{condition}\t{_cell(reached)}\t{_cell(low)}\t{_cell(high)}")
        tau = _one_tau(members)
        mean, ci_low, ci_high = _restricted_interval(members, tau, rng)
        share = sum(run.event for run in members) / len(members)
        median = "not reached" if reached is None else str(reached)
        restricted.append(
            f"{condition}\t{_fmt(mean)}\t{_fmt(ci_low)}\t{_fmt(ci_high)}\t"
            f"{tau}\t{median}\t{_fmt(share)}"
        )
    return lines + medians + restricted


def _outcome(run: _Run) -> str:
    if run.event == 1:
        return f"event at step {run.time}"
    if run.time >= run.tau:
        return f"censored at {run.tau}"
    return f"censored at step {run.time}"


def _one_tau(runs: Sequence[_Run]) -> int:
    horizons = {run.tau for run in runs}
    if len(horizons) != 1:
        raise AnalyzeError("max_cycles differs")
    return next(iter(horizons))


def _restricted_interval(
    runs: Sequence[_Run],
    tau: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    clusters: dict[str, list[tuple[int, int]]] = {}
    for run in runs:
        clusters.setdefault(run.cluster, []).append((run.time, run.event))
    groups = [clusters[name] for name in sorted(clusters)]
    pooled = [item for group in groups for item in group]
    mean = restricted_mean(pooled, tau)
    width = len(groups)
    draws = [
        restricted_mean(
            [
                item
                for group in (groups[rng.randrange(width)] for _draw in range(width))
                for item in group
            ],
            tau,
        )
        for _sample in range(_DRAWS)
    ]
    return mean, percentile(draws, _LOW), percentile(draws, _HIGH)


def _comparison_lines(runs: Sequence[_Run]) -> list[str]:
    lines = ["condition_a\tcondition_b\tchi_square\tp_value"]
    grouped = _by_condition(runs)
    names = sorted(grouped)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result = log_rank(
                [(run.time, run.event) for run in grouped[left]],
                [(run.time, run.event) for run in grouped[right]],
            )
            if result is None:
                continue
            stat, probability = result
            lines.append(f"{left}\t{right}\t{_fmt(stat)}\t{_fmt(probability)}")
    return lines


def _by_condition(runs: Sequence[_Run]) -> dict[str, list[_Run]]:
    grouped: dict[str, list[_Run]] = {}
    for run in runs:
        grouped.setdefault(run.condition, []).append(run)
    return grouped


def _write(
    directory: Path,
    curves: Sequence[str],
    survival: Sequence[str],
    comparisons: Sequence[str],
    runs: Sequence[_Run],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _text_file(directory / "curves.tsv", curves)
    _text_file(directory / "survival.tsv", survival)
    _text_file(directory / "comparisons.tsv", comparisons)
    _text_file(directory / "curves.svg", (_curves_svg(curves),))
    _text_file(directory / "survival.svg", (_survival_svg(runs),))


def _text_file(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _curves_svg(lines: Sequence[str]) -> str:
    rows = [
        fields
        for line in lines[1:]
        if len(fields := line.split("\t")) == len(_CURVE_HEADER)
    ]
    frame = _frame()
    left, top, plot_width, plot_height = frame[2:]
    steps = [int(row[1]) for row in rows if row[2] in _CURVE_SERIES]
    last = max(steps, default=1)
    parts = _frame_markup(frame)
    parts.append(_label("48", "24", "14", "#222222", "claim fidelity"))
    seen: list[str] = []
    for row in rows:
        if row[2] not in _CURVE_SERIES or row[0] in seen:
            continue
        seen.append(row[0])
    for series_index, metric in enumerate(_CURVE_SERIES):
        for color_index, condition in enumerate(seen):
            matching = [row for row in rows if row[0] == condition and row[2] == metric]
            if not matching:
                continue
            coords = " ".join(
                f"{_x(int(row[1]), last, left, plot_width):.2f},"
                f"{_y(float(row[4]), top, plot_height):.2f}"
                for row in matching
            )
            parts.append(
                _polyline(_COLORS[color_index % len(_COLORS)], coords, series_index > 0)
            )
    for index, condition in enumerate(seen):
        parts.append(
            _label(
                "560",
                str(24 + index * 16),
                "12",
                _COLORS[index % len(_COLORS)],
                condition,
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _survival_svg(runs: Sequence[_Run]) -> str:
    grouped = _by_condition(runs)
    frame = _frame()
    left, top, plot_width, plot_height = frame[2:]
    last = max((run.time for run in runs), default=1)
    parts = _frame_markup(frame)
    parts.append(_label("48", "24", "14", "#222222", "survival"))
    for index, condition in enumerate(sorted(grouped)):
        color = _COLORS[index % len(_COLORS)]
        points = kaplan_meier([(run.time, run.event) for run in grouped[condition]])
        coords = [(float(left), _y(1.0, top, plot_height))]
        previous = 1.0
        for point in points:
            x_pos = _x(point.time, last, left, plot_width)
            coords.append((x_pos, _y(previous, top, plot_height)))
            coords.append((x_pos, _y(point.survival, top, plot_height)))
            previous = point.survival
        rendered = " ".join(f"{x_pos:.2f},{y_pos:.2f}" for x_pos, y_pos in coords)
        parts.append(_polyline(color, rendered, dashed=False))
        parts.append(_label("560", str(24 + index * 16), "12", color, condition))
    parts.append("</svg>")
    return "".join(parts)


def _frame() -> tuple[int, int, int, int, int, int]:
    return 640, 400, 48, 36, 500, 320


def _frame_markup(frame: tuple[int, int, int, int, int, int]) -> list[str]:
    width, height, left, top, plot_width, plot_height = frame
    return [
        _open(
            "svg",
            (
                ("xmlns", "http://www.w3.org/2000/svg"),
                ("width", str(width)),
                ("height", str(height)),
                ("viewBox", f"0 0 {width} {height}"),
            ),
        ),
        _void("rect", (("width", "100%"), ("height", "100%"), ("fill", "#ffffff"))),
        _void(
            "line",
            (
                ("x1", str(left)),
                ("y1", str(top)),
                ("x2", str(left)),
                ("y2", str(top + plot_height)),
                ("stroke", "#222222"),
            ),
        ),
        _void(
            "line",
            (
                ("x1", str(left)),
                ("y1", str(top + plot_height)),
                ("x2", str(left + plot_width)),
                ("y2", str(top + plot_height)),
                ("stroke", "#222222"),
            ),
        ),
    ]


def _polyline(color: str, points: str, dashed: bool) -> str:
    attrs = [
        ("fill", "none"),
        ("stroke", color),
        ("stroke-width", "2"),
        ("points", points),
    ]
    if dashed:
        attrs.append(("stroke-dasharray", "4 3"))
    return _void("polyline", attrs)


def _label(x_pos: str, y_pos: str, size: str, color: str, text: str) -> str:
    return _text_tag(
        "text",
        (
            ("x", x_pos),
            ("y", y_pos),
            ("font-size", size),
            ("fill", color),
        ),
        _xml(text),
    )


def _open(name: str, attrs: Sequence[tuple[str, str]]) -> str:
    return f"<{name} {_attrs(attrs)}>"


def _void(name: str, attrs: Sequence[tuple[str, str]]) -> str:
    return f"<{name} {_attrs(attrs)}/>"


def _text_tag(name: str, attrs: Sequence[tuple[str, str]], text: str) -> str:
    return f"<{name} {_attrs(attrs)}>{text}</{name}>"


def _attrs(attrs: Sequence[tuple[str, str]]) -> str:
    return " ".join(f'{key}="{value}"' for key, value in attrs)


def _x(step: int, last: int, left: int, width: int) -> float:
    if last <= 1:
        return float(left + width / 2)
    return left + (step / last) * width


def _y(value: float, top: int, height: int) -> float:
    clipped = min(1.0, max(0.0, value))
    return top + (1.0 - clipped) * height


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _cell(value: int | None) -> str:
    if value is None:
        return "null"
    return str(value)


def _lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    ]


def _parsed(path: Path) -> object:
    try:
        return _json_value(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalyzeError(f"{path} is not JSON") from exc


def _parsed_line(path: Path, number: int, line: str) -> object:
    try:
        return _json_value(line)
    except json.JSONDecodeError as exc:
        raise AnalyzeError(f"{path}:{number} is not JSON") from exc


def _json_value(text: str) -> object:
    value: object = json.loads(text)
    return value


def _mapping(value: object, label: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AnalyzeError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AnalyzeError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise AnalyzeError(f"{label} is missing")
    return value


def _threshold(value: object) -> float:
    if not isinstance(value, dict):
        raise AnalyzeError("stop_rule is missing")
    number = _optional_number(value.get("threshold"))
    if number is None:
        raise AnalyzeError("stop_rule threshold is missing")
    return number


def _consecutive(value: object) -> int:
    if not isinstance(value, dict) or "consecutive" not in value:
        return 1
    raw = value.get("consecutive")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise AnalyzeError("stop_rule consecutive is invalid")
    return raw


def _horizon(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalyzeError("max_cycles is invalid")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
