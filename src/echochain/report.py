from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from echochain.records import Row, RunDataError, read_manifest, read_rows
from echochain.stats import (
    AnalyzeError,
    attractor_fit,
    degradation_step,
    kaplan_meier,
    log_rank,
    median_bounds,
    median_time,
    percentile,
    restricted_mean,
)

_DRAWS = 1000
_LOW = 0.025
_HIGH = 0.975
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


def report(
    directories: Sequence[Path],
    *,
    out: Path | None = None,
    rng: random.Random | None = None,
) -> str:
    if not directories:
        raise AnalyzeError("analyze needs at least one run")
    runs = tuple(_load(directory) for directory in directories)
    generator = random.Random(0) if rng is None else rng  # noqa: S311
    survival_rng = random.Random(0) if rng is None else generator  # noqa: S311
    curves = _curve_lines(runs, generator)
    survival = _survival_lines(runs, survival_rng)
    comparisons = _comparison_lines(runs)
    attractors = _attractor_lines(runs)
    text = "\n\n".join(
        (
            "curves\n" + "\n".join(curves),
            "survival\n" + "\n".join(survival),
            "comparisons\n" + "\n".join(comparisons),
            "attractors\n" + "\n".join(attractors),
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
    time: int
    event: int
    tau: int
    points: tuple[_Point, ...]


def _load(directory: Path) -> _Run:
    try:
        manifest = read_manifest(directory)
        rows = read_rows(directory)
    except RunDataError as exc:
        raise AnalyzeError(str(exc)) from exc
    if not rows:
        raise AnalyzeError(f"{directory} has no rows")
    cluster = manifest.seed_id if manifest.seed_id else manifest.run_id
    points = tuple(sorted((_point(row) for row in rows), key=lambda item: item.step))
    observed_time, observed_event = _event_time(
        points, manifest.stop_rule.threshold, manifest.stop_rule.consecutive
    )
    time, event = _cap(observed_time, observed_event, manifest.max_cycles)
    return _Run(
        run_id=manifest.run_id,
        condition=manifest.condition,
        cluster=cluster,
        time=time,
        event=event,
        tau=manifest.max_cycles,
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


def _point(row: Row) -> _Point:
    values: dict[str, float] = {}
    judge = row.judge
    named = {
        "claim_fidelity": judge.claim_fidelity,
        "claim_fidelity_vs_prev": judge.prev_claim_fidelity,
        "contradiction_rate": judge.contradiction_rate,
        "fidelity_vs_seed": judge.fidelity_vs_seed,
        "fidelity_vs_prev": judge.fidelity_vs_prev,
        "anchored": judge.anchored,
        "anchored_vs_prev": judge.anchored_vs_prev,
    }
    for name, value in named.items():
        if value is not None:
            values[name] = value
    for name in _BASELINE_METRICS:
        metric = getattr(row.baseline, name)
        if isinstance(metric, int | float):
            values[name] = float(metric)
    return _Point(step=row.step, values=values)


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
    width = 720
    panel_height = 120
    height = panel_height * len(_PANELS)
    parts = [
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
    ]
    seen = _conditions(rows)
    for index, (metric, title, ratio) in enumerate(_PANELS):
        _draw_panel(parts, rows, seen, (metric, title, ratio, index * panel_height))
    parts.append("</svg>")
    return "".join(parts)


def _conditions(rows: Sequence[Sequence[str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row[0] not in seen:
            seen.append(row[0])
    return seen


def _draw_panel(
    parts: list[str],
    rows: Sequence[Sequence[str]],
    seen: Sequence[str],
    panel: tuple[str, str, bool, int],
) -> None:
    metric, title, ratio, offset = panel
    left = 48
    top = offset + 28
    plot_width = 460
    plot_height = 80
    matching = [row for row in rows if row[2] == metric]
    last = max((int(row[1]) for row in matching), default=1)
    high = 1.0
    if ratio:
        observed = max((float(row[4]) for row in matching), default=high)
        high = max(high, observed)
    parts.extend(_axes(left, top, plot_width, plot_height))
    parts.append(_label("48", str(offset + 16), "12", "#222222", title))
    for color_index, condition in enumerate(seen):
        series = [row for row in matching if row[0] == condition]
        if not series:
            continue
        coords = " ".join(
            f"{_x(int(row[1]), last, left, plot_width):.2f},"
            f"{_y_span(float(row[4]), top, plot_height, high):.2f}"
            for row in series
        )
        parts.append(_polyline(_COLORS[color_index % len(_COLORS)], coords, False))
    if offset == 0:
        for index, condition in enumerate(seen):
            parts.append(
                _label(
                    "530",
                    str(16 + index * 14),
                    "12",
                    _COLORS[index % len(_COLORS)],
                    condition,
                )
            )


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
        *_axes(left, top, plot_width, plot_height),
    ]


def _axes(left: int, top: int, width: int, height: int) -> tuple[str, str]:
    down = _void(
        "line",
        (
            ("x1", str(left)),
            ("y1", str(top)),
            ("x2", str(left)),
            ("y2", str(top + height)),
            ("stroke", "#222222"),
        ),
    )
    across = _void(
        "line",
        (
            ("x1", str(left)),
            ("y1", str(top + height)),
            ("x2", str(left + width)),
            ("y2", str(top + height)),
            ("stroke", "#222222"),
        ),
    )
    return down, across


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
    return _y_span(value, top, height, 1.0)


def _y_span(value: float, top: int, height: int, high: float) -> float:
    span = high if high > 0.0 else 1.0
    clipped = min(span, max(0.0, value))
    return top + (1.0 - clipped / span) * height


def _attractor_lines(runs: Sequence[_Run]) -> list[str]:
    lines = ["condition\tmetric\tslope\tstrength\tposition"]
    grouped = _by_condition(runs)
    for condition in sorted(grouped):
        for metric in _ATTRACTOR_METRICS:
            pairs = [
                ends
                for run in grouped[condition]
                if (ends := _ends(run, metric)) is not None
            ]
            fit = attractor_fit(pairs)
            if fit is None:
                lines.append(f"{condition}\t{metric}\tnot estimated")
                continue
            slope, strength, position = fit
            lines.append(
                f"{condition}\t{metric}\t{_fmt(slope)}\t{_fmt(strength)}\t"
                f"{_fmt(position)}"
            )
    return lines


def _ends(run: _Run, metric: str) -> tuple[float, float] | None:
    observed = [point.values[metric] for point in run.points if metric in point.values]
    if len(observed) < _ENDPOINTS:
        return None
    return observed[0], observed[-1]


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
