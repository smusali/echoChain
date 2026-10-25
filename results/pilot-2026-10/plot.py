from __future__ import annotations

import json
from pathlib import Path

_WIDTH = 960
_HEIGHT = 560
_LEFT = 76
_RIGHT = 28
_TOP = 112
_BOTTOM = 72
_Y_MIN = 0.0
_Y_MAX = 11.0
_FONT = "ui-sans-serif, system-ui, sans-serif"
_COLORS = {
    "alternate": "#1d4ed8",
    "swap": "#c2410c",
    "same-model": "#047857",
}
_TICKS_Y = (2, 4, 6, 8, 10)
_TICKS_X = (2, 6, 10, 14, 18, 22, 26, 30)


def main() -> None:
    """Draw mean fidelity against step for each condition.

    Returns:
        None. Writes `fidelity.svg` next to this script.
    """
    root = Path(__file__).resolve().parent
    runs = _runs(root)
    (root / "fidelity.svg").write_text(_svg(runs), encoding="utf-8")


def _runs(root: Path) -> dict[str, list[list[tuple[int, float]]]]:
    grouped: dict[str, list[list[tuple[int, float]]]] = {}
    for path in sorted(root.glob("*/chain.jsonl")):
        points: list[tuple[int, float]] = []
        condition = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "":
                continue
            row = _object(json.loads(line))
            parsed = _fidelity(row)
            if parsed is None:
                continue
            condition, step, score = parsed
            points.append((step, score))
        if condition != "" and points:
            grouped.setdefault(condition, []).append(points)
    return dict(sorted(grouped.items()))


def _fidelity(row: dict[str, object]) -> tuple[str, int, float] | None:
    if row.get("score_kind") != "fidelity":
        return None
    condition = row.get("condition")
    step = row.get("step")
    score = _object(row["judge"]).get("fidelity_vs_seed")
    if not isinstance(condition, str) or condition == "":
        return None
    if isinstance(step, bool) or not isinstance(step, int):
        return None
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return condition, step, score


def _svg(runs: dict[str, list[list[tuple[int, float]]]]) -> str:
    bounds = _bounds(runs)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
            f'height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}">'
        ),
        '<defs><clipPath id="plot">'
        f'<rect x="{_LEFT}" y="{_TOP}" width="{_WIDTH - _LEFT - _RIGHT}" '
        f'height="{_HEIGHT - _TOP - _BOTTOM}"/></clipPath></defs>',
        f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="#f8fafc"/>',
        _title(),
        _legend(list(runs)),
        _panel(),
        _grid(bounds),
    ]
    for condition, repeats in runs.items():
        color = _COLORS.get(condition, "#334155")
        parts.append(_repeat_lines(repeats, bounds, color))
        parts.append(_mean_line(_mean(repeats), bounds, color))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _bounds(runs: dict[str, list[list[tuple[int, float]]]]) -> tuple[int, int]:
    steps = [
        step
        for repeats in runs.values()
        for points in repeats
        for step, _score in points
    ]
    if not steps:
        return 1, 1
    return min(steps), max(steps)


def _mean(repeats: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    buckets: dict[int, list[float]] = {}
    for points in repeats:
        for step, score in points:
            buckets.setdefault(step, []).append(score)
    return [
        (step, sum(values) / len(values)) for step, values in sorted(buckets.items())
    ]


def _title() -> str:
    return (
        f'<text x="{_LEFT}" y="36" fill="#0f172a" font-family="{_FONT}" '
        'font-size="20" font-weight="650">Mean fidelity by step</text>'
        f'<text x="{_LEFT}" y="58" fill="#64748b" font-family="{_FONT}" '
        'font-size="13">Solid line is the mean. Faint lines are repeats.</text>'
    )


def _legend(conditions: list[str]) -> str:
    parts: list[str] = []
    cursor = _LEFT
    for condition in conditions:
        color = _COLORS.get(condition, "#334155")
        parts.append(
            f'<circle cx="{cursor + 5}" cy="86" r="5" fill="{color}"/>'
            f'<text x="{cursor + 16}" y="91" fill="#0f172a" '
            f'font-family="{_FONT}" font-size="13">{condition}</text>'
        )
        cursor += 28 + len(condition) * 8
    return "".join(parts)


def _panel() -> str:
    width = _WIDTH - _LEFT - _RIGHT
    height = _HEIGHT - _TOP - _BOTTOM
    return (
        f'<rect x="{_LEFT}" y="{_TOP}" width="{width}" height="{height}" '
        'fill="#ffffff" stroke="#e2e8f0"/>'
    )


def _grid(bounds: tuple[int, int]) -> str:
    x_min, x_max = bounds
    parts: list[str] = []
    for score in _TICKS_Y:
        y = _y(score)
        parts.append(
            f'<line x1="{_LEFT}" y1="{y:.2f}" x2="{_WIDTH - _RIGHT}" '
            f'y2="{y:.2f}" stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{_LEFT - 12}" y="{y + 4:.2f}" fill="#475569" '
            f'font-family="{_FONT}" font-size="12" text-anchor="end">{score}</text>'
        )
    parts.append(
        f'<text x="22" y="{(_TOP + _HEIGHT - _BOTTOM) / 2}" fill="#475569" '
        f'font-family="{_FONT}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 22 {(_TOP + _HEIGHT - _BOTTOM) / 2})">'
        "Fidelity</text>"
    )
    for step in _TICKS_X:
        if step < x_min or step > x_max:
            continue
        x = _x(step, x_min, x_max)
        parts.append(
            f'<line x1="{x:.2f}" y1="{_TOP}" x2="{x:.2f}" '
            f'y2="{_HEIGHT - _BOTTOM}" stroke="#f1f5f9"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{_HEIGHT - _BOTTOM + 22}" fill="#475569" '
            f'font-family="{_FONT}" font-size="12" text-anchor="middle">{step}</text>'
        )
    parts.append(
        f'<text x="{(_LEFT + _WIDTH - _RIGHT) / 2}" y="{_HEIGHT - 16}" '
        f'fill="#475569" font-family="{_FONT}" font-size="13" '
        'text-anchor="middle">Step</text>'
    )
    return "\n".join(parts)


def _repeat_lines(
    repeats: list[list[tuple[int, float]]],
    bounds: tuple[int, int],
    color: str,
) -> str:
    lines: list[str] = []
    for points in repeats:
        coords = _coords(points, bounds)
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-opacity="0.35" '
            'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" '
            f'clip-path="url(#plot)" points="{coords}"/>'
        )
    return "\n".join(lines)


def _mean_line(
    points: list[tuple[int, float]],
    bounds: tuple[int, int],
    color: str,
) -> str:
    coords = _coords(points, bounds)
    marks = "\n".join(
        f'<circle cx="{_x(step, bounds[0], bounds[1]):.2f}" '
        f'cy="{_y(score):.2f}" r="3.5" fill="#ffffff" stroke="{color}" '
        'stroke-width="2"/>'
        for step, score in points
    )
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
        'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#plot)" '
        f'points="{coords}"/>\n{marks}'
    )


def _coords(points: list[tuple[int, float]], bounds: tuple[int, int]) -> str:
    x_min, x_max = bounds
    return " ".join(
        f"{_x(step, x_min, x_max):.2f},{_y(score):.2f}" for step, score in points
    )


def _x(step: int, x_min: int, x_max: int) -> float:
    span = _WIDTH - _LEFT - _RIGHT
    left = x_min - 1
    right = x_max + 1
    if right == left:
        return _LEFT + span / 2
    return _LEFT + (step - left) / (right - left) * span


def _y(score: float) -> float:
    span = _HEIGHT - _TOP - _BOTTOM
    return _TOP + (_Y_MAX - score) / (_Y_MAX - _Y_MIN) * span


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("row is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("row key is not a string")
        mapped[key] = item
    return mapped


if __name__ == "__main__":
    main()
