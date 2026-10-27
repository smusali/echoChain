from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
from echochain.analyze import (
    AnalyzeError,
    degradation_step,
    kaplan_meier,
    krippendorff_alpha,
    log_rank,
    median_time,
    percentile,
    report,
    restricted_mean,
    spearman,
    weighted_kappa,
)


def test_percentile_blends_the_four_cluster_resamples() -> None:
    clusters = ((0.0, 1.0), (1.0,))
    pooled = [value for cluster in clusters for value in cluster]
    assert sum(pooled) / len(pooled) == pytest.approx(2 / 3)
    resamples = [
        _mean([value for index in (left, right) for value in clusters[index]])
        for left in (0, 1)
        for right in (0, 1)
    ]
    assert resamples == pytest.approx([0.5, 2 / 3, 2 / 3, 1.0])
    ordered = sorted(resamples)
    low = percentile(ordered, 0.025)
    high = percentile(ordered, 0.975)
    assert low == pytest.approx(0.5 + 0.075 * (2 / 3 - 0.5))
    assert high == pytest.approx(2 / 3 + 0.925 * (1.0 - 2 / 3))


def test_kaplan_meier_keeps_the_censored_row() -> None:
    censored_at = 2
    last_event = 3
    points = kaplan_meier(((1, 1), (censored_at, 0), (last_event, 1)))
    assert points[0].survival == pytest.approx(2 / 3)
    assert points[0].variance == pytest.approx(2 / 27)
    assert points[1].time == censored_at
    assert points[1].events == 0
    assert points[1].survival == pytest.approx(2 / 3)
    assert points[2].survival == 0
    assert points[2].variance == 0
    assert median_time(points) == last_event


def test_log_rank_matches_the_hand_value() -> None:
    stat, probability = _required(log_rank(((1, 1), (2, 1)), ((2, 1), (3, 1))))
    assert stat == pytest.approx(25 / 17)
    assert probability == pytest.approx(math.erfc(math.sqrt((25 / 17) / 2)))


def test_agreement_matches_the_hand_values() -> None:
    assert weighted_kappa((1, 1, 2, 2), (1, 1, 1, 2)) == pytest.approx(0.5)
    alpha = krippendorff_alpha((("1", "1"), ("1", "2"), ("2", "2")))
    assert alpha == pytest.approx(4 / 9)
    assert spearman((1, 2, 3), (1, 2, 3)) == pytest.approx(1)
    assert spearman((1, 2, 3), (3, 2, 1)) == pytest.approx(-1)
    assert spearman((1, 2, 2), (1, 2, 3)) == pytest.approx(math.sqrt(3) / 2)


def test_report_is_deterministic(tmp_path: Path) -> None:
    first = _directory(
        tmp_path,
        _Case("run-a", "alpha", "s1", 1, event=1, fidelity=0.0),
    )
    second = _directory(
        tmp_path,
        _Case("run-b", "beta", None, 2, event=0, fidelity=1.0),
    )
    out = tmp_path / "out"
    text = report((first, second), out=out)
    first_curves = (out / "curves.svg").read_bytes()
    first_survival = (out / "survival.svg").read_bytes()
    again = report((first, second), out=out)
    assert (
        hashlib.sha256(text.encode()).hexdigest()
        == hashlib.sha256(again.encode()).hexdigest()
    )
    assert first_curves == (out / "curves.svg").read_bytes()
    assert first_survival == (out / "survival.svg").read_bytes()
    assert b"<svg" in first_curves
    assert b"<svg" in first_survival
    assert "claim_fidelity" in text
    assert "run-b\tbeta\t2\t0" in text


def test_report_refuses_mixed_rubrics(tmp_path: Path) -> None:
    first = _directory(
        tmp_path,
        _Case("run-a", "alpha", "s1", 1, event=1, fidelity=0.0),
    )
    second = _directory(
        tmp_path,
        _Case("run-c", "alpha", "s1", 1, event=1, fidelity=0.0, rubric="1"),
    )
    with pytest.raises(AnalyzeError, match="rubric versions differ"):
        report((first, second))


def test_restricted_mean_keeps_censored_runs(tmp_path: Path) -> None:
    threshold = 2.0
    streak = 2
    event_step = streak + 1
    horizon = 4
    area = 3.0
    assert (
        degradation_step(((1, None), (2, 1.0), (3, 1.0)), threshold, streak)
        == event_step
    )
    assert degradation_step(((1, 1.0), (2, 8.0)), threshold, 0) == 1
    assert restricted_mean(((2, 1), (4, 1)), horizon) == area
    assert restricted_mean(((2, 1), (4, 0)), horizon) == area
    assert restricted_mean(((4, 0), (4, 0)), horizon) == float(horizon)
    mixed = report(
        (
            _directory(
                tmp_path,
                _Case(
                    "run-low",
                    "alpha",
                    "s1",
                    2,
                    event=1,
                    fidelity=0.0,
                    max_cycles=4,
                    fidelities=((2, 0.0),),
                ),
            ),
            _directory(
                tmp_path,
                _Case(
                    "run-hold",
                    "alpha",
                    "s1",
                    4,
                    event=0,
                    fidelity=1.0,
                    max_cycles=4,
                    fidelities=((4, 8.0),),
                ),
            ),
        )
    )
    assert "alpha\t3.000000\t3.000000\t3.000000\t4\t2\t0.500000" in mixed
    assert "event at step 2" in mixed
    assert "censored at 4" in mixed
    censored = report(
        (
            _directory(
                tmp_path,
                _Case(
                    "run-c",
                    "beta",
                    "s2",
                    4,
                    event=0,
                    fidelity=1.0,
                    max_cycles=4,
                    fidelities=((4, 8.0),),
                ),
            ),
            _directory(
                tmp_path,
                _Case(
                    "run-d",
                    "beta",
                    "s2",
                    4,
                    event=0,
                    fidelity=1.0,
                    max_cycles=4,
                    fidelities=((4, 8.0),),
                ),
            ),
        )
    )
    assert "beta\t4.000000\t4.000000\t4.000000\t4\tnot reached\t0.000000" in censored


def test_judge_validation_states_the_gate() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "docs" / "judge-validation.md"
    ).read_text(encoding="utf-8")
    assert "Krippendorff's alpha at least 0.67" in text
    assert "weighted Cohen's kappa at least 0.67" in text
    assert "Spearman rho at least 0.70" in text
    assert "These labels are not collected." in text
    assert "No agreement is claimed." in text


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _required(result: tuple[float, float] | None) -> tuple[float, float]:
    if result is None:
        raise AssertionError("log-rank variance was zero")
    return result


@dataclass(frozen=True)
class _Case:
    run_id: str
    condition: str
    seed_id: str | None
    time: int
    event: int
    fidelity: float
    rubric: str = "2"
    max_cycles: int | None = None
    fidelities: tuple[tuple[int, float], ...] = ()


def _directory(root: Path, case: _Case) -> Path:
    directory = root / case.run_id
    directory.mkdir()
    score = 1.0 if case.event == 1 else 8.0
    low_anchor = 2.0
    manifest: dict[str, object] = {
        "run_id": case.run_id,
        "condition": case.condition,
        "rubric_version": case.rubric,
        "seed_id": case.seed_id,
        "stop_rule": {"threshold": 2.0},
    }
    if case.max_cycles is not None:
        manifest["max_cycles"] = case.max_cycles
    observed = case.fidelities or ((case.time, score),)
    rows = [
        {
            "step": step,
            "judge": {
                "fidelity_vs_seed": fidelity,
                "fidelity_vs_prev": None,
                "anchored": low_anchor if fidelity <= low_anchor else 4.0,
                "anchored_vs_prev": None,
                "claim_fidelity": case.fidelity,
                "prev_claim_fidelity": None,
                "contradiction_rate": 0.0,
            },
        }
        for step, fidelity in observed
    ]
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8", newline="\n"
    )
    (directory / "chain.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return directory
