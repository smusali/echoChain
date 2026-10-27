from __future__ import annotations

from pathlib import Path

import pytest

from echochain.cli import exit_code, main
from echochain.records import TerminationReason


def test_exit_code_table() -> None:
    assert exit_code(TerminationReason.max_cycles) == 0
    assert exit_code(TerminationReason.threshold) == 0
    assert exit_code(TerminationReason.generation_error) == 1
    assert exit_code(TerminationReason.judge_error) == 1
    assert exit_code(TerminationReason.budget_exceeded) == 3
    assert exit_code(TerminationReason.insufficient_credits) == 3
    assert exit_code(TerminationReason.interrupted) == 130


def test_dry_run_prints_the_call_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = main(
        [
            "run",
            "--start",
            "alpha beta",
            "--model",
            "vendor-a/model",
            "--judge",
            "vendor-b/judge",
            "--judge-samples",
            "1",
            "--max-cycles",
            "2",
            "--max-tokens",
            "8",
            "--dry-run",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "calls: 4\n" in captured.out
    assert "cost_upper_bound_usd: unknown\n" in captured.out


def test_missing_selector_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--max-cycles", "1"]) == 2
    assert "provide --start" in capsys.readouterr().err


def test_help_lists_the_run_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["run", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--start",
        "--seed",
        "--seeds",
        "--sampling-seed",
        "--resume",
        "--provider",
    ):
        assert flag in text
    assert "--split" not in text
