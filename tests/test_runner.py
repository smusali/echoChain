from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import assert_valid_run, settings_for, transport

from echochain.records import read_rows
from echochain.runner import execute, resume


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_one_run(tmp_path: Path) -> None:
    gate, _calls = transport()
    finished = execute(
        settings_for(tmp_path, start="Redis uses epoll"),
        yes=True,
        transport=gate,
    )
    assert finished.reason.value == "max_cycles"
    assert_valid_run(finished.directory)


def test_resume_keeps_claim_fidelity(tmp_path: Path, seed_file: Path) -> None:
    gate, _calls = transport(stop_after=4)
    settings = settings_for(
        tmp_path,
        seeds=seed_file,
        seed="custom",
        max_cycles=4,
    )
    interrupted = execute(settings, yes=True, transport=gate)
    assert interrupted.reason.value == "interrupted"
    rows = read_rows(interrupted.directory)
    assert [row.step for row in rows] == [1, 2]
    resumed_gate, _again = transport()
    finished = resume(
        str(interrupted.directory),
        settings,
        transport=resumed_gate,
    )
    assert finished.reason.value == "max_cycles"
    restored = read_rows(finished.directory)
    step_four = restored[-1]
    assert step_four.step == 4
    assert step_four.judge.claim_fidelity is not None


def test_batch_resume(tmp_path: Path) -> None:
    gate, _calls = transport(stop_after=0)
    settings = settings_for(tmp_path, start="Redis uses epoll", repeats=2)
    interrupted = execute(settings, yes=True, transport=gate)
    assert interrupted.reason.value == "interrupted"
    batch = next(tmp_path.glob("*.batch.json"))
    finished = resume(str(batch), settings, transport=transport()[0])
    assert finished.reason.value == "max_cycles"
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 2
