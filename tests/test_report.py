from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import assert_valid_run, settings_for, transport

from echochain.report import report
from echochain.runner import execute


def test_report_reads_a_finished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    finished = execute(
        settings_for(tmp_path, start="Redis uses epoll"),
        yes=True,
        transport=transport()[0],
    )
    assert_valid_run(finished.directory)
    out = tmp_path / "analysis"
    text = report((finished.directory,), out=out)
    assert text.startswith("curves\n")
    assert (out / "curves.svg").is_file()
    assert (out / "survival.svg").is_file()
