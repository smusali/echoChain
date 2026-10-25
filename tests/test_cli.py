from __future__ import annotations

import csv
import json
import logging
import os
import socket
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from echochain import __version__
from echochain.cli import main
from echochain.client import (
    ChatMessage,
    ChatReply,
    ModelRecord,
    parse_catalog,
)
from echochain.config import (
    JUDGE_MODEL,
    MISSING_API_KEY,
    REDACTED,
    configured_models,
    load_api_key,
)
from echochain.engine import cycle
from echochain.persistence import Recorder, mk_exp_dir
from pytest_socket import SocketBlockedError

_EXIT_CONFIG = 2


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    if not isinstance(project, dict):
        raise TypeError("project table is missing")
    value = project["version"]
    if not isinstance(value, str):
        raise TypeError("project.version must be a string")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _catalog(parameters: Mapping[str, Sequence[str]]) -> dict[str, ModelRecord]:
    payload: dict[str, object] = {
        "data": [
            {"id": model_id, "supported_parameters": list(names)}
            for model_id, names in parameters.items()
        ]
    }
    return parse_catalog(payload)


def _full_parameters() -> dict[str, list[str]]:
    return {model_id: ["response_format", "seed"] for model_id in configured_models()}


class _FakeChat:
    def __init__(self, text: str) -> None:
        self._text = text

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
    ) -> ChatReply:
        body = '{"score": 1}' if model == JUDGE_MODEL else self._text
        return ChatReply(
            text=body,
            requested_model=model,
            served_model=f"{model}-served",
        )


def test_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_version_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"echochain {__version__}"


def test_help_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert "echochain" in captured.out
    assert "--api-key" not in captured.out
    assert "--start" in captured.out


def test_import_without_api_key(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env["PYTHONPATH"] = str(_repo_root())
    completed = subprocess.run(
        [sys.executable, "-c", "import echochain.client, echochain.config"],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "OPENROUTER_API_KEY" not in completed.stderr
    assert "KeyError" not in completed.stderr


def test_missing_key_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--start", "Redis uses epoll"]) == _EXIT_CONFIG
    captured = capsys.readouterr()
    assert captured.err.strip() == MISSING_API_KEY
    assert captured.out == ""


def test_dotenv_supplies_blank_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    (tmp_path / ".env").write_text(
        '# comment\nexport OPENROUTER_API_KEY="from-file"\n',
        encoding="utf-8",
    )
    assert load_api_key() == "from-file"


def test_environment_wins_over_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=from-file\n",
        encoding="utf-8",
    )
    assert load_api_key() == "from-env"


def test_preflight_missing_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    parameters = _full_parameters()
    missing = configured_models()[0]
    del parameters[missing]

    def fetch(_api_key: str) -> dict[str, ModelRecord]:
        return _catalog(parameters)

    monkeypatch.setattr("echochain.client.fetch_catalog", fetch)
    assert main(["--start", "Redis uses epoll"]) == _EXIT_CONFIG
    message = capsys.readouterr().err
    assert missing in message
    assert "not in the OpenRouter catalog" in message
    assert "test-key" not in message


def test_preflight_missing_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    parameters = _full_parameters()
    parameters[JUDGE_MODEL] = ["response_format"]

    def fetch(_api_key: str) -> dict[str, ModelRecord]:
        return _catalog(parameters)

    monkeypatch.setattr("echochain.client.fetch_catalog", fetch)
    assert main(["--start", "Redis uses epoll"]) == _EXIT_CONFIG
    message = capsys.readouterr().err
    assert JUDGE_MODEL in message
    assert "does not support seed" in message
    assert "test-key" not in message


def test_fake_cycle_records_served_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with Recorder(mk_exp_dir("Redis uses epoll")) as recorder:
        cycle("Redis uses epoll", recorder, _FakeChat("preserved answer"))
    experiment = next(Path("experiments").iterdir())
    with (experiment / "chain.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    requested = configured_models()[0]
    assert rows[0]["model"] == requested
    assert rows[0]["served_model"] == f"{requested}-served"
    assert rows[0]["phase"] == "A→Q"
    assert float(rows[0]["score"]) == 1.0
    saved = json.loads((experiment / "chain.jsonl").read_text(encoding="utf-8"))
    assert saved["served_model"] == f"{requested}-served"


def test_sentinel_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "sk-or-sentinel-secret-value"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    load_api_key()
    caplog.set_level(logging.WARNING)
    logging.getLogger("echochain.judge").warning("saw %s", sentinel)
    with Recorder(mk_exp_dir("Redis uses epoll")) as recorder:
        cycle("Redis uses epoll", recorder, _FakeChat(f"answer {sentinel}"))
    written = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("experiments").rglob("*")
        if path.is_file()
    )
    assert sentinel not in written
    assert REDACTED in written
    assert sentinel not in caplog.text
    assert REDACTED in caplog.text


def test_network_is_blocked() -> None:
    with (
        pytest.warns(UserWarning, match="socket.socket"),
        pytest.raises(SocketBlockedError, match="socket.socket"),
    ):
        socket.socket()
