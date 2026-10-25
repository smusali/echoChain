from __future__ import annotations

import csv
import io
import json
import logging
import os
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, tzinfo
from pathlib import Path
from types import TracebackType

import httpx2
import pytest
from echochain import __version__
from echochain.cli import main
from echochain.client import (
    CallDeadlineError,
    ChatMessage,
    Completion,
    ModelRecord,
    OpenRouterClient,
    Usage,
    parse_catalog,
)
from echochain.config import (
    JUDGE_MODEL,
    MAX_TOKENS,
    MISSING_API_KEY,
    REDACTED,
    TRANSFORM_TEMPERATURE,
    configured_models,
    load_api_key,
)
from echochain.engine import cycle
from echochain.persistence import Recorder, StepRecord, mk_exp_dir, running_manifest
from jsonschema import Draft202012Validator
from openai import APIStatusError
from pytest_socket import SocketBlockedError

_EXIT_CONFIG = 2
_EXIT_BUDGET = 3
_SEED = "Redis uses epoll"
_HTTP_OK = 200
_SLOW_CHUNKS = 40
_GENERATION_TRIES = 2
_ATTEMPT_COST = 0.25


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


def _completion(
    text: str,
    model: str,
    *,
    temperature: float,
    cost: float = 0.0,
) -> Completion:
    return Completion(
        text=text,
        finish_reason="stop",
        requested_model=model,
        served_model=f"{model}-served",
        provider="Example",
        generation_id=f"gen-{model}",
        usage=Usage(prompt_tokens=3, completion_tokens=4, cost=cost),
        latency_seconds=0.01,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        sampling_seed=None,
    )


class _FakeChat:
    def __init__(self, text: str, *, cost: float = 0.0) -> None:
        self._text = text
        self._cost = cost

    def __enter__(self) -> _FakeChat:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
    ) -> Completion:
        body = '{"score": 1}' if model == JUDGE_MODEL else self._text
        return _completion(body, model, temperature=temperature, cost=self._cost)


def _schema(name: str) -> dict[str, object]:
    payload: object = json.loads(
        (_repo_root() / "schemas" / name).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError("schema is not an object")
    mapped: dict[str, object] = {}
    for key, item in payload.items():
        if not isinstance(key, str):
            raise TypeError("schema key is not a string")
        mapped[key] = item
    return mapped


def _validate(instance: object, name: str) -> None:
    Draft202012Validator(
        _schema(name),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(instance)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("value is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("key is not a string")
        mapped[key] = item
    return mapped


def _read_manifest(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "":
            continue
        rows.append(_object(json.loads(line)))
    return rows


def _whole(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value is not an integer")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("value is not a number")
    return float(value)


def _totals(rows: Sequence[Mapping[str, object]]) -> tuple[int, int, float]:
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    for row in rows:
        prompt_tokens += _whole(row["prompt_tokens"])
        completion_tokens += _whole(row["completion_tokens"])
        cost += _number(row["cost"])
        judge = _object(row["judge"])
        samples = judge["samples"]
        if not isinstance(samples, list):
            raise TypeError("samples are not a list")
        for sample in samples:
            fields = _object(sample)
            prompt_tokens += _whole(fields["prompt_tokens"])
            completion_tokens += _whole(fields["completion_tokens"])
            cost += _number(fields["cost"])
    return prompt_tokens, completion_tokens, cost


def _chat_payload(
    content: str | None,
    *,
    finish_reason: str = "stop",
    cost: float = 0.25,
) -> dict[str, object]:
    return {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 1,
        "model": "served-model",
        "provider": "Example",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "cost": cost,
        },
    }


def _step(run_id: str, text: str) -> StepRecord:
    return {
        "run_id": run_id,
        "step": 1,
        "phase": "a2q",
        "requested_model": "model",
        "served_model": "model-served",
        "provider": "Example",
        "input_text": _SEED,
        "prompt_sha256": "a" * 64,
        "output_text": text,
        "finish_reason": "stop",
        "parameters": {
            "temperature": TRANSFORM_TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "seed": None,
        },
        "latency_seconds": 0.01,
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "cost": 0.5,
        "generation_id": "gen-test",
        "judge": {
            "samples": [],
            "errors": 0,
            "fidelity_vs_seed": None,
            "fidelity_vs_prev": None,
            "answerability": None,
        },
        "timestamp": "2026-01-02T03:04:05Z",
    }


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
    assert "--out" in captured.out
    assert "--budget" in captured.out


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
    assert main(["--start", _SEED]) == _EXIT_CONFIG
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
    assert main(["--start", _SEED]) == _EXIT_CONFIG
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
    assert main(["--start", _SEED]) == _EXIT_CONFIG
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
    run = mk_exp_dir(_SEED, Path("runs"))
    manifest = running_manifest(
        run_id=run.run_id,
        slug=run.slug,
        seed=_SEED,
        budget=None,
    )
    with Recorder(run.directory, manifest) as recorder:
        reason = cycle(
            _SEED,
            recorder,
            _FakeChat("preserved answer", cost=0.5),
            run_id=run.run_id,
            budget=None,
        )
        recorder.finish(reason)
    assert reason == "threshold"
    with (run.directory / "chain.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    requested = configured_models()[0]
    assert rows[0]["model"] == requested
    assert rows[0]["served_model"] == f"{requested}-served"
    assert rows[0]["phase"] == "a2q"
    assert float(rows[0]["score"]) == 1.0
    saved = _read_jsonl(run.directory / "chain.jsonl")
    assert len(saved) == 1
    assert saved[0]["served_model"] == f"{requested}-served"
    assert saved[0]["phase"] == "a2q"
    _validate(saved[0], "row.schema.json")
    finished = _read_manifest(run.directory / "manifest.json")
    _validate(finished, "manifest.schema.json")
    prompt_tokens, completion_tokens, cost = _totals(saved)
    assert finished["prompt_tokens"] == prompt_tokens
    assert finished["completion_tokens"] == completion_tokens
    assert finished["cost"] == cost
    assert finished["termination_reason"] == "threshold"
    assert finished["status"] == "finished"
    assert cost == 1.0


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
    run = mk_exp_dir(_SEED, Path("runs"))
    manifest = running_manifest(
        run_id=run.run_id,
        slug=run.slug,
        seed=_SEED,
        budget=None,
    )
    with Recorder(run.directory, manifest) as recorder:
        cycle(
            _SEED,
            recorder,
            _FakeChat(f"answer {sentinel}"),
            run_id=run.run_id,
            budget=None,
        )
    written = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("runs").rglob("*")
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


@pytest.mark.parametrize(
    ("statuses", "retry_after", "deadline", "expected"),
    [
        ([401], None, 5.0, "status"),
        ([404], None, 5.0, "status"),
        ([429, 200], "0.2", 5.0, "slept"),
        ([503, 503, 200], "0", 5.0, "ok"),
        ([503, 503, 503], "0", 5.0, "status"),
        ([429], "30", 0.05, "deadline"),
    ],
)
def test_retries_follow_status(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[int],
    retry_after: str | None,
    deadline: float,
    expected: str,
) -> None:
    attempts = 0
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        parsed = _object(json.loads(request.content))
        bodies.append(parsed)
        status = statuses[attempts - 1]
        if status == _HTTP_OK:
            return httpx2.Response(_HTTP_OK, json=_chat_payload("hello"))
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["retry-after"] = retry_after
        return httpx2.Response(
            status,
            json={"error": {"message": "no", "type": "error"}},
            headers=headers,
        )

    slept: list[float] = []
    monkeypatch.setattr("echochain.client.time.sleep", slept.append)
    transport = httpx2.MockTransport(handler)
    with OpenRouterClient("test-key", transport=transport, deadline=deadline) as client:
        if expected == "deadline":
            with pytest.raises(CallDeadlineError):
                client.chat("model", [{"role": "user", "content": "hi"}], 0.3)
        elif expected == "status":
            with pytest.raises(APIStatusError):
                client.chat("model", [{"role": "user", "content": "hi"}], 0.3)
        else:
            reply = client.chat("model", [{"role": "user", "content": "hi"}], 0.3)
            assert reply.text == "hello"
    assert attempts == len(statuses)
    if expected == "slept":
        assert slept == [0.2]
    if expected == "deadline":
        assert slept == []
    for body in bodies:
        assert body["max_tokens"] == MAX_TOKENS
        assert body["temperature"] == TRANSFORM_TEMPERATURE


def test_slow_stream_stops_at_deadline() -> None:
    class Slow(httpx2.SyncByteStream):
        def __init__(self) -> None:
            self.produced = 0

        def __iter__(self) -> Iterator[bytes]:
            for _ in range(_SLOW_CHUNKS):
                time.sleep(0.05)
                self.produced += 1
                yield b"x"

        def close(self) -> None:
            return None

    stream = Slow()

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=stream)

    deadline = 0.12
    started = time.monotonic()
    with (
        OpenRouterClient(
            "test-key",
            transport=httpx2.MockTransport(handler),
            deadline=deadline,
        ) as client,
        pytest.raises(CallDeadlineError),
    ):
        client.chat("model", [{"role": "user", "content": "hi"}], 0.3)
    elapsed = time.monotonic() - started
    assert stream.produced < _SLOW_CHUNKS
    assert elapsed < deadline + 0.35


def test_client_reuses_one_pool() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_chat_payload("hello"))

    with OpenRouterClient(
        "test-key",
        transport=httpx2.MockTransport(handler),
        deadline=5.0,
    ) as client:
        http = client._http
        assert http is not None
        client.chat("model", [{"role": "user", "content": "one"}], 0.3)
        client.chat("model", [{"role": "user", "content": "two"}], 0.0)
        assert client._http is http
        assert not http.is_closed
    assert http.is_closed


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [
        (None, "stop"),
        ("partial", "length"),
        ("blocked", "content_filter"),
    ],
)
def test_bad_completion_retries_once(
    tmp_path: Path,
    content: str | None,
    finish_reason: str,
) -> None:
    attempts = 0
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        bodies.append(_object(json.loads(request.content)))
        return httpx2.Response(
            _HTTP_OK,
            json=_chat_payload(
                content, finish_reason=finish_reason, cost=_ATTEMPT_COST
            ),
        )

    run = mk_exp_dir(_SEED, tmp_path)
    manifest = running_manifest(
        run_id=run.run_id,
        slug=run.slug,
        seed=_SEED,
        budget=None,
    )
    with Recorder(run.directory, manifest) as recorder:
        with OpenRouterClient(
            "test-key",
            transport=httpx2.MockTransport(handler),
            deadline=5.0,
        ) as client:
            reason = cycle(
                _SEED,
                recorder,
                client,
                run_id=run.run_id,
                budget=None,
            )
        recorder.finish(reason)
    assert reason == "generation_error"
    assert attempts == _GENERATION_TRIES
    for body in bodies:
        assert body["max_tokens"] == MAX_TOKENS
        assert body["temperature"] == TRANSFORM_TEMPERATURE
    finished = _read_manifest(run.directory / "manifest.json")
    _validate(finished, "manifest.schema.json")
    assert finished["termination_reason"] == "generation_error"
    assert finished["errors"] == {"generation": 1, "judge": 0}
    saved = _read_jsonl(run.directory / "chain.jsonl")
    assert len(saved) == 1
    _validate(saved[0], "row.schema.json")
    prompt_tokens, completion_tokens, cost = _totals(saved)
    assert finished["prompt_tokens"] == prompt_tokens
    assert finished["completion_tokens"] == completion_tokens
    assert finished["cost"] == cost
    assert cost == _ATTEMPT_COST * _GENERATION_TRIES


def test_budget_stops_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert main(["--start", _SEED, "--budget", "-1"]) == _EXIT_CONFIG
    assert "budget must be zero or greater" in capsys.readouterr().err

    def fetch(_api_key: str) -> dict[str, ModelRecord]:
        return _catalog(_full_parameters())

    monkeypatch.setattr("echochain.client.fetch_catalog", fetch)
    monkeypatch.setattr(
        "echochain.client.OpenRouterClient",
        lambda _api_key: _FakeChat("preserved answer", cost=1.0),
    )
    assert main(["--start", _SEED, "--budget", "0.5"]) == _EXIT_BUDGET
    message = capsys.readouterr().err
    assert "budget_exceeded" in message
    run = next(Path("runs").iterdir())
    finished = _read_manifest(run / "manifest.json")
    saved = _read_jsonl(run / "chain.jsonl")
    _validate(finished, "manifest.schema.json")
    assert len(saved) == 1
    _validate(saved[0], "row.schema.json")
    assert finished["termination_reason"] == "budget_exceeded"
    prompt_tokens, completion_tokens, cost = _totals(saved)
    assert finished["prompt_tokens"] == prompt_tokens
    assert finished["completion_tokens"] == completion_tokens
    assert finished["cost"] == cost
    assert cost == 1.0


def test_seed_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "out"
    run = mk_exp_dir("../../escape here now", root)
    assert run.directory.resolve().is_relative_to(root.resolve())
    assert ".." not in run.directory.parts
    assert run.slug == "escape_here_now"


def test_same_second_runs_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Frozen(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Frozen:
            return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr("echochain.persistence.datetime", Frozen)
    root = tmp_path / "out"
    first = mk_exp_dir("same seed here", root)
    second = mk_exp_dir("same seed here", root)
    assert first.directory != second.directory
    assert first.directory.is_dir()
    assert second.directory.is_dir()


def test_empty_seed_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--start", "   "]) == _EXIT_CONFIG
    assert "seed is empty" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_hostile_seeds_stay_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "out"
    seeds = [
        "..",
        "../..",
        "/tmp/absolute",
        "a/b",
        "a\\b",
        "café",
        "\x00",
        "...",
        "\u200b",
        "a\nb",
    ]
    for seed in seeds:
        run = mk_exp_dir(seed, root)
        resolved = run.directory.resolve()
        assert resolved.is_relative_to(root.resolve())
        assert resolved.parent == root.resolve()


def test_utf8_row_when_default_encoding_is_cp1252(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_open = io.open

    def guarded(
        file: object,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> object:
        if "b" not in mode and encoding in {None, "locale"}:
            encoding = "cp1252"
        opener: Callable[..., object] = real_open
        return opener(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded)
    run = mk_exp_dir(_SEED, tmp_path)
    manifest = running_manifest(
        run_id=run.run_id,
        slug=run.slug,
        seed=_SEED,
        budget=None,
    )
    with Recorder(run.directory, manifest) as recorder:
        recorder.log(_step(run.run_id, "café"))
    raw = (run.directory / "chain.jsonl").read_bytes()
    assert "café".encode() in raw
    saved = _object(json.loads(raw.decode("utf-8")))
    assert saved["output_text"] == "café"
    assert saved["phase"] == "a2q"


def test_crash_leaves_running_manifest(tmp_path: Path) -> None:
    run = mk_exp_dir(_SEED, tmp_path)
    manifest = running_manifest(
        run_id=run.run_id,
        slug=run.slug,
        seed=_SEED,
        budget=None,
    )
    with pytest.raises(RuntimeError, match="boom"), Recorder(run.directory, manifest):
        raise RuntimeError("boom")
    payload = _read_manifest(run.directory / "manifest.json")
    _validate(payload, "manifest.schema.json")
    assert payload["status"] == "running"
    assert payload["termination_reason"] is None
    assert payload["ended_at"] is None
