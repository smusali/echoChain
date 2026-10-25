from __future__ import annotations

import csv
import hashlib
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
    JUDGE_ERROR_DENOMINATOR,
    JUDGE_ERROR_NUMERATOR,
    JUDGE_MODEL,
    MAX_CYCLES,
    MAX_TOKENS,
    MISSING_API_KEY,
    REDACTED,
    ROTATING_MODELS,
    SCORE_MAX,
    TRANSFORM_TEMPERATURE,
    configured_models,
    load_api_key,
)
from echochain.design import RunSpec, default_spec, load_corpus, model_for, phase_for
from echochain.engine import cycle
from echochain.judge import parse_reply
from echochain.persistence import (
    ManifestDesign,
    Recorder,
    StepRecord,
    mk_exp_dir,
    running_manifest,
)
from jsonschema import Draft202012Validator
from openai import APIStatusError
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from pytest_socket import SocketBlockedError

_EXIT_CONFIG = 2
_EXIT_BUDGET = 3
_EXIT_INTERRUPTED = 130
_MIN_STUDY_SEEDS = 10
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
        *,
        response_format: ResponseFormatJSONSchema | None = None,
    ) -> Completion:
        body = (
            '{"rationale": "preserved", "score": 1}'
            if model == JUDGE_MODEL
            else self._text
        )
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
        "condition": "alternate",
        "schedule": "alternate",
        "score_kind": "answerability",
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
        )
        recorder.finish(reason)
    assert reason == "max_cycles"
    with (run.directory / "chain.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == MAX_CYCLES
    requested = configured_models()[0]
    assert rows[0]["model"] == requested
    assert rows[0]["served_model"] == f"{requested}-served"
    assert rows[0]["phase"] == "a2q"
    assert float(rows[0]["score"]) == 1.0
    saved = _read_jsonl(run.directory / "chain.jsonl")
    assert len(saved) == MAX_CYCLES
    assert saved[0]["served_model"] == f"{requested}-served"
    assert saved[0]["phase"] == "a2q"
    assert saved[0]["score_kind"] == "answerability"
    assert saved[0]["condition"] == "alternate"
    assert saved[0]["schedule"] == "alternate"
    assert saved[1]["phase"] == "q2a"
    assert saved[1]["score_kind"] == "fidelity"
    assert saved[1]["requested_model"] == configured_models()[1]
    for row in saved:
        _validate(row, "row.schema.json")
    finished = _read_manifest(run.directory / "manifest.json")
    _validate(finished, "manifest.schema.json")
    prompt_tokens, completion_tokens, cost = _totals(saved)
    assert finished["prompt_tokens"] == prompt_tokens
    assert finished["completion_tokens"] == completion_tokens
    assert finished["cost"] == cost
    assert finished["termination_reason"] == "max_cycles"
    assert finished["status"] == "finished"
    schedule = _object(finished["schedule"])
    assert schedule["name"] == "alternate"
    assert finished["stop_rule"] == {
        "name": "fixed_length",
        "threshold": 2.0,
        "consecutive": 0,
    }
    assert cost == MAX_CYCLES


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
        assert "response_format" not in body


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
            )
        recorder.finish(reason)
    assert reason == "generation_error"
    assert attempts == _GENERATION_TRIES
    for body in bodies:
        assert body["max_tokens"] == MAX_TOKENS
        assert body["temperature"] == TRANSFORM_TEMPERATURE
        assert "response_format" not in body
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


class _Scripted:
    def __init__(
        self,
        judge_text: str | None = None,
        replies: Sequence[str] | None = None,
        *,
        echo: bool = False,
    ) -> None:
        self._replies = list(replies or [])
        self._judge_text = judge_text
        self._echo = echo
        self.seen: list[str] = []

    def __enter__(self) -> _Scripted:
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
        *,
        response_format: ResponseFormatJSONSchema | None = None,
    ) -> Completion:
        content = messages[-1]["content"]
        if model != JUDGE_MODEL:
            text = content
            if self._echo:
                marker = "Text:\n"
                text = content.split(marker, 1)[1] if marker in content else content
            return _completion(text, model, temperature=temperature)
        self.seen.append(content)
        if self._replies:
            body = self._replies.pop(0)
        elif self._judge_text is not None:
            body = self._judge_text
        else:
            body = '{"rationale": "ok", "score": 8}'
        return _completion(body, model, temperature=temperature)


class _Interrupt:
    def __enter__(self) -> _Interrupt:
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
        *,
        response_format: ResponseFormatJSONSchema | None = None,
    ) -> Completion:
        raise KeyboardInterrupt


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    client: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fetch(_api_key: str) -> dict[str, ModelRecord]:
        return _catalog(_full_parameters())

    monkeypatch.setattr("echochain.client.fetch_catalog", fetch)
    monkeypatch.setattr("echochain.client.OpenRouterClient", lambda _api_key: client)


def _run_cycle(
    tmp_path: Path,
    client: _Scripted,
    spec: RunSpec | None = None,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    plan = default_spec() if spec is None else spec
    run = mk_exp_dir(_SEED, tmp_path)
    manifest = running_manifest(
        run_id=run.run_id,
        slug=run.slug,
        seed=_SEED,
        budget=plan.budget,
        design=ManifestDesign(
            condition=plan.condition.name,
            schedule_name=plan.condition.schedule,
            schedule_models=plan.models,
            transform_temperature=plan.condition.temperature,
            seed_id=plan.seed_id,
            corpus_version=plan.corpus_version,
            corpus_sha256=plan.corpus_sha256,
            stop_consecutive=plan.stop_consecutive,
        ),
    )
    with Recorder(run.directory, manifest) as recorder:
        reason = cycle(
            _SEED,
            recorder,
            client,
            run_id=run.run_id,
            spec=plan,
        )
        recorder.finish(reason)
    saved = _read_jsonl(run.directory / "chain.jsonl")
    for row in saved:
        _validate(row, "row.schema.json")
    finished = _read_manifest(run.directory / "manifest.json")
    _validate(finished, "manifest.schema.json")
    return reason, saved, finished


def _pair_counts(
    schedule: str, models: tuple[str, ...], steps: int
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for step in range(1, steps + 1):
        phase = phase_for("telephone", step)
        key = (model_for(schedule, models, step), phase)
        counts[key] = counts.get(key, 0) + 1
    return counts


@pytest.mark.parametrize(
    ("reply", "score", "error"),
    [
        ("", None, "empty"),
        ("   ", None, "empty"),
        ('```json\n{"rationale": "kept", "score": 8}\n```', 8, None),
        ("the score is high", None, "invalid_json"),
        ('{"rationale": "no", "score": "eleven"}', None, "not_integer"),
        ('{"rationale": "no", "score": "8"}', None, "not_integer"),
        ('{"rationale": "no", "score": 8.5}', None, "not_integer"),
        ('{"rationale": "no", "score": 42}', None, "out_of_range"),
        ('{"rationale": "no", "score": 0}', None, "out_of_range"),
        ('{"rationale": "no"}', None, "missing_key"),
        ('{"score": 8}', None, "missing_key"),
        ('{"rationale": "kept", "score": 8}', 8, None),
        ('{"rationale": "kept", "score": true}', None, "not_integer"),
        ("[1, 2]", None, "not_object"),
    ],
)
def test_parse_reply(reply: str, score: int | None, error: str | None) -> None:
    assert parse_reply(reply) == (score, error)


def test_empty_judge_reply_continues(tmp_path: Path) -> None:
    reason, saved, _finished = _run_cycle(tmp_path, _Scripted(replies=[""]))
    assert reason == "max_cycles"
    assert len(saved) == MAX_CYCLES
    judge = _object(saved[0]["judge"])
    samples = judge["samples"]
    assert isinstance(samples, list)
    sample = _object(samples[0])
    assert sample["score"] is None
    assert sample["error"] == "empty"
    assert judge["answerability"] is None
    assert judge["fidelity_vs_seed"] is None


def test_judge_failures_stop_at_the_share(tmp_path: Path) -> None:
    reason, saved, finished = _run_cycle(tmp_path, _Scripted(judge_text=""))
    stopped = MAX_CYCLES * JUDGE_ERROR_NUMERATOR // JUDGE_ERROR_DENOMINATOR + 1
    assert reason == "judge_error"
    assert len(saved) == stopped
    assert finished["termination_reason"] == "judge_error"
    assert finished["errors"] == {"generation": 0, "judge": stopped}


def test_non_numeric_judge_score_finishes(tmp_path: Path) -> None:
    reason, saved, _finished = _run_cycle(
        tmp_path, _Scripted(judge_text='{"score": "eleven"}')
    )
    assert reason == "judge_error"
    samples = _object(saved[0]["judge"])["samples"]
    assert isinstance(samples, list)
    sample = _object(samples[0])
    assert sample["score"] is None
    assert sample["error"] == "not_integer"


def test_questions_use_the_answerability_rubric(tmp_path: Path) -> None:
    asked = 4
    kept = 9
    client = _Scripted(
        replies=[
            f'{{"rationale": "asks", "score": {asked}}}',
            f'{{"rationale": "kept", "score": {kept}}}',
        ]
    )
    both_phases = 2
    reason, saved, _finished = _run_cycle(
        tmp_path, client, default_spec(max_cycles=both_phases)
    )
    assert reason == "max_cycles"
    assert "Transformed Answer" not in client.seen[0]
    assert "Seed Answer" in client.seen[0]
    assert "Transformed Answer" in client.seen[1]
    assert saved[0]["score_kind"] == "answerability"
    assert saved[1]["score_kind"] == "fidelity"
    question = _object(saved[0]["judge"])
    answer = _object(saved[1]["judge"])
    assert question["answerability"] == asked
    assert question["fidelity_vs_seed"] is None
    assert answer["fidelity_vs_seed"] == kept
    assert answer["answerability"] is None


def test_schedules_fill_the_design_matrix() -> None:
    two = ("a", "b")
    three = ("a", "b", "c")
    assert set(_pair_counts("alternate", two, MAX_CYCLES)) == {
        ("a", "a2q"),
        ("b", "q2a"),
    }
    assert set(_pair_counts("swap", two, MAX_CYCLES)) == {("b", "a2q"), ("a", "q2a")}
    assert set(_pair_counts("factorial", two, MAX_CYCLES)) == {
        ("a", "a2q"),
        ("a", "q2a"),
        ("b", "a2q"),
        ("b", "q2a"),
    }
    cells = len(three) * len({"a2q", "q2a"})
    wide = _pair_counts("factorial", three, MAX_CYCLES)
    assert len(wide) == cells
    assert all(count > 0 for count in wide.values())
    latin = _pair_counts("latin", three, MAX_CYCLES)
    assert len(latin) == cells
    assert len(set(latin.values())) == 1
    assert set(_pair_counts("fixed:only", three, MAX_CYCLES)) == {
        ("only", "a2q"),
        ("only", "q2a"),
    }
    alternate = [model_for("alternate", three, step) for step in range(1, 7)]
    latin_models = [model_for("latin", three, step) for step in range(1, 7)]
    assert alternate != latin_models
    assert model_for("alternate", ROTATING_MODELS, 1) == ROTATING_MODELS[0]


def test_conditions_are_labeled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_client(monkeypatch, tmp_path, _FakeChat("preserved answer"))
    assert main(["--start", _SEED, "--condition", "all", "--out", str(tmp_path)]) == 0
    expected = {"alternate", "same-model", "identity", "paraphrase"}
    runs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(runs) == len(expected)
    labels: set[str] = set()
    for run in runs:
        finished = _read_manifest(run / "manifest.json")
        _validate(finished, "manifest.schema.json")
        saved = _read_jsonl(run / "chain.jsonl")
        _validate(saved[0], "row.schema.json")
        assert saved[0]["condition"] == finished["condition"]
        assert isinstance(finished["condition"], str)
        labels.add(finished["condition"])
        if finished["condition"] == "identity":
            assert saved[0]["phase"] == "identity"
            assert _object(saved[0]["parameters"])["temperature"] == 0
        if finished["condition"] == "paraphrase":
            assert saved[0]["phase"] == "paraphrase"
            assert saved[0]["score_kind"] == "fidelity"
    assert labels == expected


def test_identity_echo_scores_perfectly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_client(
        monkeypatch,
        tmp_path,
        _Scripted(
            judge_text=f'{{"rationale": "same", "score": {SCORE_MAX}}}',
            echo=True,
        ),
    )
    assert (
        main(["--start", _SEED, "--condition", "identity", "--out", str(tmp_path)]) == 0
    )
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    saved = _read_jsonl(run / "chain.jsonl")
    assert len(saved) == MAX_CYCLES
    for row in saved:
        assert row["phase"] == "identity"
        assert row["score_kind"] == "fidelity"
        assert row["output_text"] == _SEED
        assert _object(row["judge"])["fidelity_vs_seed"] == SCORE_MAX
        assert _object(row["parameters"])["temperature"] == 0


def test_low_fidelity_streak_stops(tmp_path: Path) -> None:
    spec = default_spec(stop_consecutive=1)
    reason, saved, finished = _run_cycle(
        tmp_path,
        _Scripted(judge_text='{"rationale": "low", "score": 1}'),
        spec,
    )
    assert reason == "threshold"
    assert len(saved) == spec.stop_consecutive + 1
    assert saved[0]["score_kind"] == "answerability"
    assert saved[1]["score_kind"] == "fidelity"
    assert finished["termination_reason"] == "threshold"
    assert _object(finished["stop_rule"])["consecutive"] == 1


def test_interrupt_records_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_client(monkeypatch, tmp_path, _Interrupt())
    assert main(["--start", _SEED, "--out", str(tmp_path)]) == _EXIT_INTERRUPTED
    assert "interrupted" in capsys.readouterr().err
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    finished = _read_manifest(run / "manifest.json")
    _validate(finished, "manifest.schema.json")
    assert finished["status"] == "finished"
    assert finished["termination_reason"] == "interrupted"


def test_dry_run_counts_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--dry-run", "--start", _SEED, "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "runs: 1" in output
    assert f"calls: {MAX_CYCLES * 2}" in output
    assert "cost_upper_bound_usd: unknown" in output
    assert list(tmp_path.iterdir()) == []


def test_dry_run_bounds_cost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fetch(_api_key: str) -> dict[str, ModelRecord]:
        priced: dict[str, ModelRecord] = {}
        for model_id, record in _catalog(_full_parameters()).items():
            priced[model_id] = ModelRecord(
                model_id=record.model_id,
                parameters=record.parameters,
                prompt_price=0.000001,
                completion_price=0.000002,
            )
        return priced

    monkeypatch.setattr("echochain.client.fetch_catalog", fetch)
    assert main(["--dry-run", "--start", _SEED]) == 0
    expected = MAX_CYCLES * 2 * MAX_TOKENS * 0.000003
    assert f"cost_upper_bound_usd: {expected:.6f}" in capsys.readouterr().out


def test_corpus_is_versioned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = load_corpus()
    ids = [seed.id for seed in corpus.seeds]
    assert len(ids) == len(set(ids))
    assert corpus.version == "v1"
    study = [seed for seed in corpus.seeds if seed.split == "study"]
    held_out = [seed for seed in corpus.seeds if seed.split == "held-out"]
    assert len(study) >= _MIN_STUDY_SEEDS
    assert held_out
    for seed in corpus.seeds:
        assert seed.domain.strip()
        assert seed.text.strip()
    digest = hashlib.sha256(corpus.path.read_bytes()).hexdigest()
    assert corpus.sha256 == digest
    assert load_corpus().sha256 == corpus.sha256
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--dry-run", "--seed", "redis-epoll"]) == 0
    assert f"calls: {MAX_CYCLES * 2}" in capsys.readouterr().out
    assert main(["--dry-run", "--seed", "missing"]) == _EXIT_CONFIG


def test_judge_requests_strict_schema(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(_object(json.loads(request.content)))
        return httpx2.Response(
            _HTTP_OK,
            json=_chat_payload('{"rationale": "ok", "score": 8}'),
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
                spec=default_spec(max_cycles=1),
            )
        recorder.finish(reason)
    assert reason == "max_cycles"
    assert "response_format" not in bodies[0]
    response_format = _object(bodies[1]["response_format"])
    assert response_format["type"] == "json_schema"
    schema = _object(response_format["json_schema"])
    assert schema["strict"] is True
    assert schema["name"] == "judge_score"
    provider = _object(bodies[1]["provider"])
    assert provider["require_parameters"] is True
