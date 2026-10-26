from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, TypedDict

from echochain import __version__
from echochain.config import (
    CALL_DEADLINE_SECONDS,
    JUDGE_MODEL,
    JUDGE_SAMPLES,
    JUDGE_TEMPERATURE,
    MAX_CYCLES,
    MAX_TOKENS,
    ROTATING_MODELS,
    STOP_CONSECUTIVE,
    STOP_SCORE,
    TRANSFORM_TEMPERATURE,
    configured_models,
    redact,
)
from echochain.prompts import (
    A_TO_Q_SOURCE,
    ANSWERABILITY_USER_SOURCE,
    IDENTITY_SOURCE,
    JUDGE_SYS,
    JUDGE_USER_SOURCE,
    PARAPHRASE_SOURCE,
    Q_TO_A_SOURCE,
    RUBRIC_VERSION,
)

_SLUG_WORDS = 3
_DEFAULT_CONDITION = "alternate"
_FIXED_LENGTH = "fixed_length"
_CONSECUTIVE_FIDELITY = "consecutive_fidelity"
_GIT_TIMEOUT_SECONDS = 5
_COMPLETE = frozenset(
    {
        "threshold",
        "max_cycles",
        "generation_error",
        "judge_error",
        "budget_exceeded",
    }
)
_PROMPT_KEYS = (
    "a2q",
    "q2a",
    "judge_system",
    "judge_user",
    "answerability",
    "identity",
    "paraphrase",
)

TerminationReason = Literal[
    "threshold",
    "max_cycles",
    "generation_error",
    "judge_error",
    "budget_exceeded",
    "interrupted",
]


class EmptySeedError(RuntimeError):
    pass


class RunDataError(RuntimeError):
    pass


class ParametersRecord(TypedDict):
    temperature: float
    max_tokens: int
    seed: None


class JudgeSampleRecord(TypedDict):
    score: int | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float
    generation_id: str
    requested_model: str
    served_model: str
    provider: str | None
    finish_reason: str


class JudgeRecord(TypedDict):
    samples: list[JudgeSampleRecord]
    errors: int
    fidelity_vs_seed: float | None
    fidelity_vs_prev: None
    answerability: float | None


class StepRecord(TypedDict):
    run_id: str
    step: int
    phase: str
    condition: str
    schedule: str
    score_kind: str
    requested_model: str
    served_model: str
    provider: str | None
    input_text: str
    prompt_sha256: str
    output_text: str
    finish_reason: str
    parameters: ParametersRecord
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    generation_id: str
    judge: JudgeRecord
    timestamp: str


class PromptHashes(TypedDict):
    a2q: str
    q2a: str
    judge_system: str
    judge_user: str
    answerability: str
    identity: str
    paraphrase: str


class Temperatures(TypedDict):
    transform: float
    judge: float


class ScheduleRecord(TypedDict):
    name: str
    models: list[str]


class StopRule(TypedDict):
    name: str
    threshold: float
    consecutive: int


class JudgePanelMember(TypedDict):
    model: str
    samples: int


class ErrorCounts(TypedDict):
    generation: int
    judge: int


class ModelProvider(TypedDict):
    model: str
    provider: str


class Manifest(TypedDict):
    status: str
    run_id: str
    slug: str
    version: str
    git_sha: str | None
    git_dirty: bool | None
    python: str
    os: str
    lock_sha256: str | None
    prompt_sha256: PromptHashes
    rubric_version: str
    seed_sha256: str
    seed_id: str | None
    corpus_version: str | None
    corpus_sha256: str | None
    condition: str
    schedule: ScheduleRecord
    requested_models: list[str]
    served_models: list[str]
    providers: list[ModelProvider]
    temperatures: Temperatures
    sampling_seed: None
    max_cycles: int
    max_tokens: int
    stop_rule: StopRule
    judge_panel: list[JudgePanelMember]
    started_at: str
    ended_at: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float
    errors: ErrorCounts
    termination_reason: TerminationReason | None
    budget: float | None
    deadline_seconds: float
    error: str | None


@dataclass(frozen=True)
class ManifestDesign:
    condition: str = _DEFAULT_CONDITION
    schedule_name: str = _DEFAULT_CONDITION
    schedule_models: tuple[str, ...] | None = None
    transform_temperature: float = TRANSFORM_TEMPERATURE
    seed_id: str | None = None
    corpus_version: str | None = None
    corpus_sha256: str | None = None
    stop_consecutive: int = STOP_CONSECUTIVE
    max_cycles: int | None = None
    max_tokens: int | None = None
    stop_score: float | None = None
    judge_models: tuple[str, ...] | None = None
    judge_samples: int | None = None
    judge_temperature: float | None = None
    deadline_seconds: float | None = None
    rubric_version: str | None = None
    requested_models: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RunDirectory:
    directory: Path
    run_id: str
    slug: str


def sha256_text(text: str) -> str:
    """Hash text.

    Args:
        text: UTF-8 text.

    Returns:
        The SHA-256 hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Return the current UTC time.

    Returns:
        An ISO-8601 timestamp with a Z suffix.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def prompt_hashes() -> PromptHashes:
    """Hash the prompt templates.

    Returns:
        SHA-256 digests for the transform and judge prompts.
    """
    return {
        "a2q": sha256_text(A_TO_Q_SOURCE),
        "q2a": sha256_text(Q_TO_A_SOURCE),
        "judge_system": sha256_text(JUDGE_SYS),
        "judge_user": sha256_text(JUDGE_USER_SOURCE),
        "answerability": sha256_text(ANSWERABILITY_USER_SOURCE),
        "identity": sha256_text(IDENTITY_SOURCE),
        "paraphrase": sha256_text(PARAPHRASE_SOURCE),
    }


def running_manifest(
    *,
    run_id: str,
    slug: str,
    seed: str,
    budget: float | None,
    design: ManifestDesign | None = None,
) -> Manifest:
    """Build the manifest written when a run starts.

    Args:
        run_id: Run UUID.
        slug: Cleaned seed label stored only in the manifest.
        seed: Seed text. Only its hash is stored.
        budget: USD cap. None means no cap.
        design: Condition, schedule, corpus, and stop rule. Defaults to a
            fixed-length alternating run.

    Returns:
        A manifest with status running.
    """
    chosen = ManifestDesign() if design is None else design
    models = list(
        ROTATING_MODELS if chosen.schedule_models is None else chosen.schedule_models
    )
    stop_name = _CONSECUTIVE_FIDELITY if chosen.stop_consecutive > 0 else _FIXED_LENGTH
    root = _project_root()
    git_sha, git_dirty = _git_state(root)
    max_cycles = MAX_CYCLES if chosen.max_cycles is None else chosen.max_cycles
    max_tokens = MAX_TOKENS if chosen.max_tokens is None else chosen.max_tokens
    stop_score = STOP_SCORE if chosen.stop_score is None else chosen.stop_score
    judge_models = (
        (JUDGE_MODEL,) if chosen.judge_models is None else chosen.judge_models
    )
    judge_samples = (
        JUDGE_SAMPLES if chosen.judge_samples is None else chosen.judge_samples
    )
    judge_temperature = (
        JUDGE_TEMPERATURE
        if chosen.judge_temperature is None
        else chosen.judge_temperature
    )
    deadline = (
        CALL_DEADLINE_SECONDS
        if chosen.deadline_seconds is None
        else chosen.deadline_seconds
    )
    rubric = RUBRIC_VERSION if chosen.rubric_version is None else chosen.rubric_version
    if chosen.requested_models is None:
        requested = list(configured_models())
    else:
        requested = list(chosen.requested_models)
    return {
        "status": "running",
        "run_id": run_id,
        "slug": redact(slug),
        "version": __version__,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "python": platform.python_version(),
        "os": platform.platform(),
        "lock_sha256": _lock_hash(root),
        "prompt_sha256": prompt_hashes(),
        "rubric_version": rubric,
        "seed_sha256": sha256_text(seed),
        "seed_id": chosen.seed_id,
        "corpus_version": chosen.corpus_version,
        "corpus_sha256": chosen.corpus_sha256,
        "condition": chosen.condition,
        "schedule": {"name": chosen.schedule_name, "models": models},
        "requested_models": requested,
        "served_models": [],
        "providers": [],
        "temperatures": {
            "transform": chosen.transform_temperature,
            "judge": judge_temperature,
        },
        "sampling_seed": None,
        "max_cycles": max_cycles,
        "max_tokens": max_tokens,
        "stop_rule": {
            "name": stop_name,
            "threshold": stop_score,
            "consecutive": chosen.stop_consecutive,
        },
        "judge_panel": [
            {"model": model, "samples": judge_samples} for model in judge_models
        ],
        "started_at": utc_now(),
        "ended_at": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "errors": {"generation": 0, "judge": 0},
        "termination_reason": None,
        "budget": budget,
        "deadline_seconds": deadline,
        "error": None,
    }


def mk_exp_dir(seed: str, root: Path) -> RunDirectory:
    """Create a unique run directory under the output root.

    Args:
        seed: Seed text. A blank seed is rejected. It is not used in the path.
        root: Directory that receives run folders.

    Returns:
        The new directory, its UUID, and the cleaned slug.

    Raises:
        EmptySeedError: The seed is blank.
    """
    if seed.strip() == "":
        raise EmptySeedError("seed is empty")
    run_id = str(uuid.uuid4())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{run_id}"
    if name != Path(name).name:
        raise OSError("run directory escapes the output root")
    destination = root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / name
    path.mkdir(exist_ok=False)
    if path.resolve().parent != destination:
        raise OSError("run directory escapes the output root")
    return RunDirectory(directory=path, run_id=run_id, slug=_slug(seed))


@dataclass(frozen=True)
class ResumePlan:
    manifest: Manifest
    seed: str
    current: str
    next_step: int
    spent: float
    rows: tuple[Mapping[str, object], ...]


def prepare_resume(directory: Path) -> ResumePlan:
    """Load a run that can continue from its last complete step.

    Args:
        directory: Run directory containing the manifest and the JSONL chain.

    Returns:
        The manifest, the seed, the text to continue from, and the next step.
    """
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RunDataError("manifest.json was not found")
    chain = directory / "chain.jsonl"
    _clip_partial(chain)
    manifest = _load_manifest(manifest_path)
    reason = manifest["termination_reason"]
    if reason in _COMPLETE or (
        manifest["status"] == "finished" and reason != "interrupted"
    ):
        raise RunDataError("run is already finished")
    rows = tuple(_read_chain(chain))
    if not rows:
        raise RunDataError("run has no completed step")
    _require_steps(rows)
    seed = _text(rows[0], "input_text", "recorded input is missing")
    if sha256_text(seed) != manifest["seed_sha256"]:
        raise RunDataError("recorded input does not match the seed hash")
    current = _text(rows[-1], "output_text", "recorded output is missing")
    manifest["status"] = "running"
    manifest["ended_at"] = None
    manifest["error"] = None
    manifest["termination_reason"] = None
    return ResumePlan(
        manifest=manifest,
        seed=seed,
        current=current,
        next_step=len(rows) + 1,
        spent=_spent(rows),
        rows=rows,
    )


class Recorder:
    def __init__(
        self,
        directory: Path,
        manifest: Manifest,
        *,
        prior: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Open the chain and write the starting manifest.

        Args:
            directory: Run directory.
            manifest: Manifest with status running.
            prior: Steps already stored. When set, new steps are appended.
        """
        self.generation_errors = 0
        self.judge_errors = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0
        self._served_models: list[str] = []
        self._providers: list[ModelProvider] = []
        self._directory = directory
        self._manifest = manifest
        self._stack = ExitStack()
        mode = "a" if prior is not None else "w"
        try:
            self.jl = self._stack.enter_context(
                (directory / "chain.jsonl").open(mode, encoding="utf-8", newline="")
            )
            if prior is not None:
                self.generation_errors = manifest["errors"]["generation"]
                self.judge_errors = _judge_errors(prior)
                self._restore(prior)
            _atomic_json(directory / "manifest.json", manifest)
        except Exception:
            self._stack.close()
            raise

    def __enter__(self) -> Recorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is not None and self._manifest["status"] == "running":
                if issubclass(exc_type, KeyboardInterrupt):
                    self.finish("interrupted")
                elif exc_val is not None:
                    self.fail(exc_val)
        finally:
            self._stack.close()

    def log(self, record: StepRecord) -> None:
        """Append one step.

        Args:
            record: Step fields to store. Text fields are redacted.
        """
        row = _redact_row(record)
        self._note(row)
        self.jl.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.jl.flush()
        os.fsync(self.jl.fileno())

    def _restore(self, rows: Sequence[Mapping[str, object]]) -> None:
        for row in rows:
            self._track(
                (
                    _count(row.get("prompt_tokens")),
                    _count(row.get("completion_tokens")),
                    _amount(row.get("cost")),
                ),
                (
                    _string(row.get("served_model"), "served_model"),
                    _string(row.get("requested_model"), "requested_model"),
                    _provider_name(row),
                ),
            )
            for sample in _sample_maps(row.get("judge")):
                self._track(
                    (
                        _count(sample.get("prompt_tokens")),
                        _count(sample.get("completion_tokens")),
                        _amount(sample.get("cost")),
                    ),
                    (
                        _string(sample.get("served_model"), "served_model"),
                        _string(sample.get("requested_model"), "requested_model"),
                        _provider_name(sample),
                    ),
                )

    def finish(self, reason: TerminationReason) -> None:
        """Write the final manifest.

        Args:
            reason: Why the run stopped.
        """
        self._manifest["status"] = "finished"
        self._manifest["ended_at"] = utc_now()
        self._manifest["termination_reason"] = reason
        self._manifest["error"] = None
        self._commit()

    def fail(self, exc: BaseException) -> None:
        """Write a failed manifest.

        Args:
            exc: Exception that stopped the run.
        """
        self._manifest["status"] = "failed"
        self._manifest["ended_at"] = utc_now()
        self._manifest["error"] = redact(f"{type(exc).__name__}: {exc}")
        self._commit()

    def _commit(self) -> None:
        self._manifest["prompt_tokens"] = self._prompt_tokens
        self._manifest["completion_tokens"] = self._completion_tokens
        self._manifest["cost"] = self._cost
        self._manifest["served_models"] = self._served_models
        self._manifest["providers"] = self._providers
        self._manifest["errors"] = {
            "generation": self.generation_errors,
            "judge": self.judge_errors,
        }
        _atomic_json(self._directory / "manifest.json", self._manifest)

    def _note(self, row: StepRecord) -> None:
        self._track(
            (row["prompt_tokens"], row["completion_tokens"], row["cost"]),
            (row["served_model"], row["requested_model"], row["provider"]),
        )
        for sample in row["judge"]["samples"]:
            self._track(
                (
                    sample["prompt_tokens"],
                    sample["completion_tokens"],
                    sample["cost"],
                ),
                (
                    sample["served_model"],
                    sample["requested_model"],
                    sample["provider"],
                ),
            )

    def _track(
        self,
        usage: tuple[int, int, float],
        call: tuple[str, str, str | None],
    ) -> None:
        prompt_tokens, completion_tokens, cost = usage
        served, model, provider = call
        self._add(prompt_tokens, completion_tokens, cost)
        _remember(self._served_models, served)
        _remember_provider(self._providers, model, provider)

    def _add(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._cost += cost


def _atomic_json(path: Path, payload: Manifest) -> None:
    temporary = path.with_suffix(".json.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _slug(seed: str) -> str:
    words: list[str] = []
    for word in seed.lower().split():
        cleaned = "".join(char if char.isalnum() else "_" for char in word).strip("_")
        if cleaned:
            words.append(cleaned)
        if len(words) == _SLUG_WORDS:
            break
    return "_".join(words)


def _redact_row(record: StepRecord) -> StepRecord:
    samples = [_redact_sample(sample) for sample in record["judge"]["samples"]]
    judge: JudgeRecord = {
        "samples": samples,
        "errors": record["judge"]["errors"],
        "fidelity_vs_seed": record["judge"]["fidelity_vs_seed"],
        "fidelity_vs_prev": None,
        "answerability": record["judge"]["answerability"],
    }
    return {
        "run_id": record["run_id"],
        "step": record["step"],
        "phase": redact(record["phase"]),
        "condition": redact(record["condition"]),
        "schedule": redact(record["schedule"]),
        "score_kind": record["score_kind"],
        "requested_model": redact(record["requested_model"]),
        "served_model": redact(record["served_model"]),
        "provider": _redact_optional(record["provider"]),
        "input_text": redact(record["input_text"]),
        "prompt_sha256": record["prompt_sha256"],
        "output_text": redact(record["output_text"]),
        "finish_reason": redact(record["finish_reason"]),
        "parameters": record["parameters"],
        "latency_seconds": record["latency_seconds"],
        "prompt_tokens": record["prompt_tokens"],
        "completion_tokens": record["completion_tokens"],
        "cost": record["cost"],
        "generation_id": redact(record["generation_id"]),
        "judge": judge,
        "timestamp": record["timestamp"],
    }


def _redact_sample(sample: JudgeSampleRecord) -> JudgeSampleRecord:
    return {
        "score": sample["score"],
        "error": _redact_optional(sample["error"]),
        "prompt_tokens": sample["prompt_tokens"],
        "completion_tokens": sample["completion_tokens"],
        "cost": sample["cost"],
        "generation_id": redact(sample["generation_id"]),
        "requested_model": redact(sample["requested_model"]),
        "served_model": redact(sample["served_model"]),
        "provider": _redact_optional(sample["provider"]),
        "finish_reason": redact(sample["finish_reason"]),
    }


def _redact_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return redact(value)


def _remember(items: list[str], value: str) -> None:
    if value != "" and value not in items:
        items.append(value)


def _remember_provider(
    items: list[ModelProvider], model: str, provider: str | None
) -> None:
    if provider is None or provider == "":
        return
    pair: ModelProvider = {"model": model, "provider": provider}
    if pair not in items:
        items.append(pair)


def _clip_partial(path: Path) -> None:
    if not path.is_file():
        return
    data = path.read_bytes()
    if data == b"" or data.endswith(b"\n"):
        return
    newline = data.rfind(b"\n")
    kept = b"" if newline < 0 else data[: newline + 1]
    path.write_bytes(kept)


def _read_chain(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.strip() == "":
            continue
        try:
            parsed: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunDataError(f"chain.jsonl:{line_number} is not JSON") from exc
        rows.append(_mapping(parsed, f"chain.jsonl:{line_number}"))
    return rows


def _require_steps(rows: Sequence[Mapping[str, object]]) -> None:
    for index, row in enumerate(rows, start=1):
        step = row.get("step")
        if isinstance(step, bool) or step != index:
            raise RunDataError("chain is missing a step")


def _text(row: Mapping[str, object], key: str, message: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise RunDataError(message)
    return value


def _sample_maps(judge: object) -> list[Mapping[str, object]]:
    if not isinstance(judge, dict):
        return []
    samples = judge.get("samples")
    if not isinstance(samples, list):
        return []
    return [sample for sample in samples if isinstance(sample, dict)]


def _spent(rows: Sequence[Mapping[str, object]]) -> float:
    total = 0.0
    for row in rows:
        total += _amount(row.get("cost"))
        for sample in _sample_maps(row.get("judge")):
            total += _amount(sample.get("cost"))
    return total


def _judge_errors(rows: Sequence[Mapping[str, object]]) -> int:
    total = 0
    for row in rows:
        judge = row.get("judge")
        if not isinstance(judge, dict):
            continue
        total += _count(judge.get("errors"))
    return total


def _provider_name(row: Mapping[str, object]) -> str | None:
    provider = row.get("provider")
    if isinstance(provider, str):
        return provider
    return None


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunDataError("token count must be an integer")
    return value


def _amount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunDataError("value must be a number")
    return float(value)


def _load_manifest(path: Path) -> Manifest:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunDataError("manifest.json is not JSON") from exc
    raw = _mapping(parsed, "manifest")
    return _manifest(raw)


def _manifest(raw: Mapping[str, object]) -> Manifest:
    if raw.get("sampling_seed") is not None:
        raise RunDataError("sampling_seed must be null")
    schedule = _mapping(raw.get("schedule"), "schedule")
    temperatures = _mapping(raw.get("temperatures"), "temperatures")
    stop = _mapping(raw.get("stop_rule"), "stop_rule")
    errors = _mapping(raw.get("errors"), "errors")
    return {
        "status": _string(raw.get("status"), "status"),
        "run_id": _string(raw.get("run_id"), "run_id"),
        "slug": _string(raw.get("slug"), "slug"),
        "version": _string(raw.get("version"), "version"),
        "git_sha": _optional_string(raw.get("git_sha"), "git_sha"),
        "git_dirty": _optional_bool(raw.get("git_dirty"), "git_dirty"),
        "python": _string(raw.get("python"), "python"),
        "os": _string(raw.get("os"), "os"),
        "lock_sha256": _optional_string(raw.get("lock_sha256"), "lock_sha256"),
        "prompt_sha256": _prompt_hashes(raw.get("prompt_sha256")),
        "rubric_version": _string(raw.get("rubric_version"), "rubric_version"),
        "seed_sha256": _string(raw.get("seed_sha256"), "seed_sha256"),
        "seed_id": _optional_string(raw.get("seed_id"), "seed_id"),
        "corpus_version": _optional_string(raw.get("corpus_version"), "corpus_version"),
        "corpus_sha256": _optional_string(raw.get("corpus_sha256"), "corpus_sha256"),
        "condition": _string(raw.get("condition"), "condition"),
        "schedule": {
            "name": _string(schedule.get("name"), "schedule name"),
            "models": _string_list(schedule.get("models"), "schedule models"),
        },
        "requested_models": _string_list(
            raw.get("requested_models"), "requested_models"
        ),
        "served_models": _string_list(raw.get("served_models"), "served_models"),
        "providers": _providers(raw.get("providers")),
        "temperatures": {
            "transform": _amount(temperatures.get("transform")),
            "judge": _amount(temperatures.get("judge")),
        },
        "sampling_seed": None,
        "max_cycles": _count(raw.get("max_cycles")),
        "max_tokens": _count(raw.get("max_tokens")),
        "stop_rule": {
            "name": _string(stop.get("name"), "stop rule"),
            "threshold": _amount(stop.get("threshold")),
            "consecutive": _count(stop.get("consecutive")),
        },
        "judge_panel": _panel(raw.get("judge_panel")),
        "started_at": _string(raw.get("started_at"), "started_at"),
        "ended_at": _optional_string(raw.get("ended_at"), "ended_at"),
        "prompt_tokens": _count(raw.get("prompt_tokens")),
        "completion_tokens": _count(raw.get("completion_tokens")),
        "cost": _amount(raw.get("cost")),
        "errors": {
            "generation": _count(errors.get("generation")),
            "judge": _count(errors.get("judge")),
        },
        "termination_reason": _termination(raw.get("termination_reason")),
        "budget": _optional_amount(raw.get("budget")),
        "deadline_seconds": _amount(raw.get("deadline_seconds")),
        "error": _optional_string(raw.get("error"), "error"),
    }


def _prompt_hashes(value: object) -> PromptHashes:
    raw = _mapping(value, "prompt_sha256")
    hashes = {key: _string(raw.get(key), key) for key in _PROMPT_KEYS}
    return {
        "a2q": hashes["a2q"],
        "q2a": hashes["q2a"],
        "judge_system": hashes["judge_system"],
        "judge_user": hashes["judge_user"],
        "answerability": hashes["answerability"],
        "identity": hashes["identity"],
        "paraphrase": hashes["paraphrase"],
    }


def _providers(value: object) -> list[ModelProvider]:
    if not isinstance(value, list):
        raise RunDataError("providers must be a list")
    pairs: list[ModelProvider] = []
    for item in value:
        raw = _mapping(item, "provider")
        pairs.append(
            {
                "model": _string(raw.get("model"), "provider model"),
                "provider": _string(raw.get("provider"), "provider name"),
            }
        )
    return pairs


def _panel(value: object) -> list[JudgePanelMember]:
    if not isinstance(value, list):
        raise RunDataError("judge_panel must be a list")
    panel: list[JudgePanelMember] = []
    for item in value:
        raw = _mapping(item, "judge")
        panel.append(
            {
                "model": _string(raw.get("model"), "judge model"),
                "samples": _count(raw.get("samples")),
            }
        )
    return panel


_REASONS: dict[str, TerminationReason] = {
    "threshold": "threshold",
    "max_cycles": "max_cycles",
    "generation_error": "generation_error",
    "judge_error": "judge_error",
    "budget_exceeded": "budget_exceeded",
    "interrupted": "interrupted",
}


def _termination(value: object) -> TerminationReason | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunDataError("termination_reason is unknown")
    reason = _REASONS.get(value)
    if reason is None:
        raise RunDataError("termination_reason is unknown")
    return reason


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RunDataError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RunDataError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RunDataError(f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RunDataError(f"{label} must be a list")
    return [_string(item, label) for item in value]


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RunDataError(f"{label} must be a boolean")
    return value


def _optional_amount(value: object) -> float | None:
    if value is None:
        return None
    return _amount(value)


def _project_root() -> Path | None:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "echochain"
        ).is_dir():
            return candidate
    return None


def _git_state(root: Path | None) -> tuple[str | None, bool | None]:
    if root is None:
        return None, None
    sha = _git(root, "rev-parse", "HEAD")
    if sha is None or sha == "":
        return None, None
    porcelain = _git(root, "status", "--porcelain")
    if porcelain is None:
        return sha, None
    return sha, porcelain != ""


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _lock_hash(root: Path | None) -> str | None:
    if root is None:
        return None
    path = root / "uv.lock"
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()
