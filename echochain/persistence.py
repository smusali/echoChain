from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import uuid
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

_FIELDNAMES = ("cycle", "phase", "model", "served_model", "text", "score")
_SLUG_WORDS = 3
_DEFAULT_CONDITION = "alternate"
_FIXED_LENGTH = "fixed_length"
_CONSECUTIVE_FIDELITY = "consecutive_fidelity"
_GIT_TIMEOUT_SECONDS = 5

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
    providers: list[str]
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


class Recorder:
    def __init__(self, directory: Path, manifest: Manifest) -> None:
        """Open the run files and write the starting manifest.

        Args:
            directory: Run directory.
            manifest: Manifest with status running.
        """
        self.generation_errors = 0
        self.judge_errors = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0
        self._served_models: list[str] = []
        self._providers: list[str] = []
        self._directory = directory
        self._manifest = manifest
        _atomic_json(directory / "manifest.json", manifest)
        self.csv = (directory / "chain.csv").open("w", newline="", encoding="utf-8")
        try:
            self.jl = (directory / "chain.jsonl").open(
                "w", encoding="utf-8", newline=""
            )
        except Exception:
            self.csv.close()
            raise
        self._writer = csv.DictWriter(self.csv, fieldnames=list(_FIELDNAMES))
        self._writer.writeheader()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.csv.close()
        self.jl.close()

    def log(self, record: StepRecord) -> None:
        """Append one step.

        Args:
            record: Step fields to store. Text fields are redacted.
        """
        row = _redact_row(record)
        self._writer.writerow(
            {
                "cycle": row["step"],
                "phase": row["phase"],
                "model": row["requested_model"],
                "served_model": row["served_model"],
                "text": row["output_text"],
                "score": _csv_score(row),
            }
        )
        self.csv.flush()
        self._note(row)
        self.jl.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.jl.flush()
        os.fsync(self.jl.fileno())

    def finish(self, reason: TerminationReason) -> None:
        """Write the final manifest.

        Args:
            reason: Why the run stopped.
        """
        self._manifest["status"] = "finished"
        self._manifest["ended_at"] = utc_now()
        self._manifest["termination_reason"] = reason
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
        self._add(row["prompt_tokens"], row["completion_tokens"], row["cost"])
        _remember(self._served_models, row["served_model"])
        provider = row["provider"]
        if provider is not None:
            _remember(self._providers, provider)
        for sample in row["judge"]["samples"]:
            self._add(
                sample["prompt_tokens"],
                sample["completion_tokens"],
                sample["cost"],
            )
            _remember(self._served_models, sample["served_model"])
            sample_provider = sample["provider"]
            if sample_provider is not None:
                _remember(self._providers, sample_provider)

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


def _csv_score(row: StepRecord) -> str:
    scores = [
        sample["score"]
        for sample in row["judge"]["samples"]
        if sample["score"] is not None
    ]
    if not scores:
        return ""
    return str(statistics.mean(scores))


def _remember(items: list[str], value: str) -> None:
    if value != "" and value not in items:
        items.append(value)


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
