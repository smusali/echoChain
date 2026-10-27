from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, ConfigDict

from echochain.settings import redact

_SLUG_WORDS = 3
_REJECTED = frozenset({"length", "content_filter"})


class RunStatus(StrEnum):
    running = "running"
    finished = "finished"
    failed = "failed"


class TerminationReason(StrEnum):
    threshold = "threshold"
    max_cycles = "max_cycles"
    generation_error = "generation_error"
    judge_error = "judge_error"
    budget_exceeded = "budget_exceeded"
    interrupted = "interrupted"
    insufficient_credits = "insufficient_credits"


class Phase(StrEnum):
    a2q = "a2q"
    q2a = "q2a"
    identity = "identity"
    paraphrase = "paraphrase"


class ScoreKind(StrEnum):
    fidelity = "fidelity"
    answerability = "answerability"


class StopName(StrEnum):
    fixed_length = "fixed_length"
    consecutive_fidelity = "consecutive_fidelity"


class VerdictName(StrEnum):
    supported = "supported"
    contradicted = "contradicted"
    not_mentioned = "not_mentioned"


_COMPLETE = frozenset(
    {
        TerminationReason.threshold,
        TerminationReason.max_cycles,
        TerminationReason.generation_error,
        TerminationReason.judge_error,
        TerminationReason.budget_exceeded,
    }
)
_RESUMABLE = frozenset(
    {TerminationReason.interrupted, TerminationReason.insufficient_credits}
)


class RunDataError(RuntimeError):
    pass


class EmptySeedError(RuntimeError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimRecord(_Model):
    id: str
    text: str


class VerdictRecord(_Model):
    id: str
    verdict: VerdictName


class Parameters(_Model):
    temperature: float
    max_tokens: int
    top_p: float
    sampling_seed: int | None
    seed_honored: bool | None


class Sample(_Model):
    score: int | None
    anchored: int | None
    verdicts: list[VerdictRecord]
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float
    generation_id: str
    requested_model: str
    served_model: str
    provider: str | None
    finish_reason: str
    sampling_seed: int | None
    top_p: float
    seed_honored: bool | None


class JudgeBlock(_Model):
    samples: list[Sample]
    prev_samples: list[Sample]
    errors: int
    fidelity_vs_seed: float | None
    fidelity_spread: float | None
    anchored: float | None
    anchored_spread: float | None
    fidelity_vs_prev: float | None
    anchored_vs_prev: float | None
    answerability: float | None
    answerability_anchored: float | None
    claims: list[VerdictRecord]
    claim_supported: int | None
    claim_contradicted: int | None
    claim_total: int | None
    claim_fidelity: float | None
    contradiction_rate: float | None
    prev_claims: list[VerdictRecord]
    prev_claim_fidelity: float | None
    prev_contradiction_rate: float | None


class BaselineMetrics(_Model):
    cosine_seed: float | None
    cosine_prev: float | None
    entail_seed_forward: float | None
    entail_seed_backward: float | None
    entail_prev_forward: float | None
    entail_prev_backward: float | None
    length_ratio: float | None
    technical_survival: float | None
    generic_share: float | None


class BaselinePins(_Model):
    embedding_model: str | None = None
    embedding_revision: str | None = None
    nli_model: str | None = None
    nli_revision: str | None = None


class Schedule(_Model):
    name: str
    models: list[str]


class StopRule(_Model):
    name: StopName
    threshold: float
    consecutive: int


class Temperatures(_Model):
    transform: float
    judge: float


class PanelMember(_Model):
    model: str
    samples: int


class ProviderUse(_Model):
    model: str
    provider: str


class ErrorCounts(_Model):
    generation: int
    judge: int


class Row(_Model):
    run_id: str
    step: int
    phase: Phase
    condition: str
    schedule: str
    score_kind: ScoreKind
    requested_model: str
    served_model: str
    provider: str | None
    input_text: str
    output_text: str
    finish_reason: str
    parameters: Parameters
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    generation_id: str
    judge: JudgeBlock
    baseline: BaselineMetrics
    timestamp: str


class Manifest(_Model):
    status: RunStatus
    run_id: str
    slug: str
    version: str
    python: str
    os: str
    seed_sha256: str
    seed_text: str
    seed_id: str | None
    claims: list[ClaimRecord]
    condition: str
    schedule: Schedule
    requested_models: list[str]
    served_models: list[str]
    providers: list[ProviderUse]
    temperatures: Temperatures
    top_p: float
    sampling_seed: int | None
    max_cycles: int
    max_tokens: int
    stop_rule: StopRule
    max_judge_error_share: float
    judge_panel: list[PanelMember]
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
    batch_id: str | None
    repeat_index: int | None
    baseline: BaselinePins


@dataclass(frozen=True)
class RunDirectory:
    directory: Path
    run_id: str
    slug: str


@dataclass(frozen=True)
class ResumeState:
    manifest: Manifest
    seed: str
    current: str
    next_step: int
    spent: float
    rows: tuple[Row, ...]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug_for(seed: str) -> str:
    words: list[str] = []
    for word in seed.lower().split():
        cleaned = "".join(char if char.isalnum() else "_" for char in word).strip("_")
        if cleaned:
            words.append(cleaned)
        if len(words) == _SLUG_WORDS:
            break
    return "_".join(words)


def make_run_dir(seed: str, root: Path) -> RunDirectory:
    if seed.strip() == "":
        raise EmptySeedError("seed is empty")
    run_id = str(uuid.uuid4())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{run_id}"
    destination = root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / name
    path.mkdir(exist_ok=False)
    if path.resolve().parent != destination:
        raise OSError("run directory escapes the output root")
    return RunDirectory(directory=path, run_id=run_id, slug=slug_for(seed))


def finished_reason(reason: TerminationReason | None) -> bool:
    return reason in _COMPLETE


def read_manifest(directory: Path) -> Manifest:
    path = directory / "manifest.json"
    if not path.is_file():
        raise RunDataError("manifest.json was not found")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def read_rows(directory: Path) -> tuple[Row, ...]:
    path = directory / "chain.jsonl"
    if not path.is_file():
        return ()
    rows: list[Row] = []
    text = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(text, start=1):
        if line.strip() == "":
            continue
        try:
            rows.append(Row.model_validate_json(line))
        except ValueError as exc:
            raise RunDataError(f"chain.jsonl:{number} is not a row") from exc
    return tuple(rows)


def prepare_resume(directory: Path) -> ResumeState:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RunDataError("manifest.json was not found")
    chain = directory / "chain.jsonl"
    _clip_partial(chain)
    manifest = read_manifest(directory)
    if _closed(manifest):
        raise RunDataError("run is already finished")
    rows = read_rows(directory)
    if sha256_text(manifest.seed_text) != manifest.seed_sha256:
        raise RunDataError("stored seed does not match its hash")
    if rows:
        if rows[0].input_text != manifest.seed_text:
            raise RunDataError("recorded input does not match the seed")
        current = rows[-1].output_text
        next_step = len(rows) + 1
        spent = _spent(rows)
    else:
        current = manifest.seed_text
        next_step = 1
        spent = 0.0
    manifest.status = RunStatus.running
    manifest.ended_at = None
    manifest.error = None
    manifest.termination_reason = None
    return ResumeState(manifest, manifest.seed_text, current, next_step, spent, rows)


def write_manifest(path: Path, manifest: Manifest) -> None:
    _atomic(path, _dumps(manifest))


def error_counts(rows: tuple[Row, ...] | list[Row]) -> ErrorCounts:
    generation = 0
    judge = 0
    for row in rows:
        judge += row.judge.errors
        failed = any(
            sample.error == "generation failed"
            for sample in (*row.judge.samples, *row.judge.prev_samples)
        )
        unusable = row.output_text.strip() == "" or row.finish_reason in _REJECTED
        if failed or (
            unusable and not row.judge.samples and not row.judge.prev_samples
        ):
            generation += 1
    return ErrorCounts(generation=generation, judge=judge)


class Recorder:
    def __init__(
        self,
        directory: Path,
        manifest: Manifest,
        *,
        prior: tuple[Row, ...] | None = None,
    ) -> None:
        self._directory = directory
        self.manifest = manifest
        self._rows: list[Row] = list(prior or ())
        self._stack = ExitStack()
        mode = "a" if prior is not None else "w"
        try:
            self._handle = self._stack.enter_context(
                (directory / "chain.jsonl").open(mode, encoding="utf-8", newline="")
            )
            write_manifest(directory / "manifest.json", manifest)
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
            if exc_type is not None and self.manifest.status == RunStatus.running:
                if issubclass(exc_type, KeyboardInterrupt):
                    self.finish(TerminationReason.interrupted)
                elif exc_val is not None:
                    self.fail(exc_val)
        finally:
            self._stack.close()

    def log(self, row: Row) -> None:
        stored = Row.model_validate(_redact(row.model_dump(mode="json")))
        self._rows.append(stored)
        payload = _redact(stored.model_dump(mode="json"))
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def finish(self, reason: TerminationReason, error: str | None = None) -> None:
        self.manifest.status = RunStatus.finished
        self.manifest.ended_at = utc_now()
        self.manifest.termination_reason = reason
        self.manifest.error = None if error is None else redact(error)
        self._commit()

    def fail(self, exc: BaseException) -> None:
        self.manifest.status = RunStatus.failed
        self.manifest.ended_at = utc_now()
        self.manifest.error = redact(f"{type(exc).__name__}: {exc}")
        self._commit()

    def _commit(self) -> None:
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        served: list[str] = []
        providers: list[ProviderUse] = []
        for row in self._rows:
            prompt_tokens, completion_tokens, cost = _track(
                (prompt_tokens, completion_tokens, cost),
                served,
                providers,
                row.prompt_tokens,
                row.completion_tokens,
                row.cost,
                row.served_model,
                row.requested_model,
                row.provider,
            )
            for sample in (*row.judge.samples, *row.judge.prev_samples):
                prompt_tokens, completion_tokens, cost = _track(
                    (prompt_tokens, completion_tokens, cost),
                    served,
                    providers,
                    sample.prompt_tokens,
                    sample.completion_tokens,
                    sample.cost,
                    sample.served_model,
                    sample.requested_model,
                    sample.provider,
                )
        self.manifest.prompt_tokens = prompt_tokens
        self.manifest.completion_tokens = completion_tokens
        self.manifest.cost = cost
        self.manifest.served_models = served
        self.manifest.providers = providers
        self.manifest.errors = error_counts(self._rows)
        write_manifest(self._directory / "manifest.json", self.manifest)


def _closed(manifest: Manifest) -> bool:
    if finished_reason(manifest.termination_reason):
        return True
    return (
        manifest.status == RunStatus.finished
        and manifest.termination_reason not in _RESUMABLE
    )


def _spent(rows: tuple[Row, ...]) -> float:
    total = 0.0
    for row in rows:
        total += row.cost
        for sample in (*row.judge.samples, *row.judge.prev_samples):
            total += sample.cost
    return total


def _track(
    totals: tuple[int, int, float],
    served: list[str],
    providers: list[ProviderUse],
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    served_model: str,
    requested_model: str,
    provider: str | None,
) -> tuple[int, int, float]:
    if served_model not in served:
        served.append(served_model)
    if provider:
        pair = ProviderUse(model=requested_model, provider=provider)
        if pair not in providers:
            providers.append(pair)
    return (
        totals[0] + prompt_tokens,
        totals[1] + completion_tokens,
        totals[2] + cost,
    )


def _clip_partial(path: Path) -> None:
    if not path.is_file():
        return
    data = path.read_bytes()
    if data == b"" or data.endswith(b"\n"):
        return
    newline = data.rfind(b"\n")
    kept = b"" if newline < 0 else data[: newline + 1]
    path.write_bytes(kept)


def _redact(value: object) -> object:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    return value


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _dumps(model: BaseModel) -> str:
    payload = _redact(model.model_dump(mode="json"))
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
