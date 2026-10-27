from __future__ import annotations

import json
import platform
import secrets
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import httpx2

from echochain import __version__
from echochain.baseline import (
    TextScorer,
    lexical_scorer,
    load_scorer,
)
from echochain.client import (
    InsufficientCreditsError,
    ModelRecord,
    OpenRouterClient,
    _ReuseTransport,
    check_models,
    fetch_catalog,
)
from echochain.cost import Work, estimate, render_estimate
from echochain.design import (
    Condition,
    PromptKind,
    RunSpec,
    Seed,
    StepNote,
    load_seeds,
    resolve_conditions,
    select_seeds,
    stop_rule_name,
)
from echochain.engine import cycle
from echochain.judge import Claim
from echochain.records import (
    BaselinePins,
    ClaimRecord,
    ErrorCounts,
    Manifest,
    PanelMember,
    Recorder,
    ResumeState,
    Row,
    RunDirectory,
    RunStatus,
    Schedule,
    StopRule,
    Temperatures,
    TerminationReason,
    finished_reason,
    make_run_dir,
    prepare_resume,
    read_manifest,
    sha256_text,
    utc_now,
)
from echochain.settings import (
    MissingAPIKeyError,
    Settings,
    load_api_key,
    redact,
    settings_from_payload,
    settings_payload,
    with_client_limits,
)

_BATCH_SUFFIX = ".batch.json"
_SAMPLING_LIMIT = 2**31 - 1
_RESUMABLE = frozenset(
    {TerminationReason.interrupted, TerminationReason.insufficient_credits}
)


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Finished:
    directory: Path
    reason: TerminationReason


@dataclass(frozen=True)
class _Cell:
    seed: Seed
    condition: Condition
    repeat_index: int


def plan_cells(settings: Settings) -> tuple[_Cell, ...]:
    if settings.start is None and settings.seed is None and settings.seeds is None:
        raise RunError("provide --start, --seed, or --seeds")
    seeds = _seeds(settings)
    conditions = resolve_conditions(
        settings.condition, settings.models, temperature=settings.temperature
    )
    cells: list[_Cell] = []
    for seed in seeds:
        for condition in conditions:
            for index in range(settings.repeats):
                cells.append(_Cell(seed, condition, index))
    return tuple(cells)


def dry_run(settings: Settings) -> str:
    cells = plan_cells(settings)
    catalog = None
    with suppress(MissingAPIKeyError):
        catalog = _catalog(load_api_key(), settings, None)
    work = tuple(_work(cell) for cell in cells)
    return render_estimate(estimate(work, settings, catalog))


def execute(
    settings: Settings,
    *,
    yes: bool,
    progress: Callable[[StepNote], None] | None = None,
    transport: httpx2.BaseTransport | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Finished:
    cells = plan_cells(settings)
    api_key = load_api_key()
    catalog = _catalog(api_key, settings, transport)
    _confirm(cells, settings, catalog, yes=yes)
    check_models(catalog, (*settings.models, *settings.judge))
    batch_id = str(uuid.uuid4())
    _write_batch(settings, batch_id)
    scorer = _scorer(settings, None)
    last: Finished | None = None
    for cell in cells:
        last = _start(
            cell, settings, api_key, batch_id, scorer, progress, transport, sleep
        )
        if _stops(last.reason):
            return last
    if last is None:
        raise RunError("nothing to run")
    return last


def resume(
    token: str,
    settings: Settings,
    *,
    progress: Callable[[StepNote], None] | None = None,
    transport: httpx2.BaseTransport | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Finished:
    batch_file = _batch_path(token, settings.out)
    if batch_file is not None:
        return _resume_batch(batch_file, settings, progress, transport, sleep)
    directory = _find_run(token, settings.out)
    return _resume_directory(directory, settings, progress, transport, sleep)


def _seeds(settings: Settings) -> tuple[Seed, ...]:
    if settings.start is not None:
        if settings.start.strip() == "":
            raise RunError("seed is empty")
        return (Seed("", "", settings.start),)
    return select_seeds(load_seeds(settings.seeds), settings.seed)


def _work(cell: _Cell) -> Work:
    return Work(cell.seed.text, cell.condition, cell.seed.claims)


def _confirm(
    cells: tuple[_Cell, ...],
    settings: Settings,
    catalog: Mapping[str, ModelRecord],
    *,
    yes: bool,
) -> None:
    bound = estimate(tuple(_work(cell) for cell in cells), settings, catalog)
    if bound.cost is None or bound.cost <= settings.confirm_above_usd or yes:
        return
    raise RunError(
        f"estimated cost {bound.cost:.6f} USD exceeds "
        f"{settings.confirm_above_usd:.6f} USD; pass --yes to start"
    )


def _start(
    cell: _Cell,
    settings: Settings,
    api_key: str,
    batch_id: str,
    scorer: TextScorer,
    progress: Callable[[StepNote], None] | None,
    transport: httpx2.BaseTransport | None,
    sleep: Callable[[float], None] | None,
) -> Finished:
    sampling = settings.sampling_seed
    if sampling is None:
        sampling = secrets.randbelow(_SAMPLING_LIMIT) + 1
    spec = _new_spec(cell, settings, sampling, progress)
    allocated = make_run_dir(cell.seed.text, settings.out)
    manifest = _manifest(allocated, spec, settings, cell, batch_id, scorer)
    reason = _drive(
        allocated.directory,
        manifest,
        cell.seed.text,
        spec,
        settings,
        api_key,
        scorer,
        None,
        transport,
        sleep,
    )
    return Finished(allocated.directory, reason)


def _resume_directory(
    directory: Path,
    settings: Settings,
    progress: Callable[[StepNote], None] | None,
    transport: httpx2.BaseTransport | None,
    sleep: Callable[[float], None] | None,
) -> Finished:
    state = prepare_resume(directory)
    spec = _resume_spec(state, progress)
    tuned = _tuned(settings, state.manifest)
    api_key = load_api_key()
    catalog = _catalog(api_key, tuned, transport)
    check_models(catalog, (*spec.models, *spec.judge_models))
    scorer = _scorer(settings, state.manifest)
    reason = _drive(
        directory,
        state.manifest,
        state.seed,
        spec,
        tuned,
        api_key,
        scorer,
        state.rows,
        transport,
        sleep,
    )
    return Finished(directory, reason)


def _resume_batch(
    path: Path,
    live: Settings,
    progress: Callable[[StepNote], None] | None,
    transport: httpx2.BaseTransport | None,
    sleep: Callable[[float], None] | None,
) -> Finished:
    batch_id, stored = _read_batch(path)
    cells = plan_cells(stored)
    found = _index_runs(stored.out, batch_id)
    last: Finished | None = None
    for cell in cells:
        identity = cell.seed.id or sha256_text(cell.seed.text)
        match = found.get((identity, cell.condition.name, cell.repeat_index))
        if match is None:
            last = _fresh_batch_cell(cell, stored, batch_id, progress, transport, sleep)
        elif _closed(read_manifest(match)):
            stored_reason = read_manifest(match).termination_reason
            if stored_reason is None:
                stored_reason = TerminationReason.max_cycles
            last = Finished(match, stored_reason)
            continue
        else:
            last = _resume_directory(match, live, progress, transport, sleep)
        if _stops(last.reason):
            return last
    if last is None:
        raise RunError("batch has no cells")
    return last


def _fresh_batch_cell(
    cell: _Cell,
    stored: Settings,
    batch_id: str,
    progress: Callable[[StepNote], None] | None,
    transport: httpx2.BaseTransport | None,
    sleep: Callable[[float], None] | None,
) -> Finished:
    api_key = load_api_key()
    catalog = _catalog(api_key, stored, transport)
    check_models(catalog, (*stored.models, *stored.judge))
    return _start(
        cell,
        stored,
        api_key,
        batch_id,
        _scorer(stored, None),
        progress,
        transport,
        sleep,
    )


def _drive(
    directory: Path,
    manifest: Manifest,
    seed: str,
    spec: RunSpec,
    settings: Settings,
    api_key: str,
    scorer: TextScorer,
    prior: tuple[Row, ...] | None,
    transport: httpx2.BaseTransport | None,
    sleep: Callable[[float], None] | None,
) -> TerminationReason:
    with Recorder(directory, manifest, prior=prior) as recorder:
        with OpenRouterClient(
            api_key, settings=settings, transport=transport, sleep=sleep
        ) as client:
            try:
                reason = cycle(
                    seed,
                    recorder,
                    client,
                    run_id=manifest.run_id,
                    spec=spec,
                    scorer=scorer,
                )
            except KeyboardInterrupt:
                recorder.finish(TerminationReason.interrupted)
                return TerminationReason.interrupted
            except InsufficientCreditsError as exc:
                message = redact(f"{type(exc).__name__}: {exc}").splitlines()[0]
                recorder.finish(TerminationReason.insufficient_credits, message)
                return TerminationReason.insufficient_credits
        recorder.finish(reason)
    return reason


def _new_spec(
    cell: _Cell,
    settings: Settings,
    sampling: int,
    progress: Callable[[StepNote], None] | None,
) -> RunSpec:
    return RunSpec(
        condition=cell.condition,
        models=tuple(settings.models),
        max_cycles=settings.max_cycles,
        stop_score=settings.stop_score,
        stop_consecutive=settings.stop_consecutive,
        max_judge_error_share=settings.max_judge_error_share,
        seed_id=cell.seed.id or None,
        budget=settings.budget,
        judge_models=tuple(settings.judge),
        judge_samples=settings.judge_samples,
        judge_temperature=settings.judge_temperature,
        top_p=settings.top_p,
        sampling_seed=sampling,
        claims=cell.seed.claims,
        on_step=progress,
    )


def _resume_spec(
    state: ResumeState, progress: Callable[[StepNote], None] | None
) -> RunSpec:
    manifest = state.manifest
    if not manifest.judge_panel:
        raise RunError("run has no judge panel")
    condition = Condition(
        manifest.condition,
        manifest.schedule.name,
        _prompts(manifest.condition),
        manifest.temperatures.transform,
    )
    return RunSpec(
        condition=condition,
        models=tuple(manifest.schedule.models),
        max_cycles=manifest.max_cycles,
        stop_score=manifest.stop_rule.threshold,
        stop_consecutive=manifest.stop_rule.consecutive,
        max_judge_error_share=manifest.max_judge_error_share,
        seed_id=manifest.seed_id,
        budget=manifest.budget,
        judge_models=tuple(member.model for member in manifest.judge_panel),
        judge_samples=manifest.judge_panel[0].samples,
        judge_temperature=manifest.temperatures.judge,
        top_p=manifest.top_p,
        sampling_seed=manifest.sampling_seed,
        claims=tuple(Claim(item.id, item.text) for item in manifest.claims),
        on_step=progress,
        start_step=state.next_step,
        opening_text=state.current,
        spent=state.spent,
        previous_answer=_previous(state.rows),
    )


def _manifest(
    allocated: RunDirectory,
    spec: RunSpec,
    settings: Settings,
    cell: _Cell,
    batch_id: str,
    scorer: TextScorer,
) -> Manifest:
    pins = scorer.models()
    return Manifest(
        status=RunStatus.running,
        run_id=allocated.run_id,
        slug=allocated.slug,
        version=__version__,
        python=platform.python_version(),
        os=platform.platform(),
        seed_sha256=sha256_text(cell.seed.text),
        seed_text=cell.seed.text,
        seed_id=spec.seed_id,
        claims=[ClaimRecord(id=item.id, text=item.text) for item in spec.claims],
        condition=spec.condition.name,
        schedule=Schedule(name=spec.condition.schedule, models=list(spec.models)),
        requested_models=[*settings.models, *settings.judge],
        served_models=[],
        providers=[],
        temperatures=Temperatures(
            transform=spec.condition.temperature, judge=spec.judge_temperature
        ),
        top_p=spec.top_p,
        sampling_seed=spec.sampling_seed,
        max_cycles=spec.max_cycles,
        max_tokens=settings.max_tokens,
        stop_rule=StopRule(
            name=stop_rule_name(spec.stop_consecutive),
            threshold=spec.stop_score,
            consecutive=spec.stop_consecutive,
        ),
        max_judge_error_share=spec.max_judge_error_share,
        judge_panel=[
            PanelMember(model=model, samples=spec.judge_samples)
            for model in spec.judge_models
        ],
        started_at=utc_now(),
        ended_at=None,
        prompt_tokens=0,
        completion_tokens=0,
        cost=0.0,
        errors=ErrorCounts(generation=0, judge=0),
        termination_reason=None,
        budget=spec.budget,
        deadline_seconds=settings.deadline_seconds,
        error=None,
        batch_id=batch_id,
        repeat_index=cell.repeat_index,
        baseline=BaselinePins.model_validate(dict(pins)),
    )


def _tuned(settings: Settings, manifest: Manifest) -> Settings:
    return with_client_limits(
        replace(
            settings,
            models=tuple(manifest.schedule.models),
            judge=tuple(member.model for member in manifest.judge_panel),
        ),
        max_tokens=manifest.max_tokens,
        deadline_seconds=manifest.deadline_seconds,
    )


def _scorer(settings: Settings, manifest: Manifest | None) -> TextScorer:
    neural = settings.baseline == "neural"
    if manifest is not None:
        neural = manifest.baseline.embedding_model is not None
    if neural:
        return load_scorer()
    return lexical_scorer()


def _prompts(name: str) -> PromptKind:
    if name == "identity":
        return "identity"
    if name == "paraphrase":
        return "paraphrase"
    return "telephone"


def _previous(rows: tuple[Row, ...]) -> str | None:
    found: str | None = None
    for row in rows:
        if row.score_kind == "fidelity":
            found = row.output_text
    return found


def _stops(reason: TerminationReason) -> bool:
    return reason not in {TerminationReason.max_cycles, TerminationReason.threshold}


def _closed(manifest: Manifest) -> bool:
    if finished_reason(manifest.termination_reason):
        return True
    return (
        manifest.status == RunStatus.finished
        and manifest.termination_reason not in _RESUMABLE
    )


def _index_runs(root: Path, batch_id: str) -> dict[tuple[str, str, int], Path]:
    found: dict[tuple[str, str, int], Path] = {}
    if not root.is_dir():
        return found
    for path in root.iterdir():
        if not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        manifest = read_manifest(path)
        if manifest.batch_id != batch_id or manifest.repeat_index is None:
            continue
        identity = manifest.seed_id or sha256_text(manifest.seed_text)
        found[(identity, manifest.condition, manifest.repeat_index)] = path
    return found


def _write_batch(settings: Settings, batch_id: str) -> None:
    settings.out.mkdir(parents=True, exist_ok=True)
    path = settings.out / f"{batch_id}{_BATCH_SUFFIX}"
    body = {"batch_id": batch_id, "settings": settings_payload(settings)}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def _read_batch(path: Path) -> tuple[str, Settings]:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunError("batch file is invalid") from exc
    if not isinstance(parsed, dict):
        raise RunError("batch file is invalid")
    batch_id = parsed.get("batch_id")
    raw = parsed.get("settings")
    if not isinstance(batch_id, str) or not isinstance(raw, dict):
        raise RunError("batch file is invalid")
    table: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise RunError("batch file is invalid")
        table[key] = value
    return batch_id, settings_from_payload(table, source=path.name)


def _find_run(token: str, root: Path) -> Path:
    if token.strip() == "":
        raise RunError("run was not found")
    candidate = Path(token)
    if (candidate / "manifest.json").is_file():
        return candidate
    if root.is_dir():
        matches = [
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path.name == token or path.name.endswith(f"_{token}"))
            and (path / "manifest.json").is_file()
        ]
        if len(matches) == 1:
            return matches[0]
    raise RunError(f"run {token} was not found")


def _batch_path(token: str, root: Path) -> Path | None:
    candidate = Path(token)
    if candidate.is_file() and candidate.name.endswith(_BATCH_SUFFIX):
        return candidate
    direct = root / f"{token}{_BATCH_SUFFIX}"
    if direct.is_file():
        return direct
    return None


def _catalog(
    api_key: str, settings: Settings, transport: httpx2.BaseTransport | None
) -> dict[str, ModelRecord]:
    if transport is None:
        return fetch_catalog(api_key, settings=settings)
    http = httpx2.Client(transport=_ReuseTransport(transport))
    try:
        return fetch_catalog(api_key, settings=settings, client=http)
    finally:
        http.close()
