from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import secrets
import statistics
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from echochain import __version__
from echochain import client as client_mod
from echochain.analyze import AnalyzeError, degradation_step, report
from echochain.baseline import (
    BaselineUnavailable,
    TextScorer,
    lexical_scorer,
    load_scorer,
)
from echochain.client import ModelRecord
from echochain.config import (
    ConfigError,
    MissingAPIKeyError,
    Settings,
    load_api_key,
    redact,
    render_settings,
    resolve_settings,
)
from echochain.design import (
    Condition,
    Corpus,
    DesignError,
    RunSpec,
    Seed,
    SeedRef,
    StepNote,
    condition_table,
    default_spec,
    load_corpus,
    model_for,
    phase_for,
    planned_calls,
    render_transform,
    resolve_conditions,
    score_kind,
    select_seeds,
)
from echochain.engine import cycle
from echochain.judge import Claim, claim_block
from echochain.persistence import (
    EmptySeedError,
    Manifest,
    ManifestDesign,
    Recorder,
    ResumePlan,
    RunDataError,
    TerminationReason,
    finished_reason,
    mk_exp_dir,
    prepare_resume,
    read_manifest,
    running_manifest,
    sha256_text,
)
from echochain.prompts import (
    ANSWERABILITY_USER_TMPL,
    JUDGE_SYS,
    JUDGE_USER_TMPL,
    render_text,
)

_EXIT_BUDGET = 3
_EXIT_CONFIG = 2
_EXIT_INTERRUPTED = 130
_BATCH_SUFFIX = ".batch.json"
_FRAMING_TOKENS = 16
_COMMANDS = frozenset({"run", "config", "export", "analyze", "models"})
_FORMAT = argparse.RawDescriptionHelpFormatter
_EPILOG = """\
examples:
  echochain --start TEXT --out runs --budget 1
  echochain run --seeds seeds/v1.toml --split study
  echochain run --resume RUN_ID
  echochain run --resume BATCH_ID --out runs

exit codes:
  0    completed, the threshold or the cycle cap was reached
  1    runtime or API failure
  2    usage or configuration error
  3    budget exceeded or insufficient credits
  130  interrupted
"""
_CSV_FIELDS = ("cycle", "phase", "model", "served_model", "text", "score")


class UsageError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedSeed:
    seed_id: str | None
    text: str
    corpus_version: str | None
    corpus_sha256: str | None
    claims: tuple[Claim, ...] = ()


@dataclass(frozen=True)
class Batch:
    settings: Settings
    seeds: tuple[PlannedSeed, ...]
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class Cell:
    seed: PlannedSeed
    condition: Condition
    repeat_index: int


def main(argv: list[str] | None = None) -> int:
    """Run one echoChain command.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. `0` is a completed run. `1` is a runtime or API
        failure. `2` is a usage or configuration error. `3` means spending
        stopped. `130` means the run was interrupted.
    """
    _configure_logging()
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(_normalize(raw))
        return _dispatch(args)
    except (
        EmptySeedError,
        UsageError,
        DesignError,
        MissingAPIKeyError,
        ConfigError,
        RunDataError,
        client_mod.ModelUnavailableError,
    ) as exc:
        return _fail(str(exc), _EXIT_CONFIG)
    except client_mod.ModelCatalogError as exc:
        return _fail(str(exc), 1)
    except KeyboardInterrupt:
        return _EXIT_INTERRUPTED
    except Exception:
        logging.getLogger("echochain").exception("run failed")
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.quiet and args.verbose:
        return _fail("--quiet and --verbose cannot be combined", _EXIT_CONFIG)
    if args.command == "export":
        return _export(args.directory)
    if args.command == "analyze":
        return _analyze(args.directory, args.out)
    overrides = _overrides(args) if args.command == "run" else None
    settings = resolve_settings(config=args.config, overrides=overrides)
    if args.command == "config":
        print(render_settings(settings), end="")
        return 0
    if args.command == "models":
        return _models(settings)
    return _run(args, settings)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echochain",
        description="Measure information degradation through LLM telephone.",
        epilog=_EPILOG,
        formatter_class=_FORMAT,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"echochain {__version__}",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide step progress.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print token counts and cost with each step.",
    )
    parser.add_argument("--config", type=Path, help="Experiment TOML file.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_flags(
        sub.add_parser(
            "run",
            help="Run one seed or a batch.",
            epilog=_EPILOG,
            formatter_class=_FORMAT,
        )
    )
    config = sub.add_parser("config", help="Show the resolved configuration.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("show", help="Print the resolved configuration.")
    export = sub.add_parser("export", help="Write CSV derived from a run's JSONL.")
    export.add_argument("directory", type=Path, metavar="RUN")
    export.add_argument("--format", default="csv", choices=("csv",))
    analyze = sub.add_parser(
        "analyze",
        help="Summarize claim fidelity, baseline curves, survival, and log-rank tests.",
    )
    analyze.add_argument("directory", nargs="+", type=Path, metavar="RUN")
    analyze.add_argument("--out", type=Path, help="Directory for tables and figures.")
    sub.add_parser("models", help="Check configured models against the catalog.")
    return parser


def _add_run_flags(run: argparse.ArgumentParser) -> None:
    run.add_argument("--start", help="Seed answer text.")
    run.add_argument("--seed", help="Seed id from the corpus.")
    run.add_argument("--seeds", type=Path, help="Seed corpus TOML.")
    run.add_argument(
        "--preregistration",
        type=Path,
        help="Pre-registration file hashed onto the manifest.",
    )
    run.add_argument("--corpus", type=Path, help="Seed corpus TOML.")
    run.add_argument("--split", choices=("study", "held-out"), help="Corpus split.")
    run.add_argument(
        "--condition",
        action="append",
        help="Condition name. Repeat to run several. 'all' runs the controls.",
    )
    run.add_argument("--repeats", type=int, help="Repetitions of each pair.")
    run.add_argument("--sampling-seed", type=int, help="Base sampling seed.")
    run.add_argument("--top-p", type=float, help="Nucleus sampling cutoff.")
    run.add_argument("--dry-run", action="store_true", help="Print the cost bound.")
    run.add_argument("--yes", action="store_true", help="Start a run over the cap.")
    run.add_argument("--out", type=Path, help="Directory that receives run folders.")
    run.add_argument("--budget", type=float, help="Stop at this many USD.")
    run.add_argument("--model", action="append", dest="models", help="Rotating model.")
    run.add_argument("--judge", action="append", help="Judge model.")
    run.add_argument("--judge-samples", type=int, help="Samples per judge.")
    run.add_argument("--judge-temperature", type=float, help="Judge temperature.")
    run.add_argument("--temperature", type=float, help="Transform temperature.")
    run.add_argument("--max-cycles", type=int, help="Planned steps in one run.")
    run.add_argument("--max-tokens", type=int, help="Completion token cap.")
    run.add_argument("--stop-score", type=float, help="Low fidelity threshold.")
    run.add_argument("--stop-consecutive", type=int, help="Low scores that stop a run.")
    run.add_argument("--score-min", type=int, help="Lowest accepted judge score.")
    run.add_argument("--score-max", type=int, help="Highest accepted judge score.")
    run.add_argument("--judge-error-numerator", type=int, help="Judge error numerator.")
    run.add_argument(
        "--judge-error-denominator", type=int, help="Judge error denominator."
    )
    run.add_argument(
        "--deadline",
        type=float,
        dest="deadline_seconds",
        help="Seconds allowed for one call.",
    )
    run.add_argument("--base-url", help="OpenRouter API base URL.")
    run.add_argument("--http-referer", help="HTTP-Referer header.")
    run.add_argument("--title", help="X-Title header.")
    run.add_argument(
        "--require",
        action="append",
        dest="required_parameters",
        help="Parameter a model must advertise.",
    )
    run.add_argument("--rubric-version", help="Rubric version stored on the run.")
    run.add_argument(
        "--confirm-above",
        type=float,
        dest="confirm_above_usd",
        help="USD estimate that requires --yes.",
    )
    run.add_argument(
        "--provider",
        action="append",
        help="Pin as model=provider,provider|true.",
    )
    run.add_argument(
        "--resume",
        help="Continue a run id, a run directory, or a batch id.",
    )


def _normalize(argv: list[str]) -> list[str]:
    if not argv:
        return ["run"]
    if argv[0] in {"-h", "--help", "--version"}:
        return argv
    leading: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"--quiet", "--verbose", "--version"}:
            leading.append(token)
            index += 1
            continue
        if token == "--config":
            leading.extend(argv[index : index + 2])
            index += 2
            continue
        if token.startswith("--config="):
            leading.append(token)
            index += 1
            continue
        rest.append(token)
        index += 1
    if not rest:
        return [*leading, "run"]
    if rest[0] not in _COMMANDS:
        rest = ["run", *rest]
    return [*leading, *rest]


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}

    def put(name: str, value: object) -> None:
        if value is not None:
            values[name] = value

    put("start", args.start)
    put("seed", args.seed)
    put("split", args.split)
    if args.seeds is not None and args.corpus is not None:
        raise UsageError("--seeds and --corpus cannot both be set")
    if args.seeds is not None:
        put("seeds", args.seeds)
    elif args.corpus is not None:
        put("seeds", args.corpus)
    put("preregistration", args.preregistration)
    put("condition", args.condition)
    put("repeats", args.repeats)
    put("sampling_seed", args.sampling_seed)
    put("top_p", args.top_p)
    put("out", args.out)
    put("budget", args.budget)
    put("models", args.models)
    put("judge", args.judge)
    put("judge_samples", args.judge_samples)
    put("judge_temperature", args.judge_temperature)
    put("temperature", args.temperature)
    put("max_cycles", args.max_cycles)
    put("max_tokens", args.max_tokens)
    put("stop_score", args.stop_score)
    put("stop_consecutive", args.stop_consecutive)
    put("score_min", args.score_min)
    put("score_max", args.score_max)
    put("judge_error_numerator", args.judge_error_numerator)
    put("judge_error_denominator", args.judge_error_denominator)
    put("deadline_seconds", args.deadline_seconds)
    put("base_url", args.base_url)
    put("http_referer", args.http_referer)
    put("title", args.title)
    put("required_parameters", args.required_parameters)
    put("rubric_version", args.rubric_version)
    put("confirm_above_usd", args.confirm_above_usd)
    if args.provider:
        values["providers"] = ";".join(args.provider)
    return values


def _run(args: argparse.Namespace, settings: Settings) -> int:
    progress = None if args.quiet else _on_step(verbose=args.verbose)
    _preregistration_hash(settings.preregistration)
    _require_baseline(settings)
    if args.resume:
        return _resume(args, settings, progress)
    batch = _batch(settings)
    cells = _expand(batch)
    if args.dry_run:
        return _dry_run(cells, settings)
    return _execute(batch, cells, settings, yes=args.yes, progress=progress)


def _preregistration_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise ConfigError(f"pre-registration was not found: {path}")
    return sha256_text(path.read_text(encoding="utf-8"))


def _require_baseline(settings: Settings) -> None:
    if settings.preregistration is None:
        return
    if importlib.util.find_spec("sentence_transformers") is None:
        raise UsageError(
            "baseline models are not installed; install the baseline extra"
        )


def _scorer(settings: Settings, manifest: Manifest | None = None) -> TextScorer:
    required = settings.preregistration is not None
    if manifest is not None and manifest["baseline"]["embedding_model"] is not None:
        required = True
    if not required:
        return lexical_scorer()
    try:
        return load_scorer()
    except BaselineUnavailable as exc:
        raise UsageError(str(exc)) from exc


def _batch(settings: Settings) -> Batch:
    conditions = resolve_conditions(
        settings.condition,
        settings.models,
        temperature=settings.temperature,
    )
    seeds: tuple[PlannedSeed, ...]
    if settings.start is not None:
        if settings.start.strip() == "":
            raise EmptySeedError("seed is empty")
        seeds = (
            PlannedSeed(
                seed_id=None,
                text=settings.start,
                corpus_version=None,
                corpus_sha256=None,
                claims=(),
            ),
        )
    elif settings.seed is not None or settings.split is not None:
        corpus = load_corpus(settings.seeds)
        seeds = tuple(
            _planned(corpus, item)
            for item in select_seeds(
                corpus, seed_id=settings.seed, split=settings.split
            )
        )
    else:
        raise UsageError("provide one of --start, --seed, or --split")
    return Batch(settings=settings, seeds=seeds, conditions=conditions)


def _planned(corpus: Corpus, seed: Seed) -> PlannedSeed:
    return PlannedSeed(
        seed_id=seed.id,
        text=seed.text,
        corpus_version=corpus.version,
        corpus_sha256=corpus.sha256,
        claims=seed.claims,
    )


def _expand(batch: Batch) -> tuple[Cell, ...]:
    return tuple(
        Cell(seed, condition, repeat)
        for seed in batch.seeds
        for condition in batch.conditions
        for repeat in range(batch.settings.repeats)
    )


def _calls(cells: Sequence[Cell], settings: Settings) -> int:
    panel = len(settings.judge) * settings.judge_samples
    return sum(
        planned_calls(cell.condition.prompts, settings.max_cycles, panel)
        for cell in cells
    )


def _dry_run(cells: Sequence[Cell], settings: Settings) -> int:
    tokens, cost = _bounds(cells, settings, _optional_catalog(settings))
    print(f"runs: {len(cells)}")
    print(f"calls: {_calls(cells, settings)}")
    print(f"tokens_upper_bound: {tokens}")
    if cost is None:
        print("cost_upper_bound_usd: unknown")
    else:
        print(f"cost_upper_bound_usd: {cost:.6f}")
    return 0


def _execute(
    batch: Batch,
    cells: Sequence[Cell],
    settings: Settings,
    *,
    yes: bool,
    progress: Callable[[StepNote], None] | None,
) -> int:
    live = _Live(
        load_api_key(),
        progress,
        _scorer(settings),
        _preregistration_hash(settings.preregistration),
    )
    catalog = client_mod.fetch_catalog(live.api_key, settings=settings)
    if _confirm(cells, settings, catalog, yes=yes):
        return _EXIT_CONFIG
    client_mod.check_models(
        catalog,
        (*settings.models, *settings.judge),
        required=settings.required_parameters,
    )
    batch_id = str(uuid.uuid4())
    _write_batch(settings.out / f"{batch_id}{_BATCH_SUFFIX}", batch_id, batch)
    for cell in cells:
        directory, reason = _one(cell, settings, live, batch_id)
        code = _report(directory, reason)
        if code != 0:
            return code
    return 0


def _confirm(
    cells: Sequence[Cell],
    settings: Settings,
    catalog: Mapping[str, client_mod.ModelRecord],
    *,
    yes: bool,
) -> bool:
    _tokens, cost = _bounds(cells, settings, catalog)
    if cost is None or cost <= settings.confirm_above_usd or yes:
        return False
    message = (
        f"estimated cost {cost:.6f} USD exceeds "
        f"{settings.confirm_above_usd:.6f} USD; pass --yes to start"
    )
    _fail(message, _EXIT_CONFIG)
    return True


@dataclass(frozen=True)
class _Live:
    api_key: str
    progress: Callable[[StepNote], None] | None
    scorer: TextScorer
    preregistration_sha256: str | None


def _one(
    cell: Cell,
    settings: Settings,
    live: _Live,
    batch_id: str,
) -> tuple[Path, TerminationReason]:
    seed = cell.seed
    base = settings.sampling_seed
    if base is None:
        base = secrets.randbelow(2**31 - 1) + 1
    spec = replace(
        default_spec(
            cell.condition,
            SeedRef(seed.seed_id, seed.corpus_version, seed.corpus_sha256),
            max_cycles=settings.max_cycles,
            stop_consecutive=settings.stop_consecutive,
            budget=settings.budget,
        ),
        models=tuple(settings.models),
        stop_score=settings.stop_score,
        judge_models=settings.judge,
        judge_samples=settings.judge_samples,
        judge_temperature=settings.judge_temperature,
        score_min=settings.score_min,
        score_max=settings.score_max,
        judge_error_numerator=settings.judge_error_numerator,
        judge_error_denominator=settings.judge_error_denominator,
        on_step=live.progress,
        claims=seed.claims,
        sampling_seed=base,
        top_p=settings.top_p,
    )
    allocated = mk_exp_dir(seed.text, settings.out)
    manifest = running_manifest(
        run_id=allocated.run_id,
        slug=allocated.slug,
        seed=seed.text,
        budget=spec.budget,
        design=replace(
            _design(spec, settings, batch_id, cell.repeat_index),
            baseline=live.scorer.models(),
            preregistration_sha256=live.preregistration_sha256,
        ),
    )
    reason = _drive(
        _Attempt(allocated.directory, manifest, seed.text, spec),
        settings,
        live,
    )
    return allocated.directory, reason


@dataclass(frozen=True)
class _Attempt:
    directory: Path
    manifest: Manifest
    seed: str
    spec: RunSpec
    prior: Sequence[Mapping[str, object]] | None = None


def _drive(attempt: _Attempt, settings: Settings, live: _Live) -> TerminationReason:
    with Recorder(attempt.directory, attempt.manifest, prior=attempt.prior) as recorder:
        with client_mod.OpenRouterClient(live.api_key, settings=settings) as client:
            try:
                reason = cycle(
                    attempt.seed,
                    recorder,
                    client,
                    run_id=attempt.manifest["run_id"],
                    spec=attempt.spec,
                    scorer=live.scorer,
                )
            except KeyboardInterrupt:
                recorder.finish("interrupted")
                return "interrupted"
            except client_mod.InsufficientCreditsError as exc:
                recorder.finish("insufficient_credits", _credit_error(exc))
                return "insufficient_credits"
        recorder.finish(reason)
    return reason


def _credit_error(exc: client_mod.InsufficientCreditsError) -> str:
    return redact(f"{type(exc).__name__}: {exc}").splitlines()[0]


def _design(
    spec: RunSpec,
    settings: Settings,
    batch_id: str | None,
    repeat_index: int | None,
) -> ManifestDesign:
    return ManifestDesign(
        condition=spec.condition.name,
        schedule_name=spec.condition.schedule,
        schedule_models=spec.models,
        transform_temperature=spec.condition.temperature,
        seed_id=spec.seed_id,
        corpus_version=spec.corpus_version,
        corpus_sha256=spec.corpus_sha256,
        stop_consecutive=spec.stop_consecutive,
        max_cycles=spec.max_cycles,
        max_tokens=settings.max_tokens,
        stop_score=spec.stop_score,
        judge_models=spec.judge_models,
        judge_samples=spec.judge_samples,
        judge_temperature=spec.judge_temperature,
        deadline_seconds=settings.deadline_seconds,
        rubric_version=settings.rubric_version,
        requested_models=(*settings.models, *settings.judge),
        sampling_seed=spec.sampling_seed,
        top_p=spec.top_p,
        batch_id=batch_id,
        repeat_index=repeat_index,
    )


def _resume(
    args: argparse.Namespace,
    settings: Settings,
    progress: Callable[[StepNote], None] | None,
) -> int:
    if args.start is not None or args.seed is not None or args.split is not None:
        raise UsageError("--resume cannot be combined with a new run")
    batch_file = _batch_path(str(args.resume), settings.out)
    if batch_file is not None:
        return _resume_batch(batch_file, args, settings, progress)
    if args.dry_run:
        raise UsageError("--resume cannot be combined with a new run")
    directory = _find_run(str(args.resume), settings.out)
    plan = prepare_resume(directory)
    tuned = _tune(settings, plan.manifest)
    live = _Live(
        load_api_key(),
        progress,
        _scorer(tuned, plan.manifest),
        None,
    )
    _check(tuned, live.api_key)
    reason = _drive(
        _Attempt(
            directory,
            plan.manifest,
            plan.seed,
            _resume_spec(plan, tuned, progress),
            plan.rows,
        ),
        tuned,
        live,
    )
    return _report(directory, reason)


def _tune(settings: Settings, manifest: Manifest) -> Settings:
    panel = manifest["judge_panel"]
    if not panel:
        raise UsageError("run has no judge panel")
    return replace(
        settings,
        models=tuple(manifest["schedule"]["models"]),
        judge=tuple(member["model"] for member in panel),
        max_tokens=manifest["max_tokens"],
        deadline_seconds=manifest["deadline_seconds"],
    )


def _check(settings: Settings, api_key: str) -> None:
    catalog = client_mod.fetch_catalog(api_key, settings=settings)
    client_mod.check_models(
        catalog,
        (*settings.models, *settings.judge),
        required=settings.required_parameters,
    )


def _resume_spec(
    plan: ResumePlan,
    settings: Settings,
    progress: Callable[[StepNote], None] | None,
) -> RunSpec:
    manifest = plan.manifest
    panel = manifest["judge_panel"]
    models = tuple(manifest["schedule"]["models"])
    condition = condition_table(
        models, temperature=manifest["temperatures"]["transform"]
    ).get(manifest["condition"])
    if condition is None:
        raise UsageError(f"unknown condition: {manifest['condition']}")
    stop = manifest["stop_rule"]
    return RunSpec(
        condition=condition,
        models=models,
        max_cycles=manifest["max_cycles"],
        stop_score=stop["threshold"],
        stop_consecutive=stop["consecutive"],
        seed_id=manifest["seed_id"],
        corpus_version=manifest["corpus_version"],
        corpus_sha256=manifest["corpus_sha256"],
        budget=manifest["budget"],
        judge_models=tuple(member["model"] for member in panel),
        judge_samples=panel[0]["samples"],
        judge_temperature=manifest["temperatures"]["judge"],
        score_min=settings.score_min,
        score_max=settings.score_max,
        judge_error_numerator=settings.judge_error_numerator,
        judge_error_denominator=settings.judge_error_denominator,
        on_step=progress,
        start_step=plan.next_step,
        opening_text=plan.current,
        spent=plan.spent,
        claims=_resume_claims(manifest),
        sampling_seed=manifest["sampling_seed"],
        top_p=manifest["top_p"],
        previous_answer=_previous_answer(plan.rows),
    )


def _find_run(token: str, root: Path) -> Path:
    if token.strip() == "":
        raise UsageError("run was not found")
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
    raise UsageError(f"run {token} was not found")


def _batch_path(token: str, root: Path) -> Path | None:
    candidate = Path(token)
    if candidate.is_file() and candidate.name.endswith(_BATCH_SUFFIX):
        return candidate
    direct = root / f"{token}{_BATCH_SUFFIX}"
    if direct.is_file():
        return direct
    return None


@dataclass(frozen=True)
class _Plan:
    batch_id: str
    seeds: tuple[PlannedSeed, ...]
    conditions: tuple[str, ...]
    repeats: int
    models: tuple[str, ...]
    judge: tuple[str, ...]
    judge_samples: int
    judge_temperature: float
    temperature: float
    max_cycles: int
    max_tokens: int
    stop_score: float
    stop_consecutive: int
    score_min: int
    score_max: int
    judge_error_numerator: int
    judge_error_denominator: int
    deadline_seconds: float
    required_parameters: tuple[str, ...]
    rubric_version: str
    budget: float | None
    confirm_above_usd: float
    top_p: float
    sampling_seed: int | None


def _resume_batch(
    path: Path,
    args: argparse.Namespace,
    settings: Settings,
    progress: Callable[[StepNote], None] | None,
) -> int:
    plan = _read_batch(path)
    tuned = _apply_batch(plan, settings)
    batch = Batch(
        settings=tuned,
        seeds=plan.seeds,
        conditions=resolve_conditions(
            plan.conditions,
            tuned.models,
            temperature=tuned.temperature,
        ),
    )
    indexed = _index_runs(tuned.out, plan.batch_id)
    remaining: list[Cell] = []
    continuations: dict[tuple[str | None, str, int], Path] = {}
    for cell in _expand(batch):
        key = (cell.seed.seed_id, cell.condition.name, cell.repeat_index)
        skip, found = _pending(indexed.get(key, ()))
        if skip:
            continue
        remaining.append(cell)
        if found is not None:
            continuations[key] = found
    chosen = tuple(remaining)
    if args.dry_run:
        return _dry_run(chosen, tuned)
    if not chosen:
        return 0
    live = _Live(
        load_api_key(),
        progress,
        _scorer(tuned),
        _preregistration_hash(tuned.preregistration),
    )
    catalog = client_mod.fetch_catalog(live.api_key, settings=tuned)
    if _confirm(chosen, tuned, catalog, yes=args.yes):
        return _EXIT_CONFIG
    client_mod.check_models(
        catalog,
        (*tuned.models, *tuned.judge),
        required=tuned.required_parameters,
    )
    for cell in chosen:
        key = (cell.seed.seed_id, cell.condition.name, cell.repeat_index)
        found = continuations.get(key)
        if found is None:
            directory, reason = _one(cell, tuned, live, plan.batch_id)
        else:
            directory, reason = _continue(found, cell.seed.text, tuned, live)
        code = _report(directory, reason)
        if code != 0:
            return code
    return 0


def _pending(paths: Sequence[Path]) -> tuple[bool, Path | None]:
    resume_at: Path | None = None
    for path in paths:
        manifest = read_manifest(path)
        if manifest["status"] == "finished" and finished_reason(
            manifest["termination_reason"]
        ):
            return True, None
        resume_at = path
    return False, resume_at


def _continue(
    directory: Path,
    seed: str,
    settings: Settings,
    live: _Live,
) -> tuple[Path, TerminationReason]:
    plan = prepare_resume(directory, seed)
    tuned = _tune(settings, plan.manifest)
    reason = _drive(
        _Attempt(
            directory,
            plan.manifest,
            plan.seed,
            _resume_spec(plan, tuned, live.progress),
            plan.rows or None,
        ),
        tuned,
        live,
    )
    return directory, reason


def _index_runs(
    root: Path, batch_id: str
) -> dict[tuple[str | None, str, int], list[Path]]:
    found: dict[tuple[str | None, str, int], list[Path]] = {}
    if not root.is_dir():
        return found
    for path in root.iterdir():
        if not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        manifest = read_manifest(path)
        repeat = manifest["repeat_index"]
        if manifest["batch_id"] != batch_id or repeat is None:
            continue
        key = (manifest["seed_id"], manifest["condition"], repeat)
        found.setdefault(key, []).append(path)
    return found


def _write_batch(path: Path, batch_id: str, batch: Batch) -> None:
    settings = batch.settings
    payload: dict[str, object] = {
        "batch_id": batch_id,
        "repeats": settings.repeats,
        "conditions": [condition.name for condition in batch.conditions],
        "seeds": [_seed_payload(seed) for seed in batch.seeds],
        "models": list(settings.models),
        "judge": list(settings.judge),
        "judge_samples": settings.judge_samples,
        "judge_temperature": settings.judge_temperature,
        "temperature": settings.temperature,
        "max_cycles": settings.max_cycles,
        "max_tokens": settings.max_tokens,
        "stop_score": settings.stop_score,
        "stop_consecutive": settings.stop_consecutive,
        "score_min": settings.score_min,
        "score_max": settings.score_max,
        "judge_error_numerator": settings.judge_error_numerator,
        "judge_error_denominator": settings.judge_error_denominator,
        "deadline_seconds": settings.deadline_seconds,
        "required_parameters": list(settings.required_parameters),
        "rubric_version": settings.rubric_version,
        "budget": settings.budget,
        "confirm_above_usd": settings.confirm_above_usd,
        "top_p": settings.top_p,
        "sampling_seed": settings.sampling_seed,
    }
    _write_object(path, payload)


def _seed_payload(seed: PlannedSeed) -> dict[str, object]:
    return {
        "seed_id": seed.seed_id,
        "text": seed.text,
        "corpus_version": seed.corpus_version,
        "corpus_sha256": seed.corpus_sha256,
        "claims": [{"id": claim.id, "text": claim.text} for claim in seed.claims],
    }


def _write_object(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_batch(path: Path) -> _Plan:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UsageError("batch plan is not JSON") from exc
    raw = _object(parsed, path.name)
    batch_id = raw.get("batch_id")
    if not isinstance(batch_id, str) or batch_id == "":
        raise UsageError("batch plan is missing a batch id")
    return _Plan(
        batch_id=batch_id,
        seeds=_plan_seeds(raw.get("seeds")),
        conditions=_plan_strings(raw.get("conditions"), "conditions"),
        repeats=_plan_int(raw.get("repeats"), "repeats"),
        models=_plan_strings(raw.get("models"), "models"),
        judge=_plan_strings(raw.get("judge"), "judge"),
        judge_samples=_plan_int(raw.get("judge_samples"), "judge_samples"),
        judge_temperature=_plan_float(
            raw.get("judge_temperature"), "judge_temperature"
        ),
        temperature=_plan_float(raw.get("temperature"), "temperature"),
        max_cycles=_plan_int(raw.get("max_cycles"), "max_cycles"),
        max_tokens=_plan_int(raw.get("max_tokens"), "max_tokens"),
        stop_score=_plan_float(raw.get("stop_score"), "stop_score"),
        stop_consecutive=_plan_int(raw.get("stop_consecutive"), "stop_consecutive"),
        score_min=_plan_int(raw.get("score_min"), "score_min"),
        score_max=_plan_int(raw.get("score_max"), "score_max"),
        judge_error_numerator=_plan_int(
            raw.get("judge_error_numerator"), "judge_error_numerator"
        ),
        judge_error_denominator=_plan_int(
            raw.get("judge_error_denominator"), "judge_error_denominator"
        ),
        deadline_seconds=_plan_float(raw.get("deadline_seconds"), "deadline_seconds"),
        required_parameters=_plan_strings(
            raw.get("required_parameters"), "required_parameters", allow_empty=True
        ),
        rubric_version=_plan_text(raw.get("rubric_version"), "rubric_version"),
        budget=_plan_optional_float(raw.get("budget"), "budget"),
        confirm_above_usd=_plan_float(
            raw.get("confirm_above_usd"), "confirm_above_usd"
        ),
        top_p=_plan_float(raw.get("top_p"), "top_p"),
        sampling_seed=_plan_optional_int(raw.get("sampling_seed"), "sampling_seed"),
    )


def _apply_batch(plan: _Plan, settings: Settings) -> Settings:
    return replace(
        settings,
        models=plan.models,
        judge=plan.judge,
        judge_samples=plan.judge_samples,
        judge_temperature=plan.judge_temperature,
        temperature=plan.temperature,
        max_cycles=plan.max_cycles,
        max_tokens=plan.max_tokens,
        stop_score=plan.stop_score,
        stop_consecutive=plan.stop_consecutive,
        score_min=plan.score_min,
        score_max=plan.score_max,
        judge_error_numerator=plan.judge_error_numerator,
        judge_error_denominator=plan.judge_error_denominator,
        deadline_seconds=plan.deadline_seconds,
        required_parameters=plan.required_parameters,
        rubric_version=plan.rubric_version,
        repeats=plan.repeats,
        budget=plan.budget,
        confirm_above_usd=plan.confirm_above_usd,
        top_p=plan.top_p,
        sampling_seed=plan.sampling_seed,
        condition=plan.conditions,
        start=None,
        seed=None,
        split=None,
    )


def _plan_seeds(value: object) -> tuple[PlannedSeed, ...]:
    if not isinstance(value, list) or not value:
        raise UsageError("batch plan seeds are invalid")
    seeds: list[PlannedSeed] = []
    for item in value:
        fields = _object(item, "seed")
        seeds.append(
            PlannedSeed(
                seed_id=_plan_optional_text(fields.get("seed_id"), "seed_id"),
                text=_plan_text(fields.get("text"), "text"),
                corpus_version=_plan_optional_text(
                    fields.get("corpus_version"), "corpus_version"
                ),
                corpus_sha256=_plan_optional_text(
                    fields.get("corpus_sha256"), "corpus_sha256"
                ),
                claims=_plan_claims(fields.get("claims")),
            )
        )
    return tuple(seeds)


def _plan_claims(value: object) -> tuple[Claim, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise UsageError("batch plan claims are invalid")
    claims: list[Claim] = []
    for item in value:
        fields = _object(item, "claim")
        claim_id = fields.get("id")
        text = fields.get("text")
        if not isinstance(claim_id, str) or not isinstance(text, str):
            raise UsageError("batch plan claims are invalid")
        claims.append(Claim(claim_id, text))
    return tuple(claims)


def _plan_strings(
    value: object, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise UsageError(f"batch plan {label} is invalid")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            raise UsageError(f"batch plan {label} is invalid")
        items.append(item)
    return tuple(items)


def _plan_text(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise UsageError(f"batch plan {label} is invalid")
    return value


def _plan_optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _plan_text(value, label)


def _plan_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UsageError(f"batch plan {label} is invalid")
    return value


def _plan_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UsageError(f"batch plan {label} is invalid")
    return float(value)


def _plan_optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _plan_float(value, label)


def _plan_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    number = _plan_int(value, label)
    if number < 1:
        raise UsageError(f"batch plan {label} is invalid")
    return number


def _models(settings: Settings) -> int:
    api_key = load_api_key()
    catalog = client_mod.fetch_catalog(api_key, settings=settings)
    model_ids = (*settings.models, *settings.judge)
    client_mod.check_models(
        catalog,
        model_ids,
        required=settings.required_parameters,
    )
    for model_id in model_ids:
        print(model_id)
    return 0


def _export(directory: Path) -> int:
    writer = csv.DictWriter(sys.stdout, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in _rows(directory):
        writer.writerow(
            {
                "cycle": _cell(row.get("step")),
                "phase": _cell(row.get("phase")),
                "model": _cell(row.get("requested_model")),
                "served_model": _cell(row.get("served_model")),
                "text": _cell(row.get("output_text")),
                "score": _csv_score(row),
            }
        )
    return 0


def _analyze(directories: Sequence[Path], out: Path | None) -> int:
    try:
        text = report(directories, out=out)
    except AnalyzeError as exc:
        raise UsageError(str(exc)) from exc
    print(text, end="")
    return 0


def _rows(directory: Path) -> list[dict[str, object]]:
    path = directory / "chain.jsonl"
    if not path.is_file():
        raise UsageError(f"chain.jsonl was not found in {directory}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.strip() == "":
            continue
        try:
            parsed: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{path}:{line_number} is not JSON") from exc
        rows.append(_object(parsed, f"{path}:{line_number}"))
    if not rows:
        raise UsageError(f"{path} has no rows")
    return rows


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UsageError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise UsageError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped


def _cell(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def _csv_score(row: Mapping[str, object]) -> str:
    judge = row.get("judge")
    if not isinstance(judge, dict):
        return ""
    samples = judge.get("samples")
    if not isinstance(samples, list):
        return ""
    scores: list[int | float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        score = sample.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            continue
        scores.append(score)
    if not scores:
        return ""
    return str(statistics.mean(scores))


def _optional_catalog(settings: Settings) -> dict[str, ModelRecord] | None:
    try:
        api_key = load_api_key()
    except MissingAPIKeyError:
        return None
    return client_mod.fetch_catalog(api_key, settings=settings)


def _bounds(
    cells: Sequence[Cell],
    settings: Settings,
    catalog: Mapping[str, ModelRecord] | None,
) -> tuple[int, float | None]:
    tokens = 0
    cost = 0.0
    priced = True
    for cell in cells:
        step_tokens, step_cost = _bound_run(
            cell.seed.text,
            cell.condition,
            settings,
            catalog,
            cell.seed.claims,
        )
        tokens += step_tokens
        if step_cost is None:
            priced = False
        else:
            cost += step_cost
    if not priced:
        return tokens, None
    return tokens, cost


def _bound_run(
    seed: str,
    condition: Condition,
    settings: Settings,
    catalog: Mapping[str, ModelRecord] | None,
    claims: tuple[Claim, ...] = (),
) -> tuple[int, float | None]:
    tokens = 0
    cost = 0.0
    known = True
    first_fidelity = 2 if condition.prompts == "telephone" else 1
    for step in range(1, settings.max_cycles + 1):
        phase = phase_for(condition.prompts, step)
        model = model_for(condition.schedule, settings.models, step)
        prompt_tokens = _transform_prompt_tokens(phase, seed, step, settings.max_tokens)
        tokens += prompt_tokens + settings.max_tokens
        transform = _priced(
            None if catalog is None else catalog.get(model),
            prompt_tokens,
            settings.max_tokens,
        )
        if transform is None:
            known = False
        else:
            cost += transform
        kind = score_kind(phase)
        panels = 2 if kind == "fidelity" and step > first_fidelity else 1
        judge_prompt = _judge_prompt_tokens(kind, seed, settings.max_tokens, claims)
        for judge in settings.judge:
            unit = _priced(
                None if catalog is None else catalog.get(judge),
                judge_prompt,
                settings.max_tokens,
            )
            for _sample in range(settings.judge_samples * panels):
                tokens += judge_prompt + settings.max_tokens
                if unit is None:
                    known = False
                else:
                    cost += unit
    if not known:
        return tokens, None
    return tokens, cost


def _priced(
    record: ModelRecord | None,
    prompt_tokens: int,
    max_tokens: int,
) -> float | None:
    if record is None or record.prompt_price is None or record.completion_price is None:
        return None
    request = 0.0 if record.request_price is None else record.request_price
    return (
        prompt_tokens * record.prompt_price
        + max_tokens * record.completion_price
        + request
    )


def _transform_prompt_tokens(phase: str, seed: str, step: int, max_tokens: int) -> int:
    inserted = len(seed) if step == 1 else max_tokens
    return len(render_transform(phase, "")) + inserted + _FRAMING_TOKENS


def _judge_prompt_tokens(
    kind: str,
    seed: str,
    max_tokens: int,
    claims: tuple[Claim, ...] = (),
) -> int:
    if kind == "answerability":
        rendered = render_text(ANSWERABILITY_USER_TMPL, question="", seed="")
    else:
        rendered = render_text(
            JUDGE_USER_TMPL,
            original="",
            transformed="",
            claims=claim_block(claims),
        )
    return len(JUDGE_SYS) + len(rendered) + len(seed) + max_tokens + _FRAMING_TOKENS


def _resume_claims(manifest: Manifest) -> tuple[Claim, ...]:
    seed_id = manifest["seed_id"]
    if seed_id is None:
        return ()
    try:
        corpus = load_corpus()
    except DesignError:
        return ()
    if corpus.sha256 != manifest["corpus_sha256"]:
        return ()
    for seed in corpus.seeds:
        if seed.id == seed_id:
            return seed.claims
    return ()


def _previous_answer(rows: tuple[Mapping[str, object], ...]) -> str | None:
    found: str | None = None
    for row in rows:
        if row.get("score_kind") != "fidelity":
            continue
        text = row.get("output_text")
        if isinstance(text, str):
            found = text
    return found


def _on_step(*, verbose: bool) -> Callable[[StepNote], None]:
    def report(note: StepNote) -> None:
        line = f"step {note.step}/{note.total} {note.phase} {note.model}"
        if verbose:
            line += (
                f" prompt_tokens={note.prompt_tokens}"
                f" completion_tokens={note.completion_tokens}"
                f" cost={note.cost:.6f}"
            )
        print(redact(line), file=sys.stderr)

    return report


def _report(directory: Path, reason: TerminationReason) -> int:
    message = _status_line(directory, reason)
    if reason in {"max_cycles", "threshold"}:
        print(message)
        return 0
    print(message, file=sys.stderr)
    if reason in {"budget_exceeded", "insufficient_credits"}:
        return _EXIT_BUDGET
    if reason == "interrupted":
        return _EXIT_INTERRUPTED
    return 1


def _status_line(directory: Path, reason: TerminationReason) -> str:
    manifest = read_manifest(directory)
    rule = manifest["stop_rule"]
    last, scores = _chain_scores(directory / "chain.jsonl")
    detail = f"{reason} at step {last} of {manifest['max_cycles']}"
    if reason == "max_cycles":
        found = degradation_step(scores, rule["threshold"], rule["consecutive"])
        if found is None:
            detail = f"{detail}, threshold not reached"
        else:
            detail = f"{detail}, threshold at step {found}"
    return f"{detail}: results saved to {directory}"


def _chain_scores(path: Path) -> tuple[int, tuple[tuple[int, float | None], ...]]:
    if not path.is_file():
        return 0, ()
    scores: list[tuple[int, float | None]] = []
    last = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        try:
            parsed: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        step = parsed.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        last = max(last, step)
        scores.append((step, _fidelity(parsed.get("judge"))))
    scores.sort()
    return last, tuple(scores)


def _fidelity(judge: object) -> float | None:
    if not isinstance(judge, dict):
        return None
    raw = judge.get("fidelity_vs_seed")
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def _configure_logging() -> None:
    logger = logging.getLogger("echochain")
    if logger.handlers:
        return
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class _StderrHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def _fail(message: str, code: int) -> int:
    print(redact(message), file=sys.stderr)
    return code
