from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from echochain import __version__
from echochain import client as client_mod
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
    default_spec,
    load_corpus,
    model_for,
    phase_for,
    render_transform,
    resolve_conditions,
    score_kind,
    select_seeds,
)
from echochain.engine import cycle
from echochain.persistence import (
    EmptySeedError,
    ManifestDesign,
    Recorder,
    TerminationReason,
    mk_exp_dir,
    running_manifest,
)
from echochain.prompts import (
    ANSWERABILITY_USER_TMPL,
    JUDGE_SYS,
    JUDGE_USER_TMPL,
)

_EXIT_BUDGET = 3
_EXIT_CONFIG = 2
_EXIT_INTERRUPTED = 130
_FRAMING_TOKENS = 16
_COMMANDS = frozenset({"run", "config", "export", "analyze", "models"})
_FORMAT = argparse.RawDescriptionHelpFormatter
_EPILOG = """\
examples:
  echochain --start TEXT --out runs --budget 1
  echochain run --seeds seeds/v1.toml --split study

exit codes:
  0    completed, the threshold or the cycle cap was reached
  1    runtime or API failure
  2    usage or configuration error
  3    budget exceeded
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


@dataclass(frozen=True)
class Batch:
    settings: Settings
    seeds: tuple[PlannedSeed, ...]
    conditions: tuple[Condition, ...]

    @property
    def run_count(self) -> int:
        """Return how many runs this batch starts.

        Returns:
            Seeds times conditions times repeats.
        """
        return len(self.seeds) * len(self.conditions) * self.settings.repeats

    @property
    def calls(self) -> int:
        """Return the maximum number of model calls.

        Returns:
            One transform and the judge samples for every planned step.
        """
        per_step = 1 + len(self.settings.judge) * self.settings.judge_samples
        return self.run_count * self.settings.max_cycles * per_step


def main(argv: list[str] | None = None) -> int:
    """Run one echoChain command.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. `0` is a completed run. `1` is a runtime or API
        failure. `2` is a usage or configuration error. `3` means the budget
        was exceeded. `130` means the run was interrupted.
    """
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
        client_mod.ModelUnavailableError,
    ) as exc:
        return _fail(str(exc), _EXIT_CONFIG)
    except client_mod.ModelCatalogError as exc:
        return _fail(str(exc), 1)
    except Exception:
        return _fail(traceback.format_exc(), 1)


def _dispatch(args: argparse.Namespace) -> int:
    if args.quiet and args.verbose:
        return _fail("--quiet and --verbose cannot be combined", _EXIT_CONFIG)
    if args.command == "export":
        return _export(args.directory)
    if args.command == "analyze":
        return _analyze(args.directory)
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
    analyze = sub.add_parser("analyze", help="Summarize fidelity for one run.")
    analyze.add_argument("directory", type=Path, metavar="RUN")
    sub.add_parser("models", help="Check configured models against the catalog.")
    return parser


def _add_run_flags(run: argparse.ArgumentParser) -> None:
    run.add_argument("--start", help="Seed answer text.")
    run.add_argument("--seed", help="Seed id from the corpus.")
    run.add_argument("--seeds", type=Path, help="Seed corpus TOML.")
    run.add_argument("--corpus", type=Path, help="Seed corpus TOML.")
    run.add_argument("--split", choices=("study", "held-out"), help="Corpus split.")
    run.add_argument(
        "--condition",
        action="append",
        help="Condition name. Repeat to run several. 'all' runs the controls.",
    )
    run.add_argument("--repeats", type=int, help="Repetitions of each pair.")
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
    put("condition", args.condition)
    put("repeats", args.repeats)
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
    batch = _batch(settings)
    if args.dry_run:
        return _dry_run(batch, settings)
    progress = None if args.quiet else _on_step(verbose=args.verbose)
    return _execute(batch, settings, yes=args.yes, progress=progress)


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
    )


def _dry_run(batch: Batch, settings: Settings) -> int:
    tokens, cost = _bounds(batch, settings, _optional_catalog(settings))
    print(f"runs: {batch.run_count}")
    print(f"calls: {batch.calls}")
    print(f"tokens_upper_bound: {tokens}")
    if cost is None:
        print("cost_upper_bound_usd: unknown")
    else:
        print(f"cost_upper_bound_usd: {cost:.6f}")
    return 0


def _execute(
    batch: Batch,
    settings: Settings,
    *,
    yes: bool,
    progress: Callable[[StepNote], None] | None,
) -> int:
    api_key = load_api_key()
    catalog = client_mod.fetch_catalog(api_key, settings=settings)
    _tokens, cost = _bounds(batch, settings, catalog)
    if cost is not None and cost > settings.confirm_above_usd and not yes:
        message = (
            f"estimated cost {cost:.6f} USD exceeds "
            f"{settings.confirm_above_usd:.6f} USD; pass --yes to start"
        )
        return _fail(message, _EXIT_CONFIG)
    client_mod.check_models(
        catalog,
        (*settings.models, *settings.judge),
        required=settings.required_parameters,
    )
    for seed in batch.seeds:
        for condition in batch.conditions:
            for _ in range(settings.repeats):
                directory, reason = _one(seed, condition, settings, api_key, progress)
                code = _report(directory, reason)
                if code != 0:
                    return code
    return 0


def _one(
    seed: PlannedSeed,
    condition: Condition,
    settings: Settings,
    api_key: str,
    progress: Callable[[StepNote], None] | None,
) -> tuple[Path, TerminationReason]:
    spec = replace(
        default_spec(
            condition,
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
        on_step=progress,
    )
    allocated = mk_exp_dir(seed.text, settings.out)
    manifest = running_manifest(
        run_id=allocated.run_id,
        slug=allocated.slug,
        seed=seed.text,
        budget=spec.budget,
        design=_design(spec, settings),
    )
    with Recorder(allocated.directory, manifest) as recorder:
        with client_mod.OpenRouterClient(api_key, settings=settings) as client:
            try:
                reason = cycle(
                    seed.text,
                    recorder,
                    client,
                    run_id=allocated.run_id,
                    spec=spec,
                )
            except KeyboardInterrupt:
                recorder.finish("interrupted")
                return allocated.directory, "interrupted"
        recorder.finish(reason)
    return allocated.directory, reason


def _design(spec: RunSpec, settings: Settings) -> ManifestDesign:
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
    )


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


def _analyze(directory: Path) -> int:
    groups: dict[tuple[str, int], list[int | float]] = {}
    for row in _rows(directory):
        value = _fidelity(row)
        condition = row.get("condition")
        step = row.get("step")
        if value is None or not isinstance(condition, str):
            continue
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        groups.setdefault((condition, step), []).append(value)
    print("condition\tstep\tcount\tmean_fidelity")
    for condition, step in sorted(groups):
        values = groups[(condition, step)]
        mean = statistics.mean(values)
        print(f"{condition}\t{step}\t{len(values)}\t{mean:.6f}")
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


def _fidelity(row: Mapping[str, object]) -> int | float | None:
    judge = row.get("judge")
    if not isinstance(judge, dict):
        return None
    value = judge.get("fidelity_vs_seed")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def _optional_catalog(settings: Settings) -> dict[str, ModelRecord] | None:
    try:
        api_key = load_api_key()
    except MissingAPIKeyError:
        return None
    return client_mod.fetch_catalog(api_key, settings=settings)


def _bounds(
    batch: Batch,
    settings: Settings,
    catalog: Mapping[str, ModelRecord] | None,
) -> tuple[int, float | None]:
    tokens = 0
    cost = 0.0
    priced = True
    for seed in batch.seeds:
        for condition in batch.conditions:
            step_tokens, step_cost = _bound_run(seed.text, condition, settings, catalog)
            tokens += step_tokens * settings.repeats
            if step_cost is None:
                priced = False
            else:
                cost += step_cost * settings.repeats
    if not priced:
        return tokens, None
    return tokens, cost


def _bound_run(
    seed: str,
    condition: Condition,
    settings: Settings,
    catalog: Mapping[str, ModelRecord] | None,
) -> tuple[int, float | None]:
    tokens = 0
    cost = 0.0
    known = True
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
        judge_prompt = _judge_prompt_tokens(kind, seed, settings.max_tokens)
        for judge in settings.judge:
            unit = _priced(
                None if catalog is None else catalog.get(judge),
                judge_prompt,
                settings.max_tokens,
            )
            for _sample in range(settings.judge_samples):
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


def _judge_prompt_tokens(kind: str, seed: str, max_tokens: int) -> int:
    if kind == "answerability":
        rendered: object = ANSWERABILITY_USER_TMPL.render(question="", seed="")
    else:
        rendered = JUDGE_USER_TMPL.render(original="", transformed="")
    if not isinstance(rendered, str):
        raise TypeError("prompt render did not return text")
    return len(JUDGE_SYS) + len(rendered) + len(seed) + max_tokens + _FRAMING_TOKENS


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
    message = f"{reason}: results saved to {directory}"
    if reason in {"max_cycles", "threshold"}:
        print(message)
        return 0
    print(message, file=sys.stderr)
    if reason == "budget_exceeded":
        return _EXIT_BUDGET
    if reason == "interrupted":
        return _EXIT_INTERRUPTED
    return 1


def _fail(message: str, code: int) -> int:
    print(redact(message), file=sys.stderr)
    return code
