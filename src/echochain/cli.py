from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path

from echochain import __version__
from echochain.baseline import BaselineUnavailable
from echochain.client import (
    ModelCatalogError,
    ModelUnavailableError,
    check_models,
    fetch_catalog,
)
from echochain.coercion import ConfigError
from echochain.design import DesignError, StepNote
from echochain.records import (
    EmptySeedError,
    Row,
    RunDataError,
    TerminationReason,
    read_manifest,
    read_rows,
)
from echochain.report import report
from echochain.runner import Finished, RunError, dry_run, execute, resume
from echochain.settings import (
    MissingAPIKeyError,
    Settings,
    load_api_key,
    log_filter,
    redact,
    render_settings,
    resolve_settings,
)
from echochain.stats import AnalyzeError, degradation_step

_EXIT_BUDGET = 3
_EXIT_CONFIG = 2
_EXIT_INTERRUPTED = 130
_COMMANDS = frozenset({"run", "config", "export", "analyze", "models"})
_FORMAT = argparse.RawDescriptionHelpFormatter
_CSV_FIELDS = ("cycle", "phase", "model", "served_model", "text", "score")
_SUCCESS = frozenset({TerminationReason.max_cycles, TerminationReason.threshold})
_BUDGET = frozenset(
    {TerminationReason.budget_exceeded, TerminationReason.insufficient_credits}
)
_EPILOG = """\
run flags:
  --start --seed --seeds --condition --repeats --sampling-seed --top-p
  --dry-run --yes --out --budget --model --judge --judge-samples
  --judge-temperature --temperature --max-cycles --max-tokens --stop-score
  --stop-consecutive --provider --resume

examples:
  echochain --start TEXT --budget 1
  echochain run --seed redis-epoll --max-cycles 4
  echochain run --resume RUN_ID

exit codes:
  0    completed, the threshold or the cycle cap was reached
  1    runtime or API failure
  2    usage or configuration error
  3    budget exceeded or insufficient credits
  130  interrupted
"""


class UsageError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(_normalize(raw))
        return _dispatch(args)
    except (
        EmptySeedError,
        UsageError,
        RunError,
        DesignError,
        MissingAPIKeyError,
        ConfigError,
        RunDataError,
        ModelUnavailableError,
        BaselineUnavailable,
    ) as exc:
        return _fail(str(exc), _EXIT_CONFIG)
    except ModelCatalogError as exc:
        return _fail(str(exc), 1)
    except KeyboardInterrupt:
        return _EXIT_INTERRUPTED
    except Exception:
        logging.getLogger("echochain").exception("run failed")
        return 1


def status_line(directory: Path, reason: TerminationReason) -> str:
    manifest = read_manifest(directory)
    rows = read_rows(directory)
    last = rows[-1].step if rows else 0
    detail = f"{reason.value} at step {last} of {manifest.max_cycles}"
    if reason == TerminationReason.max_cycles:
        scores = tuple((row.step, row.judge.fidelity_vs_seed) for row in rows)
        found = degradation_step(
            scores, manifest.stop_rule.threshold, manifest.stop_rule.consecutive
        )
        if found is None:
            detail = f"{detail}, threshold not reached"
        else:
            detail = f"{detail}, threshold at step {found}"
    return f"{detail}: results saved to {directory}"


def exit_code(reason: TerminationReason) -> int:
    if reason in _SUCCESS:
        return 0
    if reason in _BUDGET:
        return _EXIT_BUDGET
    if reason == TerminationReason.interrupted:
        return _EXIT_INTERRUPTED
    return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.quiet and args.verbose:
        raise UsageError("--quiet and --verbose cannot be combined")
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
        description="Measure how an answer changes when models pass it along.",
        epilog=_EPILOG,
        formatter_class=_FORMAT,
    )
    parser.add_argument(
        "--version", action="version", version=f"echochain {__version__}"
    )
    parser.add_argument("--quiet", action="store_true", help="Hide step progress.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print token counts and cost with each step.",
    )
    parser.add_argument("--config", type=Path, help="TOML settings file.")
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
    commands = config.add_subparsers(dest="config_command", required=True)
    commands.add_parser("show", help="Print the resolved configuration.")
    export = sub.add_parser("export", help="Write CSV derived from a run.")
    export.add_argument("directory", type=Path, metavar="RUN")
    export.add_argument("--format", default="csv", choices=("csv",))
    analyze = sub.add_parser("analyze", help="Summarize one or more finished runs.")
    analyze.add_argument("directory", nargs="+", type=Path, metavar="RUN")
    analyze.add_argument("--out", type=Path, help="Directory for tables and figures.")
    sub.add_parser("models", help="Check configured models against the catalog.")
    return parser


def _add_run_flags(run: argparse.ArgumentParser) -> None:
    for item in fields(Settings):
        flag = item.metadata.get("flag")
        if not isinstance(flag, str):
            continue
        kind = str(item.metadata.get("kind"))
        help_text = str(item.metadata.get("help") or "")
        if item.metadata.get("append"):
            run.add_argument(
                flag, action="append", type=str, dest=item.name, help=help_text
            )
            continue
        run.add_argument(flag, type=_flag_type(kind), dest=item.name, help=help_text)
    run.add_argument("--dry-run", action="store_true", help="Print the cost bound.")
    run.add_argument("--yes", action="store_true", help="Start a run over the cap.")
    run.add_argument(
        "--resume", help="Continue a run id, a run directory, or a batch id."
    )


def _flag_type(kind: str) -> type[int] | type[float] | type[str] | type[Path]:
    if kind in {"int", "optional_int"}:
        return int
    if kind in {"float", "optional_float"}:
        return float
    if kind in {"path", "optional_path"}:
        return Path
    return str


def _normalize(argv: list[str]) -> list[str]:
    if not argv:
        return ["run"]
    if argv[0] in {"-h", "--help", "--version"}:
        return argv
    leading: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in {"--quiet", "--verbose", "--version"}:
            leading.append(item)
            index += 1
            continue
        if item == "--config":
            leading.extend(argv[index : index + 2])
            index += 2
            continue
        if item.startswith("--config="):
            leading.append(item)
            index += 1
            continue
        rest.append(item)
        index += 1
    if not rest or rest[0] not in _COMMANDS:
        rest = ["run", *rest]
    return [*leading, *rest]


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for item in fields(Settings):
        if not isinstance(item.metadata.get("flag"), str):
            continue
        raw = getattr(args, item.name, None)
        if raw is None:
            continue
        if item.metadata.get("kind") == "pins":
            values[item.name] = ";".join(str(part) for part in raw)
        else:
            values[item.name] = raw
    return values


def _run(args: argparse.Namespace, settings: Settings) -> int:
    progress = None if args.quiet else _progress(bool(args.verbose))
    if args.resume:
        if args.start is not None or args.seed is not None or args.seeds is not None:
            raise UsageError("--resume cannot be combined with a new run")
        if args.dry_run:
            raise UsageError("--resume cannot be combined with --dry-run")
        finished = resume(str(args.resume), settings, progress=progress)
    elif args.dry_run:
        print(dry_run(settings), end="")
        return 0
    else:
        finished = execute(settings, yes=bool(args.yes), progress=progress)
    return _report(finished)


def _report(finished: Finished) -> int:
    code = exit_code(finished.reason)
    message = status_line(finished.directory, finished.reason)
    print(message, file=sys.stdout if code == 0 else sys.stderr)
    return code


def _progress(verbose: bool) -> Callable[[StepNote], None]:
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


def _models(settings: Settings) -> int:
    catalog = fetch_catalog(load_api_key(), settings=settings)
    names = (*settings.models, *settings.judge)
    check_models(catalog, names)
    for model_id in names:
        print(model_id)
    return 0


def _export(directory: Path) -> int:
    rows = read_rows(directory)
    if not rows:
        raise UsageError(f"chain.jsonl was not found in {directory}")
    writer = csv.DictWriter(sys.stdout, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "cycle": row.step,
                "phase": row.phase.value,
                "model": row.requested_model,
                "served_model": row.served_model,
                "text": row.output_text,
                "score": _score(row),
            }
        )
    return 0


def _score(row: Row) -> str:
    scores = [sample.score for sample in row.judge.samples if sample.score is not None]
    if not scores:
        return ""
    return str(statistics.mean(scores))


def _analyze(directories: Sequence[Path], out: Path | None) -> int:
    try:
        print(report(directories, out=out), end="")
    except AnalyzeError as exc:
        raise UsageError(str(exc)) from exc
    return 0


def _configure_logging() -> None:
    logger = logging.getLogger("echochain")
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(log_filter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _fail(message: str, code: int) -> int:
    print(redact(message), file=sys.stderr)
    return code
