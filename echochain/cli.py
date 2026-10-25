from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from echochain import __version__
from echochain import client as client_mod
from echochain.client import ModelRecord
from echochain.config import (
    JUDGE_MODEL,
    JUDGE_SAMPLES,
    MAX_CYCLES,
    MAX_TOKENS,
    ROTATING_MODELS,
    MissingAPIKeyError,
    configured_models,
    load_api_key,
    redact,
)
from echochain.design import (
    Condition,
    Corpus,
    DesignError,
    RunSpec,
    Seed,
    SeedRef,
    default_spec,
    load_corpus,
    model_for,
    resolve_conditions,
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

_EXIT_BUDGET = 3
_EXIT_CONFIG = 2
_EXIT_INTERRUPTED = 130


class InvalidBudgetError(RuntimeError):
    pass


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
    seeds: tuple[PlannedSeed, ...]
    conditions: tuple[Condition, ...]
    repeats: int
    out: Path
    budget: float | None

    @property
    def run_count(self) -> int:
        """Return how many runs this batch starts.

        Returns:
            Seeds times conditions times repeats.
        """
        return len(self.seeds) * len(self.conditions) * self.repeats

    @property
    def calls(self) -> int:
        """Return the maximum number of model calls.

        Returns:
            One transform and one judge sample for every planned step.
        """
        return self.run_count * MAX_CYCLES * (1 + JUDGE_SAMPLES)


def main(argv: list[str] | None = None) -> int:
    """Run one or more telephone experiments.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. `2` means the seed, budget, key, or a model is
        unusable. `3` means the budget was exceeded. `130` means the run was
        interrupted.
    """
    args = _parser().parse_args(argv)
    try:
        batch = _batch(args)
        if args.dry_run:
            return _dry_run(batch)
        return _execute(batch)
    except (
        EmptySeedError,
        InvalidBudgetError,
        UsageError,
        DesignError,
        MissingAPIKeyError,
        client_mod.ModelUnavailableError,
    ) as exc:
        return _fail(str(exc), _EXIT_CONFIG)
    except client_mod.ModelCatalogError as exc:
        return _fail(str(exc), 1)
    except Exception:
        return _fail(traceback.format_exc(), 1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echochain",
        description="Measure information degradation through LLM telephone.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"echochain {__version__}",
    )
    parser.add_argument("--start", help="Seed answer text.")
    parser.add_argument("--seed", help="Seed id from the corpus.")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Seed corpus TOML. Defaults to seeds/v1.toml.",
    )
    parser.add_argument(
        "--split",
        choices=("study", "held-out"),
        help="Run every seed in this split.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help=(
            "Condition name. Repeat to run several. "
            "'all' runs the treatment and the three controls."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run each seed and condition.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the call count and cost bound, then exit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="Directory that receives run folders.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Stop when spend reaches this many USD.",
    )
    return parser


def _batch(args: argparse.Namespace) -> Batch:
    sources = (
        args.start is not None,
        args.seed is not None,
        args.split is not None,
    )
    if sum(sources) != 1:
        raise UsageError("provide one of --start, --seed, or --split")
    if args.corpus is not None and args.start is not None:
        raise UsageError("--corpus is used with --seed or --split")
    if args.repeats < 1:
        raise UsageError("repeats must be at least 1")
    if args.budget is not None and args.budget < 0:
        raise InvalidBudgetError("budget must be zero or greater")
    conditions = resolve_conditions(args.condition)
    seeds: tuple[PlannedSeed, ...]
    if args.start is not None:
        if args.start.strip() == "":
            raise EmptySeedError("seed is empty")
        seeds = (
            PlannedSeed(
                seed_id=None,
                text=args.start,
                corpus_version=None,
                corpus_sha256=None,
            ),
        )
    else:
        corpus = load_corpus(args.corpus)
        seeds = tuple(
            _planned(corpus, item)
            for item in select_seeds(corpus, seed_id=args.seed, split=args.split)
        )
    return Batch(
        seeds=seeds,
        conditions=conditions,
        repeats=args.repeats,
        out=args.out,
        budget=args.budget,
    )


def _planned(corpus: Corpus, seed: Seed) -> PlannedSeed:
    return PlannedSeed(
        seed_id=seed.id,
        text=seed.text,
        corpus_version=corpus.version,
        corpus_sha256=corpus.sha256,
    )


def _dry_run(batch: Batch) -> int:
    print(f"runs: {batch.run_count}")
    print(f"calls: {batch.calls}")
    bound = _cost_upper_bound(batch)
    if bound is None:
        print("cost_upper_bound_usd: unknown")
    else:
        print(f"cost_upper_bound_usd: {bound:.6f}")
    return 0


def _cost_upper_bound(batch: Batch) -> float | None:
    try:
        api_key = load_api_key()
    except MissingAPIKeyError:
        return None
    catalog = client_mod.fetch_catalog(api_key)
    per_run = _bounded_run(catalog, batch.conditions)
    if per_run is None:
        return None
    return per_run * len(batch.seeds) * batch.repeats


def _bounded_run(
    catalog: dict[str, ModelRecord],
    conditions: tuple[Condition, ...],
) -> float | None:
    total = 0.0
    judge = _call_bound(catalog.get(JUDGE_MODEL))
    if judge is None:
        return None
    for condition in conditions:
        for step in range(1, MAX_CYCLES + 1):
            model = model_for(condition.schedule, ROTATING_MODELS, step)
            transform = _call_bound(catalog.get(model))
            if transform is None:
                return None
            total += transform + judge * JUDGE_SAMPLES
    return total


def _call_bound(record: ModelRecord | None) -> float | None:
    if record is None or record.prompt_price is None or record.completion_price is None:
        return None
    return MAX_TOKENS * (record.prompt_price + record.completion_price)


def _execute(batch: Batch) -> int:
    api_key = load_api_key()
    catalog = client_mod.fetch_catalog(api_key)
    client_mod.check_models(catalog, configured_models())
    for seed in batch.seeds:
        for condition in batch.conditions:
            for _ in range(batch.repeats):
                directory, reason = _one(seed, condition, batch, api_key)
                code = _report(directory, reason)
                if code != 0:
                    return code
    return 0


def _one(
    seed: PlannedSeed,
    condition: Condition,
    batch: Batch,
    api_key: str,
) -> tuple[Path, TerminationReason]:
    spec = default_spec(
        condition,
        SeedRef(seed.seed_id, seed.corpus_version, seed.corpus_sha256),
        budget=batch.budget,
    )
    return _run(seed.text, batch.out, spec, api_key)


def _run(
    seed: str,
    out: Path,
    spec: RunSpec,
    api_key: str,
) -> tuple[Path, TerminationReason]:
    allocated = mk_exp_dir(seed, out)
    manifest = running_manifest(
        run_id=allocated.run_id,
        slug=allocated.slug,
        seed=seed,
        budget=spec.budget,
        design=_design(spec),
    )
    with Recorder(allocated.directory, manifest) as recorder:
        with client_mod.OpenRouterClient(api_key) as client:
            try:
                reason = cycle(
                    seed,
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


def _design(spec: RunSpec) -> ManifestDesign:
    return ManifestDesign(
        condition=spec.condition.name,
        schedule_name=spec.condition.schedule,
        schedule_models=spec.models,
        transform_temperature=spec.condition.temperature,
        seed_id=spec.seed_id,
        corpus_version=spec.corpus_version,
        corpus_sha256=spec.corpus_sha256,
        stop_consecutive=spec.stop_consecutive,
    )


def _report(directory: Path, reason: TerminationReason) -> int:
    message = f"{reason}: results saved to {directory}"
    if reason == "budget_exceeded":
        print(message, file=sys.stderr)
        return _EXIT_BUDGET
    if reason == "interrupted":
        print(message, file=sys.stderr)
        return _EXIT_INTERRUPTED
    print(message)
    return 0


def _fail(message: str, code: int) -> int:
    print(redact(message), file=sys.stderr)
    return code
