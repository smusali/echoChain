from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from echochain import __version__
from echochain import client as client_mod
from echochain.config import (
    MissingAPIKeyError,
    configured_models,
    load_api_key,
    redact,
)
from echochain.engine import cycle
from echochain.persistence import (
    EmptySeedError,
    Recorder,
    mk_exp_dir,
    running_manifest,
)

_EXIT_BUDGET = 3
_EXIT_CONFIG = 2


class InvalidBudgetError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    """Run one telephone experiment.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. `2` means the seed, budget, key, or a model is
        unusable. `3` means the budget was exceeded.
    """
    parser = argparse.ArgumentParser(
        prog="echochain",
        description="Measure information degradation through LLM telephone.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"echochain {__version__}",
    )
    parser.add_argument("--start", required=True, help="Seed answer text.")
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
    args = parser.parse_args(argv)
    try:
        exp_dir, reason = _run(args.start, args.out, args.budget)
    except (
        EmptySeedError,
        InvalidBudgetError,
        MissingAPIKeyError,
        client_mod.ModelUnavailableError,
    ) as exc:
        return _fail(str(exc), _EXIT_CONFIG)
    except client_mod.ModelCatalogError as exc:
        return _fail(str(exc), 1)
    except Exception:
        return _fail(traceback.format_exc(), 1)
    if reason == "budget_exceeded":
        print(f"budget_exceeded: results saved to {exp_dir}", file=sys.stderr)
        return _EXIT_BUDGET
    print(f"Degradation complete ({reason}): results saved to {exp_dir}")
    return 0


def _run(seed: str, out: Path, budget: float | None) -> tuple[Path, str]:
    if seed.strip() == "":
        raise EmptySeedError("seed is empty")
    if budget is not None and budget < 0:
        raise InvalidBudgetError("budget must be zero or greater")
    api_key = load_api_key()
    catalog = client_mod.fetch_catalog(api_key)
    client_mod.check_models(catalog, configured_models())
    allocated = mk_exp_dir(seed, out)
    manifest = running_manifest(
        run_id=allocated.run_id,
        slug=allocated.slug,
        seed=seed,
        budget=budget,
    )
    with Recorder(allocated.directory, manifest) as recorder:
        with client_mod.OpenRouterClient(api_key) as client:
            reason = cycle(
                seed,
                recorder,
                client,
                run_id=allocated.run_id,
                budget=budget,
            )
        recorder.finish(reason)
    return allocated.directory, reason


def _fail(message: str, code: int) -> int:
    print(redact(message), file=sys.stderr)
    return code
