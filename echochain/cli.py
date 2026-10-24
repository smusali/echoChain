from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from echochain import __version__
from echochain import client as client_mod
from echochain.config import (
    MISSING_API_KEY,
    MissingAPIKeyError,
    configured_models,
    load_api_key,
    redact,
)
from echochain.engine import cycle
from echochain.persistence import Recorder, mk_exp_dir


def main(argv: list[str] | None = None) -> int:
    """Run one telephone experiment.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Process exit code. `2` means the key or a model is unusable.
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
    args = parser.parse_args(argv)
    try:
        exp_dir = _run(args.start)
    except MissingAPIKeyError:
        return _fail(MISSING_API_KEY, 2)
    except client_mod.ModelUnavailableError as exc:
        return _fail(str(exc), 2)
    except client_mod.ModelCatalogError as exc:
        return _fail(str(exc), 1)
    except Exception:
        return _fail(traceback.format_exc(), 1)
    print(f"Degradation complete: results saved to {exp_dir}")
    return 0


def _run(seed: str) -> Path:
    api_key = load_api_key()
    catalog = client_mod.fetch_catalog(api_key)
    client_mod.check_models(catalog, configured_models())
    exp_dir = mk_exp_dir(seed)
    with Recorder(exp_dir) as recorder:
        cycle(seed, recorder, client_mod.OpenRouterClient(api_key))
    return exp_dir


def _fail(message: str, code: int) -> int:
    print(redact(message), file=sys.stderr)
    return code
