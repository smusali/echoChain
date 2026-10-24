from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TypedDict

from echochain.config import redact

_FIELDNAMES = ("cycle", "phase", "model", "served_model", "text", "score")
_SLUG_WORDS = 3


class CycleRecord(TypedDict):
    cycle: int
    phase: str
    model: str
    served_model: str
    text: str
    score: float


def mk_exp_dir(seed: str) -> Path:
    """Create a timestamped experiment directory.

    Args:
        seed: Seed text used to name the directory.

    Returns:
        The new directory path.
    """
    slug = "_".join(seed.lower().split()[:_SLUG_WORDS])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = Path("experiments") / f"{slug}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Recorder:
    def __init__(self, exp_dir: Path) -> None:
        """Open the experiment output files.

        Args:
            exp_dir: Directory that receives the files.
        """
        self.csv = (exp_dir / "chain.csv").open("w", newline="", encoding="utf-8")
        self.jl = (exp_dir / "chain.jsonl").open("w", encoding="utf-8")
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

    def log(self, record: CycleRecord) -> None:
        """Append one cycle.

        Args:
            record: Cycle fields to store. Text fields are redacted.
        """
        row = {
            "cycle": record["cycle"],
            "phase": redact(record["phase"]),
            "model": redact(record["model"]),
            "served_model": redact(record["served_model"]),
            "text": redact(record["text"]),
            "score": record["score"],
        }
        self._writer.writerow(row)
        self.csv.flush()
        payload = json.dumps(row)
        self.jl.write(payload + "\n")
        self.jl.flush()
