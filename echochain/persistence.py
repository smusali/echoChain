import csv
import json
import pathlib
import datetime as dt
import traceback


def mk_exp_dir(seed: str) -> pathlib.Path:
    slug = "_".join(seed.lower().split()[:3])
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = pathlib.Path("experiments") / f"{slug}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Recorder:
    def __init__(self, exp_dir: pathlib.Path):
        self.csv = open(exp_dir / "chain.csv", "w", newline="")
        self.jl = open(exp_dir / "chain.jsonl", "w")
        self.w = csv.DictWriter(self.csv, fieldnames=["cycle", "phase", "model", "text", "score"])
        self.w.writeheader()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred in Recorder context: {exc_type.__name__}: {exc_val}")
            if exc_tb is not None:
                print("Traceback:")
                traceback.print_tb(exc_tb)

        self.csv.close()
        self.jl.close()

    def log(self, **row):
        self.w.writerow(row)
        self.csv.flush()
        self.jl.write(json.dumps(row) + "\n")
        self.jl.flush()
