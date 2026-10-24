import argparse
import textwrap

from echochain import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="echochain",
        description="Measure information degradation through LLM telephone.",
    )
    parser.add_argument("--version", action="version", version=f"echochain {__version__}")
    parser.add_argument("--start", required=True, help="Seed answer text.")
    args = parser.parse_args(argv)

    from echochain.engine import cycle
    from echochain.persistence import Recorder, mk_exp_dir

    exp_dir = mk_exp_dir(args.start)
    with Recorder(exp_dir) as recorder:
        cycle(args.start, recorder=recorder)
    print(textwrap.dedent(f"Degradation complete: results saved to {exp_dir}").strip())
    return 0
