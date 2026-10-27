from __future__ import annotations

import json
import sys
from pathlib import Path

from echochain.records import Manifest, Row

_MODELS = {"manifest": Manifest, "row": Row}
_DIRECTORY = Path("schemas")


def render() -> dict[str, str]:
    return {
        name: json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        for name, model in _MODELS.items()
    }


def main(argv: list[str] | None = None) -> int:
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    documents = render()
    _DIRECTORY.mkdir(exist_ok=True)
    for name, payload in documents.items():
        path = _DIRECTORY / f"{name}.schema.json"
        if check:
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current != payload:
                print(f"{path} does not match the models", file=sys.stderr)
                return 1
            continue
        path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
