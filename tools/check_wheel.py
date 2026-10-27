from __future__ import annotations

import zipfile
from pathlib import Path


def main() -> int:
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected one wheel, found {wheels}")
        return 1
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    needed = [
        f"echochain/prompts/{path.name}"
        for path in sorted(Path("src/echochain/prompts").glob("*.j2"))
    ]
    needed.extend(("echochain/py.typed", "echochain/seeds.toml"))
    missing = [name for name in needed if name not in names]
    if missing:
        print(f"wheel is missing {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
