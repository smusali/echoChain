from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_MERMAID = "```mermaid\n"
_PACKAGE = "@mermaid-js/mermaid-cli@12.0.0"


def main() -> int:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    blocks = list(_blocks(listed.stdout.splitlines()))
    if not blocks:
        print("no mermaid blocks found", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index, block in enumerate(blocks):
            source = root / f"diagram-{index}.mmd"
            source.write_text(block, encoding="utf-8")
            result = subprocess.run(
                [
                    "npx",
                    "-y",
                    _PACKAGE,
                    "-i",
                    str(source),
                    "-o",
                    str(source.with_suffix(".svg")),
                    "-q",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(result.stderr or result.stdout, file=sys.stderr)
                return result.returncode
    return 0


def _blocks(paths: list[str]) -> list[str]:
    found: list[str] = []
    for name in paths:
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        while _MERMAID in text:
            text = text.split(_MERMAID, 1)[1]
            block, _, text = text.partition("```")
            found.append(block)
    return found


if __name__ == "__main__":
    raise SystemExit(main())
