from __future__ import annotations

import re
import subprocess
import sys

_OLD_REPO = "llm-" + "telephone"
_OLD_PACKAGE = "llm_" + "tel"
_OLD_USER = "your" + "username"
_OLD_TITLE = "LLM " + "Telephone"
_ALLOWED = "yymss/" + _OLD_REPO
_SKIP = ("CHANGELOG.md", "docs/CHANGELOG.md")


def main() -> int:
    pattern = "|".join((_OLD_REPO, _OLD_PACKAGE, _OLD_USER, _OLD_TITLE))
    hits = _clean(_git("grep", "-nE", pattern), content=True)
    paths = _clean(_git("ls-files"), content=False, pattern=pattern)
    if hits or paths:
        print("\n".join([*hits, *paths]), file=sys.stderr)
        return 1
    return 0


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise SystemExit(result.stderr.strip() or "git failed")
    return result.stdout.splitlines()


def _clean(lines: list[str], *, content: bool, pattern: str = "") -> list[str]:
    kept: list[str] = []
    for line in lines:
        path = line.split(":", 1)[0] if content else line
        if path in _SKIP or _ALLOWED in line:
            continue
        if not content and pattern and not _matches(line, pattern):
            continue
        kept.append(line)
    return kept


def _matches(path: str, pattern: str) -> bool:
    return re.search(pattern, path) is not None


if __name__ == "__main__":
    raise SystemExit(main())
