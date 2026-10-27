from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_EM = "\u2014"
_EN = "\u2013"
_SPACED_HYPHEN = " -- "
_LABEL_LIMIT = 24
_MAX_WORDS = 3
_FENCE = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_MERMAID = re.compile(r"```mermaid\n(.*?)```", re.S)
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+\.)\s+\S")
_LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")
_END_NODE = re.compile(r"(?<![\w-])end(?![\w-])")


def problems(text: str, *, path: str) -> list[str]:
    found = _dash_problems(_plain(text))
    found.extend(_list_problems(_plain(text)))
    found.extend(_mermaid_problems(text))
    if path == "README.md":
        found.extend(_link_problems(_plain(text)))
    return found


def main() -> int:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    failed = False
    for name in listed.stdout.splitlines():
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for problem in problems(text, path=name):
            print(f"{name}: {problem}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def _plain(text: str) -> str:
    return _INLINE.sub("", _FENCE.sub("", text))


def _dash_problems(text: str) -> list[str]:
    found: list[str] = []
    if _EM in text:
        found.append("em dash")
    if _EN in text:
        found.append("en dash")
    if _SPACED_HYPHEN in text:
        found.append("spaced double hyphen")
    return found


def _list_problems(text: str) -> list[str]:
    found: list[str] = []
    group: list[str] = []
    for line in [*text.splitlines(), ""]:
        if _LIST_ITEM.match(line):
            group.append(line)
            continue
        if len(group) == 1:
            found.append("one-item list")
        group = []
    return found


def _mermaid_problems(text: str) -> list[str]:
    found: list[str] = []
    for block in _MERMAID.findall(text):
        if "\\n" in block:
            found.append("mermaid newline")
        if _END_NODE.search(block):
            found.append("mermaid end node")
        for label in re.findall(r'"([^"]*)"', block):
            words = label.split()
            if len(label) > _LABEL_LIMIT or not 1 <= len(words) <= _MAX_WORDS:
                found.append("mermaid label")
    return found


def _link_problems(text: str) -> list[str]:
    found: list[str] = []
    for match in _LINK.finditer(text):
        target = match.group(3).strip()
        if target.startswith(("https://", "http://", "mailto:")):
            continue
        kind = "image" if match.group(1) == "!" else "link"
        found.append(f"relative {kind}")
    return found


if __name__ == "__main__":
    raise SystemExit(main())
