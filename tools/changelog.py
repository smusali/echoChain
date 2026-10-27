from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_PATH = Path("docs/CHANGELOG.md")
_ORDER = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
_LIMIT = 140
_VERSION = re.compile(r"\d+\.\d+\.\d+(?:rc\d+)?")


def validate(text: str) -> list[str]:
    problems: list[str] = []
    for version, sections in _versions(text):
        seen = -1
        for name, body in sections:
            if name not in _ORDER:
                problems.append(f"{version}: unknown section {name}")
                continue
            index = _ORDER.index(name)
            if index < seen:
                problems.append(f"{version}: {name} is out of order")
            seen = index
            problems.extend(_section_problems(version, name, body))
    return problems


def notes(text: str, version: str) -> str:
    number = version.removeprefix("v")
    pattern = re.compile(
        rf"^## \[{re.escape(number)}\][^\n]*\n(.*?)(?=^## |\n\[[^\]]+\]: |\Z)",
        re.M | re.S,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"changelog has no section for {number}")
    body = match.group(1).strip()
    policy = (
        "Versioning policy: "
        f"https://github.com/smusali/echoChain/blob/v{number}/README.md#stability"
    )
    return f"{body}\n\n{policy}"


def main(argv: list[str] | None = None) -> int:
    command = (argv if argv is not None else sys.argv[1:])[:1]
    action = command[0] if command else "check"
    text = _PATH.read_text(encoding="utf-8")
    if action == "notes":
        token = _version_token(os.environ.get("REF_NAME", ""))
        if token is None:
            print("REF_NAME must be a version tag", file=sys.stderr)
            return 2
        Path("release-notes.md").write_text(notes(text, token) + "\n", encoding="utf-8")
        return 0
    if action != "check":
        print("use check or notes", file=sys.stderr)
        return 2
    problems = validate(text)
    token = _version_token(os.environ.get("REF_NAME", ""))
    if token is not None and f"## [{token}]" not in text:
        problems.append(f"missing section {token}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    return 0


def _versions(text: str) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    versions: list[tuple[str, list[tuple[str, list[str]]]]] = []
    current = ""
    sections: list[tuple[str, list[str]]] = []
    name = ""
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if name:
                sections.append((name, body))
            if current:
                versions.append((current, sections))
            current = line[3:].strip()
            sections = []
            name = ""
            body = []
            continue
        if not current:
            continue
        if line.startswith("### "):
            if name:
                sections.append((name, body))
            name = line[4:].strip()
            body = []
            continue
        if name and line.strip() and not _is_link(line):
            body.append(line)
    if name:
        sections.append((name, body))
    if current:
        versions.append((current, sections))
    return versions


def _section_problems(version: str, name: str, body: list[str]) -> list[str]:
    if not body:
        return [f"{version}: {name} is empty"]
    bullets = [line for line in body if line.startswith("- ")]
    prose = [line for line in body if not line.startswith("- ")]
    if len(bullets) == 1:
        return [f"{version}: {name} is a one-item list"]
    if bullets and prose:
        return [f"{version}: {name} mixes a sentence and a list"]
    entries = [line[2:] for line in bullets] if bullets else prose
    problems: list[str] = []
    for entry in entries:
        problems.extend(
            f"{version}: {name}: {problem}" for problem in _entry_problems(entry)
        )
    return problems


def _entry_problems(entry: str) -> list[str]:
    problems: list[str] = []
    if len(entry) > _LIMIT:
        problems.append(f"entry is {len(entry)} characters")
    if _sentences(entry) != 1:
        problems.append("entry must be one sentence")
    return problems


def _sentences(text: str) -> int:
    count = 0
    for index, char in enumerate(text):
        if char not in ".!?":
            continue
        previous = text[index - 1] if index else ""
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if char == "." and previous.isdigit() and nxt.isdigit():
            continue
        if nxt in {"", " "}:
            count += 1
    return count


def _version_token(name: str) -> str | None:
    token = name.removeprefix("v")
    if _VERSION.fullmatch(token):
        return token
    return None


def _is_link(line: str) -> bool:
    return line.startswith("[") and "]: " in line


if __name__ == "__main__":
    raise SystemExit(main())
