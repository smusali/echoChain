from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
project = data["project"]
if not isinstance(project, dict):
    raise SystemExit("project table is missing")
version = project["version"]
if not isinstance(version, str):
    raise SystemExit("project.version must be a string")
changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
heading = re.compile(
    rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$",
    re.M,
)
if heading.search(changelog) is None:
    raise SystemExit(f"CHANGELOG.md has no dated section for {version}")
args = sys.argv[1:]
if args and args[0] != f"v{version}":
    raise SystemExit(f"tag {args[0]} does not match version {version}")
