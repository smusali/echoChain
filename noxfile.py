#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox"]
# ///

import nox

nox.needs_version = "2025.10.14"
nox.options.default_venv_backend = "uv"


@nox.session(venv_backend="none")
def tests(session: nox.Session) -> None:
    session.run("uv", "run", "pytest")


if __name__ == "__main__":
    nox.main()
