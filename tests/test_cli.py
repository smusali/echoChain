import tomllib
from pathlib import Path

import pytest

from echochain import __version__
from echochain.cli import main


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    if not isinstance(project, dict):
        raise TypeError("project table is missing")
    value = project["version"]
    if not isinstance(value, str):
        raise TypeError("project.version must be a string")
    return value


def test_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"echochain {__version__}"


def test_help_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert "echochain" in captured.out
