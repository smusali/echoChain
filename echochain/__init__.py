import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version_from_pyproject() -> str:
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


def _package_version() -> str:
    try:
        return version("echochain")
    except PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _package_version()
