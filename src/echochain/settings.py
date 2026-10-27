from __future__ import annotations

import json
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import Field, dataclass, field, fields, replace
from pathlib import Path
from typing import Literal, TypeVar

from echochain.coercion import (
    ConfigError,
    ProviderPin,
    _as_float,
    _as_int,
    _as_path,
    _bound_float,
    _bound_int,
    _decode_path,
    _decode_pins,
    _env_float,
    _env_int,
    _export,
    _loose,
    _pins,
    _plain,
    _strings,
)

ROTATING_MODELS = (
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4.1-flash-1017",
    "mistralai/mistral-small-2603",
)
JUDGE_MODEL = "openai/gpt-4.1-mini"
JUDGE_SAMPLES = 3
JUDGE_TEMPERATURE = 0.5
TRANSFORM_TEMPERATURE = 0.3
TOP_P = 1.0
MAX_CYCLES = 30
MAX_TOKENS = 1024
STOP_SCORE = 2.0
STOP_CONSECUTIVE = 0
CALL_DEADLINE_SECONDS = 60.0
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/smusali/echoChain",
    "X-Title": "echoChain",
}
CONFIRM_ABOVE_USD = 1.0
MAX_JUDGE_ERROR_SHARE = 0.1
_MAX_TEMPERATURE = 2.0
MISSING_API_KEY = "OPENROUTER_API_KEY is not set; export it or add it to `.env`"
REDACTED = "[REDACTED]"
_API_KEY_ENV = "OPENROUTER_API_KEY"
_EXPORT_PREFIX = "export "
_MIN_QUOTED_LENGTH = 2
_SELECTORS = ("start", "seed")
BaselineMode = Literal["off", "neural"]


class MissingAPIKeyError(RuntimeError):
    pass


def _meta(
    kind: str,
    env: str,
    default: object,
    *,
    flag: str | None = None,
    help_text: str = "",
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
    append: bool = False,
    choices: tuple[str, ...] = (),
    selector: bool = False,
) -> dict[str, object]:
    return {
        "kind": kind,
        "env": env,
        "default": default,
        "flag": flag,
        "help": help_text,
        "minimum": minimum,
        "maximum": maximum,
        "exclusive_min": exclusive_min,
        "append": append,
        "choices": choices,
        "selector": selector,
    }


_T = TypeVar("_T")


def _setting(
    default: _T,
    kind: str,
    env: str,
    /,
    *,
    flag: str | None = None,
    help_text: str = "",
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
    append: bool = False,
    choices: tuple[str, ...] = (),
    selector: bool = False,
) -> _T:
    described = field(
        default=default,
        metadata=_meta(
            kind,
            env,
            default,
            flag=flag,
            help_text=help_text,
            minimum=minimum,
            maximum=maximum,
            exclusive_min=exclusive_min,
            append=append,
            choices=choices,
            selector=selector,
        ),
    )
    return described


@dataclass(frozen=True)
class Settings:
    # fmt: off
    models: tuple[str, ...] = _setting(
        ROTATING_MODELS, "strings", "ECHOCHAIN_MODELS",
        flag="--model", help_text="Rotating model.", append=True,
    )
    judge: tuple[str, ...] = _setting(
        (JUDGE_MODEL,), "strings", "ECHOCHAIN_JUDGE",
        flag="--judge", help_text="Judge model.", append=True,
    )
    judge_samples: int = _setting(
        JUDGE_SAMPLES, "int", "ECHOCHAIN_JUDGE_SAMPLES",
        flag="--judge-samples", help_text="Samples per judge.", minimum=1,
    )
    judge_temperature: float = _setting(
        JUDGE_TEMPERATURE, "float", "ECHOCHAIN_JUDGE_TEMPERATURE",
        flag="--judge-temperature", help_text="Judge temperature.",
        minimum=0, maximum=_MAX_TEMPERATURE,
    )
    temperature: float = _setting(
        TRANSFORM_TEMPERATURE, "float", "ECHOCHAIN_TEMPERATURE",
        flag="--temperature", help_text="Transform temperature.",
        minimum=0, maximum=_MAX_TEMPERATURE,
    )
    max_cycles: int = _setting(
        MAX_CYCLES, "int", "ECHOCHAIN_MAX_CYCLES",
        flag="--max-cycles", help_text="Planned steps in one run.", minimum=1,
    )
    max_tokens: int = _setting(
        MAX_TOKENS, "int", "ECHOCHAIN_MAX_TOKENS",
        flag="--max-tokens", help_text="Completion token cap.", minimum=1,
    )
    stop_score: float = _setting(
        STOP_SCORE, "float", "ECHOCHAIN_STOP_SCORE",
        flag="--stop-score", help_text="Low fidelity threshold.",
    )
    stop_consecutive: int = _setting(
        STOP_CONSECUTIVE, "int", "ECHOCHAIN_STOP_CONSECUTIVE",
        flag="--stop-consecutive", help_text="Low scores that stop a run.",
        minimum=0,
    )
    repeats: int = _setting(
        1, "int", "ECHOCHAIN_REPEATS",
        flag="--repeats", help_text="Repetitions of each pair.", minimum=1,
    )
    budget: float | None = _setting(
        None, "optional_float", "ECHOCHAIN_BUDGET",
        flag="--budget", help_text="Stop at this many USD.", minimum=0,
    )
    out: Path = _setting(  # noqa: RUF009
        Path("runs"), "path", "ECHOCHAIN_OUT",
        flag="--out", help_text="Directory that receives run folders.",
    )
    condition: tuple[str, ...] = _setting(
        ("alternate",), "strings", "ECHOCHAIN_CONDITION",
        flag="--condition", help_text="Condition name. Repeat to run several.",
        append=True,
    )
    seeds: Path | None = _setting(  # noqa: RUF009
        None, "optional_path", "ECHOCHAIN_SEEDS",
        flag="--seeds",
        help_text="Seed file. Every seed runs when --seed is omitted.",
    )
    seed: str | None = _setting(
        None, "optional_str", "ECHOCHAIN_SEED",
        flag="--seed", help_text="Seed id from the seed file.", selector=True,
    )
    start: str | None = _setting(
        None, "optional_str", "ECHOCHAIN_START",
        flag="--start", help_text="Seed answer text.", selector=True,
    )
    sampling_seed: int | None = _setting(
        None, "optional_int", "ECHOCHAIN_SAMPLING_SEED",
        flag="--sampling-seed", help_text="Base sampling seed.", minimum=1,
    )
    top_p: float = _setting(
        TOP_P, "float", "ECHOCHAIN_TOP_P",
        flag="--top-p", help_text="Nucleus sampling cutoff.",
        minimum=0, maximum=1,
    )
    providers: tuple[tuple[str, ProviderPin], ...] = _setting(
        (), "pins", "ECHOCHAIN_PROVIDERS",
        flag="--provider", help_text="Pin as model=provider,provider|true.",
        append=True,
    )
    deadline_seconds: float = _setting(
        CALL_DEADLINE_SECONDS, "float", "ECHOCHAIN_DEADLINE_SECONDS",
        exclusive_min=True, minimum=0,
    )
    base_url: str = _setting(OPENROUTER_API_BASE, "url", "OPENROUTER_API_BASE")
    http_referer: str = _setting(
        OPENROUTER_HEADERS["HTTP-Referer"], "str", "ECHOCHAIN_HTTP_REFERER",
    )
    title: str = _setting(OPENROUTER_HEADERS["X-Title"], "str", "ECHOCHAIN_TITLE")
    confirm_above_usd: float = _setting(
        CONFIRM_ABOVE_USD, "float", "ECHOCHAIN_CONFIRM_ABOVE", minimum=0,
    )
    baseline: str = _setting(
        "off", "choice", "ECHOCHAIN_BASELINE", choices=("off", "neural"),
    )
    max_judge_error_share: float = _setting(
        MAX_JUDGE_ERROR_SHARE, "float", "ECHOCHAIN_MAX_JUDGE_ERROR_SHARE",
        minimum=0, maximum=1,
    )
    # fmt: on


def request_headers(settings: Settings) -> dict[str, str]:
    return {"HTTP-Referer": settings.http_referer, "X-Title": settings.title}


def default_settings() -> Settings:
    return Settings()


def resolve_settings(
    *,
    config: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    env = os.environ if environ is None else environ
    merged = _defaults()
    path = _config_path(config, env)
    if path is not None:
        merged = _overlay(merged, _layer_from_toml(path))
    merged = _overlay(merged, _layer_from_env(env))
    if overrides:
        merged = _overlay(merged, _layer_from_overrides(overrides))
    return _finalize(merged)


def render_settings(settings: Settings) -> str:
    text = json.dumps(settings_payload(settings), indent=2, sort_keys=True) + "\n"
    return redact(text)


def settings_payload(settings: Settings) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in fields(Settings):
        payload[item.name] = _export(item.name, getattr(settings, item.name))
    return payload


def settings_from_payload(payload: Mapping[str, object], *, source: str) -> Settings:
    layer = _layer_from_mapping(source, payload, None)
    return _finalize(_overlay(_defaults(), layer))


def load_api_key(env_path: Path | None = None) -> str:
    env_value = os.environ.get(_API_KEY_ENV, "").strip()
    if env_value:
        return env_value
    path = Path(".env") if env_path is None else env_path
    file_value = _dotenv_value(path, _API_KEY_ENV).strip()
    if file_value:
        os.environ[_API_KEY_ENV] = file_value
        return file_value
    raise MissingAPIKeyError(MISSING_API_KEY)


def redact(text: str) -> str:
    secret = os.environ.get(_API_KEY_ENV, "").strip()
    if not secret:
        return text
    return text.replace(secret, REDACTED)


def log_filter() -> logging.Filter:
    return _RedactFilter()


def model_family(model_id: str) -> str:
    vendor, separator, _name = model_id.partition("/")
    if separator == "" or vendor == "":
        return model_id
    return vendor


def with_client_limits(
    settings: Settings, *, max_tokens: int, deadline_seconds: float
) -> Settings:
    return replace(settings, max_tokens=max_tokens, deadline_seconds=deadline_seconds)


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        secret = os.environ.get(_API_KEY_ENV, "").strip()
        if secret and isinstance(record.msg, str):
            record.msg = record.msg.replace(secret, REDACTED)
        args = record.args
        if secret and isinstance(args, dict):
            record.args = {
                key: (
                    value.replace(secret, REDACTED) if isinstance(value, str) else value
                )
                for key, value in args.items()
            }
        elif secret and isinstance(args, tuple):
            record.args = tuple(
                value.replace(secret, REDACTED) if isinstance(value, str) else value
                for value in args
            )
        return True


def _defaults() -> dict[str, object]:
    values: dict[str, object] = {}
    for item in fields(Settings):
        values[item.name] = item.metadata["default"]
    return values


def _config_path(explicit: Path | None, environ: Mapping[str, str]) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file was not found: {explicit}")
        return explicit
    raw = environ.get("ECHOCHAIN_CONFIG", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise ConfigError(f"config file was not found: {path}")
        return path
    default = Path("echochain.toml")
    if default.is_file():
        return default
    return None


def _overlay(base: dict[str, object], layer: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    if any(key in layer for key in _SELECTORS):
        for key in _SELECTORS:
            merged[key] = None
    merged.update(layer)
    return merged


def _layer_from_toml(path: Path) -> dict[str, object]:
    source = str(path)
    try:
        parsed: object = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{source}: config must be a table")
    table = {str(key): value for key, value in parsed.items()}
    return _layer_from_mapping(source, table, path)


def _layer_from_mapping(
    source: str, table: Mapping[str, object], path: Path | None
) -> dict[str, object]:
    if "api_key" in table or "OPENROUTER_API_KEY" in table:
        raise ConfigError(
            f"{source}: api_key is not a config key; export OPENROUTER_API_KEY "
            "or add it to .env"
        )
    _one_selector(source, table)
    known = {item.name for item in fields(Settings)}
    layer: dict[str, object] = {}
    for key, value in table.items():
        if key not in known:
            raise ConfigError(f"{source}: unknown key {key}")
        layer[key] = _decode(key, value, source, path)
    return layer


def _layer_from_env(environ: Mapping[str, str]) -> dict[str, object]:
    present: dict[str, str] = {}
    for item in fields(Settings):
        name = str(item.metadata["env"])
        text = environ.get(name, "").strip()
        if text:
            present[item.name] = text
    _one_selector("environment", present)
    return {
        key: _decode(key, _env_value(key, name, text), name, None)
        for key, text in present.items()
        for name in [str(_field(key).metadata["env"])]
    }


def _layer_from_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    _one_selector("command line", overrides)
    known = {item.name for item in fields(Settings)}
    layer: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in known:
            raise ConfigError(f"command line: unknown key {key}")
        source = _field(key).metadata.get("flag") or "command line"
        layer[key] = _decode(key, value, str(source), None)
    return layer


def _one_selector(source: str, keys: Mapping[str, object]) -> None:
    chosen = [key for key in _SELECTORS if key in keys and keys[key] is not None]
    if len(chosen) > 1:
        raise ConfigError(f"{source}: provide one of start or seed")


def _finalize(values: Mapping[str, object]) -> Settings:
    settings = Settings(
        models=_strings(values["models"]),
        judge=_strings(values["judge"]),
        judge_samples=_as_int(values["judge_samples"]),
        judge_temperature=_as_float(values["judge_temperature"]),
        temperature=_as_float(values["temperature"]),
        max_cycles=_as_int(values["max_cycles"]),
        max_tokens=_as_int(values["max_tokens"]),
        stop_score=_as_float(values["stop_score"]),
        stop_consecutive=_as_int(values["stop_consecutive"]),
        repeats=_as_int(values["repeats"]),
        budget=None if values["budget"] is None else _as_float(values["budget"]),
        out=_as_path(values["out"]),
        condition=_strings(values["condition"]),
        seeds=None if values["seeds"] is None else _as_path(values["seeds"]),
        seed=None if values["seed"] is None else str(values["seed"]),
        start=None if values["start"] is None else str(values["start"]),
        sampling_seed=(
            None
            if values["sampling_seed"] is None
            else _as_int(values["sampling_seed"])
        ),
        top_p=_as_float(values["top_p"]),
        providers=_pins(values["providers"]),
        deadline_seconds=_as_float(values["deadline_seconds"]),
        base_url=str(values["base_url"]),
        http_referer=str(values["http_referer"]),
        title=str(values["title"]),
        confirm_above_usd=_as_float(values["confirm_above_usd"]),
        baseline=str(values["baseline"]),
        max_judge_error_share=_as_float(values["max_judge_error_share"]),
    )
    _reject_shared_family(settings.models, settings.judge)
    if settings.start is not None and (
        settings.seed is not None or settings.seeds is not None
    ):
        raise ConfigError("start cannot be combined with seed or seeds")
    return settings


def _reject_shared_family(models: tuple[str, ...], judges: tuple[str, ...]) -> None:
    generated = {model_family(model) for model in models}
    for judge in judges:
        family = model_family(judge)
        if family in generated:
            raise ConfigError(f"judge {judge} shares a family with a generator")


def _field(name: str) -> Field[Settings]:
    for item in fields(Settings):
        if item.name == name:
            return item
    raise ConfigError(f"unknown key {name}")


def _decode(key: str, value: object, source: str, path: Path | None) -> object:
    meta = _field(key).metadata
    kind = str(meta["kind"])
    if kind == "strings":
        return _string_list(source, key, value)
    if kind == "pins":
        return _decode_pins(source, value)
    if kind in {"int", "optional_int"}:
        return _bound_int(source, key, value, meta)
    if kind in {"float", "optional_float"}:
        return _bound_float(source, key, value, meta)
    if kind == "path":
        return _decode_path(source, key, value, path)
    if kind == "optional_path":
        if value is None:
            return None
        return _decode_path(source, key, value, path)
    if kind == "optional_str":
        if value is None:
            return None
        if key == "start":
            return _loose(source, key, value)
        return _plain(source, key, value)
    if kind == "choice":
        text = _plain(source, key, value)
        choices = meta["choices"]
        if not isinstance(choices, tuple) or text not in choices:
            raise ConfigError(f"{source}: {key} is not an allowed value")
        return text
    if kind == "url":
        text = _plain(source, key, value)
        if text.startswith(("https://", "http://")):
            return text
        raise ConfigError(f"{source}: {key} must start with http:// or https://")
    return _plain(source, key, value)


def _env_value(key: str, source: str, text: str) -> object:
    kind = str(_field(key).metadata["kind"])
    if kind in {"int", "optional_int"}:
        return _env_int(source, key, text)
    if kind in {"float", "optional_float"}:
        return _env_float(source, key, text)
    if kind == "strings":
        return [part.strip() for part in text.split(",") if part.strip() != ""]
    return text


def _string_list(source: str, key: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values: list[object] = [value]
    elif isinstance(value, list | tuple):
        values = list(value)
    else:
        raise ConfigError(f"{source}: {key} must be a string or a list")
    if not values:
        raise ConfigError(f"{source}: {key} must list at least one value")
    items = tuple(_plain(source, key, item) for item in values)
    if key in {"models", "judge"}:
        for item in items:
            if any(char.isspace() for char in item):
                raise ConfigError(f"{source}: {key} must be a model id without spaces")
    return items


def _dotenv_value(path: Path, name: str) -> str:
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix(_EXPORT_PREFIX)
        key, separator, value = line.partition("=")
        if separator == "" or key.strip() != name:
            continue
        return _unquote(value.strip())
    return ""


def _unquote(value: str) -> str:
    if len(value) < _MIN_QUOTED_LENGTH:
        return value
    if value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
