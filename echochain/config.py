from __future__ import annotations

import json
import logging
import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from echochain.prompts import RUBRIC_VERSION

ROTATING_MODELS = [
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4.1-flash-1017",
    "mistralai/mistral-small-2603",
]
JUDGE_MODEL = "openai/gpt-4.1-mini"
JUDGE_SAMPLES = 1
JUDGE_TEMPERATURE = 0.0
TRANSFORM_TEMPERATURE = 0.3
MAX_CYCLES = 30
MAX_TOKENS = 1024
STOP_SCORE = 2.0
STOP_CONSECUTIVE = 0
SCORE_MIN = 1
SCORE_MAX = 10
JUDGE_ERROR_NUMERATOR = 1
JUDGE_ERROR_DENOMINATOR = 10
CALL_DEADLINE_SECONDS = 60.0
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/smusali/echoChain",
    "X-Title": "echoChain",
}
REQUIRED_PARAMETERS = ("response_format", "seed")
CONFIRM_ABOVE_USD = 1.0
_MAX_TEMPERATURE = 2.0
_PAIR_LENGTH = 2
MISSING_API_KEY = "OPENROUTER_API_KEY is not set; export it or add it to `.env`"
REDACTED = "[REDACTED]"
_API_KEY_ENV = "OPENROUTER_API_KEY"
_EXPORT_PREFIX = "export "
_MIN_QUOTED_LENGTH = 2


class MissingAPIKeyError(RuntimeError):
    pass


def configured_models() -> tuple[str, ...]:
    """Return the rotating models and the judge.

    Returns:
        Model IDs that must pass preflight.
    """
    return (*ROTATING_MODELS, JUDGE_MODEL)


def load_api_key(env_path: Path | None = None) -> str:
    """Return the OpenRouter key from the environment or `.env`.

    Args:
        env_path: `.env` path. Defaults to `.env` in the working directory.

    Returns:
        The non-empty API key.

    Raises:
        MissingAPIKeyError: The key is missing from the environment and `.env`.
    """
    env_value = os.environ.get(_API_KEY_ENV, "").strip()
    if env_value:
        _install_redaction()
        return env_value
    path = Path(".env") if env_path is None else env_path
    file_value = _dotenv_value(path, _API_KEY_ENV).strip()
    if file_value:
        os.environ[_API_KEY_ENV] = file_value
        _install_redaction()
        return file_value
    raise MissingAPIKeyError(MISSING_API_KEY)


def redact(text: str) -> str:
    """Replace the API key in text.

    Args:
        text: Text that may contain the key.

    Returns:
        Text with the key replaced by `[REDACTED]`.
    """
    secret = os.environ.get(_API_KEY_ENV, "").strip()
    if not secret:
        return text
    return text.replace(secret, REDACTED)


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


def _install_redaction() -> None:
    logging.setLoggerClass(_RedactingLogger)
    _attach_filter(logging.getLogger())
    for existing in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(existing, logging.Logger):
            _attach_filter(existing)


def _attach_filter(logger: logging.Logger) -> None:
    if any(isinstance(item, _RedactFilter) for item in logger.filters):
        return
    logger.addFilter(_RedactFilter())


class _RedactingLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self.addFilter(_RedactFilter())


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        secret = os.environ.get(_API_KEY_ENV, "").strip()
        if secret:
            _redact_record(record, secret)
        return True


def _redact_record(record: logging.LogRecord, secret: str) -> None:
    if isinstance(record.msg, str):
        record.msg = record.msg.replace(secret, REDACTED)
    args = record.args
    if isinstance(args, dict):
        record.args = {
            key: _redact_object(value, secret) for key, value in args.items()
        }
    elif isinstance(args, tuple):
        record.args = tuple(_redact_object(value, secret) for value in args)
    if record.exc_text:
        record.exc_text = record.exc_text.replace(secret, REDACTED)


def _redact_object(value: object, secret: str) -> object:
    if isinstance(value, str):
        return value.replace(secret, REDACTED)
    return value


class ConfigError(RuntimeError):
    pass


DataCollection = Literal["allow", "deny"]


@dataclass(frozen=True)
class ProviderPin:
    order: tuple[str, ...]
    allow_fallbacks: bool
    data_collection: DataCollection | None = None
    zdr: bool | None = None


@dataclass(frozen=True)
class Settings:
    models: tuple[str, ...]
    judge: tuple[str, ...]
    judge_samples: int
    judge_temperature: float
    temperature: float
    max_cycles: int
    max_tokens: int
    stop_score: float
    stop_consecutive: int
    score_min: int
    score_max: int
    judge_error_numerator: int
    judge_error_denominator: int
    deadline_seconds: float
    base_url: str
    http_referer: str
    title: str
    required_parameters: tuple[str, ...]
    rubric_version: str
    repeats: int
    budget: float | None
    out: Path
    confirm_above_usd: float
    condition: tuple[str, ...]
    providers: tuple[tuple[str, ProviderPin], ...]
    seeds: Path | None
    seed: str | None
    split: str | None
    start: str | None


@dataclass(frozen=True)
class _Bound:
    value: object
    source: str


_SELECTORS = ("start", "seed", "split")
_PATHS = frozenset({"out", "seeds"})
_LISTS = frozenset({"models", "judge", "condition", "required_parameters"})
_INTEGERS = frozenset(
    {
        "judge_samples",
        "max_cycles",
        "max_tokens",
        "stop_consecutive",
        "score_min",
        "score_max",
        "judge_error_numerator",
        "judge_error_denominator",
        "repeats",
    }
)
_FLOATS = frozenset(
    {
        "judge_temperature",
        "temperature",
        "stop_score",
        "deadline_seconds",
        "budget",
        "confirm_above_usd",
    }
)
_FLOORS = {
    "judge_samples": 1,
    "max_cycles": 1,
    "max_tokens": 1,
    "repeats": 1,
    "judge_error_denominator": 1,
    "stop_consecutive": 0,
    "judge_error_numerator": 0,
}
_PLAIN = frozenset({"http_referer", "title", "rubric_version", "seed"})
_ENV = {
    "models": "ECHOCHAIN_MODELS",
    "judge": "ECHOCHAIN_JUDGE",
    "judge_samples": "ECHOCHAIN_JUDGE_SAMPLES",
    "judge_temperature": "ECHOCHAIN_JUDGE_TEMPERATURE",
    "temperature": "ECHOCHAIN_TEMPERATURE",
    "max_cycles": "ECHOCHAIN_MAX_CYCLES",
    "max_tokens": "ECHOCHAIN_MAX_TOKENS",
    "stop_score": "ECHOCHAIN_STOP_SCORE",
    "stop_consecutive": "ECHOCHAIN_STOP_CONSECUTIVE",
    "score_min": "ECHOCHAIN_SCORE_MIN",
    "score_max": "ECHOCHAIN_SCORE_MAX",
    "judge_error_numerator": "ECHOCHAIN_JUDGE_ERROR_NUMERATOR",
    "judge_error_denominator": "ECHOCHAIN_JUDGE_ERROR_DENOMINATOR",
    "deadline_seconds": "ECHOCHAIN_DEADLINE_SECONDS",
    "base_url": "OPENROUTER_API_BASE",
    "http_referer": "ECHOCHAIN_HTTP_REFERER",
    "title": "ECHOCHAIN_TITLE",
    "required_parameters": "ECHOCHAIN_REQUIRED_PARAMETERS",
    "rubric_version": "ECHOCHAIN_RUBRIC_VERSION",
    "repeats": "ECHOCHAIN_REPEATS",
    "budget": "ECHOCHAIN_BUDGET",
    "out": "ECHOCHAIN_OUT",
    "confirm_above_usd": "ECHOCHAIN_CONFIRM_ABOVE",
    "condition": "ECHOCHAIN_CONDITION",
    "seeds": "ECHOCHAIN_SEEDS",
    "seed": "ECHOCHAIN_SEED",
    "split": "ECHOCHAIN_SPLIT",
    "start": "ECHOCHAIN_START",
    "providers": "ECHOCHAIN_PROVIDERS",
}
_FLAGS = {
    "models": "--model",
    "judge": "--judge",
    "judge_samples": "--judge-samples",
    "judge_temperature": "--judge-temperature",
    "temperature": "--temperature",
    "max_cycles": "--max-cycles",
    "max_tokens": "--max-tokens",
    "stop_score": "--stop-score",
    "stop_consecutive": "--stop-consecutive",
    "score_min": "--score-min",
    "score_max": "--score-max",
    "judge_error_numerator": "--judge-error-numerator",
    "judge_error_denominator": "--judge-error-denominator",
    "deadline_seconds": "--deadline",
    "base_url": "--base-url",
    "http_referer": "--http-referer",
    "title": "--title",
    "required_parameters": "--require",
    "rubric_version": "--rubric-version",
    "repeats": "--repeats",
    "budget": "--budget",
    "out": "--out",
    "confirm_above_usd": "--confirm-above",
    "condition": "--condition",
    "seeds": "--seeds",
    "seed": "--seed",
    "split": "--split",
    "start": "--start",
    "providers": "--provider",
}
_FILE_KEYS = frozenset(_ENV) | {"providers"}


def request_headers(settings: Settings) -> dict[str, str]:
    """Return the OpenRouter attribution headers.

    Args:
        settings: Resolved settings.

    Returns:
        The `HTTP-Referer` and `X-Title` headers.
    """
    return {"HTTP-Referer": settings.http_referer, "X-Title": settings.title}


def default_settings() -> Settings:
    """Return the settings used when nothing overrides them.

    Returns:
        Settings copied from the module constants.
    """
    return _finalize(_defaults())


def resolve_settings(
    *,
    config: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    """Resolve experiment settings.

    Args:
        config: Config file. When omitted, `ECHOCHAIN_CONFIG` or
            `echochain.toml` is used.
        environ: Environment mapping. Defaults to the process environment.
        overrides: Values set by CLI flags. These win over the environment
            and the file.

    Returns:
        The resolved settings.

    Raises:
        ConfigError: A value is missing, unknown, or invalid.
    """
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
    """Serialize settings for display.

    Args:
        settings: Resolved settings.

    Returns:
        JSON text with the API key removed.
    """
    payload = _payload(settings)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return redact(text)


def _defaults() -> dict[str, _Bound]:
    source = "defaults"
    values: dict[str, object] = {
        "models": tuple(ROTATING_MODELS),
        "judge": (JUDGE_MODEL,),
        "judge_samples": JUDGE_SAMPLES,
        "judge_temperature": JUDGE_TEMPERATURE,
        "temperature": TRANSFORM_TEMPERATURE,
        "max_cycles": MAX_CYCLES,
        "max_tokens": MAX_TOKENS,
        "stop_score": STOP_SCORE,
        "stop_consecutive": STOP_CONSECUTIVE,
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "judge_error_numerator": JUDGE_ERROR_NUMERATOR,
        "judge_error_denominator": JUDGE_ERROR_DENOMINATOR,
        "deadline_seconds": CALL_DEADLINE_SECONDS,
        "base_url": OPENROUTER_API_BASE,
        "http_referer": OPENROUTER_HEADERS["HTTP-Referer"],
        "title": OPENROUTER_HEADERS["X-Title"],
        "required_parameters": tuple(REQUIRED_PARAMETERS),
        "rubric_version": RUBRIC_VERSION,
        "repeats": 1,
        "budget": None,
        "out": Path("runs"),
        "confirm_above_usd": CONFIRM_ABOVE_USD,
        "condition": ("alternate",),
        "providers": (),
    }
    return {key: _Bound(value, source) for key, value in values.items()}


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


def _overlay(
    base: dict[str, _Bound],
    layer: dict[str, _Bound],
) -> dict[str, _Bound]:
    merged = dict(base)
    if any(key in layer for key in _SELECTORS):
        for key in _SELECTORS:
            merged.pop(key, None)
    merged.update(layer)
    return merged


def _layer_from_toml(path: Path) -> dict[str, _Bound]:
    source = str(path)
    try:
        parsed: object = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    table = _mapping(parsed, source, "config")
    _reject_secret(source, table)
    _one_selector(source, table, "provide one of start, seed, or split")
    layer: dict[str, _Bound] = {}
    for key, value in table.items():
        if key not in _FILE_KEYS:
            raise ConfigError(f"{source}: unknown key {key}")
        layer[key] = _Bound(_decode(source, key, value, path), source)
    return layer


def _layer_from_env(environ: Mapping[str, str]) -> dict[str, _Bound]:
    present = {
        key: name for key, name in _ENV.items() if environ.get(name, "").strip() != ""
    }
    _one_selector(
        "environment",
        present,
        "provide one of ECHOCHAIN_START, ECHOCHAIN_SEED, or ECHOCHAIN_SPLIT",
    )
    layer: dict[str, _Bound] = {}
    for key, name in present.items():
        text = environ[name].strip()
        value = _from_env(key, name, text)
        layer[key] = _Bound(_decode(name, key, value, None), name)
    return layer


def _layer_from_overrides(overrides: Mapping[str, object]) -> dict[str, _Bound]:
    _one_selector(
        "command line",
        overrides,
        "provide one of --start, --seed, or --split",
    )
    layer: dict[str, _Bound] = {}
    for key, value in overrides.items():
        if key not in _FILE_KEYS:
            raise ConfigError(f"command line: unknown key {key}")
        source = _FLAGS.get(key, "command line")
        layer[key] = _Bound(_decode(source, key, value, None), source)
    return layer


def _one_selector(source: str, keys: Mapping[str, object], message: str) -> None:
    chosen = [key for key in _SELECTORS if key in keys]
    if len(chosen) > 1:
        raise ConfigError(f"{source}: {message}")


def _reject_secret(source: str, table: Mapping[str, object]) -> None:
    if "api_key" in table or "OPENROUTER_API_KEY" in table:
        raise ConfigError(
            f"{source}: api_key is not a config key; export OPENROUTER_API_KEY "
            "or add it to .env"
        )


def _from_env(key: str, source: str, text: str) -> object:
    if key in _INTEGERS:
        return _env_integer(source, key, text)
    if key in _FLOATS:
        return _env_number(source, key, text)
    if key in _LISTS:
        return [part.strip() for part in text.split(",") if part.strip() != ""]
    if key == "providers":
        return _parse_provider_blob(source, text)
    return text


def _decode(source: str, key: str, value: object, base: Path | None) -> object:
    if key in _PATHS:
        return _decode_path(source, key, value, base)
    if key == "providers":
        return _decode_providers(source, value)
    if key in _LISTS:
        return _decode_list(source, key, value)
    if key in _INTEGERS or key in _FLOATS:
        return _decode_number(source, key, value)
    return _decode_text(source, key, value)


def _decode_list(source: str, key: str, value: object) -> tuple[str, ...]:
    if key in {"models", "judge"}:
        return _model_list(source, key, value, "must list at least one model")
    if key == "condition":
        return _name_list(source, key, value, "must list at least one condition")
    return _required(source, key, value)


def _required(source: str, key: str, value: object) -> tuple[str, ...]:
    if isinstance(value, list | tuple) and not value:
        return ()
    return _name_list(source, key, value, "must list parameter names")


def _decode_number(source: str, key: str, value: object) -> int | float:
    floor = _FLOORS.get(key)
    if floor is not None:
        return _at_least(source, key, value, floor)
    if key in {"score_min", "score_max"}:
        return _integer(source, key, value)
    if key in {"temperature", "judge_temperature"}:
        return _temperature(source, key, value)
    if key == "deadline_seconds":
        return _deadline(source, key, value)
    if key in {"budget", "confirm_above_usd"}:
        return _non_negative(source, key, value)
    return _number(source, key, value)


def _decode_text(source: str, key: str, value: object) -> str:
    if key == "base_url":
        return _base_url(source, key, value)
    if key == "start":
        return _loose_text(source, key, value)
    if key == "split":
        return _split(source, key, value)
    if key in _PLAIN:
        return _plain(source, key, value)
    raise ConfigError(f"{source}: unknown key {key}")


def _temperature(source: str, key: str, value: object) -> float:
    number = _number(source, key, value)
    if number < 0 or number > _MAX_TEMPERATURE:
        raise ConfigError(f"{source}: {key} must be from 0 to 2")
    return number


def _deadline(source: str, key: str, value: object) -> float:
    number = _number(source, key, value)
    if number <= 0:
        raise ConfigError(f"{source}: {key} must be greater than 0")
    return number


def _non_negative(source: str, key: str, value: object) -> float:
    number = _number(source, key, value)
    if number < 0 and key == "budget":
        raise ConfigError(f"{source}: budget must be zero or greater")
    if number < 0:
        raise ConfigError(f"{source}: {key} must be zero or greater")
    return number


def _base_url(source: str, key: str, value: object) -> str:
    text = _plain(source, key, value)
    if text.startswith(("https://", "http://")):
        return text
    raise ConfigError(f"{source}: {key} must start with http:// or https://")


def _loose_text(source: str, key: str, value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{source}: {key} must be a string")
    return value


def _split(source: str, key: str, value: object) -> str:
    text = _plain(source, key, value)
    if text not in {"study", "held-out"}:
        raise ConfigError(f"{source}: {key} must be study or held-out")
    return text


def _finalize(bounds: dict[str, _Bound]) -> Settings:
    score_min = _need_int(bounds, "score_min")
    score_max = _need_int(bounds, "score_max")
    if score_min >= score_max:
        raise ConfigError(
            f"{bounds['score_min'].source}: score_min must be less than score_max"
        )
    budget = (
        None
        if bounds["budget"].value is None
        else _as_float(bounds["budget"], "budget")
    )
    return Settings(
        models=_need_strings(bounds, "models"),
        judge=_need_strings(bounds, "judge"),
        judge_samples=_need_int(bounds, "judge_samples"),
        judge_temperature=_as_float(bounds["judge_temperature"], "judge_temperature"),
        temperature=_as_float(bounds["temperature"], "temperature"),
        max_cycles=_need_int(bounds, "max_cycles"),
        max_tokens=_need_int(bounds, "max_tokens"),
        stop_score=_as_float(bounds["stop_score"], "stop_score"),
        stop_consecutive=_need_int(bounds, "stop_consecutive"),
        score_min=score_min,
        score_max=score_max,
        judge_error_numerator=_need_int(bounds, "judge_error_numerator"),
        judge_error_denominator=_need_int(bounds, "judge_error_denominator"),
        deadline_seconds=_as_float(bounds["deadline_seconds"], "deadline_seconds"),
        base_url=_need_str(bounds, "base_url"),
        http_referer=_need_str(bounds, "http_referer"),
        title=_need_str(bounds, "title"),
        required_parameters=_need_strings(bounds, "required_parameters"),
        rubric_version=_need_str(bounds, "rubric_version"),
        repeats=_need_int(bounds, "repeats"),
        budget=budget,
        out=_need_path(bounds, "out"),
        confirm_above_usd=_as_float(bounds["confirm_above_usd"], "confirm_above_usd"),
        condition=_need_strings(bounds, "condition"),
        providers=_need_pins(bounds["providers"]),
        seeds=_optional_path(bounds, "seeds"),
        seed=_optional_str(bounds, "seed"),
        split=_optional_str(bounds, "split"),
        start=_optional_str(bounds, "start"),
    )


def _pin_payload(pin: ProviderPin) -> dict[str, object]:
    payload: dict[str, object] = {
        "allow_fallbacks": pin.allow_fallbacks,
        "order": list(pin.order),
    }
    if pin.data_collection is not None:
        payload["data_collection"] = pin.data_collection
    if pin.zdr is not None:
        payload["zdr"] = pin.zdr
    return payload


def _payload(settings: Settings) -> dict[str, object]:
    providers: dict[str, object] = {}
    for model, pin in settings.providers:
        providers[model] = _pin_payload(pin)
    return {
        "base_url": settings.base_url,
        "budget": settings.budget,
        "condition": list(settings.condition),
        "confirm_above_usd": settings.confirm_above_usd,
        "deadline_seconds": settings.deadline_seconds,
        "http_referer": settings.http_referer,
        "judge": list(settings.judge),
        "judge_error_denominator": settings.judge_error_denominator,
        "judge_error_numerator": settings.judge_error_numerator,
        "judge_samples": settings.judge_samples,
        "judge_temperature": settings.judge_temperature,
        "max_cycles": settings.max_cycles,
        "max_tokens": settings.max_tokens,
        "models": list(settings.models),
        "out": settings.out.as_posix(),
        "providers": providers,
        "repeats": settings.repeats,
        "required_parameters": list(settings.required_parameters),
        "rubric_version": settings.rubric_version,
        "score_max": settings.score_max,
        "score_min": settings.score_min,
        "seed": settings.seed,
        "seeds": None if settings.seeds is None else settings.seeds.as_posix(),
        "split": settings.split,
        "start": settings.start,
        "stop_consecutive": settings.stop_consecutive,
        "stop_score": settings.stop_score,
        "temperature": settings.temperature,
        "title": settings.title,
    }


def _need_int(bounds: dict[str, _Bound], key: str) -> int:
    value = bounds[key].value
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{bounds[key].source}: {key} must be an integer")
    return value


def _as_float(bound: _Bound, key: str) -> float:
    return _number(bound.source, key, bound.value)


def _need_str(bounds: dict[str, _Bound], key: str) -> str:
    value = bounds[key].value
    if not isinstance(value, str):
        raise ConfigError(f"{bounds[key].source}: {key} must be a string")
    return value


def _need_strings(bounds: dict[str, _Bound], key: str) -> tuple[str, ...]:
    value = bounds[key].value
    if not isinstance(value, tuple):
        raise ConfigError(f"{bounds[key].source}: {key} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{bounds[key].source}: {key} must be a list of strings")
        items.append(item)
    return tuple(items)


def _need_path(bounds: dict[str, _Bound], key: str) -> Path:
    value = bounds[key].value
    if not isinstance(value, Path):
        raise ConfigError(f"{bounds[key].source}: {key} must be a path")
    return value


def _need_pins(bound: _Bound) -> tuple[tuple[str, ProviderPin], ...]:
    value = bound.value
    if not isinstance(value, tuple):
        raise ConfigError(f"{bound.source}: providers must be a table")
    pins: list[tuple[str, ProviderPin]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != _PAIR_LENGTH:
            raise ConfigError(f"{bound.source}: providers must be a table")
        model, pin = item
        if not isinstance(model, str) or not isinstance(pin, ProviderPin):
            raise ConfigError(f"{bound.source}: providers must be a table")
        pins.append((model, pin))
    return tuple(pins)


def _optional_str(bounds: dict[str, _Bound], key: str) -> str | None:
    bound = bounds.get(key)
    if bound is None or bound.value is None:
        return None
    if not isinstance(bound.value, str):
        raise ConfigError(f"{bound.source}: {key} must be a string")
    return bound.value


def _optional_path(bounds: dict[str, _Bound], key: str) -> Path | None:
    bound = bounds.get(key)
    if bound is None or bound.value is None:
        return None
    if not isinstance(bound.value, Path):
        raise ConfigError(f"{bound.source}: {key} must be a path")
    return bound.value


def _decode_path(source: str, key: str, value: object, base: Path | None) -> Path:
    if isinstance(value, Path):
        return value
    text = _plain(source, key, value)
    candidate = Path(text)
    if base is not None and not candidate.is_absolute():
        return base.parent / candidate
    return candidate


def _decode_providers(
    source: str, value: object
) -> tuple[tuple[str, ProviderPin], ...]:
    if isinstance(value, tuple):
        return _pin_tuple(source, _pair_list(source, value))
    if isinstance(value, str):
        return _parse_provider_blob(source, value)
    return _providers_table(source, value)


def _pair_list(source: str, value: tuple[object, ...]) -> list[tuple[str, ProviderPin]]:
    pairs: list[tuple[str, ProviderPin]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != _PAIR_LENGTH:
            raise ConfigError(f"{source}: providers must be a table")
        model, pin = item
        if not isinstance(model, str) or not isinstance(pin, ProviderPin):
            raise ConfigError(f"{source}: providers must be a table")
        pairs.append((model, pin))
    return pairs


def _providers_table(source: str, value: object) -> tuple[tuple[str, ProviderPin], ...]:
    table = _mapping(value, source, "providers")
    pairs: list[tuple[str, ProviderPin]] = []
    for model, raw in table.items():
        model_id = _model_id(source, "providers", model)
        fields = _mapping(raw, source, model_id)
        for name in fields:
            if name not in {"order", "allow_fallbacks", "data_collection", "zdr"}:
                raise ConfigError(f"{source}: unknown key {name}")
        if "order" not in fields:
            raise ConfigError(f"{source}: {model_id} is missing order")
        order = _provider_order(source, fields["order"])
        allow = False
        if "allow_fallbacks" in fields:
            flag = fields["allow_fallbacks"]
            if not isinstance(flag, bool):
                raise ConfigError(f"{source}: allow_fallbacks must be a boolean")
            allow = flag
        pairs.append(
            (
                model_id,
                ProviderPin(
                    order, allow, _data_collection(source, fields), _zdr(source, fields)
                ),
            )
        )
    return _pin_tuple(source, pairs)


def _data_collection(
    source: str, fields: Mapping[str, object]
) -> DataCollection | None:
    if "data_collection" not in fields:
        return None
    value = fields["data_collection"]
    if value == "allow":
        return "allow"
    if value == "deny":
        return "deny"
    raise ConfigError(f"{source}: data_collection must be allow or deny")


def _zdr(source: str, fields: Mapping[str, object]) -> bool | None:
    if "zdr" not in fields:
        return None
    flag = fields["zdr"]
    if not isinstance(flag, bool):
        raise ConfigError(f"{source}: zdr must be a boolean")
    return flag


def _provider_order(source: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{source}: order must list at least one provider")
    return tuple(_plain(source, "order", item) for item in value)


def _parse_provider_blob(source: str, text: str) -> tuple[tuple[str, ProviderPin], ...]:
    parts = [part.strip() for part in text.split(";") if part.strip() != ""]
    if not parts:
        raise ConfigError(f"{source}: providers must list at least one model")
    return _pin_tuple(source, [_parse_provider_item(source, part) for part in parts])


def _parse_provider_item(source: str, text: str) -> tuple[str, ProviderPin]:
    model, separator, rest = text.partition("=")
    if separator == "" or model.strip() == "":
        raise ConfigError(f"{source}: providers must use model=provider")
    if "|" in rest:
        order_text, _, fallback_text = rest.rpartition("|")
        if fallback_text not in {"true", "false"}:
            raise ConfigError(f"{source}: allow_fallbacks must be true or false")
        allow = fallback_text == "true"
    else:
        order_text = rest
        allow = False
    order = tuple(part.strip() for part in order_text.split(",") if part.strip() != "")
    if not order:
        raise ConfigError(f"{source}: order must list at least one provider")
    return _model_id(source, "providers", model.strip()), ProviderPin(order, allow)


def _pin_tuple(
    source: str, pairs: list[tuple[str, ProviderPin]]
) -> tuple[tuple[str, ProviderPin], ...]:
    seen: set[str] = set()
    for model, _pin in pairs:
        if model in seen:
            raise ConfigError(f"{source}: providers lists {model} more than once")
        seen.add(model)
    return tuple(pairs)


def _model_list(source: str, key: str, value: object, empty: str) -> tuple[str, ...]:
    items = _name_list(source, key, value, empty)
    for item in items:
        if any(char.isspace() for char in item):
            raise ConfigError(f"{source}: {key} must be a model id without spaces")
    return items


def _name_list(source: str, key: str, value: object, empty: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values: list[object] = [value]
    elif isinstance(value, list | tuple):
        values = list(value)
    else:
        raise ConfigError(f"{source}: {key} must be a string or a list")
    if not values:
        raise ConfigError(f"{source}: {key} {empty}")
    return tuple(_plain(source, key, item) for item in values)


def _model_id(source: str, key: str, value: object) -> str:
    text = _plain(source, key, value)
    if any(char.isspace() for char in text):
        raise ConfigError(f"{source}: {key} must be a model id without spaces")
    return text


def _plain(source: str, key: str, value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ConfigError(f"{source}: {key} must be a non-empty string")
    return value.strip()


def _integer(source: str, key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{source}: {key} must be an integer")
    return value


def _at_least(source: str, key: str, value: object, minimum: int) -> int:
    number = _integer(source, key, value)
    if number < minimum:
        if minimum == 0:
            raise ConfigError(f"{source}: {key} must be zero or greater")
        raise ConfigError(f"{source}: {key} must be at least {minimum}")
    return number


def _number(source: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{source}: {key} must be a number")
    number = value if isinstance(value, float) else float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{source}: {key} must be a number")
    return number


def _env_integer(source: str, key: str, text: str) -> int:
    sign = 1
    body = text
    if text.startswith("-"):
        sign = -1
        body = text[1:]
    if body.isdigit() and body != "":
        return sign * int(body)
    raise ConfigError(f"{source}: {key} must be an integer")


def _env_number(source: str, key: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ConfigError(f"{source}: {key} must be a number") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{source}: {key} must be a number")
    return value


def _mapping(value: object, source: str, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: {label} must be a table")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{source}: {label} has a non-string key")
        mapped[key] = item
    return mapped
