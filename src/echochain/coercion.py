from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DataCollection = Literal["allow", "deny"]
_PIN_WIDTH = 2


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderPin:
    order: tuple[str, ...]
    allow_fallbacks: bool
    data_collection: DataCollection | None = None
    zdr: bool | None = None


def _bound_int(source: str, key: str, value: object, meta: Mapping[str, object]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{source}: {key} must be an integer")
    minimum = meta.get("minimum")
    if isinstance(minimum, int | float) and value < minimum:
        if minimum == 0:
            raise ConfigError(f"{source}: {key} must be zero or greater")
        raise ConfigError(f"{source}: {key} must be at least {int(minimum)}")
    return value


def _bound_float(
    source: str, key: str, value: object, meta: Mapping[str, object]
) -> float:
    number = _number(source, key, value)
    minimum = meta.get("minimum")
    maximum = meta.get("maximum")
    if isinstance(minimum, int | float):
        if meta.get("exclusive_min") and number <= minimum:
            raise ConfigError(f"{source}: {key} must be greater than {minimum}")
        if not meta.get("exclusive_min") and number < minimum:
            raise ConfigError(f"{source}: {key} must be zero or greater")
    if isinstance(maximum, int | float) and number > maximum:
        raise ConfigError(f"{source}: {key} is above the allowed maximum")
    return number


def _decode_path(source: str, key: str, value: object, base: Path | None) -> Path:
    if isinstance(value, Path):
        return value
    text = _plain(source, key, value)
    candidate = Path(text)
    if base is not None and not candidate.is_absolute():
        return base.parent / candidate
    return candidate


def _decode_pins(source: str, value: object) -> tuple[tuple[str, ProviderPin], ...]:
    if isinstance(value, str):
        return _pin_blob(source, value)
    if isinstance(value, tuple) and (not value or isinstance(value[0], tuple)):
        return _pin_pairs(source, value)
    if isinstance(value, dict):
        return _pin_table(source, value)
    raise ConfigError(f"{source}: providers must be a table")


def _pin_table(
    source: str, table: dict[object, object]
) -> tuple[tuple[str, ProviderPin], ...]:
    pairs: list[tuple[str, ProviderPin]] = []
    for model, raw in table.items():
        if not isinstance(model, str) or not isinstance(raw, dict):
            raise ConfigError(f"{source}: providers must be a table")
        fields_present = {str(name) for name in raw}
        allowed = {"order", "allow_fallbacks", "data_collection", "zdr"}
        if not fields_present <= allowed or "order" not in raw:
            raise ConfigError(f"{source}: {model} provider pin is invalid")
        order = raw["order"]
        if not isinstance(order, list) or not order:
            raise ConfigError(f"{source}: order must list at least one provider")
        allow = raw.get("allow_fallbacks", False)
        if not isinstance(allow, bool):
            raise ConfigError(f"{source}: allow_fallbacks must be a boolean")
        collection = raw.get("data_collection")
        if collection not in {None, "allow", "deny"}:
            raise ConfigError(f"{source}: data_collection must be allow or deny")
        zdr = raw.get("zdr")
        if zdr is not None and not isinstance(zdr, bool):
            raise ConfigError(f"{source}: zdr must be a boolean")
        chosen: DataCollection | None = None
        if collection in {"allow", "deny"}:
            chosen = collection
        pairs.append(
            (
                _plain(source, "providers", model),
                ProviderPin(
                    tuple(_plain(source, "order", item) for item in order),
                    allow,
                    chosen,
                    zdr,
                ),
            )
        )
    return _unique_pins(source, pairs)


def _pin_blob(source: str, text: str) -> tuple[tuple[str, ProviderPin], ...]:
    parts = [part.strip() for part in text.split(";") if part.strip() != ""]
    if not parts:
        raise ConfigError(f"{source}: providers must list at least one model")
    pairs: list[tuple[str, ProviderPin]] = []
    for part in parts:
        model, separator, rest = part.partition("=")
        if separator == "" or model.strip() == "":
            raise ConfigError(f"{source}: providers must use model=provider")
        allow = False
        order_text = rest
        if "|" in rest:
            order_text, _, fallback = rest.rpartition("|")
            if fallback not in {"true", "false"}:
                raise ConfigError(f"{source}: allow_fallbacks must be true or false")
            allow = fallback == "true"
        order = tuple(item.strip() for item in order_text.split(",") if item.strip())
        if not order:
            raise ConfigError(f"{source}: order must list at least one provider")
        pairs.append(
            (
                _plain(source, "providers", model.strip()),
                ProviderPin(order, allow),
            )
        )
    return _unique_pins(source, pairs)


def _pin_pairs(
    source: str, value: tuple[object, ...]
) -> tuple[tuple[str, ProviderPin], ...]:
    pairs: list[tuple[str, ProviderPin]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != _PIN_WIDTH:
            raise ConfigError(f"{source}: providers must be a table")
        model, pin = item
        if not isinstance(model, str) or not isinstance(pin, ProviderPin):
            raise ConfigError(f"{source}: providers must be a table")
        pairs.append((model, pin))
    return _unique_pins(source, pairs)


def _unique_pins(
    source: str, pairs: list[tuple[str, ProviderPin]]
) -> tuple[tuple[str, ProviderPin], ...]:
    seen: set[str] = set()
    for model, _pin in pairs:
        if model in seen:
            raise ConfigError(f"{source}: providers lists {model} more than once")
        seen.add(model)
    return tuple(pairs)


def _export(key: str, value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if key == "providers" and isinstance(value, tuple):
        exported: dict[str, object] = {}
        for model, pin in value:
            if not isinstance(pin, ProviderPin):
                continue
            body: dict[str, object] = {
                "allow_fallbacks": pin.allow_fallbacks,
                "order": list(pin.order),
            }
            if pin.data_collection is not None:
                body["data_collection"] = pin.data_collection
            if pin.zdr is not None:
                body["zdr"] = pin.zdr
            exported[model] = body
        return exported
    if isinstance(value, tuple):
        return list(value)
    return value


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    raise ConfigError("expected a list of strings")


def _pins(value: object) -> tuple[tuple[str, ProviderPin], ...]:
    if isinstance(value, tuple):
        return _pin_pairs("settings", value)
    raise ConfigError("providers must be a table")


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("expected an integer")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError("expected a number")
    return float(value)


def _as_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    raise ConfigError("expected a path")


def _plain(source: str, key: str, value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ConfigError(f"{source}: {key} must be a non-empty string")
    return value.strip()


def _loose(source: str, key: str, value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{source}: {key} must be a string")
    return value


def _number(source: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{source}: {key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{source}: {key} must be a number")
    return number


def _env_int(source: str, key: str, text: str) -> int:
    sign = -1 if text.startswith("-") else 1
    body = text[1:] if text.startswith("-") else text
    if body.isdigit() and body != "":
        return sign * int(body)
    raise ConfigError(f"{source}: {key} must be an integer")


def _env_float(source: str, key: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ConfigError(f"{source}: {key} must be a number") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{source}: {key} must be a number")
    return value
