from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import httpx2

from echochain.settings import Settings, redact, request_headers

if TYPE_CHECKING:
    from echochain.client import ModelRecord

_CATALOG_TIMEOUT_SECONDS = 10.0


def check_models(
    catalog: Mapping[str, ModelRecord],
    model_ids: Sequence[str],
    *,
    required: Sequence[str] | None = None,
) -> None:
    from echochain.client import REQUIRED_PARAMETERS, ModelRecord, ModelUnavailableError

    names = REQUIRED_PARAMETERS if required is None else required
    for model_id in model_ids:
        record = catalog.get(model_id)
        if not isinstance(record, ModelRecord):
            raise ModelUnavailableError(f"{model_id} is not in the OpenRouter catalog")
        missing = [name for name in names if name not in record.parameters]
        if missing:
            joined = ", ".join(missing)
            raise ModelUnavailableError(f"{model_id} does not support {joined}")


def parse_catalog(payload: Mapping[str, object]) -> dict[str, ModelRecord]:
    from echochain.client import ModelCatalogError, ModelRecord

    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise ModelCatalogError("model catalog is missing data")
    records: dict[str, ModelRecord] = {}
    for entry in raw_models:
        fields = _mapping(entry, "model catalog entry")
        model_id = fields.get("id")
        if not isinstance(model_id, str) or model_id == "":
            raise ModelCatalogError("model catalog entry is missing id")
        parameters = _parameter_names(fields.get("supported_parameters"), model_id)
        records[model_id] = ModelRecord(
            model_id=model_id,
            parameters=parameters,
            prompt_price=_price(fields.get("pricing"), "prompt"),
            completion_price=_price(fields.get("pricing"), "completion"),
            request_price=_price(fields.get("pricing"), "request"),
        )
    return records


def fetch_catalog(
    api_key: str,
    *,
    settings: Settings,
    client: httpx2.Client | None = None,
) -> dict[str, ModelRecord]:
    from echochain.client import ModelCatalogError

    owns = client is None
    http = httpx2.Client(timeout=_CATALOG_TIMEOUT_SECONDS) if client is None else client
    loaded: object
    try:
        try:
            response = http.get(
                f"{settings.base_url.rstrip('/')}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    **request_headers(settings),
                },
            )
            response.raise_for_status()
            loaded = response.json()
        except httpx2.HTTPError as exc:
            raise ModelCatalogError(redact(str(exc))) from exc
    finally:
        if owns:
            http.close()
    parsed = parse_catalog(_mapping(loaded, "model catalog"))
    return parsed


def _mapping(value: object, label: str) -> dict[str, object]:
    from echochain.client import ModelCatalogError

    if not isinstance(value, dict):
        raise ModelCatalogError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ModelCatalogError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped


def _price(value: object, side: str) -> float | None:
    if not isinstance(value, dict):
        return None
    raw = value.get(side)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def _parameter_names(value: object, model_id: str) -> frozenset[str]:
    from echochain.client import ModelCatalogError

    if not isinstance(value, list):
        raise ModelCatalogError(f"{model_id} is missing supported_parameters")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ModelCatalogError(f"{model_id} has a non-string parameter")
        names.append(item)
    return frozenset(names)
