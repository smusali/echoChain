from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Literal, ParamSpec, Protocol, TypedDict, TypeVar

import backoff
from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from echochain.config import (
    OPENROUTER_API_BASE,
    OPENROUTER_HEADERS,
    REQUIRED_PARAMETERS,
    load_api_key,
    redact,
)

REQUEST_TIMEOUT = 10.0
_MAX_TRIES = 3
_MAX_RETRY_SECONDS = 30
_P = ParamSpec("_P")
_R = TypeVar("_R")


class ChatMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class ChatReply:
    text: str
    requested_model: str
    served_model: str


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    parameters: frozenset[str]


class ModelCatalogError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
    ) -> ChatReply:
        """Request a completion.

        Args:
            model: Model ID to call.
            messages: Chat messages.
            temperature: Sampling temperature.

        Returns:
            The assistant reply and the model that served it.
        """


class OpenRouterClient:
    def __init__(self, api_key: str | None = None) -> None:
        """Store the key and build the SDK client on first use.

        Args:
            api_key: OpenRouter key. When omitted, the key is loaded on demand.
        """
        self._api_key = api_key
        self._sdk: OpenAI | None = None

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
    ) -> ChatReply:
        """Request a completion.

        Args:
            model: Model ID to call.
            messages: Chat messages.
            temperature: Sampling temperature.

        Returns:
            The assistant reply and the model that served it.
        """
        return _complete(self._openai(), model, messages, temperature)

    def _openai(self) -> OpenAI:
        if self._sdk is None:
            key = self._api_key if self._api_key else load_api_key()
            self._api_key = key
            self._sdk = OpenAI(
                api_key=key,
                base_url=OPENROUTER_API_BASE,
                default_headers=dict(OPENROUTER_HEADERS),
                timeout=REQUEST_TIMEOUT,
            )
        return self._sdk


def check_models(catalog: Mapping[str, ModelRecord], model_ids: Sequence[str]) -> None:
    """Reject models that are missing or lack required parameters.

    Args:
        catalog: Models keyed by ID.
        model_ids: Configured model IDs.

    Raises:
        ModelUnavailableError: An ID is missing or lacks a required parameter.
    """
    for model_id in model_ids:
        record = catalog.get(model_id)
        if record is None:
            message = f"{model_id} is not in the OpenRouter catalog"
            raise ModelUnavailableError(message)
        missing = [
            parameter
            for parameter in REQUIRED_PARAMETERS
            if parameter not in record.parameters
        ]
        if missing:
            joined = ", ".join(missing)
            message = f"{model_id} does not support {joined}"
            raise ModelUnavailableError(message)


def parse_catalog(payload: Mapping[str, object]) -> dict[str, ModelRecord]:
    """Parse an OpenRouter `/models` payload.

    Args:
        payload: JSON object returned by the models endpoint.

    Returns:
        Models keyed by ID.

    Raises:
        ModelCatalogError: The payload is not a model list.
    """
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
        records[model_id] = ModelRecord(model_id=model_id, parameters=parameters)
    return records


def fetch_catalog(api_key: str) -> dict[str, ModelRecord]:
    """Download the OpenRouter model catalog.

    Args:
        api_key: OpenRouter API key.

    Returns:
        Models keyed by ID.

    Raises:
        ModelCatalogError: The catalog request failed.
    """
    request = urllib.request.Request(
        f"{OPENROUTER_API_BASE}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            **OPENROUTER_HEADERS,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = _read_payload(response)
    except (OSError, UnicodeError) as exc:
        raise ModelCatalogError(redact(str(exc))) from exc
    return parse_catalog(payload)


def _retry(func: Callable[_P, _R]) -> Callable[_P, _R]:
    return backoff.on_exception(
        backoff.expo,
        OpenAIError,
        max_tries=_MAX_TRIES,
        max_time=_MAX_RETRY_SECONDS,
        jitter=None,
    )(func)


@_retry
def _complete(
    client: OpenAI,
    model: str,
    messages: Sequence[ChatMessage],
    temperature: float,
) -> ChatReply:
    response = client.chat.completions.create(
        model=model,
        messages=_to_openai(messages),
        temperature=temperature,
    )
    if not response.choices:
        raise OpenAIError("model returned no choices")
    content = response.choices[0].message.content
    if content is None or content.strip() == "":
        raise OpenAIError("model returned empty content")
    served_model = response.model if response.model else ""
    return ChatReply(
        text=content.strip(),
        requested_model=model,
        served_model=served_model,
    )


def _to_openai(messages: Sequence[ChatMessage]) -> list[ChatCompletionMessageParam]:
    converted: list[ChatCompletionMessageParam] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system":
            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": content,
            }
            converted.append(system_message)
            continue
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,
        }
        converted.append(user_message)
    return converted


def _read_payload(response: HTTPResponse) -> dict[str, object]:
    try:
        parsed: object = json.loads(response.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelCatalogError(redact(str(exc))) from exc
    return _mapping(parsed, "model catalog")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ModelCatalogError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ModelCatalogError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped


def _parameter_names(value: object, model_id: str) -> frozenset[str]:
    if not isinstance(value, list):
        message = f"{model_id} is missing supported_parameters"
        raise ModelCatalogError(message)
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            message = f"{model_id} has a non-string parameter"
            raise ModelCatalogError(message)
        names.append(item)
    return frozenset(names)
