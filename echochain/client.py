from __future__ import annotations

import json
import random
import time
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPResponse
from types import TracebackType
from typing import Literal, Protocol, TypedDict

import httpx2
from openai import APIConnectionError, APIStatusError, OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from echochain.config import (
    CALL_DEADLINE_SECONDS,
    MAX_TOKENS,
    OPENROUTER_API_BASE,
    OPENROUTER_HEADERS,
    REQUIRED_PARAMETERS,
    load_api_key,
    redact,
)

_CATALOG_TIMEOUT_SECONDS = 10.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_MAX_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.5
_RETRY_CAP_SECONDS = 8.0
_RETRY_STATUSES = frozenset({408, 409, 429})
_SERVER_STATUS = 500
_MILLISECONDS = 1000.0
_REJECTED_REASONS = frozenset({"length", "content_filter"})


class ChatMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cost: float


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    requested_model: str
    served_model: str
    provider: str | None
    generation_id: str
    usage: Usage
    latency_seconds: float
    temperature: float
    max_tokens: int
    sampling_seed: None


class GenerationError(Exception):
    def __init__(self, attempts: tuple[Completion, ...]) -> None:
        """Store the rejected replies.

        Args:
            attempts: Completed replies that were empty, truncated, or blocked.
        """
        self.attempts = attempts
        super().__init__("generation failed")


class CallDeadlineError(Exception):
    pass


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
        temperature: float,
    ) -> Completion:
        """Request a completion.

        Args:
            model: Model ID to call.
            messages: Chat messages.
            temperature: Sampling temperature.

        Returns:
            The completion, including usage and the model that served it.
        """


class _DeadlineStream(httpx2.SyncByteStream):
    def __init__(self, inner: httpx2.SyncByteStream, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    def __iter__(self) -> Iterator[bytes]:
        pending = iter(self._inner)
        while True:
            if time.monotonic() >= self._deadline:
                raise CallDeadlineError("call deadline exceeded")
            try:
                chunk = next(pending)
            except StopIteration:
                return
            yield chunk

    def close(self) -> None:
        self._inner.close()


class _DeadlineTransport(httpx2.BaseTransport):
    def __init__(self, inner: httpx2.BaseTransport) -> None:
        self._inner = inner
        self.deadline = float("inf")

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        if time.monotonic() >= self.deadline:
            raise CallDeadlineError("call deadline exceeded")
        response = self._inner.handle_request(request)
        stream = response.stream
        if (
            isinstance(stream, httpx2.SyncByteStream)
            and not response.is_stream_consumed
        ):
            response.stream = _DeadlineStream(stream, self.deadline)
        return response

    def close(self) -> None:
        self._inner.close()


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: httpx2.BaseTransport | None = None,
        deadline: float = CALL_DEADLINE_SECONDS,
    ) -> None:
        """Store the key and the optional transport.

        Args:
            api_key: OpenRouter key. When omitted, the key is loaded when the
                client opens.
            transport: HTTP transport. When omitted, a pooled transport is used.
            deadline: Wall-clock seconds allowed for one call, including retries.
        """
        self._api_key = api_key
        self._transport = transport
        self._deadline_seconds = deadline
        self._deadline_transport: _DeadlineTransport | None = None
        self._http: httpx2.Client | None = None
        self._sdk: OpenAI | None = None

    def __enter__(self) -> OpenRouterClient:
        inner = (
            self._transport if self._transport is not None else httpx2.HTTPTransport()
        )
        self._deadline_transport = _DeadlineTransport(inner)
        self._http = httpx2.Client(
            transport=self._deadline_transport,
            timeout=_timeout(self._deadline_seconds),
        )
        try:
            key = self._api_key if self._api_key else load_api_key()
            self._api_key = key
            self._sdk = OpenAI(
                api_key=key,
                base_url=OPENROUTER_API_BASE,
                default_headers=dict(OPENROUTER_HEADERS),
                max_retries=0,
                http_client=self._http,
                timeout=self._deadline_seconds,
            )
        except Exception:
            self._http.close()
            self._http = None
            self._deadline_transport = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._sdk is not None:
            self._sdk.close()
        if self._http is not None and not self._http.is_closed:
            self._http.close()
        self._sdk = None
        self._http = None
        self._deadline_transport = None

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
    ) -> Completion:
        """Request a completion.

        Args:
            model: Model ID to call.
            messages: Chat messages.
            temperature: Sampling temperature.

        Returns:
            The completion. A rejected reply is tried once more.

        Raises:
            GenerationError: The reply was empty, truncated, or blocked twice.
            CallDeadlineError: The call exceeded its wall-clock deadline.
        """
        first, usable = self._once(model, messages, temperature)
        if usable:
            return first
        second, second_usable = self._once(model, messages, temperature)
        if second_usable:
            return merge_attempts((first, second))
        raise GenerationError((first, second))

    def _once(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
    ) -> tuple[Completion, bool]:
        gate = self._deadline_transport
        client = self._sdk
        if gate is None or client is None:
            raise RuntimeError("client is closed")
        deadline = time.monotonic() + self._deadline_seconds
        gate.deadline = deadline
        started = time.perf_counter()
        response = _send(client, deadline, model, messages, temperature)
        latency = time.perf_counter() - started
        completion = _completion(response, model, temperature, latency)
        return completion, _usable(completion, _refused(response))


def merge_attempts(attempts: Sequence[Completion]) -> Completion:
    """Combine generation attempts into one completion.

    Args:
        attempts: Attempts in order. The last one supplies the text.

    Returns:
        A completion whose token, cost, and latency fields are sums.

    Raises:
        ValueError: No attempts were given.
    """
    if not attempts:
        raise ValueError("no generation attempts")
    last = attempts[-1]
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    latency = 0.0
    for attempt in attempts:
        prompt_tokens += attempt.usage.prompt_tokens
        completion_tokens += attempt.usage.completion_tokens
        cost += attempt.usage.cost
        latency += attempt.latency_seconds
    return Completion(
        text=last.text,
        finish_reason=last.finish_reason,
        requested_model=last.requested_model,
        served_model=last.served_model,
        provider=last.provider,
        generation_id=last.generation_id,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        ),
        latency_seconds=latency,
        temperature=last.temperature,
        max_tokens=last.max_tokens,
        sampling_seed=None,
    )


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
        with urllib.request.urlopen(
            request, timeout=_CATALOG_TIMEOUT_SECONDS
        ) as response:
            payload = _read_payload(response)
    except (OSError, UnicodeError) as exc:
        raise ModelCatalogError(redact(str(exc))) from exc
    return parse_catalog(payload)


def _send(
    client: OpenAI,
    deadline: float,
    model: str,
    messages: Sequence[ChatMessage],
    temperature: float,
) -> ChatCompletion:
    for attempt in range(_MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CallDeadlineError("call deadline exceeded")
        try:
            return client.chat.completions.create(
                model=model,
                messages=_to_openai(messages),
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                timeout=_timeout(remaining),
            )
        except CallDeadlineError:
            raise
        except APIStatusError as exc:
            if attempt + 1 >= _MAX_ATTEMPTS or not _retryable_status(exc.status_code):
                raise
            delay = _retry_delay(
                attempt,
                deadline,
                _retry_after_seconds(exc.response.headers),
            )
            if delay is None:
                raise CallDeadlineError("call deadline exceeded") from exc
            time.sleep(delay)
        except APIConnectionError as exc:
            if attempt + 1 >= _MAX_ATTEMPTS:
                raise
            delay = _retry_delay(attempt, deadline, None)
            if delay is None:
                raise CallDeadlineError("call deadline exceeded") from exc
            time.sleep(delay)
    raise CallDeadlineError("call deadline exceeded")


def _retryable_status(status: int) -> bool:
    return status in _RETRY_STATUSES or status >= _SERVER_STATUS


def _retry_after_seconds(headers: httpx2.Headers) -> float | None:
    for name, divisor in (("retry-after-ms", _MILLISECONDS), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value < 0:
            continue
        return value / divisor
    return None


def _retry_delay(
    attempt: int, deadline: float, retry_after: float | None
) -> float | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    if retry_after is not None:
        if retry_after > remaining:
            return None
        return retry_after
    cap = min(_RETRY_CAP_SECONDS, _RETRY_BASE_SECONDS * (2.0**attempt))
    return min(random.uniform(0, cap), remaining)


def _timeout(remaining: float) -> httpx2.Timeout:
    connect = min(_CONNECT_TIMEOUT_SECONDS, remaining)
    return httpx2.Timeout(remaining, connect=connect)


def _completion(
    response: ChatCompletion,
    model: str,
    temperature: float,
    latency: float,
) -> Completion:
    if not response.choices:
        text = ""
        finish_reason = ""
    else:
        message = response.choices[0].message
        content = message.content
        text = content.strip() if isinstance(content, str) else ""
        finish_reason = response.choices[0].finish_reason
    served = response.model if response.model else ""
    return Completion(
        text=text,
        finish_reason=finish_reason,
        requested_model=model,
        served_model=served,
        provider=_provider(response),
        generation_id=response.id,
        usage=_usage(response.usage),
        latency_seconds=latency,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        sampling_seed=None,
    )


def _usable(completion: Completion, refused: bool) -> bool:
    if refused or completion.text.strip() == "":
        return False
    return completion.finish_reason not in _REJECTED_REASONS


def _refused(response: ChatCompletion) -> bool:
    if not response.choices:
        return False
    refusal = response.choices[0].message.refusal
    return isinstance(refusal, str) and refusal.strip() != ""


def _provider(response: ChatCompletion) -> str | None:
    extra = response.model_extra
    if extra is None:
        return None
    raw = extra.get("provider")
    if isinstance(raw, str) and raw != "":
        return raw
    return None


def _usage(value: CompletionUsage | None) -> Usage:
    if value is None:
        return Usage(prompt_tokens=0, completion_tokens=0, cost=0.0)
    return Usage(
        prompt_tokens=value.prompt_tokens,
        completion_tokens=value.completion_tokens,
        cost=_cost(value),
    )


def _cost(value: CompletionUsage) -> float:
    extra = value.model_extra
    if extra is None:
        return 0.0
    raw = extra.get("cost")
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return 0.0
    return float(raw)


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
