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
from openai import APIConnectionError, APIStatusError, Omit, OpenAI, omit
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)

from echochain.config import (
    REQUIRED_PARAMETERS,
    DataCollection,
    ProviderPin,
    Settings,
    default_settings,
    load_api_key,
    redact,
    request_headers,
)

_CATALOG_TIMEOUT_SECONDS = 10.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_MAX_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.5
_RETRY_CAP_SECONDS = 8.0
_RETRY_STATUSES = frozenset({408, 409, 429})
_RATE_LIMIT_STATUS = 429
_CREDIT_STATUS = 402
_RATE_LIMIT_WAITS = 5
_CREDIT_WAITS = 2
_RATE_LIMIT_WAIT_SECONDS = 20.0
_MAX_STATUS_WAIT_SECONDS = 180.0
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
    sampling_seed: int | None
    top_p: float
    seed_honored: bool | None


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
    prompt_price: float | None = None
    completion_price: float | None = None
    request_price: float | None = None


class _ProviderBody(TypedDict, total=False):
    require_parameters: bool
    order: list[str]
    allow_fallbacks: bool
    data_collection: DataCollection
    zdr: bool


class _ProviderExtra(TypedDict):
    provider: _ProviderBody


class ModelCatalogError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


class ChatClient(Protocol):
    def chat(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        *,
        response_format: ResponseFormatJSONSchema | None = None,
        seed: int | None = None,
        top_p: float = 1.0,
    ) -> Completion:
        """Request a completion.

        Args:
            model: Model ID to call.
            messages: Chat messages.
            temperature: Sampling temperature.
            response_format: Structured output schema. When set, the provider
                must support it.
            seed: Sampling seed. None lets the provider choose.
            top_p: Nucleus sampling cutoff.

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
        deadline: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Store the key and the optional transport.

        Args:
            api_key: OpenRouter key. When omitted, the key is loaded when the
                client opens.
            transport: HTTP transport. When omitted, a pooled transport is used.
            deadline: Wall-clock seconds allowed for one call, including retries.
                When omitted, the settings value is used.
            settings: Base URL, headers, token cap, and provider pins. When
                omitted, the module defaults are used.
        """
        chosen = default_settings() if settings is None else settings
        self._api_key = api_key
        self._transport = transport
        self._deadline_seconds = (
            chosen.deadline_seconds if deadline is None else deadline
        )
        self._base_url = chosen.base_url
        self._headers = request_headers(chosen)
        self._max_tokens = chosen.max_tokens
        self._pins = dict(chosen.providers)
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
                base_url=self._base_url,
                default_headers=dict(self._headers),
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

    def chat(  # noqa: PLR0913
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        *,
        response_format: ResponseFormatJSONSchema | None = None,
        seed: int | None = None,
        top_p: float = 1.0,
    ) -> Completion:
        """Request a completion.

        Args:
            model: Model ID to call.
            messages: Chat messages.
            temperature: Sampling temperature.
            response_format: Structured output schema. When set, the provider
                must support it.
            seed: Sampling seed. None lets the provider choose.
            top_p: Nucleus sampling cutoff.

        Returns:
            The completion. A rejected reply is tried once more.

        Raises:
            GenerationError: The reply was empty, truncated, or blocked twice.
            CallDeadlineError: The call exceeded its wall-clock deadline.
        """
        waits = 0
        call = _Outbound(
            model,
            messages,
            temperature,
            response_format,
            self._max_tokens,
            _provider_body(self._pins, model),
            top_p,
            seed,
        )
        while True:
            try:
                return self._exchange(call)
            except APIStatusError as exc:
                limit = _wait_limit(exc.status_code)
                if limit is None or waits >= limit:
                    raise
                waits += 1
                time.sleep(_pause_seconds(exc))

    def _exchange(self, call: _Outbound) -> Completion:
        first, usable = self._once(call)
        if usable:
            return first
        second, second_usable = self._once(call)
        if second_usable:
            return merge_attempts((first, second))
        raise GenerationError((first, second))

    def _once(self, call: _Outbound) -> tuple[Completion, bool]:
        gate = self._deadline_transport
        client = self._sdk
        if gate is None or client is None:
            raise RuntimeError("client is closed")
        deadline = time.monotonic() + self._deadline_seconds
        gate.deadline = deadline
        started = time.perf_counter()
        response = _send(client, deadline, call)
        latency = time.perf_counter() - started
        completion = _completion(response, call, latency)
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
        sampling_seed=last.sampling_seed,
        top_p=last.top_p,
        seed_honored=last.seed_honored,
    )


def check_models(
    catalog: Mapping[str, ModelRecord],
    model_ids: Sequence[str],
    *,
    required: Sequence[str] | None = None,
) -> None:
    """Reject models that are missing or lack required parameters.

    Args:
        catalog: Models keyed by ID.
        model_ids: Configured model IDs.
        required: Parameter names each model must advertise. When omitted, the
            module defaults are used.

    Raises:
        ModelUnavailableError: An ID is missing or lacks a required parameter.
    """
    names = REQUIRED_PARAMETERS if required is None else required
    for model_id in model_ids:
        record = catalog.get(model_id)
        if record is None:
            message = f"{model_id} is not in the OpenRouter catalog"
            raise ModelUnavailableError(message)
        missing = [
            parameter for parameter in names if parameter not in record.parameters
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
    settings: Settings | None = None,
) -> dict[str, ModelRecord]:
    """Download the OpenRouter model catalog.

    Args:
        api_key: OpenRouter API key.
        settings: Base URL and headers. When omitted, the module defaults are
            used.

    Returns:
        Models keyed by ID.

    Raises:
        ModelCatalogError: The catalog request failed.
    """
    chosen = default_settings() if settings is None else settings
    request = urllib.request.Request(
        f"{chosen.base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            **request_headers(chosen),
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


@dataclass(frozen=True)
class _Outbound:
    model: str
    messages: Sequence[ChatMessage]
    temperature: float
    response_format: ResponseFormatJSONSchema | None
    max_tokens: int
    extra: _ProviderExtra
    top_p: float
    seed: int | None


def _provider_body(pins: Mapping[str, ProviderPin], model: str) -> _ProviderExtra:
    pin = pins.get(model)
    allow = False if pin is None else pin.allow_fallbacks
    body: _ProviderBody = {
        "require_parameters": True,
        "allow_fallbacks": allow,
    }
    if pin is None:
        return {"provider": body}
    body["order"] = list(pin.order)
    if pin.data_collection is not None:
        body["data_collection"] = pin.data_collection
    if pin.zdr is not None:
        body["zdr"] = pin.zdr
    return {"provider": body}


def _wait_limit(status: int) -> int | None:
    if status == _RATE_LIMIT_STATUS:
        return _RATE_LIMIT_WAITS
    if status == _CREDIT_STATUS:
        return _CREDIT_WAITS
    return None


def _pause_seconds(exc: APIStatusError) -> float:
    parsed = _retry_after_seconds(exc.response.headers)
    if parsed is None:
        return _RATE_LIMIT_WAIT_SECONDS
    return min(parsed, _MAX_STATUS_WAIT_SECONDS)


def _send(client: OpenAI, deadline: float, call: _Outbound) -> ChatCompletion:
    schema: ResponseFormatJSONSchema | Omit = omit
    if call.response_format is not None:
        schema = call.response_format
    for attempt in range(_MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CallDeadlineError("call deadline exceeded")
        try:
            seed: int | Omit = omit
            if call.seed is not None:
                seed = call.seed
            return client.chat.completions.create(
                model=call.model,
                messages=_to_openai(call.messages),
                temperature=call.temperature,
                top_p=call.top_p,
                seed=seed,
                max_tokens=call.max_tokens,
                timeout=_timeout(remaining),
                response_format=schema,
                extra_body=call.extra,
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
    response: ChatCompletion, call: _Outbound, latency: float
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
    echoed = _echoed_seed(response)
    honored = None if call.seed is None or echoed is None else echoed == call.seed
    return Completion(
        text=text,
        finish_reason=finish_reason,
        requested_model=call.model,
        served_model=served,
        provider=_provider(response),
        generation_id=response.id,
        usage=_usage(response.usage),
        latency_seconds=latency,
        temperature=call.temperature,
        max_tokens=call.max_tokens,
        sampling_seed=call.seed,
        top_p=call.top_p,
        seed_honored=honored,
    )


def _echoed_seed(response: ChatCompletion) -> int | None:
    extra = response.model_extra
    if extra is None:
        return None
    raw = extra.get("seed")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


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


def _price(value: object, side: str) -> float | None:
    if not isinstance(value, dict):
        return None
    raw = value.get(side)
    if isinstance(raw, str):
        return _parse_price(raw)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return raw


def _parse_price(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


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
