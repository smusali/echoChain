from __future__ import annotations

import backoff
import httpx
from openai import OpenAI, OpenAIError

from echochain.config import OPENROUTER_API_BASE, OPENROUTER_API_KEY, OPENROUTER_HEADERS


def _client() -> OpenAI:
    http_client = httpx.Client(
        timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)
    )

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_BASE,
        default_headers=OPENROUTER_HEADERS,
        http_client=http_client,
    )


def _retryable(func):
    return backoff.on_exception(
        backoff.expo,
        (OpenAIError, httpx.TimeoutException, httpx.ConnectError),
        max_tries=3,
        max_time=30,
        jitter=None,
    )(func)


@_retryable
def chat(model: str, messages: list[dict[str, str]], temperature: float = 1.0) -> str:
    resp = _client().chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()
