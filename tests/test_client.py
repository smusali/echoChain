from __future__ import annotations

import httpx2
import pytest
from tests.conftest import GENERATOR, JUDGE, transport

from echochain.client import (
    ModelUnavailableError,
    OpenRouterClient,
    check_models,
    fetch_catalog,
)
from echochain.settings import default_settings


def test_catalog_and_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gate, calls = transport()
    settings = default_settings()
    catalog = fetch_catalog("test-key", settings=settings, client=_client(gate))
    check_models(catalog, (GENERATOR, JUDGE))
    with pytest.raises(ModelUnavailableError):
        check_models(catalog, ("missing/model",))
    with OpenRouterClient("test-key", settings=settings, transport=gate) as client:
        reply = client.chat(GENERATOR, [{"role": "user", "content": "seed"}], 0.0)
    assert reply.text.startswith("Redis")
    assert calls() == 1


def _client(gate: httpx2.BaseTransport) -> httpx2.Client:
    return httpx2.Client(transport=gate)
