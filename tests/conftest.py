from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx2
import pytest

from echochain.client import Completion, ModelRecord, Usage
from echochain.design import Condition, RunSpec
from echochain.records import Manifest, Row
from echochain.settings import Settings, resolve_settings

GENERATOR = "vendor-a/model"
JUDGE = "vendor-b/judge"
_PARAMETERS = frozenset({"response_format", "seed", "temperature", "top_p"})


def assert_valid_run(directory: Path) -> None:
    text = (directory / "manifest.json").read_text(encoding="utf-8")
    Manifest.model_validate_json(text)
    lines = (directory / "chain.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        Row.model_validate_json(line)


def completion(text: str, *, model: str = GENERATOR, cost: float = 0.0) -> Completion:
    return Completion(
        text=text,
        finish_reason="stop",
        requested_model=model,
        served_model=model,
        provider="test",
        generation_id="gen",
        usage=Usage(3, 3, cost),
        latency_seconds=0.01,
        temperature=0.0,
        max_tokens=16,
        sampling_seed=1,
        top_p=1.0,
        seed_honored=True,
    )


def priced(model_id: str) -> ModelRecord:
    return ModelRecord(
        model_id,
        _PARAMETERS,
        prompt_price=1.0,
        completion_price=2.0,
        request_price=0.0,
    )


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "models": [GENERATOR],
        "judge": [JUDGE],
        "judge_samples": 1,
        "max_cycles": 1,
        "max_tokens": 32,
        "out": tmp_path,
        "budget": 1,
        "sampling_seed": 11,
        "confirm_above_usd": 100,
    }
    values.update(overrides)
    return resolve_settings(overrides=values, environ={})


def catalog_body() -> dict[str, object]:
    data = []
    for model_id in (GENERATOR, JUDGE):
        data.append(
            {
                "id": model_id,
                "supported_parameters": sorted(_PARAMETERS),
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                    "request": "0",
                },
            }
        )
    return {"data": data}


def judge_text(user: str, *, score: int = 8) -> str:
    body: dict[str, object] = {"rationale": "kept", "score": score, "anchored": 4}
    if "<claims>" in user:
        block = user.split("<claims>", 1)[1].split("</claims>", 1)[0]
        ids = re.findall(r"(?m)^([a-z0-9-]+): ", block)
        body["claims"] = [{"id": item, "verdict": "supported"} for item in ids]
    return json.dumps(body)


def transport(
    *, stop_after: int | None = None, score: int = 8
) -> tuple[httpx2.MockTransport, Callable[[], int]]:
    seen = {"chat": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json=catalog_body())
        seen["chat"] += 1
        if stop_after is not None and seen["chat"] > stop_after:
            raise KeyboardInterrupt
        payload = json.loads(request.content.decode())
        messages = payload.get("messages")
        user = ""
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        user = content
        if payload.get("response_format") is None:
            text = "Redis uses single-threaded event loops with epoll"
        else:
            text = judge_text(user, score=score)
        model = payload.get("model")
        served = model if isinstance(model, str) else GENERATOR
        return httpx2.Response(
            200,
            json={
                "id": "gen",
                "object": "chat.completion",
                "created": 0,
                "model": served,
                "provider": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 3,
                    "total_tokens": 6,
                    "cost": 0.000001,
                },
            },
        )

    return httpx2.MockTransport(handler), lambda: seen["chat"]


def identity_spec(*, budget: float | None = None, stop_consecutive: int = 0) -> RunSpec:
    return RunSpec(
        condition=Condition("identity", f"fixed:{GENERATOR}", "identity", 0.0),
        models=(GENERATOR,),
        max_cycles=1,
        stop_score=2,
        stop_consecutive=stop_consecutive,
        max_judge_error_share=0.1,
        seed_id=None,
        budget=budget,
        judge_models=(JUDGE,),
        judge_samples=1,
        judge_temperature=0,
        top_p=1,
        sampling_seed=3,
    )


@pytest.fixture(autouse=True)
def _reset_loggers() -> Iterator[None]:
    yield
    logger = logging.getLogger("echochain")
    logger.handlers.clear()
    logger.propagate = True


@pytest.fixture
def seed_file(tmp_path: Path) -> Path:
    path = tmp_path / "seeds.toml"
    path.write_text(
        "\n".join(
            (
                "[[seed]]",
                'id = "custom"',
                'domain = "demo"',
                'text = "Widgets drive sprockets in the demo"',
                "[[seed.claim]]",
                'id = "widgets"',
                'text = "widgets"',
                "[[seed.claim]]",
                'id = "sprockets"',
                'text = "sprockets"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return path
