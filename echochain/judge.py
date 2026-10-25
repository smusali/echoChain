from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from echochain.client import ChatClient, ChatMessage, Completion
from echochain.config import JUDGE_MODEL, JUDGE_TEMPERATURE, redact
from echochain.prompts import JUDGE_SYS, JUDGE_USER_TMPL

logger = logging.getLogger(__name__)

_FALLBACK_SCORE = 1.0
_JSON_FENCE = "```json"
_FENCE = "```"


@dataclass(frozen=True)
class JudgeSample:
    score: float | None
    error: str | None
    completion: Completion


@dataclass(frozen=True)
class Judgement:
    mean: float
    samples: tuple[JudgeSample, ...]
    errors: int


def score(
    original: str,
    transformed: str,
    client: ChatClient,
    charge: Callable[[float], bool],
    n: int = 1,
) -> Judgement:
    """Score how much of the original answer remains.

    Args:
        original: Seed answer.
        transformed: Current text.
        client: Chat client used for the judge model.
        charge: Records the cost of one judge call. Returns True when the
            budget is exhausted.
        n: Number of judge samples to average.

    Returns:
        Mean score from 1 to 10, the samples, and the parse-error count.
        Parse failures count as 1.
    """
    messages: list[ChatMessage] = [
        {"role": "system", "content": JUDGE_SYS},
        {
            "role": "user",
            "content": JUDGE_USER_TMPL.render(
                original=original,
                transformed=transformed,
            ),
        },
    ]
    samples: list[JudgeSample] = []
    for _ in range(n):
        reply = client.chat(JUDGE_MODEL, messages, temperature=JUDGE_TEMPERATURE)
        samples.append(_sample(reply))
        if charge(reply.usage.cost):
            break
    values = [sample.score for sample in samples if sample.score is not None]
    errors = sum(1 for sample in samples if sample.error is not None)
    mean = _FALLBACK_SCORE if not values else statistics.mean(values)
    return Judgement(mean=mean, samples=tuple(samples), errors=errors)


def _sample(reply: Completion) -> JudgeSample:
    try:
        value = _parse_score(reply.text)
    except ValueError as exc:
        logger.warning(
            "Judge parse error: %s, using fallback score",
            redact(str(exc)),
        )
        return JudgeSample(score=_FALLBACK_SCORE, error=str(exc), completion=reply)
    return JudgeSample(score=value, error=None, completion=reply)


def _parse_score(reply: str) -> float:
    try:
        payload: object = json.loads(_strip_fence(reply.strip()))
    except json.JSONDecodeError as exc:
        raise ValueError(redact(str(exc))) from exc
    if not isinstance(payload, dict) or "score" not in payload:
        raise ValueError("judge score is missing")
    score_value = payload["score"]
    if isinstance(score_value, bool) or not isinstance(score_value, int | float):
        raise ValueError("judge score must be a number")
    return float(score_value)


def _strip_fence(text: str) -> str:
    if text.startswith(_JSON_FENCE) and text.endswith(_FENCE):
        return text[len(_JSON_FENCE) : -len(_FENCE)].strip()
    if text.startswith(_FENCE) and text.endswith(_FENCE):
        return text[len(_FENCE) : -len(_FENCE)].strip()
    return text
