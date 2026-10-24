from __future__ import annotations

import json
import logging
import statistics

from echochain.client import ChatClient, ChatMessage
from echochain.config import JUDGE_MODEL, redact
from echochain.prompts import JUDGE_SYS, JUDGE_USER_TMPL

logger = logging.getLogger(__name__)

_FALLBACK_SCORE = 1.0
_JUDGE_TEMPERATURE = 0.0
_JSON_FENCE = "```json"
_FENCE = "```"


def score(original: str, transformed: str, client: ChatClient, n: int = 1) -> float:
    """Score how much of the original answer remains.

    Args:
        original: Seed answer.
        transformed: Current text.
        client: Chat client used for the judge model.
        n: Number of judge samples to average.

    Returns:
        Mean score from 1 to 10. Parse failures count as 1.
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
    scores: list[float] = []
    for _ in range(n):
        reply = client.chat(JUDGE_MODEL, messages, temperature=_JUDGE_TEMPERATURE)
        try:
            scores.append(_parse_score(reply.text))
        except ValueError as exc:
            logger.warning(
                "Judge parse error: %s, using fallback score",
                redact(str(exc)),
            )
            scores.append(_FALLBACK_SCORE)
    if not scores:
        return _FALLBACK_SCORE
    return statistics.mean(scores)


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
