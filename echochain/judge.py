import json
import logging
import statistics

from echochain.client import chat
from echochain.config import JUDGE_MODEL
from echochain.prompts import JUDGE_SYS, JUDGE_USER_TMPL

logger = logging.getLogger(__name__)


def score(original: str, transformed: str, n: int = 1) -> float:
    msgs = [
        {"role": "system", "content": JUDGE_SYS},
        {
            "role": "user",
            "content": JUDGE_USER_TMPL.render(
                original=original, transformed=transformed
            ),
        },
    ]

    scores: list[int | float] = []
    for _ in range(n):
        reply = chat(JUDGE_MODEL, msgs, temperature=0)
        if not isinstance(reply, str) or not reply.strip():
            logger.warning("Judge returned empty response, using fallback score")
            scores.append(1)
            continue
        try:
            clean_reply = reply.strip()
            if clean_reply.startswith("```json") and clean_reply.endswith("```"):
                clean_reply = clean_reply[7:-3].strip()
            elif clean_reply.startswith("```") and clean_reply.endswith("```"):
                clean_reply = clean_reply[3:-3].strip()

            score_value = json.loads(clean_reply)["score"]
            if isinstance(score_value, bool) or not isinstance(score_value, int | float):
                raise TypeError("judge score must be a number")
            scores.append(score_value)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Judge parse error: %s, using fallback score", exc)
            scores.append(1)

    return statistics.mean(scores) if scores else 1.0
