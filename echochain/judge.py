from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)

from echochain.client import ChatClient, ChatMessage, Completion
from echochain.config import (
    JUDGE_MODEL,
    JUDGE_SAMPLES,
    JUDGE_TEMPERATURE,
    SCORE_MAX,
    SCORE_MIN,
)
from echochain.prompts import ANSWERABILITY_USER_TMPL, JUDGE_SYS, JUDGE_USER_TMPL

logger = logging.getLogger(__name__)

ScoreKind = Literal["fidelity", "answerability"]
_JSON_FENCE = "```json"
_FENCE = "```"
_KEYS = frozenset({"rationale", "score"})


def judge_response_format(
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
) -> ResponseFormatJSONSchema:
    """Build the strict judge schema.

    Args:
        score_min: Lowest accepted score.
        score_max: Highest accepted score.

    Returns:
        A JSON schema response format.
    """
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rationale", "score"],
        "properties": {
            "rationale": {"type": "string"},
            "score": {"type": "integer", "minimum": score_min, "maximum": score_max},
        },
    }
    json_schema: JSONSchema = {
        "name": "judge_score",
        "strict": True,
        "schema": schema,
    }
    return {"type": "json_schema", "json_schema": json_schema}


JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rationale", "score"],
    "properties": {
        "rationale": {"type": "string"},
        "score": {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX},
    },
}
JUDGE_RESPONSE_FORMAT = judge_response_format()


@dataclass(frozen=True)
class JudgeSample:
    score: int | None
    error: str | None
    completion: Completion


@dataclass(frozen=True)
class Judgement:
    mean: float | None
    samples: tuple[JudgeSample, ...]
    errors: int


@dataclass(frozen=True)
class JudgeCall:
    kind: ScoreKind = "fidelity"
    model: str = JUDGE_MODEL
    samples: int = JUDGE_SAMPLES
    temperature: float = JUDGE_TEMPERATURE
    score_min: int = SCORE_MIN
    score_max: int = SCORE_MAX


def score(
    original: str,
    transformed: str,
    client: ChatClient,
    charge: Callable[[float], bool],
    call: JudgeCall | None = None,
) -> Judgement:
    """Score one step.

    Args:
        original: Seed answer.
        transformed: Current text. For answerability this is the question.
        client: Chat client used for the judge model.
        charge: Records the cost of one judge call. Returns True when the
            budget is exhausted.
        call: Judge model, sample count, temperature, and score bounds.

    Returns:
        The mean of the valid scores, or None when every sample failed, plus
        the samples and the parse-error count.
    """
    chosen = JudgeCall() if call is None else call
    if chosen.kind == "answerability":
        content = ANSWERABILITY_USER_TMPL.render(question=transformed, seed=original)
    else:
        content = JUDGE_USER_TMPL.render(original=original, transformed=transformed)
    messages: list[ChatMessage] = [
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": content},
    ]
    response_format = judge_response_format(chosen.score_min, chosen.score_max)
    taken: list[JudgeSample] = []
    for _ in range(chosen.samples):
        reply = client.chat(
            chosen.model,
            messages,
            temperature=chosen.temperature,
            response_format=response_format,
        )
        taken.append(_sample(reply, chosen.score_min, chosen.score_max))
        if charge(reply.usage.cost):
            break
    values = [sample.score for sample in taken if sample.score is not None]
    errors = sum(1 for sample in taken if sample.error is not None)
    mean = None if not values else statistics.mean(values)
    return Judgement(mean=mean, samples=tuple(taken), errors=errors)


def parse_reply(
    reply: str,
    *,
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
) -> tuple[int | None, str | None]:
    """Parse a judge reply.

    Args:
        reply: Raw judge text.
        score_min: Lowest accepted score.
        score_max: Highest accepted score.

    Returns:
        The score and None, or None and a stable error code.
    """
    try:
        return _parse_reply(reply, score_min, score_max)
    except Exception:
        return None, "invalid_json"


def _sample(reply: Completion, score_min: int, score_max: int) -> JudgeSample:
    value, error = parse_reply(reply.text, score_min=score_min, score_max=score_max)
    if error is not None:
        logger.warning("Judge parse error: %s", error)
    return JudgeSample(score=value, error=error, completion=reply)


def _parse_reply(
    reply: str, score_min: int, score_max: int
) -> tuple[int | None, str | None]:
    text = reply.strip()
    if text == "":
        return None, "empty"
    payload = _load_payload(text)
    if payload is None:
        return None, "invalid_json"
    return _validate(payload, score_min, score_max)


def _load_payload(text: str) -> object | None:
    for candidate in _candidates(text):
        if candidate == "":
            continue
        try:
            loaded: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return loaded
    return None


def _candidates(text: str) -> tuple[str, ...]:
    stripped = _strip_fence(text)
    extracted = _first_object(stripped)
    values: list[str] = [text]
    if stripped != text:
        values.append(stripped)
    if extracted is not None and extracted not in values:
        values.append(extracted)
    return tuple(values)


def _validate(
    payload: object, score_min: int, score_max: int
) -> tuple[int | None, str | None]:
    fields = _fields(payload)
    if isinstance(fields, str):
        return None, fields
    score_value = fields["score"]
    if isinstance(score_value, bool) or not isinstance(score_value, int):
        return None, "not_integer"
    if score_value < score_min or score_value > score_max:
        return None, "out_of_range"
    return score_value, None


def _fields(payload: object) -> dict[str, object] | str:
    if not isinstance(payload, dict):
        return "not_object"
    fields: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            return "not_object"
        fields[key] = value
    if "score" in fields:
        score_value = fields["score"]
        if isinstance(score_value, bool) or not isinstance(score_value, int):
            return "not_integer"
    if "rationale" not in fields or "score" not in fields:
        return "missing_key"
    if set(fields) != _KEYS or not isinstance(fields["rationale"], str):
        return "invalid_json"
    return fields


def _strip_fence(text: str) -> str:
    if text.startswith(_JSON_FENCE) and text.endswith(_FENCE):
        return text[len(_JSON_FENCE) : -len(_FENCE)].strip()
    if text.startswith(_FENCE) and text.endswith(_FENCE):
        return text[len(_FENCE) : -len(_FENCE)].strip()
    return text


def _first_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
