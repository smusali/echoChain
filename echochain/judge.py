from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)

from echochain.client import ChatClient, ChatMessage, Completion
from echochain.config import (
    ANCHORED_MAX,
    ANCHORED_MIN,
    JUDGE_MODEL,
    JUDGE_SAMPLES,
    JUDGE_TEMPERATURE,
    SCORE_MAX,
    SCORE_MIN,
    TOP_P,
)
from echochain.prompts import (
    ANSWERABILITY_USER_TMPL,
    JUDGE_SYS,
    JUDGE_USER_TMPL,
    render_text,
)

logger = logging.getLogger(__name__)

ScoreKind = Literal["fidelity", "answerability"]
Verdict = Literal["supported", "contradicted", "not_mentioned"]
TARGET_TRANSFORM = 0
TARGET_SEED = 1
TARGET_PREVIOUS = 2
_STEP_STRIDE = 1000
_SAMPLE_STRIDE = 10
_MIN_SPREAD = 2
_JSON_FENCE = "```json"
_FENCE = "```"
_PLAIN_KEYS = frozenset({"rationale", "score", "anchored"})
_CLAIM_KEYS = frozenset({"rationale", "score", "anchored", "claims"})


@dataclass(frozen=True)
class Claim:
    id: str
    text: str


def claim_block(claims: Sequence[Claim]) -> str | None:
    """Format frozen claims for a prompt.

    Args:
        claims: Claims in corpus order.

    Returns:
        One id and text per line, or None when `claims` is empty.
    """
    if not claims:
        return None
    return "\n".join(f"{claim.id}: {claim.text}" for claim in claims)


@dataclass(frozen=True)
class ClaimVerdict:
    id: str
    verdict: Verdict


@dataclass(frozen=True)
class ParsedJudge:
    score: int | None
    anchored: int | None
    verdicts: tuple[ClaimVerdict, ...]
    error: str | None


@dataclass(frozen=True)
class JudgeSample:
    score: int | None
    anchored: int | None
    verdicts: tuple[ClaimVerdict, ...]
    error: str | None
    completion: Completion


@dataclass(frozen=True)
class ScaleSummary:
    mean: float | None
    spread: float | None


@dataclass(frozen=True)
class ClaimSummary:
    supported: int
    contradicted: int
    total: int
    fidelity: float
    contradiction_rate: float
    verdicts: tuple[ClaimVerdict, ...]


@dataclass(frozen=True)
class Judgement:
    score: ScaleSummary
    anchored: ScaleSummary
    claims: ClaimSummary | None
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
    claims: tuple[Claim, ...] = ()
    step: int = 1
    sampling_seed: int | None = None
    top_p: float = TOP_P
    target: int = TARGET_SEED


def sampling_value(base: int, step: int, sample_index: int, target: int) -> int:
    """Return the seed sent on one call.

    Args:
        base: Run sampling seed.
        step: Step number, starting at 1.
        sample_index: Sample index, starting at 0.
        target: 0 for a transform, 1 for the seed comparison, 2 for the
            previous-answer comparison.

    Returns:
        The integer seed for the call.
    """
    return base + step * _STEP_STRIDE + sample_index * _SAMPLE_STRIDE + target


def judge_response_format(
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
    *,
    claims: bool = False,
) -> ResponseFormatJSONSchema:
    """Build the strict judge schema.

    Args:
        score_min: Lowest accepted comparison score.
        score_max: Highest accepted comparison score.
        claims: Whether the reply must include one verdict per claim.

    Returns:
        A JSON schema response format.
    """
    properties: dict[str, object] = {
        "rationale": {"type": "string"},
        "score": {"type": "integer", "minimum": score_min, "maximum": score_max},
        "anchored": {
            "type": "integer",
            "minimum": ANCHORED_MIN,
            "maximum": ANCHORED_MAX,
        },
    }
    required = ["rationale", "score", "anchored"]
    if claims:
        properties["claims"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "verdict"],
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "not_mentioned"],
                    },
                },
            },
        }
        required.append("claims")
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    json_schema: JSONSchema = {
        "name": "judge_score",
        "strict": True,
        "schema": schema,
    }
    return {"type": "json_schema", "json_schema": json_schema}


def score(
    original: str,
    transformed: str,
    client: ChatClient,
    charge: Callable[[float], bool],
    call: JudgeCall | None = None,
) -> Judgement:
    """Score one comparison.

    Args:
        original: Reference text. For answerability this is the seed.
        transformed: Current text. For answerability this is the question.
        client: Chat client used for the judge model.
        charge: Records the cost of one judge call. Returns True when the
            budget is exhausted.
        call: Judge model, sample count, claims, and sampling parameters.

    Returns:
        The aggregated scores, claim totals, samples, and parse-error count.
    """
    chosen = JudgeCall() if call is None else call
    content = _content(chosen, original, transformed)
    messages: list[ChatMessage] = [
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": content},
    ]
    response_format = judge_response_format(
        chosen.score_min,
        chosen.score_max,
        claims=bool(chosen.claims),
    )
    claim_ids = tuple(claim.id for claim in chosen.claims)
    taken: list[JudgeSample] = []
    for index in range(chosen.samples):
        sent = _sent_seed(chosen, index)
        reply = client.chat(
            chosen.model,
            messages,
            temperature=chosen.temperature,
            response_format=response_format,
            seed=sent,
            top_p=chosen.top_p,
        )
        taken.append(_sample(reply, chosen, claim_ids))
        if charge(reply.usage.cost):
            break
    return summarize(taken, claim_ids)


def summarize(samples: Sequence[JudgeSample], claim_ids: Sequence[str]) -> Judgement:
    """Aggregate judge samples.

    Args:
        samples: Samples from one comparison, including parse failures.
        claim_ids: Frozen claim ids, in corpus order. Empty when this
            comparison has no claims.

    Returns:
        Means, spreads, majority verdicts, and the parse-error count.
    """
    scores = [sample.score for sample in samples if sample.score is not None]
    anchored = [sample.anchored for sample in samples if sample.anchored is not None]
    errors = sum(1 for sample in samples if sample.error is not None)
    return Judgement(
        score=_scale(scores),
        anchored=_scale(anchored),
        claims=_claims_summary(samples, claim_ids),
        samples=tuple(samples),
        errors=errors,
    )


def parse_reply(
    reply: str,
    *,
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX,
    claim_ids: Sequence[str] = (),
) -> ParsedJudge:
    """Parse a judge reply.

    Args:
        reply: Raw judge text.
        score_min: Lowest accepted comparison score.
        score_max: Highest accepted comparison score.
        claim_ids: Frozen claim ids the reply must cover. Empty when the
            call did not ask for verdicts.

    Returns:
        The parsed scores and verdicts, or an error code with empty scores.
    """
    try:
        return _parse_reply(reply, score_min, score_max, tuple(claim_ids))
    except Exception:
        return ParsedJudge(None, None, (), "invalid_json")


def _content(call: JudgeCall, original: str, transformed: str) -> str:
    if call.kind == "answerability":
        return render_text(ANSWERABILITY_USER_TMPL, question=transformed, seed=original)
    return render_text(
        JUDGE_USER_TMPL,
        original=original,
        transformed=transformed,
        claims=claim_block(call.claims),
    )


def _sent_seed(call: JudgeCall, index: int) -> int | None:
    if call.sampling_seed is None:
        return None
    return sampling_value(call.sampling_seed, call.step, index, call.target)


def _sample(
    reply: Completion, call: JudgeCall, claim_ids: tuple[str, ...]
) -> JudgeSample:
    parsed = parse_reply(
        reply.text,
        score_min=call.score_min,
        score_max=call.score_max,
        claim_ids=claim_ids,
    )
    if parsed.error is not None:
        logger.warning("Judge parse error: %s", parsed.error)
    return JudgeSample(
        score=parsed.score,
        anchored=parsed.anchored,
        verdicts=parsed.verdicts,
        error=parsed.error,
        completion=reply,
    )


def _scale(values: list[int]) -> ScaleSummary:
    if not values:
        return ScaleSummary(None, None)
    mean = statistics.mean(values)
    spread = None if len(values) < _MIN_SPREAD else statistics.stdev(values)
    return ScaleSummary(mean, spread)


def _claims_summary(
    samples: Sequence[JudgeSample], claim_ids: Sequence[str]
) -> ClaimSummary | None:
    if not claim_ids:
        return None
    valid = [sample for sample in samples if sample.error is None and sample.verdicts]
    if not valid:
        return None
    verdicts: list[ClaimVerdict] = []
    supported = 0
    contradicted = 0
    for claim_id in claim_ids:
        chosen = _majority(valid, claim_id)
        verdicts.append(ClaimVerdict(claim_id, chosen))
        if chosen == "supported":
            supported += 1
        elif chosen == "contradicted":
            contradicted += 1
    total = len(claim_ids)
    return ClaimSummary(
        supported=supported,
        contradicted=contradicted,
        total=total,
        fidelity=supported / total,
        contradiction_rate=contradicted / total,
        verdicts=tuple(verdicts),
    )


def _majority(samples: Sequence[JudgeSample], claim_id: str) -> Verdict:
    counts: dict[Verdict, int] = {
        "supported": 0,
        "contradicted": 0,
        "not_mentioned": 0,
    }
    for sample in samples:
        for verdict in sample.verdicts:
            if verdict.id == claim_id:
                counts[verdict.verdict] += 1
    best = max(counts.values())
    winners = [name for name, count in counts.items() if count == best]
    if len(winners) != 1:
        return "not_mentioned"
    return winners[0]


def _parse_reply(
    reply: str,
    score_min: int,
    score_max: int,
    claim_ids: tuple[str, ...],
) -> ParsedJudge:
    text = reply.strip()
    if text == "":
        return ParsedJudge(None, None, (), "empty")
    payload = _load_payload(text)
    if payload is None:
        return ParsedJudge(None, None, (), "invalid_json")
    return _validate(payload, score_min, score_max, claim_ids)


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
    payload: object,
    score_min: int,
    score_max: int,
    claim_ids: tuple[str, ...],
) -> ParsedJudge:
    fields = _fields(payload, bool(claim_ids))
    if isinstance(fields, str):
        return ParsedJudge(None, None, (), fields)
    parsed = _scores(fields, score_min, score_max)
    if isinstance(parsed, str):
        return ParsedJudge(None, None, (), parsed)
    score_value, anchored_value = parsed
    verdicts, error = _verdicts(fields, claim_ids)
    if error is not None:
        return ParsedJudge(None, None, (), error)
    return ParsedJudge(score_value, anchored_value, verdicts, None)


def _scores(
    fields: dict[str, object], score_min: int, score_max: int
) -> tuple[int, int] | str:
    score_value = _integer(fields["score"], score_min, score_max)
    if isinstance(score_value, str):
        return score_value
    anchored_value = _integer(fields["anchored"], ANCHORED_MIN, ANCHORED_MAX)
    if isinstance(anchored_value, str):
        return anchored_value
    return score_value, anchored_value


def _integer(value: object, low: int, high: int) -> int | str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "not_integer"
    if value < low or value > high:
        return "out_of_range"
    return value


def _verdicts(
    fields: dict[str, object], claim_ids: tuple[str, ...]
) -> tuple[tuple[ClaimVerdict, ...], str | None]:
    if not claim_ids:
        return (), None
    found = _read_verdicts(fields.get("claims"))
    if isinstance(found, str):
        return (), found
    if set(found) != set(claim_ids):
        return (), "invalid_json"
    return tuple(
        ClaimVerdict(claim_id, found[claim_id]) for claim_id in claim_ids
    ), None


def _read_verdicts(raw: object) -> dict[str, Verdict] | str:
    if not isinstance(raw, list):
        return "missing_key"
    found: dict[str, Verdict] = {}
    for item in raw:
        parsed = _one_verdict(item, found)
        if isinstance(parsed, str):
            return parsed
        claim_id, verdict = parsed
        found[claim_id] = verdict
    return found


def _one_verdict(item: object, found: dict[str, Verdict]) -> tuple[str, Verdict] | str:
    if not isinstance(item, dict):
        return "invalid_json"
    claim_id = item.get("id")
    raw_verdict = item.get("verdict")
    if not isinstance(claim_id, str) or not isinstance(raw_verdict, str):
        return "invalid_json"
    parsed = _verdict(raw_verdict)
    if parsed is None or claim_id in found:
        return "invalid_json"
    return claim_id, parsed


def _verdict(value: str) -> Verdict | None:
    if value == "supported":
        return "supported"
    if value == "contradicted":
        return "contradicted"
    if value == "not_mentioned":
        return "not_mentioned"
    return None


def _fields(payload: object, claims: bool) -> dict[str, object] | str:
    if not isinstance(payload, dict):
        return "not_object"
    fields: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            return "not_object"
        fields[key] = value
    expected = _CLAIM_KEYS if claims else _PLAIN_KEYS
    if set(fields) != expected:
        if (
            "score" not in fields
            or "anchored" not in fields
            or "rationale" not in fields
        ):
            return "missing_key"
        return "invalid_json"
    if not isinstance(fields["rationale"], str):
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
