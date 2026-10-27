from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from echochain.judge import parse_reply

_CASES = (
    ("", "empty"),
    ("not json", "invalid_json"),
    (json.dumps({"rationale": "x", "score": "8", "anchored": 4}), "not_integer"),
    (json.dumps({"rationale": "x", "score": 11, "anchored": 4}), "out_of_range"),
    (json.dumps({"rationale": "x", "anchored": 4}), "missing_key"),
    (
        json.dumps(
            {
                "rationale": "x",
                "score": 8,
                "anchored": 4,
                "claims": [{"id": "other", "verdict": "supported"}],
            }
        ),
        "invalid_json",
    ),
)


def test_parser_table() -> None:
    for reply, error in _CASES:
        parsed = parse_reply(reply, claim_ids=("widgets",) if "claims" in reply else ())
        assert parsed.error == error
    kept = parse_reply(
        json.dumps(
            {
                "rationale": "x",
                "score": 8,
                "anchored": 4,
                "claims": [{"id": "widgets", "verdict": "supported"}],
            }
        ),
        claim_ids=("widgets",),
    )
    assert kept.error is None
    assert kept.score == 8


@given(st.text(max_size=80))
def test_parser_never_raises(reply: str) -> None:
    parse_reply(reply, claim_ids=("widgets",))
