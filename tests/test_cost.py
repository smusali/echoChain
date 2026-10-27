from __future__ import annotations

from dataclasses import replace

from tests.conftest import priced

from echochain.client import ModelRecord
from echochain.cost import Work, estimate
from echochain.design import resolve_conditions
from echochain.judge import Claim
from echochain.settings import default_settings


def test_two_step_bound_matches_the_hand_count() -> None:
    settings = replace(
        default_settings(),
        models=("vendor-a/model",),
        judge=("vendor-b/judge",),
        judge_samples=1,
        max_cycles=2,
        max_tokens=8,
    )
    condition = resolve_conditions(("alternate",), settings.models)[0]
    catalog = {
        "vendor-a/model": priced("vendor-a/model"),
        "vendor-b/judge": priced("vendor-b/judge"),
    }
    bound = estimate(
        (Work("alpha beta", condition, (Claim("c", "claim"),)),),
        settings,
        catalog,
    )
    assert bound.calls == 4
    assert bound.tokens == 3919
    assert bound.cost == 3951
    assert isinstance(catalog["vendor-a/model"], ModelRecord)
