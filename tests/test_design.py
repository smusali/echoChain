from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from echochain.design import (
    load_seeds,
    model_for,
    planned_calls,
    resolve_conditions,
    select_seeds,
)


def test_packaged_seed_and_selection(seed_file: Path) -> None:
    packaged = load_seeds()
    assert any(seed.id == "redis-epoll" for seed in packaged)
    custom = select_seeds(load_seeds(seed_file), "custom")
    assert custom[0].claims[0].id == "widgets"
    assert len(select_seeds(load_seeds(seed_file), None)) == 1


def test_planned_calls_for_two_telephone_steps() -> None:
    assert planned_calls("telephone", 2, 1) == 4


@given(
    st.integers(min_value=1, max_value=12),
    st.sampled_from(["alternate", "swap", "factorial", "latin"]),
)
def test_schedule_picks_a_configured_model(step: int, schedule: str) -> None:
    models = ["a/x", "b/y", "c/z"]
    assert model_for(schedule, models, step) in models


def test_identity_temperature_is_zero() -> None:
    condition = resolve_conditions(["identity"], ["a/x"])[0]
    assert condition.temperature == 0.0
    assert condition.prompts == "identity"
