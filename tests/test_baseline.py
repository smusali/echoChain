from __future__ import annotations

import pytest
from echochain.baseline import (
    BaselineModels,
    BaselineUnavailable,
    empty_baseline,
    empty_models,
    lexical_scorer,
    load_scorer,
    measure,
)

_SEED = "Redis uses epoll and AIMD with O(log n)"
_ANSWER = "Redis uses epoll"
_ENTAILMENT = 0.25


class _SameVector:
    def embed(self, text: str) -> tuple[float, ...] | None:
        return (3.0, 4.0)

    def entail(self, premise: str, hypothesis: str) -> float | None:
        return _ENTAILMENT

    def models(self) -> BaselineModels:
        return empty_models()


def test_length_and_survival_match_the_locked_pair() -> None:
    record = measure(_ANSWER, _SEED, None, kind="fidelity", scorer=lexical_scorer())
    assert record["length_ratio"] == pytest.approx(16 / 39)
    assert record["technical_survival"] == pytest.approx(0.5)
    assert record["cosine_seed"] is None
    assert record["entail_seed_forward"] is None
    assert record["cosine_prev"] is None


def test_question_steps_are_unset() -> None:
    record = measure(
        _ANSWER, _SEED, _ANSWER, kind="answerability", scorer=lexical_scorer()
    )
    assert record == empty_baseline()


def test_measure_is_repeatable() -> None:
    scorer = _SameVector()
    first = measure(_ANSWER, _SEED, None, kind="fidelity", scorer=scorer)
    second = measure(_ANSWER, _SEED, None, kind="fidelity", scorer=scorer)
    assert first == second
    assert first["cosine_seed"] == 1
    assert first["entail_seed_forward"] == _ENTAILMENT
    assert first["entail_seed_backward"] == _ENTAILMENT
    assert first["cosine_prev"] is None


def test_load_scorer_requires_the_extra() -> None:
    try:
        import sentence_transformers  # noqa: F401, PLC0415
    except ImportError:
        with pytest.raises(BaselineUnavailable, match="baseline extra"):
            load_scorer()
        return
    pytest.skip("baseline extra is installed")
