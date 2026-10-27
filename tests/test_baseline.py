from __future__ import annotations

from echochain.baseline import length_ratio, lexical_scorer, measure


def test_lexical_baseline_scores_a_fidelity_step() -> None:
    assert length_ratio("abcd", "ab") == 2
    record = measure(
        "Redis uses epoll",
        "Redis uses epoll for I/O",
        None,
        kind="fidelity",
        scorer=lexical_scorer(),
    )
    assert record["length_ratio"] is not None
    assert record["cosine_seed"] is None
    empty = measure(
        "answer", "seed", None, kind="answerability", scorer=lexical_scorer()
    )
    assert empty["length_ratio"] is None
