from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from tests.conftest import GENERATOR, JUDGE, completion, identity_spec

from echochain.client import ChatMessage, Completion, GenerationError
from echochain.engine import cycle
from echochain.records import (
    BaselinePins,
    ErrorCounts,
    Manifest,
    Recorder,
    RunStatus,
    Schedule,
    StopName,
    StopRule,
    Temperatures,
    TerminationReason,
    make_run_dir,
)


class _Scripted:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        *,
        response_format: ResponseFormatJSONSchema | None = None,
        seed: int | None = None,
        top_p: float = 1.0,
    ) -> Completion:
        if self.mode == "fail":
            raise GenerationError((completion(""),))
        if response_format is None:
            return completion("transformed answer", model=model)
        if self.mode == "bad-judge":
            return completion("nope", model=model)
        score = 1 if self.mode == "low" else 8
        return completion(
            f'{{"rationale":"ok","score":{score},"anchored":4}}',
            model=model,
        )


def _drive(
    tmp_path: Path,
    mode: str,
    *,
    budget: float | None = None,
    stop_consecutive: int = 0,
) -> TerminationReason:
    allocated = make_run_dir("seed text", tmp_path)
    manifest = Manifest(
        status=RunStatus.running,
        run_id=allocated.run_id,
        slug=allocated.slug,
        version="0.1.0",
        python="3.11.0",
        os="test",
        seed_sha256="0" * 64,
        seed_text="seed text",
        seed_id=None,
        claims=[],
        condition="identity",
        schedule=Schedule(name=f"fixed:{GENERATOR}", models=[GENERATOR]),
        requested_models=[GENERATOR, JUDGE],
        served_models=[],
        providers=[],
        temperatures=Temperatures(transform=0, judge=0),
        top_p=1,
        sampling_seed=3,
        max_cycles=1,
        max_tokens=16,
        stop_rule=StopRule(name=StopName.fixed_length, threshold=2, consecutive=0),
        max_judge_error_share=0.1,
        judge_panel=[],
        started_at="2026-01-01T00:00:00Z",
        ended_at=None,
        prompt_tokens=0,
        completion_tokens=0,
        cost=0,
        errors=ErrorCounts(generation=0, judge=0),
        termination_reason=None,
        budget=None,
        deadline_seconds=60,
        error=None,
        batch_id=None,
        repeat_index=None,
        baseline=BaselinePins(),
    )
    spec = identity_spec(budget=budget, stop_consecutive=stop_consecutive)
    with Recorder(allocated.directory, manifest) as recorder:
        reason = cycle(
            "seed text",
            recorder,
            _Scripted(mode),
            run_id=allocated.run_id,
            spec=spec,
            scorer=None,
        )
        recorder.finish(reason)
    return reason


def test_stop_reasons(tmp_path: Path) -> None:
    assert _drive(tmp_path / "cycles", "ok") is TerminationReason.max_cycles
    assert (
        _drive(tmp_path / "budget", "ok", budget=0) is TerminationReason.budget_exceeded
    )
    assert _drive(tmp_path / "generation", "fail") is TerminationReason.generation_error
    assert _drive(tmp_path / "judge", "bad-judge") is TerminationReason.judge_error
    assert (
        _drive(tmp_path / "threshold", "low", stop_consecutive=1)
        is TerminationReason.threshold
    )
