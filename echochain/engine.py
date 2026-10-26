from __future__ import annotations

import statistics
from dataclasses import dataclass

from echochain.client import (
    ChatClient,
    ChatMessage,
    Completion,
    GenerationError,
    merge_attempts,
)
from echochain.design import (
    RunSpec,
    StepNote,
    default_spec,
    model_for,
    phase_for,
    render_transform,
    score_kind,
)
from echochain.judge import JudgeCall, Judgement, JudgeSample, ScoreKind, score
from echochain.persistence import (
    JudgeRecord,
    JudgeSampleRecord,
    Recorder,
    StepRecord,
    TerminationReason,
    sha256_text,
    utc_now,
)


@dataclass
class _Spending:
    limit: float | None
    spent: float = 0.0

    def blocked(self) -> bool:
        return self.limit is not None and self.spent >= self.limit

    def add(self, cost: float) -> bool:
        self.spent += cost
        return self.blocked()


@dataclass
class _Run:
    plan: RunSpec
    client: ChatClient
    spending: _Spending
    seed: str
    recorder: Recorder
    run_id: str
    current: str
    failures: int = 0
    streak: int = 0


@dataclass(frozen=True)
class _Text:
    run_id: str
    index: int
    phase: str
    kind: str
    prompt: str
    source: str


@dataclass(frozen=True)
class _Step:
    run_id: str
    step: int
    phase: str
    condition: str
    schedule: str
    score_kind: str
    prompt: str
    source: str
    reply: Completion


def cycle(
    seed: str,
    recorder: Recorder,
    client: ChatClient,
    *,
    run_id: str,
    spec: RunSpec | None = None,
) -> TerminationReason:
    """Run one chain until a stop rule, a failure, or the budget.

    Args:
        seed: Original answer.
        recorder: Sink for each step.
        client: Chat client used for transformations and judging.
        run_id: Run UUID stored on every row.
        spec: Condition, schedule, budget, and stop rule. Defaults to a
            fixed-length alternating run.

    Returns:
        Why the run stopped.
    """
    plan = default_spec() if spec is None else spec
    state = _Run(
        plan=plan,
        client=client,
        spending=_Spending(plan.budget),
        seed=seed,
        recorder=recorder,
        run_id=run_id,
        current=seed,
    )
    if state.spending.blocked():
        return "budget_exceeded"
    for index in range(1, plan.max_cycles + 1):
        reason = _turn(state, index)
        if reason is not None:
            return reason
    return "max_cycles"


def _turn(state: _Run, index: int) -> TerminationReason | None:
    plan = state.plan
    phase = phase_for(plan.condition.prompts, index)
    model = model_for(plan.condition.schedule, plan.models, index)
    prompt = render_transform(phase, state.current)
    kind = score_kind(phase)
    text = _Text(state.run_id, index, phase, kind, prompt, state.current)
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    try:
        reply = state.client.chat(
            model, messages, temperature=plan.condition.temperature
        )
    except GenerationError as exc:
        reply = merge_attempts(exc.attempts)
        state.spending.add(reply.usage.cost)
        state.recorder.generation_errors += 1
        _log(state.recorder, _described(plan, text, reply), None, ())
        _notify(state, text, reply, None, ())
        return "generation_error"
    step = _described(plan, text, reply)
    if state.spending.add(reply.usage.cost):
        _log(state.recorder, step, None, ())
        _notify(state, text, reply, None, ())
        return "budget_exceeded"
    try:
        judgement = _judgement(state, reply.text, kind)
    except GenerationError as exc:
        failed = exc.attempts
        state.spending.add(merge_attempts(failed).usage.cost)
        state.recorder.generation_errors += 1
        _log(state.recorder, step, None, failed)
        _notify(state, text, reply, None, failed)
        return "generation_error"
    state.recorder.judge_errors += judgement.errors
    _log(state.recorder, step, judgement, ())
    _notify(state, text, reply, judgement, ())
    state.failures, state.streak, stop = _stop(
        state.failures, state.streak, kind, judgement.mean, plan
    )
    state.current = reply.text
    if state.spending.blocked():
        return "budget_exceeded"
    return stop


def _judgement(state: _Run, transformed: str, kind: ScoreKind) -> Judgement:
    samples: list[JudgeSample] = []
    errors = 0
    plan = state.plan
    for model in plan.judge_models:
        part = score(
            state.seed,
            transformed,
            state.client,
            state.spending.add,
            JudgeCall(
                kind=kind,
                model=model,
                samples=plan.judge_samples,
                temperature=plan.judge_temperature,
                score_min=plan.score_min,
                score_max=plan.score_max,
            ),
        )
        samples.extend(part.samples)
        errors += part.errors
        if state.spending.blocked():
            break
    values = [sample.score for sample in samples if sample.score is not None]
    mean = None if not values else statistics.mean(values)
    return Judgement(mean=mean, samples=tuple(samples), errors=errors)


def _notify(
    state: _Run,
    text: _Text,
    reply: Completion,
    judgement: Judgement | None,
    failed: tuple[Completion, ...],
) -> None:
    on_step = state.plan.on_step
    if on_step is None:
        return
    prompt_tokens = reply.usage.prompt_tokens
    completion_tokens = reply.usage.completion_tokens
    cost = reply.usage.cost
    if judgement is not None:
        for sample in judgement.samples:
            prompt_tokens += sample.completion.usage.prompt_tokens
            completion_tokens += sample.completion.usage.completion_tokens
            cost += sample.completion.usage.cost
    for attempt in failed:
        prompt_tokens += attempt.usage.prompt_tokens
        completion_tokens += attempt.usage.completion_tokens
        cost += attempt.usage.cost
    on_step(
        StepNote(
            step=text.index,
            total=state.plan.max_cycles,
            phase=text.phase,
            model=reply.requested_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )
    )


def _stop(
    failures: int,
    streak: int,
    kind: str,
    mean: float | None,
    plan: RunSpec,
) -> tuple[int, int, TerminationReason | None]:
    if mean is None:
        failures += 1
    elif kind == "fidelity":
        streak = _fidelity_streak(streak, mean, plan.stop_score)
    if (
        failures * plan.judge_error_denominator
        > plan.max_cycles * plan.judge_error_numerator
    ):
        return failures, streak, "judge_error"
    if plan.stop_consecutive > 0 and streak >= plan.stop_consecutive:
        return failures, streak, "threshold"
    return failures, streak, None


def _fidelity_streak(streak: int, mean: float, stop_score: float) -> int:
    if mean <= stop_score:
        return streak + 1
    return 0


def _described(plan: RunSpec, text: _Text, reply: Completion) -> _Step:
    return _Step(
        run_id=text.run_id,
        step=text.index,
        phase=text.phase,
        condition=plan.condition.name,
        schedule=plan.condition.schedule,
        score_kind=text.kind,
        prompt=text.prompt,
        source=text.source,
        reply=reply,
    )


def _log(
    recorder: Recorder,
    step: _Step,
    judgement: Judgement | None,
    failed_judge: tuple[Completion, ...],
) -> None:
    mean = None if judgement is None else judgement.mean
    if judgement is not None:
        samples = [_sample_record(sample) for sample in judgement.samples]
        errors = judgement.errors
    elif failed_judge:
        samples = [
            _sample_record(
                JudgeSample(score=None, error="generation failed", completion=attempt)
            )
            for attempt in failed_judge
        ]
        errors = 0
    else:
        samples = []
        errors = 0
    judge = _judge_record(samples, errors, step.score_kind, mean)
    reply = step.reply
    row: StepRecord = {
        "run_id": step.run_id,
        "step": step.step,
        "phase": step.phase,
        "condition": step.condition,
        "schedule": step.schedule,
        "score_kind": step.score_kind,
        "requested_model": reply.requested_model,
        "served_model": reply.served_model,
        "provider": reply.provider,
        "input_text": step.source,
        "prompt_sha256": sha256_text(step.prompt),
        "output_text": reply.text,
        "finish_reason": reply.finish_reason,
        "parameters": {
            "temperature": reply.temperature,
            "max_tokens": reply.max_tokens,
            "seed": None,
        },
        "latency_seconds": reply.latency_seconds,
        "prompt_tokens": reply.usage.prompt_tokens,
        "completion_tokens": reply.usage.completion_tokens,
        "cost": reply.usage.cost,
        "generation_id": reply.generation_id,
        "judge": judge,
        "timestamp": utc_now(),
    }
    recorder.log(row)


def _judge_record(
    samples: list[JudgeSampleRecord],
    errors: int,
    kind: str,
    mean: float | None,
) -> JudgeRecord:
    return {
        "samples": samples,
        "errors": errors,
        "fidelity_vs_seed": mean if kind == "fidelity" else None,
        "fidelity_vs_prev": None,
        "answerability": mean if kind == "answerability" else None,
    }


def _sample_record(sample: JudgeSample) -> JudgeSampleRecord:
    completion = sample.completion
    return {
        "score": sample.score,
        "error": sample.error,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "cost": completion.usage.cost,
        "generation_id": completion.generation_id,
        "requested_model": completion.requested_model,
        "served_model": completion.served_model,
        "provider": completion.provider,
        "finish_reason": completion.finish_reason,
    }
