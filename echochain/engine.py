from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from echochain.baseline import TextScorer, lexical_scorer, measure
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
from echochain.judge import (
    TARGET_PREVIOUS,
    TARGET_SEED,
    TARGET_TRANSFORM,
    JudgeCall,
    Judgement,
    JudgeSample,
    sampling_value,
    score,
    summarize,
)
from echochain.persistence import (
    ClaimVerdictRecord,
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
    scorer: TextScorer
    previous_answer: str | None = None
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


def cycle(  # noqa: PLR0913
    seed: str,
    recorder: Recorder,
    client: ChatClient,
    *,
    run_id: str,
    spec: RunSpec | None = None,
    scorer: TextScorer | None = None,
) -> TerminationReason:
    """Run one chain until a stop rule, a failure, or the budget.

    Args:
        seed: Original answer.
        recorder: Sink for each step.
        client: Chat client used for transformations and judging.
        run_id: Run UUID stored on every row.
        spec: Condition, schedule, budget, and stop rule. Defaults to a
            fixed-length alternating run. `start_step` and `opening_text`
            continue a chain that already has stored steps.
        scorer: Embedding and entailment scorer. None leaves those fields null.

    Returns:
        Why the run stopped.
    """
    plan = default_spec() if spec is None else spec
    state = _Run(
        plan=plan,
        client=client,
        spending=_Spending(plan.budget, plan.spent),
        seed=seed,
        recorder=recorder,
        run_id=run_id,
        current=seed if plan.opening_text is None else plan.opening_text,
        previous_answer=plan.previous_answer,
        scorer=lexical_scorer() if scorer is None else scorer,
    )
    if state.spending.blocked():
        return "budget_exceeded"
    for index in range(plan.start_step, plan.max_cycles + 1):
        reason = _turn(state, index)
        if reason is not None:
            return reason
    return "max_cycles"


@dataclass(frozen=True)
class _Judged:
    seed: Judgement | None = None
    previous: Judgement | None = None
    failed_seed: tuple[Completion, ...] = ()
    failed_previous: tuple[Completion, ...] = ()


def _turn(state: _Run, index: int) -> TerminationReason | None:
    plan = state.plan
    phase = phase_for(plan.condition.prompts, index)
    model = model_for(plan.condition.schedule, plan.models, index)
    prompt = render_transform(phase, state.current)
    kind = score_kind(phase)
    text = _Text(state.run_id, index, phase, kind, prompt, state.current)
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    sent = _call_seed(plan, index, 0, TARGET_TRANSFORM)
    try:
        reply = state.client.chat(
            model,
            messages,
            temperature=plan.condition.temperature,
            seed=sent,
            top_p=plan.top_p,
        )
    except GenerationError as exc:
        return _failed_generation(state, text, exc.attempts)
    step = _described(plan, text, reply)
    if state.spending.add(reply.usage.cost):
        _record(state, step, _Judged())
        return "budget_exceeded"
    try:
        seed_judgement = _panel(state, index, state.seed, reply.text, TARGET_SEED)
    except GenerationError as exc:
        failed = exc.attempts
        state.spending.add(merge_attempts(failed).usage.cost)
        state.recorder.generation_errors += 1
        _record(state, step, _Judged(failed_seed=failed))
        return "generation_error"
    previous, stop = _previous(state, step, seed_judgement)
    if stop is not None:
        return stop
    state.recorder.judge_errors += seed_judgement.errors
    if previous is not None:
        state.recorder.judge_errors += previous.errors
    _record(state, step, _Judged(seed=seed_judgement, previous=previous))
    state.failures, state.streak, stop = _stop(
        state.failures,
        state.streak,
        kind,
        seed_judgement.score.mean,
        plan,
    )
    state.current = reply.text
    if kind == "fidelity":
        state.previous_answer = reply.text
    if state.spending.blocked():
        return "budget_exceeded"
    return stop


def _failed_generation(
    state: _Run, text: _Text, attempts: tuple[Completion, ...]
) -> TerminationReason:
    reply = merge_attempts(attempts)
    state.spending.add(reply.usage.cost)
    state.recorder.generation_errors += 1
    _record(state, _described(state.plan, text, reply), _Judged())
    return "generation_error"


def _previous(
    state: _Run, step: _Step, seed_judgement: Judgement
) -> tuple[Judgement | None, TerminationReason | None]:
    if step.score_kind != "fidelity" or state.previous_answer is None:
        return None, None
    if state.spending.blocked():
        _record(state, step, _Judged(seed=seed_judgement))
        return None, "budget_exceeded"
    try:
        judged = _panel(
            state,
            step.step,
            state.previous_answer,
            step.reply.text,
            TARGET_PREVIOUS,
        )
    except GenerationError as exc:
        failed = exc.attempts
        state.spending.add(merge_attempts(failed).usage.cost)
        state.recorder.generation_errors += 1
        _record(state, step, _Judged(seed=seed_judgement, failed_previous=failed))
        return None, "generation_error"
    return judged, None


def _record(state: _Run, step: _Step, judged: _Judged) -> None:
    _log(state, step, judged)
    _notify(state, step, judged)


def _call_seed(plan: RunSpec, step: int, sample_index: int, target: int) -> int | None:
    if plan.sampling_seed is None:
        return None
    return sampling_value(plan.sampling_seed, step, sample_index, target)


def _panel(
    state: _Run,
    step: int,
    original: str,
    transformed: str,
    target: int,
) -> Judgement:
    samples: list[JudgeSample] = []
    plan = state.plan
    kind = score_kind(phase_for(plan.condition.prompts, step))
    claims = plan.claims if kind == "fidelity" else ()
    for model in plan.judge_models:
        part = score(
            original,
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
                claims=claims,
                step=step,
                sampling_seed=plan.sampling_seed,
                top_p=plan.top_p,
                target=target,
            ),
        )
        samples.extend(part.samples)
        if state.spending.blocked():
            break
    return summarize(samples, tuple(claim.id for claim in claims))


def _notify(state: _Run, step: _Step, judged: _Judged) -> None:
    on_step = state.plan.on_step
    if on_step is None:
        return
    reply = step.reply
    prompt_tokens = reply.usage.prompt_tokens
    completion_tokens = reply.usage.completion_tokens
    cost = reply.usage.cost
    for judgement in (judged.seed, judged.previous):
        if judgement is None:
            continue
        for sample in judgement.samples:
            prompt_tokens += sample.completion.usage.prompt_tokens
            completion_tokens += sample.completion.usage.completion_tokens
            cost += sample.completion.usage.cost
    for attempt in (*judged.failed_seed, *judged.failed_previous):
        prompt_tokens += attempt.usage.prompt_tokens
        completion_tokens += attempt.usage.completion_tokens
        cost += attempt.usage.cost
    on_step(
        StepNote(
            step=step.step,
            total=state.plan.max_cycles,
            phase=step.phase,
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


def _log(state: _Run, step: _Step, judged: _Judged) -> None:
    judge = _judge_record(
        step.score_kind,
        judged.seed,
        judged.previous,
        judged.failed_seed,
        judged.failed_previous,
    )
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
            "top_p": reply.top_p,
            "seed": reply.sampling_seed,
            "seed_honored": reply.seed_honored,
        },
        "latency_seconds": reply.latency_seconds,
        "prompt_tokens": reply.usage.prompt_tokens,
        "completion_tokens": reply.usage.completion_tokens,
        "cost": reply.usage.cost,
        "generation_id": reply.generation_id,
        "judge": judge,
        "baseline": measure(
            reply.text,
            state.seed,
            state.previous_answer,
            kind=step.score_kind,
            scorer=state.scorer,
        ),
        "timestamp": utc_now(),
    }
    state.recorder.log(row)


def _judge_record(
    kind: str,
    seed_judgement: Judgement | None,
    previous: Judgement | None,
    failed_seed: tuple[Completion, ...],
    failed_previous: tuple[Completion, ...],
) -> JudgeRecord:
    samples = _stored_samples(seed_judgement, failed_seed)
    prev_samples = _stored_samples(previous, failed_previous)
    errors = 0 if seed_judgement is None else seed_judgement.errors
    if previous is not None:
        errors += previous.errors
    fidelity = _kept(seed_judgement, kind == "fidelity", anchor=False)
    anchored = _kept(seed_judgement, kind == "fidelity", anchor=True)
    answerability = _kept(seed_judgement, kind == "answerability", anchor=False)
    answerability_anchored = _kept(seed_judgement, kind == "answerability", anchor=True)
    claims = _claim_cells(seed_judgement)
    prev_claims = _claim_cells(previous)
    previous_score = _kept(previous, True, anchor=False)
    previous_anchor = _kept(previous, True, anchor=True)
    return {
        "samples": samples,
        "prev_samples": prev_samples,
        "errors": errors,
        "fidelity_vs_seed": fidelity.mean,
        "fidelity_spread": fidelity.spread,
        "anchored": anchored.mean,
        "anchored_spread": anchored.spread,
        "fidelity_vs_prev": previous_score.mean,
        "anchored_vs_prev": previous_anchor.mean,
        "answerability": answerability.mean,
        "answerability_anchored": answerability_anchored.mean,
        "claims": claims.verdicts,
        "claim_supported": claims.supported,
        "claim_contradicted": claims.contradicted,
        "claim_total": claims.total,
        "claim_fidelity": claims.fidelity,
        "contradiction_rate": claims.rate,
        "prev_claims": prev_claims.verdicts,
        "prev_claim_fidelity": prev_claims.fidelity,
        "prev_contradiction_rate": prev_claims.rate,
    }


def _stored_samples(
    judgement: Judgement | None, failed: tuple[Completion, ...]
) -> list[JudgeSampleRecord]:
    if judgement is not None:
        return [_sample_record(sample) for sample in judgement.samples]
    return [_sample_record(_failed_sample(attempt)) for attempt in failed]


def _failed_sample(attempt: Completion) -> JudgeSample:
    return JudgeSample(
        score=None,
        anchored=None,
        verdicts=(),
        error="generation failed",
        completion=attempt,
    )


class _Pair(NamedTuple):
    mean: float | None
    spread: float | None


class _Claims(NamedTuple):
    verdicts: list[ClaimVerdictRecord]
    supported: int | None
    contradicted: int | None
    total: int | None
    fidelity: float | None
    rate: float | None


def _kept(judgement: Judgement | None, keep: bool, *, anchor: bool) -> _Pair:
    if judgement is None or not keep:
        return _Pair(None, None)
    summary = judgement.anchored if anchor else judgement.score
    return _Pair(summary.mean, summary.spread)


def _claim_cells(judgement: Judgement | None) -> _Claims:
    if judgement is None or judgement.claims is None:
        return _Claims([], None, None, None, None, None)
    claims = judgement.claims
    verdicts: list[ClaimVerdictRecord] = [
        {"id": verdict.id, "verdict": verdict.verdict} for verdict in claims.verdicts
    ]
    return _Claims(
        verdicts,
        claims.supported,
        claims.contradicted,
        claims.total,
        claims.fidelity,
        claims.contradiction_rate,
    )


def _sample_record(sample: JudgeSample) -> JudgeSampleRecord:
    completion = sample.completion
    return {
        "score": sample.score,
        "anchored": sample.anchored,
        "verdicts": [
            {"id": verdict.id, "verdict": verdict.verdict}
            for verdict in sample.verdicts
        ],
        "error": sample.error,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "cost": completion.usage.cost,
        "generation_id": completion.generation_id,
        "requested_model": completion.requested_model,
        "served_model": completion.served_model,
        "provider": completion.provider,
        "finish_reason": completion.finish_reason,
        "seed": completion.sampling_seed,
        "top_p": completion.top_p,
        "seed_honored": completion.seed_honored,
    }
