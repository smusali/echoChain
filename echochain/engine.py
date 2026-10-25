from __future__ import annotations

import itertools
from dataclasses import dataclass

from echochain.client import (
    ChatClient,
    ChatMessage,
    Completion,
    GenerationError,
    merge_attempts,
)
from echochain.config import (
    JUDGE_SAMPLES,
    MAX_CYCLES,
    ROTATING_MODELS,
    STOP_SCORE,
    TRANSFORM_TEMPERATURE,
)
from echochain.judge import Judgement, JudgeSample, score
from echochain.persistence import (
    JudgeRecord,
    JudgeSampleRecord,
    Recorder,
    StepRecord,
    sha256_text,
    utc_now,
)
from echochain.prompts import A_TO_Q, Q_TO_A

_INITIAL_AVERAGE = 10.0
_ANSWER_TO_QUESTION = "a2q"
_QUESTION_TO_ANSWER = "q2a"


@dataclass
class _Spending:
    limit: float | None
    spent: float = 0.0

    def blocked(self) -> bool:
        return self.limit is not None and self.spent >= self.limit

    def add(self, cost: float) -> bool:
        self.spent += cost
        return self.blocked()


@dataclass(frozen=True)
class _Step:
    run_id: str
    step: int
    phase: str
    prompt: str
    source: str
    reply: Completion


def cycle(
    seed: str,
    recorder: Recorder,
    client: ChatClient,
    *,
    run_id: str,
    budget: float | None,
) -> str:
    """Run telephone cycles until a stop rule, a generation failure, or the budget.

    Args:
        seed: Original answer.
        recorder: Sink for each step.
        client: Chat client used for transformations and judging.
        run_id: Run UUID stored on every row.
        budget: USD cap. None means no cap.

    Returns:
        Why the run stopped: threshold, max_cycles, generation_error, or
        budget_exceeded.
    """
    spending = _Spending(budget)
    if spending.blocked():
        return "budget_exceeded"
    current = seed
    original = seed
    models = itertools.cycle(ROTATING_MODELS)
    average = _INITIAL_AVERAGE
    reason = "max_cycles"
    for index in range(1, MAX_CYCLES + 1):
        phase = _ANSWER_TO_QUESTION if index % 2 else _QUESTION_TO_ANSWER
        model = next(models)
        prompt = (
            A_TO_Q.render(answer=current)
            if phase == _ANSWER_TO_QUESTION
            else Q_TO_A.render(question=current)
        )
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
        try:
            reply = client.chat(model, messages, temperature=TRANSFORM_TEMPERATURE)
        except GenerationError as exc:
            failed = merge_attempts(exc.attempts)
            spending.add(failed.usage.cost)
            recorder.generation_errors += 1
            _log(
                recorder, _Step(run_id, index, phase, prompt, current, failed), None, ()
            )
            reason = "generation_error"
            break
        step = _Step(run_id, index, phase, prompt, current, reply)
        if spending.add(reply.usage.cost):
            _log(recorder, step, None, ())
            reason = "budget_exceeded"
            break
        try:
            judgement = score(
                original,
                reply.text,
                client,
                spending.add,
                n=JUDGE_SAMPLES,
            )
        except GenerationError as exc:
            spending.add(merge_attempts(exc.attempts).usage.cost)
            recorder.generation_errors += 1
            _log(recorder, step, None, exc.attempts)
            reason = "generation_error"
            break
        recorder.judge_errors += judgement.errors
        _log(recorder, step, judgement, ())
        current = reply.text
        if spending.blocked():
            reason = "budget_exceeded"
            break
        average = ((average * (index - 1)) + judgement.mean) / index
        if average <= STOP_SCORE:
            reason = "threshold"
            break
    return reason


def _log(
    recorder: Recorder,
    step: _Step,
    judgement: Judgement | None,
    failed_judge: tuple[Completion, ...],
) -> None:
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
    judge = _judge_record(samples, errors)
    reply = step.reply
    row: StepRecord = {
        "run_id": step.run_id,
        "step": step.step,
        "phase": step.phase,
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


def _judge_record(samples: list[JudgeSampleRecord], errors: int) -> JudgeRecord:
    return {
        "samples": samples,
        "errors": errors,
        "fidelity_vs_seed": None,
        "fidelity_vs_prev": None,
        "answerability": None,
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
