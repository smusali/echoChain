from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from echochain.client import ModelRecord
from echochain.design import (
    Condition,
    model_for,
    phase_for,
    planned_calls,
    render_transform,
    score_kind,
)
from echochain.judge import Claim, claim_block
from echochain.prompts import (
    ANSWERABILITY_USER_TMPL,
    JUDGE_SYS,
    JUDGE_USER_TMPL,
    render_text,
)
from echochain.records import ScoreKind
from echochain.settings import Settings

_FRAMING_TOKENS = 16


@dataclass(frozen=True)
class Work:
    text: str
    condition: Condition
    claims: tuple[Claim, ...]


@dataclass(frozen=True)
class Estimate:
    runs: int
    calls: int
    tokens: int
    cost: float | None


def estimate(
    cells: Sequence[Work],
    settings: Settings,
    catalog: Mapping[str, ModelRecord] | None,
) -> Estimate:
    tokens = 0
    cost = 0.0
    priced = True
    panel = len(settings.judge) * settings.judge_samples
    calls = 0
    for cell in cells:
        step_tokens, step_cost = _bound_run(cell, settings, catalog)
        tokens += step_tokens
        calls += planned_calls(cell.condition.prompts, settings.max_cycles, panel)
        if step_cost is None:
            priced = False
        else:
            cost += step_cost
    return Estimate(len(cells), calls, tokens, cost if priced else None)


def render_estimate(bound: Estimate) -> str:
    cost = "unknown" if bound.cost is None else f"{bound.cost:.6f}"
    return (
        f"runs: {bound.runs}\n"
        f"calls: {bound.calls}\n"
        f"tokens_upper_bound: {bound.tokens}\n"
        f"cost_upper_bound_usd: {cost}\n"
    )


def _bound_run(
    cell: Work,
    settings: Settings,
    catalog: Mapping[str, ModelRecord] | None,
) -> tuple[int, float | None]:
    tokens = 0
    cost = 0.0
    known = True
    first = 2 if cell.condition.prompts == "telephone" else 1
    for step in range(1, settings.max_cycles + 1):
        phase = phase_for(cell.condition.prompts, step)
        model = _model(cell.condition, settings.models, step)
        prompt_tokens = _transform_tokens(phase, cell.text, step, settings.max_tokens)
        tokens += prompt_tokens + settings.max_tokens
        transform = _priced(_record(catalog, model), prompt_tokens, settings.max_tokens)
        if transform is None:
            known = False
        else:
            cost += transform
        kind = score_kind(phase)
        panels = 2 if kind == ScoreKind.fidelity and step > first else 1
        judge_prompt = _judge_tokens(kind, cell.text, settings.max_tokens, cell.claims)
        for judge in settings.judge:
            unit = _priced(_record(catalog, judge), judge_prompt, settings.max_tokens)
            for _sample in range(settings.judge_samples * panels):
                tokens += judge_prompt + settings.max_tokens
                if unit is None:
                    known = False
                else:
                    cost += unit
    if not known:
        return tokens, None
    return tokens, cost


def _model(condition: Condition, models: tuple[str, ...], step: int) -> str:
    return model_for(condition.schedule, models, step)


def _record(
    catalog: Mapping[str, ModelRecord] | None, model: str
) -> ModelRecord | None:
    if catalog is None:
        return None
    return catalog.get(model)


def _priced(
    record: ModelRecord | None, prompt_tokens: int, max_tokens: int
) -> float | None:
    if record is None or record.prompt_price is None or record.completion_price is None:
        return None
    request = 0.0 if record.request_price is None else record.request_price
    return (
        prompt_tokens * record.prompt_price
        + max_tokens * record.completion_price
        + request
    )


def _transform_tokens(phase: str, seed: str, step: int, max_tokens: int) -> int:
    inserted = len(seed) if step == 1 else max_tokens
    return len(render_transform(phase, "")) + inserted + _FRAMING_TOKENS


def _judge_tokens(
    kind: ScoreKind, seed: str, max_tokens: int, claims: tuple[Claim, ...]
) -> int:
    if kind == ScoreKind.answerability:
        rendered = render_text(ANSWERABILITY_USER_TMPL, question="", seed="")
    else:
        rendered = render_text(
            JUDGE_USER_TMPL,
            original="",
            transformed="",
            claims=claim_block(claims),
        )
    return len(JUDGE_SYS) + len(rendered) + len(seed) + max_tokens + _FRAMING_TOKENS
