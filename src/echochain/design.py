from __future__ import annotations

import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

from echochain.judge import Claim
from echochain.prompts import (
    A_TO_Q,
    IDENTITY_TMPL,
    PARAPHRASE_TMPL,
    Q_TO_A,
    render_text,
)
from echochain.records import ScoreKind, StopName
from echochain.settings import ROTATING_MODELS, TRANSFORM_TEMPERATURE

PromptKind = Literal["telephone", "identity", "paraphrase"]
CONTROL_NAMES = ("alternate", "same-model", "identity", "paraphrase")
_FIXED_PREFIX = "fixed:"
_SCHEDULES = frozenset({"alternate", "swap", "factorial", "latin"})


class DesignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Condition:
    name: str
    schedule: str
    prompts: PromptKind
    temperature: float


@dataclass(frozen=True)
class Seed:
    id: str
    domain: str
    text: str
    claims: tuple[Claim, ...] = ()


@dataclass(frozen=True)
class StepNote:
    step: int
    total: int
    phase: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


@dataclass(frozen=True)
class RunSpec:
    condition: Condition
    models: tuple[str, ...]
    max_cycles: int
    stop_score: float
    stop_consecutive: int
    max_judge_error_share: float
    seed_id: str | None
    budget: float | None
    judge_models: tuple[str, ...]
    judge_samples: int
    judge_temperature: float
    top_p: float
    sampling_seed: int | None = None
    claims: tuple[Claim, ...] = ()
    on_step: Callable[[StepNote], None] | None = None
    start_step: int = 1
    opening_text: str | None = None
    spent: float = 0.0
    previous_answer: str | None = None


def stop_rule_name(consecutive: int) -> StopName:
    if consecutive > 0:
        return StopName.consecutive_fidelity
    return StopName.fixed_length


def condition_table(
    models: Sequence[str] = ROTATING_MODELS,
    *,
    temperature: float = TRANSFORM_TEMPERATURE,
) -> dict[str, Condition]:
    if not models:
        raise DesignError("no models configured")
    fixed = f"{_FIXED_PREFIX}{models[0]}"
    return {
        "alternate": Condition("alternate", "alternate", "telephone", temperature),
        "swap": Condition("swap", "swap", "telephone", temperature),
        "factorial": Condition("factorial", "factorial", "telephone", temperature),
        "latin": Condition("latin", "latin", "telephone", temperature),
        "same-model": Condition("same-model", fixed, "telephone", temperature),
        "identity": Condition("identity", fixed, "identity", 0.0),
        "paraphrase": Condition("paraphrase", fixed, "paraphrase", temperature),
    }


def resolve_conditions(
    names: Sequence[str] | None = None,
    models: Sequence[str] = ROTATING_MODELS,
    *,
    temperature: float = TRANSFORM_TEMPERATURE,
) -> tuple[Condition, ...]:
    table = condition_table(models, temperature=temperature)
    chosen = list(CONTROL_NAMES) if names is not None and "all" in names else None
    if chosen is None:
        chosen = ["alternate"] if not names else list(names)
    resolved: list[Condition] = []
    for name in chosen:
        condition = table.get(name)
        if condition is None:
            raise DesignError(f"unknown condition: {name}")
        resolved.append(condition)
    return tuple(resolved)


def model_for(schedule: str, models: Sequence[str], step: int) -> str:
    if schedule.startswith(_FIXED_PREFIX):
        chosen = schedule[len(_FIXED_PREFIX) :]
        if chosen == "":
            raise ValueError("fixed schedule is missing a model")
        return chosen
    if schedule not in _SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule}")
    if not models:
        raise ValueError("schedule has no models")
    if step < 1:
        raise ValueError("step must be at least 1")
    ordered = list(reversed(models)) if schedule == "swap" else list(models)
    index = step - 1
    count = len(ordered)
    if schedule in {"alternate", "swap"}:
        return ordered[index % count]
    pair = index // 2
    answering = index % 2 == 1
    if schedule == "factorial":
        slot = pair % count if answering else (pair // count) % count
        return ordered[slot]
    slot = (pair + 1) % count if answering else pair % count
    return ordered[slot]


def phase_for(prompts: PromptKind, step: int) -> str:
    if prompts == "identity":
        return "identity"
    if prompts == "paraphrase":
        return "paraphrase"
    if step % 2:
        return "a2q"
    return "q2a"


def planned_calls(prompts: PromptKind, steps: int, panel: int) -> int:
    previous = 0
    seen_answer = False
    for step in range(1, steps + 1):
        if score_kind(phase_for(prompts, step)) != "fidelity":
            continue
        if seen_answer:
            previous += 1
        seen_answer = True
    return steps + (steps + previous) * panel


def score_kind(phase: str) -> ScoreKind:
    if phase == "a2q":
        return ScoreKind.answerability
    return ScoreKind.fidelity


def render_transform(phase: str, current: str) -> str:
    if phase == "a2q":
        return render_text(A_TO_Q, answer=current)
    if phase == "q2a":
        return render_text(Q_TO_A, question=current)
    if phase == "identity":
        return render_text(IDENTITY_TMPL, text=current)
    return render_text(PARAPHRASE_TMPL, text=current)


def load_seeds(path: Path | None = None) -> tuple[Seed, ...]:
    if path is None:
        text = files("echochain").joinpath("seeds.toml").read_text(encoding="utf-8")
    else:
        if not path.is_file():
            raise DesignError(f"seed file was not found: {path}")
        text = path.read_text(encoding="utf-8")
    try:
        parsed: object = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DesignError("seed file is not valid TOML") from exc
    payload = _object(parsed, "seed file")
    raw_seeds = payload.get("seed")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise DesignError("seed file has no seeds")
    seeds = tuple(_seed(item) for item in raw_seeds)
    ids = [seed.id for seed in seeds]
    if len(ids) != len(set(ids)):
        raise DesignError("seed ids are not unique")
    return seeds


def select_seeds(seeds: Sequence[Seed], seed_id: str | None) -> tuple[Seed, ...]:
    if seed_id is None:
        return tuple(seeds)
    matches = tuple(seed for seed in seeds if seed.id == seed_id)
    if len(matches) != 1:
        raise DesignError(f"unknown seed: {seed_id}")
    return matches


def _seed(value: object) -> Seed:
    fields = _object(value, "seed")
    seed_id = fields.get("id")
    domain = fields.get("domain")
    text = fields.get("text")
    if not isinstance(seed_id, str) or seed_id == "":
        raise DesignError("seed is missing id")
    if not isinstance(domain, str) or domain.strip() == "":
        raise DesignError(f"{seed_id} is missing domain")
    if not isinstance(text, str) or text.strip() == "":
        raise DesignError(f"{seed_id} is missing text")
    return Seed(
        id=seed_id,
        domain=domain,
        text=text,
        claims=_claims(seed_id, fields.get("claim")),
    )


def _claims(seed_id: str, value: object) -> tuple[Claim, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DesignError(f"{seed_id} claims must be a list")
    claims: list[Claim] = []
    for item in value:
        fields = _object(item, f"{seed_id} claim")
        claim_id = fields.get("id")
        text = fields.get("text")
        if not isinstance(claim_id, str) or claim_id == "":
            raise DesignError(f"{seed_id} claim is missing id")
        if not isinstance(text, str) or text.strip() == "":
            raise DesignError(f"{seed_id} claim is missing text")
        claims.append(Claim(id=claim_id, text=text))
    ids = [claim.id for claim in claims]
    if len(ids) != len(set(ids)):
        raise DesignError(f"{seed_id} claim ids are not unique")
    return tuple(claims)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DesignError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise DesignError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped
