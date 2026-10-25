from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

from jinja2 import Template

from echochain.config import (
    MAX_CYCLES,
    ROTATING_MODELS,
    STOP_CONSECUTIVE,
    STOP_SCORE,
    TRANSFORM_TEMPERATURE,
)
from echochain.judge import ScoreKind
from echochain.prompts import A_TO_Q, IDENTITY_TMPL, PARAPHRASE_TMPL, Q_TO_A

PromptKind = Literal["telephone", "identity", "paraphrase"]
FIXED_LENGTH = "fixed_length"
CONSECUTIVE_FIDELITY = "consecutive_fidelity"
CONTROL_NAMES = ("alternate", "same-model", "identity", "paraphrase")
_FIXED_PREFIX = "fixed:"
_SPLITS = frozenset({"study", "held-out"})
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
    split: str


@dataclass(frozen=True)
class Corpus:
    version: str
    path: Path
    sha256: str
    seeds: tuple[Seed, ...]


@dataclass(frozen=True)
class SeedRef:
    seed_id: str | None = None
    corpus_version: str | None = None
    corpus_sha256: str | None = None


@dataclass(frozen=True)
class RunSpec:
    condition: Condition
    models: tuple[str, ...]
    max_cycles: int
    stop_score: float
    stop_consecutive: int
    seed_id: str | None
    corpus_version: str | None
    corpus_sha256: str | None
    budget: float | None = None


def stop_rule_name(consecutive: int) -> str:
    """Return the stop rule recorded for a run.

    Args:
        consecutive: How many low fidelity scores end the run. Zero disables
            that rule.

    Returns:
        `consecutive_fidelity` when the streak rule is on, otherwise
        `fixed_length`.
    """
    if consecutive > 0:
        return CONSECUTIVE_FIDELITY
    return FIXED_LENGTH


def condition_table(
    models: Sequence[str] = ROTATING_MODELS,
) -> dict[str, Condition]:
    """Return the named conditions.

    Args:
        models: Models a rotating schedule draws from. The first model is the
            fixed-model control.

    Returns:
        Conditions keyed by name.

    Raises:
        DesignError: No models were given.
    """
    if not models:
        raise DesignError("no models configured")
    fixed = f"{_FIXED_PREFIX}{models[0]}"
    temperature = TRANSFORM_TEMPERATURE
    return {
        "alternate": Condition("alternate", "alternate", "telephone", temperature),
        "swap": Condition("swap", "swap", "telephone", temperature),
        "factorial": Condition("factorial", "factorial", "telephone", temperature),
        "latin": Condition("latin", "latin", "telephone", temperature),
        "same-model": Condition("same-model", fixed, "telephone", temperature),
        "identity": Condition("identity", fixed, "identity", 0.0),
        "paraphrase": Condition("paraphrase", fixed, "paraphrase", temperature),
    }


def resolve_conditions(names: Sequence[str] | None) -> tuple[Condition, ...]:
    """Resolve condition names.

    Args:
        names: Requested names. None selects `alternate`. `all` selects the
            treatment and the three controls.

    Returns:
        Conditions in the requested order.

    Raises:
        DesignError: A name is unknown.
    """
    table = condition_table()
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


def default_spec(
    condition: Condition | None = None,
    seed_ref: SeedRef | None = None,
    *,
    max_cycles: int = MAX_CYCLES,
    stop_consecutive: int = STOP_CONSECUTIVE,
    budget: float | None = None,
) -> RunSpec:
    """Build the spec for one run.

    Args:
        condition: Condition to run. Defaults to `alternate`.
        seed_ref: Corpus id and hash. None for an ad hoc seed.
        max_cycles: Planned step count.
        stop_consecutive: Low-fidelity streak that ends the run. Zero keeps
            the fixed length.
        budget: USD cap. None means no cap.

    Returns:
        The spec the engine runs.
    """
    selected = condition if condition is not None else condition_table()["alternate"]
    ref = SeedRef() if seed_ref is None else seed_ref
    return RunSpec(
        condition=selected,
        models=tuple(ROTATING_MODELS),
        max_cycles=max_cycles,
        stop_score=STOP_SCORE,
        stop_consecutive=stop_consecutive,
        seed_id=ref.seed_id,
        corpus_version=ref.corpus_version,
        corpus_sha256=ref.corpus_sha256,
        budget=budget,
    )


def model_for(schedule: str, models: Sequence[str], step: int) -> str:
    """Return the model assigned to a step.

    Args:
        schedule: `alternate`, `swap`, `factorial`, `latin`, or `fixed:<model>`.
        models: Models a rotating schedule draws from.
        step: Step number, starting at 1.

    Returns:
        The model id for that step.

    Raises:
        ValueError: The schedule or the step is unusable.
    """
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
    """Return the phase name for a step.

    Args:
        prompts: `telephone`, `identity`, or `paraphrase`.
        step: Step number, starting at 1.

    Returns:
        `a2q`, `q2a`, `identity`, or `paraphrase`.
    """
    if prompts == "identity":
        return "identity"
    if prompts == "paraphrase":
        return "paraphrase"
    if step % 2:
        return "a2q"
    return "q2a"


def score_kind(phase: str) -> ScoreKind:
    """Return the rubric for a phase.

    Args:
        phase: `a2q`, `q2a`, `identity`, or `paraphrase`.

    Returns:
        `answerability` for a question, otherwise `fidelity`.
    """
    if phase == "a2q":
        return "answerability"
    return "fidelity"


def render_transform(phase: str, current: str) -> str:
    """Render the transform prompt for one step.

    Args:
        phase: `a2q`, `q2a`, `identity`, or `paraphrase`.
        current: Text fed into the prompt.

    Returns:
        The rendered prompt.
    """
    if phase == "a2q":
        return _rendered(A_TO_Q, answer=current)
    if phase == "q2a":
        return _rendered(Q_TO_A, question=current)
    if phase == "identity":
        return _rendered(IDENTITY_TMPL, text=current)
    return _rendered(PARAPHRASE_TMPL, text=current)


def _rendered(template: Template, **values: str) -> str:
    rendered: object = template.render(**values)
    if not isinstance(rendered, str):
        raise TypeError("prompt render did not return text")
    return rendered


def load_corpus(path: Path | None = None) -> Corpus:
    """Load the versioned seed corpus.

    Args:
        path: Corpus file. When omitted, the repo or packaged file is used.

    Returns:
        Seeds, the corpus version, and the file hash.

    Raises:
        DesignError: The file is missing or invalid.
    """
    location = path if path is not None else _default_corpus_path()
    if not location.is_file():
        raise DesignError(f"seed corpus was not found: {location}")
    try:
        parsed: object = tomllib.loads(location.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DesignError("seed corpus is not valid TOML") from exc
    payload = _object(parsed, "seed corpus")
    version = payload.get("version")
    if not isinstance(version, str) or version == "":
        raise DesignError("seed corpus is missing version")
    raw_seeds = payload.get("seed")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise DesignError("seed corpus has no seeds")
    seeds = tuple(_seed(item) for item in raw_seeds)
    ids = [seed.id for seed in seeds]
    if len(ids) != len(set(ids)):
        raise DesignError("seed ids are not unique")
    digest = hashlib.sha256(location.read_bytes()).hexdigest()
    return Corpus(version=version, path=location, sha256=digest, seeds=seeds)


def select_seeds(
    corpus: Corpus,
    *,
    seed_id: str | None,
    split: str | None,
) -> tuple[Seed, ...]:
    """Select seeds by id or split.

    Args:
        corpus: Loaded corpus.
        seed_id: One seed id.
        split: `study` or `held-out`.

    Returns:
        The selected seeds, in file order.

    Raises:
        DesignError: Nothing matched.
    """
    if seed_id is not None:
        matches = tuple(seed for seed in corpus.seeds if seed.id == seed_id)
        if len(matches) != 1:
            raise DesignError(f"unknown seed: {seed_id}")
        return matches
    if split is None:
        raise DesignError("no seed selected")
    chosen = tuple(seed for seed in corpus.seeds if seed.split == split)
    if not chosen:
        raise DesignError(f"no seeds in split {split}")
    return chosen


def _default_corpus_path() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        path = candidate / "seeds" / "v1.toml"
        if path.is_file() and (candidate / "pyproject.toml").is_file():
            return path
    resource = files("echochain").joinpath("seeds", "v1.toml")
    if resource.is_file():
        located = Path(str(resource))
        if located.is_file():
            return located
    raise DesignError("seed corpus was not found")


def _seed(value: object) -> Seed:
    fields = _object(value, "seed")
    seed_id = fields.get("id")
    domain = fields.get("domain")
    text = fields.get("text")
    split = fields.get("split")
    if not isinstance(seed_id, str) or seed_id == "":
        raise DesignError("seed is missing id")
    if not isinstance(domain, str) or domain.strip() == "":
        raise DesignError(f"{seed_id} is missing domain")
    if not isinstance(text, str) or text.strip() == "":
        raise DesignError(f"{seed_id} is missing text")
    if not isinstance(split, str) or split not in _SPLITS:
        raise DesignError(f"{seed_id} has an unknown split")
    return Seed(id=seed_id, domain=domain, text=text, split=split)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DesignError(f"{label} is not an object")
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise DesignError(f"{label} has a non-string key")
        mapped[key] = item
    return mapped
