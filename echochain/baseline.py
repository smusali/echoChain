from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypedDict

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
_MIN_WORD = 3
_CLOSED = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "for",
        "with",
        "in",
        "on",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "use",
        "uses",
        "used",
        "using",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "into",
        "over",
        "under",
        "than",
        "then",
        "not",
        "no",
        "nor",
        "but",
        "if",
        "at",
        "via",
        "per",
        "each",
        "both",
        "such",
        "their",
        "they",
        "them",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "how",
        "can",
        "may",
        "keep",
        "keeps",
        "kept",
        "do",
        "does",
        "did",
        "done",
    ]
)
_COMPLEXITY = re.compile(r"(?i)\b[oθω]\s*\([^)]*\)")
_HYPHENATED = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*\b")
_WORD = re.compile(rf"\b[A-Za-z]{{{_MIN_WORD},}}\b")
_SURFACE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


class BaselineUnavailable(RuntimeError):
    pass


class BaselineRecord(TypedDict):
    cosine_seed: float | None
    cosine_prev: float | None
    entail_seed_forward: float | None
    entail_seed_backward: float | None
    entail_prev_forward: float | None
    entail_prev_backward: float | None
    length_ratio: float | None
    technical_survival: float | None
    generic_share: float | None


class BaselineModels(TypedDict):
    embedding_model: str | None
    embedding_revision: str | None
    nli_model: str | None
    nli_revision: str | None


class TextScorer(Protocol):
    def embed(self, text: str) -> tuple[float, ...] | None:
        """Embed one text.

        Args:
            text: Text to embed.

        Returns:
            A vector, or null when this scorer has no embedding model.
        """

    def entail(self, premise: str, hypothesis: str) -> float | None:
        """Return the entailment probability.

        Args:
            premise: The premise.
            hypothesis: The hypothesis.

        Returns:
            The entailment-class probability, or null when this scorer has
            no NLI model.
        """

    def models(self) -> BaselineModels:
        """Return the model ids this scorer loaded.

        Returns:
            Names and revisions, or nulls when no neural model was loaded.
        """


def empty_baseline() -> BaselineRecord:
    """Return a baseline object with every value unset.

    Returns:
        Nine null fields.
    """
    return {
        "cosine_seed": None,
        "cosine_prev": None,
        "entail_seed_forward": None,
        "entail_seed_backward": None,
        "entail_prev_forward": None,
        "entail_prev_backward": None,
        "length_ratio": None,
        "technical_survival": None,
        "generic_share": None,
    }


def empty_models() -> BaselineModels:
    """Return model ids for a run that did not load neural models.

    Returns:
        Four null fields.
    """
    return {
        "embedding_model": None,
        "embedding_revision": None,
        "nli_model": None,
        "nli_revision": None,
    }


def pinned_models() -> BaselineModels:
    """Return the pinned embedding and NLI model ids.

    Returns:
        The model names and revisions stored on a neural run.
    """
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "nli_model": NLI_MODEL,
        "nli_revision": NLI_REVISION,
    }


def length_ratio(answer: str, seed: str) -> float | None:
    """Return the answer length divided by the seed length.

    Args:
        answer: Answer text. Length is Unicode code points.
        seed: Seed text. An empty seed has no ratio.

    Returns:
        The ratio, or null when the seed is empty.
    """
    if seed == "":
        return None
    return len(answer) / len(seed)


def technical_tokens(text: str) -> frozenset[str]:
    """Return the technical tokens in text.

    Args:
        text: Seed or answer text.

    Returns:
        Casefolded complexity expressions, numbers, acronyms, hyphenated
        compounds, and other words of at least three letters that are not
        closed-class.
    """
    spans: list[tuple[int, int]] = []
    found: set[str] = set()
    _take(text, _COMPLEXITY, spans, found, _compact)
    _take(text, _HYPHENATED, spans, found, str.casefold)
    _take(text, _NUMBER, spans, found, str.casefold)
    _take(text, _ACRONYM, spans, found, str.casefold)
    for match in _WORD.finditer(text):
        if _overlaps(match.start(), match.end(), spans):
            continue
        token = match.group(0).casefold()
        if token in _CLOSED:
            continue
        found.add(token)
    return frozenset(found)


def technical_survival(answer: str, seed: str) -> float | None:
    """Return the share of seed technical tokens kept by the answer.

    Args:
        answer: Answer text.
        seed: Seed text.

    Returns:
        A fraction from 0 to 1, or null when the seed has no technical tokens.
    """
    needed = technical_tokens(seed)
    if not needed:
        return None
    return len(needed & technical_tokens(answer)) / len(needed)


def generic_share(text: str) -> float | None:
    """Return the share of tokens that are closed-class words.

    Args:
        text: Answer text.

    Returns:
        A fraction from 0 to 1, or null when the text has no word tokens.
    """
    tokens = tuple(match.group(0).casefold() for match in _SURFACE.finditer(text))
    if not tokens:
        return None
    closed = sum(1 for token in tokens if token in _CLOSED)
    return closed / len(tokens)


def cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return the cosine similarity of two vectors.

    Args:
        left: First vector.
        right: Second vector. It must have the same length.

    Returns:
        Similarity from -1 to 1, or null when a vector is empty or zero,
        or the lengths differ.
    """
    if not left or len(left) != len(right):
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for first, second in zip(left, right, strict=True):
        dot += first * second
        left_norm += first * first
        right_norm += second * second
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / math.sqrt(left_norm * right_norm)


def measure(
    answer: str,
    seed: str,
    previous: str | None,
    *,
    kind: str,
    scorer: TextScorer,
) -> BaselineRecord:
    """Score one step without a judge.

    Args:
        answer: Step output.
        seed: Original seed text.
        previous: Previous answer. None on the first answer.
        kind: `fidelity` or `answerability`. Question steps are all null.
        scorer: Embedding and entailment scorer.

    Returns:
        The baseline fields for one JSONL row.
    """
    if kind != "fidelity":
        return empty_baseline()
    record = empty_baseline()
    record["cosine_seed"] = _pair(scorer, answer, seed)
    record["entail_seed_forward"] = scorer.entail(seed, answer)
    record["entail_seed_backward"] = scorer.entail(answer, seed)
    record["length_ratio"] = length_ratio(answer, seed)
    record["technical_survival"] = technical_survival(answer, seed)
    record["generic_share"] = generic_share(answer)
    if previous is None:
        return record
    record["cosine_prev"] = _pair(scorer, answer, previous)
    record["entail_prev_forward"] = scorer.entail(previous, answer)
    record["entail_prev_backward"] = scorer.entail(answer, previous)
    return record


def lexical_scorer() -> TextScorer:
    """Return a scorer that leaves cosine and entailment unset.

    Returns:
        A scorer whose neural fields are null.
    """
    return _LexicalScorer()


def load_scorer() -> TextScorer:
    """Load the pinned embedding and NLI models.

    Returns:
        A scorer that fills cosine and entailment.

    Raises:
        BaselineUnavailable: The optional extra is not installed, or a model
            does not expose an entailment label.
    """
    try:
        from sentence_transformers import (  # noqa: PLC0415
            CrossEncoder,
            SentenceTransformer,
        )
    except ImportError as exc:
        raise BaselineUnavailable(
            "baseline models are not installed; install the baseline extra"
        ) from exc
    embedding = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
    nli = CrossEncoder(NLI_MODEL, revision=NLI_REVISION)
    return _TransformerScorer(embedding, nli)


class _LexicalScorer:
    def embed(self, text: str) -> tuple[float, ...] | None:
        return None

    def entail(self, premise: str, hypothesis: str) -> float | None:
        return None

    def models(self) -> BaselineModels:
        return empty_models()


class _TransformerScorer:
    def __init__(self, embedding: object, nli: object) -> None:
        self._embedding = embedding
        self._nli = nli
        self._entailment_index = _entailment_index(_label_map(nli))

    def embed(self, text: str) -> tuple[float, ...] | None:
        encode = getattr(self._embedding, "encode", None)
        if not callable(encode):
            raise BaselineUnavailable("embedding model has no encode")
        raw = encode([text], show_progress_bar=False, convert_to_numpy=True)
        rows = _matrix(raw)
        if not rows or not rows[0]:
            return None
        return tuple(rows[0])

    def entail(self, premise: str, hypothesis: str) -> float | None:
        predict = getattr(self._nli, "predict", None)
        if not callable(predict):
            raise BaselineUnavailable("nli model has no predict")
        raw = predict(
            [(premise, hypothesis)],
            show_progress_bar=False,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        return _probability(raw, self._entailment_index)

    def models(self) -> BaselineModels:
        return pinned_models()


def _pair(scorer: TextScorer, left: str, right: str) -> float | None:
    first = scorer.embed(left)
    second = scorer.embed(right)
    if first is None or second is None:
        return None
    return cosine(first, second)


def _compact(token: str) -> str:
    return "".join(token.split()).casefold()


def _take(
    text: str,
    pattern: re.Pattern[str],
    spans: list[tuple[int, int]],
    found: set[str],
    normalize: Callable[[str], str],
) -> None:
    for match in pattern.finditer(text):
        if _overlaps(match.start(), match.end(), spans):
            continue
        spans.append((match.start(), match.end()))
        found.add(normalize(match.group(0)))


def _overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in spans)


def _label_map(model: object) -> Mapping[object, object]:
    owners: tuple[object, ...] = (model, getattr(model, "model", None))
    for owner in owners:
        if owner is None:
            continue
        config = getattr(owner, "config", None)
        labels = getattr(config, "id2label", None) if config is not None else None
        if isinstance(labels, Mapping):
            return labels
    raise BaselineUnavailable("nli model has no labels")


def _entailment_index(labels: Mapping[object, object]) -> int:
    for key, name in labels.items():
        if not isinstance(name, str) or name.casefold() != "entailment":
            continue
        if isinstance(key, bool):
            continue
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.isdecimal():
            return int(key)
    raise BaselineUnavailable("nli model has no entailment label")


def _probability(raw: object, index: int) -> float:
    rows = _matrix(raw)
    if not rows:
        raise BaselineUnavailable("nli scores are empty")
    row = rows[0]
    if index < 0 or index >= len(row):
        raise BaselineUnavailable("entailment index is outside the scores")
    return row[index]


def _matrix(raw: object) -> list[list[float]]:
    values = raw
    if not isinstance(values, list | tuple):
        tolist = getattr(values, "tolist", None)
        if not callable(tolist):
            raise BaselineUnavailable("model scores are not numeric")
        values = tolist()
    if not isinstance(values, list | tuple):
        raise BaselineUnavailable("model scores are not numeric")
    if not values:
        return []
    first = values[0]
    if not isinstance(first, list | tuple):
        return [_floats(values)]
    rows: list[list[float]] = []
    for row in values:
        if not isinstance(row, list | tuple):
            raise BaselineUnavailable("model scores are not numeric")
        rows.append(_floats(row))
    return rows


def _floats(raw: list[object] | tuple[object, ...]) -> list[float]:
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise BaselineUnavailable("model scores are not numeric")
        values.append(float(item))
    return values
