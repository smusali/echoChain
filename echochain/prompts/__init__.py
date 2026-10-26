from __future__ import annotations

import hashlib
import re
from importlib.resources import files

from jinja2 import Environment, Template

_FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_FILES = (
    ("a2q", "a2q.j2"),
    ("q2a", "q2a.j2"),
    ("identity", "identity.j2"),
    ("paraphrase", "paraphrase.j2"),
    ("judge_system", "judge_system.j2"),
    ("judge_user", "judge_user.j2"),
    ("answerability", "answerability_user.j2"),
)


class PromptError(RuntimeError):
    pass


def render_text(template: Template, **values: str | None) -> str:
    """Render one template.

    Args:
        template: Compiled template.
        values: Template variables. None omits an optional section.

    Returns:
        The rendered text.

    Raises:
        TypeError: The renderer did not return text.
    """
    rendered: object = template.render(**values)
    if not isinstance(rendered, str):
        raise TypeError("prompt render did not return text")
    return rendered


def shield(text: str, tag: str) -> str:
    """Break a copy of a wrapper tag inside untrusted text.

    Args:
        text: Untrusted text.
        tag: Tag name used by the wrapper.

    Returns:
        Text whose copies of that tag cannot close the wrapper.
    """
    pattern = re.compile(rf"<\s*/?\s*{re.escape(tag)}\b[^>]*>", re.IGNORECASE)
    return pattern.sub(lambda match: "<\u200b" + match.group(0)[1:], text)


def template_bytes() -> dict[str, bytes]:
    """Read the template files.

    Returns:
        Template bytes keyed by name. Newlines are LF, so the digest does not
        depend on the checkout.
    """
    return {
        key: _lf(files("echochain.prompts").joinpath(filename).read_bytes())
        for key, filename in _FILES
    }


def template_hashes() -> dict[str, str]:
    """Hash the template files.

    Returns:
        SHA-256 hex digests keyed by template name.
    """
    return {
        key: hashlib.sha256(raw).hexdigest() for key, raw in template_bytes().items()
    }


def _lf(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def _version(block: str, filename: str) -> str:
    for line in block.splitlines():
        name, separator, value = line.partition(":")
        if separator == "" or name.strip() != "rubric_version":
            continue
        version = value.strip().strip("\"'")
        if version == "":
            break
        return version
    raise PromptError(f"{filename} is missing rubric_version")


def _body(raw: bytes, filename: str) -> tuple[str, str]:
    text = raw.decode("utf-8")
    match = _FRONT.match(text)
    if match is None:
        raise PromptError(f"{filename} is missing front matter")
    version = _version(match.group(1), filename)
    return version, text[match.end() :].strip()


def _load() -> tuple[str, dict[str, str]]:
    raw_files = template_bytes()
    versions: dict[str, str] = {}
    bodies: dict[str, str] = {}
    for key, filename in _FILES:
        version, body = _body(raw_files[key], filename)
        versions[key] = version
        bodies[key] = body
    chosen = set(versions.values())
    if len(chosen) != 1:
        raise PromptError("template rubric versions disagree")
    return next(iter(chosen)), bodies


RUBRIC_VERSION, _BODIES = _load()
_ENV = Environment(autoescape=False)
_ENV.filters["shield"] = shield
A_TO_Q = _ENV.from_string(_BODIES["a2q"])
Q_TO_A = _ENV.from_string(_BODIES["q2a"])
IDENTITY_TMPL = _ENV.from_string(_BODIES["identity"])
PARAPHRASE_TMPL = _ENV.from_string(_BODIES["paraphrase"])
JUDGE_SYS = _BODIES["judge_system"]
JUDGE_USER_TMPL = _ENV.from_string(_BODIES["judge_user"])
ANSWERABILITY_USER_TMPL = _ENV.from_string(_BODIES["answerability"])
