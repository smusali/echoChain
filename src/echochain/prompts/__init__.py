from __future__ import annotations

import re
from importlib.resources import files

from jinja2 import Environment, Template

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


def template_names() -> tuple[str, ...]:
    """Return the packaged template filenames.

    Returns:
        Filenames in package order.
    """
    return tuple(filename for _key, filename in _FILES)


def _body(filename: str) -> str:
    raw = files("echochain.prompts").joinpath(filename).read_bytes()
    return raw.replace(b"\r\n", b"\n").decode("utf-8").strip()


def _load() -> dict[str, str]:
    return {key: _body(filename) for key, filename in _FILES}


_BODIES = _load()
_ENV = Environment(autoescape=False)  # noqa: S701
_ENV.filters["shield"] = shield
A_TO_Q = _ENV.from_string(_BODIES["a2q"])
Q_TO_A = _ENV.from_string(_BODIES["q2a"])
IDENTITY_TMPL = _ENV.from_string(_BODIES["identity"])
PARAPHRASE_TMPL = _ENV.from_string(_BODIES["paraphrase"])
JUDGE_SYS = _BODIES["judge_system"]
JUDGE_USER_TMPL = _ENV.from_string(_BODIES["judge_user"])
ANSWERABILITY_USER_TMPL = _ENV.from_string(_BODIES["answerability"])
