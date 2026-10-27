from __future__ import annotations

from echochain.prompts import A_TO_Q, render_text, shield, template_names


def test_templates_render_and_shield() -> None:
    names = template_names()
    assert len(names) == 7
    assert "a2q.j2" in names
    rendered = render_text(A_TO_Q, answer="safe")
    assert "safe" in rendered
    assert "<answer" not in shield("before <answer> after", "answer")
