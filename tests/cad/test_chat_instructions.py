"""The design chat reads the one Skill at .claude/skills/opentac/, the same files
Claude Code loads, so the hosted assistant and a local agent design by the same rules."""

import pytest

pytest.importorskip("google.genai", reason="chat needs the [demo] extra")


def test_chat_reads_the_repository_skill():
    from opentac.cad.chat import context

    assert (context.SKILL_DIR / "SKILL.md").is_file()
    assert context.SKILL_DIR.parts[-3:] == (".claude", "skills", "opentac")


def test_chat_tools_import_the_skill_helpers():
    from opentac.cad.chat import tools

    assert tools.O.__file__.endswith("helpers/outline.py")
    assert tools.P.__file__.endswith("helpers/preflight.py")


def test_instructions_build_and_carry_the_rules():
    """The preamble plus the Skill body, with Setup dropped."""
    from opentac.cad.chat import context

    text = context.system_instructions()
    assert "OpenTac-CAD design assistant" in text     # preamble
    assert "## Board mode" in text                        # skill body
    assert "## Setup" not in text                         # dropped
    assert "The router is not a choice" in text           # hug-only rule
    assert len(text) > 8000
