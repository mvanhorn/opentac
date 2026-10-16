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


def test_tools_and_skill_agree_on_grow_mode_and_copper_layers():
    """The skill says grow mode unless the outline is a hard limit, and 4 layers
    when packing matters; the tool must default to grow and accept the layers,
    or the model is told about an option it cannot use (a "4-layer insole"
    request used to be rejected as an unsupported parameter)."""
    from opentac.cad.chat import context, tools

    text = context.system_instructions()
    assert "grow mode, the default" in text
    assert "use 4 layers, in grow" in text
    assert "hexagon" in text and "30-pin board" in text   # the other Arik rules

    assert tools.DEFAULT_BOARD_MODE == "expand"
    gen = next(s for s in tools.TOOL_SPECS if s["name"] == "generate_sensor_board")
    assert "copper_layers" in gen["parameters"]["properties"]
    assert "copper_layers" in tools.GEN_PARAMS
