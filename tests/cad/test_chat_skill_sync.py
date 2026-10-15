"""The vendored Skill must stay byte-identical to the canonical one.

chat/skill/ exists only because the Docker build context is the backend
directory, so the real Skill at .claude/skills/opentactilecad/ cannot be
COPYed into the image. That duplication is a drift hazard: edit the Skill,
forget the copy, and the deployed assistant quietly designs by the old rules
while Claude Code uses the new ones — with nothing failing to reveal it.

This test is what makes that failure loud. If it fails, re-copy the files:

    cp .claude/skills/opentactilecad/SKILL.md \\
       .claude/skills/opentactilecad/helpers/outline.py \\
       .claude/skills/opentactilecad/helpers/preflight.py \\
       tactile-pcb-web/backend/chat/skill/

It skips (rather than fails) when the canonical Skill is absent, which is the
normal state inside the deployed image.
"""

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "opentac" / "cad"
VENDORED = BACKEND / "chat" / "skill"
CANONICAL = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "opentactilecad"

FILES = [
    ("SKILL.md", CANONICAL / "SKILL.md"),
    ("outline.py", CANONICAL / "helpers" / "outline.py"),
    ("preflight.py", CANONICAL / "helpers" / "preflight.py"),
]


@pytest.mark.parametrize("name,source", FILES, ids=[f[0] for f in FILES])
def test_vendored_skill_matches_canonical(name, source):
    if not source.is_file():
        pytest.skip(f"canonical Skill not present ({source}) — deployed image")
    vendored = VENDORED / name
    assert vendored.is_file(), f"{name} missing from chat/skill/"
    # Compare text, not bytes: git may normalise line endings on checkout, and
    # a CRLF difference is not drift.
    assert (vendored.read_text(encoding="utf-8").replace("\r\n", "\n")
            == source.read_text(encoding="utf-8").replace("\r\n", "\n")), (
        f"chat/skill/{name} has drifted from the canonical Skill. "
        "Re-copy it — see this file's docstring.")


def test_instructions_build_and_carry_the_rules():
    """The preamble plus the Skill body, with Setup dropped."""
    pytest.importorskip("google.genai", reason="chat needs the [demo] extra")
    from opentac.cad.chat import context

    text = context.system_instructions()
    assert "OpenTactileCAD design assistant" in text     # preamble
    assert "## Board mode" in text                        # skill body
    assert "## Setup" not in text                         # dropped
    assert "The router is not a choice" in text           # hug-only rule
    assert len(text) > 8000
