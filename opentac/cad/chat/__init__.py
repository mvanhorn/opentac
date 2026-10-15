"""Plain-English sensor design: the chat routes, mounted into the main app.

The model designs; the pipeline builds. Gemini gets SKILL.md as its system
instructions and three function-calling tools, and every board comes out of
`main.generate()` — the same call the /generate route makes. Nothing here can
invent a PCB.

Import is deliberately failure-tolerant at the call site (see main.py): if
google-genai is not installed or GEMINI_API_KEY is unset, the site must keep
working exactly as before, with chat simply unavailable.
"""

from __future__ import annotations

import os
import sys
import traceback

from fastapi import APIRouter
from pydantic import BaseModel

try:
    # Local convenience only: put GEMINI_API_KEY in backend/.env and the
    # double-click launcher picks it up. On Render the env vars are real and
    # there is no file, so a missing dotenv is not an error.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

from . import context, tools
from .gemini import GeminiProvider, QuotaExhausted

def say(msg: str) -> None:
    """Print without ever raising.

    The Windows console is cp1252, and this logs model-written text that
    routinely contains characters it cannot encode ("40 x 40" comes back as
    "40 × 40"). Unencodable output would otherwise raise UnicodeEncodeError
    from inside the request and turn a finished design into a 500.
    """
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


router = APIRouter(prefix="/api/chat", tags=["chat"])

# In memory, so a restart clears them. Conversations are cheap to restart and
# the alternative is a database this project does not otherwise need.
SESSIONS: dict[str, dict] = {}
_provider: GeminiProvider | None = None

# Keeps one visitor from spending the account on a single conversation. The
# hard cap is the billing account's prepaid balance; this is the polite one.
MAX_TURNS_PER_SESSION = int(os.environ.get("OTC_CHAT_MAX_TURNS", "40"))


def available() -> bool:
    """Whether chat can serve at all — checked before advertising it."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def provider() -> GeminiProvider:
    global _provider
    if _provider is None:
        instructions = context.system_instructions()
        say(f"[chat] {len(instructions)} chars of SKILL.md as system "
            f"instructions")
        _provider = GeminiProvider(system_instruction=instructions,
                                   tool_specs=tools.TOOL_SPECS)
        say(f"[chat] model={_provider.model}")
    return _provider


class ChatIn(BaseModel):
    session_id: str = "default"
    message: str


def _starting(name: str, args: dict) -> str:
    """What is about to happen, in the user's language."""
    if name == "build_sensor_outline":
        return "Building the outline…"
    if name == "preflight_sensor":
        return "Checking whether it fits…"
    if name == "generate_sensor_board":
        return "Routing the board — this is the slow part…"
    return f"{name.replace('_', ' ')}…"


def _finished(name: str, result: dict) -> str:
    """What actually came back. Deliberately the pipeline's own numbers rather
    than anything the model says about them."""
    if "error" in result:
        return f"  → {result['error'][:120]}"
    if name == "build_sensor_outline":
        return (f"  → {result.get('n_vertices')} points, "
                f"{result.get('bbox_w_mm')} × {result.get('bbox_h_mm')} mm, "
                f"cable edge {result.get('cable_edge_len_mm')} mm")
    if name == "preflight_sensor":
        if result.get("blockers"):
            return f"  → won't work: {result['blockers'][0][:110]}"
        p = result.get("predicted", {})
        return (f"  → expects about {p.get('approx_taxels')} taxels, "
                f"{p.get('channels')} channels")
    if name == "generate_sensor_board":
        return f"  → {result.get('report', 'done')}"
    return "  → done"


@router.get("/progress")
def progress(session_id: str = "default"):
    """What the current turn is doing, for the page to poll while it waits.

    A design turn runs 10-60 seconds behind a single POST, and a motionless
    "thinking…" for that long reads as a hang. These are the pipeline's own
    steps and numbers, not the model's narration of them.
    """
    sess = SESSIONS.get(session_id)
    if sess is None:
        return {"busy": False, "steps": []}
    return {"busy": sess.get("busy", False), "steps": sess.get("progress", [])}


@router.get("/status")
def status():
    """The page asks this before showing the Describe mode at all, so a
    deployment without a key never advertises a button that cannot work."""
    return {"available": available(),
            "model": os.environ.get("OTC_MODEL", "gemini-3.6-flash")}


@router.post("")
def chat(body: ChatIn):
    if not available():
        return {"error": "The design assistant is not configured on this "
                         "server (no GEMINI_API_KEY)."}

    sess = SESSIONS.setdefault(
        body.session_id, {"tools": tools.Session(), "conversation": None,
                          "turns": 0})
    if sess["turns"] >= MAX_TURNS_PER_SESSION:
        return {"error": f"This conversation has reached its {MAX_TURNS_PER_SESSION}"
                         "-turn limit. Start a new one to keep designing."}

    say(f"\n[chat] USER: {body.message}")
    # Reset before the turn, not after: the page starts polling as soon as it
    # sends, and would otherwise be shown the previous turn's steps.
    sess["progress"] = ["Thinking…"]
    sess["busy"] = True
    try:
        p = provider()
        if sess["conversation"] is None:
            sess["conversation"] = p.new_conversation()

        def tool_fn(name: str, args: dict) -> dict:
            sess["progress"].append(_starting(name, args))
            result = sess["tools"].dispatch(name, args)
            sess["progress"].append(_finished(name, result))
            tools.log_call(name, args, result)
            return result

        turn = p.chat(sess["conversation"], body.message, tool_fn)
    except QuotaExhausted as exc:
        say(f"[chat] {exc}")
        return {"error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        # Always cleared, so a failed turn cannot leave the page polling a
        # spinner forever.
        sess["busy"] = False

    sess["turns"] += 1
    designs = sess["tools"].designs
    newest = designs[-1] if designs else None
    say(f"[chat] ASSISTANT: {turn.text[:200]}")
    return {
        "text": turn.text,
        "tool_calls": [{"name": c["name"], "args": c["args"]}
                       for c in turn.tool_calls],
        # Only the newest board is returned: the page loads it into the
        # editor, and the SVG is large enough that shipping the whole history
        # every turn would be wasteful.
        "design": None if newest is None else {
            "name": newest["name"],
            "outline": newest["outline"],
            "params": newest["params"],
            "svg": newest["svg"],
            "stats": newest["stats"],
        },
        "turns_left": MAX_TURNS_PER_SESSION - sess["turns"],
    }


@router.post("/reset")
def reset(body: ChatIn):
    SESSIONS.pop(body.session_id, None)
    return {"ok": True}
