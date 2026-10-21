"""Plain-English sensor design: the chat routes, mounted into the main app.

The model designs; the pipeline builds. Gemini gets SKILL.md as its system
instructions and three function-calling tools, and every board comes out of
`main.generate()` — the same call the /generate route makes. Nothing here can
invent a PCB.

Import is deliberately failure-tolerant at the call site (see main.py): if
google-genai is not installed or GEMINI_API_KEY is unset, the rest of the
site keeps working, with chat unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import traceback

from fastapi import APIRouter
from pydantic import BaseModel

try:
    # Local convenience only: put GEMINI_API_KEY in opentac/cad/.env and the
    # double-click launcher picks it up. On the hosted server the environment
    # variables are set directly and there is no file, so a missing dotenv is
    # not an error.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

from . import context, tools
from .gemini import GeminiProvider, QuotaExhausted

# Conversation traffic (prompts, replies, tool calls) is logged at DEBUG only,
# so a server does not record what visitors type unless its operator enables
# it. A logging handler that cannot encode a character reports the error
# itself; it never raises into the request.
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/chat", tags=["chat"])

# In memory, so a restart clears them. Conversations are cheap to restart and
# the alternative is a database this project does not otherwise need.
SESSIONS: dict[str, dict] = {}
_provider: GeminiProvider | None = None

# Limits the number of turns per conversation.
MAX_TURNS_PER_SESSION = int(os.environ.get("OTC_CHAT_MAX_TURNS", "40"))


def available() -> bool:
    """Whether chat can serve at all — checked before advertising it."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def provider() -> GeminiProvider:
    global _provider
    if _provider is None:
        instructions = context.system_instructions()
        logger.debug("%d chars of SKILL.md as system instructions",
                     len(instructions))
        _provider = GeminiProvider(system_instruction=instructions,
                                   tool_specs=tools.TOOL_SPECS)
        logger.debug("model=%s", _provider.model)
    return _provider


class ChatIn(BaseModel):
    session_id: str = "default"
    message: str
    # Echoed back by /last, so a page whose POST died mid-turn can tell its
    # own finished turn from an older one (see chat-panel.js).
    request_id: str | None = None


def _starting(name: str, args: dict) -> str:
    """What is about to happen, in the user's language."""
    if name == "build_sensor_outline":
        return "Building the outline…"
    if name == "preflight_sensor":
        return "Checking whether it fits…"
    if name == "generate_sensor_board":
        mode = args.get("board_mode") or tools.DEFAULT_BOARD_MODE
        layers = args.get("copper_layers") or 2
        layers = int(layers) if isinstance(layers, (int, float)) else layers
        return (f"Routing the board ({layers} layers, "
                f"{'grow' if mode == 'expand' else 'fixed outline'}) — "
                "this is the slow part…")
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
        # known=False after a turn was sent means the server lost its memory:
        # it restarted mid-turn (for example after running out of memory).
        return {"busy": False, "steps": [], "known": False}
    return {"busy": sess.get("busy", False), "steps": sess.get("progress", []),
            "known": True}


@router.get("/last")
def last(session_id: str = "default"):
    """The newest turn's reply, for a page whose POST died before the reply
    arrived (a dropped connection while the server kept working)."""
    sess = SESSIONS.get(session_id)
    if sess is None or sess.get("last") is None:
        return {"error": "No finished turn for this conversation on the server."}
    return sess["last"]


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

    logger.debug("USER: %s", body.message)
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

        sess["turns"] += 1
        designs = sess["tools"].designs
        newest = designs[-1] if designs else None
        logger.debug("ASSISTANT: %s", turn.text[:200])
        # Round-tripped through json here, inside the try: a value FastAPI
        # cannot encode (a numpy scalar in stats) would otherwise fail after
        # this handler returned, as a bare 500 the page cannot read.
        out = _jsonable({
            "text": turn.text,
            "tool_calls": [{"name": c["name"], "args": c["args"]}
                           for c in turn.tool_calls],
            # Only the newest board is returned: the page loads it into the
            # editor, and the SVG is large enough that returning the whole
            # history every turn would be wasteful.
            "design": None if newest is None else {
                "name": newest["name"],
                "outline": newest["outline"],
                "params": newest["params"],
                "svg": newest["svg"],
                "stats": newest["stats"],
            },
            "turns_left": MAX_TURNS_PER_SESSION - sess["turns"],
        })
    except QuotaExhausted as exc:
        logger.debug("%s", exc)
        out = {"error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        out = {"error": f"The design assistant failed: {type(exc).__name__}: {exc}"}
    finally:
        # Always cleared, so a failed turn cannot leave the page polling a
        # spinner forever.
        sess["busy"] = False

    out["request_id"] = body.request_id
    sess["last"] = out
    return out


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=_plain))


def _plain(v):
    """numpy scalars and arrays as plain numbers; anything else as text."""
    if hasattr(v, "tolist"):
        return v.tolist()
    return str(v)


@router.post("/reset")
def reset(body: ChatIn):
    SESSIONS.pop(body.session_id, None)
    return {"ok": True}
