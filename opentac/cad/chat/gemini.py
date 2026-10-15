"""Gemini provider, on the google-genai SDK.

Automatic function calling is deliberately disabled: the loop is run by hand
so every call and result can be logged to the terminal and handed back to the
UI, which is the whole point of a debugging prototype.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types


# One assistant reply, as the routes and the page see it.
@dataclass
class Turn:
    text: str
    tool_calls: list = field(default_factory=list)   # {name, args, result}
    usage: dict = field(default_factory=dict)


ToolFn = Callable[[str, dict], dict]

# gemini-2.5-flash is retired for new API keys: generateContent returns 404
# pointing here, even though the model still appears in models.list().
DEFAULT_MODEL = "gemini-3.6-flash"
MAX_TOOL_ROUNDS = 12          # a generate is slow; a runaway loop is expensive

# The free tier rate-limits per minute AND per day, and the two need opposite
# handling. A per-minute 429 clears on its own, so it is worth waiting out. A
# per-day 429 does not clear until midnight Pacific, so retrying it just burns
# five minutes before failing anyway - fail fast and say what to change.
MAX_RETRIES = 6
BACKOFF_S = (2, 5, 12, 25, 45, 60)

# Free-tier daily caps are small (20/day on gemini-3.6-flash) and one design
# conversation can spend a dozen requests, so running out is routine. Each
# model has its own pool, which makes switching model the quickest unblock.
ALTERNATIVES = ("gemini-3.5-flash", "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite", "gemini-3-flash-preview")


def _schema(node: dict) -> dict:
    """JSON Schema -> the Schema dialect Gemini's FunctionDeclaration wants.

    Same keywords, but the type names are upper-case enum members, and unknown
    keywords are rejected rather than ignored - so this walks the tree instead
    of passing the dict straight through.
    """
    out: dict = {}
    if "type" in node:
        out["type"] = node["type"].upper()
    for key in ("description", "enum", "required"):
        if key in node:
            out[key] = node[key]
    if "properties" in node:
        out["properties"] = {k: _schema(v)
                             for k, v in node["properties"].items()}
    if "items" in node:
        out["items"] = _schema(node["items"])
    return out


class GeminiProvider:
    name = "gemini"

    def __init__(self, system_instruction: str, tool_specs: list,
                 model: str | None = None, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in web-prototype/.env - "
                "see .env.example.")
        self.model = model or os.environ.get("OTC_MODEL") or DEFAULT_MODEL
        self.client = genai.Client(api_key=key)
        self.config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=s["name"], description=s["description"],
                    parameters=_schema(s["parameters"]))
                for s in tool_specs])],
            # Manual loop: see module docstring.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True),
        )

    def new_conversation(self) -> list:
        return []                      # list[types.Content]

    def _generate(self, conversation: list):
        """One API call, waiting out the failures that are about capacity
        rather than about the request: a per-minute 429 and a 503. A per-day
        429 is raised immediately - no amount of waiting fixes it today."""
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=conversation,
                    config=self.config)
            except (genai_errors.ClientError,
                    genai_errors.ServerError) as exc:
                if getattr(exc, "code", None) not in (429, 503):
                    raise
                if _quota_scope(exc) == "day":
                    raise QuotaExhausted(self.model, _quota_limit(exc)) from exc
                last = exc
                wait = _retry_delay(exc) or BACKOFF_S[
                    min(attempt, len(BACKOFF_S) - 1)]
                _log(f"[{exc.code}] "
                     f"{'rate limited' if exc.code == 429 else 'model busy'}"
                     f" - retrying in {wait:.0f}s ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
        raise last                                    # type: ignore[misc]

    def chat(self, conversation: list, user_text: str, tool_fn: ToolFn) -> Turn:
        # Everything this turn appends is rolled back if the turn dies. A
        # failed call would otherwise leave a dangling user message - or worse,
        # a model turn whose tool responses never arrived - and every later
        # message in the session would be sent on top of that broken history.
        committed = len(conversation)
        try:
            return self._chat(conversation, user_text, tool_fn)
        except Exception:
            del conversation[committed:]
            raise

    def _chat(self, conversation: list, user_text: str,
              tool_fn: ToolFn) -> Turn:
        conversation.append(types.Content(
            role="user", parts=[types.Part(text=user_text)]))

        calls_made: list[dict] = []
        usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self._generate(conversation)
            _add_usage(usage, resp)

            content = _content_of(resp)
            if content is None:
                _log_usage(usage)
                return Turn(text=_no_content_reason(resp),
                            tool_calls=calls_made, usage=usage)
            conversation.append(content)

            calls = [p.function_call for p in (content.parts or [])
                     if getattr(p, "function_call", None)]
            if not calls:
                _log_usage(usage)
                return Turn(text=_text_of(content), tool_calls=calls_made,
                            usage=usage)

            # Say the interim thought out loud - "let me preflight that first"
            # is useful in the transcript even though the turn continues.
            parts = []
            for call in calls:
                args = dict(call.args or {})
                result = tool_fn(call.name, args)
                calls_made.append({"name": call.name, "args": args,
                                   "result": result})
                parts.append(types.Part.from_function_response(
                    name=call.name, response=_jsonable(result)))
            conversation.append(types.Content(role="user", parts=parts))

        _log_usage(usage)
        return Turn(
            text=(f"Stopped after {MAX_TOOL_ROUNDS} tool rounds without a "
                  "final answer. Check the terminal log."),
            tool_calls=calls_made, usage=usage)


class QuotaExhausted(RuntimeError):
    """Today's free-tier allowance for this model is gone.

    Carries its own remedy, because the raw 429 does not distinguish "wait a
    moment" from "wait until tomorrow", and the difference is the whole
    answer.
    """

    def __init__(self, model: str, limit: str | None):
        self.model = model
        cap = f" (free tier allows {limit}/day)" if limit else ""
        others = ", ".join(m for m in ALTERNATIVES if m != model)
        super().__init__(
            f"Daily free-tier quota for {model} is used up{cap}. It resets at "
            f"midnight Pacific. To keep going now, either set OTC_MODEL in "
            f"web-prototype/.env to a model with its own separate daily pool "
            f"({others}) and restart the server, or enable billing on the "
            f"Google Cloud project at https://aistudio.google.com/apikey for "
            f"much higher limits.")


def _violations(exc) -> list:
    d = getattr(exc, "details", None)
    if not isinstance(d, dict):
        return []
    out = []
    for detail in d.get("error", {}).get("details", []):
        if "QuotaFailure" in detail.get("@type", ""):
            out += detail.get("violations", [])
    return out


def _quota_scope(exc) -> str | None:
    """'day', 'minute', or None - read from the quotaId Google returns."""
    for v in _violations(exc):
        qid = v.get("quotaId", "")
        if "PerDay" in qid:
            return "day"
        if "PerMinute" in qid:
            return "minute"
    # Fall back to the prose when the structured detail is absent.
    text = str(exc)
    if "PerDay" in text or "per day" in text:
        return "day"
    return None


def _quota_limit(exc) -> str | None:
    for v in _violations(exc):
        if "PerDay" in v.get("quotaId", ""):
            return v.get("quotaValue")
    return None


def _add_usage(usage: dict, resp) -> None:
    """Accumulate one response's token counts.

    Each tool round resends the whole conversation, system instructions
    included, so input tokens grow quadratically across a turn - which is the
    number that decides what this costs to host.
    """
    usage["requests"] += 1
    md = getattr(resp, "usage_metadata", None)
    if md is None:
        return
    usage["input_tokens"] += getattr(md, "prompt_token_count", 0) or 0
    usage["output_tokens"] += ((getattr(md, "candidates_token_count", 0) or 0)
                               + (getattr(md, "thoughts_token_count", 0) or 0))


def _log_usage(usage: dict) -> None:
    _log(f"turn used {usage['requests']} API request(s), "
         f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens")


def _log(msg: str) -> None:
    """Unbuffered: uvicorn's stdout is usually a pipe, and a buffered debug
    line that appears minutes late is worse than none."""
    print(f"  {msg}", file=sys.stderr, flush=True)


def _retry_delay(exc) -> float | None:
    """Google usually says how long to wait; prefer that over guessing.

    It arrives either as a RetryInfo detail or in the message prose, and the
    RetryInfo is sometimes '0s' while the prose carries the real number - so
    take whichever is larger, and never wait more than a minute.
    """
    best = 0.0
    for detail in (getattr(exc, "details", None) or {}).get("error", {}) \
            .get("details", []) if isinstance(getattr(exc, "details", None),
                                              dict) else []:
        if detail.get("@type", "").endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s", str(detail.get("retryDelay", "")))
            if m:
                best = max(best, float(m.group(1)))
    m = re.search(r"retry in ([\d.]+)\s*(ms|s)\b", str(exc))
    if m:
        secs = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
        best = max(best, secs)
    # A sub-second delay on a per-minute quota just burns another attempt.
    return min(max(best, 2.0), 60.0) if best else None


def _content_of(resp):
    cands = getattr(resp, "candidates", None) or []
    return cands[0].content if cands and cands[0].content else None


def _text_of(content) -> str:
    return "".join(p.text for p in (content.parts or [])
                   if getattr(p, "text", None)).strip()


def _no_content_reason(resp) -> str:
    """Empty candidate list means a block or a stop, not an outage."""
    fb = getattr(resp, "prompt_feedback", None)
    if fb and getattr(fb, "block_reason", None):
        return f"(blocked by the model: {fb.block_reason})"
    cands = getattr(resp, "candidates", None) or []
    if cands and getattr(cands[0], "finish_reason", None):
        return f"(no content returned; finish_reason={cands[0].finish_reason})"
    return "(no content returned by the model)"


def _jsonable(result: dict) -> dict:
    """Function responses must be JSON-serialisable; numpy scalars are not."""
    return json.loads(json.dumps(result, default=str))
