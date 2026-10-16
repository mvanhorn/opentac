"""Every way a design-chat turn can fail must reach the page as readable JSON.

The hosted page is cross-origin (GitHub Pages -> Render), so a response without
CORS headers, or no response at all, shows up in the browser only as
"TypeError: Failed to fetch". These pin the paths that used to do that: an
unhandled exception (Starlette's plain-text 500 sits outside CORSMiddleware), a
tool that raises, a model call that raises, and a reply FastAPI cannot encode.
They also pin the chat tool's defaults (grow mode, explicit copper layers) and
its refusal of a fixed-outline run too slow for the free hosted server.
"""
import numpy as np
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from opentac.cad import main

ORIGIN = "https://opentac.github.io"

# Size 9 insole, as the chat builds it (SKILL.md's worked example).
INSOLE = [
    (0.0, 0.0), (26.0, 6.0), (32.0, 30.0), (30.0, 62.0), (26.0, 95.0), (33.0, 130.0),
    (45.0, 160.0), (48.0, 178.0), (46.0, 205.0), (40.0, 230.0), (28.0, 247.0),
    (10.0, 254.0), (-8.0, 253.0), (-24.0, 245.0), (-33.0, 228.0), (-40.0, 205.0),
    (-48.0, 178.0), (-46.0, 155.0), (-39.0, 125.0), (-37.0, 95.0), (-36.0, 62.0),
    (-34.0, 30.0), (-26.0, 6.0),
]
DENSE = dict(pixel_w_mm=3.0, pixel_h_mm=8.4, pitch_x_mm=3.3, pitch_y_mm=8.6)


def test_unhandled_exception_is_json_with_cors_headers():
    def boom():
        raise RuntimeError("kaboom")

    # In front of the frontend's "/" mount, which would otherwise claim the path.
    main.app.router.routes.insert(0, APIRoute("/__test_boom", boom, methods=["GET"]))
    try:
        r = TestClient(main.app, raise_server_exceptions=False).get(
            "/__test_boom", headers={"Origin": ORIGIN})
    finally:
        main.app.router.routes.pop(0)
    assert r.status_code == 500
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    assert "kaboom" in r.json()["error"]


# --- the chat routes --------------------------------------------------------

class FakeProvider:
    """Stands in for Gemini: runs a fixed script of tool calls."""

    def __init__(self, script):
        self.script = script

    def new_conversation(self):
        return []

    def chat(self, conversation, text, tool_fn):
        from opentac.cad.chat.gemini import Turn
        calls = []
        for name, args in self.script:
            args = args() if callable(args) else args
            calls.append({"name": name, "args": args, "result": tool_fn(name, args)})
        last = calls[-1]["result"] if calls else {}
        return Turn(text=f"done: {last.get('error') or last.get('report')}",
                    tool_calls=calls)


@pytest.fixture
def chat(monkeypatch):
    pytest.importorskip("google.genai", reason="chat needs the [demo] extra")
    from opentac.cad import chat as chat_mod
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(chat_mod, "SESSIONS", {})
    return chat_mod


def _post(message, session="t1", request_id="r1"):
    client = TestClient(main.app, raise_server_exceptions=False)
    return client.post("/api/chat", headers={"Origin": ORIGIN},
                       json={"session_id": session, "message": message,
                             "request_id": request_id})


OUTLINE_CALL = ("build_sensor_outline", {"shape": "rectangle", "width_mm": 30,
                                         "height_mm": 30, "cable_side": "bottom"})


def test_tool_that_raises_comes_back_as_text(chat, monkeypatch):
    def broken_generate(req):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(main, "generate", broken_generate)
    monkeypatch.setattr(chat, "provider", lambda: FakeProvider([
        OUTLINE_CALL,
        ("generate_sensor_board", {"outline_id": "outline_1", "design_name": "x"}),
    ]))
    r = _post("a 30 mm square")
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    body = r.json()
    assert "router exploded" in body["text"]
    assert body["request_id"] == "r1"


def test_model_call_that_raises_is_a_json_error(chat, monkeypatch):
    class Dead(FakeProvider):
        def chat(self, *a):
            raise ConnectionError("Gemini unreachable")

    monkeypatch.setattr(chat, "provider", lambda: Dead([]))
    r = _post("hello")
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    assert "Gemini unreachable" in r.json()["error"]


def test_numpy_values_in_the_reply_still_encode(chat, monkeypatch):
    def fake_generate(req):
        return main.GenerateResponse(
            svg="<svg/>", zip_b64="", drc={"violations": np.int64(0), "details": []},
            stats={"total_pixels": np.int64(9), "col_routes": np.int64(3),
                   "row_routes": np.int64(3), "connector_pos": np.int64(30),
                   "board_mode": "expand", "copper_layers": np.int64(2)})

    monkeypatch.setattr(main, "generate", fake_generate)
    monkeypatch.setattr(chat, "provider", lambda: FakeProvider([
        OUTLINE_CALL,
        ("generate_sensor_board", {"outline_id": "outline_1", "design_name": "x"}),
    ]))
    r = _post("a 30 mm square")
    assert r.status_code == 200, r.text[:300]
    assert r.json()["design"]["stats"]["total_pixels"] == 9


def test_progress_and_last_let_the_page_recover_a_dropped_turn(chat, monkeypatch):
    monkeypatch.setattr(chat, "provider", lambda: FakeProvider([OUTLINE_CALL]))
    client = TestClient(main.app)
    # A session the server has never seen (or lost in a restart) says so.
    gone = client.get("/api/chat/progress", params={"session_id": "t9"}).json()
    assert gone["known"] is False and gone["busy"] is False

    _post("a 30 mm square", session="t9", request_id="abc")
    st = client.get("/api/chat/progress", params={"session_id": "t9"}).json()
    assert st["known"] is True and st["busy"] is False
    last = client.get("/api/chat/last", params={"session_id": "t9"}).json()
    assert last["request_id"] == "abc" and "text" in last


# --- the tool layer ----------------------------------------------------------
@pytest.fixture
def session():
    pytest.importorskip("google.genai", reason="chat needs the [demo] extra")
    from opentac.cad.chat import tools
    s = tools.Session()
    s.build_sensor_outline(shape="polygon", points=[list(p) for p in INSOLE],
                           cable_side="bottom", cable_edge_width_mm=40)
    return s


@pytest.fixture
def captured(monkeypatch):
    seen = []

    def fake_generate(req):
        seen.append(req)
        return main.GenerateResponse(
            svg="<svg/>", zip_b64="", drc={"violations": 0, "details": []},
            stats={"total_pixels": 1, "board_mode": req.board_mode,
                   "copper_layers": req.copper_layers})

    monkeypatch.setattr(main, "generate", fake_generate)
    return seen


def test_generate_defaults_to_grow_and_passes_copper_layers(session, captured):
    out = session.dispatch("generate_sensor_board", {
        "outline_id": "outline_1", "design_name": "insole4", "copper_layers": 4, **DENSE})
    assert "error" not in out, out
    req = captured[-1]
    assert req.board_mode == "expand" and req.copper_layers == 4 and req.router == "hug"
    # The page re-runs the design from these, so layers and mode are explicit.
    params = session.designs[-1]["params"]
    assert params["copper_layers"] == 4 and params["board_mode"] == "expand"
    assert params["readout_board"] == "auto"

    session.dispatch("generate_sensor_board", {"outline_id": "outline_1",
                                               "design_name": "insole2", **DENSE})
    assert captured[-1].copper_layers == 2
    assert session.designs[-1]["params"]["copper_layers"] == 2


def test_bad_copper_layers_is_a_readable_error(session, captured):
    out = session.dispatch("generate_sensor_board", {
        "outline_id": "outline_1", "design_name": "x", "copper_layers": 3})
    assert out["error"] == "copper_layers must be 2 or 4, got 3."
    assert not captured


def test_slow_fixed_four_layer_run_is_refused_on_the_hosted_server(
        session, captured, monkeypatch):
    from opentac.cad.chat import tools
    monkeypatch.setattr(tools, "FIXED_WORK_LIMIT", 1200)

    out = session.dispatch("generate_sensor_board", {
        "outline_id": "outline_1", "design_name": "x", "board_mode": "fixed_keepout",
        "copper_layers": 4, **DENSE})
    assert "Not built" in out["error"] and "grow mode" in out["error"]
    assert not captured, "the refused run must not reach the pipeline"

    # The same board in grow mode, and a coarse fixed one, are built.
    for extra in ({"copper_layers": 4, **DENSE},
                  {"board_mode": "fixed_keepout", "copper_layers": 4,
                   "pixel_w_mm": 8.4, "pixel_h_mm": 8.4,
                   "pitch_x_mm": 8.6, "pitch_y_mm": 8.6}):
        out = session.dispatch("generate_sensor_board", {
            "outline_id": "outline_1", "design_name": "x", **extra})
        assert "error" not in out, out
    assert len(captured) == 2


def test_no_work_cap_off_the_hosted_server(session, captured, monkeypatch):
    from opentac.cad.chat import tools
    monkeypatch.setattr(tools, "FIXED_WORK_LIMIT", 0)
    out = session.dispatch("generate_sensor_board", {
        "outline_id": "outline_1", "design_name": "x", "board_mode": "fixed_keepout",
        "copper_layers": 4, **DENSE})
    assert "error" not in out, out
    assert captured[-1].board_mode == "fixed_keepout"


def test_out_of_memory_inside_the_pipeline_is_named(session, monkeypatch):
    def oom(req):
        try:
            raise MemoryError()
        except MemoryError as e:        # what main.generate does with any failure
            raise main.HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    monkeypatch.setattr(main, "generate", oom)
    out = session.dispatch("generate_sensor_board", {"outline_id": "outline_1",
                                                     "design_name": "x"})
    assert "ran out of memory" in out["error"]
