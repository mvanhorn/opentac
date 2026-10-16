"""Worked example and self-test: a size 9 insole pressure sensor.

Run it after cloning to confirm the skill is wired up:

    uv run python .claude/skills/opentac/examples/insole.py /tmp/insole

Expect a clean grow-mode board of about 200 taxels (11 x 28 on the 60-pin readout
board), DRC 0, in a few seconds.

Dimensions come from published sizing data, not from memory: a US men's
size 9 foot is ~254mm long and averages 94.4mm across the ball (86-102mm
range). Note the modelling question that raises — an insole for the *shoe*
would follow the 255-281mm internal shoe length instead. Ask which.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

import outline as O          # noqa: E402
import preflight as P        # noqa: E402
import generate as G         # noqa: E402

# Right foot. Heel at y=0, toe at y=254. x>0 is the medial (big toe) side.
CONTROL_POINTS = [
    (  0.0,   0.0), ( 26.0,   6.0), ( 32.0,  30.0), ( 30.0,  62.0),
    ( 26.0,  95.0), ( 33.0, 130.0), ( 45.0, 160.0), ( 48.0, 178.0),
    ( 46.0, 205.0), ( 40.0, 230.0), ( 28.0, 247.0), ( 10.0, 254.0),
    ( -8.0, 253.0), (-24.0, 245.0), (-33.0, 228.0), (-40.0, 205.0),
    (-48.0, 178.0), (-46.0, 155.0), (-39.0, 125.0), (-37.0,  95.0),
    (-36.0,  62.0), (-34.0,  30.0), (-26.0,   6.0),
]


def build_outline() -> np.ndarray:
    pts = O.resample_closed(np.array(CONTROL_POINTS), n=260)
    # The cable exits at the heel, so the heel gets an explicit flat edge.
    # Without this the first edge is an arbitrary chord of the heel curve and
    # the whole board rotates by that angle. 40mm clears the 60-position
    # connector this grid needs (31.6mm).
    return O.with_flat_cable_edge(pts, side="bottom", width_mm=40.0)


def main(out_dir: str) -> int:
    ol = build_outline()

    problems = O.validate(ol, rows=29, cols=10)
    if problems:
        print("Outline validation failed:")
        for p in problems:
            print("  -", p)
        return 1

    # 8.6mm pitch keeps the rows within the 60-pin readout board (30 max).
    # An insole is 2.6:1, so a square grid over it is coarse
    # along the foot. Taxels here are square by choice — pressure mapping
    # wants uniform cells — and the row count is what sets the channel load.
    kw = dict(pixel_w_mm=8.4, pixel_h_mm=8.4, pitch_x_mm=8.6, pitch_y_mm=8.6)

    pf = P.preflight(ol, kw["pixel_w_mm"], kw["pixel_h_mm"],
                     kw["pitch_x_mm"], kw["pitch_y_mm"], router="hug")
    for b in pf["blockers"]:
        print("BLOCKER:", b)
    if pf["blockers"]:
        return 1
    for w in pf["warnings"]:
        print("note:", w)

    res = G.generate_board(ol, out_dir, **kw)
    print(res["report"])
    print("files in", res["out_dir"])
    return 0 if res["drc"]["violations"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "insole_out"))
