"""Move a reading from one skin to another with different taxels.

    uv run python examples/transfer.py

Skin A is a 4 mm grid, skin B a 2.5 mm board (both from tests/sim/fixtures).
A press on A is simulated with the foam model, then transferred onto B. For
comparison, the same press is simulated on B directly.
"""
from pathlib import Path

import numpy as np

from opentac import manifest
from opentac.sim import Sensor
from opentac.transfer import Transfer, cop, transfer

FIX = Path(__file__).parents[1] / "tests/sim/fixtures"
a = manifest.load(FIX / "grid_4mm.json")
b = manifest.load(FIX / "flat_square.json")

# A reads more than the 3 N pressed: at 4 mm pitch each taxel's window in the
# foam model is 4.5 mm wide (https://opentac.github.io/sim/#foam-model,
# "Integration-window convention").
frame_a = Sensor(a).press("sphere", force_n=3.0, radius_mm=5.0)      # newtons on A
frame_b = transfer(frame_a, a, b)                                      # centred on B, force conserved
direct_b = Sensor(b).press("sphere", force_n=3.0, radius_mm=5.0)     # the same press, simulated on B

# The same placement as a Transfer object, to see what fell off B's edge.
tr = Transfer.centered(a, b, frame_a.taxels)
print(f"A: {a.n} taxels, read {frame_a.taxels.sum():.2f} N")
print(f"B: {b.n} taxels, transferred {frame_b.taxels.sum():.2f} N, lost off B {tr.lost(frame_a.taxels):.2f} N")
print(f"B simulated directly: {direct_b.taxels.sum():.2f} N")
print(f"centre of pressure on B: transferred {cop(b.xy, frame_b.taxels).round(2)} mm, "
      f"direct {cop(b.xy, direct_b.taxels).round(2)} mm")

np.set_printoptions(precision=2, suppress=True)
print("transferred onto B:\n", frame_b.image(b))
print("simulated on B:\n", direct_b.image(b))
