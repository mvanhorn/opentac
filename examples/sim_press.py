"""Press a 10 mm ball into a flat sensor at 5 N and look at the reading.

    uv run python examples/sim_press.py [manifest.json]

Uses the foam model (a skin with no bump sheet). Without an argument it loads
a 48-taxel test board from tests/sim/fixtures.
"""
import sys
from pathlib import Path

import numpy as np

from opentac import manifest
from opentac.sim import Sensor

path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1] / "tests/sim/fixtures/flat_square.json"
m = manifest.load(path)
sensor = Sensor(m)                                       # model="foam"

frame = sensor.press("sphere", force_n=5.0, radius_mm=5.0)   # 10 mm ball, at the taxel centroid
k = int(np.argmax(frame.taxels))
print(f"{m.n} taxels at {sensor.pitch_mm} mm pitch")
print(f"peak {frame.taxels[k]:.3f} N on taxel {k} at {m.xy[k].round(2)} mm; "
      f"frame total {frame.taxels.sum():.2f} N of 5.00 N pressed")

image = frame.image(m)                                   # rows x cols, laid out like the sensor
print(np.array2string(image, precision=2, suppress_small=True))

import matplotlib.pyplot as plt  # noqa: E402

plt.imshow(image, cmap="magma")
plt.colorbar(label="N per taxel")
plt.title("10 mm ball, 5 N, foam model")
plt.show()
