"""Read a sensor: print the frame rate and the strongest taxel for five seconds.

    uv run python examples/read_sensor.py

Run `uv run opentac setup my_design.zip` once first, so the board and this
computer know which sensor is plugged in.
"""
import time

import numpy as np

from opentac.device import Sensor

with Sensor() as s:
    print(s)
    if s.manifest is None:
        raise SystemExit("this computer has no manifest for the board's sensor: run opentac setup")
    zero = np.mean([f.taxels for f in s.frames(limit=30)], axis=0)    # nothing touching
    t0, n, shown = time.time(), 0, 0.0
    for frame in s.frames():
        n += 1
        if frame.t - shown < 0.2:          # print five times a second
            continue
        shown = frame.t
        above = frame.taxels - zero
        k = int(np.argmax(above))
        x, y = s.manifest.xy[k]
        print(f"{n / (time.time() - t0):6.0f} fps   peak {above[k]:6.0f} counts at "
              f"({x:6.1f}, {y:6.1f}) mm")
        if time.time() - t0 > 5:
            break
