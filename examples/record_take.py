"""Record a three-second take, then read it back.

    uv run python examples/record_take.py
"""
from opentac.device import Sensor
from opentac.record import load_take, record

with Sensor() as s:
    path = record(s, "recordings", seconds=3, description="example take")

take = load_take(path)
raw = take["raw"]                      # frames x taxels, int16
print(f"{raw.shape[0]} frames of {raw.shape[1]} taxels, "
      f"{take['info']['fps']:.0f} fps, sensor {take['manifest'].name}")
print("taxel 0 is at", take["xy"][0], "mm")
