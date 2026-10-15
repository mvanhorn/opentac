"""Xampling: read 64 taxels per frame instead of all of them, and compare frame rates.

    uv run python examples/xampling.py

Keep the sensor untouched while it zeroes.
"""
import time

from opentac.device import Sensor


def fps(sensor, seconds=2.0):
    t0, n = time.time(), 0
    for _ in sensor.frames():
        n += 1
        if time.time() - t0 > seconds:
            return n / (time.time() - t0)


with Sensor() as s:
    print(f"raster: {fps(s):.0f} fps, {s.stored['active']} taxels per frame")
    z = s.zero()                              # nothing touching
    print(f"zeroed: noise median <= {z['noise_median']}, max {z['noise_max']} counts")
    s.xampling(budget=64)                     # fixed: 64 reads per frame
    frame = s.read()
    print(f"xampling: {fps(s):.0f} fps, {int(frame.mask.sum())} taxels per frame")
    s.xampling(budget=64, grow=True)          # 64 search reads, then grow around contacts
    print("press the sensor...")
    for frame in s.frames(limit=500):
        pass
    print(f"grow: last frame read {int(frame.mask.sum())} taxels")
    s.raster()
