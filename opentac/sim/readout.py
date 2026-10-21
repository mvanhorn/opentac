# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SO-101 MuJoCo Tactile contributors.
# Input checks and profile lookup from TactileSim (tactilesim/readout.py); modified for OpenTac.
"""Taxel force -> ADC counts, from a measured velostat calibration.

Everything else in this package ends in newtons per taxel. A real board streams
12-bit counts, and the map between the two is set by the piezoresistive layer,
which is a per-build choice -- so the readout is a profile the user picks, not
a constant:

    from opentac.sim import Readout

    r = Readout.from_profile("v3")              # by label, material ("yunjia"), or a path
    counts = r.counts(forces_n, pitch_m)        # (taxels,) counts above baseline
    forces = r.forces(counts, pitch_m)          # the inverse, for a real board

`pitch_m` is the taxel pitch in metres, as (px, py) or a scalar, or any object
with a `.pitch` in metres. The two shipped profiles were fitted from bench
press recordings; each JSON's `source` block records how. A profile for a
different velostat is a JSON file of the same form.

The model
---------
    counts = min(adc_max, gain * a * p ** b),   p = F / taxel_area   (kPa)
    a      = k * taxel_area_mm2 ** s

A power law in taxel pressure, clipped where the ADC tops out. `b` is the
curvature that survives normalisation.
`a` grows with taxel area, because a bigger interdigitated taxel conducts more
at the same pressure; `s` is fitted across the pitches that were calibrated.

How the profiles were fitted
----------------------------
The calibration press was not uniform: a hand press through a foam layer, no
bump sheet, and the tactile image shows it. Nor does the pattern keep its
shape as force rises (low-force vs high-force log-patterns correlate 0-0.5), so
per-taxel pressure cannot be read off a fixed weighting. What survives is the
force balance:

  a  Frame by frame, the load cell's force must equal the sum of taxel
     pressures times area. Inverting the power law per taxel gives
     F = area * sum((c/a) ** (1/b)), which fixes `a` with no assumption about
     the pressure pattern.
  b  If the model is right, that `a` is the same at every load. b is chosen
     as the exponent at which it stops trending with force. Each taxel's own
     log-log slope is the obvious alternative and reads high (0.72 against
     0.55 for v3), because taxels joining a growing contact carry a rising
     share of the load; the flat-`a` value is the one that transfers to other
     pitches and to the dead-weight rig.

A non-parametric monotone curve fitted the same way did no better: the error
that remains is press-to-press (the same 2 mm square varies +-30% between
recordings), not the shape of the curve.

What a profile does not capture
-------------------------------
- Per-taxel gain spread. Non-uniform pressure and non-uniform velostat look
  identical to this rig, so the spread is not separable and is not modelled.
- The bump sheet. Calibration pressed through foam onto the velostat; under a
  TPU bump the local pressure within one taxel is uneven, and with b < 1 a
  taxel-mean pressure overestimates the counts.
- Anything above `calibrated_kpa`. A sphere on a bump concentrates force far
  beyond what a flat press over a whole square reaches; `extrapolating()`
  says which taxels are outside the fit.
- Ghosting and scan timing. Those are topology, not the material.
"""
from dataclasses import dataclass, field
import glob
import json
import os

import numpy as np

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profiles")

MM2 = 1e-6


def available_profiles():
    """{label: material} for the profiles shipped with the package."""
    out = {}
    for path in sorted(glob.glob(os.path.join(PROFILE_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        out[d["name"]] = d.get("material", "")
    return out


def _find_profile(query):
    """A label ("v3"), a material ("Yunjia Textiles"), or its first word ("yunjia").

    The labels are the names the calibration run gave each sheet; the
    material is what others can buy, so both are accepted.
    """
    q = query.strip().lower()
    for label, material in available_profiles().items():
        m = material.lower()
        if q in (label.lower(), m) or (m and q == m.split()[0]):
            return os.path.join(PROFILE_DIR, f"{label}.json")
    return None


def _taxel_area_mm2(pitch):
    """Accept a TactileSensor, a (px, py) pitch in metres, or a scalar in metres.

    Metres because `TactileSensor.pitch` is SI. A pitch passed in mm would give
    an area 1e6 too large and every taxel would saturate, which looks like a
    velostat problem rather than a units one -- so refuse anything that is not
    plausibly metres.
    """
    pitch = getattr(pitch, "pitch", pitch)
    px, py = (pitch, pitch) if np.isscalar(pitch) else pitch
    if not (1e-4 <= px <= 2e-2 and 1e-4 <= py <= 2e-2):
        raise ValueError(f"taxel pitch {px}, {py} is not in metres "
                         "(expected 0.1-20 mm, given as 1e-4-2e-2)")
    return px * py / MM2


@dataclass(frozen=True)
class Readout:
    name: str
    material: str                   # what the sheet is; the name is a bench label
    b: float                        # pressure exponent
    k: float                        # a = k * area_mm2 ** s, counts / kPa^b
    s: float                        # area exponent of the gain
    adc_max: float                  # counts where the board's ADC tops out
    noise_counts: float             # per-taxel std at rest
    gain_range: tuple = (1.0, 1.0)  # multiplicative spread across presses + hysteresis
    calibrated_kpa: tuple = (0.0, np.inf)
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite([self.b, self.k, self.adc_max]).all() or min(self.b, self.k, self.adc_max) <= 0:
            raise ValueError("Readout exponent, gain coefficient and ADC ceiling must be positive and finite")
        if not np.isfinite([self.s, self.noise_counts]).all() or self.noise_counts < 0:
            raise ValueError("Readout area exponent and nonnegative noise must be finite")

    @staticmethod
    def _gain(gain):
        gain = np.asarray(gain, dtype=float)
        if not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError("Readout gain must be positive and finite")
        return gain

    # -- construction ----------------------------------------------------

    @classmethod
    def from_profile(cls, name_or_path):
        """A shipped profile by name (see `available_profiles()`), or a JSON path.

        A bare name (no directory, no .json) refers to a shipped profile even
        when a file or folder of that name exists in the working directory.
        """
        bare = os.path.basename(name_or_path) == name_or_path and not name_or_path.endswith(".json")
        path = ((_find_profile(name_or_path) if bare else None)
                or (name_or_path if os.path.isfile(name_or_path) else None))
        if path is None:
            shipped = ", ".join(f"{k} ({v})" for k, v in available_profiles().items())
            raise FileNotFoundError(
                f"no readout profile '{name_or_path}'. Shipped: {shipped or 'none'}; "
                "or pass the path to a profile JSON.")
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        m = d["model"]
        return cls(name=d["name"], material=d.get("material", ""), b=m["b"], k=m["k"], s=m["s"],
                   adc_max=m["adc_max"], noise_counts=m["noise_counts"],
                   gain_range=tuple(m["gain_range"]),
                   calibrated_kpa=tuple(m["calibrated_kpa"]),
                   source=d.get("source", {}))

    # -- the model -------------------------------------------------------

    def a(self, pitch):
        """counts / kPa^b for one taxel of this pitch."""
        return self.k * _taxel_area_mm2(pitch) ** self.s

    def pressure_kpa(self, forces_n, pitch):
        forces = np.asarray(forces_n, dtype=float)
        if not np.isfinite(forces).all():
            raise ValueError("Taxel forces must be finite")
        return forces / _taxel_area_mm2(pitch) * 1e3

    def counts(self, forces_n, pitch, gain=1.0, rng=None, quantize=False):
        """Counts above baseline for per-taxel forces in newtons.

        `gain` is where a domain randomiser scales the whole board; draw it from
        `gain_range`. Pass `rng` (a numpy Generator) to add read noise.
        """
        p = np.clip(self.pressure_kpa(forces_n, pitch), 0.0, None)
        c = self._gain(gain) * self.a(pitch) * p ** self.b
        if rng is not None:
            c = c + rng.normal(0.0, self.noise_counts, size=c.shape)
        c = np.clip(c, 0.0, self.adc_max)
        return np.round(c) if quantize else c

    def forces(self, counts, pitch, gain=1.0):
        """The inverse: newtons per taxel from counts above baseline.

        Taxels at `adc_max` come back as a lower bound -- the ADC has stopped
        saying how hard they are pressed. Taxels under 3x the noise read zero.
        """
        c = np.asarray(counts, dtype=float)
        if not np.isfinite(c).all():
            raise ValueError("ADC counts must be finite")
        c = np.where(c > 3.0 * self.noise_counts, c, 0.0)
        p = (c / (self._gain(gain) * self.a(pitch))) ** (1.0 / self.b)
        return p * _taxel_area_mm2(pitch) * 1e-3

    def saturation_force_n(self, pitch, gain=1.0):
        """Force on one taxel at which it reads adc_max."""
        p = (self.adc_max / (self._gain(gain) * self.a(pitch))) ** (1.0 / self.b)
        return p * _taxel_area_mm2(pitch) * 1e-3

    def extrapolating(self, forces_n, pitch):
        """True for taxels loaded above anything the calibration measured.

        Zero load is not extrapolation; below the fit's floor the reading is
        within noise anyway.
        """
        return self.pressure_kpa(forces_n, pitch) > self.calibrated_kpa[1]

    def describe(self, pitch=None):
        lines = [
            f"readout '{self.name}' ({self.material or 'material not recorded'}): "
            f"counts = a * p^{self.b:.2f}, "
            f"a = {self.k:.3g} * area_mm2^{self.s:.2f}",
            f"  ADC max {self.adc_max:.0f}, noise {self.noise_counts:.1f} counts, "
            f"gain spread {self.gain_range[0]:.2f}-{self.gain_range[1]:.2f}",
            f"  calibrated for taxel pressure up to {self.calibrated_kpa[1]:.0f} kPa "
            "(flat press through foam, no bump sheet)",
        ]
        if pitch is not None:
            area = _taxel_area_mm2(pitch)
            lines.append(f"  at {area:.2f} mm2 taxels: a = {self.a(pitch):.1f}, saturates at "
                         f"{self.saturation_force_n(pitch):.3f} N per taxel")
        return "\n".join(lines)
