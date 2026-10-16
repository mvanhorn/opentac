"""Simulate a flat OpenTac sensor: press an object into it, get a Frame.

    from opentac.sim import Sensor

    s = Sensor("manifest.json")                       # foam model
    frame = s.press("sphere", center_mm=(10, 12), force_n=5.0, radius_mm=5.0)
    frame.taxels                                      # N per taxel, manifest pixels order

Two models:

- "foam" (default): a skin with no bump sheet. Rigid indenter into a foam
  layer; normal force only. Validated against real presses in the paper.
- "mujoco": a skin with a TPU bump sheet. Rigid bumps on compliant joints,
  ported from TactileSim; normal force plus shear decoded from each bump's
  2x2 taxels. Not validated against hardware.

The manifest cannot tell the two apart: OpenTac-CAD writes a `shear` block
(where bumps WOULD go) on every export, printed sheet or not. So the default
is always "foam"; pass model="mujoco" if you print the bump sheet.

Frames come back in newtons ("N"), or in ADC counts if you give a `readout`.
See docs/sim.md.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from opentac.frame import Frame
from opentac.manifest import Manifest, ManifestError, load

from . import foam
from .readout import Readout, available_profiles

__all__ = ["Sensor", "FoamSensor", "MujocoSensor", "Readout", "available_profiles", "MODELS"]

MODELS = ("foam", "mujoco")

# Full-solver twist: the ball turns in steps of 1/3 degree, 26 solver steps
# each -- the rate TactileSim's stage-3 run used (14 degrees over 42 steps).
_TWIST_STEP_DEG = 14.0 / 42.0
_TWIST_SOLVER_STEPS = 26


def Sensor(manifest, model: str = "foam", readout=None, **params):
    """A simulated sensor for a flat manifest. Returns a FoamSensor or a MujocoSensor.

    Not to be confused with `opentac.Sensor` (`opentac.device.Sensor`), which
    opens a real readout board on USB. Both give `opentac.frame.Frame`s.

    `manifest` is a Manifest, a manifest.json path, a design export .zip, or a
    dict. `readout`, if given, turns frames into ADC counts: a profile name
    ("v2", "v3"), a profile JSON path, a `Readout`, or any object with a
    `counts(newtons)` method. `params` go to the model: `sigma_mm`,
    `k_n_per_mm3` for foam; BumpParams fields (`e_eff_mpa`, `mu`,
    `bump_height_mm`, ...) for mujoco.
    """
    m = manifest if isinstance(manifest, Manifest) else load(manifest)
    if (m.get("sim3d") or {}).get("taxels3d"):
        raise ManifestError("this sensor is mounted on a curved surface (the manifest has a "
                            "'sim3d' block); opentac.sim handles flat sensors only")
    if model == "foam":
        return FoamSensor(m, readout=readout, **params)
    if model == "mujoco":
        return MujocoSensor(m, readout=readout, **params)
    raise ValueError(f"model must be one of {MODELS}, not {model!r}")


def _resolve_readout(readout):
    if readout is None or isinstance(readout, Readout) or hasattr(readout, "counts"):
        return readout
    return Readout.from_profile(str(readout))


class _Base:
    model = ""

    def __init__(self, manifest: Manifest, readout=None):
        self.manifest = manifest
        self.readout = _resolve_readout(readout)

    @property
    def n(self) -> int:
        return self.manifest.n

    def _pitch_m(self):
        raise NotImplementedError

    def _frame(self, newtons) -> Frame:
        f = np.asarray(newtons, dtype=float)
        if self.readout is None:
            return Frame(taxels=f, t=0.0, units="N")
        if isinstance(self.readout, Readout):
            c = self.readout.counts(f, self._pitch_m())
        else:
            c = np.asarray(self.readout.counts(f), dtype=float)
        return Frame(taxels=c, t=0.0, units="counts")


class FoamSensor(_Base):
    """A skin with no bump sheet: rigid indenter into foam (see opentac.sim.foam)."""

    model = "foam"

    def __init__(self, manifest: Manifest, readout=None, sigma_mm=foam.SIGMA_MM,
                 k_n_per_mm3=foam.K_N_PER_MM3):
        from opentac.transfer import square_pitch_mm

        super().__init__(manifest, readout)
        self.pitch_mm = square_pitch_mm(manifest)
        self.sigma_mm = float(sigma_mm)
        self.k_n_per_mm3 = float(k_n_per_mm3)

    def _pitch_m(self):
        return (self.pitch_mm * 1e-3, self.pitch_mm * 1e-3)

    def press(self, shape="sphere", center_mm=None, force_n=5.0, angle_deg=0.0, *,
              radius_mm=5.0, length_mm=40.0) -> Frame:
        """Press an indenter in to a total force; one Frame.

        shape: "sphere" (radius_mm), "disc" (a flat punch of radius_mm), "bar"
        (a cylinder of radius_mm and length_mm lying on the skin, its axis at
        angle_deg from +x), a paper indenter name from `foam.INDENTERS`, or a
        geometry dict as `foam.indenter_height` takes. center_mm defaults to
        the centroid of the taxels. force_n is the load the skin carries; load
        that misses the taxels is not read, so the frame can sum to less.
        """
        center = self.manifest.xy.mean(0) if center_mm is None else np.asarray(center_mm, float)
        f = foam.taxel_forces(self.manifest.xy, self.pitch_mm, _foam_geom(shape, radius_mm, length_mm),
                              center, float(force_n), self.sigma_mm, angle_deg, self.k_n_per_mm3)
        return self._frame(f)


def _foam_geom(shape, radius_mm, length_mm):
    if isinstance(shape, dict):
        return shape
    if shape in foam.INDENTERS:
        return foam.INDENTERS[shape]
    if shape == "sphere":
        return {"kind": "sphere", "R": float(radius_mm)}
    if shape == "disc":
        return {"kind": "punch", "R": float(radius_mm)}
    if shape == "bar":
        return {"kind": "bar", "R": float(radius_mm), "L": float(length_mm)}
    raise ValueError(f"unknown foam indenter {shape!r}; expected sphere, disc, bar, "
                     f"one of {sorted(foam.INDENTERS)}, or a geometry dict")


class MujocoSensor(_Base):
    """A skin with a TPU bump sheet (see opentac.sim.mujoco). Flat sensors only.

    `tactile` is the underlying `opentac.sim.mujoco.TactileSensor`, for
    everything this wrapper does not expose. Coordinates in and out are mm;
    the bump model itself works in metres.
    """

    model = "mujoco"

    def __init__(self, manifest: Manifest, readout=None, **bump_params):
        from .mujoco import TactileSensor

        super().__init__(manifest, readout)
        ts = TactileSensor.from_manifest(manifest.raw)
        # set_params keeps the pitch read from the manifest; building a fresh
        # BumpParams would silently fall back to the 4.2 mm default.
        self.tactile = ts.set_params(**bump_params) if bump_params else ts

    def _pitch_m(self):
        return self.tactile.pitch

    def _centre_m(self, center_mm):
        if center_mm is None:
            return self.tactile.centre()
        return float(center_mm[0]) * 1e-3, float(center_mm[1]) * 1e-3

    def press(self, shape="sphere", center_mm=None, force_n=5.0, angle_deg=0.0, *,
              radius_mm=10.0, tilt_deg=(0.0, 0.0), solver="surrogate") -> Frame:
        """Press an object in to a total normal force; one Frame.

        shape: "sphere" (radius_mm), "cylinder" (radius_mm, axis along y) or
        "plane" (tilted by tilt_deg about y and x). center_mm defaults to a
        point on a bump near the middle of the board (between bumps a small
        ball can read nothing).

        solver="surrogate" (default) evaluates the bumps analytically, in
        microseconds. solver="mujoco" runs the full MuJoCo scene (sphere only;
        needs the `sim` extra) and takes about a second.
        """
        if angle_deg:
            raise ValueError("the mujoco model has no indenter angle; its cylinder runs along y")
        cx, cy = self._centre_m(center_mm)
        if solver == "surrogate":
            tilt = (math.radians(tilt_deg[0]), math.radians(tilt_deg[1]))
            f = self.tactile.probe(shape, force_n=force_n, radius_mm=radius_mm,
                                   cx_mm=cx * 1e3, cy_mm=cy * 1e3, tilt=tilt, as_image=False)
            return self._frame(f)
        scene = self._scene(shape, radius_mm, (cx, cy))
        self._press_scene(scene, force_n)
        return self._frame(scene.taxel_forces())

    def twist(self, yaw_deg, shape="sphere", center_mm=None, force_n=5.0, *,
              radius_mm=10.0, solver="surrogate") -> list[Frame]:
        """Press an object in, then turn it about the contact normal.

        One Frame per angle in `yaw_deg` (degrees, in order). Read the shear
        with `decode_shear`. Turning is continuous through the list, so stick,
        breakaway and slip appear in order.

        The surrogate holds the normal force constant while the object turns.
        The full solver holds the ball's HEIGHT, so its normal force can drift
        (12.3 N -> 16.6 N over 14 degrees on tests/sim/fixtures/flat_square.json
        at e_eff_mpa=3.5).
        """
        cx, cy = self._centre_m(center_mm)
        if solver == "surrogate":
            res = self.tactile.press_and_twist(list(yaw_deg), shape=shape, force_n=force_n,
                                               radius_mm=radius_mm, cx_mm=cx * 1e3, cy_mm=cy * 1e3)
            return [self._frame(r["forces"]) for r in res]
        scene = self._scene(shape, radius_mm, (cx, cy))
        z = self._press_scene(scene, force_n)
        out, prev = [], 0.0
        for a in yaw_deg:
            n = max(1, math.ceil(abs(a - prev) / _TWIST_STEP_DEG))
            for k in range(1, n + 1):
                scene.twist_to(z, math.radians(prev + (a - prev) * k / n), steps=_TWIST_SOLVER_STEPS)
            prev = a
            out.append(self._frame(scene.taxel_forces()))
        return out

    def decode_shear(self, frame: Frame) -> np.ndarray:
        """(n_bumps, 2) shear force per bump in N, from one frame in newtons.

        A bump the outline cut short cannot report shear along some axis;
        `degenerate_axes()` marks those components, which are noise.
        """
        if frame.units != "N":
            raise ValueError("decode_shear needs a frame in newtons; make the sensor without a readout")
        return self.tactile.decode_shear(np.nan_to_num(frame.taxels))

    def degenerate_axes(self) -> np.ndarray:
        """(n_bumps, 2) bool: True where a bump's group cannot read shear on that axis."""
        return self.tactile.degenerate_axes()

    @property
    def bump_xy_mm(self) -> np.ndarray:
        """(n_bumps, 2) bump centres, mm, in the order `decode_shear` returns."""
        return self.tactile.bump_xy * 1e3

    def _scene(self, shape, radius_mm, centre_m):
        if shape != "sphere":
            raise ValueError("solver='mujoco' presses a sphere only; use solver='surrogate'")
        from .mujoco import SensorScene

        return SensorScene(self.tactile, radius_mm * 1e-3, centre_xy=centre_m)

    @staticmethod
    def _press_scene(scene, force_n):
        z, total = scene.press_to_force(force_n)
        if total < 0.97 * force_n:
            warnings.warn(f"the ball ran out of travel at {total:.2f} N of {force_n:.2f} N",
                          RuntimeWarning, stacklevel=3)
        return z
