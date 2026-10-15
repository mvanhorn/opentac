"""Bump-sheet model of a flat OpenTac sensor (ported from TactileSim).

Each bump of the TPU bump sheet is one rigid body on a 5-DoF compliant joint
(press, shear x2, tip x2). Shear is never measured directly: a tipped bump
splits its load unevenly over the 2x2 taxels beneath it, and `decode_shear`
reads that split back.

Two ways to get a reading:

- `TactileSensor` (loader.py) evaluates the bumps with an analytic Winkler
  surrogate (surrogate.py). No MuJoCo needed; microseconds per reading.
- `SensorScene` (scene.py) builds the full MuJoCo scene for the same board
  and presses a ball into it. Needs `mujoco` (the `sim` extra).

`Lattice` (lattice.py) and `Gripper` (grip.py) are the regular-lattice studies
the model was validated with. Comments that cite NOTES.md or "Stage N" refer to
TactileSim's research notes and scripts, which are not part of this package.

Flat sensors only: a manifest with a `sim3d` block (curved) is refused.
"""
from .loader import NO_BUMP, ManifestError, TactileSensor
from .params import ESTIMATE_RANGES, BumpParams
from .surrogate import Surrogate

__all__ = ["BumpParams", "ESTIMATE_RANGES", "ManifestError", "NO_BUMP",
           "Surrogate", "TactileSensor", "Lattice", "Gripper", "SensorScene"]


def __getattr__(name):
    # These import mujoco; load them only when asked, so the surrogate path
    # works without the `sim` extra installed.
    if name == "Lattice":
        from .lattice import Lattice
        return Lattice
    if name == "Gripper":
        from .grip import Gripper
        return Gripper
    if name == "SensorScene":
        from .scene import SensorScene
        return SensorScene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
