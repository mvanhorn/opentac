"""Move a reading from one skin to another: skin A's frame, as skin B would read it.

    from opentac import manifest
    from opentac.transfer import transfer

    a = manifest.load("skin_a.json")
    b = manifest.load("skin_b.json")
    frame_b = transfer(frame_a, a, b)          # frame_a in newtons per taxel

How it works. Taxel k of a skin reads the force on its pitch cell: a square of
side `pitch` centred on pixels[k]. So a frame in newtons is a force per cell,
or a pressure F_k / pitch^2 that is constant over the cell.

A -> B, in B's millimetres:

1. Place A on B. Rotate A's frame by `angle_deg` about `about` (default: the
   centre of A's taxel bounding box), then shift by `shift_mm`. That is where
   the object sat. `transfer()` by default puts the contact's centre of
   pressure on the centroid of B's taxels (`Transfer.centered`).
2. Resample by area. Each A cell is cut into s x s sub-cells carrying equal
   shares of F_k ("box"), or spread over a 2-pitch tent ("tent", smoother when
   B is finer than A). Each share lands in the B cell that contains it. This is
   one fixed matrix per placement, F_B = M @ F_A. Every share that lands on a B
   taxel is kept, so force is conserved wherever B has taxels, at any pitch
   ratio.
3. Shares that land outside B's taxels (off the outline, or in a hole) are
   dropped: B cannot measure them. `Transfer.lost()` says how many newtons.

What this cannot do. Going to a finer B (4 mm -> 2 mm) invents nothing: detail
below A's pitch is not there, and "tent" only smooths it. Saturated taxels on A
are lower bounds, and they stay lower bounds after transfer.

Ported from the lab script `opentac_transfer.py` (Layout, Transfer, cop,
placement_cop). Layout reads an OpenTac manifest instead of a calibrated skin
object; the arithmetic is unchanged.
"""
from __future__ import annotations

import numpy as np

from opentac.frame import Frame
from opentac.manifest import Manifest, ManifestError, load


def pixel_pitch_mm(manifest: Manifest) -> tuple[float, float]:
    """Taxel pitch (x, y) in mm, from the spacing of `pixels`.

    The smallest gap between distinct x (and y) coordinates, rounded to 1 um.
    Not from the manifest's px/py: those are scan-order fields and py can be
    negative.
    """
    out = []
    for axis, values in (("x", manifest.xy[:, 0]), ("y", manifest.xy[:, 1])):
        v = np.unique(np.round(values, 6))
        if len(v) < 2:
            raise ManifestError(f"cannot derive {axis} pitch: only one distinct {axis} in pixels")
        out.append(round(float(np.diff(v).min()) * 1000.0) / 1000.0)
    return out[0], out[1]


def square_pitch_mm(manifest: Manifest) -> float:
    """The one taxel pitch of a skin with square taxel spacing, in mm."""
    px, py = pixel_pitch_mm(manifest)
    if px != py:
        raise ManifestError(f"taxel pitch differs in x and y ({px} / {py} mm); "
                            "only square taxel spacing is supported")
    return px


def _as_manifest(m) -> Manifest:
    return m if isinstance(m, Manifest) else load(m)


class Layout:
    """Where a skin's taxels are: centres, pitch, and a lattice lookup for binning."""

    def __init__(self, manifest):
        m = _as_manifest(manifest)
        self.manifest = m
        self._index(np.asarray(m.xy, dtype=float), square_pitch_mm(m))

    @classmethod
    def from_xy(cls, xy, pitch_mm: float) -> "Layout":
        """A Layout from taxel centres (mm) and a square pitch, without a manifest."""
        lay = cls.__new__(cls)
        lay.manifest = None
        lay._index(np.asarray(xy, dtype=float), float(pitch_mm))
        return lay

    def _index(self, xy, pitch):
        self.xy = xy
        self.pitch = pitch
        self.n = len(self.xy)
        self.ref = self.xy[0]
        ij = np.round((self.xy - self.ref) / self.pitch).astype(int)
        if np.abs((self.xy - self.ref) / self.pitch - ij).max() > 1e-3:
            raise ValueError("taxels are not on one regular lattice")
        self.lookup = {tuple(k): i for i, k in enumerate(ij)}
        self.ij0 = ij.min(0)
        self.grid = np.full(ij.max(0) - self.ij0 + 1, -1)
        self.grid[tuple((ij - self.ij0).T)] = np.arange(self.n)
        self.centre = (self.xy.min(0) + self.xy.max(0)) / 2
        self.centroid = self.xy.mean(0)          # equal cells, so this is the area centroid

    def index_of(self, pts):
        """Taxel index of the cell containing each point, -1 where there is no taxel."""
        ij = np.round((np.asarray(pts) - self.ref) / self.pitch).astype(int) - self.ij0
        inside = (ij >= 0).all(1) & (ij < self.grid.shape).all(1)
        out = np.full(len(ij), -1)
        out[inside] = self.grid[tuple(ij[inside].T)]
        return out


def _as_layout(x) -> Layout:
    return x if isinstance(x, Layout) else Layout(x)


class Transfer:
    """Linear force-conserving map from A's taxels to B's for one placement.

    `a` and `b` are Layouts, Manifests, or anything `opentac.manifest.load`
    accepts. `M[j, k]` is the share of A taxel k's force that lands on B
    taxel j.
    """

    def __init__(self, a, b, angle_deg=0.0, shift_mm=(0.0, 0.0), about=None, kernel="box", sub=None):
        a, b = _as_layout(a), _as_layout(b)
        self.a, self.b = a, b
        s = sub or max(4, int(np.ceil(4 * a.pitch / b.pitch)))
        if kernel == "box":
            u = (np.arange(s) + 0.5) / s - 0.5                      # sub-cell centres, in pitches
            du, dv = np.meshgrid(u, u)
            w = np.full(du.size, 1.0 / du.size)
        elif kernel == "tent":
            u = (np.arange(2 * s) + 0.5) / s - 1.0
            du, dv = np.meshgrid(u, u)
            w = ((1 - np.abs(du)) * (1 - np.abs(dv))).ravel()
            w /= w.sum()
        else:
            raise ValueError("kernel is 'box' or 'tent'")
        offs = np.stack([du.ravel(), dv.ravel()], 1) * a.pitch
        pts = a.xy[:, None, :] + offs[None]                          # nA x m x 2, A mm
        about = a.centre if about is None else np.asarray(about, float)
        th = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts_b = (pts - about) @ rot.T + about + np.asarray(shift_mm, float)
        idx = b.index_of(pts_b.reshape(-1, 2)).reshape(a.n, -1)
        self.M = np.zeros((b.n, a.n))
        cols = np.repeat(np.arange(a.n), idx.shape[1])
        keep = idx.ravel() >= 0
        np.add.at(self.M, (idx.ravel()[keep], cols[keep]), np.tile(w, a.n)[keep])
        self.kept = self.M.sum(0)                                     # per A taxel, fraction landing on B
        self.angle_deg, self.shift_mm, self.about = angle_deg, np.asarray(shift_mm, float), about

    @classmethod
    def centered(cls, a, b, fa_ref, angle_deg=0.0, target=None, cop_floor=0.1, **kw):
        """Placement putting the centre of pressure of fa_ref (A's N) on `target` (default B's centroid)."""
        a, b = _as_layout(a), _as_layout(b)
        c = placement_cop(a.xy, fa_ref, cop_floor)[0]
        if not np.all(np.isfinite(c)):
            raise ValueError("no force in the reference frame; nothing to centre")
        target = b.centroid if target is None else np.asarray(target, float)
        return cls(a, b, angle_deg=angle_deg, shift_mm=target - c, about=c, **kw)

    def force(self, fa):
        """(..., nA) N -> (..., nB) N. Unread (NaN) A taxels count as 0."""
        with np.errstate(all="ignore"):     # macOS Accelerate raises spurious matmul warnings
            return np.nan_to_num(np.asarray(fa, float)) @ self.M.T

    def lost(self, fa):
        """N of A's frame that fell outside B's taxels."""
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.asarray(fa, float)) @ (1.0 - self.kept)


def cop(xy, f):
    """Centre of pressure (mm) of each frame."""
    f = np.nan_to_num(np.asarray(f, float))
    tot = f.sum(-1, keepdims=True)
    with np.errstate(all="ignore"):     # macOS Accelerate raises spurious matmul warnings
        return (f @ xy) / np.where(tot > 0, tot, np.nan)


def placement_cop(xy, f, floor=0.1):
    """Centre of pressure for placing a contact: only taxels >= floor x the frame's peak, so
    scattered untouched-taxel noise does not pull it (the transfer still moves all the force)."""
    f = np.nan_to_num(np.atleast_2d(np.asarray(f, float)))
    return cop(xy, np.where(f >= floor * f.max(1, keepdims=True), f, 0.0))


def transfer(frame: Frame, a, b, *, center=True, angle_deg=0.0, shift_mm=(0.0, 0.0),
             target=None, kernel="box", cop_floor=0.1) -> Frame:
    """Skin A's frame, re-expressed on skin B's taxels. Force is conserved.

    `frame` must be in newtons per taxel (units "N"): resampling by area adds
    forces, which is only right for forces. `a` and `b` are the two skins'
    manifests (a Manifest, a manifest.json path, a design .zip, or a dict).

    center=True (default): the frame's centre of pressure goes to `target`
    (default: the centroid of B's taxels), rotated by `angle_deg` about itself.
    center=False: A's millimetres are used as B's millimetres, rotated by
    `angle_deg` about the centre of A's bounding box and moved by `shift_mm`.

    The result has B's taxel count, units "N", and the input's `t`. `imu` is
    not carried over: it describes A's board. Build a `Transfer` yourself to
    reuse one placement for many frames or to get `lost()`.
    """
    if frame.units != "N":
        raise ValueError(f"transfer needs a frame in newtons (units 'N'), got {frame.units!r}; "
                         "convert counts to newtons with your skin's calibration first")
    la, lb = _as_layout(a), _as_layout(b)
    if len(frame.taxels) != la.n:
        raise ValueError(f"frame has {len(frame.taxels)} taxels, skin A has {la.n}")
    fa = frame.taxels
    if center:
        if np.nansum(fa) <= 0:
            return Frame(taxels=np.zeros(lb.n), t=frame.t, units="N")
        tr = Transfer.centered(la, lb, fa, angle_deg=angle_deg, target=target,
                               cop_floor=cop_floor, kernel=kernel)
    else:
        tr = Transfer(la, lb, angle_deg=angle_deg, shift_mm=shift_mm, kernel=kernel)
    return Frame(taxels=tr.force(fa), t=frame.t, units="N")
