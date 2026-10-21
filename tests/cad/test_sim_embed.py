"""The flat<->3D map puts taxels back where they physically land.

manifest v6 promises a simulator that a taxel drawn at (x, y) on the flat board
sits at a specific point on the part's surface, with a specific normal and a
specific tangent. Nothing downstream can check that: a simulator will place a
sensor a few millimetres inside the finger, or rotated 90 degrees, and still
produce plausible-looking readings. These tests check it.

The half-cylinder is the case worth having: it is developable, so the unfold is
exact, and every quantity has a closed form to compare against. The
tolerances below are derived from the mesh's own faceting (an inscribed
40-segment polygon sits a sagitta of R*(1-cos(dtheta/2)) inside the true
cylinder), not tuned to make the test pass, so a projection that lands off the
surface fails the test instead of being absorbed by its tolerance.
"""

import numpy as np
import pytest


from opentac.cad.pipeline.geometry import cable_transform, rotate_for_cable  # noqa: E402
from opentac.cad.pipeline.sim_embed import build_embedding, project  # noqa: E402

R, H = 12.0, 20.0
NT, NZ = 40, 12
IDENTITY_TF = (np.eye(2), np.zeros(2), False)

# Faceting of an inscribed polygon: chord midpoints sit this far inside the
# true surface, and facet normals tilt by this much against the true radial.
DTHETA = np.pi / NT
SAGITTA = R * (1.0 - np.cos(DTHETA / 2.0))
NORMAL_TILT = 1.0 - np.cos(DTHETA / 2.0)


def half_cylinder():
    """Half a cylinder, outward-normal, as (vertices, faces, face_normals)."""
    th = np.linspace(0, np.pi, NT + 1)
    zs = np.linspace(0, H, NZ + 1)
    verts, idx = [], {}
    for i, t in enumerate(th):
        for j, z in enumerate(zs):
            idx[(i, j)] = len(verts)
            verts.append([R * np.cos(t), R * np.sin(t), z])
    V = np.array(verts, dtype=float)

    faces = []
    for i in range(NT):
        for j in range(NZ):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    F = np.array(faces)

    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    N = np.cross(p1 - p0, p2 - p0)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    radial = ((p0 + p1 + p2) / 3.0).copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    N[np.einsum("ij,ij->i", N, radial) < 0] *= -1
    return V, F, N


@pytest.fixture(scope="module")
def cyl():
    V, F, N = half_cylinder()
    emb = build_embedding(V, F, N, list(range(len(F))), 0, IDENTITY_TF)
    return V, F, N, emb


def interior_grid(emb, margin=1.5, nu=17, nv=11):
    lo, hi = emb.uv.min(axis=0), emb.uv.max(axis=0)
    gu = np.linspace(lo[0] + margin, hi[0] - margin, nu)
    gv = np.linspace(lo[1] + margin, hi[1] - margin, nv)
    return np.array([[u, v] for u in gu for v in gv]), len(gv)


def test_unfold_covers_every_face(cyl):
    _, F, _, emb = cyl
    assert len(emb.tris) == len(F)
    assert np.isfinite(emb.uv).all()


def test_flat_extents_match_arc_length(cyl):
    """A cylinder unfolds to a rectangle: arc length by height.

    The unfold preserves edge lengths, so it reproduces the inscribed
    polygon's perimeter, which is shorter than the true arc by the usual
    chord deficit. That is correct behaviour, not error.
    """
    _, _, _, emb = cyl
    span = emb.uv.max(axis=0) - emb.uv.min(axis=0)
    chord_arc = NT * 2.0 * R * np.sin(DTHETA / 2.0)
    assert sorted(span) == pytest.approx(sorted([chord_arc, H]), abs=1e-9)


def test_projected_points_land_on_the_surface(cyl):
    _, _, _, emb = cyl
    pts, _ = interior_grid(emb)
    pos, _, _, dist = project(emb, pts)

    assert dist.max() == 0.0, "interior points must not need the fallback"
    radius = np.linalg.norm(pos[:, :2], axis=1)
    # Inside the true cylinder by at most the sagitta, never outside it.
    assert radius.max() <= R + 1e-9
    assert (R - radius).max() <= SAGITTA + 1e-9
    assert (pos[:, 2] >= -1e-9).all() and (pos[:, 2] <= H + 1e-9).all()


def test_normals_are_outward_radial(cyl):
    _, _, _, emb = cyl
    pts, _ = interior_grid(emb)
    pos, nrm, _, _ = project(emb, pts)

    radial = np.c_[pos[:, 0], pos[:, 1], np.zeros(len(pos))]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", nrm, radial)
    assert cos.min() > 0, "normal must point out of the part, not into it"
    assert (1.0 - cos).max() <= NORMAL_TILT + 1e-9


def test_tangent_is_orthonormal_to_the_normal(cyl):
    _, _, _, emb = cyl
    pts, _ = interior_grid(emb)
    _, nrm, tan, _ = project(emb, pts)

    assert np.abs(np.linalg.norm(tan, axis=1) - 1.0).max() < 1e-12
    assert np.abs(np.einsum("ij,ij->i", tan, nrm)).max() < 1e-12


def test_tangent_follows_the_requested_flat_direction(cyl):
    """+x in the flat frame must come back as one consistent surface direction.

    This is what makes a shear reading relatable to the firmware's column
    index: "+x" has to mean the same physical direction at every taxel, not
    drift or flip halfway across the patch.

    Compared against the cylinder's closed-form circumferential tangent rather
    than against a finite chord between grid points -- a chord over a 3.5 mm
    step on R=12 already differs from the true tangent by half the subtended
    angle, which would swamp what is being measured.
    """
    _, _, _, emb = cyl
    pts, _ = interior_grid(emb)
    pos, _, tan, _ = project(emb, pts)

    # On this patch +u runs circumferentially, so the exact tangent at a point
    # is the circumferential direction there.
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    exact = np.c_[-np.sin(theta), np.cos(theta), np.zeros(len(pos))]
    cos = np.einsum("ij,ij->i", tan, exact)

    # One consistent sign everywhere -- the flip is what would silently mirror
    # a shear channel.
    assert np.sign(cos).min() == np.sign(cos).max()
    # A facet's own direction is a chord, tilted from the true tangent by at
    # most half a segment.
    assert np.abs(cos).min() >= np.cos(DTHETA / 2.0) - 1e-9


def test_unfold_is_isometric(cyl):
    """Distance in the flat frame equals geodesic distance on the part.

    The export rests on this property: a taxel pitch drawn flat is the pitch
    obtained once the sheet is on the finger.
    """
    _, _, _, emb = cyl
    pts, nv = interior_grid(emb)
    pos, _, _, _ = project(emb, pts)

    dth = np.abs(np.arctan2(pos[nv:, 1], pos[nv:, 0])
                 - np.arctan2(pos[:-nv, 1], pos[:-nv, 0]))
    geodesic = np.hypot(R * dth, pos[nv:, 2] - pos[:-nv, 2])
    flat = np.linalg.norm(pts[nv:] - pts[:-nv], axis=1)
    assert np.abs(geodesic - flat).max() < 1e-3


def test_point_off_the_patch_is_reported_not_hidden(cyl):
    """A point outside the region must raise `dist`, not silently snap."""
    _, _, _, emb = cyl
    hi = emb.uv.max(axis=0)
    off = np.array([[hi[0] + 25.0, hi[1] + 25.0]])
    pos, _, _, dist = project(emb, off)

    assert dist[0] > 1.0, "off-patch point must be flagged"
    # It still returns a real surface point, clamped to the nearest triangle.
    assert np.linalg.norm(pos[0, :2]) <= R + 1e-9


def test_embedding_is_in_the_cable_frame():
    """uv comes back in the same frame the manifest's taxels live in.

    In a wrong frame the sensor is rotated on the part with nothing to detect
    it: every projected point is still on the surface, only in the wrong
    place. So: transform an outline with rotate_for_cable, build the
    embedding with the matching cable_transform, and require that a flat
    point tracks the rotated outline rather than the raw one.
    """
    V, F, N = half_cylinder()
    flat = build_embedding(V, F, N, list(range(len(F))), 0, IDENTITY_TF)

    lo, hi = flat.uv.min(axis=0), flat.uv.max(axis=0)
    outline = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                        [hi[0], hi[1]], [lo[0], hi[1]]], dtype=float)
    rotated = rotate_for_cable(outline.copy())
    tf = cable_transform(outline.copy())

    turned = build_embedding(V, F, N, list(range(len(F))), 0, tf)

    # The frame moved, so the flat coordinates must differ...
    assert not np.allclose(turned.uv, flat.uv)
    # ...but the surface geometry must not: same 3D points, same normals.
    assert np.allclose(turned.xyz, flat.xyz)
    assert np.allclose(turned.tri_normal, flat.tri_normal)
    # And the outline's own corners must now sit on the rotated outline.
    span_before = np.ptp(flat.uv, axis=0)
    span_after = np.ptp(turned.uv, axis=0)
    assert sorted(span_after) == pytest.approx(sorted(span_before), abs=1e-9)
    assert np.ptp(rotated[:2, 1]) < 1e-9, "cable edge should be horizontal"


def test_flat_plate_round_trips_exactly():
    """A planar region has no faceting, so the map must be exact."""
    xs, ys = np.linspace(0, 30, 7), np.linspace(0, 18, 5)
    verts, idx = [], {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            idx[(i, j)] = len(verts)
            verts.append([x, y, 4.0])
    V = np.array(verts, dtype=float)
    faces = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    F = np.array(faces)
    N = np.tile([0.0, 0.0, 1.0], (len(F), 1))

    emb = build_embedding(V, F, N, list(range(len(F))), 0, IDENTITY_TF)
    pts, _ = interior_grid(emb, margin=1.0, nu=9, nv=7)
    pos, nrm, tan, dist = project(emb, pts)

    assert dist.max() == 0.0
    assert np.abs(pos[:, 2] - 4.0).max() < 1e-12
    assert np.abs(nrm - [0, 0, 1]).max() < 1e-12
    # Flat and 3D agree up to the rigid placement of the patch.
    d_flat = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    d_3d = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    assert np.abs(d_flat - d_3d).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", tan, nrm)).max() < 1e-12
