"""manifest v6 must place the REAL exported board back onto the real part.

test_sim_embed.py checks the flat<->3D map in isolation. This is the round trip
that matters: run an actual /generate on an outline that came off a curved
part, then take the taxels the exporter wrote into the manifest and put them
back on the mesh. Every one has to land on the surface, in the right place, at
the right spacing.

Same discipline as test_manifest_reproduces_board.py -- check the manifest
against the ACTUAL export rather than against a re-derivation, because a bug
that lives in the export path is invisible to a test that recomputes.

The half-cylinder is used because it is developable (so the unfold is exact and
any error is the export's own) and because every quantity has a closed form.
"""
import base64
import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest


from opentac.cad.export.sim_pack import (  # noqa: E402
    FRAME_TOL_MM, build_sim3d, mesh_sha256, stack_from_bump_params,
)
from opentac.cad.main import GenerateRequest, generate  # noqa: E402
from opentac.cad.pipeline.bump_sheet import BumpParams  # noqa: E402
from opentac.cad.pipeline.sim_embed import build_embedding  # noqa: E402

R, H = 14.0, 26.0
NT, NZ = 56, 16
DTHETA = np.pi / NT
SAGITTA = R * (1.0 - np.cos(DTHETA / 2.0))
# Positions are rounded on the way into the JSON, so an exact-surface check
# has to allow the quantum back: 4 decimals in the manifest, 5 in the
# embedding, each spread over the coordinates of a point.
ROUND_MM = 1e-4
EMB_ROUND_MM = 2e-5
IDENTITY_TF = (np.eye(2), np.zeros(2), False)


def half_cylinder():
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
def exported():
    """Unfold the part, generate a real board from it, build the v6 block."""
    V, F, N = half_cylinder()
    region = list(range(len(F)))

    # The flat frame the unfold produces, and a rectangle inside it -- this
    # stands in for the outline the frontend receives from /unroll-mesh and
    # posts to /generate.
    flat = build_embedding(V, F, N, region, 0, IDENTITY_TF)
    lo, hi = flat.uv.min(axis=0), flat.uv.max(axis=0)
    m = 1.0
    outline = [[lo[0] + m, lo[1] + m], [hi[0] - m, lo[1] + m],
               [hi[0] - m, hi[1] - m], [lo[0] + m, hi[1] - m]]

    resp = generate(GenerateRequest(
        outline=outline, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    man = json.loads(z.read("manifest.json"))

    bp = BumpParams()
    sim3d, embedding = build_sim3d(
        vertices=V, faces=F, face_normals=N,
        region_face_indices=region, start_face=0,
        source_outline_xy=outline,
        manifest_outline_xy=man.get("outline"),
        pixel_xy=man["pixels"], shear=man.get("shear"),
        bump_geom={"base_dia_mm": [3.36, 3.36], "top_dia_mm": [2.35, 2.35]},
        stack=stack_from_bump_params(bp),
        sha256=mesh_sha256(b"not-a-real-stl"), filename="half_cylinder.stl")
    return man, sim3d, embedding


def test_frame_is_verified_not_assumed(exported):
    """The outline we projected with must be the one that made the board."""
    _, sim3d, _ = exported
    assert sim3d["fit"]["outline_mm"] < FRAME_TOL_MM
    assert "warnings" not in sim3d, sim3d.get("warnings")


def test_every_taxel_lands_on_the_part_surface(exported):
    """The round trip INTEGRATION.md asked for.

    A sensor placed a millimetre inside the finger still simulates happily and
    still looks fine in a viewer, so this is the assertion that has to exist.
    """
    man, sim3d, _ = exported
    pos = np.asarray(sim3d["taxels3d"]["pos"], dtype=float)
    assert len(pos) == len(man["pixels"])

    radius = np.linalg.norm(pos[:, :2], axis=1)
    # SAGITTA is the mesh's own faceting; ROUND_MM is the manifest storing
    # positions to 4 decimals like every other coordinate in the file.
    assert radius.max() <= R + ROUND_MM, "taxel projected outside the part"
    assert (R - radius).max() <= SAGITTA + ROUND_MM, "taxel projected inside the part"
    assert (pos[:, 2] >= -ROUND_MM).all() and (pos[:, 2] <= H + ROUND_MM).all()
    assert sim3d["fit"]["worst_off_patch_mm"] == 0.0


def test_taxels3d_is_index_parallel_to_pixels(exported):
    """The manifest promises it; a consumer will rely on it silently."""
    man, sim3d, _ = exported
    assert sim3d["taxels3d"]["rc"] == [[p[2], p[3]] for p in man["pixels"]]
    for key in ("pos", "normal", "tangent"):
        assert len(sim3d["taxels3d"][key]) == len(man["pixels"])


def test_normals_point_out_of_the_part(exported):
    _, sim3d, _ = exported
    pos = np.asarray(sim3d["taxels3d"]["pos"], dtype=float)
    nrm = np.asarray(sim3d["taxels3d"]["normal"], dtype=float)
    radial = np.c_[pos[:, 0], pos[:, 1], np.zeros(len(pos))]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", nrm, radial).min() > 0.99


def test_tangents_are_orthonormal(exported):
    _, sim3d, _ = exported
    nrm = np.asarray(sim3d["taxels3d"]["normal"], dtype=float)
    tan = np.asarray(sim3d["taxels3d"]["tangent"], dtype=float)
    assert np.abs(np.linalg.norm(tan, axis=1) - 1.0).max() < 1e-5
    assert np.abs(np.einsum("ij,ij->i", tan, nrm)).max() < 1e-5


def test_pitch_survives_the_wrap(exported):
    """4.2 mm drawn flat must be 4.2 mm of arc on the finger.

    This is the claim that makes the whole export worth having -- and the one
    a user would never think to check.
    """
    man, sim3d, _ = exported
    pos = np.asarray(sim3d["taxels3d"]["pos"], dtype=float)
    rc = np.asarray(sim3d["taxels3d"]["rc"], dtype=int)
    flat = np.asarray([[p[0], p[1]] for p in man["pixels"]], dtype=float)

    by_rc = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
    pairs = [(i, by_rc[(r, c + 1)]) for (r, c), i in by_rc.items()
             if (r, c + 1) in by_rc]
    assert len(pairs) > 10, "need neighbouring taxels to compare"

    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    dth = np.abs(np.arctan2(pos[b, 1], pos[b, 0]) - np.arctan2(pos[a, 1], pos[a, 0]))
    geodesic = np.hypot(R * dth, pos[b, 2] - pos[a, 2])
    flat_d = np.linalg.norm(flat[b] - flat[a], axis=1)
    assert np.abs(geodesic - flat_d).max() < 0.01


def test_stack_matches_bump_sheet_layer_table(exported):
    """The contact surface offset is what a sim collides against."""
    _, sim3d, _ = exported
    bp = BumpParams()
    st = sim3d["stack"]
    assert st["surface_to_sheet_top_mm"] == pytest.approx(
        bp.recess_depth_mm + bp.sheet_thickness_mm)
    assert st["surface_to_bump_tip_mm"] == pytest.approx(
        bp.recess_depth_mm + bp.sheet_thickness_mm + bp.bump_height_mm)


def test_bumps_are_placed_when_the_shear_block_exists(exported):
    man, sim3d, _ = exported
    if not man.get("shear", {}).get("bumps"):
        pytest.skip("no shear block on this board")
    assert len(sim3d["bumps3d"]["pos"]) == len(man["shear"]["bumps"])
    pos = np.asarray(sim3d["bumps3d"]["pos"], dtype=float)
    radius = np.linalg.norm(pos[:, :2], axis=1)
    assert (R - radius).max() <= SAGITTA + ROUND_MM


def test_embedding_is_self_contained(exported):
    """xyz per vertex, not vertex indices -- so a re-export cannot break it."""
    _, _, emb = exported
    tris = np.asarray(emb["tris"], dtype=int)
    uv = np.asarray(emb["uv"], dtype=float)
    xyz = np.asarray(emb["xyz"], dtype=float)
    assert tris.max() < len(uv) == len(xyz)
    assert np.isfinite(uv).all() and np.isfinite(xyz).all()
    assert np.abs(np.linalg.norm(xyz[:, :2], axis=1) - R).max() < EMB_ROUND_MM


@pytest.mark.parametrize("label,mangle,expect", [
    # A different vertex 0->1 is a different CABLE EDGE, so /generate would
    # have chosen a different canonical frame. Caught by the outline check.
    ("cable edge moved", lambda o: o[1:] + o[:1], "outline_mm"),
    ("mirrored", lambda o: [[-x, y] for x, y in o], "outline_mm"),
    # cable_transform is invariant to a global rigid motion, so these two look
    # perfect to the outline check -- but the outline is then in a different
    # frame from the unfold's uv, and the taxels land off the patch.
    ("frame rotated 90deg", lambda o: [[y, -x] for x, y in o], "worst_off_patch_mm"),
    ("frame translated", lambda o: [[x + 50, y + 30] for x, y in o], "worst_off_patch_mm"),
])
def test_a_wrong_outline_is_caught_not_silently_projected(label, mangle, expect):
    """The failure this whole design exists to prevent.

    Hand the exporter an outline that did not produce this board and it has to
    say so. Nothing downstream could notice: in every case below the taxels
    still land perfectly ON the cylinder's surface -- they are simply on the
    wrong part of the finger. Being on the surface is not evidence of being in
    the right place, which is why both checks exist.

    The two checks are complementary, and the parametrisation records which one
    does the work for each kind of mistake.
    """
    V, F, N = half_cylinder()
    region = list(range(len(F)))
    flat = build_embedding(V, F, N, region, 0, IDENTITY_TF)
    lo, hi = flat.uv.min(axis=0), flat.uv.max(axis=0)
    m = 1.0
    outline = [[lo[0] + m, lo[1] + m], [hi[0] - m, lo[1] + m],
               [hi[0] - m, hi[1] - m], [lo[0] + m, hi[1] - m]]

    resp = generate(GenerateRequest(
        outline=outline, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    man = json.loads(z.read("manifest.json"))

    sim3d, _ = build_sim3d(
        vertices=V, faces=F, face_normals=N,
        region_face_indices=region, start_face=0,
        source_outline_xy=mangle(outline),
        manifest_outline_xy=man.get("outline"),
        pixel_xy=man["pixels"], stack=stack_from_bump_params(BumpParams()))

    assert sim3d["fit"][expect] > FRAME_TOL_MM, f"{label}: {sim3d['fit']}"
    assert sim3d.get("warnings"), f"{label} produced no warning"

    # And the point of the test: it looks fine on the surface regardless.
    pos = np.asarray(sim3d["taxels3d"]["pos"], dtype=float)
    radius = np.linalg.norm(pos[:, :2], axis=1)
    assert (R - radius).max() <= SAGITTA + ROUND_MM
