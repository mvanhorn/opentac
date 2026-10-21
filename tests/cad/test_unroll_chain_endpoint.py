"""Tests for /unroll-mesh-chain called in-process (no HTTP client needed):
the click_regions extension (explicit per-click face sets, for curved
selections that coplanar click-growth cannot reproduce), its validation
errors, and the clicks-only path.
"""


import asyncio
import io
import json

import numpy as np
import trimesh
from fastapi import HTTPException, UploadFile
from shapely.geometry import Polygon

from opentac.cad.main import unroll_mesh_chain
from tests.cad.test_mesh_unroll import make_filleted_corner_patch, compute_face_normals


def call_chain(vertices, faces, click_face_indices, click_regions="", connection_overrides="",
               corners_first=False):
    stl_bytes = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(file_type="stl")
    upload = UploadFile(file=io.BytesIO(stl_bytes), filename="synthetic.stl")
    return asyncio.run(unroll_mesh_chain(
        file=upload,
        click_face_indices=click_face_indices,
        connection_overrides=connection_overrides,
        click_regions=click_regions,
        corners_first=corners_first,
        angle_tol_deg=2.0,
    ))


def build_corner_mesh():
    n_a, n_arc, n_b, n_x = 4, 10, 4, 6
    vertices, faces, face_at = make_filleted_corner_patch(
        panel_a_width=30.0, fillet_radius=5.0, panel_b_height=20.0,
        sweep_length=40.0, n_x=n_x, n_arc=n_arc, n_a=n_a, n_b=n_b)
    return vertices, faces, face_at, (n_a, n_arc, n_b, n_x)


def _assert_stl_keeps_face_order(vertices, faces):
    """trimesh preserves face order through an STL round trip, on which the
    click_face_indices contract depends. Asserted on this synthetic mesh
    rather than assumed.
    """
    stl_bytes = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(file_type="stl")
    reloaded = trimesh.load(io.BytesIO(stl_bytes), file_type="stl")
    assert len(reloaded.faces) == len(faces)
    orig_centroids = vertices[faces].mean(axis=1)
    new_centroids = np.array(reloaded.vertices)[np.array(reloaded.faces)].mean(axis=1)
    assert np.allclose(orig_centroids, new_centroids, atol=1e-5), \
        "STL round trip reordered faces — click indices would be meaningless"


def test_clicks_only_strut_chain():
    """The clicks-only contract: two clicks on the flat panels, regions grown
    server-side, joined by a strut across the fillet.
    """
    vertices, faces, face_at, (n_a, n_arc, n_b, n_x) = build_corner_mesh()
    _assert_stl_keeps_face_order(vertices, faces)
    mid_xi = n_x // 2
    panel_a_seed = face_at(mid_xi, 0)
    panel_b_seed = face_at(mid_xi, n_a + n_arc)

    resp = call_chain(vertices, faces, f"{panel_a_seed},{panel_b_seed}")
    assert len(resp.connections) == 1
    assert resp.connections[0].needs_strut, "panels across a fillet should need a strut"
    assert resp.connections[0].loop_a and resp.connections[0].loop_b, \
        "strut connections must return corner-picker loops"
    poly = Polygon(resp.outline)
    assert poly.is_valid
    assert 1500 < poly.area < 2500, f"unexpected area {poly.area:.1f}mm²"
    assert len(resp.region_developability_mm) == 2
    assert all(d < 1e-6 for d in resp.region_developability_mm), \
        "flat panels should each be exactly developable"


def test_click_regions_selects_curved_patch():
    """The click_regions path: the middle click's region is the whole fillet
    band plus panel B, passed explicitly, a curved selection that
    grow_coplanar_region cannot produce (its normals rotate continuously away
    from any seed).
    Panel A (explicit too) touches the fillet directly, so the chain merges
    by union and unfolds as one patch, no strut.
    """
    vertices, faces, face_at, (n_a, n_arc, n_b, n_x) = build_corner_mesh()
    normals = compute_face_normals(vertices, faces)
    mid_xi = n_x // 2
    panel_a_seed = face_at(mid_xi, 0)
    curved_seed = face_at(mid_xi, n_a + n_arc // 2)

    panel_a = [face_at(xi, pi, w) for xi in range(n_x) for pi in range(n_a) for w in (0, 1)]
    curved = [face_at(xi, pi, w) for xi in range(n_x)
              for pi in range(n_a, n_a + n_arc + n_b) for w in (0, 1)]
    # Confirm the premise: coplanar growth from the curved seed stays tiny.
    from opentac.cad.pipeline.mesh_unroll import grow_coplanar_region
    grown = grow_coplanar_region(faces, normals, curved_seed, 2.0, vertices=vertices)
    assert len(grown) < len(curved) / 2, \
        "coplanar growth unexpectedly captured the curved band — premise broken"

    resp = call_chain(vertices, faces, f"{panel_a_seed},{curved_seed}",
                      click_regions=json.dumps([panel_a, curved]))
    assert len(resp.connections) == 1
    assert not resp.connections[0].needs_strut, \
        "panel A touches the fillet band — should merge by union, not strut"
    assert resp.region_face_count == len(set(panel_a) | set(curved))
    poly = Polygon(resp.outline)
    assert poly.is_valid
    # The full patch is a true generalized cylinder: A (30) + arc (~7.85) +
    # B (20) by 40mm sweep ≈ 2314mm², and the union path unfolds the real
    # fillet instead of cutting it, so expect close to the true area.
    assert 2100 < poly.area < 2500, f"unexpected area {poly.area:.1f}mm²"
    assert all(d < 1e-6 for d in resp.region_developability_mm), \
        "both regions are developable (flat panel; swept fillet band)"


def test_click_regions_validation_errors():
    vertices, faces, face_at, (n_a, n_arc, n_b, n_x) = build_corner_mesh()
    panel_a_seed = face_at(0, 0)
    panel_b_seed = face_at(0, n_a + n_arc)
    panel_a = [face_at(xi, pi, w) for xi in range(n_x) for pi in range(n_a) for w in (0, 1)]
    panel_b = [face_at(xi, pi, w) for xi in range(n_x)
               for pi in range(n_a + n_arc, n_a + n_arc + n_b) for w in (0, 1)]

    def expect_400(click_regions, why):
        try:
            call_chain(vertices, faces, f"{panel_a_seed},{panel_b_seed}", click_regions=click_regions)
        except HTTPException as e:
            assert e.status_code == 400, f"{why}: expected 400, got {e.status_code}"
            return e.detail
        raise AssertionError(f"{why}: expected HTTP 400, request succeeded")

    d1 = expect_400("not json", "malformed JSON")
    d2 = expect_400(json.dumps([panel_a]), "wrong list length")
    d3 = expect_400(json.dumps([panel_a, [f for f in panel_b if f != panel_b_seed]]),
                    "click face missing from its region")
    # Disconnected region: panel B plus one far-away stray face from panel A.
    d4 = expect_400(json.dumps([panel_a, panel_b + [face_at(0, 0)]]),
                    "region with a detached face")
    assert "not valid JSON" in d1 and "per click" in d2
    assert "not inside" in d3 and "connected" in d4


def test_corners_first_walks_cut_by_cut():
    """corners_first=True never runs the automatic corner heuristic:
    an unresolved strut connection comes back as pending_connection with its
    marker loops and no outline; supplying corners picked from those markers
    resolves it and completes the chain.
    """
    vertices, faces, face_at, (n_a, n_arc, n_b, n_x) = build_corner_mesh()
    mid_xi = n_x // 2
    panel_a_seed = face_at(mid_xi, 0)
    panel_b_seed = face_at(mid_xi, n_a + n_arc)
    clicks = f"{panel_a_seed},{panel_b_seed}"

    r1 = call_chain(vertices, faces, clicks, corners_first=True)
    assert r1.pending_connection == 0
    assert r1.outline == []
    conn = r1.connections[-1]
    assert conn.needs_strut and conn.loop_a and conn.loop_b, \
        "the pending connection must carry both marker loops"

    # Pick like a person would: the closest marker pair, plus the pair at
    # the other end of the two facing edges (max separation from pair 1).
    la = np.array(conn.loop_a); lb = np.array(conn.loop_b)
    d = np.linalg.norm(la[:, None, 1:4] - lb[None, :, 1:4], axis=2)
    i1, j1 = np.unravel_index(np.argmin(d), d.shape)
    i2 = int(np.argmax(np.linalg.norm(la[:, 1:4] - la[i1, 1:4], axis=1)))
    j2 = int(np.argmin(np.linalg.norm(lb[:, 1:4] - la[i2, 1:4], axis=1)))
    overrides = json.dumps({0: [[int(la[i1, 0]), int(lb[j1, 0])], [int(la[i2, 0]), int(lb[j2, 0])]]})

    r2 = call_chain(vertices, faces, clicks, connection_overrides=overrides, corners_first=True)
    assert r2.pending_connection is None
    poly = Polygon(r2.outline)
    assert poly.is_valid and 1500 < poly.area < 2500
    assert any("corners you picked" in w for w in r2.warnings)


def test_doubly_curved_region_warns_per_surface():
    from tests.cad.test_mesh_unroll import make_sphere_patch
    vertices, faces = make_sphere_patch(
        radius=20.0, n_theta=8, n_phi=8,
        theta_range=(np.pi / 4, np.pi / 2), phi_range=(0, np.pi / 2))
    all_faces = list(range(len(faces)))
    resp = call_chain(vertices, faces, "0", click_regions=json.dumps([all_faces]))
    assert len(resp.region_developability_mm) == 1
    assert resp.region_developability_mm[0] > 1.0, "sphere patch should be flagged"
    assert any("Surface 1" in w and "flattenable" in w for w in resp.warnings), \
        f"expected a per-surface warning, got {resp.warnings}"
