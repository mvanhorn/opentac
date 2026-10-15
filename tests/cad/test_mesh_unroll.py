"""Tests for triangle-unfolding mesh unroll — cylinder, hexagonal prism,
hexagonal-pyramid (angled facets, the case that broke an earlier axis-fit
approach), and a sphere patch as a negative (non-developable) control.
"""

import sys
import os

import numpy as np
from opentac.cad.pipeline.mesh_unroll import unfold_triangle_patch, check_developability, grow_coplanar_region


def make_cylinder_patch(radius, sweep_angle_deg, height, n_theta, n_z):
    thetas = np.linspace(0, np.radians(sweep_angle_deg), n_theta)
    zs = np.linspace(0, height, n_z)
    vertices = []
    idx = {}
    for zi, z in enumerate(zs):
        for ti, theta in enumerate(thetas):
            idx[(zi, ti)] = len(vertices)
            vertices.append([radius * np.cos(theta), radius * np.sin(theta), z])
    vertices = np.array(vertices)
    faces = []
    for zi in range(n_z - 1):
        for ti in range(n_theta - 1):
            v00, v01 = idx[(zi, ti)], idx[(zi, ti + 1)]
            v10, v11 = idx[(zi + 1, ti)], idx[(zi + 1, ti + 1)]
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    return vertices, np.array(faces)


def make_hex_prism_patch(radius, n_facets, height):
    """n_facets consecutive flat sides of a regular hexagonal prism."""
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pts = np.stack([radius * np.cos(hex_angles), radius * np.sin(hex_angles)], axis=1)
    vertices = []
    idx = {}
    for hi in range(n_facets + 1):
        p = hex_pts[hi % 6]
        for zi, z in enumerate([0.0, height]):
            idx[(hi, zi)] = len(vertices)
            vertices.append([p[0], p[1], z])
    vertices = np.array(vertices)
    faces = []
    for hi in range(n_facets):
        v00, v01 = idx[(hi, 0)], idx[(hi + 1, 0)]
        v10, v11 = idx[(hi, 1)], idx[(hi + 1, 1)]
        faces.append([v00, v01, v11])
        faces.append([v00, v11, v10])
    return vertices, np.array(faces)


def make_hex_pyramid_patch(base_radius, height, n_facets):
    """n_facets triangular lateral facets of a hexagonal pyramid, all
    converging to a shared apex — the angled/converging case that broke the
    earlier axis-fit approach.
    """
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pts = [[base_radius * np.cos(a), base_radius * np.sin(a), 0.0] for a in hex_angles]
    apex = [0.0, 0.0, height]
    vertices = np.array(hex_pts + [apex])
    apex_idx = 6
    faces = [[hi % 6, (hi + 1) % 6, apex_idx] for hi in range(n_facets)]
    return vertices, np.array(faces)


def make_sphere_patch(radius, n_theta, n_phi, theta_range, phi_range):
    thetas = np.linspace(*theta_range, n_theta)
    phis = np.linspace(*phi_range, n_phi)
    vertices = []
    idx = {}
    for ti, theta in enumerate(thetas):
        for pi, phi in enumerate(phis):
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            idx[(ti, pi)] = len(vertices)
            vertices.append([x, y, z])
    vertices = np.array(vertices)
    faces = []
    for ti in range(n_theta - 1):
        for pi in range(n_phi - 1):
            v00, v01 = idx[(ti, pi)], idx[(ti, pi + 1)]
            v10, v11 = idx[(ti + 1, pi)], idx[(ti + 1, pi + 1)]
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    return vertices, np.array(faces)


def make_filleted_corner_patch(panel_a_width, fillet_radius, panel_b_height,
                                sweep_length, n_x, n_arc, n_a=4, n_b=4):
    """Two flat panels — A in the z=0 plane, B in the y=fillet_radius plane
    — joined by a quarter-circle fillet, swept along x. Every cross-section
    (constant x) is the identical profile, so this whole patch is a true
    generalized cylinder (single-curvature, exactly developable), but the
    fillet is discretized into many thin, continuously-rotating triangles —
    exactly the case where grow_coplanar_region correctly refuses to walk
    from one flat panel into the other, requiring bridge_gap to connect a
    shift-click selection of both flats.

    Returns (vertices, faces, face_at) where face_at(xi, pi, which=0) gives
    the face index of the `which`-th triangle (0 or 1) of grid cell
    (xi, pi), and profile segments [0, n_a) are panel A, [n_a, n_a+n_arc)
    are the fillet, [n_a+n_arc, n_a+n_arc+n_b) are panel B.
    """
    profile = []
    for i in range(n_a, 0, -1):
        profile.append((-panel_a_width * i / n_a, 0.0))
    for i in range(n_arc + 1):
        theta = -np.pi / 2 + (np.pi / 2) * i / n_arc
        profile.append((fillet_radius * np.cos(theta),
                         fillet_radius + fillet_radius * np.sin(theta)))
    for i in range(1, n_b + 1):
        profile.append((fillet_radius, fillet_radius + panel_b_height * i / n_b))

    xs = np.linspace(0.0, sweep_length, n_x + 1)
    vertices = []
    idx = {}
    for xi, x in enumerate(xs):
        for pi, (y, z) in enumerate(profile):
            idx[(xi, pi)] = len(vertices)
            vertices.append([x, y, z])
    vertices = np.array(vertices)

    n_profile = len(profile)
    faces = []

    def face_at(xi, pi, which=0):
        return (xi * (n_profile - 1) + pi) * 2 + which

    for xi in range(n_x):
        for pi in range(n_profile - 1):
            v00, v01 = idx[(xi, pi)], idx[(xi, pi + 1)]
            v10, v11 = idx[(xi + 1, pi)], idx[(xi + 1, pi + 1)]
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    faces = np.array(faces)
    return vertices, faces, face_at


def compute_face_normals(vertices, faces):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return n


def test_filleted_corner_connects_with_strut():
    """The exact case reported (in various forms, across many real STLs):
    two flat panels joined by a rounded/chamfered corner don't naturally
    union into one connected selection, because the fillet's many thin,
    continuously-rotating triangles are correctly excluded from both flat
    regions by grow_coplanar_region's angle tolerance.

    `connect_regions_with_strut` sidesteps trying to select and flatten
    that connecting geometry (an approach that went through many iterations
    and kept finding new real-world failure modes — see NOTES.md) by just
    approximating it with straight cuts between the two regions' nearest
    corners. On this synthetic mesh the true fillet is exactly developable
    (zero distortion), so the straight-cut approximation necessarily loses
    some area (a chord is always shorter than its arc) — this test checks
    the result is a valid, reasonably-sized polygon in the right ballpark,
    not an exact geometric match (unlike the old bridge_gap tests, which
    could check for an exact match because they included the real curved
    geometry rather than approximating it away).
    """
    from shapely.geometry import Polygon
    from opentac.cad.pipeline.mesh_unroll import connect_regions_with_strut

    n_a, n_arc, n_b, n_x = 4, 10, 4, 6
    vertices, faces, face_at = make_filleted_corner_patch(
        panel_a_width=30.0, fillet_radius=5.0, panel_b_height=20.0,
        sweep_length=40.0, n_x=n_x, n_arc=n_arc, n_a=n_a, n_b=n_b)
    normals = compute_face_normals(vertices, faces)

    mid_xi = n_x // 2
    panel_a_seed = face_at(mid_xi, 0)
    panel_b_seed = face_at(mid_xi, n_a + n_arc)

    region_a = grow_coplanar_region(faces, normals, panel_a_seed, angle_tol_deg=2.0, vertices=vertices)
    region_b = grow_coplanar_region(faces, normals, panel_b_seed, angle_tol_deg=2.0, vertices=vertices)

    # The angle-tolerance BFS should correctly refuse to cross the fillet on
    # its own — this is the bug being fixed, confirmed as a real gap first.
    assert region_a.isdisjoint(region_b), "flat regions should not directly touch across the fillet"

    outline, outline_vertex_indices = connect_regions_with_strut(
        vertices, faces, region_a, panel_a_seed, region_b, panel_b_seed)
    poly = Polygon(outline)
    assert poly.is_valid, "strut-connected outline should be a valid simple polygon"
    assert len(outline_vertex_indices) == len(outline), \
        "outline_vertex_indices must be parallel to outline"
    assert all(0 <= v < len(vertices) for v in outline_vertex_indices), \
        "every outline point should map back to a real mesh vertex"

    # Sanity-check size: panel A is 30x40mm, panel B is 20x40mm, joined
    # along their 40mm edge — true combined area (including the fillet
    # strip) is a bit over 30*40 + 20*40 = 2000mm²; the straight-cut
    # approximation should be in the same ballpark, not wildly smaller
    # (a bug pulling in almost nothing) or larger (a bug wrapping around
    # the wrong way).
    assert 1500 < poly.area < 2500, f"unexpected outline area {poly.area:.1f}mm² for a ~2000mm² patch"

    print(f"test_filleted_corner_connects_with_strut: OK "
          f"(region_a={len(region_a)}, region_b={len(region_b)}, outline={len(outline)} points, "
          f"area={poly.area:.1f}mm²)")


def test_remove_needle_spikes():
    """Out-and-back needle vertices (from real sliver triangles at a region
    boundary — e.g. finger_tip_soft.stl's flat side walls carry ~4.8mm-long,
    ~0.12mm-wide ones) must be stripped from a final outline, while a
    genuinely wide tab survives.
    """
    from opentac.cad.pipeline.mesh_unroll import remove_needle_spikes

    square_with_spike = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
        [5.12, 10.0], [5.06, 14.8], [5.0, 10.0],  # 4.8mm needle, 0.12mm tip
        [0.0, 10.0],
    ])
    cleaned = remove_needle_spikes(square_with_spike)
    assert len(cleaned) == 6, f"expected the needle vertex dropped, got {len(cleaned)} points"
    assert not any(np.allclose(p, [5.06, 14.8]) for p in cleaned)

    square_with_tab = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
        [6.0, 10.0], [6.0, 14.8], [4.0, 14.8], [4.0, 10.0],  # real 2mm-wide tab
        [0.0, 10.0],
    ])
    kept = remove_needle_spikes(square_with_tab)
    assert len(kept) == len(square_with_tab), "a 2mm-wide tab must not be treated as a spike"

    # Removing one spike can expose another (nested) — must iterate to stable.
    nested = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
        [5.2, 10.0], [5.15, 13.0], [5.1, 16.0], [5.05, 13.05], [5.0, 10.1],
        [0.0, 10.0],
    ])
    stable = remove_needle_spikes(nested)
    assert len(stable) <= 7 and all(p[1] < 13.5 for p in stable), \
        f"nested spike not fully removed: {stable.tolist()}"
    print("test_remove_needle_spikes: OK (spike dropped, tab kept, nested stable)")


def test_cylinder_unrolls_flat():
    vertices, faces = make_cylinder_patch(
        radius=20.0, sweep_angle_deg=90.0, height=15.0, n_theta=12, n_z=5)
    uv, visited = unfold_triangle_patch(vertices, faces, start_face=0)
    disagreement = check_developability(vertices, faces, uv, visited)
    assert len(visited) == len(faces), "not all faces reached — mesh connectivity bug"
    assert disagreement < 1e-6, f"cylinder should be exactly developable, got {disagreement:.6f}mm"
    print(f"test_cylinder_unrolls_flat: OK (max disagreement={disagreement:.2e}mm, "
          f"{len(uv)} vertices, {len(visited)} faces)")


def test_hex_prism_unrolls_to_exact_strip_widths():
    radius = 20.0
    vertices, faces = make_hex_prism_patch(radius=radius, n_facets=4, height=10.0)
    uv, visited = unfold_triangle_patch(vertices, faces, start_face=0)
    disagreement = check_developability(vertices, faces, uv, visited)
    assert disagreement < 1e-9, f"flat prism should be exactly developable, got {disagreement:.9f}mm"

    true_edge_len = 2 * radius * np.sin(np.pi / 6)  # regular hexagon side length
    for hi in range(4):
        # vertex indices for column hi and hi+1 at z=0 are 2*hi and 2*(hi+1)
        v_a, v_b = 2 * hi, 2 * (hi + 1)
        strip_width = np.linalg.norm(uv[v_a] - uv[v_b])
        assert abs(strip_width - true_edge_len) < 1e-9, (
            f"facet {hi}: strip width {strip_width:.4f} != true edge length {true_edge_len:.4f}")
    print(f"test_hex_prism_unrolls_to_exact_strip_widths: OK "
          f"(disagreement={disagreement:.2e}mm, strip width={true_edge_len:.4f}mm exact)")


def test_hex_pyramid_angled_facets_unroll_correctly():
    """The case that broke the earlier axis-fit draft: facets converge to
    an apex instead of staying parallel. Triangle-unfolding needs no
    special-casing for this — verifies the generalization actually holds.
    """
    vertices, faces = make_hex_pyramid_patch(base_radius=20.0, height=25.0, n_facets=4)
    uv, visited = unfold_triangle_patch(vertices, faces, start_face=0)
    disagreement = check_developability(vertices, faces, uv, visited)
    assert len(visited) == len(faces)
    assert disagreement < 1e-9, f"flat pyramid facets should be exactly developable, got {disagreement:.9f}mm"

    # The apex (vertex 6) is shared by every facet — confirm it landed at
    # exactly one 2D position (not silently duplicated/inconsistent).
    assert 6 in uv
    apex_true_dists = [np.linalg.norm(vertices[6] - vertices[i]) for i in range(4)]
    apex_uv_dists = [np.linalg.norm(uv[6] - uv[i]) for i in range(4)]
    for true_d, uv_d in zip(apex_true_dists, apex_uv_dists):
        assert abs(true_d - uv_d) < 1e-9
    print(f"test_hex_pyramid_angled_facets_unroll_correctly: OK "
          f"(disagreement={disagreement:.2e}mm, apex distances exact)")


def test_sphere_patch_flagged_as_non_developable():
    vertices, faces = make_sphere_patch(
        radius=20.0, n_theta=8, n_phi=8,
        theta_range=(np.pi / 4, np.pi / 2), phi_range=(0, np.pi / 2))
    uv, visited = unfold_triangle_patch(vertices, faces, start_face=0)
    disagreement = check_developability(vertices, faces, uv, visited)
    assert disagreement > 0.1, (
        f"sphere patch should show clear non-developability, got only {disagreement:.4f}mm")
    print(f"test_sphere_patch_flagged_as_non_developable: OK "
          f"(disagreement={disagreement:.3f}mm — correctly flagged as not flattenable)")


def test_flat_cases_clearly_separated_from_sphere():
    """Sanity check the discriminating power of check_developability: the
    sphere's disagreement should be orders of magnitude larger than any of
    the genuinely-developable cases, not just "a bit more."
    """
    cyl_v, cyl_f = make_cylinder_patch(20.0, 90.0, 15.0, 12, 5)
    cyl_uv, cyl_vis = unfold_triangle_patch(cyl_v, cyl_f)
    cyl_d = check_developability(cyl_v, cyl_f, cyl_uv, cyl_vis)

    sph_v, sph_f = make_sphere_patch(20.0, 8, 8, (np.pi / 4, np.pi / 2), (0, np.pi / 2))
    sph_uv, sph_vis = unfold_triangle_patch(sph_v, sph_f)
    sph_d = check_developability(sph_v, sph_f, sph_uv, sph_vis)

    assert sph_d > cyl_d * 1000, (
        f"sphere disagreement ({sph_d:.4f}) should be >>1000x the cylinder's ({cyl_d:.2e})")
    print(f"test_flat_cases_clearly_separated_from_sphere: OK "
          f"(cylinder={cyl_d:.2e}mm vs sphere={sph_d:.4f}mm, ratio={sph_d/max(cyl_d,1e-12):.0f}x)")


if __name__ == "__main__":
    test_remove_needle_spikes()
    test_cylinder_unrolls_flat()
    test_hex_prism_unrolls_to_exact_strip_widths()
    test_hex_pyramid_angled_facets_unroll_correctly()
    test_sphere_patch_flagged_as_non_developable()
    test_flat_cases_clearly_separated_from_sphere()
    test_filleted_corner_connects_with_strut()
    print("\n=== ALL MESH UNROLL TESTS PASSED ===")
