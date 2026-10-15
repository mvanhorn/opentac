"""Unroll a triangle-mesh patch into a flat 2D layout by unfolding triangle
by triangle across shared edges — the same physical process as rolling the
object across paper. Each triangle's true flat shape is fully determined by
its 3 edge lengths (law of cosines), with zero distortion, regardless of the
overall surface family (cylinder, prism, cone, pyramid, or an irregular
faceted shape) — no axis-fitting or shape classification needed.

Used by main.py's POST /unroll-mesh endpoint. See NOTES.md's "STL/3D unroll"
section for the full design writeup, validation results, and real-STL bugs
found/fixed during development.
"""

import numpy as np


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _cross2d(a, b):
    """Scalar z-component of the 2D cross product (numpy 2.x dropped
    implicit 2D support from np.cross)."""
    return a[0] * b[1] - a[1] * b[0]


def _build_face_adjacency(faces):
    """edge (v_lo, v_hi) -> list of face indices sharing that edge."""
    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for e in (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)):
            edge_faces.setdefault(e, []).append(fi)
    return edge_faces


def _place_third_point(a2d, b2d, dist_ac, dist_bc, sign):
    """Given 2D positions of A and B and true distances |AC|, |BC|, solve C's
    2D position via law of cosines. `sign` picks which side of line AB.
    """
    ab = b2d - a2d
    c = np.linalg.norm(ab)
    u_hat = ab / c
    n_hat = np.array([-u_hat[1], u_hat[0]])

    b_, a_ = dist_ac, dist_bc  # |AC|, |BC|
    x = (b_ ** 2 - a_ ** 2 + c ** 2) / (2 * c)
    y_sq = max(0.0, b_ ** 2 - x ** 2)
    y = sign * np.sqrt(y_sq)

    return a2d + x * u_hat + y * n_hat


def _triangle_quality(vertices: np.ndarray, tri) -> float:
    """Shape quality of one triangle, 1.0 for equilateral down toward 0 for
    a degenerate sliver: `4*sqrt(3)*Area / (a^2+b^2+c^2)`, the standard
    normalized-radius-ratio-style triangle quality metric (scale-invariant,
    so it doesn't care how big the triangle is, only its shape)."""
    a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
    ab, ac = b - a, c - a
    area = 0.5 * np.linalg.norm(np.cross(ab, ac))
    perim_sq = np.sum((b - a) ** 2) + np.sum((c - b) ** 2) + np.sum((a - c) ** 2)
    if perim_sq < 1e-18:
        return 0.0
    return float(4 * np.sqrt(3) * area / perim_sq)


def grow_coplanar_region(faces: np.ndarray, face_normals: np.ndarray, start_face: int,
                          angle_tol_deg: float = 2.0, vertices: np.ndarray = None,
                          min_quality: float = 0.02) -> set:
    """BFS out from start_face across shared edges, only including neighbors
    whose normal matches the starting face's normal within angle_tol_deg.
    Captures a single flat surface of a part from one clicked triangle,
    stopping at any real crease/edge — the natural "click to select a face"
    interaction for a CAD-like part with distinct flat surfaces.

    When `vertices` is given, also refuses to grow *into* a degenerate
    sliver triangle (quality below `min_quality` — see `_triangle_quality`)
    even if its normal is within tolerance. Found live: a real STL can
    have a near-zero-area sliver triangle sitting right at a flat region's
    boundary, technically coplanar enough to pass the angle check, whose
    inclusion makes the selected region's own edge visibly jagged — a
    single stray spike poking off an otherwise clean straight edge — which
    then also becomes a bad, easy-to-misclick candidate corner for the
    manual connect-corners picker (see NOTES.md's "STL/3D unroll"
    section). Skipping sliver triangles during growth (rather than trying
    to clean them up after the fact) keeps the selected region's boundary
    clean in the first place. `vertices` is optional and defaults to no
    filtering (`None`) only for callers that don't have it handy and don't
    need this — every real call site in this app does pass it.
    """
    edge_faces = _build_face_adjacency(faces)
    start_normal = face_normals[start_face]
    cos_tol = np.cos(np.radians(angle_tol_deg))

    visited = {start_face}
    queue = [start_face]
    while queue:
        fi = queue.pop()
        tri = faces[fi]
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            key = _edge_key(v1, v2)
            for nfi in edge_faces.get(key, []):
                if nfi in visited:
                    continue
                if np.dot(face_normals[nfi], start_normal) < cos_tol:
                    continue
                if vertices is not None and _triangle_quality(vertices, faces[nfi]) < min_quality:
                    continue
                visited.add(nfi)
                queue.append(nfi)
    return visited

def unfold_face_region(vertices: np.ndarray, faces: np.ndarray, region_face_indices, start_face: int):
    """Unfold only a subset of a larger mesh's faces (e.g. the output of
    grow_coplanar_region on a full imported STL), by extracting a submesh
    (region faces + only their referenced vertices, renumbered) and running
    unfold_triangle_patch on that — so adjacency/BFS can't leak into the rest
    of the mesh outside the selected region.

    Returns (uv, visited_local, orig_vertex_index) where orig_vertex_index[i]
    maps a local submesh vertex index back to its index in the original
    `vertices` array.
    """
    region_list = [start_face] + [f for f in region_face_indices if f != start_face]
    region_faces_orig = faces[region_list]  # (R, 3), original vertex indices

    orig_vertex_index, inv = np.unique(region_faces_orig, return_inverse=True)
    sub_vertices = vertices[orig_vertex_index]
    sub_faces = inv.reshape(-1, 3)

    uv, visited_local = unfold_triangle_patch(sub_vertices, sub_faces, start_face=0)
    return uv, visited_local, orig_vertex_index


def unfold_triangle_patch(vertices: np.ndarray, faces: np.ndarray, start_face: int = 0):
    """Unfold a connected triangle patch into 2D.

    Args:
        vertices: (N, 3) array of 3D vertex positions.
        faces: (F, 3) int array of vertex indices per triangle.
        start_face: which face to lay down first (arbitrary orientation).

    Returns:
        uv: dict {vertex_index: np.array([u, v])} for every vertex reached
            from start_face via the mesh's connectivity.
        visited_faces: set of face indices that were placed.
    """
    edge_faces = _build_face_adjacency(faces)

    uv = {}
    a, b, c = faces[start_face]
    dist_ab = np.linalg.norm(vertices[a] - vertices[b])
    dist_ac = np.linalg.norm(vertices[a] - vertices[c])
    dist_bc = np.linalg.norm(vertices[b] - vertices[c])

    uv[a] = np.array([0.0, 0.0])
    uv[b] = np.array([dist_ab, 0.0])
    uv[c] = _place_third_point(uv[a], uv[b], dist_ac, dist_bc, sign=1.0)

    visited_faces = {start_face}
    queue = [start_face]

    while queue:
        fi = queue.pop()
        tri = faces[fi]
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            v3 = tri[(i + 2) % 3]  # third vertex of the current face
            key = _edge_key(v1, v2)
            for nfi in edge_faces.get(key, []):
                if nfi in visited_faces:
                    continue
                ntri = faces[nfi]
                third = [v for v in ntri if v != v1 and v != v2]
                if len(third) != 1:
                    continue  # degenerate/duplicate face, skip
                nv = third[0]

                if v1 not in uv or v2 not in uv:
                    # Shouldn't happen if we only queue faces reached via a
                    # placed edge, but guard defensively.
                    continue

                if nv not in uv:
                    dist_nv1 = np.linalg.norm(vertices[nv] - vertices[v1])
                    dist_nv2 = np.linalg.norm(vertices[nv] - vertices[v2])
                    # Unfold outward: place the new point on the opposite
                    # side of edge (v1,v2) from the current face's own third
                    # vertex v3 (already placed), like opening a hinge.
                    side_v3 = np.sign(_cross2d(uv[v2] - uv[v1], uv[v3] - uv[v1]))
                    sign = -side_v3 if side_v3 != 0 else 1.0
                    uv[nv] = _place_third_point(uv[v1], uv[v2], dist_nv1, dist_nv2, sign)

                visited_faces.add(nfi)
                queue.append(nfi)

    return uv, visited_faces


def check_developability(vertices: np.ndarray, faces: np.ndarray, uv: dict, visited_faces: set) -> float:
    """Max disagreement (mm) between true 3D edge length and 2D unfolded
    distance, over every mesh edge whose both endpoints got placed.

    Near zero means the patch is (numerically) developable — flattenable
    without distortion. A large value means real curvature in more than one
    direction somewhere in the selection (not flattenable), e.g. a sphere
    patch — different unfolding paths through the mesh disagree on where a
    vertex should end up.
    """
    max_disagreement = 0.0
    seen_edges = set()
    for fi in visited_faces:
        a, b, c = faces[fi]
        for v1, v2 in ((a, b), (b, c), (c, a)):
            key = _edge_key(v1, v2)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            if v1 not in uv or v2 not in uv:
                continue
            true_dist = np.linalg.norm(vertices[v1] - vertices[v2])
            uv_dist = np.linalg.norm(uv[v1] - uv[v2])
            max_disagreement = max(max_disagreement, abs(true_dist - uv_dist))
    return max_disagreement


def find_boundary_loops(faces: np.ndarray) -> list:
    """Boundary edges (used by exactly one triangle) of a face set, traced
    into ordered vertex-index loops. A patch with holes (e.g. bolt holes on
    a real part) has more than one loop — one outer perimeter plus one per
    hole — so this finds *all* of them, repeating the walk from any
    not-yet-consumed boundary edge, matching the demo viewer's JS logic
    (see NOTES.md's "STL/3D unroll" section for why finding only the first
    loop is a real bug on real parts, not a hypothetical one).
    """
    edge_count = {}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            key = _edge_key(x, y)
            edge_count[key] = edge_count.get(key, 0) + 1

    adj = {}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            if edge_count[_edge_key(x, y)] != 1:
                continue
            adj.setdefault(x, []).append(y)
            adj.setdefault(y, []).append(x)

    if not adj:
        return []

    visited_edges = set()
    loops = []
    for start in list(adj.keys()):
        if not any(_edge_key(start, n) not in visited_edges for n in adj[start]):
            continue
        loop = [start]
        prev, cur = None, start
        for _ in range(len(adj) + 1):
            neighbors = adj[cur]
            nxt = next((n for n in neighbors if n != prev or _edge_key(cur, n) not in visited_edges), None)
            if nxt is None:
                break
            visited_edges.add(_edge_key(cur, nxt))
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _loop_signed_area(loop, uv):
    area = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = uv[loop[i]]
        x2, y2 = uv[loop[(i + 1) % n]]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def simplify_outline_points(outline: np.ndarray, simplify_tol: float = 0.3) -> np.ndarray:
    """Douglas-Peucker-simplify an already-assembled Nx2 outline (Shapely),
    removing small zigzag/jitter from raw mesh-tessellation detail without
    meaningfully changing the shape — factored out of `extract_outline`
    (below) so strut-connected outlines can use the same cleanup: unlike
    a plain unfolded patch, a strut-joined outline never went through
    `extract_outline` at all (it's assembled directly from two loops'
    "far" arcs plus straight struts), so it never got this treatment,
    which is exactly what showed up live as a small pointless zigzag in an
    otherwise-straight strut-adjacent boundary segment. See
    `extract_outline`'s docstring for the tolerance-choice rationale.
    """
    if simplify_tol <= 0 or len(outline) <= 3:
        return outline
    from shapely.geometry import Polygon
    poly = Polygon(outline)
    if not poly.is_valid:
        return outline
    simplified = poly.simplify(simplify_tol, preserve_topology=True)
    coords = np.array(simplified.exterior.coords)[:-1]  # drop repeated closing point
    return coords if len(coords) >= 3 else outline


def remove_needle_spikes(outline: np.ndarray, tip_tol: float = 0.5,
                          min_spike_len: float = 1.5) -> np.ndarray:
    """Drop out-and-back needle vertices from an assembled Nx2 outline: any
    vertex whose two incident edges are each at least `min_spike_len` long
    but whose neighbors sit within `tip_tol` of each other — the boundary
    runs several mm out and comes straight back to (nearly) where it left.

    These come from real needle-sliver triangles at a selected region's
    boundary (e.g. `finger_tip_soft.stl`'s flat side walls carry ~4.8mm-long,
    ~0.12mm-wide slivers whose quality ~0.05 passes grow_coplanar_region's
    0.02 floor, found while chaining 4 surfaces). Whether such a spike
    reaches the final outline depends on which arcs the strut cuts happen to
    keep, so it appears and disappears with unrelated corner choices.
    Douglas-Peucker can't remove one (its deviation IS the spike length),
    hence this dedicated pass. A sub-`tip_tol`-wide outline feature can
    never be meaningful board geometry at sensor scale (trace width and
    clearance are ~0.2mm each, so nothing routable fits inside it).

    Applied to *final assembled* outlines only — same policy as
    `simplify_outline_points`, and for the same reason: earlier loops'
    per-point vertex identity must survive for corner picking. Run this
    BEFORE `simplify_outline_points`, which then merges the near-coincident
    neighbor pair the removed spike leaves behind. Iterates until stable
    (removing one spike can expose another).
    """
    outline = np.asarray(outline, dtype=float)
    changed = True
    while changed and len(outline) > 3:
        changed = False
        n = len(outline)
        keep = np.ones(n, dtype=bool)
        # width 1: one interior vertex (a single sliver triangle); width 2:
        # two interior vertices (chained slivers) — the out and back edges
        # must each be long, the middle edge of a width-2 spike may be tiny.
        for width in (1, 2):
            for i in range(n):
                interior = [(i + s) % n for s in range(1, width + 1)]
                end = (i + width + 1) % n
                if not (keep[i] and keep[end] and all(keep[j] for j in interior)):
                    continue
                if np.linalg.norm(outline[i] - outline[end]) >= tip_tol:
                    continue
                out_len = np.linalg.norm(outline[interior[0]] - outline[i])
                back_len = np.linalg.norm(outline[end] - outline[interior[-1]])
                if out_len >= min_spike_len and back_len >= min_spike_len:
                    for j in interior:
                        keep[j] = False
                    changed = True
        if changed:
            outline = outline[keep]
    return outline


def _reduce_boundary_loop(loop_orig: list, loop_uv: list, tol: float = 0.5):
    """Reduce a boundary loop to a small set of well-separated *real* mesh
    vertices, dropping tessellation-jitter points in between — the same
    cleanup idea as `simplify_outline_points`, but constrained to only ever
    select a subset of the *original* vertices rather than introduce new
    interpolated ones: every consumer of a region's boundary loop
    (`_two_connection_pairs`'s automatic corner heuristic, and — the reason
    this exists — the manual corner-picker's clickable markers) needs a
    real mesh vertex index, not an arbitrary point on the boundary. Applied
    once, at the source (`region_boundary_loop` for a fresh region,
    `connect_loop_to_region` for its own assembled result before an even
    later region chains onto it) — every consumer downstream is
    automatically clean, rather than needing its own cleanup pass.

    Found live: a real STL's boundary loop for a small/thin/jaggedly-
    tessellated region can pack many boundary vertices within a couple mm
    of each other, so a corner-picker built from the raw per-vertex list
    is easy to misclick — two visually-adjacent dots can be several
    array-positions apart, and the "wrong" one still looks like a
    plausible corner.

    Implementation: run Shapely's Douglas-Peucker simplify (as
    `simplify_outline_points` does) to find which points matter, then snap
    each simplified (possibly interpolated) point to its nearest *original*
    loop vertex instead of using the interpolated point directly, then
    dedupe.

    Returns `(loop_orig_reduced, loop_uv_reduced)`, same parallel-list
    shape as the inputs, order-preserved. Falls back to the original
    (unreduced) loop if simplification collapses it below a usable size
    (fewer than 3 points) or the loop is degenerate for Shapely.
    """
    if len(loop_uv) <= 3:
        return loop_orig, loop_uv
    from shapely.geometry import Polygon
    uv_arr = np.array(loop_uv)
    poly = Polygon(uv_arr)
    if not poly.is_valid:
        return loop_orig, loop_uv
    simplified = poly.simplify(tol, preserve_topology=True)
    simplified_coords = np.array(simplified.exterior.coords)[:-1]
    if len(simplified_coords) < 3:
        return loop_orig, loop_uv

    keep_indices = []
    for pt in simplified_coords:
        nearest = int(np.argmin(np.linalg.norm(uv_arr - pt, axis=1)))
        if nearest not in keep_indices:
            keep_indices.append(nearest)
    keep_indices.sort()
    if len(keep_indices) < 3:
        return loop_orig, loop_uv
    return [loop_orig[i] for i in keep_indices], [loop_uv[i] for i in keep_indices]


def extract_outline(faces: np.ndarray, uv: dict, simplify_tol: float = 0.3) -> np.ndarray:
    """The outer-perimeter boundary loop of an unfolded patch, as an Nx2
    array in the same coordinate frame as `uv` — directly usable as the
    pipeline's outline_xy. Any inner hole loops (bolt holes, etc.) are
    intentionally ignored for v1 — the outer perimeter is whichever
    boundary loop has the largest enclosed area (same "largest wins"
    convention as importers/dxf.py's multi-loop handling).

    `simplify_tol` (mm) removes small zigzag/jitter from the raw boundary
    trace via Shapely's Douglas-Peucker simplification. Found live on a
    real STL: even a topologically clean boundary trace (verified no
    branch points, no non-manifold edges — every boundary vertex has
    exactly 2 boundary edges) faithfully reproduces genuinely rough
    small-scale detail in the source mesh's tessellation, which shows up
    as a visually jittery/zigzagging outline segment — technically a valid
    simple polygon, but not a usable board shape. Verified this doesn't
    meaningfully alter the shape: on that same case, tol=0.3mm cut a
    169-point outline down to ~19 points while changing enclosed area by
    under 1%.
    """
    loops = find_boundary_loops(faces)
    if not loops:
        raise ValueError("No boundary found in the selected patch.")
    outer = max(loops, key=lambda loop: abs(_loop_signed_area(loop, uv)))
    outline = np.array([uv[v] for v in outer])
    return simplify_outline_points(outline, simplify_tol)


def _two_connection_pairs(p1_3d: np.ndarray, p2_3d: np.ndarray):
    """Candidate points on each loop for the two struts' endpoints, between
    `p1_3d` (loop1's boundary vertex positions) and `p2_3d` (loop2's).

    Returns `(i1, j1, extreme1, extreme2)`: `i1`/`j1` is the globally-
    closest point pair; `extreme1`/`extreme2` are each loop's own farthest
    point (in 3D), restricted to its half-plane "facing" cluster, from its
    own closest-pair point — i.e. each loop's natural facing-edge width,
    never a degenerate near-zero-width pair. `connect_regions_with_strut`
    tries pairing these two ways ("same": i1-j1 + extreme1-extreme2,
    "crossed": i1-extreme2 + extreme1-j1) since which one avoids a
    self-intersecting strut depends on the two loops' relative facing
    widths — see its docstring.
    """
    c1, c2 = p1_3d.mean(axis=0), p2_3d.mean(axis=0)

    def facing_mask(p, self_c, other_c):
        mask = np.dot(p - self_c, other_c - self_c) > 0
        if not mask.any() or mask.all():
            mask = np.ones(len(p), dtype=bool)
        return mask

    close1 = facing_mask(p1_3d, c1, c2)
    close2 = facing_mask(p2_3d, c2, c1)

    all_dists = np.linalg.norm(p1_3d[:, None, :] - p2_3d[None, :, :], axis=2)
    a, b = np.unravel_index(np.argmin(all_dists), all_dists.shape)
    i1, j1 = int(a), int(b)

    idx1 = np.where(close1)[0]
    idx2 = np.where(close2)[0]
    extreme1 = int(idx1[np.argmax(np.linalg.norm(p1_3d[idx1] - p1_3d[i1], axis=1))])
    extreme2 = int(idx2[np.argmax(np.linalg.norm(p2_3d[idx2] - p2_3d[j1], axis=1))])
    if extreme1 == i1:
        extreme1 = int(idx1[0]) if idx1[0] != i1 else (i1 + 1) % len(p1_3d)
    if extreme2 == j1:
        extreme2 = int(idx2[0]) if idx2[0] != j1 else (j1 + 1) % len(p2_3d)

    return i1, j1, extreme1, extreme2


def _loop_far_segment(s: int, e: int, n: int) -> list:
    """The longer of the two arcs between `s` and `e` around a loop of size
    `n`, inclusive of both endpoints — i.e. "the rest of the loop" once the
    short `s`-`e` connecting edge is excluded.

    Picks by vertex count rather than by argument order: `s`/`e` mark the
    two connection points, and the near (strut-replaced) edge between them
    is always the minority of the boundary, but which of the two possible
    walks (`e` forward to `s`, or `s` forward to `e`) is the *short* one
    depends on the loop's arbitrary indexing — not on which argument was
    named `s`. Picking by argument order alone silently returns the near
    (small) side about as often as the far side.
    """
    def walk(a, b):
        idxs = []
        i = a
        while True:
            idxs.append(i)
            if i == b:
                break
            i = (i + 1) % n
        return idxs

    fwd = walk(e, s)
    rev = walk(s, e)
    return fwd if len(fwd) >= len(rev) else rev


def region_boundary_loop(vertices: np.ndarray, faces: np.ndarray, region: set, start_face: int):
    """The outer boundary loop of `region` (unfolded independently, its own
    local 2D frame), as parallel lists: `loop_orig` (original mesh vertex
    index per loop point), `loop_uv` (that point's unfolded 2D position),
    `p_3d` (that point's true 3D position). Factored out of
    `connect_regions_with_strut` so `main.py` can expose a region's
    boundary to the frontend (e.g. for manual corner-picking) without
    duplicating the extraction logic.

    Deliberately returns the *full-resolution* boundary walk, not reduced
    via `_reduce_boundary_loop` — tried reducing here once, and it broke
    the automatic same/crossed corner heuristic (`_two_connection_pairs`)
    on a real case that worked fine at full resolution: reducing removes
    exactly the fine-grained detail (the true closest point between two
    regions, a cluster's true extreme) the heuristic depends on, so it's
    only safe to reduce *after* the connection math is done — either on
    the final assembled outline, or on a display-only copy for a corner-
    picker's markers. See `main.py`'s `/unroll-mesh-chain` for the
    display-only reduction.
    """
    uv, _, orig_idx = unfold_face_region(vertices, faces, region, start_face)
    region_list = [start_face] + [f for f in region if f != start_face]
    sub_faces_orig = faces[region_list]
    _, inv = np.unique(sub_faces_orig, return_inverse=True)
    sub_faces = inv.reshape(-1, 3)
    loops = find_boundary_loops(sub_faces)
    loop_local = max(loops, key=lambda loop: abs(_loop_signed_area(loop, uv)))
    loop_orig = [orig_idx[v] for v in loop_local]
    loop_uv = [np.array(uv[v]) for v in loop_local]
    p_3d = vertices[loop_orig]
    return loop_orig, loop_uv, p_3d


def connect_regions_with_strut(vertices: np.ndarray, faces: np.ndarray,
                                region1: set, start_face1: int,
                                region2: set, start_face2: int,
                                connection_pairs=None):
    """Thin wrapper around `connect_loop_to_region`: unfolds `region1` into
    its own boundary loop first, then delegates. See `connect_loop_to_region`
    for the actual algorithm and all parameter/return docs — this wrapper
    exists only for the common case where `region1` is a fresh mesh region
    rather than an already-flattened running chain (see `connect_loop_to_region`
    for that 3rd-plus-region case).
    """
    loop1_orig, loop1_uv, _ = region_boundary_loop(vertices, faces, region1, start_face1)
    return connect_loop_to_region(vertices, faces, loop1_orig, loop1_uv,
                                   region2, start_face2, connection_pairs=connection_pairs)


def connect_loop_to_region(vertices: np.ndarray, faces: np.ndarray,
                            loop1_orig: list, loop1_uv: list,
                            region2: set, start_face2: int,
                            connection_pairs=None):
    """The outline of an already-flattened loop (`loop1_orig`/`loop1_uv` —
    original mesh vertex indices and their 2D positions, in one consistent
    frame) and `region2` (unfolded independently here, exactly as validated
    for a single flat region) joined by two straight struts, approximating
    whatever real connecting surface exists between them (a fillet, a
    chamfer, several irregular facets — this doesn't care) with a
    straight-line cut instead.

    Taking `loop1` as an already-flattened loop rather than a fresh mesh
    region is what makes a 3rd (or later) region connectable onto a running
    chain: once the *first* strut connection happens, the combined shape
    isn't a real mesh region anymore (some of its "boundary" is a synthetic
    strut, not real mesh edges), so it can't be re-unfolded from a face set
    the way a single region can — the caller must instead pass the
    previous response's own outline back in. `connect_regions_with_strut`
    is the common-case wrapper for the *first* strut (region1 is still a
    real mesh region there).

    Returns `(outline, outline_vertex_indices)`: `outline` is an Nx2 array
    exactly as before; `outline_vertex_indices` is a parallel list of each
    point's *original mesh vertex index* (every outline point — whether
    from `loop1` or the newly-unfolded `region2` — is a real mesh vertex;
    only the two strut *edges* connecting them are synthetic). A caller
    extending the chain again passes this same pair back in as the next
    call's `loop1_orig`/`loop1_uv` (well, `loop1_uv` needs `outline`
    directly — `outline_vertex_indices` is what makes `vertices[...]`
    lookups against it meaningful for picking further connection points).

    Replaced `bridge_gap` (a shortest-path/distance-ellipse search that
    tried to select and include the actual connecting mesh faces) as the
    way `/unroll-mesh` handles two disjoint flat selections, after
    repeated real-world failures: every fix to the face-selection approach
    (uniform-width assumption, wandering onto unrelated curved geometry,
    non-determinism, a hair-thin pinch point, a genuinely jittery boundary
    trace from faithfully reproducing rough small-scale mesh detail) was
    followed by a *new*, different failure on the next real test case —
    see NOTES.md's "STL/3D unroll" section for the full history. Trying to
    faithfully select and flatten the true connecting geometry turned out
    to be fundamentally fragile; approximating the connection with a
    straight cut sidesteps the whole problem, because both endpoints
    (region1 and region2 individually) were *never* the source of any of
    those failures — each region alone always unfolded cleanly. This
    trades a small amount of shape fidelity (the true fillet/chamfer curve
    becomes a flat cut) for something that just works, which matches the
    existing "holes are ignored" v1 stance: approximate now, let the user
    manually adjust in the next step if they need more precision.

    Algorithm: find each region's own boundary loop (already always a
    single clean loop for a flat region grown from one click) and unfold
    it independently — its own local, always-valid 2D frame. Find two
    connection point-pairs via `_two_connection_pairs` — these are the two
    struts' endpoints. Treat the two corner pairs as two phantom triangles
    and hinge-place region2's corner points into region1's frame using the
    same law-of-cosines primitive (`_place_third_point`) used everywhere
    else in this module, from their true 3D distances — then a rigid
    transform (uniquely determined by those two now-placed points) carries
    all of region2's boundary into region1's frame. The final outline is
    region1's boundary walked around its "far" (non-facing) side, a strut
    to region2, region2's boundary walked around its far side, and a strut
    back. Since `_loop_far_segment`'s notion of "far" depends on which of
    each pair's two points is treated as the arc's start vs. end, all four
    role/direction combinations are tried and the largest-area valid
    result is kept — the wrong combination can silently pick the *near*
    (small) side of a loop instead of the far (large) one.

    **Two correspondences, not just one, are tried between the pairs**:
    "same" (loop1's closest-pair point <-> loop2's closest-pair point,
    loop1's cluster-extreme <-> loop2's cluster-extreme) and "crossed"
    (loop1's closest-pair point <-> loop2's cluster-extreme, and vice
    versa). Which one yields a non-self-intersecting strut depends on the
    two loops' relative facing-edge widths, found the hard way: "same"
    is right when both loops' facing edges are comparable widths (the
    synthetic fillet test, both ~40mm), but on a real asymmetric pair
    (an 8mm-wide small patch facing a 22mm-wide large wall,
    `finger_tip_soft.stl` faces 21478 + 6371) "same" pairing put two
    *nearly coincident* points on whichever loop's own closest-pair point
    and extreme happened to be close together, which forced the other
    loop's "far" arc to sweep across nearly its entire boundary to close
    the polygon — reading as two lobes pinched at a single point (a
    boomerang), not a self-intersection `Polygon.is_valid` would catch,
    so earlier attempts that only checked validity kept silently
    shipping that pinch. "Crossed" pairing fixed it because it keeps both
    loops' two strut points anchored to that loop's *own* natural
    facing-cluster width instead of an artifact of which point happened
    to be globally closest overall.

    **`connection_pairs`** (optional): `((v_a1, v_b1), (v_a2, v_b2))`, using
    *original mesh vertex indices* (not loop-local ones) — when given, the
    automatic corner-selection heuristic above (including the same/crossed
    trial) is skipped entirely, and strut 1 connects `v_a1` (must be on
    region1's boundary) to `v_b1` (must be on region2's), strut 2 connects
    `v_a2` to `v_b2`. This is for a manual "pick the two connection corners
    yourself" UI flow — the correspondence is then fully determined by the
    user's click order, so there's no same-vs-crossed ambiguity left to
    search over; the role/direction trial (below) still runs, since that
    part resolves a real remaining ambiguity (which arc is "far") rather
    than guessing at correspondence. Raises ValueError if a given vertex
    isn't actually on the corresponding region's boundary loop.

    Raises ValueError if no valid (non-self-intersecting) polygon can be
    formed in any combination tried — in practice this hasn't been
    observed, but a highly unusual mutual geometry could in principle
    produce one.
    """
    from shapely.geometry import Polygon

    p1_3d = vertices[loop1_orig]
    loop2_orig, loop2_uv, p2_3d = region_boundary_loop(vertices, faces, region2, start_face2)
    n1, n2 = len(loop1_uv), len(loop2_uv)

    if connection_pairs is not None:
        (v_a1, v_b1), (v_a2, v_b2) = connection_pairs
        orig1_to_local = {v: k for k, v in enumerate(loop1_orig)}
        orig2_to_local = {v: k for k, v in enumerate(loop2_orig)}
        try:
            i1, i2 = orig1_to_local[v_a1], orig1_to_local[v_a2]
            j1, j2 = orig2_to_local[v_b1], orig2_to_local[v_b2]
        except KeyError as e:
            raise ValueError(f"Picked vertex {e.args[0]} is not on the selected region's boundary.")
        pairings = [(j1, j2)]  # correspondence is fixed by the user's picks
    else:
        i1, j1, extreme1, extreme2 = _two_connection_pairs(p1_3d, p2_3d)
        i2 = extreme1
        pairings = [(j1, extreme2), (extreme2, j1)]  # "same", "crossed" — see docstring

    def place_outward(a2d, b2d, dist_ac, dist_bc, away_from):
        p_plus = _place_third_point(a2d, b2d, dist_ac, dist_bc, sign=1.0)
        p_minus = _place_third_point(a2d, b2d, dist_ac, dist_bc, sign=-1.0)
        if np.linalg.norm(p_plus - away_from) > np.linalg.norm(p_minus - away_from):
            return p_plus
        return p_minus

    def rigid_transform_2pt(src_a, src_b, dst_a, dst_b):
        src_vec, dst_vec = src_b - src_a, dst_b - dst_a
        src_len, dst_len = np.linalg.norm(src_vec), np.linalg.norm(dst_vec)
        if src_len < 1e-9 or dst_len < 1e-9:
            return None
        cos_t = np.dot(src_vec, dst_vec) / (src_len * dst_len)
        sin_t = _cross2d(src_vec, dst_vec) / (src_len * dst_len)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return lambda p: dst_a + rot @ (p - src_a)

    centroid1 = np.mean(loop1_uv, axis=0)
    best = None
    # qa must be loop2's true correspondent of s1 (and qb of e1) — Qa is
    # hinge-placed relative to (Pa, Pb) as if it's near Pa, so pairing it
    # with the "wrong" end of the (s1, e1) edge produces a twisted,
    # self-intersecting placement even though the polygon-area/validity
    # check would otherwise seem like a fair way to pick among options.
    # `pairings` holds one entry (fixed by the caller) in manual mode, or
    # both "same"/"crossed" candidates in automatic mode — see docstring.
    for jj1, jj2 in pairings:
        for (s1, e1), (qa_idx2, qb_idx2) in (((i1, i2), (jj1, jj2)), ((i2, i1), (jj2, jj1))):
            Pa, Pb = loop1_uv[s1], loop1_uv[e1]
            Qa_3d, Qb_3d = p2_3d[qa_idx2], p2_3d[qb_idx2]
            d_pa_qa = float(np.linalg.norm(p1_3d[s1] - Qa_3d))
            d_pb_qa = float(np.linalg.norm(p1_3d[e1] - Qa_3d))
            d_pb_qb = float(np.linalg.norm(p1_3d[e1] - Qb_3d))
            d_qa_qb = float(np.linalg.norm(Qa_3d - Qb_3d))
            if d_qa_qb < 1e-6:
                continue  # degenerate: both loop2 anchors resolved to the same point

            qa_frame1 = place_outward(Pa, Pb, d_pa_qa, d_pb_qa, centroid1)
            qb_frame1 = place_outward(Pb, qa_frame1, d_pb_qb, d_qa_qb, centroid1)
            transform = rigid_transform_2pt(loop2_uv[qa_idx2], loop2_uv[qb_idx2], qa_frame1, qb_frame1)
            if transform is None:
                continue
            loop2_uv_frame1 = [transform(p) for p in loop2_uv]

            far1 = _loop_far_segment(s1, e1, n1)
            far2 = _loop_far_segment(qa_idx2, qb_idx2, n2)
            for seg2 in (far2, list(reversed(far2))):
                outline = np.array([loop1_uv[i] for i in far1] + [loop2_uv_frame1[i] for i in seg2])
                poly = Polygon(outline)
                if poly.is_valid and (best is None or poly.area > best[1]):
                    outline_vidx = [loop1_orig[i] for i in far1] + [loop2_orig[i] for i in seg2]
                    best = (outline, poly.area, outline_vidx)

    if best is None:
        raise ValueError("Could not form a valid outline connecting the two selections.")
    # Deliberately not reduced here either — same reasoning as
    # region_boundary_loop's docstring: this result becomes the *next*
    # connection's loop1 in a 3rd+-region chain, and that connection's own
    # automatic heuristic needs full resolution too. Reduce a display-only
    # copy at the point a corner-picker actually needs one instead (see
    # main.py's /unroll-mesh-chain), and the final assembled outline
    # separately via `simplify_outline_points` (fine there since nothing
    # downstream needs per-point vertex identity once the whole chain is
    # done).
    return best[0], best[2]


def uv_to_outline_array(uv: dict) -> np.ndarray:
    """Convenience: stack all placed (u,v) points into an Nx2 array
    (unordered — for the real feature, the *boundary* of the patch becomes
    the outline; this prototype just needs the point cloud for verification).
    """
    keys = sorted(uv.keys())
    return np.array([uv[k] for k in keys])
