"""Pick and unroll a surface of an STL part, without a 3D viewer.

The web app asks a human to click a triangle and sends its face index. That
index is the only thing the unroller needs, and nothing about choosing it
requires a browser: grow a region from every candidate seed, score the
results, and pick. Enumerating beats clicking — a click finds out whether a
surface flattens only after it has been made.
"""

from __future__ import annotations

import numpy as np


def load(path: str):
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    return mesh


def surfaces(mesh, angle_tol_deg: float = 2.0, min_faces: int = 8,
             max_regions: int = 12) -> list[dict]:
    """Enumerate the part's distinct flat surfaces, largest first.

    Each entry carries the seed face index the unroller needs, the region's
    area, and its flattening error in millimetres. A large error means real
    curvature in two directions — that surface will not lie flat.
    """
    from opentac.cad.pipeline.mesh_unroll import (grow_coplanar_region,
                                      check_developability)

    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, float)
    normals = np.asarray(mesh.face_normals, float)
    areas = np.asarray(mesh.area_faces, float)

    seen: set[frozenset] = set()
    found: list[dict] = []
    skipped: list[str] = []
    # Seed from the largest triangles first: they sit on the big flat faces.
    for seed in np.argsort(-areas):
        if len(found) >= max_regions:
            break
        region = grow_coplanar_region(faces, normals, int(seed),
                                      angle_tol_deg=angle_tol_deg,
                                      vertices=verts)
        if len(region) < min_faces:
            continue
        key = frozenset(region)
        if key in seen:
            continue
        seen.add(key)
        try:
            uv, visited, sub_verts, sub_faces = _unfold(verts, faces, region, int(seed))
            err = check_developability(sub_verts, sub_faces, uv, visited)
        except Exception as exc:
            # Never swallow this silently: a wrong call signature here looks
            # exactly like "this part has no flat surfaces".
            skipped.append(f"seed {int(seed)}: {type(exc).__name__}: {exc}")
            continue
        found.append({
            "seed_face": int(seed),
            "faces": len(region),
            "area_mm2": float(areas[list(region)].sum()),
            "flattening_error_mm": float(err),
            "normal": normals[seed].round(3).tolist(),
        })

    if not found and skipped:
        raise RuntimeError("No surface could be unfolded. Errors:\n  "
                           + "\n  ".join(skipped[:5]))
    found.sort(key=lambda r: -r["area_mm2"])
    return found


def unroll(mesh, seed_face: int, angle_tol_deg: float = 2.0,
           simplify_tol: float = 0.3) -> np.ndarray:
    """Flatten the surface containing seed_face into a 2D outline (mm).

    The result is an organic, densely sampled boundary — exactly the case
    where the cable edge rule bites. Always pass it through
    outline.with_flat_cable_edge() before generating.
    """
    from opentac.cad.pipeline.mesh_unroll import grow_coplanar_region, extract_outline
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, float)
    normals = np.asarray(mesh.face_normals, float)

    region = grow_coplanar_region(faces, normals, int(seed_face),
                                  angle_tol_deg=angle_tol_deg, vertices=verts)
    uv, _visited, _sv, sub_faces = _unfold(verts, faces, region, int(seed_face))
    return np.asarray(extract_outline(sub_faces, uv,
                                      simplify_tol=simplify_tol), float)


def _unfold(verts, faces, region, seed):
    """unfold_face_region() works on a renumbered submesh, and returns uv keyed
    by *local* vertex index with visited faces indexed into the local face
    list. check_developability() and extract_outline() must therefore be given
    the submesh, not the original arrays. Rebuilt here exactly as main.py's
    /unroll-mesh handler does it."""
    from opentac.cad.pipeline.mesh_unroll import unfold_face_region
    uv, visited_local, orig_idx = unfold_face_region(verts, faces, region, seed)
    region_list = [seed] + [f for f in region if f != seed]
    _, inv = np.unique(faces[region_list], return_inverse=True)
    return uv, visited_local, verts[orig_idx], inv.reshape(-1, 3)


def describe(regions: list[dict]) -> str:
    """One line per surface, in terms a user can choose from."""
    out = []
    for i, r in enumerate(regions):
        flat = ("flattens cleanly" if r["flattening_error_mm"] < 0.05
                else f"distorts {r['flattening_error_mm']:.1f}mm when flattened")
        out.append(f"{i}: {r['area_mm2']:.0f} mm^2, {r['faces']} triangles, {flat}")
    return "\n".join(out) or "No flat surfaces found."
