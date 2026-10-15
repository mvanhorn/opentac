"""Flat<->3D correspondence for the sim export (manifest v6).

The unfold already computes, per selected face region, where every mesh vertex
lands in the flat frame. That map is what turns the manifest from a 2D drawing
into a 3D sensor description: with it, a taxel centre picked on the flat board
can be put back onto the surface of the part it will be glued to.

Two rules this module exists to enforce:

  * **Record the map as the unfold runs; never reconstruct it afterwards.**
    unfold_face_region already returns it and the existing callers throw it
    away. The _edge_cuts_points docstring in export/manifest.py documents the
    same "derive it again later" mistake being made in four separate places.

  * **Everything here is in the CABLE frame.** uv arrives from the unfold in an
    arbitrary frame and is carried into the canonical frame by cable_transform
    before anything is located in it, so taxel centres (which are already
    canonical) can be looked up directly. Mixing the two frames produces a
    sensor that is silently rotated on the part.

The unfold is not injective -- a curved patch can overlap itself when
flattened -- so a query point may land in more than one triangle, or, near a
simplified outline edge, in none. project() resolves both and reports how far
it had to reach, so a caller can assert on it instead of trusting it.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import apply_cable_transform


@dataclass
class SurfaceEmbedding:
    """A triangulated flat<->3D correspondence over one selected face region.

    uv is in the cable frame (mm), xyz in the source mesh's own frame (mm),
    both indexed by the same vertex id. tris indexes into them; tri_normal and
    orig_face are parallel to tris.
    """
    tris: np.ndarray        # (T, 3) int
    uv: np.ndarray          # (N, 2) float, cable frame
    xyz: np.ndarray         # (N, 3) float, source mesh frame
    tri_normal: np.ndarray  # (T, 3) float, unit, source mesh frame
    orig_face: np.ndarray   # (T,)  int, index into the source mesh's faces

    def as_json(self, ndigits: int = 5) -> dict:
        return {
            "tris": self.tris.tolist(),
            "uv": np.round(self.uv, ndigits).tolist(),
            "xyz": np.round(self.xyz, ndigits).tolist(),
        }


def _unit(v, axis=-1):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def build_embedding(vertices, faces, face_normals, region_face_indices,
                    start_face, cable_tf) -> SurfaceEmbedding:
    """Unfold region_face_indices and keep the correspondence it produces.

    cable_tf comes from geometry.cable_transform() on the same outline the
    board was generated from; it carries the unfold's arbitrary flat frame into
    the canonical one the manifest uses.

    Face normals come from the caller (trimesh's face_normals) rather than a
    cross product here, so winding is handled once, in the loader that owns it.
    """
    from .mesh_unroll import unfold_face_region

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    face_normals = np.asarray(face_normals, dtype=float)

    # Same ordering unfold_face_region builds internally, so local face i is
    # original face region_list[i]. Recomputed rather than returned because
    # that signature is shared with the existing unroll endpoints.
    region_list = [int(start_face)] + [int(f) for f in region_face_indices
                                       if int(f) != int(start_face)]
    uv_map, visited_local, orig_vertex_index = unfold_face_region(
        vertices, faces, region_face_indices, start_face)

    region_faces_orig = faces[region_list]
    _, inv = np.unique(region_faces_orig, return_inverse=True)
    sub_faces = inv.reshape(-1, 3)

    # Only faces the unfold actually placed, and only those whose three
    # vertices all got a uv -- a disconnected island reaches neither.
    keep = [i for i in sorted(visited_local)
            if all(int(v) in uv_map for v in sub_faces[i])]
    if not keep:
        raise ValueError("unfold placed no usable triangles for this region")

    uv = np.full((len(orig_vertex_index), 2), np.nan)
    for v, p in uv_map.items():
        uv[int(v)] = p

    tris = sub_faces[keep]
    orig_face = np.array([region_list[i] for i in keep], dtype=int)

    return SurfaceEmbedding(
        tris=tris.astype(int),
        uv=apply_cable_transform(uv, cable_tf),
        xyz=vertices[orig_vertex_index],
        tri_normal=_unit(face_normals[orig_face]),
        orig_face=orig_face,
    )


def project(emb: SurfaceEmbedding, pts2d, tangent2d=(1.0, 0.0)):
    """Put flat points back on the surface.

    Returns (pos, normal, tangent, dist):
      pos      (P, 3) on the part surface, source mesh frame
      normal   (P, 3) unit, outward
      tangent  (P, 3) unit, the image of tangent2d under the local flat->3D
               map, re-orthogonalised against normal
      dist     (P,)   how far outside its triangle each point fell, in mm.
               Zero for an interior point. A caller should assert this is
               small; a large value means the point is not on the patch.

    tangent2d defaults to +x, which in the cable frame is the +column
    direction -- so "shear x" in the sim means the same axis the firmware's
    column index does. Without it a 3D shear vector has no relation to
    anything the board reports.
    """
    pts2d = np.atleast_2d(np.asarray(pts2d, dtype=float))
    a2, b2, c2 = (emb.uv[emb.tris[:, i]] for i in range(3))
    a3, b3, c3 = (emb.xyz[emb.tris[:, i]] for i in range(3))

    # Barycentric of every point against every triangle. Brute force: taxel
    # counts are in the hundreds and triangles in the thousands, so this is a
    # few million flops and needs no acceleration structure.
    v0 = b2 - a2                       # (T, 2)
    v1 = c2 - a2
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) > 1e-18         # drop slivers that unfolded degenerate

    v2 = pts2d[:, None, :] - a2[None, :, :]      # (P, T, 2)
    d20 = np.einsum("ptj,tj->pt", v2, v0)
    d21 = np.einsum("ptj,tj->pt", v2, v1)
    safe = np.where(ok, denom, 1.0)[None, :]
    v = (d11[None, :] * d20 - d01[None, :] * d21) / safe
    w = (d00[None, :] * d21 - d01[None, :] * d20) / safe
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=-1)          # (P, T, 3)
    bary[:, ~ok, :] = np.nan

    # How far outside the triangle, as a barycentric shortfall scaled by the
    # triangle's size -- 0 inside, growing with distance outside. Enough to
    # rank candidates and to report, without a true point-triangle distance.
    scale = np.sqrt(np.abs(denom))               # ~ 2 * area
    short = np.clip(-bary, 0.0, None).sum(axis=-1)
    outside = short * np.sqrt(np.where(ok, scale, np.inf))[None, :]
    outside = np.where(np.isnan(outside), np.inf, outside)

    best = np.argmin(outside, axis=1)            # (P,)
    rows = np.arange(len(pts2d))
    dist = outside[rows, best]

    bc = np.clip(bary[rows, best], 0.0, None)    # (P, 3)
    bc = bc / bc.sum(axis=1, keepdims=True)      # clamp onto the triangle

    A3, B3, C3 = a3[best], b3[best], c3[best]
    pos = bc[:, 0:1] * A3 + bc[:, 1:2] * B3 + bc[:, 2:3] * C3
    normal = _unit(emb.tri_normal[best])

    # Tangent: push tangent2d through each triangle's affine flat->3D map. M
    # maps barycentric edge coefficients to the flat frame; its inverse takes
    # the requested 2D direction to coefficients on the triangle's 3D edges.
    t2 = np.asarray(tangent2d, dtype=float)
    e0, e1 = v0[best], v1[best]                  # (P, 2) each, the flat edges
    # Analytic 2x2 inverse rather than np.linalg.solve: numpy 2.x reads a
    # (P, 2) right-hand side as a single system, not P of them, and this is
    # both faster and immune to that.
    det = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    det = np.where(np.abs(det) < 1e-18, 1.0, det)
    c0 = (e1[:, 1] * t2[0] - e1[:, 0] * t2[1]) / det
    c1 = (e0[:, 0] * t2[1] - e0[:, 1] * t2[0]) / det
    tangent = c0[:, None] * (B3 - A3) + c1[:, None] * (C3 - A3)

    # Re-orthogonalise: the affine map is exact per triangle, but a tangent and
    # a normal can still drift a hair apart after normalisation.
    tangent = tangent - normal * np.einsum("ij,ij->i", tangent, normal)[:, None]
    return pos, normal, _unit(tangent), dist
