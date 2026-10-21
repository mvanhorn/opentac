"""The 3D half of the manifest (v6): where this sensor sits on the part.

manifest v5 and earlier describe a flat drawing. A simulator needs the sensor
in 3D -- each taxel's position on the part's surface, its outward normal, and a
tangent -- so it can put the sensor on the finger it will actually be glued to
and report shear along an axis that means something to the firmware.

Everything here is an annotation on the user's own STL. OTC deliberately does
NOT emit MJCF, USD, or a scene: that is the same layering as Gerber vs. KiCad,
and a physics model belongs with the simulator that owns it. In particular no
stiffness is exported -- that is a modelling choice, not a design output.

Why the caller must hand us `source_outline_xy`
-----------------------------------------------
The unfold produces uv in an arbitrary flat frame; /generate then rotates the
board into the canonical cable-edge-down frame. That rotation is chosen from
the outline's FIRST TWO VERTICES, so it cannot be recovered from the mesh, or
from the boundary loop, or from the canonical outline alone -- reconstructing
the loop gives the right point set in the wrong starting order and silently
yields a sensor rotated on the part.

So the caller passes the exact array it posted to /generate, and we re-run the
same pure function on it. `verify_frame` then checks the result against the
manifest's own outline and reports the residual, so a mismatch is a number in
the export rather than a surprise in someone's simulator.
"""
import hashlib

import numpy as np

from opentac.cad.pipeline.geometry import apply_cable_transform, cable_transform
from opentac.cad.pipeline.sim_embed import build_embedding, project

# Beyond this, the outline the caller gave us is not the one that produced the
# board, and every projected point would be placed with the wrong rotation.
FRAME_TOL_MM = 0.05

# A taxel is expected to sit inside the unfolded patch. Past this it is off the
# selected faces entirely and has been clamped to the nearest triangle.
#
# This is also the SECOND frame check, and it catches what the first cannot.
# cable_transform is invariant to a global rigid motion of its input, so an
# outline handed over rotated or translated relative to the unfold's uv frame
# still satisfies verify_frame perfectly -- and then every taxel is placed
# somewhere else entirely on the part. Measured on a half-cylinder: a 90 degree
# frame rotation leaves outline_mm at 0.0000 and pushes this to 28 mm.
ON_PATCH_TOL_MM = 0.5


def mesh_sha256(stl_bytes: bytes) -> str:
    return hashlib.sha256(stl_bytes).hexdigest()


def _hausdorff(a, b):
    """Symmetric nearest-point distance between two point sets."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def verify_frame(source_outline_xy, manifest_outline_xy, tf) -> float:
    """How far the caller's outline lands from the board's actual outline.

    Zero means the outline we were handed is the one the board was generated
    from and the frame is right. This is the check that a silent rotation
    cannot survive.
    """
    if manifest_outline_xy is None or not len(manifest_outline_xy):
        return 0.0
    moved = apply_cable_transform(np.asarray(source_outline_xy, dtype=float), tf)
    return _hausdorff(moved, np.asarray(manifest_outline_xy, dtype=float))


def stack_from_bump_params(bp) -> dict:
    """The z-stack, straight off bump_sheet's own layer table.

    From the module docstring of pipeline/bump_sheet.py:

        z = 0                                   the part surface
        z = recess_depth                        top of the sensor board
        z = recess_depth + sheet_thickness      top of the sheet; bumps start
        z = ... + bump_height                   bump tops

    so a bump tip stands `recess + sheet + bump_height` proud of the part.
    Recorded rather than derived downstream, because getting it wrong puts
    every contact at the wrong height and nothing in the sim would flag it.
    """
    recess = float(bp.recess_depth_mm)
    sheet = float(bp.sheet_thickness_mm)
    bump = float(bp.bump_height_mm)
    return {
        "recess_mm": recess,          # the sensor (board + film) sits in here
        "sheet_mm": sheet,
        "bump_height_mm": bump,
        # Convenience, and the number a loader actually wants:
        "surface_to_sheet_top_mm": round(recess + sheet, 4),
        "surface_to_bump_tip_mm": round(recess + sheet + bump, 4),
    }


def build_sim3d(*, vertices, faces, face_normals, region_face_indices,
                start_face, source_outline_xy, manifest_outline_xy,
                pixel_xy, shear=None, bump_geom=None, stack=None,
                sha256=None, filename=None, units="mm", ndigits=4):
    """Return (sim3d_block, embedding_block).

    `pixel_xy` is the manifest's own `pixels` (index-parallel), `shear` its
    `shear` block. Both are already in the canonical frame, which is why the
    embedding is carried into that frame rather than the other way round.
    """
    tf = cable_transform(np.asarray(source_outline_xy, dtype=float))
    fit_mm = verify_frame(source_outline_xy, manifest_outline_xy, tf)

    emb = build_embedding(vertices, faces, face_normals,
                          region_face_indices, start_face, tf)

    warnings = []
    if fit_mm > FRAME_TOL_MM:
        warnings.append(
            f"source outline does not match the exported board ({fit_mm:.3f} mm): "
            "the 3D placement is probably rotated. Pass the outline that was "
            "posted to /generate.")

    def place(pts2d):
        if pts2d is None or not len(pts2d):
            return None, 0.0
        pos, nrm, tan, dist = project(emb, np.asarray(pts2d, dtype=float))
        return {
            "pos": np.round(pos, ndigits).tolist(),
            "normal": np.round(nrm, 6).tolist(),
            "tangent": np.round(tan, 6).tolist(),
        }, float(np.max(dist))

    taxel_xy = [[p[0], p[1]] for p in (pixel_xy or [])]
    taxels3d, taxel_off = place(taxel_xy)
    if taxels3d is not None:
        taxels3d["rc"] = [[int(p[2]), int(p[3])] for p in pixel_xy]

    bumps3d = None
    bump_off = 0.0
    bump_xy = (shear or {}).get("bumps") if shear else None
    if bump_xy:
        bumps3d, bump_off = place(bump_xy)
        if bumps3d is not None and bump_geom:
            bumps3d.update(bump_geom)

    worst_off = max(taxel_off, bump_off)
    if worst_off > ON_PATCH_TOL_MM:
        warnings.append(
            f"{worst_off:.2f} mm: at least one taxel or bump falls outside the "
            "selected faces and was clamped to the nearest triangle. Either the "
            "board reaches past the region that was unrolled, or the outline is "
            "in a different frame from the mesh selection.")

    sim3d = {
        "mesh": {
            "sha256": sha256,
            "filename": filename,
            "units": units,
            "faces": int(len(np.asarray(faces))),
        },
        # How well the flat drawing was reconciled with the 3D map. A reader
        # can decide whether to trust the placement instead of assuming it.
        "fit": {
            "outline_mm": round(fit_mm, 5),
            "worst_off_patch_mm": round(worst_off, 5),
            "region_faces": int(len(emb.tris)),
        },
        "stack": stack,
        "taxels3d": taxels3d,
    }
    if bumps3d is not None:
        sim3d["bumps3d"] = bumps3d
    if warnings:
        sim3d["warnings"] = warnings
    return sim3d, emb.as_json()
