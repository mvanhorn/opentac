"""Bump sheet: the printable shear-sensing overlay.

These tests cover properties that an STL can violate while still looking
correct in a viewer: a sheet that is not one connected solid, a sheet that is
missing, and bumps that are separate floating bodies.
"""

import numpy as np
import pytest
from fastapi import HTTPException
from shapely.geometry import Polygon


from opentac.cad.main import BumpSheetRequest, GenerateRequest, bump_sheet, generate
from opentac.cad.pipeline.bump_sheet import (
    BumpParams, _largest_polygon, _slit_geometry, build_bump_sheet,
    bump_centers_from_taxels, bump_index_per_point, sensor_body_polygon,
)

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]


def _generate(**kw):
    body = {"outline": SQUARE, "router": "hug", "board_mode": "expand",
            "pixel_w_mm": 4, "pixel_h_mm": 4, "pitch_x_mm": 4.2, "pitch_y_mm": 4.2}
    body.update(kw)
    return generate(GenerateRequest(**body)).edit_data


def _sheet(edit_data, params=None):
    resp = bump_sheet(BumpSheetRequest(edit_data=edit_data, params=params or {}))
    return {"stats": resp.stats, "warnings": resp.warnings, "stl_b64": resp.stl_b64}


# ---- grouping ----

def test_one_bump_per_complete_2x2_block():
    # 4x4 grid on a 2mm pitch -> exactly 4 blocks, no leftovers.
    pts = [[c * 2.0, r * 2.0] for r in range(4) for c in range(4)]
    centers, partial = bump_centers_from_taxels(pts, list(range(16)), 2.0, 2.0)
    assert len(centers) == 4
    assert partial == 0
    # Each centre sits at the middle of its block, not on a taxel.
    for cx, cy in centers:
        assert cx % 2.0 == pytest.approx(1.0)
        assert cy % 2.0 == pytest.approx(1.0)


def test_partial_block_still_gets_a_bump():
    """A 3x3 grid covers four blocks, only one of them complete — and all four
    get a bump. The three partial ones overhang into the space the missing
    taxels would have occupied, which is intended: the taxels that are present
    stay under bump material, so they keep resolving shear."""
    pts = [[c * 2.0, r * 2.0] for r in range(3) for c in range(3)]
    centers, partial = bump_centers_from_taxels(pts, list(range(9)), 2.0, 2.0)
    assert len(centers) == 4
    assert partial == 3


def test_bump_index_groups_each_block_of_four():
    """The grouping the manifest publishes: four taxels per bump, and each
    taxel assigned to the bump it sits under."""
    pts = [[c * 2.0, r * 2.0] for r in range(4) for c in range(4)]
    centers, _ = bump_centers_from_taxels(pts, list(range(16)), 2.0, 2.0)
    idx = bump_index_per_point(pts, centers, 2.0, 2.0)
    assert -1 not in idx, "every taxel of a full grid is under a bump"
    counts = {}
    for i in idx:
        counts[i] = counts.get(i, 0) + 1
    assert sorted(counts) == list(range(len(centers)))
    assert set(counts.values()) == {4}
    # Assigned to the nearest centre, not merely to some centre in range.
    for (x, y), i in zip(pts, idx):
        cx, cy = centers[i]
        assert abs(x - cx) == pytest.approx(1.0)
        assert abs(y - cy) == pytest.approx(1.0)


def test_taxel_with_no_bump_over_it_reports_minus_one():
    """A 2x2 block with no bump centre gets index -1: its taxels read
    pressure only and are not attached to a neighbouring bump."""
    pts = [[c * 2.0, r * 2.0] for r in range(4) for c in range(4)]
    centers, _ = bump_centers_from_taxels(pts, list(range(16)), 2.0, 2.0)
    survivors = np.asarray([c for c in centers if not (c[0] < 2 and c[1] < 2)])
    assert len(survivors) == len(centers) - 1, "expected to drop exactly one"
    idx = bump_index_per_point(pts, survivors, 2.0, 2.0)
    # The four taxels of the dropped block, and only those, are orphaned.
    orphans = {tuple(p) for p, i in zip(pts, idx) if i == -1}
    assert orphans == {(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)}


def test_partial_bump_stays_on_the_block_lattice():
    """A partial block's bump does not move onto whichever taxels are present.

    It sits where the complete block's bump would. Averaged over the present
    taxels instead, it would no longer straddle them and could not tip against
    them, and shear sensing depends on that tipping.
    """
    full = [[c * 2.0, r * 2.0] for r in range(2) for c in range(2)]
    lone = [[0.0, 0.0]]                      # same block, one taxel left
    fc, _ = bump_centers_from_taxels(full, list(range(4)), 2.0, 2.0)
    lc, partial = bump_centers_from_taxels(lone, [0], 2.0, 2.0)
    assert partial == 1
    # Both land on the block centre, offset half a pitch from the corner taxel.
    assert lc[0][0] == pytest.approx(1.0)
    assert lc[0][1] == pytest.approx(1.0)
    assert fc[0][0] == pytest.approx(lc[0][0])
    assert fc[0][1] == pytest.approx(lc[0][1])


def test_every_taxel_ends_up_under_a_bump():
    """On a generated board every active taxel lies under a bump; none is
    left pressure-only."""
    edit_data = _generate()
    placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    removed = set(edit_data.get("removed_pix") or [])
    active = [i for i in range(len(placements)) if i not in removed]
    pitch_x = edit_data["params"]["pitch_x_mm"]
    pitch_y = edit_data["params"]["pitch_y_mm"]
    centers, partial = bump_centers_from_taxels(placements, active, pitch_x, pitch_y)
    assert partial > 0, "this board should have part-filled blocks to exercise"
    # Every active taxel sits exactly half a pitch from its block's centre in
    # each axis, i.e. squarely inside the 2x2 footprint of some bump.
    for i in active:
        x, y = placements[i]
        assert min(max(abs(x - cx) / pitch_x, abs(y - cy) / pitch_y)
                   for cx, cy in centers) <= 0.5 + 1e-6


def test_grouping_ignores_pad_order():
    """Row/col indices come from position, never from the layout's net order.

    Pad order is deliberately non-geometric (the bottom half of the rows is
    reversed so the array meets the routing on the right side), so grouping by
    it would pair taxels that are nowhere near each other.
    """
    pts = [[c * 2.0, r * 2.0] for r in range(4) for c in range(4)]
    order = list(range(16))
    shuffled = order[8:] + order[:8]
    a, _ = bump_centers_from_taxels(pts, order, 2.0, 2.0)
    b, _ = bump_centers_from_taxels(pts, shuffled, 2.0, 2.0)
    assert sorted(map(tuple, np.round(a, 6))) == sorted(map(tuple, np.round(b, 6)))


# ---- geometry ----

def test_bump_is_oval_for_non_square_taxels():
    """Each axis is sized from its own 2x2 span, so the bump covers its block."""
    edit_data = _generate(pixel_w_mm=6, pixel_h_mm=3, pitch_x_mm=6.2, pitch_y_mm=3.2)
    s = _sheet(edit_data)["stats"]
    assert s["bump_base_x_mm"] == pytest.approx(0.8 * 2 * 6.2, abs=0.01)
    assert s["bump_base_y_mm"] == pytest.approx(0.8 * 2 * 3.2, abs=0.01)
    assert s["bump_base_x_mm"] > s["bump_base_y_mm"]
    # ...and the taper is applied per axis too, not as one scalar radius.
    assert s["bump_top_x_mm"] / s["bump_base_x_mm"] == pytest.approx(0.7, abs=0.01)
    assert s["bump_top_y_mm"] / s["bump_base_y_mm"] == pytest.approx(0.7, abs=0.01)


def test_square_taxels_give_a_round_bump():
    s = _sheet(_generate())["stats"]
    assert s["bump_base_x_mm"] == pytest.approx(s["bump_base_y_mm"])


def test_sheet_is_one_connected_solid():
    """The tabs hold every tile on and the bumps join the sheet with overlap,
    so the sheet is a single watertight body. A sheet in several pieces, or
    bumps as separate bodies, would give an STL that looks correct in a
    viewer."""
    s = _sheet(_generate())["stats"]
    assert s["bodies"] == 1
    assert s["watertight"] is True


def test_sheet_clears_the_cable_tail():
    """The connector has to stay reachable, so the sheet stops at the cable edge."""
    edit_data = _generate()
    cable_y = float(np.asarray(edit_data["outline"], dtype=float)[:2, 1].min())
    body = sensor_body_polygon(edit_data["final_outline"], cable_y)
    assert body.bounds[1] >= cable_y - 1e-6
    full = Polygon(np.asarray(edit_data["final_outline"], dtype=float)).buffer(0)
    assert body.area < full.area          # the tail is removed


def test_no_material_below_the_cable_edge():
    """The recess runs out to the cable edge, with no skirt across it.

    Buffering the body for the pocket and again for the skirt, without clipping
    back, would leave a wall a full skirt width plus tolerance thick below the
    cable edge, where the tail and connector leave the board. Such a part looks
    like a closed ring in a viewer and cannot be fitted.
    """
    import trimesh
    edit_data = _generate()
    placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    removed = set(edit_data.get("removed_pix") or [])
    active = [i for i in range(len(placements)) if i not in removed]
    cable_y = float(np.asarray(edit_data["outline"], dtype=float)[:2, 1].min())
    result = build_bump_sheet(edit_data["final_outline"], cable_y, placements, active,
                              edit_data["params"]["pitch_x_mm"],
                              edit_data["params"]["pitch_y_mm"], BumpParams())
    mesh = result.mesh
    lo, hi = mesh.bounds
    assert lo[1] >= cable_y - 1e-3, "sheet material hangs past the cable edge"

    # Checked by volume as well as by the bounding box.
    probe = trimesh.creation.box(
        extents=[hi[0] - lo[0] + 4, 6.0, hi[2] - lo[2] + 4])
    probe.apply_translation([(lo[0] + hi[0]) / 2, cable_y - 3.0,
                             (lo[2] + hi[2]) / 2])
    assert trimesh.boolean.intersection([mesh, probe]).volume == pytest.approx(0.0, abs=1e-6)


def test_recess_is_hollow():
    """The pocket the sensor sits in is empty below the sheet: a solid there
    would look identical from outside and leave no room for the sensor."""
    import trimesh
    edit_data = _generate()
    placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    removed = set(edit_data.get("removed_pix") or [])
    active = [i for i in range(len(placements)) if i not in removed]
    cable_y = float(np.asarray(edit_data["outline"], dtype=float)[:2, 1].min())
    p = BumpParams()
    result = build_bump_sheet(edit_data["final_outline"], cable_y, placements, active,
                              edit_data["params"]["pitch_x_mm"],
                              edit_data["params"]["pitch_y_mm"], p)
    lo, hi = result.mesh.bounds
    # A slab spanning the whole footprint, from the bottom up to the recess
    # ceiling. Only the skirt ring may intersect it.
    probe = trimesh.creation.box(
        extents=[hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, p.recess_depth_mm - 0.05])
    probe.apply_translation([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                             (p.recess_depth_mm - 0.05) / 2])
    solid_below = trimesh.boolean.intersection([result.mesh, probe]).volume
    # The skirt ring's own volume is the expected answer; a filled recess would
    # be several times larger. Compare against the footprint area rather than a
    # hard-coded number so the test survives a parameter change.
    footprint = (hi[0] - lo[0]) * (hi[1] - lo[1])
    assert solid_below < 0.5 * footprint * (p.recess_depth_mm - 0.05)
    assert solid_below > 0            # the skirt is there


def test_perimeter_bumps_are_cut_free_like_interior_ones():
    """The outward side of a border tile is slit as much as an inward one.

    Cutting only between bump centres, and relying on a slit at the board edge
    for the rest, would leave every perimeter bump welded to the margin of
    sheet around the array. Such a bump barely moves, so it reads little shear,
    and nothing about the STL looks wrong.
    """
    from shapely.geometry import LineString

    edit_data = _generate()
    placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    removed = set(edit_data.get("removed_pix") or [])
    active = [i for i in range(len(placements)) if i not in removed]
    cable_y = float(np.asarray(edit_data["outline"], dtype=float)[:2, 1].min())
    px = edit_data["params"]["pitch_x_mm"]
    py = edit_data["params"]["pitch_y_mm"]

    p = BumpParams()
    body = sensor_body_polygon(edit_data["final_outline"], cable_y)
    recess = _largest_polygon(body.buffer(p.recess_tolerance_mm))
    centers, _ = bump_centers_from_taxels(placements, active, px, py)
    slits = _slit_geometry(recess, centers, px, py, p, open_edge_y=cable_y)

    known = {(round(c[0], 3), round(c[1], 3)) for c in centers}
    outward = []
    for cx, cy in centers:
        for seg, (dx, dy) in (
            (LineString([(cx - px, cy - py), (cx - px, cy + py)]), (-1, 0)),
            (LineString([(cx + px, cy - py), (cx + px, cy + py)]), (1, 0)),
            (LineString([(cx - px, cy - py), (cx + px, cy - py)]), (0, -1)),
            (LineString([(cx - px, cy + py), (cx + px, cy + py)]), (0, 1)),
        ):
            if (round(cx + dx * 2 * px, 3), round(cy + dy * 2 * py, 3)) in known:
                continue                       # faces another bump, not outward
            live = seg.intersection(recess).length     # edge with material behind it
            if live < 0.05:
                continue                       # nothing out there to cut
            outward.append(min(seg.intersection(slits).length / live, 1.0))

    assert outward, "this board should have perimeter bumps to exercise"
    # Only the tabs should interrupt an outward edge, exactly as inward ones.
    assert min(outward) > 0.5, f"a perimeter tile is barely cut free: {min(outward):.2f}"
    assert sum(outward) / len(outward) > 0.65


def test_a_board_that_needs_crack_repair_builds():
    """The fill_holes() branch builds a solid, not only the clean-mesh path.

    Every other board here comes out watertight at once, so only this one runs
    the repair branch, which needs networkx; without networkx, only boards that
    need repair would fail. This configuration reliably leaves a hairline crack
    for the boolean to close.
    """
    edit_data = _generate(outline=[[-25, -15], [25, -15], [25, 15], [-25, 15]],
                          pixel_w_mm=8, pixel_h_mm=2.5,
                          pitch_x_mm=8.2, pitch_y_mm=2.7)
    data = _sheet(edit_data)
    assert data["stats"]["bumps"] > 0
    assert data["stats"]["bodies"] == 1
    assert data["stats"]["volume_mm3"] > 0
    # Repair must never be fatal: a crack it cannot close is a warning, and the
    # caller still gets a solid back.
    assert not any("Could not repair" in w for w in data["warnings"]), \
        "hole repair failed outright — is networkx installed?"


# ---- endpoint ----

def test_endpoint_returns_a_parseable_binary_stl():
    import base64, struct
    data = _sheet(_generate())
    raw = base64.b64decode(data["stl_b64"])
    assert len(raw) > 84
    (count,) = struct.unpack("<I", raw[80:84])
    assert count == data["stats"]["triangles"]
    assert len(raw) == 84 + count * 50


def test_params_change_the_geometry():
    edit_data = _generate()
    base = _sheet(edit_data)["stats"]
    tall = _sheet(edit_data, {"bump_height_mm": 5.0, "skirt_width_mm": 8.0})["stats"]
    assert tall["total_height_mm"] > base["total_height_mm"]
    assert tall["sheet_w_mm"] == pytest.approx(base["sheet_w_mm"] + 10.0, abs=0.2)


def test_tab_count_changes_the_mesh():
    """tabs_per_edge, a BumpParams field set from the UI, reaches the mesh."""
    edit_data = _generate()
    one = _sheet(edit_data, {"tabs_per_edge": 1})["stats"]
    three = _sheet(edit_data, {"tabs_per_edge": 3})["stats"]
    assert three["tabs_per_edge"] == 3
    # More tabs means more material bridging the slits, and the sheet remains
    # a single connected solid.
    assert three["volume_mm3"] > one["volume_mm3"]
    assert three["bodies"] == 1


def test_fit_tolerance_widens_the_pocket():
    edit_data = _generate()
    tight = _sheet(edit_data, {"recess_tolerance_mm": 0.1})["stats"]
    loose = _sheet(edit_data, {"recess_tolerance_mm": 0.6})["stats"]
    # The pocket grows outward, so the whole sheet grows with it.
    assert loose["sheet_w_mm"] == pytest.approx(tight["sheet_w_mm"] + 1.0, abs=0.2)
    assert loose["recess_tolerance_mm"] == pytest.approx(0.6)


def test_bump_proportions_are_tunable():
    edit_data = _generate()
    base = _sheet(edit_data, {"bump_base_frac": 0.8, "bump_top_ratio": 0.7})["stats"]
    wide = _sheet(edit_data, {"bump_base_frac": 0.9, "bump_top_ratio": 0.4})["stats"]
    assert wide["bump_base_x_mm"] > base["bump_base_x_mm"]
    assert wide["bump_top_x_mm"] / wide["bump_base_x_mm"] == pytest.approx(0.4, abs=0.01)
    assert wide["bump_base_pct"] == pytest.approx(90.0)
    assert wide["bump_top_pct"] == pytest.approx(40.0)


def test_touching_bumps_are_warned_about():
    """At 100% of the block the bumps merge, which defeats the slits while the
    sheet still builds and looks correct in a viewer, so the user is warned."""
    data = _sheet(_generate(), {"bump_base_frac": 1.0})
    assert any("no longer move independently" in w for w in data["warnings"])


def test_flaring_bump_is_warned_about():
    data = _sheet(_generate(), {"bump_top_ratio": 1.4})
    assert any("support to print" in w for w in data["warnings"])


def test_unknown_param_keys_are_ignored():
    stats = _sheet(_generate(), {"not_a_real_param": 7})["stats"]
    assert stats["bump_height_mm"] == pytest.approx(2.0)


def test_missing_edit_data_is_a_400_not_a_crash():
    with pytest.raises(HTTPException) as exc:
        bump_sheet(BumpSheetRequest(edit_data={}))
    assert exc.value.status_code == 400
    assert "generate a board first" in exc.value.detail
