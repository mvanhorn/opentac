"""Hug-router fan-comb tests: every connector wire must START with a vertical
rise at its own pad's x — including pads that sit beyond the sensor field's
footprint ("spilled" pads: a wide connector under a shape that narrows toward
the tail). The old code silently fell back to a nearest-point diagonal for
spilled pads, drawing wires that crossed every neighbor's vertical stub —
seen live as row/row DRC crossings on U-shaped wraps with 3mm pixels, clean
at 4mm only because the smaller 20-pos connector still fit under the field.
"""

import sys
import os

import numpy as np
from shapely.geometry import LineString, Point

from opentac.cad.pipeline.route_hug import _pad_entry
from opentac.cad.main import generate, GenerateRequest

# The exact strut-joined fingertip wrap band (finger_tip_soft.stl, flat side +
# curved contact face + flat side) that reproduced the failure: 30 violations
# at 3mm pixels before the fix, DRC 0 at 4mm either way. Cable edge is 0->1.
WRAP_BAND = [
    [-24.193, -11.859], [-16.470, -11.935], [-16.076, -11.499],
    [16.156, -8.934], [16.604, -9.315], [24.193, -8.317],
    [21.507, 11.935], [6.329, 10.264], [-7.515, 9.141], [-24.091, 8.501],
]

# A clean synthetic upside-down U — the control that was never broken.
ARCH = [
    [0, 0], [18, 0], [18, 25], [42, 25], [42, 0], [60, 0], [60, 40], [0, 40],
]

# A second real chain outline (side + curved wrap + side, corners picked in
# the guided flow): its near-degenerate waist at x~14.6 splits the packed
# sensor field into ISLANDS at several pixel sizes — the live "F.Cu
# crossings in grow mode" report. Cable edge is 0->1.
GUIDED_WRAP = [
    [16.595, -10.221], [24.250, -10.216], [24.214, 10.214], [14.639, 10.216],
    [-24.147, 10.221], [-24.250, -10.139], [14.636, -9.784],
]


class _H:
    def __init__(self, xy):
        self.a_xy = np.asarray(xy, dtype=float)


def test_pad_entry_vertical_hit_and_spill():
    """Unit contract: ray hit -> (first point above, no elbow); ray miss
    (pad beyond the corridor footprint) -> (the corridor's extreme point on
    the pad's side — closest x, lowest such — plus an elbow at the pad's
    own x) so the caller builds an L-shaped comb stub whose horizontal run
    can never reach an interior pad's vertical."""
    ring = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])

    entry, elbow = _pad_entry(ring, _H([5.0, -8.0]))
    assert elbow is None
    assert abs(entry.x - 5.0) < 1e-9 and abs(entry.y - 0.0) < 1e-9, \
        "pad under the corridor must enter straight up at its own x"

    entry, elbow = _pad_entry(ring, _H([-4.0, -8.0]))
    assert elbow is not None, "pad beyond the corridor must get an elbow"
    assert abs(elbow[0] - (-4.0)) < 1e-9, "elbow must sit at the pad's own x"
    assert abs(elbow[1] - entry.y) < 1e-9, "elbow->entry run must be horizontal"
    assert abs(entry.x - 0.0) < 0.05 and abs(entry.y - 0.0) < 0.05, \
        "entry must be the corridor's closest-x, lowest point on the pad's side"

    entry, elbow = _pad_entry(ring, _H([14.0, -8.0]))
    assert elbow is not None and abs(elbow[0] - 14.0) < 1e-9
    assert abs(entry.x - 10.0) < 0.05 and abs(entry.y - 0.0) < 0.05, \
        "right-side spill must mirror: rightmost, lowest corridor point"
    print("test_pad_entry_vertical_hit_and_spill: OK")


def _gen(outline, px, mode="expand"):
    return generate(GenerateRequest(
        outline=outline, router="hug", board_mode=mode,
        pixel_w_mm=px, pixel_h_mm=px, pitch_x_mm=px + 0.2, pitch_y_mm=px + 0.2))


def _assert_vertical_starts(resp, label):
    """Every pad hop's first drawn segment must be vertical (the comb)."""
    routes = resp.edit_data["routes"]
    checked = 0
    for r in routes:
        poly = np.asarray(r["polyline_xy"], dtype=float)
        pad = np.asarray(r["pad_xy"], dtype=float)
        if len(poly) < 2 or np.linalg.norm(poly[0] - pad) > 0.5:
            continue                      # link hops start at a field terminal
        dx = abs(poly[1][0] - poly[0][0])
        rise = poly[1][1] - poly[0][1]
        assert dx < 0.06, (
            f"{label}: {r['net_name']} leaves its pad diagonally "
            f"(dx={dx:.2f}, first segment {poly[0]} -> {poly[1]})")
        assert rise > 0, f"{label}: {r['net_name']} first segment does not rise"
        checked += 1
    assert checked >= 4, f"{label}: expected several pad hops, checked {checked}"


def test_wrap_band_3mm_routes_clean():
    """The live failure case. Also asserts the spilled-pad path actually ran
    (some pad hop turns horizontally right after its vertical rise) so this
    test cannot silently pass by the geometry drifting out of the spill
    regime."""
    resp = _gen(WRAP_BAND, 3.0)
    assert resp.drc["violations"] == 0, \
        f"wrap band @3mm: {resp.drc['violations']} DRC violations: {resp.drc['details'][:4]}"
    _assert_vertical_starts(resp, "wrap band @3mm")
    elbows = 0
    for r in resp.edit_data["routes"]:
        poly = np.asarray(r["polyline_xy"], dtype=float)
        pad = np.asarray(r["pad_xy"], dtype=float)
        if len(poly) < 3 or np.linalg.norm(poly[0] - pad) > 0.5:
            continue
        v = abs(poly[1][0] - poly[0][0]) < 0.06 and poly[1][1] > poly[0][1]
        h = abs(poly[2][1] - poly[1][1]) < 0.06 and abs(poly[2][0] - poly[1][0]) > 0.2
        if v and h:
            elbows += 1
    assert elbows >= 1, "no L-shaped comb stub found — the spill regime was not exercised"
    print(f"test_wrap_band_3mm_routes_clean: OK "
          f"({resp.stats['total_pixels']} px, {elbows} spilled-pad stubs)")


def test_wrap_band_4mm_still_clean():
    resp = _gen(WRAP_BAND, 4.0)
    assert resp.drc["violations"] == 0
    _assert_vertical_starts(resp, "wrap band @4mm")
    print(f"test_wrap_band_4mm_still_clean: OK ({resp.stats['total_pixels']} px)")


def test_arch_3mm_still_clean():
    resp = _gen(ARCH, 3.0)
    assert resp.drc["violations"] == 0
    _assert_vertical_starts(resp, "arch @3mm")
    print(f"test_arch_3mm_still_clean: OK ({resp.stats['total_pixels']} px)")


def test_wrap_band_fixed_mode_clean():
    """Fixed-outline mode on the band, DRC 0 at both pixel sizes. Three
    mechanisms are pinned here, each found live on this exact shape:

    - No net-to-net violations (the spilled-pad L-stub fix covers this
      mode too — was 84 violations at 3mm, 41 at 4mm).
    - No phantom-seam edge flags: board_poly used to union the cable rect
      edge-touching against the outline, leaving a zero-width seam that
      the exterior ring retraced — interior row wires crossing into the
      tail flagged as 0.04-0.12mm edge violations against a boundary that
      does not exist on the exported board (fixed: 1um overlap).
    - No notch flags: the connector-width tail (10.5mm) is wider than the
      7.72mm cable edge, and the outline's bottom edge slopes up past its
      corner — the wedge between them belonged to neither shape, and the
      outermost fan stubs crossed ~0.4mm of open air through it (fixed:
      shoulder fill in layout_with_connector joins the outline bottom to
      the tail).
    """
    for px in (3.0, 4.0):
        resp = _gen(WRAP_BAND, px, mode="fixed_keepout")
        assert resp.drc["violations"] == 0, \
            f"wrap band fixed @{px}mm: {resp.drc['violations']} violations: {resp.drc['details'][:4]}"
    print("test_wrap_band_fixed_mode_clean: OK (3mm and 4mm both DRC 0)")


def test_island_field_routes_clean_across_sizes():
    """The split-field (island) saga, pinned end to end on the outline that
    reported it. At several pixel sizes the guided wrap's waist admits no
    sensor cell, so the packed field breaks into islands. Three mechanisms
    must cooperate for DRC 0 with no sensors lost:

    - field_polygon bridges each island into ONE hug contour (pitch-wide
      corridor — a thin filament leaves the contour's two sides ambiguous
      and wires hairpin the wrong way around);
    - column pad-hop directions are assigned as a coherent SPLIT of the pad
      row, not per-wire "shorter way around" (with the contour cut landing
      on an island right over the connector, per-wire choices interleaved
      directions across the pad row and the two bundles braided — the
      original F.Cu crossings);
    - routing order measures each wire's real run |span -/+ pad offset|
      from the pad's x directly, never by projecting the pad onto the ring
      (an island-bridge lobe can capture the projection and invert the
      peel order — see analyze() in route_hug.py).

    On top sits the hard emission invariant (route_face_wires): a wire that
    would cross or under-clear previously routed copper fails its net and
    is never emitted — so if some future topology defeats the plan again,
    it costs sensors (reported), never shipped crossings. Sensor counts
    are asserted so silent net-dropping can't fake a pass here.

    Historical note: the guided wrap itself no longer splits at any of
    these sizes — its "islands" were always cells touching at a CORNER,
    which the 5µm cell weld in field_polygon now fuses (the point-touch
    used to get a pitch/2 bridge disc, the half-circle "bubble" bulging
    out of flat field edges). The bridge machinery is still needed for
    REAL gaps, so it is exercised directly on a genuinely split field
    below instead of hoping an outline splits.
    """
    expected_min_sensors = {2.8: 70, 3.0: 65, 3.2: 52, 3.5: 44, 4.0: 30}
    outcomes = []
    for px, min_sensors in expected_min_sensors.items():
        resp = _gen(GUIDED_WRAP, px)
        assert resp.drc["violations"] == 0, \
            f"guided wrap @{px}mm: {resp.drc['violations']} violations: {resp.drc['details'][:4]}"
        assert resp.stats["active_pixels"] >= min_sensors, \
            f"guided wrap @{px}mm: sensors dropped ({resp.stats['active_pixels']} < {min_sensors})"
        if resp.stats.get("hug_fallback"):
            # A size hug's ordering can't yet solve cleanly: the safety
            # policy shipped the CLEAN legacy board instead (full sensor
            # count asserted above). The comb-shape assertion below only
            # applies to hug-routed geometry.
            outcomes.append(f"{px}:legacy")
        else:
            _assert_vertical_starts(resp, f"guided wrap @{px}mm")
            outcomes.append(f"{px}:hug")

    # The island machinery itself, on a REAL split: two clusters 6mm apart
    # must weld into ONE contour via a pitch-wide capsule, with the warning.
    import numpy as np
    from opentac.cad.pipeline import route_hug as rh
    from opentac.cad.pipeline.types import Params

    class _L:
        placements_xy = np.array(
            [[x, y] for x in (0.0, 3.0) for y in (0.0, 3.0)]
            + [[x + 9.0, y] for x in (0.0, 3.0) for y in (0.0, 3.0)])
    params = Params(pixel_w_mm=2.8, pixel_h_mm=2.8, pitch_x_mm=3.0, pitch_y_mm=3.0)
    warns = []
    fp = rh.field_polygon(_L, params, warns)
    assert fp.geom_type == "Polygon", "real islands were not bridged into one contour"
    assert any("island" in w for w in warns), "island bridge did not warn"
    # And a corner-touch must NOT be treated as an island (the bubble bug):
    class _T:
        placements_xy = np.array([[0.0, 0.0], [3.0, 3.0]])
    warns2 = []
    fp2 = rh.field_polygon(_T, params, warns2)
    assert fp2.geom_type == "Polygon" and not warns2, \
        f"corner-touching cells still flagged as islands: {warns2}"
    print(f"test_island_field_routes_clean_across_sizes: OK ({', '.join(outcomes)}; "
          "all DRC 0, no sensors lost; real gap bridges, corner-touch welds)")


# A live-drawn octagon whose FIRST edge (the cable edge) is an interior
# chord: the outline continues 26mm below it, so the connector sits in a
# bay with sensors on three sides. Neither router orders that topology
# cleanly today; the contract is honesty, not silence.
BAY_OCTAGON = [
    [14, -1], [-13, -1], [-29, -9], [-29, 11], [-15, 23], [15, 23], [30, 12], [30, -9],
]
# The same shape with the cable edge on the true outer boundary.
OCTAGON_OUTER = BAY_OCTAGON[7:] + BAY_OCTAGON[:7]


def test_bay_cable_edge_warns_and_never_ships_crossings():
    """The live "0 sensors, WHAT IS GOING ON" board (gapless 3.0/3.0 pitch,
    connector in a bay). Contract: whatever ships is DRC 0 (crossings can
    never ship — hard emission invariant + clean-legacy-only fallback), and
    the response TELLS the user the cable edge is interior and where to put
    it. Moving the cable edge to the outer boundary must then route the
    full board — same shape, same gapless pitch."""
    bay = generate(GenerateRequest(outline=BAY_OCTAGON, router="hug", board_mode="expand",
                                   pixel_w_mm=3.0, pixel_h_mm=3.0,
                                   pitch_x_mm=3.0, pitch_y_mm=3.0))
    assert bay.drc["violations"] == 0, f"bay board shipped violations: {bay.drc['details'][:4]}"
    warns = bay.stats.get("hug_plan_warnings", [])
    assert any("interior edge" in w for w in warns), \
        f"no interior-cable-edge warning surfaced: {warns[:3]}"

    outer = generate(GenerateRequest(outline=OCTAGON_OUTER, router="hug", board_mode="expand",
                                     pixel_w_mm=3.0, pixel_h_mm=3.0,
                                     pitch_x_mm=3.0, pitch_y_mm=3.0))
    assert outer.drc["violations"] == 0
    assert outer.stats["active_pixels"] >= 100, \
        f"outer-edge octagon should route fully, got {outer.stats['active_pixels']}"
    assert not outer.stats.get("hug_fallback"), "outer-edge octagon should not need fallback"
    print(f"test_bay_cable_edge_warns_and_never_ships_crossings: OK "
          f"(bay: {bay.stats['active_pixels']} px + warning; outer: {outer.stats['active_pixels']} px)")


SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
T_PRESET = [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]]


def test_plain_shapes_full_sensors_across_pitches():
    """The 'even a simple square fails at 3mm' regression, pinned. Plain
    shapes must route EVERY packed sensor with DRC 0 across pixel/pitch
    combos, in grow mode, on the hug router itself (no fallback). The
    regression came from measuring routing order along the assigned
    direction unconditionally: near the contour cut a pad can project a
    hair past its own anchor, the directional distance then wraps the
    whole ring, an innermost wire routes dead last across its neighbors,
    and the refusal cascade erases the board (1 of 169 sensors survived).
    Ordering now measures |span -/+ pad offset| computed from the pad's x
    (see analyze() in route_hug.py); the absolute value keeps the
    pad-beyond-anchor wires innermost, which is what this test pins.
    Full-count assertions make silent sensor loss impossible to miss here.
    """
    for name, outline in (("square", SQUARE), ("T", T_PRESET)):
        for px, pitch in ((3.0, 3.0), (3.0, 3.2), (4.0, 4.2)):
            resp = generate(GenerateRequest(outline=outline, router="hug", board_mode="expand",
                                            pixel_w_mm=px, pixel_h_mm=px,
                                            pitch_x_mm=pitch, pitch_y_mm=pitch))
            assert resp.drc["violations"] == 0, \
                f"{name} px{px}/pitch{pitch}: {resp.drc['violations']} violations"
            assert resp.stats["active_pixels"] == resp.stats["total_pixels"], \
                (f"{name} px{px}/pitch{pitch}: sensors lost "
                 f"({resp.stats['active_pixels']}/{resp.stats['total_pixels']})")
            assert not resp.stats.get("hug_fallback"), \
                f"{name} px{px}/pitch{pitch}: plain shape should not need fallback"
    print("test_plain_shapes_full_sensors_across_pitches: OK "
          "(square + T, 3.0/3.0 3.0/3.2 4.0/4.2, all full-count DRC 0)")


# The dense-board family from the 120-case grow-mode sweep that defeated
# every earlier ordering metric (and that BASELINE shipped as real copper
# crossings: 790 violations on the L, 2355 on the C, 197 on the T at
# px2.5/pitch2.7). Each pins a distinct mechanism of the final eff formula.
L_PRESET = [[-30, 0], [30, 0], [30, 16], [-6, 16], [-6, 50], [-30, 50]]
RECT_WIDE_PRESET = [[-25, -15], [25, -15], [25, 15], [-25, 15]]


def test_dense_boards_full_sensors_no_fallback():
    """The px2.5/pitch2.7 sweep failures, pinned on the hug router itself.

    - L @2.5/2.7: the packed field splits into islands at the arm junction
      and the bridge lobe used to CAPTURE the row pads' ring projections
      (ROW_16's pad projected 25mm off its overhead point), inverting the
      left row bundle's peel order — 14 braided row pairs, 21/196 sensors
      after refusals. eff is now computed from the pad's x, immune to
      contour features.
    - T @2.5/2.7: stem cols whose pads sit BEYOND their anchors must peel
      by |run|, not signed run — the signed order routes the longest such
      wire first and seals its neighbors' anchors under its lane.
    - rect @3.0/3.2: same annulus mechanism at a milder pitch.

    All must route every packed sensor, DRC 0, natively (no fallback).
    """
    for name, outline, px, pitch in (("L", L_PRESET, 2.5, 2.7),
                                     ("T", T_PRESET, 2.5, 2.7),
                                     ("rect", RECT_WIDE_PRESET, 3.0, 3.2)):
        resp = generate(GenerateRequest(outline=outline, router="hug", board_mode="expand",
                                        pixel_w_mm=px, pixel_h_mm=px,
                                        pitch_x_mm=pitch, pitch_y_mm=pitch))
        assert resp.drc["violations"] == 0, \
            f"{name} px{px}/pitch{pitch}: {resp.drc['violations']} violations: {resp.drc['details'][:3]}"
        assert resp.stats["active_pixels"] == resp.stats["total_pixels"], \
            (f"{name} px{px}/pitch{pitch}: sensors lost "
             f"({resp.stats['active_pixels']}/{resp.stats['total_pixels']})")
        assert not resp.stats.get("hug_fallback"), \
            f"{name} px{px}/pitch{pitch}: should route natively on hug"
    print("test_dense_boards_full_sensors_no_fallback: OK "
          "(L + T @2.5/2.7, rect @3.0/3.2, all full-count DRC 0, no fallback)")


def test_exported_edge_cuts_contour_is_closed():
    """The exported Edge.Cuts (board gr_lines + connector footprint
    fp_lines/fp_arcs) must form a closed contour in BOTH board modes:
    every snapped endpoint appears an even number of times. Fixed mode
    used to take a legacy export branch whose tail walls stopped at
    cable_length while the connector outline starts CONNECTOR_BODY_ABOVE
    deeper — a 2.75mm open gap per side in the gerber. All exports now go
    through stitched_outline_segments whenever a final outline exists.
    """
    import base64
    import io as _io
    import re
    import zipfile
    from collections import Counter

    num = r"[-\d.]+"
    for mode in ("fixed_keepout", "expand"):
        resp = _gen(WRAP_BAND, 3.0, mode=mode)
        pcb = zipfile.ZipFile(_io.BytesIO(base64.b64decode(resp.zip_b64))).read("board.kicad_pcb").decode()
        segs = []
        for m in re.finditer(
                rf'\(gr_line \(start ({num}) ({num})\) \(end ({num}) ({num})\) \(layer "Edge.Cuts"\)', pcb):
            a, b, c, d = map(float, m.groups())
            segs.append(((a, b), (c, d)))
        for line in pcb.splitlines():
            st = line.strip()
            for kind in ("(fp_line", "(fp_arc"):
                if st.startswith(kind) and "Edge.Cuts" in st:
                    m = re.match(rf'\({kind[1:]} \(start ({num}) ({num})\)(?: \(mid ({num}) ({num})\))? '
                                 rf'\(end ({num}) ({num})\)', st)
                    if m:
                        g = m.groups()
                        segs.append(((float(g[0]), float(g[1])), (float(g[4]), float(g[5]))))
        assert len(segs) > 10, f"{mode}: suspiciously few Edge.Cuts segments ({len(segs)})"
        deg = Counter()
        for a, b in segs:
            for p in (a, b):
                deg[(round(p[0] / 0.05), round(p[1] / 0.05))] += 1
        dangles = [(k[0] * 0.05, k[1] * 0.05) for k, n in deg.items() if n % 2 == 1]
        assert not dangles, f"{mode}: Edge.Cuts contour open at {dangles[:6]}"
    print("test_exported_edge_cuts_contour_is_closed: OK (both modes)")


# A rows-cut shape: the notch opens onto the cable edge, so rows (not columns)
# are split into segments joined by link traces.
U_NOTCH = [[-34, 0], [-12, 0], [-12, 34], [12, 34], [12, 0], [34, 0],
           [34, 56], [-34, 56]]


def test_row_link_tips_reach_their_via():
    """Every row track must END on real row copper.

    A row's only B.Cu copper is its bus, which the exporters draw via-to-via —
    and the via sits near each pixel's LEFT edge. Link terminals, however, are
    the segment's gap-facing PIXEL EDGE, so a link landing on a segment's RIGHT
    end used to stop pixel_w - 2*edge_clear - trace_w/2 - via_dia/2 short of
    anything (2.53mm at 3mm pixels, 3.53mm at 4mm) — an open circuit that DRC
    cannot see, because DRC checks clearance, not connectivity. Reported from
    an exported board; 55 such dangling ends across the preset sweep.
    """
    from shapely.geometry import LineString, MultiLineString, Point
    from opentac.cad.pipeline.geometry import contiguous_runs
    from opentac.cad.export.kicad_footprint import pixel_geometry

    for px in (3.0, 4.0):
        resp = _gen(U_NOTCH, px)
        ed = resp.edit_data
        lay = ed["layout"]
        prm = ed["params"]
        P = np.asarray(lay["placements_xy"], dtype=float)
        row_net = lay["row_net_per_pixel"]
        removed = set(ed.get("removed_pix") or [])
        active = [i for i in range(len(P)) if i not in removed]
        g = pixel_geometry(prm["pixel_w_mm"], prm["pixel_h_mm"], prm["trace_w_mm"],
                           prm["gap_mm"], prm["center_clear_mm"],
                           prm["edge_clear_mm"], prm["via_dia_mm"])

        # Rebuild the exported intra-row bus exactly as kicad_pcb.py does.
        bus = {}
        for rnet in sorted(set(row_net[i] for i in active)):
            idx = [i for i in active if row_net[i] == rnet]
            if len(idx) < 2:
                continue
            pts = P[idx][np.argsort(P[idx][:, 0])]
            segs = []
            for run in contiguous_runs(pts[:, 0], prm["pitch_x_mm"] * 1.25):
                if len(run) < 2:
                    continue
                s = pts[run]
                y = float(np.mean(s[:, 1]))
                segs.append(LineString([(s[0, 0] + g.via_x, y),
                                        (s[-1, 0] + g.via_x, y)]))
            if segs:
                bus[rnet] = MultiLineString(segs)

        linked = 0
        for r in ed["routes"]:
            net = r["net_name"]
            if not net.startswith("ROW") or net not in bus:
                continue
            poly = np.asarray(r["polyline_xy"], dtype=float)
            pad = np.asarray(r["pad_xy"], dtype=float)
            for endpoint in (poly[0], poly[-1]):
                if np.linalg.norm(endpoint - pad) < 0.5:
                    continue          # the connector-pad end
                d = bus[net].distance(Point(endpoint))
                assert d <= prm["trace_w_mm"], (
                    f"@{px}mm {net}: track ends {d:.3f}mm from its own row bus "
                    f"at {endpoint} — open circuit")
                linked += 1
        # Guard the guard: this shape must actually exercise linked rows, or
        # the assertion above would pass vacuously.
        assert linked >= 6, f"@{px}mm: expected linked row ends, saw {linked}"
        assert resp.drc["violations"] == 0,             f"@{px}mm: {resp.drc['violations']} DRC violations"
    print("test_row_link_tips_reach_their_via: OK (3mm + 4mm)")


if __name__ == "__main__":
    test_pad_entry_vertical_hit_and_spill()
    test_wrap_band_3mm_routes_clean()
    test_wrap_band_4mm_still_clean()
    test_arch_3mm_still_clean()
    test_wrap_band_fixed_mode_clean()
    test_island_field_routes_clean_across_sizes()
    test_bay_cable_edge_warns_and_never_ships_crossings()
    test_plain_shapes_full_sensors_across_pitches()
    test_dense_boards_full_sensors_no_fallback()
    test_exported_edge_cuts_contour_is_closed()
    test_row_link_tips_reach_their_via()
    print("\n=== ALL HUG FAN TESTS PASSED ===")
