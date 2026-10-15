"""Inner-layer rows for 4-layer boards (PROTOTYPE): the pipeline glue.

On a 2-layer board a row's B.Cu bus can only leave the array at its ends, so
every row wire walks around the side of the field to the connector. On a
4-layer board the rows leave on the inner layers instead:

  route_row_links_in2   In2.Cu: a row that a concavity splits into pieces
                        (petals, notches) is joined across each gap.
  route_rows_inner      In1.Cu (then In2 if asked): every row to the
                        connector pads.

Both are routed by the one engine in inner_engine.py — a top-down sweep over
the via lattice in which wires keep their order and may step sideways
between any two rows. This module supplies what the engine needs: where
inner copper may go, the via lattice, where a row may start and end, and,
below the field, the 45-degree fan into the drilled connector pads.

The rule above all others: inner-layer copper runs UNDERNEATH THE TAXELS
(inside the pitch-cell field) — copper elsewhere makes grow mode grow the
board around it. The only exception is the stretch from the bottom of the
field to the connector, which the F.Cu column fan occupies anyway. On a fixed
outline inner copper also keeps the board-edge clearance.
"""

import math

import numpy as np
from shapely.geometry import LineString, box

from opentac.cad.export.kicad_footprint import pixel_geometry
from .types import Layout, Params, Route


def wire_model():
    """OT_INNER_WIRE selects the inner-layer wire model:
      "sweep"       inner_engine.py: lanes and tracks (the staircase);
      "lines-only"  inner_lines.py: straight lines, vertical and lattice-
                    diagonal, bends where they meet;
      "lines"       the straight lines wherever they route as many rows (or
                    bridges) as the sweep does, the sweep where they do not.
      "ordered"     reserve a complete ordered channel, including the pad
                    approach; retain sweep when that certificate does not fit.
      "hug"         (the default) rows: inner_trunk.py (one bundle steered along the
                    via lattice, rows joining farthest first, in row order;
                    the rows In1 refuses on In2 when given), falling back to
                    the sweep / inner_hug.py for any it leaves; links: nested
                    U's round each notch (inner_trunk.route_links), falling
                    back to inner_hug.py / the sweep.
    All are kept until one wins."""
    import os
    return os.environ.get("OT_INNER_WIRE", "hug")


# The boards ship with KiCad's default board setup, whose hole clearance
# (copper to drill-hole edge) is 0.25 mm. With fine trace rules that, not the
# pad clearance, is what keeps a lane off a via.
HOLE_CLEARANCE_MM = 0.25


def _via_keepout(params: Params) -> float:
    """centre-to-centre distance a trace keeps from a via"""
    return (max(params.via_dia_mm / 2 + params.clearance_mm,
                params.via_drill_mm / 2 + HOLE_CLEARANCE_MM)
            + params.trace_w_mm / 2)


def taxel_cells(layout: Layout, params: Params):
    """Where inner-layer copper may go: the pitch-sized cell under each taxel,
    nothing else. Deliberately NOT route_hug.field_polygon, which bridges
    separate islands of taxels with pitch-wide corridors so the hug contour is
    one ring — a corridor with no taxels over it, which is exactly where
    inner copper must not run."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    weld = 0.005      # fuse corner-touching cells, as field_polygon does
    hx, hy = params.pitch_x_mm / 2 + weld, params.pitch_y_mm / 2 + weld
    cells = [box(x - hx, y - hy, x + hx, y + hy) for x, y in layout.placements_xy]
    return unary_union(cells).buffer(1e-3)


def _cluster(vals, tol):
    vals = np.sort(np.asarray(vals, dtype=float))
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return np.array([np.mean(c) for c in out])


def route_rows_inner(layout: Layout, params: Params, plan,
                     layers=("In1.Cu", "In2.Cu"), preplaced=None):
    """Returns ({layer: [Route]}, failed_nets, info); see _route_rows.

    OT_INNER_WIRE=ordered tries the reserved construction (inner_ordered.py)
    first. Where it does not certify, or refuses rows, the plain sweep runs
    exactly as OT_INNER_WIRE=sweep would -- same region, same checks -- and
    whichever leaves fewer rows on B.Cu is kept. The fallback is therefore
    never weaker than the sweep it falls back to."""
    model = wire_model()
    if model == "hug":
        # The trunk first (inner_trunk.py: one bundle steered along the via
        # lattice, rows joining farthest first, in row order; a second bundle
        # on In2 for the rows In1 refuses). If it leaves rows behind, the
        # sweep and the inner hug run too, and whichever leaves fewest rows on
        # B.Cu ships — the trunk on a tie, then the sweep.
        from .inner_trunk import route_feeds as trunk
        from .inner_hug import route_feeds as hug
        pads_before = dict(layout.pad_to_net)
        best, best_pads = None, None
        for name, run in (("trunk", lambda: trunk(layout, params, plan, layers, preplaced)),
                          ("sweep", lambda: _route_rows(layout, params, plan, layers, preplaced, "sweep")),
                          ("hug", lambda: hug(layout, params, plan, layers, preplaced))):
            layout.pad_to_net.clear()
            layout.pad_to_net.update(pads_before)
            got = run()
            got[2]["wire_model"] = {L: name for L in layers}
            if best is None or len(got[1]) < len(best[1]):
                best, best_pads = got, dict(layout.pad_to_net)
            if not best[1]:
                break
        layout.pad_to_net.clear()
        layout.pad_to_net.update(best_pads)
        return best
    if model != "ordered":
        return _route_rows(layout, params, plan, layers, preplaced, model)
    tried = _route_rows_ordered(layout, params, plan, layers, preplaced)
    if tried is not None and not tried[1]:
        _take_pads(layout, tried)
        return tried
    pads_before = dict(layout.pad_to_net)
    swept = _route_rows(layout, params, plan, layers, preplaced, "sweep")
    if tried is not None and len(tried[1]) < len(swept[1]):
        layout.pad_to_net.clear()
        layout.pad_to_net.update(pads_before)
        _take_pads(layout, tried)
        return tried
    swept[2]["ordered_refusal"] = ({"reason": "did not certify"} if tried is None else
                                   {"reason": "refused more rows than the sweep",
                                    "rows": sorted(tried[1])})
    return swept


def _take_pads(layout, result):
    """the pads an accepted result assigned, onto the layout's pad map"""
    for rs in result[0].values():
        for r in rs:
            layout.pad_to_net[r.pad_num] = r.net_name


def _row_terms(layout: Layout, params: Params, plan):
    """via positions and the row nets' sources (vias, own pad)"""
    Pxy = layout.placements_xy
    g = pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                       params.gap_mm, params.center_clear_mm, params.edge_clear_mm,
                       params.via_dia_mm)
    via_xy = np.column_stack([Pxy[:, 0] + g.via_x, Pxy[:, 1] - g.via_y])
    rows = []
    for net, chain in plan.chains.items():
        if chain.face != "row":
            continue
        idx = np.concatenate([np.asarray(sg, dtype=int) for sg in chain.segments])
        rows.append({
            "net": net, "pix": idx,
            "y": float(np.mean(via_xy[idx, 1])),
            "vias": np.sort(via_xy[idx, 0]),
            "pad_num": int(chain.pad_num),
            "pad_xy": np.asarray(chain.hops[0].a_xy, dtype=float),
        })
    return via_xy, rows


def _route_rows_ordered(layout: Layout, params: Params, plan, layers, preplaced):
    """The reserved construction (inner_ordered.solve) on one empty layer,
    checked whole -- stubs included -- against its own strict region: the
    taxels plus only what lies below the field's lower edge. Returns the
    result (pads in the Routes, not yet on the layout) or None when it does
    not certify. Does not touch the layout."""
    from shapely.ops import unary_union
    from .inner_ordered import allowed_region, source_order, solve
    from .inner_engine import verify
    if len(layers) != 1 or (preplaced or {}).get(layers[0]):
        return None
    via_xy, rows = _row_terms(layout, params, plan)
    if not rows:
        return None
    s = params.trace_w_mm + params.clearance_mm
    a = _via_keepout(params)
    X = _cluster(via_xy[:, 0], 0.05)
    Yr = _cluster(via_xy[:, 1], 0.05)
    via_net = list(layout.row_net_per_pixel)
    field = taxel_cells(layout, params)
    # the board as packed (its connector-shoulder wedges filled), inset
    board_in = layout.board_poly.buffer(-(params.trace_w_mm / 2 + params.board_edge_clear_mm))
    region = field
    if not params.auto_expand_board and not board_in.is_empty:
        region = field.intersection(board_in)
    pad_floor = min([float(layout.pad_y_mm)] + [float(r["pad_xy"][1]) for r in rows]
                    + [float(p[0][1]) for p in getattr(plan, "pad_stubs", {}).values() if p])
    whole_region = allowed_region(region, board_in, pad_floor - 0.001)
    ranks, missing = source_order(rows, X, via_xy, whole_region, a, s, params.pitch_x_mm)
    if missing:
        return None
    candidate, note = solve([dict(r, order=ranks[r["net"]]) for r in rows],
                            X, Yr, whole_region, a, s, params.pitch_x_mm)
    refused = set((note or {}).get("refused", {}))
    if not candidate or len(candidate) + len(refused) != len(rows):
        return None
    for net, pts in candidate.items():
        stub = getattr(plan, "pad_stubs", {}).get(note["pads"][net])
        if stub:
            pts.extend(tuple(q) for q in stub[-2::-1])
    # check the coordinates export actually writes, stubs included
    candidate = {n: np.round(p, 6) for n, p in candidate.items()}
    if verify(candidate, {n: n for n in candidate}, via_xy, via_net,
              whole_region, a, s, region_slack=0):
        return None
    result = [Route(net_name=net, pad_num=note["pads"][net], anchor_xy=pts[0], pad_xy=pts[-1],
                    polyline_xy=pts[::-1].copy())
              for net, pts in candidate.items()]
    return {layers[0]: result}, refused, {
        "rows": len(rows), "rows_failed": sorted(refused),
        "fail_reasons": note.get("refused", {}),
        "rows_bcu_fallback": len(refused), "per_layer": {layers[0]: len(result)},
        "fallback_ranks": [], "wire_model": {layers[0]: "ordered"},
        "certificate": {k: v for k, v in note.items() if k != "pads"},
    }


def _route_rows(layout: Layout, params: Params, plan, layers, preplaced, model):
    """Returns ({layer: [Route]}, failed_nets, info).

    Each row gets one wire, from any of its vias to a connector pad, on the
    first layer in `layers` that takes it. `preplaced` ({layer: [(net,
    polyline)]}) is copper already on a layer — the notch bridges on In2 —
    which the rows route around. Each Route runs pad -> tap via.
    failed_nets: rows with no inner route; the caller keeps their hug B.Cu
    wire. Mutates layout.pad_to_net: placed rows' pads are re-assigned in
    lane order (failed rows keep theirs).

    The wires come from inner_engine.sweep_escapes (they head for the row
    pads, in order, and leave the field where it ends under them); below the
    field they fan 45 degrees into the pads. Parallel 45-degree wires need
    s*sqrt(2) apart but pads are pad_pitch apart, so the pad stubs are
    staggered into a tent.
    """
    from shapely.ops import unary_union

    Pxy = layout.placements_xy
    g = pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                       params.gap_mm, params.center_clear_mm, params.edge_clear_mm,
                       params.via_dia_mm)
    via_xy = np.column_stack([Pxy[:, 0] + g.via_x, Pxy[:, 1] - g.via_y])

    s = params.trace_w_mm + params.clearance_mm
    a = _via_keepout(params)
    tol = 0.05

    rows = []
    for net, chain in plan.chains.items():
        if chain.face != "row":
            continue
        idx = np.concatenate([np.asarray(sg, dtype=int) for sg in chain.segments])
        rows.append({
            "net": net, "pix": idx,
            "y": float(np.mean(via_xy[idx, 1])),
            "vias": np.sort(via_xy[idx, 0]),
            "pad_num": int(chain.pad_num),
            "pad_xy": np.asarray(chain.hops[0].a_xy, dtype=float),
        })
    if not rows:
        return {L: [] for L in layers}, set(), {}

    X = _cluster(via_xy[:, 0], tol)
    Yr = _cluster(via_xy[:, 1], tol)
    field = taxel_cells(layout, params)
    # The drawn outline (not grow mode's working canvas) plus the tail. The
    # tail only TOUCHES the outline along the cable edge, so it is inflated a
    # hair first — a plain union of the two is two polygons with a seam, and
    # nothing would ever be allowed to cross from the field into the tail.
    drawn_parts = [p for p in (layout.outline_poly,
                               layout.cable_poly.buffer(0.01, join_style=2)
                               if layout.cable_poly is not None else None)
                   if p is not None and not p.is_empty]
    drawn = unary_union(drawn_parts) if drawn_parts else layout.board_poly
    edge_in = params.trace_w_mm / 2 + params.board_edge_clear_mm
    board_in = drawn.buffer(-edge_in)
    pad_center = float(np.mean([r["pad_xy"][0] for r in rows]))
    y_pad = float(rows[0]["pad_xy"][1])

    via_net = list(layout.row_net_per_pixel)

    # Where this router may put copper: under the taxels and, on a fixed
    # outline, inside the board by the edge clearance (a taxel cell near the
    # cut can reach almost to the edge).
    region = field
    if not params.auto_expand_board and board_in is not None and not board_in.is_empty:
        region = field.intersection(board_in)

    from functools import lru_cache
    @lru_cache(maxsize=None)
    def exit_y(x):
        hit = region.intersection(LineString([(x, -1e5), (x, 1e5)]))
        return None if hit.is_empty else float(hit.bounds[1])

    # Where a row wire may leave the taxels. "bus" (OT_INNER_EXIT=bus, trial): only straight
    # above the connector tail, where a plain drop reaches the pad row inside
    # the board; below the field the wires are a bus (bus_to_pads). "comb"
    # (default): anywhere along the bottom
    # edge, then a hug walk round the field to the pads (comb_to_pads).
    import os
    bus = os.environ.get("OT_INNER_EXIT", "comb") == "bus"
    y_row = min(float(r["pad_xy"][1]) for r in rows)
    pad_top = max(float(r["pad_xy"][1]) for r in rows)

    def exit_ok(x):
        ey = exit_y(x)
        if ey is None:
            return False
        if not bus or board_in is None or board_in.is_empty:
            return True
        return board_in.covers(LineString([(x, ey), (x, y_row + s)]))

    orig_pad = {r["net"]: (r["pad_num"], r["pad_xy"]) for r in rows}
    all_pads = sorted(orig_pad.values(), key=lambda pp: pp[1][0])

    # ---- 1) the engine: every row on the first layer, what is left on the
    # next. Copper already on a layer (the notch bridges on In2) is cut out
    # of the region, so the sweep simply routes around it. --------------------
    from .inner_engine import Lattice, chamfer_paths, sweep_escapes, verify
    lat = Lattice(X, Yr, a, s, pitch=float(params.pitch_x_mm))
    order = sorted(rows, key=lambda r: -r["y"])
    direct = {L: [] for L in layers}
    arrivals = {L: [] for L in layers}
    pending = []
    why = {}
    left = list(order)
    used = {}
    pad_xs = [pp[1][0] for pp in all_pads]
    for L in layers:
        if not left:
            break
        obst = [(net, LineString(poly)) for net, poly in (preplaced or {}).get(L, [])]
        F_L = region
        if obst:
            F_L = region.difference(unary_union([g for _, g in obst]).buffer(s - 1e-6))
        used[L] = "sweep"
        if model.startswith("lines"):
            from .inner_lines import Grid, sweep_escapes as sweep_lines
            G = Grid(X, Yr, via_xy, via_net, F_L, a, s, pitch=float(params.pitch_x_mm))
            paths, why_L = sweep_lines(left, G, exit_y, exit_ok, pad_xs)
            used[L] = "lines"
            if model == "lines":
                # the straight lines left rows behind: the staircase may not
                bad = verify(paths, {n: n for n in paths}, via_xy, via_net, F_L, a, s, others=obst)
                old_paths, old_why = sweep_escapes(left, lat, F_L, exit_y, exit_ok, pad_xs)
                old_bad = verify(old_paths, {n: n for n in old_paths}, via_xy, via_net, F_L, a, s, others=obst)
                if len(old_paths) - len(old_bad) > len(paths) - len(bad):
                    paths, why_L = old_paths, old_why
                    used[L] = "sweep (lines routed fewer rows)"
        else:
            paths, why_L = sweep_escapes(left, lat, F_L, exit_y, exit_ok, pad_xs)
        bad = verify(paths, {n: n for n in paths}, via_xy, via_net, F_L, a, s, others=obst)
        for n in bad:
            paths.pop(n, None)
            why_L[n] = "its wire failed the clearance check"
        if os.environ.get("OT_CHAMFER", "0") == "1":
            # the sweep draws square steps; cut their corners at 45 degrees
            # wherever the clearances allow (taps and exits stay put)
            paths = chamfer_paths(paths, via_xy, via_net, F_L, a, s, others=obst)
        arrivals[L] = [(r, paths[r["net"]], "sweep") for r in left if r["net"] in paths]
        for r in left:
            if r["net"] not in paths:
                why[r["net"]] = why_L.get(r["net"], "no route")
        left = [r for r in left if r["net"] not in paths]
    failed = {r["net"] for r in left}
    why = {n: why[n] for n in failed}

    # ---- 3) field exit -> pad: a nested comb along the field outline -------
    # Each wire left the taxels on the bottom edge of the field. Pads go to
    # the exits in left-to-right order; then each wire hugs the field and the
    # wires laid before it out to its pad (inner_engine.comb_to_pads), the
    # hug router's growing corridor drawn with exact geometry. Rows that fail
    # here fall back, keeping their own pad.
    from .inner_engine import bus_to_pads, comb_to_pads
    from .route_hug import _wire_clearances, field_polygon
    base_c = _wire_clearances(params)[0]
    fpoly = field_polygon(layout, params)
    # The connector, every pin of it (unused ones too) and everything below
    # its pad row (the tail extension later moves the pads further down that
    # way), is in the way of any wire that is not going into one of its pads.
    # Its sides keep a pad's half width plus a lane.
    n0, xy0 = all_pads[0]
    px = lambda n: float(xy0[0]) + (n - int(n0)) * params.pad_pitch_mm
    side = 0.15 + s
    keepout = box(px(1) - side, y_pad - 10.0, px(int(layout.connector_pos)) + side, y_pad + s)
    swept = {}
    for L in layers:
        for r, path, kind in direct[L] + arrivals[L]:
            swept[r["net"]] = (r, path, L)
    routes = {L: [] for L in layers}
    built = {L: {} for L in layers}
    complete_refusals = {}
    for _ in range(len(rows) + 1):
        live = {n: q for n, q in swept.items() if n not in failed}
        own = {orig_pad[n][0] for n in failed}
        avail = [pp for pp in all_pads if pp[0] not in own]
        exits = sorted(live, key=lambda n: live[n][1][-1][0])
        if len(exits) > len(avail):
            worst = max((exits[0], exits[-1]), key=lambda n: abs(live[n][1][-1][0] - pad_center))
            failed.add(worst)
            why[worst] = "more rows than free connector pads"
            continue
        if not exits:
            break
        xs = np.array([live[n][1][-1][0] for n in exits])
        s0 = min(range(len(avail) - len(exits) + 1),
                 key=lambda k: float(np.sum(np.abs(xs - np.array([pp[1][0] for pp in avail[k:k + len(exits)]])))))
        pad_of = {n: avail[s0 + i] for i, n in enumerate(exits)}
        refused = set()
        built = {L: {} for L in layers}
        for L in layers:
            mine = [n for n in exits if live[n][2] == L]
            if not mine:
                continue
            items = [(n, tuple(live[n][1][-1]), tuple(pad_of[n][1])) for n in mine]
            if bus:
                comb, bad = bus_to_pads(items, s, pad_top + s)
            else:
                comb, bad = comb_to_pads(items, fpoly, base_c, s, pad_center, keepout=keepout, entry=s)
            full = {n: [tuple(q) for q in comb[n]] for n in comb}
            # pad -> comb -> exit -> up through the taxels -> tap
            lines = {n: LineString(full[n][::-1] + [tuple(q) for q in reversed(live[n][1][:-1])]) for n in full}
            # clearance among this layer's finished wires (inside the field
            # they were checked already; this catches the comb)
            clash = set()
            names = list(lines)
            for i, n1 in enumerate(names):
                for n2 in names[i + 1:]:
                    if lines[n1].distance(lines[n2]) < s - 1e-6:
                        clash |= {n1, n2}
            miss = bad | clash
            refused |= miss
            built[L] = {n: full[n] for n in full if n not in miss}
        if not refused:
            break
        for n in refused:
            failed.add(n)
            why[n] = "; ".join(complete_refusals.get(n, [])) or "its wire from the field exit to its pad would cross another"

    for r in rows:
        r["pad_num"], r["pad_xy"] = orig_pad[r["net"]]
    placed = []
    for L in layers:
        for n, hpath in built[L].items():
            r, path, _ = live[n]
            pnum, pxy = pad_of[n]
            r["pad_num"], r["pad_xy"] = pnum, pxy
            layout.pad_to_net[pnum] = n
            # pad -> comb -> field exit -> up through the taxels -> tap via
            pts = [tuple(q) for q in reversed(hpath)] + [tuple(q) for q in reversed(path[:-1])]
            stub = getattr(plan, "pad_stubs", {}).get(int(pnum))
            if stub:
                # a fanned-in pad (route_hug._pad_fan): the comb ends at the
                # stub's top; the stub runs on down to the pad itself
                pts = [tuple(q) for q in stub[:-1]] + pts
                pxy = np.asarray(stub[0], dtype=float)
            clean = [pts[0]]
            for q in pts[1:]:
                if math.dist(q, clean[-1]) > 1e-6:
                    clean.append(q)
            routes[L].append(Route(
                net_name=n, pad_num=pnum,
                anchor_xy=np.asarray(path[0], dtype=float),
                pad_xy=np.asarray(pxy, dtype=float),
                polyline_xy=np.asarray(clean, dtype=float)))
            placed.append((r, path, "sweep", L))

    info = {
        "rows": len(rows),
        "rows_failed": sorted(failed),
        "fail_reasons": why,
        "rows_bcu_fallback": len(failed),
        "per_layer": {L: len(routes[L]) for L in layers},
        "direct": sum(1 for q in placed if q[2] == "direct"),
        "channel_rows": len(pending),
        "arrival": sum(1 for q in placed if q[2] == "arrival"),
        # farthest row that fell back, as a height rank (0 = farthest row)
        "fallback_ranks": sorted(i for i, r in enumerate(order) if r["net"] in failed),
        "wire_model": used,
    }
    return routes, failed, info


# ==========================================================================
# In2: row links across a concavity
# ==========================================================================


def route_row_links_in2(layout: Layout, params: Params, plan):
    """Returns (routes, nets_done, info); see _row_links. With
    OT_INNER_WIRE=ordered the reserved bridges are tried, and the plain sweep
    too: its down and up bridges are made apart and checked together, so a
    reserved down set can cost nets the sweep would have joined. The result
    joining more nets is kept (the reserved one on a tie)."""
    model = wire_model()
    if model == "hug":
        # Nested U's on the lattice first (inner_trunk.route_links); if they
        # leave links, the sweep and the inner hug run too and the result
        # joining the most nets ships — the U's on a tie, then the sweep (the
        # inner hug's walk follows the field edge, and zigzags round vias).
        from .inner_trunk import route_links as lanes
        from .inner_hug import route_links as hug
        best = lanes(layout, params, plan)
        n_split = sum(1 for ch in plan.chains.values() if ch.face == "row" and len(ch.segments) > 1)
        if len(best[1]) < n_split:
            for alt in (lambda: _row_links(layout, params, plan, "sweep"), lambda: hug(layout, params, plan)):
                got = alt()
                if len(got[1]) > len(best[1]):
                    best = got
        return best
    if model != "ordered":
        return _row_links(layout, params, plan, model)
    tried = _row_links(layout, params, plan, "ordered")
    swept = _row_links(layout, params, plan, "sweep")
    return swept if len(swept[1]) > len(tried[1]) else tried


def _row_links(layout: Layout, params: Params, plan, model):
    """Returns (routes, nets_done, info). routes go on In2.Cu; for nets_done the
    hug router's B.Cu link hops are no longer needed."""
    Pxy = layout.placements_xy
    g = pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                       params.gap_mm, params.center_clear_mm, params.edge_clear_mm,
                       params.via_dia_mm)
    via_xy = np.column_stack([Pxy[:, 0] + g.via_x, Pxy[:, 1] - g.via_y])
    s = params.trace_w_mm + params.clearance_mm
    a = _via_keepout(params)
    lane = s
    tol = 0.05
    X = _cluster(via_xy[:, 0], tol)
    Yrows = _cluster(via_xy[:, 1], tol)
    field = taxel_cells(layout, params)

    links = []
    for net, chain in plan.chains.items():
        if chain.face != "row" or len(chain.segments) < 2:
            continue
        segs = sorted((np.asarray(sg, dtype=int) for sg in chain.segments),
                      key=lambda sg: float(np.mean(via_xy[sg, 0])))
        for A, B in zip(segs[:-1], segs[1:]):
            links.append({
                "id": len(links), "net": net,
                "y": float(np.mean(via_xy[np.concatenate([A, B]), 1])),
                "xr": float(np.max(via_xy[A, 0])), "xl": float(np.min(via_xy[B, 0])),
                "pixA": A, "pixB": B,
            })
    if not links:
        return [], set(), {"links": 0}

    # Where In2 copper may go: under the taxels, and on a fixed outline inside
    # the board by the edge clearance.
    region = field
    if not params.auto_expand_board and layout.outline_poly is not None:
        edge_in = params.trace_w_mm / 2 + params.board_edge_clear_mm
        region = field.intersection(layout.outline_poly.buffer(-edge_in))

    from shapely.affinity import scale as _scale
    from .inner_engine import Lattice, sweep_bridges, verify
    pitch = float(params.pitch_x_mm)
    # A link bridges downward when there are taxels somewhere below the gap
    # between its pieces (a notch opening up); otherwise upward (a gap along
    # the bottom edge). The two kinds are swept separately: an up link's long
    # piece would otherwise tie every petal above it into one.
    def opens_up(L):
        mid = 0.5 * (L["xr"] + L["xl"])
        below = region.intersection(LineString([(mid, L["y"]), (mid, L["y"] - 1e4)]))
        return not below.is_empty
    down = [L for L in links if opens_up(L)]
    via_net = list(layout.row_net_per_pixel)
    if model.startswith("lines"):
        from .inner_lines import Grid, sweep_bridges as bridges_lines

    used = []

    def bridges(lks, Yr_, F_, vxy):
        if model == "ordered" and lks:
            from .inner_ordered import bridges as reserved_bridges
            ordered, why_ordered = reserved_bridges(
                [dict(L, A=L["pixA"], B=L["pixB"]) for L in lks],
                vxy, F_, a, lane, float(params.pitch_y_mm))
            net_of = {L["id"]: L["net"] for L in lks}
            if len(ordered) == len(lks) and not verify(
                    ordered, net_of, vxy, via_net, F_, a, lane, region_slack=0):
                used.append("ordered")
                return ordered, why_ordered
        if model.startswith("lines"):
            res = bridges_lines(lks, Grid(X, Yr_, vxy, via_net, F_, a, lane, pitch=pitch), vxy, g.via_x, pitch)
            if model == "lines-only" or len(res[0]) == len(lks):
                used.append("lines")
                return res
            # the straight lines left links unplaced: the staircase may not
            old = sweep_bridges(lks, Lattice(X, Yr_, a, lane, pitch=pitch), F_, vxy, g.via_x, pitch)
            if len(old[0]) > len(res[0]):
                used.append("sweep (lines placed fewer links)")
                return old
            used.append("lines")
            return res
        used.append("sweep")
        return sweep_bridges(lks, Lattice(X, Yr_, a, lane, pitch=pitch), F_, vxy, g.via_x, pitch)
    polylines, why = bridges(down, Yrows, region, via_xy)
    # the rest, and any down link the sweep could not place: the same sweep on
    # the mirrored board
    rest = [L for L in links if L["id"] not in polylines]
    if rest:
        flip = np.array([1.0, -1.0])
        res_up, why_up = bridges([dict(L, y=-L["y"]) for L in rest], -Yrows,
                                 _scale(region, 1.0, -1.0, origin=(0, 0)), via_xy * flip)
        for k, v in res_up.items():
            polylines[k] = [(x, -y) for x, y in v]
            why.pop(k, None)

    # independent check: every In2 wire clears other nets' wires and vias and
    # stays in the region (the down and up sweeps did not see each other)
    by_id = {L["id"]: L for L in links}
    bad = verify(polylines, {k: by_id[k]["net"] for k in polylines}, via_xy,
                 list(layout.row_net_per_pixel), region, a, lane)
    bad_nets = {by_id[k]["net"] for k in bad}

    nets_all = {L["net"] for L in links}
    nets_placed = {by_id[k]["net"] for k in polylines}
    # a net is done only if ALL its links were placed and none failed the check
    nets_missing = {L["net"] for L in links if L["id"] not in polylines}
    nets_done = nets_placed - nets_missing - bad_nets

    layout_pad_xy = {n: c.hops[0].a_xy for n, c in plan.chains.items()}
    routes = []
    for k, pts in polylines.items():
        L = by_id[k]
        if L["net"] not in nets_done:
            continue
        routes.append(Route(net_name=L["net"], pad_num=plan.chains[L["net"]].pad_num,
                            anchor_xy=np.asarray(pts[-1], dtype=float),
                            # the net's pad, like hug's link hops: a route whose
                            # first point IS its pad_xy is taken for a pad hop
                            # and extended to the connector by the tail step
                            pad_xy=np.asarray(layout_pad_xy.get(L["net"], (math.inf, math.inf)), dtype=float),
                            polyline_xy=np.asarray(pts, dtype=float)))
    info = {"links": len(links), "unplaced_links": len(links) - len(polylines), "wire_model": used,
            "why": {by_id[k]["net"]: v for k, v in why.items()},
            "nets_in2": len(nets_done), "nets_fallback_bcu": len(nets_all - nets_done)}
    return routes, nets_done, info
