"""4-layer routing: the hug router for the columns, the inner layers for the rows.

On a 4-layer board (Params.copper_layers == 4) the row wiring leaves the field
on the inner copper instead of running round it on B.Cu:

  1. the hug router plans and routes both faces as on 2 layers (columns on
     F.Cu stay exactly that);
  2. rows a notch cuts into pieces are linked on In2 (route_row_links_in2);
  3. every row is fed to the connector on In1, and the rows In1 cannot take
     on what In2 has left (route_rows_inner);
  4. a row that fits on neither keeps its hug B.Cu wire, and a row whose links
     do not all fit on In2 keeps its hug B.Cu links.

The inner routes carry their layer (Route.layer); everything downstream —
DRC, the KiCad and Gerber exports, the preview, edit data — reads it from
there. Returns what route_hug_all returns, so it is a drop-in for it."""

import re

from .route_hug import route_hug_all
from .route_inner import route_row_links_in2, route_rows_inner
from .types import Layout, Params

INNER_LAYERS = ("In1.Cu", "In2.Cu")


def route_hug_4layer(layout: Layout, outline_xy, params: Params):
    """(col_routes, row_routes, plan, failed_pixels) — route_hug_all's
    contract. row_routes mixes the B.Cu wires that remain with the inner-layer
    links and feeds; each inner route has .layer set. plan.inner records what
    went where (for stats)."""
    # the hug's own refusal report repairs its column fan (repair=True is
    # passed only here; 2-layer boards route without it)
    col, row, plan, _ = route_hug_all(layout, outline_xy, params, repair=True)
    is_pad_hop = [_starts_at_pad(r) for r in row]

    # 1) links across notches, on In2 (few, fixed by the shape, at its edge)
    in2, nets_linked, link_info = route_row_links_in2(layout, params, plan)
    for r in in2:
        r.layer = "In2.Cu"
    # 2) rows to the connector, farthest first: In1, then In2 round the links
    by_layer, inner_failed, feed_info = route_rows_inner(
        layout, params, plan, layers=INNER_LAYERS,
        preplaced={"In2.Cu": [(r.net_name, r.polyline_xy) for r in in2]})
    feeds = []
    for L in INNER_LAYERS:
        for r in by_layer.get(L, []):
            r.layer = L
            feeds.append(r)
    # 3) what stays on B.Cu: the hug's own feed for a row no inner layer took,
    #    and its links for a row whose links did not all fit on In2
    bcu = [r for r, pad_hop in zip(row, is_pad_hop)
           if (pad_hop and r.net_name in inner_failed) or (not pad_hop and r.net_name not in nets_linked)]

    # The hug's failure set counts row wires it could not draw on B.Cu, which
    # does not matter for rows routed on the inner layers: recount from what
    # was actually routed.
    col_ok = {r.net_name for r in col}
    fed = {r.net_name for r in feeds}
    bcu_fed = {r.net_name for r, pad_hop in zip(row, is_pad_hop) if pad_hop}
    bcu_links = {}
    for r, pad_hop in zip(row, is_pad_hop):
        if not pad_hop:
            bcu_links[r.net_name] = bcu_links.get(r.net_name, 0) + 1
    row_ok = set()
    for net, ch in plan.chains.items():
        if ch.face != "row":
            continue
        has_feed = net in fed or net in bcu_fed
        need = len(ch.segments) - 1
        has_links = need == 0 or net in nets_linked or bcu_links.get(net, 0) >= need
        if has_feed and has_links:
            row_ok.add(net)
    failed = {i for i in range(len(layout.placements_xy))
              if layout.col_net_per_pixel[i] not in col_ok
              or layout.row_net_per_pixel[i] not in row_ok}
    # row-face ordering warnings only matter for rows still on B.Cu
    inner_only = {n for n in row_ok if n in fed and n not in bcu_links}
    plan.warnings = [w for w in plan.warnings
                     if not (w.startswith("row:") and all(
                         n in inner_only for n in re.findall(r"(ROW_[\w]+?)/", w)))]
    plan.inner = {
        "rows": feed_info.get("rows"),
        "rows_per_layer": dict(feed_info.get("per_layer") or {}),
        "rows_on_bcu": sorted(inner_failed),
        "row_engine": dict(feed_info.get("wire_model") or {}),
        "links": link_info.get("links", 0),
        "rows_linked_on_in2": len(nets_linked),
        "rows_linked_on_bcu": link_info.get("nets_fallback_bcu", 0),
        "link_engine": link_info.get("wire_model"),
    }
    return col, bcu + in2 + feeds, plan, failed


def _starts_at_pad(route) -> bool:
    import numpy as np
    return float(np.linalg.norm(np.asarray(route.polyline_xy[0], dtype=float)
                                - np.asarray(route.pad_xy, dtype=float))) <= 1e-6


def copper_layers_of(params: Params, row_routes=()) -> int:
    """the board's copper stack: 4 when asked for, or when any row wire is on
    an inner layer (a board routed on 4 layers stays 4 whatever is asked of a
    later re-export), else 2"""
    if any(getattr(r, "layer", None) in INNER_LAYERS for r in row_routes or ()):
        return 4
    return 4 if int(getattr(params, "copper_layers", 2) or 2) == 4 else 2


def route_for(params: Params):
    """the routing function for this board's copper stack"""
    return route_hug_4layer if int(getattr(params, "copper_layers", 2) or 2) == 4 else route_hug_all


def is_feed(route) -> bool:
    """an inner-layer row wire from a connector pad (not a link)"""
    return route.layer in INNER_LAYERS and _starts_at_pad(route)


def sync_pads_to_feeds(layout: Layout, row_routes) -> None:
    """The inner feeds hand out the row pads in the order the wires leave
    the field. The router reassigns them on the layout it was given, which
    after a fixed-mode retry is a copy; the routes carry the final pad per
    net, so make this layout's pad map agree with them."""
    fix = {int(r.pad_num): r.net_name for r in row_routes if r.pad_num is not None and is_feed(r)}
    if not fix:
        return
    nets = set(fix.values())
    for pn in [pn for pn, n in layout.pad_to_net.items() if n in nets]:
        del layout.pad_to_net[pn]
    layout.pad_to_net.update(fix)
