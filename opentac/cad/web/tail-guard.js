/* Would a drawn tail extension put copper or the connector back on the board?
 *
 * A path that turns too early lands the connector (turned with the path) on the
 * board itself: its 30.5 mm length runs back along the board, over the wires
 * that wrap the board's corner, and the export shorts them. Such a path cannot
 * work at any routing, so the route editor does not let it be drawn: each
 * direction button is disabled when adding its segment would make the
 * extension's outline overlap the board's.
 *
 * Geometry, in the layout's millimetres (y up, the frame of edit_data):
 *   - the board: edit_data.final_outline, less the tail the extension replaces
 *     (the strip from the cable edge out along n_out, as wide as the connector's
 *     tab), so a path that starts in the tail and runs on past it is fine;
 *   - the extension: each segment as wide as the extension, and the connector at
 *     the path's end (as wide as its tab, 6 mm deep, centred on the end point,
 *     turned to the last segment), as the editor previews it;
 *   - overlap: any point of the extension, sampled every 0.25 mm and shrunk by
 *     TOUCH_MM, inside the board and outside that tail strip.
 *
 * Checked against KiCad's DRC on the 31 example designs x 2/4 layers x
 * grow/fixed x five directions, 20 mm: every extension KiCad fails (all of
 * them sideways on grown boards, the connector over the board's corner wiring)
 * is blocked. Also blocked: sideways ones whose wide strip runs along the
 * board's edge, merged into it; KiCad passes those, but the cable is stuck to
 * the board until it passes it, so it too should run further out first.
 *
 * Works in the browser (window.OTTailGuard) and in node (module.exports).
 */
(function (root) {
  "use strict";

  const STEP_MM = 0.25;          // sampling pitch over the extension's area
  const TOUCH_MM = 0.5;          // an overlap shallower than this is two edges touching
  const BODY_DEPTH_MM = 6.0;     // as connectorDimensions() in route-editor.js

  function inside(poly, x, y) {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }

  function unit(v) {
    const n = Math.hypot(v[0], v[1]);
    return n > 1e-9 ? [v[0] / n, v[1] / n] : null;
  }

  // Everything the check needs from edit_data, once per board.
  function prepare(editData) {
    const board = editData?.final_outline || editData?.outline;
    const outline = editData?.outline;
    const tail = editData?.layout?.tail || {};
    const p = editData?.params || {};
    if (!board?.length || !outline?.length || !tail.t_hat || !tail.n_out) return null;
    const pos = Number(editData.layout.connector_pos) || 0;
    const pitch = Number(p.pad_pitch_mm) || 0.5;
    const tab = Math.max(Number(tail.tab_tip_width_mm) || 0, Number(tail.connector_min_width_mm) || 0,
                         (pos + 1) * pitch, Number(tail.width_mm) || 0);
    // Edges that only touch are fine; an overlap deeper than this blocks.
    const margin = -TOUCH_MM;
    const tailTol = (Number(p.board_edge_clear_mm) || 0.2) + (Number(p.trace_w_mm) || 0.2) / 2;
    const mid = [(outline[0][0] + outline[1][0]) / 2, (outline[0][1] + outline[1][1]) / 2];
    // the tab is centred on the connector, which can sit off the cable edge's middle
    const pc = tail.pad_row_center || mid;
    const tabCentre = (pc[0] - mid[0]) * tail.t_hat[0] + (pc[1] - mid[1]) * tail.t_hat[1];
    return {
      board, mid,
      tHat: tail.t_hat, nOut: tail.n_out,
      tabCentre,
      tabHalf: tab / 2,
      tipHalf: (pos + 1) * pitch / 2,
      margin,             // the extension, shrunk by TOUCH_MM
      tailTol,            // the exempt tail strip, a little wider than the tab
    };
  }

  // In the tail the extension replaces: past the cable edge, within the tab's width.
  function inTail(g, x, y) {
    const dx = x - g.mid[0], dy = y - g.mid[1];
    const s = dx * g.tHat[0] + dy * g.tHat[1];
    const d = dx * g.nOut[0] + dy * g.nOut[1];
    return d > -g.tailTol && Math.abs(s - g.tabCentre) <= g.tabHalf + g.tailTol;
  }

  function onBoard(g, x, y) {
    return !inTail(g, x, y) && inside(g.board, x, y);
  }

  // A rectangle (centre c, axes u along its length, v across) sampled, grown by the margin.
  function rectHits(g, c, u, halfU, v, halfV) {
    const hu = halfU + g.margin, hv = halfV + g.margin;
    for (let a = -hu; a <= hu + 1e-9; a += STEP_MM) {
      for (let b = -hv; b <= hv + 1e-9; b += STEP_MM) {
        const x = c[0] + a * u[0] + b * v[0], y = c[1] + a * u[1] + b * v[1];
        if (onBoard(g, x, y)) return [x, y];
      }
    }
    return null;
  }

  /* The first point where `path` (mm points, the drawn extension, starting in
   * the tail) would overlap the board, or null when it clears it.
   * `widthMm`: the extension's width. */
  function overlap(g, path, widthMm) {
    if (!g || !path || path.length < 2) return null;
    const half = Math.max(0, Number(widthMm) || 0) / 2;
    for (let i = 0; i < path.length - 1; i++) {
      const a = path[i], b = path[i + 1];
      const u = unit([b[0] - a[0], b[1] - a[1]]);
      if (!u) continue;
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const hit = rectHits(g, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], u, len / 2, [-u[1], u[0]], half);
      if (hit) return hit;
    }
    const end = path[path.length - 1], prev = path[path.length - 2];
    const nOut = unit([end[0] - prev[0], end[1] - prev[1]]);
    if (!nOut) return null;
    return rectHits(g, end, [-nOut[1], nOut[0]], g.tipHalf, nOut, BODY_DEPTH_MM / 2);
  }

  const api = { prepare, overlap };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OTTailGuard = api;
})(typeof window !== "undefined" ? window : this);
