// Tests for tail-guard.js. Run with:  node opentac/cad/web/tests/tail-guard.test.mjs
//
// A grown 41 mm board whose padding reaches 6 mm past the cable edge on either
// side of a 60-pin tail, as the 2 mm calibration square does: the connector
// (30.5 mm long) turned sideways 20 mm from the pad row lands on that padding.

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const G = require("../tail-guard.js");

let fails = 0;
let checks = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { fails++; console.log("  FAIL  " + what); }
}

const editData = {
  outline: [[-20.5, 0], [20.5, 0], [20.5, 41], [-20.5, 41]],
  // the grown board: 27 mm either side, 6 mm past the cable edge, plus its tail
  final_outline: [[-27, -6], [-15.25, -6], [-15.25, -21.5], [15.25, -21.5], [15.25, -6], [27, -6], [27, 43], [-27, 43]],
  params: { pad_pitch_mm: 0.5, board_edge_clear_mm: 0.2, trace_w_mm: 0.2 },
  layout: {
    connector_pos: 60,
    tail: { t_hat: [1, 0], n_out: [0, -1], pad_split_center: [0, -18.25], pad_row_center: [0, -18.25],
            tab_tip_width_mm: 30.5, connector_min_width_mm: 30.5, width_mm: 23.7 },
  },
};
const g = G.prepare(editData);
const start = [0, -18.25];
const width = 23.0;
const go = (from, angle, len = 20) => {
  const r = angle * Math.PI / 180;
  return [from[0] + len * Math.cos(r), from[1] + len * Math.sin(r)];
};
const blocked = (path) => !!G.overlap(g, path, width);

ok(g, "prepare() reads the edit data");
ok(!blocked([start, go(start, -90)]), "straight down is fine");
ok(!blocked([start, go(start, -45)]), "down-right is fine");
ok(!blocked([start, go(start, -135)]), "down-left is fine");
ok(blocked([start, go(start, 0)]), "20 mm right lands the turned connector on the board's padding");
ok(blocked([start, go(start, 180)]), "20 mm left likewise");
ok(blocked([start, go(start, 90)]), "up runs back into the board");
const down = go(start, -90);
ok(!blocked([start, down, go(down, 0)]), "down 20 mm, then right: clear of the board");
ok(!blocked([start, down, go(down, 180)]), "down 20 mm, then left: clear of the board");
ok(!blocked([start, go(start, 0, 40)]), "40 mm right: the connector clears the board's side");
ok(G.overlap(null, [start, down], width) === null, "no board data: nothing is blocked");
ok(G.overlap(g, [start], width) === null, "a path of one point is not an extension");

console.log(`${checks - fails}/${checks} checks passed`);
if (fails) process.exit(1);
