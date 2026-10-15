// Tests for design-import.js's ZIP reader. Run with:
//   node frontend/tests/design-import.test.mjs [path/to/export.zip]
//
// The reader is hand-written binary parsing, which is where a re-import is
// most likely to break in a way nothing else catches: a wrong offset gives
// either garbage JSON or a plausible-looking string from the middle of a
// Gerber. It runs against a REAL export ZIP (generate one with
// backend/tests/make_export_zip.py, or pass a path) rather than a synthetic
// archive, so it is testing the file the user actually hands back.
//
// Like design-name.test.mjs, the module touches window/document at import
// time; stub just enough for node to load it.

import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

globalThis.window = { addEventListener() {}, dispatchEvent() {} };
globalThis.document = { addEventListener() {}, getElementById: () => null };
globalThis.location = { hostname: "localhost" };
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; Object.assign(this, init); } };

const here = dirname(fileURLToPath(import.meta.url));
const zipPath = process.argv[2] || join(here, "fixtures", "export.zip");

const { readZipEntry } = await import("../design-import.js");

let fails = 0, checks = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { fails++; console.log("  FAIL  " + what); }
}

if (!existsSync(zipPath)) {
  console.log(`SKIP design-import: no export ZIP at ${zipPath}`);
  console.log("  build one with:  python backend/tests/make_export_zip.py");
  process.exit(0);
}

const buf = readFileSync(zipPath);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

// --- the entry we actually want comes back, and parses ---
const entry = await readZipEntry(ab, ["design.json", "manifest.json"]);
ok(entry !== null, "found a design record in the export ZIP");
ok(entry?.name === "design.json", `preferred design.json (got ${entry?.name})`);

const design = JSON.parse(entry.text);
ok(design.app === "OpenTactileCAD", "design.json identifies the app");
ok(design.kind === "design", "design.json is a save file");
ok(typeof design.pipeline === "string" && design.pipeline.length > 0, "carries a pipeline stamp");
ok(design.edit_data && design.edit_data.params && design.edit_data.layout,
   "carries the editor state");
ok(Array.isArray(design.routes) && design.routes.length > 0, "carries routes");

// --- the manifest is reachable too, and is the OTHER record ---
const man = await readZipEntry(ab, ["manifest.json"]);
ok(man?.name === "manifest.json", "can pick manifest.json specifically");
const manifest = JSON.parse(man.text);
ok(manifest.design && manifest.design.params, "manifest carries the recipe");
ok(Array.isArray(manifest.pixels) && manifest.pixels.length > 0, "manifest still describes taxels");
ok(manifest.design.params.board_mode === design.edit_data.params.board_mode,
   "the two records agree on board_mode");

// --- an entry that is not there is absent, not garbage ---
ok((await readZipEntry(ab, ["nope.json"])) === null, "a missing entry returns null");

// --- offsets are right for EVERY entry, not just the one we want ---
// The bug this catches: reading name/extra lengths from the central directory
// instead of the local header. That happens to work for the first entry in an
// archive and silently misreads later ones.
const gerber = await readZipEntry(ab, ["gerber/board.drl"]);
ok(gerber !== null, "reaches a deep, later entry (gerber/board.drl)");
ok(gerber && gerber.text.startsWith("M48"), "that entry decodes to a real Excellon file");

// --- a non-ZIP is rejected clearly ---
let threw = "";
try { await readZipEntry(new Uint8Array([1, 2, 3, 4]).buffer, ["design.json"]); }
catch (e) { threw = e.message; }
ok(threw.includes("not a ZIP"), `a non-ZIP says so (got: ${threw})`);

console.log(`design-import: ${checks - fails}/${checks} checks passed`);
process.exit(fails ? 1 : 0);
