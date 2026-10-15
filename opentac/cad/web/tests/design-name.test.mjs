// Tests for design-name.js. Run with:  node frontend/tests/design-name.test.mjs
//
// The module touches `window` and `document` at import time (it binds the input
// field and listens for the mode-change event), so stub just enough of both for
// node to load it.

const listeners = {};
globalThis.window = {
  addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
  dispatchEvent(ev) { (listeners[ev.type] || []).forEach((fn) => fn(ev)); },
};
globalThis.document = {
  addEventListener() {},          // DOMContentLoaded never fires here
  getElementById: () => null,
};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; Object.assign(this, init); } };

const dn = await import("../design-name.js");

let fails = 0;
let checks = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { fails++; console.log("  FAIL  " + what); }
}
function eq(got, want, what) { ok(got === want, `${what}  (got ${got}, want ${want})`); }

// ---- derived from an uploaded file -----------------------------------------
dn.setSource("fingertip_wrap.stl");
eq(dn.downloadName("", "zip"), "fingertip_wrap.zip", "ZIP takes the STL's stem");
eq(dn.downloadName("case_top", "stl"), "fingertip_wrap_case_top.stl",
   "artefacts share the stem and differ by suffix");
eq(dn.downloadName("1to1", "pdf"), "fingertip_wrap_1to1.pdf", "PDF likewise");
eq(dn.downloadName("bump_sheet", "stl"), "fingertip_wrap_bump_sheet.stl", "bump sheet likewise");
eq(dn.downloadName("edited", "zip"), "fingertip_wrap_edited.zip", "edited ZIP likewise");

// ---- sanitising --------------------------------------------------------------
dn.setSource("My Sensor (v2).STL");
eq(dn.downloadName("", "zip"), "My_Sensor_v2.zip", "spaces and punctuation collapse to _");
dn.setSource("../../etc/passwd");
ok(!dn.downloadName("", "zip").includes("/"), "a path separator cannot survive");
ok(!dn.downloadName("", "zip").includes(".."), "nor a parent-directory hop");
dn.setSource("robot.arm.v3.dxf");
eq(dn.downloadName("", "zip"), "robot.arm.v3.zip".replace("robot.arm.v3", "robot_arm_v3"),
   "only the final extension is dropped, inner dots become _");
dn.setSource("x".repeat(200) + ".stl");
ok(dn.downloadName("", "zip").length <= 48 + 4, "an absurd name is capped");

// ---- example titles ----------------------------------------------------------
dn.setSource("Soft fingertip wrap");
eq(dn.downloadName("", "zip"), "Soft_fingertip_wrap.zip", "an example title works as a source");

// ---- drawn from scratch: timestamp ------------------------------------------
dn.clearSource();
const auto = dn.downloadName("", "zip");
ok(/^sensor_\d{8}_\d{4}\.zip$/.test(auto), `timestamp default looks right (${auto})`);
eq(dn.downloadName("case_top", "stl"),
   auto.replace(/\.zip$/, "_case_top.stl"),
   "every artefact of one drawn design shares the timestamp stem");

// ---- user override ----------------------------------------------------------
// bindInput is what the real page uses; drive it through a fake input.
const input = {
  value: "", placeholder: "", handlers: {},
  addEventListener(t, fn) { this.handlers[t] = fn; },
  type(v) { this.value = v; this.handlers.input(); },
  blur() { this.handlers.blur(); },
};
dn.bindInput(input);
ok(/^sensor_\d{8}_\d{4}$/.test(input.value), "the field pre-fills with the auto name");

input.type("thumb pad rev B");
eq(dn.downloadName("", "zip"), "thumb_pad_rev_B.zip", "the user's name wins");
eq(dn.downloadName("case_bottom", "stl"), "thumb_pad_rev_B_case_bottom.stl",
   "and applies to every artefact");
input.blur();
eq(input.value, "thumb_pad_rev_B", "blur shows what will actually be used");

// A regenerate must not clobber a name the user just typed.
const typed = dn.downloadName("", "zip");
dn.clearSource();                       // no-op path: source already cleared
eq(dn.downloadName("", "zip"), typed, "clearSource does not discard a typed name mid-design");

// A NEW source is a new design, so it does take over.
dn.setSource("other_part.stl");
eq(dn.downloadName("", "zip"), "other_part.zip", "loading a new source resets the name");
eq(input.value, "other_part", "and the field follows");

// Emptying the box falls back to automatic rather than naming a file nothing.
input.type("");
eq(dn.downloadName("", "zip"), "other_part.zip", "an empty box means automatic");

// ---- mode switching ----------------------------------------------------------
dn.setSource("loaded_mesh.stl");
window.dispatchEvent(new CustomEvent("otc:mode-selected", { detail: { mode: "draw" } }));
ok(/^sensor_\d{8}_\d{4}\.zip$/.test(dn.downloadName("", "zip")),
   "switching to Draw drops the previous source's name");

dn.setSource("loaded_mesh.stl");
window.dispatchEvent(new CustomEvent("otc:mode-selected", { detail: { mode: "stl" } }));
eq(dn.downloadName("", "zip"), "loaded_mesh.zip",
   "switching to an upload mode keeps it");

console.log(`\n${checks} checks, ${fails} failures`);
process.exit(fails ? 1 : 0);
