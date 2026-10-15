// One place that decides what a downloaded file is called.
//
// Every download used to be a hardcoded constant -- tactile_pcb.zip,
// case_top.stl, bump_sheet.stl -- so two designs in a row landed in Downloads as
// "tactile_pcb.zip" and "tactile_pcb (1).zip", and a folder of them told you
// nothing about which was which.
//
// The name comes from the best source available, in this order:
//
//   1. what the user typed, if they typed anything
//   2. the file they started from  -- fingertip_wrap.stl -> fingertip_wrap
//      (also an example's title, or an imported DXF)
//   3. a timestamp                 -- sensor_20260903_1432
//
// Every artefact of one design shares that stem and differs only by suffix, so
// they sort together:
//
//   fingertip_wrap.zip
//   fingertip_wrap_1to1.pdf
//   fingertip_wrap_bump_sheet.stl
//   fingertip_wrap_case_bottom.stl
//   fingertip_wrap_case_top.stl

const FALLBACK_STEM = "sensor";
const MAX_STEM = 48;

let sourceStem = "";   // derived from an uploaded file, example, or import
let userStem = "";     // whatever the user typed; wins over everything
let autoStem = "";     // the timestamp fallback, generated once per design
let inputEl = null;
let hintEl = null;
let printedStem = null;  // the name baked into the silkscreen of the last board

// Matches the sanitiser already used for the mesh-library name in
// stl-viewer.js, so names produced by the two paths look alike.
function sanitize(raw) {
  return String(raw || "")
    .replace(/\.[A-Za-z0-9]{1,6}$/, "")     // drop one trailing extension
    .replace(/[^A-Za-z0-9-_]+/g, "_")
    .replace(/_{2,}/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "")
    .slice(0, MAX_STEM);
}

// Local time, not UTC: this is a label a person reads next to the clock on
// their own wall. Sortable, and no separators that need escaping.
function timestampStem() {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${FALLBACK_STEM}_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
         `_${p(d.getHours())}${p(d.getMinutes())}`;
}

function ensureAuto() {
  if (!autoStem) autoStem = timestampStem();
  return autoStem;
}

/** The stem the user would see if they have not overridden it. */
export function derivedStem() {
  return sourceStem || ensureAuto();
}

/** The stem actually used for downloads. */
export function designStem() {
  return userStem || derivedStem();
}

/**
 * Record where this design came from. Called when a source is loaded, not on
 * every regenerate: a new source means a new design, so it clears both the
 * user's override and the timestamp, and the next download is named after the
 * new source.
 */
export function setSource(name) {
  const stem = sanitize(name);
  if (!stem) return;
  sourceStem = stem;
  userStem = "";
  autoStem = "";
  syncInput();
}

/**
 * Forget the source -- a design drawn or described from scratch has none, and
 * must not inherit the name of whatever was loaded before it.
 */
export function clearSource() {
  // Only meaningful if a source was actually loaded. Without this guard,
  // re-selecting the mode you are already in -- clicking "Draw" while drawing --
  // fires the mode event again and throws away a name the user just typed.
  if (!sourceStem) return;
  sourceStem = "";
  userStem = "";
  autoStem = "";
  syncInput();
}

/**
 * Record the name that went onto the board's silkscreen.
 *
 * The label is stroked into the Gerbers at generate time, so a name typed
 * afterwards changes the ZIP's filename but not the copper. Rather than
 * silently shipping a board whose silkscreen disagrees with its filename, the
 * field says so and points at the fix (regenerate).
 */
export function markPrinted(stem) {
  printedStem = stem === null ? null : sanitize(stem);
  syncHint();
}

/** Forget it: nothing is generated, so there is nothing to disagree with. */
export function clearPrinted() {
  printedStem = null;
  syncHint();
}

function syncHint() {
  if (!hintEl) return;
  const stale = printedStem !== null && sanitize(designStem()) !== printedStem;
  hintEl.hidden = !stale;
}

/**
 * Filename for one artefact of the current design.
 *   downloadName("", "zip")            -> fingertip_wrap.zip
 *   downloadName("case_top", "stl")    -> fingertip_wrap_case_top.stl
 */
export function downloadName(part, ext) {
  const stem = sanitize(designStem()) || FALLBACK_STEM;
  const suffix = part ? `_${sanitize(part)}` : "";
  return `${stem}${suffix}.${ext}`;
}

// ---- the editable field -----------------------------------------------------

function syncInput() {
  if (!inputEl) return;
  // Only overwrite what is on screen while the user has not taken it over.
  // Clobbering a name someone just typed because an unrelated regenerate fired
  // is the one behaviour that would make this feature annoying.
  if (!userStem) inputEl.value = derivedStem();
  inputEl.placeholder = derivedStem();
  syncHint();
}

export function bindInput(el) {
  if (!el) return;
  inputEl = el;
  syncInput();
  el.addEventListener("input", () => {
    const stem = sanitize(el.value);
    // An empty box means "go back to automatic" rather than "name it nothing".
    userStem = stem;
    syncHint();
  });
  el.addEventListener("blur", () => {
    // Show what will actually be used, including any sanitising, so the name in
    // the box is never a lie about the name on disk.
    el.value = designStem();
  });
}

// A design drawn or described from scratch has no source file. Listening for
// the mode change keeps that automatic instead of asking every caller to
// remember it.
window.addEventListener("otc:mode-selected", (event) => {
  const mode = event?.detail?.mode;
  if (mode === "draw" || mode === "chat") clearSource();
});

document.addEventListener("DOMContentLoaded", () => {
  hintEl = document.getElementById("design-name-hint");
  bindInput(document.getElementById("design-name"));
});
