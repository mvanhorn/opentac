// Two-piece printed case: generate, preview either half, download both.
//
// Separate from app.js because it is another output part with its own
// lifecycle, rebuilt several times against one board while tuning the fit.
//
// The shear bumps are a setting here, not a separate part: "flat lid" and
// "bumped lid" are two versions of the same top piece.
//
// Download ZIP also puts the case in the ZIP (case/, every lid), built from
// this panel's settings via otcCaseZipSettings; this panel is for previewing
// and tuning it, and for downloading single parts.
import { API_BASE } from "./config.js";
import { downloadName } from "./design-name.js";
import { resetBumpView, showBumpStl } from "./bump-viewer.js";

const enableEl = document.getElementById("case-enable");
const bodyEl = document.getElementById("case-body");
const generateBtn = document.getElementById("case-generate");
const dlBottomBtn = document.getElementById("case-dl-bottom");
const dlTopBtn = document.getElementById("case-dl-top");
const statusEl = document.getElementById("case-status");
const containerEl = document.getElementById("case-container");
const statsEl = document.getElementById("case-stats");
const bumpFieldsEl = document.getElementById("case-bump-fields");
const hatchFieldsEl = document.getElementById("case-hatch-fields");
const hatchBumpFieldsEl = document.getElementById("case-hatch-bump-fields");
const hatchBumpsEl = document.getElementById("case-hatch-bumps");
const showSel = document.getElementById("case-show");

const F = (id) => document.getElementById(id);
const num = (el, d) => {
  const v = Number.parseFloat(el.value);
  return Number.isFinite(v) ? v : d;
};

let boardReady = false;
let bottomBytes = null;
let topBytes = null;
let busy = false;
// Whether the user has picked a lid here (changed the choice, or built a case
// with it). Until then the ZIP's primary lid is the bumped one, whatever the
// radio's initial value -- see otcCaseZipSettings below.
let lidChosen = false;

function lidStyle() {
  const sel = document.querySelector('input[name="case-top"]:checked');
  return sel ? sel.value : "flat";
}

function updateButtons() {
  generateBtn.disabled = busy || !boardReady;
  dlBottomBtn.disabled = !bottomBytes;
  dlTopBtn.disabled = !topBytes;
  showSel.disabled = !bottomBytes;
  if (bumpFieldsEl) bumpFieldsEl.hidden = lidStyle() !== "bumps";
  if (hatchFieldsEl) hatchFieldsEl.hidden = lidStyle() !== "hatch";
  if (hatchBumpFieldsEl) {
    hatchBumpFieldsEl.hidden = lidStyle() !== "hatch" || !hatchBumpsEl.checked;
  }
}

/** Called by app.js when a board finishes generating. */
window.otcCaseBoardReady = (ready) => {
  boardReady = !!ready;
  if (boardReady && enableEl.checked && !bottomBytes) {
    statusEl.textContent = "Board ready — click Generate case.";
  }
  updateButtons();
};

/** Called by app.js before a new board generates: the current case will not fit it. */
window.otcCaseInvalidate = () => {
  boardReady = false;
  bottomBytes = null;
  topBytes = null;
  containerEl.hidden = true;
  statsEl.hidden = true;
  statusEl.textContent = "";
  updateButtons();
};

enableEl.addEventListener("change", () => {
  bodyEl.hidden = !enableEl.checked;
  if (enableEl.checked && !boardReady) {
    statusEl.textContent = "Generate a board first, then build a case for it.";
  }
  updateButtons();
});

for (const el of document.querySelectorAll('input[name="case-top"]')) {
  el.addEventListener("change", () => {
    lidChosen = true;
    updateButtons();
  });
}
hatchBumpsEl.addEventListener("change", updateButtons);

function preview() {
  const bytes = showSel.value === "top" ? topBytes : bottomBytes;
  if (!bytes) return;
  containerEl.hidden = false;
  showBumpStl(containerEl, bytes.buffer.slice(0));
}
showSel.addEventListener("change", preview);

function download(bytes, name) {
  const url = URL.createObjectURL(new Blob([bytes], { type: "model/stl" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

dlBottomBtn.addEventListener("click", () =>
  bottomBytes && download(bottomBytes, downloadName("case_bottom", "stl")));
dlTopBtn.addEventListener("click", () =>
  topBytes && download(topBytes, downloadName("case_top", "stl")));

// The panel's fields as CaseParams / BumpParams overrides. Read whether or not
// the panel is open: untouched fields hold the backend's defaults, so this is
// "the user's settings, or the defaults" without tracking which is which.
function caseParams() {
  return {
    recess_depth_mm: num(F("case-recess"), 0.2),
    plate_thickness_mm: num(F("case-plate"), 0.6),
    lid_thickness_mm: num(F("case-lid"), 0.8),
    skirt_width_mm: num(F("case-skirt"), 4.0),
    rib_period_mm: num(F("case-period"), 20),
    rib_gap_mm: num(F("case-gap"), 10),
    recess_tolerance_mm: num(F("case-tolerance"), 0.2),
    bumps: lidStyle() === "bumps",
    hatch: lidStyle() === "hatch",
    hatch_through: F("case-hatch-through").checked,
    hatch_depth_mm: num(F("case-hatch-depth"), 0.4),
    hatch_groove_mm: num(F("case-hatch-width"), 0.9),
    hatch_tab_mm: num(F("case-hatch-tab"), 1.2),
    hatch_bumps: hatchBumpsEl.checked,
    hatch_bump_dia_mm: num(F("case-hatch-bump-dia"), 1.8),
    hatch_bump_height_mm: num(F("case-hatch-bump-height"), 0.6),
  };
}

function bumpParams() {
  return {
    bump_height_mm: num(F("case-bump-height"), 2.0),
    bump_base_frac: num(F("case-bump-base"), 80) / 100,
    bump_top_ratio: num(F("case-bump-top"), 70) / 100,
  };
}

/**
 * Called by app.js when Download ZIP builds the case into the ZIP. Every lid
 * goes in (the backend overrides bumps/hatch per lid); the lid choice only
 * decides which one case/README.txt names as primary, and that is the bumped
 * lid until the user picks one here.
 */
window.otcCaseZipSettings = () => ({
  params: caseParams(),
  bump_params: bumpParams(),
  primary: lidChosen ? lidStyle() : "bumps",
});

generateBtn.addEventListener("click", async () => {
  const editData = window.otcGetEditData?.();
  if (!editData) {
    statusEl.innerHTML = '<span class="error">Generate a board first.</span>';
    return;
  }
  // Building a case with a lid is choosing it, even if it is the preselected one.
  lidChosen = true;
  busy = true;
  updateButtons();
  statusEl.textContent = "Building the case…";
  try {
    const params = caseParams();
    const body = { edit_data: editData, params };
    if (params.bumps) body.bump_params = bumpParams();
    const resp = await fetch(`${API_BASE}/case`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { detail = (await resp.json()).detail || detail; } catch (e) { /* keep status */ }
      throw new Error(detail);
    }
    const data = await resp.json();
    const decode = (b64) => Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
    bottomBytes = decode(data.bottom_stl_b64);
    topBytes = decode(data.top_stl_b64);
    renderStats(data.stats, data.warnings);
    preview();
    statusEl.textContent = "Done — two parts to print.";
  } catch (err) {
    statusEl.innerHTML = `<span class="error">${err.message}</span>`;
  } finally {
    busy = false;
    updateButtons();
  }
});

function renderStats(s, warnings) {
  const rows = [
    ["Case size", `${s.case_w_mm} × ${s.case_h_mm} mm`],
    ["Closed height", `${s.closed_height_mm} mm`],
    ["Rib", `${s.rib_segments} segments, ${s.rib_engaged_mm} mm engaged`],
    ["Bridges", `${s.bridges} gaps, ${s.bridge_mm} mm holding the lid`],
    ["Pattern", `${s.rib_period_mm} mm period — ${(s.rib_period_mm - s.rib_gap_mm).toFixed(0)} rib / ${s.rib_gap_mm} gap`],
    ["Undercut", `${s.undercut_mm} mm per side`],
    ["Cable clear of case", `${s.tail_flex_mm} mm of flex before the connector`],
  ];
  if (s.bumps) rows.push(["Bumps on lid", `${s.bumps}`]);
  if (s.hatch) {
    rows.push(["Hatched lid", s.hatch_through
      ? "cut through, tabs holding it"
      : `grooved, ${s.hatch_web_mm} mm web`]);
    if (s.hatch_bumps) {
      rows.push(["Bumps per taxel", `${s.hatch_bumps} on the ${s.hatch_bump_face} face`]);
    }
  }
  if (s.lid_pieces > 1) {
    rows.push(["Lid", `${s.lid_pieces} loose pieces — increase the gap`]);
  }
  const warn = (warnings || []).length
    ? `<ul class="bump-warnings">${warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`
    : "";
  statsEl.innerHTML =
    `<dl class="bump-stats">${rows.map(([k, v]) =>
      `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>${warn}`;
  statsEl.hidden = false;
}

updateButtons();
