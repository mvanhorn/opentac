// Optional shear-sensing bump sheet: toggle, generate, preview in 3D, download.
//
// Kept out of app.js because it is a second output part with its own lifecycle
// — you can rebuild the sheet several times against one board while tuning
// bump height, without re-running the board.
import { API_BASE } from "./config.js";
import { downloadName } from "./design-name.js";
import { resetBumpView, showBumpStl } from "./bump-viewer.js";

const enableEl = document.getElementById("bump-enable");
const bodyEl = document.getElementById("bump-body");
const generateBtn = document.getElementById("bump-generate");
const downloadBtn = document.getElementById("bump-download");
const resetViewBtn = document.getElementById("bump-reset-view");
const statusEl = document.getElementById("bump-status");
const containerEl = document.getElementById("bump-container");
const statsEl = document.getElementById("bump-stats");

const heightEl = document.getElementById("bump-height");
const recessEl = document.getElementById("bump-recess");
const sheetEl = document.getElementById("bump-sheet");
const skirtEl = document.getElementById("bump-skirt");
const toleranceEl = document.getElementById("bump-tolerance");
const tabWidthEl = document.getElementById("bump-tab-width");
const tabsEl = document.getElementById("bump-tabs");
const basePctEl = document.getElementById("bump-base-pct");
const topPctEl = document.getElementById("bump-top-pct");

let boardReady = false;
let lastStlBytes = null;
let busy = false;

function num(el, fallback) {
  const v = Number.parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
}

function updateButtons() {
  generateBtn.disabled = busy || !boardReady;
  downloadBtn.disabled = !lastStlBytes;
  resetViewBtn.disabled = !lastStlBytes;
}

/** Called by app.js when a board finishes generating. */
window.otcBumpBoardReady = (ready) => {
  boardReady = !!ready;
  if (boardReady && enableEl.checked && !lastStlBytes) {
    statusEl.textContent = "Board ready — click Generate bump sheet.";
  }
  updateButtons();
};

/** Called by app.js before a new board generates: the old sheet no longer fits. */
window.otcBumpInvalidate = () => {
  boardReady = false;
  lastStlBytes = null;
  containerEl.hidden = true;
  statsEl.hidden = true;
  statusEl.textContent = "";
  updateButtons();
};

enableEl.addEventListener("change", () => {
  bodyEl.hidden = !enableEl.checked;
  if (enableEl.checked && !boardReady) {
    statusEl.textContent = "Generate a board first, then build the sheet for it.";
  }
  updateButtons();
});

generateBtn.addEventListener("click", async () => {
  const editData = window.otcGetEditData?.();
  if (!editData) {
    statusEl.innerHTML = '<span class="error">No board to build a sheet for.</span>';
    return;
  }
  busy = true;
  updateButtons();
  statusEl.textContent = "Building bump sheet… (a few seconds)";
  try {
    const resp = await fetch(`${API_BASE}/bump-sheet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        edit_data: editData,
        // The two ratios are shown as percentages because "80% of the block"
        // is readable and "0.8" is not; the backend wants the fraction.
        params: {
          bump_height_mm: num(heightEl, 2.0),
          recess_depth_mm: num(recessEl, 1.0),
          sheet_thickness_mm: num(sheetEl, 0.8),
          skirt_width_mm: num(skirtEl, 3.0),
          recess_tolerance_mm: num(toleranceEl, 0.2),
          tab_width_mm: num(tabWidthEl, 1.2),
          tabs_per_edge: Math.round(num(tabsEl, 1)),
          bump_base_frac: num(basePctEl, 80) / 100,
          bump_top_ratio: num(topPctEl, 70) / 100,
        },
      }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();

    const binary = atob(data.stl_b64);
    lastStlBytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) lastStlBytes[i] = binary.charCodeAt(i);

    containerEl.hidden = false;
    // The ArrayBuffer is copied because the viewer keeps a reference to the
    // geometry it parses, while lastStlBytes is reused for the download.
    showBumpStl(containerEl, lastStlBytes.buffer.slice(0));
    renderStats(data.stats, data.warnings);
    statusEl.textContent = data.warnings.length
      ? "Built, with warnings — see below."
      : "Bump sheet ready. Drag to rotate.";
  } catch (err) {
    statusEl.innerHTML = `<span class="error">Bump sheet failed: ${err.message}</span>`;
    lastStlBytes = null;
    containerEl.hidden = true;
    statsEl.hidden = true;
  } finally {
    busy = false;
    updateButtons();
  }
});

function renderStats(s, warnings) {
  const oval = s.bump_base_x_mm === s.bump_base_y_mm
    ? `${s.bump_base_x_mm} mm round`
    : `${s.bump_base_x_mm} × ${s.bump_base_y_mm} mm oval`;
  const rows = [
    ["Bumps", `${s.bumps} (${oval} at the base, ${s.bump_height_mm} mm tall)`],
    ["Sensors covered", `all ${s.taxels_total}` +
      (s.partial_bumps > 0
        ? ` — ${s.partial_bumps} bump(s) sit over a part-filled group, so they overhang a little`
        : "")],
    ["Sheet", `${s.sheet_w_mm} × ${s.sheet_h_mm} mm, ${s.total_height_mm} mm tall overall`],
    ["Pocket", `${s.recess_depth_mm} mm deep for the sensor + ${s.recess_tolerance_mm} mm clearance, ` +
      `${s.sheet_thickness_mm} mm sheet, ${s.skirt_width_mm} mm skirt`],
    ["Decoupling", `${s.tabs_per_edge} tab(s) of ${s.tab_width_mm} mm per edge; ` +
      `bump is ${s.bump_base_pct}% of its block, tapering to ${s.bump_top_pct}%`],
    ["Mesh", `${s.triangles.toLocaleString()} triangles, ${s.bodies} solid piece(s)` +
      (s.watertight ? ", watertight" : ", NOT watertight")],
  ];
  const warnHtml = warnings.length
    ? `<ul class="bump-warnings">${warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`
    : "";
  statsEl.innerHTML =
    `<dl class="bump-stats">${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>${warnHtml}`;
  statsEl.hidden = false;
}

downloadBtn.addEventListener("click", () => {
  if (!lastStlBytes) return;
  const blob = new Blob([lastStlBytes], { type: "model/stl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = downloadName("bump_sheet", "stl");
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

resetViewBtn.addEventListener("click", () => resetBumpView());

updateButtons();
