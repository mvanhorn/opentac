// DXF outline import: uploads a .dxf file to POST /import-dxf and hands the
// chosen closed loop off to app.js via the same otc:*-outline event
// convention stl-viewer.js already uses for STL face unfolding.
import { API_BASE } from "./config.js";
import { setSource } from "./design-name.js";

const uploadBtn = document.getElementById("upload-dxf");
const fileInput = document.getElementById("dxf-input");
const infoEl = document.getElementById("dxf-info");

uploadBtn?.addEventListener("click", () => fileInput.click());

fileInput?.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  fileInput.value = "";
  if (!file) return;

  infoEl.textContent = `Importing ${file.name}…`;
  setSource(file.name);
  infoEl.classList.remove("error");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch(`${API_BASE}/import-dxf`, { method: "POST", body: formData });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const chosen = data.loops[data.chosen_index];
    infoEl.textContent = data.warnings && data.warnings.length
      ? data.warnings.join(" ")
      : `Imported ${chosen.length}-point outline from ${file.name}.`;
    window.dispatchEvent(new CustomEvent("otc:dxf-outline", {
      detail: { outline: chosen, warnings: data.warnings || [] },
    }));
  } catch (err) {
    infoEl.innerHTML = `<span class="error">Import failed: ${err.message}</span>`;
  }
});
