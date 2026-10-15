// Re-open a design that was downloaded earlier, and keep working on it.
//
// The user is handed a ZIP, so the ZIP is what they get to hand back: this
// reads the archive in the browser and pulls out whichever design record it
// contains. A loose design.json or manifest.json works too, for anyone who
// already unzipped.
//
// Two records can be in there (backend/export/design.py writes both), and
// they mean different things:
//
//   design.json    the SAVE FILE — the editor's own state. Restored through
//                  /export-edited, so the board comes back exactly as it was
//                  downloaded, hand-dragged routes and all, with nothing
//                  re-routed. Only valid against the pipeline that wrote it.
//   manifest.json  the RECIPE — the inputs. Re-generated from scratch on
//                  today's pipeline. Survives a version change; loses hand
//                  edits.
//
// design.json is preferred when present and current. The backend decides
// which is usable (/open-design) rather than this file guessing, so the
// version rule lives in one place; here we just do what it says.

import { API_BASE } from "./config.js";
import { setSource } from "./design-name.js";

const openBtn = document.getElementById("open-design");
const fileInput = document.getElementById("design-input");
const infoEl = document.getElementById("design-import-info");

const say = (html, isError = false) => {
  if (!infoEl) return;
  infoEl.innerHTML = isError ? `<span class="error">${html}</span>` : html;
};

// ---------------------------------------------------------------- ZIP -----
// A ~60-line reader instead of a zip dependency: we need exactly one small
// JSON entry out of an archive we wrote ourselves, and DecompressionStream is
// in every browser this app already requires for its other features. Anything
// unusual (Zip64, an encrypted or stored-in-a-weird-way entry) throws with a
// message telling the user to unzip by hand, which always works.
async function inflateRaw(bytes) {
  if (typeof DecompressionStream !== "function") {
    throw new Error(
      "this browser cannot open ZIPs here — unzip the file and pick design.json instead");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// `wanted` is a PREFERENCE ORDER, not a set: the whole directory is indexed
// first and then resolved against that order. Returning the first archive
// entry that happened to match instead handed back manifest.json every time,
// because design.json is appended last — so an exact restore silently
// degraded into a re-generate and any hand-edited routes were lost.
export async function readZipEntry(buffer, wanted) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // End of central directory: last 22 bytes, unless there is a trailing
  // comment, so scan backwards for the signature.
  let eocd = -1;
  for (let i = bytes.length - 22; i >= 0 && i > bytes.length - 22 - 65536; i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("not a ZIP file");
  const count = view.getUint16(eocd + 10, true);
  let ptr = view.getUint32(eocd + 16, true);
  if (ptr === 0xffffffff) throw new Error("Zip64 archives are not supported here");

  const decoder = new TextDecoder();
  const index = new Map();
  for (let n = 0; n < count; n++) {
    if (view.getUint32(ptr, true) !== 0x02014b50) throw new Error("corrupt ZIP directory");
    const nameLen = view.getUint16(ptr + 28, true);
    const extraLen = view.getUint16(ptr + 30, true);
    const commentLen = view.getUint16(ptr + 32, true);
    const name = decoder.decode(bytes.subarray(ptr + 46, ptr + 46 + nameLen));
    index.set(name, {
      method: view.getUint16(ptr + 10, true),
      compressedSize: view.getUint32(ptr + 20, true),
      localOffset: view.getUint32(ptr + 42, true),
    });
    ptr += 46 + nameLen + extraLen + commentLen;
  }

  for (const name of wanted) {
    const e = index.get(name);
    if (!e) continue;
    // The central directory's name/extra lengths are NOT the local header's —
    // the local one carries its own, and they differ in practice. Read them
    // from the local header or the data slice starts in the wrong place.
    if (view.getUint32(e.localOffset, true) !== 0x04034b50) {
      throw new Error("corrupt ZIP entry");
    }
    const dataStart = e.localOffset + 30
      + view.getUint16(e.localOffset + 26, true)
      + view.getUint16(e.localOffset + 28, true);
    const raw = bytes.subarray(dataStart, dataStart + e.compressedSize);
    return { name, text: decoder.decode(e.method === 0 ? raw : await inflateRaw(raw)) };
  }
  return null;
}

// ------------------------------------------------------------- loading ----

async function fileToDesign(file) {
  const isZip = /\.zip$/i.test(file.name);
  if (!isZip) {
    return { source: file.name, data: JSON.parse(await file.text()) };
  }
  // design.json first: it can restore the board exactly, and manifest.json
  // can only re-generate it.
  const entry = await readZipEntry(await file.arrayBuffer(), ["design.json", "manifest.json"]);
  if (!entry) {
    throw new Error(
      "this ZIP has no design.json or manifest.json — it may not be an OpenTactileCAD export");
  }
  return { source: `${file.name} (${entry.name})`, data: JSON.parse(entry.text) };
}

async function post(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

async function openDesign(file) {
  const label = file.name.replace(/\.(zip|json)$/i, "");
  say(`Reading ${file.name}…`);
  const { source, data } = await fileToDesign(file);

  const opened = await post("/open-design", { file: data });
  const warnings = (opened.warnings || []).join(" ");

  if (opened.mode === "geometry") {
    // A board from before the design block existed. Its geometry is exactly
    // what was fabricated, so it goes straight to the canvas with no round
    // trip through /export-edited — there is no layout to run through it, and
    // re-routing to manufacture a new one would give copper that does not
    // match the board the user already has.
    setSource(label);
    window.dispatchEvent(new CustomEvent("otc:restore-design", {
      detail: {
        outline: opened.outline,
        params: opened.edit_data?.params || {},
        fillRegion: null,
        editData: opened.edit_data,
        svg: opened.svg,
        label,
        geometryOnly: true,
        notes: opened.warnings || [],
      },
    }));
    say(`Reopened ${source} as a finished board — case and bump sheet are available. ` +
        `It predates stored settings, so new manufacturing files cannot be built from it.`);
    return;
  }

  if (opened.mode === "restore") {
    say(`Restoring the board from ${source}…`);
    // The same call the route editor's own "Check DRC" makes. Re-running it
    // rather than trusting the DRC stored in the file is the point: a clean
    // number in a downloaded file is not evidence that this build agrees.
    const built = await post("/export-edited", {
      edit_data: opened.edit_data,
      routes: opened.routes,
    });
    setSource(label);
    window.dispatchEvent(new CustomEvent("otc:restore-design", {
      detail: {
        outline: opened.outline,
        params: opened.params,
        fillRegion: opened.fill_region,
        editData: opened.edit_data,
        svg: built.svg,
        stats: built.stats,
        drc: built.drc,
        zipB64: built.zip_b64,
        label,
      },
    }));
    say(built.drc?.violations
      ? `Reopened ${source} exactly, with ${built.drc.violations} DRC violation(s) — the same board you downloaded.`
      : `Reopened ${source} exactly. DRC clean.`);
    return;
  }

  // Recipe: hand it to the same path an example uses, so a re-opened design
  // is editable in exactly the same way as one that was never exported.
  setSource(label);
  window.dispatchEvent(new CustomEvent("otc:load-example", {
    detail: {
      outline: opened.outline,
      params: opened.params,
      fillRegion: opened.fill_region,
      label,
      // The cable edge was decided when this design was first made, and the
      // stored outline is already in the frame that encodes it.
      requireCableEdge: false,
      autoGenerate: true,
    },
  }));
  say(warnings
    ? `Rebuilding ${source} from its saved settings. ${warnings}`
    : `Rebuilding ${source} from its saved settings…`);
}

openBtn?.addEventListener("click", () => fileInput?.click());

fileInput?.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  fileInput.value = "";
  if (!file) return;
  try {
    await openDesign(file);
  } catch (err) {
    say(`Could not open ${file.name}: ${err.message}`, true);
  }
});
