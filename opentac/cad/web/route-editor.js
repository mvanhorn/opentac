// Post-generation route editing for generated PCB previews.
// Keeps generation untouched: edits are held in browser state, then submitted to
// /export-edited so the final ZIP is rebuilt from the modified polylines.

import { downloadName } from "./design-name.js";

const SNAP_MM = 0.1;

function cloneRoutes(routes) {
  return routes.map((route) => ({
    ...route,
    anchor_xy: route.anchor_xy.slice(),
    pad_xy: route.pad_xy.slice(),
    polyline_xy: route.polyline_xy.map((p) => p.slice()),
  }));
}

function snap(v) {
  return Math.round(v / SNAP_MM) * SNAP_MM;
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function nearestSegment(points, p) {
  let best = { index: -1, distance: Infinity };
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const vx = b[0] - a[0];
    const vy = b[1] - a[1];
    const len2 = vx * vx + vy * vy;
    const t = len2 > 0
      ? Math.max(0, Math.min(1, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2))
      : 0;
    const q = [a[0] + t * vx, a[1] + t * vy];
    const d = dist(p, q);
    if (d < best.distance) best = { index: i, distance: d };
  }
  return best;
}

export function createRouteEditor({
  apiBase,
  previewEl,
  statusEl,
  downloadZip,
  renderStats,
  readParams,
  onTailExtensionApplied,
}) {
  const toolsEl = document.getElementById("route-tools");
  const viewToolsEl = document.getElementById("pcb-view-tools");
  const zoomOutBtn = document.getElementById("pcb-zoom-out");
  const zoomInBtn = document.getElementById("pcb-zoom-in");
  const zoomResetBtn = document.getElementById("pcb-zoom-reset");
  const undoBtn = document.getElementById("route-undo");
  const resetBtn = document.getElementById("route-reset");
  const resetAllBtn = document.getElementById("route-reset-all");
  const redoPaddingBtn = document.getElementById("route-redo-padding");
  const chamferInput = document.getElementById("route-chamfer-mm");
  const chamferBtn = document.getElementById("route-chamfer");
  const checkDrcBtn = document.getElementById("route-check-drc");
  const tailToggleBtn = document.getElementById("tail-extension-toggle");
  const tailPanel = document.getElementById("tail-extension-panel");
  const tailLengthInput = document.getElementById("tail-extension-length");
  const tailAngleInput = document.getElementById("tail-extension-angle");
  const tailWidthInput = document.getElementById("tail-extension-width");
  const tailAddBtn = document.getElementById("tail-extension-add");
  // Preset directions, in the same on-screen sense as the preview: 0° points
  // right, positive angles go up. (The old Vertical/Horizontal pair could only
  // reach down and right.)
  const tailDirButtons = [
    ["tail-extension-dir-down", -90],
    ["tail-extension-dir-up", 90],
    ["tail-extension-dir-right", 0],
    ["tail-extension-dir-left", 180],
    ["tail-extension-dir-down-right", -45],
    ["tail-extension-dir-down-left", -135],
    ["tail-extension-dir-up-right", 45],
    ["tail-extension-dir-up-left", 135],
  ];
  const tailUndoBtn = document.getElementById("tail-extension-undo");
  const tailResetBtn = document.getElementById("tail-extension-reset");
  const tailApplyBtn = document.getElementById("tail-extension-apply");
  const tailInfoEl = document.getElementById("tail-extension-info");

  let editData = null;
  let originalRoutes = [];
  let routes = [];
  let svg = null;
  let routeEls = [];
  let handleLayer = null;
  let selected = null;
  let drag = null;
  let undoStack = [];
  let dirty = false;
  let drcStale = false;
  let tailPath = [];
  let tailPreviewEl = null;
  let connectorPreviewEl = null;
  let originalViewBox = null;
  let currentViewBox = null;
  let panDrag = null;
  let visualPreviewOnly = false;

  function hasEditableRoutes() {
    return !!editData && routes.length > 0;
  }

  function mmToSvg(p) {
    const b = editData.preview;
    return [p[0] - b.xmin, b.ymax - p[1]];
  }

  function svgToMm(p) {
    const b = editData.preview;
    return [snap(p[0] + b.xmin), snap(b.ymax - p[1])];
  }

  function eventToSvgPoint(event) {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const out = point.matrixTransform(svg.getScreenCTM().inverse());
    return [out.x, out.y];
  }

  function routePointsAttr(route) {
    return route.polyline_xy
      .map((p) => {
        const q = mmToSvg(p);
        return `${q[0].toFixed(3)},${q[1].toFixed(3)}`;
      })
      .join(" ");
  }

  function setStatus(text, isError = false) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", isError);
  }

  function pushUndo() {
    undoStack.push(cloneRoutes(routes));
    if (undoStack.length > 40) undoStack.shift();
    updateButtons();
  }

  function markDirty() {
    dirty = true;
    drcStale = true;
    setStatus("Route edits pending. Check DRC or download edited ZIP when ready.");
    updateButtons();
  }

  function updateButtons() {
    if (!toolsEl) return;
    toolsEl.hidden = !hasEditableRoutes();
    if (viewToolsEl) viewToolsEl.hidden = !svg;
    undoBtn.disabled = visualPreviewOnly || undoStack.length === 0;
    resetBtn.disabled = visualPreviewOnly || !selected;
    resetAllBtn.disabled = visualPreviewOnly || !hasEditableRoutes() || !dirty;
    if (redoPaddingBtn) redoPaddingBtn.disabled = !hasEditableRoutes();
    checkDrcBtn.disabled = !hasEditableRoutes();
    if (tailToggleBtn) tailToggleBtn.disabled = !hasEditableRoutes();
    chamferBtn.disabled = visualPreviewOnly || !selected || selected.kind !== "vertex" || !canChamfer(selected.routeIndex, selected.pointIndex);
  }

  function updateRouteElement(index) {
    if (routeEls[index]) routeEls[index].setAttribute("points", routePointsAttr(routes[index]));
  }

  function clearHandles() {
    if (handleLayer) handleLayer.replaceChildren();
  }

  function readViewBox() {
    const raw = svg?.getAttribute("viewBox");
    if (!raw) return null;
    const parts = raw.split(/\s+/).map(Number);
    if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v))) return null;
    return { x: parts[0], y: parts[1], w: parts[2], h: parts[3] };
  }

  function setViewBox(vb) {
    if (!svg || !vb) return;
    currentViewBox = vb;
    svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
  }

  function zoomSvg(factor, center = null) {
    if (!svg || !currentViewBox) return;
    const c = center || [currentViewBox.x + currentViewBox.w / 2, currentViewBox.y + currentViewBox.h / 2];
    const nextW = currentViewBox.w * factor;
    const nextH = currentViewBox.h * factor;
    const minW = originalViewBox.w / 20;
    const maxW = originalViewBox.w * 8;
    if (nextW < minW || nextW > maxW) return;
    const rx = (c[0] - currentViewBox.x) / currentViewBox.w;
    const ry = (c[1] - currentViewBox.y) / currentViewBox.h;
    setViewBox({
      x: c[0] - rx * nextW,
      y: c[1] - ry * nextH,
      w: nextW,
      h: nextH,
    });
  }

  function resetSvgView() {
    if (originalViewBox) setViewBox({ ...originalViewBox });
  }

  function beginPan(event) {
    if (!svg || !currentViewBox) return;
    if (event.target.closest?.(".editable-route,.route-handle")) return;
    event.preventDefault();
    panDrag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      viewBox: { ...currentViewBox },
    };
    svg.setPointerCapture(event.pointerId);
  }

  function tailTrueCenter() {
    const split = editData?.layout?.tail?.pad_split_center;
    if (split?.length >= 2) return split.slice();
    const center = editData?.layout?.tail?.pad_row_center;
    if (center?.length >= 2) return center.slice();
    return [0, editData?.layout?.pad_y_mm || 0];
  }

  function tailStartPoint() {
    const ext = editData?.layout?.tail_extension;
    if (ext?.path?.length) return ext.path[0].slice();
    return tailTrueCenter();
  }

  function defaultTailExtensionWidth() {
    const explicit = Number.parseFloat(editData?.layout?.tail_extension?.width_mm);
    if (Number.isFinite(explicit) && explicit > 0) return explicit;
    const tailWidth = Number.parseFloat(editData?.layout?.tail?.width_mm);
    return Number.isFinite(tailWidth) ? Math.max(0, tailWidth - 0.7) : 0;
  }

  function tailLength() {
    let total = 0;
    for (let i = 0; i < tailPath.length - 1; i++) total += dist(tailPath[i], tailPath[i + 1]);
    return total;
  }

  function setTailInfo() {
    if (tailInfoEl) {
      tailInfoEl.textContent = `${Math.max(0, tailPath.length - 1)} segment(s) · ${tailLength().toFixed(1)} mm`;
    }
  }

  function renderTailPreview() {
    if (!svg) return;
    if (!tailPreviewEl) {
      tailPreviewEl = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      tailPreviewEl.setAttribute("class", "tail-extension-preview");
      svg.appendChild(tailPreviewEl);
    }
    tailPreviewEl.setAttribute("points", tailPath.map((p) => {
      const q = mmToSvg(p);
      return `${q[0].toFixed(3)},${q[1].toFixed(3)}`;
    }).join(" "));
    renderConnectorPreview(tailPath);
    setTailInfo();
  }

  function connectorDimensions() {
    const pos = Number(editData?.layout?.connector_pos) || 0;
    const pitch = Number(editData?.params?.pad_pitch_mm) || 0.5;
    if (!pos) return null;
    return {
      pos,
      tipWidth: (pos + 1) * pitch,
      bodyDepth: 6.0,
      earDepth: 1.1,
      earOut: 0.55,
      chamfer: 0.28,
    };
  }

  function unit(v) {
    const n = Math.hypot(v[0], v[1]);
    return n > 1e-9 ? [v[0] / n, v[1] / n] : null;
  }

  function connectorPoseFromPath(path) {
    if (!path || path.length < 2) return null;
    const end = path[path.length - 1];
    const prev = path[path.length - 2];
    const nOut = unit([end[0] - prev[0], end[1] - prev[1]]);
    if (!nOut) return null;
    return { center: end, tHat: [-nOut[1], nOut[0]], nOut };
  }

  function worldPoint(center, tHat, nOut, tx, ny) {
    return [
      center[0] + tx * tHat[0] + ny * nOut[0],
      center[1] + tx * tHat[1] + ny * nOut[1],
    ];
  }

  function polygonPointsAttr(points) {
    return points.map((p) => {
      const q = mmToSvg(p);
      return `${q[0].toFixed(3)},${q[1].toFixed(3)}`;
    }).join(" ");
  }

  function connectorEarPoints(center, tHat, nOut, side, dims) {
    const halfTip = dims.tipWidth / 2;
    const inner = side < 0 ? -halfTip : halfTip;
    const outer = inner + side * dims.earOut;
    // Ears sit near the connector's mechanical tip (the insertion edge,
    // away from the board — +ny, matching the label position below), not
    // the board-facing end. Verified against the real footprint files
    // (backend/connectors/*.kicad_mod): ears sit at local y in roughly
    // [-2.2, -1.05] out of a [-3.25 (tip), +2.75 (open/board end)] body —
    // i.e. between the tip and the middle, never near the open end. This
    // was previously mirrored to the wrong (board-facing) end — reported
    // directly against a real generated preview.
    const y1 = dims.bodyDepth / 2 - 0.8;
    const y0 = y1 - dims.earDepth;
    const c = dims.chamfer;
    const pts = side < 0
      ? [[inner, y0], [inner, y1], [outer + c, y1], [outer, y1 - c], [outer, y0 + c], [outer + c, y0]]
      : [[inner, y0], [outer - c, y0], [outer, y0 + c], [outer, y1 - c], [outer - c, y1], [inner, y1]];
    return pts.map(([tx, ny]) => worldPoint(center, tHat, nOut, tx, ny));
  }

  function renderConnectorPreview(path) {
    if (!svg) return;
    if (connectorPreviewEl) connectorPreviewEl.remove();
    connectorPreviewEl = null;
    const pose = connectorPoseFromPath(path);
    const dims = connectorDimensions();
    if (!pose || !dims) return;

    const { center, tHat, nOut } = pose;
    const halfTip = dims.tipWidth / 2;
    const halfBody = dims.bodyDepth / 2;
    connectorPreviewEl = document.createElementNS("http://www.w3.org/2000/svg", "g");
    connectorPreviewEl.setAttribute("class", "tail-connector-preview");

    const body = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    body.setAttribute("class", "tail-connector-preview__body");
    body.setAttribute("points", polygonPointsAttr([
      worldPoint(center, tHat, nOut, -halfTip, -halfBody),
      worldPoint(center, tHat, nOut, halfTip, -halfBody),
      worldPoint(center, tHat, nOut, halfTip, halfBody),
      worldPoint(center, tHat, nOut, -halfTip, halfBody),
    ]));
    connectorPreviewEl.appendChild(body);

    [-1, 1].forEach((side) => {
      const ear = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      ear.setAttribute("class", "tail-connector-preview__ear");
      ear.setAttribute("points", polygonPointsAttr(connectorEarPoints(center, tHat, nOut, side, dims)));
      connectorPreviewEl.appendChild(ear);
    });

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const labelPos = mmToSvg(worldPoint(center, tHat, nOut, 0, halfBody + 1.0));
    label.setAttribute("x", labelPos[0].toFixed(3));
    label.setAttribute("y", labelPos[1].toFixed(3));
    label.setAttribute("class", "tail-connector-preview__label");
    label.setAttribute("text-anchor", "middle");
    label.textContent = `${dims.pos}-pos connector`;
    connectorPreviewEl.appendChild(label);

    svg.appendChild(connectorPreviewEl);
  }

  function resetTailDraft() {
    const ext = editData?.layout?.tail_extension;
    tailPath = (ext?.enabled && ext?.path?.length > 1)
      ? ext.path.map((p) => p.slice())
      : [tailStartPoint()];
    if (tailWidthInput) tailWidthInput.value = defaultTailExtensionWidth() || "";
    renderTailPreview();
  }

  function addTailSegment(angleDeg = null) {
    if (!tailPath.length) tailPath = [tailStartPoint()];
    const length = Number.parseFloat(tailLengthInput?.value) || 20;
    const angle = angleDeg == null ? (Number.parseFloat(tailAngleInput?.value) || 0) : angleDeg;
    if (tailAngleInput && angleDeg != null) tailAngleInput.value = String(angleDeg);
    const last = tailPath[tailPath.length - 1];
    const rad = angle * Math.PI / 180;
    tailPath.push([snap(last[0] + length * Math.cos(rad)), snap(last[1] + length * Math.sin(rad))]);
    renderTailPreview();
  }

  async function applyTailExtension() {
    if (!editData?.layout) return;
    if (tailPath.length < 2) {
      editData.layout.tail_extension = {
        ...(editData.layout.tail_extension || {}),
        enabled: false,
        path: [tailStartPoint()],
      };
    } else {
      editData.layout.tail_extension = {
        enabled: true,
        path: tailPath.map((p, index) => index === 0 ? tailTrueCenter() : p.slice()),
        width_mm: Number.parseFloat(tailWidthInput?.value) || defaultTailExtensionWidth(),
        clearance_mm: editData.params.trace_w_mm / 2 + editData.params.board_edge_clear_mm,
      };
    }
    markDirty();
    try {
      await exportEdited({ download: false, renderPreview: true });
      onTailExtensionApplied?.();
    } catch (error) {
      setStatus(`Tail extension preview failed: ${error.message}`, true);
    }
  }

  function renderHandles() {
    clearHandles();
    if (!selected || !handleLayer) return;

    const route = routes[selected.routeIndex];
    route.polyline_xy.forEach((point, index) => {
      const [cx, cy] = mmToSvg(point);
      const handle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      handle.setAttribute("cx", cx.toFixed(3));
      handle.setAttribute("cy", cy.toFixed(3));
      handle.setAttribute("r", index === 0 || index === route.polyline_xy.length - 1 ? "0.16" : "0.28");
      handle.setAttribute("class", index === selected.pointIndex ? "route-handle route-handle--selected" : "route-handle");
      if (index > 0 && index < route.polyline_xy.length - 1) {
        handle.addEventListener("pointerdown", (event) => beginVertexDrag(event, selected.routeIndex, index));
      }
      handleLayer.appendChild(handle);
    });
  }

  function selectRoute(index, nextSelected = {}) {
    selected = { routeIndex: index, ...nextSelected };
    routeEls.forEach((el, i) => el.classList.toggle("route-edit-selected", i === index));
    renderHandles();
    updateButtons();
  }

  function beginVertexDrag(event, routeIndex, pointIndex) {
    event.preventDefault();
    event.stopPropagation();
    pushUndo();
    selectRoute(routeIndex, { kind: "vertex", pointIndex });
    drag = {
      kind: "vertex",
      routeIndex,
      pointIndex,
      originalPoint: routes[routeIndex].polyline_xy[pointIndex].slice(),
      moved: false,
    };
    svg.setPointerCapture(event.pointerId);
  }

  function beginSegmentDrag(event, routeIndex) {
    event.preventDefault();
    const mm = svgToMm(eventToSvgPoint(event));
    const nearest = nearestSegment(routes[routeIndex].polyline_xy, mm);
    selectRoute(routeIndex, { kind: "segment", segmentIndex: nearest.index });
    if (nearest.index <= 0 || nearest.index >= routes[routeIndex].polyline_xy.length - 2) {
      setStatus("End segments are locked because pad and pixel endpoints must stay fixed.");
      return;
    }
    pushUndo();
    drag = {
      kind: "segment",
      routeIndex,
      segmentIndex: nearest.index,
      startMm: mm,
      original: cloneRoutes([routes[routeIndex]])[0].polyline_xy,
      moved: false,
    };
    svg.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event) {
    if (panDrag && event.pointerId === panDrag.pointerId) {
      const rect = svg.getBoundingClientRect();
      const dx = (event.clientX - panDrag.x) * panDrag.viewBox.w / rect.width;
      const dy = (event.clientY - panDrag.y) * panDrag.viewBox.h / rect.height;
      setViewBox({
        ...panDrag.viewBox,
        x: panDrag.viewBox.x - dx,
        y: panDrag.viewBox.y - dy,
      });
      return;
    }
    if (!drag) return;
    const mm = svgToMm(eventToSvgPoint(event));
    const route = routes[drag.routeIndex];

    if (drag.kind === "vertex") {
      drag.moved = drag.moved || dist(mm, drag.originalPoint) > 1e-9;
      route.polyline_xy[drag.pointIndex] = mm;
    } else {
      const dx = snap(mm[0] - drag.startMm[0]);
      const dy = snap(mm[1] - drag.startMm[1]);
      drag.moved = drag.moved || Math.abs(dx) > 1e-9 || Math.abs(dy) > 1e-9;
      route.polyline_xy[drag.segmentIndex] = [
        snap(drag.original[drag.segmentIndex][0] + dx),
        snap(drag.original[drag.segmentIndex][1] + dy),
      ];
      route.polyline_xy[drag.segmentIndex + 1] = [
        snap(drag.original[drag.segmentIndex + 1][0] + dx),
        snap(drag.original[drag.segmentIndex + 1][1] + dy),
      ];
    }

    updateRouteElement(drag.routeIndex);
    renderHandles();
  }

  function onPointerUp(event) {
    if (panDrag && event.pointerId === panDrag.pointerId) {
      svg.releasePointerCapture(event.pointerId);
      panDrag = null;
      return;
    }
    if (!drag) return;
    svg.releasePointerCapture(event.pointerId);
    const moved = drag.moved;
    drag = null;
    if (moved) {
      markDirty();
    } else {
      undoStack.pop();
      updateButtons();
    }
  }

  function canChamfer(routeIndex, pointIndex) {
    if (routeIndex == null || pointIndex == null) return false;
    const route = routes[routeIndex];
    if (!route || pointIndex <= 0 || pointIndex >= route.polyline_xy.length - 1) return false;
    const d = Number.parseFloat(chamferInput.value) || 0.5;
    return dist(route.polyline_xy[pointIndex - 1], route.polyline_xy[pointIndex]) > d + 0.05
      && dist(route.polyline_xy[pointIndex + 1], route.polyline_xy[pointIndex]) > d + 0.05;
  }

  function applyChamfer() {
    if (!selected || selected.kind !== "vertex") return;
    const route = routes[selected.routeIndex];
    const i = selected.pointIndex;
    if (!canChamfer(selected.routeIndex, i)) {
      setStatus("Chamfer needs a selected interior corner with enough segment length.", true);
      return;
    }

    const d = Number.parseFloat(chamferInput.value) || 0.5;
    const prev = route.polyline_xy[i - 1];
    const corner = route.polyline_xy[i];
    const next = route.polyline_xy[i + 1];
    const lenPrev = dist(prev, corner);
    const lenNext = dist(next, corner);
    const p1 = [
      snap(corner[0] + ((prev[0] - corner[0]) / lenPrev) * d),
      snap(corner[1] + ((prev[1] - corner[1]) / lenPrev) * d),
    ];
    const p2 = [
      snap(corner[0] + ((next[0] - corner[0]) / lenNext) * d),
      snap(corner[1] + ((next[1] - corner[1]) / lenNext) * d),
    ];

    pushUndo();
    route.polyline_xy.splice(i, 1, p1, p2);
    selected = { routeIndex: selected.routeIndex, kind: "vertex", pointIndex: i };
    updateRouteElement(selected.routeIndex);
    renderHandles();
    markDirty();
  }

  async function exportEdited({ download = false, renderPreview = false } = {}) {
    if (!hasEditableRoutes()) return false;
    setStatus(download
      ? "Building edited ZIP..."
      : renderPreview
        ? "Building tail extension preview..."
        : "Checking edited-route DRC...");
    const resp = await fetch(`${apiBase}/export-edited`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ edit_data: editData, routes }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderStats(data.stats, data.drc);
    if (renderPreview && data.svg) {
      previewEl.innerHTML = data.svg;
      bindSvgViewOnly();
      renderConnectorPreview(editData?.layout?.tail_extension?.path || tailPath);
      // The edited export rebuilt the board, so the printable template that
      // shipped with the original generate is now for the wrong outline.
      window.otcPrintPdfUpdated?.(data.stats);
      visualPreviewOnly = true;
    }
    drcStale = false;
    if (data.drc.violations) {
      setStatus(renderPreview
        ? `Tail extension preview ready with ${data.drc.violations} DRC violation(s). Download remains available.`
        : `Edited export ready with ${data.drc.violations} DRC violation(s).`);
    } else {
      setStatus(renderPreview ? "Tail extension preview ready. DRC clean." : "Edited export ready. DRC clean.");
    }
    if (download) downloadZip(data.zip_b64, downloadName("edited", "zip"));
    updateButtons();
    return true;
  }

  async function redoPadding() {
    if (!hasEditableRoutes()) return;
    setStatus("Recalculating final board padding from current routes...");
    const resp = await fetch(`${apiBase}/redo-padding`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ edit_data: editData, routes, params: readParams ? readParams() : null }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderStats(data.stats, data.drc);
    load(data.edit_data, data.svg);
    renderConnectorPreview(editData?.layout?.tail_extension?.path || tailPath);
    setStatus(data.drc.violations
      ? `Padding updated with ${data.drc.violations} DRC violation(s). Download remains available.`
      : "Padding updated from current routes. DRC clean.");
  }

  async function resizeConnector(newConnectorPos) {
    if (!hasEditableRoutes()) return;
    setStatus(`Switching to a ${newConnectorPos}-pos connector...`);
    const resp = await fetch(`${apiBase}/redo-padding`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        edit_data: editData,
        routes,
        params: readParams ? readParams() : null,
        connector_pos_override: newConnectorPos,
      }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderStats(data.stats, data.drc);
    load(data.edit_data, data.svg);
    renderConnectorPreview(editData?.layout?.tail_extension?.path || tailPath);
    setStatus(data.drc.violations
      ? `Connector switched to ${newConnectorPos}-pos with ${data.drc.violations} DRC violation(s). Download remains available.`
      : `Connector switched to ${newConnectorPos}-pos. DRC clean.`);
    return data;
  }

  function getConnectorPos() {
    return Number(editData?.layout?.connector_pos) || null;
  }

  function bindSvgViewOnly() {
    svg = previewEl.querySelector("svg");
    if (!svg) return;
    svg.classList.add("route-editor-svg");
    originalViewBox = readViewBox();
    currentViewBox = originalViewBox ? { ...originalViewBox } : null;
    selected = null;
    drag = null;
    panDrag = null;
    routeEls = [];
    handleLayer = null;
    tailPreviewEl = null;
    connectorPreviewEl = null;

    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerup", onPointerUp);
    svg.addEventListener("pointercancel", onPointerUp);
    svg.addEventListener("pointerdown", beginPan);
    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const [cx, cy] = eventToSvgPoint(event);
      zoomSvg(event.deltaY < 0 ? 0.85 : 1.18, [cx, cy]);
    }, { passive: false });
  }

  function bindSvg() {
    svg = previewEl.querySelector("svg");
    if (!svg) return;
    svg.classList.add("route-editor-svg");
    originalViewBox = readViewBox();
    currentViewBox = originalViewBox ? { ...originalViewBox } : null;
    handleLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    handleLayer.setAttribute("class", "route-handle-layer");
    svg.appendChild(handleLayer);
    tailPreviewEl = null;
    connectorPreviewEl = null;

    const colEls = [...svg.querySelectorAll(".col-route")];
    const rowEls = [...svg.querySelectorAll(".row-route")];
    const fRoutes = routes.filter((r) => r.layer === "F.Cu");
    const bRoutes = routes.filter((r) => r.layer === "B.Cu");
    routeEls = [];

    [...fRoutes, ...bRoutes].forEach((route, orderIndex) => {
      const sourceList = route.layer === "F.Cu" ? fRoutes : bRoutes;
      const sourceIndex = sourceList.indexOf(route);
      const el = route.layer === "F.Cu" ? colEls[sourceIndex] : rowEls[sourceIndex];
      const routeIndex = routes.indexOf(route);
      if (!el || routeIndex < 0) return;
      el.dataset.routeIndex = String(routeIndex);
      el.classList.add("editable-route");
      el.setAttribute("points", routePointsAttr(route));
      el.addEventListener("pointerdown", (event) => beginSegmentDrag(event, routeIndex));
      routeEls[routeIndex] = el;
    });

    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerup", onPointerUp);
    svg.addEventListener("pointercancel", onPointerUp);
    svg.addEventListener("pointerdown", beginPan);
    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const [cx, cy] = eventToSvgPoint(event);
      zoomSvg(event.deltaY < 0 ? 0.85 : 1.18, [cx, cy]);
    }, { passive: false });
    if (tailPanel && !tailPanel.hidden) renderTailPreview();
  }

  function load(nextEditData, svgText) {
    editData = nextEditData;
    originalRoutes = cloneRoutes(editData.routes || []);
    routes = cloneRoutes(editData.routes || []);
    resetTailDraft();
    undoStack = [];
    dirty = false;
    drcStale = false;
    visualPreviewOnly = false;
    selected = null;
    drag = null;
    previewEl.innerHTML = svgText;
    bindSvg();
    setStatus(routes.length ? "Select a red or blue route to edit." : "");
    updateButtons();
  }

  undoBtn.addEventListener("click", () => {
    if (!undoStack.length) return;
    routes = undoStack.pop();
    routes.forEach((_, index) => updateRouteElement(index));
    renderHandles();
    dirty = true;
    drcStale = true;
    setStatus("Undo applied. DRC is stale.");
    updateButtons();
  });

  resetBtn.addEventListener("click", () => {
    if (!selected) return;
    pushUndo();
    routes[selected.routeIndex] = cloneRoutes([originalRoutes[selected.routeIndex]])[0];
    updateRouteElement(selected.routeIndex);
    renderHandles();
    markDirty();
  });

  resetAllBtn.addEventListener("click", () => {
    pushUndo();
    routes = cloneRoutes(originalRoutes);
    selected = null;
    routes.forEach((_, index) => updateRouteElement(index));
    renderHandles();
    markDirty();
  });

  chamferBtn.addEventListener("click", applyChamfer);
  chamferInput.addEventListener("input", updateButtons);
  redoPaddingBtn?.addEventListener("click", async () => {
    try {
      await redoPadding();
    } catch (error) {
      setStatus(`Redo padding failed: ${error.message}`, true);
    }
  });
  checkDrcBtn.addEventListener("click", async () => {
    try {
      await exportEdited({ download: false });
    } catch (error) {
      setStatus(`Edited DRC failed: ${error.message}`, true);
    }
  });
  tailToggleBtn?.addEventListener("click", () => {
    tailPanel.hidden = !tailPanel.hidden;
    if (!tailPanel.hidden) {
      resetTailDraft();
      renderTailPreview();
    }
  });
  tailAddBtn?.addEventListener("click", () => addTailSegment());
  for (const [id, angle] of tailDirButtons) {
    document.getElementById(id)?.addEventListener("click", () => addTailSegment(angle));
  }
  tailUndoBtn?.addEventListener("click", () => {
    if (tailPath.length > 1) tailPath.pop();
    renderTailPreview();
  });
  tailResetBtn?.addEventListener("click", () => {
    tailPath = [tailStartPoint()];
    renderTailPreview();
  });
  tailApplyBtn?.addEventListener("click", applyTailExtension);
  zoomOutBtn?.addEventListener("click", () => zoomSvg(1.18));
  zoomInBtn?.addEventListener("click", () => zoomSvg(0.85));
  zoomResetBtn?.addEventListener("click", resetSvgView);

  return {
    load,
    hasEditableRoutes,
    exportEdited,
    resizeConnector,
    getConnectorPos,
    // The live edit_data, including anything redo-padding or a connector swap
    // changed. The case wraps the FINAL outline, so it has to read this
    // rather than the response the board was first generated from.
    getEditData: () => editData,
  };
}
