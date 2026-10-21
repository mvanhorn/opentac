// Surface selection and true-scale unfolding for STL triangle meshes.
//
// An STL is an unindexed triangle soup. We first weld coincident vertices, merge
// connected coplanar triangles into user-facing "faces", and build adjacency from
// shared mesh edges. Selected faces are unfolded through a spanning tree: each
// child is rigidly placed on the opposite side of its shared edge from its parent.
// Lengths and angles inside every planar face are preserved exactly.
//
// Curved patches (grown from a clicked triangle, or enclosed by a loop drawn on
// the surface and optionally smoothed) unfold triangle by triangle, with a seam
// cut to open cones and tubes; that unfold is approximate and reports overlaps
// as warnings. Disconnected selections and overlapping planar nets are rejected
// instead of silently producing an invalid PCB outline.
import * as THREE from "three";

const DEFAULT_ANGLE_TOL_DEG = 1;

export function buildSurfaceModel(geometry, angleTolDeg = DEFAULT_ANGLE_TOL_DEG) {
  const pos = geometry.attributes.position;
  if (!pos || pos.count < 3 || pos.count % 3 !== 0) {
    throw new Error("The STL does not contain a valid triangle mesh.");
  }

  geometry.computeBoundingBox();
  const diag = geometry.boundingBox.min.distanceTo(geometry.boundingBox.max) || 1;
  const weldEps = Math.max(1e-5, diag * 1e-6);
  const triCount = pos.count / 3;

  const quantize = (x) => Math.round(x / weldEps);
  const vertexIds = new Map();
  const vertices = [];
  const weld = (x, y, z) => {
    const key = `${quantize(x)},${quantize(y)},${quantize(z)}`;
    let id = vertexIds.get(key);
    if (id === undefined) {
      id = vertices.length;
      vertexIds.set(key, id);
      vertices.push(new THREE.Vector3(x, y, z));
    }
    return id;
  };

  const triVerts = new Array(triCount);
  const triNormals = new Array(triCount);
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  for (let t = 0; t < triCount; t++) {
    const i = t * 3;
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    triVerts[t] = [
      weld(a.x, a.y, a.z),
      weld(b.x, b.y, b.z),
      weld(c.x, c.y, c.z),
    ];
    const normal = new THREE.Vector3()
      .subVectors(b, a)
      .cross(new THREE.Vector3().subVectors(c, a));
    if (normal.lengthSq() > 1e-20) normal.normalize();
    triNormals[t] = normal;
  }

  const triCentroids = triVerts.map((ids) => ids
    .reduce((sum, vertexId) => sum.add(vertices[vertexId]), new THREE.Vector3())
    .multiplyScalar(1 / 3));
  const triAreas = triVerts.map((ids) => triangleArea3(
    vertices[ids[0]], vertices[ids[1]], vertices[ids[2]]));

  const edgeMap = new Map();
  for (let t = 0; t < triCount; t++) {
    for (const [v1, v2] of triangleEdges(triVerts[t])) {
      const key = edgeKey(v1, v2);
      let touching = edgeMap.get(key);
      if (!touching) {
        touching = [];
        edgeMap.set(key, touching);
      }
      touching.push(t);
    }
  }

  // Flood-fill connected triangles whose normals differ by at most the tolerance.
  // abs(dot) tolerates STL files with inconsistent triangle winding on one plane.
  const cosTol = Math.cos((angleTolDeg * Math.PI) / 180);
  const triRegion = new Int32Array(triCount);
  triRegion.fill(-1);
  const regionTriangles = [];

  for (let seed = 0; seed < triCount; seed++) {
    if (triRegion[seed] !== -1 || triNormals[seed].lengthSq() === 0) continue;
    const regionId = regionTriangles.length;
    const triangles = [];
    const stack = [seed];
    triRegion[seed] = regionId;

    while (stack.length) {
      const t = stack.pop();
      triangles.push(t);
      for (const [v1, v2] of triangleEdges(triVerts[t])) {
        for (const neighbor of edgeMap.get(edgeKey(v1, v2)) || []) {
          if (triRegion[neighbor] !== -1 || triNormals[neighbor].lengthSq() === 0) continue;
          if (Math.abs(triNormals[neighbor].dot(triNormals[seed])) >= cosTol) {
            triRegion[neighbor] = regionId;
            stack.push(neighbor);
          }
        }
      }
    }
    regionTriangles.push(triangles);
  }

  // Preserve degenerate triangles as isolated, non-selectable entries in mapping.
  // A region whose boundary cannot be chained into closed loops (duplicate
  // faces, T-junctions — common in meshes converted from DAE/OBJ or scene
  // dumps) must not abort the whole model: it becomes an inert stub the
  // selection code refuses with a pointer to Curved-surface mode, which works
  // from triangle adjacency alone and does not need the region's boundary.
  // Otherwise one bad region out of thousands would fail the entire upload.
  const regions = regionTriangles.map((triangles, id) => {
    try {
      return buildRegion(id, triangles, triVerts, triNormals, vertices, pos);
    } catch (err) {
      return {
        id,
        triangles,
        degenerate: true,
        error: String(err?.message || err),
        area: 0,
        normal: triNormals[triangles[0]].clone(),
      };
    }
  });

  // Region adjacency. Every shared edge between two planar regions is a possible
  // unfolding hinge. Non-manifold edges are supported pairwise but flagged.
  const adjacencies = [];
  const adjacencySeen = new Set();
  let nonManifoldEdges = 0;
  for (const [key, touching] of edgeMap) {
    if (touching.length > 2) nonManifoldEdges++;
    for (let i = 0; i < touching.length; i++) {
      for (let j = i + 1; j < touching.length; j++) {
        const ra = triRegion[touching[i]];
        const rb = triRegion[touching[j]];
        if (ra < 0 || rb < 0 || ra === rb) continue;
        const [va, vb] = key.split("_").map(Number);
        const pairKey = `${Math.min(ra, rb)}:${Math.max(ra, rb)}:${key}`;
        if (adjacencySeen.has(pairKey)) continue;
        adjacencySeen.add(pairKey);
        adjacencies.push({ a: ra, b: rb, va, vb });
      }
    }
  }

  const adjacencyByRegion = regions.map(() => []);
  for (const adjacency of adjacencies) {
    adjacencyByRegion[adjacency.a].push(adjacency);
    adjacencyByRegion[adjacency.b].push(adjacency);
  }

  const triAdjacency = Array.from({ length: triCount }, () => []);
  for (const [key, touching] of edgeMap) {
    const [va, vb] = key.split("_").map(Number);
    for (let i = 0; i < touching.length; i++) {
      for (let j = 0; j < touching.length; j++) {
        if (i === j) continue;
        triAdjacency[touching[i]].push({
          triangle: touching[j],
          va,
          vb,
          nonManifold: touching.length > 2,
        });
      }
    }
  }

  return {
    geometry,
    positions: pos,
    vertices,
    triVerts,
    triNormals,
    triCentroids,
    triAreas,
    triAdjacency,
    triRegion,
    regions,
    adjacencies,
    adjacencyByRegion,
    weldEps,
    nonManifoldEdges,
  };
}

export function selectSmoothTrianglePatch(model, seedTriangleId, options = {}) {
  const seed = Number(seedTriangleId);
  if (!Number.isInteger(seed) || seed < 0 || seed >= model.triVerts.length) {
    throw new Error("Click a valid triangle to select a curved surface.");
  }
  if (model.triNormals[seed].lengthSq() === 0) {
    throw new Error("That triangle is degenerate and cannot seed a curved surface.");
  }

  const toleranceDeg = Number.isFinite(options.curvatureToleranceDeg)
    ? options.curvatureToleranceDeg
    : 18;
  const maxRadiusMm = Number.isFinite(options.maxRadiusMm) && options.maxRadiusMm > 0
    ? options.maxRadiusMm
    : Infinity;
  const cosTol = Math.cos((Math.max(0, toleranceDeg) * Math.PI) / 180);
  const selected = new Set([seed]);
  const distanceFromSeed = new Map([[seed, 0]]);
  const queue = [seed];
  let nonManifoldEdges = 0;

  while (queue.length) {
    const current = queue.shift();
    for (const edge of model.triAdjacency[current] || []) {
      const neighbor = edge.triangle;
      if (selected.has(neighbor) || model.triNormals[neighbor].lengthSq() === 0) continue;
      const normalDot = Math.max(-1, Math.min(1, model.triNormals[current].dot(model.triNormals[neighbor])));
      if (Math.abs(normalDot) < cosTol) continue;
      const step = model.triCentroids[current].distanceTo(model.triCentroids[neighbor]);
      const nextDistance = distanceFromSeed.get(current) + step;
      if (nextDistance > maxRadiusMm) continue;
      if (edge.nonManifold) nonManifoldEdges++;
      selected.add(neighbor);
      distanceFromSeed.set(neighbor, nextDistance);
      queue.push(neighbor);
    }
  }

  const area = [...selected].reduce((sum, triangle) => sum + model.triAreas[triangle], 0);
  return {
    triangles: selected,
    rootTriangleId: seed,
    area,
    nonManifoldEdges,
    toleranceDeg,
  };
}

export function unfoldSelectedTriangles(model, selectedTriangleIds, rootTriangleId = null, options = {}) {
  const selected = new Set([...selectedTriangleIds].map(Number));
  if (!selected.size) throw new Error("Select a curved surface before unfolding.");
  for (const id of selected) {
    if (!Number.isInteger(id) || id < 0 || id >= model.triVerts.length) {
      throw new Error("The curved selection contains an unknown triangle.");
    }
  }

  const root = selected.has(rootTriangleId) ? rootTriangleId : selected.values().next().value;
  const seam = resolveTrianglePatchSeam(model, selected, root, options);
  const local = new Map();
  for (const id of selected) local.set(id, triangleLocalFrame(model, id));

  const transforms = new Map();
  transforms.set(root, (p) => [p[0], p[1]]);
  const queue = [root];
  const usedHinges = [];
  let nonManifoldEdges = 0;
  let curvatureSum = 0;

  while (queue.length) {
    const parentId = queue.shift();
    const parentLocal = local.get(parentId);
    const parentTransform = transforms.get(parentId);

    for (const adjacency of model.triAdjacency[parentId] || []) {
      const childId = adjacency.triangle;
      if (!selected.has(childId)) continue;
      if (seam.edges.has(edgeKey(adjacency.va, adjacency.vb))) continue;
      if (adjacency.nonManifold) nonManifoldEdges++;
      curvatureSum += Math.acos(Math.max(-1, Math.min(1,
        Math.abs(model.triNormals[parentId].dot(model.triNormals[childId])))));
      if (transforms.has(childId)) continue;

      const childLocal = local.get(childId);
      const parentA = parentLocal.localByVertex.get(adjacency.va);
      const parentB = parentLocal.localByVertex.get(adjacency.vb);
      const childA = childLocal.localByVertex.get(adjacency.va);
      const childB = childLocal.localByVertex.get(adjacency.vb);
      if (!parentA || !parentB || !childA || !childB) continue;

      const placedA = parentTransform(parentA);
      const placedB = parentTransform(parentB);
      const parentCentroid = parentTransform(parentLocal.centroid);
      const childTransform = makeHingeTransform(
        childA, childB, placedA, placedB, childLocal.centroid, parentCentroid);

      transforms.set(childId, childTransform);
      queue.push(childId);
      usedHinges.push({
        parent: parentId,
        child: childId,
        va: adjacency.va,
        vb: adjacency.vb,
        a: placedA,
        b: placedB,
      });
    }
  }

  if (transforms.size !== selected.size) {
    const missing = selected.size - transforms.size;
    throw new Error(`${missing} selected triangle${missing === 1 ? " is" : "s are"} disconnected.`);
  }

  const placedFaces = [];
  for (const id of selected) {
    const triLocal = local.get(id);
    const transform = transforms.get(id);
    const ids = triLocal.ids.slice();
    let polygon = ids.map((vertexId) => transform(triLocal.localByVertex.get(vertexId)));
    if (polyArea(polygon) < 0) {
      polygon = polygon.slice().reverse();
      ids.reverse();
    }
    placedFaces.push({ id, ids, polygon, transform });
  }

  const warnings = [];
  const overlapPairs = findInteriorOverlaps(placedFaces, model.weldEps * 10);
  if (overlapPairs.length) {
    warnings.push("The approximate curved unfold overlaps; inspect the generated outline carefully.");
  }
  if (nonManifoldEdges) warnings.push("The selection touches non-manifold mesh edges.");
  if (curvatureSum > Math.PI * 2) {
    warnings.push("High accumulated curvature may cause flattening distortion.");
  }

  let outline;
  if (!seam.edges.size) {
    // Simple, robust outline for the common case (no seam cut): the
    // selection's boundary loops in vertex space always close — every
    // boundary edge belongs to exactly one selected triangle — so take
    // the largest loop, map each vertex to its unfolded 2D position, and
    // disregard every smaller loop (screw holes, vents). Re-deriving the
    // boundary from the unfolded 2D edges is fragile on a holed patch
    // (hole-rim seams, hinge drift exceeding the snap tolerance, rims
    // fragmented where tree branches meet); the vertex loop avoids all
    // three. Where the unfold's tree branches disagree about a rim
    // vertex's position, the average of its placed copies is used.
    const { loops } = selectedBoundaryLoops(model, selected);
    const pos = new Map();
    for (const face of placedFaces) {
      face.ids.forEach((vertexId, k) => {
        const p = face.polygon[k];
        const s = pos.get(vertexId) || { x: 0, y: 0, n: 0 };
        s.x += p[0]; s.y += p[1]; s.n++;
        pos.set(vertexId, s);
      });
    }
    let bestLoop = [];
    let bestPer = -1;
    for (const loop of loops) {
      const per = loopPerimeter(model, loop);
      if (per > bestPer) { bestPer = per; bestLoop = loop; }
    }
    if (loops.length > 1) {
      warnings.push(`${loops.length - 1} interior hole loop${loops.length === 2 ? "" : "s"} ignored.`);
    }
    outline = bestLoop
      .map((vertexId) => pos.get(vertexId))
      .filter(Boolean)
      .map((s) => [s.x / s.n, s.y / s.n]);
  } else {
    // A seam cut must appear in the outline (opened cones/tubes), which
    // only the placed-edge chaining can produce.
    outline = buildTrianglePatchBoundary(placedFaces, model.weldEps * 20, seam.edges);
  }
  if (outline.length < 3) throw new Error("Could not construct one closed boundary for the curved selection.");

  const xs = outline.map((p) => p[0]);
  const ys = outline.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  let centeredOutline = outline.map(([x, y]) => [x - cx, y - cy]);
  // No fold guides for a curved unfold. It hinges at every triangle, so
  // mapping the used hinges would paint the whole triangulation as
  // hundreds of "fold lines", none of them a real crease: the smooth
  // flood-fill that builds a curved selection stops at anything sharper
  // than its curvature tolerance, so every interior hinge is a micro-bend
  // of a surface that is bent continuously, not folded. (The planar
  // multi-face unfold keeps its fold lines: those are real folds between
  // flat faces.)
  let foldLines = [];

  if (polyArea(centeredOutline) < 0) centeredOutline.reverse();
  centeredOutline = simplifyCollinear(centeredOutline);
  centeredOutline = rdpSimplifyClosed(centeredOutline, 0.15);
  centeredOutline = rotateToBottomEdge(centeredOutline);

  const centeredXs = centeredOutline.map((p) => p[0]);
  const centeredYs = centeredOutline.map((p) => p[1]);
  const w = Math.max(...centeredXs) - Math.min(...centeredXs);
  const h = Math.max(...centeredYs) - Math.min(...centeredYs);
  foldLines = foldLines.filter(([a, b]) => distance2(a, b) > 1e-16);

  return {
    outline: centeredOutline,
    foldLines,
    w,
    h,
    faceCount: selected.size,
    rootTriangleId: root,
    usedHinges,
    placedFaces,
    warnings,
    seam,
  };
}

function buildRegion(id, triangles, triVerts, triNormals, vertices, positions) {
  const edgeCounts = new Map();
  for (const t of triangles) {
    for (const [a, b] of triangleEdges(triVerts[t])) {
      const key = edgeKey(a, b);
      const item = edgeCounts.get(key) || { count: 0, a, b };
      item.count++;
      edgeCounts.set(key, item);
    }
  }
  const boundaryEdges = [];
  for (const item of edgeCounts.values()) {
    if (item.count === 1) boundaryEdges.push([item.a, item.b]);
  }
  const loops = chainVertexLoops(boundaryEdges);
  if (!loops.length) throw new Error(`Planar face ${id + 1} has no closed boundary.`);

  const normal = triNormals[triangles[0]].clone();
  const u = new THREE.Vector3();
  const v = new THREE.Vector3();
  basisFromNormal(normal, u, v);
  const origin = vertices[loops[0][0]].clone();
  const project = (vertexId) => {
    const rel = vertices[vertexId].clone().sub(origin);
    return [rel.dot(u), rel.dot(v)];
  };

  let outerLoop = null;
  let outerArea = -1;
  for (const loop of loops) {
    const area = Math.abs(polyArea(loop.map(project)));
    if (area > outerArea) {
      outerArea = area;
      outerLoop = loop.slice();
    }
  }
  if (!outerLoop || outerLoop.length < 3) {
    throw new Error(`Planar face ${id + 1} has an invalid outer boundary.`);
  }
  if (polyArea(outerLoop.map(project)) < 0) outerLoop.reverse();

  const localByVertex = new Map();
  for (const vertexId of new Set(loops.flat())) localByVertex.set(vertexId, project(vertexId));
  const polygon = outerLoop.map((vertexId) => localByVertex.get(vertexId));
  const centroid = polygonCentroid(polygon);

  const highlightPositions = [];
  for (const t of triangles) {
    const base = t * 3;
    for (const j of [base, base + 1, base + 2]) {
      highlightPositions.push(positions.getX(j), positions.getY(j), positions.getZ(j));
    }
  }

  return {
    id,
    triangles,
    normal,
    loops,
    outerLoop,
    localByVertex,
    polygon,
    centroid,
    area: outerArea,
    highlightPositions,
  };
}

export function unfoldSelectedRegions(model, selectedRegionIds, rootRegionId = null) {
  const selected = new Set([...selectedRegionIds].map(Number));
  if (!selected.size) throw new Error("Select at least one face before unfolding.");
  for (const id of selected) {
    if (!Number.isInteger(id) || id < 0 || id >= model.regions.length) {
      throw new Error("The selection contains an unknown face.");
    }
    if (model.regions[id].degenerate) {
      throw new Error(
        `Face ${id + 1}'s boundary couldn't be analyzed — use Curved surface mode for that area.`);
    }
  }

  const root = selected.has(rootRegionId) ? rootRegionId : selected.values().next().value;
  const transforms = new Map();
  transforms.set(root, (p) => [p[0], p[1]]);
  const queue = [root];
  const usedHinges = [];

  while (queue.length) {
    const parentId = queue.shift();
    const parent = model.regions[parentId];
    const parentTransform = transforms.get(parentId);

    for (const adjacency of model.adjacencyByRegion[parentId]) {
      const childId = adjacency.a === parentId ? adjacency.b : adjacency.a;
      if (!selected.has(childId) || transforms.has(childId)) continue;
      const child = model.regions[childId];
      const parentA = parent.localByVertex.get(adjacency.va);
      const parentB = parent.localByVertex.get(adjacency.vb);
      const childA = child.localByVertex.get(adjacency.va);
      const childB = child.localByVertex.get(adjacency.vb);
      if (!parentA || !parentB || !childA || !childB) continue;

      const placedA = parentTransform(parentA);
      const placedB = parentTransform(parentB);
      const parentCentroid = parentTransform(parent.centroid);
      const childTransform = makeHingeTransform(
        childA, childB, placedA, placedB, child.centroid, parentCentroid);

      transforms.set(childId, childTransform);
      queue.push(childId);
      usedHinges.push({
        parent: parentId,
        child: childId,
        va: adjacency.va,
        vb: adjacency.vb,
        a: placedA,
        b: placedB,
      });
    }
  }

  if (transforms.size !== selected.size) {
    const missing = selected.size - transforms.size;
    throw new Error(
      `${missing} selected face${missing === 1 ? " is" : "s are"} disconnected. ` +
      "Select faces that share complete mesh edges.");
  }

  const placedFaces = [];
  for (const id of selected) {
    const region = model.regions[id];
    const transform = transforms.get(id);
    const ids = region.outerLoop.slice();
    let polygon = ids.map((vertexId) => transform(region.localByVertex.get(vertexId)));
    if (polyArea(polygon) < 0) {
      polygon = polygon.slice().reverse();
      ids.reverse();
    }
    placedFaces.push({ id, ids, polygon, transform });
  }

  const overlapPairs = findInteriorOverlaps(placedFaces, model.weldEps * 10);
  if (overlapPairs.length) {
    const labels = overlapPairs.slice(0, 3)
      .map(([a, b]) => `${a + 1}/${b + 1}`).join(", ");
    throw new Error(
      `This automatic net overlaps between face pairs ${labels}. ` +
      "Try selecting fewer faces or choose another first (root) face.");
  }

  const hingeEdgeKeys = new Set(usedHinges.map((h) => edgeKey(h.va, h.vb)));
  const outline = buildNetBoundary(placedFaces, hingeEdgeKeys, model.weldEps * 20);
  if (outline.length < 3) throw new Error("Could not construct one closed boundary for the unfolded net.");

  // Put the net near the origin and use its lowest boundary edge as the cable edge.
  const xs = outline.map((p) => p[0]);
  const ys = outline.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  let centeredOutline = outline.map(([x, y]) => [x - cx, y - cy]);
  let foldLines = usedHinges.map((h) => [
    [h.a[0] - cx, h.a[1] - cy],
    [h.b[0] - cx, h.b[1] - cy],
  ]);

  if (polyArea(centeredOutline) < 0) centeredOutline.reverse();
  centeredOutline = simplifyCollinear(centeredOutline);
  centeredOutline = rotateToBottomEdge(centeredOutline);

  const centeredXs = centeredOutline.map((p) => p[0]);
  const centeredYs = centeredOutline.map((p) => p[1]);
  const w = Math.max(...centeredXs) - Math.min(...centeredXs);
  const h = Math.max(...centeredYs) - Math.min(...centeredYs);

  // Remove numerical duplicate fold lines without losing different hinges.
  foldLines = foldLines.filter(([a, b]) => distance2(a, b) > 1e-16);
  return {
    outline: centeredOutline,
    foldLines,
    w,
    h,
    faceCount: selected.size,
    rootRegionId: root,
    usedHinges,
    placedFaces,
  };
}

function makeHingeTransform(qA, qB, pA, pB, childCentroid, parentCentroid) {
  const qdx = qB[0] - qA[0];
  const qdy = qB[1] - qA[1];
  const pdx = pB[0] - pA[0];
  const pdy = pB[1] - pA[1];
  const qLen = Math.hypot(qdx, qdy);
  const pLen = Math.hypot(pdx, pdy);
  if (qLen < 1e-9 || pLen < 1e-9) throw new Error("A selected face has a zero-length shared edge.");

  const qx = qdx / qLen;
  const qy = qdy / qLen;
  const px = pdx / pLen;
  const py = pdy / pLen;

  const make = (side) => (point) => {
    const rx = point[0] - qA[0];
    const ry = point[1] - qA[1];
    const along = rx * qx + ry * qy;
    const across = qx * ry - qy * rx;
    return [
      pA[0] + along * px + side * across * -py,
      pA[1] + along * py + side * across * px,
    ];
  };

  const parentSide = cross2([pdx, pdy], [
    parentCentroid[0] - pA[0], parentCentroid[1] - pA[1],
  ]);
  const positive = make(1);
  const negative = make(-1);
  const posCentroid = positive(childCentroid);
  const posSide = cross2([pdx, pdy], [posCentroid[0] - pA[0], posCentroid[1] - pA[1]]);
  return parentSide * posSide < 0 ? positive : negative;
}

function buildNetBoundary(placedFaces, hingeEdgeKeys, tolerance) {
  const scale = 1 / Math.max(tolerance, 1e-8);
  const pointKey = (p) => `${Math.round(p[0] * scale)},${Math.round(p[1] * scale)}`;
  const edges = [];

  for (const face of placedFaces) {
    const n = face.ids.length;
    for (let i = 0; i < n; i++) {
      const va = face.ids[i];
      const vb = face.ids[(i + 1) % n];
      if (hingeEdgeKeys.has(edgeKey(va, vb))) continue;
      const a = face.polygon[i];
      const b = face.polygon[(i + 1) % n];
      edges.push({ a, b, start: pointKey(a), end: pointKey(b), used: false });
    }
  }

  const outgoing = new Map();
  for (let i = 0; i < edges.length; i++) {
    if (!outgoing.has(edges[i].start)) outgoing.set(edges[i].start, []);
    outgoing.get(edges[i].start).push(i);
  }

  const loops = [];
  for (let startIndex = 0; startIndex < edges.length; startIndex++) {
    if (edges[startIndex].used) continue;
    const first = edges[startIndex];
    const loop = [first.a];
    first.used = true;
    let current = first;
    let guard = 0;

    while (current.end !== first.start && guard++ <= edges.length + 2) {
      const candidates = (outgoing.get(current.end) || []).filter((idx) => !edges[idx].used);
      if (!candidates.length) break;
      const nextIndex = chooseBoundaryContinuation(current, candidates.map((idx) => edges[idx]));
      const next = edges[candidates[nextIndex]];
      next.used = true;
      loop.push(next.a);
      current = next;
    }
    if (current.end === first.start && loop.length >= 3) loops.push(loop);
  }

  if (!loops.length) return [];
  let outer = loops[0];
  for (const loop of loops.slice(1)) {
    if (Math.abs(polyArea(loop)) > Math.abs(polyArea(outer))) outer = loop;
  }
  return removeDuplicateNeighbors(outer, tolerance);
}

function loopPerimeter(model, loop) {
  let p = 0;
  for (let i = 0; i < loop.length; i++) {
    p += model.vertices[loop[i]].distanceTo(model.vertices[loop[(i + 1) % loop.length]]);
  }
  return p;
}

// A boundary loop much smaller than the selection's largest loop is a hole
// (screw hole, vent), not a second rim that needs cutting open. A true
// tube's two rims are comparable in size; a mounting hole is a fraction.
const HOLE_LOOP_RATIO = 0.35;

function resolveTrianglePatchSeam(model, selected, rootTriangleId, options = {}) {
  const manual = Array.isArray(options.seamTriangleIds)
    ? options.seamTriangleIds.map(Number).filter((id) => selected.has(id))
    : [];
  const autoSeam = options.autoSeam !== false;
  const { loops, boundaryEdges } = selectedBoundaryLoops(model, selected);

  if (manual.length >= 2) {
    const path = shortestTrianglePath(model, selected, manual[0], manual[1]);
    if (!path.length) throw new Error("Could not build a seam between the selected seam points.");
    return {
      edges: seamEdgesFromTrianglePath(model, path),
      message: "manual seam applied",
      boundaryLoopCount: loops.length,
    };
  }

  if (!loops.length) {
    throw new Error("Closed surface selected. Select only the side surface or add a seam/cut first.");
  }

  // Holes are ignored for seam purposes: a disk-with-holes unfolds as a
  // plain tree, and treating a screw hole as a tube rim would cut an
  // outer->hole seam that breaks the unfold apart. The final outline
  // chaining already keeps only the largest loop, so ignored holes simply
  // close over.
  const byPerimeter = loops
    .map((loop) => ({ loop, perimeter: loopPerimeter(model, loop) }))
    .sort((a, b) => b.perimeter - a.perimeter);
  const significant = byPerimeter
    .filter((item) => item.perimeter >= HOLE_LOOP_RATIO * byPerimeter[0].perimeter)
    .map((item) => item.loop);
  const holeCount = loops.length - significant.length;
  const holeNote = holeCount
    ? ` (${holeCount} small hole loop${holeCount === 1 ? "" : "s"} ignored)`
    : "";

  if (significant.length === 1) {
    if (!autoSeam) {
      return { edges: new Set(), message: holeNote.trim(), boundaryLoopCount: loops.length };
    }
    const seamEdges = autoConeSeamEdges(model, selected, significant[0], rootTriangleId, loops);
    if (!seamEdges.size) {
      return { edges: new Set(), message: holeNote.trim(), boundaryLoopCount: loops.length };
    }
    return {
      edges: seamEdges,
      message: "auto seam applied" + holeNote,
      boundaryLoopCount: loops.length,
    };
  }

  if (!autoSeam) {
    throw new Error("This curved surface needs a seam. Turn on Auto seam or use Set seam.");
  }

  const seamEdges = autoLoopToLoopSeamEdges(model, selected, significant, rootTriangleId);
  if (!seamEdges.size) {
    throw new Error("Could not build an auto seam between the curved surface boundary loops.");
  }
  return {
    edges: seamEdges,
    message: "auto seam applied" + holeNote,
    boundaryLoopCount: loops.length,
    boundaryEdgeCount: boundaryEdges.length,
  };
}

function selectedBoundaryLoops(model, selected) {
  const edgeCounts = new Map();
  for (const triangleId of selected) {
    for (const [a, b] of triangleEdges(model.triVerts[triangleId])) {
      const key = edgeKey(a, b);
      const item = edgeCounts.get(key) || { count: 0, a, b };
      item.count++;
      edgeCounts.set(key, item);
    }
  }
  const boundaryEdges = [];
  for (const item of edgeCounts.values()) {
    if (item.count === 1) boundaryEdges.push([item.a, item.b]);
  }
  return { loops: chainVertexLoops(boundaryEdges), boundaryEdges };
}

function autoLoopToLoopSeamEdges(model, selected, loops, rootTriangleId) {
  const rootPoint = model.triCentroids[rootTriangleId] || selectionCentroid(model, selected);
  const start = nearestVertexToPoint(model, loops[0], rootPoint);
  const end = nearestVertexToPoint(model, loops[1], model.vertices[start]);
  return seamEdgesFromVertexPath(shortestVertexPath(model, selected, start, end));
}

function autoConeSeamEdges(model, selected, boundaryLoop, rootTriangleId, allLoops = null) {
  // Exclude the vertices of every boundary loop from apex candidacy, not
  // just the outer loop's: ignored hole rims are the densest vertices in
  // the selection (fine hole tessellation), and the degree-scored apex
  // would otherwise lock onto a screw-hole rim instead of the surface's
  // interior.
  const boundary = new Set((allLoops ?? [boundaryLoop]).flat());
  const vertexUse = new Map();
  for (const triangleId of selected) {
    for (const vertexId of model.triVerts[triangleId]) {
      if (boundary.has(vertexId)) continue;
      vertexUse.set(vertexId, (vertexUse.get(vertexId) || 0) + 1);
    }
  }
  if (!vertexUse.size) return new Set();

  const boundaryCentroid = boundaryLoop
    .reduce((sum, vertexId) => sum.add(model.vertices[vertexId]), new THREE.Vector3())
    .multiplyScalar(1 / boundaryLoop.length);
  let apex = null;
  let apexUseCount = 0;
  let bestScore = -Infinity;
  for (const [vertexId, useCount] of vertexUse) {
    const score = useCount * 1000 + model.vertices[vertexId].distanceTo(boundaryCentroid);
    if (score > bestScore) {
      bestScore = score;
      apex = vertexId;
      apexUseCount = useCount;
    }
  }
  if (apex === null || apexUseCount < 4) return new Set();

  // Cut only when the apex concentrates real curvature. The angular defect
  // (2π minus the sum of incident triangle angles) is what the cut opens
  // up: a true cone/dome apex carries a large defect and the cut gapes; a
  // gently curved pad has ~zero defect, and cutting it produces a seam
  // that "opens" by microns — two nearly coincident edge copies that
  // derail the 2D boundary walker into the "could not construct one
  // closed boundary" failure.
  let apexAngleSum = 0;
  for (const triangleId of trianglesTouchingVertices(model, selected, [apex])) {
    const ids = model.triVerts[triangleId];
    const k = ids.indexOf(apex);
    if (k === -1) continue;
    const p = model.vertices[apex];
    const e1 = model.vertices[ids[(k + 1) % 3]].clone().sub(p);
    const e2 = model.vertices[ids[(k + 2) % 3]].clone().sub(p);
    const denom = e1.length() * e2.length();
    if (denom > 1e-20) {
      apexAngleSum += Math.acos(Math.max(-1, Math.min(1, e1.dot(e2) / denom)));
    }
  }
  if (Math.abs(2 * Math.PI - apexAngleSum) < 0.15) return new Set();

  const apexTriangles = trianglesTouchingVertices(model, selected, [apex]);
  const start = nearestTriangleToPoint(model, apexTriangles, model.triCentroids[rootTriangleId] || model.vertices[apex]);
  const apexVertex = nearestVertexToPoint(model, model.triVerts[start] || [apex], model.vertices[apex]);
  const endVertex = nearestVertexToPoint(model, boundaryLoop, model.vertices[apexVertex]);
  return seamEdgesFromVertexPath(shortestVertexPath(model, selected, apexVertex, endVertex));
}

function trianglesTouchingVertices(model, selected, vertexIds) {
  const vertices = new Set(vertexIds);
  const triangles = new Set();
  for (const triangleId of selected) {
    if (model.triVerts[triangleId].some((vertexId) => vertices.has(vertexId))) {
      triangles.add(triangleId);
    }
  }
  return triangles;
}

function nearestTriangleToPoint(model, triangleIds, point) {
  let best = null;
  let bestDist = Infinity;
  for (const triangleId of triangleIds) {
    const dist = model.triCentroids[triangleId].distanceToSquared(point);
    if (dist < bestDist) {
      bestDist = dist;
      best = triangleId;
    }
  }
  return best;
}

function nearestVertexToPoint(model, vertexIds, point) {
  let best = null;
  let bestDist = Infinity;
  for (const vertexId of vertexIds) {
    const dist = model.vertices[vertexId].distanceToSquared(point);
    if (dist < bestDist) {
      bestDist = dist;
      best = vertexId;
    }
  }
  return best;
}

function selectionCentroid(model, selected) {
  const sum = new THREE.Vector3();
  for (const triangleId of selected) sum.add(model.triCentroids[triangleId]);
  return sum.multiplyScalar(1 / selected.size);
}

function shortestVertexPath(model, selected, startVertex, endVertex) {
  if (startVertex === null || endVertex === null || startVertex === undefined || endVertex === undefined) return [];
  const adjacency = selectedVertexAdjacency(model, selected);
  // Dijkstra by 3D edge length, not hop count. Hop-count BFS prefers
  // few-hop routes through large triangles: on mixed-density meshes the
  // path zigzags via long chords down a coarse side wall and back, and
  // cutting that zigzag breaks the unfold tree into pieces. A
  // length-weighted path hugs the surface like a geodesic and cuts cleanly.
  const dist = new Map([[startVertex, 0]]);
  const previous = new Map([[startVertex, null]]);
  const done = new Set();
  while (true) {
    let vertexId = null;
    let best = Infinity;
    for (const [v, d] of dist) {
      if (!done.has(v) && d < best) { best = d; vertexId = v; }
    }
    if (vertexId === null) break;
    if (vertexId === endVertex) break;
    done.add(vertexId);
    for (const next of adjacency.get(vertexId) || []) {
      if (done.has(next)) continue;
      const d = best + model.vertices[vertexId].distanceTo(model.vertices[next]);
      if (d < (dist.get(next) ?? Infinity)) {
        dist.set(next, d);
        previous.set(next, vertexId);
      }
    }
  }
  if (!previous.has(endVertex)) return [];
  const path = [];
  for (let at = endVertex; at !== null; at = previous.get(at)) path.push(at);
  return path.reverse();
}

function selectedVertexAdjacency(model, selected) {
  const adjacency = new Map();
  const add = (a, b) => {
    if (!adjacency.has(a)) adjacency.set(a, new Set());
    if (!adjacency.has(b)) adjacency.set(b, new Set());
    adjacency.get(a).add(b);
    adjacency.get(b).add(a);
  };
  for (const triangleId of selected) {
    for (const [a, b] of triangleEdges(model.triVerts[triangleId])) add(a, b);
  }
  return adjacency;
}

function shortestTrianglePath(model, selected, startTriangle, endTriangle) {
  if (startTriangle === null || endTriangle === null || startTriangle === undefined || endTriangle === undefined) return [];
  if (!selected.has(startTriangle) || !selected.has(endTriangle)) return [];
  const queue = [startTriangle];
  const previous = new Map([[startTriangle, null]]);
  while (queue.length) {
    const triangleId = queue.shift();
    if (triangleId === endTriangle) break;
    for (const adjacency of model.triAdjacency[triangleId] || []) {
      const next = adjacency.triangle;
      if (!selected.has(next) || previous.has(next)) continue;
      previous.set(next, triangleId);
      queue.push(next);
    }
  }
  if (!previous.has(endTriangle)) return [];
  const path = [];
  for (let at = endTriangle; at !== null; at = previous.get(at)) path.push(at);
  return path.reverse();
}

function seamEdgesFromTrianglePath(model, trianglePath) {
  const edges = new Set();
  for (let i = 1; i < trianglePath.length; i++) {
    const shared = sharedEdgeBetweenTriangles(model, trianglePath[i - 1], trianglePath[i]);
    if (shared) edges.add(edgeKey(shared[0], shared[1]));
  }
  return edges;
}

function seamEdgesFromVertexPath(vertexPath) {
  const edges = new Set();
  for (let i = 1; i < vertexPath.length; i++) {
    edges.add(edgeKey(vertexPath[i - 1], vertexPath[i]));
  }
  return edges;
}

function sharedEdgeBetweenTriangles(model, triangleA, triangleB) {
  const b = new Set(model.triVerts[triangleB]);
  const shared = model.triVerts[triangleA].filter((vertexId) => b.has(vertexId));
  return shared.length === 2 ? shared : null;
}

function buildTrianglePatchBoundary(placedFaces, tolerance, seamEdges = new Set()) {
  const meshEdges = new Map();
  for (const face of placedFaces) {
    const n = face.ids.length;
    for (let i = 0; i < n; i++) {
      const va = face.ids[i];
      const vb = face.ids[(i + 1) % n];
      const key = edgeKey(va, vb);
      const a = face.polygon[i];
      const b = face.polygon[(i + 1) % n];
      if (!meshEdges.has(key)) meshEdges.set(key, []);
      meshEdges.get(key).push({ a, b, used: false });
    }
  }

  // Chaining snap tolerance must scale with the mesh, not the weld
  // epsilon: each boundary edge's 2D endpoints come from the transform of
  // the one triangle that owns it, and the accumulated hinge float drift
  // across a few hundred placed triangles moves the same welded vertex by
  // more than weldEps*20 between neighboring rim triangles. With a
  // micron-scale tolerance the walk around a clean rim misses its own
  // next edge and every loop dead-ends. A fraction of the median boundary
  // edge length rides far above drift and far below real geometry.
  const lens = [];
  for (const items of meshEdges.values()) {
    if (items.length === 1) {
      const it = items[0];
      lens.push(Math.hypot(it.a[0] - it.b[0], it.a[1] - it.b[1]));
    }
  }
  lens.sort((x, y) => x - y);
  const median = lens.length ? lens[Math.floor(lens.length / 2)] : 0;
  tolerance = Math.max(tolerance, Math.min(0.2 * median, 0.5));
  const scale = 1 / Math.max(tolerance, 1e-8);
  const pointKey = (p) => `${Math.round(p[0] * scale)},${Math.round(p[1] * scale)}`;
  for (const items of meshEdges.values()) {
    for (const it of items) {
      it.start = pointKey(it.a);
      it.end = pointKey(it.b);
    }
  }

  const edges = [];
  for (const [key, items] of meshEdges) {
    if (items.length === 1) {
      edges.push(...items);
      continue;
    }
    if (!seamEdges.has(key)) continue;
    // A seam edge belongs on the boundary only where the cut actually
    // opened. On a gently curved patch the angular defect is ~zero, the
    // unfold reproduces identical positions on both sides of the cut, and
    // pooling two exactly-coincident opposite copies derails the boundary
    // walker into dead ends, so no closed loop is found. A closed cut is
    // simply not a boundary; skip it. Real cone cuts gape open and keep
    // both sides.
    const minOpen = Math.max(tolerance * 4, 0.05);
    const opened = items.length !== 2
      || Math.hypot(items[0].a[0] - items[1].b[0], items[0].a[1] - items[1].b[1]) > minOpen
      || Math.hypot(items[0].b[0] - items[1].a[0], items[0].b[1] - items[1].a[1]) > minOpen;
    if (opened) edges.push(...items);
  }
  return chainBoundaryEdges(edges, tolerance);
}

function chainBoundaryEdges(edges, tolerance) {
  const outgoing = new Map();
  for (let i = 0; i < edges.length; i++) {
    if (!outgoing.has(edges[i].start)) outgoing.set(edges[i].start, []);
    outgoing.get(edges[i].start).push(i);
  }

  const loops = [];
  const fragments = [];
  for (let startIndex = 0; startIndex < edges.length; startIndex++) {
    if (edges[startIndex].used) continue;
    const first = edges[startIndex];
    const loop = [first.a];
    first.used = true;
    let current = first;
    let guard = 0;

    while (current.end !== first.start && guard++ <= edges.length + 2) {
      const candidates = (outgoing.get(current.end) || []).filter((idx) => !edges[idx].used);
      if (!candidates.length) break;
      const nextIndex = chooseBoundaryContinuation(current, candidates.map((idx) => edges[idx]));
      const next = edges[candidates[nextIndex]];
      next.used = true;
      loop.push(next.a);
      current = next;
    }
    if (current.end === first.start && loop.length >= 3) {
      loops.push(loop);
    } else if (loop.length >= 2) {
      fragments.push([...loop, current.b]);
    }
  }

  // On strongly curved selections the approximate unfold's tree branches
  // meet along the rim misaligned — by millimeters, not float drift — so
  // the 2D boundary arrives as open fragments and the strict walk above
  // closes nothing. The unfold is still the approximation the overlap
  // warnings already flag, so make the boundary total: close a fragment
  // onto itself, or splice the nearest fragment end on, whenever the gap
  // is small relative to the geometry. Distant fragments (a hole rim vs
  // the outer rim) stay separate, and holes are discarded by the
  // largest-loop rule below.
  const gapLimit = Math.max(2.0, 20 * tolerance);
  const gapDist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const open = fragments.filter((f) => f.length >= 3);
  let stitchGuard = 0;
  while (open.length && stitchGuard++ < 1000) {
    open.sort((a, b) => b.length - a.length);
    const cur = open.shift();
    let best = null;
    for (let i = 0; i < open.length; i++) {
      const head = gapDist(cur[cur.length - 1], open[i][0]);
      const tail = gapDist(cur[cur.length - 1], open[i][open[i].length - 1]);
      const cand = Math.min(head, tail);
      if (cand <= gapLimit && (!best || cand < best.dist)) {
        best = { i, dist: cand, reversed: tail < head };
      }
    }
    if (best) {
      const piece = open.splice(best.i, 1)[0];
      if (best.reversed) piece.reverse();
      open.push(cur.concat(piece));
      continue;
    }
    if (cur.length >= 4 && gapDist(cur[0], cur[cur.length - 1]) <= gapLimit) {
      loops.push(cur.slice(0, -1));
    }
    // otherwise: an isolated sliver — drop it.
  }

  if (!loops.length) return [];
  let outer = loops[0];
  for (const loop of loops.slice(1)) {
    if (Math.abs(polyArea(loop)) > Math.abs(polyArea(outer))) outer = loop;
  }
  return removeDuplicateNeighbors(outer, tolerance);
}

function chooseBoundaryContinuation(incoming, candidates) {
  if (candidates.length === 1) return 0;
  const inAngle = Math.atan2(incoming.b[1] - incoming.a[1], incoming.b[0] - incoming.a[0]);
  let best = 0;
  let bestTurn = Infinity;
  for (let i = 0; i < candidates.length; i++) {
    const edge = candidates[i];
    const outAngle = Math.atan2(edge.b[1] - edge.a[1], edge.b[0] - edge.a[0]);
    let turn = outAngle - inAngle;
    while (turn <= 0) turn += Math.PI * 2;
    if (turn < bestTurn) {
      bestTurn = turn;
      best = i;
    }
  }
  return best;
}

function findInteriorOverlaps(placedFaces, eps) {
  const overlaps = [];
  for (let i = 0; i < placedFaces.length; i++) {
    for (let j = i + 1; j < placedFaces.length; j++) {
      if (polygonsOverlapInterior(placedFaces[i].polygon, placedFaces[j].polygon, eps)) {
        overlaps.push([placedFaces[i].id, placedFaces[j].id]);
      }
    }
  }
  return overlaps;
}

function polygonsOverlapInterior(a, b, eps) {
  for (let i = 0; i < a.length; i++) {
    const a1 = a[i];
    const a2 = a[(i + 1) % a.length];
    for (let j = 0; j < b.length; j++) {
      const b1 = b[j];
      const b2 = b[(j + 1) % b.length];
      if (segmentsProperlyIntersect(a1, a2, b1, b2, eps)) return true;
    }
  }
  if (a.some((p) => pointInPolygonStrict(p, b, eps))) return true;
  if (b.some((p) => pointInPolygonStrict(p, a, eps))) return true;
  return false;
}

function segmentsProperlyIntersect(a, b, c, d, eps) {
  const o1 = cross2([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
  const o2 = cross2([b[0] - a[0], b[1] - a[1]], [d[0] - a[0], d[1] - a[1]]);
  const o3 = cross2([d[0] - c[0], d[1] - c[1]], [a[0] - c[0], a[1] - c[1]]);
  const o4 = cross2([d[0] - c[0], d[1] - c[1]], [b[0] - c[0], b[1] - c[1]]);
  return ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
         ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps));
}

function pointInPolygonStrict(point, polygon, eps) {
  for (let i = 0; i < polygon.length; i++) {
    if (pointSegmentDistance(point, polygon[i], polygon[(i + 1) % polygon.length]) <= eps) return false;
  }
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > point[1]) !== (yj > point[1]) &&
        point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function pointSegmentDistance(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-20) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

function triangleArea3(a, b, c) {
  return new THREE.Vector3()
    .subVectors(b, a)
    .cross(new THREE.Vector3().subVectors(c, a))
    .length() / 2;
}

function triangleLocalFrame(model, triangleId) {
  const ids = model.triVerts[triangleId].slice();
  const p0 = model.vertices[ids[0]];
  const p1 = model.vertices[ids[1]];
  const p2 = model.vertices[ids[2]];
  const xAxis = p1.clone().sub(p0);
  const xLen = xAxis.length();
  if (xLen < 1e-12) throw new Error("A selected triangle has a zero-length edge.");
  xAxis.multiplyScalar(1 / xLen);
  const normal = model.triNormals[triangleId].clone();
  const yAxis = new THREE.Vector3().crossVectors(normal, xAxis).normalize();
  const project = (vertexId) => {
    const rel = model.vertices[vertexId].clone().sub(p0);
    return [rel.dot(xAxis), rel.dot(yAxis)];
  };
  const localByVertex = new Map(ids.map((vertexId) => [vertexId, project(vertexId)]));
  const polygon = ids.map((vertexId) => localByVertex.get(vertexId));
  if (polyArea(polygon) < 0) ids.reverse();
  return {
    ids,
    localByVertex,
    polygon: ids.map((vertexId) => localByVertex.get(vertexId)),
    centroid: polygonCentroid(polygon),
  };
}

// Post-process an outline computed elsewhere (the backend's strut-joined
// /unroll-mesh-chain result) the same way the client-side unfolds above
// prepare theirs: center on the origin, orient counter-clockwise, drop
// collinear points, and rotate the lowest boundary edge into the 0->1 cable
// slot. Keeps the backend handoff consistent with the client paths without
// duplicating their geometry helpers.
export function finalizeChainOutline(points) {
  if (!Array.isArray(points) || points.length < 3) {
    throw new Error("The joined outline has fewer than 3 points.");
  }
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  let outline = points.map(([x, y]) => [x - cx, y - cy]);
  if (polyArea(outline) < 0) outline.reverse();
  outline = simplifyCollinear(outline);
  outline = rdpSimplifyClosed(outline, 0.15);
  outline = rotateToBottomEdge(outline);
  const outXs = outline.map((p) => p[0]);
  const outYs = outline.map((p) => p[1]);
  return {
    outline,
    w: Math.max(...outXs) - Math.min(...outXs),
    h: Math.max(...outYs) - Math.min(...outYs),
  };
}

function triangleEdges(vertices) {
  return [[vertices[0], vertices[1]], [vertices[1], vertices[2]], [vertices[2], vertices[0]]];
}

function edgeKey(a, b) {
  return a < b ? `${a}_${b}` : `${b}_${a}`;
}

function chainVertexLoops(edges) {
  const adjacency = new Map();
  for (const [a, b] of edges) {
    if (!adjacency.has(a)) adjacency.set(a, []);
    if (!adjacency.has(b)) adjacency.set(b, []);
    adjacency.get(a).push(b);
    adjacency.get(b).push(a);
  }
  const used = new Set();
  const loops = [];
  for (const [start, neighbors] of adjacency) {
    for (const first of neighbors) {
      if (used.has(edgeKey(start, first))) continue;
      const loop = [start];
      let previous = start;
      let current = first;
      used.add(edgeKey(previous, current));
      let valid = true;
      while (current !== start) {
        loop.push(current);
        const next = (adjacency.get(current) || [])
          .find((candidate) => candidate !== previous && !used.has(edgeKey(current, candidate)));
        if (next === undefined) {
          valid = false;
          break;
        }
        used.add(edgeKey(current, next));
        previous = current;
        current = next;
        if (loop.length > edges.length + 2) {
          valid = false;
          break;
        }
      }
      if (valid && loop.length >= 3) loops.push(loop);
    }
  }
  return loops;
}

function basisFromNormal(normal, u, v) {
  const ref = Math.abs(normal.z) < 0.9
    ? new THREE.Vector3(0, 0, 1)
    : new THREE.Vector3(1, 0, 0);
  u.crossVectors(ref, normal).normalize();
  v.crossVectors(normal, u).normalize();
}

function polygonCentroid(points) {
  let area2 = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const cross = a[0] * b[1] - b[0] * a[1];
    area2 += cross;
    cx += (a[0] + b[0]) * cross;
    cy += (a[1] + b[1]) * cross;
  }
  if (Math.abs(area2) < 1e-12) {
    return [
      points.reduce((sum, p) => sum + p[0], 0) / points.length,
      points.reduce((sum, p) => sum + p[1], 0) / points.length,
    ];
  }
  return [cx / (3 * area2), cy / (3 * area2)];
}

function polyArea(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return sum / 2;
}

// Douglas-Peucker for a closed outline, tolerance in mm. Mesh-derived
// outlines walk one point per boundary vertex, so a gently curved or
// near-straight stretch arrives as dozens of sub-millimetre segments that
// are almost — but not exactly — collinear, which the relative-tolerance
// collinear pass below leaves alone. Collapsing anything that deviates
// less than a fraction of a trace width is invisible at PCB scale and
// keeps every real corner.
function rdpSimplifyClosed(points, epsilonMm) {
  if (points.length < 4) return points.slice();
  let split = 0;
  let best = -1;
  for (let i = 1; i < points.length; i++) {
    const d = distance2(points[0], points[i]);
    if (d > best) { best = d; split = i; }
  }
  const half1 = rdpSimplifyOpen(points.slice(0, split + 1), epsilonMm);
  const half2 = rdpSimplifyOpen(points.slice(split).concat([points[0]]), epsilonMm);
  const out = half1.slice(0, -1).concat(half2.slice(0, -1));
  return out.length >= 3 ? out : points.slice();
}

function rdpSimplifyOpen(points, epsilonMm) {
  if (points.length < 3) return points.slice();
  const a = points[0];
  const b = points[points.length - 1];
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const ab2 = abx * abx + aby * aby;
  let maxDist = -1;
  let idx = -1;
  for (let i = 1; i < points.length - 1; i++) {
    const p = points[i];
    let d;
    if (ab2 < 1e-20) {
      d = Math.hypot(p[0] - a[0], p[1] - a[1]);
    } else {
      const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / ab2));
      d = Math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby));
    }
    if (d > maxDist) { maxDist = d; idx = i; }
  }
  if (maxDist <= epsilonMm) return [a, b];
  const left = rdpSimplifyOpen(points.slice(0, idx + 1), epsilonMm);
  const right = rdpSimplifyOpen(points.slice(idx), epsilonMm);
  return left.slice(0, -1).concat(right);
}

function simplifyCollinear(points, relativeTolerance = 1e-5) {
  if (points.length < 3) return points.slice();
  const output = [];
  for (let i = 0; i < points.length; i++) {
    const a = points[(i - 1 + points.length) % points.length];
    const b = points[i];
    const c = points[(i + 1) % points.length];
    const ab = [b[0] - a[0], b[1] - a[1]];
    const bc = [c[0] - b[0], c[1] - b[1]];
    const la = Math.hypot(ab[0], ab[1]);
    const lb = Math.hypot(bc[0], bc[1]);
    if (la < 1e-10 || lb < 1e-10) continue;
    if (Math.abs(cross2(ab, bc)) < relativeTolerance * la * lb &&
        ab[0] * bc[0] + ab[1] * bc[1] > 0) continue;
    output.push(b);
  }
  return output.length >= 3 ? output : points.slice();
}

function removeDuplicateNeighbors(points, tolerance) {
  const output = [];
  for (const point of points) {
    if (!output.length || distance2(point, output[output.length - 1]) > tolerance * tolerance) {
      output.push(point.slice());
    }
  }
  if (output.length > 1 && distance2(output[0], output[output.length - 1]) <= tolerance * tolerance) {
    output.pop();
  }
  return output;
}

// The backend treats outline[0] -> outline[1] as the cable attachment edge.
function rotateToBottomEdge(points) {
  let best = 0;
  let bestY = Infinity;
  let bestLength = -1;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const averageY = (a[1] + b[1]) / 2;
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (averageY < bestY - 1e-7 ||
        (Math.abs(averageY - bestY) <= 1e-7 && length > bestLength)) {
      best = i;
      bestY = averageY;
      bestLength = length;
    }
  }
  return points.slice(best).concat(points.slice(0, best));
}

function cross2(a, b) {
  return a[0] * b[1] - a[1] * b[0];
}

function distance2(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

// --- Draw-a-patch-on-the-surface support ------------------------------------
// The "Draw patch" mode lets the user click points directly on the 3D surface;
// consecutive points are joined by the shortest path along the mesh, the
// closed loop splits the mesh in two, and the enclosed side becomes the
// selection — unfolded by the normal machinery, optionally after Laplacian
// smoothing so surface texture (tread bumps, ribs) does not distort the
// flattened outline. Boundary vertices are pinned during smoothing, so the
// drawn outline's true dimensions are preserved.

function vertexAdjacency(model) {
  // Cached on the model: vertexId -> [{v, length}] over unique mesh edges.
  if (model._vertexAdj) return model._vertexAdj;
  const adj = new Map();
  const seen = new Set();
  for (const ids of model.triVerts) {
    for (const [a, b] of triangleEdges(ids)) {
      const key = edgeKey(a, b);
      if (seen.has(key)) continue;
      seen.add(key);
      const length = model.vertices[a].distanceTo(model.vertices[b]);
      if (!adj.has(a)) adj.set(a, []);
      if (!adj.has(b)) adj.set(b, []);
      adj.get(a).push({ v: b, length });
      adj.get(b).push({ v: a, length });
    }
  }
  model._vertexAdj = adj;
  return adj;
}

export function shortestSurfacePath(model, startVertex, endVertex) {
  // Dijkstra over the mesh's vertex graph, weighted by true edge length.
  // Returns the vertex-id path including both endpoints.
  if (startVertex === endVertex) return [startVertex];
  const adj = vertexAdjacency(model);
  const dist = new Map([[startVertex, 0]]);
  const prev = new Map();
  const visited = new Set();
  // Simple binary-heap-free frontier: fine at mesh sizes this app handles.
  const frontier = new Map([[startVertex, 0]]);
  while (frontier.size) {
    let current = null, best = Infinity;
    for (const [v, d] of frontier) if (d < best) { best = d; current = v; }
    frontier.delete(current);
    if (current === endVertex) break;
    visited.add(current);
    for (const { v, length } of adj.get(current) || []) {
      if (visited.has(v)) continue;
      const next = best + length;
      if (next < (dist.get(v) ?? Infinity)) {
        dist.set(v, next);
        prev.set(v, current);
        frontier.set(v, next);
      }
    }
  }
  if (!prev.has(endVertex)) throw new Error("No surface path between those points.");
  const path = [endVertex];
  while (path[path.length - 1] !== startVertex) path.push(prev.get(path[path.length - 1]));
  return path.reverse();
}

function firstFaceTouchingVertex(model, vertexId) {
  for (let t = 0; t < model.triVerts.length; t++) {
    const ids = model.triVerts[t];
    if (ids[0] === vertexId || ids[1] === vertexId || ids[2] === vertexId) return t;
  }
  return -1;
}

function connectedFaces(model, seedFace) {
  // Flood fill with no edge blocked — the whole mesh shell touching `seedFace`.
  const visited = new Set([seedFace]);
  const stack = [seedFace];
  while (stack.length) {
    const t = stack.pop();
    for (const adjacency of model.triAdjacency[t] || []) {
      const n = adjacency.triangle;
      if (visited.has(n)) continue;
      visited.add(n);
      stack.push(n);
    }
  }
  return visited;
}

export function splitByLoop(model, loopVertexPath) {
  // loopVertexPath: closed vertex cycle (first == last is optional). Blocks its
  // edges and flood-fills triangles across every other shared edge; returns the
  // resulting components sorted smallest-area first. Two components = a clean
  // cut; one = the loop did not separate the surface (degenerate drawing).
  //
  // The fill is restricted to the single mesh shell the loop was drawn on
  // (a whole-mesh flood fill ignoring the cut, seeded from one of the loop's
  // own vertices). End-effector STLs routinely arrive as many disconnected
  // shells — an overmold, a separate fillet patch, stray debris islands from
  // a mesh export — that have nothing to do with the drawn patch; without
  // this restriction those islands show up as extra "sides", and if the
  // smallest one (picked as the default side) happens to be a tiny
  // watertight island, unfolding throws "Closed surface selected" for a
  // reason unrelated to what the user drew.
  const cycle = loopVertexPath.slice();
  if (cycle[0] !== cycle[cycle.length - 1]) cycle.push(cycle[0]);
  const blocked = new Set();
  for (let i = 0; i + 1 < cycle.length; i++) blocked.add(edgeKey(cycle[i], cycle[i + 1]));

  const seedFace = firstFaceTouchingVertex(model, cycle[0]);
  const shellFaces = seedFace >= 0 ? connectedFaces(model, seedFace) : null;
  const candidates = shellFaces ? [...shellFaces] : model.triVerts.map((_, i) => i);

  const assigned = new Map();
  const components = [];
  for (const seed of candidates) {
    if (assigned.has(seed)) continue;
    const id = components.length;
    const triangles = new Set([seed]);
    assigned.set(seed, id);
    const stack = [seed];
    while (stack.length) {
      const t = stack.pop();
      for (const adjacency of model.triAdjacency[t] || []) {
        const n = adjacency.triangle;
        if (shellFaces && !shellFaces.has(n)) continue;
        if (assigned.has(n)) continue;
        if (blocked.has(edgeKey(adjacency.va, adjacency.vb))) continue;
        assigned.set(n, id);
        triangles.add(n);
        stack.push(n);
      }
    }
    let area = 0;
    for (const t of triangles) area += model.triAreas[t];
    components.push({ triangles, area });
  }
  components.sort((a, b) => a.area - b.area);
  return components;
}

export function smoothedPatchModel(model, triangleIds, iterations = 12, lambda = 0.5) {
  // Shallow model copy whose patch-interior vertices are Laplacian-relaxed
  // (uniform weights, patch-internal neighbors only). Pinned: every vertex on
  // the patch's own boundary or shared with a non-patch triangle — so the
  // drawn outline keeps its true measured dimensions while bumps inside are
  // flattened out. Normals/areas are recomputed for patch triangles, which is
  // all the unfold reads.
  const patch = new Set([...triangleIds].map(Number));
  const patchVertices = new Set();
  const edgeUse = new Map();
  for (const t of patch) {
    for (const v of model.triVerts[t]) patchVertices.add(v);
    for (const [a, b] of triangleEdges(model.triVerts[t])) {
      const key = edgeKey(a, b);
      edgeUse.set(key, (edgeUse.get(key) || 0) + 1);
    }
  }
  const pinned = new Set();
  for (const [key, count] of edgeUse) {
    if (count === 1) {
      const [a, b] = key.split("_").map(Number);
      pinned.add(a); pinned.add(b);
    }
  }
  for (let t = 0; t < model.triVerts.length; t++) {
    if (patch.has(t)) continue;
    for (const v of model.triVerts[t]) if (patchVertices.has(v)) pinned.add(v);
  }
  // Patch-internal neighbor lists for the free vertices.
  const neighbors = new Map();
  for (const [key, count] of edgeUse) {
    const [a, b] = key.split("_").map(Number);
    if (!neighbors.has(a)) neighbors.set(a, []);
    if (!neighbors.has(b)) neighbors.set(b, []);
    neighbors.get(a).push(b);
    neighbors.get(b).push(a);
  }
  const positions = new Map();
  for (const v of patchVertices) positions.set(v, model.vertices[v].clone());
  for (let it = 0; it < iterations; it++) {
    const updates = [];
    for (const v of patchVertices) {
      if (pinned.has(v)) continue;
      const nbrs = neighbors.get(v) || [];
      if (!nbrs.length) continue;
      const mean = new THREE.Vector3();
      for (const n of nbrs) mean.add(positions.get(n));
      mean.multiplyScalar(1 / nbrs.length);
      const p = positions.get(v);
      updates.push([v, p.clone().lerp(mean, lambda)]);
    }
    for (const [v, p] of updates) positions.set(v, p);
  }
  const vertices = model.vertices.slice();
  for (const [v, p] of positions) vertices[v] = p;
  const triNormals = model.triNormals.slice();
  const triAreas = model.triAreas.slice();
  for (const t of patch) {
    const [a, b, c] = model.triVerts[t];
    const ab = vertices[b].clone().sub(vertices[a]);
    const ac = vertices[c].clone().sub(vertices[a]);
    const cross = new THREE.Vector3().crossVectors(ab, ac);
    const doubled = cross.length();
    triAreas[t] = doubled / 2;
    triNormals[t] = doubled > 1e-12 ? cross.multiplyScalar(1 / doubled) : model.triNormals[t].clone();
  }
  const copy = { ...model, vertices, triNormals, triAreas };
  delete copy._vertexAdj;
  return copy;
}
