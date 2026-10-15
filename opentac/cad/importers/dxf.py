"""DXF outline import — flatten arbitrary DXF geometry into closed 2D polygons.

Entities are flattened to line segments (uniformly, regardless of whether they
were originally LINE/ARC/CIRCLE/LWPOLYLINE/SPLINE/etc via ezdxf's Path API),
segment endpoints are snap-merged, and the resulting graph is walked into
closed loops. Winding order / cable-edge normalization is intentionally left
to the frontend so there's a single implementation shared with hand-drawn
outlines.
"""

import io
from dataclasses import dataclass, field

import ezdxf
from ezdxf import path as ezpath

FLATTEN_TOL_MM = 0.05
SNAP_TOL_MM = 0.01

# DXF $INSUNITS -> scale factor to mm (only the common CAD units are handled;
# anything else falls back to "assume mm" with a warning).
_UNIT_SCALE = {
    1: 25.4,     # inches
    2: 304.8,    # feet
    4: 1.0,      # millimeters
    5: 10.0,     # centimeters
    6: 1000.0,   # meters
}


class DxfImportError(ValueError):
    """Raised when a DXF file can't be parsed into a usable outline."""


@dataclass
class DxfImportResult:
    loops: list             # list of closed polygons, each Nx2 [x, y] mm (no duplicated closing vertex)
    areas: list              # abs(shoelace area) per loop, mm^2, same order as loops
    chosen_index: int        # index of the largest-area loop
    warnings: list = field(default_factory=list)


def parse_dxf_outline(content: bytes) -> DxfImportResult:
    warnings = []

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    try:
        doc = ezdxf.read(io.StringIO(text))
    except ezdxf.DXFStructureError as e:
        raise DxfImportError(f"Could not parse DXF file: {e}")
    except Exception as e:
        raise DxfImportError(f"Could not read DXF file: {e}")

    scale, unit_warning = _resolve_unit_scale(doc)
    if unit_warning:
        warnings.append(unit_warning)

    msp = doc.modelspace()

    segments = []
    skipped_types = set()
    for entity in msp:
        try:
            p = ezpath.make_path(entity)
        except TypeError:
            skipped_types.add(entity.dxftype())
            continue
        pts = [(v.x * scale, v.y * scale) for v in p.flattening(FLATTEN_TOL_MM)]
        for a, b in zip(pts[:-1], pts[1:]):
            if _dist(a, b) > 1e-9:
                segments.append((a, b))

    if skipped_types:
        warnings.append(
            "Skipped unsupported entity types: " + ", ".join(sorted(skipped_types)))

    if not segments:
        raise DxfImportError("No drawable geometry found in DXF file.")

    loops = _weld_segments_into_loops(segments, warnings)

    if not loops:
        raise DxfImportError(
            "No closed outline found — check the DXF contains a single closed "
            "polyline/region (no gaps or open ends)."
        )

    areas = [abs(_shoelace_area(loop)) for loop in loops]
    chosen_index = max(range(len(loops)), key=lambda i: areas[i])

    if len(loops) > 1:
        warnings.append(
            f"{len(loops)} closed shapes found in file — using the largest "
            f"({areas[chosen_index]:.1f} mm²)."
        )

    return DxfImportResult(loops=loops, areas=areas, chosen_index=chosen_index, warnings=warnings)


def _resolve_unit_scale(doc):
    insunits = doc.header.get("$INSUNITS", 0)
    if insunits in _UNIT_SCALE:
        return _UNIT_SCALE[insunits], None
    return 1.0, (
        "DXF has no usable unit information — assuming millimeters. "
        "Verify the imported outline's scale."
    )


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _shoelace_area(loop):
    area = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _weld_segments_into_loops(segments, warnings):
    """Snap-merge segment endpoints, then walk each degree-2 component into a closed loop."""
    node_of = {}
    node_pt = []
    adj = {}

    def get_node(pt):
        key = (round(pt[0] / SNAP_TOL_MM), round(pt[1] / SNAP_TOL_MM))
        if key not in node_of:
            node_of[key] = len(node_pt)
            node_pt.append(pt)
            adj[node_of[key]] = set()
        return node_of[key]

    for a, b in segments:
        na, nb = get_node(a), get_node(b)
        if na == nb:
            continue
        adj[na].add(nb)
        adj[nb].add(na)

    visited_nodes = set()
    loops = []
    dangling_nodes = 0

    for start in list(adj.keys()):
        if start in visited_nodes:
            continue

        stack = [start]
        component = set()
        while stack:
            n = stack.pop()
            if n in component:
                continue
            component.add(n)
            stack.extend(adj[n] - component)
        visited_nodes |= component

        if any(len(adj[n]) != 2 for n in component):
            dangling_nodes += sum(1 for n in component if len(adj[n]) != 2)
            continue

        loop_nodes = []
        prev, cur = None, start
        for _ in range(len(component) + 1):
            loop_nodes.append(cur)
            nxt_candidates = adj[cur] - ({prev} if prev is not None else set())
            if not nxt_candidates:
                break
            nxt = next(iter(nxt_candidates))
            prev, cur = cur, nxt
            if cur == start:
                break

        if len(loop_nodes) >= 3 and cur == start:
            loops.append([list(node_pt[n]) for n in loop_nodes])

    if dangling_nodes:
        warnings.append(
            f"Ignored {dangling_nodes} open/branching endpoint(s) — some geometry "
            "did not form a clean closed loop."
        )

    return loops
