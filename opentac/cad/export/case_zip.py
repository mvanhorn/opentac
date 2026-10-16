"""The printed case, added to the export ZIP when the user downloads it.

The ZIP carries a `case/` folder: the bottom piece, all three lids (bumped,
flat, hatched) and a README naming the primary lid and the settings used.
Every lid fits the same bottom, so the user can print a different one later
without coming back to the app.

Built at download time by its own endpoint (/zip-with-case), never inside
/generate. Three lids are three boolean-heavy meshes, and generate is the call
the free-tier backend can least afford to slow down; a user who regenerates
ten times and downloads once pays for the case once.

Failure is soft at every level. A lid that cannot be built is left out and
named in the README; if no lid can be built there is no bottom either, and the
ZIP is returned without a case folder. The board files are never held back
for the case.
"""

from __future__ import annotations

import base64
import io
import re
import zipfile
from dataclasses import dataclass, field, replace

from opentac.cad.pipeline.case import CaseParams, build_case

CASE_DIR = "case/"
README_NAME = CASE_DIR + "README.txt"

# Lid style -> (file suffix, what it is). The suffixes extend the names the
# case panel's own buttons use (<stem>_case_bottom.stl, <stem>_case_top.stl),
# so a file dragged out of the folder still says what it is.
LIDS = {
    "bumps": ("case_top_bumps", "lid with shear-sensing bumps, one over each 2x2 group of taxels"),
    "flat": ("case_top_flat", "flat lid, normal force only"),
    "hatch": ("case_top_hatched", "hatched lid, cut between taxels so a press stays on the taxel under it"),
}
LID_NAMES = {"bumps": "bumped lid", "flat": "flat lid", "hatch": "hatched lid"}
DEFAULT_PRIMARY = "bumps"
BOTTOM_SUFFIX = "case_bottom"


def file_stem(name) -> str:
    """The design-name rule from design-name.js's sanitize(), so names coming
    from the page and names built here look alike."""
    s = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", str(name or ""))
    s = re.sub(r"[^A-Za-z0-9\-_]+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    s = re.sub(r"^[_\-]+|[_\-]+$", "", s)
    return s[:48]


@dataclass
class CaseFiles:
    files: dict = field(default_factory=dict)      # path in the ZIP -> bytes
    left_out: dict = field(default_factory=dict)   # part name -> reason
    warnings: list = field(default_factory=list)


def _reason(exc: Exception) -> str:
    # ValueErrors from the case code are written for the user; anything else is
    # a geometry library failing and gets its type so it can be reported.
    return str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"


def _stl_is_closed(data: bytes) -> bool:
    """Watertight as the file will be read, not as the mesh was in memory.

    An STL stores only triangle corners; a mesh that is closed in memory can come
    back open once near-duplicate vertices are merged on load, and the file is
    what the slicer gets.
    """
    import trimesh
    return bool(trimesh.load(io.BytesIO(data), file_type="stl").is_watertight)


def build_case_files(final_outline, cable_y, placements, active, pitch_x, pitch_y,
                     base: CaseParams, *, primary: str = DEFAULT_PRIMARY, stem: str = "",
                     tail_extension: dict | None = None) -> CaseFiles:
    """Bottom + every lid style, as STL bytes keyed by their path in the ZIP.

    `base` carries the user's settings; its `bumps` and `hatch` flags are
    overridden per lid. The bottom does not depend on the lid style, so it is
    taken from the first lid that builds.
    """
    primary = primary if primary in LIDS else DEFAULT_PRIMARY
    order = [primary] + [k for k in LIDS if k != primary]
    stem = file_stem(stem)
    prefix = f"{stem}_" if stem else ""

    out = CaseFiles()
    bottom = None
    tops, warns = {}, {}
    for style in order:
        p = replace(base, bumps=(style == "bumps"), hatch=(style == "hatch"))
        try:
            r = build_case(final_outline, cable_y, placements, active, pitch_x, pitch_y, p,
                           tail_extension=tail_extension)
        except Exception as exc:  # noqa: BLE001 - one lid failing must not sink the rest
            out.left_out[LID_NAMES[style]] = _reason(exc)
            continue
        tops[style] = r.top
        warns[style] = list(r.warnings)
        if bottom is None:
            bottom = r.bottom

    if bottom is None:
        # Nothing built: the reasons are almost always the same shared-geometry
        # error three times, so report it once.
        reasons = list(dict.fromkeys(out.left_out.values()))
        out.left_out = {"case": "; ".join(reasons)}
        return out

    # A warning every lid raised is about the case (cable, rib pattern); one
    # only some raised is about that lid, and says which.
    shared = [w for w in warns[next(iter(warns))] if all(w in v for v in warns.values())]
    out.warnings = list(shared)
    for style, ws in warns.items():
        out.warnings += [f"{LID_NAMES[style].capitalize()}: {w}" for w in ws if w not in shared]

    names = {}
    path = f"{CASE_DIR}{prefix}{BOTTOM_SUFFIX}.stl"
    out.files[path] = bottom.export(file_type="stl")
    names["bottom"] = path
    if not _stl_is_closed(out.files[path]):
        out.warnings.append("The bottom piece is not a closed solid; it may not slice cleanly.")
    for style in order:
        if style not in tops:
            continue
        path = f"{CASE_DIR}{prefix}{LIDS[style][0]}.stl"
        out.files[path] = tops[style].export(file_type="stl")
        names[style] = path
        if not _stl_is_closed(out.files[path]):
            out.warnings.append(
                f"The {LID_NAMES[style]} is not a closed solid; it may not slice cleanly.")

    out.files[README_NAME] = _readme(stem, primary, order, names, base, out).encode()
    return out


def _readme(stem, primary, order, names, p: CaseParams, out: CaseFiles) -> str:
    base = lambda path: path[len(CASE_DIR):]  # noqa: E731
    width = max(len(base(v)) for v in names.values()) + 3
    lines = [f"Printed case for {stem or 'this sensor'}", "",
             "Print the bottom and one lid, in TPU (about Shore 85-95A). "
             "All three lids fit the same bottom.", ""]
    lines.append(f"  {base(names['bottom']):<{width}}bottom tray; the board and the Velostat sit in its pocket")
    for style in order:
        if style in names:
            mark = "  <- primary lid" if style == primary else ""
            lines.append(f"  {base(names[style]):<{width}}{LIDS[style][1]}{mark}")
    if primary not in names:
        lines += ["", f"The primary lid ({LID_NAMES[primary]}) could not be built; see below."]

    bp = p.bump_params
    lines += ["", "Settings (mm):",
              f"  pocket depth {p.recess_depth_mm:g}, floor {p.plate_thickness_mm:g}, "
              f"lid {p.lid_thickness_mm:g}, skirt {p.skirt_width_mm:g}, "
              f"fit tolerance {p.recess_tolerance_mm:g}",
              f"  rib period {p.rib_period_mm:g}, gap {p.rib_gap_mm:g}",
              f"  bumps: height {bp.bump_height_mm:g}, base {100 * bp.bump_base_frac:g}% of the "
              f"2x2 block, top {100 * bp.bump_top_ratio:g}% of the base"]
    hatch = (f"  hatch: cut through, cut width {p.hatch_groove_mm:g}, tabs {p.hatch_tab_mm:g}"
             if p.hatch_through else
             f"  hatch: grooves {p.hatch_groove_mm:g} wide, {p.hatch_depth_mm:g} deep")
    if p.hatch_bumps:
        hatch += (f"; a {p.hatch_bump_dia_mm:g} bump, {p.hatch_bump_height_mm:g} high, "
                  f"on the {p.hatch_bump_face} of each taxel")
    lines.append(hatch)

    if out.left_out:
        lines += ["", "Left out:"] + [f"  - {k}: {v}" for k, v in out.left_out.items()]
    if out.warnings:
        lines += ["", "Warnings from the case builder:"] + [f"  - {w}" for w in out.warnings]
    lines += ["", "The case is cut to this exact board. If the board is regenerated, "
              "download it again.", ""]
    return "\n".join(lines)


def zip_with_case(zip_b64: str, files: dict) -> str:
    """Return the export ZIP with its `case/` folder replaced by `files`.

    Rewrites the archive, as `zip_with_design` does, so the board files are
    copied byte for byte. Any case already in the ZIP is dropped first: it
    may have been built for other settings, and an empty `files` must leave
    no case behind rather than a stale one.
    """
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in src.infolist():
            if info.filename.startswith(CASE_DIR):
                continue
            zf.writestr(info, src.read(info.filename))
        for path, data in files.items():
            zf.writestr(path, data)
    return base64.b64encode(out.getvalue()).decode()
