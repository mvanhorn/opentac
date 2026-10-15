"""Tests for DXF outline import — polyline, multi-line, and arc-containing files."""

import io
import math
import sys
import os

import ezdxf

from opentac.cad.importers.dxf import parse_dxf_outline, DxfImportError


def _bytes_of(doc) -> bytes:
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def test_closed_lwpolyline_square():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 20), (0, 20)], close=True)

    result = parse_dxf_outline(_bytes_of(doc))
    assert len(result.loops) == 1
    assert len(result.loops[0]) == 4
    assert abs(result.areas[0] - 400.0) < 1e-6
    print("test_closed_lwpolyline_square: OK "
          f"({len(result.loops[0])} verts, area={result.areas[0]:.2f} mm^2)")


def test_square_from_separate_lines():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_line((0, 0), (20, 0))
    msp.add_line((20, 0), (20, 20))
    msp.add_line((20, 20), (0, 20))
    msp.add_line((0, 20), (0, 0))

    result = parse_dxf_outline(_bytes_of(doc))
    assert len(result.loops) == 1
    assert len(result.loops[0]) == 4
    assert abs(result.areas[0] - 400.0) < 1e-6
    print("test_square_from_separate_lines: OK "
          f"({len(result.loops[0])} verts, area={result.areas[0]:.2f} mm^2)")


def test_shape_with_arc():
    # Rounded corner: three straight edges + one arc, forming a closed loop.
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    r = 5
    msp.add_line((0, 0), (20, 0))
    msp.add_line((20, 0), (20, 20 - r))
    msp.add_arc(center=(20 - r, 20 - r), radius=r, start_angle=0, end_angle=90)
    msp.add_line((20 - r, 20), (0, 20))
    msp.add_line((0, 20), (0, 0))

    result = parse_dxf_outline(_bytes_of(doc))
    assert len(result.loops) == 1
    # Rounded-corner square area = full square minus the corner square-minus-quarter-circle cutoff
    expected_area = 20 * 20 - (r * r - math.pi * r * r / 4)
    assert abs(result.areas[0] - expected_area) < 0.5
    print("test_shape_with_arc: OK "
          f"({len(result.loops[0])} verts, area={result.areas[0]:.2f} mm^2, "
          f"expected~={expected_area:.2f})")


def test_multiple_loops_picks_largest():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 20), (0, 20)], close=True)
    msp.add_lwpolyline([(5, 5), (8, 5), (8, 8), (5, 8)], close=True)  # small inner loop

    result = parse_dxf_outline(_bytes_of(doc))
    assert len(result.loops) == 2
    assert result.areas[result.chosen_index] == max(result.areas)
    assert any("2 closed shapes" in w for w in result.warnings)
    print(f"test_multiple_loops_picks_largest: OK (chosen area={result.areas[result.chosen_index]:.2f})")


def test_unitless_warns_and_assumes_mm():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 0
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 20), (0, 20)], close=True)

    result = parse_dxf_outline(_bytes_of(doc))
    assert any("assuming millimeters" in w for w in result.warnings)
    print("test_unitless_warns_and_assumes_mm: OK")


def test_open_geometry_raises():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_line((0, 0), (20, 0))
    msp.add_line((20, 0), (20, 20))
    # missing closing edges -> no closed loop

    raised = False
    try:
        parse_dxf_outline(_bytes_of(doc))
    except DxfImportError:
        raised = True
    assert raised
    print("test_open_geometry_raises: OK")


if __name__ == "__main__":
    test_closed_lwpolyline_square()
    test_square_from_separate_lines()
    test_shape_with_arc()
    test_multiple_loops_picks_largest()
    test_unitless_warns_and_assumes_mm()
    test_open_geometry_raises()
    print("\n=== ALL DXF IMPORT TESTS PASSED ===")
