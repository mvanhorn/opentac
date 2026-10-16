"""The readout boards' JLCPCB assembly files agree with each other.

JLCPCB matches the placement file (CPL) to the BOM by designator. A part in
the CPL with no BOM line makes the order stop with "designator don't exist in
the BOM file" (the Tag-Connect programming pads J2 did this until they were
excluded from the position files too); a BOM line with no placement is never
placed.
"""
import csv
from pathlib import Path

import pytest

HW = Path(__file__).resolve().parents[2] / "hardware"
BOARDS = ["readout_60pin", "readout_30pin/top", "readout_30pin/bottom"]


def _designators(path, key):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = next(k for k in rows[0] if key in k)
    out = []
    for r in rows:
        out += [d.strip() for d in r[col].replace(";", ",").split(",") if d.strip()]
    return out


@pytest.mark.parametrize("board", BOARDS)
def test_bom_and_cpl_name_the_same_parts(board):
    prod = HW / board / "production"
    bom = _designators(next(prod.glob("*_bom.csv")), "esignator")
    cpl = _designators(next(prod.glob("*_positions.csv")), "esignator")
    assert len(cpl) == len(set(cpl)), "a designator is placed twice"
    assert sorted(set(cpl) - set(bom)) == [], "placed but not in the BOM"
    assert sorted(set(bom) - set(cpl)) == [], "in the BOM but never placed"


@pytest.mark.parametrize("board", BOARDS)
def test_bom_lines_carry_an_lcsc_part(board):
    bom = next((HW / board / "production").glob("*_bom.csv"))
    rows = list(csv.DictReader(open(bom, encoding="utf-8-sig")))
    col = next(k for k in rows[0] if "LCSC" in k)
    assert all(r[col].strip().startswith("C") for r in rows)
