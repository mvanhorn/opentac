"""Shared data types for the tactile PCB pipeline."""

from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Polygon


@dataclass
class Params:
    pixel_w_mm: float = 4.0
    pixel_h_mm: float = 4.0
    pitch_x_mm: float = 4.2
    pitch_y_mm: float = 4.2
    edge_keepout_mm: float = 0.5
    trace_w_mm: float = 0.2
    gap_mm: float = 0.2
    clearance_mm: float = 0.2
    center_clear_mm: float = 0.2
    edge_clear_mm: float = 0.1
    via_drill_mm: float = 0.15
    via_dia_mm: float = 0.35
    pad_pitch_mm: float = 0.5
    cable_length_mm: float = 12.0
    tail_width_mm: float | None = None
    tail_side_padding_mm: float = 0.0
    board_edge_clear_mm: float = 0.2
    # User-facing board-generation mode. Exactly two modes exist:
    #   "expand"        — grow the board outline until the wiring fits, keeping
    #                     the max-density sensor grid intact.
    #   "fixed_keepout" — keep the drawn outline exactly, reserve a wiring
    #                     keep-out band inside it, and drop the pixels that
    #                     reach into that band.
    # preserve_sensors/auto_expand_board below are internal flags derived from
    # this mode (see _params_for_board_mode in backend/main.py); they are no
    # longer exposed in the UI.
    board_mode: str = "expand"
    # Copper layers, the other user-facing choice (either board mode):
    #   2 — columns on F.Cu, rows on B.Cu, every wire run round the field;
    #   4 — rows leave on the inner layers instead (In1: each row to the
    #       connector; In2: links across notches, then rows In1 cannot fit),
    #       so the field needs no wiring band for rows. See
    #       pipeline/route_4layer.py.
    copper_layers: int = 2
    preserve_sensors: bool = True
    auto_expand_board: bool = True
    routing_margin_mm: float = 3.0
    # Fixed-outline mode only: minimum wiring keep-out width. None = let
    # keepout.analytic_widths decide from its own model. See fixed_keepout_floor.
    wiring_keepout_floor_mm: float | None = None
    follow_main_padding: bool = True
    follow_main_padding_mm: float = 2.0
    smooth_follow_padding: bool = True
    connector_pos_override: int | None = None
    # Fillet radius applied to the step in Edge.Cuts where a manually enlarged
    # connector's body is wider than the cable/tail leading into it. 0 disables.
    connector_shoulder_fillet_mm: float = 1.0
    # Target sensor grid size. When both are set (grow-board mode only), taxel
    # size and pitch are solved from the drawn outline instead of being read
    # from pixel_*_mm/pitch_*_mm — see solve_grid_pixel_size in pipeline/layout.py.
    target_cols: int | None = None
    target_rows: int | None = None
    # Anchor holes: Edge.Cuts cutouts at 2x2 block centres, one per shear bump,
    # so the bump sheet's posts key the cover to the board across the whole
    # array instead of only at the skirt. See pipeline/anchor_holes.py for what
    # sets the pitch (the intra-column F.Cu bus, not the taxel copper).
    anchor_holes: bool = False
    anchor_hole_dia_mm: float = 1.5
    anchor_hole_clearance_mm: float = 0.4
    # "uniform" — every taxel gap widened to fit a cutout, even though only one
    #             gap in two actually carries one.
    # "paired"  — only the intra-block gap is widened; blocks sit at the tight
    #             taxel spacing. Same cutouts, denser array.
    anchor_hole_lattice: str = "uniform"
    # Only X has to grow (the row bus clears the cutout already). Growing Y to
    # match keeps the 2x2 block square and the bump round, so shear sensitivity
    # stays isotropic; leaving Y tight buys taxels at the cost of an oval bump.
    anchor_hole_square_cells: bool = True
    # Printed on the front silkscreen just inboard of the connector, so a
    # fabricated board says which design it is. Not geometry: it changes
    # nothing the router or the packer sees. Empty means no label.
    design_name: str = ""
    # Resolved by anchor_holes.apply_to_params() at request time, not user
    # input: the step sequence the packer cycles. None = constant pitch.
    pack_pattern_x: list | None = None
    pack_pattern_y: list | None = None


@dataclass
class Layout:
    placements_xy: np.ndarray          # Nx2 pixel centers
    row_net_per_pixel: list[str] = field(default_factory=list)
    col_net_per_pixel: list[str] = field(default_factory=list)
    connector_pos: int = 0
    connector_swap: bool = False
    pad_to_net: dict[int, str] = field(default_factory=dict)
    pad_y_mm: float = 0.0
    outline_poly: Polygon = None
    cable_poly: Polygon = None
    board_poly: Polygon = None
    sensor_outline_xy: np.ndarray | None = None
    final_outline_xy: np.ndarray | None = None
    routing_margin_applied_mm: float = 0.0
    tail: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)


@dataclass
class Route:
    net_name: str
    pad_num: int
    anchor_xy: np.ndarray              # [x, y]
    pad_xy: np.ndarray                 # [x, y]
    polyline_xy: np.ndarray            # Mx2
    # Copper layer: None = the face's own (F.Cu columns, B.Cu rows);
    # "In1.Cu" / "In2.Cu" on 4-layer boards.
    layer: str | None = None


@dataclass
class PixelGeometry:
    xL: float
    xR: float
    ymid: float
    pitch: float
    y0: float
    spine_y_lo: float
    spine_y_hi: float
    total_slots: int
    x_end_left_max: float
    x_end_right_min: float
    via_x: float
    via_y: float
    # where each slot's column finger ends (cap centre x), for the odd slots
    # the column comb owns; see pixel_geometry
    col_end_x: list = field(default_factory=list)


@dataclass
class DrcViolation:
    layer: str
    net1: str
    net2: str
    type: str  # 'crossing' or 'clearance'
    seg1: np.ndarray  # 2x2
    seg2: np.ndarray  # 2x2
    distance: float
    min_required: float
