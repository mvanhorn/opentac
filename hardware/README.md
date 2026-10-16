# Hardware

KiCad 10 projects and fabrication files for the OpenTac readout boards.

| Folder | Board | Largest sensor |
|---|---|---|
| [`readout_60pin/`](readout_60pin) | 60-pin readout board | 30 rows x 30 columns |
| [`readout_30pin/`](readout_30pin) | 30-pin readout pair (top + bottom, stacked) | 15 rows x 15 columns |
| `libs/` | footprints and symbols used by the boards above | |

Each board folder has a README with the ordering steps. The pictures and turntable
videos on the website are made from these files with `scripts/make_board_media.py`.

## 3D models

The boards reference 3D models in `libs/3dmodels/`, which are not in this repository: they come
from LCSC/EasyEDA and from part manufacturers, under their own terms. The boards open, check and
export without them; only the 3D view and `scripts/make_board_media.py` need them. To get them,
fetch each part by its LCSC number (for example with
[easyeda2kicad](https://github.com/uPesy/easyeda2kicad.py)) or from the manufacturer, and save
them into `libs/3dmodels/` under the names the boards use: 5051103091, 74HC595BQ, ADG706BRUZ,
ADG732BSUZ, C160390, C191023, C2673377, C2681556, C404969, C493444, C5349955, LSM6DSV16BXTR,
STM32G491KEU6, TLV9362IDDFR, TS-1088-AR02016 (`.step`/`.stp`/`.wrl` as referenced).


## Design-rule and electrical-rule checks

KiCad 10's own checks (`kicad-cli pcb drc`, `kicad-cli sch erc`) report errors on these boards. Each class is listed here with the reason it is accepted; the boards' rule settings are left as they were drawn rather than loosened to hide them. The 30-pin pair has been fabricated by JLCPCB and works; the 60-pin board was ordered with the files in this folder. Limits quoted are JLCPCB's for 4-layer boards ([PCB capabilities](https://jlcpcb.com/capabilities/pcb-capabilities), October 2026).

**DRC**

| Board | Errors | Class | Why it is accepted |
|---|---|---|---|
| all three | 137 / 66 / 51 | `via_diameter` | Board setup asks for 0.5&nbsp;mm vias; the boards use 0.25&ndash;0.4&nbsp;mm. JLCPCB's minimum is 0.25&nbsp;mm (0.15&nbsp;mm hole). |
| 60-pin | 17 | `clearance` | 0.198&nbsp;mm between vias and the FPC connector's 0.5&nbsp;mm-pitch pads, against a 0.2&nbsp;mm setting; JLCPCB's minimum spacing is 0.09&nbsp;mm. |
| 30-pin top | 11 | `clearance` | Inside the micro-USB footprint (shield tab to its mounting hole) and between 0201 capacitor pads (0.18&nbsp;mm); both are the parts' own land patterns. |
| 30-pin top | 2 | `hole_clearance` | 0.2&nbsp;mm between the micro-USB's pins and its mounting holes, in the manufacturer footprint. |
| all three | 6 / 6 / 2 | `copper_edge_clearance` | 0.27&ndash;0.45&nbsp;mm copper to edge against a 0.5&nbsp;mm setting; JLCPCB's minimum to a routed edge is 0.2&nbsp;mm. |

Warnings (silkscreen over copper or past the edge, text height, footprints that differ from the library copy) do not affect fabrication.

**ERC**

| Board | Class | What it is |
|---|---|---|
| all three | `power_pin_not_driven` | Supply nets fed from the USB connector and the regulator carry no PWR_FLAG marker. |
| 60-pin | `single_global_label`, `label_dangling` (r1, r2, c31, c32) | The multiplexer and shift-register channels the 60-pin connector does not use: rows start at r3 and there are 30 columns. |
| 30-pin bottom | `single_global_label` (r16, c16, mux4) | The sixteenth channel of the 16:1 parts, unused with 15 rows and 15 columns. |
| 60-pin | `pin_not_connected` | The last shift register's serial output (IC12 Q7S), which nothing follows; and the solder jumpers JP3 and JP4, which are not wired on either side. |
| all three | `pin_to_pin` (warnings) | Outputs of the shift registers and multiplexer meeting on the row and column nets, which is the readout scheme. |
