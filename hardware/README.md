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

