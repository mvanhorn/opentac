# Hardware

KiCad 10 projects and fabrication files for the OpenTac readout boards.

| Folder | Board | Largest sensor |
|---|---|---|
| [`readout_60pin/`](readout_60pin) | 60-pin readout board | 30 rows x 30 columns |
| [`readout_30pin/`](readout_30pin) | 30-pin readout pair (top + bottom, stacked) | 15 rows x 15 columns |
| `libs/` | footprints, symbols and 3D models used by the boards above | |

Each board folder has a README with the ordering steps. The pictures and turntable
videos on the website are made from these files with `scripts/make_board_media.py`.
