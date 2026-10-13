# Data

The dataset is **not** stored in this repository (750 MB, owned by its authors).

## Source

TreeID RGB-D apple-tree dataset (Chen, Granland *et al.*, 2021, arXiv:2010.06879),
supplied to MMA3001 students by Dr Keenan Granland via Google Drive:
`MMA3001 Dataset - Apple Tree RGBD / TreeID / Dataset (With Summer GT)`.

## Setup

1. Download the folder `Dataset (With Summer GT)` from the unit's Drive link
   (Unit Overview Spreadsheet → Week 5 → Project Datasets → Dataset 1).
2. Extract all PNGs into `data/raw/` (sub-folders are fine; files are found recursively).
3. Build the cached arrays used by every experiment:

```bash
python scripts/build_cache.py --size 64
python scripts/build_cache.py --size 128
python scripts/build_cache.py --size 256
```

## Contents (verified by `scripts/run_audit.py`)

| Item | Value |
|---|---|
| Trees | 562 in 20 orchard rows (6 to 44 trees per row) |
| Files per tree | 5: Summer RGB, Summer depth, Summer GT, Winter RGB, Winter depth |
| RGB | 640 × 480 × 3, uint8 |
| Depth | 640 × 480, uint8 stored as 3 identical channels; **0 = no return**; larger = farther (99.6% of trees) |
| GT | 256 × 256, binary (0/255), one connected trunk + two leaders per tree |

Known issues are listed in `results/audit.json` and discussed in the report.
