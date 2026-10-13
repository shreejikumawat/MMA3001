# Seeing through the canopy: recovering apple-tree leader structure from RGB-D images

MMA3001 Numerical Methods and Machine Learning, Monash University, Semester 2 2026.
Individual project by **Shreeji Kumawat** (37573578).

## Purpose

Robotic pruning, thinning and harvesting in trellised orchards needs to know where
the woody structure of each tree is, but in summer most of it is hidden behind
leaves. This project builds and evaluates a tool that takes one summer RGB-D image
of a 2D open-V apple tree and returns:

1. a **branch mask** of the trunk and two leaders, including parts hidden by foliage;
2. two **structural quantities** a robot can plan with: the **fork height** and the
   **leader opening angle**;
3. a **3-D skeleton** (pixel coordinates plus relative depth).

Two approaches are compared on orchard rows never seen in training:

* a **classical baseline** (Hessian ridge filter + depth foreground + vegetation
  index, four parameters grid-searched), and
* a compact **U-Net** (0.48 M parameters) with RGB or RGB-D input.

The report (`doc/report.pdf`) gives the problem definition, validation design,
results, limitations and the AI-use reflection.

## Repository layout

```
├── README.md            this file
├── LICENSE              MIT (code only; the dataset is not covered)
├── requirements.txt     pinned minimum versions
├── pyproject.toml       package metadata and pytest configuration
├── src/treeid/          the library (documented with NumPy-style docstrings)
│   ├── io.py            dataset discovery, filename parsing, loading
│   ├── preprocess.py    crop/resample, excess-green, depth normalisation, network input
│   ├── align.py         FFT cross-correlation registration (sub-pixel)
│   ├── metrics.py       IoU, Dice, boundary F1, tolerant / shift-compensated centreline F1
│   ├── baseline.py      classical detector and its grid search
│   ├── model.py         U-Net and FLOP / arithmetic-intensity estimator
│   ├── train.py         training loop with early stopping
│   ├── splits.py        grouped train/val/test split by orchard row
│   └── skeleton.py      fork height, leader angle, 3-D back-projection
├── scripts/             command-line entry points (one per experiment)
├── tests/               pytest suite (unit + manufactured-solution tests)
├── doc/                 HTML API docs (pdoc), test report, AI-use log, report
├── data/                dataset instructions (data not committed)
└── results/             audit, metrics tables, figures, per-run logs
```

## Environment

Python ≥ 3.10. CPU is sufficient (all reported results were produced on a 2-core
CPU); a GPU (e.g. free Google Colab) speeds up training ~20×.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                      # optional; scripts also work without it
```

## Reproducing every result

```bash
# 0. Get the data: see data/README.md, then
python scripts/build_cache.py --size 64
python scripts/build_cache.py --size 128
python scripts/build_cache.py --size 256

# 1. Data audit (≈1 min)
python scripts/run_audit.py

# 2. How much does label mis-registration alone cost? (≈1 min)
python scripts/label_noise_study.py

# 3. Classical baseline: tune on training rows, score on test rows (≈5 min)
python scripts/run_baseline.py

# 4. All U-Net experiments (≈3 h on 2 CPU cores; ≈15 min on a Colab GPU)
bash scripts/queue.sh

# 5. Tables, statistics and figures
python scripts/analyse.py

# 6. Profiling and roofline (run with nothing else using the CPU)
python scripts/profile_pipeline.py

# 7. Tests and documentation
pytest --junitxml=doc/test_report.xml -q | tee doc/test_report.txt
pdoc src/treeid -o doc/html --docformat numpy
```

Random seeds are fixed in every script; the test rows (5, 10, 15, 20) are fixed in
`src/treeid/splits.py` before any training.

## Using the tool on one image

```python
import numpy as np, torch
from treeid.io import discover, load_sample
from treeid.preprocess import make_input
from treeid.model import UNet
from treeid.skeleton import leader_angle

recs, _ = discover("data/raw")
s = load_sample(recs["R10N05"])
x = make_input(s["rgb"], s["depth"], size=128, near_is_bright=False)

model = UNet(in_ch=5); model.load_state_dict(torch.load("results/runs/rgbd_s0/model.pt"))
model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(torch.from_numpy(x)[None]))[0, 0].numpy()
angle, info = leader_angle(prob > 0.5)
print(f"leader angle {angle:.1f} deg, fork at row {info['fork_row']} of 128")
```

### Inputs and outputs

| | Format | Units / domain | Invalid input handling |
|---|---|---|---|
| Summer RGB | PNG, 640×480×3 uint8 | 0 to 255 per channel | wrong shape → `ValueError` |
| Summer depth | PNG, 640×480 uint8 | relative, larger = farther, 0 = missing | missing pixels flagged in a validity channel |
| Branch probability | float32 H×W | [0, 1] in the 480×480 centre crop resampled to H×W | — |
| Branch mask | bool H×W | probability > validation-chosen threshold | — |
| Leader angle | float | degrees, opening of the V | `None` if no fork is found |
| Fork row | int | pixel row of the fork (0 = top) | `None` if no fork is found |
| 3-D skeleton | (K, 3) float | X, Y in relative units, Z in raw depth units (scale unknown) | pixels without depth dropped |

## Testing

```bash
pytest -q
```

Tests cover metric definitions against hand-computed values, recovery of known
shifts by the FFT registration (integer and sub-pixel), analytical V-angles for
the structure extractor, the FLOP counter against a hand count, repeatability of
training with a fixed seed, and input validation.

## AI use

AI assistance was used and is disclosed in `doc/AI_USE_LOG.md` and the report.

## Citation of data and methods

* Chen, Z., Ting, D., Newbury, R., Chen, C. (2021). *Semantic segmentation for
  partially occluded apple trees based on deep learning.* Computers and Electronics
  in Agriculture 181, 105952. arXiv:2010.06879.
* Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net.* MICCAI.
* Sato, Y. *et al.* (1998). *Three-dimensional multi-scale line filter.* Medical Image Analysis 2(2).
* Woebbecke, D. M. *et al.* (1995). *Color indices for weed identification.* Trans. ASAE 38(1).
* Libraries: NumPy, SciPy, scikit-image, OpenCV, PyTorch, pandas, Matplotlib, pdoc, pytest.
