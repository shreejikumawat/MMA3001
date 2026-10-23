# Recovering apple-tree leader structure from summer RGB-D images

MMA3001 Numerical Methods and Machine Learning
Shreeji Kumawat (37573578).

## What this project does

A pruning robot needs to know where the trunk and main branches of a tree are but in summer a lot of it is hidden behind leaves. This project takes one summer colour and depth photo of an apple tree and finds the trunk and the two main branches even where leaves cover them. It then works out the height where the trunk splits and the angle between the two branches.

Two methods are compared. The first is a simple rule based method and the second is a small U-Net. The U-Net using 64 px images did the best and was picked as the final model. The full details are in the report (`doc/Report.pdf`).

## Folders

- `src/treeid/` has the main code and each function is explained in its docstring.
- `scripts/` has the scripts that run each step.
- `tests/` has the tests.
- `results/` has the results, figures and logs.
- `doc/` has the report, the HTML documentation, the test report and the AI use log.
- `data/` explains how to get the dataset. The data itself is not uploaded.
- `notebooks/demo.ipynb` shows the tool working on one test tree.

## Setup

Python 3.10 or newer is needed. A normal CPU is enough to run everything.

```bash
python -m venv .venv
.venv\Scripts\activate        # on Mac or Linux use: source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

First get the dataset by following `data/README.md`. Then run these in order.

```bash
# 1. Prepare the data
python scripts/build_cache.py --size 64
python scripts/build_cache.py --size 128
python scripts/build_cache.py --size 256

# 2. Check the data
python scripts/run_audit.py

# 3. Run the simple baseline
python scripts/run_baseline.py

# 4. Train the U-Net models (this takes a few hours on a CPU)
bash scripts/queue.sh
python scripts/refine_labels.py
bash scripts/queue_v2.sh
bash scripts/queue_v3.sh
python scripts/validation_selection.py

# 5. Make the tables and figures
python scripts/analyse.py

# 6. Time the code and count the FLOPs
python scripts/profile_pipeline.py

# 7. Trunk position check
python scripts/trunk_check.py
```

The random seeds are fixed so the results should come out the same each time. The test rows are also fixed in `src/treeid/splits.py` before any training is done.

## Inputs and outputs

| | Format | Units | If the input is wrong |
|---|---|---|---|
| Summer colour image (input) | PNG, 640×480, 3 channels | 0 to 255 | Wrong size gives a `ValueError` |
| Summer depth image (input) | PNG, 640×480 | Relative, bigger = further, 0 = missing | Missing pixels are marked in a extra channel |
| Branch mask (output) | True/False image | Branch or not branch | n/a |
| Leader angle (output) | Number | Degrees | `None` if no fork is found |
| Fork row (output) | Whole number | Pixel row, 0 = top | `None` if no fork is found |
| 3-D skeleton (output) | List of points | Relative units | Points with no depth are dropped |

## Tests

```bash
pytest -q
```

There is 53 tests and they all pass. The latest results are saved in `doc/test_report.txt`. The tests check each part of the code against cases where the right answer is already known.

## AI use

AI was used in this project and it is explained in the report and in `doc/ai_use_log.md`.

## Licence and data

The code is under the MIT licence. The TreeID dataset is not covered by this licence.

## References

- Chen, Z., Ting, D., Newbury, R., Chen, C. (2020). Semantic segmentation for partially occluded apple trees based on deep learning. arXiv:2010.06879.
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
- Libraries used: NumPy, SciPy, scikit-image, OpenCV, PyTorch, pandas, Matplotlib, pytest, pdoc.
