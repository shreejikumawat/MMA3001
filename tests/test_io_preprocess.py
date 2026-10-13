"""Tests for filename parsing, discovery, preprocessing and input validation."""

import cv2
import numpy as np
import pytest

from treeid.io import discover, load_sample, parse_filename
from treeid.preprocess import (crop_resize, depth_foreground, excess_green, make_input,
                               resize_mask)
from treeid.splits import split_by_row


def test_parse_standard_name():
    d = parse_filename("R02N06_Summer_GT-14-22-5.png")
    assert d["tree_id"] == "R02N06" and d["row"] == 2 and d["mod"] == "GT"
    assert d["extra"] == ""


def test_parse_row4_renamed_file():
    """Row-4 winter files carry a second tree id and an underscore before the time."""
    d = parse_filename("R04N01_Winter_D-R04N05_RGB_12-17-20.png")
    assert d["tree_id"] == "R04N01" and d["extra"] == "-R04N05_RGB"


def test_parse_rejects_foreign_names():
    assert parse_filename("FANK_Sat.png") is None


def _write_tree(root, tid="R01N01"):
    rng = np.random.default_rng(0)
    for season in ("Summer", "Winter"):
        cv2.imwrite(str(root / f"{tid}_{season}_RGB-1-2-3.png"), rng.integers(0, 255, (480, 640, 3), np.uint8))
        cv2.imwrite(str(root / f"{tid}_{season}_D-1-2-3.png"), rng.integers(0, 200, (480, 640), np.uint8))
    gt = np.zeros((256, 256), np.uint8); gt[100:, 120:130] = 255
    cv2.imwrite(str(root / f"{tid}_Summer_GT-1-2-3.png"), gt)


def test_discover_and_load(tmp_path):
    _write_tree(tmp_path)
    recs, unparsed = discover(tmp_path)
    assert unparsed == [] and recs["R01N01"].complete
    s = load_sample(recs["R01N01"])
    assert s["rgb"].shape == (480, 640, 3) and s["gt"].dtype == bool and s["gt"].sum() == 1560


def test_discover_missing_root():
    with pytest.raises(FileNotFoundError):
        discover("/nonexistent/path")


def test_load_bad_season(tmp_path):
    _write_tree(tmp_path)
    recs, _ = discover(tmp_path)
    with pytest.raises(ValueError):
        load_sample(recs["R01N01"], "Autumn")


def test_crop_resize_shapes_and_nearest_for_masks():
    m = np.zeros((480, 640), np.uint8); m[:, 300:310] = 1
    out = crop_resize(m, 256, is_mask=True)
    assert out.shape == (256, 256) and set(np.unique(out)) <= {0, 1}
    with pytest.raises(ValueError):
        crop_resize(m, 256, crop=(0, 500, 0, 640))


def test_resize_mask_identity():
    m = np.random.default_rng(1).random((64, 64)) > 0.5
    assert np.array_equal(resize_mask(m, 64), m)


def test_excess_green_ordering():
    leaf = np.array([[[40, 160, 40]]], np.uint8)
    bark = np.array([[[120, 100, 90]]], np.uint8)
    assert excess_green(leaf)[0, 0] > 0.3 > excess_green(bark)[0, 0]


def test_depth_foreground_rank_and_missing():
    d = np.array([[0, 10, 20, 30]], np.uint8)
    near_far_small = depth_foreground(d, near_is_bright=False)
    assert near_far_small[0, 0] == 0                # missing stays 0
    assert near_far_small[0, 1] == 1.0              # smallest depth = nearest = 1
    assert near_far_small[0, 3] == 0.0


def test_make_input_channels_and_validation():
    rgb = np.zeros((480, 640, 3), np.uint8); dep = np.ones((480, 640), np.uint8)
    assert make_input(rgb, dep, 64).shape == (5, 64, 64)
    assert make_input(rgb, dep, 64, use_depth=False).shape == (3, 64, 64)
    with pytest.raises(ValueError):
        make_input(rgb[..., 0], dep, 64)
    with pytest.raises(ValueError):
        make_input(rgb, dep[:100], 64)


def test_split_is_disjoint_and_grouped():
    rows = np.repeat(np.arange(1, 21), 3)
    tr, va, te = split_by_row(rows)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    for part in (tr, va, te):                       # no row is split across partitions
        assert set(rows[part]).isdisjoint(set(rows[~part]))
    with pytest.raises(ValueError):
        split_by_row(rows, test_rows=(1,), val_rows=(1,))
