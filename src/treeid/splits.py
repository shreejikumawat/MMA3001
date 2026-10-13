"""Grouped train / validation / test split by orchard row.

Neighbouring trees in a row share lighting, background, camera pose and capture
time, so a random per-image split would leak this information and inflate
scores (MMA3001 Week 5.6, *grouped split*). Whole rows are therefore assigned to
one partition. Test rows are spread across the orchard (5, 10, 15, 20) so that
the test set samples the full range of conditions; they are fixed *before* any
model is trained and are used once for final reporting.
"""

from __future__ import annotations

import numpy as np

TEST_ROWS = (5, 10, 15, 20)
VAL_ROWS = (3, 13)


def split_by_row(rows: np.ndarray, test_rows=TEST_ROWS, val_rows=VAL_ROWS):
    """Boolean masks ``(train, val, test)`` over samples given their row numbers.

    Raises:
        ValueError: If test and validation rows overlap.
    """
    if set(test_rows) & set(val_rows):
        raise ValueError("test and validation rows overlap")
    rows = np.asarray(rows)
    test = np.isin(rows, test_rows)
    val = np.isin(rows, val_rows)
    train = ~(test | val)
    return train, val, test
