"""Dataset discovery, filename parsing and loading for the TreeID RGB-D apple-tree dataset.

The TreeID dataset (Chen et al., 2021, arXiv:2010.06879) supplied by the MMA3001
teaching team stores five PNG files per tree::

    R01N05_Summer_RGB-14-22-1.png   640x480x3 uint8   summer colour image
    R01N05_Summer_D-14-22-1.png     640x480   uint8   summer depth (8-bit, scaled)
    R01N05_Summer_GT-14-22-1.png    256x256   uint8   branch mask (white = branch)
    R01N05_Winter_RGB-13-26-50.png  640x480x3 uint8   winter colour image
    R01N05_Winter_D-13-26-50.png    640x480   uint8   winter depth

``R`` is the orchard row, ``N`` the tree number in that row and ``U`` marks a
tree that could not be matched between seasons. The trailing ``hh-mm-ss`` is the
capture time stamp.

All loaders return NumPy arrays and never modify files on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: Regular expression for a TreeID filename. Groups: row, kind (N/U), number,
#: season, modality, time stamp. Anything after the time stamp (e.g. the
#: ``-R04N03_RGB`` suffix seen on one file) is captured in ``extra``.
FILENAME_RE = re.compile(
    r"^R(?P<row>\d{2})(?P<kind>[NU])(?P<num>\d{2})_"
    r"(?P<season>Summer|Winter)_(?P<mod>RGB|D|GT)"
    r"(?P<extra>.*?)[-_](?P<time>\d{1,2}-\d{1,2}-\d{1,2})\.png$"
)

#: Modalities expected for a complete sample.
REQUIRED_KEYS = ("Summer_RGB", "Summer_D", "Summer_GT", "Winter_RGB", "Winter_D")


@dataclass
class TreeRecord:
    """All files belonging to one physical tree.

    Attributes:
        tree_id: Identifier such as ``"R01N05"``.
        row: Orchard row number (1-based).
        matched: ``True`` for ``N`` trees, ``False`` for unmatched ``U`` trees.
        files: Mapping from ``"<Season>_<Modality>"`` to the file path.
        anomalies: Free-text notes on naming problems found while parsing.
    """

    tree_id: str
    row: int
    matched: bool
    files: dict[str, Path] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """``True`` when all five required modalities are present."""
        return all(k in self.files for k in REQUIRED_KEYS)


def parse_filename(name: str) -> dict | None:
    """Parse a TreeID filename into its components.

    Args:
        name: Bare filename (no directory), e.g. ``"R02N06_Summer_GT-14-22-5.png"``.

    Returns:
        A dict with keys ``tree_id, row, kind, num, season, mod, time, extra``,
        or ``None`` if the name does not follow the TreeID convention.

    Example:
        >>> parse_filename("R02N06_Summer_GT-14-22-5.png")["tree_id"]
        'R02N06'
    """
    m = FILENAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["row"] = int(d["row"])
    d["tree_id"] = f"R{m['row']}{m['kind']}{m['num']}"
    return d


def discover(root: str | Path) -> tuple[dict[str, TreeRecord], list[str]]:
    """Index every TreeID PNG below ``root``.

    Args:
        root: Directory containing the extracted dataset (searched recursively).

    Returns:
        ``(records, unparsed)`` where ``records`` maps tree id to
        :class:`TreeRecord` and ``unparsed`` lists filenames that did not match
        :data:`FILENAME_RE`.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    records: dict[str, TreeRecord] = {}
    unparsed: list[str] = []
    for p in sorted(root.rglob("*.png")):
        info = parse_filename(p.name)
        if info is None:
            unparsed.append(p.name)
            continue
        rec = records.setdefault(
            info["tree_id"],
            TreeRecord(info["tree_id"], info["row"], info["kind"] == "N"),
        )
        key = f"{info['season']}_{info['mod']}"
        if info["extra"]:
            rec.anomalies.append(f"{p.name}: unexpected suffix '{info['extra']}'")
        if key in rec.files:
            rec.anomalies.append(f"duplicate {key}: {p.name}")
        rec.files[key] = p
    return records, unparsed


def read_rgb(path: str | Path) -> np.ndarray:
    """Read a PNG as an ``(H, W, 3)`` uint8 RGB array.

    Raises:
        IOError: If OpenCV cannot decode the file.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: str | Path) -> np.ndarray:
    """Read a PNG as a single-channel ``(H, W)`` uint8 array.

    Multi-channel files are reduced to their first channel (the TreeID depth and
    mask PNGs store identical values in every channel when they are RGB).

    Raises:
        IOError: If OpenCV cannot decode the file.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img


def load_sample(rec: TreeRecord, season: str = "Summer") -> dict[str, np.ndarray]:
    """Load the RGB, depth and (summer only) mask arrays of one tree.

    Args:
        rec: Tree record from :func:`discover`.
        season: ``"Summer"`` or ``"Winter"``.

    Returns:
        Dict with ``rgb`` (H, W, 3 uint8), ``depth`` (H, W uint8) and, for
        summer, ``gt`` (256, 256 bool).

    Raises:
        KeyError: If a required file is missing for this tree.
        ValueError: If ``season`` is not recognised.
    """
    if season not in ("Summer", "Winter"):
        raise ValueError(f"season must be 'Summer' or 'Winter', got {season!r}")
    out = {
        "rgb": read_rgb(rec.files[f"{season}_RGB"]),
        "depth": read_gray(rec.files[f"{season}_D"]),
    }
    if season == "Summer":
        out["gt"] = read_gray(rec.files["Summer_GT"]) > 127
    return out
