"""Training and inference loop for :class:`treeid.model.UNet`.

Design decisions (each is tested or reported):

* **Loss**: binary cross-entropy plus soft-Dice. BCE alone is dominated by the
  ~95% background pixels and converges to "predict nothing"; the Dice term is
  insensitive to class imbalance.
* **Positive weight**: by default (v1) the BCE ``pos_weight`` equals the
  background/branch pixel ratio of the training set, computed from training data
  only. Experiments showed this over-predicts (masks ~3x too wide), so v2 sets it
  to 1 and relies on the Dice term for class balance.
* **Early stopping**: training stops when the validation centreline-F1 has not
  improved for ``patience`` epochs; the best checkpoint is restored. The
  validation rows are therefore *used for model selection* and are distinct from
  the held-out test rows, which are touched exactly once.
* **Determinism**: seeds for NumPy and PyTorch are set, and ``torch`` is told to
  use deterministic algorithms where available, so results are repeatable up to
  floating-point summation order.
* **Augmentation**: horizontal flips only. Vertical flips are excluded because
  gravity makes trees vertically asymmetric (trunk at the bottom).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .metrics import tolerant_precision_recall
from .model import UNet


@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`fit`."""

    base: int = 16
    depth: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 60
    patience: int = 10
    seed: int = 0
    threshold: float = 0.5
    flip: bool = True
    threads: int = 2
    #: BCE weight on branch pixels. ``None`` = background/branch pixel ratio of
    #: the training set (v1, ~12); 1.0 = unweighted (v2, see report Section 6).
    pos_weight: float | None = None


@dataclass
class TrainResult:
    """Outputs of :func:`fit`."""

    model: UNet
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    train_seconds: float = 0.0


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """``1 - soft Dice`` averaged over the batch; differentiable everywhere."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Branch probability for a stack of inputs.

    Args:
        model: Trained network.
        x: float32 array (N, C, H, W).

    Returns:
        float32 array (N, H, W) of probabilities.
    """
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i: i + batch_size])
        out.append(torch.sigmoid(model(xb))[:, 0].numpy())
    return np.concatenate(out, axis=0)


def _val_score(model, x, y, thr) -> float:
    prob = predict_proba(model, x)
    return float(np.mean([tolerant_precision_recall(p > thr, g)[2] for p, g in zip(prob, y)]))


def fit(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
        cfg: TrainConfig = TrainConfig(), log=print) -> TrainResult:
    """Train a U-Net with early stopping on the validation centreline-F1.

    Args:
        x_tr, x_va: float32 inputs (N, C, H, W).
        y_tr, y_va: bool masks (N, H, W).
        cfg: Hyper-parameters.
        log: Callable for progress messages (``None`` to silence).

    Returns:
        :class:`TrainResult` holding the best model and per-epoch history.

    Raises:
        ValueError: On inconsistent array shapes or an empty training set.
    """
    if len(x_tr) == 0:
        raise ValueError("empty training set")
    if x_tr.shape[0] != y_tr.shape[0] or x_tr.shape[2:] != y_tr.shape[1:]:
        raise ValueError(f"x {x_tr.shape} and y {y_tr.shape} do not match")
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    model = UNet(x_tr.shape[1], cfg.base, cfg.depth)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.pos_weight is None:
        pos_frac = float(y_tr.mean())
        pos_weight = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-6))
    else:
        pos_weight = torch.tensor(float(cfg.pos_weight))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt = torch.from_numpy(x_tr)
    yt = torch.from_numpy(y_tr[:, None].astype(np.float32))
    rng = np.random.default_rng(cfg.seed)
    res = TrainResult(model=model)
    best, best_state, since = -1.0, None, 0
    t0 = time.perf_counter()
    for ep in range(cfg.max_epochs):
        model.train()
        idx = rng.permutation(len(xt))
        tot = 0.0
        for i in range(0, len(idx), cfg.batch_size):
            b = idx[i: i + cfg.batch_size]
            xb, yb = xt[b], yt[b]
            if cfg.flip:
                f = torch.from_numpy(rng.random(len(b)) < 0.5)
                xb = torch.where(f[:, None, None, None], xb.flip(-1), xb)
                yb = torch.where(f[:, None, None, None], yb.flip(-1), yb)
            logits = model(xb)
            loss = bce(logits, yb) + soft_dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        v = _val_score(model, x_va, y_va, cfg.threshold)
        res.history.append({"epoch": ep, "train_loss": tot / len(idx), "val_cl_f1": v,
                            "seconds": time.perf_counter() - t0})
        if log:
            log(f"epoch {ep:3d} loss {tot / len(idx):.4f} val clF1 {v:.4f}")
        if v > best + 1e-4:
            best, best_state, since, res.best_epoch = v, copy.deepcopy(model.state_dict()), 0, ep
        else:
            since += 1
            if since >= cfg.patience:
                break
    res.train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return res
