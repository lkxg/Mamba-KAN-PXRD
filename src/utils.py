"""Misc helpers: config loading, seeding, classification metrics."""
from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# =====================================================================
# Config
# =====================================================================

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} must be a YAML mapping (got {type(cfg)})")
    return cfg


# =====================================================================
# Seeding
# =====================================================================

def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# =====================================================================
# Metrics
# =====================================================================

def topk_accuracy(logits: np.ndarray, target: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy. ``logits`` shape (N, C), ``target`` shape (N,)."""
    k = min(k, logits.shape[1])
    top = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float((top == target[:, None]).any(axis=1).mean())


def per_class_accuracy(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """Per-class accuracy.  Classes with zero target samples get NaN."""
    correct = defaultdict(int)
    total = defaultdict(int)
    for p, t in zip(pred, target):
        total[int(t)] += 1
        if int(p) == int(t):
            correct[int(t)] += 1
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        if total[c] > 0:
            out[c] = correct[c] / total[c]
    return out


__all__ = ["load_config", "set_seed", "topk_accuracy", "per_class_accuracy"]
