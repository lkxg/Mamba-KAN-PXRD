"""Training-time utilities: losses + train/val loops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =====================================================================
# Losses
# =====================================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor(1.0))
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        target_log_p = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = log_p.exp().gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_p).clamp(min=1e-7).pow(self.gamma)
        alpha_y = self.alpha[target] if self.alpha.ndim > 0 else self.alpha
        loss = -alpha_y * focal_weight * target_log_p
        if self.ignore_index >= 0:
            loss = loss[target != self.ignore_index]
        return loss.mean()


def build_loss(
    name: Literal["ce", "weighted_ce", "focal"] = "ce",
    *,
    class_counts: np.ndarray | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Construct a loss function by name.

    - ``"ce"``           standard cross-entropy
    - ``"weighted_ce"``  CE with per-class weights = 1/sqrt(count)
    - ``"focal"``        focal loss (gamma controls down-weighting)
    """
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_ce":
        if class_counts is None:
            raise ValueError("`class_counts` required for weighted_ce")
        counts = np.asarray(class_counts, dtype=np.float64)
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        weights = weights / weights.mean()
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=gamma)
    raise ValueError(f"unknown loss: {name!r}")


# =====================================================================
# Train / eval loops
# =====================================================================

@dataclass
class StepStats:
    loss: float
    n: int
    correct_top1: int
    correct_top5: int

    @property
    def acc1(self) -> float:
        return self.correct_top1 / max(self.n, 1)

    @property
    def acc5(self) -> float:
        return self.correct_top5 / max(self.n, 1)


def _accuracy_topk(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    k = min(k, logits.shape[1])
    _, top = logits.topk(k, dim=1)
    return int((top == target.unsqueeze(1)).any(dim=1).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
    log_every: int = 100,
    progress_callback=None,
) -> StepStats:
    model.train()
    total_loss = 0.0
    n = correct1 = correct5 = 0
    use_amp = scaler is not None

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        n += bsz
        correct1 += _accuracy_topk(logits.detach(), y, k=1)
        correct5 += _accuracy_topk(logits.detach(), y, k=5)

        if progress_callback is not None and (step % log_every == 0):
            progress_callback(step, total_loss / max(n, 1), correct1 / max(n, 1))

    return StepStats(loss=total_loss / max(n, 1), n=n,
                     correct_top1=correct1, correct_top5=correct5)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = True,
) -> StepStats:
    model.eval()
    total_loss = 0.0
    n = correct1 = correct5 = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        n += bsz
        correct1 += _accuracy_topk(logits, y, k=1)
        correct5 += _accuracy_topk(logits, y, k=5)

    return StepStats(loss=total_loss / max(n, 1), n=n,
                     correct_top1=correct1, correct_top5=correct5)


__all__ = [
    "FocalLoss", "build_loss",
    "StepStats", "train_one_epoch", "evaluate",
]
