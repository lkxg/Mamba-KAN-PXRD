"""训练相关的工具函数：损失函数和训练/评估循环。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def configure_backend(cfg: dict, device: torch.device) -> None:
    """Enable optional CUDA backend optimizations from config."""
    perf_cfg = cfg.get("performance", {})
    if device.type != "cuda":
        return

    if perf_cfg.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = perf_cfg.get("cudnn_benchmark", True)

    matmul_precision = perf_cfg.get("matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)


# =====================================================================
# 损失函数
# =====================================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss: -(1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        target_log_p = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = target_log_p.exp()
        focal_weight = (1.0 - target_p).clamp(min=1e-7).pow(self.gamma)
        return (-focal_weight * target_log_p).mean()


class AsymmetricLoss(nn.Module):
    """Softmax ASL with class-weighted positives and shifted negatives."""

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        prob_shift: float = 0.05,
    ) -> None:
        super().__init__()
        if gamma_pos < 0:
            raise ValueError("gamma_pos must be non-negative")
        if gamma_neg < 0:
            raise ValueError("gamma_neg must be non-negative")
        if prob_shift < 0 or prob_shift >= 1:
            raise ValueError("prob_shift must satisfy 0 <= prob_shift < 1")
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.prob_shift = prob_shift
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=probs.dtype)
        pos_probs = (probs * one_hot).sum(dim=-1)
        pos_log_probs = (log_probs * one_hot).sum(dim=-1)
        pos_focal = (1.0 - pos_probs).clamp(min=1e-7).pow(self.gamma_pos)
        pos_loss = -self.alpha[target] * pos_focal * pos_log_probs
        neg_probs = probs * (1.0 - one_hot)
        neg_probs_shifted = (neg_probs - self.prob_shift).clamp(min=0.0)
        neg_focal = neg_probs_shifted.pow(self.gamma_neg)
        neg_log = torch.log(1.0 - neg_probs_shifted + 1e-7)
        neg_loss = -(neg_focal * neg_log).mean(dim=-1)
        return (pos_loss + neg_loss).mean()


def _positive_class_counts(class_counts: np.ndarray | None, *, name: str) -> np.ndarray:
    """Return finite positive class counts for frequency-aware losses."""
    if class_counts is None:
        raise ValueError(f"`class_counts` required for {name}")
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("class_counts must be a 1D array")
    if np.any(counts < 0):
        raise ValueError("class_counts must be non-negative")
    if not np.any(counts > 0):
        raise ValueError("class_counts must contain at least one positive count")
    return np.maximum(counts, 1.0)


def _effective_number_weights(
    class_counts: np.ndarray,
    *,
    beta: float,
    name: str,
) -> torch.Tensor:
    """Build effective-number class weights normalized over present classes."""
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"{name} beta must satisfy 0 <= beta < 1")
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("class_counts must be a 1D array")
    if np.any(counts < 0):
        raise ValueError("class_counts must be non-negative")
    weights = np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    if not present.any():
        raise ValueError("class_counts must contain at least one positive count")
    effective_num = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / effective_num
    weights[present] = weights[present] / weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss with optional DRW weights."""

    def __init__(
        self,
        class_counts: np.ndarray,
        *,
        max_m: float = 0.5,
        scale: float = 30.0,
        label_smoothing: float = 0.0,
        class_weight_beta: float = 0.9999,
        use_class_weights: bool = False,
    ) -> None:
        super().__init__()
        if max_m <= 0:
            raise ValueError("ldam_max_m must be positive")
        if scale <= 0:
            raise ValueError("ldam_scale must be positive")
        counts = _positive_class_counts(class_counts, name="ldam")
        margins = 1.0 / np.sqrt(np.sqrt(counts))
        margins = margins * (float(max_m) / margins.max())
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        self.register_buffer(
            "drw_weights",
            _effective_number_weights(
                class_counts,
                beta=class_weight_beta,
                name="ldam_drw",
            ),
        )
        self.scale = float(scale)
        self.label_smoothing = float(label_smoothing)
        self.use_class_weights = bool(use_class_weights)

    def set_class_weights(self, enabled: bool) -> None:
        self.use_class_weights = bool(enabled)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(dtype=logits.dtype, device=logits.device)
        target_margins = margins.gather(0, target)
        adjusted = logits.clone()
        adjusted.scatter_add_(1, target.unsqueeze(1), -target_margins.unsqueeze(1))
        weight = (
            self.drw_weights.to(dtype=logits.dtype, device=logits.device)
            if self.use_class_weights
            else None
        )
        return F.cross_entropy(
            self.scale * adjusted,
            target,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


def build_loss_from_config(
    loss_cfg: dict,
    *,
    class_counts: np.ndarray | None = None,
) -> nn.Module:
    """Build one of the retained losses from its YAML mapping."""
    name = str(loss_cfg["name"]).lower()
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_ce":
        if class_counts is None:
            raise ValueError("`class_counts` required for weighted_ce")
        counts = np.asarray(class_counts, dtype=np.float64)
        weights = np.zeros_like(counts, dtype=np.float64)
        present = counts > 0
        if not present.any():
            raise ValueError("class_counts must contain at least one positive count")
        class_weight_power = float(loss_cfg.get("class_weight_power", 0.5))
        if class_weight_power < 0:
            raise ValueError("class_weight_power must be non-negative")
        weights[present] = 1.0 / np.power(counts[present], class_weight_power)
        weights[present] = weights[present] / weights[present].mean()
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=label_smoothing)
    if name == "class_balanced_ce":
        if class_counts is None:
            raise ValueError("`class_counts` required for class_balanced_ce")
        w_tensor = _effective_number_weights(
            class_counts,
            beta=float(loss_cfg.get("class_weight_beta", 0.99)),
            name="class_balanced_ce",
        )
        return nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=label_smoothing)
    if name == "ldam":
        if class_counts is None:
            raise ValueError("`class_counts` required for ldam")
        return LDAMLoss(
            class_counts,
            max_m=float(loss_cfg.get("ldam_max_m", 0.5)),
            scale=float(loss_cfg.get("ldam_scale", 30.0)),
            label_smoothing=label_smoothing,
            class_weight_beta=float(loss_cfg.get("ldam_class_weight_beta", 0.9999)),
            use_class_weights=bool(loss_cfg.get("ldam_use_class_weights", False)),
        )
    if name == "focal":
        return FocalLoss(gamma=float(loss_cfg.get("focal_gamma", 2.0)))
    if name == "asl":
        if class_counts is None:
            raise ValueError("`class_counts` required for asl")
        alpha = _effective_number_weights(
            class_counts,
            beta=float(loss_cfg.get("class_weight_beta", 0.99)),
            name="asl",
        )
        return AsymmetricLoss(
            alpha=alpha,
            gamma_pos=float(loss_cfg.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(loss_cfg.get("asl_gamma_neg", 4.0)),
            prob_shift=float(loss_cfg.get("asl_prob_shift", 0.05)),
        )
    raise ValueError(f"unknown loss: {name!r}")


# =====================================================================
# 训练和评估循环
# =====================================================================

@dataclass
class StepStats:
    """单步或单 epoch 的训练统计信息。"""
    loss: float          # 平均损失值
    n: int               # 处理的样本总数
    correct_top1: int    # Top-1 正确预测数
    correct_top5: int    # Top-5 正确预测数
    class_correct_top1: np.ndarray | None = None
    class_total: np.ndarray | None = None
    contrastive_loss: float = 0.0
    rare_correct_top1: int | None = None
    rare_total: int | None = None

    @property
    def acc1(self) -> float:
        """Top-1 准确率。"""
        return self.correct_top1 / max(self.n, 1)

    @property
    def acc5(self) -> float:
        """Top-5 准确率。"""
        return self.correct_top5 / max(self.n, 1)

    @property
    def macro_acc1(self) -> float:
        """按类别平均的 Top-1 准确率，用于观察长尾类别表现。"""
        if self.class_correct_top1 is None or self.class_total is None:
            return float("nan")
        present = self.class_total > 0
        if not present.any():
            return float("nan")
        per_class = self.class_correct_top1[present] / self.class_total[present]
        return float(per_class.mean())

    @property
    def rare_acc1(self) -> float:
        """Top-1 accuracy on classes marked rare by the training split."""
        if self.rare_correct_top1 is None or self.rare_total is None:
            return float("nan")
        return self.rare_correct_top1 / max(self.rare_total, 1)


def _topk_hits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Top-1/5 hit masks from one topk call."""
    values, indices = logits.topk(min(5, logits.shape[1]), dim=1)
    hits = (indices == target.unsqueeze(1)) & torch.isfinite(values)
    return hits[:, 0], hits.any(dim=1)


def primary_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the primary classification logits from a tensor or model dict."""
    if isinstance(output, dict):
        return output["logits"]
    return output


def supervised_contrastive_loss(
    embedding: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss over same-label positives in a batch."""
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    if embedding.ndim != 2:
        raise ValueError(f"expected 2D contrastive embedding, got {tuple(embedding.shape)}")
    if embedding.shape[0] != target.shape[0]:
        raise ValueError("embedding and target batch sizes must match")

    features = F.normalize(embedding.float(), dim=-1)
    logits = features @ features.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = target.shape[0]
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=target.device)
    positive_mask = target.view(-1, 1).eq(target.view(1, -1)) & ~self_mask
    contrast_mask = ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0

    logits = logits.masked_fill(~contrast_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = (
        log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return -(positive_log_prob * valid_anchor).sum() / valid_anchor.sum().clamp_min(1)


def loss_with_contrastive(
    output: torch.Tensor | dict[str, torch.Tensor],
    target: torch.Tensor,
    loss_fn: nn.Module,
    *,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    contrastive_embedding_key: str = "embedding",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute classification loss plus optional supervised contrastive loss."""
    logits = primary_logits(output)
    main_loss = loss_fn(logits, target)
    contrastive_loss = logits.sum() * 0.0
    if contrastive_weight > 0:
        if not isinstance(output, dict):
            raise TypeError("contrastive loss requires a model output dict with embeddings")
        if contrastive_embedding_key not in output:
            raise KeyError(
                f"contrastive embedding {contrastive_embedding_key!r} missing from model output"
            )
        contrastive_loss = supervised_contrastive_loss(
            output[contrastive_embedding_key],
            target,
            temperature=contrastive_temperature,
        )
    total_loss = main_loss + float(contrastive_weight) * contrastive_loss
    return total_loss, logits, contrastive_loss.detach()


def supervised_contrastive_config(
    loss_cfg: dict,
    *,
    epoch: int,
) -> tuple[float, float, str]:
    """Return scheduled supervised-contrastive weight, temperature, and key."""
    cfg = loss_cfg.get("supervised_contrastive", {}) or {}
    temperature = float(cfg.get("temperature", 0.1))
    embedding_key = str(cfg.get("embedding_key", "embedding"))
    if not bool(cfg.get("enabled", False)):
        return 0.0, temperature, embedding_key

    weight = float(cfg.get("weight", 0.05))
    start_epoch = int(cfg.get("start_epoch", 1))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    if epoch < start_epoch:
        weight = 0.0
    elif warmup_epochs > 0:
        weight *= min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return weight, temperature, embedding_key


def rare_classes_from_counts(
    class_counts: np.ndarray | None,
    *,
    max_count: int,
    min_count: int = 1,
) -> set[int]:
    """Return 0-based class ids treated as rare under the training split."""
    if class_counts is None:
        return set()
    max_count = int(max_count)
    min_count = int(min_count)
    return {
        int(i)
        for i, c in enumerate(class_counts)
        if int(c) >= min_count and int(c) <= max_count
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    grad_clip: float | None = 1.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    contrastive_embedding_key: str = "embedding",
    rare_classes: set[int] | None = None,
) -> StepStats:
    """训练一个 epoch。

    参数:
        model: PyTorch 模型
        loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 计算设备（CPU 或 CUDA）
        grad_clip: 梯度裁剪阈值，None 表示不裁剪

    返回:
        StepStats：包含当前 epoch 的训练统计信息
    """
    model.train()
    total_loss = torch.zeros((), device=device)
    total_contrastive_loss = torch.zeros((), device=device)
    correct1 = torch.zeros((), dtype=torch.int64, device=device)
    correct5 = torch.zeros((), dtype=torch.int64, device=device)
    rare_correct1 = torch.zeros((), dtype=torch.int64, device=device)
    rare_total = torch.zeros((), dtype=torch.int64, device=device)
    n = 0
    rare_lookup: torch.Tensor | None = None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            output = model(x)
            loss, logits, contrastive_loss = loss_with_contrastive(
                output,
                y,
                loss_fn,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
                contrastive_embedding_key=contrastive_embedding_key,
            )

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bsz = y.size(0)
        total_loss += loss.detach().float() * bsz
        total_contrastive_loss += contrastive_loss.float() * bsz
        n += bsz
        top1_hits, top5_hits = _topk_hits(logits.detach(), y)
        correct1 += top1_hits.sum()
        correct5 += top5_hits.sum()
        if rare_classes:
            if rare_lookup is None:
                rare_lookup = torch.zeros(
                    logits.shape[1],
                    dtype=torch.bool,
                    device=y.device,
                )
                rare_lookup[list(rare_classes)] = True
            rare_mask = rare_lookup[y]
            rare_total += rare_mask.sum()
            rare_correct1 += top1_hits[rare_mask].sum()

    return StepStats(
        loss=float(total_loss.item()) / max(n, 1),
        n=n,
        correct_top1=int(correct1.item()),
        correct_top5=int(correct5.item()),
        contrastive_loss=float(total_contrastive_loss.item()) / max(n, 1),
        rare_correct_top1=int(rare_correct1.item()) if rare_classes else None,
        rare_total=int(rare_total.item()) if rare_classes else None,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    rare_classes: set[int] | None = None,
) -> StepStats:
    """在验证集或测试集上评估模型。

    参数:
        model: PyTorch 模型
        loader: 评估数据加载器
        loss_fn: 损失函数
        device: 计算设备
    返回:
        StepStats：包含评估统计信息
    """
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct1 = torch.zeros((), dtype=torch.int64, device=device)
    correct5 = torch.zeros((), dtype=torch.int64, device=device)
    rare_correct1 = torch.zeros((), dtype=torch.int64, device=device)
    rare_total = torch.zeros((), dtype=torch.int64, device=device)
    n = 0
    rare_lookup: torch.Tensor | None = None
    class_correct1: torch.Tensor | None = None
    class_total: torch.Tensor | None = None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            output = model(x)
            logits = primary_logits(output)
            loss = loss_fn(logits, y)

        bsz = y.size(0)
        total_loss += loss.float() * bsz
        n += bsz
        top1_hits, top5_hits = _topk_hits(logits, y)
        correct1 += top1_hits.sum()
        correct5 += top5_hits.sum()
        num_classes = logits.shape[1]
        if class_total is None:
            class_total = torch.zeros(num_classes, dtype=torch.int64, device=device)
            class_correct1 = torch.zeros(num_classes, dtype=torch.int64, device=device)
        class_total += torch.bincount(y, minlength=num_classes)
        class_correct1 += torch.bincount(y[top1_hits], minlength=num_classes)
        if rare_classes:
            if rare_lookup is None:
                rare_lookup = torch.zeros(
                    num_classes,
                    dtype=torch.bool,
                    device=y.device,
                )
                rare_lookup[list(rare_classes)] = True
            rare_mask = rare_lookup[y]
            rare_total += rare_mask.sum()
            rare_correct1 += top1_hits[rare_mask].sum()

    return StepStats(
        loss=float(total_loss.item()) / max(n, 1),
        n=n,
        correct_top1=int(correct1.item()),
        correct_top5=int(correct5.item()),
        class_correct_top1=(
            class_correct1.cpu().numpy() if class_correct1 is not None else None
        ),
        class_total=class_total.cpu().numpy() if class_total is not None else None,
        rare_correct_top1=int(rare_correct1.item()) if rare_classes else None,
        rare_total=int(rare_total.item()) if rare_classes else None,
    )
