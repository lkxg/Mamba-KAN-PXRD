"""训练相关的工具函数：损失函数和训练/评估循环。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


def amp_dtype_from_config(
    name: str | None,
    device: torch.device,
) -> torch.dtype | None:
    """Parse AMP dtype from config; return None off CUDA or for fp32."""
    if device.type != "cuda" or name is None:
        return None

    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "none"}:
        return None
    raise ValueError(f"unknown amp dtype: {name!r}")


# =====================================================================
# 损失函数
# =====================================================================

class FocalLoss(nn.Module):
    """多类别 Focal Loss（Lin et al., 2017）。

    Focal Loss 通过降低易分类样本的权重，使模型更关注难分类的样本。
    适用于类别不平衡的分类问题。

    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    其中 p_t 是真实类别的预测概率，γ 是聚焦参数，α_t 是类别权重。

    参数:
        gamma: 聚焦参数，默认 2.0。γ 越大，模型越关注难分类样本
        alpha: 类别权重张量，可选
        ignore_index: 忽略的类别索引，默认 -100
    """

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
        """计算 Focal Loss。

        参数:
            logits: 模型输出的未归一化 logits，形状 (batch_size, num_classes)
            target: 真实类别索引，形状 (batch_size,)

        返回:
            标量损失值
        """
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            logits = logits[valid_mask]
            target = target[valid_mask]
            if target.numel() == 0:
                return logits.sum() * 0.0

        # 计算 log softmax 和 softmax
        log_p = F.log_softmax(logits, dim=-1)
        target_log_p = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = log_p.exp().gather(1, target.unsqueeze(1)).squeeze(1)

        # 计算聚焦权重：(1 - p_t)^γ
        focal_weight = (1.0 - target_p).clamp(min=1e-7).pow(self.gamma)

        # 应用类别权重
        alpha_y = self.alpha[target] if self.alpha.ndim > 0 else self.alpha

        # 计算最终损失
        loss = -alpha_y * focal_weight * target_log_p
        return loss.mean()


def _positive_class_counts(class_counts: np.ndarray | None, *, name: str) -> np.ndarray:
    """Return finite positive class counts for prior/logit adjustment losses."""
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


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax loss using training-set class frequencies."""

    def __init__(
        self,
        class_counts: np.ndarray,
        *,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        counts = _positive_class_counts(class_counts, name="balanced_softmax")
        self.register_buffer(
            "log_counts",
            torch.tensor(np.log(counts), dtype=torch.float32),
        )
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.log_counts.to(dtype=logits.dtype)
        return F.cross_entropy(
            adjusted,
            target,
            label_smoothing=self.label_smoothing,
        )


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted cross entropy with configurable prior strength."""

    def __init__(
        self,
        class_counts: np.ndarray,
        *,
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if tau < 0:
            raise ValueError("logit_adjustment_tau must be non-negative")
        counts = _positive_class_counts(class_counts, name="logit_adjustment")
        priors = counts / counts.sum()
        self.register_buffer(
            "log_priors",
            torch.tensor(np.log(priors), dtype=torch.float32),
        )
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_priors.to(dtype=logits.dtype)
        return F.cross_entropy(
            adjusted,
            target,
            label_smoothing=self.label_smoothing,
        )


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


def build_loss(
    name: Literal[
        "ce",
        "label_smoothing",
        "weighted_ce",
        "class_balanced_ce",
        "balanced_softmax",
        "logit_adjustment",
        "ldam",
        "focal",
    ] = "ce",
    *,
    class_counts: np.ndarray | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    class_weight_power: float = 0.5,
    class_weight_beta: float = 0.99,
    logit_adjustment_tau: float = 1.0,
    ldam_max_m: float = 0.5,
    ldam_scale: float = 30.0,
    ldam_class_weight_beta: float = 0.9999,
    ldam_use_class_weights: bool = False,
) -> nn.Module:
    """根据名称构建损失函数。

    参数:
        name: 损失函数名称
              - "ce": 标准交叉熵损失
              - "label_smoothing": 带标签平滑的交叉熵损失
              - "weighted_ce": 加权交叉熵，权重为 1/sqrt(类别样本数)
              - "class_balanced_ce": 有效样本数加权交叉熵
              - "balanced_softmax": 按训练类别频数调整 softmax 分母
              - "logit_adjustment": 按训练类别先验调整 logits
              - "ldam": 带类别频数 margin 的 LDAM loss，可配合 DRW
              - "focal": Focal Loss
        class_counts: 各类别的样本数量，用于计算加权交叉熵的权重
        gamma: Focal Loss 的聚焦参数
        label_smoothing: 标签平滑参数，用于防止过拟合
        class_weight_power: weighted_ce 的类别频数指数
        class_weight_beta: class_balanced_ce 的有效样本数 beta
        logit_adjustment_tau: logit_adjustment 的类别先验强度

    返回:
        PyTorch 损失函数模块
    """
    name = name.lower()
    if name in {"ce", "label_smoothing"}:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_ce":
        if class_counts is None:
            raise ValueError("`class_counts` required for weighted_ce")
        counts = np.asarray(class_counts, dtype=np.float64)
        weights = np.zeros_like(counts, dtype=np.float64)
        present = counts > 0
        if not present.any():
            raise ValueError("class_counts must contain at least one positive count")
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
            beta=class_weight_beta,
            name="class_balanced_ce",
        )
        return nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=label_smoothing)
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(
            class_counts,
            label_smoothing=label_smoothing,
        )
    if name == "logit_adjustment":
        return LogitAdjustedLoss(
            class_counts,
            tau=logit_adjustment_tau,
            label_smoothing=label_smoothing,
        )
    if name == "ldam":
        if class_counts is None:
            raise ValueError("`class_counts` required for ldam")
        return LDAMLoss(
            class_counts,
            max_m=ldam_max_m,
            scale=ldam_scale,
            label_smoothing=label_smoothing,
            class_weight_beta=ldam_class_weight_beta,
            use_class_weights=ldam_use_class_weights,
        )
    if name == "focal":
        return FocalLoss(gamma=gamma)
    raise ValueError(f"unknown loss: {name!r}")


def build_loss_from_config(
    loss_cfg: dict,
    *,
    class_counts: np.ndarray | None = None,
) -> nn.Module:
    """Build a loss module from a YAML loss config mapping."""
    name = str(loss_cfg["name"]).lower()
    needs_counts = {
        "weighted_ce",
        "class_balanced_ce",
        "balanced_softmax",
        "logit_adjustment",
        "ldam",
    }
    return build_loss(
        name,
        class_counts=class_counts if name in needs_counts else None,
        gamma=loss_cfg.get("focal_gamma", 2.0),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        class_weight_power=loss_cfg.get("class_weight_power", 0.5),
        class_weight_beta=loss_cfg.get("class_weight_beta", 0.99),
        logit_adjustment_tau=loss_cfg.get("logit_adjustment_tau", 1.0),
        ldam_max_m=loss_cfg.get("ldam_max_m", 0.5),
        ldam_scale=loss_cfg.get("ldam_scale", 30.0),
        ldam_class_weight_beta=loss_cfg.get("ldam_class_weight_beta", 0.9999),
        ldam_use_class_weights=loss_cfg.get("ldam_use_class_weights", False),
    )


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
    correct_top10: int = 0  # Top-10 正确预测数
    class_correct_top1: np.ndarray | None = None
    class_total: np.ndarray | None = None
    aux_loss: float = 0.0
    gate_mean: float | None = None
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
    def acc10(self) -> float:
        """Top-10 准确率。"""
        return self.correct_top10 / max(self.n, 1)

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


def _accuracy_topk(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    """计算 Top-k 准确率。

    参数:
        logits: 模型输出的未归一化 logits，形状 (batch_size, num_classes)
        target: 真实类别索引，形状 (batch_size,)
        k: Top-k 中的 k 值

    返回:
        Top-k 正确的样本数
    """
    k = min(k, logits.shape[1])
    # 取 logits 最大的 k 个类别的索引
    _, top = logits.topk(k, dim=1)
    # 检查真实类别是否在这 k 个中
    return int((top == target.unsqueeze(1)).any(dim=1).sum().item())


def primary_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the primary classification logits from a tensor or model dict."""
    if isinstance(output, dict):
        return output["logits"]
    return output


def loss_with_auxiliary(
    output: torch.Tensor | dict[str, torch.Tensor],
    target: torch.Tensor,
    loss_fn: nn.Module,
    *,
    aux_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute main loss plus optional branch auxiliary losses."""
    logits = primary_logits(output)
    main_loss = loss_fn(logits, target)
    aux_loss = logits.sum() * 0.0
    if isinstance(output, dict):
        weights = aux_weights or {}
        for key in ("sa_logits", "wa_logits"):
            weight = float(weights.get(key, 0.0))
            if weight > 0 and key in output:
                aux_loss = aux_loss + weight * loss_fn(output[key], target)
    return main_loss + aux_loss, logits, aux_loss.detach()


def output_gate_mean(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Return batch gate means when a gated dual-range model exposes them."""
    if isinstance(output, dict) and "gate_mean" in output:
        return output["gate_mean"].detach()
    return None


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


def aux_loss_weights_from_model(model: nn.Module) -> dict[str, float]:
    """Read optional auxiliary loss weights from a raw or compiled model."""
    candidate = getattr(model, "_orig_mod", model)
    weights = getattr(candidate, "aux_loss_weights", None)
    if not weights:
        return {}
    return {str(k): float(v) for k, v in dict(weights).items() if float(v) > 0}


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
    use_amp: bool | None = None,
    amp_dtype: torch.dtype | None = None,
    aux_loss_weights: dict[str, float] | None = None,
    rare_classes: set[int] | None = None,
    freeze_batch_norm: bool = False,
) -> StepStats:
    """训练一个 epoch。

    参数:
        model: PyTorch 模型
        loader: 训练数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 计算设备（CPU 或 CUDA）
        scaler: 混合精度训练的 GradScaler，None 表示不使用 AMP
        grad_clip: 梯度裁剪阈值，None 表示不裁剪
        log_every: 日志打印频率（每多少批次打印一次）
        progress_callback: 进度回调函数，签名为 (step, loss, acc1)

    返回:
        StepStats：包含当前 epoch 的训练统计信息
    """
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    n = correct1 = correct5 = correct10 = 0
    rare_correct1 = rare_total = 0
    gate_sum = 0.0
    gate_count = 0
    if use_amp is None:
        use_amp = scaler is not None
    use_scaler = scaler is not None and scaler.is_enabled()

    for step, (x, y) in enumerate(loader):
        # 将数据移动到设备上
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)

        # 前向传播（可选混合精度）
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            output = model(x)
            loss, logits, aux_loss = loss_with_auxiliary(
                output,
                y,
                loss_fn,
                aux_weights=aux_loss_weights,
            )

        # 反向传播
        if use_scaler:
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

        # 统计信息
        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_aux_loss += float(aux_loss.item()) * bsz
        n += bsz
        correct1 += _accuracy_topk(logits.detach(), y, k=1)
        correct5 += _accuracy_topk(logits.detach(), y, k=5)
        correct10 += _accuracy_topk(logits.detach(), y, k=10)
        if rare_classes:
            pred = logits.detach().argmax(dim=1)
            rare_mask = torch.zeros_like(y, dtype=torch.bool)
            for cls in rare_classes:
                rare_mask |= y == int(cls)
            if rare_mask.any():
                rare_total += int(rare_mask.sum().item())
                rare_correct1 += int((pred[rare_mask] == y[rare_mask]).sum().item())

        gate_mean = output_gate_mean(output)
        if gate_mean is not None:
            gate_sum += float(gate_mean.float().sum().item())
            gate_count += int(gate_mean.numel())

        # 进度回调
        if progress_callback is not None and (step % log_every == 0):
            progress_callback(step, total_loss / max(n, 1), correct1 / max(n, 1))

    return StepStats(
        loss=total_loss / max(n, 1),
        n=n,
        correct_top1=correct1,
        correct_top5=correct5,
        correct_top10=correct10,
        aux_loss=total_aux_loss / max(n, 1),
        gate_mean=(gate_sum / gate_count) if gate_count else None,
        rare_correct_top1=rare_correct1 if rare_classes else None,
        rare_total=rare_total if rare_classes else None,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    aux_loss_weights: dict[str, float] | None = None,
    rare_classes: set[int] | None = None,
) -> StepStats:
    """在验证集或测试集上评估模型。

    参数:
        model: PyTorch 模型
        loader: 评估数据加载器
        loss_fn: 损失函数
        device: 计算设备
        use_amp: 是否使用混合精度

    返回:
        StepStats：包含评估统计信息
    """
    model.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    n = correct1 = correct5 = correct10 = 0
    rare_correct1 = rare_total = 0
    gate_sum = 0.0
    gate_count = 0
    class_correct1: np.ndarray | None = None
    class_total: np.ndarray | None = None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            output = model(x)
            loss, logits, aux_loss = loss_with_auxiliary(
                output,
                y,
                loss_fn,
                aux_weights=aux_loss_weights,
            )

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_aux_loss += float(aux_loss.item()) * bsz
        n += bsz
        correct1 += _accuracy_topk(logits, y, k=1)
        correct5 += _accuracy_topk(logits, y, k=5)
        correct10 += _accuracy_topk(logits, y, k=10)

        pred = logits.argmax(dim=1)
        num_classes = logits.shape[1]
        if class_total is None:
            class_total = np.zeros(num_classes, dtype=np.int64)
            class_correct1 = np.zeros(num_classes, dtype=np.int64)
        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        class_total += np.bincount(y_cpu.numpy(), minlength=num_classes)
        correct_mask = pred_cpu == y_cpu
        class_correct1 += np.bincount(
            y_cpu[correct_mask].numpy(),
            minlength=num_classes,
        )
        if rare_classes:
            rare_mask = np.isin(y_cpu.numpy(), list(rare_classes))
            rare_total += int(rare_mask.sum())
            if rare_mask.any():
                rare_correct1 += int((pred_cpu.numpy()[rare_mask] == y_cpu.numpy()[rare_mask]).sum())

        gate_mean = output_gate_mean(output)
        if gate_mean is not None:
            gate_sum += float(gate_mean.float().sum().item())
            gate_count += int(gate_mean.numel())

    return StepStats(
        loss=total_loss / max(n, 1),
        n=n,
        correct_top1=correct1,
        correct_top5=correct5,
        correct_top10=correct10,
        class_correct_top1=class_correct1,
        class_total=class_total,
        aux_loss=total_aux_loss / max(n, 1),
        gate_mean=(gate_sum / gate_count) if gate_count else None,
        rare_correct_top1=rare_correct1 if rare_classes else None,
        rare_total=rare_total if rare_classes else None,
    )


__all__ = [
    "configure_backend", "amp_dtype_from_config",
    "FocalLoss", "LDAMLoss", "build_loss", "build_loss_from_config",
    "StepStats", "primary_logits", "loss_with_auxiliary",
    "rare_classes_from_counts", "aux_loss_weights_from_model",
    "train_one_epoch", "evaluate",
]
