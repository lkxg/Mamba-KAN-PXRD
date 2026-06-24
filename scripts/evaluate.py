"""在测试集上评估已训练模型的性能。

使用方法:
    python3 scripts/evaluate.py --checkpoint checkpoints/baseline_mlp/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "matplotlib"))
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PXRDDataset, class_counts_for_rows, labels_for_rows, load_splits
from src.models import (
    BiGRUPatchClassifier,
    ConvNeXt1D,
    DualPlaneMambaClassifier,
    DualRangePXRDClassifier,
    MLPClassifier,
    PatchTSTClassifier,
    ResNet1D,
)
from src.training import (
    amp_dtype_from_config,
    aux_loss_weights_from_model,
    build_loss_from_config,
    configure_backend,
    apply_hierarchical_mask,
    hierarchical_config,
    loss_with_auxiliary,
    output_gate_mean,
    rare_classes_from_counts,
    space_group_to_crystal_system,
    StepStats,
)
from src.utils import set_seed


CRYSTAL_SYSTEM_LABELS = [
    "Triclinic",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Hexagonal",
    "Cubic",
]


def build_model(cfg: dict, *, in_dim: int, num_classes: int) -> torch.nn.Module:
    """根据配置创建模型实例。

    参数:
        cfg: 包含模型配置的字典
        in_dim: 输入特征维度
        num_classes: 分类类别数

    返回:
        PyTorch 模型实例
    """
    name = cfg["model"]["name"].lower()
    if name == "mlp":
        return MLPClassifier(in_dim=in_dim, num_classes=num_classes,
                             **cfg["model"].get("mlp", {}))
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes, **cfg["model"].get("resnet1d", {}))
    if name == "convnext1d":
        return ConvNeXt1D(num_classes=num_classes, **cfg["model"].get("convnext1d", {}))
    if name == "dual_range":
        return DualRangePXRDClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            **cfg["model"].get("dual_range", {}),
        )
    if name == "dual_plane_mamba":
        return DualPlaneMambaClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            **cfg["model"].get("dual_plane_mamba", {}),
        )
    if name == "bigru_patch":
        return BiGRUPatchClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            **cfg["model"].get("bigru_patch", {}),
        )
    if name == "patchtst":
        return PatchTSTClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            **cfg["model"].get("patchtst", {}),
        )
    raise ValueError(f"unknown model: {name!r}")


def load_model_state_compatible(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load checkpoints across the no-op mean-pooling LayerNorm cleanup."""
    result = model.load_state_dict(state_dict, strict=False)
    model_keys = set(model.state_dict())

    def is_legacy_unexpected_mean_pool_norm(key: str) -> bool:
        is_pool_norm = (
            key.endswith(".pool.norm.weight")
            or key.endswith(".pool.norm.bias")
        )
        return is_pool_norm and key not in model_keys

    missing = list(result.missing_keys)
    unexpected = [
        key for key in result.unexpected_keys
        if not is_legacy_unexpected_mean_pool_norm(key)
    ]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"Missing key(s): {missing}")
        if unexpected:
            details.append(f"Unexpected key(s): {unexpected}")
        raise RuntimeError("Error(s) in loading state_dict: " + "; ".join(details))

    ignored = len(result.missing_keys) + len(result.unexpected_keys)
    if ignored:
        print(
            "Ignored legacy mean-pooling LayerNorm keys: "
            f"missing={list(result.missing_keys)} "
            f"unexpected={list(result.unexpected_keys)}"
        )


def class_labels(task: str, num_classes: int) -> list[str]:
    """返回绘图用类别标签。"""
    if task == "space_group":
        return [str(i) for i in range(1, num_classes + 1)]
    if task == "crystal_system" and num_classes == len(CRYSTAL_SYSTEM_LABELS):
        return CRYSTAL_SYSTEM_LABELS
    return [str(i) for i in range(num_classes)]


def sparse_ticks(n: int, max_ticks: int = 40) -> np.ndarray:
    """为类别很多的图选择可读的刻度位置。"""
    if n <= max_ticks:
        return np.arange(n)
    step = int(np.ceil(n / max_ticks))
    ticks = np.arange(0, n, step)
    if ticks[-1] != n - 1:
        ticks = np.append(ticks, n - 1)
    return ticks


def angle_range_to_slice(
    *,
    start_deg: float,
    end_deg: float,
    signal_length: int,
    theta_min: float = 5.0,
    theta_max: float = 90.0,
) -> slice:
    """Map a 2theta interval to sampled PXRD indices."""
    if signal_length <= 1:
        return slice(0, signal_length)
    if theta_min >= theta_max:
        raise ValueError("theta_min must be smaller than theta_max")
    start = max(float(start_deg), float(theta_min))
    end = min(float(end_deg), float(theta_max))
    if start >= end:
        raise ValueError(
            f"empty 2theta range [{start_deg}, {end_deg}] for "
            f"coverage [{theta_min}, {theta_max}]"
        )
    step = (theta_max - theta_min) / (signal_length - 1)
    left = int(math.floor((start - theta_min) / step))
    right = int(math.ceil((end - theta_min) / step)) + 1
    left = max(0, min(signal_length - 1, left))
    right = max(left + 1, min(signal_length, right))
    return slice(left, right)


class MaskAngleRange:
    """Zero out a fixed 2theta interval for occlusion analysis."""

    def __init__(self, rng: slice, fill_value: float = 0.0) -> None:
        self.rng = rng
        self.fill_value = float(fill_value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[self.rng] = self.fill_value
        return out


def rows_metadata(labels_csv: str | Path, rows: np.ndarray) -> pd.DataFrame:
    """Load labels metadata for a sequence of dataset rows."""
    meta = pd.read_csv(
        labels_csv,
        usecols=["row", "space_group", "crystal_system", "crystal_system_id"],
    ).set_index("row")
    selected = meta.loc[np.asarray(rows, dtype=np.int64)].reset_index()
    return selected


@torch.no_grad()
def evaluate_with_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    max_topk: int = 10,
    aux_loss_weights: dict[str, float] | None = None,
    hierarchical_aux_weight: float = 0.0,
    hierarchical_consistency_weight: float = 0.0,
    hierarchical_expert_weight: float = 0.0,
    hierarchical_mask_mode: str = "none",
    rare_classes: set[int] | None = None,
) -> tuple[StepStats, dict[str, np.ndarray]]:
    """评估模型，同时收集绘图所需的预测结果。"""
    model.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    n = 0
    correct1 = 0
    correct5 = 0
    correct10 = 0
    rare_correct1 = 0
    rare_total = 0
    gate_sum = 0.0
    gate_count = 0
    class_correct1: np.ndarray | None = None
    class_total: np.ndarray | None = None
    topk_correct: np.ndarray | None = None

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    top1_conf_parts: list[np.ndarray] = []
    true_conf_parts: list[np.ndarray] = []
    correct_parts: list[np.ndarray] = []
    gate_mean_parts: list[np.ndarray] = []
    crystal_true_parts: list[np.ndarray] = []
    crystal_pred_parts: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            output = model(x)
            loss, logits, aux_loss, _contrastive_loss = loss_with_auxiliary(
                output,
                y,
                loss_fn,
                aux_weights=aux_loss_weights,
                hierarchical_aux_weight=hierarchical_aux_weight,
                hierarchical_consistency_weight=hierarchical_consistency_weight,
                hierarchical_expert_weight=hierarchical_expert_weight,
            )
            if isinstance(output, dict) and "crystal_logits" in output:
                crystal_logits = output["crystal_logits"]
                crystal_true = space_group_to_crystal_system(y)
                crystal_true_parts.append(crystal_true.detach().cpu().numpy())
                crystal_pred_parts.append(
                    crystal_logits.argmax(dim=1).detach().cpu().numpy()
                )
                logits = apply_hierarchical_mask(
                    logits,
                    crystal_logits,
                    mode=hierarchical_mask_mode,
                )

        bsz = y.size(0)
        num_classes = logits.shape[1]
        plot_topk = min(max_topk, num_classes)
        top5 = min(5, num_classes)
        top10 = min(10, num_classes)
        eval_topk = max(plot_topk, top5, top10)

        top_values, top_idx = logits.float().topk(eval_topk, dim=1)
        top_valid = torch.isfinite(top_values)
        probs = torch.softmax(logits.float(), dim=1)
        top_probs = probs.gather(1, top_idx).masked_fill(~top_valid, 0.0)
        pred = top_idx[:, 0]
        correct_mask = pred == y

        total_loss += float(loss.item()) * bsz
        total_aux_loss += float(aux_loss.item()) * bsz
        n += bsz
        correct1 += int(correct_mask.sum().item())
        correct5 += int(
            (
                (top_idx[:, :top5] == y.unsqueeze(1))
                & top_valid[:, :top5]
            ).any(dim=1).sum().item()
        )
        correct10 += int(
            (
                (top_idx[:, :top10] == y.unsqueeze(1))
                & top_valid[:, :top10]
            ).any(dim=1).sum().item()
        )

        if topk_correct is None:
            topk_correct = np.zeros(plot_topk, dtype=np.int64)
        hits = (
            (top_idx[:, :plot_topk] == y.unsqueeze(1))
            & top_valid[:, :plot_topk]
        ).cumsum(dim=1).clamp(max=1)
        topk_correct += hits.sum(dim=0).detach().cpu().numpy()

        if class_total is None:
            class_total = np.zeros(num_classes, dtype=np.int64)
            class_correct1 = np.zeros(num_classes, dtype=np.int64)

        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        correct_cpu = correct_mask.detach().cpu()
        y_np = y_cpu.numpy()

        class_total += np.bincount(y_np, minlength=num_classes)
        class_correct1 += np.bincount(
            y_cpu[correct_cpu].numpy(),
            minlength=num_classes,
        )
        if rare_classes:
            rare_mask = np.isin(y_np, list(rare_classes))
            rare_total += int(rare_mask.sum())
            if rare_mask.any():
                rare_correct1 += int((pred_cpu.numpy()[rare_mask] == y_np[rare_mask]).sum())

        gate_mean = output_gate_mean(output)
        if gate_mean is not None:
            gate_mean_cpu = gate_mean.float().detach().cpu().numpy()
            gate_mean_parts.append(gate_mean_cpu)
            gate_sum += float(gate_mean_cpu.sum())
            gate_count += int(gate_mean_cpu.size)

        y_true_parts.append(y_np)
        y_pred_parts.append(pred_cpu.numpy())
        top1_conf_parts.append(top_probs[:, 0].detach().cpu().numpy())
        true_conf_parts.append(
            probs.gather(1, y.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
        )
        correct_parts.append(correct_cpu.numpy())

    outputs = {
        "y_true": np.concatenate(y_true_parts) if y_true_parts else np.array([]),
        "y_pred": np.concatenate(y_pred_parts) if y_pred_parts else np.array([]),
        "top1_conf": (
            np.concatenate(top1_conf_parts) if top1_conf_parts else np.array([])
        ),
        "true_conf": (
            np.concatenate(true_conf_parts) if true_conf_parts else np.array([])
        ),
        "correct": np.concatenate(correct_parts) if correct_parts else np.array([]),
        "topk_correct": topk_correct if topk_correct is not None else np.array([]),
        "gate_mean": (
            np.concatenate(gate_mean_parts) if gate_mean_parts else np.array([])
        ),
        "crystal_true": (
            np.concatenate(crystal_true_parts) if crystal_true_parts else np.array([])
        ),
        "crystal_pred": (
            np.concatenate(crystal_pred_parts) if crystal_pred_parts else np.array([])
        ),
    }
    stats = StepStats(
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
    return stats, outputs


def save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    normalize: bool,
) -> None:
    """保存混淆矩阵图。"""
    n_classes = len(labels)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        data = np.divide(
            cm,
            denom,
            out=np.zeros_like(cm, dtype=np.float64),
            where=denom > 0,
        )
        title = "Normalized Confusion Matrix"
        cbar_label = "Recall by true class"
        vmin, vmax = 0.0, 1.0
    else:
        data = cm
        title = "Confusion Matrix"
        cbar_label = "Count"
        vmin, vmax = None, None

    size = min(18, max(7, n_classes * 0.12))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ticks = sparse_ticks(n_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    if n_classes <= 12:
        fmt = ".2f" if normalize else "d"
        for i in range(n_classes):
            for j in range(n_classes):
                value = data[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > data.max() * 0.5 else "black",
                )

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_per_class_accuracy(
    class_correct: np.ndarray | None,
    class_total: np.ndarray | None,
    labels: list[str],
    out_path: Path,
) -> None:
    """保存每类准确率柱状图。"""
    if class_correct is None or class_total is None:
        class_correct = np.zeros(len(labels), dtype=np.int64)
        class_total = np.zeros(len(labels), dtype=np.int64)

    present = class_total > 0
    idx = np.where(present)[0]
    acc = np.divide(
        class_correct[present],
        class_total[present],
        out=np.zeros(int(present.sum()), dtype=np.float64),
        where=class_total[present] > 0,
    )

    fig_width = min(20, max(9, len(idx) * 0.08))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
    ax.bar(np.arange(len(idx)), acc, color="#4c78a8", width=0.85)
    ticks = sparse_ticks(len(idx))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[idx[i]] for i in ticks], rotation=90, fontsize=7)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlabel("Class")
    ax.set_title("Per-Class Accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_class_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
) -> None:
    """保存真实/预测类别分布对比图。"""
    n_classes = len(labels)
    true_counts = np.bincount(y_true, minlength=n_classes)
    pred_counts = np.bincount(y_pred, minlength=n_classes)
    x = np.arange(n_classes)

    fig_width = min(20, max(9, n_classes * 0.08))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
    ax.plot(x, true_counts, color="#4c78a8", lw=1.6, label="True")
    ax.plot(x, pred_counts, color="#f58518", lw=1.6, label="Predicted")
    if max(true_counts.max(initial=0), pred_counts.max(initial=0)) > 100:
        ax.set_yscale("log")
    ticks = sparse_ticks(n_classes)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_ylabel("Samples")
    ax.set_xlabel("Class")
    ax.set_title("True vs Predicted Class Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_confidence_histogram(
    confidences: np.ndarray,
    correct: np.ndarray,
    out_path: Path,
) -> None:
    """保存预测置信度分布图。"""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 21)
    correct_conf = confidences[correct.astype(bool)]
    wrong_conf = confidences[~correct.astype(bool)]
    if len(correct_conf):
        ax.hist(correct_conf, bins=bins, alpha=0.72, label="Correct", color="#54a24b")
    if len(wrong_conf):
        ax.hist(wrong_conf, bins=bins, alpha=0.72, label="Wrong", color="#e45756")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Samples")
    ax.set_title("Prediction Confidence")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_topk_accuracy(
    topk_correct: np.ndarray,
    n_samples: int,
    out_path: Path,
) -> None:
    """保存 Top-k 准确率曲线。"""
    k = np.arange(1, len(topk_correct) + 1)
    acc = topk_correct / max(n_samples, 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(k, acc, marker="o", color="#4c78a8")
    ax.set_xticks(k)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k Accuracy")
    ax.grid(alpha=0.25)
    for x, y in zip(k, acc):
        ax.text(x, min(y + 0.025, 0.98), f"{y:.3f}", ha="center", fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_gate_mean_distribution(
    gate_mean: np.ndarray,
    out_path: Path,
) -> None:
    """保存 gated SA/WA fusion 的样本级 gate 均值分布。"""
    gate_mean = np.asarray(gate_mean, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(gate_mean, bins=bins, color="#72b7b2", alpha=0.82)
    mean = float(gate_mean.mean()) if gate_mean.size else float("nan")
    if gate_mean.size:
        ax.axvline(mean, color="#e45756", lw=1.8, label=f"Mean={mean:.3f}")
        ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean gate value per sample (SA weight; WA weight = 1 - gate)")
    ax.set_ylabel("Samples")
    ax.set_title("SA/WA Gate Mean Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_gate_by_crystal_system(
    gate_mean: np.ndarray,
    metadata: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """保存不同晶系的 gate 均值，并返回汇总表。"""
    gate_mean = np.asarray(gate_mean, dtype=np.float64)
    if gate_mean.size == 0:
        return pd.DataFrame()

    df = metadata.copy()
    df["gate_mean"] = gate_mean
    summary = (
        df.groupby(["crystal_system_id", "crystal_system"], dropna=False)["gate_mean"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .sort_values("crystal_system_id")
    )
    summary["std"] = summary["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], color="#4c78a8", width=0.72)
    ax.errorbar(
        x,
        summary["mean"],
        yerr=summary["std"],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.1,
        capsize=3,
    )
    ax.axhline(0.5, color="#e45756", lw=1.2, ls="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        summary["crystal_system"].astype(str).to_list(),
        rotation=30,
        ha="right",
    )
    ax.set_xlabel("Crystal system")
    ax.set_ylabel("Mean gate value per sample (SA weight)")
    ax.set_title("Gate Mean by Crystal System")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def save_per_class_metrics(
    stats: StepStats,
    labels: list[str],
    out_path: Path,
    *,
    rare_classes: set[int] | None = None,
) -> None:
    """保存每类样本数、正确数、准确率和 rare 标记。"""
    rare_classes = rare_classes or set()
    total = (
        stats.class_total
        if stats.class_total is not None
        else np.zeros(len(labels), dtype=np.int64)
    )
    correct = (
        stats.class_correct_top1
        if stats.class_correct_top1 is not None
        else np.zeros(len(labels), dtype=np.int64)
    )
    acc = np.divide(
        correct,
        total,
        out=np.full(len(labels), np.nan, dtype=np.float64),
        where=total > 0,
    )
    rows = pd.DataFrame({
        "class_id": np.arange(len(labels)),
        "label": labels,
        "total": total,
        "correct_top1": correct,
        "acc1": acc,
        "is_rare": [i in rare_classes for i in range(len(labels))],
    })
    rows.to_csv(out_path, index=False)


def save_confusion_cases(
    outputs: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    labels: list[str],
    out_path: Path,
    *,
    max_rows: int = 100,
) -> None:
    """保存高置信度错分案例，便于回看混淆空间群。"""
    y_true = outputs["y_true"].astype(np.int64)
    y_pred = outputs["y_pred"].astype(np.int64)
    correct = outputs["correct"].astype(bool)
    df = metadata.copy()
    df["true_class_id"] = y_true
    df["pred_class_id"] = y_pred
    df["true_label"] = [labels[i] for i in y_true]
    df["pred_label"] = [labels[i] for i in y_pred]
    df["top1_conf"] = outputs["top1_conf"]
    df["true_conf"] = outputs["true_conf"]
    df["confidence_margin"] = df["top1_conf"] - df["true_conf"]
    if outputs.get("gate_mean", np.array([])).size == len(df):
        df["gate_mean"] = outputs["gate_mean"]

    wrong = df.loc[~correct].copy()
    if wrong.empty:
        wrong.head(0).to_csv(out_path, index=False)
        return
    wrong.sort_values(
        ["top1_conf", "confidence_margin"],
        ascending=False,
    ).head(max_rows).to_csv(out_path, index=False)


def save_confusion_pair_summary(
    outputs: dict[str, np.ndarray],
    labels: list[str],
    out_path: Path,
    *,
    max_rows: int = 100,
) -> None:
    """保存最常见的 true/pred 混淆对。"""
    y_true = outputs["y_true"].astype(np.int64)
    y_pred = outputs["y_pred"].astype(np.int64)
    wrong = y_true != y_pred
    if not wrong.any():
        pd.DataFrame(
            columns=["true_class_id", "pred_class_id", "true_label", "pred_label", "count"]
        ).to_csv(out_path, index=False)
        return

    pairs = pd.DataFrame({
        "true_class_id": y_true[wrong],
        "pred_class_id": y_pred[wrong],
    })
    summary = (
        pairs.value_counts(["true_class_id", "pred_class_id"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(max_rows)
    )
    summary["true_label"] = [labels[i] for i in summary["true_class_id"]]
    summary["pred_label"] = [labels[i] for i in summary["pred_class_id"]]
    summary = summary[
        ["true_class_id", "pred_class_id", "true_label", "pred_label", "count"]
    ]
    summary.to_csv(out_path, index=False)


def save_eval_plots(
    outputs: dict[str, np.ndarray],
    stats: StepStats,
    *,
    task: str,
    num_classes: int,
    out_dir: Path,
    metadata: pd.DataFrame | None = None,
    rare_classes: set[int] | None = None,
) -> list[Path]:
    """生成并保存评估图表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = class_labels(task, num_classes)
    y_true = outputs["y_true"].astype(np.int64)
    y_pred = outputs["y_pred"].astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    paths = [
        out_dir / "01_confusion_matrix_normalized.png",
        out_dir / "02_per_class_accuracy.png",
        out_dir / "03_class_distribution_true_vs_pred.png",
        out_dir / "04_confidence_histogram.png",
        out_dir / "05_topk_accuracy.png",
    ]
    save_confusion_matrix(cm, labels, paths[0], normalize=True)
    save_per_class_accuracy(
        stats.class_correct_top1,
        stats.class_total,
        labels,
        paths[1],
    )
    save_class_distribution(y_true, y_pred, labels, paths[2])
    save_confidence_histogram(outputs["top1_conf"], outputs["correct"], paths[3])
    save_topk_accuracy(outputs["topk_correct"], stats.n, paths[4])
    per_class_path = out_dir / "per_class_metrics.csv"
    save_per_class_metrics(stats, labels, per_class_path, rare_classes=rare_classes)
    paths.append(per_class_path)

    gate_mean = outputs.get("gate_mean", np.array([]))
    if gate_mean.size:
        path = out_dir / "06_gate_mean_distribution.png"
        save_gate_mean_distribution(gate_mean, path)
        paths.append(path)
        if metadata is not None:
            system_path = out_dir / "07_gate_by_crystal_system.png"
            gate_summary = save_gate_by_crystal_system(gate_mean, metadata, system_path)
            if not gate_summary.empty:
                gate_summary.to_csv(out_dir / "gate_by_crystal_system.csv", index=False)
                paths.append(system_path)

    if metadata is not None:
        cases_path = out_dir / "confusion_cases.csv"
        pairs_path = out_dir / "confusion_pairs.csv"
        save_confusion_cases(outputs, metadata, labels, cases_path)
        save_confusion_pair_summary(outputs, labels, pairs_path)
        paths.extend([cases_path, pairs_path])
    return paths


def main():
    """主评估流程。"""
    # ---------- 命令行参数解析 ----------
    ap = argparse.ArgumentParser(description="评估 PXRD 分类模型")
    ap.add_argument("--checkpoint", required=True, help="模型检查点文件路径")
    ap.add_argument(
        "--plot-dir",
        default=None,
        help="评估图保存目录；默认保存到 checkpoint 同级的 eval_plots 目录",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="只打印指标，不生成评估图",
    )
    ap.add_argument(
        "--topk-plot-max",
        type=int,
        default=10,
        help="Top-k 准确率曲线最大 k 值",
    )
    ap.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="评估的数据划分，默认 test",
    )
    ap.add_argument(
        "--only-rare",
        action="store_true",
        help="只评估训练集频数定义下的 rare classes 样本",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅评估测试集前 N 个样本；用于快速烟测，默认全量测试集",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖 checkpoint 配置中的评估 batch size",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="覆盖 checkpoint 配置中的 DataLoader num_workers；烟测可设为 0",
    )
    ap.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="评估时禁用 pin_memory",
    )
    ap.add_argument(
        "--skip-occlusion",
        action="store_true",
        help="跳过默认 5-15 度低角遮挡复评",
    )
    ap.add_argument(
        "--occlusion-range",
        type=float,
        nargs=2,
        default=(5.0, 15.0),
        metavar=("START_DEG", "END_DEG"),
        help="低角遮挡实验的 2theta 范围，默认 5 15",
    )
    args = ap.parse_args()

    # ---------- 加载检查点 ----------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]  # 从检查点恢复训练时的配置
    set_seed(cfg["experiment"].get("seed", 42))

    # ---------- 选择计算设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(cfg, device)

    # ---------- 加载评估数据 ----------
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    class_counts = class_counts_for_rows(
        Path(cfg["data"]["root"]) / "labels.csv",
        splits["train"],
        task,
    )
    metrics_cfg = cfg.get("metrics", {})
    rare_max_count = int(ckpt.get(
        "rare_max_train_count",
        metrics_cfg.get("rare_max_train_count", 100),
    ))
    rare_min_count = int(ckpt.get(
        "rare_min_train_count",
        metrics_cfg.get("rare_min_train_count", 1),
    ))
    rare_classes = rare_classes_from_counts(
        class_counts,
        max_count=rare_max_count,
        min_count=rare_min_count,
    )

    test_rows = splits[args.split]
    if args.only_rare:
        y_eval = labels_for_rows(
            Path(cfg["data"]["root"]) / "labels.csv",
            test_rows,
            task,
        )
        test_rows = test_rows[np.isin(y_eval, list(rare_classes))]
        if len(test_rows) == 0:
            raise ValueError(f"{args.split} split has no rare-class samples")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        test_rows = test_rows[:args.max_samples]
    test_ds = PXRDDataset(cfg["data"]["root"], rows=test_rows, task=task)
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg["data"]["batch_size"])
    )
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(cfg["data"]["num_workers"])
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["data"]["pin_memory"]) and not args.no_pin_memory,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    # ---------- 加载模型权重 ----------
    model = build_model(cfg, in_dim=test_ds.signal_length,
                        num_classes=test_ds.num_classes).to(device)
    load_model_state_compatible(model, ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val acc1={ckpt.get('val_acc1', float('nan')):.4f}")
    sa_mamba_backend = ""
    wa_mamba_backend = ""
    if hasattr(model, "sa_branch") and getattr(model, "sa_branch") is not None:
        sa_mamba_backend = str(model.sa_branch.actual_mamba_backend)
        print(f"  SA mixer backend: {sa_mamba_backend}")
    if hasattr(model, "wa_branch") and getattr(model, "wa_branch") is not None:
        wa_mamba_backend = str(model.wa_branch.actual_mamba_backend)
        print(f"  WA mixer backend: {wa_mamba_backend}")
    aux_loss_weights = aux_loss_weights_from_model(model)
    if aux_loss_weights:
        print(f"Aux losses: {aux_loss_weights}")

    # ---------- 在评估集上评估 ----------
    loss_name = cfg["loss"]["name"].lower()
    print(
        f"Rare classes: {len(rare_classes)} "
        f"(train count {rare_min_count}-{rare_max_count})"
    )
    print(
        f"Eval split: {args.split}"
        f"{' rare-only' if args.only_rare else ''}  N={len(test_ds)}"
    )
    loss_fn = build_loss_from_config(
        cfg["loss"],
        class_counts=class_counts,
    ).to(device)
    if hasattr(loss_fn, "set_class_weights") and "ldam_drw_active" in ckpt:
        loss_fn.set_class_weights(bool(ckpt["ldam_drw_active"]))
        print(f"LDAM-DRW eval weights: {bool(ckpt['ldam_drw_active'])}")
    loss_cfg = cfg.get("loss", {})
    (
        hierarchical_aux_weight,
        hierarchical_consistency_weight,
        hierarchical_expert_weight,
    ) = hierarchical_config(loss_cfg)
    hierarchical_mask_mode = str(
        (loss_cfg.get("hierarchical", {}) or {}).get("inference_mask", "none")
    )
    if hierarchical_aux_weight or hierarchical_consistency_weight:
        print(
            "Hierarchical eval: "
            f"aux={hierarchical_aux_weight:.3f} "
            f"consistency={hierarchical_consistency_weight:.3f} "
            f"expert={hierarchical_expert_weight:.3f} "
            f"mask={hierarchical_mask_mode}"
        )
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    amp_dtype = amp_dtype_from_config(cfg["train"].get("amp_dtype", "float16"), device)
    stats, outputs = evaluate_with_predictions(
        model, test_loader, loss_fn, device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        max_topk=max(1, args.topk_plot_max),
        aux_loss_weights=aux_loss_weights,
        hierarchical_aux_weight=hierarchical_aux_weight,
        hierarchical_consistency_weight=hierarchical_consistency_weight,
        hierarchical_expert_weight=hierarchical_expert_weight,
        hierarchical_mask_mode=hierarchical_mask_mode,
        rare_classes=rare_classes,
    )
    macro_f1 = f1_score(
        outputs["y_true"],
        outputs["y_pred"],
        labels=np.arange(test_ds.num_classes),
        average="macro",
        zero_division=0,
    )

    crystal_acc1 = float("nan")
    if outputs["crystal_true"].size:
        crystal_acc1 = float((outputs["crystal_true"] == outputs["crystal_pred"]).mean())

    # 核心指标
    parts = [
        f"Test loss={stats.loss:.4f}",
        f"acc1={stats.acc1:.4f}",
        f"acc5={stats.acc5:.4f}",
        f"macro_f1={macro_f1:.4f}",
    ]
    # 条件指标（仅在启用时输出）
    if stats.rare_total:
        parts.append(f"rare_acc1={stats.rare_acc1:.4f}")
    if stats.aux_loss:
        parts.append(f"aux={stats.aux_loss:.4f}")
    if stats.gate_mean is not None:
        parts.append(f"gate={stats.gate_mean:.4f}")
    if math.isfinite(crystal_acc1):
        parts.append(f"crystal_acc1={crystal_acc1:.4f}")
    parts.append(f"(N={stats.n})")
    print("  ".join(parts))

    plot_dir = (
        Path(args.plot_dir)
        if args.plot_dir is not None
        else Path(args.checkpoint).resolve().parent / "eval_plots"
    )
    metadata = rows_metadata(Path(cfg["data"]["root"]) / "labels.csv", test_ds.rows)
    metrics = {
        "checkpoint": str(Path(args.checkpoint)),
        "epoch": int(ckpt["epoch"]),
        "task": task,
        "split": args.split,
        "n": int(stats.n),
        "loss": float(stats.loss),
        "acc1": float(stats.acc1),
        "acc5": float(stats.acc5),
        "macro_f1": float(macro_f1),
    }
    # 条件指标（仅在启用时输出）
    if args.only_rare:
        metrics["only_rare"] = True
    if sa_mamba_backend:
        metrics["sa_mamba_backend"] = sa_mamba_backend
    if wa_mamba_backend:
        metrics["wa_mamba_backend"] = wa_mamba_backend
    if stats.rare_total:
        metrics["rare_acc1"] = float(stats.rare_acc1)
        metrics["rare_total"] = int(stats.rare_total)
        metrics["rare_class_count"] = len(rare_classes)
    if stats.aux_loss:
        metrics["aux_loss"] = float(stats.aux_loss)
    if stats.gate_mean is not None:
        metrics["gate_mean"] = float(stats.gate_mean)
    if math.isfinite(crystal_acc1):
        metrics["crystal_acc1"] = crystal_acc1
        metrics["hierarchical_inference_mask"] = hierarchical_mask_mode

    if not args.no_plots:
        plot_paths = save_eval_plots(
            outputs,
            stats,
            task=task,
            num_classes=test_ds.num_classes,
            out_dir=plot_dir,
            metadata=metadata,
            rare_classes=rare_classes,
        )
        print(f"Saved evaluation plots to {plot_dir}")
        for path in plot_paths:
            print(f"  {path.name}")

    if not args.skip_occlusion:
        occ_start, occ_end = args.occlusion_range
        occ_slice = angle_range_to_slice(
            start_deg=occ_start,
            end_deg=occ_end,
            signal_length=test_ds.signal_length,
        )
        occ_ds = PXRDDataset(
            cfg["data"]["root"],
            rows=test_ds.rows,
            task=task,
            transform=MaskAngleRange(occ_slice),
        )
        occ_loader = DataLoader(occ_ds, **loader_kwargs)
        occ_stats, occ_outputs = evaluate_with_predictions(
            model,
            occ_loader,
            loss_fn,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_topk=max(1, args.topk_plot_max),
            aux_loss_weights=aux_loss_weights,
            hierarchical_aux_weight=hierarchical_aux_weight,
            hierarchical_consistency_weight=hierarchical_consistency_weight,
            hierarchical_expert_weight=hierarchical_expert_weight,
            hierarchical_mask_mode=hierarchical_mask_mode,
            rare_classes=rare_classes,
        )
        occ_macro_f1 = f1_score(
            occ_outputs["y_true"],
            occ_outputs["y_pred"],
            labels=np.arange(test_ds.num_classes),
            average="macro",
            zero_division=0,
        )
        metrics["occlusion"] = {
            "range_deg": [float(occ_start), float(occ_end)],
            "acc1": float(occ_stats.acc1),
            "acc5": float(occ_stats.acc5),
            "macro_f1": float(occ_macro_f1),
            "delta_acc1": float(occ_stats.acc1 - stats.acc1),
            "delta_macro_f1": float(occ_macro_f1 - macro_f1),
        }
        if occ_stats.rare_total:
            metrics["occlusion"]["rare_acc1"] = float(occ_stats.rare_acc1)
            metrics["occlusion"]["delta_rare_acc1"] = float(
                occ_stats.rare_acc1 - stats.rare_acc1
            )
        if occ_stats.gate_mean is not None:
            metrics["occlusion"]["gate_mean"] = float(occ_stats.gate_mean)
        occ_parts = [
            f"Occlusion {occ_start:.1f}-{occ_end:.1f} deg",
            f"acc1={occ_stats.acc1:.4f}",
            f"acc5={occ_stats.acc5:.4f}",
            f"macro_f1={occ_macro_f1:.4f}",
            f"delta_acc1={occ_stats.acc1 - stats.acc1:+.4f}",
            f"delta_f1={occ_macro_f1 - macro_f1:+.4f}",
        ]
        if occ_stats.rare_total:
            occ_parts.append(f"rare_acc1={occ_stats.rare_acc1:.4f}")
        print("  ".join(occ_parts))

    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = plot_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
