"""Evaluate a trained PXRD classifier and write core metrics.

Usage:
    python3 scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

mpl_config_dir = Path(
    os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "matplotlib")
)
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    PatchTSTClassifier,
    ResNet1D,
)
from src.training import (
    StepStats,
    build_loss_from_config,
    configure_backend,
    primary_logits,
    rare_classes_from_counts,
)
from src.utils import set_seed


def build_model(cfg: dict, *, in_dim: int, num_classes: int) -> torch.nn.Module:
    """Build a retained model from its checkpoint config."""
    name = cfg["model"]["name"].lower()
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes, **cfg["model"].get("resnet1d", {}))
    if name == "convnext1d":
        return ConvNeXt1D(num_classes=num_classes, **cfg["model"].get("convnext1d", {}))
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


def sparse_ticks(n: int, max_ticks: int = 40) -> np.ndarray:
    """Choose readable tick positions for many-class plots."""
    if n <= max_ticks:
        return np.arange(n)
    step = int(np.ceil(n / max_ticks))
    ticks = np.arange(0, n, step)
    if ticks[-1] != n - 1:
        ticks = np.append(ticks, n - 1)
    return ticks


@torch.no_grad()
def evaluate_with_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    rare_classes: set[int] | None = None,
) -> tuple[StepStats, np.ndarray, np.ndarray]:
    """Evaluate once while retaining labels needed for F1 and confusion matrix."""
    model.eval()
    total_loss = 0.0
    n = 0
    correct1 = 0
    correct5 = 0
    rare_correct1 = 0
    rare_total = 0
    rare_lookup: torch.Tensor | None = None
    class_correct1: np.ndarray | None = None
    class_total: np.ndarray | None = None
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            logits = primary_logits(model(x))
            loss = loss_fn(logits, y)

        bsz = y.size(0)
        num_classes = logits.shape[1]
        topk = min(5, num_classes)
        top_values, top_idx = logits.float().topk(topk, dim=1)
        top_valid = torch.isfinite(top_values)
        pred = top_idx[:, 0]
        correct_mask = pred == y

        total_loss += float(loss.item()) * bsz
        n += bsz
        correct1 += int(correct_mask.sum().item())
        correct5 += int(
            (
                (top_idx[:, : min(5, topk)] == y.unsqueeze(1))
                & top_valid[:, : min(5, topk)]
            ).any(dim=1).sum().item()
        )
        if class_total is None:
            class_total = np.zeros(num_classes, dtype=np.int64)
            class_correct1 = np.zeros(num_classes, dtype=np.int64)

        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        y_np = y_cpu.numpy()
        pred_np = pred_cpu.numpy()
        correct_np = correct_mask.detach().cpu().numpy()
        class_total += np.bincount(y_np, minlength=num_classes)
        class_correct1 += np.bincount(
            y_np[correct_np],
            minlength=num_classes,
        )

        if rare_classes:
            if rare_lookup is None:
                rare_lookup = torch.zeros(
                    num_classes,
                    dtype=torch.bool,
                    device=y.device,
                )
                rare_lookup[list(rare_classes)] = True
            rare_mask = rare_lookup[y]
            if rare_mask.any():
                rare_total += int(rare_mask.sum().item())
                rare_correct1 += int((pred[rare_mask] == y[rare_mask]).sum().item())

        y_true_parts.append(y_np)
        y_pred_parts.append(pred_np)

    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([])
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([])
    stats = StepStats(
        loss=total_loss / max(n, 1),
        n=n,
        correct_top1=correct1,
        correct_top5=correct5,
        class_correct_top1=class_correct1,
        class_total=class_total,
        rare_correct_top1=rare_correct1 if rare_classes else None,
        rare_total=rare_total if rare_classes else None,
    )
    return stats, y_true, y_pred


def save_normalized_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
) -> None:
    """Save the row-normalized space-group confusion matrix."""
    num_classes = cm.shape[0]
    denom = cm.sum(axis=1, keepdims=True)
    data = np.divide(
        cm,
        denom,
        out=np.zeros_like(cm, dtype=np.float64),
        where=denom > 0,
    )
    labels = [str(i) for i in range(1, num_classes + 1)]
    size = min(18, max(7, num_classes * 0.12))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall by true class")

    ticks = sparse_ticks(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
    ax.set_xlabel("Predicted space group")
    ax.set_ylabel("True space group")
    ax.set_title("Normalized Confusion Matrix")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_per_class_metrics(
    stats: StepStats,
    num_classes: int,
    rare_classes: set[int],
    out_path: Path,
) -> None:
    """Write per-class support, Top-1 accuracy, and rare-class membership."""
    total = (
        stats.class_total
        if stats.class_total is not None
        else np.zeros(num_classes, dtype=np.int64)
    )
    correct = (
        stats.class_correct_top1
        if stats.class_correct_top1 is not None
        else np.zeros(num_classes, dtype=np.int64)
    )
    if len(total) != num_classes or len(correct) != num_classes:
        raise ValueError("per-class metric length does not match num_classes")

    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["class_id", "label", "total", "correct_top1", "acc1", "is_rare"]
        )
        for class_id in range(num_classes):
            class_total = int(total[class_id])
            class_correct = int(correct[class_id])
            acc1: float | str = (
                class_correct / class_total if class_total > 0 else ""
            )
            writer.writerow(
                [
                    class_id,
                    class_id + 1,
                    class_total,
                    class_correct,
                    acc1,
                    class_id in rare_classes,
                ]
            )


def save_confusion_pairs(
    cm: np.ndarray,
    out_path: Path,
    *,
    max_rows: int = 100,
) -> None:
    """Write the most frequent off-diagonal true/predicted class pairs."""
    pairs = [
        (int(cm[true_id, pred_id]), int(true_id), int(pred_id))
        for true_id, pred_id in np.argwhere(cm > 0)
        if true_id != pred_id
    ]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["true_class_id", "pred_class_id", "true_label", "pred_label", "count"]
        )
        for count, true_id, pred_id in pairs[:max_rows]:
            writer.writerow([true_id, pred_id, true_id + 1, pred_id + 1, count])


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 PXRD 空间群分类模型")
    ap.add_argument("--checkpoint", required=True, help="模型检查点文件路径")
    ap.add_argument(
        "--plot-dir",
        default=None,
        help="评估输出目录；默认为 checkpoint 同级的 eval_plots",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成归一化混淆矩阵；仍写入 metrics.json 和两个 CSV",
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
        help="仅评估前 N 个样本，用于快速烟测",
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
        help="覆盖 checkpoint 配置中的 DataLoader num_workers",
    )
    ap.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="评估时禁用 pin_memory",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    set_seed(cfg["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(cfg, device)

    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    if task != "space_group":
        raise ValueError(f"evaluate.py only supports space_group, got {task!r}")
    labels_csv = Path(cfg["data"]["root"]) / "labels.csv"
    class_counts = class_counts_for_rows(labels_csv, splits["train"], task)
    metrics_cfg = cfg.get("metrics", {})
    rare_max_count = int(
        ckpt.get(
            "rare_max_train_count",
            metrics_cfg.get("rare_max_train_count", 100),
        )
    )
    rare_min_count = int(
        ckpt.get(
            "rare_min_train_count",
            metrics_cfg.get("rare_min_train_count", 1),
        )
    )
    rare_classes = rare_classes_from_counts(
        class_counts,
        max_count=rare_max_count,
        min_count=rare_min_count,
    )

    eval_rows = splits[args.split]
    if args.only_rare:
        labels = labels_for_rows(labels_csv, eval_rows, task)
        eval_rows = eval_rows[np.isin(labels, list(rare_classes))]
        if len(eval_rows) == 0:
            raise ValueError(f"{args.split} split has no rare-class samples")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        eval_rows = eval_rows[: args.max_samples]

    eval_ds = PXRDDataset(cfg["data"]["root"], rows=eval_rows, task=task)
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
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(cfg["data"]["pin_memory"]) and not args.no_pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)
    eval_loader = DataLoader(eval_ds, **loader_kwargs)

    model = build_model(
        cfg,
        in_dim=eval_ds.signal_length,
        num_classes=eval_ds.num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']}, "
        f"val acc1={ckpt.get('val_acc1', float('nan')):.4f}"
    )
    wa_mamba_backend = ""
    if hasattr(model, "wa_branch") and model.wa_branch is not None:
        wa_mamba_backend = str(model.wa_branch.actual_mamba_backend)
        print(f"  WA mixer backend: {wa_mamba_backend}")

    print(
        f"Rare classes: {len(rare_classes)} "
        f"(train count {rare_min_count}-{rare_max_count})"
    )
    print(
        f"Eval split: {args.split}"
        f"{' rare-only' if args.only_rare else ''}  N={len(eval_ds)}"
    )
    loss_fn = build_loss_from_config(
        cfg["loss"],
        class_counts=class_counts,
    ).to(device)
    if hasattr(loss_fn, "set_class_weights") and "ldam_drw_active" in ckpt:
        enabled = bool(ckpt["ldam_drw_active"])
        loss_fn.set_class_weights(enabled)
        print(f"LDAM-DRW eval weights: {enabled}")

    stats, y_true, y_pred = evaluate_with_predictions(
        model,
        eval_loader,
        loss_fn,
        device,
        rare_classes=rare_classes,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(eval_ds.num_classes),
        average="macro",
        zero_division=0,
    )

    parts = [
        f"Test loss={stats.loss:.4f}",
        f"acc1={stats.acc1:.4f}",
        f"acc5={stats.acc5:.4f}",
        f"macro_f1={macro_f1:.4f}",
    ]
    if stats.rare_total:
        parts.append(f"rare_acc1={stats.rare_acc1:.4f}")
    parts.append(f"(N={stats.n})")
    print("  ".join(parts))

    output_dir = (
        Path(args.plot_dir)
        if args.plot_dir is not None
        else Path(args.checkpoint).resolve().parent / "eval_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(eval_ds.num_classes),
    )
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
    if args.only_rare:
        metrics["only_rare"] = True
    if wa_mamba_backend:
        metrics["wa_mamba_backend"] = wa_mamba_backend
    if stats.rare_total:
        metrics["rare_acc1"] = float(stats.rare_acc1)
        metrics["rare_total"] = int(stats.rare_total)
        metrics["rare_class_count"] = len(rare_classes)

    if not args.no_plots:
        confusion_path = output_dir / "01_confusion_matrix_normalized.png"
        save_normalized_confusion_matrix(
            cm,
            confusion_path,
        )
        print(f"Saved normalized confusion matrix to {confusion_path}")

    per_class_path = output_dir / "per_class_metrics.csv"
    save_per_class_metrics(
        stats,
        eval_ds.num_classes,
        rare_classes,
        per_class_path,
    )
    print(f"Saved per-class metrics to {per_class_path}")

    confusion_pairs_path = output_dir / "confusion_pairs.csv"
    save_confusion_pairs(cm, confusion_pairs_path)
    print(f"Saved confusion pairs to {confusion_pairs_path}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
