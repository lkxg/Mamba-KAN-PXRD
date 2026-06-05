"""在测试集上评估已训练模型的性能。

使用方法:
    python3 scripts/evaluate.py --checkpoint checkpoints/baseline_mlp/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from src.data import PXRDDataset, class_counts_for_rows, load_splits
from src.models import (
    BiGRUPatchClassifier,
    MLPClassifier,
    PatchTSTClassifier,
    ResNet1D,
)
from src.training import (
    amp_dtype_from_config,
    build_loss,
    configure_backend,
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
) -> tuple[StepStats, dict[str, np.ndarray]]:
    """评估模型，同时收集绘图所需的预测结果。"""
    model.eval()
    total_loss = 0.0
    n = 0
    correct1 = 0
    correct5 = 0
    class_correct1: np.ndarray | None = None
    class_total: np.ndarray | None = None
    topk_correct: np.ndarray | None = None

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    top1_conf_parts: list[np.ndarray] = []
    true_conf_parts: list[np.ndarray] = []
    correct_parts: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            logits = model(x)
            loss = loss_fn(logits, y)

        bsz = y.size(0)
        num_classes = logits.shape[1]
        plot_topk = min(max_topk, num_classes)
        top5 = min(5, num_classes)
        eval_topk = max(plot_topk, top5)

        probs = torch.softmax(logits.float(), dim=1)
        top_probs, top_idx = probs.topk(eval_topk, dim=1)
        pred = top_idx[:, 0]
        correct_mask = pred == y

        total_loss += float(loss.item()) * bsz
        n += bsz
        correct1 += int(correct_mask.sum().item())
        correct5 += int((top_idx[:, :top5] == y.unsqueeze(1)).any(dim=1).sum().item())

        if topk_correct is None:
            topk_correct = np.zeros(plot_topk, dtype=np.int64)
        hits = (top_idx[:, :plot_topk] == y.unsqueeze(1)).cumsum(dim=1).clamp(max=1)
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
    }
    stats = StepStats(
        loss=total_loss / max(n, 1),
        n=n,
        correct_top1=correct1,
        correct_top5=correct5,
        class_correct_top1=class_correct1,
        class_total=class_total,
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


def save_eval_plots(
    outputs: dict[str, np.ndarray],
    stats: StepStats,
    *,
    task: str,
    num_classes: int,
    out_dir: Path,
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
    args = ap.parse_args()

    # ---------- 加载检查点 ----------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]  # 从检查点恢复训练时的配置
    set_seed(cfg["experiment"].get("seed", 42))

    # ---------- 选择计算设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(cfg, device)

    # ---------- 加载测试数据 ----------
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    test_ds = PXRDDataset(cfg["data"]["root"], rows=splits["test"], task=task)
    loader_kwargs = dict(
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    if cfg["data"]["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    # ---------- 加载模型权重 ----------
    model = build_model(cfg, in_dim=test_ds.signal_length,
                        num_classes=test_ds.num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val acc1={ckpt.get('val_acc1', float('nan')):.4f}")

    # ---------- 在测试集上评估 ----------
    class_counts = None
    loss_name = cfg["loss"]["name"].lower()
    if loss_name in {"weighted_ce", "class_balanced_ce"}:
        class_counts = class_counts_for_rows(
            Path(cfg["data"]["root"]) / "labels.csv",
            splits["train"],
            task,
        )
    loss_fn = build_loss(
        loss_name,
        class_counts=class_counts,
        gamma=cfg["loss"].get("focal_gamma", 2.0),
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
        class_weight_power=cfg["loss"].get("class_weight_power", 0.5),
        class_weight_beta=cfg["loss"].get("class_weight_beta", 0.99),
    ).to(device)
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    amp_dtype = amp_dtype_from_config(cfg["train"].get("amp_dtype", "float16"), device)
    stats, outputs = evaluate_with_predictions(
        model, test_loader, loss_fn, device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        max_topk=max(1, args.topk_plot_max),
    )
    macro_f1 = f1_score(
        outputs["y_true"],
        outputs["y_pred"],
        labels=np.arange(test_ds.num_classes),
        average="macro",
        zero_division=0,
    )

    # 打印测试结果
    print(f"Test loss={stats.loss:.4f}  acc1={stats.acc1:.4f}  "
          f"acc5={stats.acc5:.4f}  macro={stats.macro_acc1:.4f}  "
          f"macro_f1={macro_f1:.4f}  (N={stats.n})")

    if not args.no_plots:
        plot_dir = (
            Path(args.plot_dir)
            if args.plot_dir is not None
            else Path(args.checkpoint).resolve().parent / "eval_plots"
        )
        plot_paths = save_eval_plots(
            outputs,
            stats,
            task=task,
            num_classes=test_ds.num_classes,
            out_dir=plot_dir,
        )
        print(f"Saved evaluation plots to {plot_dir}")
        for path in plot_paths:
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
