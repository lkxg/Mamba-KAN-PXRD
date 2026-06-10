"""Post-hoc logit adjustment 评估（Menon et al., ICLR 2021）。

在已训练的 checkpoint 上做推理并缓存 logits，然后扫描温度系数 tau：

    adjusted_logits = logits - tau * log(train_prior)

其中 train_prior 是训练集的类别先验。tau=0 等价于不调整（baseline）。
协议：在 val 上扫 tau 选出 tau*（按 --select-metric），再用 tau* 报告 test
指标，避免在 test 上调参。整个过程不需要重新训练。

logits 会缓存到输出目录下的 npz 文件，重复运行（例如换 tau 网格或
选择指标）时直接复用缓存，跳过推理。

使用方法:
    python scripts/logit_adjustment.py --checkpoint checkpoints/m18_.../best.pt
    # 细化扫描并按 macro_acc1 选 tau
    python scripts/logit_adjustment.py --checkpoint ... \
        --tau-min 0.5 --tau-max 1.5 --tau-step 0.05 --select-metric macro_acc1
"""
from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import build_model
from src.data import PXRDDataset, class_counts_for_rows, load_splits
from src.training import (
    amp_dtype_from_config,
    configure_backend,
    primary_logits,
    rare_classes_from_counts,
)
from src.utils import set_seed


METRIC_KEYS = [
    "acc1", "acc5", "acc10",
    "macro_acc1", "macro_f1", "rare_acc1", "balanced",
]


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    """推理一遍，返回全量 (logits[N, C], y_true[N])。"""
    model.eval()
    logits_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            output = model(x)
            logits = primary_logits(output)
        logits_parts.append(logits.float().cpu().numpy())
        y_parts.append(y.numpy())
    return np.concatenate(logits_parts), np.concatenate(y_parts)


def cached_logits(
    cache_path: Path,
    rows: np.ndarray,
    compute_fn,
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """读取缓存的 logits；缺失、rows 不一致或 --force-recompute 时重新推理。"""
    if cache_path.exists() and not force:
        data = np.load(cache_path)
        if np.array_equal(data["rows"], rows):
            print(f"Loaded cached logits from {cache_path}")
            return data["logits"].astype(np.float32), data["y_true"]
        print(f"Cache {cache_path} has stale rows; recomputing")
    logits, y_true = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        logits=logits.astype(np.float16),
        y_true=y_true.astype(np.int64),
        rows=np.asarray(rows, dtype=np.int64),
    )
    print(f"Saved logits cache to {cache_path}")
    return logits, y_true


def metrics_for_adjusted(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    log_prior: np.ndarray,
    tau: float,
    num_classes: int,
    rare_classes: set[int],
    acc1_weight: float,
    macro_weight: float,
) -> dict[str, float]:
    """对 adjusted logits 计算与 evaluate.py 一致的指标集合。"""
    adj = logits - tau * log_prior[None, :]

    k = min(10, num_classes)
    part = np.argpartition(-adj, k - 1, axis=1)[:, :k]
    row = np.arange(adj.shape[0])[:, None]
    order = np.argsort(-adj[row, part], axis=1)
    topk = part[row, order]  # [N, k]，按 adjusted logit 降序
    pred = topk[:, 0]

    hit = topk == y_true[:, None]
    acc1 = float(hit[:, :1].any(axis=1).mean())
    acc5 = float(hit[:, : min(5, k)].any(axis=1).mean())
    acc10 = float(hit.any(axis=1).mean())

    correct = pred == y_true
    class_total = np.bincount(y_true, minlength=num_classes)
    class_correct = np.bincount(y_true[correct], minlength=num_classes)
    present = class_total > 0
    macro_acc1 = float(
        (class_correct[present] / class_total[present]).mean()
    )
    macro_f1 = float(f1_score(
        y_true,
        pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    ))

    rare_acc1 = float("nan")
    if rare_classes:
        rare_mask = np.isin(y_true, list(rare_classes))
        if rare_mask.any():
            rare_acc1 = float(correct[rare_mask].mean())

    balanced = (
        acc1_weight * acc1 + macro_weight * macro_acc1
    ) / (acc1_weight + macro_weight)
    return {
        "tau": float(tau),
        "acc1": acc1,
        "acc5": acc5,
        "acc10": acc10,
        "macro_acc1": macro_acc1,
        "macro_f1": macro_f1,
        "rare_acc1": rare_acc1,
        "balanced": balanced,
    }


def sweep_tau(
    logits: np.ndarray,
    y_true: np.ndarray,
    taus: np.ndarray,
    **kwargs,
) -> pd.DataFrame:
    """对 tau 网格逐点计算指标。"""
    rows = [
        metrics_for_adjusted(logits, y_true, tau=tau, **kwargs)
        for tau in taus
    ]
    return pd.DataFrame(rows)


def save_sweep_plot(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    tau_star: float,
    select_metric: str,
    out_path: Path,
) -> None:
    """保存指标-tau 曲线与 acc1/macro trade-off 图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)

    ax = axes[0]
    colors = {
        "acc1": "#4c78a8",
        "macro_acc1": "#f58518",
        "macro_f1": "#54a24b",
        "rare_acc1": "#e45756",
        "balanced": "#b279a2",
    }
    for key, color in colors.items():
        ax.plot(val_df["tau"], val_df[key], color=color, lw=1.6, label=f"val {key}")
        ax.plot(
            test_df["tau"], test_df[key],
            color=color, lw=1.6, ls="--", alpha=0.65, label=f"test {key}",
        )
    ax.axvline(tau_star, color="#333333", lw=1.2, ls=":",
               label=f"tau*={tau_star:.2f} (val {select_metric})")
    ax.set_xlabel("tau")
    ax.set_ylabel("Metric")
    ax.set_title("Logit Adjustment Sweep")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncols=2)

    ax = axes[1]
    for df, name, ls in ((val_df, "val", "-"), (test_df, "test", "--")):
        ax.plot(df["acc1"], df["macro_acc1"], marker="o", ms=3.5,
                ls=ls, lw=1.4, label=name)
    base = test_df.iloc[(test_df["tau"] - 0.0).abs().idxmin()]
    star = test_df.iloc[(test_df["tau"] - tau_star).abs().idxmin()]
    ax.scatter([base["acc1"]], [base["macro_acc1"]], s=70, marker="s",
               color="#333333", zorder=5, label="test tau=0")
    ax.scatter([star["acc1"]], [star["macro_acc1"]], s=90, marker="*",
               color="#e45756", zorder=5, label=f"test tau*={tau_star:.2f}")
    ax.set_xlabel("acc1")
    ax.set_ylabel("macro_acc1")
    ax.set_title("acc1 vs macro_acc1 Trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    """主流程：缓存 logits -> 扫 tau -> val 选 tau* -> 报告 test。"""
    # ---------- 命令行参数解析 ----------
    ap = argparse.ArgumentParser(description="Post-hoc logit adjustment 评估")
    ap.add_argument("--checkpoint", required=True, help="模型检查点文件路径")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="输出目录；默认保存到 checkpoint 同级的 logit_adjustment 目录",
    )
    ap.add_argument("--tau-min", type=float, default=0.0, help="tau 网格下界")
    ap.add_argument("--tau-max", type=float, default=2.0, help="tau 网格上界")
    ap.add_argument("--tau-step", type=float, default=0.1, help="tau 网格步长")
    ap.add_argument(
        "--select-metric",
        choices=["balanced", "macro_acc1", "macro_f1", "rare_acc1"],
        default="balanced",
        help="在选择划分上挑选 tau* 的指标，默认与训练 monitor 一致的 balanced",
    )
    ap.add_argument(
        "--select-split",
        choices=["val", "test"],
        default="val",
        help="挑选 tau* 的划分；默认 val（在 test 上选会高估效果）",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个划分仅评估前 N 个样本；用于快速烟测，默认全量",
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
        "--force-recompute",
        action="store_true",
        help="忽略已缓存的 logits，重新推理",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="只输出 CSV/JSON，不生成图",
    )
    args = ap.parse_args()

    if args.tau_step <= 0:
        raise ValueError("--tau-step must be positive")
    if args.tau_max < args.tau_min:
        raise ValueError("--tau-max must be >= --tau-min")
    if args.select_split == "test":
        print("WARNING: selecting tau on test overfits the test set; "
              "use --select-split val for honest reporting")

    # ---------- 加载检查点 ----------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    set_seed(cfg["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(cfg, device)

    # ---------- 训练集类先验与 rare classes ----------
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    labels_csv = Path(cfg["data"]["root"]) / "labels.csv"
    class_counts = class_counts_for_rows(labels_csv, splits["train"], task)
    prior = class_counts / max(class_counts.sum(), 1)
    log_prior = np.log(np.maximum(prior, 1e-12)).astype(np.float32)

    metrics_cfg = cfg.get("metrics", {})
    rare_classes = rare_classes_from_counts(
        class_counts,
        max_count=int(ckpt.get(
            "rare_max_train_count",
            metrics_cfg.get("rare_max_train_count", 100),
        )),
        min_count=int(ckpt.get(
            "rare_min_train_count",
            metrics_cfg.get("rare_min_train_count", 1),
        )),
    )

    balanced_cfg = cfg.get("checkpoint", {}).get("balanced_metric", {})
    acc1_weight = float(balanced_cfg.get("acc1_weight", 0.5))
    macro_weight = float(balanced_cfg.get("macro_acc1_weight", 0.5))

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(args.checkpoint).resolve().parent / "logit_adjustment"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 推理或读取缓存 ----------
    model = None
    num_classes = len(class_counts)
    split_logits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("val", "test"):
        rows = splits[split]
        if args.max_samples is not None:
            if args.max_samples <= 0:
                raise ValueError("--max-samples must be positive")
            rows = rows[: args.max_samples]

        def compute(rows=rows):
            nonlocal model
            ds = PXRDDataset(cfg["data"]["root"], rows=rows, task=task)
            if model is None:
                model = build_model(
                    cfg, in_dim=ds.signal_length, num_classes=ds.num_classes,
                ).to(device)
                model.load_state_dict(ckpt["model_state"])
                print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
                      f"val acc1={ckpt.get('val_acc1', float('nan')):.4f}")
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
            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=bool(cfg["data"]["pin_memory"]) and not args.no_pin_memory,
                persistent_workers=num_workers > 0,
            )
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)
            loader = DataLoader(ds, **loader_kwargs)
            use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
            amp_dtype = amp_dtype_from_config(
                cfg["train"].get("amp_dtype", "float16"), device,
            )
            print(f"Computing logits on {split} split (N={len(ds)})")
            return collect_logits(
                model, loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )

        split_logits[split] = cached_logits(
            out_dir / f"logits_{split}.npz",
            rows,
            compute,
            force=args.force_recompute,
        )

    # ---------- 扫描 tau ----------
    taus = np.round(
        np.arange(args.tau_min, args.tau_max + args.tau_step / 2, args.tau_step),
        decimals=6,
    )
    if 0.0 not in taus:  # 始终包含 baseline 以便对比
        taus = np.sort(np.append(taus, 0.0))
    sweep_kwargs = dict(
        log_prior=log_prior,
        num_classes=num_classes,
        rare_classes=rare_classes,
        acc1_weight=acc1_weight,
        macro_weight=macro_weight,
    )
    sweeps = {
        split: sweep_tau(logits, y_true, taus, **sweep_kwargs)
        for split, (logits, y_true) in split_logits.items()
    }
    for split, df in sweeps.items():
        path = out_dir / f"tau_sweep_{split}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}")

    # ---------- 在选择划分上挑 tau*，报告 test ----------
    select_df = sweeps[args.select_split]
    best_idx = int(select_df[args.select_metric].idxmax())
    tau_star = float(select_df.loc[best_idx, "tau"])

    def row_at(df: pd.DataFrame, tau: float) -> dict[str, float]:
        idx = int((df["tau"] - tau).abs().idxmin())
        return {k: float(df.loc[idx, k]) for k in ["tau", *METRIC_KEYS]}

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "epoch": int(ckpt["epoch"]),
        "task": task,
        "num_classes": int(num_classes),
        "rare_class_count": int(len(rare_classes)),
        "tau_grid": [float(t) for t in taus],
        "select_split": args.select_split,
        "select_metric": args.select_metric,
        "tau_star": tau_star,
        "max_samples": args.max_samples,
    }
    for split in ("val", "test"):
        baseline = row_at(sweeps[split], 0.0)
        adjusted = row_at(sweeps[split], tau_star)
        summary[split] = {
            "baseline": baseline,
            "adjusted": adjusted,
            "delta": {
                k: adjusted[k] - baseline[k]
                for k in METRIC_KEYS
                if not (np.isnan(adjusted[k]) or np.isnan(baseline[k]))
            },
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {summary_path}")

    if not args.no_plots:
        plot_path = out_dir / "tau_sweep.png"
        save_sweep_plot(
            sweeps["val"], sweeps["test"],
            tau_star=tau_star,
            select_metric=args.select_metric,
            out_path=plot_path,
        )
        print(f"Saved {plot_path}")

    # ---------- 打印对比 ----------
    print(f"\ntau* = {tau_star:.2f}  "
          f"(selected on {args.select_split} by {args.select_metric})")
    test_base = summary["test"]["baseline"]
    test_adj = summary["test"]["adjusted"]
    print(f"{'metric':<12}{'tau=0':>10}{'tau*':>10}{'delta':>10}")
    for key in METRIC_KEYS:
        delta = test_adj[key] - test_base[key]
        print(f"{key:<12}{test_base[key]:>10.4f}{test_adj[key]:>10.4f}{delta:>+10.4f}")


if __name__ == "__main__":
    main()
