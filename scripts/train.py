"""模型训练入口脚本。

使用方法:
    python3 scripts/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    PXRDDataset,
    build_augment_from_config,
    class_counts_for_rows,
    labels_for_rows,
    load_splits,
)
from src.models import (
    BiGRUPatchClassifier,
    ConvNeXt1D,
    DualPlaneMambaClassifier,
    PatchTSTClassifier,
    ResNet1D,
    XRDCTMClassifier,
)
from src.training import (
    build_loss_from_config,
    configure_backend,
    evaluate,
    rare_classes_from_counts,
    supervised_contrastive_config,
    train_one_epoch,
)
from src.utils import load_config, set_seed


def build_model(cfg: dict, *, in_dim: int, num_classes: int) -> torch.nn.Module:
    """根据配置创建模型实例。

    参数:
        cfg: 包含模型配置的字典
        in_dim: 输入特征维度（即 PXRD 信号的采样点数）
        num_classes: 分类类别数（空间群为 230，晶系为 7）

    返回:
        PyTorch 模型实例

    异常:
        ValueError: 当模型名称未知时抛出
    """
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
    if name == "xrd_ctm":
        return XRDCTMClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            **cfg["model"].get("xrd_ctm", {}),
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


def cosine_lr(
    step: int,
    warmup: int,
    total: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    """余弦退火学习率调度器。

    参数:
        step: 当前训练步数（epoch 编号）
        warmup: 预热阶段的 epoch 数
        total: 总训练 epoch 数
        base_lr: 基础学习率
        min_lr: 余弦退火阶段的最低学习率

    返回:
        当前步的学习率值

    说明:
        - 在预热阶段，学习率从很小的值线性增加到 base_lr
        - 预热结束后，使用余弦退火公式逐渐降低学习率
        - 余弦公式在 base_lr 和 min_lr 之间插值
    """
    if min_lr < 0 or min_lr > base_lr:
        raise ValueError("scheduler.min_lr must be between 0 and optim.lr")
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def build_balanced_sampler(
    cfg: dict,
    *,
    labels_csv: Path,
    train_rows,
    task: str,
    class_counts,
) -> WeightedRandomSampler | None:
    """构建可选的类均衡采样器。"""
    sampler_cfg = cfg.get("sampler", {})
    name = sampler_cfg.get("name", "none").lower()
    if name in {"none", "shuffle", ""}:
        return None
    if name != "class_balanced":
        raise ValueError(f"unknown sampler: {name!r}")

    labels = torch.tensor(labels_for_rows(labels_csv, train_rows, task), dtype=torch.long)
    counts = torch.tensor(class_counts, dtype=torch.float64)
    power = float(sampler_cfg.get("power", 0.5))
    if power < 0:
        raise ValueError("sampler.power must be non-negative")
    sample_weights = torch.pow(torch.clamp(counts[labels], min=1.0), -power)
    sample_weights = sample_weights / sample_weights.mean()

    num_samples = sampler_cfg.get("num_samples")
    if num_samples is None:
        multiplier = float(sampler_cfg.get("num_samples_multiplier", 1.0))
        num_samples = int(round(len(train_rows) * multiplier))
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("sampler.num_samples must be positive")
    replacement = bool(sampler_cfg.get("replacement", True))
    if not replacement and num_samples > len(train_rows):
        raise ValueError(
            "sampler.num_samples cannot exceed the training set size "
            "when replacement is false"
        )

    generator = torch.Generator()
    generator.manual_seed(int(cfg["experiment"].get("seed", 42)))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )


def metric_value(
    train_stats,
    val_stats,
    monitor: str,
    checkpoint_cfg: dict | None = None,
) -> float:
    """返回 checkpoint 监控指标。"""
    checkpoint_cfg = checkpoint_cfg or {}
    balanced_cfg = checkpoint_cfg.get("balanced_metric", {})
    acc1_weight = float(balanced_cfg.get("acc1_weight", 0.5))
    macro_weight = float(balanced_cfg.get("macro_acc1_weight", 0.5))
    if acc1_weight < 0 or macro_weight < 0:
        raise ValueError("balanced_metric weights must be non-negative")
    balanced_weight_sum = acc1_weight + macro_weight
    if balanced_weight_sum <= 0:
        raise ValueError("at least one balanced_metric weight must be positive")
    balanced_acc1_macro = (
        acc1_weight * val_stats.acc1 + macro_weight * val_stats.macro_acc1
    ) / balanced_weight_sum

    metrics = {
        "train_loss": train_stats.loss,
        "train_acc1": train_stats.acc1,
        "train_acc5": train_stats.acc5,
        "val_loss": val_stats.loss,
        "val_acc1": val_stats.acc1,
        "val_acc5": val_stats.acc5,
        "val_macro_acc1": val_stats.macro_acc1,
        "val_rare_acc1": val_stats.rare_acc1,
        "val_balanced_acc1_macro": balanced_acc1_macro,
    }
    if monitor not in metrics:
        raise ValueError(
            f"unknown checkpoint monitor: {monitor!r}; "
            f"choose one of {sorted(metrics)}"
        )
    return metrics[monitor]


def init_wandb(cfg: dict, run_dir: Path):
    """Initialize an optional W&B run from config."""
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled but `wandb` is not installed. "
            "Run `pip install -r requirements.txt` or set "
            "`logging.wandb.enabled: false`."
        ) from exc

    wandb_dir = Path(wandb_cfg.get("dir", "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = {
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "name": wandb_cfg.get("name") or run_dir.name,
        "group": wandb_cfg.get("group"),
        "job_type": wandb_cfg.get("job_type", "train"),
        "notes": wandb_cfg.get("notes"),
        "tags": wandb_cfg.get("tags"),
        "config": cfg,
        "dir": str(wandb_dir),
        "mode": wandb_cfg.get("mode", "online"),
        "save_code": bool(wandb_cfg.get("save_code", True)),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    run = wandb.init(**init_kwargs)
    run.summary["monitor"] = cfg["checkpoint"].get("monitor", "val_acc1")
    return run


def build_optimizer(
    cfg: dict,
    params,
) -> torch.optim.Optimizer:
    """Build the AdamW optimizer used by all retained configs."""
    optim_cfg = cfg.get("optim", {})
    name = optim_cfg.get("name", "adamw").lower()
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {name!r}; expected 'adamw'")
    return torch.optim.AdamW(
        params,
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=float(optim_cfg.get("eps", 1e-8)),
    )


def main():
    """主训练流程。"""
    # ---------- 命令行参数解析 ----------
    ap = argparse.ArgumentParser(description="训练 PXRD 分类模型")
    ap.add_argument("--config", required=True, help="YAML 配置文件路径")
    args = ap.parse_args()

    # ---------- 加载配置并设置随机种子 ----------
    cfg = load_config(args.config)
    set_seed(cfg["experiment"].get("seed", 42))

    # ---------- 选择计算设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(cfg, device)
    print(f"Device: {device}")

    # ========== 数据加载 ==========
    # 加载预先划分好的训练/验证/测试集索引
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    monitor = cfg["checkpoint"].get("monitor", "val_acc1")
    monitor_mode = cfg["checkpoint"].get("mode", "max")
    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"unknown checkpoint mode: {monitor_mode!r}")

    # 创建 PyTorch Dataset 对象
    train_ds = PXRDDataset(cfg["data"]["root"], rows=splits["train"], task=task)
    val_ds   = PXRDDataset(cfg["data"]["root"], rows=splits["val"],   task=task)
    train_augment = build_augment_from_config(
        cfg["data"].get("augment"),
        signal_length=train_ds.signal_length,
    )
    train_ds.transform = train_augment
    if train_augment is not None:
        print("Augment: lattice_scale/intensity/broaden enabled (train only)")
    if len(train_ds) == 0:
        raise ValueError("train split is empty")
    if monitor.startswith("val_") and len(val_ds) == 0:
        raise ValueError(f"{monitor} requires a non-empty validation split")

    labels_csv = Path(cfg["data"]["root"]) / "labels.csv"
    loss_name = cfg["loss"]["name"].lower()
    sampler_name = cfg.get("sampler", {}).get("name", "none").lower()
    class_counts = class_counts_for_rows(labels_csv, train_ds.rows, task)
    metrics_cfg = cfg.get("metrics", {})
    rare_max_count = int(metrics_cfg.get("rare_max_train_count", 100))
    rare_min_count = int(metrics_cfg.get("rare_min_train_count", 1))
    rare_classes = rare_classes_from_counts(
        class_counts,
        max_count=rare_max_count,
        min_count=rare_min_count,
    )

    train_sampler = build_balanced_sampler(
        cfg,
        labels_csv=labels_csv,
        train_rows=train_ds.rows,
        task=task,
        class_counts=class_counts,
    )

    # DataLoader 公共参数
    common = dict(
        batch_size=cfg["data"]["batch_size"],       # 每批次样本数
        num_workers=cfg["data"]["num_workers"],     # 数据加载线程数
        pin_memory=cfg["data"]["pin_memory"],       # 固定内存加速 GPU 读取
        persistent_workers=cfg["data"]["num_workers"] > 0,  # 保持 worker 进程
    )
    if cfg["data"]["num_workers"] > 0:
        common["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)

    batch_size = int(cfg["data"]["batch_size"])

    def make_train_loader(sampler):
        train_epoch_samples = len(sampler) if sampler is not None else len(train_ds)
        drop_last = bool(cfg["train"].get("drop_last", True))
        drop_last = drop_last and train_epoch_samples >= batch_size
        return DataLoader(
            train_ds,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=drop_last,
            **common,
        )

    # 创建训练和验证数据加载器
    train_loader = make_train_loader(train_sampler)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)

    print(f"Task: {task}  num_classes={train_ds.num_classes}")
    print(f"  train: {len(train_ds):,}  |  val: {len(val_ds):,}")
    print(f"Sampler: {sampler_name}")
    print(
        f"Rare classes: {len(rare_classes)} "
        f"(train count {rare_min_count}-{rare_max_count})"
    )

    # ========== 模型构建 ==========
    # 根据配置创建模型并移动到指定设备
    model = build_model(cfg, in_dim=train_ds.signal_length,
                        num_classes=train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}  ({n_params/1e6:.1f}M params)")
    if hasattr(model, "sa_branch") and getattr(model, "sa_branch") is not None:
        print(f"  SA mixer backend: {model.sa_branch.actual_mamba_backend}")
    if hasattr(model, "wa_branch") and getattr(model, "wa_branch") is not None:
        print(f"  WA mixer backend: {model.wa_branch.actual_mamba_backend}")
    if hasattr(model, "actual_mamba_backend"):
        print(f"  XRD-CTM Mamba backend: {model.actual_mamba_backend}")

    raw_model = model
    if cfg.get("performance", {}).get("compile", False) and device.type == "cuda":
        compile_mode = cfg.get("performance", {}).get("compile_mode", "default")
        model = torch.compile(raw_model, mode=compile_mode)
        print(f"torch.compile: enabled mode={compile_mode}")
    elif cfg.get("performance", {}).get("compile", False):
        print("torch.compile: skipped because CUDA is not available")

    # ========== 损失函数构建 ==========
    # 创建损失函数
    loss_fn = build_loss_from_config(
        cfg["loss"],
        class_counts=class_counts,
    ).to(device)
    print(f"Loss: {loss_name}")

    # ========== 优化器和学习率调度器 ==========
    optimizer = build_optimizer(cfg, raw_model.parameters())
    print(f"Optimizer: {cfg.get('optim', {}).get('name', 'adamw')}")
    print(f"AMP: enabled={device.type == 'cuda'} dtype=torch.bfloat16")

    epochs = cfg["train"]["epochs"]                              # 训练轮数
    warmup_epochs = cfg.get("scheduler", {}).get("warmup_epochs", 0)  # 预热轮数
    base_lr = float(cfg["optim"]["lr"])
    min_lr = float(cfg.get("scheduler", {}).get("min_lr", 0.0))

    loss_cfg = cfg.get("loss", {})
    auxiliary_weight = float(loss_cfg.get("auxiliary_weight", 0.0))
    if auxiliary_weight < 0:
        raise ValueError("loss.auxiliary_weight must be non-negative")
    if auxiliary_weight:
        print(f"Auxiliary branch loss weight: {auxiliary_weight}")
    ldam_drw_start_epoch = loss_cfg.get("ldam_drw_start_epoch")
    if ldam_drw_start_epoch is not None:
        ldam_drw_start_epoch = int(ldam_drw_start_epoch)
        if ldam_drw_start_epoch <= 0:
            raise ValueError("loss.ldam_drw_start_epoch must be positive")
    ldam_drw_active = bool(loss_cfg.get("ldam_use_class_weights", False))
    if hasattr(loss_fn, "set_class_weights"):
        loss_fn.set_class_weights(ldam_drw_active)
        if ldam_drw_start_epoch is not None:
            print(f"LDAM-DRW: class weights start at epoch {ldam_drw_start_epoch}")
    elif ldam_drw_start_epoch is not None:
        raise ValueError("loss.ldam_drw_start_epoch is only valid for LDAM loss")

    # ========== W&B 日志记录设置 ==========
    run_name = cfg["experiment"]["name"]
    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    ckpt_dir = Path(cfg["checkpoint"]["out_dir"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    wandb_run = init_wandb(cfg, run_dir)
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}")

    # ========== 训练循环 ==========
    best_score = -float("inf") if monitor_mode == "max" else float("inf")
    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 15))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_start_epoch = int(early_cfg.get("start_epoch", 0))
    if early_enabled:
        if early_patience <= 0:
            raise ValueError("early_stopping.patience must be positive")
        if early_min_delta < 0:
            raise ValueError("early_stopping.min_delta must be non-negative")
        print(
            "Early stopping: "
            f"monitor={monitor} patience={early_patience} "
            f"min_delta={early_min_delta} start_epoch={early_start_epoch}"
        )
    epochs_without_improvement = 0
    for epoch in range(epochs):
        epoch_num = epoch + 1
        if (
            ldam_drw_start_epoch is not None
            and epoch_num >= ldam_drw_start_epoch
            and not ldam_drw_active
        ):
            loss_fn.set_class_weights(True)
            ldam_drw_active = True
            print(f"LDAM-DRW: enabled class weights at epoch {epoch_num}", flush=True)

        # 更新学习率（使用余弦退火调度器）
        if cfg["scheduler"]["name"] == "cosine":
            lr = cosine_lr(
                epoch,
                warmup_epochs,
                epochs,
                base_lr,
                min_lr,
            )
            for g in optimizer.param_groups:
                g["lr"] = lr
        else:
            lr = base_lr

        contrastive_weight, contrastive_temperature, contrastive_embedding_key = (
            supervised_contrastive_config(loss_cfg, epoch=epoch_num)
        )
        t0 = time.time()

        # 训练一个 epoch
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            grad_clip=cfg["train"].get("grad_clip", 1.0),  # 梯度裁剪阈值
            auxiliary_weight=auxiliary_weight,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            contrastive_embedding_key=contrastive_embedding_key,
            rare_classes=rare_classes,
        )
        # 在验证集上评估
        val_stats = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            rare_classes=rare_classes,
        )
        dt = time.time() - t0

        # 打印训练统计信息
        balanced_score = metric_value(
            train_stats,
            val_stats,
            "val_balanced_acc1_macro",
            cfg["checkpoint"],
        )
        score = metric_value(train_stats, val_stats, monitor, cfg["checkpoint"])
        extra_parts = [
            f"drw={'on' if ldam_drw_active else 'off'}",
            f"rare={val_stats.rare_acc1:.4f}",
        ]
        if train_stats.contrastive_loss:
            extra_parts.append(
                f"supcon={train_stats.contrastive_loss:.4f}"
                f"x{contrastive_weight:.3f}"
            )
        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"lr={lr:.2e}  "
              f"train loss={train_stats.loss:.4f} acc1={train_stats.acc1:.4f}  "
              f"val loss={val_stats.loss:.4f} acc1={val_stats.acc1:.4f} "
              f"acc5={val_stats.acc5:.4f} "
              f"macro={val_stats.macro_acc1:.4f} "
              f"{' '.join(extra_parts)} "
              f"balanced={balanced_score:.4f}  "
              f"{monitor}={score:.4f}  "
              f"({dt:.0f}s)",
              flush=True)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "ldam_drw": int(ldam_drw_active),
                "lr": lr,
                "train_loss": train_stats.loss,
                "train_acc1": train_stats.acc1,
                "train_acc5": train_stats.acc5,
                "train_contrastive_loss": train_stats.contrastive_loss,
                "train_contrastive_weight": contrastive_weight,
                "val_loss": val_stats.loss,
                "val_acc1": val_stats.acc1,
                "val_acc5": val_stats.acc5,
                "val_macro_acc1": val_stats.macro_acc1,
                "val_rare_acc1": val_stats.rare_acc1,
                "val_balanced_acc1_macro": balanced_score,
                "epoch_time_sec": dt,
            }, step=epoch + 1)

        min_delta = early_min_delta if early_enabled else 0.0
        if monitor_mode == "max":
            is_better = score > best_score + min_delta
        else:
            is_better = score < best_score - min_delta

        # 保存最佳模型（基于配置指定的验证指标）
        if is_better:
            best_score = score
            epochs_without_improvement = 0

            if wandb_run is not None:
                wandb_run.summary["best_epoch"] = epoch + 1
                wandb_run.summary["best_monitor_score"] = best_score
                wandb_run.summary[f"best_{monitor}"] = best_score
                wandb_run.summary["best_val_acc1"] = val_stats.acc1
                wandb_run.summary["best_val_acc5"] = val_stats.acc5
                wandb_run.summary["best_val_macro_acc1"] = val_stats.macro_acc1
                wandb_run.summary["best_val_rare_acc1"] = val_stats.rare_acc1
                wandb_run.summary["best_val_balanced_acc1_macro"] = balanced_score

            if cfg["checkpoint"].get("save_best", True):
                best_path = ckpt_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "config": cfg,
                    "val_acc1": val_stats.acc1,
                    "val_acc5": val_stats.acc5,
                    "val_macro_acc1": val_stats.macro_acc1,
                    "val_rare_acc1": val_stats.rare_acc1,
                    "val_balanced_acc1_macro": balanced_score,
                    "rare_classes": sorted(rare_classes),
                    "rare_max_train_count": rare_max_count,
                    "rare_min_train_count": rare_min_count,
                    "monitor": monitor,
                    "monitor_score": score,
                    "ldam_drw_active": ldam_drw_active,
                }, best_path)
        elif early_enabled and epoch + 1 >= early_start_epoch:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no {monitor} improvement for {early_patience} epochs.",
                    flush=True,
                )
                break

    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best {monitor}: {best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
