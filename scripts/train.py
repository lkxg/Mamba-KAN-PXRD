"""模型训练入口脚本。

使用方法:
    python3 scripts/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PXRDDataset, class_counts_for_rows, labels_for_rows, load_splits
from src.models import (
    BiGRUPatchClassifier,
    DualRangePXRDClassifier,
    MLPClassifier,
    PatchTSTClassifier,
    ResNet1D,
)
from src.training import (
    amp_dtype_from_config,
    aux_loss_weights_from_model,
    build_loss,
    configure_backend,
    evaluate,
    rare_classes_from_counts,
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
    if name == "mlp":
        return MLPClassifier(in_dim=in_dim, num_classes=num_classes,
                             **cfg["model"].get("mlp", {}))
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes, **cfg["model"].get("resnet1d", {}))
    if name == "dual_range":
        return DualRangePXRDClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            **cfg["model"].get("dual_range", {}),
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


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    """余弦退火学习率调度器。

    参数:
        step: 当前训练步数（epoch 编号）
        warmup: 预热阶段的 epoch 数
        total: 总训练 epoch 数
        base_lr: 基础学习率

    返回:
        当前步的学习率值

    说明:
        - 在预热阶段，学习率从很小的值线性增加到 base_lr
        - 预热结束后，使用余弦退火公式逐渐降低学习率
        - 余弦公式: lr = base_lr * 0.5 * (1 + cos(pi * progress))
    """
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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

    labels = torch.as_tensor(labels_for_rows(labels_csv, train_rows, task), dtype=torch.long)
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
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


def _wandb_artifact_name(name: str) -> str:
    """Make a conservative W&B artifact name from an experiment/run name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") or "checkpoint"


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


def build_optimizer(cfg: dict, params) -> torch.optim.Optimizer:
    """Build the optimizer selected by config."""
    optim_cfg = cfg.get("optim", {})
    name = optim_cfg.get("name", "adamw").lower()
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
            eps=float(optim_cfg.get("eps", 1e-8)),
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
            eps=float(optim_cfg.get("eps", 1e-8)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(optim_cfg.get("nesterov", False)),
        )
    raise ValueError(f"unknown optimizer: {name!r}")


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
    train_epoch_samples = len(train_sampler) if train_sampler is not None else len(train_ds)
    drop_last = bool(cfg["train"].get("drop_last", True))
    drop_last = drop_last and train_epoch_samples >= batch_size

    # 创建训练和验证数据加载器
    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=drop_last,
        **common,
    )
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

    raw_model = model
    aux_loss_weights = aux_loss_weights_from_model(raw_model)
    if aux_loss_weights:
        print(f"Aux losses: {aux_loss_weights}")
    if cfg.get("performance", {}).get("compile", False) and device.type == "cuda":
        compile_mode = cfg.get("performance", {}).get("compile_mode", "default")
        model = torch.compile(raw_model, mode=compile_mode)
        print(f"torch.compile: enabled mode={compile_mode}")
    elif cfg.get("performance", {}).get("compile", False):
        print("torch.compile: skipped because CUDA is not available")

    # ========== 损失函数构建 ==========
    # 创建损失函数
    loss_fn = build_loss(
        loss_name,
        class_counts=class_counts,
        gamma=cfg["loss"].get("focal_gamma", 2.0),
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
        class_weight_power=cfg["loss"].get("class_weight_power", 0.5),
        class_weight_beta=cfg["loss"].get("class_weight_beta", 0.99),
        logit_adjustment_tau=cfg["loss"].get("logit_adjustment_tau", 1.0),
    ).to(device)
    print(f"Loss: {loss_name}")

    # ========== 优化器和学习率调度器 ==========
    optimizer = build_optimizer(cfg, raw_model.parameters())
    print(f"Optimizer: {cfg.get('optim', {}).get('name', 'adamw')}")
    # 混合精度训练（AMP）：减少显存占用并加速训练
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    amp_dtype = amp_dtype_from_config(cfg["train"].get("amp_dtype", "float16"), device)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler) if use_scaler else None
    print(f"AMP: enabled={use_amp} dtype={amp_dtype} scaler={use_scaler}")

    epochs = cfg["train"]["epochs"]                              # 训练轮数
    warmup_epochs = cfg.get("scheduler", {}).get("warmup_epochs", 0)  # 预热轮数

    # ========== 日志记录设置 ==========
    run_name = cfg["experiment"]["name"]
    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    logging_cfg = cfg.get("logging", {})
    writer = None
    if logging_cfg.get("tensorboard", True):
        writer = SummaryWriter(run_dir)  # TensorBoard 日志
    ckpt_dir = Path(cfg["checkpoint"]["out_dir"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if writer is not None:
        print(f"TensorBoard: {run_dir}")
    print(f"Checkpoints: {ckpt_dir}")

    wandb_run = init_wandb(cfg, run_dir)
    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}")
        if wandb_cfg.get("watch_model", False):
            import wandb

            wandb.watch(
                raw_model,
                log=wandb_cfg.get("watch_log", "gradients"),
                log_freq=int(wandb_cfg.get("watch_log_freq", 100)),
            )

    # ========== 训练循环 ==========
    best_score = -float("inf") if monitor_mode == "max" else float("inf")
    for epoch in range(epochs):
        # 更新学习率（使用余弦退火调度器）
        if cfg["scheduler"]["name"] == "cosine":
            lr = cosine_lr(epoch, warmup_epochs, epochs, cfg["optim"]["lr"])
            for g in optimizer.param_groups:
                g["lr"] = lr
        else:
            lr = cfg["optim"]["lr"]

        t0 = time.time()

        # 训练一个 epoch
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler=scaler,
            grad_clip=cfg["train"].get("grad_clip", 1.0),  # 梯度裁剪阈值
            log_every=cfg["train"].get("log_every", 100),  # 日志打印频率
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            aux_loss_weights=aux_loss_weights,
            rare_classes=rare_classes,
        )
        # 在验证集上评估
        val_stats = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            aux_loss_weights=aux_loss_weights,
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
            f"rare={val_stats.rare_acc1:.4f}",
        ]
        if val_stats.aux_loss:
            extra_parts.append(f"aux={val_stats.aux_loss:.4f}")
        if val_stats.gate_mean is not None:
            extra_parts.append(f"gate={val_stats.gate_mean:.4f}")
        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"lr={lr:.2e}  "
              f"train loss={train_stats.loss:.4f} acc1={train_stats.acc1:.4f}  "
              f"val loss={val_stats.loss:.4f} acc1={val_stats.acc1:.4f} "
              f"acc5={val_stats.acc5:.4f} macro={val_stats.macro_acc1:.4f} "
              f"{' '.join(extra_parts)} "
              f"balanced={balanced_score:.4f}  "
              f"{monitor}={score:.4f}  "
              f"({dt:.0f}s)",
              flush=True)

        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar("lr", lr, epoch)
            writer.add_scalar("train/loss", train_stats.loss, epoch)
            writer.add_scalar("train/acc1", train_stats.acc1, epoch)
            writer.add_scalar("val/loss", val_stats.loss, epoch)
            writer.add_scalar("val/acc1", val_stats.acc1, epoch)
            writer.add_scalar("val/acc5", val_stats.acc5, epoch)
            writer.add_scalar("val/macro_acc1", val_stats.macro_acc1, epoch)
            writer.add_scalar("val/rare_acc1", val_stats.rare_acc1, epoch)
            writer.add_scalar("val/aux_loss", val_stats.aux_loss, epoch)
            writer.add_scalar("val/balanced_acc1_macro", balanced_score, epoch)
            if val_stats.gate_mean is not None:
                writer.add_scalar("val/gate_mean", val_stats.gate_mean, epoch)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_stats.loss,
                "train_acc1": train_stats.acc1,
                "train_acc5": train_stats.acc5,
                "val_loss": val_stats.loss,
                "val_acc1": val_stats.acc1,
                "val_acc5": val_stats.acc5,
                "val_macro_acc1": val_stats.macro_acc1,
                "val_rare_acc1": val_stats.rare_acc1,
                "val_aux_loss": val_stats.aux_loss,
                "val_balanced_acc1_macro": balanced_score,
                "epoch_time_sec": dt,
            } | (
                {"val_gate_mean": val_stats.gate_mean}
                if val_stats.gate_mean is not None
                else {}
            ), step=epoch + 1)

        is_better = score > best_score if monitor_mode == "max" else score < best_score

        # 保存最佳模型（基于配置指定的验证指标）
        if is_better:
            best_score = score

            if wandb_run is not None:
                wandb_run.summary["best_epoch"] = epoch + 1
                wandb_run.summary["best_monitor_score"] = best_score
                wandb_run.summary[f"best_{monitor}"] = best_score
                wandb_run.summary["best_val_acc1"] = val_stats.acc1
                wandb_run.summary["best_val_macro_acc1"] = val_stats.macro_acc1
                wandb_run.summary["best_val_rare_acc1"] = val_stats.rare_acc1
                wandb_run.summary["best_val_balanced_acc1_macro"] = balanced_score
                if val_stats.gate_mean is not None:
                    wandb_run.summary["best_val_gate_mean"] = val_stats.gate_mean

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
                    "val_aux_loss": val_stats.aux_loss,
                    "val_gate_mean": val_stats.gate_mean,
                    "val_balanced_acc1_macro": balanced_score,
                    "rare_classes": sorted(rare_classes),
                    "rare_max_train_count": rare_max_count,
                    "rare_min_train_count": rare_min_count,
                    "monitor": monitor,
                    "monitor_score": score,
                }, best_path)

                if wandb_run is not None and wandb_cfg.get("log_best_checkpoint", False):
                    import wandb

                    artifact = wandb.Artifact(
                        name=_wandb_artifact_name(f"{run_dir.name}-best"),
                        type="model",
                        metadata={
                            "epoch": epoch + 1,
                            "monitor": monitor,
                            "monitor_score": score,
                        },
                    )
                    artifact.add_file(str(best_path))
                    wandb_run.log_artifact(artifact)

    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best {monitor}: {best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
