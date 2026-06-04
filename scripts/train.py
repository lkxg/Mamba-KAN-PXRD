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
from src.models import MLPClassifier, ResNet1D
from src.training import (
    amp_dtype_from_config,
    build_loss,
    configure_backend,
    evaluate,
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
        PyTorch 模型实例（MLPClassifier 或 ResNet1D）

    异常:
        ValueError: 当模型名称未知时抛出
    """
    name = cfg["model"]["name"].lower()
    if name == "mlp":
        return MLPClassifier(in_dim=in_dim, num_classes=num_classes,
                             **cfg["model"].get("mlp", {}))
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes, **cfg["model"].get("resnet1d", {}))
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
    sample_weights = torch.pow(torch.clamp(counts[labels], min=1.0), -power)
    sample_weights = sample_weights / sample_weights.mean()

    num_samples = sampler_cfg.get("num_samples")
    if num_samples is None:
        multiplier = float(sampler_cfg.get("num_samples_multiplier", 1.0))
        num_samples = int(round(len(train_rows) * multiplier))

    generator = torch.Generator()
    generator.manual_seed(int(cfg["experiment"].get("seed", 42)))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(num_samples),
        replacement=bool(sampler_cfg.get("replacement", True)),
        generator=generator,
    )


def metric_value(train_stats, val_stats, monitor: str) -> float:
    """返回 checkpoint 监控指标。"""
    metrics = {
        "train_loss": train_stats.loss,
        "train_acc1": train_stats.acc1,
        "train_acc5": train_stats.acc5,
        "val_loss": val_stats.loss,
        "val_acc1": val_stats.acc1,
        "val_acc5": val_stats.acc5,
        "val_macro_acc1": val_stats.macro_acc1,
    }
    if monitor not in metrics:
        raise ValueError(f"unknown checkpoint monitor: {monitor!r}")
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

    # 创建 PyTorch Dataset 对象
    train_ds = PXRDDataset(cfg["data"]["root"], rows=splits["train"], task=task)
    val_ds   = PXRDDataset(cfg["data"]["root"], rows=splits["val"],   task=task)

    labels_csv = Path(cfg["data"]["root"]) / "labels.csv"
    loss_name = cfg["loss"]["name"].lower()
    needs_class_counts = loss_name in {"weighted_ce", "class_balanced_ce"}
    sampler_name = cfg.get("sampler", {}).get("name", "none").lower()
    class_counts = None
    if needs_class_counts or sampler_name == "class_balanced":
        class_counts = class_counts_for_rows(labels_csv, train_ds.rows, task)

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

    # 创建训练和验证数据加载器
    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)

    print(f"Task: {task}  num_classes={train_ds.num_classes}")
    print(f"  train: {len(train_ds):,}  |  val: {len(val_ds):,}")
    print(f"Sampler: {sampler_name}")

    # ========== 模型构建 ==========
    # 根据配置创建模型并移动到指定设备
    model = build_model(cfg, in_dim=train_ds.signal_length,
                        num_classes=train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}  ({n_params/1e6:.1f}M params)")

    raw_model = model
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
    ).to(device)
    print(f"Loss: {loss_name}")

    # ========== 优化器和学习率调度器 ==========
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg["optim"]["lr"],                      # 初始学习率
        weight_decay=cfg["optim"]["weight_decay"],  # 权重衰减（L2 正则化）
        betas=tuple(cfg["optim"]["betas"]),         # Adam 动量参数
    )
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
    run_dir = Path("runs") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    logging_cfg = cfg.get("logging", {})
    writer = None
    if logging_cfg.get("tensorboard", True):
        writer = SummaryWriter(run_dir)  # TensorBoard 日志
    ckpt_dir = Path(cfg["checkpoint"]["out_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if writer is not None:
        print(f"TensorBoard: {run_dir}")

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
    monitor = cfg["checkpoint"].get("monitor", "val_acc1")
    monitor_mode = cfg["checkpoint"].get("mode", "max")
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
        )
        # 在验证集上评估
        val_stats = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        dt = time.time() - t0

        # 打印训练统计信息
        score = metric_value(train_stats, val_stats, monitor)
        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"lr={lr:.2e}  "
              f"train loss={train_stats.loss:.4f} acc1={train_stats.acc1:.4f}  "
              f"val loss={val_stats.loss:.4f} acc1={val_stats.acc1:.4f} "
              f"acc5={val_stats.acc5:.4f} macro={val_stats.macro_acc1:.4f}  "
              f"{monitor}={score:.4f}  "
              f"({dt:.0f}s)")

        # 记录到 TensorBoard
        if writer is not None:
            writer.add_scalar("lr", lr, epoch)
            writer.add_scalar("train/loss", train_stats.loss, epoch)
            writer.add_scalar("train/acc1", train_stats.acc1, epoch)
            writer.add_scalar("val/loss", val_stats.loss, epoch)
            writer.add_scalar("val/acc1", val_stats.acc1, epoch)
            writer.add_scalar("val/acc5", val_stats.acc5, epoch)
            writer.add_scalar("val/macro_acc1", val_stats.macro_acc1, epoch)

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
                "epoch_time_sec": dt,
            }, step=epoch + 1)

        if monitor_mode == "max":
            is_better = score > best_score
        elif monitor_mode == "min":
            is_better = score < best_score
        else:
            raise ValueError(f"unknown checkpoint mode: {monitor_mode!r}")

        # 保存最佳模型（基于配置指定的验证指标）
        if cfg["checkpoint"].get("save_best", True) and is_better:
            best_score = score
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "config": cfg,
                "val_acc1": val_stats.acc1,
                "val_acc5": val_stats.acc5,
                "val_macro_acc1": val_stats.macro_acc1,
                "monitor": monitor,
                "monitor_score": score,
            }, best_path)

            if wandb_run is not None:
                wandb_run.summary["best_epoch"] = epoch + 1
                wandb_run.summary["best_monitor_score"] = best_score
                wandb_run.summary[f"best_{monitor}"] = best_score
                wandb_run.summary["best_val_acc1"] = val_stats.acc1
                wandb_run.summary["best_val_macro_acc1"] = val_stats.macro_acc1

                if wandb_cfg.get("log_best_checkpoint", False):
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
    print(f"Best {monitor}: {best_score:.4f}")


if __name__ == "__main__":
    main()
