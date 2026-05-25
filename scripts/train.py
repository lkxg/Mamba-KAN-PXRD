"""模型训练入口脚本。

使用方法:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import PXRDDataset, load_splits
from src.models import MLPClassifier, ResNet1D
from src.training import build_loss, evaluate, train_one_epoch
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
    print(f"Device: {device}")

    # ========== 数据加载 ==========
    # 加载预先划分好的训练/验证/测试集索引
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]

    # 创建 PyTorch Dataset 对象
    train_ds = PXRDDataset(cfg["data"]["root"], rows=splits["train"], task=task)
    val_ds   = PXRDDataset(cfg["data"]["root"], rows=splits["val"],   task=task)

    # DataLoader 公共参数
    common = dict(
        batch_size=cfg["data"]["batch_size"],       # 每批次样本数
        num_workers=cfg["data"]["num_workers"],     # 数据加载线程数
        pin_memory=cfg["data"]["pin_memory"],       # 固定内存加速 GPU 读取
        persistent_workers=cfg["data"]["num_workers"] > 0,  # 保持 worker 进程
    )
    # 创建训练和验证数据加载器
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)

    print(f"Task: {task}  num_classes={train_ds.num_classes}")
    print(f"  train: {len(train_ds):,}  |  val: {len(val_ds):,}")

    # ========== 模型构建 ==========
    # 根据配置创建模型并移动到指定设备
    model = build_model(cfg, in_dim=train_ds.signal_length,
                        num_classes=train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}  ({n_params/1e6:.1f}M params)")

    # ========== 损失函数构建 ==========
    class_counts = None
    # 如果使用加权交叉熵损失，需要统计每个类别的样本数量
    if cfg["loss"]["name"] == "weighted_ce":
        labels_df = pd.read_csv(Path(cfg["data"]["root"]) / "labels.csv")
        if task == "space_group":
            # 空间群分类：230 个类别
            class_counts = (labels_df["space_group"]
                            .value_counts()
                            .reindex(range(1, 231), fill_value=0)
                            .to_numpy())
        else:
            # 晶系分类：7 个类别
            class_counts = (labels_df["crystal_system_id"]
                            .value_counts()
                            .reindex(range(7), fill_value=0)
                            .to_numpy())
    # 创建损失函数
    loss_fn = build_loss(
        cfg["loss"]["name"],
        class_counts=class_counts,
        gamma=cfg["loss"].get("focal_gamma", 2.0),
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
    ).to(device)

    # ========== 优化器和学习率调度器 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],                      # 初始学习率
        weight_decay=cfg["optim"]["weight_decay"],  # 权重衰减（L2 正则化）
        betas=tuple(cfg["optim"]["betas"]),         # Adam 动量参数
    )
    # 混合精度训练（AMP）：减少显存占用并加速训练
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    epochs = cfg["train"]["epochs"]                              # 训练轮数
    warmup_epochs = cfg.get("scheduler", {}).get("warmup_epochs", 0)  # 预热轮数

    # ========== 日志记录设置 ==========
    run_name = cfg["experiment"]["name"]
    run_dir = Path("runs") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(run_dir)  # TensorBoard 日志
    ckpt_dir = Path(cfg["checkpoint"]["out_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard: {run_dir}")

    # ========== 训练循环 ==========
    best_acc1 = 0.0  # 记录最佳验证准确率
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
        )
        # 在验证集上评估
        val_stats = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)
        dt = time.time() - t0

        # 打印训练统计信息
        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"lr={lr:.2e}  "
              f"train loss={train_stats.loss:.4f} acc1={train_stats.acc1:.4f}  "
              f"val loss={val_stats.loss:.4f} acc1={val_stats.acc1:.4f} "
              f"acc5={val_stats.acc5:.4f}  "
              f"({dt:.0f}s)")

        # 记录到 TensorBoard
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("train/loss", train_stats.loss, epoch)
        writer.add_scalar("train/acc1", train_stats.acc1, epoch)
        writer.add_scalar("val/loss", val_stats.loss, epoch)
        writer.add_scalar("val/acc1", val_stats.acc1, epoch)
        writer.add_scalar("val/acc5", val_stats.acc5, epoch)

        # 保存最佳模型（基于验证集 top-1 准确率）
        if cfg["checkpoint"].get("save_best", True) and val_stats.acc1 > best_acc1:
            best_acc1 = val_stats.acc1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "val_acc1": val_stats.acc1,
            }, ckpt_dir / "best.pt")

    writer.close()
    print(f"Best val acc1: {best_acc1:.4f}")


if __name__ == "__main__":
    main()
