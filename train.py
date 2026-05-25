"""Training entry point.

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
    name = cfg["model"]["name"].lower()
    if name == "mlp":
        return MLPClassifier(in_dim=in_dim, num_classes=num_classes,
                             **cfg["model"].get("mlp", {}))
    if name == "resnet1d":
        return ResNet1D(num_classes=num_classes, **cfg["model"].get("resnet1d", {}))
    raise ValueError(f"unknown model: {name!r}")


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ------------------------------------------------------------
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    train_ds = PXRDDataset(cfg["data"]["root"], rows=splits["train"], task=task)
    val_ds   = PXRDDataset(cfg["data"]["root"], rows=splits["val"],   task=task)

    common = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)

    print(f"Task: {task}  num_classes={train_ds.num_classes}")
    print(f"  train: {len(train_ds):,}  |  val: {len(val_ds):,}")

    # ---- Model -----------------------------------------------------------
    model = build_model(cfg, in_dim=train_ds.signal_length,
                        num_classes=train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}  ({n_params/1e6:.1f}M params)")

    # ---- Loss ------------------------------------------------------------
    class_counts = None
    if cfg["loss"]["name"] == "weighted_ce":
        labels_df = pd.read_csv(Path(cfg["data"]["root"]) / "labels.csv")
        if task == "space_group":
            class_counts = (labels_df["space_group"]
                            .value_counts()
                            .reindex(range(1, 231), fill_value=0)
                            .to_numpy())
        else:
            class_counts = (labels_df["crystal_system_id"]
                            .value_counts()
                            .reindex(range(7), fill_value=0)
                            .to_numpy())
    loss_fn = build_loss(
        cfg["loss"]["name"],
        class_counts=class_counts,
        gamma=cfg["loss"].get("focal_gamma", 2.0),
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
    ).to(device)

    # ---- Optim + scheduler ----------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg.get("scheduler", {}).get("warmup_epochs", 0)

    # ---- Logging ---------------------------------------------------------
    run_name = cfg["experiment"]["name"]
    run_dir = Path("runs") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(run_dir)
    ckpt_dir = Path(cfg["checkpoint"]["out_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard: {run_dir}")

    # ---- Train loop ------------------------------------------------------
    best_acc1 = 0.0
    for epoch in range(epochs):
        if cfg["scheduler"]["name"] == "cosine":
            lr = cosine_lr(epoch, warmup_epochs, epochs, cfg["optim"]["lr"])
            for g in optimizer.param_groups:
                g["lr"] = lr
        else:
            lr = cfg["optim"]["lr"]

        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler=scaler,
            grad_clip=cfg["train"].get("grad_clip", 1.0),
            log_every=cfg["train"].get("log_every", 100),
        )
        val_stats = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)
        dt = time.time() - t0

        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"lr={lr:.2e}  "
              f"train loss={train_stats.loss:.4f} acc1={train_stats.acc1:.4f}  "
              f"val loss={val_stats.loss:.4f} acc1={val_stats.acc1:.4f} "
              f"acc5={val_stats.acc5:.4f}  "
              f"({dt:.0f}s)")

        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("train/loss", train_stats.loss, epoch)
        writer.add_scalar("train/acc1", train_stats.acc1, epoch)
        writer.add_scalar("val/loss", val_stats.loss, epoch)
        writer.add_scalar("val/acc1", val_stats.acc1, epoch)
        writer.add_scalar("val/acc5", val_stats.acc5, epoch)

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
