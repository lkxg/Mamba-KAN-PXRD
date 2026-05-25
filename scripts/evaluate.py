"""在测试集上评估已训练模型的性能。

使用方法:
    python evaluate.py --checkpoint checkpoints/baseline_mlp/best.pt
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data import PXRDDataset, load_splits
from src.models import MLPClassifier, ResNet1D
from src.training import build_loss, evaluate as eval_loop
from src.utils import set_seed


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
    raise ValueError(f"unknown model: {name!r}")


def main():
    """主评估流程。"""
    # ---------- 命令行参数解析 ----------
    ap = argparse.ArgumentParser(description="评估 PXRD 分类模型")
    ap.add_argument("--checkpoint", required=True, help="模型检查点文件路径")
    args = ap.parse_args()

    # ---------- 加载检查点 ----------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]  # 从检查点恢复训练时的配置
    set_seed(cfg["experiment"].get("seed", 42))

    # ---------- 选择计算设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 加载测试数据 ----------
    splits = load_splits(cfg["data"]["splits_csv"])
    task = cfg["data"]["task"]
    test_ds = PXRDDataset(cfg["data"]["root"], rows=splits["test"], task=task)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # ---------- 加载模型权重 ----------
    model = build_model(cfg, in_dim=test_ds.signal_length,
                        num_classes=test_ds.num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val acc1={ckpt.get('val_acc1', float('nan')):.4f}")

    # ---------- 在测试集上评估 ----------
    loss_fn = build_loss("ce").to(device)
    stats = eval_loop(
        model, test_loader, loss_fn, device,
        use_amp=cfg["train"].get("amp", True) and device.type == "cuda"
    )

    # 打印测试结果
    print(f"Test loss={stats.loss:.4f}  acc1={stats.acc1:.4f}  acc5={stats.acc5:.4f}  "
          f"(N={stats.n})")


if __name__ == "__main__":
    main()
