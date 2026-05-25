"""通用辅助函数：配置加载、随机种子设置、分类指标计算。"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# =====================================================================
# 配置加载
# =====================================================================

def load_config(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载配置。

    参数:
        path: YAML 配置文件路径

    返回:
        配置字典

    异常:
        ValueError: 当配置文件不是有效的 YAML 映射时抛出
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {path} must be a YAML mapping (got {type(cfg)})")
    return cfg


# =====================================================================
# 随机种子设置
# =====================================================================

def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。

    调用此函数可以确保实验结果的可复现性。

    参数:
        seed: 随机种子值，默认 42
        deterministic: 是否使用确定性算法
                       - True: CUDNN 使用确定性算法，可能降低性能
                       - False: CUDNN 使用最快的算法（默认）
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# =====================================================================
# 分类指标
# =====================================================================

def topk_accuracy(logits: np.ndarray, target: np.ndarray, k: int = 5) -> float:
    """计算 Top-k 准确率。

    参数:
        logits: 模型预测的未归一化 logits，形状 (N, C)
        target: 真实类别索引，形状 (N,)
        k: Top-k 中的 k 值

    返回:
        Top-k 准确率（0 到 1 之间的浮点数）
    """
    k = min(k, logits.shape[1])
    # 使用 argpartition 快速找到最大的 k 个元素
    top = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    # 检查真实类别是否在这 k 个预测中
    return float((top == target[:, None]).any(axis=1).mean())


def per_class_accuracy(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """计算每个类别的准确率。

    对于没有任何真实样本的类别（测试集中），返回 NaN。

    参数:
        pred: 预测类别索引数组
        target: 真实类别索引数组
        num_classes: 总类别数

    返回:
        各类别的准确率数组，长度为 num_classes
        没有样本的类别对应位置为 NaN
    """
    correct = defaultdict(int)
    total = defaultdict(int)

    # 统计每个类别的正确预测数和总样本数
    for p, t in zip(pred, target):
        total[int(t)] += 1
        if int(p) == int(t):
            correct[int(t)] += 1

    # 计算每个类别的准确率
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        if total[c] > 0:
            out[c] = correct[c] / total[c]
    return out


__all__ = ["load_config", "set_seed", "topk_accuracy", "per_class_accuracy"]
