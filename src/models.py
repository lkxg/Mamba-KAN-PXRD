"""PXRD 分类任务的模型架构定义。

当前提供两个基准模型：
1. MLPClassifier：简单的全连接多层感知机
2. ResNet1D：适合一维信号（如 PXRD 曲线）的残差网络

这些模型为项目提供了基线性能，后续可以添加更先进的架构（如 Mamba-KAN 混合模型）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """简单的全连接多层感知机分类器。

    将展平后的 PXRD 强度曲线（10824 维向量）直接输入多层全连接网络。

    结构:
        Input(10824) -> Linear -> GELU -> Dropout -> ... -> Linear -> Output(230/7)

    参数:
        in_dim: 输入特征维度（默认 10824，对应 2θ=5°-90° 的采样点数）
        num_classes: 输出类别数（空间群为 230，晶系为 7）
        hidden: 隐藏层维度元组，默认 (1024, 512)
        dropout: Dropout 概率，默认 0.2
    """

    def __init__(
        self,
        in_dim: int = 10824,
        num_classes: int = 230,
        hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        # 构建隐藏层：Linear + GELU 激活 + Dropout
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        # 输出层
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: 输入张量，形状 (batch_size, in_dim)

        返回:
             logits 张量，形状 (batch_size, num_classes)
        """
        return self.net(x)


class _ResBlock1D(nn.Module):
    """一维残差块，包含两个卷积层和快捷连接。

    残差连接使得网络能够训练更深的架构，同时缓解梯度消失问题。

    参数:
        in_ch: 输入通道数
        out_ch: 输出通道数
        stride: 第一个卷积层的步长，用于下采样
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # 第一个卷积层：可进行下采样
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        # 第二个卷积层：保持尺寸
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # 快捷连接（shortcut）：当尺寸或通道数变化时，使用 1x1 卷积调整
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：主路径 + 残差连接。

        参数:
            x: 输入张量，形状 (batch_size, in_ch, length)

        返回:
            输出张量，形状 (batch_size, out_ch, length')
        """
        # 主路径：Conv -> BN -> GELU -> Conv -> BN
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接
        out = out + self.shortcut(x)
        return F.gelu(out)


class ResNet1D(nn.Module):
    """适合一维 PXRD 信号分类的 ResNet 风格模型。

    架构设计：
    - Stem：初始卷积层，负责降采样和特征提取
    - Stages：4 个残差阶段，每个阶段包含多个残差块
    - Head：全局平均池化 + 全连接分类层

    参数:
        num_classes: 输出类别数
        base_channels: 基础通道数，默认 32
        blocks_per_stage: 每个阶段的残差块数量，默认 (2, 2, 2, 2)
        in_channels: 输入通道数，默认 1（单通道 PXRD 曲线）
    """

    def __init__(
        self,
        num_classes: int = 230,
        base_channels: int = 32,
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        c = base_channels

        # Stem：初始特征提取
        # - Conv: kernel=15, stride=4, padding=7 -> 降采样 4 倍
        # - BN + GELU
        # - MaxPool: kernel=3, stride=2 -> 进一步降采样
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 构建残差阶段
        stages: list[nn.Module] = []
        ch = c
        for i, n_blocks in enumerate(blocks_per_stage):
            # 每个阶段开始时通道数翻倍（除第一阶段）
            out_ch = ch if i == 0 else ch * 2
            # 除第一阶段外，每个阶段开始时进行下采样（stride=2）
            stride = 1 if i == 0 else 2
            for b in range(n_blocks):
                # 第一个块可能需要下采样，后续块保持尺寸
                stages.append(_ResBlock1D(
                    ch if b == 0 else out_ch,
                    out_ch,
                    stride=stride if b == 0 else 1
                ))
            ch = out_ch
        self.stages = nn.Sequential(*stages)

        # 分类头：全局平均池化 + 全连接层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: 输入张量
               - 2D: (batch_size, signal_length) 将被扩展为 (batch_size, 1, signal_length)
               - 3D: (batch_size, 1, signal_length)

        返回:
            logits 张量，形状 (batch_size, num_classes)
        """
        # 自动扩展 2D 输入为 3D
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # Stem -> Stages -> Pool -> Flatten -> Head
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


__all__ = ["MLPClassifier", "ResNet1D"]
