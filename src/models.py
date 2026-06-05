"""PXRD 分类任务的模型架构定义。

当前提供多个基准模型：
1. MLPClassifier：简单的全连接多层感知机
2. ResNet1D：适合一维信号（如 PXRD 曲线）的残差网络
3. BiGRUPatchClassifier：patch 化 PXRD 序列 + 双向 GRU
4. PatchTSTClassifier：patch 化 PXRD 序列 + Transformer Encoder

这些模型为项目提供了基线性能，后续可以添加更先进的架构（如 Mamba-KAN 混合模型）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_patches(signal_length: int, patch_len: int, stride: int) -> int:
    if signal_length <= patch_len:
        return 1
    pad = (stride - ((signal_length - patch_len) % stride)) % stride
    return (signal_length + pad - patch_len) // stride + 1


def _patchify_1d(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """Convert a PXRD curve into non-overlapping or strided patches."""
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError("expected a single PXRD channel")
        x = x.squeeze(1)
    if x.ndim != 2:
        raise ValueError(f"expected input shape (B, L), got {tuple(x.shape)}")

    length = x.shape[-1]
    if length <= patch_len:
        pad = patch_len - length
    else:
        pad = (stride - ((length - patch_len) % stride)) % stride
    if pad:
        x = F.pad(x, (0, pad))
    return x.unfold(dimension=-1, size=patch_len, step=stride).contiguous()


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


class BiGRUPatchClassifier(nn.Module):
    """Patch 化 PXRD 序列的双向 GRU 分类器。

    直接在 10,824 个采样点上运行 RNN 代价过高，因此先把曲线切成 patch，
    再将 patch embedding 作为较短序列输入双向 GRU。
    """

    def __init__(
        self,
        in_dim: int = 10824,
        num_classes: int = 230,
        patch_len: int = 64,
        stride: int = 64,
        d_model: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_norm = nn.LayerNorm(self.patch_len)
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = _patchify_1d(x, self.patch_len, self.stride)
        z = self.patch_embed(self.patch_norm(patches))
        _, h_n = self.gru(z)
        h_n = h_n.view(self.gru.num_layers, 2, x.shape[0], self.gru.hidden_size)
        last = h_n[-1]
        feat = torch.cat([last[0], last[1]], dim=-1)
        return self.head(feat)


class PatchTSTClassifier(nn.Module):
    """PatchTST 风格的一维 PXRD 分类器。

    对单通道 PXRD 曲线进行 patch embedding，再用 Transformer Encoder 建模
    patch 间关系。这里保留 PatchTST 的核心 patch 化思想，用于序列 baseline。
    """

    def __init__(
        self,
        in_dim: int = 10824,
        num_classes: int = 230,
        patch_len: int = 64,
        stride: int = 64,
        d_model: int = 192,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        num_patches = _num_patches(in_dim, self.patch_len, self.stride)

        self.patch_norm = nn.LayerNorm(self.patch_len)
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = _patchify_1d(x, self.patch_len, self.stride)
        z = self.patch_embed(self.patch_norm(patches))
        cls = self.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1)
        if z.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"unexpected patch count: got {z.shape[1] - 1}, "
                f"expected {self.pos_embed.shape[1] - 1}"
            )
        z = z + self.pos_embed
        z = self.encoder(z)
        return self.head(z[:, 0])


__all__ = [
    "MLPClassifier",
    "ResNet1D",
    "BiGRUPatchClassifier",
    "PatchTSTClassifier",
]
