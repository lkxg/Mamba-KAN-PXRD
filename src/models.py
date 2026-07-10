"""PXRD 分类任务的模型架构定义。

当前提供多个基准模型：
1. ResNet1D：适合一维信号（如 PXRD 曲线）的残差网络
2. ConvNeXt1D：标准 ConvNeXt block 的一维 PXRD baseline
3. BiGRUPatchClassifier：patch 化 PXRD 序列 + 双向 GRU
4. PatchTSTClassifier：patch 化 PXRD 序列 + Transformer Encoder
5. DualPlaneMambaClassifier：可配置下采样前端与 Mamba token mixer

这些模型为项目提供了基线性能，后续可以添加更先进的架构（如 Mamba-KAN 混合模型）。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaSSM
    _MambaSSM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional CUDA extension
    _MambaSSM = None
    _MambaSSM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from mamba_ssm import Mamba2 as _Mamba2SSM
    _Mamba2SSM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional CUDA extension
    _Mamba2SSM = None
    _Mamba2SSM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

def _num_patches(signal_length: int, patch_len: int, stride: int) -> int:
    if signal_length <= patch_len:
        return 1
    pad = (stride - ((signal_length - patch_len) % stride)) % stride
    return (signal_length + pad - patch_len) // stride + 1


def _conv1d_out_length(
    length: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    return math.floor(
        (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


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


def _as_tuple(values: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(values, int):
        return (values,)
    return tuple(int(v) for v in values)


def _range_to_slice(
    *,
    start_deg: float,
    end_deg: float,
    signal_length: int,
    theta_min: float,
    theta_max: float,
) -> slice:
    """Map a 2theta range to a stable Python slice over the sampled curve."""
    if signal_length <= 1:
        return slice(0, signal_length)
    if theta_min >= theta_max:
        raise ValueError("theta_min must be smaller than theta_max")
    start = max(float(start_deg), float(theta_min))
    end = min(float(end_deg), float(theta_max))
    if start >= end:
        raise ValueError(
            f"empty 2theta range [{start_deg}, {end_deg}] for "
            f"coverage [{theta_min}, {theta_max}]"
        )

    step = (theta_max - theta_min) / (signal_length - 1)
    left = int(math.floor((start - theta_min) / step))
    right = int(math.ceil((end - theta_min) / step)) + 1
    left = max(0, min(signal_length - 1, left))
    right = max(left + 1, min(signal_length, right))
    return slice(left, right)


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


class _ResidualMambaSSMBlock(nn.Module):
    """LayerNorm + residual wrapper around optional uni/bidirectional Mamba."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        backend: str,
        headdim: int | None = None,
        ngroups: int | None = None,
        chunk_size: int | None = None,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.backend = backend.lower()
        if self.backend in {"auto", "mamba_ssm"}:
            mixer_cls = _MambaSSM
            import_error = _MambaSSM_IMPORT_ERROR
            backend_name = "mamba_ssm"
        elif self.backend in {"mamba2", "mamba2_ssm"}:
            mixer_cls = _Mamba2SSM
            import_error = _Mamba2SSM_IMPORT_ERROR
            backend_name = "mamba2_ssm"
        else:
            raise ValueError(
                "mamba backend must be auto, mamba_ssm, mamba2, or mamba2_ssm"
            )
        if mixer_cls is None:
            raise ImportError(
                f"{backend_name} is not importable"
                + (
                    f": {import_error}"
                    if import_error
                    else ""
                )
            )
        self.norm = nn.LayerNorm(d_model)
        mixer_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if backend_name == "mamba2_ssm":
            if headdim is not None:
                mixer_kwargs["headdim"] = int(headdim)
            if ngroups is not None:
                mixer_kwargs["ngroups"] = int(ngroups)
            if chunk_size is not None:
                mixer_kwargs["chunk_size"] = int(chunk_size)
        self.mixer = mixer_cls(**mixer_kwargs)
        self.bidirectional = bool(bidirectional)
        self.reverse_mixer = (
            mixer_cls(**mixer_kwargs)
            if self.bidirectional
            else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        mixed = self.mixer(z)
        if self.reverse_mixer is not None:
            rev = torch.flip(self.reverse_mixer(torch.flip(z, dims=[1])), dims=[1])
            mixed = 0.5 * (mixed + rev)
        return x + self.dropout(mixed)


class _MambaSequenceMixer(nn.Module):
    """Stacked Mamba blocks backed by mamba-ssm."""

    def __init__(
        self,
        d_model: int,
        *,
        num_layers: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        backend: str = "auto",
        headdim: int | None = None,
        ngroups: int | None = None,
        chunk_size: int | None = None,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.backend = backend.lower()
        self.bidirectional = bool(bidirectional)
        if self.num_layers <= 0:
            self.blocks = nn.ModuleList()
            self.actual_backend = "none"
            return
        if self.backend not in {"auto", "mamba_ssm", "mamba2", "mamba2_ssm"}:
            raise ValueError(
                "mamba backend must be auto, mamba_ssm, mamba2, or mamba2_ssm"
            )

        is_mamba2 = self.backend in {"mamba2", "mamba2_ssm"}
        if is_mamba2:
            backend_name = "mamba2_ssm"
            import_error = _Mamba2SSM_IMPORT_ERROR
            backend_missing = _Mamba2SSM is None
        else:
            backend_name = "mamba_ssm"
            import_error = _MambaSSM_IMPORT_ERROR
            backend_missing = _MambaSSM is None
        if backend_missing:
            raise ImportError(
                f"{backend_name} is not installed. Install mamba-ssm/causal-conv1d "
                "before running configs that request Mamba layers."
                + (
                    f" Import error: {import_error}"
                    if import_error
                    else ""
                )
            )

        blocks: list[nn.Module] = []
        for _ in range(self.num_layers):
            blocks.append(_ResidualMambaSSMBlock(
                d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                backend=self.backend,
                headdim=headdim,
                ngroups=ngroups,
                chunk_size=chunk_size,
                bidirectional=self.bidirectional,
            ))
        self.actual_backend = (
            f"{backend_name}_bidirectional" if self.bidirectional else backend_name
        )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.num_layers > 0:
            x = self.norm(x)
        return x


class _RickerHighFreqBranch1D(nn.Module):
    """Depthwise Ricker filters for lightweight peak/edge enhancement."""

    def __init__(self, channels: int, kernels: Sequence[int] = (7, 15)) -> None:
        super().__init__()
        channels = int(channels)
        kernels = tuple(int(k) for k in kernels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not kernels or min(kernels) <= 0 or any(k % 2 == 0 for k in kernels):
            raise ValueError("Ricker kernels must be positive odd integers")
        self.branches = nn.ModuleList()
        for kernel_size in kernels:
            conv = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
            with torch.no_grad():
                t = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
                sigma = max(1.0, kernel_size / 6.0)
                ricker = (1.0 - (t / sigma) ** 2) * torch.exp(-0.5 * (t / sigma) ** 2)
                ricker = ricker - ricker.mean()
                ricker = ricker / ricker.abs().sum().clamp_min(1.0e-6)
                conv.weight.zero_()
                conv.weight[:, 0, :] = ricker
            self.branches.append(conv)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.branches:
            return x
        high = sum(branch(x) for branch in self.branches) / len(self.branches)
        return x + self.scale.to(dtype=x.dtype) * high


class _HaarWTEBranch1D(nn.Module):
    """Token-level Haar WT/IWT branch paired with the global Mamba path."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = channels
        self.mix = nn.Sequential(
            nn.Conv1d(
                channels * 2,
                channels * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels * 2,
                bias=False,
            ),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    @staticmethod
    def _dwt(x: torch.Tensor) -> tuple[torch.Tensor, int]:
        original_length = x.shape[-1]
        if original_length % 2:
            x = F.pad(x, (0, 1), mode="replicate")
        even = x[..., 0::2]
        odd = x[..., 1::2]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        low = (even + odd) * inv_sqrt2
        high = (even - odd) * inv_sqrt2
        return torch.cat([low, high], dim=1), original_length

    @staticmethod
    def _iwt(coeffs: torch.Tensor, original_length: int) -> torch.Tensor:
        low, high = coeffs.chunk(2, dim=1)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        even = (low + high) * inv_sqrt2
        odd = (low - high) * inv_sqrt2
        x = torch.empty(
            low.shape[0],
            low.shape[1],
            low.shape[-1] * 2,
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        x[..., 0::2] = even
        x[..., 1::2] = odd
        return x[..., :original_length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        coeffs, original_length = self._dwt(seq)
        mixed = self.mix(coeffs)
        restored = self._iwt(mixed, original_length).transpose(1, 2)
        return self.scale.to(dtype=x.dtype) * restored


class _MobileXRDTokenBlock(nn.Module):
    """MobileMamba-style token block with global, local, and identity paths."""

    def __init__(
        self,
        d_model: int,
        *,
        global_channels: int,
        local_channels: int,
        identity_channels: int,
        local_kernels: Sequence[int],
        dropout: float,
        use_wavelet: bool,
        wavelet_kernels: Sequence[int],
        global_wavelet: dict | None,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_dropout: float,
        mamba_backend: str,
        mamba_headdim: int | None,
        mamba_ngroups: int | None,
        mamba_chunk_size: int | None,
        mamba_bidirectional: bool,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        global_channels = int(global_channels)
        local_channels = int(local_channels)
        identity_channels = int(identity_channels)
        if min(d_model, global_channels, local_channels, identity_channels) < 0:
            raise ValueError("MobileXRD channel counts must be non-negative")
        if global_channels + local_channels + identity_channels != d_model:
            raise ValueError(
                "global_channels + local_channels + identity_channels must equal d_model"
            )
        if global_channels <= 0:
            raise ValueError("global_channels must be positive")
        if local_channels <= 0:
            raise ValueError("local_channels must be positive")

        local_kernels = tuple(int(k) for k in local_kernels)
        if not local_kernels or min(local_kernels) <= 0 or any(k % 2 == 0 for k in local_kernels):
            raise ValueError("local_kernels must be positive odd integers")

        self.global_channels = global_channels
        self.local_channels = local_channels
        self.identity_channels = identity_channels
        self.norm = nn.LayerNorm(d_model)
        self.global_mixer = _MambaSequenceMixer(
            global_channels,
            num_layers=1,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            backend=mamba_backend,
            headdim=mamba_headdim,
            ngroups=mamba_ngroups,
            chunk_size=mamba_chunk_size,
            bidirectional=mamba_bidirectional,
        )
        global_wavelet_cfg = dict(global_wavelet or {})
        self.global_wavelet = (
            _HaarWTEBranch1D(
                global_channels,
                kernel_size=int(global_wavelet_cfg.get("kernel_size", 3)),
                init_scale=float(global_wavelet_cfg.get("init_scale", 0.1)),
            )
            if bool(global_wavelet_cfg.get("enabled", False))
            else None
        )
        base = local_channels // len(local_kernels)
        remainder = local_channels % len(local_kernels)
        self.local_splits = [
            base + (1 if idx < remainder else 0)
            for idx in range(len(local_kernels))
        ]
        local_branches: list[nn.Module] = []
        for split_channels, kernel_size in zip(self.local_splits, local_kernels):
            if split_channels <= 0:
                continue
            local_branches.append(nn.Sequential(
                nn.Conv1d(
                    split_channels,
                    split_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=split_channels,
                    bias=False,
                ),
                nn.BatchNorm1d(split_channels),
                nn.GELU(),
            ))
        self.local_branches = nn.ModuleList(local_branches)
        self.local_project = nn.Sequential(
            nn.Conv1d(local_channels, local_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(local_channels),
            nn.GELU(),
        )
        self.wavelet = (
            _RickerHighFreqBranch1D(local_channels, kernels=wavelet_kernels)
            if use_wavelet
            else nn.Identity()
        )
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(float(dropout)),
        )

    @property
    def actual_backend(self) -> str:
        return self.global_mixer.actual_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        g_end = self.global_channels
        l_end = g_end + self.local_channels
        global_tokens = z[..., :g_end]
        local_tokens = z[..., g_end:l_end]
        identity_tokens = z[..., l_end:] if self.identity_channels > 0 else None

        global_out = self.global_mixer(global_tokens)
        if self.global_wavelet is not None:
            global_out = global_out + self.global_wavelet(global_tokens)
        local_seq = local_tokens.transpose(1, 2)
        local_parts = torch.split(local_seq, self.local_splits, dim=1)
        local_seq = torch.cat([
            branch(part)
            for branch, part in zip(self.local_branches, local_parts)
        ], dim=1)
        local_seq = self.local_project(local_seq)
        local_seq = self.wavelet(local_seq)
        local_out = local_seq.transpose(1, 2)

        parts = [global_out, local_out]
        if identity_tokens is not None:
            parts.append(identity_tokens)
        return residual + self.project(torch.cat(parts, dim=-1))


class _MobileXRDTokenMixer(nn.Module):
    """Stacked MobileXRD token blocks inspired by MobileMamba MRFFI."""

    def __init__(
        self,
        d_model: int,
        *,
        num_layers: int,
        global_channels: int = 64,
        local_channels: int = 32,
        identity_channels: int = 32,
        local_kernels: Sequence[int] = (3, 7, 15),
        dropout: float = 0.0,
        use_wavelet: bool = False,
        wavelet_kernels: Sequence[int] = (7, 15),
        global_wavelet: dict | None = None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_backend: str = "mamba2_ssm",
        mamba_headdim: int | None = None,
        mamba_ngroups: int | None = None,
        mamba_chunk_size: int | None = None,
        mamba_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            self.blocks = nn.ModuleList()
            self.actual_backend = "none"
            return
        blocks = [
            _MobileXRDTokenBlock(
                int(d_model),
                global_channels=global_channels,
                local_channels=local_channels,
                identity_channels=identity_channels,
                local_kernels=local_kernels,
                dropout=dropout,
                use_wavelet=use_wavelet,
                wavelet_kernels=wavelet_kernels,
                global_wavelet=global_wavelet,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_dropout=mamba_dropout,
                mamba_backend=mamba_backend,
                mamba_headdim=mamba_headdim,
                mamba_ngroups=mamba_ngroups,
                mamba_chunk_size=mamba_chunk_size,
                mamba_bidirectional=mamba_bidirectional,
            )
            for _ in range(self.num_layers)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.actual_backend = f"mobilexrd_{self.blocks[0].actual_backend}"
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.num_layers > 0:
            x = self.norm(x)
        return x


class _RBFLayer(nn.Module):
    """Compact RBF expansion used by the local KAN head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_grids: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if num_grids < 2:
            raise ValueError("num_grids must be at least 2")
        centers = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features * num_grids, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)
        basis = torch.exp(
            -F.softplus(self.gamma) * (x_norm.unsqueeze(-1) - self.centers).pow(2)
        )
        basis = basis.reshape(*x.shape[:-1], x.shape[-1] * self.centers.numel())
        return self.norm(self.base(x) + self.spline(basis))


class KANHead(nn.Module):
    """GPU-friendly KAN-style classifier head with local RBF basis functions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden: Sequence[int] = (512,),
        num_grids: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in _as_tuple(hidden):
            layers.extend([
                _RBFLayer(prev, int(h), num_grids=num_grids),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = int(h)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AnglePositionEncoding(nn.Module):
    """Absolute 2theta-aware token position encoding."""

    def __init__(
        self,
        token_length: int,
        d_model: int,
        *,
        theta_start: float,
        theta_end: float,
        theta_min: float,
        theta_max: float,
        mode: str = "angle_mlp",
        learned: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        token_length = int(token_length)
        d_model = int(d_model)
        if token_length <= 0 or d_model <= 0:
            raise ValueError("token_length and d_model must be positive")
        if theta_min >= theta_max:
            raise ValueError("theta_min must be smaller than theta_max")
        mode = mode.lower()
        if mode not in {"angle_linear", "angle_mlp", "learned"}:
            raise ValueError("position encoding mode must be angle_linear, angle_mlp, or learned")
        angles = torch.linspace(float(theta_start), float(theta_end), token_length)
        angles = ((angles - float(theta_min)) / (float(theta_max) - float(theta_min))) * 2 - 1
        self.register_buffer("angles", angles.unsqueeze(-1))
        self.mode = mode
        self.angle_proj: nn.Module | None
        if mode == "angle_linear":
            self.angle_proj = nn.Linear(1, d_model)
        elif mode == "angle_mlp":
            self.angle_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.angle_proj = None
        self.learned = nn.Embedding(token_length, d_model) if learned or mode == "learned" else None
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected token shape (B, T, D), got {tuple(tokens.shape)}")
        pos = tokens.new_zeros(tokens.shape[1], tokens.shape[2])
        if self.angle_proj is not None:
            pos = pos + self.angle_proj(self.angles.to(dtype=tokens.dtype, device=tokens.device))
        if self.learned is not None:
            idx = torch.arange(tokens.shape[1], device=tokens.device)
            pos = pos + self.learned(idx).to(dtype=tokens.dtype)
        return tokens + self.dropout(pos.unsqueeze(0))


class _TokenSequencePool(nn.Module):
    """Mean, attention, or gated-attention pooling over token sequences."""

    def __init__(
        self,
        d_model: int,
        *,
        name: str = "mean",
        hidden: int | None = None,
        dropout: float = 0.0,
        residual_mean: bool = False,
        residual_init: float = 0.5,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.name = name.lower()
        if self.name not in {"mean", "attention", "gated_attention"}:
            raise ValueError("pooling name must be mean, attention, or gated_attention")
        self.residual_mean = bool(residual_mean)
        if self.name == "mean":
            self.norm = nn.Identity()
            self.attn = None
            self.residual_logit = None
            return
        if self.residual_mean:
            init = float(residual_init)
            init = min(max(init, 1.0e-4), 1.0 - 1.0e-4)
            self.residual_logit = nn.Parameter(
                torch.logit(torch.tensor(init, dtype=torch.float32))
            )
        else:
            self.residual_logit = None
        self.norm = nn.LayerNorm(d_model)
        hidden = int(hidden or d_model)
        if hidden <= 0:
            raise ValueError("pooling hidden must be positive")
        if self.name == "attention":
            self.attn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )
        else:
            self.value = nn.Linear(d_model, hidden)
            self.gate = nn.Linear(d_model, hidden)
            self.dropout = nn.Dropout(float(dropout))
            self.score = nn.Linear(hidden, 1)
            self.attn = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected token shape (B, T, D), got {tuple(tokens.shape)}")
        if self.name == "mean":
            return tokens.mean(dim=1)
        z = self.norm(tokens)
        if self.name == "attention":
            scores = self.attn(z)
        else:
            scores = self.score(self.dropout(torch.tanh(self.value(z)) * torch.sigmoid(self.gate(z))))
        weights = torch.softmax(scores.float(), dim=1).to(dtype=tokens.dtype)
        pooled = (tokens * weights).sum(dim=1)
        if self.residual_logit is not None:
            mean = tokens.mean(dim=1)
            alpha = torch.sigmoid(self.residual_logit).to(dtype=tokens.dtype)
            pooled = mean + alpha * (pooled - mean)
        return pooled


def _make_head(
    *,
    name: str,
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (512,),
    dropout: float = 0.1,
    kan_grids: int = 8,
) -> nn.Module:
    name = name.lower()
    if name == "linear":
        return nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
    if name == "mlp":
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in _as_tuple(hidden):
            layers.extend([nn.Linear(prev, int(h)), nn.GELU(), nn.Dropout(dropout)])
            prev = int(h)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)
    if name == "kan":
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            KANHead(
                in_dim,
                out_dim,
                hidden=hidden,
                num_grids=kan_grids,
                dropout=dropout,
            ),
        )
    raise ValueError(f"unknown head type: {name!r}")


class _SignalChannelTransform1D(nn.Module):
    """Build optional physics-inspired input channels for PXRD frontends."""

    def __init__(self, mode: str = "intensity", *, peak_window: int = 9) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        peak_window = int(peak_window)
        if peak_window <= 0:
            raise ValueError("peak_window must be positive")
        if peak_window % 2 == 0:
            peak_window += 1
        self.peak_window = peak_window
        if self.mode not in {"intensity", "peak_aware"}:
            raise ValueError("signal channel mode must be intensity or peak_aware")
        self.out_channels = 4 if self.mode == "peak_aware" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected signal shape (B, L), got {tuple(x.shape)}")
        x0 = x.unsqueeze(1)
        if self.mode == "intensity":
            return x0
        # Robust finite-difference features.  They are normalized per sample so
        # derivative channels do not dominate raw intensity.
        dx = F.pad(x[:, 1:] - x[:, :-1], (1, 0)).unsqueeze(1)
        d2 = F.pad(dx[:, :, 1:] - dx[:, :, :-1], (1, 0))
        local_mean = F.avg_pool1d(
            x0, kernel_size=self.peak_window, stride=1, padding=self.peak_window // 2
        )
        peakness = (x0 - local_mean).clamp_min(0.0)
        chans = [x0, dx, d2, peakness]
        normed = []
        for c in chans:
            scale = c.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-6)
            normed.append(c / scale)
        return torch.cat(normed, dim=1)


class _MultiScaleConv1D(nn.Module):
    """Parallel multi-kernel/dilated Conv1d stem for peak-shape extraction."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernels: Sequence[int] = (3, 7, 15),
        dilations: Sequence[int] = (1,),
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        dilations = tuple(int(d) for d in dilations)
        if not kernels or not dilations:
            raise ValueError("multi-scale kernels and dilations must be non-empty")
        branches = []
        n = len(kernels) * len(dilations)
        branch_ch = max(1, int(math.ceil(int(out_ch) / n)))
        for k in kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError("multi-scale kernels must be positive odd integers")
            for d in dilations:
                if d <= 0:
                    raise ValueError("multi-scale dilations must be positive")
                branches.append(nn.Sequential(
                    nn.Conv1d(
                        int(in_ch), branch_ch, kernel_size=k, stride=int(stride),
                        padding=(k // 2) * d, dilation=d, bias=False,
                    ),
                    nn.BatchNorm1d(branch_ch),
                    nn.GELU(),
                ))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv1d(branch_ch * len(branches), int(out_ch), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(out_ch)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class _InceptionBlock1D(nn.Module):
    """InceptionTime/XceptionTime-style 1D residual block."""

    def __init__(
        self,
        channels: int,
        *,
        kernels: Sequence[int] = (9, 19, 39),
        bottleneck: int | None = None,
        depthwise: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        kernels = tuple(int(k) for k in kernels)
        if min(kernels) <= 0 or any(k % 2 == 0 for k in kernels):
            raise ValueError("inception kernels must be positive odd integers")
        bottleneck = int(bottleneck or max(8, channels // 4))
        self.reduce = nn.Conv1d(channels, bottleneck, kernel_size=1, bias=False)
        branch_ch = max(1, channels // (len(kernels) + 1))
        branches = []
        for k in kernels:
            groups = bottleneck if depthwise and branch_ch == bottleneck else 1
            if depthwise:
                branches.append(nn.Sequential(
                    nn.Conv1d(bottleneck, bottleneck, kernel_size=k, padding=k // 2, groups=bottleneck, bias=False),
                    nn.Conv1d(bottleneck, branch_ch, kernel_size=1, bias=False),
                ))
            else:
                branches.append(nn.Conv1d(bottleneck, branch_ch, kernel_size=k, padding=k // 2, bias=False))
        branches.append(nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, branch_ch, kernel_size=1, bias=False),
        ))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.BatchNorm1d(branch_ch * len(branches)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(branch_ch * len(branches), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        outs = [branch(z) for branch in self.branches[:-1]] + [self.branches[-1](x)]
        return F.gelu(x + self.project(torch.cat(outs, dim=1)))


class _BlurPool1D(nn.Module):
    """Small anti-aliased low-pass downsampler."""

    def __init__(self, channels: int, *, stride: int = 2, filt_size: int = 5) -> None:
        super().__init__()
        if filt_size not in {3, 5, 7}:
            raise ValueError("BlurPool filt_size must be 3, 5, or 7")
        coeffs = {
            3: [1.0, 2.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }[filt_size]
        filt = torch.tensor(coeffs, dtype=torch.float32)
        filt = filt / filt.sum()
        self.register_buffer("filt", filt.view(1, 1, -1).repeat(int(channels), 1, 1))
        self.channels = int(channels)
        self.stride = int(stride)
        self.pad = filt_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.filt.to(dtype=x.dtype), stride=self.stride, padding=self.pad, groups=self.channels)


class _AntiAliasConv1D(nn.Module):
    """BlurPool followed by stride-1 convolution to reduce aliasing of narrow PXRD peaks."""

    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int, blur_size: int = 5) -> None:
        super().__init__()
        self.blur = _BlurPool1D(int(in_ch), stride=int(stride), filt_size=int(blur_size)) if int(stride) > 1 else nn.Identity()
        self.conv = nn.Conv1d(int(in_ch), int(out_ch), kernel_size=int(kernel_size), stride=1, padding=int(padding), bias=False)
        self.bn = nn.BatchNorm1d(int(out_ch))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(self.blur(x))))


class _ConvAntiAliasDownsample1D(nn.Module):
    """Stride-1 convolution followed by BlurPool downsampling for learned frontends."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        blur_size: int = 5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            int(in_ch),
            int(out_ch),
            kernel_size=int(kernel_size),
            stride=1,
            padding=int(padding),
            bias=False,
        )
        self.bn = nn.BatchNorm1d(int(out_ch))
        self.act = nn.GELU()
        self.blur = (
            _BlurPool1D(int(out_ch), stride=int(stride), filt_size=int(blur_size))
            if int(stride) > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.act(self.bn(self.conv(x))))


class _WaveletStem1D(nn.Module):
    """Learnable wavelet/scattering-like filter bank stem initialized with Ricker filters."""

    def __init__(self, in_ch: int, out_ch: int, *, kernels: Sequence[int] = (9, 17, 33), stride: int = 2) -> None:
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if min(kernels) <= 0 or any(k % 2 == 0 for k in kernels):
            raise ValueError("wavelet kernels must be positive odd integers")
        branch_ch = max(1, int(math.ceil(int(out_ch) / len(kernels))))
        branches = []
        for k in kernels:
            conv = nn.Conv1d(int(in_ch), branch_ch, kernel_size=k, stride=int(stride), padding=k // 2, bias=False)
            with torch.no_grad():
                t = torch.linspace(-(k // 2), k // 2, k)
                sigma = max(1.0, k / 6.0)
                ricker = (1.0 - (t / sigma) ** 2) * torch.exp(-0.5 * (t / sigma) ** 2)
                ricker = ricker - ricker.mean()
                ricker = ricker / ricker.abs().sum().clamp_min(1.0e-6)
                conv.weight.zero_()
                for o in range(branch_ch):
                    for i in range(int(in_ch)):
                        conv.weight[o, i, :] = ricker
            branches.append(nn.Sequential(conv, nn.BatchNorm1d(branch_ch), nn.GELU()))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv1d(branch_ch * len(branches), int(out_ch), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(out_ch)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class _CustomDownsampleMambaBranch(nn.Module):
    """Configurable PXRD-specific downsampling frontend before a Mamba mixer."""

    def __init__(
        self,
        *,
        branch_length: int,
        d_model: int = 128,
        stem_channels: int = 64,
        mid_channels: int = 128,
        final_channels: int = 256,
        blocks_per_stage: Sequence[int] | int = (2, 2),
        frontend_kind: str = "multiscale_downsample",
        frontend_cfg: dict | None = None,
        token_dropout: float = 0.0,
        branch_dropout: float = 0.1,
        mamba_layers: int = 0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_backend: str = "auto",
        mamba_headdim: int | None = None,
        mamba_ngroups: int | None = None,
        mamba_chunk_size: int | None = None,
        mamba_bidirectional: bool = False,
        position_encoding: dict | None = None,
        pooling: dict | None = None,
        theta_start: float = 5.0,
        theta_end: float = 90.0,
        theta_min: float = 5.0,
        theta_max: float = 90.0,
    ) -> None:
        super().__init__()
        self.branch_length = int(branch_length)
        frontend_kind = str(frontend_kind).lower()
        cfg = dict(frontend_cfg or {})
        blocks = _as_tuple(blocks_per_stage)
        if len(blocks) == 1:
            blocks = blocks * 2
        if len(blocks) != 2:
            raise ValueError("blocks_per_stage must be length 1 or 2")
        stem_channels, mid_channels, final_channels = map(int, (stem_channels, mid_channels, final_channels))
        if min(stem_channels, mid_channels, final_channels, int(d_model)) <= 0:
            raise ValueError("custom downsample channels and d_model must be positive")

        peak_mode = "peak_aware" if frontend_kind == "peak_aware_downsample" else "intensity"
        self.input_transform = _SignalChannelTransform1D(
            peak_mode, peak_window=int(cfg.get("peak_window", 9))
        )
        in_ch = self.input_transform.out_channels

        if frontend_kind == "multiscale_downsample":
            first = _MultiScaleConv1D(
                in_ch, stem_channels,
                kernels=cfg.get("kernels", (3, 7, 15)),
                dilations=cfg.get("dilations", (1, 2)),
                stride=2,
                dropout=float(cfg.get("dropout", 0.0)),
            )
        elif frontend_kind == "inception_downsample":
            first = nn.Sequential(
                nn.Conv1d(in_ch, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(stem_channels),
                nn.GELU(),
            )
        elif frontend_kind == "wavelet_downsample":
            first = _WaveletStem1D(in_ch, stem_channels, kernels=cfg.get("kernels", (9, 17, 33)), stride=2)
        elif frontend_kind == "antialiased_downsample":
            first = _AntiAliasConv1D(in_ch, stem_channels, kernel_size=7, stride=2, padding=3, blur_size=int(cfg.get("blur_size", 5)))
        elif frontend_kind == "peak_aware_downsample":
            first = nn.Sequential(
                nn.Conv1d(in_ch, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(stem_channels),
                nn.GELU(),
            )
        else:
            raise ValueError(f"unknown custom frontend kind: {frontend_kind}")

        layers: list[nn.Module] = [first]
        inception_kernels = cfg.get("inception_kernels", (9, 19, 39))
        depthwise = bool(cfg.get("depthwise", frontend_kind == "inception_downsample"))
        for _ in range(int(blocks[0])):
            if frontend_kind == "inception_downsample":
                layers.append(_InceptionBlock1D(stem_channels, kernels=inception_kernels, depthwise=depthwise, dropout=float(cfg.get("dropout", 0.0))))
            else:
                layers.append(_ResBlock1D(stem_channels, stem_channels, stride=1))
        if frontend_kind == "antialiased_downsample":
            layers.append(_AntiAliasConv1D(stem_channels, mid_channels, kernel_size=4, stride=2, padding=1, blur_size=int(cfg.get("blur_size", 5))))
        else:
            layers.extend([nn.Conv1d(stem_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm1d(mid_channels), nn.GELU()])
        for _ in range(int(blocks[1])):
            if frontend_kind == "inception_downsample":
                layers.append(_InceptionBlock1D(mid_channels, kernels=inception_kernels, depthwise=depthwise, dropout=float(cfg.get("dropout", 0.0))))
            else:
                layers.append(_ResBlock1D(mid_channels, mid_channels, stride=1))
        if frontend_kind == "antialiased_downsample":
            layers.append(_AntiAliasConv1D(mid_channels, final_channels, kernel_size=4, stride=2, padding=1, blur_size=int(cfg.get("blur_size", 5))))
            layers.append(nn.Conv1d(final_channels, int(d_model), kernel_size=1, bias=False))
        else:
            layers.extend([
                nn.Conv1d(mid_channels, final_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(final_channels),
                nn.GELU(),
                nn.Conv1d(final_channels, int(d_model), kernel_size=1, bias=False),
            ])
        self.frontend = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, self.branch_length)
            token_length = int(self.frontend(self.input_transform(dummy)).shape[-1])
        if token_length <= 0:
            raise ValueError("custom frontend produced non-positive token length")
        self.token_length = token_length
        self.token_stride = max(1, int(round(self.branch_length / self.token_length)))
        position_cfg = dict(position_encoding or {})
        self.position_encoding = (
            _AnglePositionEncoding(
                self.token_length, int(d_model), theta_start=float(theta_start),
                theta_end=float(theta_end), theta_min=float(theta_min), theta_max=float(theta_max),
                mode=str(position_cfg.get("mode", "angle_mlp")),
                learned=bool(position_cfg.get("learned", True)),
                dropout=float(position_cfg.get("dropout", 0.0)),
            ) if bool(position_cfg.get("enabled", False)) else None
        )
        self.token_norm = nn.LayerNorm(int(d_model))
        self.token_dropout = nn.Dropout(float(token_dropout))
        self.mixer = _MambaSequenceMixer(
            int(d_model), num_layers=mamba_layers, d_state=mamba_d_state,
            d_conv=mamba_d_conv, expand=mamba_expand, dropout=mamba_dropout,
            backend=mamba_backend, headdim=mamba_headdim, ngroups=mamba_ngroups,
            chunk_size=mamba_chunk_size, bidirectional=mamba_bidirectional,
        )
        self.actual_mamba_backend = self.mixer.actual_backend
        pooling_cfg = dict(pooling or {})
        self.pool = _TokenSequencePool(
            int(d_model), name=str(pooling_cfg.get("name", "mean")),
            hidden=pooling_cfg.get("hidden"), dropout=float(pooling_cfg.get("dropout", 0.0)),
            residual_mean=bool(pooling_cfg.get("residual_mean", False)),
            residual_init=float(pooling_cfg.get("residual_init", 0.5)),
        )
        self.dropout = nn.Dropout(float(branch_dropout))
        self.feature_norm = nn.LayerNorm(int(d_model))
        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected branch input shape (B, L), got {tuple(x.shape)}")
        if x.shape[1] != self.branch_length:
            raise ValueError(f"branch length mismatch: got {x.shape[1]}, expected {self.branch_length}")
        seq = self.frontend(self.input_transform(x))
        if seq.shape[-1] != self.token_length:
            raise RuntimeError(f"token length mismatch after custom downsampling: got {seq.shape[-1]}, expected {self.token_length}")
        tokens = self.token_norm(seq.transpose(1, 2))
        if self.position_encoding is not None:
            tokens = self.position_encoding(tokens)
        tokens = self.token_dropout(tokens)
        tokens = self.mixer(tokens)
        feat = self.pool(tokens)
        return self.feature_norm(self.dropout(feat))


class _LearnedDownsampleMambaBranch(nn.Module):
    """Learned Conv/ResBlock downsampling frontend before a Mamba mixer."""

    def __init__(
        self,
        *,
        branch_length: int,
        d_model: int = 128,
        stem_channels: int = 64,
        mid_channels: int = 128,
        final_channels: int = 256,
        blocks_per_stage: Sequence[int] | int = (2, 2),
        downsample_kernels: Sequence[int] = (7, 4, 4),
        downsample_strides: Sequence[int] = (2, 2, 2),
        downsample_paddings: Sequence[int] = (3, 1, 1),
        downsample_multiscale: dict | None = None,
        downsample_antialias: dict | None = None,
        token_dropout: float = 0.0,
        branch_dropout: float = 0.1,
        mamba_layers: int = 0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_backend: str = "auto",
        mamba_headdim: int | None = None,
        mamba_ngroups: int | None = None,
        mamba_chunk_size: int | None = None,
        mamba_bidirectional: bool = False,
        token_mixer: dict | None = None,
        position_encoding: dict | None = None,
        pooling: dict | None = None,
        theta_start: float = 5.0,
        theta_end: float = 90.0,
        theta_min: float = 5.0,
        theta_max: float = 90.0,
    ) -> None:
        super().__init__()
        if branch_length <= 0:
            raise ValueError("branch_length must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        stem_channels = int(stem_channels)
        mid_channels = int(mid_channels)
        final_channels = int(final_channels)
        if min(stem_channels, mid_channels, final_channels) <= 0:
            raise ValueError("downsample channels must be positive")
        blocks = _as_tuple(blocks_per_stage)
        if len(blocks) == 1:
            blocks = blocks * 2
        if len(blocks) != 2:
            raise ValueError("blocks_per_stage must be length 1 or 2")

        kernels = tuple(int(v) for v in downsample_kernels)
        strides = tuple(int(v) for v in downsample_strides)
        paddings = tuple(int(v) for v in downsample_paddings)
        if not (len(kernels) == len(strides) == len(paddings) == 3):
            raise ValueError(
                "downsample_kernels, downsample_strides, and downsample_paddings "
                "must each contain exactly three integers"
            )
        if min(kernels) <= 0 or min(strides) <= 0 or min(paddings) < 0:
            raise ValueError("downsample kernel/stride/padding values are invalid")
        multiscale_cfg = dict(downsample_multiscale or {})
        use_multiscale_stem = bool(multiscale_cfg.get("enabled", False))
        antialias_cfg = dict(downsample_antialias or {})
        use_antialias = bool(antialias_cfg.get("enabled", False))
        antialias_blur_size = int(antialias_cfg.get("blur_size", 5))

        self.branch_length = int(branch_length)
        length = self.branch_length
        for kernel_size, stride, padding in zip(kernels, strides, paddings):
            length = _conv1d_out_length(
                length, kernel_size=kernel_size, stride=stride, padding=padding
            )
        if length <= 0:
            raise ValueError("downsampled token length must be positive")
        self.token_stride = int(strides[0] * strides[1] * strides[2])
        self.token_length = length

        def make_downsample(in_channels: int, out_channels: int, stage_idx: int) -> nn.Module:
            if use_antialias:
                return _ConvAntiAliasDownsample1D(
                    in_channels,
                    out_channels,
                    kernel_size=kernels[stage_idx],
                    stride=strides[stage_idx],
                    padding=paddings[stage_idx],
                    blur_size=antialias_blur_size,
                )
            return nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernels[stage_idx],
                    stride=strides[stage_idx],
                    padding=paddings[stage_idx],
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )

        if use_multiscale_stem:
            layers: list[nn.Module] = [
                _MultiScaleConv1D(
                    1,
                    stem_channels,
                    kernels=multiscale_cfg.get("kernels", (3, 7, 15)),
                    dilations=multiscale_cfg.get("dilations", (1, 2)),
                    stride=strides[0],
                    dropout=float(multiscale_cfg.get("dropout", 0.0)),
                )
            ]
        else:
            layers = [make_downsample(1, stem_channels, 0)]
        for _ in range(int(blocks[0])):
            layers.append(_ResBlock1D(stem_channels, stem_channels, stride=1))
        layers.append(make_downsample(stem_channels, mid_channels, 1))
        for _ in range(int(blocks[1])):
            layers.append(_ResBlock1D(mid_channels, mid_channels, stride=1))
        layers.extend([
            make_downsample(mid_channels, final_channels, 2),
            nn.Conv1d(final_channels, int(d_model), kernel_size=1, bias=False),
        ])
        self.frontend = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, self.branch_length)
            actual_length = int(self.frontend(dummy.unsqueeze(1)).shape[-1])
        if actual_length <= 0:
            raise ValueError("learned downsample frontend produced non-positive token length")
        self.token_length = actual_length
        position_cfg = dict(position_encoding or {})
        self.position_encoding = (
            _AnglePositionEncoding(
                self.token_length,
                int(d_model),
                theta_start=float(theta_start),
                theta_end=float(theta_end),
                theta_min=float(theta_min),
                theta_max=float(theta_max),
                mode=str(position_cfg.get("mode", "angle_mlp")),
                learned=bool(position_cfg.get("learned", True)),
                dropout=float(position_cfg.get("dropout", 0.0)),
            )
            if bool(position_cfg.get("enabled", False))
            else None
        )
        self.token_norm = nn.LayerNorm(int(d_model))
        self.token_dropout = nn.Dropout(float(token_dropout))
        token_mixer_cfg = dict(token_mixer or {})
        token_mixer_name = str(token_mixer_cfg.get("name", "mamba")).lower()
        if token_mixer_name in {"mamba", "full_mamba"}:
            self.mixer = _MambaSequenceMixer(
                int(d_model),
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=mamba_dropout,
                backend=mamba_backend,
                headdim=mamba_headdim,
                ngroups=mamba_ngroups,
                chunk_size=mamba_chunk_size,
                bidirectional=mamba_bidirectional,
            )
        elif token_mixer_name in {"mobilexrd", "mobilexrd_lite", "partial_mamba"}:
            self.mixer = _MobileXRDTokenMixer(
                int(d_model),
                num_layers=mamba_layers,
                global_channels=int(token_mixer_cfg.get("global_channels", 64)),
                local_channels=int(token_mixer_cfg.get("local_channels", 32)),
                identity_channels=int(token_mixer_cfg.get("identity_channels", 32)),
                local_kernels=token_mixer_cfg.get("local_kernels", (3, 7, 15)),
                dropout=float(token_mixer_cfg.get("dropout", mamba_dropout)),
                use_wavelet=bool(token_mixer_cfg.get("use_wavelet", False)),
                wavelet_kernels=token_mixer_cfg.get("wavelet_kernels", (7, 15)),
                global_wavelet=token_mixer_cfg.get("global_wavelet"),
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_dropout=mamba_dropout,
                mamba_backend=mamba_backend,
                mamba_headdim=mamba_headdim,
                mamba_ngroups=mamba_ngroups,
                mamba_chunk_size=mamba_chunk_size,
                mamba_bidirectional=mamba_bidirectional,
            )
        else:
            raise ValueError(
                "downsample_token_mixer.name must be mamba, mobilexrd, "
                "mobilexrd_lite, or partial_mamba"
            )
        self.actual_mamba_backend = self.mixer.actual_backend
        pooling_cfg = dict(pooling or {})
        self.pool = _TokenSequencePool(
            int(d_model),
            name=str(pooling_cfg.get("name", "mean")),
            hidden=pooling_cfg.get("hidden"),
            dropout=float(pooling_cfg.get("dropout", 0.0)),
            residual_mean=bool(pooling_cfg.get("residual_mean", False)),
            residual_init=float(pooling_cfg.get("residual_init", 0.5)),
        )
        self.dropout = nn.Dropout(float(branch_dropout))
        self.feature_norm = nn.LayerNorm(int(d_model))
        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected branch input shape (B, L), got {tuple(x.shape)}")
        if x.shape[1] != self.branch_length:
            raise ValueError(
                f"branch length mismatch: got {x.shape[1]}, expected {self.branch_length}"
            )

        seq = self.frontend(x.unsqueeze(1))
        if seq.shape[-1] != self.token_length:
            raise RuntimeError(
                f"token length mismatch after learned downsampling: got {seq.shape[-1]}, "
                f"expected {self.token_length}"
            )
        tokens = seq.transpose(1, 2)
        tokens = self.token_norm(tokens)
        if self.position_encoding is not None:
            tokens = self.position_encoding(tokens)
        tokens = self.token_dropout(tokens)
        tokens = self.mixer(tokens)
        feat = self.pool(tokens)
        return self.feature_norm(self.dropout(feat))


class _ChannelLayerNorm1D(nn.Module):
    """LayerNorm over channels for tensors shaped (B, C, L)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _ConvNeXtBlock1D(nn.Module):
    """A compact ConvNeXt-style block for one-dimensional PXRD features."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 1.0e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        expansion = int(expansion)
        if channels <= 0:
            raise ValueError("ConvNeXt channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ConvNeXt kernel_size must be a positive odd integer")
        if expansion <= 0:
            raise ValueError("ConvNeXt expansion must be positive")

        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, expansion * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * channels, channels)
        self.gamma = (
            nn.Parameter(float(layer_scale_init) * torch.ones(channels))
            if float(layer_scale_init) > 0
            else None
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = self.dropout(x).transpose(1, 2)
        return residual + x


class ConvNeXt1D(nn.Module):
    """Standard ConvNeXt-style classifier adapted to one-dimensional PXRD curves."""

    def __init__(
        self,
        num_classes: int = 230,
        in_channels: int = 1,
        base_channels: int = 64,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] | None = None,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 1.0e-6,
        block_dropout: float = 0.0,
        head_dropout: float = 0.0,
        head_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        depths = _as_tuple(depths)
        if dims is None:
            dims = tuple(int(base_channels) * (2 ** i) for i in range(len(depths)))
        else:
            dims = _as_tuple(dims)
        if len(depths) != len(dims):
            raise ValueError("ConvNeXt1D depths and dims must have the same length")
        if not depths:
            raise ValueError("ConvNeXt1D must contain at least one stage")
        if min(depths) < 0 or min(dims) <= 0:
            raise ValueError("ConvNeXt1D depths must be non-negative and dims positive")

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    int(in_channels),
                    dims[0],
                    kernel_size=4,
                    stride=4,
                ),
                _ChannelLayerNorm1D(dims[0]),
            )
        )
        for idx in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    _ChannelLayerNorm1D(dims[idx]),
                    nn.Conv1d(dims[idx], dims[idx + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList([
            nn.Sequential(*[
                _ConvNeXtBlock1D(
                    dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale_init=layer_scale_init,
                    dropout=block_dropout,
                )
                for _ in range(depth)
            ])
            for dim, depth in zip(dims, depths)
        ])
        self.norm = nn.LayerNorm(dims[-1])
        self.head_dropout = nn.Dropout(float(head_dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))
        self.apply(self._init_weights)
        self.head.weight.data.mul_(float(head_init_scale))
        self.head.bias.data.mul_(float(head_init_scale))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"expected (B, L) or (B, C, L), got {tuple(x.shape)}")
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        x = x.mean(dim=-1)
        x = self.norm(x)
        return self.head(self.head_dropout(x))


class _ConvNeXtDownsampleMambaBranch(nn.Module):
    """ConvNeXt-style downsampling frontend before a Mamba mixer."""

    def __init__(
        self,
        *,
        branch_length: int,
        d_model: int = 128,
        stem_channels: int = 64,
        mid_channels: int = 128,
        final_channels: int = 256,
        blocks_per_stage: Sequence[int] | int = (2, 2),
        token_dropout: float = 0.0,
        branch_dropout: float = 0.1,
        mamba_layers: int = 0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_backend: str = "auto",
        mamba_headdim: int | None = None,
        mamba_ngroups: int | None = None,
        mamba_chunk_size: int | None = None,
        mamba_bidirectional: bool = False,
        position_encoding: dict | None = None,
        pooling: dict | None = None,
        theta_start: float = 5.0,
        theta_end: float = 90.0,
        theta_min: float = 5.0,
        theta_max: float = 90.0,
        convnext: dict | None = None,
    ) -> None:
        super().__init__()
        if branch_length <= 0:
            raise ValueError("branch_length must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        stem_channels = int(stem_channels)
        mid_channels = int(mid_channels)
        final_channels = int(final_channels)
        if min(stem_channels, mid_channels, final_channels) <= 0:
            raise ValueError("downsample channels must be positive")
        blocks = _as_tuple(blocks_per_stage)
        if len(blocks) == 1:
            blocks = blocks * 2
        if len(blocks) != 2:
            raise ValueError("blocks_per_stage must be length 1 or 2")

        convnext_cfg = dict(convnext or {})
        block_kernel = int(convnext_cfg.get("kernel_size", 7))
        expansion = int(convnext_cfg.get("expansion", 4))
        layer_scale_init = float(convnext_cfg.get("layer_scale_init", 1.0e-6))
        block_dropout = float(convnext_cfg.get("block_dropout", 0.0))

        self.branch_length = int(branch_length)
        length = self.branch_length
        length = _conv1d_out_length(length, kernel_size=7, stride=2, padding=3)
        length = _conv1d_out_length(length, kernel_size=4, stride=2, padding=1)
        length = _conv1d_out_length(length, kernel_size=4, stride=2, padding=1)
        if length <= 0:
            raise ValueError("downsampled token length must be positive")
        self.token_stride = 8
        self.token_length = length

        layers: list[nn.Module] = [
            nn.Conv1d(1, stem_channels, kernel_size=7, stride=2, padding=3),
            _ChannelLayerNorm1D(stem_channels),
        ]
        for _ in range(int(blocks[0])):
            layers.append(
                _ConvNeXtBlock1D(
                    stem_channels,
                    kernel_size=block_kernel,
                    expansion=expansion,
                    layer_scale_init=layer_scale_init,
                    dropout=block_dropout,
                )
            )
        layers.extend([
            _ChannelLayerNorm1D(stem_channels),
            nn.Conv1d(stem_channels, mid_channels, kernel_size=4, stride=2, padding=1),
        ])
        for _ in range(int(blocks[1])):
            layers.append(
                _ConvNeXtBlock1D(
                    mid_channels,
                    kernel_size=block_kernel,
                    expansion=expansion,
                    layer_scale_init=layer_scale_init,
                    dropout=block_dropout,
                )
            )
        layers.extend([
            _ChannelLayerNorm1D(mid_channels),
            nn.Conv1d(mid_channels, final_channels, kernel_size=4, stride=2, padding=1),
            _ChannelLayerNorm1D(final_channels),
            nn.Conv1d(final_channels, int(d_model), kernel_size=1),
        ])
        self.frontend = nn.Sequential(*layers)
        position_cfg = dict(position_encoding or {})
        self.position_encoding = (
            _AnglePositionEncoding(
                self.token_length,
                int(d_model),
                theta_start=float(theta_start),
                theta_end=float(theta_end),
                theta_min=float(theta_min),
                theta_max=float(theta_max),
                mode=str(position_cfg.get("mode", "angle_mlp")),
                learned=bool(position_cfg.get("learned", True)),
                dropout=float(position_cfg.get("dropout", 0.0)),
            )
            if bool(position_cfg.get("enabled", False))
            else None
        )
        self.token_norm = nn.LayerNorm(int(d_model))
        self.token_dropout = nn.Dropout(float(token_dropout))
        self.mixer = _MambaSequenceMixer(
            int(d_model),
            num_layers=mamba_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            backend=mamba_backend,
            headdim=mamba_headdim,
            ngroups=mamba_ngroups,
            chunk_size=mamba_chunk_size,
            bidirectional=mamba_bidirectional,
        )
        self.actual_mamba_backend = self.mixer.actual_backend
        pooling_cfg = dict(pooling or {})
        self.pool = _TokenSequencePool(
            int(d_model),
            name=str(pooling_cfg.get("name", "mean")),
            hidden=pooling_cfg.get("hidden"),
            dropout=float(pooling_cfg.get("dropout", 0.0)),
            residual_mean=bool(pooling_cfg.get("residual_mean", False)),
            residual_init=float(pooling_cfg.get("residual_init", 0.5)),
        )
        self.dropout = nn.Dropout(float(branch_dropout))
        self.feature_norm = nn.LayerNorm(int(d_model))
        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected branch input shape (B, L), got {tuple(x.shape)}")
        if x.shape[1] != self.branch_length:
            raise ValueError(
                f"branch length mismatch: got {x.shape[1]}, expected {self.branch_length}"
            )
        seq = self.frontend(x.unsqueeze(1))
        if seq.shape[-1] != self.token_length:
            raise RuntimeError(
                f"token length mismatch after ConvNeXt downsampling: got {seq.shape[-1]}, "
                f"expected {self.token_length}"
            )
        tokens = seq.transpose(1, 2)
        tokens = self.token_norm(tokens)
        if self.position_encoding is not None:
            tokens = self.position_encoding(tokens)
        tokens = self.token_dropout(tokens)
        tokens = self.mixer(tokens)
        feat = self.pool(tokens)
        return self.feature_norm(self.dropout(feat))


class DualPlaneMambaClassifier(nn.Module):
    """PXRD range classifier with configurable downsampling and token mixing."""

    def __init__(
        self,
        in_dim: int = 10824,
        num_classes: int = 230,
        theta_min: float = 5.0,
        theta_max: float = 90.0,
        sa_range: Sequence[float] = (5.0, 15.0),
        wa_range: Sequence[float] = (10.0, 90.0),
        use_sa: bool = True,
        use_wa: bool = True,
        frontend: str = "learned_downsample",
        sa_frontend: str | None = None,
        wa_frontend: str | None = None,
        d_model: int = 128,
        sa_d_model: int | None = None,
        wa_d_model: int | None = None,
        downsample_channels: Sequence[int] = (64, 128, 256),
        downsample_kernels: Sequence[int] = (7, 4, 4),
        downsample_strides: Sequence[int] = (2, 2, 2),
        downsample_paddings: Sequence[int] = (3, 1, 1),
        downsample_blocks_per_stage: Sequence[int] | int = (2, 2),
        downsample_token_mixer: dict | None = None,
        downsample_convnext: dict | None = None,
        downsample_position_encoding: dict | None = None,
        downsample_pooling: dict | None = None,
        downsample_multiscale: dict | None = None,
        downsample_inception: dict | None = None,
        downsample_peak_aware: dict | None = None,
        downsample_wavelet: dict | None = None,
        downsample_antialias: dict | None = None,
        token_dropout: float = 0.0,
        branch_dropout: float = 0.1,
        fusion: str = "gated",
        fusion_dim: int | None = None,
        gate_hidden: Sequence[int] = (256,),
        gate_groups: int = 16,
        gate_dropout: float = 0.1,
        kan_grids: int = 8,
        head: str = "mlp",
        head_hidden: Sequence[int] = (512,),
        head_dropout: float = 0.1,
        projection_head: str = "kan",
        projection_dim: int = 0,
        projection_hidden: Sequence[int] = (256,),
        projection_dropout: float = 0.1,
        projection_normalize: bool = True,
        mamba: dict | None = None,
    ) -> None:
        super().__init__()
        if not use_sa and not use_wa:
            raise ValueError("at least one branch must be enabled")
        if len(sa_range) != 2 or len(wa_range) != 2:
            raise ValueError("sa_range and wa_range must be [start, end]")

        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.use_sa = bool(use_sa)
        self.use_wa = bool(use_wa)
        self.frontend = frontend.lower()
        self.sa_frontend = (sa_frontend.lower() if sa_frontend else self.frontend)
        self.wa_frontend = (wa_frontend.lower() if wa_frontend else self.frontend)
        self.fusion = fusion.lower()
        self.projection_normalize = bool(projection_normalize)
        sa_d_model = int(d_model if sa_d_model is None else sa_d_model)
        wa_d_model = int(d_model if wa_d_model is None else wa_d_model)
        if sa_d_model <= 0 or wa_d_model <= 0:
            raise ValueError("sa_d_model and wa_d_model must be positive")

        self.sa_slice = _range_to_slice(
            start_deg=float(sa_range[0]),
            end_deg=float(sa_range[1]),
            signal_length=self.in_dim,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
        )
        self.wa_slice = _range_to_slice(
            start_deg=float(wa_range[0]),
            end_deg=float(wa_range[1]),
            signal_length=self.in_dim,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
        )

        mamba_cfg = dict(mamba or {})
        mamba_backend = mamba_cfg.get("backend", "auto")
        common_mamba = dict(
            mamba_d_state=int(mamba_cfg.get("d_state", 16)),
            mamba_d_conv=int(mamba_cfg.get("d_conv", 4)),
            mamba_expand=int(mamba_cfg.get("expand", 2)),
            mamba_dropout=float(mamba_cfg.get("dropout", 0.0)),
            mamba_backend=mamba_backend,
            mamba_headdim=(
                int(mamba_cfg["headdim"]) if "headdim" in mamba_cfg else None
            ),
            mamba_ngroups=(
                int(mamba_cfg["ngroups"]) if "ngroups" in mamba_cfg else None
            ),
            mamba_chunk_size=(
                int(mamba_cfg["chunk_size"]) if "chunk_size" in mamba_cfg else None
            ),
            mamba_bidirectional=bool(mamba_cfg.get("bidirectional", False)),
        )
        sa_layers = int(mamba_cfg.get("sa_layers", 0))
        wa_layers = int(mamba_cfg.get("wa_layers", 0))
        _valid_frontends = {
            "learned_downsample",
            "convnext_downsample",
            "multiscale_downsample",
            "inception_downsample",
            "peak_aware_downsample",
            "wavelet_downsample",
            "antialiased_downsample",
        }
        if self.frontend not in _valid_frontends:
            raise ValueError(f"unknown frontend: {self.frontend!r}")
        if self.sa_frontend not in _valid_frontends:
            raise ValueError(f"unknown sa_frontend: {self.sa_frontend!r}")
        if self.wa_frontend not in _valid_frontends:
            raise ValueError(f"unknown wa_frontend: {self.wa_frontend!r}")

        def make_branch(
            *,
            branch_length: int,
            branch_d_model: int,
            theta_range: Sequence[float],
            mamba_layers: int,
            frontend: str,
        ) -> nn.Module:
            if frontend == "learned_downsample":
                channels = _as_tuple(downsample_channels)
                if len(channels) != 3:
                    raise ValueError("downsample_channels must be [stem, mid, final]")
                return _LearnedDownsampleMambaBranch(
                    branch_length=branch_length,
                    d_model=branch_d_model,
                    stem_channels=channels[0],
                    mid_channels=channels[1],
                    final_channels=channels[2],
                    blocks_per_stage=downsample_blocks_per_stage,
                    downsample_kernels=downsample_kernels,
                    downsample_strides=downsample_strides,
                    downsample_paddings=downsample_paddings,
                    downsample_multiscale=downsample_multiscale,
                    downsample_antialias=downsample_antialias,
                    token_dropout=token_dropout,
                    branch_dropout=branch_dropout,
                    mamba_layers=mamba_layers,
                    token_mixer=downsample_token_mixer,
                    position_encoding=downsample_position_encoding,
                    pooling=downsample_pooling,
                    theta_start=float(theta_range[0]),
                    theta_end=float(theta_range[1]),
                    theta_min=self.theta_min,
                    theta_max=self.theta_max,
                    **common_mamba,
                )
            if frontend == "convnext_downsample":
                channels = _as_tuple(downsample_channels)
                if len(channels) != 3:
                    raise ValueError("downsample_channels must be [stem, mid, final]")
                return _ConvNeXtDownsampleMambaBranch(
                    branch_length=branch_length,
                    d_model=branch_d_model,
                    stem_channels=channels[0],
                    mid_channels=channels[1],
                    final_channels=channels[2],
                    blocks_per_stage=downsample_blocks_per_stage,
                    token_dropout=token_dropout,
                    branch_dropout=branch_dropout,
                    mamba_layers=mamba_layers,
                    position_encoding=downsample_position_encoding,
                    pooling=downsample_pooling,
                    theta_start=float(theta_range[0]),
                    theta_end=float(theta_range[1]),
                    theta_min=self.theta_min,
                    theta_max=self.theta_max,
                    convnext=downsample_convnext,
                    **common_mamba,
                )
            if frontend in {
                "multiscale_downsample",
                "inception_downsample",
                "peak_aware_downsample",
                "wavelet_downsample",
                "antialiased_downsample",
            }:
                channels = _as_tuple(downsample_channels)
                if len(channels) != 3:
                    raise ValueError("downsample_channels must be [stem, mid, final]")
                custom_cfg_map = {
                    "multiscale_downsample": downsample_multiscale,
                    "inception_downsample": downsample_inception,
                    "peak_aware_downsample": downsample_peak_aware,
                    "wavelet_downsample": downsample_wavelet,
                    "antialiased_downsample": downsample_antialias,
                }
                return _CustomDownsampleMambaBranch(
                    branch_length=branch_length,
                    d_model=branch_d_model,
                    stem_channels=channels[0],
                    mid_channels=channels[1],
                    final_channels=channels[2],
                    blocks_per_stage=downsample_blocks_per_stage,
                    frontend_kind=frontend,
                    frontend_cfg=custom_cfg_map[frontend],
                    token_dropout=token_dropout,
                    branch_dropout=branch_dropout,
                    mamba_layers=mamba_layers,
                    position_encoding=downsample_position_encoding,
                    pooling=downsample_pooling,
                    theta_start=float(theta_range[0]),
                    theta_end=float(theta_range[1]),
                    theta_min=self.theta_min,
                    theta_max=self.theta_max,
                    **common_mamba,
                )
            raise ValueError(f"unsupported frontend: {frontend!r}")

        if self.use_sa:
            self.sa_branch = make_branch(
                branch_length=self.sa_slice.stop - self.sa_slice.start,
                branch_d_model=sa_d_model,
                theta_range=sa_range,
                mamba_layers=sa_layers,
                frontend=self.sa_frontend,
            )
            sa_dim = self.sa_branch.out_dim
        else:
            self.sa_branch = None
            sa_dim = 0

        if self.use_wa:
            self.wa_branch = make_branch(
                branch_length=self.wa_slice.stop - self.wa_slice.start,
                branch_d_model=wa_d_model,
                theta_range=wa_range,
                mamba_layers=wa_layers,
                frontend=self.wa_frontend,
            )
            wa_dim = self.wa_branch.out_dim
        else:
            self.wa_branch = None
            wa_dim = 0

        if self.fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be concat or gated")

        if self.use_sa and self.use_wa:
            aligned_dim = int(fusion_dim or max(sa_dim, wa_dim))
            self.sa_align = nn.Linear(sa_dim, aligned_dim) if sa_dim != aligned_dim else nn.Identity()
            self.wa_align = nn.Linear(wa_dim, aligned_dim) if wa_dim != aligned_dim else nn.Identity()
            if self.fusion == "concat":
                fused_dim = aligned_dim * 2
                self.gate = None
                self.gate_groups = 0
            else:
                gate_groups = int(gate_groups)
                if gate_groups <= 0:
                    raise ValueError("gate_groups must be positive")
                self.gate_groups = min(gate_groups, aligned_dim)
                gate_in = aligned_dim * 4
                hidden = _as_tuple(gate_hidden)
                layers: list[nn.Module] = [nn.LayerNorm(gate_in)]
                prev = gate_in
                for h in hidden:
                    layers.extend([
                        nn.Linear(prev, int(h)),
                        nn.GELU(),
                        nn.Dropout(gate_dropout),
                    ])
                    prev = int(h)
                layers.append(nn.Linear(prev, self.gate_groups))
                self.gate = nn.Sequential(*layers)
                fused_dim = aligned_dim
            self.branch_feature_dim = aligned_dim
        else:
            only_dim = sa_dim if self.use_sa else wa_dim
            aligned_dim = int(fusion_dim or only_dim)
            self.sa_align = nn.Linear(sa_dim, aligned_dim) if self.use_sa and sa_dim != aligned_dim else nn.Identity()
            self.wa_align = nn.Linear(wa_dim, aligned_dim) if self.use_wa and wa_dim != aligned_dim else nn.Identity()
            self.gate = None
            self.gate_groups = 0
            self.branch_feature_dim = aligned_dim
            fused_dim = aligned_dim

        self.head = _make_head(
            name=head,
            in_dim=fused_dim,
            out_dim=self.num_classes,
            hidden=head_hidden,
            dropout=head_dropout,
            kan_grids=kan_grids,
        )
        projection_dim = int(projection_dim)
        self.projection_head = (
            _make_head(
                name=projection_head,
                in_dim=fused_dim,
                out_dim=projection_dim,
                hidden=projection_hidden,
                dropout=projection_dropout,
                kan_grids=kan_grids,
            )
            if projection_dim > 0
            else None
        )

    def _expand_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-1] == self.branch_feature_dim:
            return gate
        repeat = math.ceil(self.branch_feature_dim / gate.shape[-1])
        return gate.repeat_interleave(repeat, dim=-1)[:, : self.branch_feature_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError("DualPlaneMambaClassifier expects a single intensity channel")
            x = x.squeeze(1)
        if x.ndim != 2:
            raise ValueError(f"expected input shape (B, L), got {tuple(x.shape)}")

        outputs: dict[str, torch.Tensor] = {}
        f_sa = f_wa = None
        parts: list[torch.Tensor] = []

        if self.use_sa and self.sa_branch is not None:
            f_sa = self.sa_align(self.sa_branch(x[:, self.sa_slice]))
            parts.append(f_sa)
        if self.use_wa and self.wa_branch is not None:
            f_wa = self.wa_align(self.wa_branch(x[:, self.wa_slice]))
            parts.append(f_wa)

        if self.gate is not None and f_sa is not None and f_wa is not None:
            gate_input = torch.cat([f_sa, f_wa, (f_sa - f_wa).abs(), f_sa * f_wa], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            gate_expanded = self._expand_gate(gate)
            fused = gate_expanded * f_sa + (1.0 - gate_expanded) * f_wa
        elif self.fusion == "concat" and len(parts) > 1:
            fused = torch.cat(parts, dim=-1)
        else:
            fused = parts[0]

        outputs["logits"] = self.head(fused)
        if self.projection_head is not None:
            embedding = self.projection_head(fused)
            if self.projection_normalize:
                embedding = F.normalize(embedding.float(), dim=-1).to(dtype=fused.dtype)
            outputs["embedding"] = embedding
        return outputs if len(outputs) > 1 else outputs["logits"]


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
    "ResNet1D",
    "ConvNeXt1D",
    "DualPlaneMambaClassifier",
    "KANHead",
    "BiGRUPatchClassifier",
    "PatchTSTClassifier",
]
