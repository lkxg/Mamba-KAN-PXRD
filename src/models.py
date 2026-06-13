"""PXRD 分类任务的模型架构定义。

当前提供多个基准模型：
1. MLPClassifier：简单的全连接多层感知机
2. ResNet1D：适合一维信号（如 PXRD 曲线）的残差网络
3. BiGRUPatchClassifier：patch 化 PXRD 序列 + 双向 GRU
4. PatchTSTClassifier：patch 化 PXRD 序列 + Transformer Encoder
5. DualRangePXRDClassifier：SA/WA 双分支 + 可选 selective SSM + KAN head

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

try:
    from efficient_kan import KAN as _EfficientKAN
    _EFFICIENT_KAN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional external KAN package
    _EfficientKAN = None
    _EFFICIENT_KAN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_CRYSTAL_SYSTEM_SG_RANGES = (
    (0, 2),
    (2, 15),
    (15, 74),
    (74, 142),
    (142, 167),
    (167, 194),
    (194, 230),
)


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


def _make_angle_channel(
    rng: slice,
    *,
    signal_length: int,
    theta_min: float,
    theta_max: float,
) -> torch.Tensor:
    full = torch.linspace(float(theta_min), float(theta_max), int(signal_length))
    angles = full[rng]
    return ((angles - float(theta_min)) / (float(theta_max) - float(theta_min))) * 2 - 1


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


class _ResNetFeatureEncoder(nn.Module):
    """ResNet-style feature extractor that keeps the final sequence map."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        stem_kernel: int = 15,
        stem_stride: int = 4,
        stem_pool: bool = True,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        blocks_per_stage = _as_tuple(blocks_per_stage)
        if not blocks_per_stage:
            raise ValueError("blocks_per_stage must contain at least one stage")
        if stem_kernel <= 0 or stem_stride <= 0:
            raise ValueError("stem_kernel and stem_stride must be positive")

        c = int(base_channels)
        stem: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                c,
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=stem_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(c),
            nn.GELU(),
        ]
        if stem_pool:
            stem.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem)

        stages: list[nn.Module] = []
        ch = c
        for i, n_blocks in enumerate(blocks_per_stage):
            out_ch = ch if i == 0 else ch * 2
            stride = 1 if i == 0 else 2
            for b in range(int(n_blocks)):
                stages.append(_ResBlock1D(
                    ch if b == 0 else out_ch,
                    out_ch,
                    stride=stride if b == 0 else 1,
                ))
            ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.out_channels = ch

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"expected (B, L) or (B, C, L), got {tuple(x.shape)}")
        return self.stages(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.forward_sequence(x)
        return F.adaptive_avg_pool1d(seq, 1).flatten(1)


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
        basis = basis.flatten(1)
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


class EfficientKANHead(nn.Module):
    """Classifier head backed by the external efficient-kan package."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden: Sequence[int] = (512,),
        grid_size: int = 8,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        if _EfficientKAN is None:
            raise ImportError(
                "efficient-kan is not installed. Install it before using "
                "head: efficient_kan, for example: "
                "pip install git+https://github.com/Blealtan/efficient-kan.git"
                + (
                    f" Import error: {_EFFICIENT_KAN_IMPORT_ERROR}"
                    if _EFFICIENT_KAN_IMPORT_ERROR
                    else ""
                )
            )
        layers_hidden = [int(h) for h in _as_tuple(hidden)]
        self.net = _EfficientKAN(
            [int(in_dim), *layers_hidden, int(out_dim)],
            grid_size=int(grid_size),
            spline_order=int(spline_order),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_head(
    *,
    name: str,
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (512,),
    dropout: float = 0.1,
    kan_grids: int = 8,
    efficient_kan_spline_order: int = 3,
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
    if name in {"xrdmamba_repo", "xrdmamba-repo"}:
        hidden_dim = int(_as_tuple(hidden)[0])
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, out_dim))
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
    if name in {"efficient_kan", "efficient-kan"}:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            EfficientKANHead(
                in_dim,
                out_dim,
                hidden=hidden,
                grid_size=kan_grids,
                spline_order=efficient_kan_spline_order,
            ),
        )
    raise ValueError(f"unknown head type: {name!r}")


class _DualRangeBranch(nn.Module):
    """One SA/WA branch: range ResNet -> optional sequence mixer -> pooling."""

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        blocks_per_stage: Sequence[int],
        stem_kernel: int = 15,
        stem_stride: int = 4,
        stem_pool: bool = True,
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
    ) -> None:
        super().__init__()
        self.encoder = _ResNetFeatureEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            blocks_per_stage=blocks_per_stage,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            stem_pool=stem_pool,
        )
        self.mixer = _MambaSequenceMixer(
            self.encoder.out_channels,
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
        self.out_dim = self.encoder.out_channels
        self.actual_mamba_backend = self.mixer.actual_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encoder.forward_sequence(x).transpose(1, 2)
        seq = self.mixer(seq)
        return seq.mean(dim=1)


class DualRangePXRDClassifier(nn.Module):
    """SA/WA dual-range classifier for PXRD space-group classification.

    The default ranges match the current SimPOD coverage: SA=5-15 degrees and
    WA=10-90 degrees, retaining an overlap around the boundary.
    """

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
        add_angle_channel: bool = False,
        fusion: str = "concat",
        head: str = "mlp",
        head_hidden: Sequence[int] = (512,),
        head_dropout: float = 0.1,
        kan_grids: int = 8,
        efficient_kan_spline_order: int = 3,
        aux_heads: bool = False,
        sa_aux_weight: float = 0.2,
        wa_aux_weight: float = 0.2,
        sa_base_channels: int = 24,
        sa_blocks_per_stage: Sequence[int] = (2, 2, 2),
        wa_base_channels: int = 64,
        wa_blocks_per_stage: Sequence[int] = (3, 4, 6, 3),
        stem_kernel: int = 15,
        stem_stride: int = 4,
        stem_pool: bool = True,
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
        self.add_angle_channel = bool(add_angle_channel)
        self.fusion = fusion.lower()
        self.aux_heads_enabled = bool(aux_heads)
        self.aux_loss_weights = {
            "sa_logits": (
                float(sa_aux_weight)
                if self.aux_heads_enabled and self.use_sa
                else 0.0
            ),
            "wa_logits": (
                float(wa_aux_weight)
                if self.aux_heads_enabled and self.use_wa
                else 0.0
            ),
        }

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
        in_channels = 2 if self.add_angle_channel else 1

        mamba_cfg = dict(mamba or {})
        mamba_backend = mamba_cfg.get("backend", "auto")
        sa_mamba_layers = int(mamba_cfg.get("sa_layers", 0))
        wa_mamba_layers = int(mamba_cfg.get("wa_layers", 0))
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

        if self.use_sa:
            self.sa_branch = _DualRangeBranch(
                in_channels=in_channels,
                base_channels=sa_base_channels,
                blocks_per_stage=sa_blocks_per_stage,
                stem_kernel=stem_kernel,
                stem_stride=stem_stride,
                stem_pool=stem_pool,
                mamba_layers=sa_mamba_layers,
                **common_mamba,
            )
            sa_dim = self.sa_branch.out_dim
        else:
            self.sa_branch = None
            sa_dim = 0

        if self.use_wa:
            self.wa_branch = _DualRangeBranch(
                in_channels=in_channels,
                base_channels=wa_base_channels,
                blocks_per_stage=wa_blocks_per_stage,
                stem_kernel=stem_kernel,
                stem_stride=stem_stride,
                stem_pool=stem_pool,
                mamba_layers=wa_mamba_layers,
                **common_mamba,
            )
            wa_dim = self.wa_branch.out_dim
        else:
            self.wa_branch = None
            wa_dim = 0

        fusion_in = sa_dim + wa_dim
        if self.fusion == "concat" or not (self.use_sa and self.use_wa):
            fused_dim = fusion_in
            self.gate = None
        elif self.fusion == "gated":
            if sa_dim != wa_dim:
                proj_dim = max(sa_dim, wa_dim)
                self.sa_align = nn.Linear(sa_dim, proj_dim) if sa_dim != proj_dim else nn.Identity()
                self.wa_align = nn.Linear(wa_dim, proj_dim) if wa_dim != proj_dim else nn.Identity()
                sa_dim = wa_dim = proj_dim
                fusion_in = sa_dim + wa_dim
            else:
                self.sa_align = nn.Identity()
                self.wa_align = nn.Identity()
            self.gate = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Linear(fusion_in, max(64, fusion_in // 4)),
                nn.GELU(),
                nn.Linear(max(64, fusion_in // 4), sa_dim),
                nn.Sigmoid(),
            )
            fused_dim = fusion_in
        else:
            raise ValueError("fusion must be concat or gated")

        self.head = _make_head(
            name=head,
            in_dim=fused_dim,
            out_dim=self.num_classes,
            hidden=head_hidden,
            dropout=head_dropout,
            kan_grids=kan_grids,
            efficient_kan_spline_order=efficient_kan_spline_order,
        )
        if self.aux_heads_enabled:
            if self.use_sa:
                self.sa_head = _make_head(
                    name="linear",
                    in_dim=sa_dim,
                    out_dim=self.num_classes,
                    dropout=head_dropout,
                )
            if self.use_wa:
                self.wa_head = _make_head(
                    name="linear",
                    in_dim=wa_dim,
                    out_dim=self.num_classes,
                    dropout=head_dropout,
                )

        if self.add_angle_channel:
            sa_angles = _make_angle_channel(
                self.sa_slice,
                signal_length=self.in_dim,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
            )
            wa_angles = _make_angle_channel(
                self.wa_slice,
                signal_length=self.in_dim,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
            )
            self.register_buffer("sa_angles", sa_angles.view(1, 1, -1), persistent=False)
            self.register_buffer("wa_angles", wa_angles.view(1, 1, -1), persistent=False)

    def _prepare_branch_input(self, x: torch.Tensor, rng: slice, angle_name: str) -> torch.Tensor:
        z = x[:, rng]
        z = z.unsqueeze(1)
        if self.add_angle_channel:
            angles = getattr(self, angle_name).expand(z.shape[0], -1, -1)
            z = torch.cat([z, angles.to(dtype=z.dtype, device=z.device)], dim=1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError("DualRangePXRDClassifier expects a single intensity channel")
            x = x.squeeze(1)
        if x.ndim != 2:
            raise ValueError(f"expected input shape (B, L), got {tuple(x.shape)}")

        outputs: dict[str, torch.Tensor] = {}
        parts: list[torch.Tensor] = []
        f_sa = f_wa = None

        if self.use_sa and self.sa_branch is not None:
            f_sa = self.sa_branch(self._prepare_branch_input(x, self.sa_slice, "sa_angles"))
            parts.append(f_sa)
        if self.use_wa and self.wa_branch is not None:
            f_wa = self.wa_branch(self._prepare_branch_input(x, self.wa_slice, "wa_angles"))
            parts.append(f_wa)

        if self.gate is not None and f_sa is not None and f_wa is not None:
            f_sa = self.sa_align(f_sa)
            f_wa = self.wa_align(f_wa)
            gate = self.gate(torch.cat([f_sa, f_wa], dim=-1))
            fused = torch.cat([gate * f_sa, (1.0 - gate) * f_wa], dim=-1)
            outputs["gate"] = gate.detach()
            outputs["gate_mean"] = gate.detach().mean(dim=1)
        else:
            fused = torch.cat(parts, dim=-1)

        outputs["logits"] = self.head(fused)
        if self.aux_heads_enabled:
            if f_sa is not None and hasattr(self, "sa_head"):
                outputs["sa_logits"] = self.sa_head(f_sa)
            if f_wa is not None and hasattr(self, "wa_head"):
                outputs["wa_logits"] = self.wa_head(f_wa)
        return outputs if len(outputs) > 1 else outputs["logits"]


class _PlaneTokenResConvBranch(nn.Module):
    """XRDMamba-style angle-token branch followed by optional Mamba and ResConv."""

    def __init__(
        self,
        *,
        branch_length: int,
        d_model: int = 16,
        conv_channels: Sequence[int] = (32, 64, 128),
        blocks_per_stage: Sequence[int] | int = 1,
        pool_every_stage: bool = True,
        pool_type: str = "max",
        token_mode: str = "multiply",
        token_stride: int = 1,
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
    ) -> None:
        super().__init__()
        if branch_length <= 0:
            raise ValueError("branch_length must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        conv_channels = _as_tuple(conv_channels)
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel")
        blocks = _as_tuple(blocks_per_stage)
        if len(blocks) == 1 and len(conv_channels) > 1:
            blocks = blocks * len(conv_channels)
        if len(blocks) != len(conv_channels):
            raise ValueError("blocks_per_stage must be length 1 or match conv_channels")
        pool_type = pool_type.lower()
        if pool_type not in {"avg", "max"}:
            raise ValueError("pool_type must be avg or max")
        token_mode = token_mode.lower()
        if token_mode not in {"add", "multiply"}:
            raise ValueError("token_mode must be add or multiply")
        token_stride = int(token_stride)
        if token_stride <= 0:
            raise ValueError("token_stride must be positive")

        self.branch_length = int(branch_length)
        self.token_stride = token_stride
        self.token_length = math.ceil(self.branch_length / self.token_stride)
        self.token_mode = token_mode
        self.plane_embed = nn.Embedding(self.token_length, int(d_model))
        self.intensity_proj = (
            nn.Linear(1, int(d_model))
            if self.token_mode == "add"
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

        layers: list[nn.Module] = []
        ch = int(d_model)
        for out_ch, n_blocks in zip(conv_channels, blocks):
            out_ch = int(out_ch)
            for block_idx in range(int(n_blocks)):
                layers.append(_ResBlock1D(
                    ch if block_idx == 0 else out_ch,
                    out_ch,
                    stride=1,
                ))
            ch = out_ch
            if pool_every_stage:
                if pool_type == "avg":
                    layers.append(nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True))
                else:
                    layers.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
        self.resconv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(branch_dropout))
        self.feature_norm = nn.LayerNorm(ch)
        self.out_dim = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected branch input shape (B, L), got {tuple(x.shape)}")
        if x.shape[1] != self.branch_length:
            raise ValueError(
                f"branch length mismatch: got {x.shape[1]}, expected {self.branch_length}"
            )

        if self.token_stride > 1:
            x = F.avg_pool1d(
                x.unsqueeze(1),
                kernel_size=self.token_stride,
                stride=self.token_stride,
                ceil_mode=True,
                count_include_pad=False,
            ).squeeze(1)
        if x.shape[1] != self.token_length:
            raise RuntimeError(
                f"token length mismatch after downsampling: got {x.shape[1]}, "
                f"expected {self.token_length}"
            )

        idx = torch.arange(self.token_length, device=x.device)
        pos = self.plane_embed(idx).unsqueeze(0).to(dtype=x.dtype)
        if self.token_mode == "multiply":
            tokens = pos * x.unsqueeze(-1)
        else:
            if self.intensity_proj is None:
                raise RuntimeError("intensity projection is required for add token_mode")
            value = self.intensity_proj(x.unsqueeze(-1))
            tokens = value + pos.to(dtype=value.dtype)
        tokens = self.token_norm(tokens)
        tokens = self.token_dropout(tokens)
        tokens = self.mixer(tokens)

        seq = self.resconv(tokens.transpose(1, 2))
        feat = self.pool(seq).flatten(1)
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
            nn.Conv1d(1, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        ]
        for _ in range(int(blocks[0])):
            layers.append(_ResBlock1D(stem_channels, stem_channels, stride=1))
        layers.extend([
            nn.Conv1d(stem_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        ])
        for _ in range(int(blocks[1])):
            layers.append(_ResBlock1D(mid_channels, mid_channels, stride=1))
        layers.extend([
            nn.Conv1d(mid_channels, final_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(final_channels),
            nn.GELU(),
            nn.Conv1d(final_channels, int(d_model), kernel_size=1, bias=False),
        ])
        self.frontend = nn.Sequential(*layers)
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
        self.pool = nn.AdaptiveAvgPool1d(1)
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
        tokens = self.token_dropout(self.token_norm(tokens))
        tokens = self.mixer(tokens)
        feat = self.pool(tokens.transpose(1, 2)).flatten(1)
        return self.feature_norm(self.dropout(feat))


class _XRDMambaRepoResBlock1D(nn.Module):
    """ResBlock matching the public XRDMamba repository's ResConv block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pre = (
            nn.Identity()
            if int(in_ch) == int(out_ch)
            else nn.Conv1d(int(in_ch), int(out_ch), kernel_size=1, bias=False)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(int(out_ch), int(out_ch), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(int(out_ch)),
            nn.LeakyReLU(),
            nn.Conv1d(int(out_ch), int(out_ch), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(int(out_ch)),
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        return self.relu(x + self.conv(x))


class _XRDMambaRepoResConvFeature(nn.Module):
    """Feature extractor following XRDMamba's published repository ResConv."""

    def __init__(self, in_channels: int, dropout: float = 0.15) -> None:
        super().__init__()
        p = float(dropout)
        self.conv = nn.Sequential(
            _XRDMambaRepoResBlock1D(in_channels, 32),
            nn.MaxPool1d(2, 2),
            _XRDMambaRepoResBlock1D(32, 32),
            nn.MaxPool1d(2, 2),
            _XRDMambaRepoResBlock1D(32, 64),
            nn.MaxPool1d(2, 2),
            _XRDMambaRepoResBlock1D(64, 64),
            nn.AvgPool1d(2, 2, 1),
            _XRDMambaRepoResBlock1D(64, 128),
            nn.AvgPool1d(2, 2, 1),
            _XRDMambaRepoResBlock1D(128, 128),
            nn.AvgPool1d(2, 2, 1),
            _XRDMambaRepoResBlock1D(128, 256),
            nn.AvgPool1d(2, 2, 1),
            _XRDMambaRepoResBlock1D(256, 256),
            nn.AvgPool1d(2, 2),
            nn.Dropout(p),
            _XRDMambaRepoResBlock1D(256, 256),
            nn.AvgPool1d(2, 2),
            nn.Dropout(p),
            _XRDMambaRepoResBlock1D(256, 512),
            nn.AvgPool1d(2, 2),
            nn.Dropout(p),
            _XRDMambaRepoResBlock1D(512, 512),
            nn.AvgPool1d(2, 2, 1),
            nn.Dropout(p),
            _XRDMambaRepoResBlock1D(512, 1024),
            nn.AvgPool1d(2, 2, 1),
            nn.Dropout(p),
            _XRDMambaRepoResBlock1D(1024, 1024),
            nn.AvgPool1d(2, 2),
        )
        self.out_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if x.shape[-1] != 1:
            raise RuntimeError(
                "XRDMamba repo ResConv expects a final sequence length of 1; "
                f"got {x.shape[-1]}. Use xrdmamba_repo_length=5000 for repo-faithful runs."
            )
        return x.flatten(1)


class _XRDMambaRepoBranch(nn.Module):
    """Public XRDMamba repo-style plane embedding, Mamba mixer, and deep ResConv."""

    def __init__(
        self,
        *,
        branch_length: int,
        d_model: int = 8,
        repo_length: int = 5000,
        branch_dropout: float = 0.15,
        mamba_layers: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 3,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_backend: str = "auto",
        mamba_headdim: int | None = None,
        mamba_ngroups: int | None = None,
        mamba_chunk_size: int | None = None,
        mamba_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if branch_length <= 0:
            raise ValueError("branch_length must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if repo_length <= 0:
            raise ValueError("repo_length must be positive")
        self.branch_length = int(branch_length)
        self.repo_length = int(repo_length)
        self.plane_embed = nn.Embedding(self.repo_length, int(d_model))
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
        self.resconv = _XRDMambaRepoResConvFeature(
            int(d_model),
            dropout=branch_dropout,
        )
        self.out_dim = self.resconv.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected branch input shape (B, L), got {tuple(x.shape)}")
        if x.shape[1] != self.branch_length:
            raise ValueError(
                f"branch length mismatch: got {x.shape[1]}, expected {self.branch_length}"
            )
        if x.shape[1] != self.repo_length:
            x = F.interpolate(
                x.unsqueeze(1),
                size=self.repo_length,
                mode="linear",
                align_corners=True,
            ).squeeze(1)

        idx = torch.arange(self.repo_length, device=x.device)
        tokens = self.plane_embed(idx).unsqueeze(0).to(dtype=x.dtype) * x.unsqueeze(-1)
        tokens = self.mixer(tokens)
        return self.resconv(tokens.transpose(1, 2))


class DualPlaneMambaClassifier(nn.Module):
    """SA/WA diffraction-angle token model with optional Mamba and KAN fusion.

    This follows the XRDMamba ordering more closely than DualRangePXRDClassifier:
    intensity-projected angle-position tokens are mixed as a sequence and then
    summarized by a lightweight residual convolutional aggregator.
    """

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
        frontend: str = "plane_token",
        xrdmamba_repo_length: int = 5000,
        d_model: int = 16,
        sa_d_model: int | None = None,
        wa_d_model: int | None = None,
        sa_conv_channels: Sequence[int] = (32, 64, 128),
        wa_conv_channels: Sequence[int] = (32, 64, 128, 256),
        downsample_channels: Sequence[int] = (64, 128, 256),
        sa_blocks_per_stage: Sequence[int] | int = 1,
        wa_blocks_per_stage: Sequence[int] | int = 1,
        downsample_blocks_per_stage: Sequence[int] | int = (2, 2),
        pool_every_stage: bool = True,
        pool_type: str = "max",
        token_mode: str = "multiply",
        sa_token_stride: int = 1,
        wa_token_stride: int = 1,
        token_dropout: float = 0.0,
        branch_dropout: float = 0.1,
        fusion: str = "gated",
        fusion_dim: int | None = None,
        gate: str = "mlp",
        gate_hidden: Sequence[int] = (256,),
        gate_groups: int = 16,
        gate_dropout: float = 0.1,
        kan_grids: int = 8,
        efficient_kan_spline_order: int = 3,
        head: str = "mlp",
        head_hidden: Sequence[int] = (512,),
        head_dropout: float = 0.1,
        projection_head: str = "kan",
        projection_dim: int = 0,
        projection_hidden: Sequence[int] = (256,),
        projection_dropout: float = 0.1,
        projection_normalize: bool = True,
        hierarchical: dict | None = None,
        aux_heads: bool = False,
        sa_aux_weight: float = 0.1,
        wa_aux_weight: float = 0.1,
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
        self.fusion = fusion.lower()
        self.gate_type = gate.lower()
        self.aux_heads_enabled = bool(aux_heads)
        self.projection_normalize = bool(projection_normalize)
        hierarchical_cfg = dict(hierarchical or {})
        self.hierarchical_expert_heads = bool(hierarchical_cfg.get("expert_heads", False))
        self.hierarchical_enabled = bool(
            hierarchical_cfg.get("enabled", False)
            or self.hierarchical_expert_heads
        )
        self.condition_space_group = bool(
            hierarchical_cfg.get("condition_space_group", False)
        )
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
        if self.frontend not in {"plane_token", "learned_downsample", "xrdmamba_repo"}:
            raise ValueError("frontend must be plane_token, learned_downsample, or xrdmamba_repo")

        def make_branch(
            *,
            branch_length: int,
            branch_d_model: int,
            conv_channels: Sequence[int],
            blocks_per_stage: Sequence[int] | int,
            token_stride: int,
            mamba_layers: int,
        ) -> nn.Module:
            if self.frontend == "xrdmamba_repo":
                return _XRDMambaRepoBranch(
                    branch_length=branch_length,
                    d_model=branch_d_model,
                    repo_length=int(xrdmamba_repo_length),
                    branch_dropout=branch_dropout,
                    mamba_layers=mamba_layers,
                    **common_mamba,
                )
            if self.frontend == "learned_downsample":
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
                    token_dropout=token_dropout,
                    branch_dropout=branch_dropout,
                    mamba_layers=mamba_layers,
                    **common_mamba,
                )
            return _PlaneTokenResConvBranch(
                branch_length=branch_length,
                d_model=branch_d_model,
                conv_channels=conv_channels,
                blocks_per_stage=blocks_per_stage,
                pool_every_stage=pool_every_stage,
                pool_type=pool_type,
                token_mode=token_mode,
                token_stride=token_stride,
                token_dropout=token_dropout,
                branch_dropout=branch_dropout,
                mamba_layers=mamba_layers,
                **common_mamba,
            )

        if self.use_sa:
            self.sa_branch = make_branch(
                branch_length=self.sa_slice.stop - self.sa_slice.start,
                branch_d_model=sa_d_model,
                conv_channels=sa_conv_channels,
                blocks_per_stage=sa_blocks_per_stage,
                token_stride=sa_token_stride,
                mamba_layers=sa_layers,
            )
            sa_dim = self.sa_branch.out_dim
        else:
            self.sa_branch = None
            sa_dim = 0

        if self.use_wa:
            self.wa_branch = make_branch(
                branch_length=self.wa_slice.stop - self.wa_slice.start,
                branch_d_model=wa_d_model,
                conv_channels=wa_conv_channels,
                blocks_per_stage=wa_blocks_per_stage,
                token_stride=wa_token_stride,
                mamba_layers=wa_layers,
            )
            wa_dim = self.wa_branch.out_dim
        else:
            self.wa_branch = None
            wa_dim = 0

        if self.fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be concat or gated")
        if self.gate_type not in {"mlp", "kan"}:
            raise ValueError("gate must be mlp or kan")

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
                if self.gate_type == "mlp":
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
                else:
                    self.gate = nn.Sequential(
                        nn.LayerNorm(gate_in),
                        KANHead(
                            gate_in,
                            self.gate_groups,
                            hidden=hidden,
                            num_grids=kan_grids,
                            dropout=gate_dropout,
                        ),
                    )
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

        head_in_dim = fused_dim
        if self.hierarchical_enabled:
            if self.num_classes != 230:
                raise ValueError("hierarchical classification expects 230 space-group classes")
            crystal_context_dim = int(hierarchical_cfg.get("crystal_context_dim", 64))
            self.crystal_head = _make_head(
                name=str(hierarchical_cfg.get("crystal_head", "mlp")),
                in_dim=fused_dim,
                out_dim=7,
                hidden=hierarchical_cfg.get("crystal_head_hidden", (128,)),
                dropout=float(hierarchical_cfg.get("crystal_head_dropout", head_dropout)),
                kan_grids=kan_grids,
                efficient_kan_spline_order=efficient_kan_spline_order,
            )
            if self.condition_space_group and not self.hierarchical_expert_heads:
                if crystal_context_dim <= 0:
                    raise ValueError("hierarchical.crystal_context_dim must be positive")
                self.crystal_context = nn.Sequential(
                    nn.Linear(7, crystal_context_dim),
                    nn.GELU(),
                    nn.Dropout(float(hierarchical_cfg.get("crystal_context_dropout", 0.0))),
                )
                head_in_dim = fused_dim + crystal_context_dim

        if self.hierarchical_expert_heads:
            expert_head = str(hierarchical_cfg.get("expert_head", head))
            expert_hidden = hierarchical_cfg.get("expert_head_hidden", head_hidden)
            expert_dropout = float(hierarchical_cfg.get("expert_head_dropout", head_dropout))
            self.expert_heads = nn.ModuleList(
                _make_head(
                    name=expert_head,
                    in_dim=fused_dim,
                    out_dim=end - start,
                    hidden=expert_hidden,
                    dropout=expert_dropout,
                    kan_grids=kan_grids,
                    efficient_kan_spline_order=efficient_kan_spline_order,
                )
                for start, end in _CRYSTAL_SYSTEM_SG_RANGES
            )
            self.head = None
        else:
            self.head = _make_head(
                name=head,
                in_dim=head_in_dim,
                out_dim=self.num_classes,
                hidden=head_hidden,
                dropout=head_dropout,
                kan_grids=kan_grids,
                efficient_kan_spline_order=efficient_kan_spline_order,
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
                efficient_kan_spline_order=efficient_kan_spline_order,
            )
            if projection_dim > 0
            else None
        )
        self.aux_loss_weights = {
            "sa_logits": (
                float(sa_aux_weight)
                if self.aux_heads_enabled and self.use_sa
                else 0.0
            ),
            "wa_logits": (
                float(wa_aux_weight)
                if self.aux_heads_enabled and self.use_wa
                else 0.0
            ),
        }
        if self.aux_heads_enabled:
            if self.use_sa:
                self.sa_head = _make_head(
                    name="linear",
                    in_dim=self.branch_feature_dim,
                    out_dim=self.num_classes,
                    dropout=head_dropout,
                )
            if self.use_wa:
                self.wa_head = _make_head(
                    name="linear",
                    in_dim=self.branch_feature_dim,
                    out_dim=self.num_classes,
                    dropout=head_dropout,
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
            outputs["gate"] = gate.detach()
            outputs["gate_mean"] = gate.detach().mean(dim=1)
        elif self.fusion == "concat" and len(parts) > 1:
            fused = torch.cat(parts, dim=-1)
        else:
            fused = parts[0]

        head_input = fused
        if self.hierarchical_enabled:
            crystal_logits = self.crystal_head(fused)
            outputs["crystal_logits"] = crystal_logits
            if self.hierarchical_expert_heads:
                expert_logits = [head(fused) for head in self.expert_heads]
                outputs["expert_logits"] = expert_logits
                crystal_log_prob = F.log_softmax(crystal_logits.float(), dim=-1).to(dtype=fused.dtype)
                logits = fused.new_full((fused.shape[0], self.num_classes), float("-inf"))
                for idx, (start, end) in enumerate(_CRYSTAL_SYSTEM_SG_RANGES):
                    logits[:, start:end] = expert_logits[idx] + crystal_log_prob[:, idx:idx + 1]
                outputs["logits"] = logits
            elif self.condition_space_group:
                crystal_prob = torch.softmax(crystal_logits.float(), dim=-1).to(dtype=fused.dtype)
                head_input = torch.cat([fused, self.crystal_context(crystal_prob)], dim=-1)

        if "logits" not in outputs:
            outputs["logits"] = self.head(head_input)
        if self.projection_head is not None:
            embedding = self.projection_head(fused)
            if self.projection_normalize:
                embedding = F.normalize(embedding.float(), dim=-1).to(dtype=fused.dtype)
            outputs["embedding"] = embedding
        if self.aux_heads_enabled:
            if f_sa is not None and hasattr(self, "sa_head"):
                outputs["sa_logits"] = self.sa_head(f_sa)
            if f_wa is not None and hasattr(self, "wa_head"):
                outputs["wa_logits"] = self.wa_head(f_wa)
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
    "MLPClassifier",
    "ResNet1D",
    "DualRangePXRDClassifier",
    "DualPlaneMambaClassifier",
    "KANHead",
    "EfficientKANHead",
    "BiGRUPatchClassifier",
    "PatchTSTClassifier",
]
