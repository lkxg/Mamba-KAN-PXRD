"""Forward-smoke the dual-range experiment configs.

This does not train or write formal results. It only checks that each current
dual-range matrix config instantiates, accepts a PXRD-length tensor, and exposes
the expected output contract: logits and optional gate statistics.

Usage:
    python3 analysis/scripts/smoke_dual_range_forward.py
    python3 analysis/scripts/smoke_dual_range_forward.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_model
from src.utils import load_config, set_seed


CONFIGS = [
    "configs/experiments/e07_wa_only_resnet_label_smoothing.yaml",
    "configs/experiments/e14_sa_only_resnet_ablation.yaml",
    "configs/experiments/e08_dual_concat_resnet.yaml",
    "configs/experiments/e09_dual_gated_resnet.yaml",
    "configs/experiments/e11_dual_gated_kan.yaml",
]


@dataclass
class SmokeResult:
    experiment: str
    params_m: float
    output_keys: list[str]
    sa_backend: str
    wa_backend: str
    device: str


def expected_output_keys(cfg: dict) -> set[str]:
    model_cfg = cfg["model"]["dual_range"]
    keys = {"logits"}
    if (
        model_cfg.get("fusion") == "gated"
        and model_cfg.get("use_sa", True)
        and model_cfg.get("use_wa", True)
    ):
        keys.update({"gate", "gate_mean"})
    if model_cfg.get("aux_heads", False):
        if model_cfg.get("use_sa", True):
            keys.add("sa_logits")
        if model_cfg.get("use_wa", True):
            keys.add("wa_logits")
    return keys


def tensor_outputs(output: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(output, dict):
        return output
    return {"logits": output}


def smoke_config(
    path: Path,
    *,
    batch_size: int,
    in_dim: int,
    num_classes: int,
    device: torch.device,
) -> SmokeResult:
    cfg = load_config(path)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    model = build_model(cfg, in_dim=in_dim, num_classes=num_classes).to(device).eval()
    x = torch.randn(batch_size, in_dim, device=device)
    with torch.no_grad():
        outputs = tensor_outputs(model(x))

    expected = expected_output_keys(cfg)
    actual = set(outputs)
    if actual != expected:
        raise AssertionError(
            f"{path}: output keys mismatch; expected {sorted(expected)}, got {sorted(actual)}"
        )

    logits = outputs["logits"]
    if tuple(logits.shape) != (batch_size, num_classes):
        raise AssertionError(
            f"{path}: logits shape {tuple(logits.shape)} != {(batch_size, num_classes)}"
        )

    if "gate" in outputs and outputs["gate"].shape[0] != batch_size:
        raise AssertionError(f"{path}: gate batch dimension mismatch")
    if "gate_mean" in outputs and tuple(outputs["gate_mean"].shape) != (batch_size,):
        raise AssertionError(f"{path}: gate_mean shape mismatch")
    for key in ("sa_logits", "wa_logits"):
        if key in outputs and tuple(outputs[key].shape) != (batch_size, num_classes):
            raise AssertionError(f"{path}: {key} shape mismatch")

    sa_branch = getattr(model, "sa_branch", None)
    wa_branch = getattr(model, "wa_branch", None)
    sa_backend = getattr(sa_branch, "actual_mamba_backend", "") if sa_branch else ""
    wa_backend = getattr(wa_branch, "actual_mamba_backend", "") if wa_branch else ""
    params = sum(p.numel() for p in model.parameters()) / 1_000_000
    return SmokeResult(
        experiment=str(cfg["experiment"]["name"]),
        params_m=params,
        output_keys=sorted(actual),
        sa_backend=str(sa_backend),
        wa_backend=str(wa_backend),
        device=str(device),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-smoke dual-range configs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--in-dim", type=int, default=10824)
    parser.add_argument("--num-classes", type=int, default=230)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for the smoke forward pass: auto, cpu, cuda, cuda:0, etc.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    print("| experiment | params_m | device | outputs | sa_backend | wa_backend |")
    print("| --- | --- | --- | --- | --- | --- |")
    for config in CONFIGS:
        result = smoke_config(
            PROJECT_ROOT / config,
            batch_size=args.batch_size,
            in_dim=args.in_dim,
            num_classes=args.num_classes,
            device=device,
        )
        print(
            f"| {result.experiment} | {result.params_m:.2f} | {result.device} | "
            f"{','.join(result.output_keys)} | {result.sa_backend} | {result.wa_backend} |"
        )


if __name__ == "__main__":
    main()
