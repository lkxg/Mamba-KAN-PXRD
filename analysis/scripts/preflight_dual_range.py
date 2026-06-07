"""Preflight checks before launching the formal dual-range matrix.

Use this on the intended CUDA/H100 runner. It checks environment and data
requirements that are stronger than CPU smoke tests: CUDA availability,
required dataset/split files, configs, baseline checkpoint, and result schema.

Usage:
    python3 analysis/scripts/preflight_dual_range.py
    python3 analysis/scripts/preflight_dual_range.py --allow-cpu
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

from analysis.scripts.audit_dual_range import CONFIGS
from scripts.run_experiments import normalize_results_path, split_markdown_row


REQUIRED_RESULT_FIELDS = {
    "experiment",
    "config",
    "val_acc10",
    "test_acc1",
    "test_acc5",
    "test_acc10",
    "test_macro_acc1",
    "test_macro_f1",
    "test_rare_acc1",
    "occlusion_acc1",
    "occlusion_acc10",
    "occlusion_macro_acc1",
    "occlusion_macro_f1",
    "occlusion_rare_acc1",
    "test_n",
    "eval_metrics",
    "checkpoint",
    "wandb_run",
}

BASELINE_CHECKPOINT = Path(
    "checkpoints/e02_resnet_deep_label_smoothing_20260605_085108/best.pt"
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def path_exists(path: str | Path) -> bool:
    return (PROJECT_ROOT / path).exists()


def result_fields(path: Path) -> set[str]:
    if not path.exists():
        return set()
    table_lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("|")
    ]
    if not table_lines:
        return set()
    return set(split_markdown_row(table_lines[0]))


def build_checks(*, allow_cpu: bool) -> list[Check]:
    checks: list[Check] = []

    cuda_ok = torch.cuda.is_available()
    cuda_detail = f"torch={torch.__version__}, cuda_version={torch.version.cuda}, available={cuda_ok}"
    if cuda_ok:
        cuda_detail += f", devices={torch.cuda.device_count()}, device0={torch.cuda.get_device_name(0)}"
    checks.append(Check("CUDA available", cuda_ok or allow_cpu, cuda_detail))

    checks.append(Check(
        "dataset files",
        path_exists("dataset/pxrd.npy") and path_exists("dataset/labels.csv"),
        "requires dataset/pxrd.npy and dataset/labels.csv",
    ))
    checks.append(Check(
        "split file",
        path_exists("splits/splits.csv"),
        "requires splits/splits.csv",
    ))
    missing_configs = [
        str(path)
        for path in CONFIGS.values()
        if not path_exists(path)
    ]
    checks.append(Check(
        "experiment configs",
        not missing_configs,
        "all configs present" if not missing_configs else f"missing: {missing_configs}",
    ))
    checks.append(Check(
        "baseline checkpoint",
        path_exists(BASELINE_CHECKPOINT),
        str(BASELINE_CHECKPOINT),
    ))

    results_path = normalize_results_path(
        PROJECT_ROOT / "experiments/dual_range_matrix/results.md"
    )
    fields = result_fields(results_path)
    missing_fields = sorted(REQUIRED_RESULT_FIELDS - fields)
    checks.append(Check(
        "result schema",
        not missing_fields,
        f"{len(fields)} Markdown columns; missing={missing_fields or 'none'}",
    ))

    return checks


def print_checks(checks: list[Check]) -> None:
    print("# Dual-Range Preflight")
    print()
    print("| check | status | detail |")
    print("| --- | --- | --- |")
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        detail = check.detail.replace("|", "\\|")
        print(f"| {check.name} | {status} | {detail} |")
    print()
    passed = sum(c.ok for c in checks)
    print(f"Summary: PASS={passed} FAIL={len(checks) - passed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight formal dual-range matrix runs")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only preflight for local smoke/debug use.",
    )
    args = parser.parse_args()

    checks = build_checks(
        allow_cpu=args.allow_cpu,
    )
    print_checks(checks)
    if not all(c.ok for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
