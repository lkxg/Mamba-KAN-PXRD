"""Summarize the dual-range experiment matrix into a report.

Usage:
    python3 analysis/scripts/summarize_dual_range.py
    python3 analysis/scripts/summarize_dual_range.py --results /tmp/results.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASELINE = "e02_resnet_deep_label_smoothing"

MATRIX_ORDER = [
    BASELINE,
    "e07_wa_only_resnet_label_smoothing",
    "e14_sa_only_resnet_ablation",
    "e08_dual_concat_resnet",
    "e09_dual_gated_resnet",
    "e10_dual_gated_mamba",
    "e11_dual_gated_kan",
    "e12_dual_gated_mamba_kan",
    "e13_dual_gated_mamba_kan_aux",
]

ABLATIONS = [
    ("SA only vs WA only", "e14_sa_only_resnet_ablation", "e07_wa_only_resnet_label_smoothing"),
    ("SA+WA concat vs WA only", "e08_dual_concat_resnet", "e07_wa_only_resnet_label_smoothing"),
    ("Gated vs concat", "e09_dual_gated_resnet", "e08_dual_concat_resnet"),
    ("Mamba vs ResNet", "e10_dual_gated_mamba", "e09_dual_gated_resnet"),
    ("KAN vs MLP head", "e11_dual_gated_kan", "e09_dual_gated_resnet"),
    ("Mamba+KAN vs gated ResNet", "e12_dual_gated_mamba_kan", "e09_dual_gated_resnet"),
    ("Aux heads vs no aux", "e13_dual_gated_mamba_kan_aux", "e12_dual_gated_mamba_kan"),
]

METRIC_COLUMNS = [
    "test_acc1",
    "test_acc5",
    "test_macro_acc1",
    "test_macro_f1",
    "test_rare_acc1",
    "test_gate_mean",
    "sa_mamba_backend",
    "wa_mamba_backend",
    "occlusion_delta_acc1",
    "occlusion_delta_macro_acc1",
    "occlusion_delta_macro_f1",
    "occlusion_delta_rare_acc1",
]


def as_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value, *, signed: bool = False) -> str:
    value = as_float(value)
    if math.isnan(value):
        return ""
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def display_path(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def load_gate_system_summary(metrics_path: str) -> pd.DataFrame:
    if not metrics_path:
        return pd.DataFrame()
    gate_csv = Path(metrics_path).parent / "gate_by_crystal_system.csv"
    if not gate_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(gate_csv)


def load_metrics_json(metrics_path: str) -> dict:
    if not metrics_path:
        return {}
    path = Path(metrics_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def row_for(df: pd.DataFrame, experiment: str) -> pd.Series | None:
    matches = df.loc[df["experiment"] == experiment]
    if matches.empty:
        return None
    return matches.iloc[-1]


def metric_delta(row: pd.Series | None, base: pd.Series | None, column: str) -> str:
    if row is None or base is None:
        return ""
    value = as_float(row.get(column))
    base_value = as_float(base.get(column))
    if math.isnan(value) or math.isnan(base_value):
        return ""
    return fmt(value - base_value, signed=True)


def write_report(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = row_for(df, BASELINE)
    lines: list[str] = [
        "# Dual-Range Summary",
        "",
        "## Baseline",
        "",
    ]
    if base is None:
        lines.append(f"- `{BASELINE}` is missing from the results table.")
    else:
        lines.extend([
            f"- Baseline: `{BASELINE}`",
            (
                "- Test: "
                f"Top-1={fmt(base.get('test_acc1'))}, "
                f"Top-5={fmt(base.get('test_acc5'))}, "
                f"Macro Acc={fmt(base.get('test_macro_acc1'))}, "
                f"Macro F1={fmt(base.get('test_macro_f1'))}, "
                f"Rare Acc={fmt(base.get('test_rare_acc1'))}"
            ),
        ])

    lines.extend([
        "",
        "## Matrix",
        "",
        "| experiment | top1 | top5 | macro | f1 | rare | gate | SA mamba | WA mamba | occ d_top1 | occ d_macro | occ d_f1 | occ d_rare | d_top1 vs baseline | d_macro vs baseline |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for experiment in MATRIX_ORDER:
        row = row_for(df, experiment)
        if row is None:
            lines.append(f"| {experiment} | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| "
            + " | ".join([
                experiment,
                fmt(row.get("test_acc1")),
                fmt(row.get("test_acc5")),
                fmt(row.get("test_macro_acc1")),
                fmt(row.get("test_macro_f1")),
                fmt(row.get("test_rare_acc1")),
                fmt(row.get("test_gate_mean")),
                str(row.get("sa_mamba_backend", "")),
                str(row.get("wa_mamba_backend", "")),
                fmt(row.get("occlusion_delta_acc1"), signed=True),
                fmt(row.get("occlusion_delta_macro_acc1"), signed=True),
                fmt(row.get("occlusion_delta_macro_f1"), signed=True),
                fmt(row.get("occlusion_delta_rare_acc1"), signed=True),
                metric_delta(row, base, "test_acc1"),
                metric_delta(row, base, "test_macro_acc1"),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Ablations",
        "",
        "| contrast | candidate | reference | d_top1 | d_macro | d_f1 | d_rare |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for label, candidate, reference in ABLATIONS:
        cand = row_for(df, candidate)
        ref = row_for(df, reference)
        lines.append(
            "| "
            + " | ".join([
                label,
                candidate,
                reference,
                metric_delta(cand, ref, "test_acc1"),
                metric_delta(cand, ref, "test_macro_acc1"),
                metric_delta(cand, ref, "test_macro_f1"),
                metric_delta(cand, ref, "test_rare_acc1"),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Interpretation Artifacts",
        "",
        "| experiment | metrics | gate by system | confusion cases | confusion pairs |",
        "| --- | --- | --- | --- | --- |",
    ])
    for experiment in MATRIX_ORDER:
        row = row_for(df, experiment)
        if row is None:
            lines.append(f"| {experiment} | pending |  |  |  |")
            continue
        metrics_path = str(row.get("eval_metrics", ""))
        metrics_dir = Path(metrics_path).parent if metrics_path else None
        gate = metrics_dir / "gate_by_crystal_system.csv" if metrics_dir else None
        cases = metrics_dir / "confusion_cases.csv" if metrics_dir else None
        pairs = metrics_dir / "confusion_pairs.csv" if metrics_dir else None
        lines.append(
            "| "
            + " | ".join([
                experiment,
                display_path(metrics_path),
                display_path(str(gate)) if gate and gate.exists() else "",
                display_path(str(cases)) if cases and cases.exists() else "",
                display_path(str(pairs)) if pairs and pairs.exists() else "",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Gate Means By Crystal System",
        "",
    ])
    any_gate = False
    for experiment in MATRIX_ORDER:
        row = row_for(df, experiment)
        if row is None:
            continue
        summary = load_gate_system_summary(str(row.get("eval_metrics", "")))
        if summary.empty:
            continue
        any_gate = True
        lines.extend([
            f"### {experiment}",
            "",
            "| crystal_system | count | mean | std | median |",
            "| --- | --- | --- | --- | --- |",
        ])
        for _, gate_row in summary.iterrows():
            lines.append(
                "| "
                + " | ".join([
                    str(gate_row.get("crystal_system", "")),
                    str(int(gate_row.get("count", 0))),
                    fmt(gate_row.get("mean")),
                    fmt(gate_row.get("std")),
                    fmt(gate_row.get("median")),
                ])
                + " |"
            )
        lines.append("")
    if not any_gate:
        lines.append("No gated experiment with gate-by-system output is available yet.")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize dual-range experiment results")
    ap.add_argument("--results", default="experiments/dual_range_matrix/results.csv")
    ap.add_argument("--out", default="experiments/dual_range_matrix/summary.md")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} does not exist; run scripts/run_experiments.py first")
    df = pd.read_csv(results_path, keep_default_na=False)
    missing = [c for c in ["experiment", *METRIC_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{results_path} is missing columns: {missing}")
    write_report(df, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
