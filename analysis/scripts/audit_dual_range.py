"""Audit progress against the current dual-range PXRD matrix.

This script checks current repo evidence, not intent. It separates implemented
code/config from final experiment evidence so smoke results are not mistaken for
completed matrix results.

Usage:
    python3 analysis/scripts/audit_dual_range.py
    python3 analysis/scripts/audit_dual_range.py --gate implementation
    python3 analysis/scripts/audit_dual_range.py --gate evidence
    python3 analysis/scripts/audit_dual_range.py --strict
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from scripts.run_experiments import (
    RESULT_COLUMNS,
    load_existing_results,
    normalize_results_path,
    split_markdown_row,
)


BASELINE = "e02_resnet_deep_label_smoothing"

CONFIGS = {
    BASELINE: "configs/experiments/e02_resnet_deep_label_smoothing.yaml",
    "e07_wa_only_resnet_label_smoothing": "configs/experiments/e07_wa_only_resnet_label_smoothing.yaml",
    "e08_dual_concat_resnet": "configs/experiments/e08_dual_concat_resnet.yaml",
    "e09_dual_gated_resnet": "configs/experiments/e09_dual_gated_resnet.yaml",
    "e11_dual_gated_kan": "configs/experiments/e11_dual_gated_kan.yaml",
    "e14_sa_only_resnet_ablation": "configs/experiments/e14_sa_only_resnet_ablation.yaml",
}

MATRIX_ORDER = [
    "e07_wa_only_resnet_label_smoothing",
    "e14_sa_only_resnet_ablation",
    "e08_dual_concat_resnet",
    "e09_dual_gated_resnet",
    "e11_dual_gated_kan",
]

GATED_EXPERIMENTS = (
    "e09_dual_gated_resnet",
    "e11_dual_gated_kan",
)

GATE_PHASES = {
    "implementation": {"implementation"},
    "evidence": {"implementation", "evidence"},
    "final": {"implementation", "evidence", "final"},
}

METRIC_COLUMNS = [
    "test_acc1",
    "test_acc5",
    "test_acc10",
    "test_macro_acc1",
    "test_macro_f1",
    "test_rare_acc1",
]


@dataclass
class Finding:
    item: str
    status: str
    evidence: str
    phase: str = "implementation"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def exists(path: str | Path) -> bool:
    return (PROJECT_ROOT / path).exists()


def load_results(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    path = normalize_results_path(path)
    if not path.exists():
        return [], {}
    table_lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("|")
    ]
    fieldnames = split_markdown_row(table_lines[0]) if table_lines else []
    rows = {
        str(row.get("experiment", "")): row
        for row in load_existing_results(path)
        if row.get("experiment")
    }
    return fieldnames or list(RESULT_COLUMNS), rows


def nonempty(row: dict[str, str] | None, columns: list[str]) -> bool:
    if row is None:
        return False
    return all(str(row.get(col, "")).strip() for col in columns)


def cfg(exp: str) -> dict[str, Any]:
    return load_config(PROJECT_ROOT / CONFIGS[exp])


def dual_cfg(exp: str) -> dict[str, Any]:
    return cfg(exp).get("model", {}).get("dual_range", {})


def same_range(values: Any, expected: tuple[float, float]) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return False
    return all(abs(float(v) - e) < 1e-6 for v, e in zip(values, expected))


def all_configs_exist() -> bool:
    return all(exists(path) for path in CONFIGS.values())


def expected_ranges_ok() -> bool:
    for exp in MATRIX_ORDER:
        dcfg = dual_cfg(exp)
        if float(dcfg.get("theta_min", -1)) != 5.0:
            return False
        if float(dcfg.get("theta_max", -1)) != 90.0:
            return False
        if not same_range(dcfg.get("sa_range"), (5.0, 15.0)):
            return False
        if not same_range(dcfg.get("wa_range"), (10.0, 90.0)):
            return False
    return True


def long_tail_ok() -> bool:
    for exp in CONFIGS:
        config = cfg(exp)
        metrics = config.get("metrics", {})
        if int(metrics.get("rare_min_train_count", 1)) != 1:
            return False
        if int(metrics.get("rare_max_train_count", 100)) != 100:
            return False
        checkpoint = config.get("checkpoint", {})
        if checkpoint.get("monitor") != "val_balanced_acc1_macro":
            return False
    return True


def model_source_contains(*patterns: str) -> bool:
    source = (PROJECT_ROOT / "src/models.py").read_text(encoding="utf-8")
    return all(pattern in source for pattern in patterns)


def evaluate_source_contains(*patterns: str) -> bool:
    source = (PROJECT_ROOT / "scripts/evaluate.py").read_text(encoding="utf-8")
    return all(pattern in source for pattern in patterns)


def run_source_contains(*patterns: str) -> bool:
    source = (PROJECT_ROOT / "scripts/run_experiments.py").read_text(encoding="utf-8")
    return all(pattern in source for pattern in patterns)


def result_rows_complete(rows: dict[str, dict[str, str]], experiments: list[str]) -> bool:
    return all(nonempty(rows.get(exp), METRIC_COLUMNS) for exp in experiments)


def row_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def eval_metrics_exists(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    metrics_path = str(row.get("eval_metrics", "")).strip()
    return bool(metrics_path) and row_path(metrics_path).exists()


def artifact_status(rows: dict[str, dict[str, str]]) -> tuple[str, str]:
    required = [BASELINE, *MATRIX_ORDER]
    artifact_rows = [exp for exp in required if eval_metrics_exists(rows.get(exp))]
    missing_artifacts = [exp for exp in required if exp not in artifact_rows]
    missing_gate_tables = []
    for exp in GATED_EXPERIMENTS:
        row = rows.get(exp)
        if not eval_metrics_exists(row):
            missing_gate_tables.append(exp)
            continue
        metrics_dir = row_path(str(row.get("eval_metrics", ""))).parent
        if not (metrics_dir / "gate_by_crystal_system.csv").exists():
            missing_gate_tables.append(exp)

    if not missing_artifacts and not missing_gate_tables:
        return (
            "PASS",
            f"{len(artifact_rows)}/{len(required)} rows have metrics artifacts; all gated rows have gate-by-system tables.",
        )
    detail = (
        f"{len(artifact_rows)}/{len(required)} rows have metrics artifacts; "
        f"missing artifacts={missing_artifacts or 'none'}; "
        f"missing gated tables={missing_gate_tables or 'none'}."
    )
    return "PENDING", detail


def build_findings(results_path: Path) -> list[Finding]:
    fieldnames, rows = load_results(results_path)
    baseline_row = rows.get(BASELINE)
    missing_configs = [
        rel(PROJECT_ROOT / path)
        for path in CONFIGS.values()
        if not exists(path)
    ]
    findings: list[Finding] = []

    findings.append(Finding(
        "1. Fixed strong baseline",
        "PASS" if nonempty(baseline_row, METRIC_COLUMNS) else "PENDING",
        (
            f"{rel(results_path)} has {BASELINE} with "
            + ", ".join(f"{k}={baseline_row.get(k, '')}" for k in METRIC_COLUMNS)
            if baseline_row
            else f"{BASELINE} missing from {rel(results_path)}"
        ),
        "evidence",
    ))

    findings.append(Finding(
        "2. SA/WA range split",
        "PASS" if expected_ranges_ok() else "FAIL",
        "All dual-range configs use theta 5-90, SA 5-15, WA 10-90 with overlap 10-15.",
    ))

    findings.append(Finding(
        "3. DualRangePXRDClassifier module",
        "PASS" if model_source_contains("class DualRangePXRDClassifier", "self.sa_slice", "self.wa_slice") else "FAIL",
        "src/models.py defines DualRangePXRDClassifier with SA/WA slicing.",
    ))

    e08 = dual_cfg("e08_dual_concat_resnet")
    findings.append(Finding(
        "4. First non-Mamba/KAN dual branch",
        "PASS" if (
            e08.get("use_sa") is True
            and e08.get("use_wa") is True
            and e08.get("fusion") == "concat"
            and e08.get("head") == "mlp"
            and int(e08.get("mamba", {}).get("sa_layers", -1)) == 0
            and int(e08.get("mamba", {}).get("wa_layers", -1)) == 0
        ) else "FAIL",
        f"{CONFIGS['e08_dual_concat_resnet']} is SA+WA concat with MLP head and zero Mamba layers.",
    ))

    e09 = dual_cfg("e09_dual_gated_resnet")
    findings.append(Finding(
        "5. Gated fusion",
        "PASS" if e09.get("fusion") == "gated" and model_source_contains("outputs[\"gate_mean\"]") else "FAIL",
        f"{CONFIGS['e09_dual_gated_resnet']} uses gated fusion and model exposes gate_mean.",
    ))

    e11 = dual_cfg("e11_dual_gated_kan")
    findings.append(Finding(
        "6. KAN fused head",
        "PASS" if (
            e11.get("head") == "kan"
            and model_source_contains("class KANHead")
        ) else "FAIL",
        "E11 uses head: kan and src/models.py defines KANHead.",
    ))

    matrix_config_status = "PASS" if all_configs_exist() else "FAIL"
    findings.append(Finding(
        "7a. Experiment matrix configs",
        matrix_config_status,
        "All current matrix config files are present." if matrix_config_status == "PASS" else f"Missing: {missing_configs}",
    ))
    completed_matrix = [exp for exp in [BASELINE, *MATRIX_ORDER] if nonempty(rows.get(exp), METRIC_COLUMNS)]
    findings.append(Finding(
        "7b. Experiment matrix results",
        "PASS" if result_rows_complete(rows, [BASELINE, *MATRIX_ORDER]) else "PENDING",
        f"{len(completed_matrix)}/{1 + len(MATRIX_ORDER)} result rows have full test metrics in {rel(results_path)}.",
        "evidence",
    ))

    ablation_pairs = [
        ("e14_sa_only_resnet_ablation", "e07_wa_only_resnet_label_smoothing"),
        ("e08_dual_concat_resnet", "e07_wa_only_resnet_label_smoothing"),
        ("e09_dual_gated_resnet", "e08_dual_concat_resnet"),
        ("e11_dual_gated_kan", "e09_dual_gated_resnet"),
    ]
    ablations_done = sum(
        1
        for a, b in ablation_pairs
        if nonempty(rows.get(a), METRIC_COLUMNS) and nonempty(rows.get(b), METRIC_COLUMNS)
    )
    findings.append(Finding(
        "8. Ablation analysis",
        "PASS" if ablations_done == len(ablation_pairs) else "PENDING",
        f"{ablations_done}/{len(ablation_pairs)} ablation contrasts have both candidate and reference metrics.",
        "evidence",
    ))

    artifact_code_ok = evaluate_source_contains(
        "save_gate_mean_distribution",
        "save_gate_by_crystal_system",
        "save_confusion_cases",
        "MaskAngleRange",
    )
    findings.append(Finding(
        "9a. Physical interpretation code",
        "PASS" if artifact_code_ok else "FAIL",
        "Evaluation code writes gate distribution, gate-by-system, confusion cases, and low-angle occlusion.",
    ))
    artifact_gate_status, artifact_gate_detail = artifact_status(rows)
    findings.append(Finding(
        "9b. Physical interpretation artifacts",
        artifact_gate_status if artifact_code_ok else "FAIL",
        artifact_gate_detail if artifact_code_ok else "Interpretation code is incomplete.",
        "evidence",
    ))

    findings.append(Finding(
        "Long-tail handling",
        "PASS" if long_tail_ok() and run_source_contains("val_balanced_acc1_macro") else "FAIL",
        "Configs monitor val_balanced_acc1_macro and report rare classes for train counts 1-100.",
    ))

    expected_fields = {
        "config",
        "val_acc10",
        "test_acc10",
        "occlusion_acc1",
        "occlusion_acc10",
        "occlusion_macro_acc1",
        "occlusion_macro_f1",
        "occlusion_rare_acc1",
        "eval_metrics",
        "checkpoint",
        "wandb_run",
    }
    schema_source_ok = run_source_contains(*expected_fields)
    schema_csv_ok = (
        not results_path.exists()
        or expected_fields.issubset(set(fieldnames))
    )
    findings.append(Finding(
        "Result schema",
        "PASS" if schema_source_ok and schema_csv_ok else "FAIL",
        (
            f"run_experiments.py fields present={schema_source_ok}; "
            f"{rel(normalize_results_path(results_path))} has {len(fieldnames)} columns; "
            f"Markdown required fields present={schema_csv_ok}."
        ),
    ))

    return findings


def print_report(findings: list[Finding]) -> None:
    print("# Dual-Range Audit")
    print()
    print("| phase | item | status | evidence |")
    print("| --- | --- | --- | --- |")
    for finding in findings:
        evidence = finding.evidence.replace("|", "\\|").replace("\n", " ")
        print(f"| {finding.phase} | {finding.item} | {finding.status} | {evidence} |")
    print()
    counts = {status: sum(f.status == status for f in findings) for status in ["PASS", "PENDING", "FAIL"]}
    print(
        f"Summary: PASS={counts['PASS']} "
        f"PENDING={counts['PENDING']} FAIL={counts['FAIL']}"
    )


def gate_findings(findings: list[Finding], gate: str) -> list[Finding]:
    phases = GATE_PHASES[gate]
    return [finding for finding in findings if finding.phase in phases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dual-range experiment progress")
    parser.add_argument(
        "--results",
        default="experiments/dual_range_matrix/results.md",
        help="Result Markdown table to audit. Legacy .csv paths map to .md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Alias for --gate final.",
    )
    parser.add_argument(
        "--gate",
        choices=sorted(GATE_PHASES),
        default=None,
        help=(
            "Exit nonzero unless all findings in the selected phase set pass. "
            "implementation checks code/config only; evidence also requires "
            "formal result rows and artifacts; final is currently equivalent "
            "to evidence for this non-Mamba architecture matrix."
        ),
    )
    args = parser.parse_args()

    findings = build_findings(normalize_results_path(PROJECT_ROOT / args.results))
    print_report(findings)

    gate = "final" if args.strict else args.gate
    checked_findings = gate_findings(findings, gate or "implementation")
    has_fail = any(f.status == "FAIL" for f in checked_findings)
    has_pending = gate is not None and any(
        f.status == "PENDING" for f in checked_findings
    )
    if has_fail or has_pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
