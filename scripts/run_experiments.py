"""Run a sequence of training/evaluation experiments and summarize results.

Usage:
    python3 scripts/run_experiments.py
    python3 scripts/run_experiments.py --configs configs/experiments/e02_resnet_deep_label_smoothing.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


DEFAULT_RESULTS = "experiments/dual_range_matrix/results.md"
DEFAULT_LOGS_DIR = "experiments/dual_range_matrix/logs"
MAIN_RESULTS = "experiments/main_results/results.md"
MAIN_LOGS_DIR = "experiments/main_results/logs"
CS_RESULTS = "experiments/cs_results/results.md"
CS_LOGS_DIR = "experiments/cs_results/logs"
PEAK_TOKEN_RESULTS = "experiments/peak_token/results.md"
PEAK_TOKEN_LOGS_DIR = "experiments/peak_token/logs"

DEFAULT_CONFIGS = [
    "configs/experiments/e02_resnet_deep_label_smoothing.yaml",
    "configs/experiments/e07_wa_only_resnet_label_smoothing.yaml",
    "configs/experiments/e08_dual_concat_resnet.yaml",
    "configs/experiments/e09_dual_gated_resnet.yaml",
    "configs/experiments/e11_dual_gated_kan.yaml",
    "configs/experiments/e14_sa_only_resnet_ablation.yaml",
]

MAIN_CONFIGS = [
    "configs/main/m01_resnet1d_label_smoothing.yaml",
    "configs/main/m02_dual_gated_resnet_label_smoothing.yaml",
    "configs/main/m03_dual_gated_kan_label_smoothing.yaml",
    "configs/main/m04_dual_gated_mamba_label_smoothing.yaml",
    "configs/main/m05_dual_gated_mamba_kan_label_smoothing.yaml",
    "configs/main/m08_dual_gated_mamba_kan_angle_label_smoothing.yaml",
    "configs/main/m09_dual_gated_mamba_kan_relaxed_resnet_label_smoothing.yaml",
    "configs/main/m06_dual_gated_resnet_ldam_drw.yaml",
    "configs/main/m07_dual_gated_resnet_crt.yaml",
    "configs/main/m11_dual_plane_resconv_label_smoothing.yaml",
    "configs/main/m12_dual_plane_mamba_resconv_label_smoothing.yaml",
    "configs/main/m13_dual_plane_mamba_kan_gate_label_smoothing.yaml",
    "configs/main/m14_dual_plane_mamba_kan_gate_aux_label_smoothing.yaml",
    "configs/main/m15_dual_plane_mamba_kan_gate_aux_supcon.yaml",
    "configs/main/m16_dual_plane_mamba_kan_gate_aux_supcon_d64.yaml",
    "configs/main/m17_wa_only_plane_mamba_kan_aux_supcon.yaml",
    "configs/main/m18_single_plane_learned_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m19_single_plane_learned_downsample_mamba_d128_l8_kan_head_label_smoothing.yaml",
    "configs/main/m20_single_plane_learned_downsample_bidirectional_mamba_d128_l8_kan_head_label_smoothing.yaml",
    "configs/main/m21_single_plane_learned_downsample_mamba_d128_l8_efficient_kan_head_label_smoothing.yaml",
    "configs/main/m22_single_plane_learned_downsample_mamba_d128_l8_ldam_drw.yaml",
    "configs/main/m23_single_plane_learned_downsample_mamba_d128_l8_kan_proj_supcon.yaml",
    "configs/main/m25_single_plane_learned_downsample_mamba_d128_l8_hierarchical_crystal_aux.yaml",
    "configs/main/m26_single_plane_learned_downsample_mamba_d128_l8_crystal_expert_heads.yaml",
    "configs/main/m27_single_plane_learned_downsample_mamba2_d128_l8_label_smoothing.yaml",
    "configs/main/m28_single_plane_learned_downsample_mamba_d128_l8_kan_proj_supcon_lr3e4.yaml",
    "configs/main/m29_single_plane_learned_downsample_mamba_d128_l8_focal_gamma15.yaml",
    "configs/main/m32_single_plane_learned_downsample_mamba_d128_l8_kan_residual_adapter_label_smoothing.yaml",
    "configs/main/m33_single_plane_learned_downsample_mamba_d128_l8_mlp_kan_logit_residual_label_smoothing.yaml",
    "configs/main/m38_single_plane_learned_downsample_mamba_token_kan_adapter_label_smoothing.yaml",
    "configs/main/m39_single_plane_learned_downsample_mamba_local_kan_adapter_label_smoothing.yaml",
    "configs/main/m40_single_plane_learned_downsample_mamba_angle_pos_label_smoothing.yaml",
    "configs/main/m41_single_plane_learned_downsample_mamba_gated_pool_label_smoothing.yaml",
    "configs/main/m42_dual_plane_sa_raw_wa_learned_downsample_mamba_label_smoothing.yaml",
    "configs/main/m43_single_plane_convnext_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m46_single_plane_learned_downsample_mamba_d128_l8_wide_frontend_label_smoothing.yaml",
    "configs/main/m47_single_plane_learned_downsample_mamba_d128_l8_gated_pool_label_smoothing.yaml",
    "configs/main/m48_single_plane_learned_downsample_mamba_d128_l8_stride4_label_smoothing.yaml",
    "configs/main/m49_single_plane_learned_downsample_mamba_d128_l10_label_smoothing.yaml",
    "configs/main/m50_single_plane_learned_downsample_mamba_d192_l8_label_smoothing.yaml",
    "configs/main/m51_single_plane_learned_downsample_mamba_d128_l8_dstate32_label_smoothing.yaml",
    "configs/main/m52_single_plane_multiscale_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m53_single_plane_inception_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m54_single_plane_peak_aware_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m55_single_plane_wavelet_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m56_single_plane_antialiased_downsample_mamba_d128_l8_label_smoothing.yaml",
    "configs/main/m57_mobilexrd_mamba_lite_d128_l8_label_smoothing.yaml",
    "configs/main/m58_mobilexrd_mamba_wavelet_d128_l8_label_smoothing.yaml",
    "configs/main/m59_mobilexrd_mamba_global96_local32_no_identity_d128_l8_label_smoothing.yaml",
    "configs/main/m60_mobilexrd_mamba_wte_global96_local32_no_identity_d128_l8_label_smoothing.yaml",
    "configs/main/m61_single_plane_learned_downsample_mamba2_d128_l8_stride4_batch64_label_smoothing.yaml",
    "configs/main/m45_resnet1d18_label_smoothing.yaml",
]

NON_MAMBA_CONFIGS = [
    "configs/experiments/e02_resnet_deep_label_smoothing.yaml",
    "configs/experiments/e07_wa_only_resnet_label_smoothing.yaml",
    "configs/experiments/e14_sa_only_resnet_ablation.yaml",
    "configs/experiments/e08_dual_concat_resnet.yaml",
    "configs/experiments/e09_dual_gated_resnet.yaml",
    "configs/experiments/e11_dual_gated_kan.yaml",
]

CS_CONFIGS = [
    "configs/cs/c01_resnet1d_crystal_label_smoothing.yaml",
]

PEAK_TOKEN_CONFIGS = [
    "configs/main/m34_single_plane_learned_downsample_mamba_peak_mamba_label_smoothing.yaml",
    "configs/main/m35_single_plane_learned_downsample_mamba_peak_pool_label_smoothing.yaml",
]

CONFIG_PRESETS = {
    "cs": CS_CONFIGS,
    "full": DEFAULT_CONFIGS,
    "main": MAIN_CONFIGS,
    "non_mamba": NON_MAMBA_CONFIGS,
    "peak_token": PEAK_TOKEN_CONFIGS,
}

TEST_RE = re.compile(
    r"Test loss=(?P<loss>[0-9.]+)\s+"
    r"acc1=(?P<acc1>[0-9.]+)\s+"
    r"acc5=(?P<acc5>[0-9.]+)\s+"
    r"(?:acc10=(?P<acc10>[0-9.]+)\s+)?"
    r"(?:macro=(?P<macro>[0-9.]+)\s+)?"
    r"(?:macro_f1=(?P<macro_f1>[0-9.]+)\s+)?"
    r"(?:rare=(?P<rare>[0-9.]+|nan)\s+)?"
    r"(?:rare_acc1=(?P<rare_acc1>[0-9.]+|nan)\s+)?"
    r"(?:aux=(?P<aux>[0-9.]+|nan)\s+)?"
    r"(?:gate=(?P<gate>[0-9.]+|nan)\s+)?"
    r"(?:crystal_acc1=(?P<crystal_acc1>[0-9.]+|nan)\s+)?"
    r"\(N=(?P<n>[0-9]+)\)"
)

RESULT_COLUMNS = [
    "experiment",
    "test_acc1",
    "test_acc5",
    "test_macro_f1",
]

def run_and_log(cmd: list[str], log_path: Path, env: dict[str, str]) -> str:
    """Run a command, stream stdout, and write a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_parts: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
            output_parts.append(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output="".join(output_parts))
    return "".join(output_parts)


def newest_checkpoint(run_name: str, started_at: float) -> Path:
    """Return the newest best.pt for a run name after started_at."""
    candidates: list[Path] = []
    for path in (PROJECT_ROOT / "checkpoints").glob(f"{run_name}_*/best.pt"):
        if path.stat().st_mtime >= started_at:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no best.pt found for run {run_name!r}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_test_metrics(output: str) -> dict[str, str]:
    """Parse evaluate.py metrics from stdout."""
    match = TEST_RE.search(output)
    if not match:
        raise ValueError("could not parse test metrics from evaluate.py output")
    return match.groupdict()


def format_metric(value, *, digits: int = 6) -> str:
    """Format numeric metrics for stable Markdown output."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def display_path(path: Path) -> str:
    """Return a compact path when it is inside the project, else an absolute path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_eval_metrics(
    best_path: Path,
    *,
    plot_dir: Path | None = None,
) -> tuple[dict, Path | None]:
    """Load metrics.json written by evaluate.py when available."""
    metrics_path = (
        plot_dir / "metrics.json"
        if plot_dir is not None
        else best_path.parent / "eval_plots" / "metrics.json"
    )
    if not metrics_path.exists():
        return {}, None
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics, metrics_path


def normalize_results_path(path: Path) -> Path:
    """Keep legacy --results *.csv invocations on the Markdown-only output."""
    if path.suffix.lower() == ".csv":
        return path.with_suffix(".md")
    return path


def normalize_result_row(row: dict[str, str]) -> dict[str, str]:
    """Return a row with exactly the current Markdown result columns."""
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key in RESULT_COLUMNS:
            normalized[key] = "" if value is None else str(value)
    if "experiment" in normalized and normalized["experiment"]:
        normalized["experiment"] = normalized["experiment"].split("_")[0]
    return {column: normalized.get(column, "") for column in RESULT_COLUMNS}


def merge_result_rows(
    base: dict[str, str],
    overlay: dict[str, str],
) -> dict[str, str]:
    """Overlay non-empty values while preserving richer legacy data."""
    merged = normalize_result_row(base)
    for column, value in normalize_result_row(overlay).items():
        if value.strip():
            merged[column] = value
    return merged


def escape_markdown_cell(value: str) -> str:
    """Escape table metacharacters in a Markdown cell."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def split_markdown_row(line: str) -> list[str]:
    """Split one Markdown table row, respecting escaped pipes."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line:
        if escaped:
            if char in {"|", "\\"}:
                current.append(char)
            else:
                current.extend(["\\", char])
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells)


def load_markdown_results(results_path: Path) -> list[dict[str, str]]:
    """Load existing Markdown rows so resumed runs keep prior experiments."""
    if not results_path.exists() or results_path.stat().st_size == 0:
        return []
    table_lines = [
        line
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("|")
    ]
    if len(table_lines) < 2:
        return []
    headers = split_markdown_row(table_lines[0])
    rows: list[dict[str, str]] = []
    for line in table_lines[1:]:
        cells = split_markdown_row(line)
        if is_markdown_separator(cells):
            continue
        row = dict(zip(headers, cells))
        if row.get("experiment"):
            rows.append(normalize_result_row(row))
    return rows


def load_legacy_csv_results(results_path: Path) -> list[dict[str, str]]:
    """Read older CSV output as an input-only migration path."""
    csv_path = results_path.with_suffix(".csv")
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            normalize_result_row(row)
            for row in reader
            if row.get("experiment")
        ]


def load_existing_results(results_path: Path) -> list[dict[str, str]]:
    """Load prior Markdown rows, enriched from legacy CSV when present."""
    rows: list[dict[str, str]] = []
    positions: dict[str, int] = {}

    for row in load_legacy_csv_results(results_path):
        experiment = row.get("experiment", "")
        if not experiment:
            continue
        positions[experiment] = len(rows)
        rows.append(row)

    for row in load_markdown_results(results_path):
        experiment = row.get("experiment", "")
        if not experiment:
            continue
        if experiment in positions:
            rows[positions[experiment]] = merge_result_rows(
                rows[positions[experiment]],
                row,
            )
        else:
            positions[experiment] = len(rows)
            rows.append(row)
    return rows


def upsert_result(rows: list[dict[str, str]], row: dict[str, str]) -> None:
    """Add or replace one experiment row."""
    row = normalize_result_row(row)
    for idx, existing in enumerate(rows):
        if existing.get("experiment") == row.get("experiment"):
            rows[idx] = row
            return
    rows.append(row)


def write_results_markdown(results_path: Path, rows: list[dict[str, str]]) -> None:
    """Write the single Markdown result table."""
    if not rows:
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(RESULT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(RESULT_COLUMNS)) + " |",
    ]
    for row in rows:
        normalized = normalize_result_row(row)
        values = [
            escape_markdown_cell(normalized[column])
            for column in RESULT_COLUMNS
        ]
        lines.append("| " + " | ".join(values) + " |")
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_config(cfg: dict) -> dict[str, str]:
    """Return config fields used in result rows."""
    return {
        "experiment": cfg["experiment"]["name"],
    }


def mamba_layers_requested(cfg: dict) -> int:
    """Return total branch Mamba layers requested by a config."""
    model_name = cfg.get("model", {}).get("name", "").lower()
    if model_name not in {"dual_range", "dual_plane_mamba"}:
        return 0
    model_cfg = cfg.get("model", {}).get(model_name, {}) or {}
    mamba_cfg = model_cfg.get("mamba", {}) or {}
    layers = int(mamba_cfg.get("sa_layers", 0)) + int(mamba_cfg.get("wa_layers", 0))
    peak_cfg = model_cfg.get("peak_branch", {}) or {}
    if peak_cfg.get("enabled", False) and str(peak_cfg.get("encoder", "mamba")).lower() == "mamba":
        peak_mamba_cfg = peak_cfg.get("mamba", {}) or {}
        layers += int(peak_mamba_cfg.get("layers", peak_cfg.get("mamba_layers", 2)))
    return layers


def mamba_ssm_import_error(backend: str = "mamba_ssm") -> str | None:
    """Return the mamba-ssm import error, or None when the backend is usable."""
    try:
        if backend in {"mamba2", "mamba2_ssm"}:
            from mamba_ssm import Mamba2  # noqa: F401
        else:
            from mamba_ssm import Mamba  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def validate_mamba_backend(cfg: dict) -> None:
    """Require mamba-ssm whenever a config requests Mamba layers."""
    if mamba_layers_requested(cfg) <= 0:
        return

    model_name = cfg.get("model", {}).get("name", "").lower()
    model_cfg = cfg.get("model", {}).get(model_name, {}) or {}
    mamba_cfg = model_cfg.get("mamba", {}) or {}
    backend = str(mamba_cfg.get("backend", "auto")).lower()
    if backend not in {"auto", "mamba_ssm", "mamba2", "mamba2_ssm"}:
        raise ValueError(f"unknown mamba backend: {backend!r}")

    import_backend = "mamba2_ssm" if backend in {"mamba2", "mamba2_ssm"} else "mamba_ssm"
    import_error = mamba_ssm_import_error(import_backend)
    if backend == "auto" and import_error is not None:
        raise RuntimeError(
            f"{cfg['experiment']['name']} requests Mamba layers but mamba-ssm "
            "is not importable. Install mamba-ssm/causal-conv1d on the CUDA "
            "runner, or set mamba.backend: mamba_ssm to require it explicitly. "
            f"Import error: {import_error}"
        )
    if backend in {"mamba_ssm", "mamba2", "mamba2_ssm"} and import_error is not None:
        raise RuntimeError(
            f"{cfg['experiment']['name']} requires {import_backend}, but the package "
            f"is not importable. Import error: {import_error}"
        )


def metrics_row(
    metrics: dict,
    fallback: dict[str, str],
) -> dict[str, str]:
    """Build the core test result fields from metrics.json or stdout fallback."""
    if metrics:
        return {
            "test_acc1": format_metric(metrics.get("acc1")),
            "test_acc5": format_metric(metrics.get("acc5")),
            "test_macro_f1": format_metric(metrics.get("macro_f1")),
        }
    return {
        "test_acc1": fallback.get("acc1") or "",
        "test_acc5": fallback.get("acc5") or "",
        "test_macro_f1": fallback.get("macro_f1") or "",
    }


def resolve_config_paths(
    preset: str | None,
    configs: list[str] | None,
) -> list[str]:
    """Resolve config paths from a preset and/or explicit paths.

    With no CLI selection, run the full matrix. With --configs alone, run only
    those explicit paths. With --preset plus --configs, append the explicit paths
    after the selected preset.
    """
    if preset is None:
        selected = list(configs) if configs is not None else list(CONFIG_PRESETS["full"])
    else:
        selected = list(CONFIG_PRESETS[preset])
        if configs:
            selected.extend(configs)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PXRD experiment matrix")
    ap.add_argument(
        "--preset",
        choices=sorted(CONFIG_PRESETS),
        default=None,
        help=(
            "Named config group. Omit both --preset and --configs to run full. "
            "Use main for the unified main-result configs, or cs for "
            "crystal-system classifiers. Use peak_token for the compressed "
            "peak-branch validation pair."
        ),
    )
    ap.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help=(
            "Explicit config paths. Used alone, this runs only the listed paths; "
            "with --preset, they are appended after that preset."
        ),
    )
    ap.add_argument(
        "--results",
        default=None,
        help=(
            "Result Markdown path. Defaults to main_results for --preset main, "
            "cs_results for --preset cs, peak_token for --preset peak_token, "
            "else dual_range_matrix. Legacy .csv paths are mapped to .md."
        ),
    )
    ap.add_argument(
        "--logs-dir",
        default=None,
        help=(
            "Log directory. Defaults to main_results for --preset main, "
            "cs_results for --preset cs, peak_token for --preset peak_token, "
            "else dual_range_matrix."
        ),
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate latest checkpoints for the given configs.",
    )
    ap.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Forward to evaluate.py --max-samples for quick smoke tests.",
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Forward to evaluate.py --batch-size.",
    )
    ap.add_argument(
        "--eval-plot-dir-root",
        default=None,
        help="Optional root for evaluate.py plot dirs; each run gets a subdirectory.",
    )
    ap.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
        help="Forward to evaluate.py --num-workers; use 0 in restricted sandboxes.",
    )
    ap.add_argument(
        "--eval-no-pin-memory",
        action="store_true",
        help="Forward to evaluate.py --no-pin-memory.",
    )
    ap.add_argument(
        "--eval-skip-occlusion",
        action="store_true",
        help="Forward to evaluate.py --skip-occlusion.",
    )
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("WANDB_DIR", str(PROJECT_ROOT / "wandb"))

    if args.preset == "main":
        default_results = MAIN_RESULTS
        default_logs_dir = MAIN_LOGS_DIR
    elif args.preset == "cs":
        default_results = CS_RESULTS
        default_logs_dir = CS_LOGS_DIR
    elif args.preset == "peak_token":
        default_results = PEAK_TOKEN_RESULTS
        default_logs_dir = PEAK_TOKEN_LOGS_DIR
    else:
        default_results = DEFAULT_RESULTS
        default_logs_dir = DEFAULT_LOGS_DIR
    results_arg = args.results or default_results
    logs_dir_arg = args.logs_dir or default_logs_dir

    results_path = normalize_results_path(Path(results_arg))
    rows: list[dict[str, str]] = load_existing_results(results_path)
    for config_path_str in resolve_config_paths(args.preset, args.configs):
        config_path = Path(config_path_str)
        cfg = load_config(config_path)
        validate_mamba_backend(cfg)
        run_name = cfg["experiment"]["name"]
        print(f"\n=== {run_name}: {config_path} ===", flush=True)
        started_at = time.time()

        if not args.skip_train:
            train_log = Path(logs_dir_arg) / f"{run_name}.train.log"
            run_and_log(
                [sys.executable, "scripts/train.py", "--config", str(config_path)],
                train_log,
                env,
            )
            best_path = newest_checkpoint(run_name, started_at)
        else:
            best_path = newest_checkpoint(run_name, 0.0)

        eval_log = Path(logs_dir_arg) / f"{run_name}.eval.log"
        eval_cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", str(best_path)]
        eval_plot_dir = None
        if args.eval_plot_dir_root is not None:
            eval_plot_dir = Path(args.eval_plot_dir_root) / run_name
            eval_cmd.extend([
                "--plot-dir",
                str(eval_plot_dir),
            ])
        if args.eval_max_samples is not None:
            eval_cmd.extend(["--max-samples", str(args.eval_max_samples)])
        if args.eval_batch_size is not None:
            eval_cmd.extend(["--batch-size", str(args.eval_batch_size)])
        if args.eval_num_workers is not None:
            eval_cmd.extend(["--num-workers", str(args.eval_num_workers)])
        if args.eval_no_pin_memory:
            eval_cmd.append("--no-pin-memory")
        if args.eval_skip_occlusion:
            eval_cmd.append("--skip-occlusion")
        eval_output = run_and_log(
            eval_cmd,
            eval_log,
            env,
        )

        test_metrics = parse_test_metrics(eval_output)
        eval_metrics, eval_metrics_path = load_eval_metrics(
            best_path,
            plot_dir=eval_plot_dir,
        )
        row = {
            **summarize_config(cfg),
            **metrics_row(eval_metrics, test_metrics),
        }
        upsert_result(rows, row)
        write_results_markdown(results_path, rows)
        print(f"Recorded result for {run_name}: {row}", flush=True)


if __name__ == "__main__":
    main()
