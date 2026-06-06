"""Run a sequence of training/evaluation experiments and summarize results.

Usage:
    python3 scripts/run_experiments.py
    python3 scripts/run_experiments.py --preset non_mamba
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
from importlib.util import find_spec
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


DEFAULT_CONFIGS = [
    "configs/experiments/e02_resnet_deep_label_smoothing.yaml",
    "configs/experiments/e07_wa_only_resnet_label_smoothing.yaml",
    "configs/experiments/e08_dual_concat_resnet.yaml",
    "configs/experiments/e09_dual_gated_resnet.yaml",
    "configs/experiments/e10_dual_gated_mamba.yaml",
    "configs/experiments/e11_dual_gated_kan.yaml",
    "configs/experiments/e12_dual_gated_mamba_kan.yaml",
    "configs/experiments/e13_dual_gated_mamba_kan_aux.yaml",
    "configs/experiments/e14_sa_only_resnet_ablation.yaml",
]

NON_MAMBA_CONFIGS = [
    "configs/experiments/e02_resnet_deep_label_smoothing.yaml",
    "configs/experiments/e07_wa_only_resnet_label_smoothing.yaml",
    "configs/experiments/e14_sa_only_resnet_ablation.yaml",
    "configs/experiments/e08_dual_concat_resnet.yaml",
    "configs/experiments/e09_dual_gated_resnet.yaml",
    "configs/experiments/e11_dual_gated_kan.yaml",
]

MAMBA_CONFIGS = [
    "configs/experiments/e10_dual_gated_mamba.yaml",
    "configs/experiments/e12_dual_gated_mamba_kan.yaml",
    "configs/experiments/e13_dual_gated_mamba_kan_aux.yaml",
]

CONFIG_PRESETS = {
    "full": DEFAULT_CONFIGS,
    "non_mamba": NON_MAMBA_CONFIGS,
    "mamba": MAMBA_CONFIGS,
}

TEST_RE = re.compile(
    r"Test loss=(?P<loss>[0-9.]+)\s+"
    r"acc1=(?P<acc1>[0-9.]+)\s+"
    r"acc5=(?P<acc5>[0-9.]+)\s+"
    r"macro=(?P<macro>[0-9.]+)\s+"
    r"(?:macro_f1=(?P<macro_f1>[0-9.]+)\s+)?"
    r"(?:rare=(?P<rare>[0-9.]+|nan)\s+)?"
    r"(?:aux=(?P<aux>[0-9.]+|nan)\s+)?"
    r"(?:gate=(?P<gate>[0-9.]+|nan)\s+)?"
    r"\(N=(?P<n>[0-9]+)\)"
)

RESULT_FIELDNAMES = [
    "experiment",
    "config",
    "model",
    "model_params",
    "optimizer",
    "lr",
    "weight_decay",
    "loss",
    "sampler",
    "best_epoch",
    "monitor",
    "monitor_score",
    "val_acc1",
    "val_acc5",
    "val_macro_acc1",
    "val_rare_acc1",
    "val_aux_loss",
    "val_gate_mean",
    "val_balanced_acc1_macro",
    "test_loss",
    "test_acc1",
    "test_acc5",
    "test_macro_acc1",
    "test_macro_f1",
    "test_rare_acc1",
    "test_aux_loss",
    "test_gate_mean",
    "sa_mamba_backend",
    "wa_mamba_backend",
    "occlusion_acc1",
    "occlusion_macro_acc1",
    "occlusion_macro_f1",
    "occlusion_rare_acc1",
    "occlusion_delta_acc1",
    "occlusion_delta_macro_acc1",
    "occlusion_delta_macro_f1",
    "occlusion_delta_rare_acc1",
    "test_n",
    "eval_metrics",
    "checkpoint",
    "wandb_run",
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


def newest_wandb_run(started_at: float) -> str:
    """Return the newest local W&B run directory created after started_at."""
    candidates: list[Path] = []
    for wandb_dir in [PROJECT_ROOT / "wandb", PROJECT_ROOT / "wandb" / "wandb"]:
        if not wandb_dir.exists():
            continue
        candidates.extend(
            p for p in wandb_dir.glob("*run-*")
            if p.is_dir() and p.stat().st_mtime >= started_at
        )
    if not candidates:
        return ""
    return str(max(candidates, key=lambda p: p.stat().st_mtime).relative_to(PROJECT_ROOT))


def parse_test_metrics(output: str) -> dict[str, str]:
    """Parse evaluate.py metrics from stdout."""
    match = TEST_RE.search(output)
    if not match:
        raise ValueError("could not parse test metrics from evaluate.py output")
    return match.groupdict()


def format_metric(value, *, digits: int = 6) -> str:
    """Format numeric metrics for stable CSV output."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def checkpoint_metrics(best_path: Path) -> dict[str, str]:
    """Load validation metrics from a checkpoint."""
    ckpt = torch.load(best_path, map_location="cpu")
    return {
        "best_epoch": str(int(ckpt["epoch"]) + 1),
        "monitor": str(ckpt.get("monitor", "")),
        "monitor_score": format_metric(ckpt.get("monitor_score")),
        "val_acc1": format_metric(ckpt.get("val_acc1")),
        "val_acc5": format_metric(ckpt.get("val_acc5")),
        "val_macro_acc1": format_metric(ckpt.get("val_macro_acc1")),
        "val_rare_acc1": format_metric(ckpt.get("val_rare_acc1")),
        "val_aux_loss": format_metric(ckpt.get("val_aux_loss")),
        "val_gate_mean": format_metric(ckpt.get("val_gate_mean")),
        "val_balanced_acc1_macro": format_metric(
            ckpt.get("val_balanced_acc1_macro")
        ),
    }


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


def load_existing_results(results_path: Path) -> list[dict[str, str]]:
    """Load existing result rows so resumed runs keep prior experiments."""
    if not results_path.exists() or results_path.stat().st_size == 0:
        return []
    with results_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        return [
            {field: row.get(field, "") for field in RESULT_FIELDNAMES}
            for row in reader
            if row.get("experiment")
        ]


def upsert_result(rows: list[dict[str, str]], row: dict[str, str]) -> None:
    """Add or replace one experiment row."""
    for idx, existing in enumerate(rows):
        if existing.get("experiment") == row.get("experiment"):
            rows[idx] = row
            return
    rows.append(row)


def append_results(results_path: Path, rows: list[dict[str, str]]) -> None:
    """Write results CSV from all rows collected in this run."""
    if not rows:
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(results_path: Path, rows: list[dict[str, str]]) -> None:
    """Write a compact Markdown result table."""
    md_path = results_path.with_suffix(".md")
    headers = [
        "experiment",
        "model",
        "loss",
        "best_epoch",
        "val_acc1",
        "val_macro",
        "val_rare",
        "test_acc1",
        "test_acc5",
        "test_macro",
        "test_f1",
        "test_rare",
        "gate",
        "sa_mamba",
        "wa_mamba",
        "occ_delta_acc1",
        "occ_delta_macro",
        "occ_delta_f1",
        "occ_delta_rare",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["experiment"],
            row["model"],
            row["loss"],
            row["best_epoch"],
            row["val_acc1"],
            row["val_macro_acc1"],
            row.get("val_rare_acc1", ""),
            row["test_acc1"],
            row["test_acc5"],
            row["test_macro_acc1"],
            row.get("test_macro_f1", ""),
            row.get("test_rare_acc1", ""),
            row.get("test_gate_mean", ""),
            row.get("sa_mamba_backend", ""),
            row.get("wa_mamba_backend", ""),
            row.get("occlusion_delta_acc1", ""),
            row.get("occlusion_delta_macro_acc1", ""),
            row.get("occlusion_delta_macro_f1", ""),
            row.get("occlusion_delta_rare_acc1", ""),
        ]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_config(cfg: dict) -> dict[str, str]:
    """Return config fields used in result rows."""
    model_name = cfg["model"]["name"]
    model_cfg = cfg["model"].get(model_name, {})
    if model_name == "resnet1d":
        model_params = (
            f"base{model_cfg.get('base_channels')},"
            f"blocks{model_cfg.get('blocks_per_stage')}"
        )
    elif model_name == "bigru_patch":
        model_params = (
            f"patch{model_cfg.get('patch_len')},"
            f"stride{model_cfg.get('stride')},"
            f"d{model_cfg.get('d_model')},"
            f"h{model_cfg.get('hidden_size')},"
            f"layers{model_cfg.get('num_layers')}"
        )
    elif model_name == "patchtst":
        model_params = (
            f"patch{model_cfg.get('patch_len')},"
            f"stride{model_cfg.get('stride')},"
            f"d{model_cfg.get('d_model')},"
            f"heads{model_cfg.get('n_heads')},"
            f"layers{model_cfg.get('num_layers')}"
        )
    elif model_name == "dual_range":
        model_params = (
            f"sa{model_cfg.get('use_sa', True)},"
            f"wa{model_cfg.get('use_wa', True)},"
            f"fusion={model_cfg.get('fusion', 'concat')},"
            f"head={model_cfg.get('head', 'mlp')},"
            f"mamba={model_cfg.get('mamba', {})},"
            f"aux={model_cfg.get('aux_heads', False)}"
        )
    else:
        model_params = str(model_cfg)
    optim_cfg = cfg.get("optim", {})
    return {
        "experiment": cfg["experiment"]["name"],
        "model": model_name,
        "model_params": model_params,
        "optimizer": optim_cfg.get("name", "adamw"),
        "lr": str(optim_cfg.get("lr", "")),
        "weight_decay": str(optim_cfg.get("weight_decay", "")),
        "loss": cfg["loss"]["name"],
        "sampler": cfg.get("sampler", {}).get("name", "none"),
    }


def mamba_layers_requested(cfg: dict) -> int:
    """Return total branch Mamba layers requested by a config."""
    if cfg.get("model", {}).get("name", "").lower() != "dual_range":
        return 0
    mamba_cfg = cfg.get("model", {}).get("dual_range", {}).get("mamba", {}) or {}
    return int(mamba_cfg.get("sa_layers", 0)) + int(mamba_cfg.get("wa_layers", 0))


def validate_mamba_backend(cfg: dict, *, allow_fallback: bool) -> None:
    """Guard official matrix runs against accidentally using the local fallback."""
    if mamba_layers_requested(cfg) <= 0:
        return

    mamba_cfg = cfg.get("model", {}).get("dual_range", {}).get("mamba", {}) or {}
    backend = str(mamba_cfg.get("backend", "auto")).lower()
    if backend not in {"auto", "mamba_ssm", "local"}:
        raise ValueError(f"unknown mamba backend: {backend!r}")

    has_mamba_ssm = find_spec("mamba_ssm") is not None
    if backend == "local":
        if allow_fallback:
            return
        raise RuntimeError(
            f"{cfg['experiment']['name']} requests mamba.backend: local. "
            "Use --allow-mamba-fallback only for CPU/smoke runs; official "
            "Mamba rows must use mamba_ssm."
        )
    if backend == "auto" and not has_mamba_ssm and not allow_fallback:
        raise RuntimeError(
            f"{cfg['experiment']['name']} requests Mamba layers but mamba-ssm "
            "is not importable. Install mamba-ssm/causal-conv1d on the CUDA "
            "runner, set mamba.backend: mamba_ssm, or pass "
            "--allow-mamba-fallback for a non-final smoke run."
        )
    if backend == "mamba_ssm" and not has_mamba_ssm:
        raise RuntimeError(
            f"{cfg['experiment']['name']} requires mamba_ssm, but the package "
            "is not importable."
        )


def metrics_row(
    metrics: dict,
    metrics_path: Path | None,
    fallback: dict[str, str],
) -> dict[str, str]:
    """Build test/occlusion result fields from metrics.json or stdout fallback."""
    if metrics:
        occlusion = metrics.get("occlusion", {}) or {}
        return {
            "test_loss": format_metric(metrics.get("loss")),
            "test_acc1": format_metric(metrics.get("acc1")),
            "test_acc5": format_metric(metrics.get("acc5")),
            "test_macro_acc1": format_metric(metrics.get("macro_acc1")),
            "test_macro_f1": format_metric(metrics.get("macro_f1")),
            "test_rare_acc1": format_metric(metrics.get("rare_acc1")),
            "test_aux_loss": format_metric(metrics.get("aux_loss")),
            "test_gate_mean": format_metric(metrics.get("gate_mean")),
            "sa_mamba_backend": str(metrics.get("sa_mamba_backend") or ""),
            "wa_mamba_backend": str(metrics.get("wa_mamba_backend") or ""),
            "occlusion_acc1": format_metric(occlusion.get("acc1")),
            "occlusion_macro_acc1": format_metric(occlusion.get("macro_acc1")),
            "occlusion_macro_f1": format_metric(occlusion.get("macro_f1")),
            "occlusion_rare_acc1": format_metric(occlusion.get("rare_acc1")),
            "occlusion_delta_acc1": format_metric(occlusion.get("delta_acc1")),
            "occlusion_delta_macro_acc1": format_metric(
                occlusion.get("delta_macro_acc1")
            ),
            "occlusion_delta_macro_f1": format_metric(
                occlusion.get("delta_macro_f1")
            ),
            "occlusion_delta_rare_acc1": format_metric(
                occlusion.get("delta_rare_acc1")
            ),
            "test_n": str(int(metrics.get("n", 0))),
            "eval_metrics": (
                display_path(metrics_path)
                if metrics_path is not None
                else ""
            ),
        }
    return {
        "test_loss": fallback["loss"],
        "test_acc1": fallback["acc1"],
        "test_acc5": fallback["acc5"],
        "test_macro_acc1": fallback["macro"],
        "test_macro_f1": fallback.get("macro_f1") or "",
        "test_rare_acc1": fallback.get("rare") or "",
        "test_aux_loss": fallback.get("aux") or "",
        "test_gate_mean": fallback.get("gate") or "",
        "sa_mamba_backend": "",
        "wa_mamba_backend": "",
        "occlusion_acc1": "",
        "occlusion_macro_acc1": "",
        "occlusion_macro_f1": "",
        "occlusion_rare_acc1": "",
        "occlusion_delta_acc1": "",
        "occlusion_delta_macro_acc1": "",
        "occlusion_delta_macro_f1": "",
        "occlusion_delta_rare_acc1": "",
        "test_n": fallback["n"],
        "eval_metrics": "",
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
            "Use non_mamba to run baseline/WA/SA/concat/gated/KAN rows before "
            "CUDA Mamba dependencies are available."
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
    ap.add_argument("--results", default="experiments/dual_range_matrix/results.csv")
    ap.add_argument("--logs-dir", default="experiments/dual_range_matrix/logs")
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
    ap.add_argument(
        "--allow-mamba-fallback",
        action="store_true",
        help=(
            "Allow Mamba configs to run with the local selective-sequence "
            "fallback when mamba-ssm is unavailable. Use only for smoke tests, "
            "not final Mamba claims."
        ),
    )
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_DIR", str(PROJECT_ROOT / "wandb"))

    results_path = Path(args.results)
    rows: list[dict[str, str]] = load_existing_results(results_path)
    for config_path_str in resolve_config_paths(args.preset, args.configs):
        config_path = Path(config_path_str)
        cfg = load_config(config_path)
        validate_mamba_backend(cfg, allow_fallback=args.allow_mamba_fallback)
        run_name = cfg["experiment"]["name"]
        print(f"\n=== {run_name}: {config_path} ===", flush=True)
        started_at = time.time()

        if not args.skip_train:
            train_log = Path(args.logs_dir) / f"{run_name}.train.log"
            run_and_log(
                [sys.executable, "scripts/train.py", "--config", str(config_path)],
                train_log,
                env,
            )
            best_path = newest_checkpoint(run_name, started_at)
        else:
            best_path = newest_checkpoint(run_name, 0.0)

        eval_log = Path(args.logs_dir) / f"{run_name}.eval.log"
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
        val_metrics = checkpoint_metrics(best_path)
        previous_wandb_run = next(
            (
                existing.get("wandb_run", "")
                for existing in rows
                if existing.get("experiment") == run_name
            ),
            "",
        )
        row = {
            **summarize_config(cfg),
            "config": str(config_path),
            **val_metrics,
            **metrics_row(eval_metrics, eval_metrics_path, test_metrics),
            "checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
            "wandb_run": newest_wandb_run(started_at) or previous_wandb_run,
        }
        upsert_result(rows, row)
        append_results(results_path, rows)
        write_markdown(results_path, rows)
        print(f"Recorded result for {run_name}: {row}", flush=True)


if __name__ == "__main__":
    main()
