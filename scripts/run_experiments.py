"""Run a sequence of training/evaluation experiments and summarize results.

Usage:
    python3 scripts/run_experiments.py
    python3 scripts/run_experiments.py --configs configs/experiments/e02_resnet_deep_weighted_ce_pure.yaml
"""
from __future__ import annotations

import argparse
import csv
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


DEFAULT_CONFIGS = [
    "configs/experiments/e02_resnet_deep_weighted_ce_pure.yaml",
    "configs/experiments/e02_resnet_deep_label_smoothing.yaml",
    "configs/experiments/e02_resnet_deep_focal.yaml",
]

TEST_RE = re.compile(
    r"Test loss=(?P<loss>[0-9.]+)\s+"
    r"acc1=(?P<acc1>[0-9.]+)\s+"
    r"acc5=(?P<acc5>[0-9.]+)\s+"
    r"macro=(?P<macro>[0-9.]+)\s+"
    r"(?:macro_f1=(?P<macro_f1>[0-9.]+)\s+)?"
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
    "val_balanced_acc1_macro",
    "test_loss",
    "test_acc1",
    "test_acc5",
    "test_macro_acc1",
    "test_macro_f1",
    "test_n",
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


def checkpoint_metrics(best_path: Path) -> dict[str, str]:
    """Load validation metrics from a checkpoint."""
    ckpt = torch.load(best_path, map_location="cpu")
    return {
        "best_epoch": str(int(ckpt["epoch"]) + 1),
        "monitor": str(ckpt.get("monitor", "")),
        "monitor_score": f"{float(ckpt.get('monitor_score', float('nan'))):.6f}",
        "val_acc1": f"{float(ckpt.get('val_acc1', float('nan'))):.6f}",
        "val_acc5": f"{float(ckpt.get('val_acc5', float('nan'))):.6f}",
        "val_macro_acc1": f"{float(ckpt.get('val_macro_acc1', float('nan'))):.6f}",
        "val_balanced_acc1_macro": (
            f"{float(ckpt.get('val_balanced_acc1_macro', float('nan'))):.6f}"
        ),
    }


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
        "optimizer",
        "loss",
        "sampler",
        "best_epoch",
        "val_acc1",
        "val_macro",
        "test_acc1",
        "test_acc5",
        "test_macro",
        "test_f1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["experiment"],
            row["model"],
            row["optimizer"],
            row["loss"],
            row["sampler"],
            row["best_epoch"],
            row["val_acc1"],
            row["val_macro_acc1"],
            row["test_acc1"],
            row["test_acc5"],
            row["test_macro_acc1"],
            row.get("test_macro_f1", ""),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PXRD experiment matrix")
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--results", default="experiments/e02_loss_ablation/results.csv")
    ap.add_argument("--logs-dir", default="experiments/e02_loss_ablation/logs")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate latest checkpoints for the given configs.",
    )
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_DIR", str(PROJECT_ROOT / "wandb"))

    results_path = Path(args.results)
    rows: list[dict[str, str]] = load_existing_results(results_path)
    for config_path_str in args.configs:
        config_path = Path(config_path_str)
        cfg = load_config(config_path)
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
        eval_output = run_and_log(
            [sys.executable, "scripts/evaluate.py", "--checkpoint", str(best_path)],
            eval_log,
            env,
        )

        test_metrics = parse_test_metrics(eval_output)
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
            "test_loss": test_metrics["loss"],
            "test_acc1": test_metrics["acc1"],
            "test_acc5": test_metrics["acc5"],
            "test_macro_acc1": test_metrics["macro"],
            "test_macro_f1": test_metrics.get("macro_f1") or "",
            "test_n": test_metrics["n"],
            "checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
            "wandb_run": newest_wandb_run(started_at) or previous_wandb_run,
        }
        upsert_result(rows, row)
        append_results(results_path, rows)
        write_markdown(results_path, rows)
        print(f"Recorded result for {run_name}: {row}", flush=True)


if __name__ == "__main__":
    main()
