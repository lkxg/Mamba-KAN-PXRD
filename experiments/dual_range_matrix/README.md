# Dual-Range Matrix

This experiment set implements the SA/WA plan for PXRD space-group classification.

## Fixed Baseline

- `configs/experiments/e02_resnet_deep_label_smoothing.yaml`
  is the fixed current strong baseline: full-range `ResNet1D + label_smoothing`.
- Report and compare every new dual-range model against its Top-1, Top-5,
  Top-10, Macro Acc, Macro F1, and rare-class accuracy.

## Range Definition

- SA: `5-15` degrees 2theta.
- WA: `10-90` degrees 2theta.
- The `10-15` degree overlap is intentional, so peaks near the boundary are not
  cut by a hard split.

## Matrix

| ID | Config | Purpose |
| --- | --- | --- |
| E07 | `e07_wa_only_resnet_label_smoothing.yaml` | WA-only dual-range ResNet baseline |
| E08 | `e08_dual_concat_resnet.yaml` | SA + WA concat, no Mamba/KAN |
| E09 | `e09_dual_gated_resnet.yaml` | SA + WA gated fusion |
| E11 | `e11_dual_gated_kan.yaml` | Gated fusion + KAN head |
| E14 | `e14_sa_only_resnet_ablation.yaml` | SA-only ablation |

## Long-Tail Handling

- Training configs use `val_balanced_acc1_macro` as the checkpoint monitor.
- Rare classes are reported for train counts `1-100`.
- The main dual-range configs use mild `weighted_ce + label_smoothing=0.03`;
  E07 uses pure label smoothing to mirror the fixed baseline loss.
- Class-balanced sampling remains available in the shared training code, but the
  matrix defaults to no resampling so macro/rare gains are attributable to the
  architecture first.

## Main Mamba Runs

Mamba-based final comparisons now live in `configs/main/` and are launched with
`python3 scripts/run_experiments.py --preset main`. `scripts/run_experiments.py`
still guards any config that requests Mamba layers: final Mamba claims should be
based on runs whose evaluation metrics report `mamba_ssm` as the SA/WA backend.

## Evaluation Outputs

`scripts/evaluate.py` writes standard metrics and interpretation artifacts:

- `metrics.json`: Top-1, Top-5, Top-10, Macro Acc, Macro F1, rare acc, aux
  loss, gate mean, SA/WA Mamba backend, and low-angle occlusion deltas.
- `06_gate_mean_distribution.png`: sample-level SA gate distribution.
- `07_gate_by_crystal_system.png` and `gate_by_crystal_system.csv`: gate means by
  crystal system.
- `confusion_cases.csv` and `confusion_pairs.csv`: high-confidence mistakes and
  frequent true/predicted space-group confusions.
- `per_class_metrics.csv`: per-class support, Top-1 accuracy, and rare flag.
- Low-angle occlusion defaults to masking `5-15` degrees and re-evaluating.

## Completion Gate

Use the audit script to separate implemented code/config from completed formal
results:

```bash
python3 analysis/scripts/preflight_dual_range.py
python3 analysis/scripts/smoke_dual_range_forward.py
python3 analysis/scripts/audit_dual_range.py --gate implementation
python3 analysis/scripts/audit_dual_range.py --gate evidence
python3 analysis/scripts/audit_dual_range.py --gate final
```

The preflight checks CUDA, data files, configs, baseline checkpoint, and result
schema. The forward smoke checks current matrix model output contracts without
training. The audit reports `PASS`, `PENDING`, and `FAIL` evidence with phased
gates:
`implementation` checks code/config, `evidence` also requires formal result rows
and artifacts. `--strict` remains an alias for `--gate final`.

## Run

Full matrix:

```bash
python3 analysis/scripts/preflight_dual_range.py
python3 analysis/scripts/smoke_dual_range_forward.py
python3 analysis/scripts/audit_dual_range.py --gate implementation
python3 scripts/run_experiments.py
python3 analysis/scripts/summarize_dual_range.py
python3 analysis/scripts/audit_dual_range.py --gate final
```

Restricted CPU/sandbox smoke test:

```bash
python3 analysis/scripts/preflight_dual_range.py \
  --allow-cpu
python3 analysis/scripts/smoke_dual_range_forward.py
python3 scripts/run_experiments.py \
  --configs configs/experiments/e02_resnet_deep_label_smoothing.yaml \
  --skip-train \
  --eval-max-samples 8 \
  --eval-batch-size 4 \
  --eval-num-workers 0 \
  --eval-no-pin-memory \
  --eval-plot-dir-root /tmp/mkpxrd_runexp_smoke \
  --results /tmp/mkpxrd_runexp_smoke/results.md \
  --logs-dir /tmp/mkpxrd_runexp_smoke/logs
python3 analysis/scripts/summarize_dual_range.py \
  --results /tmp/mkpxrd_runexp_smoke/results.md \
  --out /tmp/mkpxrd_runexp_smoke/summary.md
python3 analysis/scripts/audit_dual_range.py \
  --results /tmp/mkpxrd_runexp_smoke/results.md \
  --gate implementation
```

On H100/CUDA, leave the evaluation overrides off so the configs use their
intended DataLoader settings.
