# Experiments

This directory stores durable experiment outputs only:

- `results.md`: canonical test metrics for retained configs.
- `figures/`: curated figures used in reports or slides.

Detailed per-class metrics, confusion pairs, and normalized confusion matrices
remain beside each checkpoint under `checkpoints/<run>/eval_plots/`.

Training and evaluation logs are transient. New runs write them to
`runs/experiment_logs/`, which is ignored by Git. `main_results/` is retained
temporarily because a legacy experiment chain is still writing there; new runs
must not use it. After that chain finishes, migrate its final row to
`results.md` and remove the legacy directory.

## Result Rules

- Use the short ID from `configs/README.md` as `experiment`.
- Keep one row per experiment; reruns replace that row.
- Prefer `checkpoints/<run>/eval_plots/metrics.json` over old Markdown tables.
- An empty metric means the historical evaluation did not record it.
- Broad exploration should rank models on validation data. Reserve repeated
  test evaluation for frozen finalists.

Run selected experiments with the canonical output paths:

```bash
python3 scripts/run_experiments.py --configs \
  configs/main/m01_mamba.yaml \
  configs/mobile/mx03_no_identity.yaml
```
