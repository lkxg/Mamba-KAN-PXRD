# Experiment Records

Use this directory as the experiment index.

## Current Main Result

- `e02_loss_ablation/` contains the clean E02 loss ablation:
  weighted CE, label smoothing, and focal loss under the same E02 base setup.
- Primary table: `e02_loss_ablation/results.md`
- Logs: `e02_loss_ablation/logs/`
- Plan/config notes: `e02_loss_ablation/plan.md`

## Dual-Range Matrix

- `dual_range_matrix/` is the current SA/WA architecture matrix.
- Fixed strong baseline: `configs/experiments/e02_resnet_deep_label_smoothing.yaml`
  (`ResNet1D + label_smoothing`).
- Matrix configs: E07 WA-only, E08 concat, E09 gated, E11 gated+KAN, and
  E14 SA-only.
- Primary table after running: `dual_range_matrix/results.md`
- Logs after running: `dual_range_matrix/logs/`
- Plan/config notes: `dual_range_matrix/README.md`
- Formal run gates: `analysis/scripts/preflight_dual_range.py`,
  `analysis/scripts/smoke_dual_range_forward.py`, and
  `analysis/scripts/audit_dual_range.py`.

## Sequence Baselines

- `sequence_baselines/` contains BiGRU and PatchTST-style baselines trained with
  the current strongest mixed E02 loss recipe.
- Primary table: `sequence_baselines/results.md`
- Logs: `sequence_baselines/logs/`
- Plan/config notes: `sequence_baselines/plan.md`

## Dual-Range Loss Control

- `dual_range_loss_control/` contains the same-architecture E09/E15 loss
  control for the gated dual-range ResNet.
- `e09_dual_gated_resnet` uses the mixed long-tail recipe,
  `weighted_ce + label_smoothing=0.03`.
- `e15_dual_gated_resnet_label_smoothing` keeps the E09 architecture fixed and
  switches only to `label_smoothing=0.05`.
- Result: label smoothing improves overall Top-1, while the mixed recipe remains
  better for test macro and rare-class accuracy.

## Main Result Configs

- `configs/main/` contains unified main-result configs M01-M07, including the
  label-smoothing dual-range Mamba and Mamba-KAN runs.
- Run them with `python3 scripts/run_experiments.py --preset main`.

## Historical Records

- `archive/20260605_initial_matrix/` contains the earlier exploratory E01-E04
  matrix. Treat it as historical context, not the current main comparison.

## Checkpoints

Model checkpoints remain under `checkpoints/` so result-table checkpoint paths stay
directly usable. Evaluation plots are inside each checkpoint directory.

## Current E02 Configs

- `configs/experiments/e02_resnet_deep_weighted_ce.yaml`: original mixed E02
  recipe, `weighted_ce + label_smoothing=0.03`.
- `configs/experiments/e02_resnet_deep_weighted_ce_pure.yaml`: pure weighted CE
  used in the clean loss ablation.
- `configs/experiments/e02_resnet_deep_label_smoothing.yaml`: label smoothing
  ablation config.
- `configs/experiments/e02_resnet_deep_focal.yaml`: focal loss ablation config.
- `configs/experiments/e05_bigru_patch_weighted_ce_mixed.yaml`: BiGRU sequence
  baseline with the mixed E02 loss.
- `configs/experiments/e06_patchtst_weighted_ce_mixed.yaml`: PatchTST-style
  sequence baseline with the mixed E02 loss.
