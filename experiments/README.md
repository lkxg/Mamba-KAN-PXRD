# Experiment Records

Use this directory as the experiment index.

## Current Main Result

- `e02_loss_ablation/` contains the clean E02 loss ablation:
  weighted CE, label smoothing, and focal loss under the same E02 base setup.
- Primary table: `e02_loss_ablation/results.md`
- Full CSV: `e02_loss_ablation/results.csv`
- Logs: `e02_loss_ablation/logs/`
- Plan/config notes: `e02_loss_ablation/plan.md`

## Sequence Baselines

- `sequence_baselines/` contains BiGRU and PatchTST-style baselines trained with
  the current strongest mixed E02 loss recipe.
- Primary table: `sequence_baselines/results.md`
- Full CSV: `sequence_baselines/results.csv`
- Logs: `sequence_baselines/logs/`
- Plan/config notes: `sequence_baselines/plan.md`

## Historical Records

- `archive/20260605_initial_matrix/` contains the earlier exploratory E01-E04
  matrix. Treat it as historical context, not the current main comparison.

## Checkpoints

Model checkpoints remain under `checkpoints/` so result CSV checkpoint paths stay
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
