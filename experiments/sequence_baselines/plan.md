# PXRD Sequence Baselines

This experiment adds two sequence-model baselines under the current strongest
loss recipe:

`weighted_ce + label_smoothing=0.03 + class_weight_power=0.30`

Both runs use the same split (`splits/splits.csv`), task (`space_group`), AdamW
(`lr=1e-3`, `weight_decay=5e-4`), cosine schedule with 3 warmup epochs, BF16 AMP,
and no sampler. Checkpoints are selected by:

`val_balanced_acc1_macro = 0.5 * val_acc1 + 0.5 * val_macro_acc1`

| ID | Config | Model | Loss | Purpose |
|---|---|---|---|---|
| E05 | `configs/experiments/e05_bigru_patch_weighted_ce_mixed.yaml` | Patchified BiGRU | weighted CE + smoothing `.03` | RNN sequence baseline. |
| E06 | `configs/experiments/e06_patchtst_weighted_ce_mixed.yaml` | PatchTST-style Transformer | weighted CE + smoothing `.03` | Transformer/patch sequence baseline. |

Results are written to `experiments/sequence_baselines/results.md`.
