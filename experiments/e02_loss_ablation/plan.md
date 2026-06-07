# PXRD Space-Group E02 Loss Ablation

All runs use the same E02 base configuration: split (`splits/splits.csv`), task
(`space_group`), ResNet1D base64 with blocks `[3,4,6,3]`, AdamW (`lr=1e-3`,
`weight_decay=5e-4`), cosine schedule with 3 warmup epochs, batch size `1024`,
BF16 AMP, and W&B offline logging. Checkpoints are
selected by:

`val_balanced_acc1_macro = 0.5 * val_acc1 + 0.5 * val_macro_acc1`

W&B is set to offline because this machine currently has no W&B API key configured.
After logging in, sync runs with:

```bash
wandb sync wandb/wandb/offline-run-*
```

| ID | Config | Model Parameters | Optimizer / Schedule | Long-Tail Handling | Purpose |
|---|---|---|---|---|---|
| E02-WCE | `configs/experiments/e02_resnet_deep_weighted_ce_pure.yaml` | ResNet1D base64, blocks `[3,4,6,3]` | AdamW, lr `1e-3`, wd `5e-4`, cosine, warmup 3 | weighted CE, power `.30`, no sampler | Verify weighted CE under the E02 base setup. |
| E02-LS | `configs/experiments/e02_resnet_deep_label_smoothing.yaml` | ResNet1D base64, blocks `[3,4,6,3]` | AdamW, lr `1e-3`, wd `5e-4`, cosine, warmup 3 | label smoothing CE, smoothing `.05`, no sampler | Verify label smoothing under the E02 base setup. |
| E02-Focal | `configs/experiments/e02_resnet_deep_focal.yaml` | ResNet1D base64, blocks `[3,4,6,3]` | AdamW, lr `1e-3`, wd `5e-4`, cosine, warmup 3 | focal loss, gamma `2.0`, no sampler | Verify focal loss under the E02 base setup. |

Results are written to `experiments/e02_loss_ablation/results.md`.

The earlier mixed E02 recipe (`weighted_ce` with `label_smoothing=0.03`) is kept at
`configs/experiments/e02_resnet_deep_weighted_ce.yaml`.
