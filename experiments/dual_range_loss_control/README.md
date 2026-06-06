# Dual-Range Loss Control

This directory records a same-architecture loss control for the dual-range
gated ResNet.

## Question

Does the current mixed long-tail recipe,
`weighted_ce + label_smoothing=0.03`, beat pure label smoothing when the
architecture is held fixed?

## Control

- Mixed-loss reference: `e09_dual_gated_resnet`
- Label-smoothing control: `e15_dual_gated_resnet_label_smoothing`
- Architecture: identical dual-range gated ResNet, MLP head, no Mamba, no KAN
- Seed, optimizer, schedule, batch size, epoch count, monitor, and data split:
  held fixed
- Only changed loss:
  - `e09`: `weighted_ce` with `label_smoothing=0.03`
  - `e15`: `label_smoothing` with `label_smoothing=0.05`

## Result

| experiment | loss | val_balanced | val_acc1 | val_macro | val_rare | test_acc1 | test_macro | test_f1 | test_rare |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e09_dual_gated_resnet | weighted_ce + smoothing .03 | 0.602994 | 0.700212 | 0.538182 | 0.522968 | 0.701821 | 0.505635 | 0.504605 | 0.475676 |
| e15_dual_gated_resnet_label_smoothing | label_smoothing .05 | 0.612916 | 0.745527 | 0.524508 | 0.494700 | 0.745062 | 0.491870 | 0.504762 | 0.461261 |

Delta `e15 - e09`:

| metric | delta |
| --- | ---: |
| val_balanced | +0.009922 |
| val_acc1 | +0.045315 |
| val_macro | -0.013674 |
| val_rare | -0.028268 |
| test_acc1 | +0.043241 |
| test_macro | -0.013765 |
| test_f1 | +0.000157 |
| test_rare | -0.014415 |

## Conclusion

Pure label smoothing is better for overall Top-1 accuracy on this architecture.
The mixed recipe remains better for macro accuracy and rare-class accuracy on
the held-out test split. Therefore, use:

- `label_smoothing=0.05` when prioritizing overall Top-1 accuracy.
- `weighted_ce + label_smoothing=0.03` when prioritizing long-tail space-group
  performance.

This result weakens the blanket claim that mixed loss is simply "better"; it is
better for long-tail metrics, not for overall accuracy.
