# Dual-Range Summary

## Baseline

- Baseline: `e02_resnet_deep_label_smoothing`
- Test: Top-1=0.7659, Top-5=0.9352, Macro Acc=0.4674, Macro F1=0.4794, Rare Acc=0.4216

## Matrix

| experiment | top1 | top5 | macro | f1 | rare | gate | SA mamba | WA mamba | occ d_top1 | occ d_macro | occ d_f1 | occ d_rare | d_top1 vs baseline | d_macro vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e02_resnet_deep_label_smoothing | 0.7659 | 0.9352 | 0.4674 | 0.4794 | 0.4216 |  |  |  |  |  |  |  | +0.0000 | +0.0000 |
| e07_wa_only_resnet_label_smoothing | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e14_sa_only_resnet_ablation | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e08_dual_concat_resnet | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e09_dual_gated_resnet | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e10_dual_gated_mamba | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e11_dual_gated_kan | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e12_dual_gated_mamba_kan | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e13_dual_gated_mamba_kan_aux | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Ablations

| contrast | candidate | reference | d_top1 | d_macro | d_f1 | d_rare |
| --- | --- | --- | --- | --- | --- | --- |
| SA only vs WA only | e14_sa_only_resnet_ablation | e07_wa_only_resnet_label_smoothing |  |  |  |  |
| SA+WA concat vs WA only | e08_dual_concat_resnet | e07_wa_only_resnet_label_smoothing |  |  |  |  |
| Gated vs concat | e09_dual_gated_resnet | e08_dual_concat_resnet |  |  |  |  |
| Mamba vs ResNet | e10_dual_gated_mamba | e09_dual_gated_resnet |  |  |  |  |
| KAN vs MLP head | e11_dual_gated_kan | e09_dual_gated_resnet |  |  |  |  |
| Mamba+KAN vs gated ResNet | e12_dual_gated_mamba_kan | e09_dual_gated_resnet |  |  |  |  |
| Aux heads vs no aux | e13_dual_gated_mamba_kan_aux | e12_dual_gated_mamba_kan |  |  |  |  |

## Interpretation Artifacts

| experiment | metrics | gate by system | confusion cases | confusion pairs |
| --- | --- | --- | --- | --- |
| e02_resnet_deep_label_smoothing |  |  |  |  |
| e07_wa_only_resnet_label_smoothing | pending |  |  |  |
| e14_sa_only_resnet_ablation | pending |  |  |  |
| e08_dual_concat_resnet | pending |  |  |  |
| e09_dual_gated_resnet | pending |  |  |  |
| e10_dual_gated_mamba | pending |  |  |  |
| e11_dual_gated_kan | pending |  |  |  |
| e12_dual_gated_mamba_kan | pending |  |  |  |
| e13_dual_gated_mamba_kan_aux | pending |  |  |  |

## Gate Means By Crystal System

No gated experiment with gate-by-system output is available yet.
