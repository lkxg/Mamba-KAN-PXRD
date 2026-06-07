# Dual-Range Summary

## Baseline

- Baseline: `e02_resnet_deep_label_smoothing`
- Test: Top-1=0.7659, Top-5=0.9352, Top-10=, Macro Acc=0.4674, Macro F1=0.4794, Rare Acc=0.4072

## Matrix

| experiment | top1 | top5 | top10 | macro | f1 | rare | gate | d_top1 vs baseline | d_macro vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e02_resnet_deep_label_smoothing | 0.7659 | 0.9352 |  | 0.4674 | 0.4794 | 0.4072 |  | +0.0000 | +0.0000 |
| e07_wa_only_resnet_label_smoothing | 0.6580 | 0.9021 |  | 0.4516 | 0.4699 | 0.4252 |  | -0.1079 | -0.0158 |
| e14_sa_only_resnet_ablation | 0.4176 | 0.8029 |  | 0.2335 | 0.2282 | 0.2613 |  | -0.3483 | -0.2339 |
| e08_dual_concat_resnet | 0.6699 | 0.9308 |  | 0.5039 | 0.4938 | 0.4703 |  | -0.0960 | +0.0365 |
| e09_dual_gated_resnet | 0.7018 | 0.9361 |  | 0.5056 | 0.5046 | 0.4757 | 0.4656 | -0.0641 | +0.0382 |
| e11_dual_gated_kan | 0.7051 | 0.9206 |  | 0.5216 | 0.5169 | 0.5063 | 0.4830 | -0.0608 | +0.0542 |

## Ablations

| contrast | candidate | reference | d_top1 | d_top10 | d_macro | d_f1 | d_rare |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SA only vs WA only | e14_sa_only_resnet_ablation | e07_wa_only_resnet_label_smoothing | -0.2404 |  | -0.2181 | -0.2417 | -0.1640 |
| SA+WA concat vs WA only | e08_dual_concat_resnet | e07_wa_only_resnet_label_smoothing | +0.0119 |  | +0.0523 | +0.0240 | +0.0450 |
| Gated vs concat | e09_dual_gated_resnet | e08_dual_concat_resnet | +0.0319 |  | +0.0017 | +0.0108 | +0.0054 |
| KAN vs MLP head | e11_dual_gated_kan | e09_dual_gated_resnet | +0.0033 |  | +0.0159 | +0.0123 | +0.0306 |

## Interpretation Artifacts

| experiment | metrics | gate by system | confusion cases | confusion pairs |
| --- | --- | --- | --- | --- |
| e02_resnet_deep_label_smoothing | checkpoints/e02_resnet_deep_label_smoothing_20260605_085108/eval_plots/metrics.json |  | checkpoints/e02_resnet_deep_label_smoothing_20260605_085108/eval_plots/confusion_cases.csv | checkpoints/e02_resnet_deep_label_smoothing_20260605_085108/eval_plots/confusion_pairs.csv |
| e07_wa_only_resnet_label_smoothing | checkpoints/e07_wa_only_resnet_label_smoothing_20260606_063750/eval_plots/metrics.json |  | checkpoints/e07_wa_only_resnet_label_smoothing_20260606_063750/eval_plots/confusion_cases.csv | checkpoints/e07_wa_only_resnet_label_smoothing_20260606_063750/eval_plots/confusion_pairs.csv |
| e14_sa_only_resnet_ablation | checkpoints/e14_sa_only_resnet_ablation_20260606_072159/eval_plots/metrics.json |  | checkpoints/e14_sa_only_resnet_ablation_20260606_072159/eval_plots/confusion_cases.csv | checkpoints/e14_sa_only_resnet_ablation_20260606_072159/eval_plots/confusion_pairs.csv |
| e08_dual_concat_resnet | checkpoints/e08_dual_concat_resnet_20260606_073246/eval_plots/metrics.json |  | checkpoints/e08_dual_concat_resnet_20260606_073246/eval_plots/confusion_cases.csv | checkpoints/e08_dual_concat_resnet_20260606_073246/eval_plots/confusion_pairs.csv |
| e09_dual_gated_resnet | checkpoints/e09_dual_gated_resnet_20260606_082222/eval_plots/metrics.json | checkpoints/e09_dual_gated_resnet_20260606_082222/eval_plots/gate_by_crystal_system.csv | checkpoints/e09_dual_gated_resnet_20260606_082222/eval_plots/confusion_cases.csv | checkpoints/e09_dual_gated_resnet_20260606_082222/eval_plots/confusion_pairs.csv |
| e11_dual_gated_kan | checkpoints/e11_dual_gated_kan_20260606_091237/eval_plots/metrics.json | checkpoints/e11_dual_gated_kan_20260606_091237/eval_plots/gate_by_crystal_system.csv | checkpoints/e11_dual_gated_kan_20260606_091237/eval_plots/confusion_cases.csv | checkpoints/e11_dual_gated_kan_20260606_091237/eval_plots/confusion_pairs.csv |

## Gate Means By Crystal System

### e09_dual_gated_resnet

| crystal_system | count | mean | std | median |
| --- | --- | --- | --- | --- |
| Triclinic | 11300 | 0.4751 | 0.0263 | 0.4785 |
| Monoclinic | 22802 | 0.4717 | 0.0286 | 0.4746 |
| Orthorhombic | 7999 | 0.4636 | 0.0318 | 0.4668 |
| Tetragonal | 1629 | 0.4297 | 0.0503 | 0.4375 |
| Trigonal | 1319 | 0.4248 | 0.0427 | 0.4277 |
| Hexagonal | 697 | 0.4085 | 0.0490 | 0.3984 |
| Cubic | 1038 | 0.3897 | 0.0680 | 0.3809 |

### e11_dual_gated_kan

| crystal_system | count | mean | std | median |
| --- | --- | --- | --- | --- |
| Triclinic | 11300 | 0.4853 | 0.0371 | 0.4863 |
| Monoclinic | 22802 | 0.4871 | 0.0379 | 0.4902 |
| Orthorhombic | 7999 | 0.4870 | 0.0388 | 0.4883 |
| Tetragonal | 1629 | 0.4657 | 0.0555 | 0.4746 |
| Trigonal | 1319 | 0.4630 | 0.0474 | 0.4727 |
| Hexagonal | 697 | 0.4471 | 0.0550 | 0.4453 |
| Cubic | 1038 | 0.4148 | 0.0858 | 0.3896 |
