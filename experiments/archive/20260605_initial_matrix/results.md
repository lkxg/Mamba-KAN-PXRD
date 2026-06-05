| experiment | model | optimizer | loss | sampler | best_epoch | val_acc1 | val_macro | test_acc1 | test_acc5 | test_macro | test_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e01_resnet_deep_cbce | resnet1d | adamw | class_balanced_ce | none | 30 | 0.707265 | 0.537855 | 0.6798 | 0.8759 | 0.5021 | 0.5195 |
| e02_resnet_deep_weighted_ce | resnet1d | adamw | weighted_ce | none | 28 | 0.755830 | 0.576961 | 0.7365 | 0.9399 | 0.5316 | 0.5239 |
| e03_resnet_small_focal_sampler | resnet1d | adam | focal | class_balanced | 28 | 0.557681 | 0.373979 | 0.5545 | 0.8816 | 0.3623 | 0.3937 |
| e04_resnet_wide_cbce_light | resnet1d | adamw | class_balanced_ce | none | 25 | 0.773208 | 0.564495 | 0.7565 | 0.8842 | 0.5295 | 0.5550 |
