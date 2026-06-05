| experiment | model | optimizer | loss | sampler | best_epoch | val_acc1 | val_macro | test_acc1 | test_acc5 | test_macro | test_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e02_resnet_deep_weighted_ce | resnet1d | adamw | weighted_ce | none | 27 | 0.714148 | 0.540387 | 0.6890 | 0.9331 | 0.5106 | 0.5008 |
| e02_resnet_deep_label_smoothing | resnet1d | adamw | label_smoothing | none | 30 | 0.769061 | 0.495244 | 0.7659 | 0.9352 | 0.4674 | 0.4794 |
| e02_resnet_deep_focal | resnet1d | adamw | focal | none | 30 | 0.747301 | 0.506008 | 0.7423 | 0.9484 | 0.4843 | 0.4951 |
