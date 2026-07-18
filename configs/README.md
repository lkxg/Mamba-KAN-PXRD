# Configuration Catalog

Experiment configs use a short, globally unique ID. The YAML filename stem,
`experiment.name`, and the primary W&B tag are identical.

| Prefix | Directory | Scope |
| --- | --- | --- |
| `b` | `baselines/` | CNN, RNN, and Transformer baselines |
| `m` | `main/` | Primary Mamba model variants |
| `l` | `losses/` | Loss and supervised-contrastive ablations |
| `a` | `ablations/` | Architecture and frontend ablations |
| `mx` | `mobile/` | MobileXRD-style variants |

`default.yaml` remains the editable starter config and uses the run name
`default_resnet`. Historical checkpoints, logs, and result rows keep their old
IDs; this table maps each retained config to that historical ID.
Evaluate an old checkpoint by passing its path directly to `scripts/evaluate.py`;
`run_experiments.py --skip-train` searches for the current short run name.

| Config | Previous ID | Purpose |
| --- | --- | --- |
| `baselines/b01_resnet.yaml` | `m01` | ResNet1D baseline |
| `baselines/b02_resnet_deep.yaml` | `e02` | Deep ResNet1D with weighted CE |
| `baselines/b03_resnet18.yaml` | `m45` | ResNet1D-18 baseline |
| `baselines/b04_convnext.yaml` | `m44` | ConvNeXt1D baseline |
| `baselines/b05_bigru.yaml` | `e05` | Patchified BiGRU baseline |
| `baselines/b06_patchtst.yaml` | `e06` | PatchTST-style baseline |
| `main/m01_mamba.yaml` | `m18` | Learned-downsample Mamba baseline |
| `main/m02_bimamba_kan.yaml` | `m20` | Bidirectional Mamba with KAN head |
| `main/m03_mamba2.yaml` | `m27` | Mamba2 baseline |
| `main/m04_mamba2_b64.yaml` | `e20` | Mamba2 with batch size 64 |
| `losses/l01_ldam.yaml` | `m22` | LDAM with deferred reweighting |
| `losses/l02_supcon.yaml` | `m23` | Supervised contrastive projection |
| `losses/l03_supcon_lr3e4.yaml` | `m28` | Lower-LR supervised contrastive run |
| `losses/l04_focal.yaml` | `m29` | Focal loss |
| `losses/l05_weighted_ce.yaml` | `m36` | Weighted cross entropy |
| `losses/l06_asl.yaml` | `m37` | Asymmetric loss |
| `ablations/a01_angle_pos.yaml` | `m40` | Absolute angle position encoding |
| `ablations/a02_gated_pool.yaml` | `m41` | Gated-attention pooling |
| `ablations/a03_convnext_frontend.yaml` | `m43` | ConvNeXt downsampling frontend |
| `ablations/a04_wide_frontend.yaml` | `m46` | Wider learned frontend |
| `ablations/a05_residual_gated_pool.yaml` | `m47` | Residual gated-attention pooling |
| `ablations/a06_stride4.yaml` | `m48` | Total frontend stride 4 |
| `ablations/a07_layers10.yaml` | `m49` | Ten Mamba layers |
| `ablations/a08_dim192.yaml` | `m50` | Model width 192 |
| `ablations/a09_dstate32.yaml` | `m51` | Mamba state size 32 |
| `ablations/a10_multiscale_frontend.yaml` | `m52` | Multiscale frontend |
| `ablations/a11_inception_frontend.yaml` | `m53` | Inception-style frontend |
| `ablations/a12_peak_frontend.yaml` | `m54` | Peak-aware frontend |
| `ablations/a13_wavelet_frontend.yaml` | `m55` | Wavelet frontend |
| `ablations/a14_antialias_frontend.yaml` | `m56` | Anti-aliased frontend |
| `ablations/a15_mamba2_stride4.yaml` | `m61` | Mamba2 with stride 4 |
| `mobile/mx01_lite.yaml` | `m57` | Lightweight MobileXRD mixer |
| `mobile/mx02_wavelet.yaml` | `m58` | MobileXRD with wavelet residual |
| `mobile/mx03_no_identity.yaml` | `m59` | MobileXRD without identity channels |
| `mobile/mx04_wte.yaml` | `m60` | MobileXRD with WTE branch |
| `mobile/mx05_wide.yaml` | `m62` | Wider MobileXRD frontend |
| `mobile/mx06_dim192.yaml` | `m63` | MobileXRD width 192 |
| `mobile/mx07_dstate64.yaml` | `m64` | MobileXRD state size 64 |
| `mobile/mx08_peak_blocks.yaml` | `m65` | Peak kernels and deeper frontend |
| `mobile/mx09_antialias.yaml` | `m67` | Anti-aliased MobileXRD frontend |
| `mobile/mx10_multikernel.yaml` | `m68` | Multi-kernel MobileXRD stem |
| `mobile/mx11_physaug.yaml` | — | Symmetry-preserving PXRD augmentation on mx03 |
| `mobile/mx12_la_physaug.yaml` | — | PeakSet gated fusion + logit-adjusted CE (augmentation disabled in the initial control) |
| `mobile/mx13_xrd_ctm.yaml` | — | Full clean XRD-CTM with CNN, unidirectional Mamba, and peak/gap Transformer |

Run selected configs explicitly:

```bash
python3 scripts/run_experiments.py --configs \
  configs/main/m01_mamba.yaml \
  configs/mobile/mx01_lite.yaml
```
