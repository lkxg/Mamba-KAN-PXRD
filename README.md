# Crystallography-Informed Sequence Models for PXRD Space Group Classification

**[English](README.md) | [中文](README_zh.md)**

A systematic study of long-sequence architectures for space group classification
from simulated powder X-ray diffraction (PXRD) patterns, built on the SIMPOD
dataset (467,861 patterns, 10,824 points each, 230 classes).

---

## Abstract

Powder X-ray diffraction is a fundamental technique for determining crystal
structures, but conventional indexing relies on expert interpretation. Two
properties of the task make it hard for standard deep learning models: PXRD
patterns are long sequences (10,824 points), which makes dense CNN and
Transformer stacks expensive, and the space group distribution is severely
long-tailed — the three most common space groups cover 63.4% of the dataset
while 103 of 230 space groups have at most 100 training samples.

This repository studies architectures that address both properties. The
strongest models combine a learned downsampling frontend, a Mamba2 selective
state space encoder with linear complexity in sequence length, and a sparse
Bragg-peak branch that reasons over detected peak positions in Q-space rather
than over the dense signal. Across 34 retained experiments the best
configuration reaches **83.4% top-1** and **95.1% top-5** accuracy on the held-out
test set; the best macro-F1 (**0.573**) is achieved by a different configuration,
reflecting a consistent trade-off between overall accuracy and rare-class
performance.

---

## Status and Scope

This is an active research repository, not a released library. Concretely:

- All results are on **simulated** SIMPOD data. No real experimental PXRD
  patterns have been evaluated yet.
- All runs use a **single seed (42)**. Differences below ~1 point should not be
  read as meaningful without repeated seeds.
- Test-set metrics are reported for all retained configs because they were
  produced during exploration. A frozen-finalist protocol is described under
  [Experimental Protocol](#experimental-protocol) and should be applied before
  any of these numbers are used in a publication.
- Checkpoints and raw logs are not tracked in Git (see `.gitignore`).

---

## Findings

**Mamba-based encoders outperform CNN, RNN, and Transformer baselines.** The
ResNet1D baseline reaches 77.4% top-1. The MobileXRD-style Mamba2 variants reach
81–83%. The patchified BiGRU (47.4%) and PatchTST-style (43.5%) baselines are far
behind; these are not tuned baselines and should be treated as lower bounds
rather than fair comparisons.

**KAN did not help on this task.** The project began as a Mamba-KAN hybrid, and
KAN was tested in the two positions the literature suggests: as a classifier head
(`m02_bimamba_kan`, 72.4% top-1) and as a SupCon feature projection head
(`l02_supcon`, 69.7% top-1, whose classifier remains an MLP). Both underperform
the plain linear head of the
otherwise identical `m01_mamba` (82.0%). The likely cause is capacity: the RBF
spline branch in [`KANHead`](src/models.py) multiplies head parameters by the
number of grid points (8 in these runs), which is difficult to fit for the 103
space groups with ≤100 training samples. This is reported as a negative result
rather than removed.

**Sparse peak reasoning is the most promising direction.** The two architectures
that encode Bragg peaks as explicit tokens — `mx14_peakset_ls` (PeakSet gated
fusion) and `mx15_xrd_ctm_ls` (CNN + Mamba + peak/gap Transformer) — achieve the
best macro-F1 and competitive rare-class accuracy. This is the part of the design
that is specific to crystallography rather than transferred from vision.

**Long-tail methods trade top-1 for rare-class accuracy, sharply.** Logit-adjusted
cross entropy with τ=1.0 (`mx12_la_physaug`) produced the best rare-class accuracy
in the entire study (55.7%) but collapsed overall top-1 to 14.1%. τ was not swept;
a post-hoc adjustment sweep over a trained checkpoint is the natural next step and
requires no retraining.

---

## Dataset

**Source**: SIMPOD (Rincón et al., *Scientific Data* 12, 1186, 2025)

| Property | Value |
|---|---|
| 2θ range | 5° – 90° |
| Sampling points | 10,824 |
| Radiation | Cu Kα (λ = 1.5406 Å) |
| Normalization | each pattern divided by its maximum → intensities ∈ [0, 1] |
| Total patterns | 467,861 |
| Classes | 230 space groups / 7 crystal systems |

### Crystal system distribution

Space groups per system are counted from `dataset/labels.csv`, not from the
nominal space group number ranges. "Largest SG share" is the fraction of that
system's samples belonging to its single most common space group.

| Crystal system | SG range | Space groups | Samples | Share | Largest SG share |
|---|---|---|---|---|---|
| Monoclinic | 3–15 | 13 | 228,011 | 48.7% | 64.9% |
| Triclinic | 1–2 | 2 | 113,002 | 24.2% | 96.9% |
| Orthorhombic | 16–74 | 59 | 79,990 | 17.1% | 32.4% |
| Tetragonal | 75–142 | 68 | 16,335 | 3.5% | 11.7% |
| Trigonal | 143–167 | 25 | 13,179 | 2.8% | 28.0% |
| Cubic | 195–230 | 36 | 10,357 | 2.2% | 25.9% |
| Hexagonal | 168–194 | 27 | 6,987 | 1.5% | 18.6% |
| **Total** | **1–230** | **230** | **467,861** | **100%** | — |

### Why crystal system is not the bottleneck

An oracle that predicts the crystal system perfectly and then guesses the most
common space group within it reaches only **62.6% top-1** — well below the 83.4%
of the best model. Coarse-to-fine hierarchical schemes therefore have less
headroom than the class counts suggest: the models already resolve a great deal
of within-system structure. The difficulty is concentrated in the systems that
are simultaneously rare and internally diverse. Tetragonal is the extreme case:
68 space groups sharing 3.5% of the data, 40 of which are rare classes.

Rare space groups (1–100 training samples, 103 of 230 total) by system:

| System | Rare SGs | Total SGs |
|---|---|---|
| Tetragonal | 40 | 68 |
| Orthorhombic | 21 | 59 |
| Cubic | 20 | 36 |
| Hexagonal | 13 | 27 |
| Trigonal | 7 | 25 |
| Monoclinic | 2 | 13 |
| Triclinic | 0 | 2 |

### Splits

Generated by `scripts/make_splits.py` with a fixed seed (42): 80% train
(374,294), 10% validation (46,783), 10% test (46,784), stratified by space
group. Seven space groups have fewer than `--min-per-class` (default 10) samples
and are placed entirely in the training set, so 223 of 230 space groups appear in
validation and test.

---

## Method

### Model families

All models are registered in [`src/models.py`](src/models.py) and selected by
`model.name` in the config.

| `model.name` | Description |
|---|---|
| `resnet1d` | 1D residual CNN baseline |
| `convnext1d` | 1D ConvNeXt baseline |
| `bigru_patch` | Patchified bidirectional GRU baseline |
| `patchtst` | PatchTST-style patch Transformer baseline |
| `dual_plane_mamba` | Learned-downsample frontend + Mamba/Mamba2 encoder; optional MobileXRD token mixer, PeakSet branch, and KAN head |
| `xrd_ctm` | Three-branch CNN + Mamba2 + peak/gap Transformer with gated fusion |

### Crystallography-specific components

These are the parts of the design motivated by the physics of diffraction rather
than adapted from vision or NLP architectures.

**Q-space peak tokens.** [`_PeakSetBranch`](src/models.py) and
[`_XRDPeakTransformer`](src/models.py) detect the top-K local maxima of the raw
pattern and build per-peak tokens from position, intensity, neighbour spacing,
and curvature. Positions are encoded as normalized Q = 4π·sin(θ)/λ rather than as
2θ index, so that an isotropic change of lattice constant becomes an affine shift
in token coordinates. Peak indices are selected without gradient; intensities and
curvatures are gathered differentiably.

**Symmetry-preserving augmentation.** [`PXRDAugment`](src/data.py) applies three
transforms that provably leave the space group label unchanged: isotropic lattice
scaling (resampling on a sinθ grid, which preserves inter-peak ratios and
systematic absences exactly), intensity perturbation via random gamma plus a
smooth low-frequency envelope (models preferred orientation / texture), and
Gaussian peak broadening (models reduced crystallite size). Enabled in
`mx11_physaug`.

**Cross-branch feedback.** In `xrd_ctm`, the peak branch emits a Gaussian-kernel
importance map over the signal axis, which modulates the CNN and Mamba feature
maps before pooling. Fusion weights over the three branches are produced by a
gate conditioned on a learned pattern-quality vector, and branch dropout
randomly replaces one branch with a learned null embedding during training.

### Losses

Implemented in [`src/training.py`](src/training.py), selected by `loss.name`:
`label_smoothing`, `weighted_ce`, `class_balanced_ce`, `focal`, `asl`, `ldam`
(with deferred reweighting), `logit_adjusted_ce`, and an optional supervised
contrastive term. Models exposing auxiliary heads additionally support
`auxiliary_weight`.

### Training setup

Actual hyperparameters of the main configs (e.g.
`configs/mobile/mx03_no_identity.yaml`):

| Setting | Value |
|---|---|
| Python / PyTorch | ≥ 3.10 / 2.8.0+cu128 |
| GPU | H100 (recorded in the W&B tags) |
| Optimizer | AdamW, lr = 3e-4, weight decay = 1e-3 |
| Schedule | cosine annealing with 20 warmup epochs |
| Batch size | 64 |
| Epochs | 120 (early stopping: patience 15, active from epoch 60) |
| Gradient clipping | 1.0 |
| Precision | TF32 with `matmul_precision: high` |
| Model selection | `val_balanced_acc1_macro` (acc1 and macro-acc1 weighted 0.5 each) |

`configs/default.yaml` is an editable starter template (ResNet1D, batch 1024,
lr 2e-3, 30 epochs) and does not reflect the main experiments.

---

## Results

Test-set metrics for all retained configurations. `rare_acc1` is top-1 accuracy
restricted to space groups with 1–100 training samples; blank cells mean the
metric predates that instrumentation. Canonical source:
[`experiments/results.md`](experiments/results.md).

### Baselines

| Config | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `b01_resnet` | 0.7736 | 0.9249 | 0.5568 | — |
| `b05_bigru` | 0.4743 | 0.8527 | 0.3371 | — |
| `b06_patchtst` | 0.4345 | 0.8337 | 0.3353 | — |

### Mamba variants

| Config | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `m01_mamba` | 0.8197 | 0.9387 | 0.5621 | — |
| `m03_mamba2` | 0.7714 | 0.9365 | 0.4939 | 0.4288 |
| `m04_mamba2_b64` | 0.8096 | 0.9339 | 0.5564 | 0.5027 |

`m02_bimamba_kan` (KAN head) is reported under [Findings](#findings); it is
retained in `configs/main/` as a negative result.

### Loss ablations

| Config | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `l04_focal` | 0.7971 | **0.9644** | 0.5289 | — |
| `l05_weighted_ce` | 0.7538 | 0.9479 | 0.5232 | 0.5009 |
| `l01_ldam` | 0.7485 | 0.9306 | 0.4840 | — |
| `l06_asl` | 0.7250 | 0.9481 | 0.4962 | 0.4793 |
| `l03_supcon_lr3e4` | 0.7257 | 0.8855 | 0.4300 | — |
| `l02_supcon` | 0.6966 | 0.9407 | 0.5165 | — |

### Architecture ablations

| Config | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `a04_wide_frontend` | 0.8259 | 0.9414 | 0.5637 | 0.5153 |
| `a11_inception_frontend` | 0.8183 | 0.9369 | 0.5525 | 0.4847 |
| `a10_multiscale_frontend` | 0.8023 | 0.9360 | 0.5439 | 0.4883 |
| `a02_gated_pool` | 0.7991 | 0.9312 | 0.5457 | 0.4901 |
| `a05_residual_gated_pool` | 0.7970 | 0.9465 | 0.5226 | 0.4757 |
| `a03_convnext_frontend` | 0.7967 | 0.9452 | 0.5454 | 0.4973 |
| `a01_angle_pos` | 0.7759 | 0.9360 | 0.5260 | 0.4811 |

### MobileXRD and XRD-CTM variants

| Config | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `mx03_no_identity` | **0.8336** | 0.9513 | 0.5694 | 0.4991 |
| `mx10_multikernel` | 0.8318 | 0.9462 | 0.5568 | 0.4883 |
| `mx08_peak_blocks` | 0.8292 | 0.9539 | 0.5491 | 0.4811 |
| `mx14_peakset_ls` | 0.8289 | 0.9589 | 0.5563 | 0.4937 |
| `mx05_wide` | 0.8278 | 0.9536 | 0.5474 | 0.4793 |
| `mx07_dstate64` | 0.8275 | 0.9554 | 0.5439 | 0.4775 |
| `mx09_antialias` | 0.8265 | 0.9545 | 0.5484 | 0.4793 |
| `mx06_dim192` | 0.8256 | 0.9406 | 0.5528 | 0.4811 |
| `mx01_lite` | 0.8138 | 0.9611 | 0.5327 | 0.4631 |
| `mx04_wte` | 0.8129 | 0.9370 | 0.5591 | 0.5207 |
| `mx11_physaug` | 0.8127 | 0.9573 | 0.5495 | 0.4775 |
| `mx02_wavelet` | 0.8103 | 0.9493 | 0.5330 | 0.4396 |
| `mx15_xrd_ctm_ls` | 0.8041 | 0.9380 | **0.5733** | 0.5135 |
| `mx12_la_physaug` | 0.1406 | 0.2901 | 0.4860 | **0.5568** |

### Reading these numbers

The three "best" columns point at three different configs: `mx03_no_identity` for
top-1, `mx15_xrd_ctm_ls` for macro-F1, `mx12_la_physaug` for rare-class accuracy.
There is no single winner, and the gaps among the top MobileXRD variants
(0.8336 / 0.8318 / 0.8292) are within the range that a second seed could plausibly
reorder.

`mx12_la_physaug` is the clearest illustration of the long-tail trade-off:
training with logit-adjusted CE at τ=1.0 moved the decision boundary so far
toward rare classes that overall accuracy collapsed, while rare-class accuracy
became the best observed. Despite the config ID, signal augmentation is disabled
in this run — it is the architecture-and-loss control described in its config
notes.

`mx13_xrd_ctm` has a config but no result row.

---

## Installation

```bash
# CUDA Toolkit 12.8 must be installed and nvcc on PATH, because
# mamba-ssm and causal-conv1d compile from source.
pip install -r requirements.txt
```

Mamba layers fall back to a pure-PyTorch implementation when
`mamba.backend: auto` is set and the CUDA kernels are unavailable, but the
`mamba2_ssm` backend used by the main configs requires the compiled package.

## Reproducing

```bash
# 1. Preprocess SIMPOD JSON into dataset/pxrd.npy + dataset/labels.csv.
#    Edit DATA_DIR / OUT_DIR at the top of the script to match your paths.
python analysis/scripts/preprocess.py

# 2. Generate the stratified split (writes splits/splits.csv).
python scripts/make_splits.py

# 3. Train a single config.
python scripts/train.py --config configs/mobile/mx03_no_identity.yaml

# 4. Evaluate a checkpoint. Writes metrics.json, per-class CSVs, and a
#    normalized confusion matrix to <checkpoint dir>/eval_plots/.
python scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt

# 5. Or run several configs end to end, appending rows to experiments/results.md.
python scripts/run_experiments.py --configs \
  configs/main/m01_mamba.yaml \
  configs/mobile/mx03_no_identity.yaml
```

Useful flags: `evaluate.py --split val` to keep the test set frozen,
`--only-rare` to evaluate only rare-class samples, and `--max-samples N` for a
smoke test.

## Repository layout

```
src/            data.py (dataset, splits, augmentation), models.py,
                training.py (losses, loop), utils.py
scripts/        train.py, evaluate.py, run_experiments.py, make_splits.py
configs/        default.yaml + baselines/ main/ losses/ ablations/ mobile/
                See configs/README.md for the ID scheme.
experiments/    results.md — canonical metrics. See experiments/README.md.
analysis/       preprocessing, dataset statistics, distribution plots
tests/          unittest suite (currently covers XRD-CTM only)
```

Configs use short globally unique IDs; the YAML filename stem,
`experiment.name`, and the primary W&B tag are identical. `configs/README.md`
maps each retained config to its historical ID.

## Experimental Protocol

The intended protocol, which the current results only partly follow:

1. Rank candidates on the **validation** split (`evaluate.py --split val`).
2. Freeze 2–3 finalists, then evaluate on test **once**.
3. Run finalists with at least 3 seeds and report mean ± std.
4. Keep one row per experiment in `experiments/results.md`; reruns replace the
   row rather than appending.

---

## Limitations

1. **Simulated data only.** SIMPOD patterns have no background, no instrumental
   broadening beyond a fixed peak width, no preferred-orientation effects, and a
   single wavelength. Performance on real diffractograms is unknown, and the
   symmetry-preserving augmentation in `PXRDAugment` is designed for exactly this
   gap but has not yet been validated against real data.
2. **Single seed.** No variance estimates; small differences are not resolvable.
3. **Test-set reuse.** See [Experimental Protocol](#experimental-protocol).
4. **Untuned sequence baselines.** BiGRU and PatchTST results should not be cited
   as evidence that recurrent or patch-Transformer models are unsuitable.
5. **No efficiency measurements.** Parameter counts, FLOPs, latency, and peak
   memory are not recorded, despite linear-complexity sequence modelling being a
   central motivation.
6. **Hardcoded preprocessing paths.** The input and output directories in
   `analysis/scripts/preprocess.py` are hardcoded Windows absolute paths and must
   be edited by hand on another machine.
7. **Partial test coverage.** `tests/` covers the XRD-CTM forward contract only.

## Planned work

- [ ] Post-hoc logit adjustment sweep over τ on a trained checkpoint, to trace the
      head/tail trade-off without retraining
- [ ] Error decomposition into cross-system and within-system components, to
      confirm where the remaining 17% of errors are concentrated
- [ ] Efficiency table: parameters, FLOPs, latency, peak memory
- [ ] Multi-seed runs for the finalists
- [ ] Crystal system (7-class) results, derivable from saved space group predictions
- [ ] Evaluation on real experimental PXRD patterns
- [ ] Stronger BiGRU and PatchTST baselines

---

## Citation

```bibtex
@article{simPOD2025,
  title={SIMPOD: A Large-Scale Simulated PXRD Dataset for Crystal System and
         Space Group Classification},
  author={Rinc\'on et al.},
  journal={Scientific Data},
  volume={12},
  pages={1186},
  year={2025},
  doi={10.57760/sciencedb.09755}
}
```

## References

1. Rincón et al., *Scientific Data* 12, 1186 (2025) — [DOI: 10.57760/sciencedb.09755](https://doi.org/10.57760/sciencedb.09755)
2. Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
3. Dao & Gu, *Transformers are SSMs (Mamba-2)* — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
4. Liu et al., *KAN: Kolmogorov-Arnold Networks* — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
5. Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021 — [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
6. Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM)*, NeurIPS 2019 — [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
7. Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 — [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
8. Dans Diffraction — [GitHub](https://github.com/DanPorter/Dans_Diffraction)
9. Crystallography Open Database — [Website](https://www.crystallography.net/cod/)
