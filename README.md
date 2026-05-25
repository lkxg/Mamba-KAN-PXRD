# PXRD-MK:Crystal-System and Space-Group Classification from PXRD

PyTorch pipeline for predicting the **crystal system** (7 classes) and
**space group** (230 classes) of an unknown crystal from its simulated
powder X-ray diffraction (PXRD) pattern, built on the
[SIMPOD dataset](https://doi.org/10.57760/sciencedb.09755) (467,861 crystals
from the Crystallography Open Database).

The repository covers the full path from raw JSON files to a trained
classifier:

1. **Preprocessing** raw SIMPOD JSONs (~30 GB) into a compact memory-mapped
   `intensities.npy` (10 GB, float16) + `labels.csv` (14 MB).
2. **Stratified split** that handles severe class imbalance (7 SGs have
   < 10 samples; the smallest has 1).
3. **Training** with mixed precision (FP16/BF16), cosine LR schedule and
   TensorBoard logging.
4. **Evaluation** with top-1 / top-5 accuracy on a held-out test split.
5. Pluggable architectures — baselines (MLP, 1D-ResNet) included; Mamba-KAN
   hybrids are the planned next step.

---

## Why this dataset is hard

| Property | Value | Implication for ML |
|---|---|---|
| Pattern length | 10,824 points (2θ = 5°–90°) | Long 1D sequence — favors CNN / SSM / sparse-attention models |
| Classes (SG) | 230 | Very fine-grained classification |
| Class imbalance (SG) | top-3 cover **63%**, 7 SGs have **< 10 samples** | Standard CE biased toward head; need re-weighting / focal loss |
| Class imbalance (system) | Monoclinic 48.7% vs Hexagonal 1.5% | Same issue, less extreme |
| Intra-class variability | Very high (different unit cells, atomic content) | Pattern complexity ≠ class label |
| Source | Simulated, no background, fixed peak width | Real-data generalisation needs care |

See `analysis/plots/01_space_group_distribution.png` and
`analysis/plots/04_peak_count_distribution.png` for the full picture.

---

## Project layout

```
SimPOD/
├── analysis/                      EDA & one-off preprocessing scripts
│   ├── preprocess.py              raw JSON → intensities.npy + labels.csv
│   ├── scan_metadata.py           fast SG-only scan (used by EDA)
│   ├── plot_distributions.py      figs 01, 02
│   ├── plot_intensity_curves.py   figs 03, 04
│   ├── check_dataset.py           sanity check on the npy
│   └── plots/                     generated figures
│
├── dataset/                       preprocessed inputs (gitignored)
│   ├── intensities.npy            (467861, 10824) float16 mmap
│   └── labels.csv                 row, id, space_group, crystal_system, crystal_system_id
│
├── Structures/                    raw SIMPOD JSON files (read-only, ~30 GB, gitignored)
├── splits/splits.csv              row, space_group, split, split_id (produced by make_splits.py)
├── checkpoints/                   model weights (gitignored)
├── runs/                          TensorBoard logs (gitignored)
│
├── src/                           importable package
│   ├── data.py                    PXRDDataset + stratified split logic
│   ├── models.py                  MLP, 1D-ResNet, (Mamba-KAN later)
│   ├── training.py                train/eval loops, losses (CE, weighted, focal)
│   └── utils.py                   YAML config, seeding, top-k metrics
│
├── configs/default.yaml           single editable training config
├── make_splits.py                 CLI: build splits/splits.csv
├── train.py                       CLI: train one experiment
├── evaluate.py                    CLI: evaluate a checkpoint on the test split
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Requirements

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.1 with CUDA support
- **GPU**: RTX 3090 / 4090 class recommended; 16 GB VRAM for baselines, 24 GB
  needed for full Mamba/KAN-scale runs at sequence length 10,824
- **Disk**: ~30 GB for raw JSONs (read-only), ~10 GB for `intensities.npy`,
  plus checkpoint + TensorBoard space
- See `requirements.txt` for the full Python dependency list

---

## Quickstart

```bash
# 1. Install dependencies (no `pip install -e .` needed — `src/` is on the PYTHONPATH automatically)
pip install -r requirements.txt

# 2. (one-off, ~20 min) Preprocess raw JSONs into a compact npy + labels CSV
python analysis/preprocess.py
# → dataset/intensities.npy   (467861 × 10824 float16, ~10 GB)
# → dataset/labels.csv        (467861 rows, 5 columns)

# 3. (one-off, < 1 min) Generate train/val/test split
python make_splits.py
# → splits/splits.csv  (80% train / 10% val / 10% test, stratified by SG)

# 4. Train (edit configs/default.yaml first)
python train.py --config configs/default.yaml
# Logs go to runs/<experiment_name>_<timestamp>/  (open with `tensorboard --logdir runs`)
# Best checkpoint saved to <out_dir>/best.pt

# 5. Evaluate on the held-out test split
python evaluate.py --checkpoint checkpoints/<run>/best.pt
```

---

## Dataset

Source: **SIMPOD** (Rincón et al., *Scientific Data* 12, 1186, 2025).
Each crystal has a simulated 1D PXRD pattern computed by
[Dans Diffraction](https://github.com/DanPorter/Dans_Diffraction) with:

- **2θ range:** 5° – 90°
- **Step size:** ≈ 0.0079°  →  10,824 intensity points
- **Source:** Cu Kα, λ = 1.5406 Å
- **Peak width:** 0.01°
- **Normalisation:** each pattern divided by its own max → intensities ∈ [0, 1]

**Crystal-system breakdown:**

| Crystal system | Samples | Share |
|----------------|--------:|------:|
| Monoclinic     | 228,011 | 48.7% |
| Triclinic      | 113,002 | 24.2% |
| Orthorhombic   |  79,990 | 17.1% |
| Tetragonal     |  16,335 |  3.5% |
| Trigonal       |  13,179 |  2.8% |
| Cubic          |  10,357 |  2.2% |
| Hexagonal      |   6,987 |  1.5% |
| **Total**      | **467,861** | **100%** |

**Top-3 space groups cover 63% of all samples** (P2₁/c, P-1, C2/c).
**7 SGs have < 10 samples** (the smallest, SG 208, has only 1).
Stratified splitting puts these rare SGs entirely in *train* (they cannot be
meaningfully validated).

> **Limitations of the simulated data**: no background, fixed peak width, single
> wavelength, flat detector. Models trained purely on SIMPOD will degrade on
> real experimental patterns (the paper reports best top-5 = 35% on 20 real
> samples vs 82.79% on simulated test data).

---

## Configuration reference (`configs/default.yaml`)

```yaml
experiment:
  name: default                  # used for run dir + checkpoint path
  seed: 42                       # seeds Python / numpy / torch / cuDNN

data:
  root: dataset                  # where intensities.npy + labels.csv live
  splits_csv: splits/splits.csv
  task: space_group              # space_group (230) | crystal_system (7)
  batch_size: 128
  num_workers: 4                 # DataLoader workers
  pin_memory: true

model:
  name: resnet1d                 # mlp | resnet1d
  resnet1d:                      # kwargs forwarded to ResNet1D(...)
    base_channels: 32
    blocks_per_stage: [2, 2, 2, 2]
  mlp:                           # kwargs forwarded to MLPClassifier(...)
    hidden: [1024, 512]
    dropout: 0.2

optim:
  lr: 1.0e-3
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]            # AdamW

scheduler:
  name: cosine                   # cosine | none
  warmup_epochs: 1

train:
  epochs: 20
  amp: true                      # mixed precision (FP16) — required on 4090
  grad_clip: 1.0
  log_every: 100                 # batches between progress prints

loss:
  name: ce                       # ce | weighted_ce | focal
  label_smoothing: 0.0
  focal_gamma: 2.0               # only for name: focal

checkpoint:
  out_dir: checkpoints/default
  save_best: true                # save the best-val-acc1 checkpoint
  monitor: val_acc1
```

To run a different experiment, **copy** `default.yaml` to e.g.
`configs/mlp_focal.yaml`, edit the values you want to change, then call
`python train.py --config configs/mlp_focal.yaml`.

---

## Models

| Model       | Where            | Notes                                                           |
|-------------|------------------|------------------------------------------------------------------|
| MLP         | `src/models.py`  | Plain feed-forward over the flattened 10,824-D vector. Reference floor. |
| ResNet1D    | `src/models.py`  | 1D-CNN with 4 strided residual stages. Decent baseline at < 5 M params. |
| Mamba-KAN   | *planned*        | Selective state-space model + KAN head — primary research target. |

Add new architectures to `src/models.py` and register them in
`build_model()` inside `train.py`.

---

## Training

**Expected wall-clock per epoch on RTX 4090 (24 GB), batch 128, AMP on:**

| Model        | Params | Throughput     | One epoch  |
|--------------|-------:|---------------:|-----------:|
| MLP          |  ~12 M | ~50 k samples/s| ~30 sec    |
| ResNet1D     |   ~5 M | ~15 k samples/s| ~1–2 min   |
| Mamba-KAN    |  ~50 M |  ~3 k samples/s| ~5–10 min  |

A typical 20-epoch baseline run finishes in **20–40 minutes**.
A 50-epoch Mamba-KAN run takes **4–8 hours**.

**Bottleneck checklist** if you see GPU under-utilisation:
- `num_workers` 0 → bump to 4–8
- raw JSON loading (without preprocessing) → fix by running `analysis/preprocess.py` first
- `amp: false` → turn on (free 1.8× speedup on 4090)

---

## Evaluation

`evaluate.py` reports loss, top-1 and top-5 accuracy on the **test split**.

For the SG task, top-5 is the more meaningful number — many SGs share
near-identical PXRD patterns (e.g., centric/non-centric variants),
and "top-5 contains the truth" is a realistic expert-tool target.

For comparison, the SIMPOD paper reports (on their own 25 k-sample test
set, 1-D diffractograms only):

|              | Acc       | Top-5     |
|--------------|----------:|----------:|
| MLP (H2O)    |     33.0  |     74.0  |
| Random Forest|     37.5  |     77.1  |

Their best radial-image model (Swin V2 + pretraining): **45.3 / 82.8**.

---

## Reproducibility

All randomness comes from a single seed in the config (`experiment.seed`).
With the same seed, identical hardware and `deterministic=True` (set in
`set_seed`), runs reproduce to within numerical noise.

Splits are deterministic for a given `--seed` passed to `make_splits.py`.
The default seed (42) is what you should use when reporting results.

If `dataset/intensities.npy` is regenerated, its row order is fixed
(filenames are sorted in `analysis/preprocess.py`), so existing
`splits.csv` remains valid.

---

## Roadmap

- [x] Preprocessing pipeline (JSON → npy)
- [x] Stratified split with low-count-SG handling
- [x] MLP + 1D-ResNet baselines
- [x] AMP training loop + TensorBoard
- [ ] **Mamba-1D + KAN classifier head**
- [ ] Hybrid Mamba-KAN with physics priors (peak position channel, etc.)
- [ ] Crystal-system task as a multi-task auxiliary
- [ ] Real-experimental-data fine-tuning study
- [ ] Cross-validation runner

---

## Citation

If you use this code, please also cite the SIMPOD dataset paper:

```bibtex
@article{rincon2025simpod,
  title   = {A new benchmark for machine learning applied to powder X-ray diffraction},
  author  = {Rinc{\'o}n, Sergio and Gonz{\'a}lez, Gabriel and Mac{\'i}as, Mario A. and Arbel{\'a}ez, Pablo},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1186},
  year    = {2025},
  doi     = {10.1038/s41597-025-05534-3}
}
```

Dataset: [https://doi.org/10.57760/sciencedb.09755](https://doi.org/10.57760/sciencedb.09755)
Original code: [https://github.com/BCV-Uniandes/SIMPOD](https://github.com/BCV-Uniandes/SIMPOD)

---

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The underlying SIMPOD data is **CC BY-NC-ND 4.0** — its licence terms apply
to any redistribution of the raw or derived data.
