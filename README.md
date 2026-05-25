# SimPOD: Crystal-system / Space-group Classification from PXRD

Deep-learning models for crystal system (7 classes) and space group (230 classes)
classification from simulated powder X-ray diffraction (PXRD) patterns, using
the [SIMPOD dataset](https://doi.org/10.57760/sciencedb.09755) (467,861
crystals from COD).
Hello

## Project layout

```
SimPOD/
├── analysis/         EDA & one-off preprocessing scripts
├── dataset/          intensities.npy + labels.csv (preprocessed)
├── Structures/       Raw SIMPOD JSON files (read-only, ~30 GB)
├── splits/           Train/val/test split CSV
├── checkpoints/      Model weights (gitignored)
├── runs/             TensorBoard logs (gitignored)
│
├── src/              Importable package
│   ├── data.py       Dataset + train/val/test split logic
│   ├── models.py     Network architectures (baselines + Mamba-KAN later)
│   ├── training.py   Train/val loop, losses, AMP helpers
│   └── utils.py      Config loading, seeding, metrics
│
├── configs/
│   └── default.yaml  Experiment config (copy & edit per run)
│
├── make_splits.py    CLI: build splits/splits.csv
├── train.py          CLI: train a model
├── evaluate.py       CLI: evaluate a checkpoint on test split
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Quickstart

```bash
# 1. Install dependencies (no `pip install -e .` needed — just import from src)
pip install -r requirements.txt

# 2. (one-off) Preprocess raw JSONs into compact npy + labels CSV
python analysis/preprocess.py
# → dataset/intensities.npy   (467861 × 10824 float16, ~10 GB)
# → dataset/labels.csv        (id, space_group, crystal_system, ...)

# 3. (one-off) Generate train/val/test split
python make_splits.py
# → splits/splits.csv

# 4. Train (edit configs/default.yaml first if needed)
python train.py --config configs/default.yaml

# 5. Evaluate on test set
python evaluate.py --checkpoint checkpoints/<run>/best.pt
```

## Dataset

Source: [SIMPOD](https://www.nature.com/articles/s41597-025-05534-3) (Rincón et al., 2025).
Each PXRD pattern has 10,824 intensity points (2θ = 5°–90°, step ≈ 0.008°,
Cu Kα 1.5406 Å), normalized to [0, 1].

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

230 SGs are heavily imbalanced (top-3 SGs cover 63%; 7 SGs have <10 samples).
See `analysis/plots/01_space_group_distribution.png` for the full distribution.

## References

- Rincón S. et al. *A new benchmark for machine learning applied to powder X-ray
  diffraction.* Scientific Data 12, 1186 (2025). DOI: 10.1038/s41597-025-05534-3
- Code: https://github.com/BCV-Uniandes/SIMPOD
