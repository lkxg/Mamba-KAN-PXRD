# Hybrid Mamba-KAN for PXRD-based Crystal Structure Classification

**[English](README.md) | [中文](README_zh.md)**

---

## Abstract

Powder X-ray diffraction (PXRD) is a fundamental technique in crystallography for determining crystal structures, but traditional methods rely on expert experience and are inefficient. Recent advances in deep learning have shown potential for PXRD analysis, yet two major challenges remain: long PXRD sequences (10,824 points) lead to high computational complexity with conventional CNNs, and extreme class imbalance (7/230 classes) makes effective learning difficult. This paper proposes a hybrid classification framework combining Mamba (Selective State Space Model) with KAN (Kolmogorov-Arnold Networks): Mamba captures long-range dependencies with linear complexity, while KAN enhances classification through learnable activation functions. 
---

## Introduction

Powder X-ray diffraction (PXRD) is a classical method in crystallography for determining crystal structures. Traditionally, identifying crystal systems and space groups requires expert knowledge and manual analysis, which is time-consuming and dependent on experience.

Recently, deep learning has shown great potential in materials science. However, research on automatic PXRD classification remains in its early stages. **Mamba**, as a selective state space model, offers linear-complexity advantages in long-range sequence modeling; **KAN**, as a novel neural architecture, demonstrates superior capability in symbolic function fitting compared to MLPs. Combining these two for PXRD analysis represents a promising research direction.

**Main Contributions:**

1. Propose the first deep learning framework combining Mamba and KAN for PXRD classification
2. Provide a stratified splitting strategy handling severe class imbalance
3. Build performance comparisons between retained CNN/RNN/Transformer baselines and the Mamba-KAN hybrid
4. Open-source complete code for data preprocessing, model training, and evaluation

---

## Related Work

| Work | Method | Accuracy | Limitation |
|------|--------|----------|------------|
| This work (Mamba-KAN) | SSM + KAN | **XX%** (Top-5) | Simulated data only |
| This work (ResNet1D) | 1D-CNN | XX% (Top-5) | Baseline |
| SIMPOD Original Paper | - | 35% (Real data) | Top-5 only |
| ... | ... | ... | ... |

---

## Method

### Core Architecture: Mamba-KAN Hybrid

Our Mamba-KAN hybrid architecture combines the strengths of two cutting-edge models:

**Mamba (Selective State Space Model)**
- Advantage: Linear-complexity long-range dependency modeling, suitable for long-sequence PXRD signals
- Mechanism: Selective scan mechanism for dynamic information filtering

**KAN (Kolmogorov-Arnold Networks)**
- Advantage: More parameter-efficient than MLP in fitting complex functions
- Mechanism: Learnable activation functions based on linear combinations

**Hybrid Strategy**
- Mamba serves as sequence encoder, extracting multi-scale features from PXRD signals
- KAN serves as classification head, leveraging powerful function fitting capability for classification decisions

### Baseline Models

For fair comparison, we retain ResNet1D, ConvNeXt1D, BiGRU-patch, and PatchTST-style baselines.

**ResNet1D**
- Structure: Stem(Conv1D + BN + GELU + MaxPool) → 4 stages × 2 ResBlocks → AdaptiveAvgPool → Linear
- Characteristics: Residual connections mitigate gradient vanishing, suitable for long sequences

### Training Strategy

- **Loss Function**: Cross-Entropy / Weighted Cross-Entropy / Focal Loss
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **LR Schedule**: Cosine annealing + warmup
- **Regularization**: Label Smoothing, Dropout, Gradient Clipping
- **Mixed Precision**: BF16 on CUDA

---

## Dataset

**Source**: SIMPOD Dataset (Rincón et al., *Scientific Data* 12, 1186, 2025)

Each crystal contains a simulated 1D PXRD pattern computed with:
- **2θ range**: 5° – 90°
- **Sampling points**: 10,824
- **Source**: Cu Kα (λ = 1.5406 Å)
- **Normalization**: Each pattern divided by its max → intensities ∈ [0, 1]

### Crystal System Distribution

| Crystal System | Space Groups | Samples | Percentage |
|----------------|--------------|---------|------------|
| Monoclinic | 3-15 | 228,011 | 48.7% |
| Triclinic | 1-2 | 113,002 | 24.2% |
| Orthorhombic | 16-74 | 79,990 | 17.1% |
| Tetragonal | 75-142 | 16,335 | 3.5% |
| Trigonal | 143-167 | 13,179 | 2.8% |
| Cubic | 195-230 | 10,357 | 2.2% |
| Hexagonal | 168-194 | 6,987 | 1.5% |
| **Total** | **1-230** | **467,861** | **100%** |

### Dataset Split

- **Training set**: 80% (including all rare space groups)
- **Validation set**: 10% (stratified by space group)
- **Test set**: 10% (stratified by space group)

**Note**: 7 space groups have fewer than 10 samples; all are placed in the training set for validity.

---

## Experiments

### Experimental Setup

| Setting | Value |
|---------|-------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| GPU | RTX 3090/4090 (16GB VRAM) |
| Batch Size | 128 |
| Epochs | 20 |
| Mixed Precision | BF16 |

### Reproducibility

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python analysis/preprocess.py

# 3. Generate splits
python make_splits.py

# 4. Train the default ResNet1D baseline
python3 scripts/train.py --config configs/default.yaml

# 5. Train Mamba-KAN model
python3 scripts/train.py --config configs/main/m01_mamba.yaml

# 6. Evaluate
python3 scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt
```

---

## Results

### Space Group Classification (230 classes)



### Crystal System Classification (7 classes)



### Discussion

1. **Mamba-KAN vs Baselines**: Mamba-KAN improves/decreases by ~XX% on space group classification compared to ResNet1D
2. **Class Imbalance Impact**: Top-3 space groups account for 63% of samples; model tends to predict these classes
3. **Simulated vs Real Data**: SIMPOD data has no noise, fixed peak width; real data performance expected to decrease
4. **Crystal System vs Space Group**: Crystal system classification (7 classes) is significantly easier than space group (230 classes)

---

## Limitations

1. **Simulated Data Limitations**: No background noise, fixed peak width, single wavelength; differs from real experimental conditions
2. **Class Imbalance**: Rare space groups have very few samples, making effective learning difficult
3. **Generalization**: Further validation on real PXRD data is needed

---

## Future Work

- [ ] Fine-tune Mamba-KAN on real PXRD data
- [ ] Explore more Mamba-KAN variants (e.g., Mamba2-KAN)
- [ ] Introduce uncertainty quantification
- [ ] Develop web interface for non-expert users

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{simPOD2025,
  title={SIMPOD: A Large-Scale Simulated PXRD Dataset for Crystal System and Space Group Classification},
  author={Rincón et al.},
  journal={Scientific Data},
  year={2025},
  doi={10.57760/sciencedb.09755}
}

@misc{mambakan2025,
  title={Mamba-KAN: Hybrid Architecture for Crystal System and Space Group Classification from PXRD},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

We thank [funding agency] for support and the [SIMPOD team] for providing the dataset.

---

## References

1. Rincón et al., *Scientific Data* 12, 1186 (2025) - [DOI: 10.57760/sciencedb.09755](https://doi.org/10.57760/sciencedb.09755)
2. Dans Diffraction - [GitHub](https://github.com/DanPorter/Dans_Diffraction)
3. Crystallography Open Database - [Website](https://www.crystallography.net/cod/)
4. Mamba: Linear-Time Sequence Modeling with Selective State Spaces - [Paper](https://arxiv.org/abs/2312.00752)
5. KAN: Kolmogorov-Arnold Networks - [Paper](https://arxiv.org/abs/2404.19756)
