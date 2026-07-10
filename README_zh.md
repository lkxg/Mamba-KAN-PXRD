# Hybrid Mamba-KAN for PXRD-based Crystal Structure Classification

**[English](README.md) | [中文](README_zh.md)**

---

## 摘要

粉末 X 射线衍射（PXRD）是晶体学中确定物质结构的核心技术，但传统方法依赖专家经验、效率低下。近年来，深度学习在PXRD分析中展现出潜力，但仍面临两大挑战：PXRD信号序列长导致传统CNN计算复杂度高，以及晶系/空间群分类任务类别极不平衡（7/230 类）。本文提出一种结合选择性状态空间模型（Mamba）与 Kolmogorov-Arnold 网络（KAN）的混合分类框架：Mamba 以线性复杂度建模长距离依赖，KAN 以可学习激活函数增强分类决策能力。

---

## 研究背景与贡献

粉末 X 射线衍射（PXRD）是晶体学中确定物质结构的经典方法。传统上，晶系和空间群的确定需要专业知识和手动分析，这是一个耗时且依赖专家经验的过程。

近年来，深度学习在材料科学领域展现出巨大潜力。然而，PXRD 自动分类的研究仍处于起步阶段。**Mamba** 作为一种选择性状态空间模型，在长序列建模上具有线性复杂度的优势；**KAN** 作为一种新型神经网络架构，在符号函数拟合上展现出超越 MLP 的能力。将两者结合用于 PXRD 分析是一个值得探索的研究方向。

**本项目的主要贡献：**

1. 提出了首个将 Mamba 与 KAN 结合用于 PXRD 分类的深度学习框架
2. 提供了处理严重类别不平衡问题的分层划分策略
3. 构建了保留的 CNN、RNN、Transformer 基准与 Mamba-KAN 混合模型的性能对比
4. 开源了数据预处理、模型训练和评估的完整代码

---

## 相关工作

| 工作 | 方法 | 准确率 | 局限性 |
|------|------|--------|--------|
| 本项目 (Mamba-KAN) | SSM + KAN | **XX%** (Top-5) | 仅模拟数据 |
| 本项目 (ResNet1D) | 1D-CNN | XX% (Top-5) | 基准模型 |
| SIMPOD 原始论文 | - | 35% (真实数据) | 仅 Top-5 |
| ... | ... | ... | ... |

---

## 方法

### 核心架构：Mamba-KAN Hybrid

我们提出的 Mamba-KAN 混合架构结合了两种前沿模型的优势：

**Mamba（选择性状态空间模型）**
- 优势：线性复杂度的长距离依赖建模，适合长序列 PXRD 信号
- 机制：选择性扫描（selective scan）机制，动态过滤无关信息

**KAN（Kolmogorov-Arnold 网络）**
- 优势：相比 MLP 在拟合复杂函数时更加参数高效
- 机制：基于可学习激活函数的线性组合

**混合策略**
- Mamba 作为序列编码器，提取 PXRD 信号的多尺度特征
- KAN 作为分类头，利用其强大的函数拟合能力进行分类决策

### 基准模型

为公平对比，我们保留了 ResNet1D、ConvNeXt1D、BiGRU-patch 和 PatchTST-style 基准。

**ResNet1D**
- 结构：Stem(Conv1D + BN + GELU + MaxPool) → 4 stages × 2 ResBlocks → AdaptiveAvgPool → Linear
- 特点：残差连接缓解梯度消失，适合长序列

### 训练策略

- **损失函数**：交叉熵 / 加权交叉熵 / Focal Loss
- **优化器**：AdamW (lr=1e-3, weight_decay=1e-4)
- **学习率调度**：余弦退火 + 预热
- **正则化**：Label Smoothing, Dropout, 梯度裁剪
- **混合精度**：CUDA 上使用 BF16

---

## 数据集

**数据来源**：SIMPOD 数据集 (Rincón et al., *Scientific Data* 12, 1186, 2025)

每个晶体包含模拟的一维 PXRD 图案，参数如下：
- **2θ 范围**：5° – 90°
- **采样点数**：10,824
- **辐射源**：Cu Kα (λ = 1.5406 Å)
- **强度归一化**：每条曲线除以最大值，∈ [0, 1]

### 晶系分布

| 晶系 | 空间群编号 | 样本数 | 占比 |
|------|-----------|--------|------|
| 单斜（Monoclinic） | 3-15 | 228,011 | 48.7% |
| 三斜（Triclinic） | 1-2 | 113,002 | 24.2% |
| 正交（Orthorhombic） | 16-74 | 79,990 | 17.1% |
| 四方（Tetragonal） | 75-142 | 16,335 | 3.5% |
| 三方（Trigonal） | 143-167 | 13,179 | 2.8% |
| 立方（Cubic） | 195-230 | 10,357 | 2.2% |
| 六方（Hexagonal） | 168-194 | 6,987 | 1.5% |
| **合计** | **1-230** | **467,861** | **100%** |

### 数据集划分

- **训练集**：80%（包括所有稀有空间群）
- **验证集**：10%（按空间群分层抽样）
- **测试集**：10%（按空间群分层抽样）

**注意**：7 个空间群样本数少于 10 个，全部划入训练集以保证有效性。

---

## 实验

### 实验环境

| 设置 | 值 |
|------|-----|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| GPU | RTX 3090/4090 (16GB VRAM) |
| Batch Size | 128 |
| Epochs | 20 |
| 混合精度 | BF16 |

### 可复现性

```bash
# 1. 环境安装
pip install -r requirements.txt

# 2. 数据预处理
python analysis/preprocess.py

# 3. 生成分划分
python make_splits.py

# 4. 训练默认 ResNet1D 基准
python3 scripts/train.py --config configs/default.yaml

# 5. Mamba-KAN 模型训练
python3 scripts/train.py --config configs/main/m01_mamba.yaml

# 6. 模型评估
python3 scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt
```

---

## 实验结果


### 结果分析

1. **Mamba-KAN vs 基准模型**：Mamba-KAN 在空间群分类任务上相比 ResNet1D 提升/下降约 XX%
2. **类别不平衡影响**：前 3 大空间群占 63% 样本，模型倾向于预测这些类别
3. **模拟 vs 真实数据**：SIMPOD 数据无噪声、峰宽固定，真实数据表现预计下降
4. **晶系 vs 空间群**：晶系分类（7 类）明显易于空间群分类（230 类）

---

## 局限性

1. **模拟数据局限**：无背景噪声、固定峰宽、单一波长，与真实实验条件存在差异
2. **类别不平衡**：稀有空间群样本极少，难以学习有效表示
3. **泛化能力**：需要进一步在真实 PXRD 数据上验证模型效果

---

## 未来工作

- [ ] 在真实 PXRD 数据上微调 Mamba-KAN
- [ ] 探索更多 Mamba-KAN 变体（如 Mamba2-KAN）
- [ ] 引入不确定性量化
- [ ] 开发 Web 界面便于非专业用户使用

---

## 引用

如果您在研究中使用本代码，请按以下格式引用：

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
  author={[您的姓名]},
  year={2025},
  url={https://github.com/[your-repo]}
}
```

---

## 许可协议

本项目采用 [MIT License](LICENSE)。

---

## 致谢

感谢 [资助机构] 的支持，感谢 [SIMPOD 团队] 提供数据集。

---

## 参考文献

1. Rincón et al., *Scientific Data* 12, 1186 (2025) - [DOI: 10.57760/sciectedb.09755](https://doi.org/10.57760/sciencedb.09755)
2. Dans Diffraction - [GitHub](https://github.com/DanPorter/Dans_Diffraction)
3. Crystallography Open Database - [Website](https://www.crystallography.net/cod/)
4. Mamba: Linear-Time Sequence Modeling with Selective State Spaces - [Paper](https://arxiv.org/abs/2312.00752)
5. KAN: Kolmogorov-Arnold Networks - [Paper](https://arxiv.org/abs/2404.19756)
