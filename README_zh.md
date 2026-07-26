# 面向 PXRD 空间群分类的晶体学启发序列模型

**[English](README.md) | [中文](README_zh.md)**

在 SIMPOD 数据集（467,861 条谱线，每条 10,824 点，230 个类别）上对长序列架构
做空间群分类的系统性研究。

---

## 摘要

粉末 X 射线衍射是测定晶体结构的基础手段，但传统指标化依赖专家解谱。这个任务
对标准深度学习模型有两个不友好的性质：PXRD 谱线是长序列（10,824 点），密集
CNN 与 Transformer 堆叠的代价很高；空间群分布是严重长尾的——最常见的 3 个空间群
占数据的 63.4%，而 230 个空间群中有 103 个的训练样本不超过 100 条。

本仓库研究同时应对这两个性质的架构。表现最好的模型由三部分组成：可学习下采样
前端、序列长度上线性复杂度的 Mamba2 选择性状态空间编码器，以及在 Q 空间中对
检测到的布拉格峰位置（而非稠密信号）进行推理的稀疏峰分支。在保留的 34 组实验中，
最佳配置在测试集上达到 **83.4% top-1** 与 **95.1% top-5**；而最佳 macro-F1
（**0.573**）由另一个配置取得——总体精度与稀有类性能之间存在稳定的权衡。

---

## 现状与适用范围

这是一个仍在推进的研究仓库，不是发布的库。具体来说：

- 所有结果都基于 **模拟** 的 SIMPOD 数据，尚未在真实实验 PXRD 谱线上评估。
- 所有实验只跑了 **单一随机种子（42）**。在没有多种子重复的情况下，1 个百分点
  以内的差异不应被当作有意义的结论。
- 表中所有配置都报告了测试集指标，因为这些数字是在探索过程中产生的。冻结候选
  模型的规范流程见[实验规范](#实验规范)；在把这些数字用于论文之前应当按该流程重跑。
- 检查点与原始日志不纳入 Git（见 `.gitignore`）。

---

## 主要发现

**基于 Mamba 的编码器优于 CNN、RNN 与 Transformer 基线。** ResNet1D 基线为
77.4% top-1，MobileXRD 风格的 Mamba2 变体达到 81–83%。分块 BiGRU（47.4%）与
PatchTST 风格（43.5%）基线差距很大；但这两个基线并未调优，只能作为下界参考，
不能视为公平比较。

**KAN 在这个任务上没有起作用。** 项目最初的设定是 Mamba-KAN 混合模型，KAN 被放在
文献建议的两个位置上分别测试：作为分类头（`m02_bimamba_kan`，72.4% top-1）以及
作为 SupCon 特征投影头（`l02_supcon`，69.7% top-1，其分类头仍是 MLP）。两者都低于
其他部分完全相同、仅使用普通线性头的 `m01_mamba`（82.0%）。最可能的原因是容量：
[`KANHead`](src/models.py) 中的 RBF 样条分支把头部参数量乘以网格点数（这些实验中
为 8），对于 103 个训练样本 ≤100 条的空间群来说很难拟合。这一结果作为负面结果
保留，而不是删除。

**稀疏峰推理是最有前景的方向。** 把布拉格峰显式编码成 token 的两个架构——
`mx14_peakset_ls`（PeakSet 门控融合）与 `mx15_xrd_ctm_ls`（CNN + Mamba + 峰/间隙
Transformer）——取得了最好的 macro-F1 和有竞争力的稀有类精度。这也是整个设计中
真正属于晶体学、而非从视觉任务迁移而来的部分。

**长尾方法在 top-1 与稀有类精度之间的权衡非常陡峭。** τ=1.0 的 logit 调整交叉熵
（`mx12_la_physaug`）取得了全研究最高的稀有类精度（55.7%），但整体 top-1 崩到
14.1%。τ 未做扫描；在已训练的检查点上做事后（post-hoc）调整扫描是自然的下一步，
且完全不需要重训。

---

## 数据集

**来源**：SIMPOD（Rincón 等，*Scientific Data* 12, 1186, 2025）

| 属性 | 数值 |
|---|---|
| 2θ 范围 | 5° – 90° |
| 采样点数 | 10,824 |
| 辐射源 | Cu Kα（λ = 1.5406 Å） |
| 归一化 | 每条谱线除以自身最大值 → 强度 ∈ [0, 1] |
| 谱线总数 | 467,861 |
| 类别数 | 230 空间群 / 7 晶系 |

### 晶系分布

每个晶系的空间群数量是从 `dataset/labels.csv` 实际统计的，而不是按空间群编号
区间推算的。"最大 SG 占比"指该晶系中最常见的单个空间群所占的样本比例。

| 晶系 | SG 编号范围 | 空间群数 | 样本数 | 占比 | 最大 SG 占比 |
|---|---|---|---|---|---|
| 单斜（Monoclinic） | 3–15 | 13 | 228,011 | 48.7% | 64.9% |
| 三斜（Triclinic） | 1–2 | 2 | 113,002 | 24.2% | 96.9% |
| 正交（Orthorhombic） | 16–74 | 59 | 79,990 | 17.1% | 32.4% |
| 四方（Tetragonal） | 75–142 | 68 | 16,335 | 3.5% | 11.7% |
| 三方（Trigonal） | 143–167 | 25 | 13,179 | 2.8% | 28.0% |
| 立方（Cubic） | 195–230 | 36 | 10,357 | 2.2% | 25.9% |
| 六方（Hexagonal） | 168–194 | 27 | 6,987 | 1.5% | 18.6% |
| **合计** | **1–230** | **230** | **467,861** | **100%** | — |

### 为什么晶系不是瓶颈

一个"完美预测晶系、再在该晶系内猜最常见空间群"的 oracle 只能达到
**62.6% top-1**——比最佳模型的 83.4% 低了 21 个点。因此由粗到细的层级方案，其
上限比类别数量给人的直觉要小得多：模型其实已经解决了大量晶系内部的结构。真正的
困难集中在那些既稀有、内部又高度多样的晶系。四方晶系是极端例子：68 个空间群共享
3.5% 的数据，其中 40 个是稀有类。

稀有空间群（训练样本 1–100 条，共 103/230 个）按晶系分布：

| 晶系 | 稀有 SG 数 | 该晶系 SG 总数 |
|---|---|---|
| 四方（Tetragonal） | 40 | 68 |
| 正交（Orthorhombic） | 21 | 59 |
| 立方（Cubic） | 20 | 36 |
| 六方（Hexagonal） | 13 | 27 |
| 三方（Trigonal） | 7 | 25 |
| 单斜（Monoclinic） | 2 | 13 |
| 三斜（Triclinic） | 0 | 2 |

### 数据划分

由 `scripts/make_splits.py` 以固定种子（42）生成：80% 训练（374,294）、10% 验证
（46,783）、10% 测试（46,784），按空间群分层。有 7 个空间群的样本数少于
`--min-per-class`（默认 10），被整体放入训练集，因此验证集与测试集中出现的空间群
为 230 个中的 223 个。

---

## 方法

### 模型族

所有模型在 [`src/models.py`](src/models.py) 中注册，通过配置文件的 `model.name`
选择。

| `model.name` | 说明 |
|---|---|
| `resnet1d` | 一维残差 CNN 基线 |
| `convnext1d` | 一维 ConvNeXt 基线 |
| `bigru_patch` | 分块双向 GRU 基线 |
| `patchtst` | PatchTST 风格的分块 Transformer 基线 |
| `dual_plane_mamba` | 可学习下采样前端 + Mamba/Mamba2 编码器；可选 MobileXRD token mixer、PeakSet 分支与 KAN 头 |
| `xrd_ctm` | CNN + Mamba2 + 峰/间隙 Transformer 三分支，带门控融合 |

### 晶体学专用组件

以下几处设计源自衍射物理本身，而非从视觉或 NLP 架构移植。

**Q 空间峰 token。** [`_PeakSetBranch`](src/models.py) 与
[`_XRDPeakTransformer`](src/models.py) 检测原始谱线的前 K 个局部极大值，并由峰位、
强度、相邻间距和曲率构造逐峰 token。峰位编码为归一化的 Q = 4π·sin(θ)/λ，而不是
2θ 索引，这样各向同性的晶格常数变化在 token 坐标中就变成一个仿射平移。峰的索引
选取不带梯度，强度与曲率则以可微的方式 gather。

**保对称性数据增强。** [`PXRDAugment`](src/data.py) 施加三种可证明不改变空间群标签
的变换：各向同性晶格缩放（在 sinθ 网格上重采样，精确保持峰间比例与系统消光）、
随机 gamma 叠加平滑低频包络的强度扰动（模拟择优取向/织构），以及高斯峰展宽（模拟
晶粒尺寸减小）。在 `mx11_physaug` 中启用。

**跨分支反馈。** 在 `xrd_ctm` 中，峰分支沿信号轴输出一张高斯核重要性图，在池化前
调制 CNN 与 Mamba 的特征图。三个分支的融合权重由一个以学习到的"谱线质量"向量为
条件的门控网络产生；训练时的 branch dropout 会随机把某一分支替换为可学习的空
嵌入。

### 损失函数

实现在 [`src/training.py`](src/training.py)，通过 `loss.name` 选择：
`label_smoothing`、`weighted_ce`、`class_balanced_ce`、`focal`、`asl`、`ldam`
（含延迟重加权 DRW）、`logit_adjusted_ce`，以及可选的有监督对比项。带辅助头的模型
另外支持 `auxiliary_weight`。

### 训练设置

主线配置（如 `configs/mobile/mx03_no_identity.yaml`）的实际超参数：

| 设置 | 值 |
|---|---|
| Python / PyTorch | ≥ 3.10 / 2.8.0+cu128 |
| GPU | H100（W&B tag 记录） |
| 优化器 | AdamW，lr = 3e-4，weight_decay = 1e-3 |
| 学习率调度 | 余弦退火 + 20 epoch 预热 |
| Batch size | 64 |
| Epochs | 120（早停：patience 15，第 60 epoch 起生效） |
| 梯度裁剪 | 1.0 |
| 混合精度 | TF32 + `matmul_precision: high` |
| 模型选择指标 | `val_balanced_acc1_macro`（acc1 与 macro-acc1 各占 0.5） |

`configs/default.yaml` 是可编辑的起始模板（ResNet1D、batch 1024、lr 2e-3、
30 epoch），与上表不同，它不代表主线实验的设置。

---

## 实验结果

下表为所有保留配置的测试集指标。`rare_acc1` 是限定在训练样本 1–100 条的空间群上的
top-1 精度；空白单元格表示该实验早于此指标的引入。规范数据来源：
[`experiments/results.md`](experiments/results.md)。

### 基线

| 配置 | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `b01_resnet` | 0.7736 | 0.9249 | 0.5568 | — |
| `b05_bigru` | 0.4743 | 0.8527 | 0.3371 | — |
| `b06_patchtst` | 0.4345 | 0.8337 | 0.3353 | — |

### Mamba 变体

| 配置 | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `m01_mamba` | 0.8197 | 0.9387 | 0.5621 | — |
| `m03_mamba2` | 0.7714 | 0.9365 | 0.4939 | 0.4288 |
| `m04_mamba2_b64` | 0.8096 | 0.9339 | 0.5564 | 0.5027 |

`m02_bimamba_kan`（KAN 头）见[主要发现](#主要发现)；它作为负面结果保留在
`configs/main/` 中。

### 损失函数消融

| 配置 | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `l04_focal` | 0.7971 | **0.9644** | 0.5289 | — |
| `l05_weighted_ce` | 0.7538 | 0.9479 | 0.5232 | 0.5009 |
| `l01_ldam` | 0.7485 | 0.9306 | 0.4840 | — |
| `l06_asl` | 0.7250 | 0.9481 | 0.4962 | 0.4793 |
| `l03_supcon_lr3e4` | 0.7257 | 0.8855 | 0.4300 | — |
| `l02_supcon` | 0.6966 | 0.9407 | 0.5165 | — |

### 架构消融

| 配置 | top-1 | top-5 | macro-F1 | rare top-1 |
|---|---|---|---|---|
| `a04_wide_frontend` | 0.8259 | 0.9414 | 0.5637 | 0.5153 |
| `a11_inception_frontend` | 0.8183 | 0.9369 | 0.5525 | 0.4847 |
| `a10_multiscale_frontend` | 0.8023 | 0.9360 | 0.5439 | 0.4883 |
| `a02_gated_pool` | 0.7991 | 0.9312 | 0.5457 | 0.4901 |
| `a05_residual_gated_pool` | 0.7970 | 0.9465 | 0.5226 | 0.4757 |
| `a03_convnext_frontend` | 0.7967 | 0.9452 | 0.5454 | 0.4973 |
| `a01_angle_pos` | 0.7759 | 0.9360 | 0.5260 | 0.4811 |

### MobileXRD 与 XRD-CTM 变体

| 配置 | top-1 | top-5 | macro-F1 | rare top-1 |
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

### 如何解读这些数字

三个"最佳"分别落在三个不同的配置上：top-1 是 `mx03_no_identity`，macro-F1 是
`mx15_xrd_ctm_ls`，稀有类精度是 `mx12_la_physaug`。没有单一赢家；排在前面的几个
MobileXRD 变体之间的差距（0.8336 / 0.8318 / 0.8292）落在换一个种子就可能重新排序
的范围内。

`mx12_la_physaug` 最清楚地展示了长尾权衡：用 τ=1.0 的 logit 调整交叉熵训练，把
决策边界推向稀有类的幅度过大，整体精度崩溃，而稀有类精度成为全研究最高。注意：
尽管配置 ID 里带 `physaug`，这一次运行实际关闭了信号增强——它是其配置注释中描述的
"架构 + 损失"对照组。

`mx13_xrd_ctm` 有配置文件但还没有结果行。

---

## 安装

```bash
# 需要预先安装 CUDA Toolkit 12.8 并把 nvcc 加入 PATH，
# 因为 mamba-ssm 与 causal-conv1d 需要从源码编译。
pip install -r requirements.txt
```

当设置 `mamba.backend: auto` 且 CUDA kernel 不可用时，Mamba 层会回退到纯 PyTorch
实现；但主线配置使用的 `mamba2_ssm` 后端需要编译好的包。

## 复现

```bash
# 1. 把 SIMPOD 的结构文件预处理成 dataset/pxrd.npy + dataset/labels.csv。
#    需要先修改脚本顶部的 DATA_DIR / OUT_DIR 以匹配你的本地路径。
python analysis/scripts/preprocess.py

# 2. 生成分层划分（写入 splits/splits.csv）。
python scripts/make_splits.py

# 3. 训练单个配置。
python scripts/train.py --config configs/mobile/mx03_no_identity.yaml

# 4. 评估检查点。会在 <检查点目录>/eval_plots/ 下写出 metrics.json、
#    逐类别 CSV 以及归一化混淆矩阵。
python scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt

# 5. 或端到端跑多个配置，结果行会追加到 experiments/results.md。
python scripts/run_experiments.py --configs \
  configs/main/m01_mamba.yaml \
  configs/mobile/mx03_no_identity.yaml
```

常用参数：`evaluate.py --split val` 用于保持测试集冻结，`--only-rare` 只评估稀有类
样本，`--max-samples N` 用于冒烟测试。

## 目录结构

```
src/            data.py（数据集、划分、增强）、models.py、
                training.py（损失、训练循环）、utils.py
scripts/        train.py、evaluate.py、run_experiments.py、make_splits.py
configs/        default.yaml + baselines/ main/ losses/ ablations/ mobile/
                ID 命名规则见 configs/README.md
experiments/    results.md — 规范指标表。说明见 experiments/README.md
analysis/       预处理、数据集统计、分布可视化
tests/          unittest 测试（目前只覆盖 XRD-CTM）
```

配置使用简短的全局唯一 ID；YAML 文件名主干、`experiment.name` 与主 W&B tag 三者
一致。`configs/README.md` 给出每个保留配置到其历史 ID 的映射。

## 实验规范

以下是预期的规范流程，当前结果只部分遵循：

1. 在 **验证集** 上排序候选模型（`evaluate.py --split val`）。
2. 冻结 2–3 个入围模型，然后在测试集上评估 **一次**。
3. 入围模型至少跑 3 个随机种子，报告均值 ± 标准差。
4. `experiments/results.md` 中每个实验只保留一行；重跑应替换该行而不是追加。

---

## 局限

1. **仅模拟数据。** SIMPOD 谱线没有背景、除固定峰宽外没有仪器展宽、没有择优取向
   效应，且只有单一波长。在真实衍射图上的表现尚不清楚；`PXRDAugment` 中的保对称性
   增强正是为弥合这一差距而设计，但还没有在真实数据上验证过。
2. **单一随机种子。** 没有方差估计，细微差异不可分辨。
3. **测试集复用。** 见[实验规范](#实验规范)。
4. **序列基线未调优。** BiGRU 与 PatchTST 的结果不能被引用为"循环模型或分块
   Transformer 不适合该任务"的证据。
5. **缺少效率测量。** 参数量、FLOPs、延迟与峰值显存均未记录，尽管"线性复杂度序列
   建模"本身是核心动机之一。
6. **预处理脚本路径写死。** `analysis/scripts/preprocess.py` 中的输入输出目录是
   硬编码的 Windows 绝对路径，换机器必须手工修改。
7. **测试覆盖不完整。** `tests/` 目前只覆盖 XRD-CTM 的前向契约。

## 后续计划

- [ ] 在已训练检查点上做事后 logit 调整的 τ 扫描，无需重训即可刻画头部/尾部权衡曲线
- [ ] 把错误分解为跨晶系与晶系内两部分，确认剩余 17% 的错误集中在哪里
- [ ] 效率表：参数量、FLOPs、延迟、峰值显存
- [ ] 入围模型的多种子实验
- [ ] 晶系（7 类）结果——可由已保存的空间群预测直接导出
- [ ] 在真实实验 PXRD 谱线上评估
- [ ] 更强的 BiGRU 与 PatchTST 基线

---

## 引用

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

## 参考文献

1. Rincón 等，*Scientific Data* 12, 1186 (2025) — [DOI: 10.57760/sciencedb.09755](https://doi.org/10.57760/sciencedb.09755)
2. Gu & Dao，*Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
3. Dao & Gu，*Transformers are SSMs (Mamba-2)* — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
4. Liu 等，*KAN: Kolmogorov-Arnold Networks* — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
5. Menon 等，*Long-tail learning via logit adjustment*，ICLR 2021 — [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
6. Cao 等，*Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM)*，NeurIPS 2019 — [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
7. Khosla 等，*Supervised Contrastive Learning*，NeurIPS 2020 — [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
8. Dans Diffraction — [GitHub](https://github.com/DanPorter/Dans_Diffraction)
9. Crystallography Open Database — [官网](https://www.crystallography.net/cod/)
