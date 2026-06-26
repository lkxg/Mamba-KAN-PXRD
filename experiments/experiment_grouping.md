# 论文实验分组建议

本文档用于把当前配置和结果整理成论文/报告中的实验分组。除特别说明外，
默认任务是 `space_group` 230 类分类，指标来自当前已有 `results.md`。

## 分组原则

- **Baseline**：读者会自然期待比较的外部或通用模型，例如 CNN、ConvNeXt、RNN、Transformer、XRDMamba 风格模型。
- **Ours**：用于支撑本文方法贡献的模型，应围绕同一条主线命名和叙事。
- **Ablation**：解释某个组件或训练策略是否有效，不建议和 baseline 混在一张主表里。
- **Appendix / Supplementary**：历史实验、旧协议实验、负结果、重复探索或还没补齐评估的实验。

## 推荐主对比表

这张表适合作为论文主结果表。`m44` 已有配置但当前结果表尚未记录 eval 结果，建议补评估后再放入正式表。

| 角色 | 论文中名称 | 配置 / 实验 | Top-1 | Top-5 | Macro F1 | 说明 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Baseline | ResNet1D | `m01_resnet1d_label_smoothing` | 0.773576 | 0.924867 | 0.556769 | 最重要强 CNN baseline |
| Baseline | ConvNeXt1D | `m44_convnext1d_label_smoothing` | TBD | TBD | TBD | 现代 CNN baseline，需补 eval |
| Baseline | BiGRU-Patch | `e05_bigru_patch_weighted_ce_mixed` | 0.474300 | 0.852700 | 0.337100 | 序列 baseline |
| Baseline | PatchTST-style | `e06_patchtst_weighted_ce_mixed` | 0.434500 | 0.833700 | 0.335300 | Transformer patch baseline |
| Baseline | XRDMamba-paper-style | `m30_xrdmamba_paper_params_ce` | 0.408537 | 0.792408 | 0.210401 | 外部 Mamba 风格对照 |
| Baseline | XRDMamba-repo-style | `m31_xrdmamba_repo_resconv_5000_ce` | 0.403792 | 0.698230 | 0.260011 | 外部 repo 风格对照 |
| Ours | Ours-Mamba backbone | `m18_single_plane_learned_downsample_mamba_d128_l8_label_smoothing` | **0.819725** | 0.938676 | **0.562054** | 当前最强主干模型 |
| Ours | Ours-Mamba-KAN | `m39_single_plane_learned_downsample_mamba_local_kan_adapter_label_smoothing` | 0.804741 | 0.933375 | 0.529165 | 最适合作为 Mamba-KAN 版本 |
| Ours | Ours-Mamba + gated pooling | `m41_single_plane_learned_downsample_mamba_gated_pool_label_smoothing` | 0.799141 | 0.931152 | 0.545715 | 增强变体，Macro F1 较稳 |

## 你的模型怎么命名

建议不要把所有 `dual_plane_mamba` 都叫 Mamba-KAN。可以这样分：

| 论文中名称 | 推荐实验 | 是否主推 | 理由 |
| --- | --- | --- | --- |
| Ours-Mamba backbone | `m18` | 是 | 当前最强，说明 learned-downsample + Mamba 主干有效 |
| Ours-Mamba-KAN | `m39` | 是 | 有明确 local KAN adapter，最贴近 Mamba-KAN 题目 |
| Ours-Mamba + gated pooling | `m41` | 可选 | 展示 pooling/attention 汇聚增强 |
| Ours-Peak-token variant | `m34`, `m35` | 可放 appendix | 结果不错，但属于扩展路线 |

如果论文标题强调 **Mamba-KAN**，建议主文写成：

> We first build a strong learned-downsample Mamba backbone (`m18`), then add a
> lightweight local KAN adapter to form the Mamba-KAN variant (`m39`).

这样既不硬说 `m18` 是 KAN，又能让结果叙事更可信。

## 结构消融组

这组用于回答“哪些结构组件真的有用”。推荐单独成表，不和 baseline 主表混在一起。

| 消融问题 | 实验 | Top-1 | Top-5 | Macro F1 | 结论用途 |
| --- | --- | ---: | ---: | ---: | --- |
| 基础主干 | `m18` | 0.819725 | 0.938676 | 0.562054 | learned-downsample + Mamba 强主干 |
| token-wise KAN adapter | `m38` | 0.775500 | 0.935576 | 0.503051 | token KAN adapter 不如主干 |
| local KAN adapter | `m39` | 0.804741 | 0.933375 | 0.529165 | 更适合作为 Mamba-KAN 版本 |
| angle position encoding | `m40` | 0.775864 | 0.936047 | 0.525953 | 位置信息有帮助但不超过主干 |
| gated pooling | `m41` | 0.799141 | 0.931152 | 0.545715 | 汇聚方式有效，Macro F1 较好 |
| dual-plane SA raw + WA learned | `m42` | 0.611534 | 0.843793 | 0.451373 | 双分支该版本明显掉点 |
| convnext downsample frontend | `m43` | TBD | TBD | TBD | 已训练，建议补 eval 后判断 |
| peak-token branch | `m34` | 0.811218 | 0.949833 | 0.540735 | 可作为扩展结构 |
| peak-token pooling | `m35` | 0.816839 | 0.939808 | 0.532999 | 可作为扩展结构 |

## 早期 Dual-Range / Mamba-KAN 原型消融

这组适合解释早期设计探索，建议放 supplementary 或作为一张小消融表。

| 实验 | 结构含义 | Top-1 | Top-5 | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| `m02` | dual gated ResNet | 0.724158 | 0.901740 | 0.493329 |
| `m03` | dual gated KAN | 0.732259 | 0.900051 | 0.521820 |
| `m04` | dual gated Mamba | 0.746730 | 0.921832 | 0.547742 |
| `m05` | dual gated Mamba-KAN | 0.736512 | 0.908216 | 0.536976 |
| `m08` | angle-aware Mamba-KAN | 0.722918 | 0.915954 | 0.540583 |
| `m09` | relaxed ResNet + Mamba-KAN | 0.766052 | 0.932049 | 0.557235 |

建议写法：这组说明 Mamba/KAN/gating 的早期组合方向是有意义的，但最终主线转向
`m18` 之后的 learned-downsample single-plane Mamba 系列。

## Loss / Long-tail 消融组

这组用于回答“长尾训练策略是否有用”。它们不是 baseline，也不是最终方法本体。

| 实验 | 策略 | Top-1 | Top-5 | Macro F1 | 建议用途 |
| --- | --- | ---: | ---: | ---: | --- |
| `m06` | LDAM-DRW on dual gated ResNet | 0.666168 | 0.900265 | 0.456669 | loss 消融 |
| `m07` | cRT on dual gated ResNet | 0.688227 | 0.905951 | 0.504379 | long-tail 消融 |
| `m22` | LDAM-DRW on Mamba backbone | 0.748461 | 0.930575 | 0.484004 | 不如 label smoothing 主干 |
| `m29` | focal loss | 0.797089 | **0.964390** | 0.528890 | Top-5 很强，可分析 |
| `m36` | weighted CE + longer schedule | 0.753826 | 0.947867 | 0.523186 | 训练策略消融 |
| `m37` | ASL | 0.724991 | 0.948081 | 0.496188 | long-tail loss 对照 |

结论建议：主训练策略仍以 `label_smoothing` 为主。长尾 loss 可以改善部分指标，
但目前没有稳定超过 `m18`。

## Crystal System 任务

`crystal_system` 是 7 类任务，不能和 230 类 `space_group` 主表混在一起。

| 任务 | 实验 | Top-1 | Top-5 | Macro F1 | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| Crystal system | `c01_resnet1d_crystal_label_smoothing` | 0.843258 | 0.989120 | 0.868897 | 单独报告任务难度或 sanity check |

## 不建议放入主表的实验

| 类型 | 实验 | 原因 |
| --- | --- | --- |
| 历史/旧协议 | `e02`, `e07`, `e08`, `e09`, `e11`, `e14`, `e15`-`e19` | 可以作为历史参照，但不建议混入最终主表 |
| 未补齐当前 eval | `m43`, `m44` | 有配置/或 checkpoint，但主结果表缺完整 eval 记录 |
| 依赖当前环境缺失 | `m21` | 需要 `efficient_kan`，否则当前环境不能复跑 |
| 负结果/探索 | `m11`-`m17`, `m25`, `m26`, `m28`, `m30`, `m31` | 可放 appendix 或 related baseline，不建议主推 |

## 推荐论文结构

1. **Main Results**：`m01`, `m44`, `e05`, `e06`, `m30`, `m31`, `m18`, `m39`, `m41`
2. **Architecture Ablation**：`m18`, `m38`, `m39`, `m40`, `m41`, `m42`, `m43`, `m34`, `m35`
3. **Mamba/KAN Prototype Ablation**：`m02`, `m03`, `m04`, `m05`, `m08`, `m09`
4. **Long-tail Training Ablation**：`m06`, `m07`, `m22`, `m29`, `m36`, `m37`
5. **Crystal System Task**：`c01`

## 最终推荐结论

主 claim 可以围绕下面这句话组织：

> Learned-downsample Mamba is the strongest backbone for long PXRD sequences,
> while the local KAN adapter provides a principled Mamba-KAN variant. Compared
> with CNN, RNN, Transformer, and XRDMamba-style baselines, the proposed model
> family achieves the best balance between overall accuracy and macro-level
> long-tail performance.

