# 实验结果汇总

本文档按当前已有结果整理 PXRD 空间群分类实验。主要测试集规模为
`N=46784`。除特别说明外，指标来自各实验目录的 `results.md` 和
checkpoint 下的 `eval_plots/metrics.json`。

## 总体结论

- 固定强基线 `ResNet1D + label_smoothing` 仍有最高 Top-1：`0.7659`。
- `Dual gated Mamba-KAN + label_smoothing` 是当前最均衡的主结果：Top-1
  接近强基线，同时显著提高 Macro Acc、Macro F1 和 rare-class accuracy。
- 结构增益主要来自三点：WA 是主信号，SA 提供低角补充，gated fusion 和
  Mamba/KAN 进一步改善长尾类。
- `balanced_softmax` 能最大化 rare accuracy，但明显牺牲 Top-1 和 Macro F1，
  更适合作为长尾分析，不适合作为主模型。
- 历史结果 `e04_resnet_wide_cbce_light` 的 Macro/F1 很强，但缺少当前协议下的
  rare、occlusion 和 `metrics.json` 细节，应视为强历史参照而非完整主对比。

## 主结果表

| 实验 | 模型 / loss | Top-1 | Top-5 | Macro Acc | Macro F1 | Rare Acc | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `e02_resnet_deep_label_smoothing` | ResNet1D / label smoothing | **0.7659** | 0.9352 | 0.4674 | 0.4794 | 0.4072 | 固定强基线，Top-1 最好 |
| `e04_resnet_wide_cbce_light` | ResNet1D wide / class-balanced CE | 0.7565 | 0.8842 | **0.5295** | **0.5550** |  | 历史强参照，缺 rare 评估 |
| `e17_dual_gated_mamba_kan_label_smoothing` | Dual gated Mamba-KAN / label smoothing | 0.7522 | 0.9186 | 0.5238 | 0.5453 | 0.4955 | 当前推荐主模型 |
| `e16_dual_gated_mamba_label_smoothing` | Dual gated Mamba / label smoothing | 0.7494 | 0.9138 | 0.5161 | 0.5328 | 0.4811 | 真实 `mamba_ssm` 后端 |
| `e15_dual_gated_resnet_label_smoothing` | Dual gated ResNet / label smoothing | 0.7451 | 0.9361 | 0.4919 | 0.5048 | 0.4613 | 同结构 loss control |
| `e18_dual_gated_resnet_balanced_softmax` | Dual gated ResNet / balanced softmax | 0.6807 | 0.9061 | 0.5295 | 0.4508 | **0.5081** | rare 最好，但整体掉点明显 |
| `e19_dual_gated_resnet_logit_adjustment` | Dual gated ResNet / logit adjustment | 0.7230 | 0.9334 | 0.5164 | 0.4875 | 0.4829 | 温和长尾校正 |
| `e11_dual_gated_kan` | Dual gated KAN / weighted CE | 0.7051 | 0.9206 | 0.5216 | 0.5169 | 0.5063 | KAN 对 rare 有帮助 |

## 相对固定强基线的变化

固定强基线为 `e02_resnet_deep_label_smoothing`。

| 实验 | Δ Top-1 | Δ Top-5 | Δ Macro Acc | Δ Macro F1 | Δ Rare Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e17_dual_gated_mamba_kan_label_smoothing` | -0.0137 | -0.0166 | **+0.0564** | **+0.0659** | +0.0883 |
| `e16_dual_gated_mamba_label_smoothing` | -0.0165 | -0.0214 | +0.0487 | +0.0535 | +0.0739 |
| `e15_dual_gated_resnet_label_smoothing` | -0.0208 | +0.0009 | +0.0245 | +0.0254 | +0.0541 |
| `e11_dual_gated_kan` | -0.0608 | -0.0146 | +0.0542 | +0.0376 | +0.0991 |
| `e18_dual_gated_resnet_balanced_softmax` | -0.0852 | -0.0291 | +0.0621 | -0.0285 | **+0.1009** |
| `e19_dual_gated_resnet_logit_adjustment` | -0.0428 | -0.0018 | +0.0490 | +0.0082 | +0.0757 |

结论：`e17` 没有赢 Top-1，但它在长尾相关指标上提升最均衡，适合作为论文主模型。

## Dual-Range 结构消融

| 实验 | 结构 | Top-1 | Top-5 | Macro Acc | Macro F1 | Rare Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `e14_sa_only_resnet_ablation` | SA only | 0.4176 | 0.8029 | 0.2335 | 0.2282 | 0.2613 |
| `e07_wa_only_resnet_label_smoothing` | WA only | 0.6580 | 0.9021 | 0.4516 | 0.4699 | 0.4252 |
| `e08_dual_concat_resnet` | SA + WA concat | 0.6699 | 0.9308 | 0.5039 | 0.4938 | 0.4703 |
| `e09_dual_gated_resnet` | SA + WA gated | 0.7018 | 0.9361 | 0.5056 | 0.5046 | 0.4757 |
| `e11_dual_gated_kan` | gated + KAN head | 0.7051 | 0.9206 | 0.5216 | 0.5169 | 0.5063 |

关键消融：

- `WA only` 明显强于 `SA only`，说明宽角范围是主要判别信号。
- `SA + WA concat` 相比 `WA only` 提高 Macro Acc `+0.0523`、Rare Acc `+0.0450`，
  说明低角信息对长尾类有补充价值。
- `gated fusion` 相比 concat 提高 Top-1 `+0.0319`，说明自适应融合优于简单拼接。
- `KAN head` 相比 gated MLP 提高 Macro Acc `+0.0159`、Rare Acc `+0.0306`。

## Loss Control

同为 `dual_gated_resnet` 架构时：

| 实验 | Loss | Top-1 | Top-5 | Macro Acc | Macro F1 | Rare Acc | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `e15_dual_gated_resnet_label_smoothing` | label smoothing | **0.7451** | **0.9361** | 0.4919 | **0.5048** | 0.4613 | 最稳，适合主线 |
| `e18_dual_gated_resnet_balanced_softmax` | balanced softmax | 0.6807 | 0.9061 | **0.5295** | 0.4508 | **0.5081** | 长尾最强，但代价大 |
| `e19_dual_gated_resnet_logit_adjustment` | logit adjustment | 0.7230 | 0.9334 | 0.5164 | 0.4875 | 0.4829 | 折中，但不如主模型均衡 |

推荐写法：label smoothing 是主训练策略；balanced softmax 和 logit adjustment
作为长尾控制实验，说明类别先验校正确实能提高 rare/macro，但会带来不同程度的
整体性能代价。

## Mamba/KAN 对照

| 实验 | Top-1 | Top-5 | Macro Acc | Macro F1 | Rare Acc | Mamba 后端 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `e15_dual_gated_resnet_label_smoothing` | 0.7451 | 0.9361 | 0.4919 | 0.5048 | 0.4613 | none / none |
| `e16_dual_gated_mamba_label_smoothing` | 0.7494 | 0.9138 | 0.5161 | 0.5328 | 0.4811 | `mamba_ssm` / `mamba_ssm` |
| `e17_dual_gated_mamba_kan_label_smoothing` | **0.7522** | 0.9186 | **0.5238** | **0.5453** | **0.4955** | `mamba_ssm` / `mamba_ssm` |

结论：在 label smoothing 设置下，Mamba 提升 macro/F1/rare；KAN 在 Mamba 基础上继续带来小幅但稳定的提升。

## Sequence Baselines

| 实验 | 模型 | Top-1 | Top-5 | Macro Acc | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `e05_bigru_patch_weighted_ce_mixed` | BiGRU patch | 0.4743 | 0.8527 | 0.3271 | 0.3371 |
| `e06_patchtst_weighted_ce_mixed` | PatchTST-style | 0.4345 | 0.8337 | 0.3332 | 0.3353 |

结论：当前 BiGRU/PatchTST 序列基线明显弱于 ResNet 和 dual-range 系列，可作为负基线或补充对照。

## 物理解释与可解释性

以 `e17_dual_gated_mamba_kan_label_smoothing` 为主：

- gate mean 为 `0.4918`，说明模型在 SA/WA 间不是单边依赖，而是接近均衡融合。
- 低角遮挡 `5-15 deg` 后：
  - Top-1 从 `0.7522` 降到 `0.3679`
  - Macro Acc 从 `0.5238` 降到 `0.1766`
  - Macro F1 从 `0.5453` 降到 `0.2055`
  - Rare Acc 从 `0.4955` 降到 `0.1514`
- 这支持低角区对空间群分类，尤其是长尾类别识别，具有关键补充作用。

## 推荐论文叙事

主 claim 建议写成：

> Dual-range gated Mamba-KAN does not simply maximize overall Top-1 accuracy;
> instead, it provides a better balance between high overall accuracy and
> long-tail robustness, improving macro-level and rare-class performance under
> severe 230-class imbalance.

中文表述：

> Mamba-KAN 不是单纯 Top-1 最高的模型，但在严重长尾的 230 类空间群分类中，
> 它在保持接近强 ResNet 基线总体准确率的同时，显著提升宏平均与稀有类识别能力。

## 待补与注意事项

- `configs/main/` 已整理主结果入口；Mamba/Mamba-KAN 主结果应从 main preset
  复跑或续跑，旧的 mixed-loss Mamba 矩阵行不再作为当前正式入口。
- `e04_resnet_wide_cbce_light` 是很强的历史结果，建议按当前评估脚本补跑
  `metrics.json`、rare-class 和 low-angle occlusion，保证公平对比。
- README 中仍有 `XX%` 占位，应在最终写作前替换为本汇总中的核心数字。
