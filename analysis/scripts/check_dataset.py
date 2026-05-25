"""数据集完整性快速检查脚本。

对预处理后的数据集进行基本验证：
- 验证 intensities.npy 和 labels.csv 的基本统计信息
- 检查数据格式是否正确
- 验证数值范围是否合理
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:/SimPOD/dataset")

# 加载数据
X = np.load(ROOT / "intensities.npy", mmap_mode="r")
labels = pd.read_csv(ROOT / "labels.csv")

# 打印基本信息
print(f"X shape : {X.shape}, dtype: {X.dtype}, size: {X.nbytes/1e9:.2f} GB")
print(f"labels  : {labels.shape}, columns: {list(labels.columns)}")
print()

# 打印晶系分布
print("Crystal-system counts:")
print(labels["crystal_system"].value_counts())
print()

# 统计有效和无效样本数
n_invalid = int((labels["crystal_system_id"] == -1).sum())
n_valid   = int((labels["crystal_system_id"] >= 0).sum())
print(f"Valid rows  : {n_valid:,}")
print(f"Invalid rows: {n_invalid}")
print()

# 打印标签文件的前几行
print("First 3 rows of labels:")
print(labels.head(3).to_string(index=False))
print()

# 打印强度数据的前几行统计
print("First 3 rows of X (stats):")
for i in range(3):
    row = np.asarray(X[i], dtype=np.float32)
    print(f"  row {i}: max={row.max():.4f}  min={row.min():.4f}  "
          f"mean={row.mean():.6f}  nonzero={(row > 0).sum()}/{X.shape[1]}")
print()

# 随机采样 1000 行，验证数值范围是否在 [0, 1] 区间
print("Sampling 1000 random rows to check value range...")
rng = np.random.default_rng(0)
sample_rows = rng.choice(X.shape[0], size=1000, replace=False)
sample = np.asarray(X[sample_rows], dtype=np.float32)
print(f"  global max: {sample.max():.4f}")
print(f"  global min: {sample.min():.4f}")
print(f"  rows with max ≥ 0.99: {(sample.max(axis=1) >= 0.99).sum()}/1000")
print(f"  rows with all zeros : {(sample.max(axis=1) == 0).sum()}/1000")
