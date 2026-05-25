"""Quick sanity check on the preprocessed dataset."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:/SimPOD/dataset")

X = np.load(ROOT / "intensities.npy", mmap_mode="r")
labels = pd.read_csv(ROOT / "labels.csv")

print(f"X shape : {X.shape}, dtype: {X.dtype}, size: {X.nbytes/1e9:.2f} GB")
print(f"labels  : {labels.shape}, columns: {list(labels.columns)}")
print()

print("Crystal-system counts:")
print(labels["crystal_system"].value_counts())
print()

n_invalid = int((labels["crystal_system_id"] == -1).sum())
n_valid   = int((labels["crystal_system_id"] >= 0).sum())
print(f"Valid rows  : {n_valid:,}")
print(f"Invalid rows: {n_invalid}")
print()

print("First 3 rows of labels:")
print(labels.head(3).to_string(index=False))
print()

print("First 3 rows of X (stats):")
for i in range(3):
    row = np.asarray(X[i], dtype=np.float32)
    print(f"  row {i}: max={row.max():.4f}  min={row.min():.4f}  "
          f"mean={row.mean():.6f}  nonzero={(row > 0).sum()}/{X.shape[1]}")
print()

# Make sure intensities are still in [0, 1] after float16 round-trip
print("Sampling 1000 random rows to check value range...")
rng = np.random.default_rng(0)
sample_rows = rng.choice(X.shape[0], size=1000, replace=False)
sample = np.asarray(X[sample_rows], dtype=np.float32)
print(f"  global max: {sample.max():.4f}")
print(f"  global min: {sample.min():.4f}")
print(f"  rows with max ≥ 0.99: {(sample.max(axis=1) >= 0.99).sum()}/1000")
print(f"  rows with all zeros : {(sample.max(axis=1) == 0).sum()}/1000")
