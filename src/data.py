"""Dataset wrapper and train/val/test split utilities.

Two things live here:
  - ``PXRDDataset``         memory-mapped PyTorch ``Dataset`` over the
                            preprocessed ``intensities.npy`` + ``labels.csv``.
  - ``make_stratified_split`` / ``load_splits``
                            stratified-by-SG split builder, with a special
                            rule that SGs with very few samples go entirely
                            to *train* (they are not split-able).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# =====================================================================
# Dataset
# =====================================================================

TaskType = Literal["space_group", "crystal_system"]


class PXRDDataset(Dataset):
    """Memory-mapped PXRD dataset.

    Loads ``intensities.npy`` lazily via mmap so the 10 GB array does not
    have to fit in RAM. Each item is converted to float32 on the fly
    (the npy is stored as float16).
    """

    def __init__(
        self,
        data_dir: str | Path,
        rows: np.ndarray | None = None,
        task: TaskType = "space_group",
        transform=None,
    ) -> None:
        data_dir = Path(data_dir)
        npy_path = data_dir / "intensities.npy"
        csv_path = data_dir / "labels.csv"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"{npy_path} not found — run analysis/preprocess.py first."
            )
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")

        self._X: np.ndarray = np.load(npy_path, mmap_mode="r")
        self._labels: pd.DataFrame = pd.read_csv(csv_path)

        if task not in ("space_group", "crystal_system"):
            raise ValueError(f"unknown task: {task!r}")
        self.task = task
        self.transform = transform

        valid_mask = self._labels["crystal_system_id"] >= 0
        if rows is None:
            rows = self._labels.loc[valid_mask, "row"].to_numpy()
        else:
            rows = np.asarray(rows, dtype=np.int64)
            rows = rows[valid_mask.iloc[rows].to_numpy()]
        self.rows = rows

        if task == "space_group":
            self._y = (self._labels["space_group"].to_numpy() - 1).astype(np.int64)
            self.num_classes = 230
        else:
            self._y = self._labels["crystal_system_id"].to_numpy().astype(np.int64)
            self.num_classes = 7

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = int(self.rows[idx])
        x = np.asarray(self._X[row], dtype=np.float32)
        x_t = torch.from_numpy(x)
        if self.transform is not None:
            x_t = self.transform(x_t)
        y_t = torch.tensor(int(self._y[row]), dtype=torch.long)
        return x_t, y_t

    @property
    def signal_length(self) -> int:
        return int(self._X.shape[1])


# =====================================================================
# Splits
# =====================================================================

SplitName = Literal["train", "val", "test"]
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}


def make_stratified_split(
    labels: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    min_per_class: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a stratified train/val/test split.

    SGs with fewer than ``min_per_class`` samples have ALL their rows
    assigned to *train* (they are not split-able).  Stratification by SG
    automatically stratifies the crystal-system label as well.
    """
    if not {"row", "space_group"}.issubset(labels.columns):
        raise ValueError("`labels` must contain 'row' and 'space_group' columns")

    counts = labels["space_group"].value_counts()
    rare_sgs = counts[counts < min_per_class].index

    rare_mask = labels["space_group"].isin(rare_sgs)
    rare_df = labels.loc[rare_mask].copy()
    splittable_df = labels.loc[~rare_mask].copy()
    train_idx_rare = rare_df["row"].to_numpy()

    if len(splittable_df) == 0:
        train_idx, val_idx, test_idx = train_idx_rare, np.array([]), np.array([])
    else:
        rows = splittable_df["row"].to_numpy()
        y = splittable_df["space_group"].to_numpy()

        rows_trainval, rows_test, y_trainval, _ = train_test_split(
            rows, y,
            test_size=test_frac,
            stratify=y,
            random_state=random_state,
        )
        rel_val_frac = val_frac / (1.0 - test_frac)
        rows_train, rows_val, _, _ = train_test_split(
            rows_trainval, y_trainval,
            test_size=rel_val_frac,
            stratify=y_trainval,
            random_state=random_state,
        )
        train_idx = np.concatenate([train_idx_rare, rows_train])
        val_idx = rows_val
        test_idx = rows_test

    out = labels[["row", "space_group"]].copy()
    out["split"] = "train"
    out.loc[out["row"].isin(val_idx), "split"] = "val"
    out.loc[out["row"].isin(test_idx), "split"] = "test"
    out["split_id"] = out["split"].map(SPLIT_TO_ID).astype(np.int8)

    return out.sample(frac=1.0, random_state=random_state).sort_index()


def load_splits(splits_csv: str | Path) -> dict[str, np.ndarray]:
    """Load ``splits.csv`` and return {split_name: row_indices}."""
    df = pd.read_csv(splits_csv)
    return {
        name: df.loc[df["split"] == name, "row"].to_numpy()
        for name in ("train", "val", "test")
    }
