"""数据集封装和训练/验证/测试划分工具。

本模块提供两个主要功能：
1. PXRDDataset 类：基于内存映射的 PyTorch Dataset，
   用于加载预处理好的 pxrd.npy + labels.csv 数据。
2. make_stratified_split / load_splits 函数：
   分层划分生成器，确保每个空间群在各个划分中的比例一致。
   对于样本数过少的空间群（少于 min_per_class），全部划入训练集。
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
# 数据集类定义
# =====================================================================

TaskType = Literal["space_group", "crystal_system"]


class PXRDDataset(Dataset):
    """基于内存映射的 PXRD 数据集。

    通过 mmap（内存映射）方式加载 pxrd.npy 文件，
    这样可以处理 10GB 以上的大文件，而不需要将整个文件加载到内存中。
    每个样本的强度数据在读取时动态转换为 float32 类型（原始存储为 float16）。

    属性:
        task: 分类任务类型，"space_group"（230 类）或 "crystal_system"（7 类）
        num_classes: 分类类别数
        signal_length: 每条 PXRD 曲线的采样点数（固定为 10824）
    """

    def __init__(
        self,
        data_dir: str | Path,
        rows: np.ndarray | None = None,
        task: TaskType = "space_group",
        transform=None,
    ) -> None:
        """
        初始化 PXRD 数据集。

        参数:
            data_dir: 数据目录路径，应包含 pxrd.npy 和 labels.csv 文件
            rows: 要加载的样本行索引，None 表示加载所有有效样本
            task: 分类任务，"space_group" 或 "crystal_system"
            transform: 可选的数据变换函数，应用于每条 PXRD 曲线
        """
        data_dir = Path(data_dir)
        npy_path = data_dir / "pxrd.npy"
        csv_path = data_dir / "labels.csv"

        # 检查必要的文件是否存在
        if not npy_path.exists():
            raise FileNotFoundError(
                f"{npy_path} not found — run analysis/preprocess.py first."
            )
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")

        # 使用内存映射加载强度数据（只读模式，不占用大量内存）
        self._X: np.ndarray = np.load(npy_path, mmap_mode="r")
        labels = pd.read_csv(csv_path)
        if not labels["row"].is_unique:
            raise ValueError(f"{csv_path} contains duplicate row values.")
        self._labels: pd.DataFrame = labels.set_index("row", drop=False)

        label_rows = self._labels.index.to_numpy(dtype=np.int64)
        if len(label_rows) and (
            label_rows.min() < 0 or label_rows.max() >= self._X.shape[0]
        ):
            raise ValueError("label rows must be valid row indices into pxrd.npy.")

        # 验证任务类型
        if task not in ("space_group", "crystal_system"):
            raise ValueError(f"unknown task: {task!r}")
        self.task = task
        self.transform = transform

        # 过滤有效样本（crystal_system_id >= 0 表示有效样本）
        valid_mask = self._labels["crystal_system_id"] >= 0
        valid_rows = self._labels.loc[valid_mask, "row"].to_numpy(dtype=np.int64)
        if rows is None:
            rows = valid_rows
        else:
            rows = np.asarray(rows, dtype=np.int64)
            missing_rows = np.setdiff1d(rows, label_rows, assume_unique=False)
            if len(missing_rows):
                raise ValueError(f"unknown row values in split: {missing_rows[:10]}")
            rows = rows[self._labels.loc[rows, "crystal_system_id"].to_numpy() >= 0]
        self.rows = rows

        # 根据任务类型设置标签和类别数
        self._y = np.full(self._X.shape[0], -100, dtype=np.int64)
        if task == "space_group":
            # 空间群标签：范围 1-230，转换为 0-229 以便 PyTorch 处理
            self._y[label_rows] = (
                self._labels["space_group"].to_numpy() - 1
            ).astype(np.int64)
            self.num_classes = 230
        else:
            # 晶系标签：范围 0-6
            self._y[label_rows] = (
                self._labels["crystal_system_id"].to_numpy()
            ).astype(np.int64)
            self.num_classes = 7

    def __len__(self) -> int:
        """返回数据集中样本的数量。"""
        return len(self.rows)

    def __getitem__(self, idx: int):
        """获取指定索引的样本。

        参数:
            idx: 样本索引（相对于当前 Dataset 的索引）

        返回:
            元组 (x, y)：x 是 PXRD 强度曲线张量，y 是类别标签张量
        """
        row = int(self.rows[idx])
        # 从内存映射数组中读取数据，转换为 float32
        x = np.asarray(self._X[row], dtype=np.float32)
        x_t = torch.from_numpy(x)
        if self.transform is not None:
            x_t = self.transform(x_t)
        y_t = torch.tensor(int(self._y[row]), dtype=torch.long)
        return x_t, y_t

    @property
    def signal_length(self) -> int:
        """返回每条 PXRD 曲线的采样点数量。"""
        return int(self._X.shape[1])


# =====================================================================
# 数据划分工具
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
    """构建分层抽样的训练/验证/测试划分。

    分层抽样确保每个空间群在各划分中的比例与总体一致。
    对于样本数过少的空间群（少于 min_per_class），全部划入训练集，
    因为它们无法进行有效的分层划分。

    参数:
        labels: 包含 'row' 和 'space_group' 列的 DataFrame
        val_frac: 验证集比例（默认 10%）
        test_frac: 测试集比例（默认 10%）
        min_per_class: 每个空间群最少样本数，少于此数则全部划入训练集
        random_state: 随机种子，确保结果可复现

    返回:
        DataFrame，包含 'row', 'space_group', 'split', 'split_id' 列

    说明:
        - 晶系标签可通过对空间群分组隐式获得（因为晶系与空间群有固定对应关系）
        - 由于按空间群分层抽样，自动保证了晶系的分层比例一致
    """
    if not {"row", "space_group"}.issubset(labels.columns):
        raise ValueError("`labels` must contain 'row' and 'space_group' columns")

    # 统计每个空间群的样本数
    counts = labels["space_group"].value_counts()
    # 找出样本数过少的空间群（不可分割）
    rare_sgs = counts[counts < min_per_class].index

    # 分离出不可分割的空间群（全部划入训练集）
    rare_mask = labels["space_group"].isin(rare_sgs)
    rare_df = labels.loc[rare_mask].copy()
    splittable_df = labels.loc[~rare_mask].copy()
    train_idx_rare = rare_df["row"].to_numpy()

    # 对可分割的空间群进行分层抽样
    if len(splittable_df) == 0:
        train_idx, val_idx, test_idx = train_idx_rare, np.array([]), np.array([])
    else:
        rows = splittable_df["row"].to_numpy()
        y = splittable_df["space_group"].to_numpy()

        # 第一步：划分测试集
        rows_trainval, rows_test, y_trainval, _ = train_test_split(
            rows, y,
            test_size=test_frac,
            stratify=y,
            random_state=random_state,
        )
        # 第二步：从剩余数据中划分验证集
        rel_val_frac = val_frac / (1.0 - test_frac)
        rows_train, rows_val, _, _ = train_test_split(
            rows_trainval, y_trainval,
            test_size=rel_val_frac,
            stratify=y_trainval,
            random_state=random_state,
        )
        # 合并不可分割空间群的训练集索引
        train_idx = np.concatenate([train_idx_rare, rows_train])
        val_idx = rows_val
        test_idx = rows_test

    # 构建输出 DataFrame
    out = labels[["row", "space_group"]].copy()
    out["split"] = "train"
    out.loc[out["row"].isin(val_idx), "split"] = "val"
    out.loc[out["row"].isin(test_idx), "split"] = "test"
    out["split_id"] = out["split"].map(SPLIT_TO_ID).astype(np.int8)

    # 打乱顺序并按索引排序
    return out.sample(frac=1.0, random_state=random_state).sort_index()


def load_splits(splits_csv: str | Path) -> dict[str, np.ndarray]:
    """加载划分文件并返回各划分的样本索引。

    参数:
        splits_csv: 划分 CSV 文件路径（由 make_stratified_split 生成）

    返回:
        字典，键为划分名称（"train", "val", "test"），值为对应的样本行索引数组
    """
    df = pd.read_csv(splits_csv)
    return {
        name: df.loc[df["split"] == name, "row"].to_numpy()
        for name in ("train", "val", "test")
    }


def class_counts_for_rows(
    labels_csv: str | Path,
    rows: np.ndarray,
    task: TaskType,
) -> np.ndarray:
    """统计指定样本行的类别数量。"""
    labels = pd.read_csv(labels_csv).set_index("row")
    selected = labels.loc[np.asarray(rows, dtype=np.int64)]

    if task == "space_group":
        return (selected["space_group"]
                .value_counts()
                .reindex(range(1, 231), fill_value=0)
                .to_numpy())
    if task == "crystal_system":
        return (selected["crystal_system_id"]
                .value_counts()
                .reindex(range(7), fill_value=0)
                .to_numpy())
    raise ValueError(f"unknown task: {task!r}")


def labels_for_rows(
    labels_csv: str | Path,
    rows: np.ndarray,
    task: TaskType,
) -> np.ndarray:
    """返回指定样本行的 0-based 分类标签。"""
    labels = pd.read_csv(labels_csv).set_index("row")
    selected = labels.loc[np.asarray(rows, dtype=np.int64)]

    if task == "space_group":
        return (selected["space_group"].to_numpy() - 1).astype(np.int64)
    if task == "crystal_system":
        return selected["crystal_system_id"].to_numpy().astype(np.int64)
    raise ValueError(f"unknown task: {task!r}")
