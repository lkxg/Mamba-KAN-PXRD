"""根据 labels.csv 生成分层抽样划分文件 splits/splits.csv。

使用方法:
    python make_splits.py
    python make_splits.py --val-frac 0.05 --test-frac 0.05 --seed 0

输出文件格式:
    splits/splits.csv - 包含 row, space_group, split 三列
    其中 split 列的值为 "train", "val" 或 "test"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import make_stratified_split


def main():
    """主函数：生成分层抽样划分。"""
    # ---------- 命令行参数解析 ----------
    ap = argparse.ArgumentParser(description="构建训练/验证/测试划分")
    ap.add_argument("--labels", default="dataset/labels.csv",
                    help="标签 CSV 文件路径")
    ap.add_argument("--out", default="splits/splits.csv",
                    help="输出 CSV 文件路径")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="验证集比例（默认 10%%）")
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="测试集比例（默认 10%%）")
    ap.add_argument("--min-per-class", type=int, default=10,
                    help="每个空间群最少样本数，少于此数的空间群全部划入训练集")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子，用于保证结果可复现")
    args = ap.parse_args()

    # ---------- 加载标签数据 ----------
    labels = pd.read_csv(args.labels)
    # 只保留有效的晶系分类（crystal_system_id >= 0）
    labels = labels[labels["crystal_system_id"] >= 0].reset_index(drop=True)
    print(f"Loaded {len(labels):,} valid rows from {args.labels}")

    # ---------- 生成分层抽样划分 ----------
    # 按空间群进行分层抽样，确保每个划分中各空间群的比例一致
    splits = make_stratified_split(
        labels,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_per_class=args.min_per_class,
        random_state=args.seed,
    )

    # ---------- 保存结果 ----------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    # ---------- 打印划分统计信息 ----------
    print("Split summary:")
    print(splits["split"].value_counts())
    print()
    for s in ("train", "val", "test"):
        sub = splits[splits["split"] == s]
        n_sgs = sub["space_group"].nunique()
        print(f"  {s:5s}: {len(sub):>7,} samples,  {n_sgs}/230 SGs covered")


if __name__ == "__main__":
    main()
