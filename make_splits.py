"""Generate splits/splits.csv from dataset/labels.csv.

Usage
-----
    python make_splits.py
    python make_splits.py --val-frac 0.05 --test-frac 0.05 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import make_stratified_split


def main():
    ap = argparse.ArgumentParser(description="Build train/val/test splits.")
    ap.add_argument("--labels", default="dataset/labels.csv",
                    help="Path to labels.csv")
    ap.add_argument("--out", default="splits/splits.csv",
                    help="Output CSV path")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--min-per-class", type=int, default=10,
                    help="SGs with fewer samples than this go entirely to train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    labels = labels[labels["crystal_system_id"] >= 0].reset_index(drop=True)
    print(f"Loaded {len(labels):,} valid rows from {args.labels}")

    splits = make_stratified_split(
        labels,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_per_class=args.min_per_class,
        random_state=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    print("Split summary:")
    print(splits["split"].value_counts())
    print()
    for s in ("train", "val", "test"):
        sub = splits[splits["split"] == s]
        n_sgs = sub["space_group"].nunique()
        print(f"  {s:5s}: {len(sub):>7,} samples,  {n_sgs}/230 SGs covered")


if __name__ == "__main__":
    main()
