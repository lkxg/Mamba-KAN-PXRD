"""PXRD pattern visualisations for the dataset.

Two figures are produced:

  03_intensity_curves_per_system.png
      Two representative PXRD patterns chosen at the symmetry extremes —
      Triclinic (lowest symmetry, dense forest of peaks) and Cubic
      (highest symmetry, very few sharp peaks).  Together they bracket
      what every PXRD pattern in the dataset looks like.

  04_peak_count_distribution.png
      Per-system box plot of "number of peaks per pattern" (peaks defined as
      local maxima with intensity > 5% of pattern max).  Peak count is one
      of the strongest hand-crafted features for crystal-system / SG
      classification: it tracks symmetry directly (high symmetry → fewer,
      sharper peaks).  The figure quantifies how separable the systems are
      on this single feature alone.
"""
import json
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

DATA = Path(r"d:/SimPOD/Structures/Structures")
CSV  = Path(r"d:/SimPOD/analysis/metadata.csv")
OUT  = Path(r"d:/SimPOD/analysis/plots")
OUT.mkdir(parents=True, exist_ok=True)

TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 90.0

SYSTEMS = [
    ("Triclinic",    1,   2,  "三斜", "#e41a1c"),
    ("Monoclinic",   3,  15,  "单斜", "#ff7f00"),
    ("Orthorhombic", 16, 74,  "正交", "#d4b106"),
    ("Tetragonal",   75, 142, "四方", "#4daf4a"),
    ("Trigonal",     143,167, "三方", "#00b8d4"),
    ("Hexagonal",    168,194, "六方", "#377eb8"),
    ("Cubic",        195,230, "立方", "#984ea3"),
]

N_PEAKS_PER_SYSTEM  = 60  # samples per system used for the figure-04 boxplot

# Peak-detection thresholds (applied per pattern, on raw intensities)
PEAK_HEIGHT_FRAC = 0.05   # peak must reach >= 5% of that pattern's max
PEAK_MIN_DIST    = 3      # min distance between peaks, in bins


def sg_to_system(sg: int) -> str:
    for name, lo, hi, *_ in SYSTEMS:
        if lo <= sg <= hi:
            return name
    return "Invalid"


def load_intensities(file_id: str) -> np.ndarray:
    with open(DATA / f"{file_id}.json", "r") as f:
        d = json.load(f)
    return np.asarray(d["intensities"], dtype=np.float64)


def main():
    random.seed(0)
    np.random.seed(0)

    df = pd.read_csv(CSV)
    df = df[(df["space_group"] >= 1) & (df["space_group"] <= 230)].copy()
    df["system"] = df["space_group"].map(sg_to_system)

    # =============================================================
    # Figure 03: two extreme-symmetry systems (Triclinic + Cubic)
    # =============================================================
    # Triclinic = lowest symmetry → dense forest of overlapping reflections.
    # Cubic     = highest symmetry → very few, sharp, well-separated peaks.
    # The other 5 systems lie on the continuum between these two; their
    # statistics (peak count, etc.) are summarised in figure 04.
    print("Loading curves for figure 03 ...")
    extreme_names = ("Triclinic", "Cubic")
    extreme_picks = []
    for name, lo, hi, cn, color in SYSTEMS:
        if name not in extreme_names:
            continue
        sub = df[df["system"] == name]
        if len(sub) == 0:
            continue
        row = sub.sample(1, random_state=0).iloc[0]
        fid = str(row["ID"])
        sg = int(row["space_group"])
        y = load_intensities(fid)
        x = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, len(y))
        extreme_picks.append((name, cn, color, fid, sg, x, y))
        print(f"  {name:12s}  ID={fid}  SG={sg}")

    fig, axes = plt.subplots(len(extreme_picks), 1, figsize=(13, 6),
                             sharex=True, constrained_layout=True)
    if len(extreme_picks) == 1:
        axes = [axes]
    for ax, (name, cn, color, fid, sg, x, y) in zip(axes, extreme_picks):
        ax.plot(x, y, color=color, lw=0.7)
        ax.fill_between(x, 0, y, color=color, alpha=0.25)
        ax.set_ylabel("强度", fontsize=10)
        ax.set_title(f"{name} ({cn})    ID {fid},  空间群 {sg}",
                     loc="left", fontsize=11, color=color, weight="bold")
        ax.set_xlim(TWO_THETA_MIN, TWO_THETA_MAX)
        ax.grid(alpha=0.25, lw=0.4)
    axes[-1].set_xlabel(r"$2\theta$ (度)")
    fig.suptitle(r"对称性两极的代表 PXRD 谱  ($2\theta = 5^\circ$–$90^\circ$)",
                 fontsize=13, weight="bold")
    out = OUT / "03_intensity_curves_per_system.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    # =============================================================
    # Figure 04: peak-count distribution per system (box plot)
    # =============================================================
    print(f"\nCounting peaks ({N_PEAKS_PER_SYSTEM} samples / system) ...")
    peak_counts = {name: [] for name, *_ in SYSTEMS}
    for name, lo, hi, cn, color in SYSTEMS:
        sub = df[df["system"] == name]
        if len(sub) == 0:
            continue
        rows = sub.sample(n=min(N_PEAKS_PER_SYSTEM, len(sub)), random_state=42)
        for _, row in rows.iterrows():
            fid = str(row["ID"])
            try:
                y = load_intensities(fid)
                if y.max() <= 0:
                    continue
                peaks, _ = find_peaks(y,
                                      height=PEAK_HEIGHT_FRAC * y.max(),
                                      distance=PEAK_MIN_DIST)
                peak_counts[name].append(len(peaks))
            except Exception:
                continue
        c = peak_counts[name]
        if c:
            print(f"  {name:12s}  n={len(c):3d}  median={int(np.median(c)):4d}  "
                  f"min={min(c):4d}  max={max(c):4d}")

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    box_data = [peak_counts[n] for (n, *_) in SYSTEMS]
    box_labels = [f"{n}\n({cn})" for (n, _, _, cn, _) in SYSTEMS]
    box_colors = [c for (*_, c) in SYSTEMS]

    bp = ax.boxplot(box_data, tick_labels=box_labels,
                    patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="#333333", lw=0.8),
                    capprops=dict(color="#333333", lw=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="#888888",
                                    markeredgecolor="none", alpha=0.45))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    # Annotate medians on top of each box
    for i, data in enumerate(box_data, start=1):
        if not data:
            continue
        med = int(np.median(data))
        ax.text(i, max(data) * 1.04, f"中位数 {med}",
                ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_ylabel("每条 PXRD 谱的峰数  (强度 > 5% max)", fontsize=11)
    ax.set_title(f"各晶系 PXRD 峰数分布  (每晶系 {N_PEAKS_PER_SYSTEM} 个随机样本)",
                 fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    out = OUT / "04_peak_count_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
