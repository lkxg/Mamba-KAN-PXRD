"""Distribution plots: space group (1-230) and crystal system (7)."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path

# Use a font that supports CJK so Chinese labels render correctly on Windows
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

CSV = Path(r"d:/SimPOD/analysis/metadata.csv")
OUT = Path(r"d:/SimPOD/analysis/plots")
OUT.mkdir(parents=True, exist_ok=True)


# Crystal system ranges (space group number -> system)
SYSTEMS = [
    ("Triclinic",    1,   2,  "三斜"),
    ("Monoclinic",   3,  15,  "单斜"),
    ("Orthorhombic", 16, 74,  "正交"),
    ("Tetragonal",   75, 142, "四方"),
    ("Trigonal",     143,167, "三方"),
    ("Hexagonal",    168,194, "六方"),
    ("Cubic",        195,230, "立方"),
]
SYSTEM_COLORS = {
    "Triclinic":    "#e41a1c",
    "Monoclinic":   "#ff7f00",
    "Orthorhombic": "#ffd92f",
    "Tetragonal":   "#4daf4a",
    "Trigonal":     "#00b8d4",
    "Hexagonal":    "#377eb8",
    "Cubic":        "#984ea3",
}


def sg_to_system(sg: int) -> str:
    for name, lo, hi, _ in SYSTEMS:
        if lo <= sg <= hi:
            return name
    return "Invalid"


def main():
    df = pd.read_csv(CSV)
    df = df[(df["space_group"] >= 1) & (df["space_group"] <= 230)].copy()
    df["system"] = df["space_group"].map(sg_to_system)
    total = len(df)
    print(f"Loaded {total:,} valid samples")

    # ---------- 1. Space group distribution (1..230), log scale ----------
    counts = df["space_group"].value_counts().reindex(range(1, 231), fill_value=0)
    bar_colors = [SYSTEM_COLORS[sg_to_system(sg)] for sg in counts.index]

    # Hermann-Mauguin symbols for the top-3 space groups by count
    # (annotated above their bars on the chart). Together they cover
    # ~63% of the entire dataset.
    SG_SYMBOLS = {
        14: r"$P2_1/c$",   # Monoclinic, ~31.6%
        2:  r"$P\bar{1}$", # Triclinic,  ~23.4%
        15: r"$C2/c$",     # Monoclinic, ~8.4%
    }

    fig, ax = plt.subplots(figsize=(17, 6.2), constrained_layout=True)

    # Alternating soft background bands per crystal system for visual grouping
    for i, (name, lo, hi, _) in enumerate(SYSTEMS):
        ax.axvspan(lo - 0.5, hi + 0.5,
                   facecolor=SYSTEM_COLORS[name], alpha=0.06, zorder=0)

    ax.bar(counts.index, counts.values.clip(min=1), color=bar_colors,
           width=0.95, edgecolor="none", zorder=2)

    ax.set_yscale("log")
    # Tight ceiling: just enough headroom above the in-plot labels.
    ax.set_ylim(0.7, counts.max() * 9)
    ax.set_xlim(0.5, 230.5)
    ax.set_xlabel("空间群编号", fontsize=12)
    ax.set_ylabel("样本数量（对数刻度）", fontsize=12)
    ax.set_title(f"230 个空间群的样本数量分布   (N = {total:,})",
                 fontsize=14, weight="bold", pad=14)

    # Crystal-system region dividers + labels.
    #   - Triclinic (only 2 SGs wide) is too narrow for an in-plot label, so it
    #     lives in the header strip ABOVE the plot with a thin leader line.
    #   - All other systems get their labels placed INSIDE the plot, just above
    #     the tallest bar in their band.
    header_tf = ax.get_xaxis_transform()
    y_inside = counts.max() * 5      # data-y for in-plot system labels (~7.5e5)
    y_outside = 1.015                # axes-fraction y for the Triclinic label
    for name, lo, hi, cn in SYSTEMS:
        ax.axvline(hi + 0.5, color="gray", lw=0.6, ls=":", alpha=0.6, zorder=1)
        n_sys = int(((df["system"] == name).sum()))
        pct = 100.0 * n_sys / total
        label = f"{name}  {cn}\n{n_sys:,}  ({pct:.1f}%)"

        if name == "Triclinic":
            # Outside the plot (band too narrow), with a leader line down
            x_c = (lo + hi) / 2
            ax.text(x_c, y_outside, label,
                    ha="center", va="bottom", fontsize=9.5,
                    color=SYSTEM_COLORS[name], weight="bold",
                    transform=header_tf, clip_on=False, zorder=4)
            ax.plot([x_c, x_c], [1.0, y_outside - 0.003],
                    color=SYSTEM_COLORS[name], lw=0.6, ls=":", alpha=0.7,
                    transform=header_tf, clip_on=False, zorder=3)
        else:
            # Inside the plot, centered horizontally in the band
            ax.text((lo + hi) / 2, y_inside, label,
                    ha="center", va="top", fontsize=9.5,
                    color=SYSTEM_COLORS[name], weight="bold", zorder=4)

    # ---- Annotate Hermann-Mauguin symbols on the tallest bars ----
    # The dataset is dominated by a handful of low-symmetry space groups; mark
    # them so the reader can immediately read off which SGs the spikes are.
    for sg, sym in SG_SYMBOLS.items():
        c = int(counts.loc[sg])
        if c <= 0:
            continue
        ax.annotate(
            sym, xy=(sg, c), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
            color=SYSTEM_COLORS[sg_to_system(sg)], weight="bold", zorder=5,
        )

    ax.grid(axis="y", which="major", color="gray", lw=0.3, alpha=0.4, zorder=0)
    ax.grid(axis="y", which="minor", color="gray", lw=0.2, alpha=0.2, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    out = OUT / "01_space_group_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

    # ---------- 2. Crystal system distribution ----------
    sys_counts = df["system"].value_counts().reindex([s[0] for s in SYSTEMS], fill_value=0)
    sys_pct = sys_counts / total * 100

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)

    colors = [SYSTEM_COLORS[n] for n in sys_counts.index]
    bars = ax.bar(range(len(sys_counts)), sys_counts.values,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(sys_counts)))
    ax.set_xticklabels([f"{n}\n({cn})" for (n, _, _, cn) in SYSTEMS], rotation=0)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_title(f"各晶系样本数量分布 (N = {total:,})", fontsize=13, weight="bold")
    for bar, c, p in zip(bars, sys_counts.values, sys_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}\n{p:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    out = OUT / "02_crystal_system_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved {out}")

    # ---------- Console summary ----------
    print("\nCrystal system summary")
    print("-" * 60)
    summary = pd.DataFrame({
        "system": sys_counts.index,
        "count": sys_counts.values,
        "pct":   sys_pct.round(2).values,
    })
    print(summary.to_string(index=False))

    top10 = counts.sort_values(ascending=False).head(10)
    print("\nTop-10 space groups")
    print("-" * 60)
    for sg, c in top10.items():
        print(f"  SG {sg:3d} ({sg_to_system(sg):12s})  {c:>8,}  {100*c/total:5.2f}%")

    bot = counts[counts > 0].sort_values().head(10)
    print("\nBottom-10 non-empty space groups")
    print("-" * 60)
    for sg, c in bot.items():
        print(f"  SG {sg:3d} ({sg_to_system(sg):12s})  {c:>8,}")

    n_empty = int((counts == 0).sum())
    print(f"\nSpace groups with zero samples: {n_empty}/230")


if __name__ == "__main__":
    main()
