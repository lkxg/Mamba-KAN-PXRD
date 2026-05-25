"""数据分布可视化脚本。

生成两个图表：
1. 空间群分布图（01_space_group_distribution.png）
   - 展示 230 个空间群的样本数量分布
   - 使用对数刻度y轴，以便显示大范围的数值差异
   - 按晶系用不同颜色标注区域

2. 晶系分布图（02_crystal_system_distribution.png）
   - 展示 7 个晶系的样本数量分布
   - 包含样本数量和百分比标注
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path

# 设置中文字体支持（Windows 平台）
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# 路径配置
CSV = Path(r"d:/SimPOD/analysis/metadata.csv")
OUT = Path(r"d:/SimPOD/analysis/plots")
OUT.mkdir(parents=True, exist_ok=True)


# 晶系定义：名称、空间群下限、空间群上限、中文名称
SYSTEMS = [
    ("Triclinic",    1,   2,  "三斜"),
    ("Monoclinic",   3,  15,  "单斜"),
    ("Orthorhombic", 16, 74,  "正交"),
    ("Tetragonal",   75, 142, "四方"),
    ("Trigonal",     143,167, "三方"),
    ("Hexagonal",    168,194, "六方"),
    ("Cubic",        195,230, "立方"),
]

# 晶系对应的颜色
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
    """根据空间群编号返回晶系名称。

    参数:
        sg: 空间群编号（1-230）

    返回:
        晶系名称，如果不在有效范围内返回 "Invalid"
    """
    for name, lo, hi, _ in SYSTEMS:
        if lo <= sg <= hi:
            return name
    return "Invalid"


def main():
    """主函数：加载数据并生成可视化图表。"""
    # 加载元数据
    df = pd.read_csv(CSV)
    # 过滤有效空间群（1-230）
    df = df[(df["space_group"] >= 1) & (df["space_group"] <= 230)].copy()
    df["system"] = df["space_group"].map(sg_to_system)
    total = len(df)
    print(f"Loaded {total:,} valid samples")

    # ========== 图 1：空间群分布（1-230），对数刻度 ==========
    # 统计每个空间群的样本数，填充没有样本的空间群为 0
    counts = df["space_group"].value_counts().reindex(range(1, 231), fill_value=0)
    bar_colors = [SYSTEM_COLORS[sg_to_system(sg)] for sg in counts.index]

    # 标注最常见的三个空间群（占总数据集约 63%）
    SG_SYMBOLS = {
        14: r"$P2_1/c$",   # 单斜晶系，约 31.6%
        2:  r"$P\bar{1}$", # 三斜晶系，约 23.4%
        15: r"$C2/c$",     # 单斜晶系，约 8.4%
    }

    # 创建图表
    fig, ax = plt.subplots(figsize=(17, 6.2), constrained_layout=True)

    # 添加晶系背景色带，助于视觉分组
    for i, (name, lo, hi, _) in enumerate(SYSTEMS):
        ax.axvspan(lo - 0.5, hi + 0.5,
                   facecolor=SYSTEM_COLORS[name], alpha=0.06, zorder=0)

    # 绘制柱状图
    ax.bar(counts.index, counts.values.clip(min=1), color=bar_colors,
           width=0.95, edgecolor="none", zorder=2)

    # 设置对数刻度和轴范围
    ax.set_yscale("log")
    ax.set_ylim(0.7, counts.max() * 9)
    ax.set_xlim(0.5, 230.5)
    ax.set_xlabel("空间群编号", fontsize=12)
    ax.set_ylabel("样本数量（对数刻度）", fontsize=12)
    ax.set_title(f"230 个空间群的样本数量分布   (N = {total:,})",
                 fontsize=14, weight="bold", pad=14)

    # 绘制晶系分隔线和标签
    # 三斜晶系太窄（只有 2 个空间群），标签放在图表外部
    # 其他晶系标签放在图表内部
    header_tf = ax.get_xaxis_transform()
    y_inside = counts.max() * 5      # 内部标签的 y 位置
    y_outside = 1.015                # 外部标签的 y 位置（axes 分数）

    for name, lo, hi, cn in SYSTEMS:
        ax.axvline(hi + 0.5, color="gray", lw=0.6, ls=":", alpha=0.6, zorder=1)
        n_sys = int(((df["system"] == name).sum()))
        pct = 100.0 * n_sys / total
        label = f"{name}  {cn}\n{n_sys:,}  ({pct:.1f}%)"

        if name == "Triclinic":
            # 外部标签 + 引导线
            x_c = (lo + hi) / 2
            ax.text(x_c, y_outside, label,
                    ha="center", va="bottom", fontsize=9.5,
                    color=SYSTEM_COLORS[name], weight="bold",
                    transform=header_tf, clip_on=False, zorder=4)
            ax.plot([x_c, x_c], [1.0, y_outside - 0.003],
                    color=SYSTEM_COLORS[name], lw=0.6, ls=":", alpha=0.7,
                    transform=header_tf, clip_on=False, zorder=3)
        else:
            # 内部标签
            ax.text((lo + hi) / 2, y_inside, label,
                    ha="center", va="top", fontsize=9.5,
                    color=SYSTEM_COLORS[name], weight="bold", zorder=4)

    # 标注空间群符号
    for sg, sym in SG_SYMBOLS.items():
        c = int(counts.loc[sg])
        if c <= 0:
            continue
        ax.annotate(
            sym, xy=(sg, c), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
            color=SYSTEM_COLORS[sg_to_system(sg)], weight="bold", zorder=5,
        )

    # 网格线和边框
    ax.grid(axis="y", which="major", color="gray", lw=0.3, alpha=0.4, zorder=0)
    ax.grid(axis="y", which="minor", color="gray", lw=0.2, alpha=0.2, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    # 保存图表
    out = OUT / "01_space_group_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

    # ========== 图 2：晶系分布 ==========
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

    # 在柱子上标注数量和百分比
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

    # ========== 打印统计摘要 ==========
    print("\nCrystal system summary")
    print("-" * 60)
    summary = pd.DataFrame({
        "system": sys_counts.index,
        "count": sys_counts.values,
        "pct":   sys_pct.round(2).values,
    })
    print(summary.to_string(index=False))

    # Top-10 空间群
    top10 = counts.sort_values(ascending=False).head(10)
    print("\nTop-10 space groups")
    print("-" * 60)
    for sg, c in top10.items():
        print(f"  SG {sg:3d} ({sg_to_system(sg):12s})  {c:>8,}  {100*c/total:5.2f}%")

    # Bottom-10 非空空间群
    bot = counts[counts > 0].sort_values().head(10)
    print("\nBottom-10 non-empty space groups")
    print("-" * 60)
    for sg, c in bot.items():
        print(f"  SG {sg:3d} ({sg_to_system(sg):12s})  {c:>8,}")

    n_empty = int((counts == 0).sum())
    print(f"\nSpace groups with zero samples: {n_empty}/230")


if __name__ == "__main__":
    main()
