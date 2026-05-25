"""PXRD 曲线可视化脚本。

生成两个图表：

1. 03_intensity_curves_per_system.png
   - 展示两个对称性极端的晶系的代表性 PXRD 曲线
   - Triclinic（三斜）：最低对称性，峰密集成群
   - Cubic（立方）：最高对称性，峰少而尖锐

2. 04_peak_count_distribution.png
   - 各晶系每条 PXRD 曲线的峰数分布箱线图
   - 峰定义：强度 > 5% 最大值的局部最大值
   - 峰数量是晶系/空间群分类的重要手工特征
"""
import json
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

# 设置中文字体
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# 路径配置
DATA = Path(r"d:/SimPOD/Structures/Structures")
CSV  = Path(r"d:/SimPOD/analysis/metadata.csv")
OUT  = Path(r"d:/SimPOD/analysis/plots")
OUT.mkdir(parents=True, exist_ok=True)

# 2θ 角度范围
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 90.0

# 晶系定义：名称、空间群下限、上限、中文名称、颜色
SYSTEMS = [
    ("Triclinic",    1,   2,  "三斜", "#e41a1c"),
    ("Monoclinic",   3,  15,  "单斜", "#ff7f00"),
    ("Orthorhombic", 16, 74,  "正交", "#d4b106"),
    ("Tetragonal",   75, 142, "四方", "#4daf4a"),
    ("Trigonal",     143,167, "三方", "#00b8d4"),
    ("Hexagonal",    168,194, "六方", "#377eb8"),
    ("Cubic",        195,230, "立方", "#984ea3"),
]

# 每个晶系用于箱线图的随机样本数
N_PEAKS_PER_SYSTEM  = 60

# 峰检测参数
PEAK_HEIGHT_FRAC = 0.05   # 峰必须达到该条曲线最大值的 5%
PEAK_MIN_DIST    = 3      # 峰之间的最小距离（采样点）


def sg_to_system(sg: int) -> str:
    """根据空间群编号返回晶系名称。"""
    for name, lo, hi, *_ in SYSTEMS:
        if lo <= sg <= hi:
            return name
    return "Invalid"


def load_intensities(file_id: str) -> np.ndarray:
    """从 JSON 文件加载 PXRD 强度数据。

    参数:
        file_id: 文件 ID（不含扩展名）

    返回:
        强度数组，形状 (10824,)
    """
    with open(DATA / f"{file_id}.json", "r") as f:
        d = json.load(f)
    return np.asarray(d["intensities"], dtype=np.float64)


def main():
    """主函数：生成可视化图表。"""
    random.seed(0)
    np.random.seed(0)

    # 加载元数据
    df = pd.read_csv(CSV)
    df = df[(df["space_group"] >= 1) & (df["space_group"] <= 230)].copy()
    df["system"] = df["space_group"].map(sg_to_system)

    # =============================================================
    # 图 03：两个极端对称性晶系的 PXRD 曲线
    # =============================================================
    print("Loading curves for figure 03 ...")
    extreme_names = ("Triclinic", "Cubic")
    extreme_picks = []
    for name, lo, hi, cn, color in SYSTEMS:
        if name not in extreme_names:
            continue
        sub = df[df["system"] == name]
        if len(sub) == 0:
            continue
        # 随机选择一个样本
        row = sub.sample(1, random_state=0).iloc[0]
        fid = str(row["ID"])
        sg = int(row["space_group"])
        y = load_intensities(fid)
        # 生成 2θ 角度数组
        x = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, len(y))
        extreme_picks.append((name, cn, color, fid, sg, x, y))
        print(f"  {name:12s}  ID={fid}  SG={sg}")

    # 绘制图表
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
    # 图 04：各晶系峰数分布箱线图
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
                # 峰检测：局部最大值且强度 > 5% 最大值
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

    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    box_data = [peak_counts[n] for n, *_ in SYSTEMS]
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

    # 在箱体上方标注中位数
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
