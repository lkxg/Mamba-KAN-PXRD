"""数据预处理脚本：将 SIMPOD JSON 文件转换为机器学习可用的格式。

功能说明：
- 从 Structures/Structures/ 目录读取所有 JSON 文件
- 提取关键字段：id（文件名）、space_group（空间群，1-230）、intensities（强度数据）
- 丢弃其他字段（晶胞参数、原子坐标等）

输出文件：
- dataset/pxrd.npy：内存映射文件，形状 (N, 10824)，float16 格式，约 10GB
- dataset/labels.csv：包含 row, id, space_group, crystal_system, crystal_system_id 列

技术细节：
- 使用多进程并行处理大量 JSON 文件
- float16 存储：[0,1] 范围内的数据使用 float16 可将文件大小减半
- 内存映射（memmap）方式写入，避免一次性占用大量内存
"""
import csv
import json
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 数据目录和输出目录路径
DATA_DIR = Path(r"d:/SimPOD/Structures/Structures")
OUT_DIR  = Path(r"d:/SimPOD/dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 常量定义
INTENSITY_LEN = 10824   # PXRD 强度数据的长度（2θ = 5°-90°，步长约 0.008°）
CHUNK_SIZE    = 500     # 每个 worker 处理的文件批次大小

# 晶系定义：名称、空间群范围下限、空间群范围上限
SYSTEMS = [
    ("Triclinic",    1,   2),     # 三斜晶系
    ("Monoclinic",   3,  15),    # 单斜晶系
    ("Orthorhombic", 16, 74),     # 正交晶系
    ("Tetragonal",   75, 142),    # 四方晶系
    ("Trigonal",     143, 167),   # 三方晶系
    ("Hexagonal",    168, 194),   # 六方晶系
    ("Cubic",        195, 230),   # 立方晶系
]
SYSTEM_TO_ID = {name: i for i, (name, *_) in enumerate(SYSTEMS)}


def sg_to_system(sg: int):
    """根据空间群编号返回对应的晶系名称和 ID。

    参数:
        sg: 空间群编号（1-230）

    返回:
        元组 (晶系名称, 晶系 ID)
        如果空间群不在有效范围内，返回 ("Invalid", -1)
    """
    for name, lo, hi in SYSTEMS:
        if lo <= sg <= hi:
            return name, SYSTEM_TO_ID[name]
    return "Invalid", -1


def parse_chunk(paths):
    """Worker 函数：解析一批 JSON 文件。

    参数:
        paths: JSON 文件路径列表

    返回:
        列表，每个元素为 (文件ID, 空间群编号, 强度数组或None)
    """
    out = []
    for p in paths:
        fid = Path(p).stem
        try:
            with open(p, "r") as f:
                d = json.load(f)
            sg = int(d.get("space_group", 0))
            inten = d.get("intensities")
            # 检查数据有效性：空间群必须在 1-230 范围内，且必须有强度数据
            if inten is None or not (1 <= sg <= 230):
                out.append((fid, sg, None))
                continue
            arr = np.asarray(inten, dtype=np.float32)
            # 检查强度数据长度是否正确
            if arr.shape != (INTENSITY_LEN,):
                out.append((fid, sg, None))
                continue
            out.append((fid, sg, arr.astype(np.float16)))
        except Exception:
            out.append((fid, -1, None))
    return out


def main():
    """主函数：执行数据预处理流程。"""
    t_start = time.time()

    # ---------- 步骤 1：列出所有 JSON 文件 ----------
    print(f"Listing files in {DATA_DIR} ...")
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix == ".json")
    n = len(files)
    print(f"  found {n:,} JSON files  ({time.time()-t_start:.1f}s)")

    # ---------- 步骤 2：分配输出内存映射文件 ----------
    npy_path = OUT_DIR / "pxrd.npy"
    n_bytes = n * INTENSITY_LEN * 2  # float16 占 2 字节
    print(f"\nAllocating output:")
    print(f"  {npy_path}")
    print(f"  shape=({n}, {INTENSITY_LEN}), dtype=float16  →  {n_bytes/1e9:.2f} GB")
    intensities = np.lib.format.open_memmap(
        str(npy_path), mode="w+",
        dtype=np.float16, shape=(n, INTENSITY_LEN),
    )

    # ---------- 步骤 3：分批并行处理文件 ----------
    # 构建批次：(文件路径列表, 起始行号)
    chunks = []
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunks.append(([str(files[i]) for i in range(start, end)], start))

    # 启动多进程处理
    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"\nDispatching {len(chunks)} chunks across {workers} workers ...")

    labels = [None] * n          # row → (id_str, sg, sys_name, sys_id)
    n_invalid = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_start = {
            ex.submit(parse_chunk, paths): start
            for paths, start in chunks
        }

        done_count = 0
        for fut in as_completed(future_to_start):
            start = future_to_start[fut]
            try:
                results = fut.result()
            except Exception as e:
                print(f"  WARNING: chunk starting at row {start} failed: {e}")
                results = []

            for offset, (fid, sg, arr16) in enumerate(results):
                row = start + offset
                if arr16 is None:
                    intensities[row] = 0  # 无效行填充为零
                    labels[row] = (fid, sg, "Invalid", -1)
                    n_invalid += 1
                    continue
                # 写入有效数据
                intensities[row] = arr16
                sys_name, sys_id = sg_to_system(sg)
                labels[row] = (fid, sg, sys_name, sys_id)

            done_count += 1
            # 打印进度
            if done_count % 25 == 0 or done_count == len(chunks):
                pct = 100 * done_count / len(chunks)
                elapsed = time.time() - t0
                eta = elapsed * (len(chunks) - done_count) / done_count if done_count else 0
                print(f"  chunks {done_count}/{len(chunks)} ({pct:5.1f}%)  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

    # 确保数据写入磁盘
    intensities.flush()
    del intensities  # 释放内存映射句柄

    # ---------- 步骤 4：写入标签 CSV ----------
    csv_path = OUT_DIR / "labels.csv"
    print(f"\nWriting {csv_path} ...")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "id", "space_group", "crystal_system", "crystal_system_id"])
        for row, lab in enumerate(labels):
            if lab is None:
                w.writerow([row, "", -1, "Invalid", -1])
                continue
            fid, sg, sn, sid = lab
            w.writerow([row, fid, sg, sn, sid])

    # ---------- 打印统计信息 ----------
    valid_rows = n - n_invalid
    print(f"\n{'='*60}")
    print(f"Done in {time.time()-t_start:.1f}s")
    print(f"  total rows : {n:,}")
    print(f"  valid rows : {valid_rows:,}")
    print(f"  invalid    : {n_invalid:,}")
    print(f"  npy size   : {n_bytes/1e9:.2f} GB")
    print(f"  output dir : {OUT_DIR}")
    print(f"{'='*60}")

    # ---------- 随机抽样验证数据质量 ----------
    print("\nSpot check (5 random rows):")
    arr = np.load(npy_path, mmap_mode="r")
    rng = np.random.default_rng(0)
    valid_idxs = [i for i, l in enumerate(labels) if l is not None and l[3] >= 0]
    for idx in rng.choice(valid_idxs, size=min(5, len(valid_idxs)), replace=False):
        fid, sg, sn, sid = labels[idx]
        row = arr[idx]
        # 统计峰值数量（强度大于最大值 5% 的局部最大值）
        n_peaks_above_5pct = int(((row > 0.05 * row.max()) & (row > 0)).sum())
        print(f"  row {idx:7d}: id={fid:>10s}  sg={sg:3d}  {sn:12s}  "
              f"max={float(row.max()):.4f}  peaks(>5%)≈{n_peaks_above_5pct}")


if __name__ == "__main__":
    main()
