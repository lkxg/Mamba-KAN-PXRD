"""快速并行扫描脚本：从所有 JSON 文件中提取 ID 和空间群信息。

功能说明：
- 使用正则表达式仅读取每个文件的前 256 字节，避免解析完整的 JSON
- 多进程并行处理大量文件
- 输出 CSV 文件包含 ID 和 space_group 两列

技术优势：
- 比完整 JSON 解析快 10 倍以上
- 内存占用极低
- 可处理数十万个文件
"""
import os
import re
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(r"d:/SimPOD/Structures/Structures")
OUT_CSV = Path(r"d:/SimPOD/analysis/metadata.csv")

# 正则表达式：匹配 "space_group": 数字 模式
SG_RE = re.compile(rb'"space_group"\s*:\s*(\d+)')


def scan_one(path: str):
    """扫描单个文件，提取空间群编号。

    参数:
        path: JSON 文件路径

    返回:
        元组 (文件ID, 空间群编号)，无效时空间群编号为 -1
    """
    try:
        with open(path, "rb") as f:
            head = f.read(256)  # 只读取文件开头
        m = SG_RE.search(head)
        if not m:
            return (Path(path).stem, -1)
        return (Path(path).stem, int(m.group(1)))
    except Exception:
        return (Path(path).stem, -1)


def scan_chunk(paths):
    """扫描一批文件。

    参数:
        paths: 文件路径列表

    返回:
        扫描结果列表
    """
    return [scan_one(p) for p in paths]


def chunked(seq, n):
    """将序列分块。

    参数:
        seq: 输入序列
        n: 每块大小

    返回:
        分块迭代器
    """
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    """主函数：执行并行扫描。"""
    t0 = time.time()

    # 列出所有 JSON 文件
    print(f"Listing files in {DATA_DIR} ...")
    files = [str(p) for p in DATA_DIR.iterdir() if p.suffix == ".json"]
    print(f"Found {len(files):,} files in {time.time()-t0:.1f}s")

    # 准备并行扫描
    t0 = time.time()
    rows = []
    chunk_size = 2000  # 每块 2000 个文件
    chunks = list(chunked(files, chunk_size))
    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Scanning with {workers} workers, {len(chunks)} chunks ...")

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(scan_chunk, c) for c in chunks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 25 == 0 or done == len(chunks):
                pct = 100 * done / len(chunks)
                print(f"  {done}/{len(chunks)} chunks ({pct:.1f}%) in {time.time()-t0:.1f}s")

    print(f"Scanned {len(rows):,} files in {time.time()-t0:.1f}s")

    # 写入 CSV 文件
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # 按文件 ID 排序（数字部分）
    rows.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 0)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "space_group"])
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}")

    # 统计无效条目
    bad = sum(1 for _, sg in rows if sg < 1 or sg > 230)
    print(f"Invalid/missing space_group rows: {bad}")


if __name__ == "__main__":
    main()
