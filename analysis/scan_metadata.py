"""Fast parallel scanner: extract (ID, space_group) from every JSON file.

Reads only the first ~200 bytes of each file with a regex to avoid full
JSON parsing of the huge intensities array.
"""
import os
import re
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(r"d:/SimPOD/Structures/Structures")
OUT_CSV = Path(r"d:/SimPOD/analysis/metadata.csv")

SG_RE = re.compile(rb'"space_group"\s*:\s*(\d+)')


def scan_one(path: str):
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        m = SG_RE.search(head)
        if not m:
            return (Path(path).stem, -1)
        return (Path(path).stem, int(m.group(1)))
    except Exception:
        return (Path(path).stem, -1)


def scan_chunk(paths):
    return [scan_one(p) for p in paths]


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    t0 = time.time()
    print(f"Listing files in {DATA_DIR} ...")
    files = [str(p) for p in DATA_DIR.iterdir() if p.suffix == ".json"]
    print(f"Found {len(files):,} files in {time.time()-t0:.1f}s")

    t0 = time.time()
    rows = []
    chunk_size = 2000
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

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 0)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "space_group"])
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}")

    bad = sum(1 for _, sg in rows if sg < 1 or sg > 230)
    print(f"Invalid/missing space_group rows: {bad}")


if __name__ == "__main__":
    main()
