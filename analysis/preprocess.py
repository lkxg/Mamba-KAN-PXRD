"""Preprocess SIMPOD JSON files into a compact (npy + CSV) ML-ready format.

For each crystal in `Structures/Structures/*.json` we keep ONLY the three
fields that matter for crystal-system / space-group classification:

    - id              (from the filename)
    - space_group     (1 .. 230)
    - intensities     (length 10824, already normalized to [0,1] per paper)

and discard cell parameters, atom types/coordinates, etc.

Outputs (in `dataset/`):

    intensities.npy   memmap, shape (N, 10824), dtype float16  (~10 GB)
    labels.csv        columns: row, id, space_group, crystal_system,
                               crystal_system_id

Row order in `intensities.npy` matches the `row` column in `labels.csv`,
so they can be loaded together with:

    X      = np.load("dataset/intensities.npy", mmap_mode="r")
    labels = pd.read_csv("dataset/labels.csv")

float16 is intentional — the data is already in [0, 1] and a 10⁻³ resolution
is far below any meaningful peak-height precision; halves the disk size with
no measurable impact on classification accuracy.
"""
import csv
import json
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---- Paths --------------------------------------------------------------
DATA_DIR = Path(r"d:/SimPOD/Structures/Structures")
OUT_DIR  = Path(r"d:/SimPOD/dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants ----------------------------------------------------------
INTENSITY_LEN = 10824   # paper spec (2θ = 5°–90°, step ≈ 0.008°)
CHUNK_SIZE    = 500     # files per worker batch (balances IO and pickling)

SYSTEMS = [
    ("Triclinic",    1,   2),
    ("Monoclinic",   3,  15),
    ("Orthorhombic", 16, 74),
    ("Tetragonal",   75, 142),
    ("Trigonal",     143, 167),
    ("Hexagonal",    168, 194),
    ("Cubic",        195, 230),
]
SYSTEM_TO_ID = {name: i for i, (name, *_) in enumerate(SYSTEMS)}


def sg_to_system(sg: int):
    """Return (system_name, system_id 0..6)."""
    for name, lo, hi in SYSTEMS:
        if lo <= sg <= hi:
            return name, SYSTEM_TO_ID[name]
    return "Invalid", -1


def parse_chunk(paths):
    """Worker: parse a list of JSON paths. Returns list of
    (file_id_str, space_group_int, intensities_float16_or_None)."""
    out = []
    for p in paths:
        fid = Path(p).stem
        try:
            with open(p, "r") as f:
                d = json.load(f)
            sg = int(d.get("space_group", 0))
            inten = d.get("intensities")
            if inten is None or not (1 <= sg <= 230):
                out.append((fid, sg, None))
                continue
            arr = np.asarray(inten, dtype=np.float32)
            if arr.shape != (INTENSITY_LEN,):
                out.append((fid, sg, None))
                continue
            out.append((fid, sg, arr.astype(np.float16)))
        except Exception:
            out.append((fid, -1, None))
    return out


def main():
    t_start = time.time()
    print(f"Listing files in {DATA_DIR} ...")
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix == ".json")
    n = len(files)
    print(f"  found {n:,} JSON files  ({time.time()-t_start:.1f}s)")

    # Open output memmap (allocates the file on disk; won't blow up RAM)
    npy_path = OUT_DIR / "intensities.npy"
    n_bytes = n * INTENSITY_LEN * 2  # float16
    print(f"\nAllocating output:")
    print(f"  {npy_path}")
    print(f"  shape=({n}, {INTENSITY_LEN}), dtype=float16  →  {n_bytes/1e9:.2f} GB")
    intensities = np.lib.format.open_memmap(
        str(npy_path), mode="w+",
        dtype=np.float16, shape=(n, INTENSITY_LEN),
    )

    # Build (paths, start_row) chunks
    chunks = []
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunks.append(([str(files[i]) for i in range(start, end)], start))

    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"\nDispatching {len(chunks)} chunks across {workers} workers ...")

    labels = [None] * n          # row → (id_str, sg, sys_name, sys_id)
    n_invalid = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # submit() returns futures; map start index for each
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
                    intensities[row] = 0  # zero out invalid row
                    labels[row] = (fid, sg, "Invalid", -1)
                    n_invalid += 1
                    continue
                intensities[row] = arr16
                sys_name, sys_id = sg_to_system(sg)
                labels[row] = (fid, sg, sys_name, sys_id)

            done_count += 1
            if done_count % 25 == 0 or done_count == len(chunks):
                pct = 100 * done_count / len(chunks)
                elapsed = time.time() - t0
                eta = elapsed * (len(chunks) - done_count) / done_count if done_count else 0
                print(f"  chunks {done_count}/{len(chunks)} ({pct:5.1f}%)  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

    intensities.flush()
    del intensities  # release memmap handle

    # Write labels CSV in row order
    csv_path = OUT_DIR / "labels.csv"
    print(f"\nWriting {csv_path} ...")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "id", "space_group", "crystal_system", "crystal_system_id"])
        for row, lab in enumerate(labels):
            if lab is None:
                # File missing or scan failed — should be rare; record anyway
                w.writerow([row, "", -1, "Invalid", -1])
                continue
            fid, sg, sn, sid = lab
            w.writerow([row, fid, sg, sn, sid])

    # ---- Summary ----------------------------------------------------------
    valid_rows = n - n_invalid
    print(f"\n{'='*60}")
    print(f"Done in {time.time()-t_start:.1f}s")
    print(f"  total rows : {n:,}")
    print(f"  valid rows : {valid_rows:,}")
    print(f"  invalid    : {n_invalid:,}")
    print(f"  npy size   : {n_bytes/1e9:.2f} GB")
    print(f"  output dir : {OUT_DIR}")
    print(f"{'='*60}")

    # ---- Quick sanity check on 5 random valid rows -----------------------
    print("\nSpot check (5 random rows):")
    arr = np.load(npy_path, mmap_mode="r")
    rng = np.random.default_rng(0)
    valid_idxs = [i for i, l in enumerate(labels) if l is not None and l[3] >= 0]
    for idx in rng.choice(valid_idxs, size=min(5, len(valid_idxs)), replace=False):
        fid, sg, sn, sid = labels[idx]
        row = arr[idx]
        n_peaks_above_5pct = int(((row > 0.05 * row.max()) & (row > 0)).sum())
        print(f"  row {idx:7d}: id={fid:>10s}  sg={sg:3d}  {sn:12s}  "
              f"max={float(row.max()):.4f}  peaks(>5%)≈{n_peaks_above_5pct}")


if __name__ == "__main__":
    main()
