"""
Precompute Laplacian PE + enhanced node features for an entire shard directory.

Writes `<shard>.cache.pt` next to each `mis_shard_XXXX.pt`, containing graphs
with `x` (12-dim) and `pe` (16-dim) fields already populated.

Usage
-----
python -m experiments.preprocess_features \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --pe_dim 16 --workers 8

python -m experiments.preprocess_features \
    --data_path data/difusco_benchmark/datasets/er_700_800/train \
    --pe_dim 16 --workers 8

Rationale
---------
MISDataset streams features on-the-fly for >3000-graph datasets, causing
~92% CPU idle and ~8% GPU utilisation. Running this script once (~tens of
minutes) lets subsequent training runs skip the expensive eigendecomposition
and NetworkX shortest-path work entirely.

Idempotent: skips shards whose cache already exists and is newer than the
source shard. Delete `*.cache.pt` to force rebuild.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import time
from functools import partial

import torch

from dataset.mis_dataset import compute_laplacian_pe, compute_node_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Directory containing mis_shard_*.pt files.")
    p.add_argument("--pe_dim", type=int, default=16)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel graphs processed per shard via multiprocessing.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache exists.")
    return p.parse_args()


def _enhance_one(sample: dict, pe_dim: int) -> dict:
    out = dict(sample)
    out["x"] = compute_node_features(sample["edge_index"], sample["n"], sample["x"])
    out["pe"] = compute_laplacian_pe(sample["edge_index"], sample["n"], pe_dim=pe_dim)
    return out


def process_shard(shard_path: str, pe_dim: int, workers: int, force: bool) -> tuple[str, float, int]:
    cache_path = shard_path.replace(".pt", ".cache.pt")
    if os.path.exists(cache_path) and not force:
        src_mtime = os.path.getmtime(shard_path)
        dst_mtime = os.path.getmtime(cache_path)
        if dst_mtime >= src_mtime:
            return shard_path, 0.0, -1  # skipped

    t0 = time.time()
    payload = torch.load(shard_path, weights_only=False)
    graphs = payload["data"]

    if workers > 1 and len(graphs) > 1:
        with mp.Pool(workers) as pool:
            enhanced = pool.map(partial(_enhance_one, pe_dim=pe_dim), graphs)
    else:
        enhanced = [_enhance_one(g, pe_dim) for g in graphs]

    new_payload = dict(payload)
    new_payload["data"] = enhanced
    new_payload["precomputed_features"] = True
    new_payload["pe_dim"] = pe_dim

    tmp_path = cache_path + ".tmp"
    torch.save(new_payload, tmp_path)
    os.replace(tmp_path, cache_path)
    return shard_path, time.time() - t0, len(enhanced)


def main() -> None:
    args = parse_args()
    shard_paths = sorted(glob.glob(os.path.join(args.data_path, "mis_shard_*.pt")))
    # Exclude any existing cache files
    shard_paths = [p for p in shard_paths if not p.endswith(".cache.pt")]
    if not shard_paths:
        raise FileNotFoundError(f"No mis_shard_*.pt in {args.data_path}")

    print(f"[preprocess] {len(shard_paths)} shards in {args.data_path}")
    print(f"[preprocess] pe_dim={args.pe_dim}, workers={args.workers}, force={args.force}")

    t_start = time.time()
    total_graphs = 0
    skipped = 0
    for i, sp in enumerate(shard_paths, 1):
        sp, dt, n = process_shard(sp, args.pe_dim, args.workers, args.force)
        if n < 0:
            skipped += 1
            print(f"  [{i}/{len(shard_paths)}] {os.path.basename(sp)}: cached, skipped")
        else:
            total_graphs += n
            rate = n / max(1e-6, dt)
            print(f"  [{i}/{len(shard_paths)}] {os.path.basename(sp)}: "
                  f"{n} graphs in {dt:.1f}s ({rate:.2f} g/s)")
    total = time.time() - t_start
    print(f"[preprocess] done. {total_graphs} graphs written, {skipped} skipped, "
          f"{total:.1f}s total")


if __name__ == "__main__":
    main()
