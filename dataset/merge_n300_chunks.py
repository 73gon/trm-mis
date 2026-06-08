#!/usr/bin/env python3
"""Merge data/smallmis_n300_multi/{train_chunk_*,test_chunk_*}/mis_shard_*.pt
into data/smallmis_n300_multi/{train,test}/mis_shard_<idx>.pt with sequential
indices. Concatenates the `data` lists across chunks; metadata is taken from
the first chunk and `cfg.num_instances` is replaced with the total.
"""
import glob
import os
import sys
import torch

ROOT = "data/smallmis_n300_multi"


def merge(split: str) -> int:
    out_dir = os.path.join(ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    chunk_dirs = sorted(glob.glob(os.path.join(ROOT, f"{split}_chunk_*")))
    if not chunk_dirs:
        print(f"[merge] no chunks for {split}", file=sys.stderr)
        return 0

    shards = []
    for cd in chunk_dirs:
        shards.extend(sorted(glob.glob(os.path.join(cd, "mis_shard_*.pt"))))
    print(f"[merge] {split}: {len(shards)} input shards from {len(chunk_dirs)} chunks")

    out_idx = 0
    total_graphs = 0
    first_meta = None
    for shard_path in shards:
        blob = torch.load(shard_path, weights_only=False)
        meta = blob["meta"]
        data = blob["data"]
        if first_meta is None:
            first_meta = dict(meta)
        out_path = os.path.join(out_dir, f"mis_shard_{out_idx:04d}.pt")
        torch.save({"meta": meta, "data": data}, out_path)
        out_idx += 1
        total_graphs += len(data)
    print(f"[merge] {split}: wrote {out_idx} shards, {total_graphs} graphs to {out_dir}")
    return total_graphs


if __name__ == "__main__":
    n_train = merge("train")
    n_test = merge("test")
    print(f"[merge] DONE. train={n_train} test={n_test}")
