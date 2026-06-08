"""Compute MIS label distribution (1/0 ratio) per graph for the difusco
benchmark shards.

Aggregates per-split (mean/std/min/p10/p50/p90/max + global pos_frac) and
also reports the same stats for the first {1, 250, 1500, 9000, 25000} graphs
(the loading order overfit_sl uses), so we can see whether smaller-N runs
see a biased subsample.

Usage:
    python -m dataset.compute_label_ratio
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np
import torch

ROOT = "data/difusco_benchmark/datasets"
SPLITS = [
    ("satlib", "train"),
    ("satlib", "test"),
    ("er_700_800", "train"),
    ("er_700_800", "test"),
]
SUBSET_SIZES = [1, 250, 1500, 9000, 25000]


def iter_graphs(data_path: str):
    shard_paths = sorted(glob.glob(os.path.join(data_path, "mis_shard_*.pt")))
    shard_paths = [p for p in shard_paths if not p.endswith(".cache.pt")]
    for sp in shard_paths:
        cp = sp.replace(".pt", ".cache.pt")
        path = cp if (os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(sp)) else sp
        payload = torch.load(path, weights_only=False)
        for g in payload["data"]:
            yield g


def graph_pos_n(g: dict[str, Any]) -> tuple[int, int]:
    y = g["y"]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = np.asarray(y).reshape(-1).astype(np.float64)
    n = int(g.get("n", y.shape[0]))
    return int(y.sum()), n


def stats_block(items: list[tuple[int, int]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    fracs = np.array([p / n for p, n in items], dtype=np.float64)
    pos_sum = int(sum(p for p, _ in items))
    node_sum = int(sum(n for _, n in items))
    return {
        "count": len(items),
        "pos_frac_global": pos_sum / node_sum,
        "node_sum": node_sum,
        "pos_sum": pos_sum,
        "mean": float(fracs.mean()),
        "std": float(fracs.std()),
        "min": float(fracs.min()),
        "p10": float(np.percentile(fracs, 10)),
        "p50": float(np.percentile(fracs, 50)),
        "p90": float(np.percentile(fracs, 90)),
        "max": float(fracs.max()),
    }


def compute_split(dataset: str, split: str) -> dict[str, Any]:
    data_path = os.path.join(ROOT, dataset, split)
    items: list[tuple[int, int]] = []
    for g in iter_graphs(data_path):
        items.append(graph_pos_n(g))
    full = stats_block(items)
    subsets = {k: stats_block(items[:k]) for k in SUBSET_SIZES if k <= len(items)}
    return {"full": full, "subsets": subsets}


def fmt_block(b: dict[str, Any]) -> str:
    if b.get("count", 0) == 0:
        return "(empty)"
    return (
        f"N={b['count']:>6d} "
        f"global_pos_frac={b['pos_frac_global']:.4f} "
        f"per_graph mean={b['mean']:.4f} std={b['std']:.4f} "
        f"min={b['min']:.4f} p10={b['p10']:.4f} p50={b['p50']:.4f} "
        f"p90={b['p90']:.4f} max={b['max']:.4f}"
    )


def main() -> None:
    out: dict[str, Any] = {}
    for dataset, split in SPLITS:
        key = f"{dataset}/{split}"
        print(f"\n=== {key} ===", flush=True)
        try:
            r = compute_split(dataset, split)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        out[key] = r
        print(f"FULL : {fmt_block(r['full'])}")
        for k in sorted(r["subsets"]):
            print(f"N={k:<6d} : {fmt_block(r['subsets'][k])}")
    os.makedirs("logs", exist_ok=True)
    with open("logs/label_ratio.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved logs/label_ratio.json")


if __name__ == "__main__":
    main()
