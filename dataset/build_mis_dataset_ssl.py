"""
Build MIS Dataset for Self-Supervised Learning (No Labels Required)

This script generates graph instances for self-supervised MIS training.
Unlike the supervised version, it does NOT compute optimal MIS labels,
making it much faster and not requiring Gurobi.

The output format is compatible with MISDataset but without ground truth labels.
For evaluation against optimal solutions, use the original build_mis_dataset.py.

Usage:
    python dataset/build_mis_dataset_ssl.py --num_instances 50000 --output_dir data/mis-ssl-50k
"""

import os
import random
import time

import networkx as nx
import numpy as np
import torch
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

cli = ArgParser()


class MISSSLProcessConfig(BaseModel):
    # output
    output_dir: str = "data/mis-ssl-10k"
    shard_size: int = 250

    # dataset size / seeds
    num_instances: int = 10_000
    seed_start: int = 0

    # Erdos-Renyi distribution via expected degree d ~ p*(n-1)
    n_min: int = 50
    n_max: int = 250
    d_min: float = 6.0
    d_max: float = 14.0

    # node features
    use_degree_feature: bool = True  # x = [1, degree_norm] else x = [1]

    # logging
    log_every: int = 500

    # For compatibility with supervised dataset - set dummy labels
    # When True, y will be all zeros (placeholder)
    include_dummy_labels: bool = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_erdos_renyi_graph(seed: int, cfg: MISSSLProcessConfig):
    rng = random.Random(seed)

    n = rng.randint(cfg.n_min, cfg.n_max)
    d = rng.uniform(cfg.d_min, cfg.d_max)
    p = max(0.0, min(1.0, d / (n - 1)))

    G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**31 - 1))
    return G, n, p, d


def nx_to_edge_index(G: nx.Graph):
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges_rev = edges[:, ::-1]
    edges_all = np.vstack([edges, edges_rev])
    return torch.from_numpy(edges_all.T).long()


def make_node_features(G: nx.Graph, cfg: MISSSLProcessConfig):
    n = G.number_of_nodes()
    ones = np.ones(n, dtype=np.float32)

    if not cfg.use_degree_feature:
        x = ones[:, None]  # [n, 1]
        return torch.from_numpy(x)

    deg = np.array([G.degree(i) for i in range(n)], dtype=np.float32)
    deg_norm = deg / max(1.0, (n - 1))
    x = np.stack([ones, deg_norm], axis=1)  # [n, 2]
    return torch.from_numpy(x)


def save_shard(cfg: MISSSLProcessConfig, shard_idx: int, shard: list[dict], meta: dict):
    path = os.path.join(cfg.output_dir, f"mis_shard_{shard_idx:04d}.pt")
    torch.save({"meta": meta, "data": shard}, path)
    return path


@cli.command(singleton=True)
def preprocess_data(cfg: MISSSLProcessConfig):
    ensure_dir(cfg.output_dir)

    meta = {
        "cfg": cfg.model_dump(),
        "created_unix": time.time(),
        "format": "dict(meta, data=list_of_samples)",
        "training_mode": "self-supervised",
        "fields": {
            "x": "FloatTensor[n,1 or 2] (ones[, degree_norm])",
            "edge_index": "LongTensor[2, 2|E|] undirected both directions",
            "y": "LongTensor[n] - dummy zeros for SSL (not used in training)",
            "opt_value": "int - dummy 0 for SSL (not computed)",
            "n,p,d_target,num_edges,seed": "scalars",
        },
        "note": "Labels are dummy values (zeros). For evaluation with ground truth, use build_mis_dataset.py",
    }

    shard: list[dict] = []
    shard_idx = 0

    # timing stats
    gen_times = []
    total_t0 = time.time()

    pbar = tqdm(range(cfg.num_instances), desc="Generating MIS SSL dataset", dynamic_ncols=True)

    for i in pbar:
        seed = cfg.seed_start + i

        t_gen0 = time.perf_counter()
        G, n, p, d = generate_erdos_renyi_graph(seed, cfg)
        t_gen1 = time.perf_counter()

        gen_times.append(t_gen1 - t_gen0)

        # Create dummy labels (zeros) - not used in SSL training
        if cfg.include_dummy_labels:
            y = torch.zeros(n, dtype=torch.long)
            opt_value = 0
        else:
            y = None
            opt_value = None

        sample = {
            "x": make_node_features(G, cfg),
            "edge_index": nx_to_edge_index(G),
            "y": y,
            "opt_value": opt_value,
            "n": n,
            "p": p,
            "d_target": d,
            "num_edges": G.number_of_edges(),
            "seed": seed,
        }
        shard.append(sample)

        # update tqdm stats occasionally
        if (i + 1) % cfg.log_every == 0:
            avg_gen = float(np.mean(gen_times[-cfg.log_every :]))
            elapsed = time.time() - total_t0
            rate = (i + 1) / max(elapsed, 1e-9)
            pbar.set_postfix(
                {
                    "inst/s": f"{rate:.2f}",
                    "gen_ms": f"{avg_gen * 1000:.2f}",
                    "last_n": n,
                    "last_E": G.number_of_edges(),
                }
            )

        # shard save
        if len(shard) >= cfg.shard_size:
            out_path = save_shard(cfg, shard_idx, shard, meta)
            shard = []
            shard_idx += 1
            pbar.write(f"Saved {out_path}")

    if shard:
        out_path = save_shard(cfg, shard_idx, shard, meta)
        pbar.write(f"Saved {out_path}")

    # final summary
    total_elapsed = time.time() - total_t0
    print("\nDone (Self-Supervised Dataset - No Labels).")
    print(f"Total instances: {cfg.num_instances}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Avg gen time: {np.mean(gen_times) * 1000:.2f} ms")
    print(f"Output dir: {cfg.output_dir}")
    print("\nNote: This dataset has dummy labels. Use original build_mis_dataset.py for evaluation data.")


if __name__ == "__main__":
    cli()
