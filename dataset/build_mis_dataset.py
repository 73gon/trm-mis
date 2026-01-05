from typing import Optional
import os
import time
import random

import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser

from gurobi_optimods.mwis import maximum_weighted_independent_set


cli = ArgParser()


class MISProcessConfig(BaseModel):
    # output
    output_dir: str = "data/mis-10k"
    shard_size: int = 250  # 10k -> 40 shards

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
    log_every: int = 250  # update postfix every N instances

    # safety / debug
    assert_valid_labels: bool = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_erdos_renyi_graph(seed: int, cfg: MISProcessConfig):
    rng = random.Random(seed)

    n = rng.randint(cfg.n_min, cfg.n_max)
    d = rng.uniform(cfg.d_min, cfg.d_max)
    p = max(0.0, min(1.0, d / (n - 1)))

    G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**31 - 1))
    return G, n, p, d


def label_with_gurobi_mis(G: nx.Graph):
    n = G.number_of_nodes()
    weights = np.ones(n, dtype=float)

    res = maximum_weighted_independent_set(G, weights)

    y = np.zeros(n, dtype=np.int64)
    y[np.array(res.x, dtype=int)] = 1
    opt_value = int(res.f)
    return y, opt_value


def nx_to_edge_index(G: nx.Graph):
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges_rev = edges[:, ::-1]
    edges_all = np.vstack([edges, edges_rev])
    return torch.from_numpy(edges_all.T).long()


def make_node_features(G: nx.Graph, cfg: MISProcessConfig):
    n = G.number_of_nodes()
    ones = np.ones(n, dtype=np.float32)

    if not cfg.use_degree_feature:
        x = ones[:, None]  # [n, 1]
        return torch.from_numpy(x)

    deg = np.array([G.degree(i) for i in range(n)], dtype=np.float32)
    deg_norm = deg / max(1.0, (n - 1))
    x = np.stack([ones, deg_norm], axis=1)  # [n, 2]
    return torch.from_numpy(x)


def check_independent_set(G: nx.Graph, y: np.ndarray) -> bool:
    for u, v in G.edges():
        if y[u] == 1 and y[v] == 1:
            return False
    return True


def save_shard(cfg: MISProcessConfig, shard_idx: int, shard: list[dict], meta: dict):
    path = os.path.join(cfg.output_dir, f"mis_shard_{shard_idx:04d}.pt")
    torch.save({"meta": meta, "data": shard}, path)
    return path


@cli.command(singleton=True)
def preprocess_data(cfg: MISProcessConfig):
    ensure_dir(cfg.output_dir)

    meta = {
        "cfg": cfg.model_dump(),
        "created_unix": time.time(),
        "format": "dict(meta, data=list_of_samples)",
        "fields": {
            "x": "FloatTensor[n,1 or 2] (ones[, degree_norm])",
            "edge_index": "LongTensor[2, 2|E|] undirected both directions",
            "y": "LongTensor[n] in {0,1}",
            "opt_value": "int (MIS size)",
            "n,p,d_target,num_edges,seed": "scalars",
        },
    }

    shard: list[dict] = []
    shard_idx = 0

    # timing stats
    gen_times = []
    solve_times = []
    total_t0 = time.time()

    pbar = tqdm(range(cfg.num_instances), desc="Generating MIS dataset", dynamic_ncols=True)

    for i in pbar:
        seed = cfg.seed_start + i

        t_gen0 = time.perf_counter()
        G, n, p, d = generate_erdos_renyi_graph(seed, cfg)
        t_gen1 = time.perf_counter()

        t_sol0 = time.perf_counter()
        y_np, opt_value = label_with_gurobi_mis(G)
        t_sol1 = time.perf_counter()

        gen_times.append(t_gen1 - t_gen0)
        solve_times.append(t_sol1 - t_sol0)

        if cfg.assert_valid_labels:
            if not (int(y_np.sum()) == opt_value and check_independent_set(G, y_np)):
                raise RuntimeError(f"Invalid label at i={i}, seed={seed}")

        sample = {
            "x": make_node_features(G, cfg),
            "edge_index": nx_to_edge_index(G),
            "y": torch.from_numpy(y_np).long(),
            "opt_value": opt_value,
            "n": n,
            "p": p,
            "d_target": d,
            "num_edges": G.number_of_edges(),
            "seed": seed,
        }
        shard.append(sample)

        # update tqdm stats occasionally (keeps it fast)
        if (i + 1) % cfg.log_every == 0:
            avg_gen = float(np.mean(gen_times[-cfg.log_every:]))
            avg_sol = float(np.mean(solve_times[-cfg.log_every:]))
            elapsed = time.time() - total_t0
            rate = (i + 1) / max(elapsed, 1e-9)
            pbar.set_postfix({
                "inst/s": f"{rate:.2f}",
                "gen_ms": f"{avg_gen*1000:.1f}",
                "solve_ms": f"{avg_sol*1000:.1f}",
                "last_n": n,
                "last_p": f"{p:.4f}",
                "last_E": G.number_of_edges(),
            })

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
    print("\nDone.")
    print(f"Total instances: {cfg.num_instances}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Avg gen time: {np.mean(gen_times)*1000:.2f} ms")
    print(f"Avg solve time: {np.mean(solve_times)*1000:.2f} ms")
    print(f"Output dir: {cfg.output_dir}")


if __name__ == "__main__":
    cli()
