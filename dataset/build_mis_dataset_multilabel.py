"""Build MIS dataset with multi-MIS soft labels.

Each graph is solved with Gurobi in solution pool mode to collect up to K
maximum-cardinality independent sets. Each node receives a soft label equal to
the fraction of returned optimal MISs containing it. This removes the
single-label arbitrariness that puts a Bayes-error floor on BCE.

CLI mirrors build_mis_dataset.py (single-graph-size variant) with extra flags:
    --pool_size        max optimal MISs to collect (default 16)
    --pool_gap         relative gap from optimum to accept (0.0 = strict opt)
    --pool_time_limit  per-graph Gurobi time limit in seconds
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser

import gurobipy as gp

cli = ArgParser()


class Cfg(BaseModel):
    output_dir: str = "data/smallmis_n50_multi/train"
    shard_size: int = 250

    num_instances: int = 5000
    seed_start: int = 1_000_000  # avoid collision with single-label datasets

    n_min: int = 50
    n_max: int = 50
    d_min: float = 7.35
    d_max: float = 7.35

    pool_size: int = 16
    pool_gap: float = 0.0  # 0 = only strictly optimal solutions
    pool_time_limit: float = 30.0
    threads: int = 1

    use_degree_feature: bool = True
    log_every: int = 50
    assert_valid_labels: bool = True


def _silent_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()
    return env


def _gen_graph(seed: int, cfg: Cfg):
    rng = np.random.RandomState(seed)
    n = int(rng.randint(cfg.n_min, cfg.n_max + 1))
    d = float(rng.uniform(cfg.d_min, cfg.d_max))
    p = max(0.0, min(1.0, d / max(1, n - 1)))
    G = nx.gnp_random_graph(n, p, seed=int(rng.randint(0, 2**31 - 1)))
    return G, n, p, d


def _solve_mis_pool(G: nx.Graph, env: gp.Env, cfg: Cfg):
    """Return (soft_y[n] in [0,1], opt_value, num_solutions)."""
    n = G.number_of_nodes()
    m = gp.Model("mis", env=env)
    m.Params.Threads = cfg.threads
    m.Params.PoolSearchMode = 2  # systematic search for n-best
    m.Params.PoolSolutions = cfg.pool_size
    m.Params.PoolGap = cfg.pool_gap
    m.Params.TimeLimit = cfg.pool_time_limit

    x = m.addVars(n, vtype=gp.GRB.BINARY, name="x")
    for u, v in G.edges():
        if u != v:
            m.addConstr(x[u] + x[v] <= 1)
    m.setObjective(gp.quicksum(x[i] for i in range(n)), gp.GRB.MAXIMIZE)
    m.optimize()

    if m.SolCount == 0:
        raise RuntimeError("Gurobi returned no solutions")

    opt_value = int(round(m.ObjVal))
    # collect only solutions matching the optimum (when PoolGap=0 Gurobi
    # already enforces this, but cheap to double-check)
    soft = np.zeros(n, dtype=np.float32)
    kept = 0
    for k in range(m.SolCount):
        m.Params.SolutionNumber = k
        if int(round(m.PoolObjVal)) != opt_value:
            continue
        sol = np.array([x[i].Xn for i in range(n)], dtype=np.float32)
        sol = (sol > 0.5).astype(np.float32)
        soft += sol
        kept += 1
    if kept == 0:
        raise RuntimeError("No optimal solutions retained from pool")
    soft /= kept
    return soft, opt_value, kept


def _edge_index(G: nx.Graph) -> torch.Tensor:
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    both = np.vstack([edges, edges[:, ::-1]])
    return torch.from_numpy(both.T).long()


def _node_features(G: nx.Graph, cfg: Cfg) -> torch.Tensor:
    n = G.number_of_nodes()
    ones = np.ones(n, dtype=np.float32)
    if not cfg.use_degree_feature:
        return torch.from_numpy(ones[:, None])
    deg = np.array([G.degree(i) for i in range(n)], dtype=np.float32) / max(1.0, n - 1)
    return torch.from_numpy(np.stack([ones, deg], axis=1))


def _check_soft(G: nx.Graph, soft: np.ndarray, opt_value: int) -> bool:
    # Each pure 1.0 entry must be in some MIS; expected mass = opt_value.
    if not np.isfinite(soft).all():
        return False
    if soft.min() < -1e-6 or soft.max() > 1 + 1e-6:
        return False
    return abs(soft.sum() - opt_value) < 1e-3 or soft.sum() <= opt_value + 1e-3


def _save_shard(cfg: Cfg, idx: int, shard, meta):
    path = os.path.join(cfg.output_dir, f"mis_shard_{idx:04d}.pt")
    torch.save({"meta": meta, "data": shard}, path)
    return path


@cli.command(singleton=True)
def build(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    env = _silent_env()

    meta = {
        "cfg": cfg.model_dump(),
        "created_unix": time.time(),
        "format": "dict(meta, data=list_of_samples)",
        "fields": {
            "x": "FloatTensor[n,1 or 2]",
            "edge_index": "LongTensor[2, 2|E|] undirected both directions",
            "y": "FloatTensor[n] in [0,1] (fraction of optimal MISs containing node)",
            "opt_value": "int (MIS size)",
            "num_optimal": "int (number of MISs averaged into y)",
            "n,p,d_target,num_edges,seed": "scalars",
        },
    }

    shard: list[dict] = []
    shard_idx = 0
    gen_times, solve_times, pool_counts = [], [], []
    t0 = time.time()

    pbar = tqdm(range(cfg.num_instances), desc=f"build {cfg.output_dir}", dynamic_ncols=True)
    for i in pbar:
        seed = cfg.seed_start + i
        tg0 = time.perf_counter()
        G, n, p, d = _gen_graph(seed, cfg)
        tg1 = time.perf_counter()
        gen_times.append(tg1 - tg0)

        ts0 = time.perf_counter()
        soft_y, opt_value, kept = _solve_mis_pool(G, env, cfg)
        ts1 = time.perf_counter()
        solve_times.append(ts1 - ts0)
        pool_counts.append(kept)

        if cfg.assert_valid_labels and not _check_soft(G, soft_y, opt_value):
            raise RuntimeError(f"Invalid soft label at i={i}, seed={seed}")

        shard.append({
            "x": _node_features(G, cfg),
            "edge_index": _edge_index(G),
            "y": torch.from_numpy(soft_y).float(),
            "opt_value": opt_value,
            "num_optimal": int(kept),
            "n": int(n),
            "p": float(p),
            "d_target": float(d),
            "num_edges": int(G.number_of_edges()),
            "seed": int(seed),
        })

        if (i + 1) % cfg.log_every == 0:
            avg_g = float(np.mean(gen_times[-cfg.log_every:]))
            avg_s = float(np.mean(solve_times[-cfg.log_every:]))
            avg_k = float(np.mean(pool_counts[-cfg.log_every:]))
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            pbar.set_postfix(gen=f"{avg_g*1000:.0f}ms", solve=f"{avg_s*1000:.0f}ms", K=f"{avg_k:.1f}", rate=f"{rate:.1f}/s")

        if len(shard) >= cfg.shard_size:
            _save_shard(cfg, shard_idx, shard, meta)
            shard = []
            shard_idx += 1

    if shard:
        _save_shard(cfg, shard_idx, shard, meta)

    total = time.time() - t0
    print(f"[done] {cfg.num_instances} graphs in {total:.1f}s  ({cfg.num_instances/total:.2f}/s)")
    print(f"[stats] mean solve={np.mean(solve_times)*1000:.0f}ms  mean pool size={np.mean(pool_counts):.2f}  max={max(pool_counts)}  min={min(pool_counts)}")


if __name__ == "__main__":
    cli()
