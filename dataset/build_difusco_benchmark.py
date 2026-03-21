#!/usr/bin/env python3
"""
Generate the DIFUSCO MIS benchmark datasets for comparison.

Two datasets:
  1) SATLIB CBS (Controlled Backbone SAT) → MIS conflict graphs
     - Downloads CNF files from SATLIB website
     - Converts 3-SAT formulas to conflict graphs
     - Splits into train / test following DIMES (500 test, rest train)
     - Labels with KaMIS (same solver as DIFUSCO paper)

  2) ER-[700-800] Erdős–Rényi random graphs with p=0.15
     - Generates random ER graphs, n ∈ [700,800], p=0.15
     - Labels with KaMIS (same solver as DIFUSCO paper)
     - Training: 163,840 graphs (as in DIFUSCO paper)
     - Test: 500 graphs

Output format: PyTorch shard files (.pt) matching the project's format.
Solver: KaMIS (compiled from source, same as DIFUSCO benchmark framework).

Usage:
  # Generate SATLIB dataset
  python dataset/build_difusco_benchmark.py --mode satlib

  # Generate ER dataset
  python dataset/build_difusco_benchmark.py --mode er --num_train 163840

  # Test run (small subset)
  python dataset/build_difusco_benchmark.py --mode satlib --test_run
  python dataset/build_difusco_benchmark.py --mode er --test_run

  # Parallel generation (recommended for full dataset)
  python dataset/build_difusco_benchmark.py --mode satlib --num_workers 20
  python dataset/build_difusco_benchmark.py --mode er --num_train 163840 --num_workers 20
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# KaMIS solver
# ---------------------------------------------------------------------------
KAMIS_BINARY = os.path.expanduser("~/trm/data/difusco_benchmark/KaMIS/deploy/redumis")


def solve_mis_kamis(G: nx.Graph, time_limit: int = 60) -> tuple:
    """
    Solve MIS using KaMIS (redumis).
    Returns (y: np.ndarray of 0/1 labels, mis_size: int).
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Relabel to contiguous 0..n-1
    G_relab = nx.relabel_nodes(G, node_to_idx)
    G_relab.remove_edges_from(nx.selfloop_edges(G_relab))
    m = G_relab.number_of_edges()

    # Write METIS format
    with tempfile.NamedTemporaryFile(mode="w", suffix=".graph", delete=False) as f:
        f.write(f"{n} {m}\n")
        for v in range(n):
            neighbors = sorted([u + 1 for u in G_relab.neighbors(v)])
            f.write(" ".join(map(str, neighbors)) + "\n")
        graph_file = f.name

    output_file = graph_file + ".mis"

    try:
        subprocess.run(
            [
                KAMIS_BINARY,
                graph_file,
                f"--output={output_file}",
                f"--time_limit={time_limit}",
            ],
            capture_output=True,
            text=True,
            timeout=time_limit + 30,
        )

        if not os.path.exists(output_file):
            raise RuntimeError(f"KaMIS did not produce output for {graph_file}")

        with open(output_file) as f:
            labels = [int(line.strip()) for line in f if line.strip()]

        y = np.array(labels, dtype=np.int64)
        mis_size = int(y.sum())
        return y, mis_size

    finally:
        if os.path.exists(graph_file):
            os.unlink(graph_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def check_independent_set(G: nx.Graph, y: np.ndarray, nodes: list) -> bool:
    """Verify that y is actually an independent set."""
    idx = {v: i for i, v in enumerate(nodes)}
    for u, v in G.edges():
        if y[idx[u]] == 1 and y[idx[v]] == 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Graph → PyTorch Data conversion
# ---------------------------------------------------------------------------
def nx_to_edge_index(G: nx.Graph, nodes: list):
    """Convert networkx graph to [2, 2E] undirected edge_index tensor."""
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    edges = []
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        edges.append((i, j))
        edges.append((j, i))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    e = np.array(edges, dtype=np.int64).T
    return torch.from_numpy(e).long()


def make_node_features(n: int, G: nx.Graph, nodes: list):
    """Create node features: [1, degree_norm] for each node."""
    ones = np.ones(n, dtype=np.float32)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    deg = np.zeros(n, dtype=np.float32)
    for v in nodes:
        deg[node_to_idx[v]] = G.degree(v)
    deg_norm = deg / max(1.0, (n - 1))
    x = np.stack([ones, deg_norm], axis=1)
    return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# SATLIB CNF → MIS conflict graph
# ---------------------------------------------------------------------------
def cnf_to_mis_graph(cnf_clauses, nv):
    """
    Convert a 3-SAT formula to a MIS conflict graph.
    Nodes = one per literal occurrence in a clause.
    Edges connect: (1) literals in same clause, (2) complementary literals.
    """
    ind = {}
    for k in range(1, nv + 1):
        ind[k] = []
        ind[-k] = []

    edges = []
    for i, clause in enumerate(cnf_clauses):
        if len(clause) != 3:
            continue
        a, b, c = clause[0], clause[1], clause[2]
        aa = 3 * i + 0
        bb = 3 * i + 1
        cc = 3 * i + 2
        ind[a].append(aa)
        ind[b].append(bb)
        ind[c].append(cc)
        # Intra-clause edges
        edges.append((aa, bb))
        edges.append((aa, cc))
        edges.append((bb, cc))

    # Inter-clause edges (complementary literals)
    for k in range(1, nv + 1):
        for u in ind[k]:
            for v in ind[-k]:
                edges.append((u, v))

    G = nx.from_edgelist(edges)
    return G


def parse_cnf_dimacs(content: str):
    """Parse a DIMACS CNF file content string. Returns (clauses, nv)."""
    clauses = []
    nv = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("c") or line.startswith("%"):
            continue
        if line.startswith("p"):
            parts = line.split()
            nv = int(parts[2])
            continue
        lits = list(map(int, line.split()))
        if lits and lits[-1] == 0:
            lits = lits[:-1]
        if lits:
            clauses.append(lits)
    return clauses, nv


# ---------------------------------------------------------------------------
# SATLIB CBS download & processing
# ---------------------------------------------------------------------------
CBS_BASE_URL = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/"
CBS_CONFIGS = [f"CBS_k3_n100_m{m}_b{b}" for m in [403, 411, 418, 423, 429, 435, 441, 449] for b in [10, 30, 50, 70, 90]]


def download_and_extract_cbs(config_name: str, cache_dir: Path) -> list:
    """Download a CBS tar.gz archive and extract CNF files."""
    archive_path = cache_dir / f"{config_name}.tar.gz"
    url = CBS_BASE_URL + f"{config_name}.tar.gz"

    if not archive_path.exists():
        print(f"  Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:
            print(f"  WARNING: Failed to download {url}: {e}")
            return []

    results = []
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".cnf"):
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read().decode("utf-8", errors="replace")
                        basename = os.path.basename(member.name)
                        stem = basename.replace(".cnf", "")
                        results.append((stem, content))
    except Exception as e:
        print(f"  WARNING: Failed to read {archive_path}: {e}")

    return results


def build_satlib_dataset(args):
    """Build the SATLIB CBS benchmark dataset."""
    output_dir = Path(args.output_dir) / "satlib"
    cache_dir = Path(args.output_dir) / "_satlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load the DIMES test split file list
    test_list_path = Path("data/difusco_benchmark/DIMES/MIS/data/sat_test_gpickle_files.txt")
    if not test_list_path.exists():
        print(f"ERROR: {test_list_path} not found. Clone DIMES repo first.")
        sys.exit(1)

    test_names = set()
    with open(test_list_path) as f:
        for line in f:
            name = line.strip().replace(".gpickle", "")
            if name:
                test_names.add(name)
    print(f"DIMES test split: {len(test_names)} graphs")

    # Download and parse all CBS CNF files
    all_graphs = {}
    configs_to_download = CBS_CONFIGS
    if args.test_run:
        configs_to_download = CBS_CONFIGS[:2]
        print(f"TEST RUN: Only processing {len(configs_to_download)} CBS configs")

    for config in tqdm(configs_to_download, desc="Downloading CBS archives"):
        cnf_files = download_and_extract_cbs(config, cache_dir)
        for stem, content in cnf_files:
            clauses, nv = parse_cnf_dimacs(content)
            if clauses:
                G = cnf_to_mis_graph(clauses, nv)
                all_graphs[stem] = (G, nv)

    print(f"Total graphs parsed: {len(all_graphs)}")

    # Split into train/test
    train_graphs = {}
    test_graphs = {}
    for name, (G, nv) in all_graphs.items():
        if name in test_names:
            test_graphs[name] = (G, nv)
        else:
            train_graphs[name] = (G, nv)

    print(f"Train: {len(train_graphs)}, Test: {len(test_graphs)}")
    if args.test_run:
        train_items = list(train_graphs.items())[:5]
        test_items = list(test_graphs.items())[:5]
    else:
        train_items = list(train_graphs.items())
        test_items = list(test_graphs.items())

    # Process and save
    for split_name, items in [("train", train_items), ("test", test_items)]:
        if not items:
            print(f"  No {split_name} items to process, skipping.")
            continue
        _process_and_save_satlib(items, output_dir / split_name, args)

    if not args.keep_cache:
        print("Cleaning up SATLIB cache...")
        shutil.rmtree(cache_dir, ignore_errors=True)


def _solve_single_satlib(args_tuple):
    """Worker function for parallel SATLIB solving. Returns a sample dict."""
    name, G, time_limit = args_tuple
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    t0 = time.perf_counter()
    y, opt_value = solve_mis_kamis(G, time_limit=time_limit)
    t1 = time.perf_counter()

    assert check_independent_set(G, y, nodes), f"Invalid IS for {name}"

    sample = {
        "x": make_node_features(n, G, nodes),
        "edge_index": nx_to_edge_index(G, nodes),
        "y": torch.from_numpy(y).long(),
        "opt_value": opt_value,
        "name": name,
        "n": n,
        "num_edges": G.number_of_edges(),
    }
    return sample, t1 - t0


def _process_and_save_satlib(items, output_path: Path, args):
    """Label and save SATLIB graphs as shards (supports parallel workers)."""
    output_path.mkdir(parents=True, exist_ok=True)
    shard_size = args.shard_size
    num_workers = args.num_workers

    meta = {
        "dataset": "satlib_cbs",
        "source": "SATLIB CBS k3 n100 (DIFUSCO benchmark)",
        "created_unix": time.time(),
        "format": "dict(meta, data=list_of_samples)",
        "fields": {
            "x": "FloatTensor[n,2] (ones, degree_norm)",
            "edge_index": "LongTensor[2, 2|E|] undirected both directions",
            "y": "LongTensor[n] in {0,1}",
            "opt_value": "int (MIS size)",
            "name": "str (original CBS instance name)",
            "n": "int (number of nodes)",
            "num_edges": "int (number of edges)",
        },
    }

    # Check for existing shards (resume support)
    existing_shards = sorted(output_path.glob("mis_shard_*.pt"))
    already_done = 0
    shard_idx = 0
    if existing_shards:
        for sp in existing_shards:
            d = torch.load(sp, weights_only=False)
            already_done += len(d["data"])
            shard_idx += 1
        print(f"  Resuming: {already_done} already saved in {len(existing_shards)} shards")
        if already_done >= len(items):
            print(f"  All {len(items)} graphs already done, skipping.")
            return
        items = items[already_done:]

    shard = []
    solve_times = []

    work_args = [(name, G, args.kamis_time_limit) for name, (G, _nv) in items]

    if num_workers <= 1:
        # Sequential
        pbar = tqdm(work_args, desc=f"Labeling {output_path.name}", dynamic_ncols=True)
        for wa in pbar:
            sample, dt = _solve_single_satlib(wa)
            solve_times.append(dt)
            shard.append(sample)

            if len(solve_times) % 10 == 0:
                avg_t = np.mean(solve_times[-10:])
                pbar.set_postfix({"solve_s": f"{avg_t:.1f}", "MIS": sample["opt_value"]})

            if len(shard) >= shard_size:
                path = output_path / f"mis_shard_{shard_idx:04d}.pt"
                torch.save({"meta": meta, "data": shard}, path)
                pbar.write(f"Saved {path}")
                shard = []
                shard_idx += 1
    else:
        # Parallel
        print(f"  Parallel solving with {num_workers} workers ...")
        pbar = tqdm(total=len(work_args), desc=f"Labeling {output_path.name}", dynamic_ncols=True)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_solve_single_satlib, wa): i for i, wa in enumerate(work_args)}
            for future in as_completed(futures):
                sample, dt = future.result()
                solve_times.append(dt)
                shard.append(sample)
                pbar.update(1)

                if len(solve_times) % 50 == 0:
                    avg_t = np.mean(solve_times[-50:])
                    pbar.set_postfix({"solve_s": f"{avg_t:.1f}", "done": len(solve_times)})

                if len(shard) >= shard_size:
                    path = output_path / f"mis_shard_{shard_idx:04d}.pt"
                    torch.save({"meta": meta, "data": shard}, path)
                    pbar.write(f"Saved {path}")
                    shard = []
                    shard_idx += 1
        pbar.close()

    if shard:
        path = output_path / f"mis_shard_{shard_idx:04d}.pt"
        torch.save({"meta": meta, "data": shard}, path)
        print(f"Saved {path}")

    total_done = already_done + len(solve_times)
    if solve_times:
        print(f"  {output_path.name} done: {total_done} graphs, avg solve: {np.mean(solve_times):.1f}s, total: {sum(solve_times) / 3600:.1f}h (wall)")


# ---------------------------------------------------------------------------
# ER-[700-800] dataset
# ---------------------------------------------------------------------------
def _generate_single_er(args_tuple):
    """Worker function for parallel ER generation. Returns a sample dict."""
    idx, seed, n_min, n_max, p, time_limit = args_tuple
    rng = random.Random(seed)
    n = rng.randint(n_min, n_max)
    G = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 2**31 - 1))
    nodes = sorted(G.nodes())
    num_nodes = G.number_of_nodes()

    y, opt_value = solve_mis_kamis(G, time_limit=time_limit)
    assert check_independent_set(G, y, nodes), f"Invalid IS at seed={seed}"

    sample = {
        "x": make_node_features(num_nodes, G, nodes),
        "edge_index": nx_to_edge_index(G, nodes),
        "y": torch.from_numpy(y).long(),
        "opt_value": opt_value,
        "n": num_nodes,
        "p": p,
        "num_edges": G.number_of_edges(),
        "seed": seed,
    }
    return idx, sample


def build_er_dataset(args):
    """Build the ER-[700-800] p=0.15 benchmark dataset."""
    output_dir = Path(args.output_dir) / "er_700_800"

    n_min, n_max = 700, 800
    p = 0.15
    num_train = args.num_train
    num_test = args.num_test

    if args.test_run:
        num_train = 3
        num_test = 2
        print(f"TEST RUN: {num_train} train + {num_test} test")

    # Generate test set first (seed 0..num_test-1)
    _generate_er_split(
        output_dir / "test",
        n_min,
        n_max,
        p,
        num_instances=num_test,
        seed_start=0,
        shard_prefix="er_test",
        shard_size=args.shard_size,
        split_name="test",
        time_limit=args.kamis_time_limit,
        num_workers=args.num_workers,
    )

    # Generate train set (seed 100000..100000+num_train-1)
    _generate_er_split(
        output_dir / "train",
        n_min,
        n_max,
        p,
        num_instances=num_train,
        seed_start=100_000,
        shard_prefix="er_train",
        shard_size=args.shard_size,
        split_name="train",
        time_limit=args.kamis_time_limit,
        num_workers=args.num_workers,
    )


def _generate_er_split(
    output_path: Path,
    n_min: int,
    n_max: int,
    p: float,
    num_instances: int,
    seed_start: int,
    shard_prefix: str,
    shard_size: int,
    split_name: str,
    time_limit: int,
    num_workers: int,
):
    """Generate and save ER graphs for one split."""
    output_path.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": "er_700_800",
        "source": f"ER n∈[{n_min},{n_max}] p={p} (DIFUSCO benchmark)",
        "created_unix": time.time(),
        "format": "dict(meta, data=list_of_samples)",
        "fields": {
            "x": "FloatTensor[n,2] (ones, degree_norm)",
            "edge_index": "LongTensor[2, 2|E|] undirected both directions",
            "y": "LongTensor[n] in {0,1}",
            "opt_value": "int (MIS size)",
            "n": "int",
            "p": "float",
            "num_edges": "int",
            "seed": "int",
        },
    }

    # Check for existing shards to support resuming
    existing_shards = sorted(output_path.glob("mis_shard_*.pt"))
    resume_from = 0
    if existing_shards:
        total_existing = 0
        for sp in existing_shards:
            sd = torch.load(sp, weights_only=False)
            total_existing += len(sd["data"])
        resume_from = total_existing
        print(f"  Resuming {split_name} from instance {resume_from} ({len(existing_shards)} existing shards)")

    if resume_from >= num_instances:
        print(f"  {split_name}: Already complete ({resume_from}/{num_instances})")
        return

    shard_idx = len(existing_shards)
    remaining = num_instances - resume_from
    total_t0 = time.time()

    # Prepare task arguments
    tasks = [(i, seed_start + resume_from + i, n_min, n_max, p, time_limit) for i in range(remaining)]

    shard = []
    completed = 0

    if num_workers <= 1:
        # Sequential mode
        pbar = tqdm(
            tasks,
            desc=f"ER {split_name}",
            dynamic_ncols=True,
            initial=resume_from,
            total=num_instances,
        )
        for task in pbar:
            idx, sample = _generate_single_er(task)
            shard.append(sample)
            completed += 1

            if completed % 5 == 0:
                elapsed = time.time() - total_t0
                rate = completed / max(elapsed, 1e-9)
                eta_h = (remaining - completed) / max(rate, 1e-9) / 3600
                pbar.set_postfix(
                    {
                        "inst/s": f"{rate:.3f}",
                        "ETA_h": f"{eta_h:.1f}",
                        "n": sample["n"],
                        "MIS": sample["opt_value"],
                    }
                )

            if len(shard) >= shard_size:
                path = output_path / f"mis_shard_{shard_idx:04d}.pt"
                torch.save({"meta": meta, "data": shard}, path)
                pbar.write(f"Saved {path}")
                shard = []
                shard_idx += 1
    else:
        # Parallel mode
        pbar = tqdm(
            total=remaining,
            desc=f"ER {split_name}",
            dynamic_ncols=True,
            initial=resume_from,
        )
        # Buffer to collect results in order
        buffer = {}
        next_expected = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate_single_er, task): task[0] for task in tasks}
            for future in as_completed(futures):
                idx, sample = future.result()
                buffer[idx] = sample
                pbar.update(1)
                completed += 1

                # Drain buffer in order
                while next_expected in buffer:
                    shard.append(buffer.pop(next_expected))
                    next_expected += 1

                    if len(shard) >= shard_size:
                        path = output_path / f"mis_shard_{shard_idx:04d}.pt"
                        torch.save({"meta": meta, "data": shard}, path)
                        pbar.write(f"Saved {path}")
                        shard = []
                        shard_idx += 1

                if completed % 10 == 0:
                    elapsed = time.time() - total_t0
                    rate = completed / max(elapsed, 1e-9)
                    eta_h = (remaining - completed) / max(rate, 1e-9) / 3600
                    pbar.set_postfix(
                        {
                            "inst/s": f"{rate:.3f}",
                            "ETA_h": f"{eta_h:.1f}",
                        }
                    )

        # Drain remaining buffer
        while next_expected in buffer:
            shard.append(buffer.pop(next_expected))
            next_expected += 1

        pbar.close()

    if shard:
        path = output_path / f"mis_shard_{shard_idx:04d}.pt"
        torch.save({"meta": meta, "data": shard}, path)
        print(f"Saved {path}")

    total_elapsed = time.time() - total_t0
    if completed > 0:
        print(f"\n  {split_name} done: {completed} instances in {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate DIFUSCO MIS benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["satlib", "er", "both"],
        help="Which dataset to generate: satlib, er, or both",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/difusco_benchmark/datasets",
        help="Root output directory",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=250,
        help="Number of instances per shard file",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=163_840,
        help="[ER only] Number of training graphs to generate",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=500,
        help="[ER only] Number of test graphs to generate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for KaMIS solving",
    )
    parser.add_argument(
        "--kamis_time_limit",
        type=int,
        default=60,
        help="KaMIS time limit per graph (seconds)",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Generate only a tiny subset for testing the pipeline",
    )
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        help="Keep downloaded SATLIB archives after processing",
    )
    args = parser.parse_args()

    # Verify KaMIS binary exists
    if not os.path.exists(KAMIS_BINARY):
        print(f"ERROR: KaMIS binary not found at {KAMIS_BINARY}")
        print("Build KaMIS first:")
        print("  cd ~/trm/data/difusco_benchmark/KaMIS && bash compile_withcmake.sh")
        sys.exit(1)

    print("=" * 60)
    print("DIFUSCO MIS Benchmark Dataset Generator")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    print(f"Solver: KaMIS ({KAMIS_BINARY})")
    print(f"Workers: {args.num_workers}")
    print(f"KaMIS time limit: {args.kamis_time_limit}s")
    if args.test_run:
        print("*** TEST RUN MODE ***")
    print()

    if args.mode in ("satlib", "both"):
        print("--- SATLIB CBS Dataset ---")
        build_satlib_dataset(args)
        print()

    if args.mode in ("er", "both"):
        print("--- ER-[700-800] Dataset ---")
        build_er_dataset(args)
        print()

    print("All done!")


if __name__ == "__main__":
    main()
