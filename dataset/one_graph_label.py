import random
import numpy as np
import networkx as nx
import torch
from gurobi_optimods.mwis import maximum_weighted_independent_set

def generate_erdos_renyi_graph(seed: int, n_min=50, n_max=250, d_min=6.0, d_max=14.0):
    rng = random.Random(seed)

    n = rng.randint(n_min, n_max)
    d = rng.uniform(d_min, d_max)
    p = max(0.0, min(1.0, d / (n - 1)))  # expected degree ~ d

    G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**31 - 1))
    return G, n, p, d

def label_with_gurobi_mis(G: nx.Graph):
    n = G.number_of_nodes()
    weights = np.ones(n, dtype=float)
    res = maximum_weighted_independent_set(G, weights)

    y = np.zeros(n, dtype=int)
    y[np.array(res.x, dtype=int)] = 1
    opt_value = int(res.f)
    return y, opt_value

def check_independent_set(G: nx.Graph, y: np.ndarray):
    for u, v in G.edges():
        if y[u] == 1 and y[v] == 1:
            return False
    return True

def nx_to_edge_index(G: nx.Graph):
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges_rev = edges[:, ::-1]
    edges_all = np.vstack([edges, edges_rev])
    return torch.from_numpy(edges_all.T).long()

def make_node_features(G: nx.Graph):
    n = G.number_of_nodes()
    deg = np.array([G.degree(i) for i in range(n)], dtype=np.float32)
    deg_norm = deg / max(1.0, (n - 1))
    ones = np.ones(n, dtype=np.float32)
    x = np.stack([ones, deg_norm], axis=1)
    return torch.from_numpy(x)

if __name__ == "__main__":
    seed = 42

    # --- Generate + label ---
    G, n, p, d = generate_erdos_renyi_graph(seed)
    y, opt_value = label_with_gurobi_mis(G)

    assert y.sum() == opt_value
    assert check_independent_set(G, y)

    # --- Convert to tensors ---
    edge_index = nx_to_edge_index(G)
    x = make_node_features(G)
    y_t = torch.from_numpy(y).long()

    # --- Pack sample ---
    sample = {
        "x": x,                       # [n, 2]
        "edge_index": edge_index,     # [2, 2|E|]
        "y": y_t,                     # [n]
        "opt_value": opt_value,
        "n": n,
        "p": p,
        "d_target": d,
        "num_edges": G.number_of_edges(),
        "seed": seed,
    }

    torch.save(sample, "one_sample.pt")

    # --- Print summary ---
    print("=== Instance ===")
    print(f"seed={seed}")
    print(f"n={n}")
    print(f"target expected degree d~{d:.2f}")
    print(f"p={p:.5f}")
    print(f"|E|={G.number_of_edges()}")

    print("\n=== Gurobi label (MIS) ===")
    print(f"opt_value (MIS size) = {opt_value}")
    print(f"selected nodes        = {int(y.sum())}")
    print(f"edge violations       = 0")

    print("\nSaved -> one_sample.pt")
