from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import argparse
import torch
import networkx as nx


# ------------------------
# Shard loading utilities
# ------------------------

def shard_path(shard_dir: str | Path, shard_id: int) -> Path:
    return Path(shard_dir) / f"mis_shard_{shard_id:04d}.pt"


def load_shard(shard_dir: str | Path, shard_id: int, map_location: str = "cpu") -> Dict[str, Any]:
    path = shard_path(shard_dir, shard_id)
    if not path.exists():
        raise FileNotFoundError(f"Shard not found: {path}")

    obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, dict) or "data" not in obj:
        raise ValueError(f"Unexpected shard format in {path} (expected dict with key 'data')")

    return obj


# ------------------------
# Visualization helpers
# ------------------------

def sample_to_nx(sample: Dict[str, Any]) -> nx.Graph:
    n = int(sample["n"])
    ei = sample["edge_index"]
    if isinstance(ei, torch.Tensor):
        ei = ei.cpu().numpy()

    edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def save_sample_plot(sample: Dict[str, Any], out_path: str, layout_seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    G = sample_to_nx(sample)

    pos = nx.spring_layout(G, seed=layout_seed)

    y = sample["y"].cpu().numpy() if isinstance(sample["y"], torch.Tensor) else sample["y"]
    node_sizes = [40 if yi == 0 else 140 for yi in y]
    node_colors = ["lightgray" if yi == 0 else "tab:orange" for yi in y]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    plt.title(
        f"n={sample['n']}  "
        f"E={sample['num_edges']}  "
        f"opt={sample['opt_value']}  "
        f"seed={sample['seed']}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------------
# Pretty print
# ------------------------

def describe_sample(sample: Dict[str, Any]) -> str:
    x = sample["x"]
    ei = sample["edge_index"]
    y = sample["y"]
    return (
        f"n={int(sample['n'])}  E={int(sample['num_edges'])}  "
        f"p={float(sample['p']):.4f}  d_target={float(sample['d_target']):.2f}  seed={int(sample['seed'])}\n"
        f"opt_value={int(sample['opt_value'])}  y_sum={int(y.sum().item())}\n"
        f"x.shape={tuple(x.shape)}  edge_index.shape={tuple(ei.shape)}  y.shape={tuple(y.shape)}"
    )


# ------------------------
# Interactive CLI
# ------------------------

def repl(shard_dir: str, shard_id: int):
    shard = load_shard(shard_dir, shard_id, map_location="cpu")
    data = shard["data"]
    n = len(data)

    i = 1  # 1-based index

    print(f"\nLoaded shard {shard_id:04d} with {n} samples")
    print(f"Path: {shard_path(shard_dir, shard_id)}\n")
    print("Commands:")
    print("  [Enter]        next")
    print("  p              previous")
    print("  j <k>          jump to local index k (1..N)")
    print("  g <idx>        show where global index idx lives")
    print("  v              visualize current graph (PNG)")
    print("  q              quit\n")

    while True:
        i = max(1, min(n, i))
        sample = data[i - 1]

        print(f"\n--- shard {shard_id:04d} | local {i}/{n} ---")
        print(describe_sample(sample))

        cmd = input("> ").strip()

        if cmd == "":
            i += 1

        elif cmd in {"q", "quit", "exit"}:
            break

        elif cmd == "p":
            i -= 1

        elif cmd.startswith("j "):
            try:
                k = int(cmd.split()[1])
                i = k
            except Exception:
                print("Usage: j <local_index>")

        elif cmd.startswith("g "):
            try:
                global_idx = int(cmd.split()[1])
                shard_size = n
                z = global_idx - 1
                target_shard = z // shard_size
                target_local = (z % shard_size) + 1
                print(f"Global idx {global_idx} → shard {target_shard:04d}, local {target_local}")
                if target_shard != shard_id:
                    print("Re-run with --shard to view that shard.")
                else:
                    i = target_local
            except Exception:
                print("Usage: g <global_index>")

        elif cmd == "v":
            out = f"vis/shard{shard_id:04d}_local{i:03d}_seed{int(sample['seed'])}.png"
            save_sample_plot(sample, out_path=out, layout_seed=int(sample["seed"]))
            print(f"Saved visualization to: {out}")

        else:
            print("Unknown command. Type q to quit.")


# ------------------------
# Entrypoint
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/mis-10k", help="dataset directory")
    ap.add_argument("--shard", type=int, required=True, help="shard id (e.g. 0 for mis_shard_0000.pt)")
    args = ap.parse_args()

    repl(args.dir, args.shard)


if __name__ == "__main__":
    main()
