"""
Baseline Comparison: Evaluate trivial strategies vs trained model on SATLIB.

Tests whether the SSL model is actually learning or if greedy decoding does all the work.

Baselines:
1. All-ones: every node prob = 1.0  (trivial select-all)
2. Random uniform: prob ~ U(0,1)
3. Degree heuristic: prob = 1/(1+deg)  (same as TRM y_init)
4. Trained SSL model checkpoint
5. Trained Supervised model checkpoint (if available)
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.utils import degree

from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.pp import greedy_decode


def evaluate_baseline(name, prob_fn, val_dataloader, device="cuda"):
    """Evaluate a baseline probability assignment strategy."""
    total_pp_approx = 0
    total_pp_pred = 0
    total_opt = 0
    total_raw_feas = 0
    num_graphs = 0

    for batch_name, batch, batch_size in val_dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        edge_index = batch["edge_index"]
        labels = batch["y"].float()
        batch_vec = batch["batch"]
        ptr = batch["ptr"]

        n_graphs = len(ptr) - 1
        for g in range(n_graphs):
            node_mask = batch_vec == g
            graph_labels = labels[node_mask]
            graph_opt = graph_labels.sum().item()
            if graph_opt == 0:
                continue

            # Get edges for this graph
            edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
            graph_edges = edge_index[:, edge_mask]
            graph_num_nodes = node_mask.sum().item()

            # Remap edge indices to local
            node_offset = ptr[g].item()
            graph_edges_local = graph_edges - node_offset

            # Generate probabilities using the baseline strategy
            probs = prob_fn(graph_num_nodes, graph_edges_local, device)

            # Raw feasibility
            preds_binary = (probs > 0.5).float()
            src, dst = graph_edges_local[0], graph_edges_local[1]
            if preds_binary.sum() > 0 and graph_edges_local.size(1) > 0:
                violations = (preds_binary[src] * preds_binary[dst]).sum()
                raw_feas = 1.0 - (violations / graph_edges_local.size(1)).clamp(max=1.0).item()
            else:
                raw_feas = 1.0

            # Greedy decode
            pp_size, _ = greedy_decode(probs, graph_edges_local, graph_num_nodes)

            total_pp_approx += pp_size / graph_opt
            total_pp_pred += pp_size
            total_opt += graph_opt
            total_raw_feas += raw_feas
            num_graphs += 1

    avg_pp_approx = total_pp_approx / max(num_graphs, 1)
    avg_pp_pred = total_pp_pred / max(num_graphs, 1)
    avg_opt = total_opt / max(num_graphs, 1)
    avg_raw_feas = total_raw_feas / max(num_graphs, 1)

    print(f"  {name:30s} | PP Approx: {avg_pp_approx:.4f} | PP Pred: {avg_pp_pred:.1f} | Opt: {avg_opt:.1f} | Raw Feas: {avg_raw_feas:.4f} | Graphs: {num_graphs}")
    return {"name": name, "pp_approx_ratio": avg_pp_approx, "pp_pred_size": avg_pp_pred, "opt_size": avg_opt, "raw_feasibility": avg_raw_feas}


def evaluate_model_checkpoint(name, checkpoint_path, model_class, model_config, val_dataloader, device="cuda"):
    """Evaluate a trained model checkpoint."""
    model = model_class(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    total_pp_approx = 0
    total_pp_pred = 0
    total_opt = 0
    total_raw_feas = 0
    num_graphs = 0

    with torch.no_grad():
        for batch_name, batch, batch_size in val_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            carry = model.initial_carry(batch)
            carry, _, _, preds, _ = model(carry, batch)
            probs = preds["preds"].squeeze()

            edge_index = batch["edge_index"]
            labels = batch["y"].float()
            batch_vec = batch["batch"]
            ptr = batch["ptr"]

            n_graphs = len(ptr) - 1
            for g in range(n_graphs):
                node_mask = batch_vec == g
                graph_probs = probs[node_mask]
                graph_labels = labels[node_mask]
                graph_opt = graph_labels.sum().item()
                if graph_opt == 0:
                    continue

                edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
                graph_edges = edge_index[:, edge_mask]
                graph_num_nodes = node_mask.sum().item()
                node_offset = ptr[g].item()
                graph_edges_local = graph_edges - node_offset

                # Raw feasibility
                preds_binary = (graph_probs > 0.5).float()
                src, dst = graph_edges_local[0], graph_edges_local[1]
                if preds_binary.sum() > 0 and graph_edges_local.size(1) > 0:
                    violations = (preds_binary[src] * preds_binary[dst]).sum()
                    raw_feas = 1.0 - (violations / graph_edges_local.size(1)).clamp(max=1.0).item()
                else:
                    raw_feas = 1.0

                pp_size, _ = greedy_decode(graph_probs, graph_edges_local, graph_num_nodes)

                total_pp_approx += pp_size / graph_opt
                total_pp_pred += pp_size
                total_opt += graph_opt
                total_raw_feas += raw_feas
                num_graphs += 1

    avg_pp_approx = total_pp_approx / max(num_graphs, 1)
    avg_pp_pred = total_pp_pred / max(num_graphs, 1)
    avg_opt = total_opt / max(num_graphs, 1)
    avg_raw_feas = total_raw_feas / max(num_graphs, 1)

    print(f"  {name:30s} | PP Approx: {avg_pp_approx:.4f} | PP Pred: {avg_pp_pred:.1f} | Opt: {avg_opt:.1f} | Raw Feas: {avg_raw_feas:.4f} | Graphs: {num_graphs}")
    return {"name": name, "pp_approx_ratio": avg_pp_approx, "pp_pred_size": avg_pp_pred, "opt_size": avg_opt, "raw_feasibility": avg_raw_feas}


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison for MIS greedy decoding")
    parser.add_argument("--max_shards", type=int, default=10, help="Max shards to load")
    parser.add_argument("--ssl_checkpoint", type=str, default=None, help="Path to SSL model checkpoint")
    parser.add_argument("--sup_checkpoint", type=str, default=None, help="Path to supervised model checkpoint")
    parser.add_argument("--n_random_trials", type=int, default=5, help="Number of random trials to average")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load validation data (use_pe=True to match training config)
    val_config = MISDatasetConfig(
        dataset_paths=["data/difusco_benchmark/datasets/satlib/train"],
        global_batch_size=32,
        rank=0,
        num_replicas=1,
        drop_last=False,
        val_split=0.1,
        seed=0,
        use_pe=False,
        use_enhanced_features=False,
        max_shards=args.max_shards,
    )
    val_dataset = MISDataset(val_config, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)
    print(f"Validation graphs: {val_dataset.num_graphs}")

    print("\n" + "=" * 110)
    print("BASELINE COMPARISON")
    print("=" * 110)

    results = []

    # 1. All-ones baseline
    def all_ones(n, edges, dev):
        return torch.ones(n, device=dev)

    results.append(evaluate_baseline("All-Ones (trivial)", all_ones, val_dataloader, device))

    # 2. Random uniform (averaged over trials)
    random_approxes = []
    for trial in range(args.n_random_trials):
        torch.manual_seed(trial + 42)
        np.random.seed(trial + 42)

        def random_uniform(n, edges, dev):
            return torch.rand(n, device=dev)

        r = evaluate_baseline(f"Random Uniform (trial {trial + 1})", random_uniform, val_dataloader, device)
        random_approxes.append(r["pp_approx_ratio"])
    avg_random = sum(random_approxes) / len(random_approxes)
    print(f"  {'Random Uniform (AVERAGE)':30s} | PP Approx: {avg_random:.4f}")
    results.append({"name": "Random Uniform (avg)", "pp_approx_ratio": avg_random})

    # 3. Degree heuristic (same as TRM y_init: 1/(1+deg))
    def degree_heuristic(n, edges, dev):
        if edges.size(1) > 0:
            deg = degree(edges[0], num_nodes=n, dtype=torch.float).to(dev)
        else:
            deg = torch.zeros(n, device=dev)
        return (1.0 / (1.0 + deg)).clamp(0.01, 0.99)

    results.append(evaluate_baseline("Degree Heuristic 1/(1+d)", degree_heuristic, val_dataloader, device))

    # 4. Inverse degree squared (stronger bias to low-degree nodes)
    def degree_squared(n, edges, dev):
        if edges.size(1) > 0:
            deg = degree(edges[0], num_nodes=n, dtype=torch.float).to(dev)
        else:
            deg = torch.zeros(n, device=dev)
        return (1.0 / (1.0 + deg) ** 2).clamp(0.01, 0.99)

    results.append(evaluate_baseline("Degree Heuristic 1/(1+d)^2", degree_squared, val_dataloader, device))

    # 5. SSL model checkpoint
    if args.ssl_checkpoint:
        from models.graph_transformer_trm_ssl import GraphTransformerTRM_SSL

        ssl_config = {
            "input_dim": val_dataset.metadata.input_dim,
            "pe_input_dim": 0,
            "pe_dim": 16,
            "hidden_dim": 256,
            "num_layers": 2,
            "H_cycles": 2,
            "L_cycles": 6,
            "dropout": 0.2,
            "attn_dropout": 0.2,
            "mu": 5.0,
            "feasibility_weight": 2.0,
            "selection_weight": 5.0,
            "feasibility_loss_type": "log_barrier",
        }
        results.append(evaluate_model_checkpoint("SSL Model (checkpoint)", args.ssl_checkpoint, GraphTransformerTRM_SSL, ssl_config, val_dataloader, device))

    # 6. Supervised model checkpoint
    if args.sup_checkpoint:
        from models.graph_transformer_trm import GraphTransformerTRM

        sup_config = {
            "input_dim": val_dataset.metadata.input_dim,
            "pe_input_dim": 0,
            "pe_dim": 16,
            "hidden_dim": 256,
            "num_layers": 2,
            "H_cycles": 2,
            "L_cycles": 6,
            "dropout": 0.2,
            "attn_dropout": 0.2,
            "feasibility_weight": 1.0,
            "feasibility_loss_type": "log_barrier",
        }
        results.append(evaluate_model_checkpoint("Supervised Model (checkpoint)", args.sup_checkpoint, GraphTransformerTRM, sup_config, val_dataloader, device))

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY (sorted by PP Approx Ratio)")
    print("=" * 110)
    results_sorted = sorted(results, key=lambda x: x["pp_approx_ratio"], reverse=True)
    for r in results_sorted:
        print(f"  {r['name']:30s} | PP Approx: {r['pp_approx_ratio']:.4f}")


if __name__ == "__main__":
    main()
