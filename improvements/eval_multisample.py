"""
Multi-Sample Inference Evaluation for MIS

Instead of a single deterministic forward pass, this evaluator:
1. Runs multiple forward passes with temperature scaling / dropout noise
2. Applies enhanced post-processing to each sample
3. Returns the best solution across all samples

This is the key technique DIFUSCO uses to achieve SOTA — we replicate it here.
"""

import argparse
import sys
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.graph_transformer_trm_ssl import GraphTransformerTRM_SSL
from models.graph_transformer_trm import GraphTransformerTRM
from improvements.pp_enhanced import (
    build_adjacency,
    greedy_decode_core,
    local_search_1swap,
)
from models.pp import greedy_decode


def multisample_decode(probs_list, edge_index, num_nodes, decode_fn):
    """
    Given a list of probability vectors from multiple forward passes,
    decode each and return the best (largest) independent set.
    """
    best_size = 0
    best_tensor = None

    for probs in probs_list:
        size, tensor = decode_fn(probs, edge_index, num_nodes)
        if size > best_size:
            best_size = size
            best_tensor = tensor

    return best_size, best_tensor


def run_model_multisample(model, batch_dict, n_samples=16, temperatures=None,
                          use_dropout=True, n_supervision=1):
    """
    Run model multiple times with different noise sources:
    - Temperature scaling: vary sigmoid sharpness
    - Dropout noise: keep dropout enabled during inference
    - Additive noise: small Gaussian perturbation on logits

    Returns list of probability tensors.
    """
    if temperatures is None:
        # Default: sweep temperatures around 1.0
        base_temps = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]
        # Repeat to fill n_samples, cycling through temps
        temperatures = [base_temps[i % len(base_temps)] for i in range(n_samples)]

    all_probs = []

    for i in range(n_samples):
        temp = temperatures[i] if i < len(temperatures) else 1.0

        if use_dropout and i > 0:
            model.train()  # Enable dropout for stochastic samples
        else:
            model.eval()  # First sample is deterministic

        with torch.no_grad():
            carry = model.initial_carry(batch_dict)

            for _ in range(n_supervision):
                carry, _, _, preds, _ = model(carry, batch_dict)

            # NOTE: model returns post-sigmoid probabilities in preds["preds"],
            # NOT logits. Recover logits to apply temperature correctly, then
            # re-apply sigmoid. Previously this code double-sigmoided the probs,
            # saturating all outputs to [0.5, 0.73] and producing meaningless
            # raw confusion-matrix metrics.
            probs_raw = preds["preds"].squeeze().clamp(1e-6, 1.0 - 1e-6)
            if abs(temp - 1.0) < 1e-8:
                probs = probs_raw
            else:
                logits = torch.logit(probs_raw)
                probs = torch.sigmoid(logits / temp)

            # For samples > 0, add small noise for diversity
            if i > 0:
                noise = torch.randn_like(probs) * 0.05
                probs = (probs + noise).clamp(0, 1)

            all_probs.append(probs)

    model.eval()  # Restore eval mode
    return all_probs


def _decode_greedy(probs_np, adj, num_nodes, threshold=0.5):
    """Fast greedy decode with precomputed adjacency."""
    candidate_nodes = np.where(probs_np > threshold)[0]
    selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)
    return len(selected)


def _decode_enhanced(probs_np, adj, num_nodes, threshold=0.5):
    """Greedy + 1-swap with precomputed adjacency."""
    candidate_nodes = np.where(probs_np > threshold)[0]
    selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)
    selected = local_search_1swap(selected, adj, num_nodes, max_iters=20)
    return len(selected)


def _decode_multi_threshold(probs_np, adj, num_nodes, thresholds=(0.0, 0.2, 0.5)):
    """Multi-threshold + 1-swap with precomputed adjacency."""
    best = 0
    for t in thresholds:
        candidate_nodes = np.where(probs_np > t)[0]
        selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)
        selected = local_search_1swap(selected, adj, num_nodes, max_iters=20)
        best = max(best, len(selected))
    return best


def compute_raw_confusion(probs, labels):
    """Compute confusion matrix BEFORE decoding (raw threshold at 0.5)."""
    preds_binary = (probs > 0.5).float()
    tp = (preds_binary * labels).sum().item()
    tn = ((1 - preds_binary) * (1 - labels)).sum().item()
    fp = (preds_binary * (1 - labels)).sum().item()
    fn = ((1 - preds_binary) * labels).sum().item()
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def compute_raw_feasibility(probs, edge_index):
    """Compute feasibility BEFORE decoding (violations / edges)."""
    preds_binary = (probs > 0.5).float()
    pred_mask = preds_binary == 1
    if pred_mask.sum() > 0 and edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        violations = (pred_mask[src] & pred_mask[dst]).sum().float().item()
        total_edges = edge_index.size(1)
        return 1.0 - (violations / max(total_edges, 1))
    return 1.0


def evaluate_single_graph(graph_probs_list, graph_edges, graph_num_nodes, graph_opt,
                          graph_labels=None):
    """Evaluate a single graph with multiple decoding strategies.

    Precomputes adjacency once and reuses for all strategies.
    Also computes before-decoding metrics (confusion matrix, feasibility).
    """
    results = {}
    edge_index_np = graph_edges.cpu().numpy()
    adj = build_adjacency(edge_index_np, graph_num_nodes)

    # Convert first sample to numpy once
    probs0_np = graph_probs_list[0].cpu().numpy()

    # === Before decoding metrics (raw threshold at 0.5) ===
    probs0 = graph_probs_list[0]
    raw_pred_size = (probs0 > 0.5).float().sum().item()
    results["raw_pred_size"] = raw_pred_size
    results["raw_approx_ratio"] = raw_pred_size / max(graph_opt, 1e-8)
    results["raw_feasibility"] = compute_raw_feasibility(probs0, graph_edges)

    if graph_labels is not None:
        cm = compute_raw_confusion(probs0, graph_labels)
        for k, v in cm.items():
            results[f"raw_{k}"] = v

    # === After decoding metrics (post-processed) ===

    # 1. Baseline: original greedy (single sample, threshold=0.5)
    baseline_size = _decode_greedy(probs0_np, adj, graph_num_nodes)
    results["baseline_greedy"] = baseline_size / max(graph_opt, 1e-8)
    results["baseline_greedy_size"] = baseline_size

    # 2. Enhanced greedy with 1-swap (single sample)
    enhanced_size = _decode_enhanced(probs0_np, adj, graph_num_nodes)
    results["enhanced_greedy"] = enhanced_size / max(graph_opt, 1e-8)
    results["enhanced_greedy_size"] = enhanced_size

    # 3. Multi-threshold + 1-swap (single sample)
    mt_size = _decode_multi_threshold(probs0_np, adj, graph_num_nodes)
    results["multi_threshold"] = mt_size / max(graph_opt, 1e-8)
    results["multi_threshold_size"] = mt_size

    # 4. Multi-sample + baseline greedy (fast)
    best_ms_base = 0
    for probs in graph_probs_list:
        pnp = probs.cpu().numpy()
        best_ms_base = max(best_ms_base, _decode_greedy(pnp, adj, graph_num_nodes))
    results["multisample_greedy"] = best_ms_base / max(graph_opt, 1e-8)
    results["multisample_greedy_size"] = best_ms_base

    # 5. Multi-sample + enhanced greedy
    best_ms_enh = 0
    for probs in graph_probs_list:
        pnp = probs.cpu().numpy()
        best_ms_enh = max(best_ms_enh, _decode_enhanced(pnp, adj, graph_num_nodes))
    results["multisample_enhanced"] = best_ms_enh / max(graph_opt, 1e-8)
    results["multisample_enhanced_size"] = best_ms_enh

    # 6. Multi-sample + multi-threshold (BEST)
    best_ms_mt = 0
    for probs in graph_probs_list:
        pnp = probs.cpu().numpy()
        best_ms_mt = max(best_ms_mt, _decode_multi_threshold(pnp, adj, graph_num_nodes))
    results["multisample_multi_threshold"] = best_ms_mt / max(graph_opt, 1e-8)
    results["multisample_multi_threshold_size"] = best_ms_mt

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Sample MIS Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/difusco_benchmark/datasets/satlib/test")
    parser.add_argument("--model_type", type=str, default="ssl", choices=["ssl", "supervised"])
    parser.add_argument("--n_samples", type=int, default=16, help="Number of forward pass samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 for per-graph eval)")
    parser.add_argument("--n_supervision", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--H_cycles", type=int, default=2)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="MIS-TRM", help="Wandb project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Multi-Sample MIS Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")

    # Initialize wandb if requested
    if args.wandb:
        run_name = args.wandb_run_name or f"eval_{os.path.basename(args.checkpoint)}_{os.path.basename(args.data_path)}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Load dataset — batch_size=1 for per-graph evaluation
    ds_config = MISDatasetConfig(
        dataset_paths=[args.data_path],
        global_batch_size=args.batch_size,
        rank=0, num_replicas=1,
        drop_last=False,
        val_split=0.0,
        seed=42,
        use_pe=False,
        use_enhanced_features=False,
        max_shards=args.max_shards,
    )
    dataset = MISDataset(ds_config, split="train")
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    print(f"Graphs: {dataset.num_graphs}")
    print(f"Features: input_dim={dataset.metadata.input_dim}, pe_dim={dataset.metadata.pe_dim}")

    # Detect input_dim and pe_dim from checkpoint weights
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "x_embed.weight" in state_dict:
        ckpt_input_dim = state_dict["x_embed.weight"].shape[1]
        ckpt_pe_dim = state_dict["pe_embed.weight"].shape[1] if "pe_embed.weight" in state_dict else 0
    else:
        ckpt_input_dim = dataset.metadata.input_dim
        ckpt_pe_dim = dataset.metadata.pe_dim

    print(f"Checkpoint dims: input_dim={ckpt_input_dim}, pe_dim={ckpt_pe_dim}")

    # Rebuild dataset if needed to match checkpoint
    if ckpt_input_dim != dataset.metadata.input_dim or ckpt_pe_dim != dataset.metadata.pe_dim:
        print(f"⚠️ Checkpoint dims ({ckpt_input_dim},{ckpt_pe_dim}) != dataset dims ({dataset.metadata.input_dim},{dataset.metadata.pe_dim})")
        print(f"   Rebuilding dataset with use_pe={ckpt_pe_dim > 0}, use_enhanced_features={ckpt_input_dim > 2}")
        ds_config = MISDatasetConfig(
            dataset_paths=[args.data_path],
            global_batch_size=args.batch_size,
            rank=0, num_replicas=1,
            drop_last=False, val_split=0.0, seed=42,
            use_pe=(ckpt_pe_dim > 0),
            use_enhanced_features=(ckpt_input_dim > 2),
            max_shards=args.max_shards,
        )
        dataset = MISDataset(ds_config, split="train")
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        print(f"Rebuilt: input_dim={dataset.metadata.input_dim}, pe_dim={dataset.metadata.pe_dim}")

    # Build model
    model_config = {
        "input_dim": ckpt_input_dim,
        "pe_input_dim": ckpt_pe_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "L_cycles": args.L_cycles,
        "H_cycles": args.H_cycles,
    }

    if args.model_type == "ssl":
        model_config.update({"mu": 5.0, "feasibility_weight": 1.0, "selection_weight": 1.0})
        model = GraphTransformerTRM_SSL(model_config).to(device)
    else:
        model_config.update({"feasibility_weight": 0.0, "feasibility_loss_type": "soft"})
        model = GraphTransformerTRM(model_config).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Evaluate
    strategy_names = [
        "baseline_greedy",
        "enhanced_greedy",
        "multi_threshold",
        "multisample_greedy",
        "multisample_enhanced",
        "multisample_multi_threshold",
    ]
    raw_metric_names = [
        "raw_pred_size", "raw_approx_ratio", "raw_feasibility",
        "raw_tp", "raw_tn", "raw_fp", "raw_fn",
        "raw_precision", "raw_recall", "raw_f1",
    ]
    size_metric_names = [f"{s}_size" for s in strategy_names]
    all_metric_names = strategy_names + raw_metric_names + size_metric_names
    accumulators = {name: [] for name in all_metric_names}
    opt_sizes = []
    graph_count = 0

    start_time = time.time()

    for batch_name, batch, batch_size in tqdm(dataloader, desc="Evaluating"):
        batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}

        # Get graph info
        edge_index = batch_dict["edge_index"]
        labels = batch_dict["y"].float()
        batch_vec = batch_dict["batch"]
        ptr = batch_dict["ptr"]

        # Process per-graph
        n_graphs = len(ptr) - 1
        for g in range(n_graphs):
            node_mask = batch_vec == g
            graph_labels = labels[node_mask]
            graph_opt = graph_labels.sum().item()
            if graph_opt == 0:
                continue

            edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
            graph_edges = edge_index[:, edge_mask]
            graph_num_nodes = node_mask.sum().item()
            node_offset = ptr[g].item()
            graph_edges_local = graph_edges - node_offset

            # Run multi-sample inference
            all_probs = run_model_multisample(
                model, batch_dict, n_samples=args.n_samples,
                use_dropout=True, n_supervision=args.n_supervision,
            )

            # Extract per-graph probs from each sample
            graph_probs_list = []
            for probs in all_probs:
                graph_probs = probs[node_mask]
                graph_probs_list.append(graph_probs)

            # Evaluate all strategies (with labels for confusion matrix)
            results = evaluate_single_graph(
                graph_probs_list, graph_edges_local, graph_num_nodes, graph_opt,
                graph_labels=graph_labels,
            )

            for name in all_metric_names:
                if name in results:
                    accumulators[name].append(results[name])

            opt_sizes.append(graph_opt)
            graph_count += 1

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print(f"RESULTS ({graph_count} graphs, {elapsed:.1f}s)")
    print("=" * 70)

    # Before-decoding metrics
    print(f"\n--- Before Decoding (raw threshold=0.5) ---")
    for name in raw_metric_names:
        vals = accumulators.get(name, [])
        if vals:
            print(f"  {name:<25} {np.mean(vals):>10.4f}")

    # After-decoding metrics (approx ratios)
    print(f"\n--- After Decoding (approx ratio & sizes) ---")
    print(f"\n{'Strategy':<35} {'Mean AR':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Mean Size':>10}")
    print("-" * 85)

    results_summary = {}
    for name in strategy_names:
        vals = accumulators[name]
        size_vals = accumulators.get(f"{name}_size", [])
        if vals:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            mean_size = np.mean(size_vals) if size_vals else 0
            print(f"{name:<35} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {mean_size:>10.1f}")
            results_summary[name] = {"mean": mean_val, "std": std_val, "min": min_val, "max": max_val, "mean_size": mean_size}

    avg_opt = np.mean(opt_sizes) if opt_sizes else 0
    print(f"\nAvg optimal size: {avg_opt:.1f}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/max(graph_count,1):.2f}s/graph)")

    # Wandb logging
    if args.wandb:
        log_data = {
            "eval/n_graphs": graph_count,
            "eval/n_samples": args.n_samples,
            "eval/elapsed_s": elapsed,
            "eval/avg_opt_size": avg_opt,
        }
        # Before-decoding metrics
        for name in raw_metric_names:
            vals = accumulators.get(name, [])
            if vals:
                log_data[f"eval/{name}"] = np.mean(vals)

        # After-decoding metrics (per strategy)
        for name in strategy_names:
            vals = accumulators[name]
            if vals:
                log_data[f"eval/{name}_approx_ratio"] = np.mean(vals)
                log_data[f"eval/{name}_gap_pct"] = (1.0 - np.mean(vals)) * 100
            size_vals = accumulators.get(f"{name}_size", [])
            if size_vals:
                log_data[f"eval/{name}_size"] = np.mean(size_vals)

        wandb.log(log_data)
        wandb.finish()

    # Save results if output specified
    if args.output:
        import json
        output_data = {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "n_samples": args.n_samples,
            "n_graphs": graph_count,
            "elapsed_seconds": elapsed,
            "avg_opt_size": avg_opt,
            "raw_metrics": {name: float(np.mean(accumulators[name]))
                            for name in raw_metric_names if accumulators.get(name)},
            "results": {k: {"mean": float(v["mean"]), "std": float(v["std"]),
                            "min": float(v["min"]), "max": float(v["max"]),
                            "mean_size": float(v["mean_size"])}
                        for k, v in results_summary.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
