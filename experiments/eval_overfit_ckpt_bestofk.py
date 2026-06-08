"""Evaluate an overfit_sl checkpoint with best-of-K stochastic decoding.

Identical to eval_overfit_ckpt.py except instead of a single greedy_decode per
graph, we sample K Gumbel-perturbed orderings and keep the largest feasible IS.

Greedy_decode always returns a feasible IS, so "best" means largest set size.
Larger K -> more chances to escape bad orderings introduced by tied probs.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import torch
import wandb

from experiments.overfit_sl import (
    load_first_n_graphs,
    enhance_graph,
    to_batch,
)
from models.graph_transformer_trm import GraphTransformerTRM
from models.pp import greedy_decode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--num_graphs", type=int, default=10_000_000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--K", type=int, default=16, help="Best-of-K samples per graph.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Gumbel noise scale (0=greedy; 1=standard Gumbel-softmax).")
    p.add_argument("--threshold", type=float, default=0.5)

    # Architecture (must match training)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=2)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--use_pe", type=int, default=1)
    p.add_argument("--use_enhanced_features", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=16)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--feasibility_weight", type=float, default=0.0)
    p.add_argument("--feasibility_loss_type", type=str, default="hinge")

    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="MIS-TRM")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default="eval_bok")
    p.add_argument("--max_steps", type=int, default=12)
    return p.parse_args()


def best_of_k_decode_per_graph(probs, edge_index, num_nodes, K, temperature, threshold):
    """Returns (best_size, best_selected_tensor, sizes_per_sample)."""
    if num_nodes == 0:
        return 0, torch.zeros(0, device=probs.device), [0]
    # Convert probs to logits-ish for Gumbel perturbation; clamp for stability.
    p_clamped = probs.clamp(1e-6, 1 - 1e-6)
    logits = torch.log(p_clamped) - torch.log(1 - p_clamped)  # logit
    best_size = -1
    best_tensor = None
    sizes = []
    for k in range(K):
        if temperature > 0:
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
            perturbed_logits = logits + temperature * g
            perturbed_probs = torch.sigmoid(perturbed_logits)
        else:
            perturbed_probs = probs
        size, tensor = greedy_decode(perturbed_probs, edge_index, num_nodes,
                                     threshold=threshold)
        sizes.append(size)
        if size > best_size:
            best_size = size
            best_tensor = tensor
    return best_size, best_tensor, sizes


def compute_pp_metrics_bok(probs, edge_index, labels, batch_vec, ptr,
                           K, temperature, threshold):
    num_graphs = len(ptr) - 1
    total_opt = 0
    total_pp_pred = 0
    total_approx_ratio = 0.0
    total_violations = 0.0
    total_edges = 0
    sum_mean_size = 0.0
    sum_max_minus_mean = 0.0

    for g in range(num_graphs):
        node_mask = batch_vec == g
        graph_probs = probs[node_mask]
        graph_labels = labels[node_mask]
        edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
        graph_edge_index = edge_index[:, edge_mask]
        node_indices = torch.where(node_mask)[0]
        if len(node_indices) > 0:
            local_idx_map = torch.zeros(batch_vec.size(0), dtype=torch.long, device=probs.device)
            local_idx_map[node_indices] = torch.arange(len(node_indices), device=probs.device)
            graph_edge_index = local_idx_map[graph_edge_index]

        graph_opt = graph_labels.sum().item()
        graph_num_nodes = graph_probs.size(0)
        graph_num_edges = graph_edge_index.size(1)
        total_edges += graph_num_edges

        if graph_num_nodes > 0 and graph_opt > 0:
            best_size, best_tensor, sizes = best_of_k_decode_per_graph(
                graph_probs, graph_edge_index, graph_num_nodes,
                K=K, temperature=temperature, threshold=threshold)

            src, dst = graph_edge_index[0], graph_edge_index[1]
            sel_mask = best_tensor == 1.0
            if sel_mask.sum() > 0 and graph_num_edges > 0:
                v = (sel_mask[src] & sel_mask[dst]).sum().float().item()
            else:
                v = 0.0
            total_violations += v
            total_opt += graph_opt
            total_pp_pred += best_size
            total_approx_ratio += best_size / graph_opt
            mean_size = sum(sizes) / len(sizes)
            sum_mean_size += mean_size
            sum_max_minus_mean += best_size - mean_size

    feasibility = 1.0 - (total_violations / max(total_edges, 1))
    return {
        "pp_pred_size": total_pp_pred / max(num_graphs, 1),
        "pp_approx_ratio": total_approx_ratio / max(num_graphs, 1),
        "pp_feasibility": feasibility,
        "pp_mean_sample_size": sum_mean_size / max(num_graphs, 1),
        "pp_bok_lift_size": sum_max_minus_mean / max(num_graphs, 1),
    }


def aggregate(per_batch, graph_counts):
    total_graphs = sum(graph_counts)
    keys = [k for k in per_batch[0].keys() if isinstance(per_batch[0][k], (int, float))]
    out = {}
    for k in keys:
        s = sum(m[k] * c for m, c in zip(per_batch, graph_counts))
        out[k] = s / max(total_graphs, 1)
    out["num_graphs"] = total_graphs
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-bok] device={device} K={args.K} T={args.temperature} "
          f"ckpt={args.checkpoint} data={args.data_path}", flush=True)

    try:
        samples, precomputed = load_first_n_graphs(args.data_path, args.num_graphs)
    except RuntimeError:
        import glob, torch as _t
        shard_paths = sorted(p for p in glob.glob(os.path.join(args.data_path, "mis_shard_*.pt"))
                             if not p.endswith(".cache.pt"))
        samples = []
        precomputed = True
        for sp in shard_paths:
            cp = sp.replace(".pt", ".cache.pt")
            if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(sp):
                payload = _t.load(cp, weights_only=False)
            else:
                payload = _t.load(sp, weights_only=False)
                precomputed = False
            samples.extend(payload["data"])
        if not samples:
            raise
    print(f"[eval-bok] loaded {len(samples)} graphs (precomputed={precomputed})", flush=True)

    samples = [enhance_graph(s, use_pe=bool(args.use_pe),
                             use_enhanced_features=bool(args.use_enhanced_features),
                             pe_dim=args.pe_dim,
                             already_precomputed=precomputed) for s in samples]
    input_dim = int(samples[0]["x"].shape[1])
    pe_dim_actual = int(samples[0]["pe"].shape[1]) if "pe" in samples[0] else 0

    model_config = {
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "H_cycles": args.H_cycles, "L_cycles": args.L_cycles,
        "num_heads": args.num_heads, "dropout": 0.0, "attn_dropout": 0.0,
        "input_dim": input_dim, "pe_dim": pe_dim_actual,
        "pe_input_dim": args.pe_dim, "pos_weight": args.pos_weight,
        "feasibility_weight": args.feasibility_weight,
        "feasibility_loss_type": args.feasibility_loss_type,
        "label_smoothing": 0.0, "selection_weight": 0.0, "track_steps": False,
    }
    model = GraphTransformerTRM(model_config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval-bok] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args))

    per_batch = []
    graph_counts = []
    t0 = time.time()
    bs = max(1, args.batch_size)
    n = len(samples)

    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = samples[i:i + bs]
            batch = to_batch(chunk, device)
            carry = model.initial_carry(batch)
            all_finish = False
            preds = None
            steps = 0
            while not all_finish and steps < args.max_steps:
                carry, _l, _m, preds, all_finish = model(carry, batch)
                steps += 1
            probs = preds["preds"].squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            metrics = compute_pp_metrics_bok(
                probs=probs,
                edge_index=batch["edge_index"],
                labels=batch["y"].float(),
                batch_vec=batch["batch"],
                ptr=batch["ptr"],
                K=args.K, temperature=args.temperature, threshold=args.threshold,
            )
            per_batch.append(metrics)
            graph_counts.append(len(chunk))
            print(f"  batch {i//bs+1}/{(n+bs-1)//bs}: graphs {i+1}-{i+len(chunk)}/{n} "
                  f"pp_AR={metrics['pp_approx_ratio']:.4f} "
                  f"mean_size={metrics['pp_mean_sample_size']:.2f} "
                  f"bok_lift={metrics['pp_bok_lift_size']:.2f} "
                  f"feas={metrics['pp_feasibility']:.4f} steps={steps}", flush=True)

    elapsed = time.time() - t0
    agg = aggregate(per_batch, graph_counts)
    print("\n=== Aggregate over", agg["num_graphs"], "graphs ===")
    print(f"  K               = {args.K}  T={args.temperature}")
    print(f"  pp_approx_ratio = {agg['pp_approx_ratio']:.4f}  (best-of-K)")
    print(f"  pp_mean_sample_size = {agg['pp_mean_sample_size']:.2f}")
    print(f"  pp_bok_lift_size    = {agg['pp_bok_lift_size']:.2f}")
    print(f"  pp_feasibility  = {agg['pp_feasibility']:.4f}")
    print(f"  elapsed={elapsed:.1f}s ({agg['num_graphs']/elapsed:.2f} graphs/s)")

    if args.wandb:
        wandb.log({f"eval/{k}": v for k, v in agg.items()})
        wandb.finish()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint, "data_path": args.data_path,
                "K": args.K, "temperature": args.temperature,
                "elapsed_s": elapsed, "aggregate": agg,
            }, f, indent=2)
        print(f"[eval-bok] wrote {args.output_json}")


if __name__ == "__main__":
    main()
