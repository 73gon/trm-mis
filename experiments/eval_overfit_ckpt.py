"""Evaluate an overfit_sl checkpoint on any data path (train or test).

Reuses helpers from experiments.overfit_sl for parity with training.
Reports aggregate raw + post-processed metrics, plus per-graph trace if asked.
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
    compute_all_metrics,
)
from models.graph_transformer_trm import GraphTransformerTRM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True,
                   help="e.g. data/difusco_benchmark/datasets/satlib/test")
    p.add_argument("--num_graphs", type=int, default=10_000_000,
                   help="Cap; will load all available shards up to this count.")
    p.add_argument("--batch_size", type=int, default=16)

    # Architecture (must match training)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=2)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--use_pe", type=int, default=1)
    p.add_argument("--use_enhanced_features", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=16)

    # Loss config (only used to build the model object; eval is loss-free)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--feasibility_weight", type=float, default=0.0)
    p.add_argument("--feasibility_loss_type", type=str, default="hinge")

    # Output / logging
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="MIS-TRM")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default="eval")
    p.add_argument("--max_steps", type=int, default=12,
                   help="Hard cap on TRM inner loop steps (overrides ACT halting).")
    return p.parse_args()


def aggregate(per_batch: list[dict], graph_counts: list[int]) -> dict:
    """Weight per-batch metrics by graph count to get per-graph averages."""
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
    print(f"[eval] device={device} ckpt={args.checkpoint} data={args.data_path}")

    # 1. Load data — load all available shards regardless of cap.
    try:
        samples, precomputed = load_first_n_graphs(args.data_path, args.num_graphs)
    except RuntimeError as e:
        # The helper raises when N > available; manually load everything instead.
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
    print(f"[eval] loaded {len(samples)} graphs (precomputed={precomputed})")

    # 2. Enhance features (no-op if cached)
    samples = [enhance_graph(s, use_pe=bool(args.use_pe),
                             use_enhanced_features=bool(args.use_enhanced_features),
                             pe_dim=args.pe_dim,
                             already_precomputed=precomputed) for s in samples]
    input_dim = int(samples[0]["x"].shape[1])
    pe_dim_actual = int(samples[0]["pe"].shape[1]) if "pe" in samples[0] else 0
    print(f"[eval] input_dim={input_dim} pe_dim={pe_dim_actual}")

    # 3. Build model + load checkpoint
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "num_heads": args.num_heads,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "input_dim": input_dim,
        "pe_dim": pe_dim_actual,
        "pe_input_dim": args.pe_dim,
        "pos_weight": args.pos_weight,
        "feasibility_weight": args.feasibility_weight,
        "feasibility_loss_type": args.feasibility_loss_type,
        "label_smoothing": 0.0,
        "selection_weight": 0.0,
        "track_steps": False,
    }
    model = GraphTransformerTRM(model_config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[eval]   first missing: {missing[:3]}")
    model.eval()

    # 4. Wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args))

    # 5. Run eval in mini-batches
    per_batch: list[dict] = []
    graph_counts: list[int] = []
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
                carry, _loss, _metrics, preds, all_finish = model(carry, batch)
                steps += 1
            probs = preds["preds"].squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            metrics = compute_all_metrics(
                probs=probs,
                labels=batch["y"].float(),
                edge_index=batch["edge_index"],
                batch_vec=batch["batch"],
                ptr=batch["ptr"],
            )
            per_batch.append(metrics)
            graph_counts.append(len(chunk))
            pp_ar = metrics["pp_approx_ratio"]
            pp_feas = metrics["pp_feasibility"]
            print(f"  batch {i//bs+1}/{(n+bs-1)//bs}: graphs {i+1}-{i+len(chunk)}/{n} "
                  f"pp_AR={pp_ar:.4f} pp_feas={pp_feas:.4f} steps={steps}", flush=True)

    elapsed = time.time() - t0

    # 6. Aggregate and report
    agg = aggregate(per_batch, graph_counts)
    print("\n=== Aggregate over", agg["num_graphs"], "graphs ===")
    print(f"  pp_approx_ratio = {agg['pp_approx_ratio']:.4f}")
    print(f"  pp_feasibility  = {agg['pp_feasibility']:.4f}")
    print(f"  raw_approx_ratio= {agg['raw_approx_ratio']:.4f}")
    print(f"  raw_feasibility = {agg['raw_feasibility']:.4f}")
    print(f"  f1              = {agg['f1']:.4f}")
    print(f"  precision/recall= {agg['precision']:.4f} / {agg['recall']:.4f}")
    print(f"  cm%[tp={agg['tp_pct']:.2f} tn={agg['tn_pct']:.2f} fp={agg['fp_pct']:.2f} fn={agg['fn_pct']:.2f}]")
    print(f"  elapsed={elapsed:.1f}s ({agg['num_graphs']/elapsed:.2f} graphs/s)")

    if args.wandb:
        wandb.log({f"eval/{k}": v for k, v in agg.items()})
        wandb.finish()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "data_path": args.data_path,
                "elapsed_s": elapsed,
                "aggregate": agg,
            }, f, indent=2)
        print(f"[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
