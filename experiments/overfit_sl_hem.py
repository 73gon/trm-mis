"""Hard-Example Mining variant of overfit_sl.py.

Differences from overfit_sl.py:

* `--init_from <ckpt>`  load weights at start (warm-start)
* `--hem_alpha <float>` weight power for biased sampling: w_i = (1 - pp_AR_i + 0.05) ** alpha
* `--hem_eval_every <iters>` re-rank graphs by per-graph pp_AR every N iters
                              (initial rank at iter 0, refresh at every multiple)
* `--hem_eval_max_steps <int>` inner-loop cap during HEM eval pass
* `--hem_eval_batch <int>` batch size for the per-graph eval pass

Trains with biased mini-batch sampling: each iter samples `bs` graph indices
according to weights w_i.  Easy graphs (high pp_AR) get visited rarely, hard
graphs frequently.  When `hem_alpha=0` this collapses to uniform sampling.

Original overfit_sl.py is untouched.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import time
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from models.graph_transformer_trm import GraphTransformerTRM
from models.pp import greedy_decode

# Reuse helpers from the original (untouched) trainer.
from experiments.overfit_sl import (
    load_first_n_graphs,
    enhance_graph,
    to_batch,
    compute_all_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--num_graphs", type=int, default=25000)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Architecture (must match warm-start ckpt)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=2)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n_supervision", type=int, default=1)

    # Loss
    p.add_argument("--pos_weight", type=float, default=5.0)
    p.add_argument("--feasibility_weight", type=float, default=2.0)
    p.add_argument("--feasibility_loss_type", type=str, default="hinge")

    # Features
    p.add_argument("--use_pe", type=int, default=1)
    p.add_argument("--use_enhanced_features", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=16)

    # Logging / output
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/overfit_sl_hem")
    p.add_argument("--run_name", type=str, default="overfit_sl_hem")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="MIS-TRM")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="onecycle",
                   choices=["none", "onecycle"])
    p.add_argument("--warmup_pct", type=float, default=0.1)

    # HEM-specific
    p.add_argument("--init_from", type=str, default=None,
                   help="Checkpoint to warm-start from (model_state_dict).")
    p.add_argument("--hem_alpha", type=float, default=2.0,
                   help="Power for sample weighting; 0=uniform, 2=hard graphs sampled ~4x more.")
    p.add_argument("--hem_eval_every", type=int, default=50_000,
                   help="Re-rank graphs every this many iters. Set huge to rank only at start.")
    p.add_argument("--hem_eval_max_steps", type=int, default=12)
    p.add_argument("--hem_eval_batch", type=int, default=16)
    p.add_argument("--max_steps_train", type=int, default=12,
                   help="Inner-loop cap during training (matches eval).")

    # Eval triggers
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--pp_ar_threshold", type=float, default=0.999)
    p.add_argument("--bce_threshold", type=float, default=0.01)
    p.add_argument("--disable_early_stop", action="store_true")
    return p.parse_args()


def per_graph_pp_ar_eval(model, samples, device, max_steps: int, batch_size: int):
    """Run the model on every graph once and return a list of per-graph pp_AR.
    Mirrors eval_overfit_ckpt logic but returns per-graph (not aggregated)."""
    model.eval()
    n = len(samples)
    out = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = samples[i:i + batch_size]
            batch = to_batch(chunk, device)
            carry = model.initial_carry(batch)
            all_finish = False
            preds = None
            steps = 0
            while not all_finish and steps < max_steps:
                carry, _l, _m, preds, all_finish = model(carry, batch)
                steps += 1
            probs = preds["preds"].squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            # Per-graph greedy decode
            edge_index = batch["edge_index"]
            batch_vec = batch["batch"]
            labels = batch["y"].float()
            for g in range(len(chunk)):
                node_mask = batch_vec == g
                graph_probs = probs[node_mask]
                graph_labels = labels[node_mask]
                edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
                graph_edge_index = edge_index[:, edge_mask]
                node_indices = torch.where(node_mask)[0]
                if len(node_indices) > 0:
                    local_idx_map = torch.zeros(batch_vec.size(0), dtype=torch.long, device=device)
                    local_idx_map[node_indices] = torch.arange(len(node_indices), device=device)
                    graph_edge_index = local_idx_map[graph_edge_index]
                graph_opt = graph_labels.sum().item()
                graph_num_nodes = graph_probs.size(0)
                if graph_num_nodes > 0 and graph_opt > 0:
                    pp_size, _ = greedy_decode(graph_probs, graph_edge_index, graph_num_nodes)
                    out[i + g] = pp_size / graph_opt
                else:
                    out[i + g] = 1.0
    print(f"[hem-eval] per-graph pp_AR over {n} graphs in {time.time()-t0:.1f}s "
          f"(mean={out.mean():.4f} min={out.min():.4f} bottom-decile={np.quantile(out, 0.1):.4f})",
          flush=True)
    model.train()
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hem] device={device}", flush=True)

    # 1. Load
    try:
        raw_samples, already_precomputed = load_first_n_graphs(args.data_path, args.num_graphs)
    except RuntimeError:
        # fallback: load all
        shard_paths = sorted(p for p in glob.glob(os.path.join(args.data_path, "mis_shard_*.pt"))
                             if not p.endswith(".cache.pt"))
        raw_samples = []
        already_precomputed = True
        for sp in shard_paths:
            cp = sp.replace(".pt", ".cache.pt")
            if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(sp):
                payload = torch.load(cp, weights_only=False)
            else:
                payload = torch.load(sp, weights_only=False)
                already_precomputed = False
            raw_samples.extend(payload["data"])
        raw_samples = raw_samples[:args.num_graphs]
    print(f"[hem] loaded {len(raw_samples)} graphs", flush=True)

    samples = [enhance_graph(s, use_pe=bool(args.use_pe),
                             use_enhanced_features=bool(args.use_enhanced_features),
                             pe_dim=args.pe_dim,
                             already_precomputed=already_precomputed)
               for s in raw_samples]
    input_dim = int(samples[0]["x"].shape[1])
    pe_dim_actual = int(samples[0]["pe"].shape[1]) if "pe" in samples[0] else 0
    print(f"[hem] input_dim={input_dim} pe_dim={pe_dim_actual}", flush=True)

    N = len(samples)
    bs = max(1, min(args.batch_size, N))

    # 2. Model
    model_config = {
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "H_cycles": args.H_cycles, "L_cycles": args.L_cycles,
        "num_heads": args.num_heads, "dropout": args.dropout,
        "attn_dropout": args.dropout,
        "input_dim": input_dim, "pe_dim": pe_dim_actual,
        "pe_input_dim": args.pe_dim,
        "pos_weight": args.pos_weight,
        "feasibility_weight": args.feasibility_weight,
        "feasibility_loss_type": args.feasibility_loss_type,
        "label_smoothing": 0.0, "selection_weight": 0.0, "track_steps": False,
    }
    model = GraphTransformerTRM(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hem] model params={n_params/1e6:.2f}M", flush=True)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        miss, unx = model.load_state_dict(state, strict=False)
        print(f"[hem] warm-start from {args.init_from}: missing={len(miss)} unexpected={len(unx)}",
              flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if args.lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr, total_steps=args.iterations,
            pct_start=args.warmup_pct, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=100.0)
    else:
        scheduler = None

    wb = None
    if args.wandb:
        import wandb as _wb
        wb = _wb.init(project=args.wandb_project, entity=args.wandb_entity,
                      name=args.run_name,
                      config=vars(args) | {"input_dim": input_dim,
                                            "pe_dim_actual": pe_dim_actual,
                                            "num_params": n_params, "N": N},
                      tags=["overfit", "hem", f"N{N}"])

    # 3. Initial per-graph pp_AR eval (defines sampling weights)
    print(f"[hem] initial per-graph eval...", flush=True)
    per_pp_ar = per_graph_pp_ar_eval(
        model, samples, device,
        max_steps=args.hem_eval_max_steps,
        batch_size=args.hem_eval_batch,
    )
    weights = (1.0 - per_pp_ar + 0.05) ** args.hem_alpha
    weights = weights / weights.sum()
    print(f"[hem] alpha={args.hem_alpha} weights: "
          f"min={weights.min():.6f} max={weights.max():.6f} "
          f"top10frac={np.sort(weights)[-N//10:].sum():.3f}", flush=True)
    if wb is not None:
        wb.log({"hem/init_pp_ar_mean": float(per_pp_ar.mean()),
                "hem/init_pp_ar_min": float(per_pp_ar.min()),
                "hem/init_pp_ar_p10": float(np.quantile(per_pp_ar, 0.1))}, step=0)

    # 4. Training loop
    os.makedirs(args.checkpoint_path, exist_ok=True)
    best_approx = float(per_pp_ar.mean())
    best_step = 0
    t_start = time.time()
    recent_pp_ar: collections.deque = collections.deque(maxlen=args.early_stop_patience)
    early_stopped_at = -1
    stop_reason = ""

    model.train()
    for it in range(1, args.iterations + 1):
        ids = rng.choice(N, size=bs, replace=False, p=weights)
        sel = [samples[i] for i in ids]
        batch = to_batch(sel, device)

        carry = model.initial_carry(batch)
        optim.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        last_metrics: dict = {}
        preds = None
        steps = 0
        for _ in range(args.n_supervision):
            carry, loss, metrics, preds, all_finish = model(carry, batch)
            total_loss = total_loss + loss
            last_metrics = metrics
            steps += 1
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if it % args.log_every == 0 or it == 1 or it == args.iterations:
            probs = preds["preds"].squeeze()
            m = compute_all_metrics(probs, batch["y"].float(), batch["edge_index"],
                                    batch["batch"], batch["ptr"])
            elapsed = time.time() - t_start
            it_per_s = it / max(1e-6, elapsed)
            loss_val = float(total_loss.detach())
            bce = float(last_metrics.get("loss_bce", torch.tensor(0.0)).detach()) if last_metrics else 0.0
            feas_l = float(last_metrics.get("loss_feasibility", torch.tensor(0.0)).detach()) if last_metrics else 0.0
            print(f"[it {it:6d}/{args.iterations}] loss={loss_val:.4f} bce={bce:.4f} "
                  f"feas_l={feas_l:.4f} pp[AR={m['pp_approx_ratio']:.4f}] "
                  f"raw[AR={m['raw_approx_ratio']:.4f}] "
                  f"F1={m['f1']:.3f} it/s={it_per_s:.2f}", flush=True)
            cur_lr = optim.param_groups[0]["lr"]
            if wb is not None:
                wb.log({"iter": it, "loss/total": loss_val, "loss/bce": bce,
                        "loss/feasibility": feas_l, "it_per_s": it_per_s,
                        "lr": cur_lr,
                        **{f"metrics/{k}": v for k, v in m.items()}}, step=it)

            score = m["pp_approx_ratio"]
            if score > best_approx:
                best_approx = score
                best_step = it
                torch.save({"model_state_dict": model.state_dict(),
                            "config": model_config, "iter": it, "metrics": m},
                           os.path.join(args.checkpoint_path, "best.pt"))

            recent_pp_ar.append(m["pp_approx_ratio"])
            if not args.disable_early_stop and len(recent_pp_ar) == args.early_stop_patience:
                if all(v >= args.pp_ar_threshold for v in recent_pp_ar):
                    early_stopped_at = it
                    stop_reason = "pp_ar"
                    print(f"[hem] EARLY STOP at iter {it}", flush=True)
                    break

        # Periodic re-ranking
        if args.hem_eval_every > 0 and it % args.hem_eval_every == 0 and it < args.iterations:
            print(f"[hem] re-ranking at iter {it}", flush=True)
            per_pp_ar = per_graph_pp_ar_eval(
                model, samples, device,
                max_steps=args.hem_eval_max_steps,
                batch_size=args.hem_eval_batch,
            )
            weights = (1.0 - per_pp_ar + 0.05) ** args.hem_alpha
            weights = weights / weights.sum()
            if wb is not None:
                wb.log({"hem/refresh_pp_ar_mean": float(per_pp_ar.mean()),
                        "hem/refresh_pp_ar_min": float(per_pp_ar.min()),
                        "hem/refresh_pp_ar_p10": float(np.quantile(per_pp_ar, 0.1))}, step=it)

    # Final eval
    print(f"[hem] final per-graph eval...", flush=True)
    final_per = per_graph_pp_ar_eval(
        model, samples, device,
        max_steps=args.hem_eval_max_steps,
        batch_size=args.hem_eval_batch,
    )
    elapsed = time.time() - t_start
    print(f"[hem] done in {elapsed:.1f}s ({args.iterations/elapsed:.2f} it/s)")
    print(f"[hem] best pp_AR (train batch)={best_approx:.4f} at iter {best_step}")
    print(f"[hem] final per-graph pp_AR mean={final_per.mean():.4f} "
          f"min={final_per.min():.4f} p10={np.quantile(final_per, 0.1):.4f}")

    if wb is not None:
        wb.summary["best_pp_approx_ratio"] = best_approx
        wb.summary["best_iter"] = best_step
        wb.summary["final_per_graph_mean"] = float(final_per.mean())
        wb.summary["final_per_graph_p10"] = float(np.quantile(final_per, 0.1))
        wb.summary["early_stopped_at"] = early_stopped_at
        wb.summary["total_seconds"] = elapsed
        wb.finish()


if __name__ == "__main__":
    main()
