"""
Stage-A/B/C overfit probe for supervised MIS model.

Purpose
-------
Diagnostic: train GraphTransformerTRM on a tiny fixed set of graphs (1, 10,
100, ...) to verify that the architecture is capable of representing a perfect
solution before scaling up. If the model cannot overfit N=1, no amount of data
will help.

Design choices
--------------
- Loads graphs directly from shards (no val split, no streaming).
- Feature extraction (Laplacian PE + enhanced node features) happens ONCE at
  start-up and is kept in GPU memory. This eliminates the ~0.1-1s/graph CPU
  tax that tanks GPU utilization in the standard MISDataset streaming path.
- Trains in a pure in-memory loop: every "epoch" is one forward/backward on
  every cached graph. With N=1 this is effectively `iterations` backward
  passes on the same sample.
- Prints raw (no greedy decode) metrics every few steps so we can see whether
  the model actually learns to emit the true MIS, rather than just producing
  logits that get rescued by post-processing.
- Optional `nvidia-smi` polling via the companion SLURM script.

CLI
---
python -m experiments.overfit_sl \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --num_graphs 1 --iterations 2000 \
    --use_pe 1 --use_enhanced_features 1 --precomputed 1

Notes
-----
- We do NOT touch train_mis.py; this script is self-contained.
- Saves best (by raw approx ratio) checkpoint to `--checkpoint_path`.
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
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from dataset.mis_dataset import compute_laplacian_pe, compute_node_features
from models.graph_transformer_trm import GraphTransformerTRM
from models.metrics import compute_metrics, compute_pp_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Directory containing mis_shard_*.pt")
    p.add_argument("--num_graphs", type=int, default=1,
                   help="How many graphs to use (takes the first N).")
    p.add_argument("--iterations", type=int, default=2000,
                   help="Total forward/backward passes. Each pass touches all "
                        "N graphs (one mini-batch if num_graphs is small).")
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 => use all selected graphs in a single batch.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Architecture
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--H_cycles", type=int, default=2)
    p.add_argument("--L_cycles", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Keep at 0 for overfit probe (we WANT it to overfit).")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--n_supervision", type=int, default=1)

    # Loss
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--feasibility_weight", type=float, default=1.0)
    p.add_argument("--feasibility_loss_type", type=str, default="hinge")

    # Features
    p.add_argument("--use_pe", type=int, default=1)
    p.add_argument("--use_enhanced_features", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=16)
    p.add_argument("--precomputed", type=int, default=1,
                   help="If 1, compute features once at start (small N). If 0, "
                        "recompute on every iteration (mainly for debugging).")

    # Logging / output
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/overfit_sl")
    p.add_argument("--run_name", type=str, default="overfit_sl")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="MIS-TRM")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_from", type=str, default=None,
                   help="path to a best.pt to warm-start weights from")

    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="onecycle",
                   choices=["none", "onecycle"],
                   help="onecycle: warmup (10%%) + cosine decay to lr/100.")
    p.add_argument("--warmup_pct", type=float, default=0.1)

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=5,
                   help="Number of consecutive evals that must satisfy stop criterion.")
    p.add_argument("--pp_ar_threshold", type=float, default=0.999)
    p.add_argument("--bce_threshold", type=float, default=0.01)
    p.add_argument("--disable_early_stop", action="store_true")
    return p.parse_args()


def load_first_n_graphs(data_path: str, n: int) -> tuple[list[dict[str, Any]], bool]:
    """Load the first `n` graphs. Prefers `*.cache.pt` siblings when present
    (features already baked in). Returns (graphs, precomputed_flag).
    """
    shard_paths = sorted(glob.glob(os.path.join(data_path, "mis_shard_*.pt")))
    # Exclude cache files from the main list
    shard_paths = [p for p in shard_paths if not p.endswith(".cache.pt")]
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {data_path}")

    out: list[dict[str, Any]] = []
    precomputed_seen = True
    for sp in shard_paths:
        cp = sp.replace(".pt", ".cache.pt")
        if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(sp):
            payload = torch.load(cp, weights_only=False)
        else:
            payload = torch.load(sp, weights_only=False)
            precomputed_seen = False
        for g in payload["data"]:
            out.append(g)
            if len(out) >= n:
                return out, precomputed_seen
    raise RuntimeError(f"Only found {len(out)} graphs in {data_path}, need {n}")


def enhance_graph(sample: dict[str, Any], *, use_pe: bool,
                  use_enhanced_features: bool, pe_dim: int,
                  already_precomputed: bool) -> dict[str, Any]:
    """Attach precomputed features to a graph dict. No-op if already baked in."""
    s = dict(sample)  # shallow copy
    # If features were loaded from *.cache.pt, we keep them as-is regardless of
    # the CLI flags. Otherwise, optionally compute them now.
    if already_precomputed:
        return s
    if use_enhanced_features:
        s["x"] = compute_node_features(s["edge_index"], s["n"], s["x"])
    if use_pe:
        s["pe"] = compute_laplacian_pe(s["edge_index"], s["n"], pe_dim=pe_dim)
    return s


def to_batch(samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Collate into the dict format expected by GraphTransformerTRM."""
    data_list = []
    for s in samples:
        d = Data(
            x=s["x"],
            edge_index=s["edge_index"],
            y=s["y"],
            num_nodes=s["n"],
        )
        d.opt_value = torch.tensor([s["opt_value"]])
        if "pe" in s:
            d.pe = s["pe"]
        data_list.append(d)
    b = Batch.from_data_list(data_list)
    out = {
        "x": b.x.to(device),
        "edge_index": b.edge_index.to(device),
        "batch": b.batch.to(device),
        "y": b.y.to(device),
        "ptr": b.ptr.to(device),
        "num_graphs": b.num_graphs,
        "opt_value": b.opt_value.to(device),
    }
    if hasattr(b, "pe") and b.pe is not None:
        out["pe"] = b.pe.to(device)
    return out


def compute_all_metrics(probs: torch.Tensor, labels: torch.Tensor,
                        edge_index: torch.Tensor, batch_vec: torch.Tensor,
                        ptr: torch.Tensor) -> dict[str, float]:
    """Full metric dict: raw + post-processed + confusion-matrix percentages.

    Keys:
      raw_pred_size, raw_opt_size, raw_gap, raw_approx_ratio, raw_feasibility,
      tp_pct, tn_pct, fp_pct, fn_pct, precision, recall, f1,
      pp_pred_size, pp_gap, pp_approx_ratio, pp_feasibility.
    """
    raw = compute_metrics(probs, edge_index, labels, batch_vec=batch_vec, ptr=ptr)
    pp = compute_pp_metrics(probs, edge_index, labels, batch_vec=batch_vec, ptr=ptr)

    # compute_metrics returns tp/tn/fp/fn as per-graph averages (counts).
    # Convert to % of nodes per graph, so the four sum to ~100%.
    num_graphs = int(len(ptr) - 1)
    nodes_per_graph = float(batch_vec.numel()) / max(1, num_graphs)
    denom = max(nodes_per_graph, 1e-8)
    tp_pct = 100.0 * raw["tp"] / denom
    tn_pct = 100.0 * raw["tn"] / denom
    fp_pct = 100.0 * raw["fp"] / denom
    fn_pct = 100.0 * raw["fn"] / denom

    return {
        # Raw (pre-post-processing)
        "raw_opt_size": raw["opt_size"],
        "raw_pred_size": raw["pred_size"],
        "raw_gap": raw["gap"],
        "raw_approx_ratio": raw["approx_ratio"],
        "raw_feasibility": raw["feasibility"],
        "raw_size_ratio": raw["pred_size"] / max(raw["opt_size"], 1e-8),
        # Confusion-matrix percentages of node count
        "tp_pct": tp_pct,
        "tn_pct": tn_pct,
        "fp_pct": fp_pct,
        "fn_pct": fn_pct,
        "precision": raw["precision"],
        "recall": raw["recall"],
        "f1": raw["f1"],
        # Post-processed (greedy decode -> always feasible)
        "pp_pred_size": pp["pp_pred_size"],
        "pp_gap": pp["pp_gap"],
        "pp_approx_ratio": pp["pp_approx_ratio"],
        "pp_feasibility": pp["pp_feasibility"],
        "pp_size_ratio": pp["pp_pred_size"] / max(raw["opt_size"], 1e-8),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[overfit_sl] device={device}")
    if device.type == "cuda":
        print(f"[overfit_sl] GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Load graphs (first N) and compute features once
    # ------------------------------------------------------------------
    t0 = time.time()
    raw_samples, already_precomputed = load_first_n_graphs(args.data_path, args.num_graphs)
    print(f"[overfit_sl] loaded {len(raw_samples)} graphs in {time.time() - t0:.2f}s "
          f"(precomputed_cache={already_precomputed})")
    for i, s in enumerate(raw_samples):
        print(f"  graph[{i}]: n={s['n']} edges={s['edge_index'].size(1)} opt={s['opt_value']}")

    t0 = time.time()
    samples = [
        enhance_graph(s, use_pe=bool(args.use_pe),
                      use_enhanced_features=bool(args.use_enhanced_features),
                      pe_dim=args.pe_dim,
                      already_precomputed=already_precomputed)
        for s in raw_samples
    ]
    print(f"[overfit_sl] feature extraction (PE={bool(args.use_pe)}, "
          f"enhanced={bool(args.use_enhanced_features)}, "
          f"precomputed={already_precomputed}): "
          f"{time.time() - t0:.2f}s")

    input_dim = samples[0]["x"].shape[1]
    pe_dim_actual = args.pe_dim if args.use_pe else 0
    print(f"[overfit_sl] input_dim={input_dim}, pe_dim={pe_dim_actual}")

    # ------------------------------------------------------------------
    # 2. Build batching strategy
    # ------------------------------------------------------------------
    N = len(samples)
    if args.batch_size > 0:
        bs = min(args.batch_size, N)
    else:
        # Default: N/25 graphs per mini-batch, at least 1, capped at 100.
        bs = max(1, min(100, N // 25)) if N >= 25 else N
    # Pre-collate the full set on GPU only when small (memory bound).
    if bs == N and N <= 50:
        cached_batch = to_batch(samples, device)
        print(f"[overfit_sl] pre-collated fixed batch of {bs} graphs kept on GPU")
    else:
        cached_batch = None
        print(f"[overfit_sl] streaming mini-batches: N={N}, batch_size={bs}, "
              f"~{(N + bs - 1) // bs} batches/epoch")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    if args.pos_weight is None:
        total_pos = sum(float(s["y"].sum().item()) for s in samples)
        total_neg = sum(float(s["y"].numel() - s["y"].sum().item()) for s in samples)
        pos_weight = total_neg / max(1.0, total_pos)
    else:
        pos_weight = args.pos_weight

    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "attn_dropout": args.dropout,
        "input_dim": input_dim,
        "pe_dim": pe_dim_actual,
        "pe_input_dim": args.pe_dim,
        "pos_weight": pos_weight,
        "feasibility_weight": args.feasibility_weight,
        "feasibility_loss_type": args.feasibility_loss_type,
        "label_smoothing": 0.0,
        "selection_weight": 0.0,
        "track_steps": False,
    }
    model = GraphTransformerTRM(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[overfit_sl] model built, params={n_params/1e6:.2f}M, pos_weight={pos_weight:.2f}")

    if getattr(args, "init_from", None):
        print(f"[overfit_sl_warm] loading weights from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in ckpt:
                    sd = ckpt[k]
                    print(f"[overfit_sl_warm] using key '{k}' (n={len(sd)})")
                    break
            else:
                sd = ckpt
        else:
            sd = ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[overfit_sl_warm] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[overfit_sl_warm] first missing: {missing[:5]}")
        if unexpected:
            print(f"[overfit_sl_warm] first unexpected: {unexpected[:5]}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = None
    if args.lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr, total_steps=args.iterations,
            pct_start=args.warmup_pct, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=100.0,
        )
        print(f"[overfit_sl] OneCycleLR: max_lr={args.lr}, warmup_pct={args.warmup_pct}")

    # ------------------------------------------------------------------
    # 4. Optional wandb
    # ------------------------------------------------------------------
    wb = None
    if args.wandb:
        import wandb as _wb  # lazy
        wb = _wb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args) | {"input_dim": input_dim, "pe_dim_actual": pe_dim_actual,
                                  "num_params": n_params, "num_graphs": len(samples)},
            tags=["overfit", "sl", f"N{len(samples)}"],
        )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.checkpoint_path, exist_ok=True)
    best_approx = -1.0
    best_step = -1
    solved_at = -1
    t_start = time.time()

    # Index cycling for streaming mini-batches
    rng = np.random.default_rng(args.seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    ptr_idx = 0

    # Early-stop bookkeeping
    recent_pp_ar: collections.deque = collections.deque(maxlen=args.early_stop_patience)
    recent_bce: collections.deque = collections.deque(maxlen=args.early_stop_patience)
    recent_feas: collections.deque = collections.deque(maxlen=args.early_stop_patience)
    early_stopped_at = -1
    stop_reason = ""

    model.train()
    for it in range(1, args.iterations + 1):
        if cached_batch is not None:
            batch = cached_batch
        else:
            if ptr_idx + bs > N:
                rng.shuffle(indices)
                ptr_idx = 0
            ids = indices[ptr_idx:ptr_idx + bs]
            ptr_idx += bs
            sel = [samples[i] for i in ids]
            batch = to_batch(sel, device)

        carry = (
            torch.zeros(batch["x"].size(0), 1, device=device),
            torch.zeros(batch["x"].size(0), args.hidden_dim, device=device),
            0,
        )

        optim.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        last_metrics: dict = {}
        for _ in range(args.n_supervision):
            carry, loss, metrics, preds, all_finish = model(carry, batch)
            total_loss = total_loss + loss
            last_metrics = metrics
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if it % args.log_every == 0 or it == 1 or it == args.iterations:
            probs = preds["preds"].squeeze()
            m = compute_all_metrics(
                probs, batch["y"].float(), batch["edge_index"],
                batch["batch"], batch["ptr"],
            )
            elapsed = time.time() - t_start
            it_per_s = it / max(1e-6, elapsed)
            loss_val = float(total_loss.detach())
            bce = float(last_metrics.get("loss_bce", torch.tensor(0.0)).detach()) if last_metrics else 0.0
            feas_l = float(last_metrics.get("loss_feasibility", torch.tensor(0.0)).detach()) if last_metrics else 0.0

            print(
                f"[it {it:5d}/{args.iterations}] "
                f"loss={loss_val:.4f} bce={bce:.4f} feas_l={feas_l:.4f} "
                f"raw[AR={m['raw_approx_ratio']:.4f} feas={m['raw_feasibility']:.3f} "
                f"pred={m['raw_pred_size']:.1f}/opt={m['raw_opt_size']:.1f}] "
                f"pp[AR={m['pp_approx_ratio']:.4f} pred={m['pp_pred_size']:.1f}] "
                f"cm%[tp={m['tp_pct']:.1f} tn={m['tn_pct']:.1f} "
                f"fp={m['fp_pct']:.1f} fn={m['fn_pct']:.1f}] "
                f"F1={m['f1']:.3f} it/s={it_per_s:.2f}",
                flush=True,
            )

            cur_lr = optim.param_groups[0]["lr"]
            if wb is not None:
                wb.log({
                    "iter": it,
                    "loss/total": loss_val,
                    "loss/bce": bce,
                    "loss/feasibility": feas_l,
                    "it_per_s": it_per_s,
                    "lr": cur_lr,
                    **{f"metrics/{k}": v for k, v in m.items()},
                })

            # Best checkpoint = highest pp_approx_ratio (always feasible, capped at 1.0)
            score = m["pp_approx_ratio"]
            if score > best_approx:
                best_approx = score
                best_step = it
                ckpt_path = os.path.join(args.checkpoint_path, "best.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "iter": it,
                    "metrics": m,
                }, ckpt_path)

            # Solved gate: raw AR == 1.0 AND feasibility == 1.0 AND size_ratio == 1.0
            if (
                m["raw_approx_ratio"] >= 0.9999
                and m["raw_feasibility"] >= 0.9999
                and abs(m["raw_size_ratio"] - 1.0) < 1e-3
                and solved_at < 0
            ):
                solved_at = it
                print(f"[overfit_sl] SOLVED at iteration {it} "
                      f"(raw AR=1.0, feasibility=1.0, size_ratio=1.0)", flush=True)

            # Early stop criteria
            recent_pp_ar.append(m["pp_approx_ratio"])
            recent_bce.append(bce)
            recent_feas.append(m["pp_feasibility"])
            if not args.disable_early_stop and len(recent_pp_ar) == args.early_stop_patience:
                cond_ar = all(v >= args.pp_ar_threshold for v in recent_pp_ar)
                cond_bce = all(
                    b < args.bce_threshold and f >= 0.9999
                    for b, f in zip(recent_bce, recent_feas)
                )
                if cond_ar or cond_bce:
                    early_stopped_at = it
                    stop_reason = "pp_ar" if cond_ar else "bce+feas"
                    print(f"[overfit_sl] EARLY STOP at iter {it} (reason={stop_reason}) "
                          f"last pp_AR={list(recent_pp_ar)} last bce={list(recent_bce)}",
                          flush=True)
                    break

    elapsed = time.time() - t_start
    print(f"[overfit_sl] done in {elapsed:.1f}s "
          f"({args.iterations/elapsed:.2f} it/s)")
    print(f"[overfit_sl] best pp_AR={best_approx:.4f} at iter {best_step}")
    print(f"[overfit_sl] solved_at={solved_at}")
    print(f"[overfit_sl] early_stopped_at={early_stopped_at} reason={stop_reason}")

    if wb is not None:
        wb.summary["best_pp_approx_ratio"] = best_approx
        wb.summary["best_iter"] = best_step
        wb.summary["solved_at"] = solved_at
        wb.summary["early_stopped_at"] = early_stopped_at
        wb.summary["stop_reason"] = stop_reason
        wb.summary["total_seconds"] = elapsed
        wb.finish()


if __name__ == "__main__":
    main()
