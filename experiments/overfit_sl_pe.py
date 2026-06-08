"""Periodic-Eval variant of overfit_sl.py.

Adds:
* `--eval_data_path <dir>` — directory of test shards to eval on
* `--eval_every <iters>` — run a full test eval every this many iters (default 5000)
* `--eval_max_steps <int>` — TRM inner-loop cap for eval forward (default 12)
* `--eval_batch_size <int>` — batch size for eval (default 16)
* `--benchmark_md <path>` — append a row to this markdown file at run end
* `--benchmark_dataset_label <str>` — label for the row (e.g. "SAT" or "ER")

Logs `eval_test/*` metrics to wandb at the same `step=it` as `metrics/*`, so
they overlay on the same chart.

Tracks the best checkpoint by **eval_test/pp_approx_ratio** (not train), since
that's what we actually care about. Saves best.pt + final.pt.

Original overfit_sl.py is untouched. Reuses helpers (`load_first_n_graphs`,
`enhance_graph`, `to_batch`, `compute_all_metrics`).
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import glob
import os
import time

import numpy as np
import torch

from experiments.overfit_sl import (
    compute_all_metrics,
    enhance_graph,
    load_first_n_graphs,
    to_batch,
)
from models.graph_transformer_trm import GraphTransformerTRM


def _build_optimizer(model, args) -> torch.optim.Optimizer:
    """AdamW (default) or Muon (hybrid: Muon for 2D weight matrices, AdamW for the rest)."""
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer == "muon":
        try:
            from muon import Muon  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Muon requested but `muon-optimizer` is not installed. "
                "Install with: pip install git+https://github.com/KellerJordan/Muon"
            ) from e
        muon_params, adamw_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Muon only handles 2D weight matrices (no bias/norm/embedding tables).
            if p.ndim == 2 and "embed" not in n.lower() and "embedding" not in n.lower():
                muon_params.append(p)
            else:
                adamw_params.append(p)
        # Standard hybrid: Muon for 2D weights, AdamW for the rest.
        opt = Muon(
            muon_params, lr=args.lr,
            adamw_params=adamw_params, adamw_lr=args.lr,
            adamw_betas=(args.beta1, args.beta2), adamw_wd=args.weight_decay,
        )
        return opt
    raise ValueError(f"unknown --optimizer {args.optimizer}")


def _sample_gpu_util(duration_s: float = 60.0, interval_s: float = 5.0) -> dict:
    """Poll nvidia-smi for util/mem over `duration_s`. Returns mean/min."""
    import subprocess
    import time as _t
    utils, mems = [], []
    t_end = _t.time() + duration_s
    while _t.time() < t_end:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip()
            for line in out.splitlines():
                u, m = line.split(",")
                utils.append(float(u.strip()))
                mems.append(float(m.strip()))
        except Exception:
            break
        _t.sleep(interval_s)
    if not utils:
        return {}
    import statistics as _st
    return {
        "gpu/util_pct_mean": float(_st.mean(utils)),
        "gpu/util_pct_min": float(min(utils)),
        "gpu/util_pct_max": float(max(utils)),
        "gpu/mem_mb_mean": float(_st.mean(mems)),
        "gpu/mem_mb_max": float(max(mems)),
        "gpu/samples": float(len(utils)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--num_graphs", type=int, default=25000)
    p.add_argument("--iterations", type=int, default=400_000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Architecture
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

    # Precision / optimizer
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"], help="Forward/loss autocast dtype.")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="Optimizer. 'muon' requires `pip install muon-optimizer`.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--gpu_util_poll", action="store_true", help="Sample nvidia-smi for 60s after iter 200 and log gpu/util_pct, gpu/mem_mb to wandb.")

    # Features
    p.add_argument("--use_pe", type=int, default=1)
    p.add_argument("--use_enhanced_features", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=16)

    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="onecycle", choices=["none", "onecycle"])
    p.add_argument("--warmup_pct", type=float, default=0.1)

    # Logging / output
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/overfit_sl_pe")
    p.add_argument("--run_name", type=str, default="overfit_sl_pe")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="MIS-TRM")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    # Periodic eval (NEW)
    p.add_argument("--eval_data_path", type=str, required=True, help="Directory of test shards (mis_shard_*.pt).")
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--eval_max_steps", type=int, default=12)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--eval_num_graphs", type=int, default=10_000_000, help="Cap on # eval graphs; default = all.")
    # Uniform eval cadence (+ optional iter=0 to capture random-init loss).
    p.add_argument("--eval_at_zero", type=int, default=1,
                   help="If 1, run a test eval at iter=0 BEFORE any optimizer step (captures random-init loss).")

    # Benchmark MD (NEW)
    p.add_argument("--benchmark_md", type=str, default=None, help="Append a final-results row to this markdown file.")
    p.add_argument("--benchmark_dataset_label", type=str, default="?")

    # Decoder-only baseline (NEW): quantifies how much of pp_AR is the model
    # vs the greedy decoder + structural prior. Computed once on eval set.
    p.add_argument(
        "--decoder_baseline",
        type=str,
        default="degree",
        choices=["none", "degree"],
        help="Baseline 'probs' source for decoder-only pp_AR. 'degree' = normalized node degree (matches greedy bias).",
    )

    # Early stop (kept but more conservative — mostly we want the full curve)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--pp_ar_threshold", type=float, default=0.999)
    p.add_argument("--bce_threshold", type=float, default=0.01)
    p.add_argument("--disable_early_stop", action="store_true")
    # Eval-loss based early stop (separate from pp_AR plateau).
    p.add_argument("--eval_loss_patience", type=int, default=0,
                   help="If >0, stop after N consecutive evals with no eval-loss improvement.")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="BCE label smoothing in [0, 0.5). Helps prevent overconfidence/eval-loss divergence.")
    return p.parse_args()


def load_eval_samples(data_path: str, max_n: int, *, use_pe: bool, use_enhanced_features: bool, pe_dim: int) -> list[dict]:
    """Load ALL eval shards (with cache fallback) up to max_n graphs."""
    try:
        raw, precomputed = load_first_n_graphs(data_path, max_n)
    except RuntimeError:
        shard_paths = sorted(p for p in glob.glob(os.path.join(data_path, "mis_shard_*.pt")) if not p.endswith(".cache.pt"))
        raw = []
        precomputed = True
        for sp in shard_paths:
            cp = sp.replace(".pt", ".cache.pt")
            if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(sp):
                payload = torch.load(cp, weights_only=False)
            else:
                payload = torch.load(sp, weights_only=False)
                precomputed = False
            raw.extend(payload["data"])
    raw = raw[:max_n]
    return [enhance_graph(s, use_pe=use_pe, use_enhanced_features=use_enhanced_features, pe_dim=pe_dim, already_precomputed=precomputed) for s in raw]


def run_eval(model: GraphTransformerTRM, samples: list[dict], device: torch.device, *, max_steps: int, batch_size: int) -> dict:
    """Run a forward pass on every eval graph; aggregate compute_all_metrics."""
    model.eval()
    n = len(samples)
    bs = max(1, batch_size)
    per_batch_metrics: list[dict] = []
    graph_counts: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = samples[i : i + bs]
            batch = to_batch(chunk, device)
            carry = model.initial_carry(batch)
            all_finish = False
            preds = None
            steps = 0
            last_metrics: dict = {}
            while not all_finish and steps < max_steps:
                carry, _l, _m, preds, all_finish = model(carry, batch)
                last_metrics = _m
                steps += 1
            probs = preds["preds"].squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            m = compute_all_metrics(probs, batch["y"].float(), batch["edge_index"], batch["batch"], batch["ptr"])
            # Capture model losses from the last forward pass so eval reports
            # bce / feasibility / total alongside metrics.
            for k in ("loss_total", "loss_bce", "loss_feasibility"):
                v = last_metrics.get(k)
                if v is not None:
                    m[k] = float(v.item()) if hasattr(v, "item") else float(v)
            per_batch_metrics.append(m)
            graph_counts.append(len(chunk))
    model.train()
    total = sum(graph_counts)
    keys = [k for k, v in per_batch_metrics[0].items() if isinstance(v, (int, float))]
    agg = {k: sum(m[k] * c for m, c in zip(per_batch_metrics, graph_counts)) / max(total, 1) for k in keys}
    agg["eval_seconds"] = time.time() - t0
    agg["eval_num_graphs"] = total
    return agg


def compute_decoder_baseline(samples: list[dict], device: torch.device, batch_size: int, kind: str = "degree") -> dict:
    """Decoder-only baseline: feed structural-prior probs (no model) into the
    same greedy decoder + metric pipeline. Quantifies how much pp_AR comes from
    the decoder + graph structure alone, vs from the model.

    For 'degree': probs[i] = degree(i) / max_degree(graph), so the greedy
    decoder picks high-degree nodes first within each graph. This matches the
    degree-based heuristic baseline commonly cited for MIS.
    """
    if kind == "none":
        return {}
    n = len(samples)
    bs = max(1, batch_size)
    per_batch: list[dict] = []
    counts: list[int] = []
    for i in range(0, n, bs):
        chunk = samples[i : i + bs]
        batch = to_batch(chunk, device)
        edge_index = batch["edge_index"]
        num_nodes = int(batch["x"].shape[0])
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))
        # normalize per-graph by max degree
        ptr = batch["ptr"]
        probs = torch.zeros_like(deg)
        for g in range(int(len(ptr) - 1)):
            s, e = int(ptr[g]), int(ptr[g + 1])
            d = deg[s:e]
            mx = float(d.max().item()) if d.numel() > 0 else 1.0
            probs[s:e] = d / max(mx, 1e-8)
        m = compute_all_metrics(probs, batch["y"].float(), edge_index, batch["batch"], ptr)
        per_batch.append(m)
        counts.append(len(chunk))
    total = sum(counts)
    keys = [k for k, v in per_batch[0].items() if isinstance(v, (int, float))]
    return {k: sum(m[k] * c for m, c in zip(per_batch, counts)) / max(total, 1) for k in keys}


def append_benchmark_row(md_path: str, row: dict) -> None:
    """Append a row to the benchmark markdown table. Creates header if absent.

    File-locked so two parallel jobs can write safely.
    """
    header = (
        "| Run | Dataset | Iters | Epochs | Best iter | "
        "train pp_AR | train F1 | train bce | "
        "test pp_AR | decoder_AR | model_lift | test pp_feas | test raw_AR | test F1 | "
        "test prec | test rec | test tp% | test tn% | test fp% | test fn% | "
        "wall (s) | it/s |\n"
        "|-----|---------|------:|-------:|----------:|"
        "------------:|---------:|----------:|"
        "-----------:|-----------:|-----------:|------------:|------------:|---------:|"
        "----------:|---------:|---------:|---------:|---------:|---------:|"
        "---------:|-----:|\n"
    )
    cells = [
        row["run_name"],
        row["dataset"],
        f"{row['iters']:,}",
        f"{row['epochs']:.2f}",
        f"{row['best_iter']:,}",
        f"{row['train_pp_AR']:.4f}",
        f"{row['train_f1']:.4f}",
        f"{row['train_bce']:.4f}",
        f"{row['test_pp_AR']:.4f}",
        f"{row.get('decoder_AR', float('nan')):.4f}" if row.get("decoder_AR") is not None else "—",
        f"{row.get('model_lift', float('nan')):+.4f}" if row.get("model_lift") is not None else "—",
        f"{row['test_pp_feas']:.4f}",
        f"{row['test_raw_AR']:.4f}",
        f"{row['test_f1']:.4f}",
        f"{row['test_precision']:.4f}",
        f"{row['test_recall']:.4f}",
        f"{row['test_tp_pct']:.2f}",
        f"{row['test_tn_pct']:.2f}",
        f"{row['test_fp_pct']:.2f}",
        f"{row['test_fn_pct']:.2f}",
        f"{row['wall_seconds']:.0f}",
        f"{row['it_per_s']:.2f}",
    ]
    line = "| " + " | ".join(cells) + " |\n"
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write("# Benchmark — periodic-eval training runs\n\n")
                f.write("Auto-generated by `experiments/overfit_sl_pe.py`. Each row is one run.\n\n")
                f.write(header)
            f.write(line)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pe] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[pe] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 1. Load training set
    t0 = time.time()
    raw_samples, already_precomputed = load_first_n_graphs(args.data_path, args.num_graphs)
    print(f"[pe] loaded {len(raw_samples)} train graphs in {time.time() - t0:.2f}s (precomputed={already_precomputed})", flush=True)

    samples = [
        enhance_graph(s, use_pe=bool(args.use_pe), use_enhanced_features=bool(args.use_enhanced_features), pe_dim=args.pe_dim, already_precomputed=already_precomputed)
        for s in raw_samples
    ]
    input_dim = int(samples[0]["x"].shape[1])
    pe_dim_actual = int(samples[0]["pe"].shape[1]) if "pe" in samples[0] else 0
    print(f"[pe] input_dim={input_dim} pe_dim={pe_dim_actual}", flush=True)

    # 2. Load eval set
    t0 = time.time()
    eval_samples = load_eval_samples(
        args.eval_data_path,
        args.eval_num_graphs,
        use_pe=bool(args.use_pe),
        use_enhanced_features=bool(args.use_enhanced_features),
        pe_dim=args.pe_dim,
    )
    print(f"[pe] loaded {len(eval_samples)} eval graphs from {args.eval_data_path} in {time.time() - t0:.2f}s", flush=True)

    # 2b. Decoder-only baseline (computed once, used for model_lift on every eval)
    baseline_pp_ar: float | None = None
    baseline_metrics: dict = {}
    if args.decoder_baseline != "none":
        t0 = time.time()
        baseline_metrics = compute_decoder_baseline(eval_samples, device, args.eval_batch_size, kind=args.decoder_baseline)
        baseline_pp_ar = float(baseline_metrics.get("pp_approx_ratio", 0.0))
        print(
            f"[pe] decoder-only baseline ({args.decoder_baseline}): "
            f"pp_AR={baseline_pp_ar:.4f} "
            f"raw_AR={baseline_metrics.get('raw_approx_ratio', 0.0):.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    N = len(samples)
    bs = max(1, min(args.batch_size, N))

    # 3. Build model
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
        "label_smoothing": args.label_smoothing,
        "selection_weight": 0.0,
        "track_steps": False,
    }
    model = GraphTransformerTRM(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pe] model params={n_params / 1e6:.2f}M, pos_weight={pos_weight:.2f}", flush=True)

    optim = _build_optimizer(model, args)
    scheduler = None
    if args.lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr, total_steps=args.iterations, pct_start=args.warmup_pct, anneal_strategy="cos", div_factor=25.0, final_div_factor=100.0
        )
        print(f"[pe] OneCycleLR: max_lr={args.lr}, warmup_pct={args.warmup_pct}", flush=True)

    wb = None
    if args.wandb:
        import wandb as _wb

        wb = _wb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args) | {"input_dim": input_dim, "pe_dim_actual": pe_dim_actual, "num_params": n_params, "num_train_graphs": N, "num_eval_graphs": len(eval_samples)},
            tags=["overfit", "sl", "pe", f"N{N}", args.benchmark_dataset_label],
        )

    # 4. Training loop
    os.makedirs(args.checkpoint_path, exist_ok=True)
    indices = np.arange(N)
    rng.shuffle(indices)
    ptr_idx = 0

    best_test_pp_ar = -1.0
    best_test_metrics: dict = {}
    best_train_metrics_at_best: dict = {}
    best_iter = -1

    recent_pp_ar: collections.deque = collections.deque(maxlen=args.early_stop_patience)
    early_stopped_at = -1
    best_eval_loss = float("inf")
    evals_since_loss_improve = 0
    stop_reason = ""

    last_train_metrics: dict = {}
    last_train_loss = 0.0
    last_train_bce = 0.0
    last_train_feas_l = 0.0

    # Mixed precision (bf16 autocast). bf16 needs no GradScaler.
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    autocast_enabled = args.precision != "fp32"
    if autocast_enabled:
        print(f"[pe] autocast: cuda/{args.precision}", flush=True)

    # GPU util sampler thread (one-shot, runs once after iter 200)
    util_thread = None
    util_result: dict = {}
    if args.gpu_util_poll and device.type == "cuda":
        import threading
        def _poll():
            util_result.update(_sample_gpu_util(duration_s=60.0, interval_s=5.0))
        util_thread = threading.Thread(target=_poll, daemon=True)

    # Helper: run one eval pass and print the [eval-test] line. Returns the agg dict.
    def _do_eval(cur_it: int) -> dict:
        was_training = model.training
        model.eval()
        agg = run_eval(model, eval_samples, device, max_steps=args.eval_max_steps, batch_size=args.eval_batch_size)
        if was_training:
            model.train()
        lift_str = f" lift={agg['pp_approx_ratio'] - baseline_pp_ar:+.4f}" if baseline_pp_ar is not None else ""
        loss_str = ""
        if "loss_total" in agg:
            loss_str = f" loss={agg['loss_total']:.4f} bce={agg['loss_bce']:.4f} feas_l={agg['loss_feasibility']:.4f}"
        print(
            f"[eval-test it={cur_it}] pp_AR={agg['pp_approx_ratio']:.4f}{lift_str} "
            f"pp_feas={agg['pp_feasibility']:.4f} "
            f"raw_AR={agg['raw_approx_ratio']:.4f} F1={agg['f1']:.4f} "
            f"prec={agg['precision']:.4f} rec={agg['recall']:.4f}{loss_str} "
            f"({agg['eval_seconds']:.1f}s)",
            flush=True,
        )
        if wb is not None:
            log_d = {f"eval_test/{k}": v for k, v in agg.items()}
            log_d["iter"] = cur_it
            if baseline_pp_ar is not None:
                log_d["eval_test/decoder_only_pp_AR"] = baseline_pp_ar
                log_d["eval_test/model_lift"] = agg["pp_approx_ratio"] - baseline_pp_ar
            wb.log(log_d)
        return agg

    t_start = time.time()
    # Initial eval at iter=0 (random-init) BEFORE any optimizer step, so the full
    # learning curve (typically starting near BCE = ln(2) ≈ 0.693) is recorded.
    if args.eval_at_zero:
        _do_eval(0)
    model.train()
    for it in range(1, args.iterations + 1):
        if ptr_idx + bs > N:
            rng.shuffle(indices)
            ptr_idx = 0
        ids = indices[ptr_idx : ptr_idx + bs]
        ptr_idx += bs
        sel = [samples[i] for i in ids]
        batch = to_batch(sel, device)

        carry = model.initial_carry(batch)
        optim.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        last_metrics: dict = {}
        preds = None
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=autocast_dtype, enabled=autocast_enabled):
            for _ in range(args.n_supervision):
                carry, loss, metrics, preds, all_finish = model(carry, batch)
                total_loss = total_loss + loss
                last_metrics = metrics
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if scheduler is not None:
            scheduler.step()

        # Launch GPU util sampler thread once at iter 200 (after warm-up).
        if util_thread is not None and it == 200:
            util_thread.start()
        if util_thread is not None and not util_thread.is_alive() and util_result and "_gpu_logged" not in util_result:
            if wb is not None:
                wb.log(dict(util_result))
            print(f"[pe] gpu util: mean={util_result.get('gpu/util_pct_mean', 0):.1f}% "
                  f"min={util_result.get('gpu/util_pct_min', 0):.1f}% "
                  f"mem={util_result.get('gpu/mem_mb_mean', 0):.0f}MB", flush=True)
            util_result["_gpu_logged"] = True

        # Periodic train-batch logging
        if it % args.log_every == 0 or it == 1 or it == args.iterations:
            probs = preds["preds"].squeeze().float()
            m = compute_all_metrics(probs, batch["y"].float(), batch["edge_index"], batch["batch"], batch["ptr"])
            elapsed = time.time() - t_start
            it_per_s = it / max(1e-6, elapsed)
            loss_val = float(total_loss.detach())
            bce = float(last_metrics.get("loss_bce", torch.tensor(0.0)).detach()) if last_metrics else 0.0
            feas_l = float(last_metrics.get("loss_feasibility", torch.tensor(0.0)).detach()) if last_metrics else 0.0

            last_train_metrics = m
            last_train_loss = loss_val
            last_train_bce = bce
            last_train_feas_l = feas_l

            print(
                f"[it {it:6d}/{args.iterations}] loss={loss_val:.4f} bce={bce:.4f} "
                f"feas_l={feas_l:.4f} pp[AR={m['pp_approx_ratio']:.4f}] "
                f"raw[AR={m['raw_approx_ratio']:.4f}] F1={m['f1']:.3f} it/s={it_per_s:.2f}",
                flush=True,
            )

            cur_lr = optim.param_groups[0]["lr"]
            if wb is not None:
                wb.log(
                    {
                        "iter": it,
                        "loss/total": loss_val,
                        "loss/bce": bce,
                        "loss/feasibility": feas_l,
                        "it_per_s": it_per_s,
                        "lr": cur_lr,
                        **{f"metrics/{k}": v for k, v in m.items()},
                    },
                )

        # Periodic test eval — uniform cadence every eval_every iters.
        do_eval = (it % args.eval_every == 0) or (it == args.iterations)
        if do_eval:
            agg = _do_eval(it)

            # Best ckpt = highest test pp_AR
            if agg["pp_approx_ratio"] > best_test_pp_ar:
                best_test_pp_ar = agg["pp_approx_ratio"]
                best_test_metrics = dict(agg)
                best_train_metrics_at_best = dict(last_train_metrics) if last_train_metrics else {}
                best_train_metrics_at_best["loss_bce"] = last_train_bce
                best_iter = it
                ckpt_path = os.path.join(args.checkpoint_path, "best.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model_config,
                        "iter": it,
                        "test_metrics": agg,
                        "train_metrics": best_train_metrics_at_best,
                    },
                    ckpt_path,
                )

            recent_pp_ar.append(agg["pp_approx_ratio"])
            if not args.disable_early_stop and len(recent_pp_ar) == args.early_stop_patience:
                if all(v >= args.pp_ar_threshold for v in recent_pp_ar):
                    early_stopped_at = it
                    stop_reason = "test_pp_ar"
                    print(f"[pe] EARLY STOP at iter {it} (reason={stop_reason})", flush=True)
                    break

            # Eval-loss based early stop: detect calibration overfit.
            cur_eval_loss = float(agg.get("loss_bce", agg.get("loss_total", float("inf"))))
            if cur_eval_loss + 1e-6 < best_eval_loss:
                best_eval_loss = cur_eval_loss
                evals_since_loss_improve = 0
            else:
                evals_since_loss_improve += 1
            if (args.eval_loss_patience > 0
                    and not args.disable_early_stop
                    and evals_since_loss_improve >= args.eval_loss_patience):
                early_stopped_at = it
                stop_reason = "eval_loss_plateau"
                print(f"[pe] EARLY STOP at iter {it} (reason={stop_reason}, "
                      f"best_eval_loss={best_eval_loss:.4f}, evals_since_improve={evals_since_loss_improve})",
                      flush=True)
                break

    elapsed = time.time() - t_start
    iters_done = it
    epochs_done = iters_done * bs / max(N, 1)
    print(f"[pe] done in {elapsed:.1f}s ({iters_done / elapsed:.2f} it/s) epochs~{epochs_done:.2f}", flush=True)
    print(f"[pe] best test pp_AR={best_test_pp_ar:.4f} at iter {best_iter}", flush=True)

    # Final ckpt
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "iter": iters_done,
        },
        os.path.join(args.checkpoint_path, "final.pt"),
    )

    if wb is not None:
        wb.summary["best_test_pp_AR"] = best_test_pp_ar
        wb.summary["best_iter"] = best_iter
        wb.summary["epochs"] = epochs_done
        wb.summary["total_seconds"] = elapsed
        wb.summary["early_stopped_at"] = early_stopped_at
        wb.summary["stop_reason"] = stop_reason
        if baseline_pp_ar is not None:
            wb.summary["decoder_only_pp_AR"] = baseline_pp_ar
            wb.summary["best/model_lift"] = best_test_pp_ar - baseline_pp_ar
        for k, v in best_test_metrics.items():
            wb.summary[f"best/test/{k}"] = v
        for k, v in (best_train_metrics_at_best or {}).items():
            wb.summary[f"best/train/{k}"] = v
        wb.finish()

    if args.benchmark_md and best_test_metrics:
        row = {
            "run_name": args.run_name,
            "dataset": args.benchmark_dataset_label,
            "iters": iters_done,
            "epochs": epochs_done,
            "best_iter": best_iter,
            "train_pp_AR": float(best_train_metrics_at_best.get("pp_approx_ratio", 0.0)),
            "train_f1": float(best_train_metrics_at_best.get("f1", 0.0)),
            "train_bce": float(best_train_metrics_at_best.get("loss_bce", 0.0)),
            "test_pp_AR": float(best_test_metrics.get("pp_approx_ratio", 0.0)),
            "decoder_AR": baseline_pp_ar,
            "model_lift": (None if baseline_pp_ar is None else float(best_test_metrics.get("pp_approx_ratio", 0.0)) - baseline_pp_ar),
            "test_pp_feas": float(best_test_metrics.get("pp_feasibility", 0.0)),
            "test_raw_AR": float(best_test_metrics.get("raw_approx_ratio", 0.0)),
            "test_f1": float(best_test_metrics.get("f1", 0.0)),
            "test_precision": float(best_test_metrics.get("precision", 0.0)),
            "test_recall": float(best_test_metrics.get("recall", 0.0)),
            "test_tp_pct": float(best_test_metrics.get("tp_pct", 0.0)),
            "test_tn_pct": float(best_test_metrics.get("tn_pct", 0.0)),
            "test_fp_pct": float(best_test_metrics.get("fp_pct", 0.0)),
            "test_fn_pct": float(best_test_metrics.get("fn_pct", 0.0)),
            "wall_seconds": elapsed,
            "it_per_s": iters_done / max(elapsed, 1e-6),
        }
        append_benchmark_row(args.benchmark_md, row)
        print(f"[pe] appended benchmark row to {args.benchmark_md}", flush=True)


if __name__ == "__main__":
    main()
