"""One-shot decoder-only baseline evaluator.

Computes greedy-decoder pp_approx_ratio on a test shard directory using a
non-model probability source (degree-normalized). Quantifies how much of the
periodic-eval `pp_AR` numbers come from the structural prior + decoder vs the
model itself.

Use this to back-fill `decoder_AR` and `model_lift` columns in
`docs/BENCHMARK.md` for runs that pre-date the in-loop instrumentation.

Example:
    python -m experiments.decoder_baseline \
        --eval_data_path data/difusco_benchmark/datasets/satlib/test \
        --label SAT
"""

from __future__ import annotations

import argparse
import time

import torch

from experiments.overfit_sl_pe import (
    compute_decoder_baseline,
    load_eval_samples,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_data_path", type=str, required=True)
    p.add_argument("--eval_num_graphs", type=int, default=10_000_000)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--use_pe", type=int, default=1)
    p.add_argument("--use_enhanced_features", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=16)
    p.add_argument("--label", type=str, default="?", help="Dataset label for the printed summary line.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[decoder_baseline] device={device}", flush=True)

    t0 = time.time()
    eval_samples = load_eval_samples(
        args.eval_data_path,
        args.eval_num_graphs,
        use_pe=bool(args.use_pe),
        use_enhanced_features=bool(args.use_enhanced_features),
        pe_dim=args.pe_dim,
    )
    print(f"[decoder_baseline] loaded {len(eval_samples)} graphs from {args.eval_data_path} in {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    m = compute_decoder_baseline(eval_samples, device, args.eval_batch_size, kind="degree")
    print(f"[decoder_baseline] computed in {time.time() - t0:.1f}s", flush=True)
    print(
        f"[{args.label}] decoder-only (degree) pp_AR={m['pp_approx_ratio']:.4f} "
        f"pp_feas={m['pp_feasibility']:.4f} "
        f"raw_AR={m['raw_approx_ratio']:.4f} "
        f"f1={m['f1']:.4f} "
        f"(N={len(eval_samples)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
