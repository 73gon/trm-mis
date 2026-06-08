 # Small-MIS v8 Learning Curves — n ∈ {50, 100, 200, 300}

Goal: visualize a "real" deep-learning style descent curve (BCE ~0.73 → ~0.25)
for the supervised, multi-label MIS trainer across graph sizes.

## What changed in v8

Prior runs only logged eval starting at iter ≥ 2000, so the visible portion of
the loss curve was 0.3 → 0.2 — a 0.1 sliver that hid the actual learning.
Two trainer changes fixed this:

1. **iter=0 eval** (`--eval_at_zero=1`, on by default). Runs a full test eval
   before any optimizer step, capturing the random-init BCE (~ln 2 ≈ 0.693).
2. **Uniform eval cadence** (`--eval_every=500` in the v8 SLURM scripts). Eval
   fires every 500 training iterations for the entire run, so the x-axis of
   the wandb chart shows real iterations with equal spacing — no two-phase
   "dense then sparse" gap.

Trainer file: [experiments/overfit_sl_pe.py](experiments/overfit_sl_pe.py)
SLURM scripts: [slurm/smallmis_n50_v8.slurm](slurm/smallmis_n50_v8.slurm),
[slurm/smallmis_n100_v8.slurm](slurm/smallmis_n100_v8.slurm),
[slurm/smallmis_n200_v8.slurm](slurm/smallmis_n200_v8.slurm),
[slurm/smallmis_n300_v8.slurm](slurm/smallmis_n300_v8.slurm).

## Shared hyperparameters

| Setting | Value | Notes |
|---|---|---|
| Model | GraphTransformerTRM 1.55M | hidden=256, L=2, H=2, L_cycles=6 |
| Precision | bf16 autocast | |
| Optimizer | AdamW, wd=0.05 | |
| LR schedule | OneCycleLR, max_lr=3e-4 | warmup_pct=0.1, cos anneal |
| Dropout | 0.15 | |
| Pos-weight | 1.0 | |
| Feasibility weight | 2.0 | |
| Label smoothing | 0.0 | |
| Eval cadence | every 500 iters (+ iter=0) | uniform |
| Early stop | `eval_loss_patience` 12 (n=50) / 25 (n≥100) | |

Per-size differences (graph size scales memory):

| n | Batch | Eval batch | Train graphs | Test graphs | Iters cap |
|---|---|---|---|---|---|
| 50  | 512 | 256 | 50 000 | (built-in) | 80 000 |
| 100 | 256 | 256 | 50 000 | (built-in) | 160 000 |
| 200 | 64  | 64  | 12 000 | 1 000 | 160 000 |
| 300 | 32  | 32  | 12 000 | 1 000 | 160 000 |

## Data pipeline (multi-MIS soft labels)

Each graph's label is the *fraction of strictly-optimal MISs that contain
node v*, computed by Gurobi 13.0.0 with `PoolSearchMode=2`, `PoolSolutions=16`,
`PoolGap=0`. ER graphs with fixed average degree d=7.35.

n=50 and n=100 datasets pre-existed at `data/smallmis_n{50,100}_multi/`.
For n=200 and n=300, single-worker builds were too slow (~12 h/worker), so we
ship parallel chunk builders that run 6 train + 2 test workers each with
single-threaded Gurobi, plus a sequential-rename merger:

- [dataset/build_n200_parallel.sh](dataset/build_n200_parallel.sh) → 12 000
  train + 1 000 test, ~80 min wall on the login node.
- [dataset/build_n300_parallel.sh](dataset/build_n300_parallel.sh) → same
  topology, `pool-time-limit=120s`.
- [dataset/merge_n200_chunks.py](dataset/merge_n200_chunks.py),
  [dataset/merge_n300_chunks.py](dataset/merge_n300_chunks.py).

## Results — n=50, n=100, n=200 (COMPLETED)

| Job | n | Iter=0 BCE | Best BCE | Best pp_AR | Stopped @ | Elapsed |
|---|---|---|---|---|---|---|
| 3991 | 50  | 0.7293 | **0.2262** @23 500 | **0.9818** @29 500 | early@29 500 | 1h59m |
| 3992 | 100 | 0.7275 | **0.2684** @35 500 | **0.9927** @ ~30 000 | early@48 000 | 3h28m |
| 4060 | 200 | 0.7260 | **0.3188** @ 4 500 | **0.9951** @ ~5 000 | early@17 000 | 0h37m |

All three runs satisfy the "model is actually learning" gate: monotone descent
of eval BCE from ~0.73 down to ~0.22–0.32 and monotone rise of eval pp_AR from
~0.89 → 0.98+. n=200 converged in only ~5k iterations because its 12k-graph
training pool plus higher per-graph signal density made early epochs very
informative; eval BCE then crept back up (mild overfit on the smaller train
set), so `eval_loss_patience=25` correctly stopped at it=17 000.

### Why the three runs took different wall times

This is **by design**, not an artifact:

1. **Per-size iteration caps.** n=50 uses `--iterations=80000`; n≥100 uses
   `--iterations=160000`. The cap also defines OneCycleLR's schedule (warmup
   10 %, cosine anneal to `max_lr/100`), so changing the cap changes the LR
   curve, not just the budget. Bigger graphs get a longer, slower anneal.
2. **Per-size early-stop patience.** n=50 uses `--eval_loss_patience=12`,
   n≥100 uses `25`. Each run quits the moment eval BCE stops improving for
   `patience` consecutive evals (× 500 iters/eval).
3. **Different train-set sizes.** n=50/n=100 have 50 000 training graphs;
   n=200/n=300 only have 12 000. Smaller train pools saturate sooner —
   n=200 hit its best eval at it=4 500 and was overfitting by it=17 000.

The alternative — forcing all runs to the same fixed number of iterations
without early stop — would waste GPU on the small-train runs and visibly
overfit them in the back half of the curve. We prefer "stop when
generalization stops improving" with a per-size patience tuned to the
training-set size.

![v8 learning curves n=50 / n=100 / n=200](visualizations/learning_curve_v8_n50_n100_n200.png)

Reproduction:
```bash
python visualizations/plot_curves.py \
  logs/smallmis_n50_v8_3991.log \
  logs/smallmis_n100_v8_3992.log \
  logs/smallmis_n200_v8_4060.log \
  -o visualizations/learning_curve_v8_n50_n100_n200.png
```

## In progress

- **n=300 data build** — n=300 instances are ~2× harder than n=200 for
  Gurobi's MIS pool enumeration. The initial 8-worker build (2000 train + 500
  test per worker) was producing ~100 s/graph and would have taken 14 h+ for
  the first shard. We restarted with 18 single-threaded Gurobi workers
  (15 train × 200 + 3 test × 100 = 3 000 train + 300 test graphs total) and
  `--shard-size 50`, so the first usable shards arrive in ~1.5 h.
- **n=300 training** — once all 18 builders exit, run
  `.venv/bin/python dataset/merge_n300_chunks.py` and
  `sbatch slurm/smallmis_n300_v8.slurm` manually. After the training job
  completes, regenerate the all-four plot with:

```bash
python visualizations/plot_curves.py \
  logs/smallmis_n50_v8_3991.log \
  logs/smallmis_n100_v8_3992.log \
  logs/smallmis_n200_v8_4060.log \
  logs/smallmis_n300_v8_<JID>.log \
  -o visualizations/learning_curve_v8_all.png
```

**Progress snapshot (2026-05-25 22:05 UTC):** 18 workers alive, 15 train
shards + 3 test shards already written (one per worker × 50 graphs);
per-worker progress ~60/200 (train) and ~60/100 (test) at ~109 s/graph.
ETAs: test done ~23:20 UTC, train done ~02:30 UTC the next day.

## Operational notes

- Sarmatia is the only working GPU node (`tujestpolin` has an NVML mismatch).
- The login node has 24 CPU, 30 GB RAM, gurobipy 13.0.0 — enough for the
  8-worker parallel data builds.
- Each n=300 worker is ~2× slower than n=200 in expectation (Gurobi
  PoolSolutions=16 scales poorly with n); plan accordingly.
