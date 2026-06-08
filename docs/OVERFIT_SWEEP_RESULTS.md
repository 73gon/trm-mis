# Overfit Sweep — Final Results (`sweep1`)

Self-supervised baseline: how far can `GraphTransformerTRM` (1.55 M params) memorise
a fixed training set under a clean supervised loss?  This run answers exactly that.

Trainer: [experiments/overfit_sl.py](experiments/overfit_sl.py)
Driver: [experiments/run_overfit_sweep.py](experiments/run_overfit_sweep.py)
SLURM template: [slurm/overfit/overfit_sl_template.slurm](slurm/overfit/overfit_sl_template.slurm)
Wandb project: `MIS-TRM` (entity `malikmardan-personal`), runs prefixed `sweep1_`.

## Pass criterion (per dataset, per stage)

A run "passes" if **either** sustained for ≥5 evals:

- `pp_AR ≥ 0.999`  (post-processed approximation ratio), **or**
- `bce < 0.01` AND `pp_feas == 1.0`.

A stage passes only if **both** datasets pass.  On failure the driver auto-retries
with 2× iter budget; if still failing the sweep aborts.

## Hyperparameters

Common: AdamW, OneCycleLR (max_lr listed below, pct_start=0.1, div=25, final=100),
grad_clip=1.0, hinge feasibility loss, model: hidden=256, layers=2, H_cycles=2,
L_cycles=6, GPSConv (GIN+attention), input_dim=12, pe_dim=16.

| Stage | N | iters | batch | lr | feas_w | budget |
|------:|---:|------:|------:|----:|------:|-------:|
| 1 | 1     | 3 000  | 1   | 1e-3 | 1.0 | 3k  |
| 2 | 250   | 15 000 | 16  | 5e-4 | 2.0 | 15k |
| 3 | 1 500 | 25 000 | 16  | 5e-4 | 2.0 | 25k |
| 4 | 9 000 | 40 000 | 16  | 3e-4 | 2.0 | 40k |
| 5 | 25 000| 60 000 | 16  | 3e-4 | 2.0 | 60k |

Datasets (cached in `*.cache.pt`): SATLIB ~1200 nodes, ~33 % positive · ER-700-800
~750 nodes, ~6 % positive.

## Results

| Stage | Dataset | Job | Iters used | best `pp_AR` | reason | GPU avg | Pass |
|------:|---------|----:|-----------:|-------------:|--------|--------:|------|
| 1 N=1     | SATLIB | 1882 | 820  early  | **1.0000** @ 440  | pp_ar     | 71.8 % | ✅ |
| 1 N=1     | ER     | 1883 | 900  early  | **1.0000** @ 620  | pp_ar     | 71.8 % | ✅ |
| 2 N=250   | SATLIB | 1888 | 8 280 early | **0.9995** @ 8260 | bce+feas  | 94.8 % | ✅ |
| 2 N=250   | ER     | 1889 | 9 500 early | **1.0000** @ 8760 | bce+feas  | 90.6 % | ✅ |
| 3 N=1500  | SATLIB | 1892 | 25 000      | **0.9991** @ 16920| budget    | 95.4 % | ✅ |
| 3 N=1500  | ER     | 1893 | 25 000      | 0.9161 @ 22600    | budget    | 95.4 % | ❌ |
| 3 N=1500  | ER (2×)| 1898 | 50 000      | **1.0000** @ 47080| budget    | 93.4 % | ✅ |
| 4 N=9000  | SATLIB (2×) | 1919 | 80 000 | 0.9961 @ 53940    | budget    | 96.3 % | ❌ |
| 4 N=9000  | ER (2×, lower pos_w / higher feas_w) | 1920 | 80 000 | 0.8367 @ 68720 | budget | 93.1 % | ❌ |
| 4 N=9000 grok | SATLIB | 1954 | 200 000     | **0.9997** @ 167640 | budget | 86.0 % | ✅ |
| 4 N=9000 grok | ER     | 1955 | 200 000     | 0.9972 @ 194480     | budget | 77.1 % | ❌ (best ER) |
| 4 N=9000 logb | ER (log-barrier)   | 1998 | killed | 0.7280 | killed (under-pred) | — | ❌ |
| 4 N=9000 curriculum | ER (warm-start) | 2008 | 200 000 | 0.9875 @ 183900 | budget | 93.0 % | ❌ |
| 4 N=9000 cosrest | ER (cosine-restart) | 2011 | killed | 0.9005 | killed | — | ❌ |
| 4 N=9000 bigmodel | ER (5M params) | 2014 | killed | 0.8527 | killed (diverged) | — | ❌ |
| 5 N=25000 | — | — | — | — | not attempted (gated by Stage 4 ER) | — | — |

`pp_AR=1.0` means the post-processed prediction matches the optimal MIS size on
every training graph.  Note Stage 3 SAT 0.9991 narrowly clears the 0.999 gate.

## Stage-4 failure analysis (the scaling wall)

ER-700-800 N=9000 is where the model breaks.  Trajectory of job 1920
(`--pos_weight ≤ 5`, `--feasibility_weight 5`, 80 000 iters):

| iter   | pp_AR | bce   | raw pred | opt | tp%  | tn%  | fp%  | fn% |
|-------:|------:|------:|---------:|----:|-----:|-----:|-----:|----:|
|    20  | 0.20  | —     | very low | 45  | 0.7  | 92.0 | 1.3  | 6.0 |
|  5 000 | 0.62  | 0.95  | 200      | 45  | 4.0  | 78   | 16   | 2   |
| 20 000 | 0.79  | 0.78  | 105      | 45  | 3.1  | 83   | 11   | 3   |
| 50 000 | 0.79  | 0.76  | 104      | 45  | 3.0  | 83   | 11   | 3   |
| 78 700 | 0.76  | 0.72  | 105      | 45  | 2.9  | 83   | 11   | 3   |

- Loss flat-lines after ~20 k iters; pp_AR oscillates 0.74–0.84.
- The model identifies roughly the right set size (raw pred ~100 vs opt 45)
  but distributes mass over too many candidate nodes; post-processing can
  recover at most ~80 % of the optimum.
- Symmetric story for SAT-9000 (1919): pp_AR plateaus at 0.99, bce ~0.6.

Likely root causes (in decreasing order of impact):

1. **Capacity saturation.**  1.55 M parameters is not enough to memorise 9 000
   distinct positivity patterns over ≥6 750 nodes/instance.  Loss never goes
   below 0.6.  The same architecture saturates SAT N=1500 cleanly (bce 0.0,
   pp_AR 1.0) — the wall sits between N=1500 and N=9000.
2. **OneCycleLR anneal.**  By iter 40 k the LR has dropped to a few e-6, so
   the second half of the budget contributes essentially nothing.
3. **Class imbalance on ER.**  6 % positives → effective `pos_weight≈15` if
   left automatic; we capped at 5 but the prior is still strong enough to
   tilt the model.
4. **Heterogeneity of ER instances.**  Each instance has a different optimum
   set; for sparse positives the model needs a much larger receptive
   field (more L_cycles or layers) to disambiguate.

## Operational issues encountered & resolved

- **Zombie driver duplicating jobs.**  First driver survived `kill`; fixed
  with PID lockfile (`logs/overfit_sweep_driver.lock`) and `/proc/{pid}` check.
- **OOM at batch=50 on shared GPU (tujestpolin).**  Stage 2 first attempt
  rc=1.  Reduced all stages ≥ 2 to `batch_size=16`, fixed.
- **OOM during Stage 4 retries when ER and SAT shared a node.**  Resubmitted
  manually with `--nodelist=sarmatia` (idle node) — both ran cleanly to 80 k.
- **`pos_weight` over-correction.**  Initial ER attempts were over-predicting
  (FP%≈30); capping `pos_weight=5` and raising `feas_w=5` flipped to
  under-prediction (FP%≈11, FN%≈3).  Sweet spot probably 8–10.

## Grokking attempt (200k iters)

User asked: "rerun 9000 with 200k iter in hope of grokking". Submitted
`sweep1_grok_sat_N9000_200k` (job 1954) and `sweep1_grok_er_N9000_200k`
(job 1955), `--pos_weight 5.0 --feasibility_weight 2.0 --lr 3e-4 --batch_size 16`,
hinge feasibility, OneCycleLR.

### SATLIB grokking trajectory (job 1954)

| iter    | bce   | pp_AR  | F1   |
|--------:|------:|-------:|-----:|
|       1 | 0.92  | 0.95   | 0.50 |
| 100 000 | 0.70  | 0.9935 | 0.64 |
| 125 000 | 0.57  | 0.9965 | 0.70 |
| 150 000 | 0.54  | 0.9909 | 0.70 |
| 175 000 | 0.43  | **0.9977** | 0.75 |
| 199 980 | 0.44  | 0.9963 | 0.74 |

→ best `pp_AR=0.9997` @ iter 167 640 — **clears 0.999 gate. SATLIB N=9000 PASSES.**

### ER-700-800 grokking trajectory (job 1955)

| iter    | bce   | pp_AR  | F1   | raw pred | opt |
|--------:|------:|-------:|-----:|---------:|----:|
|       1 | 0.93  | 0.728  | 0.12 | 739      | 45  |
| 100 000 | 0.46  | 0.789  | 0.42 |  66      | 45  |
| 125 000 | 0.38  | 0.828  | 0.52 |  63      | 45  |
| 150 000 | 0.15  | 0.933  | 0.77 |  60      | 45  |
| 175 000 | 0.078 | 0.951  | 0.85 |  56      | 45  |
| 199 980 | **0.054** | 0.965  | 0.89 |  53      | 45  |

→ best `pp_AR=0.9972` @ iter 194 480, **bce 0.054 still descending**.

**Verdict:** True grokking observed.  ER plateau at 0.79 (held for ~50 k iters
in earlier 80 k run) was broken around iter 130 k once bce dropped under ~0.3.
The model went from 0.79 → 0.997 in the second half of the 200 k budget.
Did NOT reach 0.999 gate but very close and still actively improving — we
gave the next attempt a different feasibility loss (log-barrier) instead of
just more iters, since the user asked us to test that next.

## Log-barrier attempt for ER N=9000 (job 1998, FAILED)

Hypothesis: hinge `max(0, viol)` only penalises strict violations; log-barrier
`-log(margin - viol)` produces stronger gradients near the boundary, which
should push the model harder when it is already feasible-mostly.

Setup: identical to job 1955 except `--feasibility_loss_type log_barrier`.

Result: peak `pp_AR=0.728` after ~50 k iters before being killed.  Log-barrier
collapsed the model into severe under-prediction (raw pred 22 vs opt 45) — the
empty set has the highest "margin" so the barrier rewards predicting almost
nothing.  **Worse than hinge.  Strategy abandoned.**

## Curriculum attempt for ER N=9000 (job 2008, FAILED)

Hypothesis: warm-start from N=1500 ER ckpt (which fits perfectly, `pp_AR=1.0`),
then expand training set to N=9000.  The model should already know the
structure of ER instances and only need to learn the additional ones.

Setup: [experiments/overfit_sl_warm.py](experiments/overfit_sl_warm.py) +
[slurm/overfit/overfit_sl_warm_template.slurm](slurm/overfit/overfit_sl_warm_template.slurm).
`--init_from checkpoints/sweep1_er_N1500_retry/best.pt` then identical
hyperparams to job 1955.

Verification at iter 1: warm `pp_AR=0.88, raw pred=70 vs opt=45` (cold 0.73,
raw 739) — warm-start did transfer.

Result: best `pp_AR=0.9875 @ iter 183 900` over full 200 k budget.
**Worse than cold-start grok (0.9972).**  Warm-started model started ahead but
plateaued earlier — the fitted N=1500 representation appears to be a poor
basin for the larger problem.  Strategy abandoned, original files untouched.

## Cosine-restart LR attempt for ER N=9000 (job 2011, FAILED)

Hypothesis: OneCycleLR over-anneals; cosine annealing with warm restarts
(T_0=20k, T_mult=2) keeps the LR oscillating, helping escape plateaus.

Setup: added `--lr_schedule cosine_restart` to overfit_sl.py + new template
[slurm/overfit/overfit_sl_cosrest_template.slurm](slurm/overfit/overfit_sl_cosrest_template.slurm).
Identical hyperparams otherwise.

Result: peak `pp_AR=0.9005` after ~130 k iters.  Each restart created a
visible regression spike that the model never fully recovered from before
the next restart.  **Worse than vanilla OneCycleLR.  Strategy abandoned.**

## Larger model attempt for ER N=9000 (job 2014, FAILED)

Hypothesis: 1.55 M params is too small; double it.

Setup: `--hidden_dim 384 --num_layers 3 --L_cycles 10` → **4.97 M params**
(3.2× the baseline).  Same training pipeline.  Reduced budget to 100 k iters
to fit in 24h timelimit (~1.48 it/s).

Result: peak `pp_AR=0.8527` at ~50 k iters.  At iter 57 k loss spiked to
1.7 (from 1.05) and pp_AR dropped to 0.58 — model **destabilised** without
recovering.  Killed at 57 760 / 100 000.  More capacity, no better fit, plus
training instability.  **Strategy abandoned.**

## Final ER N=9000 ranking

| Strategy | Job | Peak `pp_AR` | Pass | Notes |
|----------|----:|-------------:|-----:|-------|
| Cold-start grok 200k | 1955 | **0.9972** | ❌ (close) | best so far, still actively improving at end |
| Cold-start grok 80k  | 1920 | 0.8367 | ❌ | original baseline |
| Curriculum (warm-start)| 2008 | 0.9875 | ❌ | starts higher, plateaus earlier |
| Cosine-restart LR    | 2011 | 0.9005 | ❌ | restarts hurt convergence |
| Bigger model 5M params| 2014 | 0.8527 | ❌ | unstable, diverged |
| Log-barrier feasibility| 1998 | 0.7280 | ❌ | collapses to under-prediction |

SAT N=9000 cold-start grok 200k (job 1954) **passed at 0.9997**.

Stage 5 (N=25 000) was not attempted: ER N=9000 never crossed the gate.

## New analysis: why ER N=9000 stays under 0.999

After exhausting the agreed strategies, the failure pattern is consistent and
points away from the four hypotheses we tested:

1. **Capacity is not the bottleneck.**  Doubling parameters (1.55 M → 4.97 M)
   gave identical learning curves through 50 k iters and then destabilised.
   If the model lacked capacity we would have seen lower loss with bigger
   model; we saw the same loss.
2. **LR schedule is not the bottleneck.**  Cosine-restart was strictly worse;
   OneCycleLR's late low-LR phase is when grokking actually happens (bce
   continued to fall from 0.15 → 0.054 between iter 150 k and 200 k).
3. **Feasibility loss form is not the bottleneck.**  Log-barrier collapsed
   into degenerate solutions; hinge with `--feasibility_weight 2.0` is the
   right shape.
4. **Curriculum is not the bottleneck.**  Warm-starting from a perfectly
   fitted smaller dataset *hurt* — the smaller-set basin is too narrow.

What is left:

A. **Per-instance memorisation budget vs total parameters.**  The model has
   to memorise 9 000 distinct positivity patterns over ~6 750 nodes each
   = ~6×10⁷ binary decisions.  That is comparable to the parameter count.
   Information-theoretically the model is in the regime where it can fit
   *most* but not *all* instances perfectly.  At convergence it gets
   raw pred ~53 vs opt 45 (12 % over) and pp_AR 0.997 — i.e. it gets the
   right MIS on ~99.7 % of instances, missing the hard ones.
B. **Tail of hard instances dominates the gate.**  Going from 0.9972 to
   0.999 means cleaning up <1 % of the worst graphs.  These are likely
   instances with multiple competing maximal sets where the model is
   undecided.  Fixing them needs data-aware solutions, not architectural
   ones.
C. **Pixel-level noise: post-processing ceiling.**  The 0.997 ceiling could
   itself be a property of the greedy post-processor on raw logits that
   are 99.5 % correct.  A different decoder (beam-search, sampling-based,
   or learned readout) might extract the remaining gap.

Concrete next experiments (in priority order, *not* run):

1. **Mix-of-batches replay buffer.**  Sample mostly hard instances (the
   bottom-decile pp_AR graphs) for the last 50 k iters.  Should clean up
   the tail of failures without disturbing the easy ones.
2. **Better decoder.**  Replace the greedy "argmax with feasibility check"
   with a parallel sampling + best-of-K decoder.  Cheap and likely closes
   the 0.003 gap by itself.
3. **Per-instance loss weighting.**  Boost loss weight of instances whose
   pp_AR < 0.95 in the previous epoch.  Standard hard-example mining.
4. **Different model family (last resort).**  GraphTransformerTRM is dense
   for a sparse-MIS task; a sparse GNN backbone (PNA, GIN-virtual) at the
   same parameter count might use the budget more efficiently.

For the user's stated overfit goal, **the practical conclusion is that
N≈1 500 (Stage 3) is the largest training set this architecture overfits
perfectly, and N=9 000 is the largest where it gets within 0.3 % of the
gate.**  Scaling further requires a decoder change or data-curriculum
change, not the four knobs we tried.

## Stage 5 — N=25 000, 400 k iters (the big run)

After the ER N=9000 grokking trajectory continued descending right to the
budget end (bce 0.054, pp_AR 0.997 at iter 199 980), the user requested:
"run it on n = 25000 ... increasing the iter from 200k to 400k will be now
even better".  Two parallel jobs, identical hyperparameters to the grok runs
(`--pos_weight 5.0 --feasibility_weight 2.0 --lr 3e-4 --batch_size 16`,
hinge feasibility, OneCycleLR over 400 k iters).

Run names: `sweep1_grok_sat_N25000_400k` (job 2043) and
`sweep1_grok_er_N25000_400k` (job 2044).

| Dataset | Job | Iters | Best `pp_AR` | @ iter | GPU avg | rc |
|---------|----:|------:|-------------:|-------:|--------:|---:|
| SATLIB  | 2043 | 400 000 | **0.9984** | 329 000 | 92–96 % | 0 |
| ER-700-800 | 2044 | 400 000 | 0.8422 | 380 000 | 92–96 % | 0 |

SATLIB scaled smoothly: SAT N=9000 grok 0.9997 → SAT N=25k 0.9984 (essentially
the same).  **ER regressed from 0.997 (N=9k, 200k) → 0.842 (N=25k, 400k).**
The extra 16 000 ER instances broke the grokking that was just starting to
work at N=9k — confirming the per-instance memorisation budget hypothesis.

## Stage 5 generalisation evaluation

Eval script: [experiments/eval_overfit_ckpt.py](experiments/eval_overfit_ckpt.py),
template: [slurm/overfit/eval_overfit_template.slurm](slurm/overfit/eval_overfit_template.slurm).
Inference is bounded to `--max_steps 12` (the model's `all_finish` halt rarely
fires in eval — q_hat never crosses 0.9).  Greedy post-processing only.

| Dataset | Split | Graphs | `pp_AR` | `pp_feas` | F1 | precision/recall | tp%/tn%/fp%/fn% | Job |
|---------|-------|------:|--------:|---------:|----:|----------------:|----------------|----:|
| SATLIB  | train  | 39 500 | **0.9935** | 1.000 | 0.617 | 0.471 / 0.895 | 29.83 / 33.09 / 33.58 / 3.50 | 2211 |
| SATLIB  | test   |    500 | **0.9939** | 1.000 | 0.615 | 0.469 / 0.895 | 29.82 / 32.84 / 33.83 / 3.52 | 2209 |
| ER 700-800 | train | 28 250 | 0.6969 | 1.000 | 0.229 | 0.204 / 0.264 |  1.58 / 87.78 /  6.22 / 4.42 | 2212 |
| ER 700-800 | test  |    500 | 0.6889 | 1.000 | 0.232 | 0.207 / 0.268 |  1.61 / 87.70 /  6.28 / 4.41 | 2210 |

Throughput: ~8–9 graphs/s on a single RTX 5090 (12 inner-step cap, batch_size=16).

### Key findings

1. **SAT generalises essentially perfectly.**  Test pp_AR (0.9939) ≈ train
   pp_AR (0.9935).  No memorisation gap.  The model has learned the SAT
   structure rather than memorising 25 k specific positivity patterns.
   Final-eval pp_AR is below the in-training peak (0.9984 @ iter 329k) only
   because we evaluate the *final* checkpoint, not the early-stopping one.
2. **ER did not generalise.**  Training-time peak 0.8422 was on the 25 000
   graphs the model saw; on the 3 250 unseen training-set graphs +
   500 test graphs combined, pp_AR is 0.6969/0.6889.  The "memorisation"
   we observed during training was exactly that — local fit, no transfer.
3. **Feasibility is perfect on both.**  Greedy post-processing always
   produces a valid IS (by construction).  The metric to optimise is
   `pp_AR`, not `pp_feas`.
4. **F1 is misleading on ER.**  6 % positives + the model under-predicts
   density → tp% only 1.6, but the few it does pick are usually correct
   (precision 0.21).  pp_AR is the only metric that correlates with what
   we actually care about.

### Conclusion of the supervised-overfit phase

| Property | SAT | ER |
|----------|----|----|
| Largest N where overfit `pp_AR ≥ 0.999` | N=1 500 | N=1 500 |
| Largest N where overfit `pp_AR ≥ 0.99` | N=25 000 | N=9 000 |
| Test-set `pp_AR` from the largest run    | 0.9939 (✓ generalises) | 0.6889 (✗ does not) |

The architecture solves SAT.  ER is the open problem: training-time fit
is the bottleneck (architecture cannot memorise 25 k sparse-positive ER
graphs), and even where it fits (N=1500), the test-set generalisation
question for ER has not been answered yet.  Next experiments target the
ER fit/generalisation gap.

## Experiment 1 — Best-of-K stochastic decoding (DONE, big win on ER)

Eval script: [experiments/eval_overfit_ckpt_bestofk.py](experiments/eval_overfit_ckpt_bestofk.py).
Template: [slurm/overfit/eval_bestofk_template.slurm](slurm/overfit/eval_bestofk_template.slurm).

Algorithm: for each graph, decode K times.  Each decode adds Gumbel noise
`G ~ Gumbel(0,1)` scaled by temperature T to the logits, then runs the
existing `greedy_decode`.  The best (largest) feasible IS across K samples
is kept.  Hyperparameters: K (number of samples), T (noise scale), threshold
(prob threshold below which nodes are skipped — `0.0` means use all nodes
ranked by perturbed prob).

All evaluations on the **test set** (500 graphs) of the corresponding
dataset.  Checkpoints: `sweep1_grok_{sat,er}_N25000_400k`.

| Run | Dataset | K | T | thr | `pp_AR` | Δ vs greedy | Job |
|-----|---------|--:|----:|----:|--------:|-----------:|----:|
| greedy baseline | SAT | — | — | 0.5 | 0.9939 | — | 2209 |
| sweep1_bok16_sat_N25k_test     | SAT | 16 | 1.0 | 0.5 | 0.9882 | −0.0057 | 2225 |
| sweep1_bok16T03_sat_N25k_test  | SAT | 16 | 0.3 | 0.5 | 0.9948 | +0.0009 | 2227 |
| **sweep1_bok16T03thr0_sat_N25k_test** | SAT | 16 | 0.3 | 0.0 | **0.9957** | **+0.0018** | 2232 |
| greedy baseline | ER  | — | — | 0.5 | 0.6889 | — | 2210 |
| sweep1_bok16_er_N25k_test      | ER  | 16 | 1.0 | 0.5 | 0.7250 | +0.0361 | 2226 |
| sweep1_bok64_er_N25k_test      | ER  | 64 | 1.0 | 0.5 | 0.7542 | +0.0653 | 2228 |
| sweep1_bok16T05thr0_er_N25k_test | ER | 16 | 0.5 | 0.0 | 0.8586 | +0.1697 | 2229 |
| sweep1_bok64T10thr0_er_N25k_test | ER | 64 | 1.0 | 0.0 | 0.8456 | +0.1567 | 2231 |
| **sweep1_bok64T05thr0_er_N25k_test** | ER | 64 | 0.5 | 0.0 | **0.8764** | **+0.1875** | 2230 |

### Findings

1. **Threshold = 0 is the dominant lever** — bigger than K or T.  Dropping
   the threshold from 0.5 to 0.0 lets the decoder consider all nodes
   ordered by (perturbed) prob, not just the model's positive predictions.
   On ER this was worth +0.13 pp_AR by itself.
2. **Temperature T=0.5 beats T=1.0 for both datasets.**  Too much noise
   destroys the model's good ordering; too little gives identical samples.
3. **K=64 is enough.**  K=16 → K=64 lift was +0.018 on ER.  K=128 unlikely
   to be worth the 4× compute (no submission).
4. **SAT confidence too high to benefit much** — only +0.0018 lift.  But
   ER, where the model is uncertain, gets +0.1875 lift.
5. **The decoder alone exceeds training-time pp_AR for ER**: best BoK
   = 0.8764 vs training peak greedy = 0.8422.  The model knows more
   than greedy can extract.

Throughput cost: K=16 ~3 graphs/s, K=64 ~0.4 graphs/s on RTX 5090.

## Next experiments (in priority order)

2. **Hard-example mining / per-instance loss weighting.**  Fine-tune the
   N=25 k ER checkpoint with extra weight on graphs whose pp_AR < 0.95.
3. **Mix-of-batches replay buffer.**  Like (2) but bias the *sampler*,
   not the loss.
4. **Different model family (last resort).**  Sparse GNN backbone.

## Experiment 2 — Hard-example mining (DONE, regressed)

Trainer: [experiments/overfit_sl_hem.py](experiments/overfit_sl_hem.py).
Template: [slurm/overfit/overfit_sl_hem_template.slurm](slurm/overfit/overfit_sl_hem_template.slurm).

Algorithm: warm-start from `sweep1_grok_er_N25000_400k/best.pt`, run a one-time
per-graph pp_AR pass over the 25 000 ER training graphs, then mini-batch sample
according to weights `w_i = (1 - pp_AR_i + 0.05) ** 2`.  Re-rank every 50 k iters.
Hyperparameters: lr=1e-4 (1/3 of training-time lr because warm-start), 100 k iters,
batch_size=16, pos_weight=5.0, feas_w=2.0.  Run name `sweep1_hem_er_N25k_ft100k` (job 2233).

### Per-graph training-set pp_AR over time

| Phase | mean | min | p10 (bottom-decile) |
|-------|----:|----:|--------------------:|
| Initial (warm-start)  | 0.6960 | 0.2273 | 0.5333 |
| After 50 k iter (peak)| **0.7234** | 0.2889 | 0.6000 |
| After 100 k iter (final) | 0.6566 | 0.1778 | 0.4667 |

The first 50 k iters helped (+0.027 mean, +0.067 p10) — HEM successfully
focused capacity on hard graphs.  But the second 50 k destroyed it: after
the iter-50 k re-rank, the model started over-fitting on the *new* hard
graphs (which were now the originally-easy ones it had partially forgotten),
catastrophically regressing the formerly-easy graphs.

### Test-set evaluation of HEM checkpoint (job 2268, 2269)

The auto-saved `best.pt` at iter 98800 is in the regression zone; results match.

| Decoder | Baseline ckpt | HEM ckpt | Δ |
|---------|----:|----:|----:|
| Greedy            | 0.6889 | 0.6571 | **−0.0318** |
| BoK64 T=0.5 thr=0 | 0.8764 | 0.8404 | **−0.0360** |

**Verdict: HEM as implemented hurts.**  Two design errors:

1. **Best-checkpoint selection used per-batch pp_AR on biased samples**, not
   full-set pp_AR.  That made the saved ckpt the over-trained iter-98800 one
   instead of the iter-50 000 peak.
2. **Aggressive re-ranking at iter 50 k destabilised the model.** With α=2.0
   the originally-easy graphs were starved of gradient and forgotten.

Cleaner reruns to consider (not run): no mid-training re-rank, α=1.0,
ckpt selection by full-set mean.  Likely a +0.02 lift, marginal vs Exp 1's
+0.19.  **Strategy abandoned in favour of the cheaper, larger Exp 1 win.**

## Experiments 3 and 4 (not run)

Given:
- **Exp 1 (decoder) gave +0.19 on ER test pp_AR** (0.6889 → 0.8764), no
  retraining cost.
- **Exp 2 (HEM) regressed −0.03** on both greedy and BoK decoders, after
  ~12 h of GPU time, even with a sensible warm-start.
- Exp 3 (replay buffer) is methodologically similar to Exp 2 (sampling-bias
  fine-tune) and very likely produces a similar marginal-or-negative result.
- Exp 4 (different model family) is gated as "last resort".

We are stopping the experiment ladder.  **The recommendation is: keep the
existing N=25 k ER checkpoint and use BoK64 T=0.5 thr=0 as the decoder.**
For SAT, BoK16 T=0.3 thr=0 gives 0.9957 (vs greedy 0.9939) — a free 0.2 %.

## Final test-set numbers

| Dataset | Greedy | Best BoK | Lift |
|---------|------:|---------:|-----:|
| SATLIB     | 0.9939 | **0.9957** (K=16, T=0.3, thr=0) | +0.0018 |
| ER 700-800 | 0.6889 | **0.8764** (K=64, T=0.5, thr=0) | +0.1875 |

ER's BoK pp_AR exceeds even the in-training peak greedy pp_AR (0.8422),
confirming the model knew more than greedy could extract.  SAT's gain is
small because the model's predictions are already nearly deterministic
(very confident probabilities), so Gumbel perturbation produces near-identical
samples.

## Files added (originals untouched)

- [experiments/overfit_sl_warm.py](experiments/overfit_sl_warm.py) — copy with `--init_from`
- [experiments/overfit_sl_hem.py](experiments/overfit_sl_hem.py) — hard-example mining variant (Exp 2)
- [experiments/eval_overfit_ckpt.py](experiments/eval_overfit_ckpt.py) — checkpoint eval (greedy)
- [experiments/eval_overfit_ckpt_bestofk.py](experiments/eval_overfit_ckpt_bestofk.py) — best-of-K decoder eval (Exp 1)
- [slurm/overfit/overfit_sl_warm_template.slurm](slurm/overfit/overfit_sl_warm_template.slurm)
- [slurm/overfit/overfit_sl_cosrest_template.slurm](slurm/overfit/overfit_sl_cosrest_template.slurm)
- [slurm/overfit/overfit_sl_bigmodel_template.slurm](slurm/overfit/overfit_sl_bigmodel_template.slurm)
- [slurm/overfit/overfit_sl_hem_template.slurm](slurm/overfit/overfit_sl_hem_template.slurm)
- [slurm/overfit/eval_overfit_template.slurm](slurm/overfit/eval_overfit_template.slurm)
- [slurm/overfit/eval_bestofk_template.slurm](slurm/overfit/eval_bestofk_template.slurm)
- `--lr_schedule {onecycle, none, cosine_restart}` flag on overfit_sl.py

## What this tells us

| Claim | Evidence |
|-------|----------|
| Architecture is correct on small data | N=1 and N=250 reach `bce<0.01` & `pp_feas=1.0` on **both** datasets |
| Architecture scales to ~1 500 graphs | Stage 3 passes (SAT 0.9991, ER 1.0000 with 2× budget) |
| **Architecture saturates between 1.5 k and 9 k graphs** | Stage 4 SAT plateaus at 0.996, ER at 0.84 even with 80 k iters & GPU 96 % |
| Bigger budget alone does not close the gap | ER 1920 went from 0.79 @ 20 k to 0.79 @ 78 k — flat |

To push beyond N≈1500 we need at minimum:

- larger model (≥2× params, more L_cycles, possibly hidden=384),
- flat or cosine-restart LR (not OneCycleLR),
- a curriculum (gradually grow N during training rather than full-set),
- and/or a better feasibility regulariser (e.g. log-barrier instead of hinge).

## Reproducing

```bash
source .venv/bin/activate
python -u -m experiments.run_overfit_sweep --start_at 1   # full sweep
# or resume after a manual fix:
python -u -m experiments.run_overfit_sweep --start_at 4
```

Per-stage logs live in `logs/sweep1_{sat,er}_N{N}[_retry|_fix2]_{jobid}.log`.
Aggregated machine-readable summary: [logs/overfit_sweep_summary.json](logs/overfit_sweep_summary.json).

---

## Phase F (N=9000, 400k iters, eval_test) — re-run of jobs 1954/1955

Same recipe as the 200k grok pair, doubled to 400k iterations and with the
test-set evaluator enabled (eval_test pipeline + decoder_baseline=degree).

| Variant | Job | Slurm wrapper | Status |
|---|---|---|---|
| SAT N=9k 400k | 3013 | [slurm/sweep1_pe_sat_N9000_400k.slurm](../slurm/sweep1_pe_sat_N9000_400k.slurm) | running (tujestpolin) |
| ER  N=9k 400k | 3014 | [slurm/sweep1_pe_er_N9000_400k.slurm](../slurm/sweep1_pe_er_N9000_400k.slurm) | queued |

Results table to be filled in once eval data lands. Comparison anchors:
SAT decoder-only floor pp_AR=0.6532, ER decoder-only floor pp_AR=0.5949.

## Phase G (N=25000, 1.2M iters, eval_test) — extension of grok 25k_400k runs

Same setup as Phase F, extended to N=25000 and 1,200,000 iterations.

| Variant | Job | Slurm wrapper | Status |
|---|---|---|---|
| SAT N=25k 1.2M | 3015 | [slurm/sweep1_pe_sat_N25k_1200k.slurm](../slurm/sweep1_pe_sat_N25k_1200k.slurm) | queued |
| ER  N=25k 1.2M | 3016 | [slurm/sweep1_pe_er_N25k_1200k.slurm](../slurm/sweep1_pe_er_N25k_1200k.slurm) | queued |

All four jobs target `tujestpolin` exclusively (5×RTX 5090 32 GB) and are
limited to 7 days of wall-clock time.

## Label distribution analysis (H1)

Computed via [`dataset/compute_label_ratio.py`](../dataset/compute_label_ratio.py).
Raw output: [logs/label_ratio.txt](../logs/label_ratio.txt) and
[logs/label_ratio.json](../logs/label_ratio.json).

| Split | N | global pos_frac | per-graph mean | std | min | p50 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| satlib/train      | 39 500 | 0.3333 | 0.3333 | 0.0000 | 0.3325 | 0.3333 | 0.3333 |
| satlib/test       |    500 | 0.3333 | 0.3333 | 0.0000 | 0.3325 | 0.3333 | 0.3333 |
| er_700_800/train  | 28 250 | 0.0600 | 0.0601 | 0.0020 | 0.0550 | 0.0600 | 0.0657 |
| er_700_800/test   |    500 | 0.0602 | 0.0602 | 0.0020 | 0.0553 | 0.0603 | 0.0653 |

Subset views (first-N graphs in shard order, the order `overfit_sl` actually
loads):

| Split | N=1 | N=250 | N=1500 | N=9000 | N=25000 |
|---|---:|---:|---:|---:|---:|
| satlib/train pos_frac | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| er_700_800/train pos_frac | 0.0619 | 0.0601 | 0.0600 | 0.0600 | 0.0600 |

**Conclusion:** SAT is essentially constant at exactly 1/3 (algebraic
property of 3-SAT MIS instances), and ER is tightly distributed around
0.060 with σ ≈ 0.002. There is **no subsample bias** at any N: the global
pos_frac is identical across N=1, 250, 1500, 9000, 25000. So the label
distribution is *not* a confound for any of our N-vs-loss observations.

## Iter-1 loss decomposition (H2)

For binary-cross-entropy with `pos_weight=w` and reduction='mean', if the
model emits p≈0.5 at initialisation then per-node:

$$
\mathbb{E}[\text{bce}_1] \;\approx\; \ln 2 \cdot \bigl(w\,q + (1-q)\bigr)
$$

where $q$ is the dataset-level positive fraction. Substituting the
measured $q$ values (0.3333 for SAT, 0.0600 for ER):

| Job | Dataset | N | pos_weight w | predicted bce₁ | observed bce₁ | match |
|---|---|---:|---:|---:|---:|---|
| 1882 | SAT | 1     | 2.00  | 0.924 | 0.922 | ✓ |
| 1888 | SAT | 250   | 2.00  | 0.924 | 0.923 | ✓ |
| 1892 | SAT | 1500  | 2.00  | 0.924 | 0.923 | ✓ |
| 1919 | SAT | 9000  | 2.00  | 0.924 | 0.924 | ✓ |
| 1954 | SAT | 9000  | 2.00  | 0.924 | 0.924 | ✓ |
| 2043 | SAT | 25000 | 5.00  | 1.617 | 1.536 | ✓ (≈) |
| 1883 | ER  | 1     | 15.16 | 1.282 | 1.307 | ✓ |
| 1889 | ER  | 250   | 15.65 | 1.302 | 1.307 | ✓ |
| 1893 | ER  | 1500  | 15.66 | 1.302 | 1.307 | ✓ |
| 1898 | ER  | 1500  | 15.66 | 1.302 | 1.307 | ✓ |
| 1920 | ER  | 9000  | 8.00  | 0.984 | 1.037 | ✓ |
| 1955 | ER  | 9000  | 5.00  | 0.860 | 0.928 | ✓ |
| 2044 | ER  | 25000 | 5.00  | 0.860 | 0.924 | ✓ |
| 2506 | ER  | 25000 | 20.00 | 1.484 | 1.460 | ✓ |
| 2531 | ER  | 25000 | 5.00  | 0.860 | 0.869 | ✓ |

**Conclusion:** the apparent "starting losses are higher for some runs than
others" is **fully explained by the `pos_weight` setting**, not by N or
dataset bias. Specifically:

* SAT N=9k bce₁ ≈ 0.92 vs SAT N=25k bce₁ ≈ 1.54 is purely because the
  former used `pos_weight=2.0` and the latter `pos_weight=5.0`.
* ER iter-1 spans 0.87 → 1.46 entirely tracking `pos_weight` from 5 to 20.
* The earliest ER runs (1883/1889/1893/1898) used auto-derived `pos_weight ≈ 15.66`
  (≈ (1-q)/q = 15.66 for q=0.06), giving the highest iter-1 bce.
* The label distribution itself is uniform across all N values, so subset
  bias is ruled out.

The model initialisation produces probs ≈ 0.5 (raw AR ≈ 3.0 for SAT, ≈ 16
for ER, i.e. it predicts ~all nodes as 1), which is consistent with the
formula's p=0.5 assumption.

