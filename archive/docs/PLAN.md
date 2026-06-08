# Master Plan — Honest MIS Model Surpassing DIFUSCO

**Status:** Active. Created April 16, 2026.

## Goal

Train MIS models that:
- Beat DIFUSCO on SATLIB (target AR > 0.999, DIFUSCO = 0.9979)
- Beat DIFUSCO on ER-700-800 (target AR > 0.97, DIFUSCO = 0.9706)
- **Actually learn**: raw feasibility > 0.95 AND pred_size/opt_size ∈ [0.95, 1.10]
  (not "predict all ones and let greedy fix it")
- All runs visible in wandb with clear naming

## Constraints (user-confirmed)

- **Do NOT modify** `train_mis.py` or `train_mis_ssl.py` until final merge with user approval
- All experimental code lives in new `experiments/` folder
- 20-epoch pilots → pick winners → 100-epoch final runs (early-stop)
- SLURM queue (sequential jobs, cluster handles parallelism)
- Checkpoint-resume chaining for unlimited effective compute
- Separate models per dataset (SATLIB, ER-700)
- Final runs use `use_pe=1 use_enhanced_features=1`

## Wandb naming convention

- Project: `MIS-TRM`, entity: `malikmardan-personal`
- Group: `pilot_<technique>` or `final_runs`
- Tags: one of `pilot`/`final`, one of `ssl`/`sl`, dataset (`satlib`/`er700`), technique name
- Run name: `<phase>_<paradigm>_<dataset>_<technique>` e.g. `pilot_sl_satlib_cardinality`

## Phase 0 — Bug fix & audit (BLOCKING)

1. Fix double-sigmoid in `improvements/eval_multisample.py` (~L85-95).
   Model already returns post-sigmoid probs; eval re-applies sigmoid → saturates to [0.5, 0.73].
   Fix: treat input as probs, apply temperature via `logit(probs)/temp → sigmoid`.
2. Audit `models/metrics.py` and `models/metrics_ssl.py` for same bug.
3. Re-run eval on existing best checkpoints with fixed code (parallel to pilots):
   `mis_ft_high`, `sl_fw1_pw2`, `mis_ssl`, `mis_full_pw5`.
   Expectation: decoded AR unchanged; raw confusion matrix becomes meaningful.

## Phase 1 — Pilots (20 epochs each)

Eight pilot experiments in `experiments/` folder. Each = minimal copy of `train_mis.py` or `train_mis_ssl.py` with one change. Pilots run on SATLIB only (fast, clean signal).

### Baselines (must include — user request)
- `pilot_baseline_sl` — identity copy of train_mis.py, 20 ep
- `pilot_baseline_ssl` — identity copy of train_mis_ssl.py, 20 ep

### Techniques (SL, since it converges faster than SSL)
1. `pilot_cardinality` — add `λ_c · (Σprobs − opt_size)²`, λ_c = 0.01
2. `pilot_entropy` — minimize entropy, λ_e = 0.01 (push probs to 0/1)
3. `pilot_temp_anneal` — linear temp 5.0 → 0.5 + cosine LR schedule (addresses plateau)
4. `pilot_focal` — focal loss α=0.25, γ=2.0 replacing BCE
5. `pilot_curriculum` — sort training graphs by |V|, start smallest 30%, +10%/epoch
6. `pilot_reinforce` — Bernoulli sampling + policy gradient w/ baseline

### Exit rule per pilot
Keep if: raw_feasibility > 0.7 OR pred_size/opt_size < 1.5 OR decoded AR > baseline + 0.001.

## Phase 2 — Combine winners (final training files)

- `experiments/train_mis_final.py` (SL with top 2-3 winners)
- `experiments/train_mis_ssl_final.py` (SSL with top 2-3 winners)
- Each winner technique added as CLI flag (opt-in)

### Final runs (4 SLURM jobs)

| Job name | File | Dataset | Checkpoint dir |
|----------|------|---------|----------------|
| final_sl_satlib | train_mis_final.py | SATLIB | checkpoints/final_sl_satlib |
| final_sl_er700 | train_mis_final.py | ER-700 | checkpoints/final_sl_er700 |
| final_ssl_satlib | train_mis_ssl_final.py | SATLIB | checkpoints/final_ssl_satlib |
| final_ssl_er700 | train_mis_ssl_final.py | ER-700 | checkpoints/final_ssl_er700 |

- Max 100 epochs, early-stop patience 15 on val AR
- `use_pe=1 use_enhanced_features=1`
- 48h wall + auto-resume from latest checkpoint

## Phase 3 — Evaluation

For each final checkpoint:
- 4 parallel Bernoulli samples × 3 thresholds × `multisample_enhanced` decoder
- Per-graph metrics: AR, std, raw pred_size/opt_size, raw feasibility, time
- Comparison table vs DIFUSCO (Table 3), KaMIS, Gurobi
- Honest-learning plots: raw-prob histogram (should be bimodal), pred_size ratio distribution

## Phase 4 — Documentation

- `docs/PLAN.md` (this file)
- `docs/RESEARCH_LOG.md` — chronological per-experiment log (hypothesis, config, result, verdict, wandb link)
- `docs/FINAL_RESULTS.md` — comparison tables + plots + best numbers
- `docs/FAILED_EXPERIMENTS.md` — detailed per-failure analysis
- `docs/HONEST_LEARNING.md` — over-prediction problem + fix explanation
- Wandb report (shareable link)

## Phase 5 — Merge winners back to originals

Only after user approval:
- Add winner techniques to `train_mis.py` + `train_mis_ssl.py` as optional CLI flags (default OFF, backward compat preserved)
- User reviews diff before merge

## Risks & mitigations

1. **No pilot beats baseline honestly** → escalate to architectural changes (bigger H_cycles, more GPS layers).
2. **ER-700 gap too large** → add test-time search (Metropolis-Hastings over top-K probs) as Phase 3 fallback.
3. **Pilot compute too slow** → drop `pilot_reinforce` first (highest variance).
4. **Checkpoint chain breaks** → all final scripts auto-detect latest .pt and resume.

## Success criteria

- Raw feasibility > 0.95, pred_size/opt ∈ [0.95, 1.10] on at least one final model
- Decoded AR > 0.999 on SATLIB OR > 0.97 on ER-700
- All wandb runs organized with consistent tags
- 4 documentation markdown files written with concrete numbers and plots
