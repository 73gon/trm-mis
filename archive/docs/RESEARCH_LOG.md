# Research Log — MIS TRM (chronological)

Every experiment recorded here includes hypothesis, config, key result, verdict, and wandb link.

## Phase 0 — Bug fix & audit (April 16, 2026)

### 0.1 Double-sigmoid in `improvements/eval_multisample.py`
- **Symptom**: raw_feasibility=0, raw_pred_size = num_nodes for every graph during eval.
- **Root cause**: Model returns post-sigmoid probs (see [models/graph_transformer_trm.py](models/graph_transformer_trm.py#L311) and [models/graph_transformer_trm_ssl.py](models/graph_transformer_trm_ssl.py#L481)). Eval code renamed them `logits` and applied `torch.sigmoid(logits / temp)` → saturates every value into `[0.5, 0.73]` so all nodes cross the 0.5 threshold.
- **Fix**: treat the model output as probs, recover logits with `torch.logit(probs.clamp(1e-6, 1-1e-6))`, apply temperature, resigmoid. See [improvements/eval_multisample.py](improvements/eval_multisample.py#L72-L95).
- **Effect on history**: decoded AR metrics unchanged (greedy uses ordering, preserved by sigmoid), but every "raw" metric reported before the fix is meaningless.
- **Verdict**: ✅ fixed.

### 0.2 Audit of training-time metrics
- [models/metrics.py](models/metrics.py) and [models/metrics_ssl.py](models/metrics_ssl.py) apply `> 0.5` directly to input probs — no sigmoid inside. Both training files pass the post-sigmoid output. No double-sigmoid there.
- **Verdict**: ✅ clean, no change needed.

## Phase 1 — Pilots (20 epochs each on SATLIB)

Submitted April 16, 2026.

First submission (jobs `1825-1831`) ran on the full 158-shard SATLIB dataset, which at 1-2 it/s meant ~37 min/epoch and 20 epochs ≈ 12h — it would not fit in the 8h SLURM window. Cancelled after 13 min and resubmitted with `--max_shards 50` (~1/3 of the data), which gives ~4h for 20 epochs.

| ID | Pilot | Wandb run name | Status |
|----|-------|----------------|--------|
| 1832 | `pilot_baseline_sl` | pilot_sl_satlib_baseline | running |
| 1833 | `pilot_baseline_ssl` | pilot_ssl_satlib_baseline | running |
| 1834 | `pilot_cardinality_sl` | pilot_sl_satlib_cardinality | running |
| 1835 | `pilot_entropy_ssl` | pilot_ssl_satlib_entropy | running |
| 1836 | `pilot_focal_sl` | pilot_sl_satlib_focal | pending |
| 1837 | `pilot_reinforce_ssl` | pilot_ssl_satlib_reinforce | pending |
| 1838 | `pilot_tempanneal_ssl` | pilot_ssl_satlib_temp_anneal | pending |

(Curriculum pilot deferred — requires dataset refactor; will return if other techniques plateau.)

Results will be filled in once jobs complete.

### Pilot results (epoch-20 validation, EMA model, SATLIB val split, 50 shards)

| Pilot | PP AR | Raw Pred/Opt | Raw Feas | Verdict |
|-------|-------|--------------|----------|---------|
| **focal_sl** | 0.9637 | **1.126** | **0.984** | 🏆 best honest learning — nearly meets both final gates |
| cardinality_sl | 0.9913 | 1.916 | 0.794 | best PP AR, small honest improvement vs SL baseline |
| baseline_sl | 0.9903 | 2.147 | 0.665 | high AR but severe over-prediction (greedy rescues) |
| reinforce_ssl | 0.9537 | 1.874 | 0.900 | moderate honest gain, AR slightly above SSL baseline |
| baseline_ssl | 0.9491 | 1.541 | 0.940 | SSL baseline — already decently honest, lower AR |
| entropy_ssl (w=0.01) | 0.9492 | 1.542 | 0.940 | indistinguishable from SSL baseline — weight too low |
| tempanneal_ssl (2→0.5) | 0.9497 | 1.541 | 0.940 | indistinguishable from SSL baseline — anneal range too narrow |

### Takeaways
1. **Focal loss is the standout**: at 20 epochs on 1/3 of SATLIB it already nearly satisfies the honest-learning gates (ratio 1.13 vs target 1.10; raw feas 0.98 vs target 0.95). PP AR dropped from 0.990 → 0.964, but the model is doing the real work instead of relying on greedy rescue.
2. **Cardinality helps SL honesty without hurting AR** (ratio 2.15 → 1.92, raw feas 0.67 → 0.79, PP AR 0.990 → 0.991).
3. **Entropy and temperature annealing at pilot hyperparameters had no measurable effect on SSL** — likely too small. Will revisit with larger coefficients (or higher temperature start) if the SSL final run needs more regularization.
4. **REINFORCE SSL** slightly beats SSL baseline on PP AR (0.954 vs 0.949) and improves raw feasibility (0.90 vs 0.94). Interesting to keep as a candidate.
5. **Supervised > self-supervised on PP AR** at 20 epochs (0.99 vs 0.95). Expected — labels are strong signal.

### Winner selection for Phase 2 finals
- SL final: **focal + cardinality** combined (honest learning + high AR).
- SSL final: **reinforce + cranked-up entropy + temp annealing** (push the SSL plateau harder; weak-individual techniques may combine non-trivially).

## Phase 2 — Final runs

Submitted April 17, 2026.

| ID | Run | Wandb | Dataset | Config |
|----|-----|-------|---------|--------|
| 1840 | final_sl_satlib | final_sl_satlib | SATLIB (158 shards) | focal γ=2, cardinality λ=0.1, PE+enhanced, 100 ep |
| 1839 | final_sl_er700 | final_sl_er700 | ER-700-800 (113 shards) | same + pos_weight=15, bs=8 |
| 1842 | final_ssl_satlib | final_ssl_satlib | SATLIB | REINFORCE + entropy 0.05 + temp 3→0.3, PE+enhanced, 50 ep |
| 1841 | final_ssl_er700 | final_ssl_er700 | ER-700-800 | same, bs=8 |

SLURM wall: 48h. SSL epochs reduced to 50 (plateaus early). SL targets 100 epochs, early-stop by validation AR.

### Phase 2 — Incidents

- **1839 killed (OOM)**: ER-700 SL with bs=8 + PE + enhanced on RTX 4090 (24GB) was OOM-killed mid-epoch 1. Training crashed, eval crashed on missing best.pt.
- **Fix**: resubmitted as **1843** with bs=4 (same config otherwise). Queued behind 1840 on sarmatia.
- **Speed observation**: With PE+enhanced, full-dataset epoch times are ~2.4h (SL SATLIB 158 shards, bs=8, RTX 4090) and ~3.8h (SSL SATLIB 158 shards, bs=8, RTX 5090, reinforce). 48h wall will yield ~20 SL epochs / ~12 SSL epochs — less than planned 100/50 but pilots showed convergence by epoch 15–20. Best.pt is saved per val improvement, so partial runs produce usable checkpoints.
