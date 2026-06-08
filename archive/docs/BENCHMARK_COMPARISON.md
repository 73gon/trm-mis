# MIS Model Benchmark Results — Comparison with DIFUSCO SOTA

## Executive Summary

We evaluate our Graph Transformer TRM model on the Maximum Independent Set (MIS) 
problem against DIFUSCO (Sun & Yang, NeurIPS 2023 Spotlight), the current state-of-the-art 
neural solver. Results are reported on the standard SATLIB CBS and ER-[700-800] benchmarks.

---

## 1. Model Overview

| Model | Architecture | Params | Training | Diffusion Steps |
|-------|-------------|--------|----------|----------------|
| **DIFUSCO** | 12-layer AGNN, width=256 | ~2M | SL (49.5K SATLIB, 50 epochs) | 50 |
| **Ours (TRM)** | GPS (GIN+MHA) + TRM recursion, width=256 | 1.55M | SSL pretrain + SL fine-tune | 0 (direct) |

**Key architectural differences:**
- DIFUSCO: Diffusion model, generates heatmap via iterative denoising (50 steps), edge-based AGNN
- TRM: Single forward pass, GPS layers with GIN convolution + multi-head attention, TRM recursion (H=2, L=6)

---

## 2. SATLIB CBS Benchmark

### DIFUSCO Results (from paper, Table 3)

| Method | Type | MIS Size ↑ | Gap ↓ |
|--------|------|-----------|-------|
| KaMIS (heuristic) | Baseline* | 425.96 | — |
| Gurobi (exact) | Exact | 425.95 | 0.00% |
| Intel | SL+Greedy | 420.66 | 1.48% |
| DGL | SL+Greedy | — | 0.63% |
| **DIFUSCO** | SL+Greedy (50 steps × 1 sample) | ~424.6 | ~0.32% |
| **DIFUSCO** | SL+Sampling (50 steps × 4 samples) | **~425.07** | **0.21%** |

*Note: DIFUSCO reports NO post-processing (no local search, no graph reduction).*

### Our Results

#### Preliminary (SATLIB train, 249 graphs, 8 samples)

| Checkpoint | Strategy | Approx Ratio | Gap |
|-----------|----------|-------------|-----|
| mis_finetune | baseline_greedy (no PP) | 0.9894 | 1.06% |
| mis_finetune | enhanced_greedy (+1-swap) | 0.9923 | 0.77% |
| mis_finetune | multisample_enhanced (8 samples) | 0.9930 | 0.70% |
| mis_ft_high | baseline_greedy (no PP) | 0.9898 | 1.02% |
| mis_ft_high | enhanced_greedy (+1-swap) | 0.9927 | 0.73% |
| mis_ft_high | multisample_enhanced (8 samples) | 0.9930 | 0.70% |

#### Final Benchmark (SATLIB test, 499 graphs, 32 samples)

| Checkpoint | Strategy | Approx Ratio | Gap↓ |
|-----------|----------|-------------|------|
| mis_finetune | baseline_greedy (no PP) | 0.9882 | 1.18% |
| mis_finetune | enhanced_greedy (+1-swap) | 0.9914 | 0.86% |
| mis_finetune | multisample_greedy (32s) | 0.9907 | 0.93% |
| **mis_finetune** | **multisample_enhanced (32s)** | **0.9935** | **0.65%** |
| mis_ft_high | baseline_greedy (no PP) | 0.9889 | 1.11% |
| mis_ft_high | enhanced_greedy (+1-swap) | 0.9918 | 0.82% |
| mis_ft_high | multisample_greedy (32s) | 0.9903 | 0.97% |
| **mis_ft_high** | **multisample_enhanced (32s)** | **0.9930** | **0.70%** |

---

## 3. ER-[700-800] Benchmark

### DIFUSCO Results

DIFUSCO "does not perform well" on ER graphs. The paper states: "we hypothesize 
that this is because previous methods use node-based GNNs (GCN, GraphSage) while 
we use an edge-based AGNN whose inductive bias may not be suitable for ER graphs."

| Method | Type | MIS Size↑ | Gap↓ |
|--------|------|----------|------|
| KaMIS (heuristic) | Baseline* | 44.87 | — |
| Gurobi (exact) | Exact | 41.38 | 7.78% |
| Intel | SL+Greedy | 34.86 | 22.31% |
| DIFUSCO | SL+Greedy | — | (poor, not clearly stated) |

### Our Results (ER-700-800 test, 499 graphs, 32 samples)

**Note:** Our model was trained only on SATLIB CBS data (~430-node structured graphs).
These ER results represent **zero-shot transfer** to unseen ER-type random graphs (700-800 nodes).

| Checkpoint | Strategy | Approx Ratio | Gap↓ |
|-----------|----------|-------------|------|
| mis_finetune | baseline_greedy (no PP) | 0.7400 | 26.00% |
| mis_finetune | enhanced_greedy (+1-swap) | 0.8327 | 16.73% |
| mis_finetune | multisample_enhanced (32s) | 0.8587 | 14.13% |
| mis_ft_high | baseline_greedy (no PP) | 0.7388 | 26.12% |
| mis_ft_high | enhanced_greedy (+1-swap) | 0.8306 | 16.94% |
| **mis_ft_high** | **multisample_enhanced (32s)** | **0.8622** | **13.78%** |

**Key insight:** 1-swap local search provides a massive **+9.3 percentage point** improvement 
on ER-700-800 graphs (from 0.74 to 0.83), demonstrating that post-processing is 
especially valuable on out-of-distribution graphs.

---

## 4. Compute Efficiency Comparison

| Metric | DIFUSCO | TRM (Ours) |
|--------|---------|------------|
| Forward passes per sample | 50 (diffusion steps) | 1 |
| Typical samples | 4 | 32 |
| **Total network evaluations** | **200** | **32** |
| Post-processing | None | 1-swap local search |
| Hardware (original) | 8× V100 train, 1× V100 eval | 1× RTX 5090 |

Our model is **6.25× more network-efficient** per evaluation (32 vs 200 forward passes),
though we use post-processing which DIFUSCO does not.

---

## 5. Analysis

### Post-Processing Impact

| Dataset | Without PP | With 1-swap | Improvement |
|---------|-----------|-------------|-------------|
| SATLIB (best) | 0.9882 | 0.9914 | +0.32 pp |
| ER-700-800 (best) | 0.7400 | 0.8327 | +9.27 pp |

1-swap local search improves our approx ratio by +0.32 points on SATLIB and a 
massive +9.27 points on ER-700-800. Multi-sample (32 samples) provides an 
additional +0.21 points on SATLIB and +2.95 points on ER.

DIFUSCO explicitly disabled post-processing, noting that "all models perform 
similarly with local search post-processing" (citing Böther et al., 2022).

### SATLIB Summary (Head-to-Head)

| Method | Approx Ratio | Gap↓ | Network Evals | Post-Processing |
|--------|-------------|------|---------------|-----------------|
| **DIFUSCO** (SL+S, 4 samples) | **0.9979** | **0.21%** | 200 | None |
| **Ours** (ms_enhanced, 32s) | 0.9935 | 0.65% | 32 | 1-swap |
| Ours (ms_enhanced, 8s) | 0.9930 | 0.70% | 8 | 1-swap |
| Ours (enhanced, 1 sample) | 0.9914 | 0.86% | 1 | 1-swap |
| Ours (baseline, no PP) | 0.9882 | 1.18% | 1 | None |

### Strengths of Our Approach
- **Computational efficiency**: Single forward pass vs 50 diffusion steps.
  At 32 samples, we use 32 network evaluations vs DIFUSCO's 200 (6.25× fewer).
- **Simpler architecture**: No noise scheduling, no denoising, direct prediction
- **Self-supervised pretraining**: Reduces label dependency
- **Post-processing amplification**: 1-swap local search is especially effective 
  on hard/OOD graphs (+9 pp on ER-700-800)

### Gaps vs DIFUSCO
- **SATLIB**: Our Gap 0.65% vs DIFUSCO's 0.21% — DIFUSCO is ~3× better in gap terms
- **Architecture advantage**: DIFUSCO's diffusion process captures multimodal solution 
  distributions naturally, enabling more diverse sampling than our temperature perturbation
- **Training scale**: DIFUSCO trained on 49,500 SATLIB examples for 50 epochs; 
  our model was trained on fewer examples
- **ER-700-800**: Not comparable — our model was not trained on ER data

### Efficiency-Normalized Comparison

If we normalize by number of network evaluations:
- DIFUSCO: Gap=0.21% with 200 evals → 0.00105 gap/eval
- Ours: Gap=0.65% with 32 evals → 0.0203 gap/eval

Per evaluation, our model achieves reasonable results but DIFUSCO is more 
effective at utilizing additional compute through its diffusion framework.

---

## 6. Conclusion

Our Graph Transformer TRM model achieves **0.9935 approximation ratio (0.65% gap)** 
on the SATLIB CBS benchmark with 32-sample inference and 1-swap local search. This 
compares to DIFUSCO's **0.9979 (0.21% gap)** with 200 network evaluations and no 
post-processing.

While we do not surpass DIFUSCO on SATLIB, our approach offers significant advantages:
1. **6.25× fewer network evaluations** (32 vs 200)
2. **Simpler architecture** without diffusion complexity
3. **Strong post-processing synergy** — 1-swap local search provides +0.32 pp on 
   SATLIB and +9.27 pp on ER-700-800

On ER-700-800 (zero-shot, no ER training), our model achieves 0.8622 AR (13.78% gap),
demonstrating reasonable out-of-distribution generalization, especially with post-processing.

### Potential Improvements to Close the Gap
1. **Train on full SATLIB dataset** (49.5K examples, matching DIFUSCO's training data)
2. **Scale up model** (12 layers like DIFUSCO vs our 2 GPS layers + TRM recursion)
3. **Add learnable 2-opt/local search** during training
4. **Train on ER-700-800** for competitive ER benchmark results
5. **Increase samples to 64-128** to further exploit multimodal coverage

---

### Raw Outputs

All JSON results saved in `improvements/results/`:
- `bench_finetune_satlib_test.json` — mis_finetune on SATLIB test
- `bench_ft_high_satlib_test.json` — mis_ft_high on SATLIB test
- `bench_finetune_er700_test.json` — mis_finetune on ER-700-800 test
- `bench_ft_high_er700_test.json` — mis_ft_high on ER-700-800 test

SLURM job: #1088 on tujestpolin (RTX 5090)

---

*Document generated: April 1, 2026*
*Benchmark completed successfully: All 4 evaluations (2 checkpoints × 2 datasets)*
