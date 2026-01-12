# MIS Training: Professor Feedback Analysis & Implementation Plan

**Date**: January 8, 2026
**Status**: 10,000 graphs generated, initial training complete
**Goal**: Overfit on training data with very good feasibility and very small BCE loss

---

## 📋 Summary of Feedback Points

| # | Point | Category | Current Status |
|---|-------|----------|----------------|
| 1 | Data difficulty (LP vs ILP optimum) | Data Quality | ❓ Not analyzed |
| 2 | Feasibility loss weight tuning | Loss Function | ✅ Configurable (cfg.loss.feasibility_weight) |
| 3 | Feasibility loss ablation | Experiments | ⚠️ Ready to run |
| 4 | Sparsity loss ablation | Experiments | ⚠️ Ready to run |
| 5 | Post-processing gap logging | Metrics | ✅ Implemented |
| 6 | Train/Val logging per epoch | Infrastructure | ✅ Implemented |
| 7 | EMA model validation | Infrastructure | ❌ Not implemented |
| 8 | Node degree feature | Features | ✅ Partially (deg_norm) |
| 9 | Graph Laplacian feature | Features | ❌ Not implemented |
| 10 | Graph Transformer Networks | Architecture | ❌ Using GIN |
| 11 | y_init and z_init | Architecture | ❌ Not implemented |
| 12 | Smaller learning rate | Hyperparameters | ⚠️ LR=1e-3 |
| 13 | Log model size | Logging | ✅ Implemented |
| 14 | Model size <= 7M params | Architecture | ✅ Verified (count_parameters) |
| 15 | Weight decay tuning | Hyperparameters | ⚠️ WD=0.1 |

---

## 🎯 Priority Ranking & Evaluation

### **Priority 1: Critical Infrastructure (Must Have First)**

These items are foundational - without them, we cannot properly evaluate other changes.

#### 1.1 Train/Validation Logging Per Epoch
**Importance**: ⭐⭐⭐⭐⭐ (5/5)
**Effort**: Medium
**Why Critical**: Currently we only log training metrics. Without validation logging, we cannot:
- Detect overfitting (which is our GOAL)
- Compare model generalization
- Make informed hyperparameter decisions

**Current State**:
- Training logging exists via wandb
- No validation split or evaluation loop
- No per-epoch validation metrics

**Implementation Tasks**:
- [ ] Split dataset into train/validation (e.g., 90/10)
- [ ] Add validation loop after each epoch
- [ ] Log validation metrics: loss, F1, precision, recall, feasibility, approx_ratio
- [ ] Add early stopping based on validation metrics (optional)

---

#### 1.2 Log Model Size (Parameters, Layers, Width)
**Importance**: ⭐⭐⭐⭐⭐ (5/5)
**Effort**: Low
**Why Critical**: We need to know current model size to:
- Verify we're under 7M parameters
- Track architecture changes
- Compare with other methods

**Current State**:
- `hidden_dim=256`, `num_layers=2`, `cycles=18`
- No parameter count logged

**Implementation Tasks**:
- [ ] Add `count_parameters()` utility function
- [ ] Log to wandb at training start: total params, trainable params, layers, hidden_dim
- [ ] Print model summary

---

#### 1.3 Post-Processing Gap Logging
**Importance**: ⭐⭐⭐⭐⭐ (5/5)
**Effort**: Medium
**Why Critical**: The raw model predictions may have violations. After post-processing (greedy repair), we need to know:
- How many nodes were removed to achieve feasibility?
- What is the final set size vs optimal?
- This is the TRUE performance metric

**Current State**:
- `approx_ratio` logged but computed on RAW predictions (may be infeasible)
- No post-processing step
- No gap logging

**Implementation Tasks**:
- [ ] Implement greedy post-processing (remove nodes with most violated edges)
- [ ] Log: `optimal_size`, `raw_pred_size`, `postprocessed_size`, `gap = optimal - postprocessed`
- [ ] Log feasibility before and after post-processing

---

### **Priority 2: Loss Function Tuning (Core Hypothesis)**

These directly impact whether the model learns the right thing.

#### 2.1 Feasibility Loss Weight Analysis & Tuning
**Importance**: ⭐⭐⭐⭐⭐ (5/5)
**Effort**: Medium
**Why Important**: Professor's key insight: **Both loss terms should be same magnitude**

**Current State** (from `graph_trm.py`):
```python
feasibility_weight = 1.0
sparsity_weight = 0.3
loss = bce_loss + feasibility_weight * feasibility_loss + sparsity_weight * sparsity_loss
```

**Analysis Needed**:
1. What is typical magnitude of `bce_loss`? (likely 0.3-0.7)
2. What is typical magnitude of `feasibility_loss`? (likely 0.01-0.1)
3. What is typical magnitude of `sparsity_loss`? (likely 0.001-0.01)

**If feasibility_loss << bce_loss**, the model ignores feasibility!

**Implementation Tasks**:
- [ ] Add logging for raw loss magnitudes (before weighting)
- [ ] Analyze loss magnitudes over first few epochs
- [ ] Calculate weight to make `feasibility_weight * feasibility_loss ≈ bce_loss`
- [ ] Experiment with weights: [1.0, 5.0, 10.0, 20.0, 50.0]
- [ ] Track feasibility metric vs weight

---

#### 2.2 Feasibility Loss Ablation
**Importance**: ⭐⭐⭐⭐ (4/5)
**Effort**: Low
**Why Important**: Does feasibility loss actually help? Need empirical evidence.

**Experiments**:
| Experiment | feasibility_weight | Expected Result |
|------------|-------------------|-----------------|
| Baseline | 0.0 | Many violations, high recall |
| Low | 1.0 | Current behavior |
| Medium | 5.0 | Fewer violations |
| High | 20.0 | Very few violations, lower recall |

**Implementation Tasks**:
- [ ] Create config for ablation experiments
- [ ] Run 4 experiments (can be parallel on different GPUs)
- [ ] Plot: feasibility vs F1 trade-off curve
- [ ] Determine optimal weight

---

#### 2.3 Sparsity Loss Ablation
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Low
**Why Important**: Sparsity loss may be unnecessary or even harmful.

**Hypothesis**:
- If BCE loss already has class imbalance correction (pos_weight), sparsity loss may be redundant
- Sparsity loss could conflict with BCE objective

**Experiments**:
| Experiment | sparsity_weight | Expected Result |
|------------|-----------------|-----------------|
| No sparsity | 0.0 | May be more greedy |
| Current | 0.3 | Baseline |
| Higher | 1.0 | More conservative |

**Implementation Tasks**:
- [ ] Run experiments with sparsity_weight ∈ {0.0, 0.3, 1.0}
- [ ] Analyze impact on set_size_ratio metric
- [ ] Decide whether to keep sparsity loss

---

### **Priority 3: Data Quality Analysis**

#### 3.1 LP-Optimum vs ILP-Optimum Analysis
**Importance**: ⭐⭐⭐⭐ (4/5)
**Effort**: High
**Why Important**: If LP relaxation = ILP solution for most graphs, the problems are "easy" and don't test the model properly.

**Background**:
- **ILP (Integer Linear Program)**: True MIS solution (NP-hard)
- **LP (Linear Program relaxation)**: Polynomial-time approximation
- **LP-ILP Gap**: If gap is 0 for most graphs, they're "easy" (bipartite, trees, etc.)

**Current State**:
- Labels are computed via some solver (need to verify which)
- LP gap not computed or logged

**Implementation Tasks**:
- [ ] Compute LP relaxation for each graph (using scipy or cvxpy)
- [ ] Compare LP solution size vs ILP solution size
- [ ] Log distribution of LP-ILP gaps
- [ ] If most gaps are 0, generate harder graphs (denser, more complex structure)

**How to Generate Harder Graphs**:
- Increase edge density
- Use Erdős-Rényi with higher p
- Use structured graphs (e.g., random regular graphs)
- Avoid bipartite structures

---

### **Priority 4: Feature Engineering**

#### 4.1 Node Degree Feature Enhancement
**Importance**: ⭐⭐⭐⭐ (4/5)
**Effort**: Low
**Why Important**: Degree is the most informative local feature for MIS.

**Current State** (from `mis_dataset.py`, need to verify):
- Features: `[1, deg_norm]` where deg_norm = degree / max_degree
- `input_dim = 2`

**Enhancement Options**:
- Raw degree (integer)
- Normalized degree (current)
- Log degree: `log(1 + degree)`
- Degree centrality
- Local clustering coefficient

**Implementation Tasks**:
- [ ] Verify current feature construction
- [ ] Add log degree as additional feature
- [ ] Experiment with feature combinations

---

#### 4.2 Graph Laplacian Features
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Medium-High
**Why Important**: Laplacian eigenvectors capture global graph structure (spectral information).

**Options**:
1. **Laplacian Positional Encoding (LPE)**: Top-k eigenvectors of normalized Laplacian
2. **Random Walk Positional Encoding (RWPE)**: `diag(RW^k)` for k=1,...,K

**Trade-offs**:
- LPE is expensive to compute (eigendecomposition)
- RWPE is cheaper but less informative
- For 10k graphs, precomputation is feasible

**Implementation Tasks**:
- [ ] Implement LPE computation (k=8 eigenvectors)
- [ ] Add to dataset preprocessing (can be cached)
- [ ] Update `input_dim` accordingly
- [ ] Experiment with/without LPE

---

#### 4.3 y_init and z_init
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Medium
**Why Important**: Initialize predictions based on heuristics instead of zeros.

**Explanation**:
- **y_init**: Initial guess for node inclusion probability
- **z_init**: Initial hidden state

**Options for y_init**:
1. Degree-based: `y_init = 1 / (1 + degree)` (low-degree nodes more likely in MIS)
2. LP-based: Use LP relaxation solution as initialization
3. Greedy-based: Run greedy algorithm, use as soft labels

**Implementation Tasks**:
- [ ] Implement degree-based y_init: `y_init[i] = 1 / (1 + deg[i])`
- [ ] Modify `initial_carry()` to use y_init instead of zeros
- [ ] Optionally: precompute LP solution for z_init
- [ ] Compare convergence speed with/without initialization

---

### **Priority 5: Architecture Experiments**

#### 5.1 Graph Transformer Networks (GTN) vs GIN
**Importance**: ⭐⭐⭐⭐ (4/5)
**Effort**: High
**Why Important**: Graph Transformers can capture long-range dependencies better than message-passing GNNs.

**Current Architecture**: GIN (Graph Isomorphism Network)
- Good at local structure
- Limited receptive field (num_layers hops)
- May miss global patterns

**Graph Transformer Options** (PyG):
1. `TransformerConv`: Basic graph transformer layer
2. `GPSConv`: General, Powerful, Scalable (combines MPNN + Transformer)
3. `GATv2Conv`: Graph Attention with dynamic attention

**Recommended**: Start with `GPSConv` as it's designed for this use case.

**Reference**: https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/graph_transformer.html

**Implementation Tasks**:
- [ ] Read PyG Graph Transformer tutorial
- [ ] Implement `GraphTransformerTRM` variant using `GPSConv`
- [ ] Add positional encoding (required for transformers)
- [ ] Compare: GIN vs GTN on same data
- [ ] Track: performance vs compute cost

---

### **Priority 6: Hyperparameter Tuning**

#### 6.1 Learning Rate Experiments
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Low
**Why Important**: Current LR=1e-3 may be too high, causing instability or overshooting.

**Current State**:
- `lr = 1e-3`
- Warmup: 200 steps
- Cosine schedule with min_ratio=0.1

**Experiments**:
| LR | Expected Behavior |
|----|-------------------|
| 1e-3 | Current (possibly too fast) |
| 5e-4 | Slower, more stable |
| 1e-4 | Very slow, may need more epochs |
| 3e-4 | Good middle ground |

**Implementation Tasks**:
- [ ] Run experiments with LR ∈ {1e-4, 3e-4, 5e-4, 1e-3}
- [ ] Track loss curves for each
- [ ] Choose LR with best final loss (we want overfitting!)

---

#### 6.2 Weight Decay Experiments
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Low
**Why Important**: TRM paper uses WD=1.0, we use WD=0.1. Since we WANT overfitting, lower WD may be better.

**Hypothesis**:
- High WD prevents overfitting (regularization)
- For our goal (overfit on training), lower WD is better

**Experiments**:
| WD | Expected Behavior |
|----|-------------------|
| 0.01 | Minimal regularization, easy overfit |
| 0.1 | Current |
| 0.5 | More regularization |
| 1.0 | TRM paper default |

**Implementation Tasks**:
- [ ] Run experiments with WD ∈ {0.01, 0.1, 0.5, 1.0}
- [ ] Track training loss (lower is better for overfitting)
- [ ] Choose WD that gives best training performance

---

#### 6.3 Model Size Experiments
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Medium
**Why Important**: Model should be <= 7M parameters. Smaller models are faster and may generalize better.

**Current Config**:
- `hidden_dim = 256`
- `num_layers = 2`
- `cycles = 18`

**Experiments**:
| Config | hidden_dim | layers | Approx Params |
|--------|------------|--------|---------------|
| Tiny | 64 | 2 | ~50K |
| Small | 128 | 2 | ~200K |
| Medium | 256 | 2 | ~800K |
| Large | 256 | 4 | ~1.5M |
| XL | 512 | 4 | ~6M |

**Implementation Tasks**:
- [ ] First: count current model parameters
- [ ] Experiment with smaller configs
- [ ] Find smallest model that achieves good training performance

---

### **Priority 7: Advanced Infrastructure**

#### 7.1 EMA Model for Validation
**Importance**: ⭐⭐⭐ (3/5)
**Effort**: Medium
**Why Important**: EMA (Exponential Moving Average) often gives smoother, better-generalizing predictions.

**What is EMA?**
- Maintain shadow weights: `ema_weight = decay * ema_weight + (1-decay) * current_weight`
- Typical decay: 0.999 or 0.9999
- Use EMA weights for validation/inference

**Current State**:
- `models/ema.py` exists (need to check implementation)
- Not integrated into training loop

**Implementation Tasks**:
- [ ] Review existing `models/ema.py`
- [ ] Integrate EMA into training loop
- [ ] Log validation metrics for both regular and EMA model
- [ ] Compare: regular vs EMA performance

---

## 📊 Implementation Phases

### Phase 1: Infrastructure & Logging (Days 1-2)
**Goal**: Proper experiment tracking

| Task | File | Priority |
|------|------|----------|
| Add parameter counting | `train_mis.py` | ✅ Done |
| Implement train/val split | `mis_dataset.py` | ✅ Done |
| Add validation loop | `train_mis.py` | ✅ Done |
| Add post-processing | `train_mis.py` | ✅ Done |
| Log loss magnitudes (raw) | `graph_trm.py` | ✅ Done |

**Deliverables**:
- [x] Model size logged at start
- [x] Validation metrics after each epoch
- [x] Post-processed metrics logged
- [x] Raw loss magnitudes visible

---

### Phase 2: Loss Function Experiments (Days 3-4)
**Goal**: Find optimal loss weights

| Experiment | Config Changes |
|------------|----------------|
| Feasibility weight sweep | `feasibility_weight ∈ {0, 1, 5, 10, 20}` |
| Sparsity ablation | `sparsity_weight ∈ {0, 0.3, 1.0}` |

**Deliverables**:
- [ ] Feasibility weight vs F1 curve
- [ ] Optimal weight determined
- [ ] Decision on sparsity loss

---

### Phase 3: Data Quality Analysis (Days 5-6)
**Goal**: Verify data is hard enough

| Task | Tool |
|------|------|
| Compute LP relaxation | scipy/cvxpy |
| Compare LP vs ILP | Analysis script |
| Generate harder data if needed | Dataset script |

**Deliverables**:
- [ ] LP-ILP gap distribution
- [ ] Decision on data quality
- [ ] Harder dataset if needed

---

### Phase 4: Feature Engineering (Days 7-8)
**Goal**: Better input features

| Feature | Implementation |
|---------|----------------|
| Enhanced degree features | Dataset update |
| Laplacian Positional Encoding | New preprocessing |
| y_init from degree | Model update |

**Deliverables**:
- [ ] New feature set
- [ ] Comparison: old vs new features

---

### Phase 5: Architecture Experiments (Days 9-12)
**Goal**: Test Graph Transformers

| Task | Implementation |
|------|----------------|
| Implement GPSConv variant | New model file |
| Add positional encoding | Dataset + model |
| Compare GIN vs GTN | Experiments |

**Deliverables**:
- [ ] `graph_transformer_trm.py`
- [ ] GIN vs GTN comparison table

---

### Phase 6: Hyperparameter Sweep (Days 13-14)
**Goal**: Final tuning

| Hyperparameter | Values |
|----------------|--------|
| Learning rate | {1e-4, 3e-4, 5e-4, 1e-3} |
| Weight decay | {0.01, 0.1, 0.5, 1.0} |
| Model size | Various configs |

**Deliverables**:
- [ ] Optimal hyperparameters
- [ ] Final model configuration

---

## 📝 Immediate Next Steps (Today)

1. **Count model parameters** - 5 minutes
2. **Add validation split to dataset** - 30 minutes
3. **Add validation loop to training** - 1 hour
4. **Log raw loss magnitudes** - 15 minutes
5. **Implement post-processing** - 1 hour

---

## 🔬 Success Criteria

For **overfitting goal**:
- [ ] Training BCE loss < 0.1
- [ ] Training feasibility > 0.99
- [ ] Training F1 > 0.9
- [ ] Post-processed approx_ratio > 0.95

For **overall progress**:
- [ ] All experiments documented
- [ ] Clear winner for each ablation
- [ ] Final model configuration determined

---

## 📚 References

1. [PyG Graph Transformer Tutorial](https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/graph_transformer.html)
2. [GPS: General, Powerful, Scalable Graph Transformers](https://arxiv.org/abs/2205.12454)
3. [TRM Paper](link-to-trm-paper)
4. [MIS LP Relaxation](https://en.wikipedia.org/wiki/Maximum_independent_set#Linear_programming_relaxation)
