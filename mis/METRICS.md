# Training Metrics Documentation

This document explains all metrics logged during training (`train_mis.py`).

## Quick Reference

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| `loss_total` | 0+ | ↓ decreasing | Combined loss |
| `loss_bce` | 0+ | 0.3-0.5 | Binary cross-entropy |
| `loss_feasibility` | 0-1 | < 0.1 | Edge violation penalty |
| `loss_sparsity` | 0-1 | < 0.01 | Size matching penalty |
| `f1` | 0-1 | > 0.8 | Classification F1 score |
| `precision` | 0-1 | > 0.7 | % predictions correct |
| `recall` | 0-1 | > 0.7 | % true nodes found |
| `feasibility` | 0-1 | > 0.95 | Independence quality |
| `approx_ratio` | 0+ | ~1.0 | Predicted/True size (RAW) |
| `num_violations` | 0+ | ↓ decreasing | Edge violations count |
| `grad_norm` | 0+ | 1-50 | Gradient magnitude |

---

## Loss Components

### `loss_total`

**Formula:** `loss_bce + 1.0 * loss_feasibility + 0.3 * loss_sparsity`

This is the total loss used for backpropagation. Decrease indicates learning.

---

### `loss_bce` (Binary Cross-Entropy)

**What it measures:** How well the model predicts which nodes are in the MIS.

**Formula:** Weighted BCE with `pos_weight` for class imbalance.

**Range:** 0 to ∞ (typically 0.3-0.8)

**Interpretation:**
- **< 0.4** = Good node classification
- **0.4-0.7** = Acceptable
- **> 0.8** = Poor, model struggling

**Note:** Uses **global pos_weight** computed from dataset (not per-batch) for stable training.

---

### `loss_feasibility`

**What it measures:** Penalty for selecting adjacent nodes (violating independence constraint).

**Formula:** `mean(prob[u] * prob[v])` for all edges (u, v)

**Range:** 0 to 1 (typically 0.05-0.2)

**Interpretation:**
- **< 0.05** = Excellent, few violations
- **0.05-0.15** = Good
- **> 0.2** = Too many constraint violations

**Weight in total loss:** 1.0 (can increase to 2.0-3.0 if violations are high)

---

### `loss_sparsity`

**What it measures:** How well predicted set size matches true set size.

**Formula:** `(mean(predictions) - mean(labels))²`

**Range:** 0 to 1 (typically 0.0-0.01)

**Interpretation:**
- **< 0.005** = Excellent size matching
- **< 0.02** = Good
- **> 0.05** = Size mismatch (too greedy or conservative)

**Weight in total loss:** 0.3

---

## Classification Metrics

### `f1` (F1 Score) ⭐ **KEY METRIC**

**Formula:** `2 * (precision * recall) / (precision + recall)`

**Range:** 0 to 1

**Target:** > 0.8

**Interpretation:** Balance between precision and recall. Main indicator of node-level performance.

---

### `precision`

**Formula:** `TP / (TP + FP)`

**What it measures:** Of nodes predicted as MIS, what % are actually in true MIS?

**Range:** 0 to 1

- **High precision** = Model is selective/conservative
- **Low precision** = Too many false positives

---

### `recall`

**Formula:** `TP / (TP + FN)`

**What it measures:** Of true MIS nodes, what % did the model find?

**Range:** 0 to 1

- **High recall** = Model finds most true nodes
- **Low recall** = Missing too many true nodes

---

## MIS-Specific Metrics

### `feasibility` (Raw Feasibility)

**What it measures:** Fraction of predicted nodes that form a valid independent set.

**Formula:** `1 - (edge_violations / num_predicted_nodes)`

**Range:** 0 to 1

**Target:** > 0.95

**⚠️ IMPORTANT:** This is the **HONEST** metric. It measures what the model actually predicted, not after any post-processing.

**Interpretation:**
- **1.0** = Perfect independence (no adjacent nodes selected)
- **0.95** = 95% valid predictions
- **< 0.8** = Many constraint violations, increase `feasibility_weight`

---

### `approx_ratio` (Prediction Size Ratio)

**What it measures:** Raw prediction count vs true MIS size.

**Formula:** `num_pred_1s / num_true_1s`

**Range:** 0 to ∞ (typically 0.8-1.2)

**Target:** ~1.0

**⚠️ IMPORTANT:** This is the **HONEST** metric - it measures raw predictions without any post-processing. If `feasibility` is low, this metric can be misleading because many predictions may be invalid.

**Interpretation:**
- **1.0** = Predicting correct number of nodes
- **> 1.0** = Predicting too many nodes (greedy)
- **< 1.0** = Predicting too few nodes (conservative)

---

### `num_violations`

**What it measures:** Number of edges where both endpoints are selected.

**Formula:** `count(edges where pred[u]==1 AND pred[v]==1) / 2`

**Range:** 0 to num_edges

**Target:** 0 (or as low as possible)

**Interpretation:**
- **0** = Perfect independence
- **> 0** = Model is selecting adjacent nodes
- Track this to see if feasibility is improving

---

### `set_size_ratio`

**What it measures:** Same as `approx_ratio` (kept for compatibility).

---

## Training Health Metrics

### `grad_norm`

**What it measures:** L2 norm of gradients before clipping.

**Range:** 0 to ∞

**Healthy range:** 1-50

**Interpretation:**
- **0.01-0.1** = Vanishing gradients (learning too slow)
- **1-50** = Healthy training
- **100+** = Gradient explosion (may destabilize training)
- **After clipping** = Capped at `grad_clip` value (default: 1.0)

---

## Wandb Logging

All metrics are logged to wandb under `train/` prefix:

```
train/loss_total
train/loss_bce
train/loss_feasibility
train/loss_sparsity
train/f1
train/precision
train/recall
train/feasibility
train/approx_ratio
train/num_violations
train/grad_norm
train/num_pred_1s
train/num_true_1s
train/set_size_ratio
```

---

## Interpreting Training Progress

### Good Training Trajectory

```
Epoch 1:   loss=1.5, f1=0.3, feasibility=0.6
Epoch 10:  loss=0.8, f1=0.6, feasibility=0.85
Epoch 50:  loss=0.5, f1=0.75, feasibility=0.92
Epoch 100: loss=0.4, f1=0.82, feasibility=0.97
```

✅ Loss decreasing, F1 increasing, feasibility improving.

### Bad Signs

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Loss stuck > 1.0 | Learning rate too low/high | Try LR 1e-4 to 1e-2 |
| F1 stuck < 0.5 | Model not learning patterns | Check data, increase capacity |
| Feasibility stuck < 0.7 | Ignoring constraints | Increase `feasibility_weight` to 2.0+ |
| approx_ratio > 1.5 | Too greedy | Increase `sparsity_weight` |
| approx_ratio < 0.6 | Too conservative | Decrease `feasibility_weight` |
| grad_norm spikes | Unstable training | Reduce LR, check for NaN |

---

## Key Changes from Previous Version

1. **`approx_ratio` is now HONEST** - It shows raw prediction size, not post-processed.
2. **Removed `valid_set_size`** - This was a "cheated" metric that subtracted violations.
3. **Added `num_violations`** - Direct count of constraint violations.
4. **Global `pos_weight`** - Computed once from dataset, not per-batch.
