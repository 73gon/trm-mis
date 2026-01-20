# MIS-TRM Metrics Calculations Reference

This document explains all metric calculations in the MIS-TRM codebase. Use this to verify correctness and understand what each metric means.

---

## Table of Contents
1. [Graph Notation](#graph-notation)
2. [Loss Functions](#loss-functions)
3. [Feasibility Metrics](#feasibility-metrics)
4. [Classification Metrics](#classification-metrics)
5. [Set Size Metrics](#set-size-metrics)
6. [Post-Processing Metrics](#post-processing-metrics)
7. [TRM-Specific Metrics](#trm-specific-metrics)

---

## Graph Notation

| Symbol | Meaning |
|--------|---------|
| `N` | Number of nodes in graph |
| `E` | Number of edges (undirected) |
| `edge_index` | `[2, 2E]` tensor - each undirected edge appears twice: (u,v) and (v,u) |
| `labels` | Ground truth MIS membership: 1 = in MIS, 0 = not in MIS |
| `probs` | Model output probabilities after sigmoid, range [0, 1] |
| `preds_binary` | Binary predictions: `(probs > 0.5).float()` |

---

## Loss Functions

### 1. Binary Cross-Entropy Loss (BCE)

**Location**: `models/graph_transformer_trm.py` line ~443-449

```python
bce_loss = F.binary_cross_entropy_with_logits(
    logits,
    labels,
    pos_weight=pos_weight
)
```

**Formula**:
$$\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w \cdot y_i \cdot \log(\sigma(z_i)) + (1-y_i) \cdot \log(1-\sigma(z_i)) \right]$$

Where:
- $y_i$ = ground truth label (0 or 1)
- $z_i$ = model logits (before sigmoid)
- $\sigma(z_i)$ = sigmoid(logits) = probability
- $w$ = `pos_weight` (handles class imbalance)

**pos_weight calculation**:
$$\text{pos\_weight} = \frac{\text{negative\_count}}{\text{positive\_count}} = \frac{N - |MIS|}{|MIS|}$$

**Purpose**: Higher pos_weight penalizes missing MIS nodes more than false positives.

**What it does**: Measures how well the model predicts node membership. Lower BCE means better classification on individual nodes, regardless of feasibility.

---

### 2. Feasibility Loss

**Location**: `models/graph_transformer_trm.py` line ~453-459

```python
edge_violations = probs[src] * probs[dst]
feasibility_loss = edge_violations.mean()
```

**Formula**:
$$\text{Feas\_Loss} = \frac{1}{2E} \sum_{(u,v) \in \text{edge\_index}} p_u \cdot p_v$$

Where:
- $p_u, p_v$ = probability of node u, v being in MIS
- Sum is over all directed edges (each undirected edge counted twice)

**Purpose**: Penalizes selecting adjacent nodes. When both $p_u$ and $p_v$ are high, the product is high → high loss.

**Range**: [0, 1] (0 = no violations, 1 = all edges violated)

**What it does**: Guides the model to learn which edges are violated. Directly optimizes for feasibility constraint satisfaction during training. Complements BCE by explicitly penalizing adjacent node selection.

---

### 3. Total Loss

```python
loss = bce_loss + feasibility_weight * feasibility_loss
```

$$\text{Total\_Loss} = \text{BCE} + \lambda \cdot \text{Feas\_Loss}$$

Where $\lambda$ = `feasibility_weight` (default: 0.0, increase to enforce feasibility)

**What it does**: Combined objective function. BCE trains the model to identify MIS nodes accurately, while feasibility loss trains it to avoid selecting adjacent nodes. Together they minimize both classification errors and constraint violations.

---

## Feasibility Metrics

### 1. Violations Count

**Location**: `models/graph_transformer_trm.py` line ~488-494

```python
violations = (pred_mask[src] & pred_mask[dst]).sum().float()
num_violations = (violations / 2).ceil()
```

**Formula**:
$$\text{violations\_raw} = \sum_{(u,v) \in \text{edge\_index}} \mathbb{1}[\text{pred}_u = 1] \cdot \mathbb{1}[\text{pred}_v = 1]$$

$$\text{num\_violations} = \lceil \frac{\text{violations\_raw}}{2} \rceil$$

**Explanation**:
- `violations_raw` counts each violating edge twice (once for (u,v), once for (v,u))
- `num_violations` is the actual number of violated undirected edges

**What it does**: Counts how many edges were violated by the predictions. Lower is better (0 = fully feasible). Essential metric for evaluating constraint satisfaction.

---

### 2. Feasibility Loss

**Location**: `models/graph_transformer_trm.py` line ~453-459

```python
probs = torch.sigmoid(logits_clamped)
src, dst = edge_index[0], edge_index[1]
edge_violations = probs[src] * probs[dst]
feasibility_loss = edge_violations.mean()
```

**Formula**:
$$\text{Feas\_Loss} = \frac{1}{2E} \sum_{(u,v) \in \text{edge\_index}} p_u \cdot p_v$$

Where:
- $p_u, p_v$ = probability of node u, v being in MIS (after sigmoid)
- Sum is over all directed edges in `edge_index` (each undirected edge counted twice)
- Taking mean automatically handles the double-counting

**Purpose**: Penalizes selecting adjacent nodes. When both $p_u$ and $p_v$ are high, their product is high → high loss.

**Range**: [0, 1] where 0 = no violations, 1 = all edges violated

**Scale Behavior**:
- With few violations: loss ≈ 0 → easy to optimize
- With many violations: loss ≈ 0.5-1.0 → significant gradient signal
- Balanced with BCE loss via `feasibility_weight` parameter

**Double-Counting Note**: Each undirected edge appears twice in `edge_index`, so the product $p_u \cdot p_v$ is computed twice. However, taking the mean normalizes this out correctly. This is equivalent to computing the mean over unique edges only.

**Location**: `models/graph_transformer_trm.py` line ~490-493

```python
total_edges = float(edge_index.size(1))
feasibility_raw = 1.0 - (violations / total_edges).clamp(max=1.0)
```

**Formula**:
$$\text{Feasibility} = 1 - \min\left(\frac{\text{violations\_raw}}{2E}, 1\right)$$

Where $2E$ = `edge_index.size(1)` (total directed edges)

**Range**: [0, 1]
- 1.0 = no violations (perfect feasibility)
- 0.0 = all edges violated

**Note**: We use `violations_raw / total_edges` (not `violations_raw / 2E`) because both numerator and denominator include the double-counting, so they cancel out correctly.

**What it does**: Reports the feasibility ratio in the forward pass. Directly shows whether the model's predictions satisfy the edge constraints. Used for real-time monitoring during training.

---

### 3. Feasibility Diagnostics

**Location**: `train_mis.py` line ~24-69

```python
violation_rate = violations / total_edges
raw_feasibility = 1.0 - min(violation_rate, 1.0)
```

Same formula as above, used for logging/debugging.

**What it does**: Computes feasibility with diagnostic output. Helps identify constraint violations early during debugging and monitoring.

---

## Classification Metrics

### 1. Accuracy

**Location**: `models/graph_transformer_trm.py` line ~500

```python
acc = (preds_binary == labels).float().mean()
```

**Formula**:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{pred}_i = y_i]$$

**What it does**: Measures the fraction of correct node predictions overall. Balanced metric when class distribution is equal, but can be misleading with class imbalance.

---

### 2. Precision, Recall, F1

**Location**: `models/graph_transformer_trm.py` line ~476-482

```python
tp = (preds_binary * labels).sum().float()
fp = (preds_binary * (1 - labels)).sum().float()
fn = ((1 - preds_binary) * labels).sum().float()

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
```

**Formulas**:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where:
- **TP** (True Positive): Correctly predicted MIS nodes
- **FP** (False Positive): Predicted MIS but not in ground truth
- **FN** (False Negative): In ground truth MIS but not predicted

**What they do**:
- **Precision**: Of predicted MIS nodes, how many are correct? (penalizes false positives)
- **Recall**: Of ground truth MIS nodes, how many did we find? (penalizes false negatives)
- **F1**: Harmonic mean balances precision and recall. Use when you want both high accuracy and coverage.

---

## Set Size Metrics

### 1. Predicted Set Size

```python
num_pred_1s = preds_binary.sum()
```

$$|\text{Pred}| = \sum_{i=1}^{N} \mathbb{1}[\text{pred}_i = 1]$$

**What it does**: Counts how many nodes the model selected. Larger values indicate more aggressive selection; lower values indicate more conservative predictions.

### 2. Optimal Set Size (Ground Truth)

```python
num_true_1s = labels.sum()
```

$$|\text{MIS}^*| = \sum_{i=1}^{N} y_i$$

**What it does**: Reports the ground truth MIS size. Used as baseline for comparison. Ideal solution has this many nodes selected.

### 3. Set Size Ratio

```python
set_size_ratio = num_pred_1s / (num_true_1s + 1e-8)
```

$$\text{Size\_Ratio} = \frac{|\text{Pred}|}{|\text{MIS}^*|}$$

**Interpretation**:
- < 1.0: Under-predicting (missing MIS nodes)
- = 1.0: Same size as optimal
- > 1.0: Over-predicting (but may have violations)

**What it does**: Compares prediction size relative to optimal. Indicates whether the model is selecting too few or too many nodes.

### 4. Approximation Ratio (Raw)

```python
approx_ratio_raw = num_pred_1s / (num_true_1s + 1e-8)
```

Same as set_size_ratio. **Note**: This is NOT a true approximation ratio because the raw prediction may be infeasible.

**What it does**: Raw quality metric before any post-processing. Useful for understanding model's raw output quality before constraint satisfaction is enforced.

---

## Post-Processing Metrics

**Location**: `train_mis.py` function `compute_postprocessing_metrics()`

### 1. Greedy Decode

**Location**: `train_mis.py` line ~92-131

```python
def greedy_decode(probs, edge_index, num_nodes):
    sorted_nodes = np.argsort(-probs_np)  # Sort by probability (descending)
    for node in sorted_nodes:
        if node not in blocked_nodes:
            selected_set.add(node)
            blocked_nodes.add(node)
            for neighbor in adj[node]:
                blocked_nodes.add(neighbor)
    return len(selected_set), selected_tensor
```

**Algorithm**:
1. Sort nodes by probability (highest first)
2. Greedily select nodes if not blocked
3. When selecting a node, block all its neighbors

**Guarantee**: Always produces a **valid** (feasible) independent set.

**What it does**: Post-processing algorithm to convert soft predictions to hard constraints. Ensures the final solution is always feasible by greedily selecting high-probability nodes while respecting edge constraints.

---

### 2. Postprocessed Size

```python
postprocessed_size, _ = greedy_decode(probs, edge_index, num_nodes)
```

Size of the independent set after greedy decoding.

**What it does**: Reports the size of the feasible solution produced by greedy decode. Always feasible but may be suboptimal due to greedy nature.

---

### 3. Gap

```python
gap = optimal_size - postprocessed_size
```

$$\text{Gap} = |\text{MIS}^*| - |\text{Greedy}|$$

**Interpretation**: How many nodes short of optimal.

**What it does**: Measures the absolute quality loss from post-processing. Smaller gap = better approximation quality.

---

### 4. Gap Ratio

```python
gap_ratio = gap / (optimal_size + 1e-8)
```

$$\text{Gap\_Ratio} = \frac{|\text{MIS}^*| - |\text{Greedy}|}{|\text{MIS}^*|}$$

**Range**: [0, 1]
- 0.0 = optimal solution found
- 1.0 = empty set (worst case)

**What it does**: Normalized gap. Shows what fraction of the optimal solution we're losing. Directly comparable across different graph sizes.

---

### 5. Approximation Ratio (Postprocessed)

```python
approx_ratio_postprocessed = postprocessed_size / (optimal_size + 1e-8)
```

$$\text{Approx\_Ratio} = \frac{|\text{Greedy}|}{|\text{MIS}^*|}$$

**Range**: [0, 1]
- 1.0 = optimal solution found
- 0.0 = empty set

**This is the TRUE approximation ratio** because the greedy solution is always feasible.

**What it does**: Key quality metric showing how close the greedy solution is to optimal. This is the main metric used to evaluate approximation quality.

---

## TRM-Specific Metrics

### 1. Q_hat (Confidence)

**Location**: `models/graph_transformer_trm.py` line ~467-473

```python
correct = (preds_binary == labels).float()
confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]
q_hat = (correct * confidence).mean()
```

**Formula**:
$$\text{Q\_hat} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{pred}_i = y_i] \cdot 2 \cdot |p_i - 0.5|$$

**Interpretation**:
- High Q_hat = model is confident AND correct
- Used for early stopping in deep supervision

**What it does**: Combines correctness with confidence to identify when the model makes reliable predictions. Used as an early stopping criterion in deep supervision (stops gradient flow to early steps if Q_hat is high).

---

### 2. Steps to Solve

**Location**: `models/graph_transformer_trm.py` function `deep_recursion_with_tracking()`

Tracks at which recursion step the model first achieves a "perfect" prediction (100% accuracy + 0 violations).

$$\text{Steps\_to\_solve} = \min\{t : \text{Acc}_t = 1 \land \text{Violations}_t = 0\}$$

**Range**: [1, H_cycles × L_cycles]

**What it does**: Measures model efficiency in reasoning. Lower values (solves in fewer steps) indicate better recursion performance. Shows whether increasing recursion depth helps the model find solutions faster.

---

### 3. Solved

```python
is_solved = (acc == 1.0) and (num_violations == 0)
```

Binary indicator: 1 if perfect solution, 0 otherwise.

**What it does**: Reports whether the model found a perfect solution (all nodes correctly classified AND feasible). Success indicator for the entire model pipeline.

---

## Summary Table

| Metric | Range | Optimal Value | Formula |
|--------|-------|---------------|---------|
| `loss_bce` | [0, ∞) | 0 | Cross-entropy |
| `loss_feasibility` | [0, 1] | 0 | Mean edge violation prob |
| `feasibility` | [0, 1] | 1.0 | 1 - violations/edges |
| `num_violations` | [0, E] | 0 | Count of violated edges |
| `accuracy` | [0, 1] | 1.0 | Correct / Total |
| `precision` | [0, 1] | 1.0 | TP / (TP + FP) |
| `recall` | [0, 1] | 1.0 | TP / (TP + FN) |
| `f1` | [0, 1] | 1.0 | Harmonic mean of P & R |
| `approx_ratio_postprocessed` | [0, 1] | 1.0 | Greedy size / Optimal size |
| `gap` | [0, MIS*] | 0 | Optimal - Greedy |
| `gap_ratio` | [0, 1] | 0 | Gap / Optimal |
| `q_hat` | [0, 1] | 1.0 | Confidence × Correctness |
| `steps_to_solve` | [1, H×L] | 1 | First step with perfect solution |

---

## Edge Index Double-Counting Note

**IMPORTANT**: In PyTorch Geometric, `edge_index` for undirected graphs contains each edge **twice**:
- Edge (u, v) appears as both `[u, v]` and `[v, u]`
- So `edge_index.size(1) = 2 * E` where E = actual undirected edges

This affects:
1. **Violation counting**: Raw violations are counted twice, so divide by 2 for actual count
2. **Feasibility normalization**: Using `violations / edge_index.size(1)` is correct because both are doubled

---

## Code Locations Summary

| Calculation | File | Lines |
|-------------|------|-------|
| BCE Loss | `models/graph_transformer_trm.py` | ~443-449 |
| Feasibility Loss | `models/graph_transformer_trm.py` | ~453-459 |
| Feasibility Metric | `models/graph_transformer_trm.py` | ~487-496 |
| Precision/Recall/F1 | `models/graph_transformer_trm.py` | ~476-482 |
| Greedy Decode | `train_mis.py` | ~92-131 |
| Postprocessing Metrics | `train_mis.py` | ~133-302 |
| Feasibility Diagnostics | `train_mis.py` | ~24-69 |

---

## Changelog

- **2026-01-20**: Fixed feasibility calculation to normalize by total edges instead of selected nodes
- **2026-01-20**: Made `use_postprocessing=False` report raw prediction size (not size minus violations)
- **2026-01-20**: Unified feasibility calculation across all functions
