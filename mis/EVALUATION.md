# Evaluation Documentation: MIS with Graph TRM

This document provides a comprehensive explanation of how evaluation works for the Maximum Independent Set (MIS) model, including greedy decoding, metrics computation, and result interpretation.

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Greedy Decoding](#greedy-decoding)
3. [Metrics Computation](#metrics-computation)
4. [Running Evaluation](#running-evaluation)
5. [Interpreting Results](#interpreting-results)
6. [Comparison with Training](#comparison-with-training)
7. [Visualization Tools](#visualization-tools)
8. [Advanced Topics](#advanced-topics)

---

## Evaluation Overview

### What Evaluation Does

```
Trained Model + Test Graphs → Predictions → Metrics
```

1. **Load checkpoint** from training
2. **Run inference** on test graphs (or training graphs for sanity check)
3. **Compute raw predictions** (probabilities per node)
4. **Apply greedy decoding** to get valid independent sets
5. **Calculate metrics** comparing to optimal solutions
6. **Log to wandb** for visualization

### Why Evaluation is Different from Training

| Aspect | Training | Evaluation |
|--------|----------|------------|
| Goal | Minimize loss | Measure quality |
| Gradients | Computed | Not computed (`torch.no_grad()`) |
| Batching | Drop partial | Include all samples |
| Output | Loss values | Metrics (approx_ratio, F1, etc.) |
| Post-processing | Raw logits | Greedy decode to valid set |

---

## Greedy Decoding

### Why Greedy Decoding?

Raw model predictions may **violate the independence constraint**:

```
Model predicts: {A, B, C, D, E}
But edge (B, D) exists in graph!
→ Invalid solution (not an independent set)
```

Greedy decoding **fixes violations** to produce a valid MIS:

```
After greedy: {A, B, C, E}  (D removed because B was selected first)
→ Valid independent set!
```

### Algorithm

```python
def greedy_decode(probs, edge_index, num_nodes):
    """
    Convert probabilities to valid independent set.

    Strategy:
    1. Sort nodes by probability (highest first)
    2. Greedily select nodes
    3. Block neighbors of selected nodes
    """
    # Build adjacency list
    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Sort by probability (descending)
    sorted_nodes = argsort(-probs)

    selected = set()
    blocked = set()

    for node in sorted_nodes:
        if node in blocked:
            continue  # Skip if neighbor already selected

        selected.add(node)
        blocked.add(node)
        blocked.update(adj[node])  # Block all neighbors

    return selected
```

### Properties

| Property | Value |
|----------|-------|
| **Always valid?** | ✅ Yes, produces valid independent set |
| **Optimal?** | ❌ No, greedy doesn't guarantee optimality |
| **Deterministic?** | ✅ Yes, same probs → same result |
| **Time complexity** | O(n log n + m) where n=nodes, m=edges |

### Example Walkthrough

```
Graph:  A --- B --- C
        |     |
        D --- E

Probabilities: A=0.9, B=0.8, C=0.7, D=0.6, E=0.5

Sorted order: [A, B, C, D, E]

Step 1: Select A → Block {A, B, D}
Step 2: B blocked, skip
Step 3: Select C → Block {C, B, E}  (B already blocked)
Step 4: D blocked, skip
Step 5: E blocked, skip

Result: {A, C} (valid independent set, size 2)
Optimal: {A, C, E} or {B, D} (size 3)
→ approx_ratio = 2/3 ≈ 0.67
```

---

## Metrics Computation

### Classification Metrics (Raw Predictions)

These compare raw model predictions (threshold at 0.5) to ground truth labels:

```python
preds_binary = (sigmoid(logits) > 0.5).float()
labels = ground_truth

# True Positives: Correctly predicted as MIS
TP = (preds_binary * labels).sum()

# False Positives: Predicted as MIS but not in true MIS
FP = (preds_binary * (1 - labels)).sum()

# False Negatives: In true MIS but not predicted
FN = ((1 - preds_binary) * labels).sum()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

### Feasibility Metrics

```python
# Raw feasibility: Before greedy decoding
pred_mask = (preds_binary == 1)
violations = 0
for u, v in edges:
    if pred_mask[u] and pred_mask[v]:
        violations += 1

feasibility_raw = 1.0 - violations / num_predicted_nodes

# Greedy feasibility: After greedy decoding (should always be 1.0)
greedy_set = greedy_decode(probs, edges, num_nodes)
feasibility_greedy = is_valid_independent_set(greedy_set, edges)
```

### Size Metrics

```python
# How many nodes did model predict vs ground truth?
num_pred_1s = preds_binary.sum()
num_true_1s = labels.sum()
set_size_ratio = num_pred_1s / num_true_1s

# Raw prediction size vs optimal (NOT a valid approx ratio if infeasible)
pred_size_vs_opt = num_pred_1s / optimal_size

# After greedy: Valid approximation ratio
greedy_size = len(greedy_decode(probs, edges, num_nodes))
approx_ratio_greedy = greedy_size / optimal_size
```

### Metric Summary

| Metric | What it measures | Range | Target |
|--------|------------------|-------|--------|
| `precision` | % predictions correct | 0-1 | > 0.7 |
| `recall` | % true nodes found | 0-1 | > 0.7 |
| `f1` | Harmonic mean of P & R | 0-1 | > 0.8 |
| `feasibility_raw` | Independence of raw preds | 0-1 | > 0.95 |
| `feasibility_greedy` | Independence after greedy | 0-1 | 1.0 |
| `set_size_ratio` | Pred size / true size | 0-∞ | ~1.0 |
| `pred_size_vs_opt` | Pred size / optimal | 0-∞ | ~1.0 |
| `approx_ratio_greedy` | Greedy size / optimal | 0-1 | > 0.95 |

---

## Running Evaluation

### Basic Usage

```bash
# Evaluate on test set
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt \
                   --data_path data/test_mis \
                   --wandb

# Evaluate on training set (sanity check)
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt \
                   --data_path data/mis-10k \
                   --wandb
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/mis/epoch_99.pt` | Model checkpoint path |
| `--data_path` | `data/test_mis` | Dataset directory |
| `--hidden_dim` | 256 | Model hidden dimension |
| `--num_layers` | 2 | GNN layers per cycle |
| `--cycles` | 18 | Recursive reasoning steps |
| `--batch_size` | 64 | Batch size for eval |
| `--max_samples` | 999999 | Max samples to evaluate |
| `--wandb` | False | Enable wandb logging |
| `--wandb_project` | `MIS-TRM` | Wandb project name |
| `--wandb_run_name` | Auto | Wandb run name |

### Output

```
============================================================
EVALUATION RESULTS
============================================================

📊 KEY METRICS (n=1000 samples):
--------------------------------------------------
  approx_ratio_greedy      : 0.9234 ± 0.0567  (min=0.7500, max=1.0000)
  pred_size_vs_opt         : 1.0123 ± 0.1234  (min=0.6000, max=1.5000)
  f1                       : 0.8012 ± 0.0987  (min=0.5000, max=0.9500)
  precision                : 0.7856 ± 0.1123  (min=0.4500, max=0.9800)
  recall                   : 0.8234 ± 0.0876  (min=0.5500, max=0.9900)
  feasibility_raw          : 0.9567 ± 0.0345  (min=0.8000, max=1.0000)
  feasibility_greedy       : 1.0000 ± 0.0000  (min=1.0000, max=1.0000)
  set_size_ratio           : 1.0234 ± 0.1567  (min=0.5000, max=1.8000)

📈 SIZE METRICS:
--------------------------------------------------
  Total Optimal Nodes:    45678
  Total Predicted Nodes:  46789
  Total Valid Set Size:   42345
  Overall Approx Ratio:   0.9270

✅ Results logged to wandb: MIS-TRM
```

---

## Interpreting Results

### Good Results

```
approx_ratio_greedy: 0.95 ± 0.05
pred_size_vs_opt:    1.00 ± 0.10
feasibility_raw:     0.97 ± 0.03
f1:                  0.85 ± 0.08
```

✅ **Interpretation:**
- Model finds 95% of optimal solution on average
- Predicts correct number of nodes
- Almost all predictions are feasible
- Good node-level classification

### Overly Greedy Model

```
approx_ratio_greedy: 0.72 ± 0.15
pred_size_vs_opt:    1.80 ± 0.50  ← Much higher than 1.0!
feasibility_raw:     0.55 ± 0.20  ← Low!
recall:              0.90 ± 0.05  ← High recall but...
precision:           0.45 ± 0.15  ← Low precision
```

⚠️ **Interpretation:**
- Model predicts 80% more nodes than optimal
- Many predictions violate independence
- High recall (finds true nodes) but low precision (too many false positives)
- Greedy decoding removes many nodes → low final approx ratio

**Fix:** Increase `feasibility_weight` or `sparsity_weight` during training

### Too Conservative Model

```
approx_ratio_greedy: 0.60 ± 0.15
pred_size_vs_opt:    0.55 ± 0.20  ← Much lower than 1.0!
feasibility_raw:     0.99 ± 0.01  ← Almost perfect
recall:              0.50 ± 0.15  ← Low!
precision:           0.95 ± 0.05  ← High precision
```

⚠️ **Interpretation:**
- Model predicts only 55% of optimal nodes
- Almost all predictions are feasible (too cautious)
- Low recall (missing many true nodes)
- High precision (what it predicts is correct)

**Fix:** Decrease `feasibility_weight` during training

### Understanding the Gap

The gap between `pred_size_vs_opt` and `approx_ratio_greedy` shows feasibility losses:

```
pred_size_vs_opt:    1.20  (model predicts 120% of optimal)
approx_ratio_greedy: 0.85  (after greedy, only 85% remains)
Gap:                 0.35  (35% lost to feasibility violations)
```

---

## Comparison with Training

### Overfitting Detection

Run evaluation on both training and test data:

```bash
# Training data
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt \
                   --data_path data/mis-10k --wandb

# Test data
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt \
                   --data_path data/test_mis --wandb
```

Compare in wandb:

| Metric | Train | Test | Status |
|--------|-------|------|--------|
| approx_ratio_greedy | 0.92 | 0.90 | ✅ OK |
| approx_ratio_greedy | 0.92 | 0.75 | ⚠️ Overfitting |
| approx_ratio_greedy | 0.92 | 0.60 | ❌ Severe overfitting |

### Metric Correspondence

| Training Metric | Eval Metric | Notes |
|-----------------|-------------|-------|
| `train/f1` | `eval/f1` | Should be similar |
| `train/feasibility` | `eval/feasibility_raw` | Should be similar |
| `train/approx_ratio` | `eval/pred_size_vs_opt` | Raw prediction ratio |
| N/A | `eval/approx_ratio_greedy` | Only computed at eval |

---

## Visualization Tools

### Interactive Visualizer

```bash
python mis/visualize_predictions.py \
    --checkpoint checkpoints/mis/epoch_99.pt \
    --data_path data/test_mis
```

**Features:**
- Navigate through samples: `n` (next), `p` (previous), `g` (go to)
- View predictions vs ground truth
- Color-coded nodes:
  - 🔴 Red: True MIS (model missed)
  - 🔵 Blue: Predicted (not in true MIS)
  - 🟣 Purple: Correct prediction
  - ⚪ Gray: Correctly not selected

**Output:**
```
======================================================================
SAMPLE 42/1000
======================================================================
Graph seed: 12345
Nodes: 150, Edges: 450
True MIS size: 45
Predicted MIS size: 43
Approx Ratio: 0.9556
Predicted set: ✅ Valid (Independent)
True set: ✅ Valid (Independent)

📊 Creating visualization: visualizations/mis_predictions/sample_0042.png
```

### Batch Visualization

```bash
# In visualizer, press 's' to save all samples
```

Creates PNG files for all samples in `visualizations/mis_predictions/`.

---

## Advanced Topics

### Batched Evaluation

Evaluation processes multiple graphs in parallel:

```python
# PyG batches multiple graphs
batch = Batch.from_data_list([graph1, graph2, ..., graph64])

# Single forward pass for all graphs
probs = model(batch)

# Separate results by graph
for i in range(batch.num_graphs):
    start = batch.ptr[i]
    end = batch.ptr[i+1]
    graph_probs = probs[start:end]
    graph_edges = extract_edges(batch, i)
    result = greedy_decode(graph_probs, graph_edges)
```

### Optimal Values

We use the **optimal MIS size** from Gurobi (stored during data generation):

```python
# From batch
optimal_size = batch["opt_value"][i].item()

# Not the label sum! (labels might be different optimal solution)
# optimal_size ≠ labels.sum() in general
```

**Why?**
- MIS can have multiple optimal solutions
- Labels store ONE optimal solution
- opt_value stores the SIZE of optimal (consistent)

### Per-Batch vs Final Metrics

Evaluation logs **per-batch** to wandb for smooth curves:

```python
for batch in dataloader:
    # Compute metrics for this batch
    metrics = compute_metrics(batch)

    # Log immediately (creates datapoint)
    wandb.log({
        "eval/approx_ratio_greedy": metrics["approx_ratio"],
        "eval/sample_num": sample_count,
    })
```

**Benefits:**
- See metric distribution across samples
- Identify outliers (some graphs harder than others)
- Smooth curves for monitoring

### Validation During Training

You can add validation during training:

```python
# In train_mis.py, after each epoch:
if epoch % 10 == 0:
    model.eval()
    val_metrics = evaluate(model, val_dataloader)
    wandb.log({"val/approx_ratio": val_metrics["approx_ratio"]})
    model.train()
```

---

## Summary

**Evaluation pipeline:**

1. **Load model** from checkpoint
2. **Inference** on test graphs (no gradients)
3. **Raw metrics** from thresholded predictions (F1, feasibility_raw)
4. **Greedy decode** to get valid independent sets
5. **Final metrics** comparing greedy sets to optimal (approx_ratio_greedy)
6. **Log to wandb** per-batch for smooth curves

**Key metrics:**

| Metric | What to look for |
|--------|------------------|
| `approx_ratio_greedy` | Main metric, target > 0.95 |
| `feasibility_raw` | Model quality, target > 0.95 |
| `f1` | Node classification, target > 0.80 |
| Gap between raw and greedy | Feasibility losses |

**Debugging:**

| Issue | Check |
|-------|-------|
| Low approx_ratio | Compare pred_size_vs_opt to see if greedy or conservative |
| High pred_size_vs_opt, low approx_ratio | Too many violations, increase feasibility_weight |
| Low pred_size_vs_opt | Too conservative, decrease feasibility_weight |
| Train >> Test | Overfitting, reduce epochs or increase regularization |
