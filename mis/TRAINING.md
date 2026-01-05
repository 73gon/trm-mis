# Training Documentation: MIS with Graph TRM

This document provides a comprehensive explanation of how the Maximum Independent Set (MIS) training works, covering the problem, model architecture, training loop, and all implementation details.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Model Architecture](#model-architecture)
3. [Training Loop](#training-loop)
4. [Loss Functions](#loss-functions)
5. [Dataset and Batching](#dataset-and-batching)
6. [Hyperparameters](#hyperparameters)
7. [Implementation Details](#implementation-details)
8. [Troubleshooting](#troubleshooting)

---

## Problem Definition

### Maximum Independent Set (MIS)

Given an undirected graph G = (V, E), find the largest set of vertices S ⊆ V such that no two vertices in S are adjacent.

**Example:**
```
Graph:      1 --- 2 --- 3
            |     |
            4 --- 5

Optimal MIS: {1, 3, 5} or {2, 4} (size 3 is optimal here)
```

**Why is this hard?**
- MIS is NP-hard (no known polynomial algorithm)
- Many local optima (greedy can get stuck)
- Requires global reasoning about graph structure

### Our Approach: Learning to Solve MIS

Instead of hand-crafted algorithms, we train a neural network to:
1. Take a graph as input
2. Output probability for each node: "Should this node be in the MIS?"
3. Learn from optimal solutions computed by Gurobi solver

---

## Model Architecture

### GraphTRM Overview

```
Input Graph (nodes, edges)
         ↓
┌─────────────────────────┐
│    Node Embedding       │  x_embed: [n, input_dim] → [n, hidden_dim]
│    + LayerNorm          │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│   Recursive Reasoning   │  ← Runs for `cycles` iterations (default: 18)
│   (GNN + Hidden State)  │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│    Prediction Head      │  head: [n, hidden_dim] → [n, 1] (logits)
│    + Sigmoid            │
└─────────────────────────┘
         ↓
Output: Probability per node [0, 1]
```

### Component Details

#### 1. Node Embedding (`x_embed`)

```python
# Input features: [1, normalized_degree]
x = [[1.0, 0.12], [1.0, 0.08], ...]  # shape: [num_nodes, 2]

# Embed to hidden dimension
x_emb = LayerNorm(Linear(x))  # shape: [num_nodes, 256]
```

**Why these features?**
- `1.0` constant: Allows model to learn bias
- Degree: High-degree nodes are usually NOT in MIS (too many neighbors)

#### 2. Recursive Core (GNN Layers)

Each recursion step:

```python
# 1. Concatenate inputs
h_concat = [x_emb, h_prev, sigmoid(y_prev)]  # [n, 256 + 256 + 1]

# 2. Project to hidden dim
h_in = LayerNorm(Linear(h_concat))  # [n, 256]

# 3. GNN message passing (GIN layers)
for layer in gnn_layers:
    h_out = GINConv(h_in, edge_index)  # Aggregate neighbor info
    h_out = GELU(h_out)
    h_in = LayerNorm(h_in + h_out)  # Residual connection

# 4. Predict
logits = head(h_in)  # [n, 1]
```

**Key Design Choices:**

| Choice | Reason |
|--------|--------|
| **GIN (Graph Isomorphism Network)** | Most expressive standard GNN, can distinguish non-isomorphic graphs |
| **Residual connections** | Prevents gradient vanishing in deep models |
| **LayerNorm** | Stabilizes training, prevents activation explosion |
| **GELU activation** | Smoother than ReLU, better gradients |
| **Detached sigmoid on y_prev** | Prevents gradient explosion through recurrence |

#### 3. Why Recursive Reasoning?

MIS requires **global reasoning** - a node's decision depends on distant nodes' decisions.

**Without recursion:** Single GNN pass only sees local neighborhood
```
Node A's decision depends on: A's 1-hop neighbors
```

**With 18 cycles:** Information propagates across entire graph
```
Cycle 1: A sees 1-hop neighbors
Cycle 2: A sees 2-hop neighbors (via hidden state)
...
Cycle 18: A sees entire graph structure
```

**Mathematical intuition:** After k cycles, each node "sees" information from k-hop neighborhood, like computing a fixed point.

---

## Training Loop

### Overview

```python
for epoch in range(100):
    dataset.set_epoch(epoch)  # New shuffle each epoch

    for batch in dataloader:
        # 1. Initialize carry (hidden state, predictions)
        carry = model.initial_carry(batch)

        # 2. Recursive forward pass
        for cycle in range(18):
            carry, loss, metrics, preds, done = model(carry, batch)

        # 3. Use FINAL loss only (not accumulated)
        optimizer.zero_grad()
        loss.backward()

        # 4. Gradient clipping
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)

        # 5. Update
        optimizer.step()
```

### Key Training Decisions

#### Only Use Final Loss

**Wrong approach (we fixed this):**
```python
total_loss = 0
for cycle in range(18):
    carry, step_loss, ... = model(carry, batch)
    total_loss += step_loss  # ❌ Accumulates 18x!

total_loss.backward()  # Gradient explosion!
```

**Correct approach:**
```python
for cycle in range(18):
    carry, loss, ... = model(carry, batch)  # Only keep last

loss.backward()  # ✅ Single loss, stable gradients
```

#### Learning Rate Schedule

```python
# Cosine schedule with warmup
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)  # Linear warmup
else:
    # Cosine decay from base_lr to min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

| Phase | Steps | LR |
|-------|-------|-----|
| Warmup | 0-200 | 0 → 1e-3 |
| Cosine | 200-3900 | 1e-3 → 1e-4 |

#### Per-Epoch Shuffling

```python
# In MISDataset.__iter__():
rng = RandomState(seed + rank + epoch * 1000)  # Different each epoch!
rng.shuffle(my_shards)
rng.shuffle(data_list)
```

**Why this matters:**
- Same shuffle every epoch → model memorizes order
- Different shuffle → better generalization

---

## Loss Functions

### Total Loss

```python
loss = loss_bce + 1.0 * loss_feasibility + 0.3 * loss_sparsity
```

### 1. Binary Cross-Entropy Loss

```python
# With class imbalance handling
pos_weight = num_negatives / num_positives  # ~3-5 typically

loss_bce = BCEWithLogits(logits, labels, pos_weight=pos_weight)
```

**Why pos_weight?**
- MIS nodes are minority (~20-30% of nodes)
- Without weighting, model learns to predict all 0s
- pos_weight upweights positive samples

**Global vs Per-Batch pos_weight:**
- **Per-batch:** Varies significantly, causes training instability
- **Global (our approach):** Computed once from dataset, stable training

### 2. Feasibility Loss

```python
# Penalize selecting adjacent nodes
probs = sigmoid(logits)
edge_violations = probs[src_nodes] * probs[dst_nodes]  # Product of endpoints
loss_feasibility = mean(edge_violations)
```

**Interpretation:**
- If edge (u, v) exists and both prob[u] and prob[v] are high → penalty
- Drives model to learn: "Don't select neighbors"

**Weight:** 1.0 (increase to 2.0-3.0 if violations are high)

### 3. Sparsity Loss

```python
# Match predicted set size to true set size
true_sparsity = mean(labels)  # Fraction of nodes in MIS
pred_sparsity = mean(sigmoid(logits))  # Predicted fraction

loss_sparsity = (pred_sparsity - true_sparsity)²
```

**Why needed?**
- Without this, model might select 80% of nodes (greedy)
- Or select 5% of nodes (too conservative)
- Sparsity loss encourages matching the right size

**Weight:** 0.3 (smaller than feasibility, just a regularizer)

---

## Dataset and Batching

### Graph Generation

```python
# Erdos-Renyi random graphs
n = random(50, 250)  # Number of nodes
d = random(6, 14)    # Expected degree
p = d / (n - 1)      # Edge probability

G = nx.gnp_random_graph(n, p)
```

**Why these parameters?**
- n=50-250: Reasonable graph sizes for training
- d=6-14: Medium density (sparse graphs are easier, dense are harder)

### Optimal Labels (Gurobi)

```python
from gurobi_optimods.mwis import maximum_weighted_independent_set

y, opt_value = maximum_weighted_independent_set(G, weights=ones)
```

**Important:** Labels are **optimal** solutions from Gurobi solver, not approximations.

### Batching with PyG

```python
# Multiple graphs batched together
batch = Batch.from_data_list([graph1, graph2, ...])

# batch.x: [total_nodes, features]
# batch.edge_index: [2, total_edges] (with offset)
# batch.batch: [total_nodes] - which graph each node belongs to
```

**Why PyG batching?**
- Efficiently handles variable-size graphs
- Automatically offsets edge indices
- Single forward pass for all graphs in batch

### Epoch-Based Shuffling

```python
# Different random order each epoch
rng = RandomState(seed + rank + epoch * 1000)

# Shuffle shards
rng.shuffle(my_shards)

# Shuffle within each shard
for shard in my_shards:
    data = load(shard)
    rng.shuffle(data)
```

**Benefits:**
- Prevents model from memorizing training order
- Better gradient diversity each epoch
- Improves generalization

### Drop Last Batch

```python
# Don't yield partial batches during training
if len(buffer) == batch_size:
    yield batch
# else: drop partial batch
```

**Why?**
- Partial batch (e.g., 37 samples instead of 256) has different gradient variance
- Can destabilize training
- Evaluation uses `drop_last=False` to test all samples

---

## Hyperparameters

### Current Configuration

```python
# Model
hidden_dim = 256
num_layers = 2  # GNN layers per cycle
cycles = 18     # Recursive steps

# Optimizer
lr = 1e-3
weight_decay = 0.1
betas = (0.9, 0.95)
grad_clip = 1.0

# Schedule
lr_warmup_steps = 200
lr_min_ratio = 0.1

# Loss weights
feasibility_weight = 1.0
sparsity_weight = 0.3

# Training
batch_size = 256
epochs = 100
```

### Hyperparameter Tuning Guide

| Problem | Try |
|---------|-----|
| Loss not decreasing | Increase LR to 3e-3, check data |
| Loss exploding | Decrease LR to 3e-4, increase grad_clip |
| Poor feasibility | Increase feasibility_weight to 2.0-3.0 |
| Too greedy | Increase sparsity_weight to 0.5 |
| Too conservative | Decrease feasibility_weight to 0.5 |
| Overfitting | Increase weight_decay to 0.2, reduce epochs |
| Underfitting | Increase hidden_dim to 512, more cycles |

---

## Implementation Details

### Gradient Stability

Several techniques prevent training collapse:

1. **Logit clamping:** `clamp(logits, -10, 10)` prevents extreme values
2. **Detached sigmoid:** `sigmoid(y_prev).detach()` breaks gradient chain
3. **LayerNorm everywhere:** Normalizes activations
4. **Residual connections:** Helps gradient flow
5. **Xavier init with small gain:** `gain=0.1` for small initial weights

### Memory Efficiency

```python
# Reuse dataloader iterator each epoch
dataset.set_epoch(epoch)
# Instead of creating new DataLoader

# Use in-place operations where possible
h_in = norm(h_in + h_out)  # Reuses h_in
```

### Distributed Training

```python
if "LOCAL_RANK" in os.environ:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Shard data across GPUs
    my_shards = shards[rank::world_size]

    # Wrap model
    model = DistributedDataParallel(model)
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| `loss = nan` | Gradient explosion | Reduce LR, increase grad_clip |
| `grad_norm = 0.01` | Vanishing gradients | Increase LR, check for dead ReLU |
| `feasibility = 0.5` | Not learning constraints | Increase feasibility_weight |
| `approx_ratio = 1.5` | Too greedy | Increase sparsity_weight |
| `approx_ratio = 0.5` | Too conservative | Decrease feasibility_weight |
| `loss stuck > 1.0` | Not converging | Check data, try different LR |
| Training too slow | Small batch size | Increase batch_size to 256-512 |

### Debugging Commands

```bash
# Check a single batch
python -c "
from dataset.mis_dataset import MISDataset, MISDatasetConfig
ds = MISDataset(MISDatasetConfig(['data/mis-10k'], 8))
for _, batch, _ in ds:
    print('x:', batch['x'].shape)
    print('y:', batch['y'].shape, batch['y'].sum().item(), '/', batch['y'].numel())
    break
"

# Test model forward pass
python -c "
from models.graph_trm import GraphTRM
import torch
model = GraphTRM({'input_dim': 2, 'hidden_dim': 64, 'cycles': 3})
x = torch.randn(10, 2)
edge_index = torch.randint(0, 10, (2, 20))
batch = {'x': x, 'edge_index': edge_index, 'y': torch.randint(0, 2, (10,)).float()}
carry = model.initial_carry(batch)
carry, loss, metrics, preds, done = model(carry, batch)
print('loss:', loss.item())
print('preds shape:', preds['preds'].shape)
"
```

---

## Summary

The MIS training pipeline:

1. **Data:** Erdos-Renyi graphs with optimal MIS labels from Gurobi
2. **Model:** GraphTRM with recursive GNN reasoning (18 cycles)
3. **Loss:** BCE + Feasibility + Sparsity constraints
4. **Training:** AdamW with cosine LR, gradient clipping, per-epoch shuffling
5. **Output:** Per-node probabilities, converted to MIS via greedy decode

Key innovations:
- **Recursive reasoning** for global graph understanding
- **Multi-component loss** for learning constraints
- **Honest metrics** for accurate performance tracking
- **Stable training** via normalization and gradient techniques
