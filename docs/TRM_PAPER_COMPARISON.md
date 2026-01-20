# Our MIS-TRM Implementation

**IMPORTANT:** This document describes OUR implementation for the Maximum Independent Set problem. We do NOT claim to replicate any specific paper. The TRM paper solves different tasks (Sudoku, Maze, ARC-AGI) which are NOT graph problems.

---

## What We Built

We built a graph neural network for MIS using:

1. **GPS Backbone** (GIN + Attention)
2. **Iterative Refinement** (H × L cycles)
3. **Enhanced Node Features** (8 features)
4. **Laplacian Positional Encoding** (16 dimensions)

---

## Our Architecture

### Model: GraphTransformerTRM

```
Input: Graph (nodes, edges)
  │
  ▼
Feature Embedding (8 features → 240 dim)
  +
PE Embedding (16 dim → 16 dim)
  │
  ▼
GPS Layers (GIN + MultiheadAttention)
  │
  ▼
Output Head → Node probabilities
```

### GPS Layer

Each GPS layer combines:
- **GINConv**: Local message passing (aggregates from neighbors)
- **MultiheadAttention**: Global attention (all nodes see all nodes)
- **LayerNorm + Residual**: Stability

### Iterative Refinement

```python
for h in range(H_cycles):      # Outer loop (default: 2)
    for l in range(L_cycles):  # Inner loop (default: 6)
        z = GPS(x, y, z, edges)
    y = OutputHead(y, z)
```

Total: 2 × 6 = 12 refinement steps

---

## Our Design Choices

| Component | Our Choice | Why |
|-----------|------------|-----|
| Backbone | GPS (GIN + Attention) | Both local and global reasoning |
| y_init | 1 / (1 + degree) | MIS heuristic: low-degree nodes are good |
| z_init | zeros | Clean starting point |
| Loss | BCE + Feasibility | Classification + constraint satisfaction |
| Features | 8 enhanced | More information for decision |
| PE | Laplacian (16 dim) | Global graph structure |

---

## Our Features

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | 1 | Bias term |
| 1 | degree_norm | degree / (n-1) |
| 2 | raw_deg_norm | degree / max_degree |
| 3 | clustering | Clustering coefficient |
| 4 | avg_neighbor_deg | Mean neighbor degree |
| 5 | max_neighbor_deg | Max neighbor degree |
| 6 | min_neighbor_deg | Min neighbor degree |
| 7 | core_number | k-core number |

---

## Our Loss Function

```python
loss = BCE(logits, labels, pos_weight) + λ * feasibility_loss
```

- **BCE**: Binary cross-entropy for node classification
- **pos_weight**: Handles class imbalance (MIS nodes are minority)
- **Feasibility**: Penalizes selecting adjacent nodes

```python
# Feasibility loss
probs = sigmoid(logits)
feasibility_loss = mean(probs[src] * probs[dst])  # For all edges
```

---

## Summary

This is our custom implementation for MIS on graphs. We use iterative refinement and GPS layers because they work well for this problem. We make no claims about matching any specific paper.
