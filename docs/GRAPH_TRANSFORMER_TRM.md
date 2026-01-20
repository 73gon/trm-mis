# GraphTransformerTRM - Complete Technical Documentation

This document provides an exhaustive explanation of the `graph_transformer_trm.py` model architecture.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Breakdown](#3-component-breakdown)
4. [TRM Recursion Explained](#4-trm-recursion-explained)
5. [Loss Functions](#5-loss-functions)
6. [Comparison with Paper](#6-comparison-with-paper)
7. [Configuration Reference](#7-configuration-reference)

---

## 1. Overview

**GraphTransformerTRM** is a neural network for solving the Maximum Independent Set (MIS) problem on graphs. It combines:

- **GPS Layers** (General, Powerful, Scalable): Hybrid local-global message passing
- **TRM Framework** (Tiny recursive model): Iterative refinement through recursion
- **Deep Supervision**: Gradient flow through multiple refinement steps

### What is MIS?

Given a graph G = (V, E), find the largest subset S ⊆ V such that no two nodes in S are adjacent.

**Input:** Graph with node features and edges
**Output:** Probability for each node being in the MIS (0 to 1)

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GRAPH TRANSFORMER TRM                            │
└─────────────────────────────────────────────────────────────────────────┘

INPUT:
  x: [N, 8]           - Node features (8 dimensions after enhancement)
  pe: [N, 16]         - Laplacian Positional Encoding
  edge_index: [2, E]  - Graph connectivity
  batch_vec: [N]      - Batch assignment (for batched graphs)

┌─────────────────────────────────────────────────────────────────────────┐
│                         FEATURE EMBEDDING                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   x ─────┬───► x_embed ───► GELU ───► x_norm ───► x_emb [N, 240]       │
│          │     (8 → 240)                                                │
│          │                                                              │
│   pe ────┴───► pe_embed ──► GELU ───► pe_norm ──► pe_emb [N, 16]       │
│                (16 → 16)                                                │
│                                                                         │
│   x_emb = concat[x_emb, pe_emb] ───────────────────► [N, 256]          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   y_init = 1 / (1 + degree)  → logits                                  │
│   z_init = zeros([N, 256])                                             │
│                                                                         │
│   Low-degree nodes start with higher probability (MIS heuristic)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRM DEEP RECURSION (H_cycles)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   for h in range(H_cycles):  # Default: 2 horizontal cycles             │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │              LATENT RECURSION (L_cycles)                        │  │
│   ├─────────────────────────────────────────────────────────────────┤  │
│   │                                                                 │  │
│   │   for l in range(L_cycles):  # Default: 6 vertical cycles       │  │
│   │                                                                 │  │
│   │     input = concat[x_emb, sigmoid(y), z]    # [N, 256+1+256]   │  │
│   │              │                                                  │  │
│   │              ▼                                                  │  │
│   │     h = latent_proj(input)                  # [N, 513] → [N, 256]
│   │     h = LayerNorm(GELU(h))                                      │  │
│   │              │                                                  │  │
│   │              ▼                                                  │  │
│   │     ┌───────────────────────────────────────────────────────┐  │  │
│   │     │             GPS LAYER (x num_layers)                  │  │  │
│   │     ├───────────────────────────────────────────────────────┤  │  │
│   │     │                                                       │  │  │
│   │     │  ┌─────────────┐        ┌───────────────────────┐    │  │  │
│   │     │  │  GINConv    │        │  MultiHead Attention  │    │  │  │
│   │     │  │  (Local)    │   +    │  (Global)             │    │  │  │
│   │     │  │             │        │  heads=4              │    │  │  │
│   │     │  └─────────────┘        └───────────────────────┘    │  │  │
│   │     │         │                        │                    │  │  │
│   │     │         └────────┬───────────────┘                   │  │  │
│   │     │                  ▼                                    │  │  │
│   │     │          LayerNorm + Residual                        │  │  │
│   │     │                  │                                    │  │  │
│   │     │                  ▼                                    │  │  │
│   │     │              z_new [N, 256]                           │  │  │
│   │     └───────────────────────────────────────────────────────┘  │  │
│   │                                                                 │  │
│   │   end for (L_cycles)                                           │  │
│   │                                                                 │  │
│   │   ┌─────────────────────────────────────────────────────────┐  │  │
│   │   │                OUTPUT STEP                              │  │  │
│   │   ├─────────────────────────────────────────────────────────┤  │  │
│   │   │   input = concat[y, z]              # [N, 1+256]        │  │  │
│   │   │   h = output_proj(input)            # [N, 257] → [N, 256]
│   │   │   h = LayerNorm(GELU(h))                                │  │  │
│   │   │   y_new = output_head(h)            # [N, 256] → [N, 1] │  │  │
│   │   └─────────────────────────────────────────────────────────┘  │  │
│   │                                                                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   end for (H_cycles)                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   logits = y_new.squeeze()          # [N]                              │
│   probs = sigmoid(logits)           # [N], values in [0, 1]            │
│                                                                         │
│   LOSS = BCE(logits, labels) + λ * FeasibilityLoss(probs, edges)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Feature Embedding (`embed_features`)

**Purpose:** Convert raw node features + positional encodings into a rich embedding.

```python
def embed_features(self, batch):
    x = batch["x"]  # [N, 8] - node features

    # Embed base features: Linear(8 → 240) + GELU + LayerNorm
    x_emb = self.x_norm(F.gelu(self.x_embed(x)))  # [N, 240]

    # Embed positional encoding: Linear(16 → 16) + GELU + LayerNorm
    pe = batch["pe"]  # [N, 16]
    pe_emb = self.pe_norm(F.gelu(self.pe_embed(pe)))  # [N, 16]

    # Concatenate: [N, 240] + [N, 16] = [N, 256]
    x_emb = torch.cat([x_emb, pe_emb], dim=-1)

    return x_emb
```

**Why split into 240 + 16?**
- Total hidden_dim = 256
- PE dimension = 16 (captures global structure)
- Node features get remaining 240 dimensions
- This ensures PE information isn't overwhelmed

### 3.2 Initialization (`initial_carry`, `compute_y_init`)

**Purpose:** Create initial state for TRM recursion.

```python
def compute_y_init(self, batch):
    """Degree-based heuristic: low-degree nodes are likely in MIS"""
    deg = degree(edge_index[0], num_nodes)  # Node degrees
    y_prob_init = 1.0 / (1.0 + deg)         # High prob for low degree
    y_logits_init = logit(y_prob_init)      # Convert to logits
    return y_logits_init

def initial_carry(self, batch):
    z = torch.zeros(N, hidden_dim)  # Start with zero hidden state
    y = self.compute_y_init(batch)  # Degree-based initialization
    return (y, z, step_count=0)
```

**Why degree-based init?**
- Low-degree nodes have fewer neighbors to conflict with
- This is a well-known MIS heuristic
- Gives the model a strong starting point

### 3.3 Latent Step (`latent_step`)

**Purpose:** One iteration of message passing to update hidden state.

```python
def latent_step(self, x_emb, y_logits, z, edge_index, batch_vec):
    y_prob = sigmoid(y_logits)  # Convert logits to probabilities

    # Concatenate all information
    h_in = concat([x_emb, y_prob, z])  # [N, 256+1+256] = [N, 513]

    # Project down to hidden dimension
    h = self.latent_proj(h_in)         # [N, 513] → [N, 256]
    h = LayerNorm(GELU(h))

    # Apply GPS layers (local + global attention)
    for gps_layer in self.gps_layers:
        h = gps_layer(h, edge_index, batch=batch_vec)

    return h  # New z
```

**What's inside GPS Layer?**
```
GPSConv = GINConv (Local MPNN) + MultiheadAttention (Global)
        + LayerNorm + Residual connections
```

- **GINConv (Local):** Aggregates neighbor information using MLP
- **MultiheadAttention (Global):** All nodes can attend to all nodes in same graph
- **Combined:** Captures both local structure and global patterns

### 3.4 Output Step (`output_step`)

**Purpose:** Refine the probability prediction using current hidden state.

```python
def output_step(self, y_logits, z):
    h_in = concat([y_logits, z])       # [N, 1+256] = [N, 257]
    h = self.output_proj(h_in)         # [N, 257] → [N, 256]
    h = LayerNorm(GELU(h))
    y_new = self.output_head(h)        # [N, 256] → [N, 1]
    return y_new
```

The output head is:
```
Linear(256 → 128) → GELU → Linear(128 → 1)
```

### 3.5 Latent Recursion (`latent_recursion`)

**Purpose:** Multiple iterations of latent steps followed by output refinement.

```python
def latent_recursion(self, x_emb, y, z, edge_index, batch_vec):
    # L_cycles iterations of message passing (default: 6)
    for _ in range(self.L_cycles):
        z = self.latent_step(x_emb, y, z, edge_index, batch_vec)

    # Then refine output
    y = self.output_step(y, z)
    return y, z
```

**L_cycles = 6** means 6 rounds of GPS message passing before updating y.

### 3.6 Deep Recursion (`deep_recursion`)

**Purpose:** Outer loop for H_cycles repetitions of latent recursion.

```python
def deep_recursion(self, x_emb, y, z, edge_index, batch_vec):
    # H_cycles = 2 repetitions (all with gradients in our implementation)
    for _ in range(self.H_cycles):
        y, z = self.latent_recursion(x_emb, y, z, edge_index, batch_vec)
    return y, z
```

**Total iterations:** H_cycles × L_cycles = 2 × 6 = **12 GPS layer applications**

---

## 4. TRM Recursion Explained

### What is TRM?

TRM (Tiny Recursive Model) is a framework for iterative refinement:

1. **Start with initial guess** (y_init from degree heuristic)
2. **Iteratively refine** through multiple "thinking" cycles
3. **Each cycle** propagates information and updates predictions
4. **Stop when confident** or after max iterations

### Our Implementation vs. Paper TRM

| Aspect | Paper TRM | Our Implementation |
|--------|-----------|-------------------|
| **H_cycles (Outer)** | No-grad for H-1, grad for 1 | All with gradients |
| **L_cycles (Inner)** | Multiple latent updates | 6 GPS applications |
| **Early Stopping** | q_hat threshold | Implemented but not used in training |
| **Supervision** | Deep supervision at each H | Single supervision per forward |

### Why All Gradients?

The original TRM uses no-grad for H-1 outer cycles to:
- Save memory
- Prevent gradient explosion

We keep all gradients because:
- Overfitting test needs full gradient flow
- Smaller model (256 hidden) can handle it
- Better debugging visibility

---

## 5. Loss Functions

### 5.1 Binary Cross-Entropy (BCE) Loss

```python
bce_loss = F.binary_cross_entropy_with_logits(
    logits, labels,
    pos_weight=pos_weight  # Handle class imbalance
)
```

**pos_weight** = neg_count / pos_count ≈ 3-4 for typical MIS

This increases the penalty for false negatives (missing MIS nodes).

### 5.2 Feasibility Loss

```python
probs = sigmoid(logits)
src, dst = edge_index[0], edge_index[1]
edge_violations = probs[src] * probs[dst]
feasibility_loss = edge_violations.mean()
```

**Purpose:** Penalize selecting both endpoints of an edge.

If both nodes have high probability, their product is high → loss increases.

### 5.3 Total Loss

```python
loss = bce_loss + feasibility_weight * feasibility_loss
```

Default `feasibility_weight = 50.0`, but set to 0 during overfitting tests.

---

## 6. Comparison with Paper

### 6.1 Architecture Comparison

| Component | Paper (TRM) | Our Implementation |
|-----------|-------------|-------------------|
| **Base Model** | Custom MPNN | GPSConv (GIN + Attention) |
| **Hidden Dim** | 512 | 256 |
| **Num Layers** | 4-6 | 2 (inside GPS) |
| **y_init** | Uniform or degree | Degree-based: 1/(1+deg) |
| **z_init** | Zeros | Zeros |
| **H_cycles** | 3-5 | 2 |
| **L_cycles** | 6-10 | 6 |
| **Positional Encoding** | Random Walk PE | Laplacian PE |
| **Attention** | None in base | MultiheadAttention in GPS |

### 6.2 Similarities

1. **Iterative Refinement:** Both use nested loops (H × L)
2. **Degree Heuristic:** Both can use degree-based initialization
3. **Zero z_init:** Both start with zero hidden state
4. **BCE + Feasibility Loss:** Same loss formulation

### 6.3 Key Differences

1. **GPS vs Pure MPNN:**
   - Paper uses pure GNN without attention
   - We add global attention for long-range dependencies

2. **Gradient Flow:**
   - Paper: no-grad for H-1 steps (memory efficient)
   - Ours: all gradients (full flow for debugging)

3. **Positional Encoding:**
   - Paper: Random Walk PE (local k-hop info)
   - Ours: Laplacian PE (global spectral position)

4. **Node Features:**
   - Paper: Typically just degree
   - Ours: 8 features including clustering, k-core, neighbor stats

### 6.4 Why These Differences?

| Difference | Rationale |
|------------|-----------|
| GPS over MPNN | Better expressivity for graph-level reasoning |
| Smaller hidden_dim | GPU memory constraints, faster iteration |
| Laplacian PE | Global structure matters for MIS |
| Rich node features | More information for decision-making |

---

## 7. Configuration Reference

```python
config = {
    # Dimensions
    "input_dim": 8,           # Node feature dimension
    "pe_dim": 16,             # Positional encoding dimension
    "hidden_dim": 256,        # Hidden state dimension

    # GPS Layers
    "num_layers": 2,          # GPS layers per latent step
    "num_heads": 4,           # Attention heads
    "dropout": 0.1,           # Dropout rate

    # TRM Recursion
    "L_cycles": 6,            # Inner loop iterations
    "H_cycles": 2,            # Outer loop iterations

    # Initialization
    "use_degree_init": True,  # Use degree-based y_init

    # Loss
    "pos_weight": None,       # Auto-compute from data
    "feasibility_weight": 50.0,

    # Early stopping
    "early_stop_threshold": 0.9,
}
```

---

## Summary

GraphTransformerTRM is a powerful architecture for MIS that:

1. **Embeds** rich node features + positional encodings
2. **Initializes** with degree-based heuristic
3. **Iterates** through H×L = 12 GPS applications
4. **Predicts** node probabilities with BCE + feasibility loss

The model combines local message passing (GINConv) with global attention, allowing it to reason about both neighborhood structure and graph-wide patterns.
