# MIS Dataset (mis_dataset.py) - Complete Documentation

This document provides a detailed explanation of the MIS dataset implementation.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Format](#2-data-format)
3. [Feature Engineering](#3-feature-engineering)
4. [Dataset Configuration](#4-dataset-configuration)
5. [Data Loading and Splitting](#5-data-loading-and-splitting)
6. [Batching and Collation](#6-batching-and-collation)
7. [Iteration Modes](#7-iteration-modes)

---

## 1. Overview

`mis_dataset.py` provides:

1. **Laplacian Positional Encoding** computation
2. **Enhanced Node Features** (clustering, k-core, neighbor stats)
3. **Graph-level train/val splitting**
4. **Efficient batching** with PyG

### Key Classes and Functions

```python
compute_laplacian_pe(edge_index, num_nodes, pe_dim)  # Positional encoding
compute_node_features(edge_index, num_nodes, x)       # Feature enhancement
MISDatasetConfig                                       # Configuration
MISDataset                                            # Main dataset class
```

---

## 2. Data Format

### 2.1 Shard Files

Data is stored in `.pt` (PyTorch) shard files:

```
data/mis-10k/
├── mis_shard_0.pt
├── mis_shard_1.pt
├── mis_shard_2.pt
└── ...
```

Each shard contains:
```python
payload = torch.load("mis_shard_0.pt")
# payload = {
#     "data": [graph1, graph2, ...],  # List of graph dicts
# }
```

### 2.2 Graph Dictionary

Each graph is a dictionary:
```python
graph = {
    "x": tensor[N, 2],           # Node features [1, degree_norm]
    "edge_index": tensor[2, E],  # Edge list (undirected, both directions)
    "y": tensor[N],              # Labels (1 = in MIS, 0 = not)
    "n": int,                    # Number of nodes
    "opt_value": float,          # Optimal MIS size
}
```

### 2.3 Understanding Edge Index

```python
# For a triangle graph A-B-C:
edge_index = tensor([
    [0, 1, 1, 2, 0, 2],  # Source nodes
    [1, 0, 2, 1, 2, 0],  # Target nodes
])
# Each edge appears twice (undirected)
# Edge (0,1) → row [0,1] and [1,0]
```

---

## 3. Feature Engineering

### 3.1 Laplacian Positional Encoding

```python
def compute_laplacian_pe(edge_index, num_nodes, pe_dim=16):
    """
    Compute eigenvectors of normalized Laplacian matrix.

    Steps:
    1. Build adjacency matrix A from edge_index
    2. Compute degree matrix D
    3. Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    4. Compute smallest pe_dim+1 eigenvectors
    5. Skip first (trivial) eigenvector
    6. Return [N, pe_dim] tensor
    """
```

**What Laplacian PE captures:**

| Eigenvector | Information |
|-------------|-------------|
| 1st (skip) | Trivial, constant |
| 2nd | Graph connectivity (Fiedler vector) |
| 3rd-5th | Cluster structure |
| Higher | Local oscillations |

**Why it helps MIS:**
- Nodes in same cluster are close in PE space
- Can only pick one node from dense cluster
- PE helps identify cluster boundaries

### 3.2 Node Features

```python
def compute_node_features(edge_index, num_nodes, x_original):
    """
    Compute enhanced features using NetworkX.

    Returns [N, 8] tensor with:
    - Original: [1, degree_norm]
    - Enhanced: [raw_deg_norm, clustering, avg_neigh, max_neigh, min_neigh, core]
    """
```

#### Feature Breakdown

| Index | Feature | Range | Description | Why it helps MIS |
|-------|---------|-------|-------------|------------------|
| 0 | `1` | 1 | Bias term | Learnable offset |
| 1 | `degree_norm` | [0,1] | degree / (n-1) | Low degree = good MIS candidate |
| 2 | `raw_deg_norm` | [0,1] | degree / max_degree | Relative degree within graph |
| 3 | `clustering` | [0,1] | Clustering coefficient | High = dense neighborhood = only 1 in MIS |
| 4 | `avg_neighbor_deg` | [0,1] | Mean neighbor degree | Low-deg node with high-deg neighbors = good |
| 5 | `max_neighbor_deg` | [0,1] | Max neighbor degree | Identifies hub connections |
| 6 | `min_neighbor_deg` | [0,1] | Min neighbor degree | Sparse neighborhood indicator |
| 7 | `core_number` | [0,1] | k-core number | High = deeply embedded = bad MIS candidate |

#### Feature Computation Details

**Clustering Coefficient:**
```
C(v) = 2 × triangles(v) / (deg(v) × (deg(v) - 1))
```
Measures how connected a node's neighbors are to each other.

**k-Core Number:**
```
Largest k such that v is in the k-core
(subgraph where all nodes have degree ≥ k)
```

Example:
```
    A -- B -- C
    |    |
    D -- E

k-cores:
- 1-core: All nodes (all have degree ≥ 1)
- 2-core: {A, B, D, E} (all have degree ≥ 2)
- 3-core: {} (no node has degree ≥ 3)

Core numbers: A=2, B=2, C=1, D=2, E=2
```

---

## 4. Dataset Configuration

```python
class MISDatasetConfig:
    def __init__(
        self,
        dataset_paths,           # List of paths to search for shards
        global_batch_size,       # Graphs per batch
        rank=0,                  # GPU rank (for distributed)
        num_replicas=1,          # Total GPUs
        seed=42,                 # Random seed
        epoch=0,                 # Current epoch (for shuffling)
        drop_last=True,          # Drop partial batches
        val_split=0.1,           # Validation fraction
        max_shards=None,         # Limit shards (for testing)
        inject_noise=False,      # Add random noise features
        pe_dim=16,               # Laplacian PE dimension
    ):
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `global_batch_size` | 128 | How many graphs per batch |
| `val_split` | 0.1 | 10% validation, 90% training |
| `drop_last` | True | Training drops partial batches |
| `max_shards` | None | Limit data for overfitting tests |
| `inject_noise` | False | Add random features for symmetry breaking |
| `pe_dim` | 16 | Dimension of positional encoding |

---

## 5. Data Loading and Splitting

### 5.1 Loading Flow

```python
# 1. Find all shards
all_shards = glob.glob("data/mis-10k/mis_shard_*.pt")

# 2. Shuffle and optionally limit
rng = np.random.RandomState(seed)
rng.shuffle(shard_indices)
if max_shards:
    shard_indices = shard_indices[:max_shards]

# 3. Load all graphs from selected shards
all_graphs = []
for shard in selected_shards:
    payload = torch.load(shard)
    all_graphs.extend(payload["data"])

# 4. Split at GRAPH level
graph_indices = np.arange(len(all_graphs))
rng.shuffle(graph_indices)

val_count = int(len(all_graphs) * val_split)
val_indices = graph_indices[:val_count]
train_indices = graph_indices[val_count:]
```

### 5.2 Why Graph-Level Splitting?

**Bad approach (shard-level):**
```
Shard 0 → Train
Shard 1 → Train
Shard 2 → Val
```
Problem: If shards have different difficulty, val is biased.

**Good approach (graph-level):**
```
All graphs shuffled together
First 10% → Val
Remaining 90% → Train
```
Ensures representative split.

### 5.3 Caching for Small Datasets

```python
if num_graphs <= 3000:
    # Cache with feature enhancement
    self.cached_data = []
    for graph in all_graphs:
        # Compute PE
        pe = compute_laplacian_pe(...)
        # Compute enhanced features
        x_enhanced = compute_node_features(...)
        # Store
        self.cached_data.append({...})
```

Benefits:
- No recomputation each epoch
- Consistent features across epochs
- Faster iteration

---

## 6. Batching and Collation

### 6.1 Collation Function

```python
def _collate_graph_batch(self, sample_list):
    # Convert to PyG Data objects
    data_list = []
    for s in sample_list:
        d = Data(
            x=s["x"],
            edge_index=s["edge_index"],
            y=s["y"],
            num_nodes=s["n"],
            pe=s.get("pe"),
            opt_value=s["opt_value"],
        )
        data_list.append(d)

    # PyG batching (handles edge index offsets)
    batch = Batch.from_data_list(data_list)

    return {
        "x": batch.x,
        "edge_index": batch.edge_index,
        "batch": batch.batch,
        "y": batch.y,
        "ptr": batch.ptr,
        "num_graphs": batch.num_graphs,
        "pe": batch.pe,
        "opt_value": batch.opt_value,
    }
```

### 6.2 Understanding PyG Batching

When batching multiple graphs, PyG:

1. **Concatenates node features:**
```
Graph 0: x = [[1,2], [3,4]]      # 2 nodes
Graph 1: x = [[5,6], [7,8], [9,0]]  # 3 nodes

Batched x = [[1,2], [3,4], [5,6], [7,8], [9,0]]  # 5 nodes
```

2. **Offsets edge indices:**
```
Graph 0: edge_index = [[0,1], [1,0]]
Graph 1: edge_index = [[0,1], [1,2], [2,0], ...]

Batched edge_index = [[0,1], [1,0], [2,3], [3,4], [4,2], ...]
                       └─Graph 0─┘  └────Graph 1 (offset by 2)────┘
```

3. **Creates batch vector:**
```
batch = [0, 0, 1, 1, 1]  # Node → Graph mapping
```

4. **Creates pointer array:**
```
ptr = [0, 2, 5]  # Graph boundaries
# Graph 0: nodes 0-1 (ptr[0] to ptr[1]-1)
# Graph 1: nodes 2-4 (ptr[1] to ptr[2]-1)
```

---

## 7. Iteration Modes

### 7.1 Cached Mode (Small Datasets)

```python
if self.cached_data is not None:
    indices = np.arange(len(self.cached_data))
    my_indices = indices[rank::world_size]  # Distribute across GPUs
    rng.shuffle(my_indices)                  # Shuffle per epoch

    buffer = []
    for i in my_indices:
        buffer.append(self.cached_data[i])
        if len(buffer) >= global_batch_size:
            yield "mis", self._collate_graph_batch(buffer), len(buffer)
            buffer = []
```

### 7.2 Streaming Mode (Large Datasets)

```python
else:
    my_graphs = self.all_graphs[rank::world_size]
    indices = list(range(len(my_graphs)))
    rng.shuffle(indices)

    buffer = []
    for idx in indices:
        sample = my_graphs[idx]

        # Compute features on-the-fly
        pe = compute_laplacian_pe(...)
        x_enhanced = compute_node_features(...)

        buffer.append({...})
        if len(buffer) >= global_batch_size:
            yield "mis", self._collate_graph_batch(buffer), len(buffer)
            buffer = []
```

### 7.3 Comparison

| Aspect | Cached Mode | Streaming Mode |
|--------|-------------|----------------|
| When used | ≤3000 graphs | >3000 graphs |
| Feature computation | Once at init | Every epoch |
| Memory | Higher (stores all) | Lower |
| Speed | Faster iteration | Slower |
| Consistency | Same features each epoch | Same (deterministic) |

---

## 8. Class Imbalance Handling

### 8.1 Computing pos_weight

```python
def _compute_class_imbalance(self):
    total_pos = 0
    total_neg = 0

    for sample in graphs_to_sample:
        y = sample["y"]
        pos = (y == 1).sum().item()
        neg = (y == 0).sum().item()
        total_pos += pos
        total_neg += neg

    # pos_weight = neg / pos (typically 3-4 for MIS)
    self.metadata.pos_weight = total_neg / max(1.0, total_pos)
    self.metadata.class_ratio = total_pos / (total_pos + total_neg)
```

### 8.2 Why pos_weight Matters

In MIS, typically ~25% of nodes are in the solution:
- 100 nodes → ~25 in MIS (pos), ~75 not (neg)
- pos_weight = 75/25 = 3

Without weighting:
- Model can achieve 75% accuracy by predicting all zeros
- Misses all MIS nodes

With pos_weight=3:
- False negatives penalized 3× more
- Model forced to learn MIS nodes

---

## Summary

The MIS dataset provides:

1. **Rich Features:**
   - Laplacian PE (global structure)
   - 8 node features (degree, clustering, k-core, etc.)

2. **Proper Splitting:**
   - Graph-level (not shard-level)
   - Deterministic based on seed

3. **Efficient Batching:**
   - PyG-compatible
   - Handles variable-size graphs

4. **Class Imbalance:**
   - Auto-computed pos_weight
   - Passed to model for BCE loss
