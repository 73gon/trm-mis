# How Graphs Flow Through the Neural Network

## 1. GRAPH STORAGE (Shards on Disk)

```
data/mis-10k/
  ├── mis_shard_0.pt  ← Contains ~100 graphs
  ├── mis_shard_1.pt  ← Contains ~100 graphs
  └── ...
```

Each shard is a dict with `payload["data"]`:
```python
{
  "data": [
    {
      "x": torch.Tensor([N, 2]),           # Node features: [1, normalized_degree]
      "edge_index": torch.Tensor([2, E]),  # Edges: [[u1,u2,...], [v1,v2,...]]
      "y": torch.Tensor([N]),              # Labels: 0/1 for each node (MIS membership)
      "n": int,                            # Number of nodes
      "opt_value": int,                    # Optimal MIS size (ground truth)
    },
    {...},  # Next graph
    ...
  ]
}
```

---

## 2. DATASET LOADING & PREPROCESSING

### Step 2a: Load All Shards into Memory
```python
# mis_dataset.py, __init__()
all_shards = ["mis_shard_0.pt", "mis_shard_1.pt", ...]
for shard_path in selected_shards:
    payload = torch.load(shard_path)
    all_graphs.extend(payload["data"])  # Now have ~10k dicts
```

### Step 2b: Train/Val Split at GRAPH Level
```python
# Split 10k graphs: 90% train, 10% val
val_count = int(10000 * 0.1) = 1000
train_graphs = all_graphs[1000:]  # 9000 graphs
val_graphs = all_graphs[:1000]    # 1000 graphs
```

### Step 2c: Compute Features & PE (Per-Graph)
**For each graph in cache:**
```python
# mis_dataset.py, __init__() caching loop
for sample in all_graphs:
    sample_copy = sample.copy()

    # 1. Compute Laplacian Positional Encoding (16-dim)
    pe = compute_laplacian_pe(edge_index, n, pe_dim=16)
    sample_copy["pe"] = pe

    # 2. Compute enhanced node features (6 more features)
    x_enhanced = compute_node_features(edge_index, n, x)
    # x_enhanced shape: [N, 8]
    # Combines: [original_2, raw_degree_norm, clustering, avg_neighbor_deg,
    #            max_neighbor_deg, min_neighbor_deg, core_number]
    sample_copy["x"] = x_enhanced

    cached_data.append(sample_copy)
```

**Updated sample structure:**
```python
{
  "x": torch.Tensor([N, 8]),           # ← NOW 8 features (was 2)
  "pe": torch.Tensor([N, 16]),         # ← NEW: Laplacian PE
  "edge_index": torch.Tensor([2, E]),  # ← Unchanged
  "y": torch.Tensor([N]),              # ← Unchanged
  "n": int,
  "opt_value": int,
}
```

---

## 3. BATCHING (Multiple Graphs → One Batch)

### Step 3a: Buffer Accumulation (mis_dataset.py, __iter__)
```python
buffer = []
for i in my_indices:
    sample = cached_data[i]
    buffer.append(sample)
    if len(buffer) >= global_batch_size (128):
        # Collate this buffer
        yield "mis", self._collate_graph_batch(buffer), 128
        buffer = []
```

**Buffer content after 128 graphs:**
```
buffer = [
  {"x": [N1, 8], "pe": [N1, 16], "edge_index": [2, E1], "y": [N1], ...},
  {"x": [N2, 8], "pe": [N2, 16], "edge_index": [2, E2], "y": [N2], ...},
  ...
  {"x": [N128, 8], "pe": [N128, 16], "edge_index": [2, E128], "y": [N128], ...},
]
```

### Step 3b: PyG Batching (mis_dataset.py, _collate_graph_batch)
```python
# Convert to PyG Data objects
data_list = []
for s in sample_list:
    d = Data(x=s["x"], edge_index=s["edge_index"], y=s["y"], num_nodes=s["n"])
    d.pe = s["pe"]
    data_list.append(d)

# PyG Batch.from_data_list() handles:
# 1. Node concatenation
# 2. Edge index offsetting
# 3. Batch vector creation
batch = Batch.from_data_list(data_list)
```

**Key PyG operations:**

| Operation | Input | Output |
|-----------|-------|--------|
| **x concatenation** | [N1,8], [N2,8], ... | [N_total, 8] where N_total = N1+N2+... |
| **edge_index offsetting** | [2, E1], [2, E2], ... | [2, E_total] with adjusted node IDs |
| **batch vector** | N1 nodes from graph 0, N2 from graph 1, ... | [0,0,...0, 1,1,...1, 2,2,...2, ...] |
| **ptr vector** | 4 graphs batched | [0, N1, N1+N2, N1+N2+N3, N1+N2+N3+N4] |

### Step 3c: Return as Dict
```python
result = {
    "x": torch.Tensor([N_total, 8]),           # All nodes stacked
    "edge_index": torch.Tensor([2, E_total]),  # All edges, adjusted indices
    "batch": torch.Tensor([N_total]),          # Graph assignment per node
    "y": torch.Tensor([N_total]),              # All labels stacked
    "ptr": torch.Tensor([129]),                # Pointers: [0, N1, N1+N2, ...]
    "pe": torch.Tensor([N_total, 16]),         # All PEs stacked
    "num_graphs": 128,
    "opt_value": torch.Tensor([128]),          # All optimal sizes
}
```

**Example with 3 graphs:**
```
Graph 0: 10 nodes, 15 edges
Graph 1: 8 nodes, 12 edges
Graph 2: 12 nodes, 18 edges

After batching:
  x: [30, 8]              (10+8+12 = 30 nodes)
  edge_index: [2, 45]     (15+12+18 = 45 edges)
  batch: [0]*10 + [1]*8 + [2]*12
  ptr: [0, 10, 18, 30]
  y: [10 labels] + [8 labels] + [12 labels]
```

---

## 4. FORWARD PASS (train_mis.py)

### Step 4a: Move Batch to GPU
```python
batch = {
    k: v.cuda() if isinstance(v, torch.Tensor) else v
    for k, v in batch.items()
}
```

### Step 4b: Initialize Carry State
```python
carry = raw_model.initial_carry(batch)
# Returns: (y_logits, z_latent, step_count)
# y_logits: [N_total, 1] - initial node probabilities (degree-based)
# z_latent: [N_total, 256] - zero hidden state
```

### Step 4c: Deep Recursion (H_cycles=2)
```python
# graph_transformer_trm.py, forward()
x_emb = self.embed_features(batch)  # [N_total, 256]
# - Embeds x: [N_total, 8] → [N_total, 240]
# - Embeds pe: [N_total, 16] → [N_total, 16]
# - Concatenates: [N_total, 256]

# H_cycle 1
for _ in range(L_cycles=6):  # Latent recursion
    z = latent_step(x_emb, y, z, edge_index, batch)
    # GPS layer applies: local GINConv + global MultiheadAttention
    # Uses batch vector to know which nodes belong to which graph
y = output_step(y, z)

# H_cycle 2
for _ in range(L_cycles=6):
    z = latent_step(x_emb, y, z, edge_index, batch)
y = output_step(y, z)
```

**Key: How GPS Knows Graph Boundaries**
```python
# In GPS layer call:
gps_layer(h, edge_index, batch=batch_vec)
#                                ↓
# batch_vec tells GPS which nodes belong to which graph
# → Attention only within-graph (no cross-graph communication)
# → GINConv respects edge topology (already encoded in edge_index)
```

### Step 4d: Loss Calculation
```python
y_new: [N_total, 1]  # Final logits

# BCE Loss
bce_loss = F.binary_cross_entropy_with_logits(
    y_new.squeeze(-1),  # [N_total]
    labels,             # [N_total]
    pos_weight=pos_weight  # scalar
)

# Feasibility Loss
probs = torch.sigmoid(y_new.squeeze(-1))  # [N_total]
edge_violations = probs[edge_index[0]] * probs[edge_index[1]]  # [E_total]
feasibility_loss = edge_violations.mean()

# Total Loss
loss = bce_loss + feasibility_weight * feasibility_loss
```

---

## 5. BACKWARD PASS & OPTIMIZATION

```python
optimizer.zero_grad()
loss.backward()  # Backprop through all 30 nodes simultaneously
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip=0.5)
optimizer.step()
```

---

## 📊 Data Shapes Summary

| Stage | Tensor Name | Shape | Meaning |
|-------|-------------|-------|---------|
| **Raw Graph** | x | [N, 2] | Node features (before enhancement) |
| | edge_index | [2, E] | Edge pairs |
| | y | [N] | Node labels (0/1) |
| **After PE + Features** | x | [N, 8] | Enhanced features |
| | pe | [N, 16] | Positional encoding |
| **After Batching** | x | [N_total, 8] | All nodes concatenated |
| | edge_index | [2, E_total] | All edges (with offset indices) |
| | batch | [N_total] | Graph ID per node [0,0,...,1,1,...,2,...] |
| | ptr | [num_graphs+1] | Graph boundary pointers |
| **In Model** | x_emb | [N_total, 256] | Embedded features |
| | z | [N_total, 256] | Hidden state |
| | y | [N_total, 1] | Node predictions |
| **Loss** | probs | [N_total] | Sigmoid(y) ∈ [0, 1] |
| | edge_violations | [E_total] | probs[u] * probs[v] per edge |

---

## 🔄 Multi-Graph Gradient Flow

**Critical point:** When you backprop through batched graphs:

```python
loss.backward()
```

Gradients flow **independently per graph** because:
1. ✅ edge_index only connects nodes within same graph
2. ✅ batch vector prevents cross-graph attention
3. ✅ Each graph's nodes have independent loss contributions

**Result:** Batching is purely for compute efficiency, not learning!

---

## Example: 3-Graph Batch Processing

```
Input batch (3 graphs):
  Graph 0: 10 nodes, [1,0,1,0,1,1,0,1,0,1]  ← 6 nodes in MIS
  Graph 1: 8 nodes,  [1,1,0,1,0,1,0,1]      ← 5 nodes in MIS
  Graph 2: 12 nodes, [1,0,1,0,1,0,1,0,1,0,1,0]  ← 6 nodes in MIS

Batched representation:
  x:    [30, 8]
  edge_index: [2, 45]  (edges from all 3 graphs, with adjusted IDs)
  batch: [0]*10 + [1]*8 + [2]*12
  y:    [1,0,1,0,1,1,0,1,0,1, 1,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1,0,1,0]

Model processes all 30 nodes:
  → GPS layers see full graphs (edges only within each graph)
  → Attention only within each graph (batch vector)
  → Output: [30, 1] predictions

Loss:
  loss = BCE(preds, y) + feasibility * violations
  Backprop: updates parameters based on all 30 nodes' gradients
```
