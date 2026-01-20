# Training Script (train_mis.py) - Complete Documentation

This document provides a detailed explanation of the training script for the MIS Graph Transformer.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration System](#2-configuration-system)
3. [Data Pipeline](#3-data-pipeline)
4. [Training Loop](#4-training-loop)
5. [Metrics Explained](#5-metrics-explained)
6. [Validation Phase](#6-validation-phase)
7. [Key Functions](#7-key-functions)

---

## 1. Overview

`train_mis.py` is the main training script that:

1. Loads MIS graph data
2. Configures the model
3. Runs training with deep supervision
4. Computes metrics and logs to wandb
5. Saves checkpoints

### Execution Flow

```
main()
  │
  ├─► init_distributed()          # Setup multi-GPU if available
  │
  ├─► Create Config               # Hyperparameters
  │
  ├─► Create Datasets             # Train + Val
  │     ├─ MISDatasetConfig
  │     └─ MISDataset
  │
  ├─► Create Model                # GraphTransformerTRM
  │
  ├─► Create Optimizer            # AdamW
  │
  └─► Training Loop
        │
        ├─► For each epoch:
        │     │
        │     ├─► Training Phase
        │     │     ├─ Forward pass (TRM recursion)
        │     │     ├─ Backward pass
        │     │     ├─ Gradient clipping
        │     │     ├─ Optimizer step
        │     │     └─ Log metrics
        │     │
        │     └─► Validation Phase
        │           ├─ Forward pass (no grad)
        │           └─ Log metrics
        │
        └─► Save checkpoint every 50 epochs
```

---

## 2. Configuration System

### 2.1 LossConfig

```python
class LossConfig(BaseModel):
    feasibility_weight: float = 1.0
```

Controls the weight of feasibility loss relative to BCE loss.

- **High weight (50-100):** Strong penalty for edge violations
- **Low weight (0-1):** Focus on classification accuracy
- **Zero weight:** Used for overfitting tests

### 2.2 ArchConfig

```python
class ArchConfig(BaseModel):
    name: str = "graph_trm"
    hidden_dim: int = 256      # Hidden state dimension
    num_layers: int = 2        # GPS layers per latent step
    L_cycles: int = 6          # Inner loop iterations
    H_cycles: int = 2          # Outer loop iterations
```

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `hidden_dim` | Width of hidden layers | More = more capacity, more memory |
| `num_layers` | GPS layers per L-cycle | More = deeper per iteration |
| `L_cycles` | Latent update iterations | More = more message passing |
| `H_cycles` | Outer refinement loops | More = more refinement stages |

**Total GPS applications** = num_layers × L_cycles × H_cycles = 2 × 6 × 2 = **24**

### 2.3 Main Config

```python
class Config(BaseModel):
    # Data
    data_paths: List[str]           # Paths to dataset shards
    val_split: float = 0.1          # 10% validation
    global_batch_size: int = 128    # Graphs per batch

    # Training
    lr: float = 0.0005              # Learning rate
    lr_min_ratio: float = 0.1       # Min LR = base * 0.1
    lr_warmup_steps: int = 50       # Warmup steps
    epochs: int = 1000              # Training epochs

    # Optimizer
    weight_decay: float = 0.0       # L2 regularization
    beta1: float = 0.9              # Adam β1
    beta2: float = 0.95             # Adam β2
    grad_clip: float = 0.5          # Gradient clipping

    # Deep Supervision
    n_supervision: int = 1          # Forward passes per batch
```

---

## 3. Data Pipeline

### 3.1 Dataset Creation

```python
# Training dataset
train_ds_config = MISDatasetConfig(
    dataset_paths=cfg.data_paths,
    global_batch_size=cfg.global_batch_size,
    rank=rank,
    num_replicas=world_size,
    drop_last=True,           # Drop partial batches (training)
    val_split=cfg.val_split,
    seed=cfg.seed,
)
train_dataset = MISDataset(train_ds_config, split="train")

# Validation dataset
val_ds_config = MISDatasetConfig(
    ...,
    drop_last=False,          # Keep partial batches (validation)
)
val_dataset = MISDataset(val_ds_config, split="val")
```

### 3.2 Data Flow

```
Shards (.pt files)
      │
      ▼
  Load graphs
      │
      ▼
  Graph-level split (90% train, 10% val)
      │
      ▼
  Feature enhancement (Laplacian PE + node features)
      │
      ▼
  Batch collation (PyG Batch)
      │
      ▼
  DataLoader yields (batch_name, batch_dict, batch_size)
```

### 3.3 Batch Dictionary

```python
batch = {
    "x": tensor[N, 8],           # Enhanced node features
    "edge_index": tensor[2, E],  # Edge connectivity
    "batch": tensor[N],          # Graph assignment (0, 0, 1, 1, 1, 2, ...)
    "y": tensor[N],              # Labels (1 = in MIS, 0 = not)
    "ptr": tensor[G+1],          # Graph boundaries in batch
    "num_graphs": int,           # Number of graphs in batch
    "pe": tensor[N, 16],         # Laplacian positional encoding
    "opt_value": tensor[G],      # Optimal MIS size per graph
}
```

Where:
- N = total nodes across all graphs in batch
- E = total edges across all graphs in batch
- G = number of graphs in batch

---

## 4. Training Loop

### 4.1 Per-Step Training

```python
for batch_name, batch, batch_size in train_dataloader:
    # 1. Update learning rate (warmup + cosine)
    current_lr = cosine_schedule_with_warmup(...)

    # 2. Move batch to GPU
    batch = {k: v.cuda() for k, v in batch.items()}

    # 3. Initialize carry state
    carry = raw_model.initial_carry(batch)

    # 4. Forward pass (TRM recursion)
    while not all_finish:
        carry, step_loss, metrics, preds, all_finish = model(carry, batch)
        if loop_step >= n_supervision:
            all_finish = True

    # 5. Backward pass
    optimizer.zero_grad()
    final_loss.backward()

    # 6. Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # 7. Optimizer step
    optimizer.step()
```

### 4.2 Learning Rate Schedule

```python
def cosine_schedule_with_warmup(current_step, base_lr, num_warmup_steps,
                                 num_training_steps, min_ratio=0.1):
    # Warmup phase: linear increase
    if current_step < num_warmup_steps:
        return base_lr * (current_step / num_warmup_steps)

    # Cosine decay phase
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(π * progress)))
```

Visual:
```
LR
 │
 │    /\
 │   /  \___
 │  /       \___
 │ /            \___
 └─────────────────────► Steps
   │   │              │
   0  warmup     total_steps
```

### 4.3 Deep Supervision

The `n_supervision` parameter controls how many forward passes per batch:

```python
n_supervision = 1  # Default: single forward pass

while not all_finish:
    carry, step_loss, metrics, preds, all_finish = model(carry, batch)
    loop_step += 1
    if loop_step >= n_supervision:
        all_finish = True
```

With `n_supervision > 1`:
- Multiple forward passes with gradients
- Each pass refines from previous prediction
- More computation, potentially better learning

---

## 5. Metrics Explained

### 5.1 Loss Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| `loss_total` | BCE + λ × Feasibility | Minimize |
| `loss_bce` | Weighted BCE (with pos_weight) | Minimize |
| `loss_bce_raw` | Unweighted BCE | Monitor |
| `loss_feasibility` | Edge violation penalty | Minimize |
| `loss_feasibility_raw` | Same as above | Monitor |
| `loss_feasibility_weighted` | λ × feasibility | Monitor |

### 5.2 Classification Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `precision` | TP / (TP + FP) | Of predicted MIS nodes, how many are correct? |
| `recall` | TP / (TP + FN) | Of true MIS nodes, how many did we find? |
| `f1` | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| `acc` | Correct / Total | Overall accuracy |

### 5.3 MIS-Specific Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `optimal_size` | Ground truth MIS size | What we're trying to match |
| `postprocessed_size` | Size after greedy decode | What we actually achieve |
| `gap` | optimal - postprocessed | How many nodes we're missing |
| `gap_ratio` | gap / optimal | Relative error |
| `approx_ratio_postprocessed` | postprocessed / optimal | Solution quality (closer to 1 = better) |
| `feasibility` | 1 - violations / predicted | Are predictions valid? (1 = no violations) |

### 5.4 Why These Metrics Matter

**Raw model output** (probs > 0.5) may be **infeasible** - it might select adjacent nodes.

**Post-processing** (greedy decode) ensures feasibility:
1. Sort nodes by probability
2. Greedily select highest prob node
3. Remove its neighbors from consideration
4. Repeat

So we track:
- `raw_pred_size`: What model outputs (may be invalid)
- `postprocessed_size`: After greedy fix (always valid)
- `optimal_size`: Ground truth (always valid)

---

## 6. Validation Phase

### 6.1 Validation Loop

```python
if cfg.validate_every_epoch:
    model.eval()

    with torch.no_grad():
        for batch_name, batch, batch_size in val_dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            carry = raw_model.initial_carry(batch)

            # Fixed number of supervision steps (no early stopping)
            for _ in range(cfg.n_supervision):
                carry, step_loss, metrics, preds, _ = model(carry, batch)

            # Accumulate metrics
            val_metrics["loss_total"] += metrics["loss_total"].item()
            ...
```

### 6.2 Key Differences from Training

| Aspect | Training | Validation |
|--------|----------|------------|
| Gradients | Enabled | Disabled (`torch.no_grad()`) |
| Dropout | Active | Disabled (`model.eval()`) |
| drop_last | True | False |
| Early stopping | Based on q_hat | Fixed iterations |
| Batch size | Same | Same |

---

## 7. Key Functions

### 7.1 `greedy_decode`

```python
def greedy_decode(probs, edge_index, num_nodes):
    """
    Convert probabilities to valid Independent Set.

    Algorithm:
    1. Sort nodes by probability (descending)
    2. For each node (highest prob first):
       - If not blocked: add to set, block all neighbors
       - If blocked: skip
    3. Return set size and selection mask
    """
```

**Example:**
```
Graph: A -- B -- C
Probs: [0.8, 0.9, 0.7]

Step 1: Select B (prob=0.9), block A and C
Step 2: A blocked, skip
Step 3: C blocked, skip

Result: {B}, size=1
```

### 7.2 `compute_postprocessing_metrics`

```python
def compute_postprocessing_metrics(probs, edge_index, labels, batch_vec=None, ptr=None):
    """
    Compute MIS metrics per graph, then average.

    Per-graph computation:
    1. Extract nodes/edges for graph g
    2. Remap edge indices to local (0-indexed)
    3. Run greedy_decode
    4. Compute optimal_size, postprocessed_size, gap

    Average across all graphs in batch.
    """
```

**Why per-graph?**
- Batches contain graphs of varying sizes
- A 50-node graph and 200-node graph shouldn't be weighted equally
- Per-graph averaging gives fair comparison

### 7.3 `count_parameters`

```python
def count_parameters(model):
    """
    Count model parameters with per-module breakdown.

    Returns:
        total_params: Total parameter count
        trainable_params: Parameters with requires_grad=True
        breakdown: Dict of {module_name: param_count}
    """
```

---

## Summary

The training script orchestrates:

1. **Configuration:** Pydantic-based config for type safety
2. **Data:** Graph batching with enhanced features and PE
3. **Training:** TRM recursion with warmup + cosine LR
4. **Metrics:** Comprehensive tracking of loss, classification, and MIS quality
5. **Validation:** Per-epoch evaluation with detailed logging

Key design decisions:
- Gradient clipping (0.5) for stability with deep recursion
- Cosine schedule with warmup for smooth training
- Per-graph metrics for fair evaluation
- Checkpoint every 50 epochs to save disk space
