# Comprehensive Comparison: SSL vs Normal Approach

## Table of Contents
1. [High-Level Differences](#1-high-level-differences)
2. [Architecture Comparison](#2-architecture-comparison)
3. [Loss Calculation - Detailed Breakdown](#3-loss-calculation---detailed-breakdown)
4. [Loss Weights & Optimal Values](#4-loss-weights--optimal-values)
5. [Concrete Example: 5-Node Graph](#5-concrete-example-5-node-graph)
6. [Metrics Tracked](#6-metrics-tracked)
7. [When to Use Each](#7-when-to-use-each)

---

## 1. **HIGH-LEVEL DIFFERENCES**

### **Normal (Supervised) Approach**
- **Training Signal**: Ground truth labels ✓ (we know the optimal MIS)
- **Loss**: Binary Cross-Entropy (BCE) + optional feasibility loss
- **Goal**: Predict which nodes are in the optimal MIS

### **SSL (Self-Supervised) Approach**
- **Training Signal**: NO labels ✗ (we DON'T know the optimal MIS)
- **Loss**: Exponential penalty + selection maximization
- **Goal**: Learn to find feasible MIS solutions through unsupervised constraints

---

## 2. **ARCHITECTURE COMPARISON**

Both use the **same core architecture**:

```
Input Nodes
    ↓
Feature Embedding (node features + positional encoding)
    ↓
Deep Recursion: H_cycles × L_cycles = total thinking steps
    │
    └─→ GPS Layers (local MPNN + global attention)
    │
    └─→ Output Head refinement
    ↓
Output Logits (raw predictions)
    ↓
[DIVERGES HERE]
```

**Key Parameters (Same for Both)**:
- `hidden_dim`: 256 (hidden size of GPS layers)
- `num_layers`: 2 (GPS blocks per latent cycle)
- `L_cycles`: 6 (inner recursion steps)
- `H_cycles`: 2 (outer recursion steps)
- **Total recursion steps**: 2 × 6 = 12 thinking steps

---

## 3. **LOSS CALCULATION - DETAILED BREAKDOWN**

### **NORMAL (Supervised)**

**Formula:**
```
L_total = L_BCE + λ_feas · L_feasibility
```

#### **A. Binary Cross-Entropy (BCE) Loss**

```
L_BCE = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]
```

Where:
- `y` = ground truth label (0 or 1)
- `ŷ` = model logit
- `σ` = sigmoid function
- `pos_weight` = adjusts penalty for false negatives

**With pos_weight:**
```
L_BCE = pos_weight · 𝟙[y=1] · (-log(σ(ŷ))) + 𝟙[y=0] · (-log(1-σ(ŷ)))
```

**Why pos_weight?** In MIS, positive examples (nodes in MIS) are rare. If `neg_count=9000` and `pos_count=1000`, then `pos_weight = 9/1 = 9`. This penalizes missing a positive node 9× more than false positives.

#### **B. Feasibility Loss (Hinge or Soft)**

**Hinge Loss** (better for accuracy):
```
L_feas,hinge = (1/|E|) · Σ_{(u,v)∈E} max(0, p_u - 0.5) · max(0, p_v - 0.5)
```

Only penalizes edges where **BOTH** nodes are predicted as selected (> 0.5).

**Soft Loss** (guarantees feasibility):
```
L_feas,soft = (1/|E|) · Σ_{(u,v)∈E} p_u · p_v
```

Penalizes **all** edges continuously.

**Example (Hinge)**:
- If `p_u=0.3, p_v=0.9`: loss = `max(0, 0.3-0.5) × max(0, 0.9-0.5)` = `0 × 0.4` = **0** (no penalty)
- If `p_u=0.7, p_v=0.8`: loss = `max(0, 0.2) × max(0, 0.3)` = `0.2 × 0.3` = **0.06** (penalized!)

---

### **SSL (Self-Supervised)**

**Formula:**
```
L_total = λ_feas · L_feasibility + λ_select · L_selection
```

#### **A. Selection Loss (Maximize Set Size)**

```
L_selection = -(1/n) · Σ_{i=1}^{n} p_i
```

- **Negative** because we minimize loss = maximize sum
- Encourages the model to select as many nodes as possible
- Using **mean** (not sum) for batch-size invariance

**Example**:
- If predictions: `p = [0.9, 0.2, 0.8, 0.1, 0.95]`
- Mean = `(0.9 + 0.2 + 0.8 + 0.1 + 0.95) / 5` = `0.57`
- Selection loss = `-0.57` (we want to maximize this)

#### **B. Exponential Penalty Loss (Enforce Feasibility)**

```
L_feasibility = (1/|E|) · Σ_{(u,v)∈E} exp(μ · (p_u + p_v - 1)) / μ
```

Where:
- `μ (mu)` = penalty parameter (typically 3-5)
- `p_u + p_v - 1` = constraint violation for edge (u,v)

**Why exponential?**
- When `p_u + p_v ≤ 1`: violation ≤ 0 → `exp(μ·violation) ≤ 1` → small loss ✓
- When `p_u + p_v > 1`: violation > 0 → `exp(μ·violation) >> 1` → HUGE loss ✗

**Example (μ=3)**:

| p_u | p_v | Violation | exp(3·violation) | Loss |
|-----|-----|-----------|------------------|------|
| 0.2 | 0.3 | -0.5      | exp(-1.5) ≈ 0.22 | 0.07 |
| 0.5 | 0.5 | 0.0       | exp(0) = 1.0     | 0.33 |
| 0.7 | 0.7 | 0.4       | exp(1.2) ≈ 3.32  | 1.11 |
| 0.9 | 0.9 | 0.8       | exp(2.4) ≈ 11.0  | 3.67 |

**Critical threshold**: `p_u + p_v = 1.0`
- Below: loss stays small (feasible region)
- Above: loss explodes exponentially (infeasible region)

---

## 4. **LOSS WEIGHTS & OPTIMAL VALUES**

### **Normal Approach Config:**

```python
class LossConfig:
    pos_weight: float = 1.0                    # Adjust for class imbalance
    feasibility_weight: float = 0.0            # Usually 0 (BCE enough)
    feasibility_loss_type: str = "hinge"       # Better than "soft"
```

**Typical tuning:**
- `pos_weight = neg_count / pos_count` (auto-computed if 1.0)
- `feasibility_weight = 0-1` (only if violations are common)

### **SSL Approach Config:**

```python
class LossConfig:
    mu: float = 3.0                            # Penalty parameter
    feasibility_weight: float = 1.0            # Must be > 0
    selection_weight: float = 30.0             # Strongly encourage selection
```

**Key insight**: `selection_weight >> feasibility_weight`
- We want to find **large** MIS, so selection is more important
- Feasibility is a constraint, not the main objective

---

## 5. **CONCRETE EXAMPLE: 5-NODE GRAPH**

Suppose we have this graph:

```
1---2
|\ /|
| 3 |
|/ \|
4---5

Edges: (1,2), (1,3), (1,4), (2,3), (2,5), (3,4), (3,5), (4,5)
Optimal MIS: {1, 5} or {2, 4} (size = 2)
```

### **Scenario A: Model outputs raw logits**

```
logits = [2.0, -1.5, 0.5, -0.8, 1.2]
probs = sigmoid(logits) = [0.88, 0.18, 0.62, 0.31, 0.77]
```

### **NORMAL LOSS CALCULATION:**

**Assume labels = [1, 0, 0, 0, 1]** (ground truth is {1,5})

#### **1. BCE Loss:**

```
For node 1 (y=1, p=0.88): -log(0.88) ≈ 0.128
For node 2 (y=0, p=0.18): -log(1-0.18) ≈ 0.198
For node 3 (y=0, p=0.62): -log(1-0.62) ≈ 0.963  ← high! (false positive)
For node 4 (y=0, p=0.31): -log(1-0.31) ≈ 0.371
For node 5 (y=1, p=0.77): -log(0.77) ≈ 0.262

L_BCE = (0.128 + 0.198 + 0.963 + 0.371 + 0.262) / 5 ≈ 0.384

With pos_weight=2:
L_BCE = (2×0.128 + 0.198 + 0.963 + 0.371 + 2×0.262) / 5 ≈ 0.459
         (penalizes false negatives more)
```

#### **2. Feasibility Loss (Hinge):**

```
Threshold activations:
- Node 1: 0.88 > 0.5 ✓
- Node 2: 0.18 < 0.5 ✗
- Node 3: 0.62 > 0.5 ✓
- Node 4: 0.31 < 0.5 ✗
- Node 5: 0.77 > 0.5 ✓

Selected: {1, 3, 5}

Check edges for violations (nodes that are both > 0.5):
- (1,2): 0.88 × 0 = 0
- (1,3): (0.88-0.5) × (0.62-0.5) = 0.38 × 0.12 = 0.046 ✗✗✗
- (1,4): 0.38 × 0 = 0
- (2,3): 0 × 0.12 = 0
- (2,5): 0 × 0.27 = 0
- (3,4): 0.12 × 0 = 0
- (3,5): (0.62-0.5) × (0.77-0.5) = 0.12 × 0.27 = 0.032 ✗✗✗
- (4,5): 0 × 0.27 = 0

L_feasibility = (0.046 + 0.032) / 8 ≈ 0.010
```

#### **Total Loss:**

```
L_total = L_BCE + λ_feas × L_feasibility
        = 0.459 + 0.1 × 0.010
        = 0.460
```

---

### **SSL LOSS CALCULATION:**

**Same predictions, but NO labels used!**

#### **1. Selection Loss:**

```
L_selection = -mean(p) = -(0.88 + 0.18 + 0.62 + 0.31 + 0.77) / 5
            = -0.552
```

#### **2. Feasibility Loss (Exponential Penalty, μ=3):**

```
For each edge, compute constraint violation:

(1,2): p1 + p2 - 1 = 0.88 + 0.18 - 1 = 0.06
       exp(3 × 0.06) / 3 = exp(0.18) / 3 ≈ 1.20 / 3 = 0.40

(1,3): p1 + p3 - 1 = 0.88 + 0.62 - 1 = 0.50
       exp(3 × 0.50) / 3 = exp(1.5) / 3 ≈ 4.48 / 3 = 1.49 ✗✗✗ HUGE!

(1,4): p1 + p4 - 1 = 0.88 + 0.31 - 1 = 0.19
       exp(3 × 0.19) / 3 = exp(0.57) / 3 ≈ 1.77 / 3 = 0.59

(2,3): p2 + p3 - 1 = 0.18 + 0.62 - 1 = -0.20
       exp(3 × -0.20) / 3 = exp(-0.6) / 3 ≈ 0.55 / 3 = 0.18

(2,5): p2 + p5 - 1 = 0.18 + 0.77 - 1 = -0.05
       exp(3 × -0.05) / 3 = exp(-0.15) / 3 ≈ 0.86 / 3 = 0.29

(3,4): p3 + p4 - 1 = 0.62 + 0.31 - 1 = -0.07
       exp(3 × -0.07) / 3 = exp(-0.21) / 3 ≈ 0.81 / 3 = 0.27

(3,5): p3 + p5 - 1 = 0.62 + 0.77 - 1 = 0.39
       exp(3 × 0.39) / 3 = exp(1.17) / 3 ≈ 3.22 / 3 = 1.07 ✗✗✗

(4,5): p4 + p5 - 1 = 0.31 + 0.77 - 1 = 0.08
       exp(3 × 0.08) / 3 = exp(0.24) / 3 ≈ 1.27 / 3 = 0.42

L_feasibility = (0.40 + 1.49 + 0.59 + 0.18 + 0.29 + 0.27 + 1.07 + 0.42) / 8
              = 4.71 / 8 ≈ 0.589
```

#### **Total Loss:**

```
L_total = λ_feas × L_feasibility + λ_select × L_selection
        = 1.0 × 0.589 + 30.0 × (-0.552)
        = 0.589 - 16.56
        = -15.97
```

**Why is this negative?** Because we're maximizing selection (negative loss = good!). The optimizer will push probabilities UP to make selection more negative, while feasibility loss constrains violations.

---

## 6. **METRICS TRACKED**

### **Normal (Supervised)**

```
✓ loss_bce              - Main training loss
✓ loss_feasibility      - Constraint penalty
✓ accuracy              - % nodes correctly classified
✓ pred_size             - Predicted set size
✓ opt_size              - Ground truth MIS size
✓ gap                   - opt_size - pred_size
✓ approx_ratio          - pred_size / opt_size (should be ≈1)
✓ feasibility           - % edges without violations
```

### **SSL (Self-Supervised)**

```
✓ loss_feasibility      - Exponential penalty
✓ loss_selection        - Negative (maximize)
✗ accuracy              - NO LABELS!
✓ pred_size             - Predicted set size
✗ opt_size              - NO LABELS!
✗ gap                   - Can't compute without labels
✗ approx_ratio          - Can't compute without labels
✓ feasibility           - % edges without violations
✓ selection_ratio       - % nodes selected
```

---

## 7. **WHEN TO USE EACH**

| Aspect | **Normal** | **SSL** |
|--------|-----------|--------|
| **Data** | Needs labeled optimal solutions | Works with unlabeled graphs |
| **Speed** | Faster (direct BCE signal) | Slower (indirect exponential signal) |
| **Accuracy** | Better (ground truth guidance) | Worse (no ground truth) |
| **Feasibility** | May need post-processing | Designed to find feasible solutions |
| **Cost** | Expensive (need optimal labels) | Cheap (auto-unsupervised) |
| **Use Case** | Small, labeled datasets | Large, unlabeled datasets |

---

## Summary: Key Takeaways

### **Loss Design Philosophy**

**Normal**: "Tell me what's right, I'll learn to predict it"
- Direct supervision from ground truth
- BCE gives clear signal: "this node should be 1, that should be 0"
- Simple and fast

**SSL**: "Find large feasible solutions on your own"
- Indirect supervision through constraints
- Exponential penalties force feasibility (p_u + p_v ≤ 1)
- Selection loss encourages maximizing set size
- Works without labels but requires careful tuning of μ and loss weights

### **The Critical μ Parameter**

The exponential penalty's strength depends on μ:
- **μ = 1**: Weak penalty, violations tolerated
- **μ = 3-5**: Good balance (your default)
- **μ = 10+**: Very strict, almost impossible to violate

A higher μ makes the loss landscape sharper, harder to optimize, but guarantees feasibility.

### **Why selection_weight = 30 in SSL?**

With this weight, `-30 × (-0.552) = 16.56` dominates the loss, pushing the optimizer to maximize selection. The feasibility loss (≈0.6) keeps it in check by penalizing violations. The balance drives the model to find the **largest feasible set**.

---
