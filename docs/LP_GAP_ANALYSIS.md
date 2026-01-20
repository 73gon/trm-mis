# LP Gap Analysis (analyze_lp_gap.py) - Documentation

This document explains how to analyze whether your MIS dataset is "hard enough" using LP-ILP gap analysis.

---

## Table of Contents

1. [What is LP-ILP Gap?](#1-what-is-lp-ilp-gap)
2. [Why Does Hardness Matter?](#2-why-does-hardness-matter)
3. [How the Analysis Works](#3-how-the-analysis-works)
4. [Interpreting Results](#4-interpreting-results)
5. [Usage](#5-usage)

---

## 1. What is LP-ILP Gap?

### 1.1 The MIS Problem

**Maximum Independent Set (MIS)** is an **Integer Linear Program (ILP)**:

```
Maximize:    Σ x_i           (maximize number of selected nodes)
Subject to:  x_i + x_j ≤ 1   for all edges (i,j)
             x_i ∈ {0, 1}    (binary: selected or not)
```

### 1.2 LP Relaxation

If we **relax** the binary constraint to continuous:

```
Maximize:    Σ x_i
Subject to:  x_i + x_j ≤ 1   for all edges (i,j)
             0 ≤ x_i ≤ 1     (continuous: fractional selection)
```

This is the **Linear Program (LP) relaxation**.

### 1.3 The Gap

- **ILP solution**: True optimal MIS size (integer)
- **LP solution**: Optimal value of relaxation (can be fractional)

**LP value ≥ ILP value** always (LP is less constrained)

**Gap = LP value - ILP value**

### 1.4 Example

```
Graph: A -- B -- C (path graph)

ILP (true MIS):
  Optimal: {A, C}
  Value: 2

LP (relaxation):
  Optimal: x_A = 0.5, x_B = 0.5, x_C = 0.5
  Each constraint satisfied: 0.5 + 0.5 = 1 ≤ 1
  Value: 1.5

Gap = 1.5 - 2 = -0.5???
```

Wait, this shows LP < ILP which is wrong. Let me reconsider...

Actually for this path graph:
```
LP optimal: x_A = 1, x_B = 0, x_C = 1
Constraints: x_A + x_B = 1 ≤ 1 ✓, x_B + x_C = 0 + 1 = 1 ≤ 1 ✓
LP value = 2

ILP value = 2 (same!)

Gap = 0
```

This graph is "easy" - LP = ILP.

### 1.5 When Gap > 0

```
Graph: Triangle A -- B -- C -- A

ILP (true MIS):
  Optimal: {A} or {B} or {C}
  Value: 1

LP (relaxation):
  Optimal: x_A = x_B = x_C = 0.5
  Constraints: 0.5 + 0.5 = 1 ≤ 1 for all edges ✓
  LP value = 1.5

Gap = 1.5 - 1 = 0.5
Gap ratio = 0.5 / 1 = 50%
```

The triangle is "hard" - LP overestimates.

---

## 2. Why Does Hardness Matter?

### 2.1 Easy Graphs

If most graphs have **LP = ILP**:
- A simple LP solver could find optimal
- No need for neural network
- Model won't learn anything useful

Examples of easy graphs:
- **Bipartite graphs**: LP = ILP always
- **Trees**: LP = ILP always
- **Sparse graphs**: Often LP ≈ ILP

### 2.2 Hard Graphs

If graphs have **significant LP-ILP gap**:
- No polynomial-time algorithm known
- Neural network can learn useful heuristics
- Real-world value in solving these

Examples of hard graphs:
- **Dense random graphs**: Large gaps
- **Triangles/cliques**: Maximum gap
- **Community-structured**: Moderate gaps

### 2.3 Target Difficulty

For training a useful model:
- **< 20% easy graphs**: Good dataset
- **20-50% easy graphs**: Acceptable
- **> 50% easy graphs**: Too easy, regenerate

---

## 3. How the Analysis Works

### 3.1 LP Solver

```python
def compute_lp_relaxation(edge_index, num_nodes):
    # Objective: maximize sum(x_i) = minimize -sum(x_i)
    c = -np.ones(num_nodes)

    # Constraints: x_i + x_j <= 1 for each edge
    A_ub = build_constraint_matrix(edge_index)
    b_ub = np.ones(num_edges)

    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(num_nodes)]

    # Solve with scipy
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return -result.fun  # LP optimal value
```

### 3.2 ILP Value

The ILP optimal value is stored in the dataset as `opt_value`:
- Pre-computed during dataset generation
- Uses exact solver (e.g., Gurobi, CPLEX)

### 3.3 Metrics Computed

```python
results = {
    "lp_values": [],      # LP relaxation solutions
    "ilp_values": [],     # True optimal (from dataset)
    "gaps": [],           # LP - ILP
    "gap_ratios": [],     # (LP - ILP) / ILP
    "num_nodes": [],      # Graph sizes
    "densities": [],      # Edge density
}
```

---

## 4. Interpreting Results

### 4.1 Output Example

```
📊 Dataset Statistics (n=500 samples):
  Nodes: 150.3 ± 58.2
  Edges: 1123.4 ± 892.1
  Density: 0.089 ± 0.045

📈 LP vs ILP Analysis:
  LP value (mean): 52.3 ± 18.4
  ILP value (mean): 47.8 ± 16.9
  Gap (mean): 4.5 ± 3.2
  Gap ratio (mean): 9.42% ± 6.83%

🎯 Hardness Analysis:
  'Easy' graphs (gap ratio < 1%): 45/500 (9.0%)
  'Medium' graphs (gap ratio 1-10%): 312/500 (62.4%)
  'Hard' graphs (gap ratio > 10%): 143/500 (28.6%)

💡 Interpretation:
  ✅ GOOD: Only 9% of graphs are 'easy'
  → Dataset difficulty is appropriate
```

### 4.2 What the Numbers Mean

| Metric | Good Range | Warning |
|--------|-----------|---------|
| Easy % | < 20% | > 50% too easy |
| Medium % | 40-70% | - |
| Hard % | 20-40% | > 60% may be too hard |
| Mean gap ratio | 5-15% | < 2% too easy |

### 4.3 Density Correlation

Higher density typically → larger gap:
- Dense graphs have more constraints
- More cliques and triangles
- Harder for LP to approximate

Expected correlation: **0.2 - 0.5**

---

## 5. Usage

### 5.1 Command Line

```bash
# Analyze default dataset
python dataset/analyze_lp_gap.py --data_path data/mis-10k --max_samples 500

# Analyze custom dataset
python dataset/analyze_lp_gap.py --data_path /path/to/data --max_samples 1000
```

### 5.2 Output Files

The script saves:
- `lp_ilp_analysis.npz`: Raw numpy arrays
- `lp_ilp_analysis.png`: Visualization plots

### 5.3 Plots Generated

1. **Gap Distribution**: Histogram of LP-ILP gaps
2. **Gap Ratio Distribution**: Histogram of relative gaps
3. **Density vs Gap**: Scatter plot showing correlation
4. **LP vs ILP**: Scatter plot with y=x reference line

---

## 6. Recommendations

### If Dataset is Too Easy

1. **Increase density**: Generate graphs with more edges
2. **Avoid bipartite**: Use Erdős-Rényi or Barabási-Albert models
3. **Add cliques**: Force triangles and dense subgraphs
4. **Larger graphs**: Bigger graphs tend to be harder

### If Dataset is Too Hard

1. **Decrease density**: Sparser graphs
2. **Smaller graphs**: 50-100 nodes instead of 200+
3. **Use tree-like structures**: More bipartite-ish

### Ideal Dataset Characteristics

```
Nodes: 50-250 (variable)
Density: 0.05-0.15
Easy graphs: < 20%
Mean gap ratio: 5-15%
```

---

## Summary

LP-ILP gap analysis tells you whether your MIS dataset is **challenging enough** for training a neural network:

- **Zero gap** = LP finds optimal = easy graph
- **Large gap** = LP overestimates = hard graph

Use this tool to validate your dataset before training. If too many graphs are easy, the model won't learn meaningful patterns.
