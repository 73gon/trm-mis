# Evaluation Methodology: Measuring MIS Solution Quality

## Metrics Overview

### What You're Measuring

```
Ground Truth: Optimal MIS of size OPT
Your Prediction: Greedy decoded MIS of size PRED

approx_ratio = PRED / OPT

Example:
OPT = 15 (optimal MIS for this graph)
PRED = 13 (your model found)
approx_ratio = 13/15 = 0.867 (86.7% of optimal)
```

---

## Raw vs Greedy Metrics

### Before Greedy Decode (Raw)
```python
probs = [0.9, 0.8, 0.3, 0.7, ...]  # Model output
preds_binary = (probs > 0.5).float()  # Threshold
# [1, 1, 0, 1, ...]

# Can this form valid independent set?
# Check edges: Maybe nodes 0 and 1 are adjacent!
# Feasibility = 0.8 (20% of nodes violate constraints)
```

### After Greedy Decode (Final)
```python
greedy_set = greedy_decode(probs, edges)
# [1, 0, 0, 1, ...]  (removed node 1 to avoid conflict)

# Is this valid?
# Feasibility = 1.0 (guaranteed by greedy algorithm)

# How good?
# approx_ratio = 3 / 4 = 0.75 (75% of optimal)
```

---

## Key Metrics Explained

### 1. **feasibility_raw** (Raw Predictions)
```
Measures: What % of raw model predictions form independent set?

Formula: 1 - (edge_violations / num_predicted_nodes)

Range: 0 to 1
- 1.0 = perfect (no adjacent nodes predicted)
- 0.8 = 20% of nodes violate constraints
- 0.0 = completely invalid

Why it matters:
- Shows if model learned to respect constraints
- 0.8 is actually good! Shows model learned MIS structure
```

### 2. **feasibility_greedy** (After Decode)
```
Measures: Is greedy decoded set valid independent set?

Formula: 1.0 (always)

Range: Always 1.0
- Greedy decode GUARANTEES validity

Why it matters:
- Confirms final output is valid
- Should always be 1.0 (sanity check)
```

### 3. **pred_size_vs_opt** (Raw Predictions)
```
Measures: How many nodes did raw predictions select?
           Compared to optimal size?

Formula: num_predicted_nodes / optimal_size

Range: 0 to ∞
- 1.0 = predicting same size as optimal
- 0.8 = underpredicting by 20%
- 1.2 = overpredicting by 20%

Why it matters:
- Shows if model is conservative or greedy
- Doesn't account for validity (can be invalid)
- Honest measure of raw output
```

### 4. **approx_ratio_greedy** (True Metric) ⭐
```
Measures: After greedy decode, how close to optimal?

Formula: valid_set_size / optimal_size

Range: 0 to 1
- 1.0 = found optimal MIS!
- 0.95 = 95% of optimal (excellent)
- 0.85 = 85% of optimal (good)
- 0.5 = 50% of optimal (okay, random is ~40%)

Why it matters:
- THIS IS YOUR REAL METRIC
- Measures actual solution quality
- Accounts for validity (always valid)
- Comparable across graphs of different sizes
```

### 5. **num_violations**
```
Measures: How many edge violations in raw predictions?

Formula: count(edges where both endpoints predicted=1)

Range: 0 to num_edges
- 0 = perfect (no violations)
- 100 = many violations

Why it matters:
- Shows where model struggles
- Lower is better
- Together with feasibility: tells full story
```

---

## Full Workflow Example

### Graph Setup
```
Nodes: {0, 1, 2, 3, 4, 5}
Edges: {(0,1), (1,2), (2,3), (3,4), (4,5)}
Optimal MIS: {0, 2, 4} size = 3

(Linear chain: 0-1-2-3-4-5)
```

### Model Prediction
```
Forward pass through 18 cycles:
probs = [0.9, 0.85, 0.7, 0.6, 0.8, 0.3]
        Nodes: 0   1   2   3   4   5

Interpretation:
- Node 0: 90% confidence (should be in MIS)
- Node 1: 85% confidence (should NOT, conflicts with 0)
- Node 2: 70% confidence (should be in MIS)
- Node 3: 60% confidence (should NOT, conflicts with 2)
- Node 4: 80% confidence (should be in MIS)
- Node 5: 30% confidence (should NOT, conflicts with 4)
```

### Raw Metrics (Before Greedy)
```
Threshold at 0.5:
raw_preds = [1, 1, 1, 1, 1, 0]
Nodes selected: {0, 1, 2, 3, 4}

Check violations:
- Edge (0,1): both=1 ✗ VIOLATION
- Edge (1,2): both=1 ✗ VIOLATION
- Edge (2,3): both=1 ✗ VIOLATION
- Edge (3,4): both=1 ✗ VIOLATION
- Edge (4,5): 1,0 ✓ OK

num_violations = 4
num_predicted = 5
feasibility_raw = 1 - (4/5) = 0.2  (only 20% valid)
pred_size_vs_opt = 5 / 3 = 1.67 (67% over-predicting)
```

### Greedy Decode
```
Step 1: Sort by probability
Sorted: [0(0.9), 4(0.8), 1(0.85), 2(0.7), 3(0.6), 5(0.3)]

Step 2: Greedy select
- Node 0: no neighbors selected → SELECT ✓
- Node 4: no neighbors selected → SELECT ✓
- Node 1: adjacent to 0 → SKIP ✗
- Node 2: adjacent to 1 (skipped), not adjacent to 0,4 → SELECT ✓
- Node 3: adjacent to 2 and 4 → SKIP ✗
- Node 5: adjacent to 4 → SKIP ✗

greedy_set = {0, 2, 4} (size = 3)
```

### Final Metrics (After Greedy)
```
feasibility_greedy = 1.0 ✓ (guaranteed valid)

num_violations_greedy = 0 ✓ (no adjacent nodes)

valid_set_size = 3

approx_ratio_greedy = 3 / 3 = 1.0 ✓✓✓ OPTIMAL!
```

---

## Metric Interpretation Guide

### Perfect Performance
```
approx_ratio_greedy = 1.0
feasibility_greedy = 1.0
num_violations = 0

Interpretation: Found optimal solution ✓✓✓
```

### Good Performance
```
approx_ratio_greedy = 0.90-0.95
feasibility_greedy = 1.0
num_violations = 0

Interpretation: Near-optimal solution ✓✓
```

### Decent Performance
```
approx_ratio_greedy = 0.75-0.90
feasibility_greedy = 1.0
num_violations = 0

Interpretation: Good approximation ✓
```

### Poor Performance
```
approx_ratio_greedy = 0.50-0.75
feasibility_greedy = 1.0
num_violations = 0

Interpretation: Okay but room for improvement
```

### Very Poor Performance
```
approx_ratio_greedy < 0.50
feasibility_greedy = 1.0
num_violations = 0

Interpretation: Model not learning well ✗
```

---

## Train vs Test Comparison

### No Overfitting (Good)
```
train approx_ratio = 0.88
test approx_ratio = 0.87
Difference = 0.01

Interpretation: Generalizes well ✓
```

### Moderate Overfitting
```
train approx_ratio = 0.92
test approx_ratio = 0.85
Difference = 0.07

Interpretation: Decent generalization ✓
```

### Severe Overfitting
```
train approx_ratio = 0.98
test approx_ratio = 0.70
Difference = 0.28

Interpretation: Model memorized training data ✗
Need: More regularization, better dataset
```

---

## Comparison to Baselines

### Random Selection
```
For random independent set:
approx_ratio ≈ 0.40-0.50 (depends on density)

Your model should significantly beat this
```

### Greedy by Degree
```
Standard greedy algorithm:
approx_ratio ≈ 0.50-0.70 (varies by graph)

Your model should be competitive or better
```

### SDP (Theoretical)
```
SDP algorithm:
approx_ratio ≥ log²(n) / n (guaranteed)
On typical instances: 0.80-0.95

Your model should be comparable or better in practice
```

---

## Summary

| Metric | Measures | Ideal | Warning |
|--------|----------|-------|---------|
| **feasibility_raw** | Raw model quality | > 0.8 | < 0.5 means bad training |
| **feasibility_greedy** | Validity guarantee | 1.0 | Should always be 1.0 |
| **pred_size_vs_opt** | Prediction bias | 1.0 | > 1.2 = too greedy |
| **approx_ratio_greedy** | **Solution quality** | **> 0.85** | **< 0.60 = poor** |
| **num_violations** | Constraint violations | 0 | Should be 0 after decode |

**Focus on:** `approx_ratio_greedy` is your main metric!
