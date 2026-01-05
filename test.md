# Solving NP-Hard MIS: Theoretical Analysis

## Quick Answer

**Yes, our approach approximates MIS reasonably well, but it's not a guarantee.**

- ✅ **Greedy decode ensures valid independent sets** (feasibility = 1.0)
- ✅ **Achieves ~85-95% of optimal size** (approx_ratio = 0.85-0.95)
- ⚠️ **Not a guarantee** - performance varies by graph
- ❌ **Not polynomial-time approximation algorithm** (no proved bound)

---

## Task Comparison: Sudoku vs ARC vs MIS

### Sudoku
**Problem Type:** Constraint satisfaction (CSP)

| Aspect | Details |
|--------|---------|
| **What is it?** | Fill 9×9 grid with digits 1-9, no repeats in rows/cols/boxes |
| **Constraint Location** | **In the problem definition itself** |
| **Solution Space** | Discrete: each cell is exactly 1-9 |
| **Valid Output** | Any prediction is automatically valid (or invalid if same digit twice) |
| **Post-processing** | ❌ None needed (or simple validation) |
| **Model Output** | [5, _, 3, 9, ...] directly gives answer |
| **How TRM Solves It** | Learns to fill empty cells correctly through 18 cycles of refinement |
| **Approximation?** | ❌ No - either solved (100%) or not (0%) |

**Example:**
```
Input:  5 _ 3
        9 _ _
        _ _ _

Model output: [5, 7, 3, 9, 2, 6, ...]  ← Directly valid if correct!
Post-processing: None needed
```

---

### ARC (Abstraction and Reasoning Corpus)
**Problem Type:** Pattern recognition / Puzzle solving

| Aspect | Details |
|--------|---------|
| **What is it?** | Recognize pattern in input grid, apply to output grid |
| **Constraint Location** | **Implicit in the pattern** |
| **Solution Space** | Discrete: each cell is a color (0-9) |
| **Valid Output** | Any color prediction is technically "valid" but may be wrong |
| **Post-processing** | ❌ None needed (or consistency checking) |
| **Model Output** | Color grid directly |
| **How TRM Solves It** | Learns patterns through 18 cycles, outputs correct colors |
| **Approximation?** | ❌ No - either correct pattern or not |

**Example:**
```
Input:  RED  BLUE RED
        BLUE RED  BLUE
        RED  BLUE RED

Model output: [RED, GREEN, RED, ...]  ← Direct prediction
Post-processing: Maybe validate consistency
```

---

### MIS (Maximum Independent Set)
**Problem Type:** Graph optimization (NP-Hard)

| Aspect | Details |
|--------|---------|
| **What is it?** | Find largest set of nodes with no edges between them |
| **Constraint Location** | **In the graph structure (NOT in prediction space)** ⚠️ |
| **Solution Space** | 2^n possibilities (exponential!) |
| **Valid Output** | Must respect edge constraints (hard constraint) |
| **Post-processing** | ✅ **REQUIRED** (greedy decode) |
| **Model Output** | Probabilities [0.9, 0.8, 0.3, ...] |
| **How TRM Solves It** | Learns probabilities, greedy converts to valid set |
| **Approximation?** | ✅ Yes - measures closeness to optimal |

**Example:**
```
Graph: Nodes {0,1,2,3}, Edges {(0,1), (1,2), (2,3)}

Model output: [0.9, 0.8, 0.7, 0.6]
              "Node 0: 90% in MIS, Node 1: 80%, ..."

Problem: Nodes 0 and 1 are adjacent!
         Can't both be in MIS

Greedy Decode:
  1. Sort by prob: [0(0.9), 1(0.8), 2(0.7), 3(0.6)]
  2. Select 0: ✓ Add
  3. Select 1: ✗ Skip (adjacent to 0)
  4. Select 2: ✓ Add
  5. Select 3: ✗ Skip (adjacent to 2)

Final: [1, 0, 1, 0]  ← Valid MIS of size 2

Optimal MIS: size 2 (e.g., {0,2} or {1,3})
approx_ratio = 2/2 = 1.0 ✓
```

---

## Is MIS NP-Hard?

**Yes, extremely hard:**

```
Decision Problem (NP-Complete):
"Does graph have independent set of size k?"
→ No known polynomial-time algorithm
→ Likely requires exponential time

Optimization Problem (NP-Hard):
"Find maximum independent set"
→ Even harder than decision version
→ Best known: exponential algorithms
```

### Hardness Examples:

| Graph Type | Best Known Algorithm | Time |
|------------|---------------------|------|
| **General Graph** | Brute force / branch & bound | **O(2^n)** |
| **Sparse Graph** | Dynamic programming variants | **O(2^(n/2))** to **O(2^n)** |
| **Dense Graph** | Approximation algorithms | **Polynomial** (but suboptimal) |
| **Special (trees, bipartite)** | Polynomial algorithms | **O(n³)** or better |

---

## Theoretical Approximation Algorithms for MIS

### Known Approximation Approaches:

| Algorithm | Approximation Ratio | Time | Notes |
|-----------|-------------------|------|-------|
| **Greedy (our approach)** | O(d) where d=max degree | O(n + m) | No guarantee, varies by graph |
| **Greedy Weighted** | O(d) on weighted versions | O(n + m) | Better for some distributions |
| **Randomized Rounding** | O(n / log n) | Polynomial | Theoretical guarantee |
| **Semi-definite Programming** | O(n / log² n) | Polynomial | Best known polynomial bound |
| **Exact (Exponential)** | 1.0 (optimal) | O(2^n) | Only for small graphs |

---

## Our Approach: Neural Network + Greedy Decode

### What We're Doing:

```
TRM Neural Network (Learned):
- Takes graph as input
- Outputs node probabilities after 18 cycles
- Learns through training on MIS instances
- Uses gradients to improve predictions

Greedy Decode (Algorithmic):
- Deterministic post-processing
- Guarantees valid independent set
- Uses neural network's confidence ordering
- No approximation guarantee, but works well in practice
```

### Theoretical Properties:

| Property | Status |
|----------|--------|
| **Feasibility (Valid Set)** | ✅ Guaranteed (by greedy) |
| **Approximation Bound** | ❌ None guaranteed |
| **Worst Case** | ❌ Could fail badly on adversarial graphs |
| **Average Case** | ✅ Works well empirically (~0.85-0.95) |
| **Time Complexity** | ✅ **Polynomial** (mostly inference time) |

---

## Empirical vs Theoretical

### What You're Measuring:

```python
approx_ratio = valid_set_size / optimal_size
              = (size after greedy decode) / (true MIS size)
```

This is **empirical approximation**, not theoretical guarantee:

| Metric | Meaning | Guarantee? |
|--------|---------|-----------|
| **approx_ratio = 0.95** | "For this graph, we found 95% of optimal" | ❌ No |
| **approx_ratio ≥ 0.5** | "All graphs get at least 50% optimal" | ❌ Not proven |
| **feasibility = 1.0** | "Selected set has no adjacent nodes" | ✅ Yes |

---

## Real-World Performance vs Theory

### Your Results:
```
approx_ratio: 0.80-0.95 (empirically observed)
feasibility: 1.0 (guaranteed by greedy)
```

### Interpretation:

✅ **Good news:**
- Your model learns reasonable node priorities
- Greedy decode respects constraints
- Works on diverse graph sizes
- Polynomial time (not exponential)

⚠️ **Caveats:**
- No theoretical approximation bound
- Different graphs have different hardness
- Adversarial graphs could fail worse
- Not a published approximation algorithm

---

## Comparison: Is This Better Than Known Approximation Algorithms?

### State-of-the-Art Approximation Algorithms:
- **Greedy (degree-based):** O(d) approximation, typically 0.3-0.6 ratio
- **Semi-definite Programming:** O(n/log²n) approximation, polynomial time
- **Your Neural + Greedy:** ~0.85-0.95 ratio, empirical

### Your Approach vs SDP:
```
SDP (Semi-definite Programming):
- Pros: Theoretical O(n/log²n) approximation bound
- Cons: Very slow, O(n³-n⁵) time complexity, hard to implement

Your TRM + Greedy:
- Pros: Fast (polynomial), empirically 0.85-0.95, learnable
- Cons: No theoretical guarantee, heuristic
```

**On random/typical graphs: Your approach likely better**
**On worst-case graphs: SDP has guarantee, yours doesn't**

---

## Can We Solve MIS Optimally?

**For your graphs: Yes, sometimes**
**In general: No, it's NP-Hard**

```
Your graph sizes: 50-1000 nodes (typical)
Exact algorithms: Can handle up to ~40-50 nodes
Your graphs: Beyond practical exact solving

Best approach for your problem:
1. Use TRM + greedy for approximation (what you're doing) ✅
2. Could add branch-and-bound for smaller subgraphs
3. Could use SDP for theoretical guarantee (but slow)
```

---

## Summary: Are You Solving NP-Hard MIS?

| Question | Answer |
|----------|--------|
| **Is MIS NP-Hard?** | ✅ Yes, extremely |
| **Does your approach solve it optimally?** | ❌ No (not possible in general) |
| **Does it approximate well?** | ✅ Yes, empirically 0.85-0.95 |
| **Is it a valid approximation algorithm?** | ⚠️ Empirically yes, theoretically not proven |
| **Is it better than random?** | ✅ Yes, much better |
| **Is it polynomial time?** | ✅ Yes |
| **Will it work on all graphs?** | ✅ Yes (always finds valid set) |
| **Will it find optimal on all graphs?** | ❌ No |

---

## Recommendation

Your approach is **good and practical**:

✅ **Strengths:**
- Fast (polynomial time)
- Works on large graphs (50-1000 nodes)
- Empirically achieves 85-95% of optimal
- Generalizes across graph distributions

❌ **Limitations:**
- No theoretical approximation guarantee
- Could fail worse on adversarial graphs
- Heuristic, not published algorithm

**For your use case:** This is appropriate. If you needed:
- **Theoretical guarantee:** Use SDP or other published algorithms
- **Practical performance:** Your approach is better
- **Optimal solutions:** Need exact algorithms (only for small graphs)

---

## Future Improvements

To strengthen your approach:

1. **Hybrid: TRM + Exact**
   - Use TRM for initial solution
   - Run local search / branch-and-bound to improve

2. **Hybrid: TRM + SDP**
   - Use TRM for fast approximation
   - Use SDP for theoretical bound

3. **Better Post-processing**
   - Current: Greedy (deterministic)
   - Could add: Local search, tabu search, genetic algorithms

4. **Theoretical Analysis**
   - Analyze: When does your approach work well?
   - When does it fail?
   - Characterize graph properties


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

### 4. **approx_ratio_greedy** (True Metric)
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
