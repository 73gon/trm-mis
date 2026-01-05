# Task Comparison: Sudoku vs ARC vs MIS

## Overview

The TRM model is task-agnostic, but different tasks require different post-processing strategies:

| Task | Type | Constraints | Post-processing | Approximation? |
|------|------|-------------|-----------------|----------------|
| **Sudoku** | CSP | In problem | ❌ None | ❌ No (binary) |
| **ARC** | Pattern Recognition | Implicit | ❌ None/Light | ❌ No (binary) |
| **MIS** | Graph Optimization | In structure | ✅ **Greedy** | ✅ Yes |

---

## Sudoku

### Problem Structure
- **Input:** Partially filled 9×9 grid (digits 1-9)
- **Constraint:** No repeats in rows, columns, 3×3 boxes
- **Output:** Fully filled valid grid

### How TRM Solves It

```
Cycle 1:  Random guesses for empty cells
          [_, 5, _] → [3, 5, 7]

Cycle 2:  Refine based on constraints
          Check: Does 3 appear elsewhere in row? Fix if needed

...

Cycle 18: Polished solution
          [_, 5, _] → [1, 5, 9]  (Constraint-aware)
```

### Key Insight
**Constraints are naturally encoded in the problem space:**
- Each cell independently valid (value 1-9)
- Constraints checked at inference time
- Model learns to satisfy constraints implicitly

### Output Validation
```python
model_output = [1, 5, 3, 9, 2, ...]  # Raw model output
# Is this valid?
# - Check rows: unique 1-9 ✓
# - Check cols: unique 1-9 ✓
# - Check boxes: unique 1-9 ✓
# Valid! ✓

# No post-processing needed
final_output = model_output
```

### Post-processing: ❌ None Needed
- Model output is already valid (or can be validated)
- Could optionally check correctness, but not needed for validity

---

## ARC (Abstraction and Reasoning Corpus)

### Problem Structure
- **Input:** Pattern in source grid
- **Constraint:** Pattern/rule is implicit
- **Output:** Apply pattern to larger grid

### How TRM Solves It

```
Cycle 1:  Recognize rough pattern
          "Color seems to rotate?"

Cycle 2:  Refine understanding
          "Pattern repeats every 3 steps"

...

Cycle 18: Apply pattern correctly
          Output grid matches pattern from input
```

### Key Insight
**Constraints are implicit in the learned pattern:**
- Model learns to recognize and reproduce patterns
- Each output color is individually valid
- Consistency emerges from learned pattern

### Output Validation
```python
model_output = [RED, BLUE, RED, BLUE, ...]  # Raw colors
# Is this valid?
# - Each cell has valid color ✓
# - Follows learned pattern (if learned correctly) ✓

# No post-processing needed
final_output = model_output
```

### Post-processing: ❌ None Needed
- Colors are all valid by themselves
- Could optionally validate pattern consistency
- But model output is already complete

---

## MIS (Maximum Independent Set)

### Problem Structure
- **Input:** Graph (nodes + edges)
- **Constraint:** No selected nodes can be adjacent (IN GRAPH STRUCTURE)
- **Output:** Set of non-adjacent nodes

### How TRM Solves It

```
Cycle 1:  Random node scores
          [0.2, 0.3, 0.1, 0.4, ...]

Cycle 2:  Refine scores
          [0.4, 0.5, 0.2, 0.6, ...]

...

Cycle 18: Final scores
          [0.9, 0.8, 0.3, 0.7, ...]
          "Nodes 0, 1 should be in MIS"

Problem: Nodes 0 and 1 are adjacent! ⚠️
         Model outputs don't satisfy constraint
```

### Key Insight
**⚠️ Constraints are NOT in the prediction space:**
- Model outputs probabilities
- Graph structure is external
- No guarantee probabilities satisfy adjacency constraints
- **POST-PROCESSING REQUIRED**

### Output Validation (Raw Model)
```python
model_output = [0.9, 0.8, 0.3, 0.7, 0.2, ...]
edges = [[0,1], [1,2], [2,3]]

# Is this valid?
# Check: Nodes 0 (0.9) and 1 (0.8) adjacent?
# Yes! → NOT valid ❌
# Check: Nodes 1 (0.8) and 2 (0.3) adjacent?
# Yes! → NOT valid ❌

# Raw output violates constraints!
```

### Post-processing: ✅ REQUIRED (Greedy Decode)
```python
# Step 1: Sort by confidence
sorted_nodes = [0, 1, 3, 2, 4]  # by probability
              [0.9, 0.8, 0.7, 0.3, 0.2]

# Step 2: Greedy select with constraint checking
selected = []
for node in sorted_nodes:
    if not adjacent_to_any(node, selected, edges):
        selected.append(node)

# Result
greedy_output = [1, 0, 0, 1, 0]  # Valid! ✓

# Feasibility = 1.0 (guaranteed by greedy)
# approx_ratio = len(selected) / optimal_size
```

---

## Why Different Post-processing?

### Sudoku/ARC: Constraints in Prediction Space
```
Cell value ∈ {1,2,...,9}
Each value independently valid
Constraints are about RELATIONSHIPS between cells
Model learns relationships implicitly

Example: Model predicts "5" for a cell
Is "5" valid by itself? YES ✓
Does it violate row constraint? Maybe, but model learned this
```

### MIS: Constraints NOT in Prediction Space
```
Node selection ∈ [0, 1] (probability)
Each probability independently valid
Constraints are about GRAPH STRUCTURE
Model doesn't "know" about adjacencies

Example: Model predicts 0.9 for node 0
Is 0.9 valid by itself? YES ✓
Does it violate edge constraint? Only if we select it AND adjacent node selected
```

---

## Summary Table

| Aspect | Sudoku | ARC | MIS |
|--------|--------|-----|-----|
| **Constraint Type** | Structural (CSP) | Implicit (Pattern) | Structural (Graph) |
| **Constraint Location** | In problem definition | Learned by model | External (edges) |
| **Model Output Valid?** | Usually yes | Usually yes | ❌ Not guaranteed |
| **Post-processing** | None | None | ✅ Greedy Decode |
| **Why Different?** | Constraints in cell values | Constraints in pattern | Constraints in edges |
| **Approximation?** | No (binary) | No (binary) | Yes (optimization) |

---

## Conclusion

Your MIS approach correctly identifies that **graph constraints are external** and requires post-processing. This is fundamentally different from Sudoku/ARC where constraints are naturally satisfied by the model's output structure.
