# Evaluation Metrics Documentation

This document explains all metrics used in evaluation (`eval_mis.py`).

## How to Run Evaluation

```bash
# Evaluate on test set with wandb logging
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt --data_path data/test_mis --wandb

# Evaluate on training set (quick sanity check)
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt --data_path data/mis-10k --wandb

# With custom batch size (increase for speed)
python eval_mis.py --checkpoint checkpoints/mis/epoch_99.pt --batch_size 256 --wandb
```

## Wandb Logging

All metrics are logged **per-batch** to wandb, creating smooth curves showing performance across all samples. You should see many datapoints, not just one final value.

Metrics logged to wandb:
- `eval/approx_ratio_greedy` - Main MIS metric after greedy decode (target: ~1.0)
- `eval/f1` - Classification quality (0 to 1, target: ~0.8+)
- `eval/precision` - % of predictions that are correct (0 to 1)
- `eval/recall` - % of true nodes found (0 to 1)
- `eval/feasibility_raw` - Independence before greedy decode (0 to 1)
- `eval/feasibility_greedy` - Independence after greedy decode (should be 1.0)
- `eval/set_size_ratio` - Predicted size vs true size (target: ~1.0)
- `eval/sample_num` - Sample counter (for x-axis)

Plus final summary statistics logged at the end.

---

## Key Metrics

### `approx_ratio_greedy` ⭐ **PRIMARY METRIC**

**What it measures:** How close your solution is to the optimal MIS **after greedy decoding**.

**Formula:** `greedy_set_size / optimal_size`

**Range:** 0 to ∞ (typically 0.7-1.0)

**Interpretation:**
- **1.0** = Found optimal solution (perfect!)
- **0.95** = Found 95% of optimal (excellent)
- **0.85** = Found 85% of optimal (good)
- **0.75** = Found 75% of optimal (acceptable)
- **< 0.7** = Poor, needs improvement

**Why greedy decode?** Your raw predictions might violate the independence constraint. Greedy decoding fixes violations and produces a **valid** MIS. This is the honest evaluation metric for a combinatorial optimization problem.

---

### `pred_size_vs_opt` (Raw Prediction Size Ratio)

**What it measures:** Raw prediction count vs optimal size (ignoring feasibility).

**Formula:** `num_pred_1s / optimal_size`

**Range:** 0 to ∞

**⚠️ CAUTION:** This is NOT a valid approximation ratio! An invalid set (with adjacent nodes) cannot be a solution to MIS. This metric is useful for understanding model behavior:

- **> 1.0** = Model predicts more nodes than optimal (greedy behavior)
- **< 1.0** = Model predicts fewer nodes than optimal (conservative)

Compare with `approx_ratio_greedy` to see how much is "lost" to feasibility violations.

---

### `f1` (F1 Score)

**What it measures:** Balance between precision and recall for node classification.

**Formula:** `2 * (precision * recall) / (precision + recall)`

**Range:** 0 to 1

**Interpretation:**
- **0.8+** = Good
- **0.7-0.8** = Acceptable
- **< 0.7** = Needs improvement

**Why it matters:** Even if `approx_ratio_greedy` is good, F1 tells you if individual node predictions are correct.

---

### `precision`

**What it measures:** Of nodes you predicted as MIS, what % are actually in the true MIS?

**Formula:** `TP / (TP + FP)`

**Range:** 0 to 1

**Interpretation:**
- **High precision, low recall** = Too conservative, only selects nodes you're very sure about
- **Low precision, high recall** = Too greedy, selecting too many wrong nodes

---

### `recall`

**What it measures:** Of all nodes in the true MIS, what % did you find?

**Formula:** `TP / (TP + FN)`

**Range:** 0 to 1

**Interpretation:**
- **High recall, low precision** = Finding most true nodes but with many false positives
- **Low recall** = Missing true MIS nodes (too conservative)

---

### `feasibility_raw`

**What it measures:** Fraction of predicted nodes that form a valid independent set (no adjacent nodes both selected).

**Formula:** `1 - (edge_violations / num_predicted_nodes)`

**Range:** 0 to 1

**Interpretation:**
- **1.0** = All predictions respect independence constraint
- **0.9** = 90% of predictions are valid, 10% violate constraint
- **< 0.8** = Many constraint violations, model needs better feasibility training

**Note:** This is measured BEFORE greedy decoding. This is the HONEST metric of model quality.

---

### `feasibility_greedy`

**What it measures:** Whether the greedy-decoded set is a valid independent set.

**Range:** 0 or 1

**Expected:** Always 1.0 (greedy decoding guarantees validity)

**If < 1.0:** Bug in greedy_decode function.

---

### `set_size_ratio`

**What it measures:** How many times larger/smaller is your prediction vs true MIS size?

**Formula:** `num_predicted_1s / num_true_1s`

**Range:** 0 to ∞ (typically 0.5-2.0)

**Interpretation:**
- **1.0** = Perfect size match
- **0.8** = Predicting 80% of true size (too conservative)
- **1.2** = Predicting 120% of true size (too greedy)

---

## Train vs Eval Comparison

To check for overfitting, compare these metrics on **train** vs **test** data:

| Metric | Train | Test | Interpretation |
|--------|-------|------|----------------|
| approx_ratio_greedy | 0.85 | 0.83 | ✅ Similar = Good generalization |
| approx_ratio_greedy | 0.85 | 0.75 | ⚠️ Drop = Some overfitting |
| approx_ratio_greedy | 0.85 | 0.60 | ❌ Large drop = Severe overfitting |
| f1 | 0.78 | 0.75 | ✅ Similar = OK |
| feasibility_raw | 0.88 | 0.80 | ⚠️ Slightly worse on test |

**Rule of thumb:** Train and test metrics should be within 5% of each other.

---

## Understanding the Difference: Raw vs Greedy Metrics

| Scenario | `pred_size_vs_opt` | `feasibility_raw` | `approx_ratio_greedy` | Interpretation |
|----------|-------------------|-------------------|----------------------|----------------|
| Perfect | 1.0 | 1.0 | 1.0 | Model predicts optimal valid set |
| Good | 1.0 | 0.95 | 0.92 | Small loss from fixing violations |
| Greedy | 1.5 | 0.6 | 0.75 | Model over-predicts, loses a lot to violations |
| Conservative | 0.7 | 1.0 | 0.7 | Model under-predicts but all valid |

**Key insight:** If `pred_size_vs_opt > 1.0` but `approx_ratio_greedy < 1.0`, the model is wasting predictions on infeasible nodes.

---

## Interpreting Results

### Good Results (Target Performance)
```
approx_ratio_greedy: 0.95 ± 0.05 (min=0.80, max=1.00)
pred_size_vs_opt:    1.0 ± 0.1
f1:                  0.80 ± 0.10
feasibility_raw:     0.95 ± 0.05
set_size_ratio:      1.0 ± 0.1
```
✅ Model is working well!

### Overly Greedy (Selecting Too Many Nodes)
```
approx_ratio_greedy: 0.70 ± 0.15 (min=0.40, max=1.05)
pred_size_vs_opt:    1.8 ± 0.5    (way > 1.0!)
feasibility_raw:     0.50 ± 0.3   (too low)
```
⚠️ Increase `sparsity_weight` in training, or increase `feasibility_weight`

### Too Conservative (Selecting Too Few Nodes)
```
approx_ratio_greedy: 0.60 ± 0.15
pred_size_vs_opt:    0.6 ± 0.2    (< 1.0)
recall:              0.50 ± 0.2   (should be > 0.8)
```
⚠️ Decrease `feasibility_weight` in training, or increase learning rate

### Poor Feasibility
```
feasibility_raw:     0.70 ± 0.1   (should be > 0.9)
pred_size_vs_opt:    1.2          (> optimal)
approx_ratio_greedy: 0.75 ± 0.05  (losses from fixing)
```
⚠️ Increase `feasibility_weight` from 1.0 to 2.0-3.0 in training

---

## Wandb Charts to Monitor

Create these charts in wandb for easy monitoring:

1. **Main Performance Chart**
   - X: eval/sample_num
   - Y: eval/approx_ratio_greedy
   - Should trend toward 1.0

2. **Raw vs Greedy Comparison**
   - X: eval/sample_num
   - Y1: eval/pred_size_vs_opt (raw)
   - Y2: eval/approx_ratio_greedy (after greedy)
   - Gap shows loss from feasibility violations

3. **Feasibility Trajectory**
   - X: eval/sample_num
   - Y: eval/feasibility_raw
   - Should be high (> 0.9)

4. **F1 Trajectory**
   - X: eval/sample_num
   - Y: eval/f1
   - Should trend toward 0.8+

---

## Troubleshooting

| Issue | Check | Solution |
|-------|-------|----------|
| Only 1 datapoint in wandb charts | Are you using `--wandb` flag? | Re-run with `python eval_mis.py --checkpoint ... --wandb` |
| Flat lines (no improvement) | Check raw eval output in terminal | Model may have bad checkpoint or test data missing |
| Test metrics much worse than train | Overfitting | Train for fewer epochs, increase regularization |
| Very noisy eval curves | Check batch_size | Increase to 64-256 for smoother curves |
| `approx_ratio_greedy` much lower than `pred_size_vs_opt` | Many feasibility violations | Increase `feasibility_weight` during training |

---

## Key Changes from Previous Version

1. **Renamed `approx_ratio` → `pred_size_vs_opt`** - Clearer that this is raw prediction size, not a valid approximation ratio.
2. **Added explanation of raw vs greedy metrics** - Understanding the difference is crucial for debugging.
3. **Updated tables and examples** - Reflect actual metric behavior.
