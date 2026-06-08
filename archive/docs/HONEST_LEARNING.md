# Honest Learning — The Over-Prediction Problem

## The trap

On paper our models hit 0.99+ approximation ratios. But a closer look at the **raw** model output (before any post-processing) revealed that:

- Raw `pred_size` was close to `num_nodes` (the model labeled almost every node as "in the set").
- Raw feasibility was ≈ 0 (adjacent nodes are happily both selected).
- Only after `greedy_decode` converted the probability vector into an independent set did the approximation ratio shoot up.

This looks alarmingly like the model "cheated" by predicting all-ones and letting the greedy algorithm do the real work. Two root causes were separately responsible for what we observed; separating them was essential.

## Root cause 1 — a measurement bug

[improvements/eval_multisample.py](improvements/eval_multisample.py) used to compute:

```python
logits = preds["preds"].squeeze()
probs  = torch.sigmoid(logits / temp)
```

But `preds["preds"]` is **already** post-sigmoid (see [models/graph_transformer_trm.py](models/graph_transformer_trm.py#L311)). Applying sigmoid a second time squashes every value into `[0.5, 0.7311]`, guaranteeing every node crosses the 0.5 threshold → raw_pred_size = num_nodes, raw_feasibility = 0.

**Fix**: treat the output as probs and apply temperature through the logit space: `sigmoid(logit(probs) / temp)`. Greedy decoding was never affected because it uses the *ordering* of probabilities, not their absolute values — that is why our headline AR numbers were fine even though the raw metrics were garbage.

## Root cause 2 — a real learning problem

Training-time metrics (which compute the confusion matrix from probs directly, with no second sigmoid) also showed the model over-predicting — pred_size/opt_size ≈ 2 on SATLIB. So even without the measurement bug there is real over-prediction: the model relies on greedy to trim the fat.

This is why all experiments in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) track two honest-learning metrics alongside AR:

1. **Raw feasibility** — fraction of edges whose endpoints are not *both* predicted.
2. **pred_size / opt_size** — closer to 1.0 is better; > 1.5 means the decoder is carrying the model.

## Targets for "honest learning"

A model only counts as actually solving MIS when, without any post-processing:

| Metric | Target |
|--------|--------|
| Raw feasibility | > 0.95 |
| pred_size / opt_size | in [0.95, 1.10] |

These gates are enforced in the final-run acceptance checklist. Even if a model hits > 0.999 AR, if it cannot satisfy both gates we treat it as a greedy-rescue pipeline, not a learned solver.
