# Failed & Deferred Experiments

This document tracks techniques that were tried and dropped, plus ones deferred for later rounds. Each entry includes hypothesis, configuration, observed behavior, analysis, and lesson learned.

## Deferred to a later round

### Curriculum learning (smaller → larger graphs)
- **Hypothesis**: Training order matters; starting on small graphs accelerates early convergence and reduces the plateau at epoch ~10.
- **Why deferred**: Requires a dataset refactor (per-shard graph-size sort + epoch-dependent shard subsetting). Dropping into the first pilot round would have delayed everything by ≥1 day. Will return in round 2 if the round-1 winners still plateau.
- **Plan if revisited**: wrap `MISDataset` with a `CurriculumDatasetWrapper` that exposes `set_epoch(epoch)` and restricts sampling to the smallest `30% + epoch·r%` of graphs.

## Abandoned techniques (will be filled in after pilots complete)

_Each finalized entry will follow the template below._

### Template
- **Hypothesis**: what we thought the change would do.
- **Config**: CLI/diff that produced the run.
- **Wandb run**: link.
- **Observed behavior**: loss curve trajectory, key raw metrics, final decoded AR.
- **Why it failed**: concrete analysis.
- **Lesson learned**: actionable takeaway.

_Entries will be appended once 20-epoch pilot results are in._
