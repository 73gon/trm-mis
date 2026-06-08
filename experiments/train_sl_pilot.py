"""
Pilot launcher for SL experiments with extra loss terms.

This script is a thin wrapper around train_mis.py that:
1. Monkey-patches GraphTransformerTRM with GraphTransformerTRM_Pilot.
2. Exposes three extra CLI flags (cardinality_weight, extra_entropy_weight,
   focal_gamma) that map to the new opt-in loss terms.
3. Hands every other flag back to train_mis.py unchanged.

train_mis.py itself is NOT modified.
"""

from __future__ import annotations

import argparse
import os
import sys

# Patch before importing train_mis (train_mis imports GraphTransformerTRM at top)
import models.graph_transformer_trm as _gtrm_mod
from experiments.pilot_model import GraphTransformerTRM_Pilot

_gtrm_mod.GraphTransformerTRM = GraphTransformerTRM_Pilot  # type: ignore[misc]

import train_mis  # noqa: E402  (imports the now-patched class name)

# Ensure train_mis's own reference is patched too
train_mis.GraphTransformerTRM = GraphTransformerTRM_Pilot  # type: ignore[attr-defined]


def _build_config_from_args() -> "train_mis.Config":
    parser = argparse.ArgumentParser()
    # Pilot-specific
    parser.add_argument("--cardinality_weight", type=float, default=0.0)
    parser.add_argument("--extra_entropy_weight", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--cosine_temp_start", type=float, default=None,
                        help="Unused in SL pilot (placeholder for symmetry)")
    # Passthrough (mirrors train_mis.py CLI)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--feasibility_weight", type=float, default=None)
    parser.add_argument("--feasibility_loss_type", type=str, default=None)
    parser.add_argument("--use_pe", type=int, default=None)
    parser.add_argument("--use_enhanced_features", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--selection_weight", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--n_supervision", type=int, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override train data paths (semicolon-separated)")
    args = parser.parse_args()

    cfg = train_mis.Config()

    # Store pilot loss weights in config for the model to pick up.
    # We stash them on loss.__dict__ via arch config (passed as model_config_dict).
    # Simpler: train_mis builds model_config_dict from cfg.arch + loss fields.
    # We already pass these via an injected env variable — see monkey-patched
    # initializer below.
    os.environ["PILOT_CARDINALITY_WEIGHT"] = str(args.cardinality_weight)
    os.environ["PILOT_EXTRA_ENTROPY_WEIGHT"] = str(args.extra_entropy_weight)
    os.environ["PILOT_FOCAL_GAMMA"] = str(args.focal_gamma)
    os.environ["PILOT_FOCAL_ALPHA"] = str(args.focal_alpha)

    if args.epochs is not None: cfg.epochs = args.epochs
    if args.run_name is not None: cfg.run_name = args.run_name
    if args.use_pe is not None: cfg.use_pe = bool(args.use_pe)
    if args.use_enhanced_features is not None: cfg.use_enhanced_features = bool(args.use_enhanced_features)
    if args.feasibility_weight is not None: cfg.loss.feasibility_weight = args.feasibility_weight
    if args.feasibility_loss_type is not None: cfg.loss.feasibility_loss_type = args.feasibility_loss_type
    if args.max_shards is not None: cfg.max_shards = args.max_shards
    if args.batch_size is not None: cfg.global_batch_size = args.batch_size
    if args.pos_weight is not None: cfg.loss.pos_weight = args.pos_weight
    if args.selection_weight is not None: cfg.loss.selection_weight = args.selection_weight
    if args.label_smoothing is not None: cfg.loss.label_smoothing = args.label_smoothing
    if args.n_supervision is not None: cfg.n_supervision = args.n_supervision
    if args.pretrained is not None: cfg.pretrained = args.pretrained
    if args.checkpoint_path is not None: cfg.checkpoint_path = args.checkpoint_path
    if args.data_path is not None:
        cfg.data_paths = [p for p in args.data_path.split(";") if p]
    return cfg


# Patch the GraphTransformerTRM_Pilot.__init__ to read extra weights from env
_orig_pilot_init = GraphTransformerTRM_Pilot.__init__


def _patched_init(self, config):  # type: ignore[override]
    config["cardinality_weight"] = float(os.environ.get("PILOT_CARDINALITY_WEIGHT", 0.0))
    config["extra_entropy_weight"] = float(os.environ.get("PILOT_EXTRA_ENTROPY_WEIGHT", 0.0))
    config["focal_gamma"] = float(os.environ.get("PILOT_FOCAL_GAMMA", 0.0))
    config["focal_alpha"] = float(os.environ.get("PILOT_FOCAL_ALPHA", 0.25))
    _orig_pilot_init(self, config)


GraphTransformerTRM_Pilot.__init__ = _patched_init  # type: ignore[method-assign]


def main():
    cfg = _build_config_from_args()
    train_mis.main(cfg)


if __name__ == "__main__":
    main()
