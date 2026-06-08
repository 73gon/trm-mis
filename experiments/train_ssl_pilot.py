"""
Thin wrapper around train_mis_ssl.py that adds a --data_path override.

train_mis_ssl.py is NOT modified. We parse --data_path here, stash it on
cfg.data_paths, and call train_mis_ssl.main(cfg) directly.
"""

from __future__ import annotations

import argparse

import train_mis_ssl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override training data paths (semicolon-separated)")
    # Passthrough (superset of train_mis_ssl.py args)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--feasibility_weight", type=float, default=None)
    parser.add_argument("--selection_weight", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--feasibility_loss_type", type=str, default=None)
    parser.add_argument("--use_pe", type=int, default=None)
    parser.add_argument("--use_enhanced_features", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--entropy_weight", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temp_start", type=float, default=None)
    parser.add_argument("--temp_end", type=float, default=None)
    parser.add_argument("--use_deep_supervision", type=int, default=None)
    parser.add_argument("--use_loss_schedule", type=int, default=None)
    parser.add_argument("--fw_start", type=float, default=None)
    parser.add_argument("--sw_start", type=float, default=None)
    parser.add_argument("--n_supervision", type=int, default=None)
    parser.add_argument("--loss_mode", type=str, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    cfg = train_mis_ssl.Config()
    if args.data_path is not None:
        cfg.data_paths = [p for p in args.data_path.split(";") if p]
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.run_name is not None: cfg.run_name = args.run_name
    if args.use_pe is not None: cfg.use_pe = bool(args.use_pe)
    if args.use_enhanced_features is not None: cfg.use_enhanced_features = bool(args.use_enhanced_features)
    if args.feasibility_weight is not None: cfg.loss.feasibility_weight = args.feasibility_weight
    if args.selection_weight is not None: cfg.loss.selection_weight = args.selection_weight
    if args.mu is not None: cfg.loss.mu = args.mu
    if args.feasibility_loss_type is not None: cfg.loss.feasibility_loss_type = args.feasibility_loss_type
    if args.max_shards is not None: cfg.max_shards = args.max_shards
    if args.batch_size is not None: cfg.global_batch_size = args.batch_size
    if args.entropy_weight is not None: cfg.loss.entropy_weight = args.entropy_weight
    if args.temperature is not None: cfg.loss.temperature = args.temperature
    if args.temp_start is not None: cfg.loss.temp_start = args.temp_start
    if args.temp_end is not None: cfg.loss.temp_end = args.temp_end
    if args.use_deep_supervision is not None: cfg.loss.use_deep_supervision = bool(args.use_deep_supervision)
    if args.use_loss_schedule is not None: cfg.loss.use_loss_schedule = bool(args.use_loss_schedule)
    if args.fw_start is not None: cfg.loss.fw_start = args.fw_start
    if args.sw_start is not None: cfg.loss.sw_start = args.sw_start
    if args.n_supervision is not None: cfg.n_supervision = args.n_supervision
    if args.loss_mode is not None: cfg.loss.loss_mode = args.loss_mode
    if args.noise_scale is not None: cfg.loss.noise_scale = args.noise_scale
    if args.pretrained is not None: cfg.pretrained = args.pretrained
    if args.checkpoint_path is not None: cfg.checkpoint_path = args.checkpoint_path

    train_mis_ssl.main(cfg)


if __name__ == "__main__":
    main()
