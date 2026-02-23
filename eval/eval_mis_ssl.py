"""
Evaluation Script for Self-Supervised MIS Models

This script evaluates a self-supervised trained model against
ground truth MIS labels to compute:
- Approximation ratio (pred_size / opt_size)
- Gap (opt_size - pred_size)
- Feasibility

Works with both:
1. SSL models trained with train_mis_ssl.py
2. Supervised models trained with train_mis.py

Usage:
    python eval_mis_ssl.py --checkpoint checkpoints/mis_ssl/epoch_499.pt \
                           --data_path data/test_mis \
                           --model_type ssl
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.graph_transformer_trm_ssl import GraphTransformerTRM_SSL
from models.metrics_ssl import compute_metrics_ssl, compute_pp_metrics_ssl


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL MIS Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/test_mis", help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--n_supervision", type=int, default=1, help="Number of supervision steps")

    # Model architecture (should match training config)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--H_cycles", type=int, default=2)

    args = parser.parse_args()

    print("=" * 70)
    print("MIS SSL Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_path}")

    # Load dataset
    ds_config = MISDatasetConfig(
        dataset_paths=[args.data_path],
        global_batch_size=args.batch_size,
        rank=0,
        num_replicas=1,
        drop_last=False,
        val_split=0.0,  # Use all data for evaluation
        seed=42,
    )
    dataset = MISDataset(ds_config, split="train")  # split doesn't matter with val_split=0

    print(f"\nDataset: {dataset.num_graphs} graphs")
    print(f"Features: input_dim={dataset.metadata.input_dim}, pe_dim={dataset.metadata.pe_dim}")

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Build model config
    model_config = {
        "input_dim": dataset.metadata.input_dim,
        "pe_input_dim": dataset.metadata.pe_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "L_cycles": args.L_cycles,
        "H_cycles": args.H_cycles,
        "mu": 5.0,  # Default, not used in eval
        "feasibility_weight": 1.0,
        "selection_weight": 1.0,
    }

    # Load model
    model = GraphTransformerTRM_SSL(model_config).cuda()
    state_dict = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\nModel loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Evaluation
    all_metrics = {
        "pred_size": 0,
        "opt_size": 0,
        "gap": 0,
        "approx_ratio": 0,
        "feasibility": 0,
        "num_violations": 0,
        "pp_pred_size": 0,
        "pp_gap": 0,
        "pp_approx_ratio": 0,
        "pp_feasibility": 0,
    }
    count = 0

    with torch.no_grad():
        for batch_name, batch, batch_size in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            carry = model.initial_carry(batch)

            for _ in range(args.n_supervision):
                carry, _, _, preds, _ = model(carry, batch)

            probs = preds["preds"].squeeze()
            labels = batch["y"].float()

            # Raw metrics (vs ground truth)
            raw_metrics = compute_metrics_ssl(
                probs,
                batch["edge_index"],
                labels,
                batch_vec=batch["batch"],
                ptr=batch["ptr"],
            )

            # PP metrics (vs ground truth)
            pp_metrics = compute_pp_metrics_ssl(
                probs,
                batch["edge_index"],
                labels,
                batch_vec=batch["batch"],
                ptr=batch["ptr"],
            )

            for k, v in raw_metrics.items():
                all_metrics[k] += v
            for k, v in pp_metrics.items():
                all_metrics[k] += v

            count += 1

    # Average
    n = max(count, 1)
    for k in all_metrics:
        all_metrics[k] /= n

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n📊 Raw Model Predictions:")
    print(f"  Opt Size (avg):     {all_metrics['opt_size']:.2f}")
    print(f"  Pred Size (avg):    {all_metrics['pred_size']:.2f}")
    print(f"  Gap:                {all_metrics['gap']:.2f}")
    print(f"  Approx Ratio:       {all_metrics['approx_ratio']:.4f}")
    print(f"  Feasibility:        {all_metrics['feasibility']:.4f}")

    print("\n📊 Post-Processed (Greedy Decode):")
    print(f"  PP Pred Size (avg): {all_metrics['pp_pred_size']:.2f}")
    print(f"  PP Gap:             {all_metrics['pp_gap']:.2f}")
    print(f"  PP Approx Ratio:    {all_metrics['pp_approx_ratio']:.4f}")
    print(f"  PP Feasibility:     {all_metrics['pp_feasibility']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
