import torch
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

# Imports from your project
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.graph_trm import GraphTRM


def greedy_decode(probs, edge_index, num_nodes):
    """
    Turns probabilities into a valid Independent Set using a greedy strategy.
    1. Sort nodes by probability (highest first).
    2. Pick the best node.
    3. Remove its neighbors.
    4. Repeat.

    Returns: (set_size, selected_nodes_set)
    """
    probs = probs.cpu().numpy()
    edge_index = edge_index.cpu().numpy()

    # Create adjacency list
    adj = {i: set() for i in range(num_nodes)}
    for u, v in zip(edge_index[0], edge_index[1]):
        adj[u].add(v)
        adj[v].add(u)

    # Sort nodes by probability (descending)
    sorted_nodes = np.argsort(-probs)

    selected_set = set()
    blocked_nodes = set()

    for node in sorted_nodes:
        if node in blocked_nodes:
            continue

        selected_set.add(node)
        blocked_nodes.add(node)

        for neighbor in adj[node]:
            blocked_nodes.add(neighbor)

    return len(selected_set), selected_set


def validate_independent_set(selected_set, edge_index):
    """Check that no two selected nodes are adjacent"""
    edge_index = edge_index.cpu().numpy()
    for u, v in zip(edge_index[0], edge_index[1]):
        if u in selected_set and v in selected_set:
            return False
    return True


def compute_metrics(preds_binary, labels, edge_index):
    """Compute all metrics matching training"""
    # Classification metrics
    tp = (preds_binary * labels).sum().float()
    fp = (preds_binary * (1 - labels)).sum().float()
    fn = ((1 - preds_binary) * labels).sum().float()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))

    # Set size metrics
    num_pred_1s = preds_binary.sum().item()
    num_true_1s = labels.sum().item()
    set_size_ratio = num_pred_1s / (num_true_1s + 1e-8)

    # Feasibility (before greedy decode)
    pred_mask = (preds_binary == 1)
    if pred_mask.sum() > 0 and edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        violations = (pred_mask[src] & pred_mask[dst]).sum().float()
        feasibility = (1.0 - (violations / (pred_mask.sum() + 1e-8)).clamp(max=1.0)).item()
    else:
        feasibility = 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pred_1s": num_pred_1s,
        "num_true_1s": num_true_1s,
        "set_size_ratio": set_size_ratio,
        "feasibility_raw": feasibility,  # Before greedy decode
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIS model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mis/epoch_99.pt",
                        help="Path to checkpoint")
    parser.add_argument("--data_path", type=str, default="data/test_mis",
                        help="Path to test data")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=18)
    parser.add_argument("--max_samples", type=int, default=999999,
                        help="Maximum samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation (increase for speed)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="MIS-TRM",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name (default: eval_<checkpoint>)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Initialize wandb if requested
    if args.wandb:
        run_name = args.wandb_run_name or f"eval_{os.path.basename(args.checkpoint)}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Check if test data exists
    test_shards = glob.glob(os.path.join(args.data_path, "mis_shard_*.pt"))

    if not test_shards:
        print(f"No test data found in {args.data_path}")
        print("Falling back to training data (data/mis-10k)")
        args.data_path = "data/mis-10k"

    # Load Data
    ds_config = MISDatasetConfig(
        dataset_paths=[args.data_path],
        global_batch_size=args.batch_size
    )

    try:
        dataset = MISDataset(ds_config)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Load Model
    model_config = {
        "input_dim": dataset.metadata.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "cycles": args.cycles
    }
    model = GraphTRM(model_config).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        available = sorted(glob.glob("checkpoints/mis/epoch_*.pt"))
        if available:
            print(f"Available: {available[-5:]}")
        return

    model.eval()

    # Metric accumulators
    all_metrics = {
        "pred_size_vs_opt": [],  # Raw prediction count vs optimal (NOT a valid approx ratio)
        "approx_ratio_greedy": [],  # After greedy decode (VALID approx ratio)
        "f1": [],
        "precision": [],
        "recall": [],
        "feasibility_raw": [],
        "feasibility_greedy": [],  # After greedy decode (should be 1.0)
        "set_size_ratio": [],
        "num_pred_1s": [],
        "num_true_1s": [],
        "valid_set_size": [],
        "optimal_size": [],
    }

    pbar = tqdm(dataloader, desc="Evaluating")
    sample_count = 0

    with torch.no_grad():
        for _, batch, _ in pbar:
            sample_count += 1
            if sample_count > args.max_samples:
                break

            # Move to device
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            # Inference
            carry = model.initial_carry(batch_dict)
            all_finish = False
            while not all_finish:
                carry, _, _, preds, all_finish = model(carry, batch_dict)

            # Get predictions
            final_probs = preds["preds"].squeeze()
            if final_probs.dim() == 0:
                final_probs = final_probs.unsqueeze(0)

            preds_binary = (final_probs > 0.5).float()
            labels = batch_dict["y"].float()
            edge_index = batch_dict["edge_index"]

            # Get optimal size
            if "opt_value" in batch_dict:
                opt_val = batch_dict["opt_value"]
                gt_size = opt_val.item() if opt_val.numel() == 1 else opt_val[0].item()
            else:
                gt_size = labels.sum().item()

            num_nodes = batch_dict["x"].size(0)

            # Compute raw metrics (before greedy decode)
            raw_metrics = compute_metrics(preds_binary, labels, edge_index)

            # Greedy decode for valid set
            greedy_size, selected_set = greedy_decode(final_probs, edge_index, num_nodes)

            # Validate greedy decode (should always be valid)
            is_valid = validate_independent_set(selected_set, edge_index)

            # Approximation ratios
            # pred_size_vs_opt: Raw prediction count vs optimal (HONEST - ignores feasibility)
            # Note: This is NOT a valid approximation ratio since invalid sets aren't solutions
            pred_size_vs_opt = preds_binary.sum().item() / (gt_size + 1e-8)
            # approx_ratio_greedy: After greedy decoding (valid set vs optimal)
            greedy_approx = greedy_size / (gt_size + 1e-8)

            # Store metrics
            all_metrics["pred_size_vs_opt"].append(pred_size_vs_opt)
            all_metrics["approx_ratio_greedy"].append(greedy_approx)
            all_metrics["f1"].append(raw_metrics["f1"])
            all_metrics["precision"].append(raw_metrics["precision"])
            all_metrics["recall"].append(raw_metrics["recall"])
            all_metrics["feasibility_raw"].append(raw_metrics["feasibility_raw"])
            all_metrics["feasibility_greedy"].append(1.0 if is_valid else 0.0)
            all_metrics["set_size_ratio"].append(raw_metrics["set_size_ratio"])
            all_metrics["num_pred_1s"].append(raw_metrics["num_pred_1s"])
            all_metrics["num_true_1s"].append(raw_metrics["num_true_1s"])
            all_metrics["valid_set_size"].append(greedy_size)
            all_metrics["optimal_size"].append(gt_size)

            # Update progress bar with running averages
            pbar.set_postfix({
                "approx_greedy": f"{np.mean(all_metrics['approx_ratio_greedy']):.4f}",
                "f1": f"{np.mean(all_metrics['f1']):.4f}",
                "feasibility": f"{np.mean(all_metrics['feasibility_raw']):.4f}",
            })

            # Log to wandb after each batch (creates smooth curves)
            if args.wandb:
                wandb.log({
                    "eval/approx_ratio_greedy": greedy_approx,
                    "eval/f1": raw_metrics["f1"],
                    "eval/precision": raw_metrics["precision"],
                    "eval/recall": raw_metrics["recall"],
                    "eval/feasibility_raw": raw_metrics["feasibility_raw"],
                    "eval/feasibility_greedy": 1.0 if is_valid else 0.0,
                    "eval/set_size_ratio": raw_metrics["set_size_ratio"],
                    "eval/sample_num": sample_count,
                })

    # Compute final statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    results = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            results[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    # Print key metrics
    print(f"\n📊 KEY METRICS (n={sample_count} samples):")
    print("-" * 50)

    key_metrics = ["approx_ratio_greedy", "pred_size_vs_opt", "f1", "precision", "recall",
                   "feasibility_raw", "feasibility_greedy", "set_size_ratio"]

    for key in key_metrics:
        if key in results:
            r = results[key]
            print(f"  {key:25s}: {r['mean']:.4f} ± {r['std']:.4f}  (min={r['min']:.4f}, max={r['max']:.4f})")

    print("\n📈 SIZE METRICS:")
    print("-" * 50)
    print(f"  Total Optimal Nodes:    {sum(all_metrics['optimal_size']):.0f}")
    print(f"  Total Predicted Nodes:  {sum(all_metrics['num_pred_1s']):.0f}")
    print(f"  Total Valid Set Size:   {sum(all_metrics['valid_set_size']):.0f}")
    print(f"  Overall Approx Ratio:   {sum(all_metrics['valid_set_size']) / sum(all_metrics['optimal_size']):.4f}")

    # Log final results to wandb
    if args.wandb:
        wandb_results = {}
        for key in key_metrics:
            if key in results:
                wandb_results[f"eval/{key}_mean"] = results[key]["mean"]
                wandb_results[f"eval/{key}_std"] = results[key]["std"]

        wandb_results["eval/total_optimal"] = sum(all_metrics["optimal_size"])
        wandb_results["eval/total_predicted"] = sum(all_metrics["num_pred_1s"])
        wandb_results["eval/total_valid"] = sum(all_metrics["valid_set_size"])
        wandb_results["eval/overall_approx_ratio"] = sum(all_metrics["valid_set_size"]) / sum(all_metrics["optimal_size"])
        wandb_results["eval/num_samples"] = sample_count

        wandb.log(wandb_results)

        # Create summary table
        wandb.summary.update(wandb_results)

        print(f"\n✅ Results logged to wandb: {args.wandb_project}")
        wandb.finish()

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
