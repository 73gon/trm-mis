"""
LP vs ILP Analysis Script for MIS Dataset

This script analyzes whether the generated MIS problems are "hard enough"
by comparing the LP relaxation solution with the true ILP (optimal) solution.

If LP == ILP for most graphs, the problems are "easy" (e.g., bipartite, trees)
and we need to generate harder graphs.

Usage:
    python dataset/analyze_lp_gap.py --data_path data/mis-10k --max_samples 500
"""

import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def compute_lp_relaxation(edge_index, num_nodes):
    """
    Compute LP relaxation of Maximum Independent Set.

    LP Relaxation:
        maximize: sum(x_i)
        subject to: x_i + x_j <= 1 for all edges (i,j)
                    0 <= x_i <= 1

    Returns: LP optimal value (sum of fractional variables)
    """
    if num_nodes == 0:
        return 0.0

    # Objective: maximize sum(x_i) = minimize -sum(x_i)
    c = -np.ones(num_nodes)

    # Constraints: x_i + x_j <= 1 for each edge
    edge_index_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    num_edges = edge_index_np.shape[1] // 2  # Each edge appears twice

    if num_edges == 0:
        # No edges = all nodes can be in MIS
        return float(num_nodes)

    # Build constraint matrix (only use each edge once)
    seen_edges = set()
    A_ub_rows = []
    A_ub_cols = []
    A_ub_data = []
    row_idx = 0

    for k in range(edge_index_np.shape[1]):
        i, j = edge_index_np[0, k], edge_index_np[1, k]
        if i > j:
            i, j = j, i
        edge = (i, j)
        if edge not in seen_edges:
            seen_edges.add(edge)
            A_ub_rows.extend([row_idx, row_idx])
            A_ub_cols.extend([i, j])
            A_ub_data.extend([1, 1])
            row_idx += 1

    if row_idx == 0:
        return float(num_nodes)

    A_ub = csr_matrix((A_ub_data, (A_ub_rows, A_ub_cols)), shape=(row_idx, num_nodes))
    b_ub = np.ones(row_idx)

    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1) for _ in range(num_nodes)]

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return -result.fun  # Negate because we minimized -sum
        else:
            return float(num_nodes)  # Fallback
    except Exception as e:
        print(f"LP solver error: {e}")
        return float(num_nodes)


def analyze_dataset(data_path, max_samples=500):
    """
    Analyze LP-ILP gap for MIS dataset.
    """
    # Find all shards
    shards = sorted(glob.glob(os.path.join(data_path, "mis_shard_*.pt")))
    if not shards:
        print(f"No shards found in {data_path}")
        return

    print(f"Found {len(shards)} shards in {data_path}")

    results = {
        "lp_values": [],
        "ilp_values": [],
        "gaps": [],
        "gap_ratios": [],
        "num_nodes": [],
        "num_edges": [],
        "densities": [],
    }

    sample_count = 0

    for shard_path in tqdm(shards, desc="Analyzing shards"):
        payload = torch.load(shard_path, weights_only=False)

        for sample in payload["data"]:
            if sample_count >= max_samples:
                break

            edge_index = sample["edge_index"]
            y = sample["y"]
            n = int(sample["n"])
            opt_value = sample.get("opt_value", y.sum().item())

            # Compute LP relaxation
            lp_value = compute_lp_relaxation(edge_index, n)

            # ILP value is the ground truth optimal
            ilp_value = float(opt_value)

            # Compute gap
            gap = lp_value - ilp_value
            gap_ratio = gap / (ilp_value + 1e-8)

            # Compute graph density
            num_edges = edge_index.shape[1] // 2
            max_edges = n * (n - 1) // 2
            density = num_edges / max(1, max_edges)

            results["lp_values"].append(lp_value)
            results["ilp_values"].append(ilp_value)
            results["gaps"].append(gap)
            results["gap_ratios"].append(gap_ratio)
            results["num_nodes"].append(n)
            results["num_edges"].append(num_edges)
            results["densities"].append(density)

            sample_count += 1

        if sample_count >= max_samples:
            break

    # Analysis
    print("\n" + "=" * 70)
    print("LP vs ILP ANALYSIS RESULTS")
    print("=" * 70)

    gaps = np.array(results["gaps"])
    gap_ratios = np.array(results["gap_ratios"])
    densities = np.array(results["densities"])
    lp_values = np.array(results["lp_values"])
    ilp_values = np.array(results["ilp_values"])

    print(f"\n📊 Dataset Statistics (n={sample_count} samples):")
    print(f"  Nodes: {np.mean(results['num_nodes']):.1f} ± {np.std(results['num_nodes']):.1f}")
    print(f"  Edges: {np.mean(results['num_edges']):.1f} ± {np.std(results['num_edges']):.1f}")
    print(f"  Density: {np.mean(densities):.3f} ± {np.std(densities):.3f}")

    print(f"\n📈 LP vs ILP Analysis:")
    print(f"  LP value (mean): {np.mean(lp_values):.2f} ± {np.std(lp_values):.2f}")
    print(f"  ILP value (mean): {np.mean(ilp_values):.2f} ± {np.std(ilp_values):.2f}")
    print(f"  Gap (mean): {np.mean(gaps):.2f} ± {np.std(gaps):.2f}")
    print(f"  Gap ratio (mean): {np.mean(gap_ratios):.2%} ± {np.std(gap_ratios):.2%}")

    # Count "easy" graphs (gap = 0 or very small)
    easy_threshold = 0.01
    easy_count = np.sum(gap_ratios < easy_threshold)
    easy_pct = easy_count / sample_count * 100

    print(f"\n🎯 Hardness Analysis:")
    print(f"  'Easy' graphs (gap ratio < {easy_threshold:.0%}): {easy_count}/{sample_count} ({easy_pct:.1f}%)")
    print(f"  'Medium' graphs (gap ratio 1-10%): {np.sum((gap_ratios >= 0.01) & (gap_ratios < 0.1))}/{sample_count}")
    print(f"  'Hard' graphs (gap ratio > 10%): {np.sum(gap_ratios >= 0.1)}/{sample_count}")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if easy_pct > 50:
        print(f"  ⚠️ WARNING: {easy_pct:.0f}% of graphs are 'easy' (LP = ILP)")
        print(f"  → Consider generating harder graphs (higher density, avoid bipartite)")
    elif easy_pct > 20:
        print(f"  ⚠️ CAUTION: {easy_pct:.0f}% of graphs are 'easy'")
        print(f"  → Dataset may be too easy for challenging the model")
    else:
        print(f"  ✅ GOOD: Only {easy_pct:.0f}% of graphs are 'easy'")
        print(f"  → Dataset difficulty is appropriate")

    # Correlation with density
    correlation = np.corrcoef(densities, gap_ratios)[0, 1]
    print(f"\n📉 Correlation (density vs gap ratio): {correlation:.3f}")
    if correlation > 0.3:
        print(f"  → Higher density correlates with larger LP-ILP gap (expected)")

    # Save results
    output_path = os.path.join(data_path, "lp_ilp_analysis.npz")
    np.savez(output_path, **results)
    print(f"\n💾 Results saved to: {output_path}")

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Gap distribution
        axes[0, 0].hist(gaps, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('LP - ILP Gap')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of LP-ILP Gap')
        axes[0, 0].axvline(x=0, color='r', linestyle='--', label='Easy (gap=0)')

        # 2. Gap ratio distribution
        axes[0, 1].hist(gap_ratios * 100, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Gap Ratio (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of LP-ILP Gap Ratio')

        # 3. Density vs Gap
        axes[1, 0].scatter(densities, gap_ratios, alpha=0.5, s=10)
        axes[1, 0].set_xlabel('Graph Density')
        axes[1, 0].set_ylabel('Gap Ratio')
        axes[1, 0].set_title(f'Density vs Gap Ratio (corr={correlation:.3f})')

        # 4. LP vs ILP
        axes[1, 1].scatter(ilp_values, lp_values, alpha=0.5, s=10)
        axes[1, 1].plot([0, max(ilp_values)], [0, max(ilp_values)], 'r--', label='LP=ILP')
        axes[1, 1].set_xlabel('ILP (Optimal) Value')
        axes[1, 1].set_ylabel('LP Relaxation Value')
        axes[1, 1].set_title('LP vs ILP Values')
        axes[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(data_path, "lp_ilp_analysis.png")
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create plot: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze LP vs ILP gap for MIS dataset")
    parser.add_argument("--data_path", type=str, default="data/mis-10k",
                        help="Path to MIS dataset")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum samples to analyze")
    args = parser.parse_args()

    analyze_dataset(args.data_path, args.max_samples)


if __name__ == "__main__":
    main()
