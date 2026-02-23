"""
Self-Supervised Metrics for MIS

These metrics are designed for self-supervised learning.
The loss doesn't use ground truth labels, but we compute comparison metrics
using labels for visualization/evaluation purposes.

Metrics computed:
- pred_size: Number of selected nodes (what we want to maximize)
- feasibility: Whether the solution is valid (no adjacent nodes selected)
- opt_size: Ground truth MIS size (FOR COMPARISON ONLY)
- gap: opt_size - pred_size (FOR COMPARISON ONLY)
- approx_ratio: pred_size / opt_size (FOR COMPARISON ONLY)

Post-processed metrics (pp_*) use greedy decode to ensure feasibility.
"""

import torch

from models.pp import greedy_decode


def compute_metrics_ssl(probs, edge_index, labels, batch_vec=None, ptr=None):
    """
    Compute self-supervised model metrics.

    Labels are used ONLY for comparison metrics (opt_size, gap, approx_ratio),
    NOT for training loss.

    Returns dict with:
    - pred_size: Average predicted set size per graph (thresholded at 0.5)
    - feasibility: 1 - (violations / total_edges)
    - opt_size: Ground truth MIS size (for comparison)
    - gap: opt_size - pred_size (for comparison)
    - approx_ratio: pred_size / opt_size (for comparison)
    """
    # If batch info not provided, fall back to single-graph metrics
    if batch_vec is None or ptr is None:
        preds_binary = (probs > 0.5).float()
        pred_size = preds_binary.sum().item()
        opt_size = labels.sum().item()

        # Compute violations
        src, dst = edge_index[0], edge_index[1]
        pred_mask = preds_binary == 1
        total_edges = edge_index.size(1)

        if pred_mask.sum() > 0 and total_edges > 0:
            violations = (pred_mask[src] & pred_mask[dst]).sum().float().item()
        else:
            violations = 0.0

        feasibility = 1.0 - (violations / max(total_edges, 1))
        gap = opt_size - pred_size
        approx_ratio = pred_size / (opt_size + 1e-8)

        return {
            "pred_size": pred_size,
            "feasibility": feasibility,
            "opt_size": opt_size,
            "gap": gap,
            "approx_ratio": approx_ratio,
        }

    # Compute per-graph metrics
    num_graphs = len(ptr) - 1
    total_pred = 0
    total_opt = 0
    total_gap = 0
    total_approx_ratio = 0
    total_violations = 0
    total_edges = 0

    for g in range(num_graphs):
        # Extract subgraph data
        node_mask = batch_vec == g
        graph_probs = probs[node_mask]
        graph_labels = labels[node_mask]

        # Get edges for this graph
        edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
        graph_edge_index = edge_index[:, edge_mask]

        # Remap edge indices to local node indices
        node_indices = torch.where(node_mask)[0]
        if len(node_indices) > 0:
            local_idx_map = torch.zeros(batch_vec.size(0), dtype=torch.long, device=probs.device)
            local_idx_map[node_indices] = torch.arange(len(node_indices), device=probs.device)
            graph_edge_index = local_idx_map[graph_edge_index]

        # Compute metrics for this graph
        graph_preds_binary = (graph_probs > 0.5).float()
        graph_pred_size = graph_preds_binary.sum().item()
        graph_opt_size = graph_labels.sum().item()

        # Compute violations
        src, dst = graph_edge_index[0], graph_edge_index[1]
        pred_mask = graph_preds_binary == 1
        graph_num_edges = graph_edge_index.size(1)
        total_edges += graph_num_edges

        if pred_mask.sum() > 0 and graph_num_edges > 0:
            graph_violations = (pred_mask[src] & pred_mask[dst]).sum().float().item()
        else:
            graph_violations = 0.0
        total_violations += graph_violations

        total_pred += graph_pred_size
        total_opt += graph_opt_size

        if graph_opt_size > 0:
            total_gap += graph_opt_size - graph_pred_size
            total_approx_ratio += graph_pred_size / graph_opt_size

    # Average over graphs
    avg_pred = total_pred / max(num_graphs, 1)
    avg_opt = total_opt / max(num_graphs, 1)
    avg_gap = total_gap / max(num_graphs, 1)
    avg_approx_ratio = total_approx_ratio / max(num_graphs, 1)

    # Compute feasibility
    if total_edges > 0:
        feasibility = 1.0 - (total_violations / total_edges)
    else:
        feasibility = 1.0

    return {
        "pred_size": avg_pred,
        "feasibility": feasibility,
        "opt_size": avg_opt,
        "gap": avg_gap,
        "approx_ratio": avg_approx_ratio,
    }


def compute_pp_metrics_ssl(probs, edge_index, labels, batch_vec=None, ptr=None):
    """
    Compute post-processed metrics using greedy decode.

    Labels are used ONLY for comparison metrics (pp_gap, pp_approx_ratio),
    NOT for training loss.

    GUARANTEES (by construction of greedy decode):
    - pp_feasibility == 1.0 (no violations possible)

    Returns dict with:
    - pp_pred_size: Average greedy-decoded set size per graph
    - pp_feasibility: 1 - (violations / edges) (ALWAYS 1.0)
    - pp_gap: opt_size - pp_pred_size (for comparison)
    - pp_approx_ratio: pp_pred_size / opt_size (for comparison)
    """
    # If batch info not provided, fall back to single-graph metrics
    if batch_vec is None or ptr is None:
        num_nodes = probs.size(0)
        opt_size = labels.sum().item()

        # Run greedy decode
        pp_pred_size, greedy_decoded = greedy_decode(probs, edge_index, num_nodes)

        # Verify feasibility of greedy-decoded solution
        src, dst = edge_index[0], edge_index[1]
        greedy_mask = greedy_decoded == 1.0
        total_edges = edge_index.size(1)

        if greedy_mask.sum() > 0 and total_edges > 0:
            violations = (greedy_mask[src] & greedy_mask[dst]).sum().float().item()
        else:
            violations = 0.0

        feasibility = 1.0 - (violations / max(total_edges, 1))
        gap = opt_size - pp_pred_size
        approx_ratio = pp_pred_size / (opt_size + 1e-8)

        return {
            "pp_pred_size": pp_pred_size,
            "pp_feasibility": feasibility,
            "pp_gap": gap,
            "pp_approx_ratio": approx_ratio,
        }

    # Compute per-graph metrics
    num_graphs = len(ptr) - 1
    total_pp_pred = 0
    total_opt = 0
    total_gap = 0
    total_approx_ratio = 0
    total_violations = 0
    total_edges = 0

    for g in range(num_graphs):
        # Extract subgraph data
        node_mask = batch_vec == g
        graph_probs = probs[node_mask]
        graph_labels = labels[node_mask]

        # Get edges for this graph
        edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
        graph_edge_index = edge_index[:, edge_mask]

        # Remap edge indices to local node indices
        node_indices = torch.where(node_mask)[0]
        if len(node_indices) > 0:
            local_idx_map = torch.zeros(batch_vec.size(0), dtype=torch.long, device=probs.device)
            local_idx_map[node_indices] = torch.arange(len(node_indices), device=probs.device)
            graph_edge_index = local_idx_map[graph_edge_index]

        # Compute metrics for this graph
        graph_num_nodes = graph_probs.size(0)
        graph_num_edges = graph_edge_index.size(1)
        graph_opt_size = graph_labels.sum().item()
        total_edges += graph_num_edges
        total_opt += graph_opt_size

        if graph_num_nodes > 0:
            # Run greedy decode for this graph
            graph_pp_size, greedy_decoded = greedy_decode(graph_probs, graph_edge_index, graph_num_nodes)

            # Verify feasibility of greedy-decoded solution
            src, dst = graph_edge_index[0], graph_edge_index[1]
            greedy_mask = greedy_decoded == 1.0

            if greedy_mask.sum() > 0 and graph_num_edges > 0:
                graph_violations = (greedy_mask[src] & greedy_mask[dst]).sum().float().item()
            else:
                graph_violations = 0.0
            total_violations += graph_violations
            total_pp_pred += graph_pp_size

            if graph_opt_size > 0:
                total_gap += graph_opt_size - graph_pp_size
                total_approx_ratio += graph_pp_size / graph_opt_size

    # Average over graphs
    avg_pp_pred = total_pp_pred / max(num_graphs, 1)
    avg_gap = total_gap / max(num_graphs, 1)
    avg_approx_ratio = total_approx_ratio / max(num_graphs, 1)

    # Compute feasibility (should be 1.0 if greedy decode works correctly)
    if total_edges > 0:
        feasibility = 1.0 - (total_violations / total_edges)
    else:
        feasibility = 1.0

    return {
        "pp_pred_size": avg_pp_pred,
        "pp_feasibility": feasibility,
        "pp_gap": avg_gap,
        "pp_approx_ratio": avg_approx_ratio,
    }
