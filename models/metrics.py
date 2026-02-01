import torch

from models.pp import greedy_decode


def compute_metrics(probs, edge_index, labels, batch_vec=None, ptr=None):
    """
    Compute raw model metrics (no post-processing).

    This function runs ALWAYS, regardless of use_postprocessing setting.
    These metrics reflect the model's actual predictions before any greedy decode.

    Returns dict with:
    - opt_size: Average ground truth MIS size per graph
    - pred_size: Average predicted set size per graph (thresholded at 0.5)
    - gap: Average (opt_size - pred_size) per graph
    - approx_ratio: Average pred_size / opt_size per graph (CAN BE > 1 if infeasible)
    - feasibility: 1 - (violations / total_edges)
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
            "opt_size": opt_size,
            "pred_size": pred_size,
            "gap": gap,
            "approx_ratio": approx_ratio,
            "feasibility": feasibility,
        }

    # Compute per-graph metrics
    num_graphs = len(ptr) - 1
    total_opt = 0
    total_pred = 0
    total_gap = 0
    total_gap_ratio = 0
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
        graph_opt = graph_labels.sum().item()
        graph_num_nodes = graph_probs.size(0)
        graph_preds_binary = (graph_probs > 0.5).float()
        graph_pred_size = graph_preds_binary.sum().item()

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

        if graph_num_nodes > 0 and graph_opt > 0:
            graph_gap = graph_opt - graph_pred_size
            graph_gap_ratio = graph_gap / graph_opt
            graph_approx_ratio = graph_pred_size / graph_opt

            total_opt += graph_opt
            total_pred += graph_pred_size
            total_gap += graph_gap
            total_gap_ratio += graph_gap_ratio
            total_approx_ratio += graph_approx_ratio

    # Average over graphs
    avg_opt = total_opt / max(num_graphs, 1)
    avg_pred = total_pred / max(num_graphs, 1)
    avg_gap = total_gap / max(num_graphs, 1)
    avg_approx_ratio = total_approx_ratio / max(num_graphs, 1)

    # Compute feasibility
    if total_edges > 0:
        feasibility = 1.0 - (total_violations / total_edges)
    else:
        feasibility = 1.0

    return {
        "opt_size": avg_opt,
        "pred_size": avg_pred,
        "gap": avg_gap,
        "approx_ratio": avg_approx_ratio,
        "feasibility": feasibility,
    }


def compute_pp_metrics(probs, edge_index, labels, batch_vec=None, ptr=None):
    """
    Compute post-processed metrics using greedy decode.

    This function should ONLY run when use_postprocessing=True.
    These metrics reflect the greedy-decoded feasible solution.

    GUARANTEES (by construction of greedy decode):
    - pp_feasibility == 1.0 (no violations possible)
    - pp_num_violations == 0 (no adjacent nodes both selected)
    - pp_approx_ratio <= 1.0 (feasible set can't exceed optimal)
    - pp_pred_size <= opt_size (for same reason)

    Returns dict with:
    - pp_pred_size: Average greedy-decoded set size per graph
    - pp_gap: Average (opt_size - pp_pred_size) per graph
    - pp_approx_ratio: Average pp_pred_size / opt_size per graph (ALWAYS <= 1)
    - pp_feasibility: 1 - (violations / edges) (ALWAYS 1.0 - verified)
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
            "pp_gap": gap,
            "pp_approx_ratio": approx_ratio,
            "pp_feasibility": feasibility,
        }

    # Compute per-graph metrics
    num_graphs = len(ptr) - 1
    total_opt = 0
    total_pp_pred = 0
    total_gap = 0
    total_gap_ratio = 0
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
        graph_opt = graph_labels.sum().item()
        graph_num_nodes = graph_probs.size(0)
        graph_num_edges = graph_edge_index.size(1)
        total_edges += graph_num_edges

        if graph_num_nodes > 0 and graph_opt > 0:
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

            graph_gap = graph_opt - graph_pp_size
            graph_gap_ratio = graph_gap / graph_opt
            graph_approx_ratio = graph_pp_size / graph_opt

            total_opt += graph_opt
            total_pp_pred += graph_pp_size
            total_gap += graph_gap
            total_gap_ratio += graph_gap_ratio
            total_approx_ratio += graph_approx_ratio

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
        "pp_gap": avg_gap,
        "pp_approx_ratio": avg_approx_ratio,
        "pp_feasibility": feasibility,
    }
