import numpy as np
import torch


def greedy_decode(probs, edge_index, num_nodes):
    """
    Turns probabilities into a valid Independent Set using a greedy strategy.

    IMPORTANT: Only considers nodes with prob > 0.5 (model's actual predictions).
    This ensures pp_pred_size <= pred_size (post-processing refines, not expands).

    Algorithm:
    1. Filter to only nodes with prob > 0.5
    2. Sort these nodes by probability (highest first)
    3. Greedily pick nodes, blocking neighbors
    4. Result is a feasible subset of the model's predictions

    Returns: (set_size, selected_nodes_tensor)
    """
    probs_np = probs.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()

    # Create adjacency list
    adj = {i: set() for i in range(num_nodes)}
    for u, v in zip(edge_index_np[0], edge_index_np[1]):
        adj[u].add(v)
        adj[v].add(u)

    # Only consider nodes that the model predicted (prob > 0.5)
    candidate_nodes = np.where(probs_np > 0.5)[0]

    # Sort candidate nodes by probability (descending)
    if len(candidate_nodes) > 0:
        candidate_probs = probs_np[candidate_nodes]
        sorted_indices = np.argsort(-candidate_probs)
        sorted_candidates = candidate_nodes[sorted_indices]
    else:
        sorted_candidates = np.array([], dtype=np.int64)

    selected_set = set()
    blocked_nodes = set()

    for node in sorted_candidates:
        if node in blocked_nodes:
            continue
        selected_set.add(node)
        blocked_nodes.add(node)
        for neighbor in adj[node]:
            blocked_nodes.add(neighbor)

    # Return as tensor for consistency
    selected_tensor = torch.zeros(num_nodes, device=probs.device)
    for node in selected_set:
        selected_tensor[node] = 1.0

    return len(selected_set), selected_tensor
