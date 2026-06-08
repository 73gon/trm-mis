"""
Enhanced Post-Processing for MIS

Improvements over the original greedy_decode in models/pp.py:

1. **Local Search (1-swap & 2-swap)**: After greedy decode, iteratively try to
   improve the solution by swapping nodes in/out of the independent set.

2. **Multi-Threshold Greedy**: Run greedy decode at multiple thresholds and
   return the best solution.

3. **Graph Reduction Rules**: Apply pendant/domination rules before greedy
   to reduce the problem size.

These are all inference-time improvements — no retraining needed.
"""

import numpy as np
import torch


def build_adjacency(edge_index_np, num_nodes):
    """Build adjacency list from edge index."""
    adj = [set() for _ in range(num_nodes)]
    for u, v in zip(edge_index_np[0], edge_index_np[1]):
        adj[u].add(v)
        adj[v].add(u)
    return adj


def greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes):
    """Core greedy decode operating on numpy arrays with prebuilt adjacency."""
    if len(candidate_nodes) > 0:
        candidate_probs = probs_np[candidate_nodes]
        sorted_indices = np.argsort(-candidate_probs)
        sorted_candidates = candidate_nodes[sorted_indices]
    else:
        sorted_candidates = np.array([], dtype=np.int64)

    selected = set()
    blocked = set()

    for node in sorted_candidates:
        if node in blocked:
            continue
        selected.add(node)
        blocked.add(node)
        for neighbor in adj[node]:
            blocked.add(neighbor)

    return selected


def local_search_1swap(selected, adj, num_nodes, max_iters=20):
    """
    1-swap local search: try adding a non-selected node by removing
    its conflicting selected neighbors. Accept if net gain >= 1
    (i.e., we remove 0 neighbors and add 1 node — the node has no
    selected neighbors, meaning greedy missed it).

    Also try removing a selected node if it frees up >= 2 neighbors
    that can all be added.
    """
    selected = set(selected)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1

        # Phase 1: Try adding unselected nodes that have no conflicts
        for node in range(num_nodes):
            if node in selected:
                continue
            # Check how many selected neighbors this node has
            selected_neighbors = selected & adj[node]
            if len(selected_neighbors) == 0:
                # Free node — add it (greedy missed it)
                selected.add(node)
                improved = True

        # Phase 2: Try 1-swap: remove 1 selected node, add >= 2 freed nodes
        for node in list(selected):
            # What nodes become unblocked if we remove this node?
            # A neighbor v of `node` becomes a candidate if:
            # - v is not selected
            # - v's only selected neighbor was `node`
            freed = []
            for v in adj[node]:
                if v in selected:
                    continue
                selected_neighbors_of_v = selected & adj[v]
                # After removing `node`, v's selected neighbors = selected_neighbors_of_v - {node}
                if selected_neighbors_of_v == {node}:
                    freed.append(v)

            # Check if freed nodes are mutually independent
            # Greedily add freed nodes
            if len(freed) >= 2:
                temp_selected = selected - {node}
                gained = set()
                blocked_freed = set()
                for v in freed:
                    if v in blocked_freed:
                        continue
                    # Check v has no conflicts with temp_selected + gained
                    conflict = False
                    for u in adj[v]:
                        if u in gained:
                            conflict = True
                            break
                    if not conflict:
                        gained.add(v)
                        for u in adj[v]:
                            if u in set(freed) - gained:
                                blocked_freed.add(u)

                if len(gained) >= 2:  # Net gain: +gained - 1 (removed node) >= 1
                    selected = temp_selected | gained
                    improved = True
                    break  # Restart outer loop after modification

    return selected


def local_search_2swap(selected, adj, num_nodes, max_iters=50):
    """
    2-swap local search: try removing 2 adjacent selected nodes and adding
    3+ freed nodes. More expensive but can escape deeper local optima.
    """
    selected = set(selected)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for node_a in list(selected):
            if improved:
                break
            for node_b in adj[node_a]:
                if node_b not in selected or node_b <= node_a:
                    continue

                # Try removing both node_a and node_b
                temp_selected = selected - {node_a, node_b}

                # Find freed nodes: non-selected neighbors whose only
                # selected neighbors were among {node_a, node_b}
                freed = []
                candidates = (adj[node_a] | adj[node_b]) - temp_selected - {node_a, node_b}
                for v in candidates:
                    sel_neighbors = adj[v] & temp_selected
                    if len(sel_neighbors) == 0:
                        freed.append(v)

                # Also node_a and node_b themselves might be re-added
                # (but that's just undoing the removal, skip)

                # Greedily add freed nodes
                if len(freed) >= 3:  # Need net gain: +freed - 2 >= 1
                    gained = set()
                    for v in freed:
                        conflict = False
                        for u in adj[v]:
                            if u in gained or u in temp_selected:
                                conflict = True
                                break
                        if not conflict:
                            gained.add(v)

                    if len(gained) >= 3:
                        selected = temp_selected | gained
                        improved = True
                        break

    return selected


def greedy_decode_enhanced(probs, edge_index, num_nodes, threshold=0.5,
                           use_local_search=True, use_2swap=False):
    """
    Enhanced greedy decode with local search.

    Args:
        probs: Node probabilities [N]
        edge_index: Edge index [2, E]
        num_nodes: Number of nodes
        threshold: Probability threshold for candidates
        use_local_search: Enable 1-swap local search
        use_2swap: Enable 2-swap local search (more expensive)

    Returns: (set_size, selected_nodes_tensor)
    """
    probs_np = probs.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()
    adj = build_adjacency(edge_index_np, num_nodes)

    # Greedy decode
    candidate_nodes = np.where(probs_np > threshold)[0]
    selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)

    # Local search
    if use_local_search:
        selected = local_search_1swap(selected, adj, num_nodes)

    if use_2swap:
        selected = local_search_2swap(selected, adj, num_nodes)

    # Convert to tensor
    selected_tensor = torch.zeros(num_nodes, device=probs.device)
    for node in selected:
        selected_tensor[node] = 1.0

    return len(selected), selected_tensor


def multi_threshold_greedy(probs, edge_index, num_nodes,
                           thresholds=None, use_local_search=True,
                           use_2swap=False):
    """
    Run greedy decode at multiple thresholds and return the best solution.

    Args:
        thresholds: List of thresholds to try. Default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        use_local_search: Apply 1-swap after each greedy decode
        use_2swap: Apply 2-swap after each greedy decode

    Returns: (best_size, best_selected_tensor)
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    probs_np = probs.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()
    adj = build_adjacency(edge_index_np, num_nodes)

    best_size = 0
    best_selected = set()

    for threshold in thresholds:
        candidate_nodes = np.where(probs_np > threshold)[0]
        selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)

        if use_local_search:
            selected = local_search_1swap(selected, adj, num_nodes)

        if use_2swap:
            selected = local_search_2swap(selected, adj, num_nodes)

        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected

    # Convert to tensor
    selected_tensor = torch.zeros(num_nodes, device=probs.device)
    for node in best_selected:
        selected_tensor[node] = 1.0

    return best_size, selected_tensor


def apply_reduction_rules(adj, num_nodes):
    """
    Apply standard MIS reduction rules.

    1. Pendant Rule: Degree-1 nodes are always in the MIS (add them,
       remove their neighbor)
    2. Domination Rule: If N(u) ⊆ N(v), then v dominates u, and we
       can safely remove v (u is at least as good)

    Returns:
        forced_in: Set of nodes forced into the MIS
        removed: Set of nodes removed from consideration
    """
    forced_in = set()
    removed = set()
    active = set(range(num_nodes))
    changed = True

    while changed:
        changed = False

        # Pendant rule: degree-1 nodes
        for node in list(active):
            if node in removed:
                continue
            active_neighbors = adj[node] & active - removed
            if len(active_neighbors) == 0:
                # Isolated node — always in MIS
                forced_in.add(node)
                removed.add(node)
                active.discard(node)
                changed = True
            elif len(active_neighbors) == 1:
                # Pendant node — always in MIS
                forced_in.add(node)
                removed.add(node)
                active.discard(node)
                neighbor = next(iter(active_neighbors))
                removed.add(neighbor)
                active.discard(neighbor)
                # Remove the neighbor's neighbors from active consideration
                for nn in adj[neighbor]:
                    if nn in active and nn not in removed:
                        pass  # They remain active but their degree changed
                changed = True

    return forced_in, removed


def greedy_with_reduction(probs, edge_index, num_nodes,
                          threshold=0.0, use_local_search=True):
    """
    Apply reduction rules first, then greedy decode on the residual graph.

    Returns: (set_size, selected_nodes_tensor)
    """
    probs_np = probs.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()
    adj = build_adjacency(edge_index_np, num_nodes)

    # Apply reductions
    forced_in, removed = apply_reduction_rules(adj, num_nodes)

    # Build residual graph candidates
    remaining = set(range(num_nodes)) - removed
    candidate_nodes = np.array([n for n in remaining if probs_np[n] > threshold],
                               dtype=np.int64)

    # Run greedy on residual
    selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)

    # Combine forced + greedy
    # Make sure forced nodes don't conflict with greedy (they shouldn't by construction)
    final = forced_in | selected

    # Verify independence (shouldn't fail if reduction rules are correct)
    # Just in case, do a quick check and remove conflicts
    verified = set()
    blocked = set()
    # Process forced nodes first
    for node in forced_in:
        if node not in blocked:
            verified.add(node)
            blocked.add(node)
            for n in adj[node]:
                blocked.add(n)
    # Then greedy nodes
    sorted_greedy = sorted(selected - forced_in, key=lambda n: -probs_np[n])
    for node in sorted_greedy:
        if node not in blocked:
            verified.add(node)
            blocked.add(node)
            for n in adj[node]:
                blocked.add(n)

    if use_local_search:
        verified = local_search_1swap(verified, adj, num_nodes)

    selected_tensor = torch.zeros(num_nodes, device=probs.device)
    for node in verified:
        selected_tensor[node] = 1.0

    return len(verified), selected_tensor


def full_enhanced_decode(probs, edge_index, num_nodes):
    """
    Full enhanced pipeline: reduction + multi-threshold greedy + local search.

    This is the best-effort decoding combining all improvements.
    Used for final evaluation / SOTA comparison.
    """
    probs_np = probs.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()
    adj = build_adjacency(edge_index_np, num_nodes)

    # Step 1: Apply reductions
    forced_in, removed = apply_reduction_rules(adj, num_nodes)

    # Step 2: Multi-threshold greedy on residual
    remaining = set(range(num_nodes)) - removed
    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    best_size = 0
    best_selected = set()

    for threshold in thresholds:
        candidate_nodes = np.array([n for n in remaining if probs_np[n] > threshold],
                                   dtype=np.int64)
        selected = greedy_decode_core(probs_np, adj, num_nodes, candidate_nodes)

        # Combine with forced
        combined = set()
        blocked = set()
        for node in forced_in:
            if node not in blocked:
                combined.add(node)
                blocked.add(node)
                for n in adj[node]:
                    blocked.add(n)
        sorted_greedy = sorted(selected - forced_in, key=lambda n: -probs_np[n])
        for node in sorted_greedy:
            if node not in blocked:
                combined.add(node)
                blocked.add(node)
                for n in adj[node]:
                    blocked.add(n)

        # Local search
        combined = local_search_1swap(combined, adj, num_nodes)

        if len(combined) > best_size:
            best_size = len(combined)
            best_selected = combined

    # Step 3: 2-swap on best solution
    best_selected = local_search_2swap(best_selected, adj, num_nodes, max_iters=30)
    best_size = len(best_selected)

    selected_tensor = torch.zeros(num_nodes, device=probs.device)
    for node in best_selected:
        selected_tensor[node] = 1.0

    return best_size, selected_tensor
