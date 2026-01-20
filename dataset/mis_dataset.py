import glob
import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import torch
from scipy.sparse.linalg import eigsh
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch, Data


def compute_laplacian_pe(edge_index, num_nodes, pe_dim=16):
    """
    Compute Laplacian Positional Encoding (LapPE) for a graph.

    LapPE uses the eigenvectors of the graph Laplacian matrix to encode
    structural position. Unlike Random Walk PE (local), LapPE captures
    global graph structure.

    Args:
        edge_index: [2, E] tensor of edges
        num_nodes: Number of nodes in graph
        pe_dim: Number of eigenvectors to use (default: 16)

    Returns:
        pe: [N, pe_dim] tensor of positional encodings
    """
    # Handle edge cases
    if num_nodes <= pe_dim + 1 or edge_index.size(1) == 0:
        # Graph too small for meaningful PE, return zeros
        return torch.zeros(num_nodes, pe_dim)

    # Build adjacency matrix
    edge_index_np = edge_index.cpu().numpy()
    row, col = edge_index_np[0], edge_index_np[1]

    # Create sparse adjacency matrix
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # Compute degree matrix
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute smallest eigenvalues/eigenvectors (excluding the trivial one)
    # We want pe_dim eigenvectors, so compute pe_dim + 1 and skip first
    try:
        # Use 'SM' for smallest magnitude eigenvalues
        eigenvalues, eigenvectors = eigsh(L, k=pe_dim + 1, which="SM", tol=1e-3, maxiter=500)
        # Skip the first (trivial) eigenvector, take next pe_dim
        pe = eigenvectors[:, 1 : pe_dim + 1]
    except Exception:
        # Fallback: if eigendecomposition fails, return zeros
        return torch.zeros(num_nodes, pe_dim)

    # Handle sign ambiguity: random sign flip for augmentation robustness
    # (eigenvectors are only defined up to sign)
    pe = torch.from_numpy(pe).float()

    # Pad if we got fewer eigenvectors than requested
    if pe.size(1) < pe_dim:
        padding = torch.zeros(num_nodes, pe_dim - pe.size(1))
        pe = torch.cat([pe, padding], dim=1)

    return pe


def compute_node_features(edge_index, num_nodes, x_original):
    """
    Compute additional node features for MIS.

    Features computed (Section 1.2 from IMPROVEMENT_PLAN.md):
    - A. Raw degree (un-normalized)
    - C. Clustering coefficient
    - E. Local degree statistics (avg/max/min neighbor degree)
    - F. Core number (k-core)

    Args:
        edge_index: [2, E] tensor of edges
        num_nodes: Number of nodes in graph
        x_original: Original node features [N, 2] with [1, degree_norm]

    Returns:
        x_enhanced: [N, num_features] tensor with all features
    """
    # Build NetworkX graph for algorithms
    edge_index_np = edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index_np[0], edge_index_np[1]))
    G.add_edges_from(edges)

    # Initialize feature arrays
    raw_degree = np.zeros(num_nodes, dtype=np.float32)
    clustering = np.zeros(num_nodes, dtype=np.float32)
    avg_neighbor_deg = np.zeros(num_nodes, dtype=np.float32)
    max_neighbor_deg = np.zeros(num_nodes, dtype=np.float32)
    min_neighbor_deg = np.zeros(num_nodes, dtype=np.float32)
    core_number = np.zeros(num_nodes, dtype=np.float32)

    # Raw degree
    degrees = dict(G.degree())
    for node, deg in degrees.items():
        raw_degree[node] = deg

    # Normalize raw degree by max degree in graph for stability
    max_deg = max(raw_degree.max(), 1)
    raw_degree_norm = raw_degree / max_deg

    # Clustering coefficient
    clustering_dict = nx.clustering(G)
    for node, cc in clustering_dict.items():
        clustering[node] = cc

    # Local degree statistics
    for node in range(num_nodes):
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            neighbor_degs = [degrees[n] for n in neighbors]
            avg_neighbor_deg[node] = np.mean(neighbor_degs) / max_deg  # Normalize
            max_neighbor_deg[node] = np.max(neighbor_degs) / max_deg
            min_neighbor_deg[node] = np.min(neighbor_degs) / max_deg
        # else: stays 0

    # Core number (k-core)
    try:
        core_dict = nx.core_number(G)
        max_core = max(core_dict.values()) if core_dict else 1
        max_core = max(max_core, 1)  # Avoid division by zero
        for node, k in core_dict.items():
            core_number[node] = k / max_core  # Normalize
    except nx.NetworkXError:
        # Graph might have issues, keep zeros
        pass

    # =========================================================================
    # RELATIVE PE: Shortest path statistics (node-level aggregation)
    # =========================================================================
    # Compute shortest path distances and aggregate to per-node statistics
    avg_sp_dist = np.zeros(num_nodes, dtype=np.float32)  # Average shortest path
    max_sp_dist = np.zeros(num_nodes, dtype=np.float32)  # Eccentricity
    min_sp_dist = np.zeros(num_nodes, dtype=np.float32)  # Min distance to other nodes
    closeness = np.zeros(num_nodes, dtype=np.float32)  # Closeness centrality

    try:
        # Build sparse adjacency for shortest paths
        data = np.ones(len(edge_index_np[0]), dtype=np.float32)
        adj_sparse = sp.csr_matrix((data, (edge_index_np[0], edge_index_np[1])), shape=(num_nodes, num_nodes))

        # Compute all-pairs shortest paths (returns inf for disconnected pairs)
        sp_matrix = csgraph.shortest_path(adj_sparse, directed=False, unweighted=True)

        # Replace inf with a large finite value (graph diameter + 1)
        finite_mask = np.isfinite(sp_matrix)
        if finite_mask.any():
            max_finite = sp_matrix[finite_mask].max()
        else:
            max_finite = num_nodes  # Fallback
        sp_matrix = np.where(finite_mask, sp_matrix, max_finite + 1)

        # Compute per-node statistics (excluding self-distance = 0)
        for node in range(num_nodes):
            distances = sp_matrix[node, :]
            other_dists = np.concatenate([distances[:node], distances[node + 1 :]])
            if len(other_dists) > 0:
                avg_sp_dist[node] = np.mean(other_dists)
                max_sp_dist[node] = np.max(other_dists)  # Eccentricity
                min_sp_dist[node] = np.min(other_dists)
                # Closeness: inverse of average distance (normalized)
                closeness[node] = (num_nodes - 1) / (np.sum(other_dists) + 1e-8)

        # Normalize by graph diameter for stability
        graph_diameter = max(max_sp_dist.max(), 1)
        avg_sp_dist = avg_sp_dist / graph_diameter
        max_sp_dist = max_sp_dist / graph_diameter
        min_sp_dist = min_sp_dist / graph_diameter
        # Closeness is already normalized (0-1 range)

    except Exception:
        # Fallback: keep zeros if shortest path computation fails
        pass

    # Stack new features: [raw_deg_norm, clustering, avg_neigh, max_neigh, min_neigh, core,
    #                      avg_sp_dist, max_sp_dist, min_sp_dist, closeness]
    new_features = np.stack(
        [
            raw_degree_norm,
            clustering,
            avg_neighbor_deg,
            max_neighbor_deg,
            min_neighbor_deg,
            core_number,
            avg_sp_dist,  # Relative PE: average shortest path distance
            max_sp_dist,  # Relative PE: eccentricity
            min_sp_dist,  # Relative PE: min distance to other nodes
            closeness,  # Relative PE: closeness centrality
        ],
        axis=1,
    )  # [N, 10]

    new_features_tensor = torch.from_numpy(new_features).float()

    # Concatenate with original features [1, degree_norm]
    # Final: [1, degree_norm, raw_deg_norm, clustering, avg_neigh, max_neigh, min_neigh, core,
    #         avg_sp_dist, max_sp_dist, min_sp_dist, closeness]
    x_enhanced = torch.cat([x_original, new_features_tensor], dim=1)

    return x_enhanced


class MISDatasetConfig:
    def __init__(
        self,
        dataset_paths,
        global_batch_size,
        rank=0,
        num_replicas=1,
        seed=42,
        epoch=0,
        drop_last=True,
        val_split=0.1,
        max_shards=None,
        inject_noise=False,  # Inject random noise for symmetry breaking
        pe_dim=16,  # Laplacian PE dimension
    ):
        self.dataset_paths = dataset_paths
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = epoch  # For per-epoch shuffling
        self.drop_last = drop_last  # Drop partial batches for consistent gradients
        self.val_split = val_split  # Fraction of data for validation (0.1 = 10%)
        self.max_shards = max_shards  # Limit shards for debugging/testing
        self.inject_noise = inject_noise
        self.pe_dim = pe_dim


class MISDataset(IterableDataset):
    def __init__(self, config: MISDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split  # "train" or "val"

        # 1. Find all shards
        all_shards = []
        for path in config.dataset_paths:
            # We assume shards are directly in the path or a subdir
            # Adjust pattern if your shards are in "train" subdir
            search_path = os.path.join(path, "mis_shard_*.pt")
            found = sorted(glob.glob(search_path))
            all_shards.extend(found)

        if not all_shards:
            raise FileNotFoundError(f"No mis_shard_*.pt files found in {config.dataset_paths}")

        # 2. Apply max_shards FIRST (before splitting)
        # Use deterministic shuffle based on seed
        rng = np.random.RandomState(config.seed)
        shard_indices = np.arange(len(all_shards))
        rng.shuffle(shard_indices)

        if config.max_shards is not None and config.max_shards > 0:
            shard_indices = shard_indices[: config.max_shards]

        selected_shards = [all_shards[i] for i in sorted(shard_indices)]

        # 3. Load ALL graphs from selected shards for graph-level splitting
        all_graphs = []
        for shard_path in selected_shards:
            payload = torch.load(shard_path, weights_only=False)
            all_graphs.extend(payload["data"])

        # 4. Split at GRAPH level (not shard level)
        num_graphs = len(all_graphs)
        graph_indices = np.arange(num_graphs)
        rng2 = np.random.RandomState(config.seed + 1)  # Different seed for graph shuffle
        rng2.shuffle(graph_indices)

        val_count = max(1, int(num_graphs * config.val_split))
        if split == "val":
            my_indices = sorted(graph_indices[:val_count])
        else:  # train
            my_indices = sorted(graph_indices[val_count:])

        self.all_graphs = [all_graphs[i] for i in my_indices]
        self.num_graphs = len(self.all_graphs)
        self.shards = selected_shards  # Keep for reference/logging
        self.pe_dim = config.pe_dim  # Store PE dimension

        # Metadata for the trainer (simple namespace)
        class DatasetMetadata:
            def __init__(self):
                self.total_groups = 0
                self.num_graphs = 0
                self.sets = []
                self.input_dim = 12  # 2 original + 10 enhanced (including relative PE)
                self.pe_dim = 16
                self.pos_weight = 1.0
                self.class_ratio = 0.5

        self.metadata = DatasetMetadata()
        self.metadata.total_groups = len(self.shards)
        self.metadata.num_graphs = self.num_graphs
        self.metadata.sets = ["mis"]

        # ---------------------------------------------------------------------
        # CACHING, PE COMPUTATION & NOISE INJECTION
        # ---------------------------------------------------------------------
        self.cached_data = None
        # Cache small datasets for faster iteration
        if self.num_graphs <= 3000:
            noise_msg = " (with noise injection)" if config.inject_noise else ""
            print(f"[{split.upper()}] Caching {self.num_graphs} graphs with Laplacian PE (dim={config.pe_dim}){noise_msg}...")
            self.cached_data = []

            # Static generator for consistent noise across epochs (if enabled)
            static_rng = torch.Generator()
            static_rng.manual_seed(12345)

            for sample in self.all_graphs:
                sample_copy = sample.copy()

                # Compute Laplacian PE for this graph
                pe = compute_laplacian_pe(sample["edge_index"], sample["n"], pe_dim=config.pe_dim)
                sample_copy["pe"] = pe

                # Compute enhanced node features (clustering, k-core, neighbor stats)
                x_enhanced = compute_node_features(sample["edge_index"], sample["n"], sample["x"])
                sample_copy["x"] = x_enhanced

                # Only inject noise if explicitly enabled
                if config.inject_noise:
                    n_nodes = sample["n"]
                    noise = torch.randn(n_nodes, 16, generator=static_rng)
                    sample_copy["x"] = torch.cat([sample_copy["x"], noise], dim=1)

                self.cached_data.append(sample_copy)

            # Update input dim from modified data
            if self.cached_data:
                self.metadata.input_dim = self.cached_data[0]["x"].shape[1]
                self.metadata.pe_dim = config.pe_dim
            else:
                self.metadata.input_dim = 2  # Fallback
                self.metadata.pe_dim = config.pe_dim

            print(f"[{split.upper()}] Caching complete. input_dim={self.metadata.input_dim}, pe_dim={self.metadata.pe_dim}")

        else:
            # Standard streaming setup - compute enhanced input_dim
            # Features: original (2) + enhanced (10) = 12
            # Original: [1, degree_norm]
            # Enhanced: [raw_deg_norm, clustering, avg_neigh, max_neigh, min_neigh, core,
            #            avg_sp_dist, max_sp_dist, min_sp_dist, closeness]
            self.metadata.input_dim = 12  # 2 original + 10 enhanced features
            self.metadata.pe_dim = config.pe_dim
            print(f"[{split.upper()}] Streaming mode. input_dim={self.metadata.input_dim}, pe_dim={self.metadata.pe_dim}")

        # Compute global class imbalance for pos_weight
        # Sample from first shard to estimate
        self._compute_class_imbalance()

    def _compute_class_imbalance(self):
        """Compute global pos_weight once from dataset (not per-batch)"""
        total_pos = 0
        total_neg = 0

        # Use cached data if available, otherwise sample from all_graphs
        graphs_to_sample = self.cached_data if self.cached_data else self.all_graphs[: min(1000, len(self.all_graphs))]

        for sample in graphs_to_sample:
            y = sample["y"]
            pos = (y == 1).sum().item()
            neg = (y == 0).sum().item()
            total_pos += pos
            total_neg += neg

        # pos_weight = neg_count / pos_count (for BCE loss)
        self.metadata.pos_weight = total_neg / max(1.0, total_pos)
        self.metadata.class_ratio = total_pos / max(1.0, total_pos + total_neg)

    def set_epoch(self, epoch: int):
        """Update epoch for per-epoch shuffling (call before each epoch)"""
        self.config.epoch = epoch

    def _collate_graph_batch(self, sample_list):
        # Convert list of dicts to list of PyG Data objects
        data_list = []
        for s in sample_list:
            d = Data(x=s["x"], edge_index=s["edge_index"], y=s["y"], num_nodes=s["n"])
            # Store opt_value for metrics if needed
            d.opt_value = torch.tensor([s["opt_value"]])
            # Include PE if available
            if "pe" in s:
                d.pe = s["pe"]
            data_list.append(d)

        # PyG Batching (automatically handles edge_index offsets)
        batch = Batch.from_data_list(data_list)

        # Return as a dict compatible with your trainer
        result = {
            "x": batch.x,
            "edge_index": batch.edge_index,
            "batch": batch.batch,  # The vector [0,0,0, 1,1,1...] mapping nodes to graph ID
            "y": batch.y,
            "ptr": batch.ptr,  # specific to PyG, helpful for pooling
            "num_graphs": batch.num_graphs,
            "opt_value": batch.opt_value,  # Include opt_value for evaluation
        }

        # Include PE if it was computed
        if hasattr(batch, "pe") and batch.pe is not None:
            result["pe"] = batch.pe

        return result

    def __iter__(self):
        rank = self.config.rank
        world_size = self.config.num_replicas
        epoch = self.config.epoch

        # Deterministic shuffling based on epoch AND rank
        # Different shuffle each epoch for better generalization
        rng = np.random.RandomState(self.config.seed + rank + epoch * 1000)

        # ---------------------------------------------------------------------
        # USE CACHED DATA IF AVAILABLE (for small datasets)
        # ---------------------------------------------------------------------
        if self.cached_data is not None:
            indices = np.arange(len(self.cached_data))
            # Shard the indices for multi-gpu/worker
            my_indices = indices[rank::world_size]
            rng.shuffle(my_indices)  # Shuffle locally

            buffer = []
            for i in my_indices:
                sample = self.cached_data[i]
                buffer.append(sample)
                if len(buffer) >= self.config.global_batch_size:
                    yield "mis", self._collate_graph_batch(buffer), len(buffer)
                    buffer = []
            if buffer and not self.config.drop_last:
                yield "mis", self._collate_graph_batch(buffer), len(buffer)
            return

        # ---------------------------------------------------------------------
        # STANDARD STREAMING (from pre-loaded graphs)
        # ---------------------------------------------------------------------

        # Split graphs across ranks
        my_graphs = self.all_graphs[rank::world_size]

        # Shuffle graph indices for this epoch
        indices = list(range(len(my_graphs)))
        rng.shuffle(indices)

        buffer = []

        for idx in indices:
            sample = my_graphs[idx]

            # Apply feature enhancement on-the-fly (same as caching path)
            sample_copy = sample.copy()

            # Compute Laplacian PE
            pe = compute_laplacian_pe(sample["edge_index"], sample["n"], pe_dim=self.config.pe_dim)
            sample_copy["pe"] = pe

            # Compute enhanced node features
            x_enhanced = compute_node_features(sample["edge_index"], sample["n"], sample["x"])
            sample_copy["x"] = x_enhanced

            buffer.append(sample_copy)

            if len(buffer) >= self.config.global_batch_size:
                # Yield a batch
                yield "mis", self._collate_graph_batch(buffer), len(buffer)
                buffer = []

        # Handle remaining samples
        # drop_last=True (default): Drop partial batch to avoid gradient variance
        # drop_last=False: Yield partial batch (for evaluation)
        if buffer and not self.config.drop_last:
            yield "mis", self._collate_graph_batch(buffer), len(buffer)
