import os
import glob
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data, Batch

class MISDatasetConfig:
    def __init__(self, dataset_paths, global_batch_size, rank=0, num_replicas=1, seed=42, epoch=0, drop_last=True, **kwargs):
        self.dataset_paths = dataset_paths
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = epoch  # For per-epoch shuffling
        self.drop_last = drop_last  # Drop partial batches to avoid gradient variance

class MISDataset(IterableDataset):
    def __init__(self, config: MISDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # 1. Find all shards
        self.shards = []
        for path in config.dataset_paths:
            # We assume shards are directly in the path or a subdir
            # Adjust pattern if your shards are in "train" subdir
            search_path = os.path.join(path, "mis_shard_*.pt")
            found = sorted(glob.glob(search_path))
            self.shards.extend(found)

        if not self.shards:
            raise FileNotFoundError(f"No mis_shard_*.pt files found in {config.dataset_paths}")

        # Metadata for the trainer (dummy values to satisfy interfaces)
        self.metadata = type('Metadata', (), {})()
        self.metadata.total_groups = len(self.shards)
        self.metadata.mean_puzzle_examples = 250 # Approximation
        self.metadata.sets = ["mis"]

        # Needed for TRM sizing (input dim is usually 2: ones + degree)
        # We load one sample to check dims
        dummy_data = torch.load(self.shards[0], weights_only=False)["data"][0]
        self.metadata.input_dim = dummy_data["x"].shape[1]

        # Compute global class imbalance for pos_weight
        # Sample from first shard to estimate
        self._compute_class_imbalance()

    def _compute_class_imbalance(self):
        """Compute global pos_weight once from dataset (not per-batch)"""
        total_pos = 0
        total_neg = 0

        # Sample first few shards to estimate imbalance
        sample_shards = self.shards[:min(5, len(self.shards))]
        for shard_path in sample_shards:
            payload = torch.load(shard_path, weights_only=False)
            for sample in payload["data"]:
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
            d = Data(
                x=s["x"],
                edge_index=s["edge_index"],
                y=s["y"],
                num_nodes=s["n"]
            )
            # Store opt_value for metrics if needed
            d.opt_value = torch.tensor([s["opt_value"]])
            data_list.append(d)

        # PyG Batching (automatically handles edge_index offsets)
        batch = Batch.from_data_list(data_list)

        # Return as a dict compatible with your trainer
        return {
            "x": batch.x,
            "edge_index": batch.edge_index,
            "batch": batch.batch, # The vector [0,0,0, 1,1,1...] mapping nodes to graph ID
            "y": batch.y,
            "ptr": batch.ptr, # specific to PyG, helpful for pooling
            "num_graphs": batch.num_graphs,
            "opt_value": batch.opt_value  # Include opt_value for evaluation
        }

    def __iter__(self):
        worker_info = get_worker_info()
        rank = self.config.rank
        world_size = self.config.num_replicas
        epoch = self.config.epoch

        # Deterministic shuffling based on epoch AND rank
        # Different shuffle each epoch prevents ordering overfitting
        rng = np.random.RandomState(self.config.seed + rank + epoch * 1000)

        # Shard splitting across ranks
        my_shards = self.shards[rank::world_size]
        rng.shuffle(my_shards)

        buffer = []

        for shard_path in my_shards:
            payload = torch.load(shard_path, weights_only=False)
            data_list = payload["data"]
            rng.shuffle(data_list)

            for sample in data_list:
                buffer.append(sample)

                if len(buffer) >= self.config.global_batch_size:
                    # Yield a batch
                    yield "mis", self._collate_graph_batch(buffer), len(buffer)
                    buffer = []

        # Handle remaining samples
        # drop_last=True (default): Drop partial batch to avoid gradient variance
        # drop_last=False: Yield partial batch (for evaluation)
        if buffer and not self.config.drop_last:
            yield "mis", self._collate_graph_batch(buffer), len(buffer)
