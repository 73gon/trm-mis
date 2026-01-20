import math
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.ema import EMAHelper
from models.graph_transformer_trm import GraphTransformerTRM

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def greedy_decode(probs, edge_index, num_nodes):
    """
    Turns probabilities into a valid Independent Set using a greedy strategy.
    1. Sort nodes by probability (highest first).
    2. Pick the best node.
    3. Remove its neighbors.
    4. Repeat.

    Returns: (set_size, selected_nodes_tensor)
    """
    probs_np = probs.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()

    # Create adjacency list
    adj = {i: set() for i in range(num_nodes)}
    for u, v in zip(edge_index_np[0], edge_index_np[1]):
        adj[u].add(v)
        adj[v].add(u)

    # Sort nodes by probability (descending)
    sorted_nodes = np.argsort(-probs_np)

    selected_set = set()
    blocked_nodes = set()

    for node in sorted_nodes:
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


def compute_postprocessing_metrics(probs, edge_index, labels, batch_vec=None, ptr=None, use_postprocessing=True):
    """
    Compute post-processing metrics using greedy decode.

    When batch_vec and ptr are provided, computes metrics PER GRAPH and averages.
    This gives a fair comparison across batches with different numbers of graphs.

    When use_postprocessing=False, reports model outputs without greedy decode.
    This allows the model to learn to produce valid MIS directly.

    Returns dict with:
    - optimal_size: Average ground truth MIS size per graph
    - size: Average predicted set size per graph (after greedy decode if use_postprocessing=True)
    - gap: Average (optimal_size - size) per graph
    - gap_ratio: Average gap / optimal_size per graph
    - approx_ratio: Average size / optimal_size per graph
    - feasibility: Feasibility of predictions (for monitoring)
    - violations: Number of constraint violations in predictions
    """
    # If batch info not provided, fall back to batch-level metrics
    if batch_vec is None or ptr is None:
        num_nodes = probs.size(0)
        preds_binary = (probs > 0.5).float()
        pred_size = preds_binary.sum().item()
        optimal_size = labels.sum().item()

        # Compute feasibility (violations in predictions)
        src, dst = edge_index[0], edge_index[1]
        pred_mask = preds_binary == 1
        total_edges = edge_index.size(1)
        if pred_mask.sum() > 0 and total_edges > 0:
            violations = (pred_mask[src] & pred_mask[dst]).sum().float().item()
        else:
            violations = 0.0
        # Each undirected edge appears twice in edge_index, so divide by 2
        violations_count = violations / 2
        # Feasibility = 1 - (violations / total_edges), consistent with batched version
        feasibility = 1.0 - (violations / max(total_edges, 1))

        if use_postprocessing:
            final_size, _ = greedy_decode(probs, edge_index, num_nodes)
        else:
            final_size = pred_size

        gap = optimal_size - final_size
        gap_ratio = gap / (optimal_size + 1e-8)
        approx_ratio = final_size / (optimal_size + 1e-8)
        return {
            "optimal_size": optimal_size,
            "size": final_size,
            "gap": gap,
            "gap_ratio": gap_ratio,
            "approx_ratio": approx_ratio,
            "feasibility": feasibility,
            "violations": violations_count,
        }

    # Compute per-graph metrics
    num_graphs = len(ptr) - 1
    total_optimal = 0
    total_postprocessed = 0
    total_gap = 0
    total_gap_ratio = 0
    total_approx_ratio = 0

    # Track individual graph values for min/max/std
    optimal_sizes = []
    final_sizes = []
    total_violations = 0
    total_edges = 0  # Track total edges for proper feasibility calculation

    for g in range(num_graphs):
        # Get node indices for this graph
        start, end = ptr[g].item(), ptr[g + 1].item()

        # Extract subgraph data
        node_mask = (batch_vec >= start) & (batch_vec < end)
        # Actually batch_vec contains graph IDs, not node indices
        node_mask = batch_vec == g

        graph_probs = probs[node_mask]
        graph_labels = labels[node_mask]

        # Get edges for this graph (edges where both nodes belong to graph g)
        edge_mask = (batch_vec[edge_index[0]] == g) & (batch_vec[edge_index[1]] == g)
        graph_edge_index = edge_index[:, edge_mask]

        # Remap edge indices to local node indices
        node_indices = torch.where(node_mask)[0]
        if len(node_indices) > 0:
            # Create mapping from global to local indices
            local_idx_map = torch.zeros(batch_vec.size(0), dtype=torch.long, device=probs.device)
            local_idx_map[node_indices] = torch.arange(len(node_indices), device=probs.device)
            graph_edge_index = local_idx_map[graph_edge_index]

        # Compute metrics for this graph
        graph_optimal = graph_labels.sum().item()
        graph_num_nodes = graph_probs.size(0)
        graph_preds_binary = (graph_probs > 0.5).float()
        graph_pred_size = graph_preds_binary.sum().item()

        # Compute violations for this graph
        # Note: edge_index contains each undirected edge twice (u,v) and (v,u)
        src, dst = graph_edge_index[0], graph_edge_index[1]
        pred_mask = graph_preds_binary == 1
        graph_num_edges = graph_edge_index.size(1)
        total_edges += graph_num_edges  # Always count edges

        if pred_mask.sum() > 0 and graph_num_edges > 0:
            # Count violations (each violating edge counted twice)
            graph_violations = (pred_mask[src] & pred_mask[dst]).sum().float().item()
        else:
            graph_violations = 0.0
        total_violations += graph_violations  # Keep as count (will normalize at end)

        if graph_num_nodes > 0 and graph_optimal > 0:
            if use_postprocessing:
                graph_final_size, _ = greedy_decode(graph_probs, graph_edge_index, graph_num_nodes)
            else:
                # No postprocessing: use prediction size
                # Note: This may be infeasible (violations exist), but that's the point of training without postprocessing - model must learn feasibility
                graph_final_size = graph_pred_size

            graph_gap = graph_optimal - graph_final_size
            graph_gap_ratio = graph_gap / graph_optimal
            graph_approx_ratio = graph_final_size / graph_optimal

            total_optimal += graph_optimal
            total_postprocessed += graph_final_size
            total_gap += graph_gap
            total_gap_ratio += graph_gap_ratio
            total_approx_ratio += graph_approx_ratio

            # Track for statistics
            optimal_sizes.append(graph_optimal)
            final_sizes.append(graph_final_size)

    # Average over graphs
    avg_optimal = total_optimal / max(num_graphs, 1)
    avg_gap = total_gap / max(num_graphs, 1)
    avg_gap_ratio = total_gap_ratio / max(num_graphs, 1)
    avg_approx_ratio = total_approx_ratio / max(num_graphs, 1)

    # Compute feasibility: violations / total_edges (normalized by total edges in batch)
    if total_edges > 0:
        feasibility = 1.0 - (total_violations / total_edges)
    else:
        feasibility = 1.0

    # Compute min/max for insight into variation
    min_optimal = min(optimal_sizes) if optimal_sizes else 0
    max_optimal = max(optimal_sizes) if optimal_sizes else 0
    min_final = min(final_sizes) if final_sizes else 0
    max_final = max(final_sizes) if final_sizes else 0

    return {
        "optimal_size": avg_optimal,
        "optimal_size_min": min_optimal,
        "optimal_size_max": max_optimal,
        "size": (probs > 0.5).float().sum().item() / max(num_graphs, 1),
        "size_min": min_final,
        "size_max": max_final,
        "gap": avg_gap,
        "gap_ratio": avg_gap_ratio,
        "approx_ratio": avg_approx_ratio,
        "feasibility": feasibility,
        "violations": total_violations,
    }


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


class LossConfig(BaseModel):
    # BCE loss weight for positive class (accounts for class imbalance)
    # pos_weight = neg_count / pos_count
    pos_weight: float = 1.0

    # Feasibility loss weight: penalizes selecting adjacent nodes
    feasibility_weight: float = 0.0


class ArchConfig(BaseModel):
    name: str = "graph_trm"
    # Matches 'hidden_size' (set to 256 to be safe on GPU, 512 is heavy for graphs)
    hidden_dim: int = 256
    # Matches 'L_layers' from your TRM config
    num_layers: int = 2
    # TRM recursion: H_cycles * L_cycles = total thinking steps
    L_cycles: int = 6
    H_cycles: int = 2
    dropout: float = 0.2  # Dropout in GPS layers
    attn_dropout: float = 0.2  # Attention dropout for multi-head attention


class Config(BaseModel):
    # Data
    data_paths: List[str] = ["data/mis-10k"]
    val_split: float = 0.1
    global_batch_size: int = 256

    # Training
    lr: float = 0.0001
    lr_min_ratio: float = 0.1  # Minimum LR ratio for cosine schedule
    lr_warmup_steps: int = 50
    epochs: int = 1000
    seed: int = 0  # Matches your config

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 0.5  # Gradient clipping for stability with deep recursion

    # Deep Supervision
    n_supervision: int = 1

    # Model
    arch: ArchConfig = ArchConfig()
    loss: LossConfig = LossConfig()

    # Postprocessing - When False, model must learn to produce valid MIS directly - When True, greedy decode post-processes raw predictions
    use_postprocessing: bool = False

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.999  # Standard decay for EMA

    # Logging
    project_name: str = "MIS-TRM"
    run_name: str = "mis_trm_v4"
    checkpoint_path: str = "checkpoints/mis"
    log_every: int = 1

    # Validation
    validate_every_epoch: bool = True  # Run validation after each epoch


def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.1,
):
    """Cosine learning rate schedule with warmup, matching pretrain.py"""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def init_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, world_size
    return 0, 1


def main():
    # 1. Setup
    rank, world_size = init_distributed()

    cfg = Config()

    if rank == 0:
        print("=" * 70)
        print("MIS-TRM Training")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Config: LR={cfg.lr}, WD={cfg.weight_decay}, Betas=({cfg.beta1}, {cfg.beta2}), GradClip={cfg.grad_clip}")
        print(f"Arch: Dim={cfg.arch.hidden_dim}, Layers={cfg.arch.num_layers}, H={cfg.arch.H_cycles}, L={cfg.arch.L_cycles}")
        print(f"Loss Weights: feasibility={cfg.loss.feasibility_weight}")
        print(f"Validation: {cfg.val_split * 100:.0f}% of data")
        pp_mode = "ON (greedy decode)" if cfg.use_postprocessing else "OFF (raw predictions)"
        print(f"Postprocessing: {pp_mode}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)

    torch.manual_seed(cfg.seed + rank)

    # 2. Data - Create train and validation datasets

    train_ds_config = MISDatasetConfig(
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        rank=rank,
        num_replicas=world_size,
        drop_last=True,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    train_dataset = MISDataset(train_ds_config, split="train")

    val_ds_config = MISDatasetConfig(
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        rank=rank,
        num_replicas=world_size,
        drop_last=False,  # Don't drop for validation
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    val_dataset = MISDataset(val_ds_config, split="val")

    # Pass 'input_dim', 'pos_weight', and loss weights from config to model
    model_config_dict = cfg.arch.model_dump()
    model_config_dict["input_dim"] = train_dataset.metadata.input_dim
    model_config_dict["pos_weight"] = train_dataset.metadata.pos_weight
    model_config_dict["feasibility_weight"] = cfg.loss.feasibility_weight
    model_config_dict["pe_input_dim"] = train_dataset.metadata.pe_dim  # Laplacian PE dimension

    if rank == 0:
        print("\\n📊 Dataset Summary:")
        print(f"  Total shards used: {len(train_dataset.shards)}")
        print(f"  Training graphs: {train_dataset.num_graphs}")
        print(f"  Validation graphs: {val_dataset.num_graphs}")
        print(f"  Val split: {cfg.val_split * 100:.0f}% ({val_dataset.num_graphs}/{train_dataset.num_graphs + val_dataset.num_graphs} graphs)")
        print(f"  Class imbalance: pos_weight={train_dataset.metadata.pos_weight:.2f}, class_ratio={train_dataset.metadata.class_ratio:.2%}")
        print(f"  Features: input_dim={train_dataset.metadata.input_dim}, pe_dim={train_dataset.metadata.pe_dim}")

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    # 3. Model configuration - all values from Config classes
    model_config_dict["pos_weight"] = cfg.loss.pos_weight
    model_config_dict["feasibility_weight"] = cfg.loss.feasibility_weight
    model_config_dict["dropout"] = cfg.arch.dropout
    model_config_dict["attn_dropout"] = cfg.arch.attn_dropout

    if rank == 0:
        print("\n🚀 Model Configuration:")
        print(f"  pos_weight: {cfg.loss.pos_weight}")
        print(f"  feasibility_weight: {cfg.loss.feasibility_weight}")
        print(f"  dropout: {cfg.arch.dropout}")
        print(f"  attn_dropout: {cfg.arch.attn_dropout}")
        print(f"  Training on {len(train_dataset.shards)} shards")

    model = GraphTransformerTRM(model_config_dict).cuda()
    raw_model = model

    # Initialize EMA (Exponential Moving Average) model
    ema_helper = None
    if cfg.use_ema:
        ema_helper = EMAHelper(mu=cfg.ema_decay)
        ema_helper.register(model)
        if rank == 0:
            print(f"📈 EMA enabled with decay={cfg.ema_decay}")

    # Count and log model parameters (Task 1.2)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n📊 Model Size:")
        print(f"  Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        raw_model = model.module

    # Initialize wandb after model creation to include param count
    if rank == 0:
        wandb_config = cfg.model_dump()
        wandb_config["model_params"] = total_params
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=wandb_config)

        # Log model architecture info
        wandb.log(
            {
                "model/total_params": total_params,
            }
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    # 4. Calculate actual steps for LR scheduling (based on real graph count)
    steps_per_epoch = train_dataset.num_graphs // cfg.global_batch_size
    val_steps_per_epoch = (val_dataset.num_graphs + cfg.global_batch_size - 1) // cfg.global_batch_size  # ceil division
    total_steps = steps_per_epoch * cfg.epochs

    if rank == 0:
        print(f"Steps per epoch: {steps_per_epoch} (train), {val_steps_per_epoch} (val)")
        print(f"Total training steps: {total_steps}")

    # 5. Training Loop
    step = 0

    for epoch in range(cfg.epochs):
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        model.train()
        train_dataset.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{cfg.epochs} - Training")
            print(f"{'=' * 70}")
            pbar = tqdm(total=steps_per_epoch, desc="Train")

        # Accumulators for epoch-level metrics
        epoch_metrics = {
            "loss_total": 0,
            "loss_bce": 0,
            "loss_feasibility": 0,
            "loss_bce_unweighted": 0,
            "loss_feasibility_unweighted": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "feasibility": 0,
            "size": 0,
            "optimal_size": 0,
            "gap": 0,
        }
        count = 0

        for batch_name, batch, batch_size in train_dataloader:
            step += 1

            # Compute LR with warmup + cosine schedule
            current_lr = cosine_schedule_with_warmup(
                current_step=step,
                base_lr=cfg.lr,
                num_warmup_steps=cfg.lr_warmup_steps,
                num_training_steps=total_steps,
                min_ratio=cfg.lr_min_ratio,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Move batch to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Initialize Carry
            carry = raw_model.initial_carry(batch)

            # Forward (Recursive Loop)
            all_finish = False
            final_loss = None
            final_metrics = None
            final_preds = None
            loop_step = 0

            while not all_finish:
                carry, step_loss, metrics, preds, all_finish = model(carry, batch)
                final_loss = step_loss
                final_metrics = metrics
                final_preds = preds

                loop_step += 1
                if loop_step >= cfg.n_supervision:
                    all_finish = True

            # Backward
            optimizer.zero_grad()
            final_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # Update EMA model after each optimizer step
            if ema_helper is not None:
                ema_helper.update(raw_model)

            # Compute post-processing metrics (per-graph averages)
            with torch.no_grad():
                pp_metrics = compute_postprocessing_metrics(
                    final_preds["preds"].squeeze(),
                    batch["edge_index"],
                    batch["y"].float(),
                    batch_vec=batch["batch"],
                    ptr=batch["ptr"],
                    use_postprocessing=cfg.use_postprocessing,
                )

            # Accumulate metrics
            epoch_metrics["loss_total"] += final_metrics["loss_total"].item()
            epoch_metrics["loss_bce"] += final_metrics["loss_bce"].item()
            epoch_metrics["loss_feasibility"] += final_metrics["loss_feasibility"].item()
            epoch_metrics["loss_bce_unweighted"] += final_metrics["loss_bce_unweighted"].item()
            epoch_metrics["loss_feasibility_unweighted"] += final_metrics["loss_feasibility_unweighted"].item()
            epoch_metrics["f1"] += final_metrics["f1"].item()
            epoch_metrics["precision"] += final_metrics["precision"].item()
            epoch_metrics["recall"] += final_metrics["recall"].item()
            epoch_metrics["feasibility"] += final_metrics["feasibility"].item()
            epoch_metrics["size"] += pp_metrics["size"]
            epoch_metrics["optimal_size"] += pp_metrics["optimal_size"]
            epoch_metrics["gap"] += pp_metrics["gap"]
            count += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{final_metrics['loss_total'].item():.4f}",
                        "bce": f"{final_metrics['loss_bce'].item():.4f}",
                        "f1": f"{final_metrics['f1'].item():.4f}",
                    }
                )

                if step % cfg.log_every == 0:
                    wandb.log(
                        {
                            # Standard metrics
                            "train/loss_total": final_metrics["loss_total"].item(),
                            "train/loss_bce": final_metrics["loss_bce"].item(),
                            "train/loss_feasibility": final_metrics["loss_feasibility"].item(),
                            # Unweighted losses for tuning weights
                            "train/loss_bce_unweighted": final_metrics["loss_bce_unweighted"].item(),
                            "train/loss_feasibility_unweighted": final_metrics["loss_feasibility_unweighted"].item(),
                            # Classification metrics
                            "train/f1": final_metrics["f1"].item(),
                            "train/precision": final_metrics["precision"].item(),
                            "train/recall": final_metrics["recall"].item(),
                            "train/feasibility": final_metrics["feasibility"].item(),
                            "train/approx_ratio": final_metrics["approx_ratio"].item(),
                            "train/num_violations": final_metrics["num_violations"].item(),
                            # Post-processing metrics
                            "train/size": pp_metrics["size"],
                            "train/optimal_size": pp_metrics["optimal_size"],
                            "train/gap": pp_metrics["gap"],
                            "train/gap_ratio": pp_metrics["gap_ratio"],
                            # Training info
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "epoch": epoch,
                            "lr": current_lr,
                        }
                    )

        if rank == 0:
            pbar.close()

            # Log epoch-level training metrics
            n = max(count, 1)

            print(f"\n📈 Training Epoch {epoch + 1} Summary:")
            print(
                f"   Loss: {epoch_metrics['loss_total'] / n:.4f} (BCE: {epoch_metrics['loss_bce_unweighted'] / n:.4f}, Feas: {epoch_metrics['loss_feasibility_unweighted'] / n:.4f})"
            )
            print(f"  Feasibility: {epoch_metrics['feasibility'] / n:.4f}")

            # Conditional logging based on use_postprocessing flag
            if cfg.use_postprocessing:
                print(f"  Post-processed: {epoch_metrics['size'] / n:.1f} / {epoch_metrics['optimal_size'] / n:.1f} (gap: {epoch_metrics['gap'] / n:.2f})")
            else:
                print(f"  Size: {epoch_metrics['size'] / n:.1f} / {epoch_metrics['optimal_size'] / n:.1f} (gap: {epoch_metrics['gap'] / n:.2f})")

            # Build log dict - always log normal metrics
            log_dict = {
                "train_epoch/loss_total": epoch_metrics["loss_total"] / n,
                "train_epoch/loss_bce": epoch_metrics["loss_bce_unweighted"] / n,
                "train_epoch/loss_feasibility": epoch_metrics["loss_feasibility_unweighted"] / n,
                "train_epoch/f1": epoch_metrics["f1"] / n,
                "epoch": epoch,
                # Always log normal metrics
                "train_epoch/feasibility": epoch_metrics["feasibility"] / n,
                "train_epoch/size": epoch_metrics["size"] / n,
                "train_epoch/approx_ratio": epoch_metrics["size"] / max(epoch_metrics["optimal_size"], 1),
                "train_epoch/gap": epoch_metrics["gap"] / n,
                "train_epoch/optimal_size": epoch_metrics["optimal_size"] / n,
            }

            # Add pp_ prefixed metrics when using post-processing
            if cfg.use_postprocessing:
                log_dict["train_epoch/pp_feasibility"] = epoch_metrics["feasibility"] / n
                log_dict["train_epoch/pp_size"] = epoch_metrics["size"] / n
                log_dict["train_epoch/pp_approx_ratio"] = epoch_metrics["size"] / max(epoch_metrics["optimal_size"], 1)
                log_dict["train_epoch/pp_gap"] = epoch_metrics["gap"] / n

            wandb.log(log_dict)

        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        if cfg.validate_every_epoch and rank == 0:

            def run_validation(eval_model, prefix="val", model_name="Model"):
                """Run validation and return metrics dict"""
                eval_model.eval()

                val_metrics = {
                    "loss_total": 0,
                    "loss_bce_unweighted": 0,
                    "loss_feasibility_unweighted": 0,
                    "f1": 0,
                    "precision": 0,
                    "recall": 0,
                    "feasibility": 0,
                    "size": 0,
                    "optimal_size": 0,
                    "gap": 0,
                }
                val_count = 0

                with torch.no_grad():
                    for batch_name, batch, batch_size in tqdm(
                        val_dataloader,
                        desc=f"Validate ({model_name})",
                        total=val_steps_per_epoch,
                    ):
                        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        carry = eval_model.initial_carry(batch)

                        # Run fixed number of supervision steps (no early stopping in validation)
                        for _ in range(cfg.n_supervision):
                            carry, step_loss, metrics, preds, _ = eval_model(carry, batch)

                        # Post-processing metrics (per-graph averages)
                        pp_metrics = compute_postprocessing_metrics(
                            preds["preds"].squeeze(),
                            batch["edge_index"],
                            batch["y"].float(),
                            batch_vec=batch["batch"],
                            ptr=batch["ptr"],
                            use_postprocessing=cfg.use_postprocessing,
                        )

                        val_metrics["loss_total"] += metrics["loss_total"].item()
                        val_metrics["loss_bce_unweighted"] += metrics["loss_bce_unweighted"].item()
                        val_metrics["loss_feasibility_unweighted"] += metrics["loss_feasibility_unweighted"].item()
                        val_metrics["f1"] += metrics["f1"].item()
                        val_metrics["precision"] += metrics["precision"].item()
                        val_metrics["recall"] += metrics["recall"].item()
                        val_metrics["feasibility"] += metrics["feasibility"].item()
                        val_metrics["size"] += pp_metrics["size"]
                        val_metrics["optimal_size"] += pp_metrics["optimal_size"]
                        val_metrics["gap"] += pp_metrics["gap"]
                        val_count += 1

                n = max(val_count, 1)
                return val_metrics, n

            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{cfg.epochs} - Validation")
            print(f"{'=' * 70}")

            # --- Regular Model Validation ---
            val_metrics, n = run_validation(raw_model, prefix="val", model_name="Regular")

            print(f"\n📊 Validation (Regular Model) Epoch {epoch + 1}:")
            print(f"Loss: {val_metrics['loss_total'] / n:.4f} (BCE: {val_metrics['loss_bce_unweighted'] / n:.4f}, Feas: {val_metrics['loss_feasibility_unweighted'] / n:.4f})")

            # Conditional logging based on use_postprocessing flag
            print(f"  Feasibility: {val_metrics['feasibility'] / n:.4f}")
            if cfg.use_postprocessing:
                print(f"  Post-processed: {val_metrics['size'] / n:.1f} / {val_metrics['optimal_size'] / n:.1f} (gap: {val_metrics['gap'] / n:.2f})")
                print(f"  Approx Ratio: {val_metrics['size'] / max(val_metrics['optimal_size'], 1):.4f}")
            else:
                print(f"  Size: {val_metrics['size'] / n:.1f} / {val_metrics['optimal_size'] / n:.1f} (gap: {val_metrics['gap'] / n:.2f})")

            # Build log dict - always log normal metrics
            val_log_dict = {
                "val/loss_total": val_metrics["loss_total"] / n,
                "val/loss_bce": val_metrics["loss_bce_unweighted"] / n,
                "val/loss_feasibility": val_metrics["loss_feasibility_unweighted"] / n,
                "val/f1": val_metrics["f1"] / n,
                "val/precision": val_metrics["precision"] / n,
                "val/recall": val_metrics["recall"] / n,
                "epoch": epoch,
                # Always log normal metrics
                "val/feasibility": val_metrics["feasibility"] / n,
                "val/size": val_metrics["size"] / n,
                "val/approx_ratio": val_metrics["size"] / max(val_metrics["optimal_size"], 1),
                "val/gap": val_metrics["gap"] / n,
                "val/optimal_size": val_metrics["optimal_size"] / n,
            }

            # Add pp_ prefixed metrics when using post-processing
            if cfg.use_postprocessing:
                val_log_dict["val/pp_feasibility"] = val_metrics["feasibility"] / n
                val_log_dict["val/pp_size"] = val_metrics["size"] / n
                val_log_dict["val/pp_approx_ratio"] = val_metrics["size"] / max(val_metrics["optimal_size"], 1)
                val_log_dict["val/pp_gap"] = val_metrics["gap"] / n

            wandb.log(val_log_dict)

            # --- EMA Model Validation ---
            if ema_helper is not None:
                # Create a copy of model with EMA weights
                ema_model = ema_helper.ema_copy(raw_model)
                ema_model.cuda()

                ema_val_metrics, ema_n = run_validation(ema_model, prefix="val_ema", model_name="EMA")

                print(f"\n📊 Validation (EMA Model) Epoch {epoch + 1}:")
                print(
                    f"Loss: {ema_val_metrics['loss_total'] / ema_n:.4f} (BCE: {ema_val_metrics['loss_bce_unweighted'] / ema_n:.4f}, Feas: {ema_val_metrics['loss_feasibility_unweighted'] / ema_n:.4f})"
                )
                print(f"  Feasibility: {ema_val_metrics['feasibility'] / ema_n:.4f}")
                print(f"  Post-processed: {ema_val_metrics['size'] / ema_n:.1f} / {ema_val_metrics['optimal_size'] / ema_n:.1f} (gap: {ema_val_metrics['gap'] / ema_n:.2f})")
                print(f"  Approx Ratio (post-processed): {ema_val_metrics['size'] / max(ema_val_metrics['optimal_size'], 1):.4f}")

                # Compare Regular vs EMA
                reg_f1 = val_metrics["f1"] / n
                ema_f1 = ema_val_metrics["f1"] / ema_n
                diff = ema_f1 - reg_f1
                print(f"\n  🔄 EMA vs Regular F1: {ema_f1:.4f} vs {reg_f1:.4f} (diff: {diff:+.4f})")

                ema_log_dict = {
                    "val_ema/loss_total": ema_val_metrics["loss_total"] / ema_n,
                    "val_ema/loss_bce": ema_val_metrics["loss_bce_unweighted"] / ema_n,
                    "val_ema/loss_feasibility": ema_val_metrics["loss_feasibility_unweighted"] / ema_n,
                    "val_ema/f1": ema_val_metrics["f1"] / ema_n,
                    "val_ema/precision": ema_val_metrics["precision"] / ema_n,
                    "val_ema/recall": ema_val_metrics["recall"] / ema_n,
                    # Always log normal metrics
                    "val_ema/feasibility": ema_val_metrics["feasibility"] / ema_n,
                    "val_ema/approx_ratio": ema_val_metrics["size"] / max(ema_val_metrics["optimal_size"], 1),
                    "val_ema/gap": ema_val_metrics["gap"] / ema_n,
                    "val_ema/size": ema_val_metrics["size"] / ema_n,
                    "val_ema/optimal_size": ema_val_metrics["optimal_size"] / ema_n,
                    "epoch": epoch,
                }

                if cfg.use_postprocessing:
                    ema_log_dict["val_ema/pp_feasibility"] = ema_val_metrics["feasibility"] / ema_n
                    ema_log_dict["val_ema/pp_size"] = ema_val_metrics["size"] / ema_n
                    ema_log_dict["val_ema/pp_approx_ratio"] = ema_val_metrics["size"] / max(ema_val_metrics["optimal_size"], 1)
                    ema_log_dict["val_ema/pp_gap"] = ema_val_metrics["gap"] / ema_n

                wandb.log(ema_log_dict)

                # Free EMA model copy
                del ema_model

            model.train()  # Set back to training mode

        # Save checkpoint every 50 epochs (and always save the last one)
        if rank == 0 and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
            torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}.pt")
            print(f"  💾 Checkpoint saved: epoch_{epoch}.pt")

            # Save EMA model separately
            if ema_helper is not None:
                torch.save(
                    ema_helper.state_dict(),
                    f"{cfg.checkpoint_path}/epoch_{epoch}_ema.pt",
                )
                print(f"  💾 EMA Checkpoint saved: epoch_{epoch}_ema.pt")

    if rank == 0:
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        wandb.finish()


if __name__ == "__main__":
    main()
