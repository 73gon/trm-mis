import os
import sys
import yaml
import time
import shutil
import math
import torch
import torch.distributed as dist
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Optional

# --- Import our new modules ---
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.graph_trm import GraphTRM


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count model parameters and return detailed breakdown"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-module breakdown
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "breakdown": breakdown
    }


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


def compute_postprocessing_metrics(probs, edge_index, labels, batch_ptr=None):
    """
    Compute post-processing metrics using greedy decode.

    Returns dict with:
    - optimal_size: Ground truth MIS size
    - raw_pred_size: Number of nodes with prob > 0.5 (may be infeasible)
    - postprocessed_size: Size after greedy decode (always feasible)
    - gap: optimal_size - postprocessed_size
    - gap_ratio: gap / optimal_size
    """
    num_nodes = probs.size(0)
    preds_binary = (probs > 0.5).float()

    raw_pred_size = preds_binary.sum().item()
    optimal_size = labels.sum().item()

    # Greedy decode to get feasible set
    postprocessed_size, _ = greedy_decode(probs, edge_index, num_nodes)

    gap = optimal_size - postprocessed_size
    gap_ratio = gap / (optimal_size + 1e-8)
    approx_ratio = postprocessed_size / (optimal_size + 1e-8)

    return {
        "optimal_size": optimal_size,
        "raw_pred_size": raw_pred_size,
        "postprocessed_size": postprocessed_size,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "approx_ratio_postprocessed": approx_ratio,
    }

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class LossConfig(BaseModel):
    """
    Centralized loss weight configuration for easy experimentation.
    Professor's advice: tune weights so bce_loss and feasibility_weight * feasibility_loss
    are approximately the same magnitude.
    """
    # Feasibility loss weight: penalizes selecting adjacent nodes
    # Increase if model produces too many violations (infeasible solutions)
    # Typical range: 20.0 - 100.0 depending on loss magnitudes
    feasibility_weight: float = 50.0


class ArchConfig(BaseModel):
    name: str = "graph_trm"
    # Matches 'hidden_size' (set to 256 to be safe on GPU, 512 is heavy for graphs)
    hidden_dim: int = 256
    # Matches 'L_layers' from your TRM config
    num_layers: int = 2
    # Matches 'H_cycles' (3) * 'L_cycles' (6) = 18 total steps
    cycles: int = 18


class Config(BaseModel):
    # Data
    data_paths: List[str]
    val_split: float = 0.1  # 10% for validation
    # Increase batch size to utilize full GPU memory (was 64, now 256)
    global_batch_size: int = 256

    # Training - Matches 'cfg_pretrain' settings
    lr: float = 1e-3          # Increased from 3e-4 to prevent vanishing gradients
    lr_min_ratio: float = 0.1 # Minimum LR ratio for cosine schedule
    lr_warmup_steps: int = 200 # Warmup steps
    epochs: int = 100         # MIS learns faster than ARC, 100 epochs is usually plenty
    seed: int = 0             # Matches your config

    # Optimizer - Matches 'cfg_pretrain' Llama-style settings
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0    # Increased from 0.5 for better gradient flow

    # Model
    arch: ArchConfig

    # Loss weights (centralized for easy experimentation)
    loss: LossConfig = LossConfig()

    # Logging
    project_name: str = "MIS-TRM"
    run_name: str = "mis_trm_v2_stable"
    checkpoint_path: str = "checkpoints/mis"
    log_every: int = 10

    # Validation
    validate_every_epoch: bool = True  # Run validation after each epoch

def cosine_schedule_with_warmup(current_step: int, base_lr: float, num_warmup_steps: int,
                                 num_training_steps: int, min_ratio: float = 0.1):
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

    # =========================================================================
    # HYPERPARAMETERS - EDIT HERE FOR EXPERIMENTS
    # =========================================================================
    cfg = Config(
        data_paths=["data/mis-10k"],
        val_split=0.1,  # 10% for validation
        arch=ArchConfig(),
        # Loss weights - tune these so BCE and weighted feasibility are ~same magnitude
        loss=LossConfig(
            feasibility_weight=50.0,  # Try: 1.0, 5.0, 10.0, 20.0, 50.0
            sparsity_weight=0.3,     # Try: 0.0, 0.3, 1.0 (0.0 for ablation)
        ),
    )

    if rank == 0:
        print("=" * 70)
        print("MIS-TRM Training")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Config: LR={cfg.lr}, WD={cfg.weight_decay}, Betas=({cfg.beta1}, {cfg.beta2})")
        print(f"Arch: Dim={cfg.arch.hidden_dim}, Layers={cfg.arch.num_layers}, Cycles={cfg.arch.cycles}")
        print(f"Loss Weights: feasibility={cfg.loss.feasibility_weight}, sparsity={cfg.loss.sparsity_weight}")
        print(f"Validation: {cfg.val_split*100:.0f}% of data")
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
    model_config_dict["sparsity_weight"] = cfg.loss.sparsity_weight

    if rank == 0:
        print(f"Train shards: {len(train_dataset.shards)}, Val shards: {len(val_dataset.shards)}")
        print(f"Dataset class imbalance: pos_weight={train_dataset.metadata.pos_weight:.2f}, "
              f"class_ratio={train_dataset.metadata.class_ratio:.2%}")

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    # 3. Model
    model = GraphTRM(model_config_dict).cuda()
    raw_model = model

    # Count and log model parameters (Task 1.2)
    if rank == 0:
        param_info = count_parameters(model)
        print(f"\n📊 Model Size:")
        print(f"  Total Parameters: {param_info['total_params']:,} ({param_info['total_params']/1e6:.2f}M)")
        print(f"  Trainable Parameters: {param_info['trainable_params']:,}")
        print(f"  Breakdown:")
        for name, count in param_info['breakdown'].items():
            print(f"    {name}: {count:,}")
        print()

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        raw_model = model.module

    # Initialize wandb after model creation to include param count
    if rank == 0:
        wandb_config = cfg.model_dump()
        wandb_config["model_params"] = param_info
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=wandb_config)

        # Log model architecture info
        wandb.log({
            "model/total_params": param_info['total_params'],
            "model/trainable_params": param_info['trainable_params'],
            "model/hidden_dim": cfg.arch.hidden_dim,
            "model/num_layers": cfg.arch.num_layers,
            "model/cycles": cfg.arch.cycles,
        })

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2)
    )

    # 4. Estimate total steps for LR scheduling
    estimated_samples = 10000 * (1 - cfg.val_split)  # Training samples only
    steps_per_epoch = int(estimated_samples // cfg.global_batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    if rank == 0:
        print(f"Estimated steps per epoch: {steps_per_epoch}, total: {total_steps}")

    # 5. Training Loop
    step = 0

    for epoch in range(cfg.epochs):
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        model.train()
        train_dataset.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{cfg.epochs} - Training")
            print(f"{'='*70}")
            pbar = tqdm(total=steps_per_epoch, desc="Train")

        # Accumulators for epoch-level metrics
        epoch_metrics = {
            "loss_total": 0, "loss_bce": 0, "loss_feasibility": 0,
            "loss_bce_raw": 0, "loss_feasibility_raw": 0,
            "f1": 0, "precision": 0, "recall": 0, "feasibility": 0,
            "postprocessed_size": 0, "optimal_size": 0, "gap": 0,
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
                min_ratio=cfg.lr_min_ratio
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Move batch to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            # Initialize Carry
            carry = raw_model.initial_carry(batch)

            # Forward (Recursive Loop)
            all_finish = False
            final_loss = None
            final_metrics = None
            final_preds = None

            while not all_finish:
                carry, step_loss, metrics, preds, all_finish = model(carry, batch)
                final_loss = step_loss
                final_metrics = metrics
                final_preds = preds

            # Backward
            optimizer.zero_grad()
            final_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # Compute post-processing metrics (Task 1.3)
            with torch.no_grad():
                pp_metrics = compute_postprocessing_metrics(
                    final_preds["preds"].squeeze(),
                    batch["edge_index"],
                    batch["y"].float()
                )

            # Accumulate metrics
            epoch_metrics["loss_total"] += final_metrics["loss_total"].item()
            epoch_metrics["loss_bce"] += final_metrics["loss_bce"].item()
            epoch_metrics["loss_feasibility"] += final_metrics["loss_feasibility"].item()
            epoch_metrics["loss_bce_raw"] += final_metrics["loss_bce_raw"].item()
            epoch_metrics["loss_feasibility_raw"] += final_metrics["loss_feasibility_raw"].item()
            epoch_metrics["f1"] += final_metrics["f1"].item()
            epoch_metrics["precision"] += final_metrics["precision"].item()
            epoch_metrics["recall"] += final_metrics["recall"].item()
            epoch_metrics["feasibility"] += final_metrics["feasibility"].item()
            epoch_metrics["postprocessed_size"] += pp_metrics["postprocessed_size"]
            epoch_metrics["optimal_size"] += pp_metrics["optimal_size"]
            epoch_metrics["gap"] += pp_metrics["gap"]
            count += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{final_metrics['loss_total'].item():.4f}",
                    "bce": f"{final_metrics['loss_bce_raw'].item():.4f}",
                    "feas": f"{final_metrics['loss_feasibility_raw'].item():.4f}",
                    "f1": f"{final_metrics['f1'].item():.4f}",
                })

                if step % cfg.log_every == 0:
                    wandb.log({
                        # Standard metrics
                        "train/loss_total": final_metrics["loss_total"].item(),
                        "train/loss_bce": final_metrics["loss_bce"].item(),
                        "train/loss_feasibility": final_metrics["loss_feasibility"].item(),
                        # Raw (unweighted) losses for tuning weights
                        "train/loss_bce_raw": final_metrics["loss_bce_raw"].item(),
                        "train/loss_feasibility_raw": final_metrics["loss_feasibility_raw"].item(),
                        # Weighted feasibility loss (for comparison)
                        "train/loss_feasibility_weighted": final_metrics["loss_feasibility_weighted"].item(),
                        # Classification metrics
                        "train/f1": final_metrics["f1"].item(),
                        "train/precision": final_metrics["precision"].item(),
                        "train/recall": final_metrics["recall"].item(),
                        "train/feasibility": final_metrics["feasibility"].item(),
                        "train/approx_ratio": final_metrics["approx_ratio"].item(),
                        "train/num_violations": final_metrics["num_violations"].item(),
                        # Post-processing metrics
                        "train/postprocessed_size": pp_metrics["postprocessed_size"],
                        "train/optimal_size": pp_metrics["optimal_size"],
                        "train/gap": pp_metrics["gap"],
                        "train/gap_ratio": pp_metrics["gap_ratio"],
                        "train/approx_ratio_postprocessed": pp_metrics["approx_ratio_postprocessed"],
                        # Training info
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "epoch": epoch,
                        "lr": current_lr,
                    })

        if rank == 0:
            pbar.close()

            # Log epoch-level training metrics
            n = max(count, 1)
            print(f"\n📈 Training Epoch {epoch+1} Summary:")
            print(f"  Loss: {epoch_metrics['loss_total']/n:.4f} (BCE: {epoch_metrics['loss_bce_raw']/n:.4f}, Feas: {epoch_metrics['loss_feasibility_raw']/n:.4f})")
            print(f"  F1: {epoch_metrics['f1']/n:.4f}, Precision: {epoch_metrics['precision']/n:.4f}, Recall: {epoch_metrics['recall']/n:.4f}")
            print(f"  Feasibility: {epoch_metrics['feasibility']/n:.4f}")
            print(f"  Post-processed: {epoch_metrics['postprocessed_size']/n:.1f} / {epoch_metrics['optimal_size']/n:.1f} (gap: {epoch_metrics['gap']/n:.2f})")

            wandb.log({
                "train_epoch/loss_total": epoch_metrics["loss_total"] / n,
                "train_epoch/loss_bce_raw": epoch_metrics["loss_bce_raw"] / n,
                "train_epoch/loss_feasibility_raw": epoch_metrics["loss_feasibility_raw"] / n,
                "train_epoch/f1": epoch_metrics["f1"] / n,
                "train_epoch/feasibility": epoch_metrics["feasibility"] / n,
                "train_epoch/approx_ratio_postprocessed": epoch_metrics["postprocessed_size"] / max(epoch_metrics["optimal_size"], 1),
                "train_epoch/gap": epoch_metrics["gap"] / n,
                "epoch": epoch,
            })

        # =====================================================================
        # VALIDATION PHASE (Task 1.1)
        # =====================================================================
        if cfg.validate_every_epoch and rank == 0:
            model.eval()

            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{cfg.epochs} - Validation")
            print(f"{'='*70}")

            val_metrics = {
                "loss_total": 0, "loss_bce_raw": 0, "loss_feasibility_raw": 0,
                "f1": 0, "precision": 0, "recall": 0, "feasibility": 0,
                "postprocessed_size": 0, "optimal_size": 0, "gap": 0,
            }
            val_count = 0

            with torch.no_grad():
                for batch_name, batch, batch_size in tqdm(val_dataloader, desc="Validate"):
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

                    carry = raw_model.initial_carry(batch)
                    all_finish = False

                    while not all_finish:
                        carry, step_loss, metrics, preds, all_finish = model(carry, batch)

                    # Post-processing metrics
                    pp_metrics = compute_postprocessing_metrics(
                        preds["preds"].squeeze(),
                        batch["edge_index"],
                        batch["y"].float()
                    )

                    val_metrics["loss_total"] += metrics["loss_total"].item()
                    val_metrics["loss_bce_raw"] += metrics["loss_bce_raw"].item()
                    val_metrics["loss_feasibility_raw"] += metrics["loss_feasibility_raw"].item()
                    val_metrics["f1"] += metrics["f1"].item()
                    val_metrics["precision"] += metrics["precision"].item()
                    val_metrics["recall"] += metrics["recall"].item()
                    val_metrics["feasibility"] += metrics["feasibility"].item()
                    val_metrics["postprocessed_size"] += pp_metrics["postprocessed_size"]
                    val_metrics["optimal_size"] += pp_metrics["optimal_size"]
                    val_metrics["gap"] += pp_metrics["gap"]
                    val_count += 1

            n = max(val_count, 1)
            print(f"\n📊 Validation Epoch {epoch+1} Summary:")
            print(f"  Loss: {val_metrics['loss_total']/n:.4f} (BCE: {val_metrics['loss_bce_raw']/n:.4f}, Feas: {val_metrics['loss_feasibility_raw']/n:.4f})")
            print(f"  F1: {val_metrics['f1']/n:.4f}, Precision: {val_metrics['precision']/n:.4f}, Recall: {val_metrics['recall']/n:.4f}")
            print(f"  Feasibility: {val_metrics['feasibility']/n:.4f}")
            print(f"  Post-processed: {val_metrics['postprocessed_size']/n:.1f} / {val_metrics['optimal_size']/n:.1f} (gap: {val_metrics['gap']/n:.2f})")
            print(f"  Approx Ratio (post-processed): {val_metrics['postprocessed_size'] / max(val_metrics['optimal_size'], 1):.4f}")

            wandb.log({
                "val/loss_total": val_metrics["loss_total"] / n,
                "val/loss_bce_raw": val_metrics["loss_bce_raw"] / n,
                "val/loss_feasibility_raw": val_metrics["loss_feasibility_raw"] / n,
                "val/f1": val_metrics["f1"] / n,
                "val/precision": val_metrics["precision"] / n,
                "val/recall": val_metrics["recall"] / n,
                "val/feasibility": val_metrics["feasibility"] / n,
                "val/approx_ratio_postprocessed": val_metrics["postprocessed_size"] / max(val_metrics["optimal_size"], 1),
                "val/gap": val_metrics["gap"] / n,
                "val/postprocessed_size": val_metrics["postprocessed_size"] / n,
                "val/optimal_size": val_metrics["optimal_size"] / n,
                "epoch": epoch,
            })

        # Save checkpoint
        if rank == 0:
            torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}.pt")

    if rank == 0:
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        wandb.finish()


if __name__ == "__main__":
    main()
