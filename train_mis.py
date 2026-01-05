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

# --- Configuration Classes ---
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

    # Logging
    project_name: str = "MIS-TRM"
    run_name: str = "mis_trm_v2_stable"
    checkpoint_path: str = "checkpoints/mis"
    log_every: int = 10

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

    # --- HYPERPARAMETERS SET HERE ---
    cfg = Config(
        data_paths=["data/mis-10k"],
        arch=ArchConfig()
    )

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Config: LR={cfg.lr}, WD={cfg.weight_decay}, Betas=({cfg.beta1}, {cfg.beta2})")
        print(f"Arch: Dim={cfg.arch.hidden_dim}, Layers={cfg.arch.num_layers}, Cycles={cfg.arch.cycles}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=cfg.model_dump())

    torch.manual_seed(cfg.seed + rank)

    # 2. Data
    ds_config = MISDatasetConfig(
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        rank=rank,
        num_replicas=world_size,
        drop_last=True  # Drop partial batches for consistent gradient updates
    )
    dataset = MISDataset(ds_config)

    # Pass 'input_dim' and 'pos_weight' from dataset metadata to model config
    model_config_dict = cfg.arch.model_dump()
    model_config_dict["input_dim"] = dataset.metadata.input_dim
    model_config_dict["pos_weight"] = dataset.metadata.pos_weight  # Global class imbalance

    if rank == 0:
        print(f"Dataset class imbalance: pos_weight={dataset.metadata.pos_weight:.2f}, "
              f"class_ratio={dataset.metadata.class_ratio:.2%}")

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # 3. Model
    model = GraphTRM(model_config_dict).cuda()

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # --- OPTIMIZER SETUP (MATCHING YOUR YAML) ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2)
    )

    # 4. Estimate total steps for LR scheduling
    # Approx: 10k samples / batch_size * epochs
    estimated_samples = 10000  # Adjust based on your dataset
    steps_per_epoch = estimated_samples // cfg.global_batch_size
    total_steps = steps_per_epoch * cfg.epochs

    if rank == 0:
        print(f"Estimated total steps: {total_steps}")

    # 5. Training Loop
    step = 0
    model.train()

    for epoch in range(cfg.epochs):
        # Update dataset epoch for different shuffle each epoch
        dataset.set_epoch(epoch)

        if rank == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs}")
            pbar = tqdm(total=steps_per_epoch)

        epoch_loss = 0
        epoch_acc = 0
        count = 0

        for batch_name, batch, batch_size in dataloader:
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
            raw_model = model.module if hasattr(model, "module") else model
            carry = raw_model.initial_carry(batch)

            # Forward (Recursive Loop)
            # KEY FIX: Only use FINAL loss, not accumulated loss over all steps
            # This prevents gradient explosion from 18x loss accumulation
            all_finish = False
            final_loss = None
            final_metrics = None

            while not all_finish:
                carry, step_loss, metrics, preds, all_finish = model(carry, batch)
                final_loss = step_loss
                final_metrics = metrics

            # Don't normalize by batch size - BCE loss is already averaged over nodes
            # Dividing by 256 was crushing gradients to 0.018

            # Backward
            optimizer.zero_grad()
            final_loss.backward()

            # Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            # Logging (use un-normalized loss for display)
            epoch_loss += final_loss.item()
            epoch_acc += final_metrics["f1"].item()  # Track F1 instead of misleading acc
            count += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{final_metrics['loss_total'].item():.4f}",
                    "f1": f"{final_metrics['f1'].item():.4f}",
                    "feasibility": f"{final_metrics['feasibility'].item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })

                if step % cfg.log_every == 0:
                    wandb.log({
                        "train/loss_total": final_metrics["loss_total"].item(),
                        "train/loss_bce": final_metrics["loss_bce"].item(),
                        "train/loss_feasibility": final_metrics["loss_feasibility"].item(),
                        "train/loss_sparsity": final_metrics["loss_sparsity"].item(),
                        "train/f1": final_metrics["f1"].item(),
                        "train/precision": final_metrics["precision"].item(),
                        "train/recall": final_metrics["recall"].item(),
                        "train/num_pred_1s": final_metrics["num_pred_1s"].item(),
                        "train/num_true_1s": final_metrics["num_true_1s"].item(),
                        "train/set_size_ratio": final_metrics["set_size_ratio"].item(),
                        "train/feasibility": final_metrics["feasibility"].item(),
                        "train/approx_ratio": final_metrics["approx_ratio"].item(),
                        "train/num_violations": final_metrics["num_violations"].item(),
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "epoch": epoch,
                        "lr": current_lr
                    })

        if rank == 0:
            pbar.close()
            # Save Checkpoint
            torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}.pt")
            print(f"Mean Epoch Loss: {epoch_loss/max(count,1):.4f} | Mean F1: {epoch_acc/max(count,1):.4f}")

    if rank == 0:
        print("Done.")
        wandb.finish()

if __name__ == "__main__":
    main()
