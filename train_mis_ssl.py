"""
Self-Supervised Training for Maximum Independent Set (MIS)

This training script uses self-supervised learning - NO ground truth labels needed!

The model learns to find MIS solutions through two loss signals:
1. Feasibility Loss: Penalize selecting adjacent nodes (exponential penalty)
2. Selection Loss: Encourage selecting as many nodes as possible

Key differences from supervised version:
- No BCE loss (no labels)
- No accuracy metrics (no ground truth to compare against)
- No opt_size, gap, approx_ratio (all require ground truth)
- Metrics focus on: pred_size, feasibility, num_violations

The architecture and training loop remain similar to the supervised version.
"""

import math
import os
from typing import List

import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.ema import EMAHelper
from models.graph_transformer_trm_ssl import GraphTransformerTRM_SSL
from models.metrics_ssl import compute_metrics_ssl, compute_pp_metrics_ssl

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


class LossConfig(BaseModel):
    """Self-supervised loss configuration"""

    # Exponential penalty parameter (mu)
    # Higher = stricter constraint enforcement, but harder to optimize
    # Lower = softer constraints, easier optimization but may have more violations
    mu: float = 5.0

    # Weight for feasibility loss (exponential penalty term)
    feasibility_weight: float = 2.0

    # Weight for selection loss (maximize set size term)
    selection_weight: float = 5.0


class ArchConfig(BaseModel):
    name: str = "graph_trm_ssl"
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
    lr: float = 1e-4
    lr_min_ratio: float = 0.1  # Minimum LR ratio for cosine schedule
    lr_warmup_steps: int = 50
    epochs: int = 500
    seed: int = 0

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 0.5  # Gradient clipping for stability with deep recursion

    # Deep Supervision
    n_supervision: int = 1

    # Model
    arch: ArchConfig = ArchConfig()
    loss: LossConfig = LossConfig()

    use_postprocessing: bool = True

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.999

    # Logging
    project_name: str = "MIS-TRM"
    run_name: str = "mis_trm_ssl_s5_f1_log"
    checkpoint_path: str = "checkpoints/mis_ssl"
    log_every: int = 1

    # Validation
    validate_every_epoch: bool = True


def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.1,
):
    """Cosine learning rate schedule with warmup"""
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
        print("MIS-TRM Self-Supervised Training")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Config: LR={cfg.lr}, WD={cfg.weight_decay}, Betas=({cfg.beta1}, {cfg.beta2}), GradClip={cfg.grad_clip}")
        print(f"Arch: Dim={cfg.arch.hidden_dim}, Layers={cfg.arch.num_layers}, H={cfg.arch.H_cycles}, L={cfg.arch.L_cycles}")
        print(f"Loss (SSL): mu={cfg.loss.mu}, feasibility_w={cfg.loss.feasibility_weight}, selection_w={cfg.loss.selection_weight}")
        print(f"Validation: {cfg.val_split * 100:.0f}% of data")
        pp_mode = "ON (greedy decode)" if cfg.use_postprocessing else "OFF (raw predictions)"
        print(f"Postprocessing: {pp_mode}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)

    torch.manual_seed(cfg.seed + rank)

    # 2. Data - Create train and validation datasets
    # Note: We still use the same dataset, but we won't use the labels for training

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
        drop_last=False,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    val_dataset = MISDataset(val_ds_config, split="val")

    # Build model config
    model_config_dict = cfg.arch.model_dump()
    model_config_dict["input_dim"] = train_dataset.metadata.input_dim
    model_config_dict["pe_input_dim"] = train_dataset.metadata.pe_dim

    # Self-supervised loss parameters
    model_config_dict["mu"] = cfg.loss.mu
    model_config_dict["feasibility_weight"] = cfg.loss.feasibility_weight
    model_config_dict["selection_weight"] = cfg.loss.selection_weight
    model_config_dict["dropout"] = cfg.arch.dropout
    model_config_dict["attn_dropout"] = cfg.arch.attn_dropout

    if rank == 0:
        print("\n📊 Dataset Summary:")
        print(f"  Total shards used: {len(train_dataset.shards)}")
        print(f"  Training graphs: {train_dataset.num_graphs}")
        print(f"  Validation graphs: {val_dataset.num_graphs}")
        print(f"  Val split: {cfg.val_split * 100:.0f}% ({val_dataset.num_graphs}/{train_dataset.num_graphs + val_dataset.num_graphs} graphs)")
        print(f"  Features: input_dim={train_dataset.metadata.input_dim}, pe_dim={train_dataset.metadata.pe_dim}")
        print("  Note: Labels exist in dataset but NOT used for training (self-supervised)")

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    if rank == 0:
        print("\n🚀 Model Configuration (Self-Supervised):")
        print(f"  mu (penalty param): {cfg.loss.mu}")
        print(f"  feasibility_weight: {cfg.loss.feasibility_weight}")
        print(f"  selection_weight: {cfg.loss.selection_weight}")
        print(f"  dropout: {cfg.arch.dropout}")
        print(f"  attn_dropout: {cfg.arch.attn_dropout}")
        print(f"  Training on {len(train_dataset.shards)} shards")

    # 3. Create model
    model = GraphTransformerTRM_SSL(model_config_dict).cuda()
    raw_model = model

    # Initialize EMA
    ema_helper = None
    if cfg.use_ema:
        ema_helper = EMAHelper(mu=cfg.ema_decay)
        ema_helper.register(model)
        if rank == 0:
            print(f"  EMA enabled with decay={cfg.ema_decay}")

    # Count model parameters
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n📊 Model Size:")
        print(f"  Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        raw_model = model.module

    # Initialize wandb
    if rank == 0:
        wandb_config = cfg.model_dump()
        wandb_config["model_params"] = total_params
        wandb_config["training_mode"] = "self-supervised"
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=wandb_config)

        wandb.log({"model/total_params": total_params})

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    # 4. Calculate steps for LR scheduling
    steps_per_epoch = train_dataset.num_graphs // cfg.global_batch_size
    val_steps_per_epoch = (val_dataset.num_graphs + cfg.global_batch_size - 1) // cfg.global_batch_size
    total_steps = steps_per_epoch * cfg.epochs

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
            print(f"Epoch {epoch + 1}/{cfg.epochs} - Training (Self-Supervised)")
            print(f"{'=' * 70}")
            pbar = tqdm(total=steps_per_epoch, desc="Train")

        # Accumulators for epoch-level metrics (self-supervised)
        epoch_metrics = {
            "loss_total": 0,
            "loss_feasibility": 0,
            "loss_selection": 0,
            "pred_size": 0,
            "feasibility": 0,
            "opt_size": 0,
            "gap": 0,
            "approx_ratio": 0,
        }
        # PP metrics (when use_postprocessing=True)
        epoch_pp_metrics = {
            "pp_pred_size": 0,
            "pp_feasibility": 0,
            "pp_gap": 0,
            "pp_approx_ratio": 0,
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

            # Move batch to GPU (labels present but not used)
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

            # Gradient clipping
            if cfg.grad_clip > 0:
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            else:
                final_loss.backward()

            optimizer.step()

            # Update EMA
            if ema_helper is not None:
                ema_helper.update(raw_model)

            # Compute metrics (labels used for comparison only, not training)
            with torch.no_grad():
                raw_metrics = compute_metrics_ssl(
                    final_preds["preds"].squeeze(),
                    batch["edge_index"],
                    batch["y"].float(),
                    batch_vec=batch["batch"],
                    ptr=batch["ptr"],
                )

                if cfg.use_postprocessing:
                    pp_metrics = compute_pp_metrics_ssl(
                        final_preds["preds"].squeeze(),
                        batch["edge_index"],
                        batch["y"].float(),
                        batch_vec=batch["batch"],
                        ptr=batch["ptr"],
                    )

            # Accumulate metrics
            epoch_metrics["loss_total"] += final_metrics["loss_total"].item()
            epoch_metrics["loss_feasibility"] += final_metrics["loss_feasibility"].item()
            epoch_metrics["loss_selection"] += final_metrics["loss_selection"].item()
            epoch_metrics["pred_size"] += raw_metrics["pred_size"]
            epoch_metrics["feasibility"] += raw_metrics["feasibility"]
            epoch_metrics["opt_size"] += raw_metrics["opt_size"]
            epoch_metrics["gap"] += raw_metrics["gap"]
            epoch_metrics["approx_ratio"] += raw_metrics["approx_ratio"]

            if cfg.use_postprocessing:
                epoch_pp_metrics["pp_pred_size"] += pp_metrics["pp_pred_size"]
                epoch_pp_metrics["pp_feasibility"] += pp_metrics["pp_feasibility"]
                epoch_pp_metrics["pp_gap"] += pp_metrics["pp_gap"]
                epoch_pp_metrics["pp_approx_ratio"] += pp_metrics["pp_approx_ratio"]

            count += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{final_metrics['loss_total'].item():.4f}",
                        "feas": f"{raw_metrics['feasibility']:.3f}",
                        "size": f"{raw_metrics['pred_size']:.1f}",
                    }
                )

                if step % cfg.log_every == 0:
                    log_data = {
                        # Loss metrics (self-supervised)
                        "train/loss_total": final_metrics["loss_total"].item(),
                        "train/loss_feasibility": final_metrics["loss_feasibility"].item(),
                        "train/loss_selection": final_metrics["loss_selection"].item(),
                        # Raw model metrics
                        "train/pred_size": raw_metrics["pred_size"],
                        "train/feasibility": raw_metrics["feasibility"],
                        # Comparison metrics (for visualization only)
                        "train/opt_size": raw_metrics["opt_size"],
                        "train/gap": raw_metrics["gap"],
                        "train/approx_ratio": raw_metrics["approx_ratio"],
                        # Training info
                        "epoch": epoch,
                        "lr": current_lr,
                    }

                    if cfg.use_postprocessing:
                        log_data.update(
                            {
                                "train/pp_pred_size": pp_metrics["pp_pred_size"],
                                "train/pp_feasibility": pp_metrics["pp_feasibility"],
                                "train/pp_gap": pp_metrics["pp_gap"],
                                "train/pp_approx_ratio": pp_metrics["pp_approx_ratio"],
                            }
                        )

                    wandb.log(log_data)

        if rank == 0:
            pbar.close()

            # Log epoch-level training metrics
            n = max(count, 1)

            print(f"\n📈 Training Epoch {epoch + 1} Summary (Self-Supervised):")
            print(f"   Loss: {epoch_metrics['loss_total'] / n:.4f} (Feas: {epoch_metrics['loss_feasibility'] / n:.4f}, Sel: {epoch_metrics['loss_selection'] / n:.4f})")
            print("  Raw Metrics:")
            print(f"    Pred Size: {epoch_metrics['pred_size'] / n:.1f} / Opt Size: {epoch_metrics['opt_size'] / n:.1f}")
            print(f"    Gap: {epoch_metrics['gap'] / n:.2f}, Approx Ratio: {epoch_metrics['approx_ratio'] / n:.4f}")
            print(f"    Feasibility: {epoch_metrics['feasibility'] / n:.4f}")

            if cfg.use_postprocessing:
                print("  PP Metrics (greedy-decoded):")
                print(f"    PP Pred Size: {epoch_pp_metrics['pp_pred_size'] / n:.1f}")
                print(f"    PP Gap: {epoch_pp_metrics['pp_gap'] / n:.2f}, PP Approx Ratio: {epoch_pp_metrics['pp_approx_ratio'] / n:.4f}")
                print(f"    PP Feasibility: {epoch_pp_metrics['pp_feasibility'] / n:.4f}")

            log_dict = {
                "train_epoch/loss_total": epoch_metrics["loss_total"] / n,
                "train_epoch/loss_feasibility": epoch_metrics["loss_feasibility"] / n,
                "train_epoch/loss_selection": epoch_metrics["loss_selection"] / n,
                "epoch": epoch,
                "train_epoch/pred_size": epoch_metrics["pred_size"] / n,
                "train_epoch/feasibility": epoch_metrics["feasibility"] / n,
                "train_epoch/opt_size": epoch_metrics["opt_size"] / n,
                "train_epoch/gap": epoch_metrics["gap"] / n,
                "train_epoch/approx_ratio": epoch_metrics["approx_ratio"] / n,
            }

            if cfg.use_postprocessing:
                log_dict.update(
                    {
                        "train_epoch/pp_pred_size": epoch_pp_metrics["pp_pred_size"] / n,
                        "train_epoch/pp_feasibility": epoch_pp_metrics["pp_feasibility"] / n,
                        "train_epoch/pp_gap": epoch_pp_metrics["pp_gap"] / n,
                        "train_epoch/pp_approx_ratio": epoch_pp_metrics["pp_approx_ratio"] / n,
                    }
                )

            wandb.log(log_dict)

        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        if cfg.validate_every_epoch and rank == 0:

            def run_validation(eval_model, prefix="val", model_name="Model"):
                """Run validation and return metrics dict (self-supervised)"""
                eval_model.eval()

                val_metrics = {
                    "loss_total": 0,
                    "loss_feasibility": 0,
                    "loss_selection": 0,
                    "pred_size": 0,
                    "feasibility": 0,
                    "opt_size": 0,
                    "gap": 0,
                    "approx_ratio": 0,
                }
                val_pp_metrics = {
                    "pp_pred_size": 0,
                    "pp_feasibility": 0,
                    "pp_gap": 0,
                    "pp_approx_ratio": 0,
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

                        for _ in range(cfg.n_supervision):
                            carry, step_loss, metrics, preds, _ = eval_model(carry, batch)

                        raw_metrics = compute_metrics_ssl(
                            preds["preds"].squeeze(),
                            batch["edge_index"],
                            batch["y"].float(),
                            batch_vec=batch["batch"],
                            ptr=batch["ptr"],
                        )

                        if cfg.use_postprocessing:
                            pp_metrics = compute_pp_metrics_ssl(
                                preds["preds"].squeeze(),
                                batch["edge_index"],
                                batch["y"].float(),
                                batch_vec=batch["batch"],
                                ptr=batch["ptr"],
                            )

                        val_metrics["loss_total"] += metrics["loss_total"].item()
                        val_metrics["loss_feasibility"] += metrics["loss_feasibility"].item()
                        val_metrics["loss_selection"] += metrics["loss_selection"].item()
                        val_metrics["pred_size"] += raw_metrics["pred_size"]
                        val_metrics["feasibility"] += raw_metrics["feasibility"]
                        val_metrics["opt_size"] += raw_metrics["opt_size"]
                        val_metrics["gap"] += raw_metrics["gap"]
                        val_metrics["approx_ratio"] += raw_metrics["approx_ratio"]

                        if cfg.use_postprocessing:
                            val_pp_metrics["pp_pred_size"] += pp_metrics["pp_pred_size"]
                            val_pp_metrics["pp_feasibility"] += pp_metrics["pp_feasibility"]
                            val_pp_metrics["pp_gap"] += pp_metrics["pp_gap"]
                            val_pp_metrics["pp_approx_ratio"] += pp_metrics["pp_approx_ratio"]

                        val_count += 1

                n = max(val_count, 1)
                return val_metrics, val_pp_metrics, n

            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{cfg.epochs} - Validation (Self-Supervised)")
            print(f"{'=' * 70}")

            # Regular Model Validation
            val_metrics, val_pp_metrics, n = run_validation(raw_model, prefix="val", model_name="Regular")

            print(f"\n📊 Validation (Regular Model) Epoch {epoch + 1}:")
            print(f"Loss: {val_metrics['loss_total'] / n:.4f} (Feas: {val_metrics['loss_feasibility'] / n:.4f}, Sel: {val_metrics['loss_selection'] / n:.4f})")
            print("  Raw Metrics:")
            print(f"    Pred Size: {val_metrics['pred_size'] / n:.1f} / Opt Size: {val_metrics['opt_size'] / n:.1f}")
            print(f"    Gap: {val_metrics['gap'] / n:.2f}, Approx Ratio: {val_metrics['approx_ratio'] / n:.4f}")
            print(f"    Feasibility: {val_metrics['feasibility'] / n:.4f}")

            if cfg.use_postprocessing:
                print("  PP Metrics (greedy-decoded):")
                print(f"    PP Pred Size: {val_pp_metrics['pp_pred_size'] / n:.1f}")
                print(f"    PP Gap: {val_pp_metrics['pp_gap'] / n:.2f}, PP Approx Ratio: {val_pp_metrics['pp_approx_ratio'] / n:.4f}")
                print(f"    PP Feasibility: {val_pp_metrics['pp_feasibility'] / n:.4f}")

            val_log_dict = {
                "val/loss_total": val_metrics["loss_total"] / n,
                "val/loss_feasibility": val_metrics["loss_feasibility"] / n,
                "val/loss_selection": val_metrics["loss_selection"] / n,
                "epoch": epoch,
                "val/pred_size": val_metrics["pred_size"] / n,
                "val/feasibility": val_metrics["feasibility"] / n,
                "val/opt_size": val_metrics["opt_size"] / n,
                "val/gap": val_metrics["gap"] / n,
                "val/approx_ratio": val_metrics["approx_ratio"] / n,
            }

            if cfg.use_postprocessing:
                val_log_dict.update(
                    {
                        "val/pp_pred_size": val_pp_metrics["pp_pred_size"] / n,
                        "val/pp_feasibility": val_pp_metrics["pp_feasibility"] / n,
                        "val/pp_gap": val_pp_metrics["pp_gap"] / n,
                        "val/pp_approx_ratio": val_pp_metrics["pp_approx_ratio"] / n,
                    }
                )

            wandb.log(val_log_dict)

            # EMA Model Validation
            if ema_helper is not None:
                ema_model = ema_helper.ema_copy(raw_model)
                ema_model.cuda()

                ema_val_metrics, ema_val_pp_metrics, ema_n = run_validation(ema_model, prefix="val_ema", model_name="EMA")

                print(f"\n📊 Validation (EMA Model) Epoch {epoch + 1}:")
                print(
                    f"Loss: {ema_val_metrics['loss_total'] / ema_n:.4f} (Feas: {ema_val_metrics['loss_feasibility'] / ema_n:.4f}, Sel: {ema_val_metrics['loss_selection'] / ema_n:.4f})"
                )
                print("  Raw Metrics:")
                print(f"    Pred Size: {ema_val_metrics['pred_size'] / ema_n:.1f} / Opt Size: {ema_val_metrics['opt_size'] / ema_n:.1f}")
                print(f"    Gap: {ema_val_metrics['gap'] / ema_n:.2f}, Approx Ratio: {ema_val_metrics['approx_ratio'] / ema_n:.4f}")
                print(f"    Feasibility: {ema_val_metrics['feasibility'] / ema_n:.4f}")

                if cfg.use_postprocessing:
                    print("  PP Metrics (greedy-decoded):")
                    print(f"    PP Pred Size: {ema_val_pp_metrics['pp_pred_size'] / ema_n:.1f}")
                    print(f"    PP Gap: {ema_val_pp_metrics['pp_gap'] / ema_n:.2f}, PP Approx Ratio: {ema_val_pp_metrics['pp_approx_ratio'] / ema_n:.4f}")
                    print(f"    PP Feasibility: {ema_val_pp_metrics['pp_feasibility'] / ema_n:.4f}")

                ema_log_dict = {
                    "val_ema/loss_total": ema_val_metrics["loss_total"] / ema_n,
                    "val_ema/loss_feasibility": ema_val_metrics["loss_feasibility"] / ema_n,
                    "val_ema/loss_selection": ema_val_metrics["loss_selection"] / ema_n,
                    "epoch": epoch,
                    "val_ema/pred_size": ema_val_metrics["pred_size"] / ema_n,
                    "val_ema/feasibility": ema_val_metrics["feasibility"] / ema_n,
                    "val_ema/opt_size": ema_val_metrics["opt_size"] / ema_n,
                    "val_ema/gap": ema_val_metrics["gap"] / ema_n,
                    "val_ema/approx_ratio": ema_val_metrics["approx_ratio"] / ema_n,
                }

                if cfg.use_postprocessing:
                    ema_log_dict.update(
                        {
                            "val_ema/pp_pred_size": ema_val_pp_metrics["pp_pred_size"] / ema_n,
                            "val_ema/pp_feasibility": ema_val_pp_metrics["pp_feasibility"] / ema_n,
                            "val_ema/pp_gap": ema_val_pp_metrics["pp_gap"] / ema_n,
                            "val_ema/pp_approx_ratio": ema_val_pp_metrics["pp_approx_ratio"] / ema_n,
                        }
                    )

                wandb.log(ema_log_dict)
                del ema_model

            model.train()

        # Save checkpoint every 50 epochs
        if rank == 0 and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
            torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}.pt")
            print(f"  💾 Checkpoint saved: epoch_{epoch}.pt")

            if ema_helper is not None:
                torch.save(
                    ema_helper.state_dict(),
                    f"{cfg.checkpoint_path}/epoch_{epoch}_ema.pt",
                )
                print(f"  💾 EMA Checkpoint saved: epoch_{epoch}_ema.pt")

    if rank == 0:
        print("\n" + "=" * 70)
        print("Self-Supervised Training Complete!")
        print("=" * 70)
        wandb.finish()


if __name__ == "__main__":
    main()
