import math
import os
from typing import List, Optional

import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.ema import EMAHelper
from models.graph_transformer_trm import GraphTransformerTRM
from models.metrics import compute_metrics, compute_pp_metrics

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


class LossConfig(BaseModel):
    # BCE loss weight for positive class (None = auto from dataset class balance)
    pos_weight: Optional[float] = None

    # Feasibility loss weight: penalizes selecting adjacent nodes
    feasibility_weight: float = 0.0

    # Feasibility loss type: "soft" or "hinge" or "log_barrier"
    feasibility_loss_type: str = "hinge"

    # Selection loss weight (hybrid: adds SSL-style selection loss)
    selection_weight: float = 0.0

    # Label smoothing (0 = disabled, e.g. 0.1 = use 0.1/0.9 targets)
    label_smoothing: float = 0.0


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
    data_paths: List[str] = ["data/difusco_benchmark/datasets/satlib/train"]
    val_split: float = 0.1
    global_batch_size: int = 32

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
    run_name: str = "mis_trm_hinge_w0_satlib"
    checkpoint_path: str = "checkpoints/mis"
    log_every: int = 1

    # Validation
    validate_every_epoch: bool = True

    # Feature flags (Phase 0: CPU bottleneck elimination)
    use_pe: bool = True  # Compute Laplacian PE (CPU-heavy)
    use_enhanced_features: bool = True  # Compute enhanced features (CPU-heavy)

    # Dataset
    max_shards: int = None  # Limit shards for debugging (None = all)

    # Pretrained weights
    pretrained: Optional[str] = None  # Path to checkpoint for weight initialization


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


def main(cfg=None):
    # 1. Setup
    rank, world_size = init_distributed()

    if cfg is None:
        cfg = Config()

    if rank == 0:
        print("=" * 70)
        print("MIS-TRM Training")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Config: LR={cfg.lr}, WD={cfg.weight_decay}, Betas=({cfg.beta1}, {cfg.beta2}), GradClip={cfg.grad_clip}")
        print(f"Arch: Dim={cfg.arch.hidden_dim}, Layers={cfg.arch.num_layers}, H={cfg.arch.H_cycles}, L={cfg.arch.L_cycles}")
        print(f"Loss: feasibility_weight={cfg.loss.feasibility_weight}, loss_type={cfg.loss.feasibility_loss_type}")
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
        use_pe=cfg.use_pe,
        use_enhanced_features=cfg.use_enhanced_features,
        max_shards=cfg.max_shards,
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
        use_pe=cfg.use_pe,
        use_enhanced_features=cfg.use_enhanced_features,
        max_shards=cfg.max_shards,
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
    # Use CLI-specified pos_weight if set, otherwise keep auto-computed from dataset
    if cfg.loss.pos_weight is not None:
        model_config_dict["pos_weight"] = cfg.loss.pos_weight
    # else: keep the auto-computed value from dataset (set at line 180)
    model_config_dict["feasibility_weight"] = cfg.loss.feasibility_weight
    model_config_dict["feasibility_loss_type"] = cfg.loss.feasibility_loss_type
    model_config_dict["dropout"] = cfg.arch.dropout
    model_config_dict["attn_dropout"] = cfg.arch.attn_dropout
    model_config_dict["selection_weight"] = cfg.loss.selection_weight
    model_config_dict["label_smoothing"] = cfg.loss.label_smoothing

    effective_pos_weight = model_config_dict["pos_weight"]
    if rank == 0:
        print("\n🚀 Model Configuration:")
        print(f"  pos_weight: {effective_pos_weight} ({'auto from dataset' if cfg.loss.pos_weight is None else 'manual'})")
        print(f"  feasibility_weight: {cfg.loss.feasibility_weight}")
        print(f"  feasibility_loss_type: {cfg.loss.feasibility_loss_type}")
        print(f"  dropout: {cfg.arch.dropout}")
        print(f"  attn_dropout: {cfg.arch.attn_dropout}")
        print(f"  Training on {len(train_dataset.shards)} shards")

    model = GraphTransformerTRM(model_config_dict).cuda()
    raw_model = model

    # Load pretrained weights if specified (skip shape-mismatched keys)
    if cfg.pretrained is not None:
        state_dict = torch.load(cfg.pretrained, map_location="cuda", weights_only=False)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        if rank == 0:
            skipped = set(state_dict.keys()) - set(filtered.keys())
            print(f"📦 Loaded {len(filtered)}/{len(state_dict)} pretrained weights from {cfg.pretrained}")
            if skipped:
                print(f"  ⚠️  Skipped {len(skipped)} keys (shape mismatch): {skipped}")

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

    # 5. Training Loop
    step = 0
    best_val_pp_approx = 0.0
    best_ema_pp_approx = 0.0

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

        # Accumulators for epoch-level metrics (raw model metrics - always computed)
        epoch_metrics = {
            "loss_total": 0,
            "loss_bce": 0,
            "loss_feasibility": 0,
            "opt_size": 0,
            "pred_size": 0,
            "gap": 0,
            "approx_ratio": 0,
            "feasibility": 0,
        }
        # Accumulators for pp metrics (only used when use_postprocessing=True)
        epoch_pp_metrics = {
            "pp_pred_size": 0,
            "pp_gap": 0,
            "pp_approx_ratio": 0,
            "pp_feasibility": 0,
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
            optimizer.step()

            # Update EMA model after each optimizer step
            if ema_helper is not None:
                ema_helper.update(raw_model)

            # Compute raw metrics (always)
            with torch.no_grad():
                raw_metrics = compute_metrics(
                    final_preds["preds"].squeeze(),
                    batch["edge_index"],
                    batch["y"].float(),
                    batch_vec=batch["batch"],
                    ptr=batch["ptr"],
                )

                # Compute pp metrics only when use_postprocessing is True
                if cfg.use_postprocessing:
                    pp_metrics = compute_pp_metrics(
                        final_preds["preds"].squeeze(),
                        batch["edge_index"],
                        batch["y"].float(),
                        batch_vec=batch["batch"],
                        ptr=batch["ptr"],
                    )

            # Accumulate raw metrics (always)
            epoch_metrics["loss_total"] += final_metrics["loss_total"].item()
            epoch_metrics["loss_bce"] += final_metrics["loss_bce"].item()
            epoch_metrics["loss_feasibility"] += final_metrics["loss_feasibility"].item()
            epoch_metrics["opt_size"] += raw_metrics["opt_size"]
            epoch_metrics["pred_size"] += raw_metrics["pred_size"]
            epoch_metrics["gap"] += raw_metrics["gap"]
            epoch_metrics["approx_ratio"] += raw_metrics["approx_ratio"]
            epoch_metrics["feasibility"] += raw_metrics["feasibility"]

            # Accumulate pp metrics (only when use_postprocessing=True)
            if cfg.use_postprocessing:
                epoch_pp_metrics["pp_pred_size"] += pp_metrics["pp_pred_size"]
                epoch_pp_metrics["pp_gap"] += pp_metrics["pp_gap"]
                epoch_pp_metrics["pp_approx_ratio"] += pp_metrics["pp_approx_ratio"]
                epoch_pp_metrics["pp_feasibility"] += pp_metrics["pp_feasibility"]

            count += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{final_metrics['loss_total'].item():.4f}",
                        "bce": f"{final_metrics['loss_bce'].item():.4f}",
                    }
                )

                if step % cfg.log_every == 0:
                    # Always log raw metrics
                    log_data = {
                        # Loss metrics
                        "train/loss_total": final_metrics["loss_total"].item(),
                        "train/loss_bce": final_metrics["loss_bce"].item(),
                        "train/loss_feasibility": final_metrics["loss_feasibility"].item(),
                        # Raw model metrics (can have violations, approx_ratio > 1)
                        "train/opt_size": raw_metrics["opt_size"],
                        "train/pred_size": raw_metrics["pred_size"],
                        "train/gap": raw_metrics["gap"],
                        "train/approx_ratio": raw_metrics["approx_ratio"],
                        "train/feasibility": raw_metrics["feasibility"],
                        # Confusion matrix (before decoding)
                        "train/precision": raw_metrics["precision"],
                        "train/recall": raw_metrics["recall"],
                        "train/f1": raw_metrics["f1"],
                        "train/tp": raw_metrics["tp"],
                        "train/fp": raw_metrics["fp"],
                        "train/fn": raw_metrics["fn"],
                        "train/tn": raw_metrics["tn"],
                        # Training info
                        "epoch": epoch,
                        "lr": current_lr,
                    }

                    # Add pp metrics when use_postprocessing is True
                    if cfg.use_postprocessing:
                        log_data.update(
                            {
                                "train/pp_pred_size": pp_metrics["pp_pred_size"],
                                "train/pp_gap": pp_metrics["pp_gap"],
                                "train/pp_approx_ratio": pp_metrics["pp_approx_ratio"],
                                "train/pp_feasibility": pp_metrics["pp_feasibility"],
                            }
                        )

                    wandb.log(log_data)

        if rank == 0:
            pbar.close()

            # Log epoch-level training metrics
            n = max(count, 1)

            print(f"\n📈 Training Epoch {epoch + 1} Summary:")
            print(f"   Loss: {epoch_metrics['loss_total'] / n:.4f} (BCE: {epoch_metrics['loss_bce'] / n:.4f}, Feas: {epoch_metrics['loss_feasibility'] / n:.4f})")
            print("  Raw Metrics:")
            print(f"    Pred Size: {epoch_metrics['pred_size'] / n:.1f} / Opt Size: {epoch_metrics['opt_size'] / n:.1f}")
            print(f"    Gap: {epoch_metrics['gap'] / n:.2f}, Approx Ratio: {epoch_metrics['approx_ratio'] / n:.4f}")
            print(f"    Feasibility: {epoch_metrics['feasibility'] / n:.4f}")

            if cfg.use_postprocessing:
                print("  PP Metrics (greedy-decoded):")
                print(f"    PP Gap: {epoch_pp_metrics['pp_gap'] / n:.2f}, PP Approx Ratio: {epoch_pp_metrics['pp_approx_ratio'] / n:.4f}")
                print(f"    PP Feasibility: {epoch_pp_metrics['pp_feasibility'] / n:.4f}")

            # Build log dict - always log raw metrics
            log_dict = {
                "train_epoch/loss_total": epoch_metrics["loss_total"] / n,
                "train_epoch/loss_bce": epoch_metrics["loss_bce"] / n,
                "train_epoch/loss_feasibility": epoch_metrics["loss_feasibility"] / n,
                "epoch": epoch,
                # Raw metrics
                "train_epoch/opt_size": epoch_metrics["opt_size"] / n,
                "train_epoch/pred_size": epoch_metrics["pred_size"] / n,
                "train_epoch/gap": epoch_metrics["gap"] / n,
                "train_epoch/approx_ratio": epoch_metrics["approx_ratio"] / n,
                "train_epoch/feasibility": epoch_metrics["feasibility"] / n,
            }

            # Add pp metrics when using post-processing
            if cfg.use_postprocessing:
                log_dict.update(
                    {
                        "train_epoch/pp_pred_size": epoch_pp_metrics["pp_pred_size"] / n,
                        "train_epoch/pp_gap": epoch_pp_metrics["pp_gap"] / n,
                        "train_epoch/pp_approx_ratio": epoch_pp_metrics["pp_approx_ratio"] / n,
                        "train_epoch/pp_feasibility": epoch_pp_metrics["pp_feasibility"] / n,
                    }
                )

            wandb.log(log_dict)

        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        if cfg.validate_every_epoch and rank == 0:

            def run_validation(eval_model, prefix="val", model_name="Model"):
                """Run validation and return metrics dict"""
                eval_model.eval()

                # Raw metrics (always computed)
                val_metrics = {
                    "loss_total": 0,
                    "loss_bce": 0,
                    "loss_feasibility": 0,
                    "opt_size": 0,
                    "pred_size": 0,
                    "gap": 0,
                    "approx_ratio": 0,
                    "feasibility": 0,
                }
                # PP metrics (only when use_postprocessing=True)
                val_pp_metrics = {
                    "pp_pred_size": 0,
                    "pp_gap": 0,
                    "pp_approx_ratio": 0,
                    "pp_feasibility": 0,
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

                        # Raw metrics (always)
                        raw_metrics = compute_metrics(
                            preds["preds"].squeeze(),
                            batch["edge_index"],
                            batch["y"].float(),
                            batch_vec=batch["batch"],
                            ptr=batch["ptr"],
                        )

                        # PP metrics (only when use_postprocessing=True)
                        if cfg.use_postprocessing:
                            pp_metrics = compute_pp_metrics(
                                preds["preds"].squeeze(),
                                batch["edge_index"],
                                batch["y"].float(),
                                batch_vec=batch["batch"],
                                ptr=batch["ptr"],
                            )

                        # Accumulate raw metrics
                        val_metrics["loss_total"] += metrics["loss_total"].item()
                        val_metrics["loss_bce"] += metrics["loss_bce"].item()
                        val_metrics["loss_feasibility"] += metrics["loss_feasibility"].item()
                        val_metrics["opt_size"] += raw_metrics["opt_size"]
                        val_metrics["pred_size"] += raw_metrics["pred_size"]
                        val_metrics["gap"] += raw_metrics["gap"]
                        val_metrics["approx_ratio"] += raw_metrics["approx_ratio"]
                        val_metrics["feasibility"] += raw_metrics["feasibility"]

                        # Accumulate pp metrics
                        if cfg.use_postprocessing:
                            val_pp_metrics["pp_pred_size"] += pp_metrics["pp_pred_size"]
                            val_pp_metrics["pp_gap"] += pp_metrics["pp_gap"]
                            val_pp_metrics["pp_approx_ratio"] += pp_metrics["pp_approx_ratio"]
                            val_pp_metrics["pp_feasibility"] += pp_metrics["pp_feasibility"]

                        val_count += 1

                n = max(val_count, 1)
                return val_metrics, val_pp_metrics, n

            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{cfg.epochs} - Validation")
            print(f"{'=' * 70}")

            # --- Regular Model Validation ---
            val_metrics, val_pp_metrics, n = run_validation(raw_model, prefix="val", model_name="Regular")

            print(f"\n📊 Validation (Regular Model) Epoch {epoch + 1}:")
            print(f"Loss: {val_metrics['loss_total'] / n:.4f} (BCE: {val_metrics['loss_bce'] / n:.4f}, Feas: {val_metrics['loss_feasibility'] / n:.4f})")
            print("  Raw Metrics:")
            print(f"    Pred Size: {val_metrics['pred_size'] / n:.1f} / Opt Size: {val_metrics['opt_size'] / n:.1f}")
            print(f"    Gap: {val_metrics['gap'] / n:.2f}, Approx Ratio: {val_metrics['approx_ratio'] / n:.4f}")
            print(f"    Feasibility: {val_metrics['feasibility'] / n:.4f}")

            if cfg.use_postprocessing:
                print("  PP Metrics (greedy-decoded):")
                print(f"    PP Gap: {val_pp_metrics['pp_gap'] / n:.2f}, PP Approx Ratio: {val_pp_metrics['pp_approx_ratio'] / n:.4f}")
                print(f"    PP Feasibility: {val_pp_metrics['pp_feasibility'] / n:.4f}")

            # Build log dict - always log raw metrics
            val_log_dict = {
                "val/loss_total": val_metrics["loss_total"] / n,
                "val/loss_bce": val_metrics["loss_bce"] / n,
                "val/loss_feasibility": val_metrics["loss_feasibility"] / n,
                "epoch": epoch,
                # Raw metrics
                "val/opt_size": val_metrics["opt_size"] / n,
                "val/pred_size": val_metrics["pred_size"] / n,
                "val/gap": val_metrics["gap"] / n,
                "val/approx_ratio": val_metrics["approx_ratio"] / n,
                "val/feasibility": val_metrics["feasibility"] / n,
            }

            # Add pp metrics when using post-processing
            if cfg.use_postprocessing:
                val_log_dict.update(
                    {
                        "val/pp_pred_size": val_pp_metrics["pp_pred_size"] / n,
                        "val/pp_gap": val_pp_metrics["pp_gap"] / n,
                        "val/pp_approx_ratio": val_pp_metrics["pp_approx_ratio"] / n,
                        "val/pp_feasibility": val_pp_metrics["pp_feasibility"] / n,
                    }
                )

            wandb.log(val_log_dict)

            # Save best regular model checkpoint
            if cfg.use_postprocessing:
                current_pp_approx = val_pp_metrics["pp_approx_ratio"] / n
                if current_pp_approx > best_val_pp_approx:
                    best_val_pp_approx = current_pp_approx
                    torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/best.pt")
                    print(f"  🏆 New best model! PP Approx: {best_val_pp_approx:.4f} (epoch {epoch + 1})")

            # --- EMA Model Validation ---
            if ema_helper is not None:
                # Create a copy of model with EMA weights
                ema_model = ema_helper.ema_copy(raw_model)
                ema_model.cuda()

                ema_val_metrics, ema_val_pp_metrics, ema_n = run_validation(ema_model, prefix="val_ema", model_name="EMA")

                print(f"\n📊 Validation (EMA Model) Epoch {epoch + 1}:")
                print(
                    f"Loss: {ema_val_metrics['loss_total'] / ema_n:.4f} (BCE: {ema_val_metrics['loss_bce'] / ema_n:.4f}, Feas: {ema_val_metrics['loss_feasibility'] / ema_n:.4f})"
                )
                print("  Raw Metrics:")
                print(f"    Pred Size: {ema_val_metrics['pred_size'] / ema_n:.1f} / Opt Size: {ema_val_metrics['opt_size'] / ema_n:.1f}")
                print(f"    Gap: {ema_val_metrics['gap'] / ema_n:.2f}, Approx Ratio: {ema_val_metrics['approx_ratio'] / ema_n:.4f}")
                print(f"    Feasibility: {ema_val_metrics['feasibility'] / ema_n:.4f}")

                if cfg.use_postprocessing:
                    print("  PP Metrics (greedy-decoded):")
                    print(f"    PP Gap: {ema_val_pp_metrics['pp_gap'] / ema_n:.2f}, PP Approx Ratio: {ema_val_pp_metrics['pp_approx_ratio'] / ema_n:.4f}")
                    print(f"    PP Feasibility: {ema_val_pp_metrics['pp_feasibility'] / ema_n:.4f}")

                ema_log_dict = {
                    "val_ema/loss_total": ema_val_metrics["loss_total"] / ema_n,
                    "val_ema/loss_bce": ema_val_metrics["loss_bce"] / ema_n,
                    "val_ema/loss_feasibility": ema_val_metrics["loss_feasibility"] / ema_n,
                    "epoch": epoch,
                    # Raw metrics
                    "val_ema/opt_size": ema_val_metrics["opt_size"] / ema_n,
                    "val_ema/pred_size": ema_val_metrics["pred_size"] / ema_n,
                    "val_ema/gap": ema_val_metrics["gap"] / ema_n,
                    "val_ema/approx_ratio": ema_val_metrics["approx_ratio"] / ema_n,
                    "val_ema/feasibility": ema_val_metrics["feasibility"] / ema_n,
                }

                if cfg.use_postprocessing:
                    ema_log_dict.update(
                        {
                            "val_ema/pp_pred_size": ema_val_pp_metrics["pp_pred_size"] / ema_n,
                            "val_ema/pp_gap": ema_val_pp_metrics["pp_gap"] / ema_n,
                            "val_ema/pp_approx_ratio": ema_val_pp_metrics["pp_approx_ratio"] / ema_n,
                            "val_ema/pp_feasibility": ema_val_pp_metrics["pp_feasibility"] / ema_n,
                        }
                    )

                wandb.log(ema_log_dict)

                # Save best EMA model checkpoint
                if cfg.use_postprocessing:
                    current_ema_pp_approx = ema_val_pp_metrics["pp_approx_ratio"] / ema_n
                    if current_ema_pp_approx > best_ema_pp_approx:
                        best_ema_pp_approx = current_ema_pp_approx
                        torch.save(ema_helper.state_dict(), f"{cfg.checkpoint_path}/best_ema.pt")
                        print(f"  🏆 New best EMA! PP Approx: {best_ema_pp_approx:.4f} (epoch {epoch + 1})")

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--feasibility_weight", type=float, default=None)
    parser.add_argument("--feasibility_loss_type", type=str, default=None)
    parser.add_argument("--use_pe", type=int, default=None, help="0 or 1")
    parser.add_argument("--use_enhanced_features", type=int, default=None, help="0 or 1")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Global batch size")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive class weight for BCE (None=auto from dataset)")
    # New Phase C args
    parser.add_argument("--selection_weight", type=float, default=None, help="Hybrid SSL selection loss weight")
    parser.add_argument("--label_smoothing", type=float, default=None, help="Label smoothing factor")
    parser.add_argument("--n_supervision", type=int, default=None, help="Number of forward passes per step")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint for fine-tuning")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.use_pe is not None:
        cfg.use_pe = bool(args.use_pe)
    if args.use_enhanced_features is not None:
        cfg.use_enhanced_features = bool(args.use_enhanced_features)
    if args.feasibility_weight is not None:
        cfg.loss.feasibility_weight = args.feasibility_weight
    if args.feasibility_loss_type is not None:
        cfg.loss.feasibility_loss_type = args.feasibility_loss_type
    if args.max_shards is not None:
        cfg.max_shards = args.max_shards
    if args.batch_size is not None:
        cfg.global_batch_size = args.batch_size
    if args.pos_weight is not None:
        cfg.loss.pos_weight = args.pos_weight
    # New Phase C args
    if args.selection_weight is not None:
        cfg.loss.selection_weight = args.selection_weight
    if args.label_smoothing is not None:
        cfg.loss.label_smoothing = args.label_smoothing
    if args.n_supervision is not None:
        cfg.n_supervision = args.n_supervision
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path

    main(cfg)
