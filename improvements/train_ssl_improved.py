"""
Improved Self-Supervised Training for MIS

Key improvements over the original train_mis_ssl.py:

1. **Lagrangian Dual Ascent**: Instead of fixed feasibility_weight, we maintain
   a learnable dual variable λ that auto-balances selection vs feasibility.
   Update: λ ← λ + α * (mean_violation - target_violation)
   This prevents the collapse seen in ssl_hlr (fw=20) and ssl_pignn (fw=100).

2. **Feasibility Warmup**: feasibility_weight starts at 0 and ramps up over
   first 20% of training. The model first learns to select nodes (easy signal),
   then gradually learns to respect constraints.

3. **Weight Decay**: L2 regularization (1e-5) to prevent overfitting.

4. **Gradient Norm Monitoring**: Track gradient norms to detect collapse early.

5. **Early Stopping on Collapse**: If pred_size drops to 0 for 5 consecutive
   steps, reduce feasibility weight by 50%.

This file is standalone — it imports from the original codebase but has its
own training loop. Run via SLURM or directly.
"""

import math
import os
import sys
from typing import List, Optional

import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.ema import EMAHelper
from models.graph_transformer_trm_ssl import GraphTransformerTRM_SSL
from models.metrics_ssl import compute_metrics_ssl, compute_pp_metrics_ssl


# ============================================================================
# CONFIGURATION
# ============================================================================

class LossConfig(BaseModel):
    mu: float = 5.0
    feasibility_weight: float = 2.0
    selection_weight: float = 5.0
    feasibility_loss_type: str = "log_barrier"  # Changed default to log_barrier (best from experiments)
    loss_mode: str = "default"
    noise_scale: float = 0.0
    degree_weighted: bool = False
    reinforce_samples: int = 8
    entropy_weight: float = 0.0
    temperature: float = 1.0
    temp_start: float = 1.0
    temp_end: float = 1.0
    use_loss_schedule: bool = False
    fw_start: float = 0.5
    sw_start: float = 10.0
    use_deep_supervision: bool = False

    # NEW: Lagrangian dual ascent
    use_lagrangian: bool = True
    lagrangian_lr: float = 0.01  # Dual variable learning rate
    lagrangian_target: float = 0.05  # Target violation rate (5%)
    lagrangian_max: float = 50.0  # Maximum dual variable value
    lagrangian_min: float = 0.1  # Minimum dual variable value

    # NEW: Feasibility warmup
    use_feasibility_warmup: bool = True
    warmup_fraction: float = 0.2  # Fraction of training for warmup


class ArchConfig(BaseModel):
    name: str = "graph_trm_ssl"
    hidden_dim: int = 256
    num_layers: int = 2
    L_cycles: int = 6
    H_cycles: int = 2
    dropout: float = 0.2
    attn_dropout: float = 0.2


class Config(BaseModel):
    data_paths: List[str] = ["data/difusco_benchmark/datasets/satlib/train"]
    val_split: float = 0.1
    global_batch_size: int = 32
    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 50
    epochs: int = 500
    seed: int = 0
    weight_decay: float = 1e-5  # Changed: enable weight decay
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 0.5
    n_supervision: int = 1
    arch: ArchConfig = ArchConfig()
    loss: LossConfig = LossConfig()
    use_postprocessing: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    project_name: str = "MIS-TRM"
    run_name: str = "improved_ssl"
    checkpoint_path: str = "checkpoints/improved_ssl"
    log_every: int = 1
    validate_every_epoch: bool = True
    use_pe: bool = False  # Disabled for speed (50x)
    use_enhanced_features: bool = False
    max_shards: Optional[int] = None
    pretrained: Optional[str] = None  # Path to pretrained checkpoint


def cosine_schedule_with_warmup(current_step, base_lr, num_warmup_steps,
                                  num_training_steps, min_ratio=0.1):
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
    rank, world_size = init_distributed()
    if cfg is None:
        cfg = Config()

    if rank == 0:
        print("=" * 70)
        print("IMPROVED MIS-TRM Self-Supervised Training")
        print("=" * 70)
        print(f"Improvements: Lagrangian={cfg.loss.use_lagrangian}, "
              f"FeasWarmup={cfg.loss.use_feasibility_warmup}, "
              f"WeightDecay={cfg.weight_decay}")
        print(f"Loss: {cfg.loss.feasibility_loss_type}, fw={cfg.loss.feasibility_weight}, sw={cfg.loss.selection_weight}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)

    torch.manual_seed(cfg.seed + rank)

    # Data
    train_ds_config = MISDatasetConfig(
        dataset_paths=cfg.data_paths, global_batch_size=cfg.global_batch_size,
        rank=rank, num_replicas=world_size, drop_last=True,
        val_split=cfg.val_split, seed=cfg.seed,
        use_pe=cfg.use_pe, use_enhanced_features=cfg.use_enhanced_features,
        max_shards=cfg.max_shards,
    )
    train_dataset = MISDataset(train_ds_config, split="train")

    val_ds_config = MISDatasetConfig(
        dataset_paths=cfg.data_paths, global_batch_size=cfg.global_batch_size,
        rank=0, num_replicas=1, drop_last=False,
        val_split=cfg.val_split, seed=cfg.seed,
        use_pe=cfg.use_pe, use_enhanced_features=cfg.use_enhanced_features,
        max_shards=cfg.max_shards,
    )
    val_dataset = MISDataset(val_ds_config, split="val")

    # Model config
    model_config_dict = {
        "input_dim": train_dataset.metadata.input_dim,
        "pe_input_dim": train_dataset.metadata.pe_dim,
        "hidden_dim": cfg.arch.hidden_dim,
        "num_layers": cfg.arch.num_layers,
        "L_cycles": cfg.arch.L_cycles,
        "H_cycles": cfg.arch.H_cycles,
        "dropout": cfg.arch.dropout,
        "attn_dropout": cfg.arch.attn_dropout,
        "mu": cfg.loss.mu,
        "feasibility_weight": cfg.loss.feasibility_weight,
        "selection_weight": cfg.loss.selection_weight,
        "feasibility_loss_type": cfg.loss.feasibility_loss_type,
        "use_deep_supervision": cfg.loss.use_deep_supervision,
        "temperature": cfg.loss.temperature,
        "entropy_weight": cfg.loss.entropy_weight,
        "loss_mode": cfg.loss.loss_mode,
        "noise_scale": cfg.loss.noise_scale,
        "degree_weighted": cfg.loss.degree_weighted,
        "reinforce_samples": cfg.loss.reinforce_samples,
    }

    if rank == 0:
        print(f"\n📊 Dataset: {train_dataset.num_graphs} train, {val_dataset.num_graphs} val")

    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    # Model
    model = GraphTransformerTRM_SSL(model_config_dict).cuda()
    raw_model = model

    # Load pretrained weights if specified
    if cfg.pretrained is not None:
        state_dict = torch.load(cfg.pretrained, map_location="cuda", weights_only=False)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        if rank == 0:
            print(f"📦 Loaded {len(filtered)}/{len(state_dict)} pretrained weights from {cfg.pretrained}")

    # EMA
    ema_helper = None
    if cfg.use_ema:
        ema_helper = EMAHelper(mu=cfg.ema_decay)
        ema_helper.register(model)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model: {total_params:,} params ({total_params / 1e6:.2f}M)")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        raw_model = model.module

    # wandb
    if rank == 0:
        wandb_config = cfg.model_dump()
        wandb_config["model_params"] = total_params
        wandb_config["training_mode"] = "improved_ssl"
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=wandb_config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    # Lagrangian dual variable
    lagrangian_lambda = cfg.loss.feasibility_weight  # Start at configured fw

    steps_per_epoch = train_dataset.num_graphs // cfg.global_batch_size
    val_steps_per_epoch = (val_dataset.num_graphs + cfg.global_batch_size - 1) // cfg.global_batch_size
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.loss.warmup_fraction) if cfg.loss.use_feasibility_warmup else 0

    # Collapse detection
    zero_pred_count = 0

    step = 0
    best_val_pp_approx = 0.0
    best_ema_pp_approx = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        train_dataset.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{cfg.epochs} — Improved SSL Training")
            print(f"  fw={raw_model.feasibility_weight:.4f}, λ={lagrangian_lambda:.4f}")
            print(f"{'=' * 70}")
            pbar = tqdm(total=steps_per_epoch, desc="Train")

        epoch_metrics = {
            "loss_total": 0, "loss_feasibility": 0, "loss_selection": 0,
            "pred_size": 0, "feasibility": 0, "opt_size": 0,
            "gap": 0, "approx_ratio": 0, "grad_norm": 0,
        }
        epoch_pp_metrics = {
            "pp_pred_size": 0, "pp_feasibility": 0,
            "pp_gap": 0, "pp_approx_ratio": 0,
        }
        count = 0

        for batch_name, batch, batch_size in train_dataloader:
            step += 1

            # LR schedule
            current_lr = cosine_schedule_with_warmup(
                step, cfg.lr, cfg.lr_warmup_steps, total_steps, cfg.lr_min_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # === IMPROVEMENT 1: Feasibility Warmup ===
            if cfg.loss.use_feasibility_warmup and step <= warmup_steps:
                warmup_progress = step / max(warmup_steps, 1)
                effective_fw = lagrangian_lambda * warmup_progress
            else:
                effective_fw = lagrangian_lambda

            raw_model.feasibility_weight = effective_fw

            # Temperature annealing
            if cfg.loss.temp_start != cfg.loss.temp_end:
                progress = step / max(total_steps, 1)
                raw_model.temperature = cfg.loss.temp_start + (cfg.loss.temp_end - cfg.loss.temp_start) * progress

            # Forward
            carry = raw_model.initial_carry(batch)
            all_finish = False
            final_loss = final_metrics = final_preds = None
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

            # Gradient norm monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            if ema_helper is not None:
                ema_helper.update(raw_model)

            # Compute metrics
            with torch.no_grad():
                raw_metrics = compute_metrics_ssl(
                    final_preds["preds"].squeeze(), batch["edge_index"],
                    batch["y"].float(), batch_vec=batch["batch"], ptr=batch["ptr"])

                if cfg.use_postprocessing:
                    pp_metrics = compute_pp_metrics_ssl(
                        final_preds["preds"].squeeze(), batch["edge_index"],
                        batch["y"].float(), batch_vec=batch["batch"], ptr=batch["ptr"])

            # === IMPROVEMENT 2: Lagrangian Dual Ascent ===
            if cfg.loss.use_lagrangian and step > warmup_steps:
                violation_rate = 1.0 - raw_metrics["feasibility"]
                lagrangian_lambda += cfg.loss.lagrangian_lr * (violation_rate - cfg.loss.lagrangian_target)
                lagrangian_lambda = max(cfg.loss.lagrangian_min,
                                        min(cfg.loss.lagrangian_max, lagrangian_lambda))

            # === IMPROVEMENT 3: Collapse Detection ===
            if raw_metrics["pred_size"] < 1.0:
                zero_pred_count += 1
                if zero_pred_count >= 5:
                    lagrangian_lambda *= 0.5
                    lagrangian_lambda = max(cfg.loss.lagrangian_min, lagrangian_lambda)
                    if rank == 0:
                        print(f"  ⚠️ Collapse detected! Reducing λ to {lagrangian_lambda:.4f}")
                    zero_pred_count = 0
            else:
                zero_pred_count = 0

            # Accumulate
            epoch_metrics["loss_total"] += final_metrics["loss_total"].item()
            epoch_metrics["loss_feasibility"] += final_metrics["loss_feasibility"].item()
            epoch_metrics["loss_selection"] += final_metrics["loss_selection"].item()
            epoch_metrics["pred_size"] += raw_metrics["pred_size"]
            epoch_metrics["feasibility"] += raw_metrics["feasibility"]
            epoch_metrics["opt_size"] += raw_metrics["opt_size"]
            epoch_metrics["gap"] += raw_metrics["gap"]
            epoch_metrics["approx_ratio"] += raw_metrics["approx_ratio"]
            epoch_metrics["grad_norm"] += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            if cfg.use_postprocessing:
                epoch_pp_metrics["pp_pred_size"] += pp_metrics["pp_pred_size"]
                epoch_pp_metrics["pp_feasibility"] += pp_metrics["pp_feasibility"]
                epoch_pp_metrics["pp_gap"] += pp_metrics["pp_gap"]
                epoch_pp_metrics["pp_approx_ratio"] += pp_metrics["pp_approx_ratio"]

            count += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{final_metrics['loss_total'].item():.4f}",
                    "feas": f"{raw_metrics['feasibility']:.3f}",
                    "size": f"{raw_metrics['pred_size']:.1f}",
                    "λ": f"{lagrangian_lambda:.2f}",
                })

                if step % cfg.log_every == 0:
                    log_data = {
                        "train/loss_total": final_metrics["loss_total"].item(),
                        "train/loss_feasibility": final_metrics["loss_feasibility"].item(),
                        "train/loss_selection": final_metrics["loss_selection"].item(),
                        "train/pred_size": raw_metrics["pred_size"],
                        "train/feasibility": raw_metrics["feasibility"],
                        "train/opt_size": raw_metrics["opt_size"],
                        "train/gap": raw_metrics["gap"],
                        "train/approx_ratio": raw_metrics["approx_ratio"],
                        "train/precision": raw_metrics["precision"],
                        "train/recall": raw_metrics["recall"],
                        "train/f1": raw_metrics["f1"],
                        "train/tp": raw_metrics["tp"],
                        "train/fp": raw_metrics["fp"],
                        "train/fn": raw_metrics["fn"],
                        "train/tn": raw_metrics["tn"],
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/lagrangian_lambda": lagrangian_lambda,
                        "train/effective_fw": effective_fw,
                        "epoch": epoch,
                        "lr": current_lr,
                    }
                    if cfg.use_postprocessing:
                        log_data.update({
                            "train/pp_pred_size": pp_metrics["pp_pred_size"],
                            "train/pp_feasibility": pp_metrics["pp_feasibility"],
                            "train/pp_gap": pp_metrics["pp_gap"],
                            "train/pp_approx_ratio": pp_metrics["pp_approx_ratio"],
                        })
                    wandb.log(log_data)

        if rank == 0:
            pbar.close()
            n = max(count, 1)
            print(f"\n📈 Epoch {epoch + 1} Summary:")
            print(f"  Loss: {epoch_metrics['loss_total']/n:.4f}, GradNorm: {epoch_metrics['grad_norm']/n:.4f}")
            print(f"  Pred: {epoch_metrics['pred_size']/n:.1f}, Opt: {epoch_metrics['opt_size']/n:.1f}, "
                  f"Feas: {epoch_metrics['feasibility']/n:.4f}")
            if cfg.use_postprocessing:
                print(f"  PP Approx: {epoch_pp_metrics['pp_approx_ratio']/n:.4f}, "
                      f"PP Gap: {epoch_pp_metrics['pp_gap']/n:.2f}")

        # =====================================================================
        # VALIDATION
        # =====================================================================
        if cfg.validate_every_epoch and rank == 0:
            def run_validation(eval_model, model_name="Model"):
                eval_model.eval()
                vm = {"loss_total": 0, "loss_feasibility": 0, "loss_selection": 0,
                      "pred_size": 0, "feasibility": 0, "opt_size": 0, "gap": 0, "approx_ratio": 0}
                vpp = {"pp_pred_size": 0, "pp_feasibility": 0, "pp_gap": 0, "pp_approx_ratio": 0}
                vc = 0

                with torch.no_grad():
                    for batch_name, batch, batch_size in tqdm(val_dataloader, desc=f"Val ({model_name})", total=val_steps_per_epoch):
                        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        carry = eval_model.initial_carry(batch)
                        for _ in range(cfg.n_supervision):
                            carry, sl, met, preds, _ = eval_model(carry, batch)
                        rm = compute_metrics_ssl(preds["preds"].squeeze(), batch["edge_index"],
                                                 batch["y"].float(), batch_vec=batch["batch"], ptr=batch["ptr"])
                        if cfg.use_postprocessing:
                            pm = compute_pp_metrics_ssl(preds["preds"].squeeze(), batch["edge_index"],
                                                        batch["y"].float(), batch_vec=batch["batch"], ptr=batch["ptr"])
                        for k in vm: vm[k] += (met[k].item() if k in met else rm.get(k, 0))
                        if cfg.use_postprocessing:
                            for k in vpp: vpp[k] += pm[k]
                        vc += 1
                return vm, vpp, max(vc, 1)

            val_m, val_pp, vn = run_validation(raw_model, "Regular")
            print(f"\n📊 Val Epoch {epoch+1}:")
            print(f"  Approx: {val_m['approx_ratio']/vn:.4f}, Feas: {val_m['feasibility']/vn:.4f}")
            if cfg.use_postprocessing:
                print(f"  PP Approx: {val_pp['pp_approx_ratio']/vn:.4f}")

            val_log = {
                "val/loss_total": val_m["loss_total"] / vn,
                "val/pred_size": val_m["pred_size"] / vn,
                "val/feasibility": val_m["feasibility"] / vn,
                "val/approx_ratio": val_m["approx_ratio"] / vn,
                "val/opt_size": val_m["opt_size"] / vn,
                "epoch": epoch,
            }
            if cfg.use_postprocessing:
                val_log.update({
                    "val/pp_approx_ratio": val_pp["pp_approx_ratio"] / vn,
                    "val/pp_pred_size": val_pp["pp_pred_size"] / vn,
                    "val/pp_gap": val_pp["pp_gap"] / vn,
                })
            wandb.log(val_log)

            if cfg.use_postprocessing:
                cur = val_pp["pp_approx_ratio"] / vn
                if cur > best_val_pp_approx:
                    best_val_pp_approx = cur
                    torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/best.pt")
                    print(f"  🏆 New best! PP Approx: {best_val_pp_approx:.4f}")

            # EMA validation
            if ema_helper is not None:
                ema_model = ema_helper.ema_copy(raw_model)
                ema_model.cuda()
                ema_m, ema_pp, en = run_validation(ema_model, "EMA")
                if cfg.use_postprocessing:
                    ema_cur = ema_pp["pp_approx_ratio"] / en
                    print(f"  EMA PP Approx: {ema_cur:.4f}")
                    ema_log = {"val_ema/pp_approx_ratio": ema_cur, "val_ema/feasibility": ema_m["feasibility"]/en, "epoch": epoch}
                    wandb.log(ema_log)
                    if ema_cur > best_ema_pp_approx:
                        best_ema_pp_approx = ema_cur
                        torch.save(ema_helper.state_dict(), f"{cfg.checkpoint_path}/best_ema.pt")
                        print(f"  🏆 New best EMA! PP Approx: {best_ema_pp_approx:.4f}")
                del ema_model

            model.train()

        # Checkpoint every 50 epochs
        if rank == 0 and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
            torch.save(raw_model.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}.pt")
            if ema_helper:
                torch.save(ema_helper.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}_ema.pt")

    if rank == 0:
        print("\n" + "=" * 70)
        print(f"Training Complete! Best PP Approx: {best_val_pp_approx:.4f}")
        print("=" * 70)
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--feasibility_weight", type=float, default=None)
    parser.add_argument("--selection_weight", type=float, default=None)
    parser.add_argument("--feasibility_loss_type", type=str, default=None)
    parser.add_argument("--use_pe", type=int, default=None)
    parser.add_argument("--use_enhanced_features", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    # Lagrangian args
    parser.add_argument("--use_lagrangian", type=int, default=None, help="0 or 1")
    parser.add_argument("--lagrangian_lr", type=float, default=None)
    parser.add_argument("--lagrangian_target", type=float, default=None)
    parser.add_argument("--use_feasibility_warmup", type=int, default=None, help="0 or 1")
    parser.add_argument("--warmup_fraction", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--loss_mode", type=str, default=None)
    parser.add_argument("--H_cycles", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.lr is not None: cfg.lr = args.lr
    if args.run_name is not None: cfg.run_name = args.run_name
    if args.max_shards is not None: cfg.max_shards = args.max_shards
    if args.batch_size is not None: cfg.global_batch_size = args.batch_size
    if args.checkpoint_path is not None: cfg.checkpoint_path = args.checkpoint_path
    if args.grad_clip is not None: cfg.grad_clip = args.grad_clip
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.pretrained is not None: cfg.pretrained = args.pretrained
    if args.use_pe is not None: cfg.use_pe = bool(args.use_pe)
    if args.use_enhanced_features is not None: cfg.use_enhanced_features = bool(args.use_enhanced_features)
    if args.feasibility_weight is not None: cfg.loss.feasibility_weight = args.feasibility_weight
    if args.selection_weight is not None: cfg.loss.selection_weight = args.selection_weight
    if args.feasibility_loss_type is not None: cfg.loss.feasibility_loss_type = args.feasibility_loss_type
    if args.use_lagrangian is not None: cfg.loss.use_lagrangian = bool(args.use_lagrangian)
    if args.lagrangian_lr is not None: cfg.loss.lagrangian_lr = args.lagrangian_lr
    if args.lagrangian_target is not None: cfg.loss.lagrangian_target = args.lagrangian_target
    if args.use_feasibility_warmup is not None: cfg.loss.use_feasibility_warmup = bool(args.use_feasibility_warmup)
    if args.warmup_fraction is not None: cfg.loss.warmup_fraction = args.warmup_fraction
    if args.mu is not None: cfg.loss.mu = args.mu
    if args.loss_mode is not None: cfg.loss.loss_mode = args.loss_mode
    if args.H_cycles is not None: cfg.arch.H_cycles = args.H_cycles

    main(cfg)
