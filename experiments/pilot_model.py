"""
Pilot model subclass for MIS experiments.

Adds opt-in loss terms to `GraphTransformerTRM` without modifying the original
model file. Controlled via three new config keys:

- cardinality_weight: λ_c · (Σ probs - Σ labels)^2 / num_graphs
- extra_entropy_weight: λ_e · mean(-p log p - (1-p) log (1-p))  (minimize → saturate to 0/1)
- focal_gamma: replace BCE with focal loss when > 0 (α = focal_alpha, default 0.25)

Every extra term is OFF by default (weight/gamma = 0), so the subclass matches
the base model bit-for-bit unless the pilot script flips a flag.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.graph_transformer_trm import GraphTransformerTRM


class GraphTransformerTRM_Pilot(GraphTransformerTRM):
    """Drop-in replacement that exposes three extra opt-in loss terms."""

    def __init__(self, config):
        super().__init__(config)
        self.cardinality_weight: float = float(config.get("cardinality_weight", 0.0))
        self.extra_entropy_weight: float = float(config.get("extra_entropy_weight", 0.0))
        self.focal_gamma: float = float(config.get("focal_gamma", 0.0))
        self.focal_alpha: float = float(config.get("focal_alpha", 0.25))

    def forward(self, carry, batch):
        x = batch["x"]
        edge_index = batch["edge_index"]
        labels = batch["y"].float()
        batch_vec = batch.get("batch", None)
        ptr = batch.get("ptr", None)

        y_prev, z_prev, H_step = carry

        track_steps = self.config.get("track_steps", True)
        early_stop_on_solve = self.config.get("early_stop_on_solve", False)

        x_emb = self.embed_features(batch)

        if track_steps:
            y_new, z_new, steps_to_solve, step_accs, step_feas = (
                self.deep_recursion_with_tracking(
                    x_emb, y_prev, z_prev, edge_index, batch_vec, labels,
                    early_stop=early_stop_on_solve,
                )
            )
        else:
            y_new, z_new = self.deep_recursion(x_emb, y_prev, z_prev, edge_index, batch_vec)
            step_accs, step_feas = [], []

        logits = y_new.squeeze(-1)

        # pos_weight for class imbalance (same as base)
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=x.device, dtype=x.dtype)
        else:
            with torch.no_grad():
                pos_count = labels.sum().clamp(min=1.0)
                neg_count = (labels.numel() - pos_count).clamp(min=1.0)
                pos_weight = torch.tensor(
                    [neg_count / pos_count], device=x.device, dtype=x.dtype
                )

        smooth_labels = labels
        if self.label_smoothing > 0:
            smooth_labels = (
                labels * (1 - self.label_smoothing)
                + (1 - labels) * self.label_smoothing
            )

        # --- Focal loss replaces BCE when focal_gamma > 0 --------------------
        if self.focal_gamma > 0:
            probs_for_focal = torch.sigmoid(logits)
            p_t = probs_for_focal * smooth_labels + (1 - probs_for_focal) * (1 - smooth_labels)
            alpha_t = self.focal_alpha * smooth_labels + (1 - self.focal_alpha) * (1 - smooth_labels)
            # Keep the pos_weight-like scaling factor so positives stay in the loss
            pw = smooth_labels * pos_weight + (1 - smooth_labels)
            focal_term = alpha_t * (1 - p_t).pow(self.focal_gamma) * (
                -torch.log(p_t.clamp(min=1e-8))
            )
            bce_loss = (pw * focal_term).mean()
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, smooth_labels, pos_weight=pos_weight,
            )

        # --- Feasibility loss (unchanged) ------------------------------------
        probs = torch.sigmoid(logits)
        src, dst = edge_index[0], edge_index[1]
        if self.feasibility_loss_type == "log_barrier":
            edge_violations = -torch.log(
                1 - (probs[src] * probs[dst]).clamp(max=1 - 1e-8)
            )
        elif self.feasibility_loss_type == "hinge":
            edge_violations = F.relu(probs[src] - 0.5) * F.relu(probs[dst] - 0.5)
        else:
            edge_violations = probs[src] * probs[dst]
        feasibility_loss = (
            edge_violations.mean()
            if edge_violations.numel() > 0
            else torch.tensor(0.0, device=x.device)
        )

        # --- Selection loss (unchanged) --------------------------------------
        selection_loss = torch.tensor(0.0, device=x.device)
        if self.selection_weight > 0:
            log_probs = torch.log(probs.clamp(min=1e-8))
            selection_loss = -log_probs.mean()

        # --- Cardinality loss (NEW) ------------------------------------------
        # Encourages Σ probs per graph to match Σ labels per graph.
        cardinality_loss = torch.tensor(0.0, device=x.device)
        if self.cardinality_weight > 0:
            if ptr is not None:
                num_graphs = ptr.numel() - 1
                per_graph = torch.zeros(num_graphs, device=x.device, dtype=probs.dtype)
                per_graph_opt = torch.zeros_like(per_graph)
                for g in range(num_graphs):
                    s, e = int(ptr[g].item()), int(ptr[g + 1].item())
                    per_graph[g] = probs[s:e].sum()
                    per_graph_opt[g] = labels[s:e].sum()
                diff = per_graph - per_graph_opt
                cardinality_loss = (diff.pow(2) / (per_graph_opt.clamp(min=1.0) ** 2)).mean()
            else:
                pred_sum = probs.sum()
                opt_sum = labels.sum().clamp(min=1.0)
                cardinality_loss = ((pred_sum - opt_sum) / opt_sum) ** 2

        # --- Entropy regularization (NEW) ------------------------------------
        # We MINIMIZE entropy → probs are pushed to 0 or 1 (no mode collapse at 0.5).
        entropy_loss = torch.tensor(0.0, device=x.device)
        if self.extra_entropy_weight > 0:
            p = probs.clamp(min=1e-6, max=1 - 1e-6)
            entropy_loss = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

        loss = (
            bce_loss
            + self.feasibility_weight * feasibility_loss
            + self.selection_weight * selection_loss
            + self.cardinality_weight * cardinality_loss
            + self.extra_entropy_weight * entropy_loss
        )

        # --- Metrics (mostly unchanged) --------------------------------------
        with torch.no_grad():
            preds_binary = (probs > 0.5).float()
            correct = (preds_binary == labels).float()
            confidence = torch.abs(probs - 0.5) * 2
            q_hat = (correct * confidence).mean()

            num_pred_1s = preds_binary.sum()
            num_true_1s = labels.sum()
            set_size_ratio = num_pred_1s / (num_true_1s + 1e-8)

            pred_mask = preds_binary == 1
            if pred_mask.sum() > 0 and edge_index.size(1) > 0:
                violations = (pred_mask[src] & pred_mask[dst]).sum().float()
                total_edges = float(edge_index.size(1))
                feasibility_pred = 1.0 - (violations / total_edges).clamp(max=1.0)
                num_violations = (violations / 2).ceil()
            else:
                feasibility_pred = torch.tensor(1.0, device=x.device)
                num_violations = torch.tensor(0.0, device=x.device)

            approx_ratio_pred = num_pred_1s / (num_true_1s + 1e-8)
            acc = (preds_binary == labels).float().mean()

            metrics = {
                "loss_total": loss.detach(),
                "loss_bce": bce_loss.detach(),
                "loss_feasibility": (self.feasibility_weight * feasibility_loss).detach(),
                "loss_selection": (self.selection_weight * selection_loss).detach(),
                "loss_cardinality": (self.cardinality_weight * cardinality_loss).detach(),
                "loss_entropy": (self.extra_entropy_weight * entropy_loss).detach(),
                "acc": acc,
                "num_pred_1s": num_pred_1s,
                "num_true_1s": num_true_1s,
                "set_size_ratio": set_size_ratio,
                "feasibility": feasibility_pred,
                "approx_ratio": approx_ratio_pred,
                "num_violations": num_violations,
                "step": torch.tensor(H_step, device=x.device),
                "q_hat": q_hat,
            }

            for i, (step_acc, step_fea) in enumerate(zip(step_accs, step_feas)):
                metrics[f"step_{i + 1}_acc"] = torch.tensor(step_acc, device=x.device)
                metrics[f"step_{i + 1}_feasibility"] = torch.tensor(step_fea, device=x.device)

        H_step += 1
        early_stop_threshold = self.config.get("early_stop_threshold", 0.9)
        all_finish = (q_hat > early_stop_threshold).item()
        new_carry = (y_new.detach(), z_new.detach(), H_step)

        return new_carry, loss, metrics, {"preds": probs.detach()}, all_finish
