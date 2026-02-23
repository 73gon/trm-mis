"""
Graph Transformer TRM - Self-Supervised Learning for Maximum Independent Set

This model uses self-supervised learning with NO ground truth labels.
Instead of BCE loss with labels, we use:

1. **Selection Loss (Maximize Set Size)**: Encourage selecting as many nodes as possible
   L_selection = -sum(p_i) / n (negative because we minimize loss)

2. **Feasibility Loss (Exponential Penalty)**: Penalize constraint violations using
   the exponential penalty method from optimization theory:

   L_feasibility = sum_{(u,v) in E} exp[mu * (p_u + p_v - 1)] / mu

Inherits shared architecture from BaseGraphTRM.

Reference:
- Penalty Methods: Nocedal & Wright, Numerical Optimization
- GraphGPS: https://arxiv.org/abs/2205.12454
- BaseGraphTRM: shared architecture components
"""

import torch

from models.base_graph_trm import BaseGraphTRM


class GraphTransformerTRM_SSL(BaseGraphTRM):
    """
    Graph Transformer TRM for Maximum Independent Set (Self-Supervised).

    Uses self-supervised learning with NO ground truth labels.
    Inherits shared architecture from BaseGraphTRM.

    Config:
        mu: Exponential penalty parameter (default: 5.0)
        feasibility_weight: Weight for feasibility loss (default: 1.0)
        selection_weight: Weight for selection (maximize set) loss (default: 1.0)
    """

    def __init__(self, config):
        super().__init__(config)

        # Self-supervised-specific loss parameters
        self.mu = config.get("mu", 5.0)  # Exponential penalty parameter
        self.feasibility_weight = config.get("feasibility_weight", 1.0)
        self.selection_weight = config.get("selection_weight", 1.0)

    def check_feasibility(self, y, edge_index):
        """
        Check if current prediction is feasible (no adjacent nodes both selected).

        Returns: (is_feasible, feasibility_ratio, num_violations)
        """
        probs = torch.sigmoid(y.squeeze(-1))
        preds_binary = (probs > 0.5).float()

        # Check feasibility (no violations)
        src, dst = edge_index[0], edge_index[1]
        pred_mask = preds_binary == 1
        if pred_mask.sum() > 0 and edge_index.size(1) > 0:
            violations = (pred_mask[src] & pred_mask[dst]).sum().float()
            total_edges = float(edge_index.size(1))
            feasibility_ratio = 1.0 - (violations / total_edges).clamp(max=1.0)
        else:
            violations = torch.tensor(0.0, device=y.device)
            feasibility_ratio = torch.tensor(1.0, device=y.device)

        is_feasible = violations == 0
        return is_feasible, feasibility_ratio, violations

    def deep_recursion_with_tracking(self, x_emb, y, z, edge_index, batch_vec, early_stop=False):
        """
        Deep recursion with step-by-step tracking (for self-supervised).

        Tracks feasibility at each step (no accuracy since no labels).

        Returns:
            y: Final prediction logits
            z: Final latent state
            step_feasibilities: List of feasibilities after each output step
            steps_to_feasible: Number of steps to achieve feasibility (-1 if never)
        """
        steps_to_feasible = -1  # -1 means never achieved feasibility
        step_feasibilities = []
        achieved_feasibility = False
        total_latent_step = 0

        for _ in range(self.H_cycles):
            # Run L_cycles of latent recursion
            for _ in range(self.L_cycles):
                z = self.latent_step(x_emb, y, z, edge_index, batch_vec)
                total_latent_step += 1

            # Output refinement at end of each H_cycle
            y = self.output_step(y, z)

            # Track progress at end of each H_cycle
            with torch.no_grad():
                is_feasible, feas_ratio, _ = self.check_feasibility(y, edge_index)
                step_feasibilities.append(feas_ratio.item() if torch.is_tensor(feas_ratio) else feas_ratio)

                if is_feasible and not achieved_feasibility:
                    steps_to_feasible = total_latent_step
                    achieved_feasibility = True

                    if early_stop:
                        break

        return y, z, step_feasibilities, steps_to_feasible

    def compute_ssl_loss(self, logits, edge_index, batch_vec=None, ptr=None):
        """
        Compute self-supervised loss for MIS.

        Loss = L_feasibility - selection_weight * L_selection

        Where:
        1. L_feasibility = sum_{(u,v) in E} exp[mu * (p_u + p_v - 1)] / mu
           - Exponential penalty for constraint violations
           - Small when p_u + p_v <= 1 (feasible)
           - Large when p_u + p_v > 1 (infeasible)

        2. L_selection = -mean(log(p_i))
           - Negative because we want to MAXIMIZE selection
           - Log probabilities provide stronger gradient signal for low-probability nodes
           - We use mean instead of sum for scale-invariance across different graph sizes

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        probs = torch.sigmoid(logits)
        src, dst = edge_index[0], edge_index[1]
        device = logits.device

        # =====================================================================
        # SELECTION LOSS: Maximize number of selected nodes
        # L_selection = -mean(log(p_i))
        # =====================================================================
        # Use log probs for stronger gradient signal; clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-8))
        selection_loss = -log_probs.mean()

        # =====================================================================
        # FEASIBILITY LOSS: Exponential penalty for constraint violations
        # L_feasibility = sum_{(u,v)} exp[mu * (p_u + p_v - 1)] / mu
        # =====================================================================
        if edge_index.size(1) > 0:
            # For each edge, compute constraint violation: p_u + p_v - 1
            # When p_u + p_v <= 1: violation <= 0, exp(mu * violation) <= 1
            # When p_u + p_v > 1: violation > 0, exp(mu * violation) >> 1
            constraint_violation = probs[src] + probs[dst] - 1.0

            # Exponential penalty: exp[mu * violation] / mu
            # The /mu term normalizes the penalty scale
            exp_penalty = torch.exp(self.mu * constraint_violation) / self.mu

            # Average over edges for scale-invariance
            feasibility_loss = exp_penalty.mean()
        else:
            feasibility_loss = torch.tensor(0.0, device=device)

        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total_loss = self.feasibility_weight * feasibility_loss + self.selection_weight * selection_loss

        loss_dict = {
            "loss_total": total_loss,
            "loss_feasibility": feasibility_loss,
            "loss_selection": selection_loss,  # This is negative (we maximize)
        }

        return total_loss, loss_dict

    def forward(self, carry, batch):
        """
        Forward pass with TRM recursion (self-supervised).

        Returns:
            new_carry: (y_detached, z_detached, step+1) for next supervision step
            loss: Total self-supervised loss
            metrics: Dict of metrics (no accuracy-based metrics)
            preds: Dict with "preds" probability tensor
            all_finish: Whether to stop (based on feasibility)
        """
        x = batch["x"]
        edge_index = batch["edge_index"]
        batch_vec = batch.get("batch", None)
        ptr = batch.get("ptr", None)

        # Unpack carry
        y_prev, z_prev, H_step = carry

        # Check if we should track steps
        track_steps = self.config.get("track_steps", True)
        early_stop_on_feasible = self.config.get("early_stop_on_feasible", False)

        # 1. Embed features (including PE)
        x_emb = self.embed_features(batch)

        # 2. Deep recursion
        if track_steps:
            y_new, z_new, step_feas, steps_to_feasible = self.deep_recursion_with_tracking(
                x_emb,
                y_prev,
                z_prev,
                edge_index,
                batch_vec,
                early_stop=early_stop_on_feasible,
            )
        else:
            y_new, z_new = self.deep_recursion(x_emb, y_prev, z_prev, edge_index, batch_vec)
            step_feas = []
            steps_to_feasible = -1

        # 3. Compute self-supervised loss
        logits = y_new.squeeze(-1)
        loss, loss_dict = self.compute_ssl_loss(logits, edge_index, batch_vec, ptr)

        # 4. Compute metrics
        probs = torch.sigmoid(logits)
        with torch.no_grad():
            preds_binary = (probs > 0.5).float()
            num_pred_1s = preds_binary.sum()
            num_nodes = probs.size(0)

            # Feasibility check
            src, dst = edge_index[0], edge_index[1]
            pred_mask = preds_binary == 1
            if pred_mask.sum() > 0 and edge_index.size(1) > 0:
                violations = (pred_mask[src] & pred_mask[dst]).sum().float()
                total_edges = float(edge_index.size(1))
                feasibility_pred = 1.0 - (violations / total_edges).clamp(max=1.0)
                num_violations = (violations / 2).ceil()  # Each violation counted twice (u,v) and (v,u)
            else:
                feasibility_pred = torch.tensor(1.0, device=x.device)
                num_violations = torch.tensor(0.0, device=x.device)

            # Selection ratio: what fraction of nodes are selected
            selection_ratio = num_pred_1s / max(num_nodes, 1)

            # Q_hat for early stopping: high if feasible and selecting reasonable amount
            # In SSL, we want high feasibility AND reasonable selection
            q_hat = feasibility_pred * (selection_ratio.clamp(min=0.1))

            metrics = {
                "loss_total": loss_dict["loss_total"].detach(),
                "loss_feasibility": (self.feasibility_weight * loss_dict["loss_feasibility"]).detach(),
                "loss_selection": (self.selection_weight * loss_dict["loss_selection"]).detach(),
                "pred_size": num_pred_1s,
                "num_nodes": torch.tensor(num_nodes, device=x.device),
                "selection_ratio": selection_ratio,
                "feasibility": feasibility_pred,
                "num_violations": num_violations,
                "step": torch.tensor(H_step, device=x.device),
                "q_hat": q_hat,
                "avg_prob": probs.mean(),
            }

            # Add per-step feasibilities
            for i, step_f in enumerate(step_feas):
                metrics[f"step_{i + 1}_feasibility"] = torch.tensor(step_f, device=x.device)

            if steps_to_feasible > 0:
                metrics["steps_to_feasible"] = torch.tensor(steps_to_feasible, device=x.device)

        # Update carry
        H_step += 1

        # Early stopping: if fully feasible and selecting enough
        early_stop_threshold = self.config.get("early_stop_threshold", 0.95)
        all_finish = (feasibility_pred > early_stop_threshold and selection_ratio > 0.1).item()

        new_carry = (y_new.detach(), z_new.detach(), H_step)

        return new_carry, loss, metrics, {"preds": probs.detach()}, all_finish
