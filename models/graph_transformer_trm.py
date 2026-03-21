"""
Graph Transformer TRM - GPS-based Architecture for Maximum Independent Set

This model combines the TRM (Thinking-Reasoning Machine) framework with
Graph Transformers (specifically GPSConv from PyG) for better expressivity.

Key Components:
1. **GPSConv Layer**: Combines local MPNN + global self-attention
2. **Positional Encodings**: Random Walk PE (RWPE) or Laplacian PE (LPE)
3. **Enhanced Node Features**: Degree, log-degree, clustering coefficient
4. **TRM Recursion**: Deep recursion with no-grad steps + latent recursion

Architecture:
    Input Features [x, pe, degree_features]
           ↓
    Feature Embedding (Linear + LayerNorm)
           ↓
    ┌─────────────────────────────────────┐
    │  Deep Recursion (T steps)           │
    │  ┌─────────────────────────────────┐│
    │  │ Latent Recursion (L steps)      ││
    │  │  z = GPS(x_emb, y, z, edge_idx) ││
    │  │  ...                            ││
    │  │  y = OutputHead(y, z)           ││
    │  └─────────────────────────────────┘│
    │  (T-1 steps without grad)           │
    │  (1 step with grad)                 │
    └─────────────────────────────────────┘
           ↓
    Output Logits → Binary Cross-Entropy + Feasibility Loss

Reference:
- GraphGPS: https://arxiv.org/abs/2205.12454
- PyG Tutorial: https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/graph_transformer.html
"""

import torch
import torch.nn.functional as F

from models.base_graph_trm import BaseGraphTRM


class GraphTransformerTRM(BaseGraphTRM):
    """
    Graph Transformer TRM for Maximum Independent Set (Supervised).

    Uses supervised learning with ground truth labels.
    Inherits shared architecture from BaseGraphTRM.

    Config:
        pos_weight: Weight for positive class in BCE (default: auto-computed)
        feasibility_weight: Weight for feasibility loss (default: 1.0)
        feasibility_loss_type: "soft" or "hinge" (default: "soft")
    """

    def __init__(self, config):
        super().__init__(config)

        # Supervised-specific loss weights
        self.pos_weight = config.get("pos_weight", None)
        self.feasibility_weight = config.get("feasibility_weight", 1.0)
        self.feasibility_loss_type = config.get("feasibility_loss_type", "soft")

        # Hybrid: add SSL-style selection loss to encourage larger sets
        self.selection_weight = config.get("selection_weight", 0.0)

        # Label smoothing: use softer targets (e.g., 0.1/0.9 instead of 0/1)
        self.label_smoothing = config.get("label_smoothing", 0.0)

    def check_perfect_prediction(self, y, labels, edge_index):
        """
        Check if current prediction is perfect:
        - All nodes correctly classified
        - No feasibility violations (no adjacent nodes both selected)

        Returns: (is_perfect, accuracy, feasibility)
        """
        probs = torch.sigmoid(y.squeeze(-1))
        preds_binary = (probs > 0.5).float()

        # Check accuracy (all nodes correct)
        accuracy = (preds_binary == labels).float().mean()
        is_correct = accuracy == 1.0

        # Check feasibility (no violations)
        src, dst = edge_index[0], edge_index[1]
        pred_mask = preds_binary == 1
        if pred_mask.sum() > 0 and edge_index.size(1) > 0:
            violations = (pred_mask[src] & pred_mask[dst]).sum().float()
        else:
            violations = 0.0
        is_feasible = violations == 0

        is_perfect = is_correct and is_feasible
        return is_perfect, accuracy.item(), (1.0 if is_feasible else 0.0)

    def deep_recursion_with_tracking(self, x_emb, y, z, edge_index, batch_vec, labels, early_stop=False):
        """
        Deep recursion with step-by-step tracking.

        Tracks at which LATENT STEP the model achieves a perfect prediction.
        Counts total latent steps across all H_cycles (1 to H_cycles * L_cycles).
        Optionally stops early if perfect prediction is achieved.

        Returns:
            y: Final prediction logits
            z: Final latent state
            step_accuracies: List of accuracies after each output step (after each H_cycle)
            step_feasibilities: List of feasibilities after each output step
        """
        max_steps = self.H_cycles * self.L_cycles
        steps_to_solve = max_steps  # Default: took all steps
        step_accuracies = []
        step_feasibilities = []
        solved = False
        total_latent_step = 0

        for _ in range(self.H_cycles):
            # Run L_cycles of latent recursion, checking after EACH latent step
            for _ in range(self.L_cycles):
                z = self.latent_step(x_emb, y, z, edge_index, batch_vec)
                total_latent_step += 1

                # Check for perfect prediction after each latent step
                with torch.no_grad():
                    is_perfect, acc, feas = self.check_perfect_prediction(y, labels, edge_index)

                    if is_perfect and not solved:
                        steps_to_solve = total_latent_step
                        solved = True

                        if early_stop:
                            # Stop early - we found the solution
                            return (
                                y,
                                z,
                                steps_to_solve,
                                step_accuracies,
                                step_feasibilities,
                            )

            # Output refinement at end of each H_cycle
            y = self.output_step(y, z)

            # Track progress at end of each H_cycle (for per-H_cycle monitoring)
            with torch.no_grad():
                is_perfect, acc, feas = self.check_perfect_prediction(y, labels, edge_index)
                step_accuracies.append(acc)
                step_feasibilities.append(feas)

                # Re-check if we just achieved perfect after output step
                if is_perfect and not solved:
                    steps_to_solve = total_latent_step + 1  # +1 for output step
                    solved = True

                    if early_stop:
                        break

        return y, z, steps_to_solve, step_accuracies, step_feasibilities

    def forward(self, carry, batch):
        """
        Forward pass with TRM recursion.

        This implements ONE step of deep supervision.
        The training loop should call this N_supervision times.

        Returns:
            new_carry: (y_detached, z_detached, step+1) for next supervision step
            loss: Total loss for this step
            metrics: Dict of metrics
            preds: Dict with "preds" probability tensor
            all_finish: Whether to stop (based on q_hat if implemented)
        """
        x = batch["x"]
        edge_index = batch["edge_index"]
        labels = batch["y"].float()
        batch_vec = batch.get("batch", None)

        # Unpack carry
        y_prev, z_prev, H_step = carry

        # Check if we should track steps (for validation/eval mode)
        track_steps = self.config.get("track_steps", True)
        early_stop_on_solve = self.config.get("early_stop_on_solve", False)

        # 1. Embed features (including PE) - only needed once
        x_emb = self.embed_features(batch)

        # 2. Deep recursion with step tracking
        if track_steps:
            y_new, z_new, steps_to_solve, step_accs, step_feas = self.deep_recursion_with_tracking(
                x_emb,
                y_prev,
                z_prev,
                edge_index,
                batch_vec,
                labels,
                early_stop=early_stop_on_solve,
            )
        else:
            y_new, z_new = self.deep_recursion(x_emb, y_prev, z_prev, edge_index, batch_vec)
            step_accs = []
            step_feas = []

        # 3. Compute loss and metrics
        logits = y_new.squeeze(-1)

        # BCE loss - use pos_weight as a SCALAR, not expanded tensor
        # pos_weight > 1 increases the penalty for false negatives (missing MIS nodes)
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=x.device, dtype=x.dtype)
        else:
            with torch.no_grad():
                pos_count = labels.sum().clamp(min=1.0)
                neg_count = (labels.numel() - pos_count).clamp(min=1.0)
                pos_weight = torch.tensor([neg_count / pos_count], device=x.device, dtype=x.dtype)

        # Weighted BCE for training (accounts for class imbalance)
        # Apply label smoothing if configured
        smooth_labels = labels
        if self.label_smoothing > 0:
            smooth_labels = labels * (1 - self.label_smoothing) + (1 - labels) * self.label_smoothing

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            smooth_labels,
            pos_weight=pos_weight,
        )

        # Feasibility loss
        probs = torch.sigmoid(logits)
        src, dst = edge_index[0], edge_index[1]
        # Feasibility loss: penalize edges where both endpoints are predicted as selected
        if self.feasibility_loss_type == "log_barrier":
            # Log-barrier: -log(1 - p_u * p_v) — direct penalty, no exp wrapping
            edge_violations = -torch.log(1 - (probs[src] * probs[dst]).clamp(max=1 - 1e-8))
        elif self.feasibility_loss_type == "hinge":
            # Hinge loss: only penalize when BOTH endpoints > 0.5 (would be selected)
            edge_violations = F.relu(probs[src] - 0.5) * F.relu(probs[dst] - 0.5)
        else:
            # Soft loss: penalize all edges proportionally (original behavior)
            edge_violations = probs[src] * probs[dst]
        feasibility_loss = edge_violations.mean() if edge_violations.numel() > 0 else torch.tensor(0.0, device=x.device)

        # Selection loss (hybrid): encourages selecting MORE nodes
        selection_loss = torch.tensor(0.0, device=x.device)
        if self.selection_weight > 0:
            log_probs = torch.log(probs.clamp(min=1e-8))
            selection_loss = -log_probs.mean()

        loss = bce_loss + self.feasibility_weight * feasibility_loss + self.selection_weight * selection_loss

        # Compute Q_hat: confidence that prediction is correct
        # Q_hat = 1 if model is confident and correct, 0 otherwise
        # Used for early stopping in deep supervision
        with torch.no_grad():
            preds_binary = (probs > 0.5).float()
            correct = (preds_binary == labels).float()
            confidence = torch.abs(probs - 0.5) * 2  # 0-1 scale
            q_hat = (correct * confidence).mean()  # Average confidence on correct predictions

            num_pred_1s = preds_binary.sum()
            num_true_1s = labels.sum()
            set_size_ratio = num_pred_1s / (num_true_1s + 1e-8)

            pred_mask = preds_binary == 1
            if pred_mask.sum() > 0 and edge_index.size(1) > 0:
                violations = (pred_mask[src] & pred_mask[dst]).sum().float()
                # Normalize by total edges, not selected nodes
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
                "acc": acc,
                "num_pred_1s": num_pred_1s,
                "num_true_1s": num_true_1s,
                "set_size_ratio": set_size_ratio,
                "feasibility": feasibility_pred,
                "approx_ratio": approx_ratio_pred,
                "num_violations": num_violations,
                "step": torch.tensor(H_step, device=x.device),
                "q_hat": q_hat,  # For early stopping
            }

            # Add per-step accuracies for detailed analysis
            for i, (step_acc, step_fea) in enumerate(zip(step_accs, step_feas)):
                metrics[f"step_{i + 1}_acc"] = torch.tensor(step_acc, device=x.device)
                metrics[f"step_{i + 1}_feasibility"] = torch.tensor(step_fea, device=x.device)

        # Update carry - detach for next supervision step
        H_step += 1

        # Early stopping: if q_hat > threshold, we can stop - This implements the TRM early-stopping mechanism
        early_stop_threshold = self.config.get("early_stop_threshold", 0.9)
        all_finish = (q_hat > early_stop_threshold).item()

        new_carry = (y_new.detach(), z_new.detach(), H_step)

        return new_carry, loss, metrics, {"preds": probs.detach()}, all_finish
