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
        self.feasibility_loss_type = config.get("feasibility_loss_type", "exponential")

        # Loss mode: "default", "pi_gnn", "reinforce"
        self.loss_mode = config.get("loss_mode", "default")

        # Deep supervision: compute loss at each H-cycle output (not just final)
        self.use_deep_supervision = config.get("use_deep_supervision", False)

        # Temperature annealing: controls sigmoid sharpness
        # tau > 1 = softer (spread probs), tau < 1 = sharper (more binary)
        self.temperature = config.get("temperature", 1.0)

        # Entropy regularization: penalize uniform high/low probs
        self.entropy_weight = config.get("entropy_weight", 0.0)

        # Exploration noise: add Gaussian noise to logits during training
        self.noise_scale = config.get("noise_scale", 0.0)

        # Degree-weighted selection: weight selection loss by 1/(1+degree)
        self.degree_weighted = config.get("degree_weighted", False)

        # REINFORCE: number of samples per graph for variance reduction
        self.reinforce_samples = config.get("reinforce_samples", 8)

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

        Loss = fw * L_feasibility + sw * L_selection + ew * L_entropy

        Where:
        1. L_feasibility: penalize adjacent node co-selection
        2. L_selection = -mean(log(p_i)): maximize selection
        3. L_entropy = -mean(H(p_i)): penalize low-entropy (uniform) predictions
           H(p) = -p*log(p) - (1-p)*log(1-p)  (maximized at p=0.5)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        probs = torch.sigmoid(logits / self.temperature)
        src, dst = edge_index[0], edge_index[1]
        device = logits.device

        # =====================================================================
        # SELECTION LOSS: Maximize number of selected nodes
        # L_selection = -mean(log(p_i))
        # =====================================================================
        log_probs = torch.log(probs.clamp(min=1e-8))
        selection_loss = -log_probs.mean()

        # =====================================================================
        # FEASIBILITY LOSS: Penalize constraint violations
        # =====================================================================
        if edge_index.size(1) > 0:
            if self.feasibility_loss_type == "log_barrier":
                feasibility_loss = -torch.log(1 - (probs[src] * probs[dst]).clamp(max=1 - 1e-8)).mean()
            elif self.feasibility_loss_type == "hinge":
                edge_violations = torch.relu(probs[src] - 0.5) * torch.relu(probs[dst] - 0.5)
                feasibility_loss = edge_violations.mean()
            else:
                constraint_violation = -torch.log(1 - (probs[src] * probs[dst]).clamp(max=1 - 1e-8))
                exp_penalty = torch.exp(self.mu * constraint_violation) / self.mu
                feasibility_loss = exp_penalty.mean()
        else:
            feasibility_loss = torch.tensor(0.0, device=device)

        # =====================================================================
        # ENTROPY REGULARIZATION: penalize uniform predictions
        # Negative entropy: low when probs are near 0 or 1, high when near 0.5
        # We MINIMIZE this → pushes probs AWAY from 0.5 toward 0 or 1
        # This breaks the equilibrium where all probs cluster at ~0.96
        # =====================================================================
        entropy_loss = torch.tensor(0.0, device=device)
        if self.entropy_weight > 0:
            p_clamped = probs.clamp(1e-8, 1 - 1e-8)
            # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
            # We want to MAXIMIZE entropy (push toward differentiation)
            # So we ADD -H(p) to loss (minimize negative entropy = maximize entropy)
            entropy = -p_clamped * torch.log(p_clamped) - (1 - p_clamped) * torch.log(1 - p_clamped)
            entropy_loss = -entropy.mean()  # Negative: maximize entropy

        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        total_loss = self.feasibility_weight * feasibility_loss + self.selection_weight * selection_loss
        if self.entropy_weight > 0:
            total_loss = total_loss + self.entropy_weight * entropy_loss

        loss_dict = {
            "loss_total": total_loss,
            "loss_feasibility": feasibility_loss,
            "loss_selection": selection_loss,
            "loss_entropy": entropy_loss,
        }

        return total_loss, loss_dict

    def compute_pi_gnn_loss(self, logits, edge_index, batch_vec=None, ptr=None):
        """
        PI-GNN-style loss (Schuetz et al. 2022):
            L = -sum(p_i)/n + fw * sum(p_u * p_v)/|E|

        Key differences from default SSL loss:
        - Linear selection (p_i instead of log(p_i)) — gradient doesn't saturate at high p
        - Simple quadratic feasibility (p_u*p_v instead of exponential) — cleaner gradients
        """
        probs = torch.sigmoid(logits / self.temperature)
        src, dst = edge_index[0], edge_index[1]
        device = logits.device

        # Add noise during training for exploration
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(logits) * self.noise_scale
            probs = torch.sigmoid((logits + noise) / self.temperature)

        # Selection: -mean(p_i) — linear, gradient = 1 everywhere
        if self.degree_weighted and edge_index.size(1) > 0:
            from torch_geometric.utils import degree

            deg = degree(edge_index[0], num_nodes=probs.size(0), dtype=torch.float)
            weights = 1.0 / (1.0 + deg)
            weights = weights / weights.mean()  # Normalize to mean=1
            selection_loss = -(probs * weights).mean()
        else:
            selection_loss = -probs.mean()

        # Feasibility: mean(p_u * p_v) — simple quadratic
        if edge_index.size(1) > 0:
            feasibility_loss = (probs[src] * probs[dst]).mean()
        else:
            feasibility_loss = torch.tensor(0.0, device=device)

        total_loss = self.selection_weight * selection_loss + self.feasibility_weight * feasibility_loss

        loss_dict = {
            "loss_total": total_loss,
            "loss_feasibility": feasibility_loss,
            "loss_selection": -selection_loss,  # Log positive for readability
            "loss_entropy": torch.tensor(0.0, device=device),
        }
        return total_loss, loss_dict

    def compute_reinforce_loss(self, logits, edge_index, batch_vec=None, ptr=None):
        """
        REINFORCE loss: sample discrete solutions, compute reward, use policy gradient.

        For each graph, sample K solutions from Bernoulli(sigmoid(logits)).
        Reward = set_size if feasible, else set_size * (1 - violation_ratio).
        Use mean reward as baseline for variance reduction.

        This fundamentally breaks the plateau because:
        - Discrete samples force the model to commit to choices
        - Reward signal comes from actual MIS quality, not continuous relaxation
        """
        probs = torch.sigmoid(logits / self.temperature)
        device = logits.device
        K = self.reinforce_samples

        # Sample K binary solutions
        probs_expanded = probs.unsqueeze(0).expand(K, -1)  # [K, N]
        samples = torch.bernoulli(probs_expanded)  # [K, N] binary

        # Compute reward for each sample
        src, dst = edge_index[0], edge_index[1]
        rewards = []
        for k in range(K):
            s = samples[k]  # [N] binary
            set_size = s.sum()

            # Count violations
            if edge_index.size(1) > 0:
                violations = (s[src] * s[dst]).sum() / 2  # Each edge counted twice
            else:
                violations = torch.tensor(0.0, device=device)

            # Reward: set_size if feasible, penalized otherwise
            if violations > 0:
                total_edges = edge_index.size(1) / 2
                penalty = violations / max(total_edges, 1)
                reward = set_size * (1.0 - penalty)
            else:
                reward = set_size
            rewards.append(reward)

        rewards = torch.stack(rewards)  # [K]
        baseline = rewards.mean()  # Variance reduction

        # REINFORCE gradient: -mean_k[(r_k - baseline) * log_prob_k]
        log_probs_pos = torch.log(probs.clamp(min=1e-8))
        log_probs_neg = torch.log((1 - probs).clamp(min=1e-8))

        reinforce_loss = torch.tensor(0.0, device=device)
        for k in range(K):
            s = samples[k]
            log_prob = (s * log_probs_pos + (1 - s) * log_probs_neg).sum()
            reinforce_loss = reinforce_loss - (rewards[k] - baseline) * log_prob

        reinforce_loss = reinforce_loss / K

        # Also compute continuous feasibility for logging
        if edge_index.size(1) > 0:
            feasibility_loss = (probs[src] * probs[dst]).mean()
        else:
            feasibility_loss = torch.tensor(0.0, device=device)

        loss_dict = {
            "loss_total": reinforce_loss,
            "loss_feasibility": feasibility_loss,
            "loss_selection": -probs.mean(),
            "loss_entropy": rewards.mean(),  # Log reward as "entropy" slot
        }
        return reinforce_loss, loss_dict

    def _compute_loss(self, logits, edge_index, batch_vec=None, ptr=None):
        """Dispatch to the appropriate loss function based on loss_mode."""
        if self.loss_mode == "pi_gnn":
            return self.compute_pi_gnn_loss(logits, edge_index, batch_vec, ptr)
        elif self.loss_mode == "reinforce":
            return self.compute_reinforce_loss(logits, edge_index, batch_vec, ptr)
        else:
            # Add noise to logits in default mode too if configured
            if self.training and self.noise_scale > 0:
                noise = torch.randn_like(logits) * self.noise_scale
                logits = logits + noise
            return self.compute_ssl_loss(logits, edge_index, batch_vec, ptr)

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
        if self.use_deep_supervision and track_steps:
            # Deep supervision: compute loss at EACH H-cycle output
            y_cur, z_cur = y_prev, z_prev
            step_feas = []
            steps_to_feasible = -1
            achieved_feasibility = False
            total_latent_step = 0
            accumulated_loss = torch.tensor(0.0, device=x.device)
            accumulated_loss_dict = None

            for h in range(self.H_cycles):
                for _ in range(self.L_cycles):
                    z_cur = self.latent_step(x_emb, y_cur, z_cur, edge_index, batch_vec)
                    total_latent_step += 1
                y_cur = self.output_step(y_cur, z_cur)

                # Compute loss at this H-cycle
                h_logits = y_cur.squeeze(-1)
                h_loss, h_loss_dict = self._compute_loss(h_logits, edge_index, batch_vec, ptr)
                accumulated_loss = accumulated_loss + h_loss

                if accumulated_loss_dict is None:
                    accumulated_loss_dict = {k: v.clone() for k, v in h_loss_dict.items()}
                else:
                    for k in accumulated_loss_dict:
                        accumulated_loss_dict[k] = accumulated_loss_dict[k] + h_loss_dict[k]

                with torch.no_grad():
                    is_feasible, feas_ratio, _ = self.check_feasibility(y_cur, edge_index)
                    step_feas.append(feas_ratio.item() if torch.is_tensor(feas_ratio) else feas_ratio)
                    if is_feasible and not achieved_feasibility:
                        steps_to_feasible = total_latent_step
                        achieved_feasibility = True

            # Average over H-cycles
            y_new = y_cur
            z_new = z_cur
            loss = accumulated_loss / self.H_cycles
            loss_dict = {k: v / self.H_cycles for k, v in accumulated_loss_dict.items()}
            loss_dict["loss_total"] = loss
        elif track_steps:
            y_new, z_new, step_feas, steps_to_feasible = self.deep_recursion_with_tracking(
                x_emb,
                y_prev,
                z_prev,
                edge_index,
                batch_vec,
                early_stop=early_stop_on_feasible,
            )
            logits = y_new.squeeze(-1)
            loss, loss_dict = self._compute_loss(logits, edge_index, batch_vec, ptr)
        else:
            y_new, z_new = self.deep_recursion(x_emb, y_prev, z_prev, edge_index, batch_vec)
            step_feas = []
            steps_to_feasible = -1
            logits = y_new.squeeze(-1)
            loss, loss_dict = self._compute_loss(logits, edge_index, batch_vec, ptr)

        # 4. Compute metrics (always use final y_new)
        final_logits = y_new.squeeze(-1)
        probs = torch.sigmoid(final_logits)
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
                "loss_entropy": (self.entropy_weight * loss_dict.get("loss_entropy", torch.tensor(0.0))).detach(),
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
