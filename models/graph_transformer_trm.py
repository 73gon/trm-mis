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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GPSConv
from torch_geometric.utils import degree


class GraphTransformerTRM(nn.Module):
    """
    Graph Transformer TRM for Maximum Independent Set.

    Uses GPSConv (General, Powerful, Scalable) layers that combine:
    - Local MPNN (GINConv) for neighborhood aggregation
    - Global Multi-Head Self-Attention for long-range dependencies
    - LayerNorm and residual connections for stability

    Config:
        input_dim: Base node feature dimension (default: 2)
        pe_dim: Positional encoding dimension (default: 16)
        hidden_dim: Hidden state dimension (default: 256)
        num_layers: GPS layers per latent step (default: 2)
        num_heads: Attention heads (default: 4)
        L_cycles: Latent recursion steps (default: 6)
        H_cycles: Deep recursion steps (default: 3)
        dropout: Dropout rate (default: 0.1)
        use_degree_init: Use degree-based y_init (default: True)
        feasibility_weight: Feasibility loss weight (default: 50.0)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dimensions
        input_dim = config.get("input_dim", 2)
        pe_dim = config.get("pe_dim", 16)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 2)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)

        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim

        # TRM recursion settings
        self.L_cycles = config.get("L_cycles", 6)
        self.H_cycles = config.get("H_cycles", 3)

        # Initialization settings
        self.use_degree_init = config.get("use_degree_init", True)

        # Loss weights
        self.pos_weight = config.get("pos_weight", None)
        self.feasibility_weight = config.get("feasibility_weight", 50.0)

        # =====================================================================
        # FEATURE EMBEDDINGS
        # =====================================================================

        # Node feature embedding (base features like [1, degree_norm])
        self.x_embed = nn.Linear(input_dim, hidden_dim - pe_dim)
        self.x_norm = nn.LayerNorm(hidden_dim - pe_dim)

        # Positional encoding embedding
        # PE dimension from dataset (typically 16-20 for RWPE)
        pe_input_dim = config.get("pe_input_dim", 16)
        self.pe_embed = nn.Linear(pe_input_dim, pe_dim)
        self.pe_norm = nn.LayerNorm(pe_dim)

        # =====================================================================
        # GPS LAYERS (Local MPNN + Global Attention)
        # =====================================================================

        # Latent update: z = GPS(concat[x_emb, y, z], edge_index)
        latent_input_dim = hidden_dim + 1 + hidden_dim  # x_emb + y_prob + z
        self.latent_proj = nn.Linear(latent_input_dim, hidden_dim)
        self.latent_norm = nn.LayerNorm(hidden_dim)

        # GPS layers for latent update
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Inner MLP for GINConv
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # GPSConv combines GINConv (local) + MultiheadAttention (global)
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=GINConv(gin_mlp),
                heads=num_heads,
                dropout=dropout,
                attn_type="multihead",  # Use standard multi-head attention
                norm="layer_norm",  # Use LayerNorm for stability
            )
            self.gps_layers.append(gps_layer)

        # =====================================================================
        # OUTPUT HEAD
        # =====================================================================

        # Output refinement: y = MLP(concat[y, z])
        self.output_proj = nn.Linear(1 + hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def compute_y_init(self, batch):
        """
        Compute degree-based initialization for y.

        Nodes with lower degree are more likely to be in MIS:
            y_init = 1 / (1 + degree)

        This provides a strong inductive bias based on the well-known
        heuristic that low-degree nodes are good MIS candidates.
        """
        edge_index = batch["edge_index"]
        num_nodes = batch["x"].size(0)
        device = batch["x"].device

        # Compute node degrees
        if edge_index.size(1) > 0:
            deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        else:
            deg = torch.zeros(num_nodes, device=device)

        # y_init = 1 / (1 + degree), then convert to logits
        y_prob_init = 1.0 / (1.0 + deg)
        y_prob_init = y_prob_init.clamp(0.01, 0.99)
        y_logits_init = torch.log(y_prob_init / (1.0 - y_prob_init))

        return y_logits_init.unsqueeze(-1)

    def initial_carry(self, batch):
        """
        Initialize carry state (y, z, step_count).

        y_init: Degree-based heuristic or zeros
        z_init: Zero hidden state
        """
        num_nodes = batch["x"].size(0)
        device = batch["x"].device
        dtype = batch["x"].dtype

        # z_init: zero hidden state
        z = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype)

        # y_init: degree-based or zeros
        if self.use_degree_init:
            y = self.compute_y_init(batch)
        else:
            y = torch.zeros(num_nodes, 1, device=device, dtype=dtype)

        return (y, z, 0)

    def embed_features(self, batch):
        """
        Embed input features including positional encodings.

        Returns concatenation of:
        - Embedded node features (hidden_dim - pe_dim)
        - Embedded positional encoding (pe_dim)
        """
        x = batch["x"]

        # Embed base features
        x_emb = self.x_norm(F.gelu(self.x_embed(x)))

        # Embed positional encoding if available
        if "pe" in batch and batch["pe"] is not None:
            pe = batch["pe"]
            pe_emb = self.pe_norm(F.gelu(self.pe_embed(pe)))
            x_emb = torch.cat([x_emb, pe_emb], dim=-1)
        else:
            # No PE available - pad with zeros
            pe_emb = torch.zeros(x.size(0), self.pe_dim, device=x.device, dtype=x.dtype)
            x_emb = torch.cat([x_emb, pe_emb], dim=-1)

        return x_emb

    def latent_step(self, x_emb, y_logits, z, edge_index, batch_vec):
        """
        One step of latent recursion using GPS layers.

        z_new = GPS(concat[x_emb, sigmoid(y), z], edge_index)
        """
        y_prob = torch.sigmoid(y_logits)

        # Concatenate inputs
        h_in = torch.cat([x_emb, y_prob, z], dim=-1)
        h = self.latent_norm(F.gelu(self.latent_proj(h_in)))

        # Apply GPS layers (local MPNN + global attention)
        for gps_layer in self.gps_layers:
            h = gps_layer(h, edge_index, batch=batch_vec)

        return h  # New z

    def output_step(self, y_logits, z):
        """
        Output refinement: y_new = MLP(concat[y, z])
        """
        h_in = torch.cat([y_logits, z], dim=-1)
        h = self.output_norm(F.gelu(self.output_proj(h_in)))
        y_new = self.output_head(h)
        return y_new

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

    def latent_recursion(self, x_emb, y, z, edge_index, batch_vec):
        """
        Latent recursion (inner loop): L steps of latent updates, then output.

        for i in range(L):
            z = GPS(x_emb, y, z, edge_index)
        y = OutputHead(y, z)
        return y, z
        """
        for _ in range(self.L_cycles):
            z = self.latent_step(x_emb, y, z, edge_index, batch_vec)

        y = self.output_step(y, z)
        return y, z

    def deep_recursion(self, x_emb, y, z, edge_index, batch_vec):
        """
        Deep recursion (outer loop): All H_cycles steps with gradient.
        Each step runs L_cycles of latent recursion.
        """
        # Run all H_cycles with gradients
        for _ in range(self.H_cycles - 1):
            y, z = self.latent_recursion(x_emb, y, z, edge_index, batch_vec)

        # Final step (already had grad)
        y, z = self.latent_recursion(x_emb, y, z, edge_index, batch_vec)

        return y, z

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

        for h_cycle in range(self.H_cycles):
            # Run L_cycles of latent recursion, checking after EACH latent step
            for l_cycle in range(self.L_cycles):
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
        logits_clamped = y_new.squeeze(-1)

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
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_clamped,
            labels,
            pos_weight=pos_weight,  # Scalar, NOT expanded
        )

        # Unweighted BCE for monitoring
        bce_loss_unweighted = F.binary_cross_entropy_with_logits(logits_clamped, labels)

        # Feasibility loss
        probs = torch.sigmoid(logits_clamped)
        src, dst = edge_index[0], edge_index[1]
        edge_violations = probs[src] * probs[dst]
        feasibility_loss = edge_violations.mean() if edge_violations.numel() > 0 else torch.tensor(0.0, device=x.device)

        loss = bce_loss + self.feasibility_weight * feasibility_loss

        # Compute Q_hat: confidence that prediction is correct
        # Q_hat = 1 if model is confident and correct, 0 otherwise
        # Used for early stopping in deep supervision
        with torch.no_grad():
            preds_binary = (probs > 0.5).float()
            correct = (preds_binary == labels).float()
            confidence = torch.abs(probs - 0.5) * 2  # 0-1 scale
            q_hat = (correct * confidence).mean()  # Average confidence on correct predictions

            tp = (preds_binary * labels).sum().float()
            fp = (preds_binary * (1 - labels)).sum().float()
            fn = ((1 - preds_binary) * labels).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

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
                "loss_bce_unweighted": bce_loss_unweighted.detach(),
                "loss_feasibility_unweighted": feasibility_loss.detach(),
                "loss_feasibility": (self.feasibility_weight * feasibility_loss).detach(),
                "acc": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
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

        # Early stopping: if q_hat > threshold, we can stop
        # This implements the TRM early-stopping mechanism
        early_stop_threshold = self.config.get("early_stop_threshold", 0.9)
        all_finish = (q_hat > early_stop_threshold).item()

        new_carry = (y_new.detach(), z_new.detach(), H_step)

        return new_carry, loss, metrics, {"preds": probs.detach()}, all_finish
