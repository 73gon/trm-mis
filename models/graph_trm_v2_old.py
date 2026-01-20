"""
GraphTRM Version 2 - Proper TRM Implementation following Paper Version 1

Key differences from v1:
1. Deep Recursion: T-1 steps without gradients, then 1 step with gradients
2. Latent Recursion: n inner steps per deep recursion call
3. y_init based on degree heuristic (low degree = more likely in MIS)
4. z_init can be precomputed or learned
5. Proper separation of y (output) and z (latent state)

Paper pseudocode (Version 1):
    def latent_recursion(x, y, z, n=6):
        for i in range(n):
            z = net(x, y, z)
        y = net(y, z)
        return y, z

    def deep_recursion(x, y, z, n=6, T=3):
        with torch.no_grad():
            for j in range(T-1):
                y, z = latent_recursion(x, y, z, n)
        y, z = latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), output_head(y), Q_head(y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


def make_mlp(in_channels, hidden_channels, out_channels, num_layers=2):
    """Helper MLP for GIN with LayerNorm for stability"""
    layers = []
    layers.append(nn.Linear(in_channels, hidden_channels))
    layers.append(nn.LayerNorm(hidden_channels))
    layers.append(nn.GELU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_channels, hidden_channels))
        layers.append(nn.LayerNorm(hidden_channels))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_channels, out_channels))
    return nn.Sequential(*layers)


class GraphTRMv2(nn.Module):
    """
    Graph Thinking-Reasoning Machine (TRM) - Version 2

    Implements Paper Version 1 with proper deep and latent recursion.

    Config parameters:
        input_dim: Node feature dimension (default: 2)
        hidden_dim: Hidden state dimension (default: 256)
        num_layers: GNN layers per latent step (default: 2)
        L_cycles: Latent recursion steps (n in paper) (default: 6)
        H_cycles: Deep recursion steps (T in paper) (default: 3)
        use_degree_init: Whether to use degree-based y_init (default: True)
        feasibility_weight: Weight for feasibility loss (default: 50.0)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        input_dim = config.get("input_dim", 2)
        hidden_dim = config.get("hidden_dim", 256)
        self.hidden_dim = hidden_dim

        # TRM recursion settings (matching paper)
        self.L_cycles = config.get("L_cycles", 6)  # Latent recursion steps (n)
        self.H_cycles = config.get("H_cycles", 3)  # Deep recursion steps (T)

        # Backward compatibility: if 'cycles' is provided, use it
        if "cycles" in config and "L_cycles" not in config:
            # Old config: cycles = L * H, assume H=3
            total_cycles = config.get("cycles", 18)
            self.H_cycles = 3
            self.L_cycles = total_cycles // self.H_cycles

        # Initialization settings
        self.use_degree_init = config.get("use_degree_init", True)

        # Loss weights
        self.pos_weight = config.get("pos_weight", None)
        self.feasibility_weight = config.get("feasibility_weight", 50.0)

        # --- Encoder (X -> Embedding) ---
        self.x_embed = nn.Linear(input_dim, hidden_dim)
        self.x_norm = nn.LayerNorm(hidden_dim)

        # --- Latent Update Network: z = net(x, y, z) ---
        # Input: [x_emb, y_prob, z_prev]
        latent_input_dim = hidden_dim + 1 + hidden_dim
        self.latent_proj = nn.Linear(latent_input_dim, hidden_dim)
        self.latent_norm = nn.LayerNorm(hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(config.get("num_layers", 2)):
            mlp = make_mlp(hidden_dim, hidden_dim * 2, hidden_dim)
            self.gnn_layers.append(GINConv(mlp, train_eps=True))
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        # --- Output Refinement Network: y = net(y, z) ---
        # Input: [y_logits, z]
        self.output_proj = nn.Linear(1 + hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_y_init(self, batch):
        """
        Compute y_init based on degree heuristic.
        Low degree nodes are more likely to be in MIS.

        y_init[i] = 1 / (1 + degree[i])

        This gives higher probability to low-degree nodes.
        """
        x = batch["x"]
        edge_index = batch["edge_index"]
        num_nodes = x.size(0)
        device = x.device
        dtype = x.dtype

        # Compute degree for each node
        # x[:, 1] should be deg_norm = degree / max_degree if features are [1, deg_norm]
        # But let's compute from edge_index to be safe
        if edge_index.size(1) > 0:
            # Count edges per node
            degree = torch.zeros(num_nodes, device=device, dtype=dtype)
            src = edge_index[0]
            degree.scatter_add_(0, src, torch.ones_like(src, dtype=dtype))
        else:
            degree = torch.zeros(num_nodes, device=device, dtype=dtype)

        # y_init = 1 / (1 + degree), then convert to logits
        y_prob_init = 1.0 / (1.0 + degree)
        # Convert probability to logits: logit = log(p / (1-p))
        y_prob_init = y_prob_init.clamp(0.01, 0.99)  # Avoid log(0)
        y_logits_init = torch.log(y_prob_init / (1.0 - y_prob_init))

        return y_logits_init.unsqueeze(-1)  # [N, 1]

    def initial_carry(self, batch):
        """
        Initialize carry state (y, z, step_count).

        y_init: Degree-based heuristic or zeros
        z_init: Zero hidden state (could be learned or precomputed)
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

        return (y, z, 0)  # (y, z, H_step_count)

    def latent_step(self, x_emb, y_logits, z, edge_index):
        """
        One step of latent recursion: z = net(x, y, z)

        This updates the hidden state z based on:
        - x_emb: static node features
        - y_logits: current output prediction (as probability)
        - z: current hidden state
        """
        y_prob = torch.sigmoid(y_logits)

        # Concatenate inputs
        h_in = torch.cat([x_emb, y_prob, z], dim=-1)
        h_in = self.latent_norm(self.latent_proj(h_in))

        # GNN message passing
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_out = conv(h_in, edge_index)
            h_out = F.gelu(h_out)
            h_in = norm(h_in + h_out)  # Residual

        return h_in  # New z

    def output_step(self, y_logits, z):
        """
        Output refinement: y = net(y, z)

        Refines the output prediction based on current hidden state.
        """
        # Concatenate current prediction with hidden state
        h_in = torch.cat([y_logits, z], dim=-1)
        h_in = self.output_norm(self.output_proj(h_in))

        # Predict new logits
        y_new = self.output_head(h_in)
        return y_new

    def latent_recursion(self, x_emb, y, z, edge_index):
        """
        Latent recursion (inner loop): n steps of latent updates, then output refinement.

        for i in range(n):
            z = net(x, y, z)
        y = net(y, z)
        return y, z
        """
        for _ in range(self.L_cycles):
            z = self.latent_step(x_emb, y, z, edge_index)

        y = self.output_step(y, z)
        return y, z

    def deep_recursion(self, x_emb, y, z, edge_index):
        """
        Deep recursion (outer loop): T-1 steps without gradients, 1 step with gradients.

        with torch.no_grad():
            for j in range(T-1):
                y, z = latent_recursion(x, y, z, n)
        y, z = latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), y
        """
        # T-1 steps without gradients (thinking)
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                y, z = self.latent_recursion(x_emb, y, z, edge_index)

        # 1 step with gradients (learning)
        y, z = self.latent_recursion(x_emb, y, z, edge_index)

        return y, z

    def forward(self, carry, batch, return_keys=None):
        """
        Forward pass with TRM recursion.

        Note: This is called ONCE per training step (not in a loop).
        The deep_recursion handles all the internal looping.
        """
        x = batch["x"]
        edge_index = batch["edge_index"]
        labels = batch["y"].float()

        # Unpack carry
        y_prev, z_prev, H_step = carry

        # 1. Embed static features
        x_emb = self.x_norm(self.x_embed(x))

        # 2. Deep recursion (T steps total, T-1 without grad)
        y_new, z_new = self.deep_recursion(x_emb, y_prev, z_prev, edge_index)

        # 3. Compute loss and metrics
        logits_clamped = torch.clamp(y_new.squeeze(-1), min=-10.0, max=10.0)

        # BCE loss with class imbalance correction
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=x.device, dtype=x.dtype)
        else:
            with torch.no_grad():
                pos_count = labels.sum().clamp(min=1.0)
                neg_count = (labels.numel() - pos_count).clamp(min=1.0)
                pos_weight = neg_count / pos_count

        bce_loss = F.binary_cross_entropy_with_logits(
            logits_clamped, labels,
            pos_weight=pos_weight.expand_as(labels)
        )

        # Feasibility loss
        probs = torch.sigmoid(logits_clamped)
        src, dst = edge_index[0], edge_index[1]
        edge_violations = probs[src] * probs[dst]
        feasibility_loss = edge_violations.mean() if edge_violations.numel() > 0 else torch.tensor(0.0, device=x.device)

        loss = bce_loss + self.feasibility_weight * feasibility_loss

        # Metrics
        with torch.no_grad():
            preds_binary = (probs > 0.5).float()

            tp = (preds_binary * labels).sum().float()
            fp = (preds_binary * (1 - labels)).sum().float()
            fn = ((1 - preds_binary) * labels).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            num_pred_1s = preds_binary.sum()
            num_true_1s = labels.sum()
            set_size_ratio = num_pred_1s / (num_true_1s + 1e-8)

            pred_mask = (preds_binary == 1)
            if pred_mask.sum() > 0 and edge_index.size(1) > 0:
                violations = (pred_mask[src] & pred_mask[dst]).sum().float()
                feasibility_raw = 1.0 - (violations / (pred_mask.sum() + 1e-8)).clamp(max=1.0)
                num_violations = (violations / 2).ceil()
            else:
                feasibility_raw = torch.tensor(1.0, device=x.device)
                num_violations = torch.tensor(0.0, device=x.device)

            approx_ratio_raw = num_pred_1s / (num_true_1s + 1e-8)
            acc = (preds_binary == labels).float().mean()

            metrics = {
                "loss_total": loss.detach(),
                "loss_bce": bce_loss.detach(),
                "loss_feasibility": feasibility_loss.detach(),
                "loss_bce_raw": bce_loss.detach(),
                "loss_feasibility_raw": feasibility_loss.detach(),
                "loss_feasibility_weighted": (self.feasibility_weight * feasibility_loss).detach(),
                "acc": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "num_pred_1s": num_pred_1s,
                "num_true_1s": num_true_1s,
                "set_size_ratio": set_size_ratio,
                "feasibility": feasibility_raw if isinstance(feasibility_raw, torch.Tensor) else torch.tensor(feasibility_raw, device=x.device),
                "approx_ratio": approx_ratio_raw if isinstance(approx_ratio_raw, torch.Tensor) else torch.tensor(approx_ratio_raw, device=x.device),
                "num_violations": num_violations if isinstance(num_violations, torch.Tensor) else torch.tensor(num_violations, device=x.device),
                "step": torch.tensor(H_step, device=x.device)
            }

        # Update carry (detach y and z for next supervision step if using deep supervision)
        H_step += 1
        all_finish = True  # Deep recursion handles all steps internally

        new_carry = (y_new.detach(), z_new.detach(), H_step)

        return new_carry, loss, metrics, {"preds": probs.detach()}, all_finish
