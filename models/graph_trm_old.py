import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

# Helper MLP for GIN with proper initialization and LayerNorm for stability
def make_mlp(in_channels, hidden_channels, out_channels, num_layers=2):
    layers = []
    layers.append(nn.Linear(in_channels, hidden_channels))
    layers.append(nn.LayerNorm(hidden_channels))  # Add LayerNorm for stability
    layers.append(nn.GELU())  # GELU is smoother than ReLU
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_channels, hidden_channels))
        layers.append(nn.LayerNorm(hidden_channels))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_channels, out_channels))
    return nn.Sequential(*layers)


class GraphTRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        input_dim = config.get("input_dim", 2) # [1, deg_norm]
        hidden_dim = config.get("hidden_dim", 64)
        self.hidden_dim = hidden_dim

        # Global class imbalance weight (computed once from dataset)
        # If not provided, will be computed per-batch as fallback
        self.pos_weight = config.get("pos_weight", None)

        # --- Loss Weights (centralized for easy experimentation) ---
        # These should be tuned so that weighted losses have similar magnitudes
        # Professor's advice: bce_loss and feasibility_loss should be ~same magnitude
        self.feasibility_weight = config.get("feasibility_weight", 50.0)

        # --- Encoder (X -> Embedding) ---
        self.x_embed = nn.Linear(input_dim, hidden_dim)
        self.x_norm = nn.LayerNorm(hidden_dim)

        # --- Recursive Core ---
        # Input to recursion: [x_emb, hidden_state_prev, prob_prev]
        # prob_prev is 1 dim (probability of being in set)
        recur_input_dim = hidden_dim + hidden_dim + 1

        # Input projection to reduce dimension before GNN
        self.input_proj = nn.Linear(recur_input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        # We use GIN (Graph Isomorphism Network) for strong expressivity
        for _ in range(config.get("num_layers", 3)):
            mlp = make_mlp(hidden_dim, hidden_dim * 2, hidden_dim)
            self.gnn_layers.append(GINConv(mlp, train_eps=True))
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        # --- Decoder (Hidden -> Probability) ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # TRM Settings
        self.max_cycles = config.get("cycles", 5)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def initial_carry(self, batch):
        # Initial state: Zero hidden state, Zero probability
        num_nodes = batch["x"].size(0)
        device = batch["x"].device
        dtype = batch["x"].dtype

        h = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype)
        y_pred = torch.zeros(num_nodes, 1, device=device, dtype=dtype)

        # Carry is a tuple
        return (h, y_pred, 0) # 0 is the current cycle count

    def forward(self, carry, batch, return_keys=None):
        x = batch["x"]
        edge_index = batch["edge_index"]
        labels = batch["y"].float()

        # Unpack carry
        h_prev, y_prev, step_count = carry

        # 1. Embed static features with normalization
        x_emb = self.x_norm(self.x_embed(x)) # [N, H]

        # 2. Prepare recursive input: Concat(x_emb, h_prev, y_prev)
        # Use detached sigmoid for stability (no gradient through y_prev sigmoid)
        y_in = torch.sigmoid(y_prev).detach()  # Detach to prevent gradient explosion

        # [N, H + H + 1]
        h_concat = torch.cat([x_emb, h_prev, y_in], dim=-1)

        # Project to hidden dim
        h_in = self.input_norm(self.input_proj(h_concat))

        # 3. GNN Pass with residual connections
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_out = conv(h_in, edge_index)
            h_out = F.gelu(h_out)
            h_in = norm(h_in + h_out)  # Residual connection + norm

        h_new = h_in # New hidden state

        # 4. Predict
        logits = self.head(h_new)

        # 5. Loss calculation (Binary Cross Entropy with label smoothing for stability)
        # Clamp logits to prevent extreme values
        logits_clamped = torch.clamp(logits.squeeze(-1), min=-10.0, max=10.0)

        # Use global pos_weight (computed once from dataset) for stable training
        # Falls back to per-batch computation if not provided
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=x.device, dtype=x.dtype)
        else:
            # Fallback: compute per-batch (less stable, may vary significantly)
            with torch.no_grad():
                pos_count = labels.sum().clamp(min=1.0)
                neg_count = (labels.numel() - pos_count).clamp(min=1.0)
                pos_weight = neg_count / pos_count

        # Main BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_clamped,
            labels,
            pos_weight=pos_weight.expand_as(labels)
        )

        # Feasibility loss: penalize selecting adjacent nodes
        # If edge (u,v) exists and both u and v are selected (logits > 0), penalize it
        probs_for_loss = torch.sigmoid(logits_clamped)
        src, dst = edge_index[0], edge_index[1]

        # Penalty: product of probabilities at edge endpoints
        # High if both endpoints are likely selected
        edge_violations = probs_for_loss[src] * probs_for_loss[dst]
        feasibility_loss = edge_violations.mean() if edge_violations.numel() > 0 else torch.tensor(0.0, device=x.device)

        # Combined loss: only BCE + feasibility
        loss = bce_loss + self.feasibility_weight * feasibility_loss

        # Metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits_clamped)
            preds_binary = (probs > 0.5).float()

            # Basic classification metrics
            tp = (preds_binary * labels).sum().float()
            fp = (preds_binary * (1 - labels)).sum().float()
            fn = ((1 - preds_binary) * labels).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            # Set size metrics
            num_pred_1s = preds_binary.sum()
            num_true_1s = labels.sum()
            set_size_ratio = num_pred_1s / (num_true_1s + 1e-8)  # >1 = too greedy, <1 = too conservative

            # Feasibility: check if predicted set is independent (no edges between predicted 1s)
            # This is expensive, so we sample check
            pred_mask = (preds_binary == 1)
            if pred_mask.sum() > 0 and edge_index.size(1) > 0:
                src, dst = edge_index[0], edge_index[1]
                violations = (pred_mask[src] & pred_mask[dst]).sum().float()
                # Raw feasibility - how independent is the predicted set?
                # This is the HONEST metric - what the model actually predicted
                feasibility_raw = 1.0 - (violations / (pred_mask.sum() + 1e-8)).clamp(max=1.0)
                num_violations = (violations / 2).ceil()  # Each edge counted twice
            else:
                feasibility_raw = torch.tensor(1.0, device=x.device)
                num_violations = torch.tensor(0.0, device=x.device)

            # Raw approx ratio - HONEST metric (what the model predicted, ignoring feasibility)
            # This shows if the model is learning to select the right NUMBER of nodes
            approx_ratio_raw = num_pred_1s / (num_true_1s + 1e-8)

            acc = (preds_binary == labels).float().mean()

            metrics = {
                "loss_total": loss.detach(),
                "loss_bce": bce_loss.detach(),
                "loss_feasibility": feasibility_loss.detach(),
                # Raw (unweighted) loss magnitudes for tuning loss weights
                # Professor's advice: bce_loss_raw and feasibility_loss_raw should be ~same magnitude
                "loss_bce_raw": bce_loss.detach(),
                "loss_feasibility_raw": feasibility_loss.detach(),
                # Weighted feasibility loss (what actually contributes to total loss)
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
                "step": torch.tensor(step_count, device=x.device)
            }

        # 6. Check termination
        step_count += 1
        all_finish = (step_count >= self.max_cycles)

        new_carry = (h_new, logits, step_count)

        # Return format expected by trainer
        # carry, loss, metrics, preds, all_finish
        return new_carry, loss, metrics, {"preds": probs.detach()}, all_finish
