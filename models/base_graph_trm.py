"""
Base Graph Transformer TRM - Shared Architecture for MIS

This base class contains all shared architecture components between:
- GraphTransformerTRM (supervised)
- GraphTransformerTRM_SSL (self-supervised)

Both variants use the same:
- GPS layers (local MPNN + global attention)
- Feature embeddings
- TRM recursion logic
- Output head structure

Reference:
- GraphGPS: https://arxiv.org/abs/2205.12454
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GPSConv
from torch_geometric.utils import degree


class BaseGraphTRM(nn.Module):
    """
    Base class for Graph Transformer TRM models.

    Contains all shared architecture and methods for both supervised and
    self-supervised variants.
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

        # =====================================================================
        # FEATURE EMBEDDINGS
        # =====================================================================

        # Node feature embedding (base features like [1, degree_norm])
        self.x_embed = nn.Linear(input_dim, hidden_dim - pe_dim)
        self.x_norm = nn.LayerNorm(hidden_dim - pe_dim)

        # Positional encoding embedding
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
                attn_type="multihead",
                norm="layer_norm",
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

    # =========================================================================
    # SHARED UTILITY METHODS
    # =========================================================================

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

    # =========================================================================
    # SHARED RECURSION METHODS
    # =========================================================================

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

        # Final step
        y, z = self.latent_recursion(x_emb, y, z, edge_index, batch_vec)

        return y, z
