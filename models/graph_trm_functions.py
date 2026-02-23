import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


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
