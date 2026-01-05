import torch
import networkx as nx
import matplotlib.pyplot as plt

# === Load sample ===
sample = torch.load("one_sample.pt")

edge_index = sample["edge_index"].numpy()
y = sample["y"].numpy()
n = sample["x"].shape[0]

# === Reconstruct undirected graph ===
G = nx.Graph()
G.add_nodes_from(range(n))

# edge_index has both directions -> take half
edges = edge_index.T[: edge_index.shape[1] // 2]
G.add_edges_from(edges)

# === Layout ===
pos = nx.spring_layout(G, seed=42)

# === Coloring ===
node_colors = ["red" if y[i] == 1 else "lightgray" for i in range(n)]

# === Plot to file ===
plt.figure(figsize=(12, 12))
nx.draw(
    G,
    pos,
    node_color=node_colors,
    node_size=35,
    with_labels=False,
    edge_color="gray",
    alpha=0.9,
)
plt.title("Maximum Independent Set (red = selected nodes)")
plt.tight_layout()

plt.savefig("graph_with_mis.png", dpi=300)
plt.close()

print("Saved -> graph_with_mis.png")
