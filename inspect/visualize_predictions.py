"""
Interactive MIS Visualization Tool
Visualize predictions vs ground truth for individual graphs from shards
Similar to inspect_shards.py but with model predictions
"""

import argparse
import os
import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import glob

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.graph_trm import GraphTRM


def greedy_decode(probs, edge_index, num_nodes):
    """Greedy decoding to valid independent set"""
    probs = probs.cpu().numpy()
    edge_index = edge_index.cpu().numpy()

    adj = {i: set() for i in range(num_nodes)}
    for u, v in zip(edge_index[0], edge_index[1]):
        adj[u].add(v)
        adj[v].add(u)

    sorted_nodes = np.argsort(-probs)
    selected_set = set()
    blocked_nodes = set()

    for node in sorted_nodes:
        if node in blocked_nodes:
            continue
        selected_set.add(node)
        blocked_nodes.add(node)
        for neighbor in adj[node]:
            blocked_nodes.add(neighbor)

    return selected_set


def is_independent_set(node_set, edge_index):
    """Check if a set of nodes is truly independent (no edges between them)"""
    if len(node_set) <= 1:
        return True

    edge_index = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    node_set_array = np.array(sorted(list(node_set)))

    # Build adjacency set for quick lookup
    edges_in_set = set()
    for u, v in zip(edge_index[0], edge_index[1]):
        if u in node_set and v in node_set:
            edges_in_set.add((u, v))
            edges_in_set.add((v, u))

    return len(edges_in_set) == 0


def sample_to_networkx(sample):
    """Convert sample dict to networkx graph"""
    n = int(sample["n"])
    ei = sample["edge_index"]
    if isinstance(ei, torch.Tensor):
        ei = ei.cpu().numpy()

    edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def visualize_prediction(sample, pred_set, true_set, out_path, layout_seed=42):
    """
    Visualize graph with:
    - Red nodes: True MIS
    - Blue nodes: Predicted MIS
    - Purple nodes: Both (should be empty if perfect)
    - White nodes: Neither
    """
    G = sample_to_networkx(sample)

    # Layout
    np.random.seed(layout_seed)
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=layout_seed)

    # Node colors
    node_colors = []
    for node in G.nodes():
        in_true = node in true_set
        in_pred = node in pred_set

        if in_true and in_pred:
            node_colors.append('#AA00AA')  # Purple - correct selection
        elif in_true:
            node_colors.append('#FF0000')  # Red - missed node
        elif in_pred:
            node_colors.append('#0000FF')  # Blue - false positive
        else:
            node_colors.append('#CCCCCC')  # Gray - correctly not selected

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)

    # Title and stats
    title = f"Graph {sample.get('seed', 'unknown')} | n={int(sample['n'])} | e={G.number_of_edges()}\n"
    title += f"True MIS: {len(true_set)} | Predicted: {len(pred_set)} | Approx Ratio: {len(pred_set)/(len(true_set)+1e-8):.3f}"

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF0000', label='Missed (True but not Pred)'),
        Patch(facecolor='#0000FF', label='False Positive (Pred but not True)'),
        Patch(facecolor='#AA00AA', label='Correct (Both)'),
        Patch(facecolor='#CCCCCC', label='Correct Negative (Neither)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


class MISVisualizerApp:
    def __init__(self, checkpoint, data_path, hidden_dim, num_layers, cycles, output_dir):
        self.checkpoint = checkpoint
        self.data_path = data_path
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cycles = cycles
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

        # Load dataset
        self._load_dataset()

        # Cache for predictions
        self.pred_cache = {}

    def _load_model(self):
        """Load checkpoint"""
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        # Will configure input_dim after loading dataset
        self.model = None

    def _load_dataset(self):
        """Load data"""
        import glob

        # Find all shards directly
        self.shards = sorted(glob.glob(os.path.join(self.data_path, "mis_shard_*.pt")))
        if not self.shards:
            raise FileNotFoundError(f"No mis_shard_*.pt files found in {self.data_path}")

        # Load one sample to get input_dim
        dummy_data = torch.load(self.shards[0], weights_only=False)["data"][0]
        self.input_dim = dummy_data["x"].shape[1]

        # Now load model with correct input_dim
        model_config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "cycles": self.cycles
        }
        self.model = GraphTRM(model_config).to(self.device)

        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✅ Model loaded: {self.checkpoint}")

    def predict_on_sample(self, sample):
        """Get prediction for a sample"""
        # Convert sample to batch - sample dict from shard has tensors/scalars
        x = sample["x"]
        edge_index = sample["edge_index"]
        y = sample["y"]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        y = y.to(self.device).long()

        num_nodes = x.size(0)

        batch = {
            "x": x,
            "edge_index": edge_index,
            "y": y,
            "batch": torch.zeros(num_nodes, dtype=torch.long, device=self.device),
            "num_graphs": 1,
            "ptr": torch.tensor([0, num_nodes], dtype=torch.long, device=self.device),
        }

        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            all_finish = False
            while not all_finish:
                carry, _, _, preds, all_finish = self.model(carry, batch)

            final_probs = preds["preds"].squeeze()

        return final_probs

    def interactive_loop(self):
        """Main interactive loop"""
        all_samples = []

        print("\n" + "="*70)
        print("LOADING ALL SAMPLES FROM DATASET...")
        print("="*70)

        # Load all samples from shards
        for shard_path in self.shards:
            payload = torch.load(shard_path, weights_only=False)
            all_samples.extend(payload["data"])

        print(f"✅ Loaded {len(all_samples)} samples")

        if len(all_samples) == 0:
            print("No samples found!")
            return

        current_idx = 0

        while True:
            sample = all_samples[current_idx]

            print("\n" + "="*70)
            print(f"SAMPLE {current_idx + 1}/{len(all_samples)}")
            print("="*70)

            # Get true labels
            true_set = set(np.where(sample["y"].numpy())[0])
            opt_val = sample.get("opt_value", len(true_set))

            print(f"Graph seed: {sample.get('seed', 'unknown')}")
            print(f"Nodes: {int(sample['n'])}, Edges: {int(sample['num_edges'])}")
            print(f"True MIS size: {len(true_set)}")
            print(f"True MIS nodes: {sorted(list(true_set))[:20]}{'...' if len(true_set) > 20 else ''}")

            # Get predictions
            print("\nGenerating prediction...")
            with torch.no_grad():
                final_probs = self.predict_on_sample(sample)

            # Compute raw feasibility (before greedy decode)
            preds_binary = (final_probs > 0.5).float().cpu().numpy()
            edge_index_np = sample["edge_index"].cpu().numpy()
            src, dst = edge_index_np[0], edge_index_np[1]
            raw_violations = 0
            for u, v in zip(src, dst):
                if preds_binary[u] == 1 and preds_binary[v] == 1:
                    raw_violations += 1
            num_raw_pred = int(preds_binary.sum())
            raw_feasibility = 1.0 - (raw_violations / max(1, num_raw_pred)) if num_raw_pred > 0 else 1.0

            pred_set = greedy_decode(final_probs, sample["edge_index"], int(sample["n"]))

            print(f"Predicted MIS size: {len(pred_set)}")
            print(f"Predicted MIS nodes: {sorted(list(pred_set))[:20]}{'...' if len(pred_set) > 20 else ''}")
            print(f"Approx Ratio: {len(pred_set)/(len(true_set)+1e-8):.4f}")

            # Check if predicted set is independent (greedy should always be valid)
            is_pred_independent = is_independent_set(pred_set, sample["edge_index"])
            is_true_independent = is_independent_set(true_set, sample["edge_index"])

            pred_status = "✅ Valid (Independent)" if is_pred_independent else "❌ Invalid (Contains edges)"
            true_status = "✅ Valid (Independent)" if is_true_independent else "❌ Invalid (Contains edges)"

            print(f"\nPredicted set (after greedy): {pred_status}")
            print(f"Raw feasibility (before greedy): {raw_feasibility:.4f} ({int(raw_violations)} violations in {num_raw_pred} predictions)")
            print(f"True set: {true_status}")

            # Visualize
            out_path = self.output_dir / f"sample_{current_idx:04d}.png"
            print(f"\n📊 Creating visualization: {out_path}")
            visualize_prediction(sample, pred_set, true_set, str(out_path), layout_seed=42)
            print(f"✅ Saved to {out_path}")

            # Menu
            print("\n" + "-"*70)
            print("Commands:")
            print(f"  [n] Next     [p] Previous    [g] Go to (0-{len(all_samples)-1})")
            print(f"  [q] Quit     [s] Save all    [r] Regenerate")
            print("-"*70)

            cmd = input(">> ").strip().lower()

            if cmd == 'q':
                print("Goodbye!")
                break
            elif cmd == 'n':
                current_idx = (current_idx + 1) % len(all_samples)
            elif cmd == 'p':
                current_idx = (current_idx - 1) % len(all_samples)
            elif cmd == 's':
                print("\n🔄 Generating visualizations for all samples...")
                for idx, smp in enumerate(all_samples):
                    out_path = self.output_dir / f"sample_{idx:04d}.png"
                    if out_path.exists():
                        continue
                    true_set = set(np.where(smp["y"].numpy())[0])
                    with torch.no_grad():
                        final_probs = self.predict_on_sample(smp)
                    pred_set = greedy_decode(final_probs, smp["edge_index"], int(smp["n"]))
                    visualize_prediction(smp, pred_set, true_set, str(out_path), layout_seed=42)
                    if (idx + 1) % 10 == 0:
                        print(f"  Generated {idx + 1}/{len(all_samples)}")
                print(f"✅ All visualizations saved to {self.output_dir}")
            elif cmd == 'g':
                try:
                    idx = int(input("Enter sample number (0-{}): ".format(len(all_samples)-1)))
                    if 0 <= idx < len(all_samples):
                        current_idx = idx
                    else:
                        print(f"Invalid index! Range: 0-{len(all_samples)-1}")
                except ValueError:
                    print("Invalid input!")
            elif cmd == 'r':
                print("🔄 Regenerating prediction...")
                # Already done above, just redisplay
                pass


def main():
    parser = argparse.ArgumentParser(description="Visualize MIS predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mis/epoch_99.pt")
    parser.add_argument("--data_path", type=str, default="data/mis-10k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=18)
    parser.add_argument("--output_dir", type=str, default="visualizations/mis_predictions")
    args = parser.parse_args()

    print("="*70)
    print("MIS PREDICTION VISUALIZER")
    print("="*70)

    app = MISVisualizerApp(
        args.checkpoint,
        args.data_path,
        args.hidden_dim,
        args.num_layers,
        args.cycles,
        args.output_dir
    )

    app.interactive_loop()


if __name__ == "__main__":
    main()
