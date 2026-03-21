#!/bin/bash
#SBATCH --job-name=test_threshold
#SBATCH --output=logs/test_threshold_%j.log
#SBATCH --error=logs/test_threshold_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u -c "
import torch
from dataset.mis_dataset import MISDataset, MISDatasetConfig
from models.graph_transformer_trm import GraphTransformerTRM
from models.pp import greedy_decode
import numpy as np

paths = ['data/difusco_benchmark/datasets/satlib/train/']
ds_config = MISDatasetConfig(dataset_paths=paths, global_batch_size=16, rank=0, num_replicas=1, drop_last=False, val_split=0.1, seed=42, use_pe=False, use_enhanced_features=False, max_shards=10)
val_dataset = MISDataset(ds_config, split='val')
from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Test Supervised model (best checkpoint epoch_49.pt)
print('=== Supervised Model (pw=5.0, epoch_49) ===')
model_config = {'input_dim': 2, 'hidden_dim': 256, 'num_layers': 2, 'n_heads': 2, 'L': 6, 'pe_dim': 16, 'pe_input_dim': 0, 'pos_weight': 5.0, 'feasibility_weight': 0.0, 'feasibility_loss_type': 'hinge'}
model = GraphTransformerTRM(model_config).cuda()
state_dict = torch.load('checkpoints/mis/epoch_49.pt', map_location='cuda', weights_only=False)
model.load_state_dict(state_dict)
model.eval()

results = {t: [] for t in thresholds}
with torch.no_grad():
    for batch_name, batch, batch_size in val_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        carry = model.initial_carry(batch)
        for step in range(model.config.get('num_layers', 2)):
            carry, _, _, preds, _ = model(carry, batch)
        logits = preds['preds'].squeeze(-1)
        probs = torch.sigmoid(logits)
        ptr = batch['ptr']
        edge_index = batch['edge_index']
        y_labels = batch['y']
        for i in range(ptr.shape[0] - 1):
            start, end = ptr[i].item(), ptr[i+1].item()
            g_probs = probs[start:end]
            mask = (edge_index[0] >= start) & (edge_index[0] < end)
            g_edges = edge_index[:, mask] - start
            opt_size = y_labels[start:end].sum().item()
            num_nodes = end - start
            for t in thresholds:
                pp_size, _ = greedy_decode(g_probs, g_edges, num_nodes, threshold=t)
                results[t].append(pp_size / max(1, opt_size))

for t in thresholds:
    print(f'  threshold={t:.1f}: PP Approx = {np.mean(results[t]):.4f} (std={np.std(results[t]):.4f})')

# EMA model
print()
print('=== Supervised EMA Model (pw=5.0, epoch_49_ema) ===')
state_dict_ema = torch.load('checkpoints/mis/epoch_49_ema.pt', map_location='cuda', weights_only=False)
model.load_state_dict(state_dict_ema)
model.eval()

results_ema = {t: [] for t in thresholds}
with torch.no_grad():
    for batch_name, batch, batch_size in val_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        carry = model.initial_carry(batch)
        for step in range(model.config.get('num_layers', 2)):
            carry, _, _, preds, _ = model(carry, batch)
        logits = preds['preds'].squeeze(-1)
        probs = torch.sigmoid(logits)
        ptr = batch['ptr']
        edge_index = batch['edge_index']
        y_labels = batch['y']
        for i in range(ptr.shape[0] - 1):
            start, end = ptr[i].item(), ptr[i+1].item()
            g_probs = probs[start:end]
            mask = (edge_index[0] >= start) & (edge_index[0] < end)
            g_edges = edge_index[:, mask] - start
            opt_size = y_labels[start:end].sum().item()
            num_nodes = end - start
            for t in thresholds:
                pp_size, _ = greedy_decode(g_probs, g_edges, num_nodes, threshold=t)
                results_ema[t].append(pp_size / max(1, opt_size))

for t in thresholds:
    print(f'  threshold={t:.1f}: PP Approx = {np.mean(results_ema[t]):.4f} (std={np.std(results_ema[t]):.4f})')

# All-Ones baseline
print()
print('=== All-Ones Baseline ===')
results_base = {t: [] for t in thresholds}
with torch.no_grad():
    for batch_name, batch, batch_size in val_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        ptr = batch['ptr']
        edge_index = batch['edge_index']
        y_labels = batch['y']
        for i in range(ptr.shape[0] - 1):
            start, end = ptr[i].item(), ptr[i+1].item()
            num_nodes = end - start
            g_probs = torch.ones(num_nodes, device='cuda')
            mask = (edge_index[0] >= start) & (edge_index[0] < end)
            g_edges = edge_index[:, mask] - start
            opt_size = y_labels[start:end].sum().item()
            for t in thresholds:
                pp_size, _ = greedy_decode(g_probs, g_edges, num_nodes, threshold=t)
                results_base[t].append(pp_size / max(1, opt_size))

for t in thresholds:
    print(f'  threshold={t:.1f}: PP Approx = {np.mean(results_base[t]):.4f} (std={np.std(results_base[t]):.4f})')

print()
print('Done!')
"
