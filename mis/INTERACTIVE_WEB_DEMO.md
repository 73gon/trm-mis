# Interactive Web Demo: Real-Time MIS Prediction Visualization

## Overview

This document details the plan for creating an interactive web application that:
1. Accepts graph input (upload or generate random)
2. Runs the trained GraphTRM model in real-time
3. Visualizes node selection probabilities as color fill (0-100%)
4. Compares predictions to ground truth (if available)
5. Shows approximation ratio and feasibility metrics

## Yes, It's Completely Doable ✅

This is a straightforward project combining:
- **Backend**: FastAPI + PyTorch model serving
- **Frontend**: React + TanStack Query + D3.js/Cytoscape for visualization
- **Real-time**: ~100-500ms inference for 50-500 node graphs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   WEB BROWSER                           │
│  ┌─────────────────────────────────────────────────────┐│
│  │  React App (TypeScript)                             ││
│  │  ├─ Graph input (upload/generate)                   ││
│  │  ├─ TanStack Query (API requests)                   ││
│  │  ├─ Cytoscape (graph visualization)                 ││
│  │  ├─ Node color intensity (probability fill)         ││
│  │  └─ Comparison view (pred vs ground truth)          ││
│  └─────────────────────────────────────────────────────┘│
│         ↕ JSON over HTTP                                 │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI)                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │ POST /api/predict                                   ││
│  │  ├─ Receive: adjacency matrix + node features      ││
│  │  ├─ Load model from checkpoint                      ││
│  │  ├─ Run inference                                   ││
│  │  ├─ Greedy decode (optional)                        ││
│  │  ├─ Compute metrics                                 ││
│  │  └─ Return: probabilities, selected set, metrics    ││
│  └─────────────────────────────────────────────────────┘│
│         ↕ PyTorch GPU                                    │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Trained Model (graph_trm.py)                        ││
│  │  ├─ Load checkpoint (e.g., epoch_50.pt)            ││
│  │  ├─ Device: CUDA if available, else CPU            ││
│  │  └─ Batch size: 1 for single inference             ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend
- **Framework**: FastAPI (async, fast, great for ML serving)
- **Server**: Uvicorn (or Gunicorn for production)
- **Model Loading**: PyTorch with model checkpointing
- **Graph Handling**: PyTorch Geometric (same as training)
- **CORS**: Enable cross-origin for frontend
- **Caching**: Optional Redis for model caching

### Frontend
- **Framework**: React 18+ with TypeScript
- **State Management**: TanStack Query (React Query v5)
- **Graph Visualization**: Cytoscape.js or D3.js + Force-Graph
- **UI Components**: Tailwind CSS + shadcn/ui (optional)
- **Build Tool**: Vite
- **HTTP Client**: TanStack Query with built-in Axios

---

## Data Flow Diagram

### Step 1: User Input → Backend Request
```
User Input (Upload or Generate)
    ↓
Adjacency Matrix (JSON)
    ↓
POST /api/predict
{
  "adjacency_matrix": [[0, 1, 0, ...], [...], ...],
  "features": [[...], [...], ...],  # optional node features
  "num_nodes": 50,
  "use_greedy_decode": true
}
```

### Step 2: Backend Inference
```
Parse JSON
    ↓
Convert to PyTorch tensors
    ↓
Create PyG Data object (edge_index from adjacency)
    ↓
Load model (cached if already loaded)
    ↓
Forward pass (single sample, batch_size=1)
    ↓
Extract node probabilities [0.0-1.0 per node]
    ↓
Optional: Greedy decode to get final set
    ↓
Compute metrics (F1, feasibility, approx_ratio if optimal known)
    ↓
Return JSON response
```

### Step 3: Backend Response
```
{
  "node_probabilities": [0.95, 0.12, 0.87, ...],  # per-node scores
  "selected_nodes": [0, 2, 4, 7, ...],             # after greedy decode
  "metrics": {
    "feasibility_raw": 0.85,
    "feasibility_greedy": 1.0,
    "num_selected": 23,
    "f1_score": 0.89
  },
  "inference_time_ms": 145,
  "model_checkpoint": "epoch_50.pt"
}
```

### Step 4: Frontend Visualization
```
Receive response
    ↓
Parse probabilities
    ↓
Update node colors:
  - 0.0 = white/light gray
  - 0.5 = light blue
  - 1.0 = dark blue (fully filled)
    ↓
Highlight selected nodes (border highlight)
    ↓
Display metrics panel
    ↓
If ground truth available:
  Compare prediction to optimal
  Show agreement/disagreement
```

---

## Implementation Plan

### Phase 1: Backend Setup (2-3 hours)

#### 1.1 Create FastAPI server skeleton

```python
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from typing import List, Optional
import time

app = FastAPI(title="MIS Predictor API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def get_model(checkpoint_path: str = "checkpoints/mis/epoch_50.pt"):
    global _model
    if _model is None:
        device = get_device()
        # Import from your project
        from models.graph_trm import GraphTRM

        _model = GraphTRM(...)  # args from config
        state = torch.load(checkpoint_path, map_location=device)
        _model.load_state_dict(state)
        _model.eval()
        _model.to(device)
    return _model

# Request/Response models
class PredictRequest(BaseModel):
    adjacency_matrix: List[List[int]]  # 0/1 matrix
    num_nodes: int
    use_greedy_decode: bool = True
    checkpoint: Optional[str] = "epoch_50.pt"

class MetricsResponse(BaseModel):
    feasibility_raw: float
    feasibility_greedy: float
    num_selected: int
    f1_score: Optional[float] = None

class PredictResponse(BaseModel):
    node_probabilities: List[float]
    selected_nodes: List[int]
    metrics: MetricsResponse
    inference_time_ms: float
    model_checkpoint: str

@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        start_time = time.time()
        device = get_device()

        # Convert to tensors
        import numpy as np
        adj_matrix = torch.tensor(request.adjacency_matrix, dtype=torch.long, device=device)

        # Create edge index from adjacency matrix
        edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()

        # Create PyG Data object
        from torch_geometric.data import Data
        data = Data(
            x=torch.ones((request.num_nodes, 1), device=device),  # node features
            edge_index=edge_index
        )

        # Inference
        model = get_model(request.checkpoint)
        with torch.no_grad():
            logits = model(data)  # [num_nodes, 1] or similar
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Post-processing
        selected_nodes = []
        if request.use_greedy_decode:
            selected_nodes = greedy_decode(probs, edge_index.cpu().numpy(), request.num_nodes)

        # Compute metrics
        metrics = compute_metrics(probs, selected_nodes, edge_index)

        inference_time = (time.time() - start_time) * 1000

        return PredictResponse(
            node_probabilities=probs.tolist(),
            selected_nodes=selected_nodes,
            metrics=metrics,
            inference_time_ms=inference_time,
            model_checkpoint=request.checkpoint
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(get_device())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 1.2 Helper functions

```python
# backend/inference.py
import numpy as np
from scipy.sparse import csr_matrix

def greedy_decode(node_probs: np.ndarray, edge_index: np.ndarray, num_nodes: int):
    """
    Greedy decoding: sort by probability, select non-adjacent nodes
    Args:
        node_probs: [num_nodes] array of probabilities
        edge_index: [2, num_edges] edge list
        num_nodes: number of nodes
    Returns:
        selected_nodes: list of selected node indices
    """
    # Build adjacency from edge index
    adj = csr_matrix(
        (np.ones(edge_index.shape[1]), edge_index),
        shape=(num_nodes, num_nodes)
    )

    # Sort by probability descending
    sorted_nodes = np.argsort(-node_probs)

    selected = []
    selected_set = set()

    for node in sorted_nodes:
        # Check if any neighbor is selected
        neighbors = adj[node].nonzero()[1]
        if not any(n in selected_set for n in neighbors):
            selected.append(int(node))
            selected_set.add(node)

    return sorted(selected)

def compute_metrics(node_probs, selected_nodes, edge_index):
    """Compute feasibility and quality metrics"""
    num_nodes = len(node_probs)

    # Check feasibility (no two selected nodes are adjacent)
    selected_set = set(selected_nodes)
    feasibility = 1.0
    violations = 0

    for u, v in zip(edge_index[0], edge_index[1]):
        if u in selected_set and v in selected_set:
            violations += 1

    if violations > 0:
        feasibility = 1.0 - (violations / len(selected_nodes))

    return {
        "feasibility_raw": max(node_probs),  # dummy
        "feasibility_greedy": feasibility,
        "num_selected": len(selected_nodes),
        "f1_score": None
    }
```

#### 1.3 Run backend
```bash
cd backend
pip install fastapi uvicorn torch torch-geometric scipy
python main.py
# Server runs on http://localhost:8000
```

---

## Graph Builder Feature 🎨

### Overview

Add an interactive graph editor that allows users to:
1. **Create nodes** (click on canvas)
2. **Connect nodes** (click node → click another node)
3. **Delete nodes/edges** (right-click or delete button)
4. **Drag nodes** (move layout around)
5. **Clear graph** (reset)
6. **Save/Load** (export as JSON)

### UI/UX Design

```
┌─────────────────────────────────────────────────────────────┐
│                    GRAPH BUILDER TAB                        │
├─────────────────────────────────────────────────────────────┤
│  [+ Add Node] [+ Add Edge] [🗑️ Delete] [Clear] [Save] [Load]│
├─────────────────────────────────────────────────────────────┤
│  Mode: ◉ Select  ○ Add Node  ○ Add Edge  ○ Delete           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│              Canvas (Cytoscape)                             │
│              ┌─────────────────────────────┐               │
│              │     ○ 0 - 90%               │               │
│              │    / \                      │               │
│              │   /   \                     │               │
│              │  ○ 1  ○ 2 - 45%             │               │
│              │  |     |  \                 │               │
│              │  |     |   ○ 3 - 75%        │               │
│              │   \   /                     │               │
│              │    ○ 4 - 60%                │               │
│              │                             │               │
│              └─────────────────────────────┘               │
├─────────────────────────────────────────────────────────────┤
│  Stats: Nodes=5  Edges=4  Density=0.40                     │
│  [Predict MIS]  [Export JSON]  [Download Image]            │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### Mode System
```typescript
type GraphEditorMode = 'select' | 'add-node' | 'add-edge' | 'delete'
// select: drag nodes, select
// add-node: click to add node
// add-edge: click node1 -> click node2 to connect
// delete: click to delete node/edge
```

#### State Management
```typescript
interface GraphBuilderState {
  nodes: Array<{ id: string; label: string; x: number; y: number }>
  edges: Array<{ source: string; target: string }>
  selectedNode: string | null
  mode: GraphEditorMode
  adjacencyMatrix: number[][]
}
```

#### Event Handling
```
User Action          →  Handler              →  Update State
────────────────────────────────────────────────────────
Click canvas         →  addNode()            →  nodes.push()
Click node 1         →  selectNode(1)        →  edges pending
Click node 2         →  completeEdge(1, 2)   →  edges.push()
Right-click node     →  deleteNode()         →  nodes.filter()
Drag node            →  moveNode()           →  update coordinates
Clear button         →  clearGraph()         →  reset all
```

### Key Components

#### GraphBuilderCanvas Component
```typescript
// src/components/GraphBuilderCanvas.tsx
import React, { useRef, useEffect, useState } from 'react'
import CytoscapeComponent from 'react-cytoscape'
import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'

cytoscape.use(fcose)

interface Props {
  mode: 'select' | 'add-node' | 'add-edge' | 'delete'
  onGraphChange: (adj: number[][]) => void
  onNodesEdgesChange: (data: { nodes: number; edges: number }) => void
}

export default function GraphBuilderCanvas({ mode, onGraphChange, onNodesEdgesChange }: Props) {
  const cyRef = useRef<cytoscape.Core | null>(null)
  const [nodes, setNodes] = useState<cytoscape.NodeDefinition[]>([])
  const [edges, setEdges] = useState<cytoscape.EdgeDefinition[]>([])
  const [pendingEdge, setPendingEdge] = useState<string | null>(null)
  const [nodeCounter, setNodeCounter] = useState(0)

  // Initialize Cytoscape
  useEffect(() => {
    const cy = cytoscape({
      container: document.getElementById('cy-builder'),
      elements: [...nodes, ...edges],
      style: cytoscape.stylesheet()
        .selector('node')
        .style({
          'background-color': '#3b82f6',
          'label': 'data(label)',
          'width': 40,
          'height': 40,
          'font-size': 12,
          'color': '#fff',
          'text-opacity': 1
        })
        .selector('node.selected')
        .style({
          'background-color': '#10b981',
          'border-width': 3,
          'border-color': '#059669'
        })
        .selector('edge')
        .style({
          'width': 2,
          'line-color': '#666',
          'target-arrow-color': '#666',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier'
        }),
      layout: {
        name: 'fcose',
        animate: true,
        animationDuration: 300
      }
    })

    cyRef.current = cy

    // Mode-specific handlers
    if (mode === 'add-node') {
      cy.on('click', (event: cytoscape.EventObject) => {
        if (event.target === cy) {
          const pos = event.position
          const id = `n${nodeCounter}`
          const newNode: cytoscape.NodeDefinition = {
            data: { id, label: nodeCounter.toString() },
            position: pos
          }
          setNodes(prev => [...prev, newNode])
          setNodeCounter(prev => prev + 1)
          updateAdjacencyMatrix([...nodes, newNode], edges)
        }
      })
    }

    if (mode === 'add-edge') {
      cy.on('click', 'node', (event: cytoscape.EventObject) => {
        const node = event.target
        if (!pendingEdge) {
          setPendingEdge(node.id())
          node.addClass('selected')
        } else if (node.id() !== pendingEdge) {
          // Create edge
          const newEdge: cytoscape.EdgeDefinition = {
            data: { source: pendingEdge, target: node.id() }
          }
          setEdges(prev => [...prev, newEdge])
          updateAdjacencyMatrix(nodes, [...edges, newEdge])

          // Reset
          cy.getElementById(pendingEdge).removeClass('selected')
          setPendingEdge(null)
        }
      })
    }

    if (mode === 'delete') {
      cy.on('click', (event: cytoscape.EventObject) => {
        const target = event.target
        if (target !== cy) {
          if (target.isNode()) {
            setNodes(prev => prev.filter(n => n.data.id !== target.id()))
            setEdges(prev => prev.filter(
              e => e.data.source !== target.id() && e.data.target !== target.id()
            ))
          } else {
            setEdges(prev => prev.filter(
              e => !(e.data.source === target.source().id() && e.data.target === target.target().id())
            ))
          }
          updateAdjacencyMatrix(nodes, edges)
        }
      })
    }

    if (mode === 'select') {
      // Default: drag to move
      cy.on('drag', 'node', () => {
        updateAdjacencyMatrix(nodes, edges)
      })
    }
  }, [mode, nodes, edges, pendingEdge, nodeCounter])

  function updateAdjacencyMatrix(nodeList: any[], edgeList: any[]) {
    const n = nodeList.length
    const adj = Array(n).fill(null).map(() => Array(n).fill(0))

    edgeList.forEach(edge => {
      const sourceIdx = nodeList.findIndex(node => node.data.id === edge.data.source)
      const targetIdx = nodeList.findIndex(node => node.data.id === edge.data.target)
      if (sourceIdx !== -1 && targetIdx !== -1) {
        adj[sourceIdx][targetIdx] = 1
        adj[targetIdx][sourceIdx] = 1
      }
    })

    onGraphChange(adj)
    onNodesEdgesChange({ nodes: n, edges: edgeList.length })
  }

  return (
    <div
      id="cy-builder"
      style={{ width: '100%', height: '500px' }}
      className="bg-slate-900 rounded border border-slate-600"
    />
  )
}
```

#### GraphBuilderControls Component
```typescript
// src/components/GraphBuilderControls.tsx
import React, { useState } from 'react'

interface Props {
  mode: 'select' | 'add-node' | 'add-edge' | 'delete'
  setMode: (mode: 'select' | 'add-node' | 'add-edge' | 'delete') => void
  onClear: () => void
  onSave: (adj: number[][]) => void
  onLoad: (adj: number[][]) => void
  nodes: number
  edges: number
}

export default function GraphBuilderControls({
  mode,
  setMode,
  onClear,
  onSave,
  onLoad,
  nodes,
  edges
}: Props) {
  const [filename, setFilename] = useState('graph.json')

  const modeButtons = [
    { value: 'select', label: '👆 Select', icon: 'pointer' },
    { value: 'add-node', label: '➕ Add Node', icon: 'plus-circle' },
    { value: 'add-edge', label: '🔗 Add Edge', icon: 'link' },
    { value: 'delete', label: '🗑️ Delete', icon: 'trash' }
  ]

  return (
    <div className="space-y-4">
      {/* Mode Selector */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-sm font-semibold text-white mb-3">Drawing Mode</h3>
        <div className="grid grid-cols-2 gap-2">
          {modeButtons.map(btn => (
            <button
              key={btn.value}
              onClick={() => setMode(btn.value as any)}
              className={`py-2 px-3 rounded text-sm transition ${
                mode === btn.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {btn.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-400 mt-3">
          {mode === 'select' && '👆 Drag nodes to move, click to select'}
          {mode === 'add-node' && '➕ Click on canvas to add nodes'}
          {mode === 'add-edge' && '🔗 Click two nodes to create edge'}
          {mode === 'delete' && '🗑️ Click node or edge to delete'}
        </p>
      </div>

      {/* Graph Stats */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-sm font-semibold text-white mb-3">Graph Stats</h3>
        <dl className="space-y-2 text-sm text-slate-300">
          <div className="flex justify-between">
            <dt>Nodes:</dt>
            <dd className="font-semibold text-blue-400">{nodes}</dd>
          </div>
          <div className="flex justify-between">
            <dt>Edges:</dt>
            <dd className="font-semibold text-blue-400">{edges}</dd>
          </div>
          <div className="flex justify-between">
            <dt>Density:</dt>
            <dd className="font-semibold text-blue-400">
              {nodes > 1 ? ((2 * edges) / (nodes * (nodes - 1))).toFixed(3) : '0'}
            </dd>
          </div>
        </dl>
      </div>

      {/* Actions */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 space-y-3">
        <button
          onClick={onClear}
          className="w-full bg-red-600 hover:bg-red-700 text-white py-2 rounded text-sm"
        >
          Clear Graph
        </button>

        <button
          onClick={() => onSave([])}
          className="w-full bg-green-600 hover:bg-green-700 text-white py-2 rounded text-sm"
        >
          Export as JSON
        </button>

        <div>
          <input
            type="file"
            accept=".json"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) {
                const reader = new FileReader()
                reader.onload = (ev) => {
                  try {
                    const data = JSON.parse(ev.target?.result as string)
                    onLoad(data.adjacency_matrix)
                  } catch {
                    alert('Invalid JSON format')
                  }
                }
                reader.readAsText(file)
              }
            }}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="block text-center w-full bg-purple-600 hover:bg-purple-700 text-white py-2 rounded text-sm cursor-pointer"
          >
            Import JSON
          </label>
        </div>
      </div>
    </div>
  )
}
```

#### Integrated App with Tabs
```typescript
// src/App.tsx - Updated with Graph Builder
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import GraphVisualizer from './components/GraphVisualizer'
import PredictionPanel from './components/PredictionPanel'
import InputPanel from './components/InputPanel'
import GraphBuilderCanvas from './components/GraphBuilderCanvas'
import GraphBuilderControls from './components/GraphBuilderControls'
import axios from 'axios'

interface PredictionResult {
  node_probabilities: number[]
  selected_nodes: number[]
  metrics: any
  inference_time_ms: number
  model_checkpoint: string
}

type Tab = 'builder' | 'predict'

export default function App() {
  const [tab, setTab] = useState<Tab>('builder')

  // Builder state
  const [adjacencyMatrix, setAdjacencyMatrix] = useState<number[][] | null>(null)
  const [builderMode, setBuilderMode] = useState<'select' | 'add-node' | 'add-edge' | 'delete'>('select')
  const [nodesEdgesCount, setNodesEdgesCount] = useState({ nodes: 0, edges: 0 })

  // Predictor state
  const [numNodes, setNumNodes] = useState(50)
  const [useGreedyDecode, setUseGreedyDecode] = useState(true)
  const [checkpoint, setCheckpoint] = useState('epoch_50.pt')

  // Prediction query
  const { data, isLoading, error } = useQuery({
    queryKey: ['predict', adjacencyMatrix, useGreedyDecode, checkpoint],
    queryFn: async () => {
      if (!adjacencyMatrix) return null
      const response = await axios.post<PredictionResult>(
        'http://localhost:8000/api/predict',
        {
          adjacency_matrix: adjacencyMatrix,
          num_nodes: adjacencyMatrix.length,
          use_greedy_decode: useGreedyDecode,
          checkpoint: checkpoint
        }
      )
      return response.data
    },
    enabled: tab === 'builder' && adjacencyMatrix !== null
  })

  const handleClearGraph = () => {
    setAdjacencyMatrix(null)
    setNodesEdgesCount({ nodes: 0, edges: 0 })
  }

  const handleLoadGraph = (adj: number[][]) => {
    setAdjacencyMatrix(adj)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <header className="border-b border-slate-700 bg-slate-900 p-6">
        <h1 className="text-3xl font-bold text-white">MIS Solver</h1>
        <p className="text-slate-300 mt-2">Interactive graph builder + real-time prediction</p>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-slate-700 px-6 bg-slate-800">
        <button
          onClick={() => setTab('builder')}
          className={`px-6 py-3 font-semibold transition ${
            tab === 'builder'
              ? 'border-b-2 border-blue-500 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          🎨 Graph Builder
        </button>
        <button
          onClick={() => setTab('predict')}
          className={`px-6 py-3 font-semibold transition ${
            tab === 'predict'
              ? 'border-b-2 border-blue-500 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          🔮 Predict MIS
        </button>
      </div>

      <div className="p-6">
        {tab === 'builder' && (
          <div className="grid grid-cols-4 gap-6">
            {/* Builder Controls */}
            <GraphBuilderControls
              mode={builderMode}
              setMode={setBuilderMode}
              onClear={handleClearGraph}
              onSave={() => {}} // Implement export
              onLoad={handleLoadGraph}
              nodes={nodesEdgesCount.nodes}
              edges={nodesEdgesCount.edges}
            />

            {/* Canvas */}
            <div className="col-span-3">
              <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Canvas</h3>
                <GraphBuilderCanvas
                  mode={builderMode}
                  onGraphChange={setAdjacencyMatrix}
                  onNodesEdgesChange={setNodesEdgesCount}
                />
              </div>

              {/* Prediction Results */}
              {adjacencyMatrix && (
                <div className="mt-6">
                  <PredictionPanel result={data} isLoading={isLoading} />
                </div>
              )}
            </div>
          </div>
        )}

        {tab === 'predict' && (
          <div className="grid grid-cols-3 gap-6">
            <InputPanel
              numNodes={numNodes}
              setNumNodes={setNumNodes}
              useGreedyDecode={useGreedyDecode}
              setUseGreedyDecode={setUseGreedyDecode}
              checkpoint={checkpoint}
              setCheckpoint={setCheckpoint}
              onGraphGenerated={setAdjacencyMatrix}
              isLoading={isLoading}
            />
            <div className="col-span-2">
              <GraphVisualizer
                adjacencyMatrix={adjacencyMatrix}
                nodeProbs={data?.node_probabilities}
                selectedNodes={data?.selected_nodes}
                isLoading={isLoading}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
```

### Features Breakdown

| Feature | Implementation | Complexity |
|---------|---|---|
| Add Node | Click canvas, append to nodes array | Easy ✅ |
| Add Edge | Select two nodes, create edge, update adjacency | Medium |
| Delete Node/Edge | Right-click or mode-based, filter array | Easy ✅ |
| Drag Nodes | Cytoscape built-in, update coordinates | Easy ✅ |
| Auto-Layout | fcose library, Cytoscape integration | Medium |
| Save/Load JSON | File I/O, JSON serialization | Easy ✅ |
| Graph Stats | Count nodes/edges, calculate density | Easy ✅ |
| Real-time Predict | TanStack Query auto-refetch on change | Medium |

### User Workflow

```
1. User lands on app → Graph Builder tab
2. Clicks "Add Node" mode → clicks canvas multiple times → nodes appear
3. Clicks "Add Edge" mode → clicks node1 → clicks node2 → edge appears
4. Auto-predicts MIS as graph builds (real-time)
5. Sees nodes colored by probability (high prob = dark blue)
6. Sees selected nodes with green border
7. Can delete nodes (switch to "Delete" mode)
8. Can export built graph as JSON
9. Can import previously saved graphs
10. Switch to "Predict" tab to compare with random/uploaded graphs
```

### Advanced Options

#### Undo/Redo Stack
```typescript
interface HistoryState {
  nodes: any[]
  edges: any[]
}

const [history, setHistory] = useState<HistoryState[]>([])
const [historyIndex, setHistoryIndex] = useState(-1)

function addToHistory(state: HistoryState) {
  setHistory(prev => [...prev.slice(0, historyIndex + 1), state])
  setHistoryIndex(prev => prev + 1)
}

function undo() {
  if (historyIndex > 0) {
    setHistoryIndex(prev => prev - 1)
    // Restore state
  }
}
```

#### Template Graphs
```typescript
const templates = {
  'star': generateStarGraph(10),           // Central node connected to all
  'cycle': generateCycleGraph(10),         // Ring of nodes
  'grid': generateGridGraph(5, 5),         // 5x5 grid
  'complete': generateCompleteGraph(8),    // Fully connected
  'bipartite': generateBipartiteGraph(5, 5) // Two groups
}
```

#### Validation
```typescript
function validateGraph(adj: number[][]) {
  const issues = []

  // Check symmetric
  for (let i = 0; i < adj.length; i++) {
    for (let j = 0; j < adj.length; j++) {
      if (adj[i][j] !== adj[j][i]) {
        issues.push('Graph must be undirected (symmetric adjacency)')
      }
    }
  }

  // Check no self-loops
  for (let i = 0; i < adj.length; i++) {
    if (adj[i][i] !== 0) {
      issues.push('No self-loops allowed')
    }
  }

  return issues
}
```

---

### Phase 2: Frontend Setup (3-4 hours)

#### 2.1 Create React + Vite project

```bash
npm create vite@latest mis-demo -- --template react-ts
cd mis-demo
npm install
npm install @tanstack/react-query axios cytoscape react-cytoscape tailwindcss
npm run dev
# Frontend runs on http://localhost:5173
```

#### 2.2 Main App Component

```typescript
// src/App.tsx
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import GraphVisualizer from './components/GraphVisualizer'
import PredictionPanel from './components/PredictionPanel'
import InputPanel from './components/InputPanel'
import axios from 'axios'

interface PredictionResult {
  node_probabilities: number[]
  selected_nodes: number[]
  metrics: {
    feasibility_raw: number
    feasibility_greedy: number
    num_selected: number
    f1_score: number | null
  }
  inference_time_ms: number
  model_checkpoint: string
}

export default function App() {
  const [adjacencyMatrix, setAdjacencyMatrix] = useState<number[][] | null>(null)
  const [numNodes, setNumNodes] = useState(50)
  const [useGreedyDecode, setUseGreedyDecode] = useState(true)
  const [checkpoint, setCheckpoint] = useState('epoch_50.pt')

  const { data, isLoading, error } = useQuery({
    queryKey: ['predict', adjacencyMatrix, useGreedyDecode, checkpoint],
    queryFn: async () => {
      if (!adjacencyMatrix) return null

      const response = await axios.post<PredictionResult>(
        'http://localhost:8000/api/predict',
        {
          adjacency_matrix: adjacencyMatrix,
          num_nodes: numNodes,
          use_greedy_decode: useGreedyDecode,
          checkpoint: checkpoint
        }
      )
      return response.data
    },
    enabled: adjacencyMatrix !== null,
  })

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <header className="border-b border-slate-700 bg-slate-900 p-6">
        <h1 className="text-3xl font-bold text-white">
          MIS Prediction Visualizer
        </h1>
        <p className="text-slate-300 mt-2">
          Real-time Maximum Independent Set solver with GraphTRM
        </p>
      </header>

      <div className="grid grid-cols-3 gap-6 p-6">
        {/* Input Panel */}
        <InputPanel
          numNodes={numNodes}
          setNumNodes={setNumNodes}
          useGreedyDecode={useGreedyDecode}
          setUseGreedyDecode={setUseGreedyDecode}
          checkpoint={checkpoint}
          setCheckpoint={setCheckpoint}
          onGraphGenerated={setAdjacencyMatrix}
          isLoading={isLoading}
        />

        {/* Graph Visualization */}
        <div className="col-span-2">
          <GraphVisualizer
            adjacencyMatrix={adjacencyMatrix}
            nodeProbs={data?.node_probabilities}
            selectedNodes={data?.selected_nodes}
            isLoading={isLoading}
          />
        </div>
      </div>

      {/* Prediction Results */}
      {data && (
        <div className="p-6">
          <PredictionPanel result={data} />
        </div>
      )}

      {error && (
        <div className="p-6 text-red-400">
          Error: {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      )}
    </div>
  )
}
```

#### 2.3 GraphVisualizer Component (Cytoscape)

```typescript
// src/components/GraphVisualizer.tsx
import React, { useEffect, useRef } from 'react'
import CytoscapeComponent from 'react-cytoscape'
import cytoscape from 'cytoscape'

interface Props {
  adjacencyMatrix: number[][] | null
  nodeProbs?: number[]
  selectedNodes?: number[]
  isLoading: boolean
}

export default function GraphVisualizer({
  adjacencyMatrix,
  nodeProbs,
  selectedNodes,
  isLoading
}: Props) {
  const cyRef = useRef<cytoscape.Core | null>(null)

  useEffect(() => {
    if (!adjacencyMatrix) return

    // Build elements for Cytoscape
    const elements: any[] = []
    const numNodes = adjacencyMatrix.length

    // Add nodes
    for (let i = 0; i < numNodes; i++) {
      const prob = nodeProbs?.[i] ?? 0.5
      const isSelected = selectedNodes?.includes(i)

      // Color intensity based on probability (0 = white, 1 = dark blue)
      const hue = 210 // blue
      const saturation = Math.round(prob * 100)
      const lightness = Math.round(100 - prob * 50)
      const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`

      elements.push({
        data: { id: `n${i}` },
        style: {
          'background-color': color,
          'border-width': isSelected ? 3 : 1,
          'border-color': isSelected ? '#10b981' : '#666',
          'label': i.toString(),
          'width': 30,
          'height': 30,
          'font-size': 10,
          'color': '#fff'
        }
      })
    }

    // Add edges
    for (let i = 0; i < numNodes; i++) {
      for (let j = i + 1; j < numNodes; j++) {
        if (adjacencyMatrix[i][j]) {
          elements.push({
            data: { source: `n${i}`, target: `n${j}` },
            style: {
              'stroke': '#666',
              'width': 1,
              'opacity': 0.5
            }
          })
        }
      }
    }

    // Initialize Cytoscape
    const cy = cytoscape({
      container: document.getElementById('cytoscape-container'),
      elements: elements,
      style: cytoscape.stylesheet()
        .selector('node')
        .style({
          'text-opacity': 1,
          'text-valign': 'center',
          'text-halign': 'center'
        })
        .selector('edge')
        .style({
          'target-arrow-shape': 'none'
        }),
      layout: {
        name: 'cose-bilkent',
        animate: true,
        animationDuration: 500
      }
    })

    cyRef.current = cy
  }, [adjacencyMatrix, nodeProbs, selectedNodes])

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-white">Graph Visualization</h2>
        <p className="text-sm text-slate-400">
          Color intensity = node probability | Green border = selected
        </p>
      </div>
      {isLoading && (
        <div className="flex items-center justify-center h-96 text-slate-400">
          <div className="animate-spin">⏳ Running inference...</div>
        </div>
      )}
      <div
        id="cytoscape-container"
        style={{ height: '600px', display: isLoading ? 'none' : 'block' }}
        className="bg-slate-900 rounded border border-slate-600"
      />
    </div>
  )
}
```

#### 2.4 InputPanel Component

```typescript
// src/components/InputPanel.tsx
import React from 'react'
import { generateRandomGraph } from '../utils/graphGenerator'

interface Props {
  numNodes: number
  setNumNodes: (n: number) => void
  useGreedyDecode: boolean
  setUseGreedyDecode: (b: boolean) => void
  checkpoint: string
  setCheckpoint: (s: string) => void
  onGraphGenerated: (adj: number[][]) => void
  isLoading: boolean
}

export default function InputPanel({
  numNodes,
  setNumNodes,
  useGreedyDecode,
  setUseGreedyDecode,
  checkpoint,
  setCheckpoint,
  onGraphGenerated,
  isLoading
}: Props) {
  const handleGenerateGraph = () => {
    const graph = generateRandomGraph(numNodes, 0.15)
    onGraphGenerated(graph)
  }

  const handleUploadFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string)
        onGraphGenerated(data.adjacency_matrix)
      } catch (err) {
        alert('Invalid JSON format. Expected: { "adjacency_matrix": [[...]] }')
      }
    }
    reader.readAsText(file)
  }

  const checkpoints = [
    'epoch_50.pt',
    'epoch_75.pt',
    'epoch_90.pt'
  ]

  return (
    <div className="space-y-6">
      {/* Generate Graph */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Input Graph</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-300 mb-2">
              Number of Nodes: {numNodes}
            </label>
            <input
              type="range"
              min="10"
              max="500"
              value={numNodes}
              onChange={(e) => setNumNodes(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <button
            onClick={handleGenerateGraph}
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded"
          >
            Generate Random Graph
          </button>

          <div className="relative">
            <input
              type="file"
              accept=".json"
              onChange={handleUploadFile}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="w-full block text-center bg-slate-700 hover:bg-slate-600 text-white py-2 rounded cursor-pointer"
            >
              Upload JSON
            </label>
          </div>
        </div>
      </div>

      {/* Model Settings */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Model Settings</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-300 mb-2">
              Checkpoint
            </label>
            <select
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
              className="w-full bg-slate-700 text-white p-2 rounded"
            >
              {checkpoints.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          <label className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={useGreedyDecode}
              onChange={(e) => setUseGreedyDecode(e.target.checked)}
            />
            <span className="text-slate-300">Use Greedy Decode</span>
          </label>

          <p className="text-xs text-slate-400">
            Greedy decode ensures all selected nodes form a valid independent set
          </p>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Legend</h3>

        <div className="space-y-2 text-sm">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-100 to-blue-900 rounded" />
            <span className="text-slate-300">Probability (white=low, blue=high)</span>
          </div>

          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 border-4 border-green-500 rounded" />
            <span className="text-slate-300">Selected (after greedy)</span>
          </div>

          <div className="flex items-center space-x-3">
            <div className="flex-1 h-px bg-slate-600" />
            <span className="text-slate-300">Edge (no conflicts)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
```

#### 2.5 PredictionPanel Component

```typescript
// src/components/PredictionPanel.tsx
import React from 'react'

interface PredictionResult {
  node_probabilities: number[]
  selected_nodes: number[]
  metrics: {
    feasibility_raw: number
    feasibility_greedy: number
    num_selected: number
    f1_score: number | null
  }
  inference_time_ms: number
  model_checkpoint: string
}

export default function PredictionPanel({ result }: { result: PredictionResult }) {
  const avgProb = result.node_probabilities.reduce((a, b) => a + b) / result.node_probabilities.length

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
      <h2 className="text-2xl font-semibold text-white mb-6">Prediction Results</h2>

      <div className="grid grid-cols-4 gap-4 mb-6">
        {/* Metric Cards */}
        <MetricCard
          label="Model Feasibility"
          value={`${(result.metrics.feasibility_greedy * 100).toFixed(1)}%`}
          description="After greedy decode"
        />

        <MetricCard
          label="Selected Nodes"
          value={result.metrics.num_selected}
          description={`of ${result.node_probabilities.length} total`}
        />

        <MetricCard
          label="Avg Probability"
          value={`${(avgProb * 100).toFixed(1)}%`}
          description="Across all nodes"}
        />

        <MetricCard
          label="Inference Time"
          value={`${result.inference_time_ms.toFixed(0)}ms`}
          description="On single GPU/CPU"
        />
      </div>

      {/* Details */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold text-white mb-3">Model Info</h3>
          <dl className="space-y-2 text-sm text-slate-300">
            <div className="flex justify-between">
              <dt>Checkpoint:</dt>
              <dd className="font-mono text-blue-400">{result.model_checkpoint}</dd>
            </div>
            <div className="flex justify-between">
              <dt>Feasibility (Raw):</dt>
              <dd>{(result.metrics.feasibility_raw * 100).toFixed(1)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt>Feasibility (Greedy):</dt>
              <dd>{(result.metrics.feasibility_greedy * 100).toFixed(1)}%</dd>
            </div>
          </dl>
        </div>

        <div>
          <h3 className="font-semibold text-white mb-3">Selected Nodes</h3>
          <div className="bg-slate-900 rounded p-3 max-h-32 overflow-y-auto">
            <code className="text-xs text-slate-300 font-mono">
              [{result.selected_nodes.join(', ')}]
            </code>
          </div>
        </div>
      </div>

      {/* Probability Distribution */}
      <div className="mt-6">
        <h3 className="font-semibold text-white mb-3">Probability Distribution</h3>
        <div className="bg-slate-900 rounded p-4">
          <div className="flex items-end gap-1">
            {result.node_probabilities.map((prob, i) => (
              <div
                key={i}
                className="flex-1 bg-blue-500 rounded-t"
                style={{ height: `${prob * 100}px` }}
                title={`Node ${i}: ${(prob * 100).toFixed(1)}%`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ label, value, description }: any) {
  return (
    <div className="bg-slate-700 rounded p-4">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-2xl font-bold text-white mt-2">{value}</p>
      <p className="text-xs text-slate-400 mt-1">{description}</p>
    </div>
  )
}
```

#### 2.6 Utility: Graph Generator

```typescript
// src/utils/graphGenerator.ts
export function generateRandomGraph(
  numNodes: number,
  edgeProbability: number = 0.15
): number[][] {
  const adj = Array(numNodes)
    .fill(null)
    .map(() => Array(numNodes).fill(0))

  for (let i = 0; i < numNodes; i++) {
    for (let j = i + 1; j < numNodes; j++) {
      if (Math.random() < edgeProbability) {
        adj[i][j] = 1
        adj[j][i] = 1
      }
    }
  }

  return adj
}

export function downloadGraphJSON(adj: number[][], filename: string = 'graph.json') {
  const data = { adjacency_matrix: adj }
  const json = JSON.stringify(data, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
}
```

---

### Phase 3: Integration (1-2 hours)

#### 3.1 Start both servers

**Terminal 1: Backend**
```bash
cd backend
python main.py
# http://localhost:8000
```

**Terminal 2: Frontend**
```bash
cd mis-demo
npm run dev
# http://localhost:5173
```

#### 3.2 Test the flow

1. Generate random graph (50 nodes)
2. Watch inference run (~200ms)
3. See nodes colored by probability
4. See selected nodes with green border
5. View metrics panel

---

## Feature Expansion (Phase 4+)

### Optional Enhancements

#### A. Ground Truth Comparison
```typescript
// If you have optimal solutions
interface ComparisonResult {
  optimalNodes: number[]
  approximationRatio: number
  agreementPercentage: number
  falsePositives: number[]
  falseNegatives: number[]
}

// Add UI to show:
// - Optimal set vs predicted set
// - Color: green=both agree, yellow=only pred, red=only optimal
```

#### B. Multiple Checkpoints Comparison
```typescript
// Compare predictions from different epochs
// Side-by-side visualization
// Show how model improves over training
```

#### C. Graph Statistics
```typescript
// Display in input panel:
// - Number of nodes
// - Number of edges
// - Edge density
// - Graph diameter
// - Clustering coefficient
```

#### D. Batch Processing
```
POST /api/predict_batch
- Upload multiple graphs
- Predict on all
- Download results CSV
```

#### E. Interactive Ground Truth Labeling
```
- User can manually mark nodes as "must include/exclude"
- Visualize constraint violations
- See how constraints affect predictions
```

#### F. Performance Analytics
```
- Log inference times per graph size
- Show model throughput
- Compare GPU vs CPU
- Benchmark against alternatives
```

---

## Architecture Decisions Explained

### Why FastAPI?
- ✅ Built for ML serving (async, fast)
- ✅ Automatic OpenAPI docs (`/docs`)
- ✅ Easy CORS, validation, error handling
- ✅ Integrates well with PyTorch

### Why TanStack Query?
- ✅ Request caching (don't re-run if inputs same)
- ✅ Loading/error states automatic
- ✅ Background refetching possible
- ✅ TypeScript support

### Why Cytoscape?
- ✅ Built for graphs, not generic visualization
- ✅ Fast even for 500+ nodes
- ✅ Easy styling (color, size, etc.)
- ✅ Interaction: drag, zoom, pan

### Why React?
- ✅ Component-based (clean UI)
- ✅ TanStack Query integration
- ✅ Fast re-renders
- ✅ Large ecosystem

---

## Performance Considerations

### Backend Optimization
```python
# Model loading cache
_model = None  # Loads once, reused

# Batch inference (if needed later)
@app.post("/api/predict_batch")
async def predict_batch(graphs: List[PredictRequest]):
    # Stack graphs into single batch
    # Run once through model
    # Split outputs
    # Much faster than multiple requests
```

### Frontend Optimization
```typescript
// TanStack Query caching
const { data } = useQuery({
  queryKey: ['predict', adj, greedy, checkpoint],  // Unique per input
  queryFn: fetchPredictions,
  staleTime: 60 * 1000  // Cache 1 minute
})

// Only refetch when inputs change
```

### Network Optimization
```python
# Compress response
from fastapi.middleware.gzip import GZIPMiddleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)

# Example: 50 nodes = ~200 bytes (compressed: ~50 bytes)
```

---

## Deployment Options

### Option 1: Local Development
```bash
Backend: localhost:8000
Frontend: localhost:5173
Perfect for demo and testing
```

### Option 2: Docker Containers
```dockerfile
# backend/Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]

# frontend/Dockerfile
FROM node:18
WORKDIR /app
COPY . .
RUN npm install && npm run build
CMD ["npm", "run", "preview"]
```

### Option 3: Cloud Deployment
- **Backend**: AWS SageMaker, HuggingFace Spaces, or Railway
- **Frontend**: Vercel, Netlify

---

## Estimated Timeline

| Phase | Component | Time | Difficulty |
|-------|-----------|------|------------|
| 1 | Backend (FastAPI + Model) | 2-3h | Medium |
| 2a | Frontend Base (React + Cytoscape) | 2-3h | Medium |
| 2b | Graph Builder Feature | 2-3h | Medium |
| 3 | Integration | 1-2h | Easy |
| **Total** | **MVP with Builder** | **8-12h** | **Medium** |
| 4+ | Enhancements (optional) | 2-5h each | Variable |

---

## Code Structure
```
project/
├── backend/
│   ├── main.py                     # FastAPI server
│   ├── inference.py                # Helper functions
│   ├── requirements.txt            # Dependencies
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Main component (tabs)
│   │   ├── components/
│   │   │   ├── GraphVisualizer.tsx           # Prediction view
│   │   │   ├── GraphBuilderCanvas.tsx        # Builder canvas
│   │   │   ├── GraphBuilderControls.tsx      # Builder controls
│   │   │   ├── InputPanel.tsx                # Input controls
│   │   │   └── PredictionPanel.tsx           # Results
│   │   ├── utils/
│   │   │   ├── graphGenerator.ts             # Random graphs
│   │   │   └── graphTemplates.ts             # Predefined templates
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
│
└── README.md
```

---

## Example Workflow

### User Interaction Flow
```
1. Visit http://localhost:5173
2. Adjust "Number of Nodes" slider → 150 nodes
3. Click "Generate Random Graph"
   → Frontend calls generateRandomGraph(150, 0.15)
   → Generates 150x150 adjacency matrix
   → Sends to backend
4. Backend receives POST /api/predict with adjacency
   → Converts to PyG Data object
   → Loads model (if not loaded)
   → Runs forward pass
   → Returns: probabilities, selected_nodes, metrics
5. Frontend receives response
   → Updates node colors (intensity = probability)
   → Highlights selected nodes (green border)
   → Shows metrics (feasibility, size, time)
6. User sees:
   → Graph with blue nodes (darker = higher prob)
   → Selected nodes with green borders
   → Bottom panel: metrics, inference time, etc.
7. Optional: User can try different checkpoint
   → Changes dropdown from epoch_50 to epoch_90
   → Auto-triggers new prediction (different results)
```

---

## Why This Approach Works

### Separation of Concerns
- **Backend**: Model serving, inference, metrics
- **Frontend**: Visualization, user input, interaction
- **Communication**: Clean JSON API

### Real-Time Feedback
- Graph generation: instant (client-side)
- Inference: ~200ms (server-side)
- Visualization: instant (client-side)
- Total UX: feels responsive

### Scalability
- Backend can serve multiple frontend clients
- Model loaded once, reused
- TanStack Query prevents duplicate requests
- Easy to add batch processing later

---

## Testing Checklist

- [ ] Backend starts without errors
- [ ] Frontend connects to backend
- [ ] Generate graph and predict (should show results)
- [ ] Probabilities displayed as color intensity
- [ ] Selected nodes have green border
- [ ] Metrics panel shows correct values
- [ ] Try different checkpoints (results change)
- [ ] Try with/without greedy decode
- [ ] Upload custom JSON graph
- [ ] Inference time reasonable (~100-500ms)
- [ ] No CORS errors
- [ ] Mobile responsive (optional)

---

## Known Limitations & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow inference | Large graph | Limit to max 1000 nodes |
| Duplicate requests | Missing cache | TanStack Query handles this |
| Stale model | Checkpoint changed | Auto-reload model on checkpoint change |
| Large graphs unreadable | Too many nodes | Force-directed layout with clustering |
| No ground truth | Optimal not computed | Make optional, skip metrics if missing |
| Slow first request | Model cold start | Warm up on backend startup |
| Graph builder lag | Too many nodes | Limit builder to 500 nodes for smooth UX |
| Edge creation complexity | Manual clicks | Add "connect selected" batch mode |
| Graph validation | Invalid inputs | Validate before sending to model |

---

## Graph Builder Testing Checklist

- [ ] Add nodes by clicking canvas
- [ ] Nodes have unique IDs and labels
- [ ] Add edges by selecting two nodes
- [ ] Edges bidirectional (undirected graph)
- [ ] Delete mode removes nodes and connected edges
- [ ] Drag nodes to rearrange layout
- [ ] Clear button resets everything
- [ ] Export creates valid JSON with adjacency matrix
- [ ] Import loads graph correctly from JSON
- [ ] Real-time prediction as graph updates
- [ ] Node colors update with probabilities
- [ ] Graph stats (nodes, edges, density) accurate
- [ ] Mode switching doesn't lose data
- [ ] Tab switching (builder ↔ predict) works
- [ ] Large graphs (100+ nodes) stay responsive

---

## Graph Builder UI Reference

### Mode Guide
```
┌─────────────────────────────────────────┐
│ 👆 SELECT Mode                          │
│ ─ Drag nodes to move                   │
│ ─ Click to select                      │
│ ─ Right-click for context menu         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ ➕ ADD NODE Mode                        │
│ ─ Click anywhere on canvas to add     │
│ ─ Node auto-positioned & labeled      │
│ ─ ID: n0, n1, n2, ... nN              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 🔗 ADD EDGE Mode                        │
│ ─ Click first node (turns green)       │
│ ─ Click second node to connect         │
│ ─ Edge automatically created           │
│ ─ Nodes deselected                     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 🗑️ DELETE Mode                          │
│ ─ Click node to delete (+ all edges)  │
│ ─ Click edge to delete only edge      │
│ ─ Cannot undo (add undo stack later)   │
└─────────────────────────────────────────┘
```

### Color Scheme
```
Blue nodes      = Normal state
Green nodes     = Selected (add edge mode)
Dark blue fill  = High probability (90%+)
Light blue fill = Low probability (10%-)
Green border    = Selected in final prediction
Gray edges      = Connections
```

---

## Advanced Graph Builder Features (Phase 4+)

### A. Graph Templates
Prebuilt graphs for quick testing:
```typescript
const templates = {
  'star-10': generateStarGraph(10),
  'cycle-12': generateCycleGraph(12),
  'grid-5x5': generateGridGraph(5, 5),
  'complete-8': generateCompleteGraph(8),
  'bipartite-6x6': generateBipartiteGraph(6, 6),
  'random-20': generateRandomGraph(20, 0.15)
}
```

### B. Undo/Redo
```typescript
// Maintain history of states
const [history, setHistory] = useState<GraphState[]>([])
const [currentIndex, setCurrentIndex] = useState(0)

// User action → save to history
// Ctrl+Z → undo
// Ctrl+Shift+Z → redo
```

### C. Copy/Paste Nodes
```typescript
// Select nodes → Ctrl+C
// Click canvas → Ctrl+V
// Duplicate with offset
```

### D. Auto-Layout Algorithms
```
Force-directed (fcose)  - Default, organic
Hierarchical (layering) - For DAGs, tree-like
Circular                - Nodes in circle
Grid                    - Regular grid pattern
```

### E. Batch Operations
```
- Select multiple nodes (Shift+Click)
- Delete all selected
- Export selection as subgraph
- Change properties (color, size)
```

### F. Import Formats
```
✅ JSON (adjacency matrix)
✅ CSV (edge list: node1,node2)
✅ GML (Graph Modeling Language)
✅ GraphML (XML-based)
✅ Paste from clipboard
```

---

## Integration with Existing Features

### Graph Builder → Predictor
```
1. User builds graph in Builder tab
2. Auto-predicts MIS as they build
3. User sees probabilities in real-time
4. Can export built graph
5. User can switch to Predictor tab
6. Graph persists between tabs
7. Can compare with other graphs
```

### Predictor → Graph Builder
```
1. User loads graph in Predictor tab
2. Can export it
3. Switch to Builder tab
4. Import the exported graph
5. Edit and modify
6. Auto-predict changes
```

### Share & Collaborate (Phase 4+)
```
1. Export graph as JSON
2. Share JSON via link/email
3. Others import JSON
4. See predictions on their copy
5. Compare models/checkpoints
```

---

## Conclusion

**This is absolutely doable!** 🚀

You now have:
1. ✅ Trained model ready
2. ✅ Loss functions computed on backend
3. ✅ Greedy decode algorithm ready
4. ✅ Metrics pipeline in place
5. ✅ **Complete Graph Builder specifications**

Just need to:
1. Wrap model in FastAPI
2. Create React frontend with TanStack Query
3. Use Cytoscape for visualization + graph building
4. Connect them with HTTP
5. Add real-time prediction as graphs update

**Estimated effort: 8-12 hours for MVP with graph builder**

### Why Graph Builder is Valuable

✅ **Educational**: Users learn how graph structure affects MIS
✅ **Interactive**: Build graphs, see probabilities instantly
✅ **Intuitive**: Visual interface beats JSON uploading
✅ **Testable**: Users test specific graph patterns
✅ **Shareable**: Export/import graphs easily
✅ **Demonstration**: Perfect for talks/demos

The graph builder transforms your demo from a **prediction tool** into an **interactive learning experience**.

Good luck! 🎨
