# Graph Builder Quick Start Guide

## What is the Graph Builder?

An interactive visual tool that lets users:
- ✅ Draw graphs by clicking nodes and edges
- ✅ Visualize probabilities as color intensity (90% = dark blue)
- ✅ See predictions update in real-time
- ✅ Export/import graphs as JSON
- ✅ Test specific graph patterns

Perfect for: demos, education, quick testing, presentations

---

## Installation (See INTERACTIVE_WEB_DEMO.md)

```bash
# Backend
cd backend
pip install fastapi uvicorn torch torch-geometric scipy
python main.py  # http://localhost:8000

# Frontend
cd frontend
npm install @tanstack/react-query cytoscape tailwindcss
npm run dev  # http://localhost:5173
```

---

## 5-Minute Quick Start

### 1. Start Servers
```bash
Terminal 1: python main.py
Terminal 2: npm run dev
```

### 2. Open Browser
```
http://localhost:5173
```

### 3. Use Graph Builder
```
1. Click "🎨 Graph Builder" tab
2. Click "➕ Add Node" button
3. Click on canvas 5 times → 5 nodes appear
4. Click "🔗 Add Edge" button
5. Click node 0 → turns green
6. Click node 1 → edge created
7. Repeat for more edges
8. Watch predictions update!
9. Nodes colored by probability
```

---

## Modes Guide

### 👆 SELECT Mode (Default)
```
• Drag nodes to move them around
• Organize graph layout
• Click to select nodes (optional)
• Don't add or remove anything
```

### ➕ ADD NODE Mode
```
• Click anywhere on canvas
• Node appears at click location
• Auto-labeled: 0, 1, 2, 3, ...
• Connected to nothing yet
```

### 🔗 ADD EDGE Mode
```
Step 1: Click first node → turns green
Step 2: Click second node → edge appears
Step 3: Edge is bidirectional (undirected)
Step 4: Both nodes return to blue
Step 5: Repeat for more edges
```

### 🗑️ DELETE Mode
```
• Click a node → node + all connected edges deleted
• Click an edge → only edge deleted
• Cannot undo (feature coming)
```

---

## Live Example: Build a Triangle

**Goal**: Create 3 nodes all connected to each other

**Steps**:
```
1. Mode: ➕ Add Node
   → Click canvas 3 times
   → You have nodes 0, 1, 2

2. Mode: 👆 Select
   → Drag nodes to triangle shape
   → Nodes positioned nicely

3. Mode: 🔗 Add Edge
   → Click 0 → click 1 → edge appears ✓
   → Click 1 → click 2 → edge appears ✓
   → Click 2 → click 0 → edge appears ✓

4. View Results
   → Triangle visible with 3 nodes, 3 edges
   → Each node probability displayed as color
   → Model predicts: max independent set = 1 node
```

---

## Visual Feedback

### Node Colors
```
🔵 Blue       = Normal node
🟢 Green      = Selected (temporarily in add-edge mode)
⚫ Dark blue   = High probability (80-100%)
⚪ Light blue  = Low probability (0-20%)
```

### Node Borders
```
─── Thin border    = Not selected
═══ Thick border   = Selected in final prediction (MIS)
🟩 Green border    = Selected in model output
```

### Graph Stats (Right Panel)
```
Nodes:    5
Edges:    4
Density:  0.40  (40% possible edges)
```

Density = 2×Edges / (Nodes×(Nodes-1))

---

## Common Tasks

### Save Your Graph
```
1. Click "Export as JSON" button
2. Browser downloads: graph.json
3. Contains: adjacency matrix [5x5 array]
4. Can share or backup
```

### Load a Previously Saved Graph
```
1. Click "Import JSON" button
2. Select graph.json file
3. Graph appears on canvas
4. Can now edit it
```

### Clear Everything
```
1. Click "Clear Graph" button
2. Canvas becomes empty
3. All nodes and edges removed
4. Start fresh
```

### Test a Specific Pattern
```
Option A: Build manually using modes
Option B: Upload JSON template
Option C: Use predefined templates (future)
```

---

## What Happens When You Predict?

### Behind the Scenes
```
Your Graph (adjacency matrix)
    ↓
[Upload to backend via HTTP]
    ↓
Load trained model
    ↓
Run inference (150-300ms)
    ↓
Get probabilities for each node
    ↓
Greedy decode (sort by prob, select non-adjacent)
    ↓
Return results
    ↓
[Visualize in browser]
```

### What You See
```
1. Nodes change color (blue = low prob, dark blue = high)
2. Some nodes get green borders (selected in MIS)
3. Metrics panel shows:
   - Feasibility: % of constraints satisfied
   - Selected: how many nodes chosen
   - Size ratio: predicted size / optimal
   - Inference time: how fast was prediction
```

---

## Tips for Testing

### Test 1: Star Graph
```
Create: 1 central node connected to all others
Predict: Central node probably high prob
MIS: All outer nodes (central not selected)
Why: Center has max edges, so excluding it maximizes set
```

### Test 2: Independent Nodes
```
Create: N nodes with NO edges
Predict: All nodes high probability (~0.9)
MIS: All N nodes selected
Why: No conflicts possible
```

### Test 3: Cycle (Ring)
```
Create: Nodes 0-1-2-...-N-0 in a circle
Predict: Alternating probabilities
MIS: Roughly every other node
Why: Greedy picks high-prob nodes, skips neighbors
```

### Test 4: Complete Graph
```
Create: Every node connected to every other
Predict: One node very high, others very low
MIS: Exactly 1 node
Why: Can't select any two nodes (all are connected)
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Nodes don't appear | Wrong mode | Check mode = ➕ Add Node |
| Can't add edges | Wrong mode | Check mode = 🔗 Add Edge |
| Predictions stale | Network lag | Wait 2-3 seconds |
| Graph too messy | Layout bad | Use Select mode to rearrange |
| Can't delete anything | Wrong mode | Check mode = 🗑️ Delete |
| Browser slow | Too many nodes | Limit to <500 nodes |
| Predictions nonsensical | Server down | Check backend (http://localhost:8000) |
| JSON doesn't load | Format wrong | Use exported JSON from this tool |

---

## JSON Format (For Manual Export)

```json
{
  "adjacency_matrix": [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
  ]
}
```

**Meaning**:
- 4 nodes (0, 1, 2, 3)
- Edges: 0-1, 1-2, 2-3, 3-0 (cycle)
- Matrix is symmetric (undirected)
- Diagonal is all zeros (no self-loops)

---

## Advanced Features (Coming Soon)

### Phase 2 Enhancements
```
✅ Undo/Redo (Ctrl+Z, Ctrl+Shift+Z)
✅ Copy/Paste nodes (Ctrl+C, Ctrl+V)
✅ Multi-select (Shift+Click)
✅ Batch delete selected
✅ Auto-layout algorithms (force-directed, hierarchical)
✅ Graph templates (star, cycle, grid, etc.)
✅ Import formats (CSV, GML, GraphML)
✅ Touch/mobile support
✅ Undo history
```

### Phase 3: Collaboration
```
✅ Share graph via URL
✅ Compare predictions from different models
✅ A/B test two checkpoints
✅ Visualize ground truth vs prediction
✅ Download prediction as image
```

---

## Key Components in Code

### GraphBuilderCanvas.tsx
- Cytoscape setup and initialization
- Mouse event handlers (click, drag)
- Mode-specific logic (add/delete/edge)
- Adjacency matrix updates

### GraphBuilderControls.tsx
- Mode buttons (select, add-node, add-edge, delete)
- Graph stats display (nodes, edges, density)
- Clear, export, import buttons
- Mode explanation text

### App.tsx Integration
- Tab switching (builder ↔ predict)
- State management (graph, mode)
- TanStack Query for predictions
- Real-time updates

---

## Real-World Use Cases

### 1. Educational Demo
```
Teaching graph algorithms?
→ Let students build graphs
→ Show MIS predictions
→ Discuss why each node selected
→ Perfect for learning!
```

### 2. Conference Talk
```
Presenting your research?
→ Live build graph on screen
→ Show real-time predictions
→ Audience impressed 🎉
→ Ask them to suggest graphs
```

### 3. Reproducibility
```
Want to test specific cases?
→ Build graphs deliberately
→ Export as JSON
→ Include in paper appendix
→ Others can verify results
```

### 4. Model Comparison
```
Testing new checkpoint?
→ Build same graph
→ Compare predictions
→ See improvements
→ Validate model progress
```

---

## Performance Notes

| Graph Size | Inference Time | UX Quality |
|-----------|---|---|
| 10-50 nodes | 50-100ms | ⭐⭐⭐⭐⭐ |
| 50-150 nodes | 100-200ms | ⭐⭐⭐⭐ |
| 150-300 nodes | 200-400ms | ⭐⭐⭐ |
| 300-500 nodes | 400-600ms | ⭐⭐ |
| 500+ nodes | 600ms+ | ⭐ (slow) |

**Recommendation**: Keep graphs < 200 nodes for smooth experience

---

## Next Steps

1. **Follow INTERACTIVE_WEB_DEMO.md** for complete implementation
2. **Start with Phase 1 & 2** (backend + frontend basics)
3. **Add graph builder** (Phase 2b, ~2-3 hours)
4. **Test with examples** (star, cycle, complete graphs)
5. **Deploy and share** with colleagues

---

## Questions?

Refer to main documents:
- **Architecture**: INTERACTIVE_WEB_DEMO.md
- **Model Details**: TRAINING.md
- **Metrics Meaning**: EVAL_METRICS.md
- **Visualization Code**: Components section in INTERACTIVE_WEB_DEMO.md

Good luck! 🚀
