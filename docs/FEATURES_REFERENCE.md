# Node Features Reference

This document explains all node features used in the MIS-TRM model, their calculations, pros/cons, and what they represent.

---

## Table of Contents

1. [Original Features](#original-features)
2. [Local Graph Features](#local-graph-features)
3. [Relative Positional Encodings (Shortest Path)](#relative-positional-encodings)
4. [Global Positional Encodings (Laplacian)](#global-positional-encodings)
5. [Feature Summary Table](#feature-summary-table)

---

## Original Features

### 1. Constant Unit Feature

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Calculation**:
```python
features[:, 0] = 1.0
```

**Formula**:
$$f_0(i) = 1$$

**Dimension**: 1

**What it does**: Provides a learnable bias term for each node. Acts as an intercept in the neural network.

**Pros**:
- ✓ Allows the model to learn node-specific biases
- ✓ Helps with gradient flow during backpropagation
- ✓ Minimal computational cost

**Cons**:
- ✗ Carries no structural information
- ✗ Not differentiable w.r.t. graph structure

---

### 2. Normalized Degree

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Calculation**:
```python
degrees = torch.tensor(graph.degree(), dtype=torch.float32)
max_degree = degrees.max()
normalized_degree = degrees / (max_degree + 1e-8)
features[:, 1] = normalized_degree
```

**Formula**:
$$f_1(i) = \frac{\text{deg}(i)}{\max_j \text{deg}(j)}$$

**Dimension**: 1

**Range**: [0, 1]

**What it does**: Indicates local connectivity. High values = hub nodes with many neighbors; low values = peripheral nodes.

**Pros**:
- ✓ Captures local graph structure
- ✓ Fast to compute (O(|V|))
- ✓ Interpretable - directly relates to node importance
- ✓ Normalized so scale-invariant across different graphs

**Cons**:
- ✗ Loses information about degree distribution (e.g., in star graphs, all non-center nodes have degree=1)
- ✗ Doesn't capture higher-order structure (doesn't know about triangles, cliques, etc.)
- ✗ Identical for nodes with same degree (not node-specific)

---

## Local Graph Features

### 3. Clustering Coefficient

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Calculation**:
```python
nx_graph = nx.from_numpy_array(adj_matrix)
clustering = nx.clustering(nx_graph)
clustering_values = torch.tensor([clustering[i] for i in range(num_nodes)])
features[:, 2] = clustering_values
```

**Formula**:
$$\text{CC}(i) = \frac{2 \cdot |\{(u,v) : u, v \in N(i), (u,v) \in E\}|}{|N(i)|(|N(i)|-1)}$$

Where $N(i)$ = neighbors of node $i$

**Dimension**: 1

**Range**: [0, 1]

**What it does**: Measures how "clustered" a node's neighborhood is. High values = node's neighbors are highly interconnected; low values = neighbors form a sparse neighborhood.

**Pros**:
- ✓ Captures local dense structure (important for MIS - want sparse regions)
- ✓ Helps identify bottleneck nodes
- ✓ Can indicate good candidates for MIS (low clustering = sparser neighborhood)
- ✓ Geometric interpretation - fraction of possible triangles present

**Cons**:
- ✗ Computationally expensive (O(|E| + |V|·deg²))
- ✗ Not meaningful for degree-1 nodes (returns 0)
- ✗ Similar clustering coefficient can hide very different local structures
- ✗ Biased towards low-degree neighborhoods

---

### 4. K-Core Degree

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Calculation**:
```python
nx_graph = nx.from_numpy_array(adj_matrix)
k_core = nx.core_number(nx_graph)
k_core_values = torch.tensor([k_core[i] for i in range(num_nodes)])
features[:, 3] = k_core_values / (num_nodes)  # Normalized
```

**Formula**:
$$\text{k-core}(i) = \max\{k : v \text{ is in } k\text{-core}\}$$

A $k$-core is a maximal subgraph with minimum degree $k$.

**Dimension**: 1

**Range**: [0, num_nodes]

**What it does**: Measures how "deep" in the graph structure a node is. Nodes in dense cores have higher k-core values; peripheral nodes have low values.

**Pros**:
- ✓ Captures multi-scale structure (unlike degree which is local)
- ✓ Robust to outliers in degree distribution
- ✓ Identifies core vs. periphery nodes
- ✓ More stable than clustering coefficient

**Cons**:
- ✗ Computationally expensive (O(|E| + |V|))
- ✗ Only discrete values (not continuous gradient information)
- ✗ Hard to interpret in isolation
- ✗ Normalized by num_nodes which makes it scale-dependent

---

### 5. Mean Neighbor Degree

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Calculation**:
```python
degrees = torch.tensor(graph.degree(), dtype=torch.float32)
mean_neighbor_degree = torch.zeros(num_nodes)
for i in range(num_nodes):
    neighbors = list(graph.neighbors(i))
    if neighbors:
        mean_neighbor_degree[i] = degrees[neighbors].mean()
features[:, 4] = mean_neighbor_degree / (max_degree + 1e-8)
```

**Formula**:
$$f_{\text{mnd}}(i) = \frac{1}{|N(i)|} \sum_{j \in N(i)} \text{deg}(j)$$

**Dimension**: 1

**Range**: [0, 1] (after normalization)

**What it does**: Indicates whether a node is surrounded by hubs (high values) or peripheral nodes (low values). Captures local "importance context".

**Pros**:
- ✓ Captures information about neighborhood quality
- ✓ Helps identify bridge nodes connecting different density regions
- ✓ Fast to compute (O(|E|))
- ✓ Interpretable - know what kind of neighbors the node has

**Cons**:
- ✗ Correlated with degree (hubs tend to neighbor hubs)
- ✗ Doesn't capture the diversity of neighbor degrees
- ✗ Loses transitivity information (grandparent structure)
- ✗ Normalized globally which can hide local patterns

---

### 6. Neighbor Degree Variance

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Calculation**:
```python
degrees = torch.tensor(graph.degree(), dtype=torch.float32)
neighbor_degree_variance = torch.zeros(num_nodes)
for i in range(num_nodes):
    neighbors = list(graph.neighbors(i))
    if len(neighbors) > 1:
        neighbor_degrees = degrees[neighbors]
        neighbor_degree_variance[i] = neighbor_degrees.var()
    else:
        neighbor_degree_variance[i] = 0.0
features[:, 5] = neighbor_degree_variance / (max_degree + 1e-8)
```

**Formula**:
$$f_{\text{ndv}}(i) = \text{Var}\left(\{\text{deg}(j) : j \in N(i)\}\right)$$

**Dimension**: 1

**Range**: [0, ∞) (unbounded before normalization)

**What it does**: Measures how diverse the degrees of neighbors are. High variance = mixed neighborhoods (hubs and peripherals); low variance = homogeneous neighborhoods.

**Pros**:
- ✓ Captures neighborhood heterogeneity
- ✓ Identifies bridge nodes (typically have high variance)
- ✓ Helps distinguish similar-degree nodes in different contexts
- ✓ Zero-initialized for degree-1 nodes (sensible default)

**Cons**:
- ✗ Very expensive to compute variance for all neighbors
- ✗ Unbounded - normalization factor can be unstable
- ✗ Zero for low-degree nodes (less informative)
- ✗ Highly correlated with mean neighbor degree
- ✗ Sensitive to outliers in degree distribution

---

## Relative Positional Encodings

These features capture shortest-path-based distances between nodes. Computed for each node relative to all other nodes in the graph.

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

**Computation Method**:
```python
from scipy.sparse.csgraph import shortest_path
shortest_paths = shortest_path(adj_matrix, directed=False)
# Compute statistics for each node
```

### 7. Average Shortest Path Distance

**Calculation**:
```python
avg_sp_dist = shortest_paths.mean(axis=1)
max_avg_sp_dist = avg_sp_dist.max()
features[:, 6] = avg_sp_dist / (max_avg_sp_dist + 1e-8)
```

**Formula**:
$$f_{\text{avg\_sp}}(i) = \frac{1}{N} \sum_{j=1}^{N} d(i, j)$$

Where $d(i,j)$ = shortest path distance between nodes $i$ and $j$

**Dimension**: 1

**Range**: [0, 1] (after normalization)

**What it does**: Measures global "centrality" based on average distance to all nodes. Central nodes have low values; peripheral nodes have high values.

**Pros**:
- ✓ Captures multi-scale structure across entire graph
- ✓ Geometric interpretation - closeness to graph center
- ✓ Identifies central nodes important for MIS reasoning
- ✓ Robust to local structure variations

**Cons**:
- ✗ Very expensive to compute (O(|V|²|E|) for dense graphs, or Floyd-Warshall O(|V|³))
- ✗ Sensitive to disconnected components (infinite distances)
- ✗ Loses information about distance distribution (only uses mean)
- ✗ Relatively stable across graphs (less discriminative)

---

### 8. Maximum Shortest Path Distance

**Calculation**:
```python
max_sp_dist = shortest_paths.max(axis=1)
max_max_sp_dist = max_sp_dist.max()
features[:, 7] = max_sp_dist / (max_max_sp_dist + 1e-8)
```

**Formula**:
$$f_{\text{max\_sp}}(i) = \max_{j} d(i, j)$$

**Dimension**: 1

**Range**: [0, 1] (after normalization)

**What it does**: Measures how far a node is from its farthest neighbor (eccentricity-related). High values = peripheral nodes; low values = central nodes in a compact region.

**Pros**:
- ✓ Indicates eccentricity (related to graph diameter)
- ✓ Identifies truly peripheral vs. central nodes
- ✓ Helps distinguish nodes in different parts of the graph
- ✓ Compact representation (single value per node)

**Cons**:
- ✗ Expensive to compute (same as avg distance)
- ✗ Highly correlated with average distance
- ✗ Sensitive to outliers (one distant node affects all)
- ✗ Can be misleading in highly clustered graphs
- ✗ Doesn't capture distribution shape

---

### 9. Minimum Shortest Path Distance

**Calculation**:
```python
min_sp_dist = torch.zeros(num_nodes)
for i in range(num_nodes):
    dists = shortest_paths[i, :]
    nonzero_dists = dists[dists > 0]  # Exclude distance to self
    if len(nonzero_dists) > 0:
        min_sp_dist[i] = nonzero_dists.min()
max_min_sp_dist = min_sp_dist.max()
features[:, 8] = min_sp_dist / (max_min_sp_dist + 1e-8)
```

**Formula**:
$$f_{\text{min\_sp}}(i) = \min_{j \neq i} d(i, j)$$

**Dimension**: 1

**Range**: [0, 1] (after normalization)

**What it does**: Identifies how far the nearest neighbor is (in terms of path distance, not direct adjacency). Indicates local density - isolated clusters have high values.

**Pros**:
- ✓ Captures isolation/clustering information
- ✓ Different from degree (can be 1 even for high-degree nodes in sparse regions)
- ✓ Helps identify weakly connected components
- ✓ Interpretable - "closest other node" is intuitive

**Cons**:
- ✗ Expensive to compute (same as other shortest path features)
- ✗ Will be 1 for all nodes except those with direct neighbors
- ✗ Low discriminative power (most nodes have min_sp_dist = 1)
- ✗ Redundant with degree information
- ✗ Doesn't add much value in practice

---

### 10. Closeness Centrality

**Calculation**:
```python
closeness = torch.zeros(num_nodes)
for i in range(num_nodes):
    dists = shortest_paths[i, :]
    nonzero_dists = dists[dists > 0]
    if len(nonzero_dists) > 0:
        closeness[i] = (len(nonzero_dists) - 1) / nonzero_dists.sum()
    else:
        closeness[i] = 0.0
features[:, 9] = closeness / (closeness.max() + 1e-8)
```

**Formula**:
$$f_{\text{closeness}}(i) = \frac{N-1}{\sum_j d(i,j)}$$

**Dimension**: 1

**Range**: [0, 1] (after normalization)

**What it does**: Harmonic mean of distances to all nodes. Measures how "efficient" a node is at reaching others. Central nodes have high closeness; isolated nodes have low closeness.

**Pros**:
- ✓ Comprehensive centrality measure combining distance and connectivity
- ✓ Identifies high-impact nodes for information flow
- ✓ Robust to outliers (uses sum, not max)
- ✓ Well-studied metric with theoretical foundations

**Cons**:
- ✗ Very expensive to compute (O(|V|²|E|))
- ✗ Highly correlated with average shortest path
- ✗ Undefined for disconnected graphs (needs special handling)
- ✗ Dominated by degree (high-degree nodes tend to have high closeness)
- ✗ Dense feature (sensitive to all distances simultaneously)

---

## Global Positional Encodings

### 11-26. Laplacian Eigenvector Features

**Location**: `dataset/mis_dataset.py` function `compute_laplacian_pe()` and `compute_node_features()`

**Calculation**:
```python
# Compute Laplacian of adjacency matrix
laplacian = np.diag(adj_matrix.sum(axis=1)) - adj_matrix
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
# Sort by eigenvalues
idx = np.argsort(eigenvalues)
eigenvectors = eigenvectors[:, idx]
# Take first 16 eigenvectors as features
laplacian_pe = torch.tensor(eigenvectors[:, :16], dtype=torch.float32)
```

**Formula**:
$$f_{\text{lap}}(i) = [v_1(i), v_2(i), \ldots, v_{16}(i)]$$

Where $v_k$ = $k$-th eigenvector of graph Laplacian

**Dimension**: 16

**Range**: [-1, 1] (approximately, eigenvectors are orthonormal)

**What it does**: Global graph structure encoding. Eigenvectors capture fundamental graph properties like connectivity, clusters, and symmetries.

**Pros**:
- ✓ Captures global graph structure holistically
- ✓ Invariant to node permutations (isomorphic graphs have same eigenvectors)
- ✓ Theoretically well-founded (spectral graph theory)
- ✓ Captures clustering structure naturally (small eigenvalues → cluster boundaries)
- ✓ 16 dimensions is compact yet expressive
- ✓ Provides orthonormal basis for graph space

**Cons**:
- ✗ Expensive to compute (O(|V|³) for dense eigendecomposition)
- ✗ Computed globally - doesn't adapt to node importance
- ✗ Eigenvectors are determined up to sign (not unique)
- ✗ First few eigenvectors may be nearly identical across different graphs
- ✗ Can't distinguish some very different graphs (spectral coarsening)
- ✗ Requires careful numerical handling for small eigenvalues

---

## Feature Summary Table

| Feature | Dimension | Type | Cost | Range | What It Captures |
|---------|-----------|------|------|-------|-----------------|
| Constant Unit | 1 | Learnable Bias | O(1) | [1, 1] | Bias term for MLP |
| Normalized Degree | 1 | Local Structure | O(1) | [0, 1] | Node connectivity |
| Clustering Coefficient | 1 | Local Structure | O(\|E\|) | [0, 1] | Neighborhood density |
| K-Core Degree | 1 | Local Structure | O(\|E\|) | [0, num_nodes] | Structural depth |
| Mean Neighbor Degree | 1 | Local Structure | O(\|E\|) | [0, 1] | Hub neighborhood |
| Neighbor Degree Var | 1 | Local Structure | O(\|E\|) | [0, ∞) | Neighborhood diversity |
| Avg Shortest Path | 1 | Relative PE | O(\|V\|²) | [0, 1] | Global centrality |
| Max Shortest Path | 1 | Relative PE | O(\|V\|²) | [0, 1] | Eccentricity |
| Min Shortest Path | 1 | Relative PE | O(\|V\|²) | [0, 1] | Local isolation |
| Closeness Centrality | 1 | Relative PE | O(\|V\|²) | [0, 1] | Reachability |
| Laplacian PE (16x) | 16 | Global PE | O(\|V\|³) | [-1, 1] | Global structure |
| **TOTAL** | **26** | **Mixed** | **Varies** | **Varies** | **Comprehensive** |

---

## Input Dimension Breakdown

The model receives features in this order:

```
[Local Features: 2] + [Local Enhancements: 4] + [Relative PE: 4]
+ [Global PE: 16]
= 26 total dimensions
```

Then appended: Graph transformer layers use these 26-dim features in first layer.

---

## Feature Engineering Recommendations

### Best Practices

1. **Always normalize** features to similar ranges (most are [0,1], Laplacian is [-1,1])
2. **Use relative PE for MIS** - shortest path features are theoretically motivated for independent set problems
3. **Include Laplacian PE** - provides necessary global context that local features miss
4. **Monitor cost vs. benefit** - Some features (closeness) are expensive but may not add value
5. **Consider graph type** - For sparse graphs, local features dominate; for dense graphs, global features matter more

### Potential Improvements

1. **Remove redundant features** - Min shortest path and neighbor degree variance have low discriminative power
2. **Add edge-based features** - Features encoding edge properties (edge betweenness, etc.)
3. **Add triangle-based features** - Richer indicator of local structure
4. **Use adaptive encodings** - Adjust number of Laplacian eigenvectors based on graph size
5. **Add graph-level features** - Normalize local features by global statistics
6. **Use higher-order PE** - Beyond shortest paths (e.g., 2-hop neighborhoods)

---

## Feature Computation Pipeline

**Location**: `dataset/mis_dataset.py` function `compute_node_features()`

1. **Input**: Adjacency matrix, number of nodes
2. **Local features** (fast): degree, clustering, k-core, neighbor stats
3. **Relative PE** (medium): shortest paths via scipy
4. **Global PE** (expensive): Laplacian eigendecomposition
5. **Output**: [N, 10] tensor concatenated with [N, 16] Laplacian PE

Total time complexity: O(|V|³) dominated by Laplacian computation.

---

