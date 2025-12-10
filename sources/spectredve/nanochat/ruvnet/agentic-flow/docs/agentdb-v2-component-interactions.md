# AgentDB v2 Component Interaction Diagrams

**Version:** 2.0.0-alpha
**Last Updated:** 2025-11-28

## Table of Contents

- [System Overview](#system-overview)
- [Backend Selection Flow](#backend-selection-flow)
- [Vector Insert Flow](#vector-insert-flow)
- [Vector Search Flow](#vector-search-flow)
- [GNN Enhancement Flow](#gnn-enhancement-flow)
- [Graph Query Flow](#graph-query-flow)
- [Data Flow Diagrams](#data-flow-diagrams)

## System Overview

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  (User Code, MCP Tools, CLI Commands)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     High-Level Controllers                       │
│  ReasoningBank  │  SkillLibrary  │  CausalMemoryGraph          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend Abstraction Layer                     │
│  VectorBackend  │  LearningBackend  │  GraphBackend            │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   RuVector Backends      │    │   HNSWLib Backend        │
│  (Native/WASM)           │    │   (Node.js)              │
└──────────────────────────┘    └──────────────────────────┘
                │                           │
                ▼                           ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   SQLite Database        │    │   File System            │
│   (Metadata Storage)     │    │   (Index Persistence)    │
└──────────────────────────┘    └──────────────────────────┘
```

## Backend Selection Flow

### Sequence Diagram

```
┌────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User  │    │ Factory │    │ Detector │    │ RuVector │    │ HNSWLib  │
└───┬────┘    └────┬────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
    │              │              │               │               │
    │ createBackend('auto', config)              │               │
    ├─────────────>│              │               │               │
    │              │              │               │               │
    │              │ detectBackends()             │               │
    │              ├─────────────>│               │               │
    │              │              │               │               │
    │              │              │ import('@ruvector/core')      │
    │              │              ├──────────────>│               │
    │              │              │               │               │
    │              │              │ isNative()    │               │
    │              │              ├──────────────>│               │
    │              │              │ <─────────────┤               │
    │              │              │   true        │               │
    │              │              │               │               │
    │              │              │ import('@ruvector/gnn')       │
    │              │              ├──────────────>│               │
    │              │              │ <─────────────┤               │
    │              │              │   success     │               │
    │              │              │               │               │
    │              │ <────────────┤               │               │
    │              │ {backend: 'ruvector',       │               │
    │              │  features: {gnn: true}}     │               │
    │              │              │               │               │
    │              │ new RuVectorBackend(config) │               │
    │              ├──────────────────────────────>               │
    │              │              │               │               │
    │              │ <────────────────────────────┤               │
    │              │   backend instance           │               │
    │              │              │               │               │
    │ <────────────┤              │               │               │
    │  backend     │              │               │               │
    │              │              │               │               │
```

### Decision Tree

```
Start
  │
  ├─ Backend Type?
  │   │
  │   ├─ 'auto' ──> Detect Available Backends
  │   │              │
  │   │              ├─ RuVector Core Available? ──> Yes ──> Use RuVector
  │   │              │                              │
  │   │              │                              └─> Check Native/WASM
  │   │              │
  │   │              └─ No ──> HNSWLib Available? ──> Yes ──> Use HNSWLib
  │   │                         │
  │   │                         └─ No ──> Error: No backend available
  │   │
  │   ├─ 'ruvector' ──> Check RuVector
  │   │                 │
  │   │                 ├─ Available? ──> Yes ──> Use RuVector
  │   │                 │
  │   │                 └─ No ──> Error: Install @ruvector/core
  │   │
  │   └─ 'hnswlib' ──> Check HNSWLib
  │                    │
  │                    ├─ Available? ──> Yes ──> Use HNSWLib
  │                    │
  │                    └─ No ──> Error: Install hnswlib-node
  │
  └─ Optional Features?
      │
      ├─ GNN Learning? ──> Check @ruvector/gnn
      │                    │
      │                    ├─ Available? ──> Enable LearningBackend
      │                    │
      │                    └─ Not Available ──> Disable (optional)
      │
      └─ Graph Database? ──> Check @ruvector/graph-node
                             │
                             ├─ Available? ──> Enable GraphBackend
                             │
                             └─ Not Available ──> Disable (optional)
```

## Vector Insert Flow

### Sequence Diagram

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌────────┐
│ ReasoningBank│    │  Vector  │    │  RuVector  │    │  Index │
│              │    │  Backend │    │  Backend   │    │  Store │
└──────┬───────┘    └────┬─────┘    └─────┬──────┘    └───┬────┘
       │                 │                │               │
       │ storePattern(pattern)            │               │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ Generate Embedding             │
       │                 │ (EmbeddingService)             │
       │                 ├───────────┐    │               │
       │                 │           │    │               │
       │                 │<──────────┘    │               │
       │                 │ embedding      │               │
       │                 │                │               │
       │                 │ insert(id, embedding, metadata)│
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │                │ Map ID to Label
       │                 │                ├──────────┐    │
       │                 │                │          │    │
       │                 │                │<─────────┘    │
       │                 │                │ label: 42     │
       │                 │                │               │
       │                 │                │ addPoint(vector, label)
       │                 │                ├──────────────>│
       │                 │                │               │
       │                 │                │ <─────────────┤
       │                 │                │   success     │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │   void         │               │
       │                 │                │               │
       │ <───────────────┤                │               │
       │  patternId      │                │               │
       │                 │                │               │
```

### Batch Insert Optimization

```
┌──────────────┐    ┌──────────┐    ┌────────────┐
│   User Code  │    │  Vector  │    │  RuVector  │
│              │    │  Backend │    │  Backend   │
└──────┬───────┘    └────┬─────┘    └─────┬──────┘
       │                 │                │
       │ insertBatch([1000 items])       │
       ├────────────────>│                │
       │                 │                │
       │                 │ Batch Processing (parallel)
       │                 ├───────────┐    │
       │                 │           │    │
       │                 │  - Map IDs to labels
       │                 │  - Prepare vectors
       │                 │  - Optimize memory
       │                 │           │    │
       │                 │<──────────┘    │
       │                 │                │
       │                 │ addPoints(vectors, labels)
       │                 ├───────────────>│
       │                 │                │
       │                 │                │ SIMD Batch Insert
       │                 │                ├──────────┐
       │                 │                │          │
       │                 │                │ 10-20x faster
       │                 │                │          │
       │                 │                │<─────────┘
       │                 │                │
       │                 │ <──────────────┤
       │                 │   success      │
       │                 │                │
       │ <───────────────┤                │
       │  void           │                │
       │                 │                │

Performance: 50ms for 1000 vectors vs 500ms sequential
```

## Vector Search Flow

### Sequence Diagram

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌────────┐
│ ReasoningBank│    │  Vector  │    │  RuVector  │    │  Index │
│              │    │  Backend │    │  Backend   │    │  Store │
└──────┬───────┘    └────┬─────┘    └─────┬──────┘    └───┬────┘
       │                 │                │               │
       │ searchPatterns(query, k=10)     │               │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ search(embedding, 10, options) │
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │                │ searchKnn(vector, k, ef)
       │                 │                ├──────────────>│
       │                 │                │               │
       │                 │                │               │ HNSW Graph Traversal
       │                 │                │               ├──────────┐
       │                 │                │               │          │
       │                 │                │               │ - Start at entry
       │                 │                │               │ - Navigate layers
       │                 │                │               │ - Collect candidates
       │                 │                │               │ - SIMD distance calc
       │                 │                │               │          │
       │                 │                │               │<─────────┘
       │                 │                │               │
       │                 │                │ <─────────────┤
       │                 │                │ [labels, distances]
       │                 │                │               │
       │                 │                │ Map Labels to IDs
       │                 │                ├──────────┐    │
       │                 │                │          │    │
       │                 │                │ - label 42 -> "id1"
       │                 │                │ - label 15 -> "id2"
       │                 │                │          │    │
       │                 │                │<─────────┘    │
       │                 │                │               │
       │                 │                │ Normalize Similarities
       │                 │                ├──────────┐    │
       │                 │                │          │    │
       │                 │                │ cosine: 1-dist
       │                 │                │ l2: exp(-dist)
       │                 │                │          │    │
       │                 │                │<─────────┘    │
       │                 │                │               │
       │                 │                │ Apply Threshold
       │                 │                ├──────────┐    │
       │                 │                │          │    │
       │                 │                │ filter < 0.7
       │                 │                │          │    │
       │                 │                │<─────────┘    │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │ SearchResult[] │               │
       │                 │ {id, distance, similarity, metadata}
       │                 │                │               │
       │ <───────────────┤                │               │
       │ ReasoningPattern[]              │               │
       │ (enriched with DB data)         │               │
       │                 │                │               │

Performance: 0.5-2ms per search (RuVector native)
```

## GNN Enhancement Flow

### Sequence Diagram

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐
│ ReasoningBank│    │  Vector  │    │  Learning  │    │   GNN   │
│              │    │  Backend │    │  Backend   │    │  Model  │
└──────┬───────┘    └────┬─────┘    └─────┬──────┘    └────┬────┘
       │                 │                │               │
       │ searchPatterns(query, k=10)     │               │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ search(query, k+20)  // Get more candidates
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │ initial results│               │
       │                 │                │               │
       │                 │ enhance(query, neighbors, weights)
       │                 ├───────────────────────────────>│
       │                 │                │               │
       │                 │                │               │ GNN Forward Pass
       │                 │                │               ├──────────┐
       │                 │                │               │          │
       │                 │                │               │ - Multi-head attention
       │                 │                │               │ - Aggregate neighbors
       │                 │                │               │ - Apply learned weights
       │                 │                │               │ - Generate enhanced query
       │                 │                │               │          │
       │                 │                │               │<─────────┘
       │                 │                │               │
       │                 │ <──────────────────────────────┤
       │                 │ enhanced_query │               │
       │                 │                │               │
       │                 │ Re-search with enhanced query  │
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │ better results │               │
       │                 │ (higher quality)│              │
       │                 │                │               │
       │ <───────────────┤                │               │
       │ ReasoningPattern[]              │               │
       │                 │                │               │
       │                 │                │               │
       │ Feedback: success=true, reward=0.9              │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ addSample({embedding, label, reward})
       │                 ├───────────────────────────────>│
       │                 │                │               │
       │                 │                │               │ Accumulate Sample
       │                 │                │               ├──────────┐
       │                 │                │               │          │
       │                 │                │               │<─────────┘
       │                 │                │               │
       │                 │ <──────────────────────────────┤
       │                 │   void         │               │
       │                 │                │               │

Periodic Training (every 1000 samples or nightly):
       │                 │                │               │
       │                 │ train({epochs: 50})           │
       ├────────────────────────────────────────────────>│
       │                 │                │               │
       │                 │                │               │ Backpropagation
       │                 │                │               ├──────────┐
       │                 │                │               │          │
       │                 │                │               │ - Compute gradients
       │                 │                │               │ - Update weights
       │                 │                │               │ - Reduce loss
       │                 │                │               │          │
       │                 │                │               │<─────────┘
       │                 │                │               │
       │ <──────────────────────────────────────────────┤
       │ TrainingResult {epochs: 50, finalLoss: 0.15,   │
       │                 improvement: 45%, duration: 2.3s}
       │                 │                │               │
```

## Graph Query Flow

### Sequence Diagram

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐
│  Causal      │    │  Graph   │    │  RuVector  │    │  Graph  │
│  Memory      │    │  Backend │    │  Graph     │    │  Store  │
└──────┬───────┘    └────┬─────┘    └─────┬──────┘    └────┬────┘
       │                 │                │               │
       │ Create causal relationship       │               │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ createNode(['Memory'], props) │
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │                │ Insert Node   │
       │                 │                ├──────────────>│
       │                 │                │               │
       │                 │                │ <─────────────┤
       │                 │                │  nodeId1      │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │  nodeId1       │               │
       │                 │                │               │
       │                 │ createRelationship(node1, node2, 'CAUSES')
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │                │ Insert Edge   │
       │                 │                ├──────────────>│
       │                 │                │               │
       │                 │                │ <─────────────┤
       │                 │                │  relId        │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │  relId         │               │
       │                 │                │               │
       │ <───────────────┤                │               │
       │  relId          │                │               │
       │                 │                │               │
       │                 │                │               │
       │ Query: Find related memories     │               │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ traverse(startId, pattern, {maxDepth: 2})
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │                │ Graph Traversal
       │                 │                ├──────────────>│
       │                 │                │               │
       │                 │                │               │ DFS/BFS Search
       │                 │                │               ├──────────┐
       │                 │                │               │          │
       │                 │                │               │ - Follow edges
       │                 │                │               │ - Apply pattern
       │                 │                │               │ - Collect nodes
       │                 │                │               │          │
       │                 │                │               │<─────────┘
       │                 │                │               │
       │                 │                │ <─────────────┤
       │                 │                │ [node1, node2]│
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │ GraphNode[]    │               │
       │                 │                │               │
       │ <───────────────┤                │               │
       │ Memory[]        │                │               │
       │                 │                │               │
       │                 │                │               │
       │ Hybrid Query: Vector + Graph    │               │
       ├────────────────>│                │               │
       │                 │                │               │
       │                 │ vectorSearch(query, k=10, contextNode)
       │                 ├───────────────>│               │
       │                 │                │               │
       │                 │                │ 1. Get graph neighborhood
       │                 │                ├──────────────>│
       │                 │                │               │
       │                 │                │ <─────────────┤
       │                 │                │ [neighbors]   │
       │                 │                │               │
       │                 │                │ 2. Vector search on neighbors
       │                 │                ├──────────┐    │
       │                 │                │          │    │
       │                 │                │ HNSW + filter
       │                 │                │          │    │
       │                 │                │<─────────┘    │
       │                 │                │               │
       │                 │ <──────────────┤               │
       │                 │ GraphNode[] (ranked by similarity)
       │                 │                │               │
       │ <───────────────┤                │               │
       │ Memory[] (semantically + structurally relevant)  │
       │                 │                │               │
```

## Data Flow Diagrams

### Pattern Storage Data Flow

```
User Pattern Input
       │
       ▼
[Embedding Generation]
       │ (Float32Array)
       ▼
[VectorBackend.insert()]
       │
       ├──> [ID → Label Mapping] ──> [In-Memory Map]
       │
       ├──> [Vector → Index] ──> [HNSW Graph Structure]
       │
       └──> [Metadata → Storage] ──> [Separate Metadata Store]
       │
       ▼
[Optional: Persist to Disk]
       │
       ├──> [Index File] (.hnsw)
       │
       └──> [Metadata File] (.json)
```

### Pattern Search Data Flow

```
Query Embedding
       │
       ▼
[Optional: GNN Enhancement]
       │ (Enhanced Query)
       ▼
[VectorBackend.search()]
       │
       ├──> [HNSW Graph Traversal]
       │    │
       │    ├──> [Entry Point Selection]
       │    ├──> [Layer-by-Layer Navigation]
       │    └──> [Candidate Collection]
       │
       ▼
[Distance Calculation] (SIMD)
       │
       ▼
[Label → ID Mapping]
       │
       ▼
[Distance → Similarity Normalization]
       │
       ▼
[Threshold Filtering]
       │
       ▼
[Metadata Enrichment]
       │
       ▼
Search Results
```

### Learning Feedback Loop

```
Query + Results
       │
       ▼
[User Interaction]
       │
       ├──> Success? ──> reward = 0.9
       │
       └──> Failure? ──> reward = 0.1
       │
       ▼
[LearningBackend.addSample()]
       │
       ├──> [Sample Buffer] (accumulate)
       │
       └──> [When Buffer Full or Nightly]
            │
            ▼
       [LearningBackend.train()]
            │
            ├──> [Backpropagation]
            ├──> [Weight Updates]
            └──> [Loss Reduction]
            │
            ▼
       [Improved Query Enhancement]
            │
            └──> Better Results Next Time
```

---

**Next Steps:**
1. Implement missing components (RuVectorGraph)
2. Add telemetry and monitoring
3. Create performance dashboards
4. Build integration tests for all flows
