# RuVector Usage Examples

## Quick Start Examples

### Basic Vector Operations

```typescript
import { VectorDB } from '@ruvector/core';

// Create a vector database with 384 dimensions
const db = new VectorDB(384, { metric: 'cosine' });

// Insert vectors
db.insert('doc-1', [0.1, 0.2, 0.3, /* ... 384 values */]);
db.insert('doc-2', [0.2, 0.3, 0.4, /* ... 384 values */]);
db.insert('doc-3', [0.3, 0.4, 0.5, /* ... 384 values */]);

// Search for similar vectors
const query = [0.15, 0.25, 0.35, /* ... 384 values */];
const results = db.search(query, 3);

console.log(results);
// [
//   { id: 'doc-1', distance: 0.02 },
//   { id: 'doc-2', distance: 0.08 },
//   { id: 'doc-3', distance: 0.15 }
// ]
```

### Batch Operations

```typescript
import { VectorDB } from '@ruvector/core';

const db = new VectorDB(384);

// Batch insert for better performance
const vectors = [
  { id: 'doc-1', embedding: generateEmbedding('Hello world') },
  { id: 'doc-2', embedding: generateEmbedding('Machine learning') },
  { id: 'doc-3', embedding: generateEmbedding('Vector databases') },
  // ... thousands more
];

db.insertBatch(vectors);
console.log(`Inserted ${db.count()} vectors`);
```

### Persistence

```typescript
import { VectorDB } from '@ruvector/core';

// Create and populate
const db = new VectorDB(384);
db.insert('doc-1', embedding1);
db.insert('doc-2', embedding2);

// Save to disk
db.save('./mydb.ruvec');

// Load later
const db2 = new VectorDB(384);
db2.load('./mydb.ruvec');

// Continue using
const results = db2.search(query, 10);
```

---

## Integration with Agentic-Flow

### Agent Memory Storage

```typescript
// src/memory/AgentMemoryStore.ts
import { VectorDB } from '@ruvector/core';
import { EmbeddingService } from 'agentdb';

export class AgentMemoryStore {
  private vectorDB: VectorDB;
  private embedder: EmbeddingService;
  private memories: Map<string, AgentMemory>;

  constructor(embedder: EmbeddingService) {
    this.vectorDB = new VectorDB(embedder.dimension, {
      metric: 'cosine',
      maxElements: 100000
    });
    this.embedder = embedder;
    this.memories = new Map();
  }

  async storeMemory(memory: AgentMemory): Promise<void> {
    // Generate embedding from memory content
    const embedding = await this.embedder.embed(
      `${memory.type}: ${memory.content}`
    );

    // Store in vector database
    this.vectorDB.insert(memory.id, Array.from(embedding));
    this.memories.set(memory.id, memory);
  }

  async recallSimilar(query: string, k: number = 5): Promise<AgentMemory[]> {
    const queryEmbedding = await this.embedder.embed(query);
    const results = this.vectorDB.search(Array.from(queryEmbedding), k);

    return results.map(r => this.memories.get(r.id)!);
  }

  async recallByContext(context: {
    agentId?: string;
    taskType?: string;
    timeRange?: [number, number];
  }): Promise<AgentMemory[]> {
    const filtered = Array.from(this.memories.values()).filter(m => {
      if (context.agentId && m.agentId !== context.agentId) return false;
      if (context.taskType && m.taskType !== context.taskType) return false;
      if (context.timeRange) {
        const [start, end] = context.timeRange;
        if (m.timestamp < start || m.timestamp > end) return false;
      }
      return true;
    });

    return filtered;
  }
}

interface AgentMemory {
  id: string;
  agentId: string;
  type: 'experience' | 'learning' | 'observation';
  taskType: string;
  content: string;
  outcome?: 'success' | 'failure';
  timestamp: number;
}
```

### ReasoningBank with GNN Enhancement

```typescript
// src/reasoning/GNNReasoningBank.ts
import { VectorDB } from '@ruvector/core';
import { GNNLayer } from '@ruvector/gnn';
import { EmbeddingService, ReasoningPattern } from 'agentdb';

export class GNNReasoningBank {
  private vectorDB: VectorDB;
  private gnn: GNNLayer;
  private embedder: EmbeddingService;
  private patterns: Map<string, ReasoningPattern>;
  private trainingBuffer: TrainingSample[];

  constructor(dimension: number, embedder: EmbeddingService) {
    this.vectorDB = new VectorDB(dimension, { metric: 'cosine' });
    this.gnn = new GNNLayer(dimension, dimension, 4);
    this.embedder = embedder;
    this.patterns = new Map();
    this.trainingBuffer = [];
  }

  async storePattern(pattern: ReasoningPattern): Promise<void> {
    const embedding = await this.embedder.embed(
      `${pattern.taskType}: ${pattern.approach}`
    );

    this.vectorDB.insert(pattern.id!, Array.from(embedding));
    this.patterns.set(pattern.id!, { ...pattern, embedding });
  }

  async searchWithGNN(query: string, k: number = 10): Promise<ReasoningPattern[]> {
    const queryEmbedding = await this.embedder.embed(query);

    // Get initial candidates
    const candidates = this.vectorDB.search(Array.from(queryEmbedding), k * 3);

    if (candidates.length === 0) return [];

    // Extract neighbor embeddings and weights
    const neighborEmbeddings = candidates.map(c => {
      const pattern = this.patterns.get(c.id)!;
      return Array.from(pattern.embedding!);
    });
    const weights = candidates.map(c => {
      const pattern = this.patterns.get(c.id)!;
      return pattern.successRate * (1 - c.distance);
    });

    // Apply GNN enhancement
    const enhanced = this.gnn.forward(
      Array.from(queryEmbedding),
      neighborEmbeddings,
      weights
    );

    // Re-search with enhanced query
    const reranked = this.vectorDB.search(enhanced, k);

    return reranked.map(r => this.patterns.get(r.id)!);
  }

  async recordOutcome(patternId: string, success: boolean): Promise<void> {
    const pattern = this.patterns.get(patternId);
    if (!pattern) return;

    // Update pattern success rate
    const alpha = 0.1;
    pattern.successRate = (1 - alpha) * pattern.successRate + alpha * (success ? 1 : 0);

    // Add to training buffer
    this.trainingBuffer.push({
      embedding: Array.from(pattern.embedding!),
      label: success ? 1 : 0
    });

    // Auto-train when buffer is full
    if (this.trainingBuffer.length >= 100) {
      await this.train();
    }
  }

  async train(epochs: number = 50): Promise<void> {
    if (this.trainingBuffer.length < 10) return;

    await this.gnn.train(this.trainingBuffer, {
      epochs,
      learningRate: 0.001,
      batchSize: 32
    });

    // Clear buffer after training
    this.trainingBuffer = [];
  }
}

interface TrainingSample {
  embedding: number[];
  label: number;
}
```

### Agent Collaboration Graph

```typescript
// src/graph/AgentCollaborationGraph.ts

export class AgentCollaborationGraph {
  private graphDB: any; // @ruvector/graph-node

  async createAgent(agent: {
    id: string;
    name: string;
    type: string;
    capabilities: string[];
  }): Promise<void> {
    await this.execute(`
      CREATE (a:Agent {
        id: $id,
        name: $name,
        type: $type,
        capabilities: $capabilities,
        createdAt: timestamp()
      })
    `, agent);
  }

  async recordCollaboration(
    agent1Id: string,
    agent2Id: string,
    task: string,
    success: boolean
  ): Promise<void> {
    await this.execute(`
      MATCH (a:Agent {id: $agent1Id}), (b:Agent {id: $agent2Id})
      MERGE (a)-[r:COLLABORATED_WITH]->(b)
      ON CREATE SET r.count = 1, r.successCount = $success
      ON MATCH SET r.count = r.count + 1,
                   r.successCount = r.successCount + $success
      SET r.lastTask = $task, r.lastTimestamp = timestamp()
    `, { agent1Id, agent2Id, task, success: success ? 1 : 0 });
  }

  async findBestCollaborators(
    agentId: string,
    forTask: string,
    limit: number = 5
  ): Promise<Array<{ agent: any; score: number }>> {
    const result = await this.execute(`
      MATCH (a:Agent {id: $agentId})-[r:COLLABORATED_WITH]-(b:Agent)
      WHERE b.type = $forTask OR any(cap IN b.capabilities WHERE cap CONTAINS $forTask)
      WITH b, r, r.successCount * 1.0 / r.count as successRate
      RETURN b as agent,
             successRate * r.count * 0.01 as score
      ORDER BY score DESC
      LIMIT $limit
    `, { agentId, forTask, limit });

    return result;
  }

  async getAgentNetwork(agentId: string): Promise<{
    agents: any[];
    relationships: any[];
  }> {
    const result = await this.execute(`
      MATCH (center:Agent {id: $agentId})
      OPTIONAL MATCH (center)-[r:COLLABORATED_WITH*1..2]-(connected:Agent)
      WITH center, collect(DISTINCT connected) as agents,
           collect(DISTINCT r) as rels
      RETURN agents, rels as relationships
    `, { agentId });

    return result;
  }

  async recommendTeam(
    taskDescription: string,
    teamSize: number = 4
  ): Promise<any[]> {
    // Find agents with high success rates on similar tasks
    const result = await this.execute(`
      MATCH (a:Agent)-[r:COLLABORATED_WITH]-(b:Agent)
      WHERE r.lastTask CONTAINS $taskDescription
      WITH a, b, r.successCount * 1.0 / r.count as pairSuccess
      ORDER BY pairSuccess DESC
      LIMIT 100
      WITH collect(DISTINCT a) + collect(DISTINCT b) as candidates
      UNWIND candidates as agent
      WITH agent, count(*) as frequency
      ORDER BY frequency DESC
      LIMIT $teamSize
      RETURN agent
    `, { taskDescription, teamSize });

    return result;
  }

  private async execute(cypher: string, params: Record<string, any>): Promise<any> {
    // Implementation using @ruvector/graph-node
    throw new Error('Requires @ruvector/graph-node');
  }
}
```

---

## CLI Examples

### Create and Populate Database

```bash
# Create database
npx ruvector create ./agent-memory --dimension 384 --metric cosine

# Insert vectors from JSON file
cat > vectors.json << 'EOF'
[
  {"id": "task-1", "embedding": [0.1, 0.2, ...], "metadata": {"type": "code_review"}},
  {"id": "task-2", "embedding": [0.2, 0.3, ...], "metadata": {"type": "debugging"}},
  {"id": "task-3", "embedding": [0.3, 0.4, ...], "metadata": {"type": "feature_dev"}}
]
EOF

npx ruvector insert ./agent-memory vectors.json

# Check stats
npx ruvector stats ./agent-memory
```

### Search Operations

```bash
# Search by vector
npx ruvector search ./agent-memory --query "[0.15, 0.25, ...]" --k 5

# Search by text (uses built-in embeddings)
npx ruvector search ./agent-memory --text "fix authentication bug" --k 10

# Search with threshold
npx ruvector search ./agent-memory --text "optimize database" --k 5 --threshold 0.7
```

### GNN Operations

```bash
# Train GNN on existing data
npx ruvector gnn train ./agent-memory --epochs 100

# Enhanced search using GNN
npx ruvector gnn search ./agent-memory --text "implement feature" --k 10

# Export trained model
npx ruvector gnn export ./agent-memory --output gnn-model.bin
```

### Server Mode

```bash
# Start HTTP server
npx ruvector server --database ./agent-memory --port 8080

# Start with gRPC
npx ruvector server --database ./agent-memory --port 8080 --grpc --grpc-port 50051

# Start cluster node
npx ruvector server --database ./agent-memory --port 8080 --cluster --cluster-port 9000
```

### Graph Operations

```bash
# Create agent nodes
npx ruvector graph query "CREATE (a:Agent {name: 'coder', type: 'specialist'})"
npx ruvector graph query "CREATE (a:Agent {name: 'reviewer', type: 'specialist'})"

# Create relationships
npx ruvector graph query "
  MATCH (a:Agent {name: 'coder'}), (b:Agent {name: 'reviewer'})
  CREATE (a)-[:COLLABORATES_WITH {since: timestamp()}]->(b)
"

# Query relationships
npx ruvector graph query "
  MATCH (a:Agent)-[:COLLABORATES_WITH*1..3]-(related)
  WHERE a.name = 'coder'
  RETURN DISTINCT related.name
"
```

---

## Production Configuration

### Docker Compose

```yaml
version: '3.8'
services:
  ruvector:
    image: ruvnet/ruvector:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - ./data:/data
    environment:
      - RUVECTOR_LOG_LEVEL=info
      - RUVECTOR_THREADS=4
      - RUVECTOR_CACHE_SIZE=512
      - RUVECTOR_COMPRESSION=true
    command: server --database /data/vectors --port 8080 --grpc --grpc-port 50051
```

### Cluster Configuration

```yaml
# cluster.yaml
nodes:
  - host: node1.example.com
    port: 9000
  - host: node2.example.com
    port: 9000
  - host: node3.example.com
    port: 9000

replication:
  factor: 3
  consistency: quorum

sharding:
  count: 8
  algorithm: consistent-hash

failover:
  timeout: 5000
  retries: 3
```

```bash
# Initialize cluster
npx ruvector cluster init --config cluster.yaml

# Check status
npx ruvector cluster status
```

---

## Benchmarking

```bash
# Quick benchmark
npx ruvector benchmark --vectors 10000 --dimension 384

# Comprehensive benchmark
npx ruvector benchmark --comprehensive \
  --vectors 100000 \
  --dimension 384 \
  --iterations 1000 \
  --output benchmark-report.json

# Compare with existing implementation
npx ruvector benchmark --compare-hnswlib \
  --vectors 50000 \
  --dimension 384
```

Sample Output:
```
╔══════════════════════════════════════════════════════════╗
║                  RuVector Benchmark Report               ║
╠══════════════════════════════════════════════════════════╣
║ Vectors: 100,000 | Dimension: 384 | Metric: cosine       ║
╠══════════════════════════════════════════════════════════╣
║ Operation          │ Latency (p50) │ Latency (p99) │ QPS  ║
╠═══════════════════════════════════════════════════════════╣
║ Insert             │ 0.021ms       │ 0.089ms       │ 47K  ║
║ Search (k=10)      │ 0.061ms       │ 0.142ms       │ 16K  ║
║ Search (k=100)     │ 0.164ms       │ 0.312ms       │ 6.1K ║
║ Batch Insert (100) │ 1.2ms         │ 2.4ms         │ 83K  ║
╠══════════════════════════════════════════════════════════╣
║ Memory Usage: 156 MB (1.56 KB/vector)                    ║
║ With Compression: 48 MB (0.48 KB/vector) - 3.25x savings ║
╚══════════════════════════════════════════════════════════╝
```
