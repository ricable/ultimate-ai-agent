# RuVector

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![npm downloads](https://img.shields.io/npm/dm/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org/)

**A distributed vector database that learns.** Store embeddings, query with Cypher, scale horizontally, and let the index improve itself through Graph Neural Networks.

```bash
npx ruvector
```

> **All-in-One Package**: The `ruvector` package includes everything — vector search, graph queries, GNN layers, AI agent routing, and WASM support. No additional packages needed.

## Why RuVector?

Traditional vector databases just store and search. When you ask "find similar items," they return results but never get smarter. They can't handle complex relationships. They don't optimize your AI costs.

**RuVector is built for the agentic AI era:**

| Challenge | RuVector Solution |
|-----------|-------------------|
| RAG retrieval quality plateaus | **Self-learning GNN** improves results over time |
| Knowledge graphs need separate DB | **Cypher queries** built-in (Neo4j syntax) |
| LLM costs spiral out of control | **AI Router** sends simple queries to cheaper models |
| Memory usage explodes at scale | **Adaptive compression** (2-32x reduction) |
| Can't run AI in the browser | **Full WASM support** for client-side inference |

## Quick Start

### Installation

```bash
# Install the package
npm install ruvector

# Or try instantly without installing
npx ruvector

# With yarn
yarn add ruvector

# With pnpm
pnpm add ruvector
```

### Basic Vector Search

```javascript
const { VectorDB } = require('ruvector');

// Create a vector database (384 = OpenAI ada-002 dimensions)
const db = new VectorDB(384);

// Insert vectors with metadata
await db.insert('doc1', embedding1, {
  title: 'Introduction to AI',
  category: 'tech',
  date: '2024-01-15'
});

// Semantic search
const results = await db.search(queryEmbedding, 10);

// Filter by metadata
const filtered = await db.search(queryEmbedding, 10, {
  category: 'tech',
  date: { $gte: '2024-01-01' }
});
```

### RAG (Retrieval-Augmented Generation)

```javascript
const { VectorDB } = require('ruvector');
const OpenAI = require('openai');

const db = new VectorDB(1536); // text-embedding-3-small dimensions
const openai = new OpenAI();

// Index your documents
async function indexDocument(doc) {
  const embedding = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: doc.content
  });
  await db.insert(doc.id, embedding.data[0].embedding, {
    title: doc.title,
    content: doc.content
  });
}

// RAG query
async function ragQuery(question) {
  // 1. Embed the question
  const questionEmb = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: question
  });

  // 2. Retrieve relevant context
  const context = await db.search(questionEmb.data[0].embedding, 5);

  // 3. Generate answer with context
  const response = await openai.chat.completions.create({
    model: 'gpt-4-turbo',
    messages: [{
      role: 'user',
      content: `Context:\n${context.map(c => c.metadata.content).join('\n\n')}

Question: ${question}
Answer based only on the context above:`
    }]
  });

  return response.choices[0].message.content;
}
```

### Knowledge Graphs (Cypher)

```javascript
const { GraphDB } = require('ruvector');

const graph = new GraphDB();

// Create entities and relationships
graph.execute(`
  CREATE (alice:Person {name: 'Alice', role: 'Engineer'})
  CREATE (bob:Person {name: 'Bob', role: 'Manager'})
  CREATE (techcorp:Company {name: 'TechCorp', industry: 'AI'})
  CREATE (alice)-[:WORKS_AT {since: 2022}]->(techcorp)
  CREATE (bob)-[:WORKS_AT {since: 2020}]->(techcorp)
  CREATE (alice)-[:REPORTS_TO]->(bob)
`);

// Query relationships
const team = graph.execute(`
  MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: 'TechCorp'})
  RETURN p.name, p.role
`);

// Find paths
const chain = graph.execute(`
  MATCH path = (a:Person {name: 'Alice'})-[:REPORTS_TO*1..3]->(manager)
  RETURN path
`);

// Combine with vector search
const similarPeople = graph.execute(`
  MATCH (p:Person)
  WHERE vector.similarity(p.embedding, $queryEmbedding) > 0.8
  RETURN p ORDER BY vector.similarity(p.embedding, $queryEmbedding) DESC
  LIMIT 10
`);
```

### GNN-Enhanced Search (Self-Learning)

```javascript
const { GNNLayer, VectorDB } = require('ruvector');

// Create GNN layer for query enhancement
const gnn = new GNNLayer(384, 512, 4); // input_dim, output_dim, num_heads

// The GNN learns from your search patterns
async function enhancedSearch(query) {
  // Get initial results
  const neighbors = await db.search(query, 20);

  // Compute attention weights based on user clicks/relevance
  const weights = computeRelevanceWeights(neighbors);

  // GNN enhances the query using graph structure
  const enhancedQuery = gnn.forward(query,
    neighbors.map(n => n.embedding),
    weights
  );

  // Re-rank with enhanced understanding
  return db.search(enhancedQuery, 10);
}

// Train on user feedback
gnn.train({
  queries: historicalQueries,
  clicks: userClickData,
  relevance: expertLabels
}, { epochs: 100 });
```

### AI Agent Routing (Tiny Dancer)

Route queries to the optimal LLM based on complexity — save 60-80% on API costs:

```javascript
const { Router } = require('ruvector');

const router = new Router({
  confidenceThreshold: 0.85,
  maxUncertainty: 0.15,
  enableCircuitBreaker: true
});

// Define your model candidates
const models = [
  { id: 'gpt-4-turbo', embedding: gpt4Emb, cost: 0.03, quality: 0.95 },
  { id: 'gpt-3.5-turbo', embedding: gpt35Emb, cost: 0.002, quality: 0.80 },
  { id: 'claude-3-haiku', embedding: haikuEmb, cost: 0.001, quality: 0.75 },
  { id: 'llama-3-8b', embedding: llamaEmb, cost: 0.0005, quality: 0.70 }
];

async function smartComplete(prompt) {
  const promptEmb = await embed(prompt);

  // Router decides optimal model
  const decision = router.route(promptEmb, models);

  console.log(`Routing to ${decision.candidateId} (confidence: ${decision.confidence})`);
  // Output: "Routing to gpt-3.5-turbo (confidence: 0.92)"

  // Call the selected model
  return callModel(decision.candidateId, prompt);
}
```

### Compression (2-32x Memory Savings)

```javascript
const { compress, decompress, CompressionTier } = require('ruvector');

// Automatic tier selection
const auto = compress(embedding, 0.3); // 30% quality threshold

// Explicit tiers
const f16 = compress(embedding, CompressionTier.F16);     // 2x compression
const pq8 = compress(embedding, CompressionTier.PQ8);     // 8x compression
const pq4 = compress(embedding, CompressionTier.PQ4);     // 16x compression
const binary = compress(embedding, CompressionTier.Binary); // 32x compression

// Adaptive tiering based on access frequency
db.enableAdaptiveCompression({
  hotThreshold: 0.8,    // Keep hot data in f32
  warmThreshold: 0.4,   // Compress to f16
  coldThreshold: 0.1,   // Compress to PQ8
  archiveThreshold: 0.01 // Compress to binary
});
```

## CLI Usage

```bash
# Show system info and backend status
npx ruvector info

# Initialize a new index
npx ruvector init my-index --dimension 384 --type hnsw

# Insert vectors from JSON/JSONL
npx ruvector insert my-index vectors.json
npx ruvector insert my-index vectors.jsonl --format jsonl

# Search with a query
npx ruvector search my-index --query "[0.1, 0.2, ...]" -k 10
npx ruvector search my-index --text "machine learning" -k 10  # Auto-embed

# Show index statistics
npx ruvector stats my-index

# Run performance benchmarks
npx ruvector benchmark --dimension 384 --num-vectors 10000

# Export/import
npx ruvector export my-index backup.bin
npx ruvector import backup.bin restored-index
```

## Integrations

### LangChain

```javascript
const { RuVectorStore } = require('ruvector/langchain');
const { OpenAIEmbeddings } = require('@langchain/openai');

const vectorStore = new RuVectorStore(
  new OpenAIEmbeddings(),
  { dimension: 1536 }
);

await vectorStore.addDocuments(documents);
const results = await vectorStore.similaritySearch("query", 5);
```

### LlamaIndex

```javascript
const { RuVectorIndex } = require('ruvector/llamaindex');

const index = new RuVectorIndex({
  dimension: 384,
  enableGNN: true
});

await index.insert(documents);
const queryEngine = index.asQueryEngine();
const response = await queryEngine.query("What is machine learning?");
```

### OpenAI / Anthropic

```javascript
const { createEmbedder } = require('ruvector');

// OpenAI
const openaiEmbed = createEmbedder('openai', {
  model: 'text-embedding-3-small'
});

// Anthropic (via Voyage)
const anthropicEmbed = createEmbedder('voyage', {
  model: 'voyage-2'
});

// Cohere
const cohereEmbed = createEmbedder('cohere', {
  model: 'embed-english-v3.0'
});
```

## Benchmarks

| Operation | Dimensions | Time | Throughput |
|-----------|------------|------|------------|
| **HNSW Search (k=10)** | 384 | 61µs | 16,400 QPS |
| **HNSW Search (k=100)** | 384 | 164µs | 6,100 QPS |
| **Cosine Similarity** | 1536 | 143ns | 7M ops/sec |
| **Dot Product** | 384 | 33ns | 30M ops/sec |
| **Insert** | 384 | 20µs | 50,000/sec |
| **GNN Forward** | 384→512 | 89µs | 11,200/sec |
| **Compression (PQ8)** | 384 | 12µs | 83,000/sec |

Run your own benchmarks:
```bash
npx ruvector benchmark --dimension 384 --num-vectors 100000
```

## Comparison

| Feature | RuVector | Pinecone | Qdrant | ChromaDB | Milvus | Weaviate |
|---------|----------|----------|--------|----------|--------|----------|
| **Latency (p50)** | **61µs** | ~2ms | ~1ms | ~50ms | ~5ms | ~3ms |
| **Graph Queries** | ✅ Cypher | ❌ | ❌ | ❌ | ❌ | ✅ GraphQL |
| **Self-Learning** | ✅ GNN | ❌ | ❌ | ❌ | ❌ | ❌ |
| **AI Routing** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Browser/WASM** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Compression** | 2-32x | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Hybrid Search** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Multi-tenancy** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Open Source** | ✅ MIT | ❌ | ✅ Apache | ✅ Apache | ✅ Apache | ✅ BSD |
| **Pricing** | Free | $70+/mo | Free | Free | Free | Free |

## npm Packages

| Package | Description |
|---------|-------------|
| [`ruvector`](https://www.npmjs.com/package/ruvector) | **All-in-one package (recommended)** |
| [`@ruvector/wasm`](https://www.npmjs.com/package/@ruvector/wasm) | Browser/WASM bindings |
| [`@ruvector/graph`](https://www.npmjs.com/package/@ruvector/graph) | Graph database with Cypher |
| [`@ruvector/gnn`](https://www.npmjs.com/package/@ruvector/gnn) | Graph Neural Network layers |
| [`@ruvector/tiny-dancer`](https://www.npmjs.com/package/@ruvector/tiny-dancer) | AI agent routing (FastGRNN) |
| [`@ruvector/router`](https://www.npmjs.com/package/@ruvector/router) | Semantic routing engine |

```bash
# Install all-in-one (recommended)
npm install ruvector

# Or install specific packages
npm install @ruvector/graph @ruvector/gnn
```

## API Reference

### VectorDB

```typescript
class VectorDB {
  constructor(dimension: number, options?: VectorDBOptions);

  // CRUD operations
  insert(id: string, values: number[], metadata?: object): Promise<void>;
  insertBatch(vectors: Vector[], options?: BatchOptions): Promise<void>;
  get(id: string): Promise<Vector | null>;
  update(id: string, values?: number[], metadata?: object): Promise<void>;
  delete(id: string): Promise<boolean>;

  // Search
  search(query: number[], k?: number, filter?: Filter): Promise<SearchResult[]>;
  hybridSearch(query: number[], text: string, k?: number): Promise<SearchResult[]>;

  // Persistence
  save(path: string): Promise<void>;
  static load(path: string): Promise<VectorDB>;

  // Management
  stats(): Promise<IndexStats>;
  optimize(): Promise<void>;
  clear(): Promise<void>;
}
```

### GraphDB

```typescript
class GraphDB {
  constructor(options?: GraphDBOptions);

  // Cypher execution
  execute(cypher: string, params?: object): QueryResult;

  // Direct API
  createNode(label: string, properties: object): string;
  createRelationship(from: string, to: string, type: string, props?: object): void;
  createHyperedge(nodeIds: string[], type: string, props?: object): string;

  // Traversal
  shortestPath(from: string, to: string): Path | null;
  neighbors(nodeId: string, depth?: number): Node[];
}
```

### GNNLayer

```typescript
class GNNLayer {
  constructor(inputDim: number, outputDim: number, numHeads: number);

  // Inference
  forward(query: number[], neighbors: number[][], weights: number[]): number[];

  // Training
  train(data: TrainingData, config?: TrainingConfig): TrainingMetrics;
  save(path: string): void;
  static load(path: string): GNNLayer;
}
```

### Router

```typescript
class Router {
  constructor(config?: RouterConfig);

  // Routing
  route(query: number[], candidates: Candidate[]): RoutingDecision;
  routeBatch(queries: number[][], candidates: Candidate[]): RoutingDecision[];

  // Management
  reloadModel(): void;
  circuitBreakerStatus(): 'closed' | 'open' | 'half-open';
  resetCircuitBreaker(): void;
}
```

## Use Cases

### Agentic AI / Multi-Agent Systems

```javascript
// Route tasks to specialized agents
const agents = [
  { id: 'researcher', embedding: researchEmb, capabilities: ['search', 'summarize'] },
  { id: 'coder', embedding: codeEmb, capabilities: ['code', 'debug'] },
  { id: 'analyst', embedding: analysisEmb, capabilities: ['data', 'visualize'] }
];

const taskEmb = await embed("Write a Python script to analyze sales data");
const decision = router.route(taskEmb, agents);
// Routes to 'coder' agent with high confidence
```

### Recommendation Systems

```javascript
const recommendations = graph.execute(`
  MATCH (user:User {id: $userId})-[:VIEWED]->(item:Product)
  MATCH (item)-[:SIMILAR_TO]->(rec:Product)
  WHERE NOT (user)-[:VIEWED]->(rec)
    AND vector.similarity(rec.embedding, $userPreference) > 0.7
  RETURN rec
  ORDER BY vector.similarity(rec.embedding, $userPreference) DESC
  LIMIT 10
`);
```

### Semantic Caching

```javascript
const cache = new VectorDB(1536);

async function cachedLLMCall(prompt) {
  const promptEmb = await embed(prompt);

  // Check semantic cache
  const cached = await cache.search(promptEmb, 1);
  if (cached[0]?.score > 0.95) {
    return cached[0].metadata.response; // Cache hit
  }

  // Cache miss - call LLM
  const response = await llm.complete(prompt);
  await cache.insert(generateId(), promptEmb, { prompt, response });

  return response;
}
```

### Document Q&A with Sources

```javascript
async function qaWithSources(question) {
  const results = await db.search(await embed(question), 5);

  const answer = await llm.complete({
    prompt: `Answer based on these sources:\n${results.map(r =>
      `[${r.id}] ${r.metadata.content}`
    ).join('\n')}\n\nQuestion: ${question}`,
  });

  return {
    answer,
    sources: results.map(r => ({
      id: r.id,
      title: r.metadata.title,
      relevance: r.score
    }))
  };
}
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         ruvector                              │
│                  (All-in-One npm Package)                     │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   VectorDB   │   GraphDB    │   GNNLayer   │     Router      │
│   (Search)   │   (Cypher)   │    (ML)      │  (AI Routing)   │
├──────────────┴──────────────┴──────────────┴─────────────────┤
│                      Rust Core Engine                         │
│  • HNSW Index  • Cypher Parser  • Attention  • FastGRNN      │
│  • SIMD Ops    • Hyperedges     • Training   • Uncertainty   │
└──────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
      ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
      │  Native │       │  WASM   │       │   FFI   │
      │(napi-rs)│       │(wasm32) │       │   (C)   │
      └─────────┘       └─────────┘       └─────────┘
           │                  │                  │
      ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
      │ Node.js │       │ Browser │       │ Python  │
      │   Bun   │       │  Deno   │       │   Go    │
      └─────────┘       └─────────┘       └─────────┘
```

## Platform Support

| Platform | Backend | Installation |
|----------|---------|--------------|
| **Node.js 16+** | Native (napi-rs) | `npm install ruvector` |
| **Node.js (fallback)** | WASM | Automatic if native fails |
| **Bun** | Native | `bun add ruvector` |
| **Deno** | WASM | `import from "npm:ruvector"` |
| **Browser** | WASM | `npm install @ruvector/wasm` |
| **Cloudflare Workers** | WASM | `npm install @ruvector/wasm` |
| **Vercel Edge** | WASM | `npm install @ruvector/wasm` |

## Documentation

- [Getting Started Guide](https://github.com/ruvnet/ruvector/blob/main/docs/guide/GETTING_STARTED.md)
- [Cypher Reference](https://github.com/ruvnet/ruvector/blob/main/docs/api/CYPHER_REFERENCE.md)
- [GNN Architecture](https://github.com/ruvnet/ruvector/blob/main/docs/gnn-layer-implementation.md)
- [Performance Tuning](https://github.com/ruvnet/ruvector/blob/main/docs/optimization/PERFORMANCE_TUNING_GUIDE.md)
- [API Reference](https://github.com/ruvnet/ruvector/tree/main/docs/api)

## Contributing

```bash
# Clone repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector

# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build

# Benchmarks
npm run bench
```

See [CONTRIBUTING.md](https://github.com/ruvnet/ruvector/blob/main/docs/development/CONTRIBUTING.md) for guidelines.

## License

MIT License — free for commercial and personal use.

---

<div align="center">

**Built by [rUv](https://ruv.io)** • [GitHub](https://github.com/ruvnet/ruvector) • [npm](https://npmjs.com/package/ruvector)

*Vector search that gets smarter over time.*

**[⭐ Star on GitHub](https://github.com/ruvnet/ruvector)** if RuVector helps your project!

</div>
