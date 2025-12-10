# RuVector API Reference

## CLI Commands

### Database Operations

#### `npx ruvector create <path>`
Create a new vector database.

```bash
npx ruvector create ./mydb --dimension 384 --metric cosine
npx ruvector create ./mydb -d 1536 -m l2
```

Options:
- `-d, --dimension <number>`: Vector dimension (default: 384)
- `-m, --metric <type>`: Distance metric: cosine, l2, ip (default: cosine)
- `--max-elements <number>`: Maximum vectors (default: 100000)
- `--ef-construction <number>`: HNSW build quality (default: 200)
- `--M <number>`: HNSW max connections (default: 16)

#### `npx ruvector insert <database> <file>`
Insert vectors from JSON file.

```bash
npx ruvector insert ./mydb vectors.json
npx ruvector insert ./mydb vectors.json --batch-size 1000
```

Options:
- `--batch-size <number>`: Batch insert size (default: 100)
- `--id-field <string>`: JSON field for ID (default: "id")
- `--embedding-field <string>`: JSON field for embedding (default: "embedding")

#### `npx ruvector search <database>`
Search for similar vectors.

```bash
npx ruvector search ./mydb --query "[0.1, 0.2, ...]" --k 10
npx ruvector search ./mydb --text "machine learning" --k 5
```

Options:
- `-q, --query <vector>`: Query vector as JSON array
- `-t, --text <string>`: Query text (uses built-in embeddings)
- `-k, --k <number>`: Number of results (default: 10)
- `--threshold <number>`: Minimum similarity (0-1)
- `--ef <number>`: Search quality parameter

#### `npx ruvector stats <database>`
Show database statistics.

```bash
npx ruvector stats ./mydb
```

Output:
```
Database: ./mydb
Vectors: 50,000
Dimension: 384
Metric: cosine
Index Size: 76.2 MB
Compression: 2.3x (tiered)
```

### GNN Operations

#### `npx ruvector gnn`
Graph Neural Network operations.

```bash
# Train GNN on existing data
npx ruvector gnn train ./mydb --epochs 100 --lr 0.001

# Apply GNN layer to query
npx ruvector gnn enhance --query "[...]" --neighbors 10

# Export trained model
npx ruvector gnn export ./mydb --output model.bin
```

Options:
- `train`: Train GNN layers on database
- `enhance`: Apply GNN to query
- `export`: Export trained model
- `import`: Import pre-trained model

### Graph Operations

#### `npx ruvector graph`
Execute Cypher queries (requires @ruvector/graph-node).

```bash
# Execute Cypher query
npx ruvector graph query "MATCH (n:Agent) RETURN n LIMIT 10"

# Create nodes and relationships
npx ruvector graph query "CREATE (a:Agent {name: 'coder'})"

# Interactive Cypher shell
npx ruvector graph shell
```

### Server Operations

#### `npx ruvector server`
Start HTTP/gRPC server.

```bash
npx ruvector server --port 8080 --database ./mydb
npx ruvector server -p 8080 -d ./mydb --grpc --grpc-port 50051
```

Options:
- `-p, --port <number>`: HTTP port (default: 8080)
- `-d, --database <path>`: Database path
- `--grpc`: Enable gRPC server
- `--grpc-port <number>`: gRPC port (default: 50051)
- `--tls`: Enable TLS
- `--cert <path>`: TLS certificate
- `--key <path>`: TLS private key

### Cluster Operations

#### `npx ruvector cluster`
Distributed cluster management.

```bash
# Initialize cluster
npx ruvector cluster init --nodes 3

# Add node
npx ruvector cluster add-node node2:5000

# Check cluster status
npx ruvector cluster status

# Rebalance shards
npx ruvector cluster rebalance
```

### Utility Commands

#### `npx ruvector benchmark`
Run performance benchmarks.

```bash
npx ruvector benchmark --vectors 100000 --dimension 384
npx ruvector benchmark --comprehensive --output report.json
```

#### `npx ruvector doctor`
Check system health.

```bash
npx ruvector doctor
```

Output:
```
✓ Native bindings: linux-x64-gnu
✓ SIMD support: AVX2, AVX-512
✓ Memory: 16GB available
✓ CPU cores: 8
✓ Node.js: v20.10.0
```

#### `npx ruvector embed`
Generate embeddings from text.

```bash
npx ruvector embed "machine learning" --model all-MiniLM-L6-v2
npx ruvector embed --file texts.txt --output embeddings.json
```

---

## JavaScript/TypeScript API

### @ruvector/core

#### VectorDB Class

```typescript
import { VectorDB } from '@ruvector/core';

// Create database
const db = new VectorDB(dimension: number, options?: VectorDBOptions);

// Options
interface VectorDBOptions {
  metric?: 'cosine' | 'l2' | 'ip';  // Distance metric
  maxElements?: number;              // Max capacity
  efConstruction?: number;           // Build quality (default: 200)
  M?: number;                        // Max connections (default: 16)
  enableCompression?: boolean;       // Enable tiered compression
  compressionTiers?: CompressionConfig;
}
```

#### Core Methods

```typescript
// Insert vector
db.insert(id: string, embedding: number[]): void;

// Batch insert
db.insertBatch(items: Array<{ id: string; embedding: number[] }>): void;

// Search k-nearest neighbors
db.search(query: number[], k: number): SearchResult[];

interface SearchResult {
  id: string;
  distance: number;
}

// Remove vector
db.remove(id: string): boolean;

// Get count
db.count(): number;

// Set search quality
db.setEfSearch(ef: number): void;

// Save to file
db.save(path: string): void;

// Load from file
db.load(path: string): void;

// Get compression stats
db.getCompressionStats(): CompressionStats;
```

### @ruvector/gnn

#### GNNLayer Class

```typescript
import { GNNLayer } from '@ruvector/gnn';

// Create GNN layer
const layer = new GNNLayer(
  inputDim: number,
  outputDim: number,
  heads?: number  // Attention heads (default: 4)
);
```

#### GNN Methods

```typescript
// Forward pass with attention
layer.forward(
  query: number[],
  neighbors: number[][],
  weights: number[]
): number[];

// Train on labeled data
await layer.train(
  data: Array<{ embedding: number[]; label: number }>,
  options?: {
    epochs?: number;
    learningRate?: number;
    batchSize?: number;
  }
): Promise<TrainingStats>;

// Save model
layer.save(path: string): void;

// Load model
layer.load(path: string): void;
```

### @ruvector/graph-node

#### GraphDB Class

```typescript
import { GraphDB } from '@ruvector/graph-node';

// Create graph database
const graph = new GraphDB(path: string);
```

#### Graph Methods

```typescript
// Execute Cypher query
graph.execute(cypher: string, params?: Record<string, any>): QueryResult;

// Transaction support
graph.beginTransaction(): Transaction;
graph.commit(): void;
graph.rollback(): void;

// Create node
graph.createNode(labels: string[], properties: Record<string, any>): string;

// Create relationship
graph.createRelationship(
  fromId: string,
  toId: string,
  type: string,
  properties?: Record<string, any>
): string;
```

### Compression API

```typescript
import { compress, decompress } from '@ruvector/gnn';

// Compress embedding
const compressed = compress(
  embedding: number[],
  ratio: number  // 0.0-1.0 compression level
): CompressedEmbedding;

// Decompress
const restored = decompress(compressed): number[];
```

---

## REST API (Server Mode)

### Endpoints

#### POST /vectors
Insert vectors.

```bash
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc-1",
    "embedding": [0.1, 0.2, ...],
    "metadata": {"title": "Example"}
  }'
```

#### POST /search
Search vectors.

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, ...],
    "k": 10,
    "threshold": 0.7
  }'
```

Response:
```json
{
  "results": [
    {"id": "doc-1", "distance": 0.05, "similarity": 0.95},
    {"id": "doc-2", "distance": 0.12, "similarity": 0.88}
  ],
  "searchTime": "0.12ms"
}
```

#### DELETE /vectors/:id
Remove vector.

```bash
curl -X DELETE http://localhost:8080/vectors/doc-1
```

#### GET /stats
Get database statistics.

```bash
curl http://localhost:8080/stats
```

#### POST /graph/query
Execute Cypher query (requires graph module).

```bash
curl -X POST http://localhost:8080/graph/query \
  -H "Content-Type: application/json" \
  -d '{
    "cypher": "MATCH (n:Agent) RETURN n LIMIT 10"
  }'
```

---

## gRPC API (Server Mode)

### Service Definition

```protobuf
service RuVector {
  rpc Insert(InsertRequest) returns (InsertResponse);
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc Delete(DeleteRequest) returns (DeleteResponse);
  rpc BatchInsert(stream InsertRequest) returns (BatchResponse);
  rpc StreamSearch(SearchRequest) returns (stream SearchResult);
}

message InsertRequest {
  string id = 1;
  repeated float embedding = 2;
  map<string, string> metadata = 3;
}

message SearchRequest {
  repeated float query = 1;
  int32 k = 2;
  float threshold = 3;
}

message SearchResult {
  string id = 1;
  float distance = 2;
  float similarity = 3;
}
```

### Node.js gRPC Client

```typescript
import { RuVectorClient } from '@ruvector/core/grpc';

const client = new RuVectorClient('localhost:50051');

// Insert
await client.insert({
  id: 'doc-1',
  embedding: [0.1, 0.2, ...]
});

// Search
const results = await client.search({
  query: [0.1, 0.2, ...],
  k: 10
});
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUVECTOR_LOG_LEVEL` | Logging level | `info` |
| `RUVECTOR_THREADS` | Worker threads | CPU cores |
| `RUVECTOR_CACHE_SIZE` | Cache size MB | `256` |
| `RUVECTOR_COMPRESSION` | Enable compression | `false` |
| `RUVECTOR_SIMD` | SIMD instruction set | `auto` |
