# 3GPP Knowledge Graph - RuVector/PostgreSQL Integration

Advanced 3GPP Knowledge Graph integration powered by **RuVector** - the most advanced AI vector database for PostgreSQL with Graph Neural Networks, Cypher queries, hyperbolic embeddings, and 39 attention mechanisms.

## Overview

This project downloads the 3GPP knowledge graph dataset from Hugging Face (`otellm/3gpp_knowledgeGraph`) in GraphML format and loads it into RuVector, enabling:

- **Cypher Queries**: Neo4j-compatible graph query language
- **Graph Neural Networks**: GCN, GraphSAGE, GAT for learning on graphs
- **Hyperbolic Embeddings**: Poincaré and Lorentz for hierarchical data
- **Attention Mechanisms**: 39 types including Flash Attention, Multi-Head, etc.
- **Vector Similarity**: Semantic search with HNSW and IVFFlat indexes
- **Hybrid Search**: BM25 + semantic vector search
- **Graph Algorithms**: Shortest path, traversal, statistics

## Features

✨ **Graph Database**
- Native graph storage with Cypher query support
- Graph Neural Networks (GCN, GraphSAGE, GAT)
- Advanced graph algorithms and analytics

🔍 **Vector Search**
- 53+ SQL functions for vector operations
- HNSW and IVFFlat indexing
- 8+ distance metrics including hyperbolic

🧠 **AI/ML Integration**
- 39 attention mechanisms (Flash, Multi-Head, Cross, etc.)
- Self-learning with ReasoningBank
- Agent routing with Tiny Dancer
- Vector quantization for compression

📊 **3GPP Specific**
- Automated GraphML import from Hugging Face
- Specification dependency analysis
- Protocol relationship learning
- Version evolution tracking

## Prerequisites

- **Node.js** v14 or higher
- **PostgreSQL** 14+ (or use RuVector's built-in installation)
- Internet connection for downloading dataset

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Install RuVector PostgreSQL

**Option A: Automatic Installation (Recommended)**

```bash
npm run install-ruvector
```

This installs PostgreSQL + RuVector extension automatically via Docker or native installation.

**Option B: Use Existing PostgreSQL**

If you already have PostgreSQL, skip the installation and update `.env` with your credentials.

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work with RuVector installation)
```

### 4. Run Complete Setup

```bash
npm start
```

This will:
1. Download the 3GPP dataset from Hugging Face
2. Parse the GraphML data
3. Load it into RuVector graph database
4. Create the `3gpp_knowledge_graph` graph

## Usage

### Run Example Queries

```bash
# Cypher queries and graph operations
npm run query

# Advanced: GNN, embeddings, attention mechanisms
npm run advanced
```

### Individual Steps

```bash
# Download dataset only
npm run download

# Load data into database (requires downloaded data)
npm run load
```

### Direct CLI Usage

```bash
# Cypher queries
npx @ruvector/postgres-cli graph query 3gpp_knowledge_graph "MATCH (n) RETURN n LIMIT 10"

# Graph statistics
npx @ruvector/postgres-cli graph stats 3gpp_knowledge_graph

# Shortest path
npx @ruvector/postgres-cli graph shortest-path 3gpp_knowledge_graph --from "node1" --to "node2"

# List all graphs
npx @ruvector/postgres-cli graph list
```

## Cypher Query Examples

### Find Nodes

```cypher
# Get all nodes
MATCH (n) RETURN n LIMIT 10

# Find nodes with specific property
MATCH (n) WHERE n.type = 'specification' RETURN n

# Count nodes by type
MATCH (n) WHERE n.type IS NOT NULL
RETURN n.type, count(*) as count
ORDER BY count DESC
```

### Explore Relationships

```cypher
# Get relationship types and counts
MATCH (n)-[r]->(m)
RETURN type(r) as relationship_type, count(*) as count
GROUP BY type(r)
ORDER BY count DESC

# 2-hop neighborhood
MATCH (n)-[r1]->(m)-[r2]->(o)
RETURN n, type(r1), m, type(r2), o
LIMIT 10

# Find triangles
MATCH (a)-[]->(b)-[]->(c)-[]->(a)
RETURN a, b, c
LIMIT 5
```

### Pattern Matching

```cypher
# Find specification dependencies
MATCH (spec:Specification)-[:DEPENDS_ON]->(dep:Specification)
RETURN spec.name, collect(dep.name) as dependencies

# Search by property value
MATCH (n) WHERE toString(n) CONTAINS "5G"
RETURN n LIMIT 10
```

## Graph Neural Networks (GNN)

Create and use GNN layers for learning on your graph:

```bash
# Create GCN layer
npx @ruvector/postgres-cli gnn create 3gpp_gcn \
  --type gcn \
  --input-dim 128 \
  --hidden-dim 64 \
  --output-dim 32

# Create GraphSAGE layer
npx @ruvector/postgres-cli gnn create 3gpp_sage \
  --type graphsage \
  --input-dim 128 \
  --hidden-dim 64

# Create GAT (Graph Attention Network)
npx @ruvector/postgres-cli gnn create 3gpp_gat \
  --type gat \
  --input-dim 128 \
  --hidden-dim 64 \
  --num-heads 4
```

## Vector Embeddings

Add semantic search to your graph:

```bash
# Create vector table for node embeddings
npx @ruvector/postgres-cli vector create 3gpp_embeddings \
  --dim 384 \
  --index hnsw

# Insert embeddings (after generating with your embedding model)
npx @ruvector/postgres-cli vector insert 3gpp_embeddings \
  --file embeddings.json

# Semantic search
npx @ruvector/postgres-cli vector search 3gpp_embeddings \
  --query "[0.1, 0.2, ..., 0.384]" \
  --top-k 10 \
  --metric cosine
```

## Hyperbolic Embeddings

Perfect for hierarchical 3GPP specification structures:

```bash
# Poincaré distance
npx @ruvector/postgres-cli hyperbolic poincare-distance \
  --x "[0.1, 0.2, 0.3]" \
  --y "[0.4, 0.5, 0.6]"

# Lorentz distance
npx @ruvector/postgres-cli hyperbolic lorentz-distance \
  --x "[0.1, 0.2, 0.3]" \
  --y "[0.4, 0.5, 0.6]"
```

## Attention Mechanisms

39 types available for various AI tasks:

```bash
# Scaled Dot-Product Attention
npx @ruvector/postgres-cli attention scaled-dot-product \
  --query "[1.0, 2.0, 3.0]" \
  --key "[4.0, 5.0, 6.0]" \
  --value "[7.0, 8.0, 9.0]"

# Multi-Head Attention
npx @ruvector/postgres-cli attention multi-head \
  --query "[1.0, 2.0, 3.0, 4.0]" \
  --key "[5.0, 6.0, 7.0, 8.0]" \
  --value "[9.0, 10.0, 11.0, 12.0]" \
  --num-heads 2
```

## Hybrid Search (BM25 + Vectors)

Combine keyword and semantic search:

```bash
# Create sparse vector for BM25
npx @ruvector/postgres-cli sparse create \
  --indices "[0, 5, 10, 15]" \
  --values "[0.8, 0.6, 0.4, 0.2]" \
  --dim 10000

# BM25 scoring
npx @ruvector/postgres-cli sparse bm25 \
  --query '{"indices": [1,5,10], "values": [0.8,0.5,0.3]}' \
  --doc '{"indices": [1,5], "values": [2,1]}' \
  --doc-len 150 \
  --avg-doc-len 200
```

## Practical Use Cases

### 1. Semantic Specification Search

```javascript
// Generate embeddings for all specs
// Use vector similarity to find related specifications
// Combine with BM25 for hybrid search
```

### 2. Protocol Dependency Analysis

```cypher
# Find all dependencies
MATCH (spec)-[:DEPENDS_ON*]->(dep)
RETURN spec.name, collect(dep.name)

# Critical paths
MATCH path = shortestPath((s:Spec)-[*]-(t:Spec))
WHERE s.name = "TS 38.401" AND t.name = "TS 38.331"
RETURN path
```

### 3. Version Evolution Tracking

```cypher
# Track changes across releases
MATCH (v1:Version)-[:EVOLVES_TO]->(v2:Version)
RETURN v1.release, v2.release, v2.changes
```

### 4. Intelligent Q&A System

Combine:
- Cypher for graph traversal
- Vector search for semantic matching
- Attention mechanisms for context
- ReasoningBank for learning

## Project Structure

```
.
├── index.js                 # Main integration script
├── download-dataset.js      # Hugging Face dataset downloader
├── load-to-postgres.js      # GraphML parser & RuVector loader
├── query-examples.js        # Cypher and graph query examples
├── advanced-examples.js     # GNN, embeddings, attention examples
├── package.json            # Dependencies and npm scripts
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── data/                   # Downloaded GraphML files (runtime)
└── README.md               # This file
```

## RuVector vs pgvector

| Feature | pgvector | RuVector |
|---------|----------|----------|
| Vector Search | HNSW, IVFFlat | HNSW, IVFFlat |
| Distance Metrics | 3 | 8+ (including hyperbolic) |
| Graph Database | ❌ | ✅ Cypher queries |
| Attention Mechanisms | ❌ | ✅ 39 types |
| Graph Neural Networks | ❌ | ✅ GCN, GraphSAGE, GAT |
| Hyperbolic Embeddings | ❌ | ✅ Poincaré, Lorentz |
| Sparse Vectors / BM25 | ❌ | ✅ Full support |
| Self-Learning | ❌ | ✅ ReasoningBank |
| Agent Routing | ❌ | ✅ Tiny Dancer |

## Troubleshooting

### Connection Error

```bash
# Install RuVector
npm run install-ruvector

# Or check your PostgreSQL credentials in .env
```

### Dataset Download Issues

- Check internet connection
- Verify dataset exists: https://huggingface.co/datasets/otellm/3gpp_knowledgeGraph
- For private datasets, set `HF_TOKEN` in `.env`

### GraphML Parsing Errors

- Check the GraphML file in `data/` directory
- Verify file isn't corrupted
- Check console output for specific errors

## Dataset Information

- **Dataset**: otellm/3gpp_knowledgeGraph
- **Format**: GraphML
- **Source**: Hugging Face Datasets
- **URL**: https://huggingface.co/datasets/otellm/3gpp_knowledgeGraph

## Dependencies

- `@ruvector/postgres-cli` - Advanced AI vector database for PostgreSQL
- `@huggingface/hub` - Hugging Face dataset integration
- `pg` - PostgreSQL client
- `xml2js` - GraphML XML parsing
- `dotenv` - Environment variable management

## Documentation

- [RuVector Documentation](https://github.com/ruvnet/ruvector)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [3GPP Specifications](https://www.3gpp.org/specifications)

## License

ISC

## Contributing

Contributions welcome! Please submit a Pull Request.

## Support

- GitHub Issues: [Report issues](https://github.com/your-repo/issues)
- RuVector: [@ruvector/postgres-cli](https://www.npmjs.com/package/@ruvector/postgres-cli)
