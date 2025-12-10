# RuVector Telecom RAG

**Cognitive Automation Platform for Ericsson RAN - Self-Learning RAG System**

A production-ready, self-learning Retrieval-Augmented Generation (RAG) system for Ericsson Radio Access Network (RAN) technical documentation. Built on the RuVector ecosystem for high-performance vector operations and Graph Neural Networks.

## 🎯 Key Features

- **ELEX Documentation Processing**: Ingests Ericsson ELEX HTML documentation from ZIP files with embedded images
- **3GPP MOM XML Parsing**: Extracts Managed Object Model definitions with full attribute schemas
- **Self-Learning RAG**: Adaptive retrieval that improves based on user feedback
- **GNN-Based Optimization**: Graph Neural Networks for network topology analysis
- **Bayesian Uncertainty Quantification**: Confidence intervals for all predictions
- **Agent Swarm System**: Autonomous optimization agents (Optimizer, Validator, Auditor)
- **Tuning Paradox Solver**: Specifically designed to solve the P0/Alpha optimization challenge

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RuVector Telecom RAG Platform                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ ELEX Parser │  │ 3GPP Parser │  │ ENM Adapter │ ← Data Ingestion│
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│              ┌───────────────────────┐                              │
│              │  Self-Learning RAG    │ ← Vector Store (RuVector)    │
│              │  with Adaptive        │                              │
│              │  Relevance Weighting  │                              │
│              └───────────┬───────────┘                              │
│                          │                                          │
│         ┌────────────────┼────────────────┐                         │
│         ▼                ▼                ▼                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Network     │  │ Bayesian    │  │ Agent       │                 │
│  │ Graph       │  │ GNN Engine  │  │ Swarm       │ ← Optimization  │
│  │ Builder     │  │             │  │ Controller  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  REST API (Fastify) │ CLI Tools │ WebSocket (Real-time)             │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Node.js 20+
- npm or pnpm
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ruvector-telecom-rag.git
cd ruvector-telecom-rag

# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### Ingest Documentation

```bash
# Ingest ELEX documentation (place ZIP files in ./data/elex/)
npm run ingest:elex

# Ingest 3GPP MOM XML (place XML files in ./data/3gpp/)
npm run ingest:3gpp
```

### Start the Server

```bash
# Development mode
npm run dev

# Production mode
npm run build
npm start
```

### Use the CLI

```bash
# Interactive query mode
npx tsx src/cli/query.ts

# Single query
npx tsx src/cli/query.ts "What is the optimal alpha value for urban deployments?"

# Run optimization
npx tsx src/cli/optimize.ts -c ./data/cells.json -o ./output/recommendations.json
```

## 📡 API Endpoints

### RAG Queries

```bash
# Query documentation
POST /api/v1/rag/query
{
  "query": "How does pZeroNominalPusch affect uplink SINR?",
  "topK": 10,
  "technologies": ["LTE", "NR"]
}

# Parameter lookup
GET /api/v1/rag/parameter/pZeroNominalPusch
```

### Optimization

```bash
# Create network graph
POST /api/v1/optimization/graph
{
  "cells": [...],
  "neighborRelations": {...}
}

# Run optimization
POST /api/v1/optimization/optimize
{
  "graphId": "uuid-here"
}

# Simulate changes
POST /api/v1/optimization/simulate
{
  "graphId": "uuid-here",
  "changes": {
    "cell-1": { "alpha": 0.7, "pZeroNominalPusch": -98 }
  }
}
```

## 🧠 The Tuning Paradox

The platform is specifically designed to solve the **Tuning Paradox** in uplink power control:

- **Reducing α** (alpha) decreases inter-cell interference but risks coverage holes
- **Increasing P0** compensates for reduced alpha but may increase the noise floor

The GNN-based optimizer treats (P0, α) as a coupled vector, finding the optimal balance using:

1. **Genetic Algorithm** for candidate generation
2. **Bayesian GNN** for outcome prediction with uncertainty
3. **3-ROP Stability Protocol** for safe deployment validation

## 🔧 Configuration

Key configuration options in `.env`:

```env
# LLM Configuration
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# GNN Configuration
GNN_HIDDEN_DIM=256
GNN_NUM_LAYERS=4
BAYESIAN_MC_SAMPLES=100

# Optimization
OPTIMIZATION_POPULATION_SIZE=50
OPTIMIZATION_GENERATIONS=100

# 3GPP Parameters
ALPHA_MIN=0.4
ALPHA_MAX=1.0
P0_MIN=-126
P0_MAX=24
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# With monitoring stack
docker-compose --profile with-monitoring up -d

# Check logs
docker-compose logs -f ruvector-rag
```

## 📊 Key Technologies

- **RuVector**: High-performance vector database with HNSW indexing
- **@ruvector/gnn**: Graph Neural Network operations
- **@ruvector/graph-node**: Hypergraph support with Cypher queries
- **Fastify**: High-performance HTTP server
- **OpenAI/Anthropic**: LLM integration for generation
- **Cheerio/fast-xml-parser**: Document parsing

## 📚 Documentation Sources

The system processes:
- **ELEX Documentation**: Ericsson technical documentation in HTML format
- **3GPP MOM XML**: Managed Object Model specifications (TS 36.xxx / TS 38.xxx)
- **ENM Configuration**: Network Manager exports (CM/PM data)

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for Telecom Engineers**

*Solving the complexity crisis in modern RAN, one parameter at a time.*
