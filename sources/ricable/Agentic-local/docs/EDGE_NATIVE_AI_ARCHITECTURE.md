# Edge-Native AI Architecture

## Decentralized AI Platform with Kairos, SpinKube, K3s, and Ruvnet Ecosystem

This document provides a comprehensive overview of the Edge-Native AI architecture, implementing the framework described in "Decentralized Edge-Native AI Architectures: A Comprehensive Framework for Distributed SaaS."

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Infrastructure Layer (Kairos)](#infrastructure-layer-kairos)
3. [Orchestration Layer (K3s)](#orchestration-layer-k3s)
4. [Compute Layer (SpinKube)](#compute-layer-spinkube)
5. [Application Layer (Ruvnet Ecosystem)](#application-layer-ruvnet-ecosystem)
6. [Connectivity Layer (A2A & MCP)](#connectivity-layer-a2a--mcp)
7. [AI Gateway (LiteLLM)](#ai-gateway-litellm)
8. [Secure Execution (E2B)](#secure-execution-e2b)
9. [Deployment Guide](#deployment-guide)
10. [API Reference](#api-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Edge-Native AI Platform                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   FastAPI   │  │   LiteLLM   │  │  A2A Proto  │  │    MCP     │ │
│  │   Backend   │  │   Gateway   │  │   Handler   │  │   Server   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                │                │                │        │
│  ┌──────┴────────────────┴────────────────┴────────────────┴──────┐ │
│  │                     Ruvnet Orchestrator                         │ │
│  │  ┌──────────┐ ┌───────────┐ ┌─────────┐ ┌──────────┐           │ │
│  │  │ Agentic  │ │  Claude   │ │ AgentDB │ │ RuVector │           │ │
│  │  │  Flow    │ │   Flow    │ │         │ │          │           │ │
│  │  └──────────┘ └───────────┘ └─────────┘ └──────────┘           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     SpinKube (WebAssembly)                      │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │ │
│  │  │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │ │  ...   │       │ │
│  │  │ Wasm   │ │ Wasm   │ │ Wasm   │ │ Wasm   │ │        │       │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                         K3s Cluster                             │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │ │
│  │  │  Redis   │ │ Postgres │ │ LocalAI  │ │ Kong AI  │          │ │
│  │  │          │ │ +pgvector│ │          │ │ Gateway  │          │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Kairos (Immutable OS)                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │ │
│  │  │  x86 Node    │  │  ARM64 Node  │  │  Edge Node   │         │ │
│  │  │  (Linux)     │  │  (Mac/RPi)   │  │  (Embedded)  │         │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │ │
│  │                   P2P Mesh (EdgeVPN)                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Layer (Kairos)

### Overview

Kairos provides an immutable, container-based Linux distribution that treats the OS as a container image. This enables:

- **Atomic Upgrades**: A/B partition scheme for safe rollbacks
- **P2P Mesh Networking**: Decentralized cluster formation via EdgeVPN
- **Heterogeneous Support**: Both amd64 and arm64 architectures

### Configuration Files

```
infrastructure/kairos/
├── cloud-config.yaml           # Main configuration
├── node-profiles/
│   ├── control-plane.yaml      # Control plane nodes
│   ├── worker-x86.yaml         # x86 worker nodes
│   └── worker-arm64.yaml       # ARM64 worker nodes
└── generate-token.sh           # Network token generator
```

### Key Features

1. **Immutable Root Filesystem**: No configuration drift
2. **Declarative Configuration**: GitOps-compatible
3. **Zero-Touch Provisioning**: Automatic cluster joining
4. **Self-Healing Mesh**: Automatic route updates on IP changes

### Quick Start

```bash
# Generate cluster tokens
./infrastructure/kairos/generate-token.sh

# Flash Kairos ISO to node
# Boot node - it automatically joins the cluster
```

---

## Orchestration Layer (K3s)

### Overview

K3s is a lightweight, certified Kubernetes distribution optimized for edge and IoT deployments.

### Manifests

```
infrastructure/k3s/
├── 00-namespaces.yaml          # Namespace definitions
├── 01-storage.yaml             # Storage classes and PVCs
├── 02-secrets.yaml             # Secret configuration
├── 10-redis.yaml               # Redis StatefulSet
├── 11-postgres.yaml            # PostgreSQL with pgvector
├── 20-localai.yaml             # LocalAI inference server
├── 21-litellm.yaml             # LiteLLM proxy
└── 30-spinkube-operator.yaml   # SpinKube operator
```

### Namespaces

- `edge-ai-system`: Core platform components
- `edge-ai-agents`: WebAssembly agent workloads
- `edge-ai-inference`: Local AI inference
- `edge-ai-data`: Data stores (Redis, PostgreSQL)
- `edge-ai-gateway`: API gateway components

### Deploy

```bash
kubectl apply -f infrastructure/k3s/
```

---

## Compute Layer (SpinKube)

### Overview

SpinKube enables running WebAssembly (Wasm) workloads on Kubernetes, providing:

- **Millisecond Cold Starts**: vs seconds for containers
- **Extreme Density**: Thousands of agents per node
- **Memory Safety**: Sandboxed execution
- **Portability**: Same binary on x86 and ARM

### Configuration

```
infrastructure/spinkube/
├── spin.toml                   # Spin application manifest
└── spinapp-agent.yaml          # SpinApp Kubernetes CRD
```

### SpinApp Example

```yaml
apiVersion: core.spinoperator.dev/v1alpha1
kind: SpinApp
metadata:
  name: ai-agent-controller
  namespace: edge-ai-agents
spec:
  image: "ghcr.io/edge-ai/agent-controller:latest"
  replicas: 3
  executor: containerd-shim-spin
```

---

## Application Layer (Ruvnet Ecosystem)

### Components

#### 1. Agentic Flow (`src/ruvnet/agentic-flow.js`)

High-performance agent swarm orchestration with Agent Booster (352x WASM speedup).

```javascript
import { AgenticFlowIntegration } from './ruvnet/index.js';

const agenticFlow = new AgenticFlowIntegration({
  booster: { enabled: true },
  providers: ['local', 'openai']
});

// Create agent
const agent = await agenticFlow.createAgent({
  name: 'coder-agent',
  type: 'coder',
  capabilities: ['code-generation', 'code-review']
});

// Execute task
const result = await agenticFlow.executeTask(agent.id, {
  type: 'generate',
  input: 'Create a Python function to sort a list'
});
```

#### 2. Claude Flow (`src/ruvnet/claude-flow.js`)

Enterprise workflow orchestration with ReasoningBank and SPARC methodology.

```javascript
import { ClaudeFlowIntegration } from './ruvnet/index.js';

const claudeFlow = new ClaudeFlowIntegration({
  reasoningBank: { enabled: true },
  sparc: { enabled: true }
});

// Create SPARC workflow
const workflow = await claudeFlow.createWorkflow('sparc-development', {
  task: 'Build a REST API for user management',
  model: 'qwen-coder'
});

// Execute workflow
const result = await claudeFlow.executeWorkflow(workflow.id);
```

#### 3. AgentDB (`src/ruvnet/agentdb.js`)

Distributed agent state management with SQLite local + Redis sync.

```javascript
import { AgentDBIntegration } from './ruvnet/index.js';

const agentdb = new AgentDBIntegration({
  local: { path: './data/agentdb.sqlite' },
  distributed: { redisUrl: 'redis://localhost:6379' }
});

// Store agent state
await agentdb.createAgent({
  id: 'agent-001',
  name: 'Research Agent',
  type: 'researcher'
});

// Add memory
await agentdb.addMemory('agent-001', {
  type: 'conversation',
  content: { role: 'user', text: 'Find papers on transformers' }
});
```

#### 4. RuVector (`src/ruvnet/ruvector.js`)

High-performance vector database for semantic search.

```javascript
import { RuVectorIntegration } from './ruvnet/index.js';

const ruvector = new RuVectorIntegration({
  backend: 'local', // or 'postgres'
  dimensions: 1536
});

// Add knowledge
await ruvector.add('doc-001', 'WebAssembly enables portable binary code', {
  type: 'documentation',
  source: 'wiki'
});

// Search
const results = await ruvector.search('What is WASM?', { k: 5 });
```

#### 5. Unified Orchestrator (`src/ruvnet/orchestrator.js`)

Master orchestrator integrating all components.

```javascript
import { initializeRuvnet } from './ruvnet/index.js';

const orchestrator = await initializeRuvnet();

// Create intelligent agent
const agent = await orchestrator.createAgent({
  name: 'full-stack-agent',
  capabilities: ['code', 'research', 'analysis']
});

// Chat with RAG
const response = await orchestrator.chat(conversationId, 'Explain SpinKube');
```

---

## Connectivity Layer (A2A & MCP)

### A2A Protocol (`src/a2a/protocol.js`)

Google's Agent-to-Agent protocol for cross-agent communication.

#### Agent Card (`/.well-known/agent.json`)

```json
{
  "name": "Edge-Native-AI",
  "description": "Decentralized AI agent platform",
  "url": "http://agent.edge-ai.local",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "code-generation",
      "name": "Code Generation",
      "description": "Generate code using local models"
    }
  ]
}
```

#### Send Task to Another Agent

```javascript
import { A2AProtocol } from './a2a/protocol.js';

const a2a = new A2AProtocol({
  agentId: 'my-agent',
  discovery: { method: 'kubernetes' }
});

// Discover agents
const agents = await a2a.findAgentsByCapability('code-generation');

// Send task
const result = await a2a.sendTask(agents[0].url, {
  input: 'Generate a sorting algorithm'
});
```

### MCP Server (`src/api/routers/mcp.py`)

Model Context Protocol for tool and resource access.

#### Available Tools

- `execute_code`: Run code in secure sandbox
- `search_knowledge`: Query vector database
- `create_agent`: Create new AI agent
- `execute_workflow`: Run SPARC workflow
- `chat_with_agent`: Conversational interface

---

## AI Gateway (LiteLLM)

### Configuration (`config/litellm/config.yaml`)

Local-first routing with cloud fallback:

```yaml
model_list:
  # Local models (free)
  - model_name: qwen-coder
    litellm_params:
      model: openai/qwen2.5-coder-7b
      api_base: http://localai:8080/v1

  # Cloud fallback
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  routing_strategy: "cost-based-routing"
  fallbacks:
    qwen-coder: ["gpt-4o"]
```

### Usage

```javascript
import { AIGateway } from './gateway/index.js';

const gateway = new AIGateway({
  litellmUrl: 'http://localhost:4000',
  routing: { strategy: 'local-first' }
});

// Auto-routes to cheapest available model
const response = await gateway.chatCompletion({
  model: 'coder', // Uses qwen-coder locally, falls back to gpt-4o
  messages: [{ role: 'user', content: 'Write a Python function' }]
});
```

---

## Secure Execution (E2B)

### Overview (`src/e2b/sandbox.js`)

E2B provides Firecracker microVMs for secure code execution.

### Usage

```javascript
import { HybridSandbox } from './e2b/sandbox.js';

const sandbox = new HybridSandbox({
  preferLocal: false,
  cloudForDataScience: true
});

// Automatically routes:
// - Simple code → Local Docker
// - Data science → E2B cloud
const result = await sandbox.executeCode(`
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df.describe())
`, { language: 'python' });
```

---

## Deployment Guide

### Prerequisites

- Docker or containerd
- K3s or full Kubernetes cluster
- Node.js >= 20.0.0
- Python >= 3.11 (for FastAPI)

### Step 1: Infrastructure Setup

```bash
# Generate Kairos tokens
./infrastructure/kairos/generate-token.sh

# Deploy K3s manifests
kubectl apply -f infrastructure/k3s/

# Deploy SpinKube
kubectl apply -f infrastructure/spinkube/
```

### Step 2: Application Setup

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Start Services

```bash
# Start FastAPI backend
npm run start:api

# Start AI Gateway
npm run start:gateway

# Start agent orchestration
npm run start:agent
```

### Environment Variables

```bash
# .env
LITELLM_URL=http://localhost:4000
LITELLM_MASTER_KEY=sk-your-key
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/edgeai
E2B_API_KEY=your-e2b-key  # Optional
OPENAI_API_KEY=sk-...      # Optional cloud fallback
```

---

## API Reference

### FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | A2A Agent Card |
| `/api/v1/agents` | GET/POST | Agent management |
| `/api/v1/agents/{id}/chat` | POST | Chat with agent |
| `/api/v1/workflows` | GET/POST | Workflow management |
| `/a2a` | POST | A2A JSON-RPC endpoint |
| `/mcp/tools` | GET | List MCP tools |
| `/mcp/tools/{name}` | POST | Execute MCP tool |
| `/health` | GET | Health check |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/agent/{id}` | Real-time agent communication |
| `/a2a/stream/{task_id}` | A2A task streaming (SSE) |

---

## License

MIT License - See LICENSE file for details.
