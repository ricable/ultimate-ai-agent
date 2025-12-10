# Phase 3: Architecture

## System Overview

The Ultimate AI Agent platform is a distributed, multi-layered architecture combining agent orchestration, vector memory, and self-learning capabilities.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  CLI Tools    │   Web UI    │   MCP Clients   │   SDK Integrations      │
└───────┬───────┴──────┬──────┴────────┬────────┴──────────┬──────────────┘
        │              │               │                   │
        ▼              ▼               ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ claude-flow │  │agentic-flow │  │ flow-nexus  │  │  ruv-swarm  │    │
│  │   v2.7.47   │  │   v1.10.2   │  │  v0.1.128   │  │   v1.0.20   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└───────┬───────────────────┬────────────────────┬───────────────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   agentdb   │  │  dspy.ts    │  │ midstreamer │  │   aidefence │    │
│  │   v1.6.1    │  │   v2.1.1    │  │   v0.2.4    │  │   v2.1.1    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└───────┬───────────────────┬────────────────────┬───────────────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEMORY & LEARNING LAYER                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────┐  ┌─────────────────────────────┐  │
│  │          RUVECTOR               │  │      NEURAL-TRADER          │  │
│  │  npm: ruvector v0.1.33          │  │  npm: neural-trader v2.6.3  │  │
│  │  npm: @ruvector/core v0.1.17    │  │  @neural-trader/core v2.0   │  │
│  │  npm: @ruvector/gnn v0.1.22     │  │  @neural-trader/mcp v2.1    │  │
│  │  crate: ruvector-core v0.1.22   │  │  @neural-trader/risk v2.6   │  │
│  │  crate: ruvector-wasm v0.1.22   │  └─────────────────────────────┘  │
│  │  crate: ruvector-node v0.1.22   │                                    │
│  └─────────────────────────────────┘                                    │
└───────┬─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Supabase   │  │   Fly.io    │  │ Cloudflare  │  │   Docker    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Published Packages (Last 2 Months)

### NPM Packages (ruvnet)

| Package | Version | Updated | Description |
|---------|---------|---------|-------------|
| `claude-flow` | 2.7.47 | 2025-12-09 | Agent orchestration platform |
| `agentic-flow` | 1.10.2 | 2025-11-10 | Model switching for Claude SDK |
| `ruvector` | 0.1.33 | 2025-12-09 | Distributed vector database |
| `@ruvector/core` | 0.1.17 | 2025-12-03 | Core vector operations |
| `@ruvector/gnn` | 0.1.22 | 2025-12-03 | Graph neural network integration |
| `@ruvector/postgres-cli` | 0.2.6 | 2025-12-08 | PostgreSQL CLI tools |
| `@ruvector/agentic-synth` | 0.1.6 | 2025-12-01 | Agentic synthesis module |
| `neural-trader` | 2.6.3 | 2025-11-25 | Neural trading system |
| `@neural-trader/core` | 2.0.0 | 2025-11-14 | Trading core module |
| `@neural-trader/mcp` | 2.1.0 | 2025-11-14 | MCP integration |
| `dspy.ts` | 2.1.1 | 2025-11-15 | Declarative self-learning JS |
| `agentdb` | 1.6.1 | 2025-10-25 | Agent database with RL |
| `flow-nexus` | 0.1.128 | 2025-09-18 | Competitive agentic platform |
| `ruv-swarm` | 1.0.20 | 2025-09-10 | Swarm orchestration |
| `agentic-robotics` | 0.2.4 | 2025-11-18 | Robotics framework |
| `agentic-jujutsu` | 2.3.6 | 2025-11-24 | Version control for agents |
| `midstreamer` | 0.2.4 | 2025-10-29 | Real-time streaming |
| `aidefence` | 2.1.1 | 2025-10-30 | AI security module |
| `agentic-payments` | 0.1.13 | 2025-10-27 | Payment processing |
| `sublinear-time-solver` | 1.5.0 | 2025-09-27 | Optimization algorithms |

### Rust Crates (ruvnet)

| Crate | Version | Updated | Description |
|-------|---------|---------|-------------|
| `ruvector-core` | 0.1.22 | 2025-12-09 | Core vector operations |
| `ruvector-wasm` | 0.1.22 | 2025-12-09 | WASM bindings |
| `ruvector-node` | 0.1.22 | 2025-12-09 | Node.js bindings |
| `ruvector-router-ffi` | 0.1.22 | 2025-12-09 | Router FFI |
| `ruvector-router-wasm` | 0.1.22 | 2025-12-09 | Router WASM |
| `temporal-compare` | 0.5.0 | 2025-09-27 | Temporal comparison |
| `bit-parallel-search` | 0.1.0 | 2025-09-28 | Bit-parallel search |
| `intrinsic-dim` | 0.0.0 | 2025-09-27 | Intrinsic dimensionality |

## Component Details

### 1. Claude-Flow (Orchestration Core)

```
claude-flow/
├── src/
│   ├── orchestrator/      # Multi-agent coordination
│   ├── swarm/             # Swarm intelligence
│   ├── mcp/               # MCP protocol implementation
│   ├── memory/            # Agent memory management
│   └── providers/         # LLM provider integrations
├── examples/
└── docs/
```

**Key Capabilities:**
- Multi-agent swarm deployment
- MCP protocol support
- Distributed task execution
- Real-time agent communication

### 2. Ruvector (Vector Memory)

```
ruvector/
├── ruvector-core/         # Rust core (Cargo)
├── ruvector-wasm/         # WASM compilation
├── ruvector-node/         # Node.js bindings (NAPI)
├── @ruvector/core/        # npm package
├── @ruvector/gnn/         # Graph neural networks
└── @ruvector/postgres-cli # PostgreSQL integration
```

**Key Capabilities:**
- Distributed vector storage
- GNN-based index learning
- Cypher query support
- Horizontal scaling with Raft

### 3. AgentDB (Reinforcement Learning)

```
agentdb/
├── agents/                # Agent definitions
├── learning/              # RL algorithms
├── consciousness/         # Temporal reasoning
└── swarm/                 # Swarm intelligence
```

**Key Capabilities:**
- Reinforcement learning integration
- Temporal reasoning
- Cognitive consciousness simulation
- Swarm intelligence

### 4. Agentic-Flow (Model Switching)

```
agentic-flow/
├── providers/             # LLM providers
│   ├── anthropic/
│   ├── openai/
│   ├── openrouter/
│   └── local/
├── router/                # Request routing
└── cost/                  # Cost optimization
```

**Key Capabilities:**
- Multi-provider support
- Cost-based routing
- Latency optimization
- Automatic failover

## Integration Points

### MCP Protocol Integration

```typescript
// MCP Server Configuration
{
  "mcpServers": {
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@latest", "mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    },
    "ruvector": {
      "command": "npx",
      "args": ["ruvector", "serve"],
      "env": {
        "RUVECTOR_DB_URL": "${DATABASE_URL}"
      }
    }
  }
}
```

### Cross-Package Dependencies

```
┌──────────────┐     depends on     ┌──────────────┐
│ claude-flow  │───────────────────▶│   ruvector   │
└──────────────┘                    └──────────────┘
       │                                   │
       │ depends on                        │ depends on
       ▼                                   ▼
┌──────────────┐                    ┌──────────────┐
│ agentic-flow │                    │@ruvector/gnn │
└──────────────┘                    └──────────────┘
       │
       │ depends on
       ▼
┌──────────────┐
│    agentdb   │
└──────────────┘
```

## Deployment Architecture

### Single Node (Development)

```bash
# Start all services locally
npx claude-flow start --mode=development
npx ruvector serve --port=6333
docker-compose up -d postgres
```

### Distributed (Production)

```yaml
# fly.toml for production deployment
[env]
  RUVECTOR_CLUSTER_SIZE = "3"
  CLAUDE_FLOW_WORKERS = "10"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

## Security Architecture

### Authentication Flow

```
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│ Client │────▶│  Auth  │────▶│  API   │────▶│ Agent  │
│        │     │Gateway │     │Gateway │     │ Pool   │
└────────┘     └────────┘     └────────┘     └────────┘
                   │
                   ▼
              ┌────────┐
              │Supabase│
              │  Auth  │
              └────────┘
```

### Key Security Features
- JWT-based authentication (Supabase)
- API key rotation
- Rate limiting per agent
- Audit logging
- Secrets management via environment

---

*SPARC Phase 3 Complete - Proceed to [04-refinement.md](04-refinement.md)*
