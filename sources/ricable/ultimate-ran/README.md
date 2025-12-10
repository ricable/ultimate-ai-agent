# TITAN: Neuro-Symbolic RAN Platform

[![Version](https://img.shields.io/badge/version-7.0--alpha-blue.svg)](./plan.md)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Unfinished Coverage](https://img.shields.io/badge/coverage-unfinished-yellow)]()

> Autonomous 5G/6G RAN optimization via Multi-Provider AI and LLM Council Architecture

## 🚀 Multi-Provider AI Integration

TITAN now supports concurrent execution with multiple AI providers:

- **Claude Code PRO MAX** - Primary reasoning and optimization
- **Google AI Pro (Gemini 2.0)** - Multimodal analysis and anomaly detection
- **E2B Sandboxes** - Isolated safety validation and digital twin simulation
- **OpenRouter** - Fallback and additional model access
- **Agentic Flow** - Multi-agent coordination with QUIC transport
- **Claude Flow** - Swarm orchestration and consensus

### Quick Setup (5 minutes)

**New users:** See [Authentication Guide](docs/AUTH.md) for OAuth subscriptions, API keys, or free tier setup. Also available: [Quick Start Guide](docs/QUICK-START.md) for complete setup.

```bash
# 1. Configure API keys
cp config/.env.template config/.env
nano config/.env  # Add your API keys

# 2. Install and test
npm install
npm run test:integration  # Verify all APIs work

# 3. Start TITAN (choose one)
npm run start:local       # Local Mac Silicon
npm run start:devpod      # DevPod with Docker
npm run docker:up         # Docker Compose
```

**Existing users:**

```bash
npm install
npm start        # Start orchestrator
npm test         # Run all tests (Vitest)
npm run coverage # Generate coverage report
npm run benchmark # Run benchmarks
```

## Architecture

### Five-Layer Stack
```
Layer 5: AG-UI Glass Box Interface (real-time visualization)
Layer 4: LLM Council (Multi-agent debate & consensus)
         ├─ Claude Code PRO MAX (primary reasoning)
         ├─ Google Gemini 2.0 (multimodal analysis)
         └─ OpenRouter (fallback & additional models)
Layer 3: SPARC Governance (5-gate validation)
Layer 2: Cognitive Memory (agentdb + HNSW vectors)
Layer 1: QUIC Transport (agentic-flow with 0-RTT)
```

### Multi-Provider AI Flow
```
┌─────────────────────────────────────────────────────────┐
│  Claude PRO MAX  ◄──── Consensus Mode ────►  Gemini 2.0 │
└─────────────────────────────────────────────────────────┘
         │                                    │
         └──────►  E2B Sandboxes  ◄───────────┘
              (Digital Twin Validation)
                      │
         ┌────────────▼────────────┐
         │  Agentic Flow (QUIC)    │
         │  AgentDB + HNSW         │
         │  QuDAG Consensus        │
         └─────────────────────────┘
```

### SPARC Methodology
```
Specification → Pseudocode → Architecture → Refinement → Completion
```

### Key Modules

- **SMO (Service Management & Orchestration):** Handles PM (Performance Management) and FM (Fault Management) data pipelines.
- **Knowledge:** Loads and indexes 3GPP specifications using Vector Search.
- **GNN (Graph Neural Networks):** Optimizes uplink parameters using spatial embeddings.
- **Dojo App UI:** Frontend component for visualizing and interacting with TITAN.
- **Learning:** Self-learning agents using Reinforcement Learning (RL) and Q-tables.

### 3-ROP Closed-Loop Governance

Parameter changes are monitored across three Roll-Out Periods:
- **ROP 1:** Observe and collect PM counters
- **ROP 2:** Compare to predictions (confidence interval check)
- **ROP 3:** Confirm success or trigger rollback

---

## Agent Types

See the full [Agent Documentation](./AGENTS.md) for detailed roles and capabilities.

TITAN employs a hierarchy of specialized agents, including **Architects** (planning), **Guardians** (safety), and **Sentinels** (monitoring).

## Performance

| Metric | Result | Target |
|--------|--------|--------|
| Vector Search | 0.12ms | <10ms |
| Consensus | <500ms | <5s |
| Safety Check | 0.05ms | <100ms |
| Test Coverage | >40% | 80% |

## AI Council & Providers

### Primary Council Members
- **Analyst** (Claude Sonnet 4.5): Deep reasoning and Lyapunov chaos detection
- **Historian** (Gemini 2.0): Multimodal episode retrieval and pattern recognition
- **Strategist** (Claude/OpenRouter): RAN parameter proposals and optimization
- **Chairman**: Multi-LLM consensus synthesis

### AI Strategies
- **Consensus** (95%+ confidence): Both Claude & Gemini must agree - recommended for production
- **Claude Primary** (85-90%): Claude leads with Gemini validation - fast and reliable
- **Gemini Primary** (80-85%): Gemini leads with multimodal analysis - visual insights
- **Parallel** (70-80%): Both run independently - maximum speed

Configure strategy in `config/.env`:
```bash
AGENTIC_FLOW_STRATEGY=consensus  # or claude_primary, gemini_primary, parallel
```

## Key Scripts

### Multi-Provider AI
```bash
npm run start:local        # Start locally on Mac Silicon
npm run start:devpod       # Start in DevPod environment
npm run test:integration   # Test all API integrations
npm run ui:integration     # Test AI integration with live data
```

### System Management
```bash
npm run orchestrate        # claude-flow orchestration
npm run db:status          # AgentDB memory status
npm run sentinel:monitor   # Chaos detection
npm run agui:start         # AG-UI dashboard server
npm run agui:frontend      # Open AG-UI in browser
npm run swarm:spawn        # Spawn agent swarm
npm run hive:status        # Check hive-mind status
```

### Docker & DevPod
```bash
npm run docker:up          # Start Docker Compose
npm run docker:down        # Stop Docker Compose
npm run docker:logs        # View container logs
devpod up titan-ran        # Start DevPod workspace
devpod ssh titan-ran       # SSH into DevPod
```

## Technology Stack

### AI Providers
- [Anthropic Claude](https://console.anthropic.com/) - Claude Code PRO MAX (Sonnet 4.5)
- [Google AI](https://ai.google.dev/) - Gemini 2.0 Flash Exp
- [E2B](https://e2b.dev/) - Sandboxed code execution and digital twins
- [OpenRouter](https://openrouter.ai/) - Multi-model access and fallback

### Orchestration & Memory
- [claude-flow](https://github.com/ruvnet/claude-flow) - Multi-agent orchestration
- [agentic-flow](https://github.com/ruvnet/agentic-flow) - QUIC transport with 0-RTT
- [agentdb](https://npmjs.com/package/agentdb) - Cognitive memory and persistence
- [ruvector](https://npmjs.com/package/ruvector) - HNSW vector indexing (150x faster)

### Security & Consensus
- [QuDAG](https://github.com/ruvnet/QuDAG) - Quantum-resistant ledger
- ML-DSA-87 - Post-quantum digital signatures
- ML-KEM-768 - Post-quantum key encapsulation

### Development Environment
- **Mac Silicon Support**: ARM64 optimized with NEON intrinsics
- **DevPod**: Workspace isolation with Docker provider
- **E2B Sandboxes**: Isolated execution environments
---

### Documentation

### Setup & Configuration
- **[Free AI Setup Guide](docs/FREE-SETUP-GUIDE.md)** - Get TITAN running with zero API costs
- **[Quick Start](docs/QUICK-START.md)** - Get running in 5 minutes
- **[AI Swarm Quick Start](docs/AI-SWARM-QUICK-START.md)** - Get the multi-agent consensus running
- **[Multi-Provider Setup](docs/MULTI-PROVIDER-SETUP.md)** - Complete configuration guide

### Architecture & Development
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and SPARC methodology
- **[Agent Documentation](AGENTS.md)** - Detailed agent roles and architecture
- **[Full PRD](plan.md)** - Complete Product Requirements Document (if exists)

### External Resources
- **[SPARC Methodology](https://github.com/ruvnet/claude-flow/wiki/SPARC-Methodology)** - Development workflow
- **[AG-UI Protocol](https://docs.ag-ui.com/introduction)** - Frontend integration
- **[claude-flow Wiki](https://github.com/ruvnet/claude-flow/wiki)** - Orchestration details


---

## Success Metrics

| Metric | Target |
|:-------|:-------|
| UL SINR Improvement | +26% |
| Spectral Efficiency | +7% |
| System Uptime | >=99.9% |
| Self-Healing Success | >=85% |
| URLLC PLR | <=10^-5 |
| Decision Latency | <5 minutes |
| Vector Query Latency | <10ms (p95) |

---

## Security

- **Quantum-Resistant:** ML-DSA-87 signatures, ML-KEM-768 encryption
- **Immutable Audit:** QuDAG DAG-based ledger for all changes
- **3GPP Compliant:** TS 28.552, TS 28.310, TS 23.288
- **Sandboxed Execution:** All simulations in E2B isolated environments

---

## Contributing

This is an internal Ericsson project. For contribution guidelines, contact the Autonomous Networks Division.

---

## License

Proprietary - Ericsson AB
