# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🎯 Claude Code vs MCP Tools

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY COORDINATE:
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

**KEY**: MCP coordinates the strategy, Claude Code's Task tool executes with real agents.

## 🚀 Quick Setup

```bash
# Add MCP servers (Claude Flow required, others optional)
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start  # Optional: Enhanced coordination
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # Optional: Cloud features
```

## MCP Tool Categories

### Coordination
`swarm_init`, `agent_spawn`, `task_orchestrate`

### Monitoring
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

### System
`benchmark_run`, `features_detect`, `swarm_monitor`

### Flow-Nexus MCP Tools (Optional Advanced Features)
Flow-Nexus extends MCP capabilities with 70+ cloud-based orchestration tools:

**Key MCP Tool Categories:**
- **Swarm & Agents**: `swarm_init`, `swarm_scale`, `agent_spawn`, `task_orchestrate`
- **Sandboxes**: `sandbox_create`, `sandbox_execute`, `sandbox_upload` (cloud execution)
- **Templates**: `template_list`, `template_deploy` (pre-built project templates)
- **Neural AI**: `neural_train`, `neural_patterns`, `seraphina_chat` (AI assistant)
- **GitHub**: `github_repo_analyze`, `github_pr_manage` (repository management)
- **Real-time**: `execution_stream_subscribe`, `realtime_subscribe` (live monitoring)
- **Storage**: `storage_upload`, `storage_list` (cloud file management)

**Authentication Required:**
- Register: `mcp__flow-nexus__user_register` or `npx flow-nexus@latest register`
- Login: `mcp__flow-nexus__user_login` or `npx flow-nexus@latest login`
- Access 70+ specialized MCP tools for advanced orchestration

## 🚀 Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages

### Example Full-Stack Development:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")
  
  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }
  
  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## 📋 Agent Coordination Protocol

### Every Agent Spawned via Task Tool MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT WORKFLOW: MCP Coordinates, Claude Code Executes

```javascript
// Step 1: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 2: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")
  
  // Batch ALL todos in ONE call
  TodoWrite { todos: [
    {id: "1", content: "Research API patterns", status: "in_progress", priority: "high"},
    {id: "2", content: "Design database schema", status: "in_progress", priority: "high"},
    {id: "3", content: "Implement authentication", status: "pending", priority: "high"},
    {id: "4", content: "Build REST endpoints", status: "pending", priority: "high"},
    {id: "5", content: "Write unit tests", status: "pending", priority: "medium"},
    {id: "6", content: "Integration tests", status: "pending", priority: "medium"},
    {id: "7", content: "API documentation", status: "pending", priority: "low"},
    {id: "8", content: "Performance optimization", status: "pending", priority: "low"}
  ]}
  
  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"
```

### ❌ WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.0.0)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training
- 📊 Bottleneck Analysis
- 🤖 Smart Auto-Spawning
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TITAN** (Ericsson Gen 7.0) is a Neuro-Symbolic RAN (Radio Access Network) platform using a "Cognitive Mesh" of AI agents to autonomously optimize 5G/6G networks. The system combines LLM-based reasoning (DeepSeek, Gemini, Claude) with symbolic safety verification (Lyapunov analysis, digital twins) to propose and validate network parameter changes.

## Architecture Overview

### Five-Layer Stack
```
Layer 5: AG-UI Glass Box Interface (real-time visualization)
Layer 4: LLM Council (Multi-agent debate & consensus)
Layer 3: SPARC Governance (5-gate validation before deployment)
Layer 2: Cognitive Memory (agentdb + HNSW vector search)
Layer 1: QUIC Transport (agentic-flow for agent communication)
```

### Core Agent Types
- **Architect**: Strategic planning, cognitive decomposition into Product Requirements Prompts (PRPs)
- **Guardian**: Safety gatekeeper; simulates changes in digital twin before deployment using Lyapunov stability analysis
- **Sentinel**: Real-time system observer; circuit breaker for chaos detection; implements RIV (Robust Isolation Verifier) pattern
- **Cluster Orchestrator**: Coordinates multi-cell optimization (Phase 2 focus)
- **Self-Healing Agent**: Anomaly detection and remediation (src/smo/fm-handler.ts)
- **Self-Learning Agent**: Continuous optimization via Q-Learning (src/learning/self-learner.ts)

### Key Technologies
- **AI Providers**: Claude Code PRO MAX, Google Gemini 2.0, E2B Sandboxes, OpenRouter
- **Orchestration**: `claude-flow@alpha` (multi-agent CRDT memory)
- **Transport**: `agentic-flow@alpha` (QUIC with 0-RTT)
- **Memory**: `agentdb@alpha` (PostgreSQL/WASM persistence), `ruvector` (HNSW spatial indexing)
- **Consensus**: `QuDAG` (quantum-resistant DAG-based ledger)
- **Code Generation**: SPARC methodology (Specification → Pseudocode → Architecture → Refinement → Completion)

### Multi-Provider Configuration
TITAN supports multiple AI providers working in concert:

**Configuration Files:**
- `config/.env` - API keys and runtime configuration
- `config/agentic-flow.config.ts` - Multi-provider AI settings
- `config/devpod.yaml` - DevPod workspace configuration
- `config/docker-compose.devpod.yml` - Docker Compose setup

**AI Strategies:**
Set in `config/.env` via `AGENTIC_FLOW_STRATEGY`:
- `consensus` - Both Claude & Gemini must agree (95%+ confidence, recommended for production)
- `claude_primary` - Claude leads with Gemini validation (85-90% confidence, fast)
- `gemini_primary` - Gemini leads with multimodal analysis (80-85% confidence)
- `parallel` - Both run independently (70-80% confidence, fastest)

**Setup Documentation:**
- Quick Start: `docs/QUICK-START.md` (5-minute setup)
- Full Guide: `docs/MULTI-PROVIDER-SETUP.md` (comprehensive documentation)
- Setup Reference: `README-SETUP.md` (overview and scripts)

## Development Commands

### Build & Compilation
```bash
npm run build              # Compile TypeScript to dist/
npm run build:watch       # Watch mode compilation
```

### Testing
```bash
npm test                  # Run all tests (Vitest)
npm run test:watch       # Watch mode testing
npm run coverage         # Generate coverage report (target: 80%)
npm run benchmark        # Performance benchmarks
npm run test:safety      # Safety-specific tests (src/hooks/safety.test.ts)
```

### Running the System

#### Multi-Provider AI Setup (New in v7.0)
```bash
# First-time setup (see docs/QUICK-START.md)
cp config/.env.template config/.env  # Configure API keys
npm run test:integration             # Test all integrations

# Start TITAN (choose one runtime mode)
npm run start:local                  # Local Mac Silicon (development)
npm run start:devpod                 # DevPod with Docker (production)
npm run docker:up                    # Docker Compose (alternative)

# AI Integration
npm run ui:integration               # Test AI integration with live data
EXAMPLE=3 npm run ui:integration     # Test consensus mode specifically
```

#### Standard Operations
```bash
npm start                            # Start main orchestrator
npm run orchestrate                  # Run claude-flow orchestration engine
npm run db:status                    # Check AgentDB memory status
npm run db:train                     # Train AgentDB models
npm run agui:start                   # Start AG-UI server
npm run agui:frontend                # Open AG-UI in browser
npm run sentinel:monitor             # Start Sentinel chaos detection
npm run sparc:validate               # Validate SPARC methodology compliance
npm run agents:list                  # List available agents
npm run hive:status                  # Check hive-mind status
npm run swarm:spawn                  # Spawn swarm with intent (e.g., --intent="optimize_coverage")
```

#### Docker & DevPod Management
```bash
npm run docker:up                    # Start all Docker containers
npm run docker:down                  # Stop all containers
npm run docker:logs                  # View container logs
npm run env:validate                 # Validate environment configuration

# DevPod commands (requires devpod CLI)
devpod up titan-ran                  # Create/start workspace
devpod stop titan-ran                # Stop workspace
devpod ssh titan-ran                 # SSH into workspace
devpod delete titan-ran              # Delete workspace
```

### Single Test Execution
```bash
npm test -- tests/smo.test.ts                    # Run specific test file
npm test -- tests/gnn.test.ts --reporter=verbose # Run with verbose output
```

## Source Structure

```
src/
├── agents/                    # Agent implementations (Architect, Guardian, Sentinel, etc.)
│   ├── base-agent.js
│   ├── architect/
│   ├── guardian/
│   ├── sentinel/
│   └── cluster_orchestrator/
├── smo/                       # Service Management & Orchestration
│   ├── pm-collector.ts        # Performance Management data pipelines
│   ├── fm-handler.ts          # Fault Management & anomaly detection
│   └── index.ts
├── learning/                  # Self-learning agents
│   └── self-learner.ts        # Q-Learning implementation
├── knowledge/                 # 3GPP spec indexing & vector search
├── gnn/                       # Graph Neural Networks for parameter optimization
├── interference/              # GNN-based interference modeling
├── memory/                    # Cognitive memory schema & vector indexing
│   ├── schema.ts
│   └── vector-index.ts
├── council/                   # LLM Council consensus
│   ├── orchestrator.ts
│   ├── chairman.ts            # Multi-LLM consensus synthesis
│   ├── debate-protocol.ts
│   └── router.ts
├── consensus/                 # Consensus mechanisms
│   ├── voting.js
│   └── qudag.js               # Quantum-resistant ledger
├── racs/                      # Remote Agentic Coding System
│   ├── orchestrator.js
│   ├── swarm_manager.js
│   └── slicing.js
├── ui/                        # UI components (dashboard, demo)
├── agui/                      # AG-UI Glass Box interface server
├── security/                  # Quantum-resistant signatures & psycho-symbolic analysis
├── hooks/                     # Safety hooks & pre-commit validation
├── sparc/                     # SPARC methodology validation & simulation
├── transport/                 # QUIC transport layer
├── cognitive/                 # AgentDB & Ruvector clients
│   ├── agentdb-client.js
│   └── ruvector-engine.js
├── enm/                       # Element Management (ENM) integration
├── governance/                # Network governance policies
├── ml/                        # ML utilities & model management
└── index.js                   # Main orchestrator entry point

tests/
├── smo.test.ts                # SMO (PM/FM) functionality
├── gnn.test.ts                # GNN accuracy & interference modeling
├── knowledge.test.ts          # Vector search & 3GPP indexing
├── self-learning.test.ts      # Q-Learning agent tests
├── integration.test.js        # Multi-agent integration
├── phase2.test.js             # Multi-cell swarm tests
├── phase3_test.js             # Network-wide slicing tests
├── phase4_test.js             # Production autonomy tests
├── enm-integration.test.ts     # ENM integration tests
├── ml.test.ts                 # ML model tests
└── benchmark.test.js          # Performance benchmarks
```

## 3-ROP Closed-Loop Governance

Parameter changes are monitored across three **Roll-Out Periods (ROP)**:
- **ROP 1**: Observe and collect PM counters from baseline
- **ROP 2**: Compare results to predictions (confidence interval check)
- **ROP 3**: Confirm success or trigger automatic rollback

This ensures safe, validated deployments without manual intervention.

## Key Patterns

### Agent Communication Loop
1. **Action**: Agent executes a task (proposes parameter change)
2. **Observation**: Sentinel/Guardian observes effects via PM counters
3. **Reflexion**: Agent critiques performance, logs to AgentDB for future learning

### Safety-First SPARC Gate Validation
All code generated by agents passes through 5 gates:
1. Specification clarity
2. Pseudocode logic flow
3. Architecture component interaction
4. Refinement & optimization
5. Completion & deployment simulation

### Digital Twin Pre-Deployment
Guardian Agent simulates all changes in E2B isolated environments before live deployment. Uses Lyapunov exponent analysis to detect stability issues early.

## Performance Targets

| Metric | Target |
|:-------|:-------|
| Vector Search Latency | <10ms (p95) |
| LLM Council Consensus | <5s |
| Safety Check Execution | <100ms |
| Test Coverage | 80% |
| UL SINR Improvement | +26% |
| System Uptime | >=99.9% |
| URLLC Packet Loss Rate | <=10^-5 |

## Security & Compliance

- **Quantum-Resistant**: ML-DSA-87 signatures, ML-KEM-768 encryption
- **Immutable Audit**: QuDAG ledger for all parameter changes
- **3GPP Compliance**: TS 28.552, TS 28.310, TS 23.288
- **Sandboxed Execution**: All simulations in E2B isolated environments

## Important Notes for Development

### TypeScript Configuration
- Target: ES2020, Module: ES2020
- Output: `dist/` directory
- Source: `src/` directory
- Test files excluded from compilation
- Strict mode enabled

### Test Framework
- **Test Runner**: Vitest
- **Coverage Provider**: v8
- **Coverage Reports**: text, json, html (in coverage/)
- **Test Location**: `tests/` or colocated `.test.ts` files

### Environment Setup
- **Node.js**: >= 18.0.0 (20.x recommended for Mac Silicon)
- **Package Manager**: npm
- **Private Registry**: Requires access to ruvnet registry for `claude-flow`, `agentdb`, `agentic-flow`
- **API Keys Required**:
  - `ANTHROPIC_API_KEY` - Claude Code PRO MAX subscription
  - `GOOGLE_AI_API_KEY` - Google AI Studio (free tier available)
  - `E2B_API_KEY` - E2B sandboxes (100 hours/month free)
  - `OPENROUTER_API_KEY` - OpenRouter (optional, pay-per-use)
- **Optional Tools**:
  - Docker Desktop 4.25+ (for DevPod mode)
  - DevPod CLI (for containerized development)

**First-time Setup:**
1. Copy `config/.env.template` to `config/.env`
2. Add your API keys to `config/.env`
3. Run `npm run test:integration` to verify all connections
4. Choose runtime mode: `start:local`, `start:devpod`, or `docker:up`

### Claude Code Features
TITAN uses Claude Code hooks and commands in `.claude/`:
- Agent-specific prompts and validation rules
- Hive-mind coordination across agents
- Safety hooks for pre-commit validation
- Monitoring and logging integrations
- Swarm spawning and fleet management

### Roadmap Context
- **Phase 1** (COMPLETED): Single-cell agents, infrastructure setup
- **Phase 2** (CURRENT): Multi-cell swarm, cluster coordination, GNN interference modeling
- **Phase 3** (UPCOMING): Network-wide (50+ cells), slicing, QuDAG integration
- **Phase 4+** (FUTURE): Production autonomy (100+ cells), 6G readiness, full self-evolution

### Debugging Tips
- Check AgentDB memory status: `npm run db:status`
- Monitor Sentinel chaos detection: `npm run sentinel:monitor`
- View AG-UI dashboard for real-time agent behavior: `npm run agui:frontend`
- Run specific test with verbose output: `npm test -- <file> --reporter=verbose`
- Test API integrations: `npm run test:integration`
- Test AI providers: `npm run ui:integration`
- Verify environment: `npm run env:validate`
- View Docker logs: `npm run docker:logs`
- SSH into DevPod: `devpod ssh titan-ran`

### Runtime Modes

**Local Mode (Development):**
```bash
npm run start:local
# Services run directly on Mac Silicon
# Fastest for development
# Full access to system resources
```

**DevPod Mode (Production):**
```bash
npm run start:devpod
# Isolated Docker workspace
# Consistent environment
# Includes PostgreSQL + Redis
# VS Code integration
```

**Docker Compose Mode (Alternative):**
```bash
npm run docker:up
# All services in containers
# Easy management
# Production-like environment
```
