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

**Environment Manager**: This project uses **mise** for reproducible environments across all agent categories.

## mise Environment Manager

mise is the default environment manager for zgents. All development, builds, and deployments should use mise for consistency.

### Quick Reference

```bash
# Setup environment (first time)
mise trust && mise run setup

# Common tasks
mise run dev              # Start local development
mise run build            # Build all agents
mise run test             # Run all tests
mise run lint             # Lint and format code
mise run k8s-apply        # Deploy to Kubernetes
```

### Agent Category Environments

Each agent category has its own mise.toml with category-specific tools:

| Category | Location | Key Tools |
|----------|----------|-----------|
| WASM | `agents/wasm/mise.toml` | node, bun, rust, wasm-pack, spin |
| Python | `agents/python/mise.toml` | python@3.12, uv, ruff, pytest |
| Rust | `agents/rust/mise.toml` | rust, cargo-component, cross |
| Infra | `agents/infra/mise.toml` | kubectl, helm, k9s, terraform |

### mise Task Categories

**Agent Lifecycle:**
- `mise run agent-create <name> <type>` - Create new agent
- `mise run agent-list` - List agents
- `mise run agent-deploy <path>` - Deploy agent

**Build & Test:**
- `mise run build-wasm` - Build WASM agents
- `mise run build-python` - Build Python agents
- `mise run test-wasm` - Test WASM agents
- `mise run test-python` - Test Python agents

**Kubernetes/kagenti:**
- `mise run k8s-apply` / `mise run k8s-delete`
- `mise run kagenti-install` - Install operator
- `mise run kagenti-platform <file>` - Deploy Platform CR
- `mise run kagenti-status` - Check resources

**Secrets (age encryption):**
- `mise run secrets-init` - Generate age key
- `mise run secrets-set <name>` - Set encrypted secret

**Shims (for Docker/CI/CD):**
- `mise run shims-setup` - Setup shims
- `mise run shims-verify` - Verify shims work

## Ruvnet Ecosystem NPX Tools

The project integrates the full ruvnet npm package ecosystem for vector operations, agent management, and distributed AI:

### Core Tools
```bash
# SPARC Development & Agent Orchestration
npx claude-flow@alpha          # SPARC methodology + agent coordination
npx agentic-flow@alpha         # Multi-agent workflow orchestration
npx agentdb@alpha              # Agent state management database

# Vector Database & Graph Operations
npx ruvector                   # Vector database CLI
npx ruvector-gnn               # Graph neural network tools
npx ruvector-graph             # Graph operations and traversal
npx @ruvector/postgres-cli     # PostgreSQL vector operations
```

### mise Tasks for Ruvnet
```bash
mise run ruvector              # Run ruvector vector database CLI
mise run ruvector-gnn          # Run ruvector-gnn graph neural network tools
mise run ruvector-graph        # Run ruvector-graph for graph operations
mise run ruvector-postgres     # Run @ruvector/postgres-cli
mise run agentdb               # Run agentdb for agent state management
mise run agentic-flow          # Run agentic-flow for multi-agent orchestration
mise run claude-flow-alpha     # Run claude-flow@alpha for SPARC development
```

## Talos Linux / Siderolabs Deployment

This project supports deployment on Talos Linux with Sidero Omni for production-ready, immutable infrastructure.

### Talos Quick Reference
```bash
# Generate Talos configuration
mise run talos-genconfig                    # Generate config with Wasm extensions

# Apply configuration to nodes
mise run talos-apply <node-ip> [config]     # Apply config to a node
mise run talos-bootstrap <control-plane-ip> # Bootstrap the cluster
mise run talos-kubeconfig <control-plane-ip> # Get kubeconfig

# Cluster management
mise run talos-status                       # Check cluster status
mise run talos-upgrade <node-ip> [image]    # Upgrade Talos on node

# System extensions
mise run talos-extensions-list              # List available extensions
mise run talos-extensions-add <image>       # Add extension to config
```

### Sidero Omni Integration
```bash
# Omni cluster management
mise run omni-cluster-create [name] [template]  # Create cluster via Omni
mise run omni-status                            # Check Omni clusters
mise run omni-machine-classes                   # Apply machine classes
mise run omni-kubeconfig [cluster-name]         # Get kubeconfig

# Deploy complete stack
mise run deploy-talos-wasm      # Deploy Talos with Wasm runtimes
mise run deploy-ruvector-stack  # Deploy full ruvector agent stack
```

### Proxmox Infrastructure Provider
```bash
mise run proxmox-provider-setup    # Setup Proxmox provider
mise run proxmox-provider-start    # Start Proxmox provider
```

### Machine Classes (infrastructure/talos/machine-classes/)
- `control-plane.yaml` - Control plane nodes (4 CPU, 8GB RAM, 50GB disk)
- `worker.yaml` - Standard workers (4 CPU, 16GB RAM, 100GB disk)
- `gpu-worker.yaml` - GPU workers with NVIDIA support
- `llamaedge-worker` - LLM inference optimized nodes

## WasmEdge / LlamaEdge Runtimes

### WasmEdge for JavaScript/TypeScript
```bash
mise run wasmedge-install          # Install WasmEdge runtime
mise run wasmedge-run <file.js>    # Run JS with WasmEdge QuickJS
mise run wasmedge-build            # Build WasmEdge JS agent
```

### LlamaEdge for LLM Inference
```bash
mise run llamaedge-install         # Install WasmEdge + GGML plugin
mise run llamaedge-run [model] [port]  # Run LlamaEdge server

# Using the helper script
./scripts/llamaedge/run-llamaedge.sh install        # Install WasmEdge
./scripts/llamaedge/run-llamaedge.sh download llama-2-7b-chat  # Download model
./scripts/llamaedge/run-llamaedge.sh run models/model.gguf     # Start server
./scripts/llamaedge/run-llamaedge.sh k8s-deploy    # Deploy to Kubernetes
```

### Spin / SpinKube
```bash
mise run spin-new [name] [template]  # Create new Spin app
mise run spin-build                  # Build Spin application
mise run spin-up                     # Run locally
mise run spin-deploy [app-path]      # Deploy to SpinKube cluster
```

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ruvector Edge AI Platform                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Sidero Omni   │    │   Kairos/K3s    │    │  Docker/Local   │ │
│  │  (Talos Linux)  │    │ (Immutable OS)  │    │  Development    │ │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘ │
│           │                      │                      │           │
│           ▼                      ▼                      ▼           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Kubernetes / K3s                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │  WasmEdge   │  │    Spin     │  │  Container Runtime  │  │  │
│  │  │  Runtime    │  │   Runtime   │  │       (runc)        │  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │  │
│  └─────────┼────────────────┼────────────────────┼─────────────┘  │
│            │                │                    │                 │
│            ▼                ▼                    ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Agent Workloads                            │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │  │
│  │  │ LlamaEdge │  │  Ruvector │  │  AgentDB  │  │Claude-Flow│ │  │
│  │  │   LLMs    │  │  Vectors  │  │   State   │  │   SPARC   │ │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Using mise in Agent Development

When developing agents, always use mise tasks:

```bash
# WASM Agent Development
cd apps/wasmedge-js
mise run build-wasmedge   # Build for WasmEdge
mise run test             # Run tests

# Python Agent Development
cd apps/fastapi
mise run setup            # Setup venv
mise run run              # Run locally

# Rust Agent Development
cd agents/rust
mise run build-wasm       # Build WASM
mise run cross-build      # Cross-compile
```

### Environment Variables

mise automatically loads environment variables from mise.toml:

```toml
[env]
NODE_ENV = "development"
ZGENTS_ROOT = "{{config_root}}"
_.path = ["{{config_root}}/node_modules/.bin"]
```

### Secrets with age Encryption

```bash
# Initialize (one-time)
mise run secrets-init

# Set secret interactively
mise run secrets-set ANTHROPIC_API_KEY

# Secrets stored encrypted in mise.local.toml
# NEVER commit: config/age-key.txt, config/secrets.toml
```

## SPARC Commands

### Core Commands
- `npx claude-flow@alpha sparc modes` - List available modes
- `npx claude-flow@alpha sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow@alpha sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow@alpha sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow@alpha sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow@alpha sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow@alpha sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

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

- Claude Flow Documentation: https://github.com/ruvnet/claude-flow
- Claude Flow Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

### Siderolabs / Talos Resources
- Talos Linux: https://www.talos.dev/
- Sidero Omni: https://github.com/siderolabs/omni
- Omni Proxmox Provider: https://github.com/siderolabs/omni-infra-provider-proxmox
- Talos Extensions: https://github.com/siderolabs/extensions
- Proxmox Starter Kit: https://github.com/mitchross/sidero-omni-talos-proxmox-starter

### WasmEdge / LlamaEdge Resources
- WasmEdge: https://wasmedge.org/
- LlamaEdge: https://github.com/LlamaEdge/LlamaEdge
- SpinKube: https://spinkube.dev/

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
