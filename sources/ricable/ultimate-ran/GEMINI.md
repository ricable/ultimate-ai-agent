# TITAN: Neuro-Symbolic RAN Platform (Ericsson Gen 7.0)

## Project Overview
TITAN is an autonomous 5G/6G Radio Access Network (RAN) optimization platform. It leverages a "Cognitive Mesh" of AI agents to optimize network parameters (like SINR, Spectral Efficiency) using a Neuro-Symbolic approach.
- **Core Concept:** Combines LLM reasoning (Claude Code PRO MAX, Google Gemini 2.0) with symbolic safety checks (Lyapunov analysis, Digital Twins) and Quantum-resistant ledgers (QuDAG).
- **Goal:** Autonomous network optimization with 5-gate safety validation (SPARC).

## Architecture
The system follows a 5-Layer Stack:
1.  **Layer 1: QUIC Transport** (`agentic-flow`): High-speed, 0-RTT agent communication.
2.  **Layer 2: Cognitive Memory** (`agentdb` + `ruvector`): Persistence and HNSW vector search for 3GPP specs and past episodes.
3.  **Layer 3: SPARC Governance**: 5-step validation (Spec -> Pseudocode -> Architecture -> Refinement -> Completion).
4.  **Layer 4: LLM Council**: Multi-agent debate and consensus engine (Claude + Gemini).
5.  **Layer 5: AG-UI**: "Glass Box" real-time visualization interface.

### Multi-Provider AI Strategy
TITAN uses multiple models concurrently:
- **Claude Code PRO MAX:** Primary reasoning & coding.
- **Google Gemini 2.0:** Multimodal analysis & anomaly detection.
- **E2B Sandboxes:** Isolated execution for safety checks.

**Consensus Modes (`AGENTIC_FLOW_STRATEGY`):**
- `consensus`: Claude & Gemini must agree (Production).
- `claude_primary`: Claude leads, Gemini validates (Speed/Reliability balance).
- `gemini_primary`: Gemini leads (Visual/Multimodal focus).
- `parallel`: Independent execution (Max speed).

## Development Conventions & Mandates
**CRITICAL:** This project enforces strict agentic workflows.

1.  **SPARC Methodology:** All features must follow the Specification -> Completion pipeline.
2.  **Concurrent Execution:**
    - **1 Message = All Operations:** Batch all `Task`, `TodoWrite`, `Write`, and `Bash` commands in a single turn.
    - **Task Tool:** Use `Task(description, instruction, agent_type)` for parallel agent execution.
    - **MCP Tools:** Use ONLY for high-level coordination (`swarm_init`), not for doing the actual work.
3.  **File Management:**
    - **NEVER** save files to the root directory (except config/meta files).
    - Use `src/` for code, `tests/` for tests, `docs/` for documentation.
4.  **Testing:** TDD is mandatory. Write tests (`tests/`) before implementation.

## Build, Run, and Test

### Core Commands
- **Install:** `npm install`
- **Build:** `npm run build` (Compiles TypeScript to `dist/`)
- **Start (Local):** `npm run start:local` (Mac Silicon optimized)
- **Start (DevPod):** `npm run start:devpod` (Dockerized env)
- **Orchestrate:** `npm run orchestrate` (Runs `claude-flow`)

### Testing (Vitest)
- **Run All:** `npm test`
- **Integration:** `npm run test:integration` (Verifies API/AI connections)
- **Safety:** `npm run test:safety` (Hooks & guardrails)
- **Coverage:** `npm run coverage` (Target: >80%)
- **Benchmark:** `npm run benchmark`

## Key Directories
- `src/agents/`: specialized agents (Architect, Guardian, Sentinel).
- `src/council/`: Multi-LLM consensus logic.
- `src/knowledge/`: Vector search & 3GPP specification indexing.
- `src/smo/`: Service Management & Orchestration (PM/FM pipelines).
- `src/gnn/`: Graph Neural Networks for parameter optimization.
- `config/`: Configuration files (`.env`, `agentic-flow.config.ts`).
- `.claude/`: Claude Code specific hooks, commands, and agent definitions.

## Agent Types
- **Architect:** Planning & Decomposition (PRPs).
- **Guardian:** Safety & Digital Twin simulation.
- **Sentinel:** Real-time monitoring & Chaos detection.
- **Cluster Orchestrator:** Multi-cell coordination.
- **Self-Learning:** Q-Learning optimization.

## 3-ROP Governance
Changes are deployed in 3 Roll-Out Periods:
1.  **ROP 1:** Observation.
2.  **ROP 2:** Prediction vs Actuals check.
3.  **ROP 3:** Confirmation or Rollback.
