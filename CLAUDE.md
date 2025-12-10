# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A unified AI agent development platform consolidating **160 projects** from ricable (50), ruvnet (105), and spectredve (5) into a cohesive monorepo covering June-December 2024. Key systems include:

- **claude-flow** - Multi-agent swarm orchestration with MCP protocol
- **agentic-flow** - LLM provider switching (Claude, OpenAI, local models)
- **ruvector** - Distributed vector database with GNN learning (150x faster search)
- **agentdb** - Reinforcement learning with swarm intelligence
- **flow-nexus** - Competitive agentic platform with 70+ MCP tools
- **ARCADIA** - AI-driven game engine (Rust/WASM)
- **Synaptic-Mesh** - Self-evolving P2P neural fabric
- **nanochat** - Full-stack ChatGPT alternative

## Repository Structure

```
sources/
├── ricable/      # 50 repos: agentdb, ruvector, claude-flow, ultimate-ran, skills, etc.
├── ruvnet/       # 105 repos (see full list below)
└── spectredve/   # 5 repos: nanochat, FANN, Synaptic-Mesh, claude-fann, dotfiles
docs/sparc-phases/  # SPARC methodology documentation
prompts/            # Multi-step phased prompts (discovery → deployment)
```

### ruvnet Sources (105 repositories)

**Orchestration**: claude-flow, agentic-flow, flow-nexus, sparc, sparc-ide, roomodes, Roo-Code-Docs

**AI/ML**: ruvector, dspy.ts, Synaptic-Mesh, ARCADIA, QuDAG, ruv-FANN, SAFLA, SynthLang, omnipotent, onnx-agent, llamastack, strawberry-phi

**Agentic Apps**: agentic-artifacts, agentic-dashboard, agentic-devops, agentic-difusion, agentic-employment, agentic-flows, agentic-gradio, agentic-music, agentic-preview, agentic-robotics, agentic-scraper, agentic-search, agentic-security, agentic-voice, agenticsjs, agentXNG, agileagents

**Infrastructure**: federated-mcp, vsc-remote-mcp, dynamo-mcp, open-claude-code, ruv-code, rUv-dev, ruv-engineer, ruv.io

**Domain**: musicai, vibecast, tribe-knowledgegraph, swarm-world, genesis, alienator, dreamfactory, hacker-league, tariffic, inflight, nova, voicebot, phone-agent, image-agent

**Quantum**: quantum-magnetic-navigation, Quantum-Virtual-Machine, quantum_cryptocurrency, sublinear-time-solver

**Web/UI**: ai-browse, auto-browser, infinity-ui, drupal, drupaljs, demo-proxy-app, phoenix, vibing

**Data**: contextual-retrevial, swirl-search, local-logic, hft, AiCodeCalc, reflective-engineer

**Elections**: Electo1, electo1-js, FACT

**Utilities**: Agent-Name-Service, agentics-meetup, anthropic-quickstarts, chatgpt-dev-mode, claude-test, code-mesh, codespaces-jupyter, codex-one, daa, deco, fireflies-webook, GenAI-Superstream, gpts, hello_world_agent, midstream, open-agentics, runme, ruvnet, supabase-authentication, symbolic-scribe, test-react-lib, ultrasonic, wifi-densepose, yyz-agentics-june, aido, aihl

## Build Commands

### TypeScript/JavaScript Projects (sources/ricable/agentdb, sources/ruvnet/claude-flow)
```bash
npm run build          # Compile TypeScript
npm run test           # Run Jest tests
npm run lint           # ESLint
npm run typecheck      # Type checking
```

### Rust Projects (sources/ruvnet/ARCADIA)
```bash
cargo build --release  # Optimized build
cargo test             # Run tests
cargo bench            # Run benchmarks
make build             # Build native + WASM
wasm-pack build        # Web/Node WASM targets
```

### Python Projects (sources/spectredve/nanochat)
```bash
uv sync                # Install dependencies (use astral uv)
uv run pytest tests/   # Run tests
bash speedrun.sh       # Full training pipeline (~4 hours)
uv run python -m scripts.chat_web  # Launch web UI
```

### SPARC Workflow Commands
```bash
npx claude-flow sparc modes              # List available modes
npx claude-flow sparc run <mode> "<task>" # Execute mode
npx claude-flow sparc tdd "<feature>"    # Run TDD workflow
npx claude-flow sparc batch <modes> "<task>"  # Parallel execution
```

## Testing

### Running Tests
```bash
# TypeScript projects
npm run test                    # All tests
npm run test -- --watch         # Watch mode
npm run test -- path/to/file    # Single file

# Rust projects
cargo test                      # All tests
cargo test <test_name>          # Single test
wasm-pack test --headless --firefox  # WASM tests

# Python projects
uv run pytest tests/            # All tests
uv run pytest tests/test_foo.py -v  # Single file verbose
```

### Jest Configurations (agentdb)
- `jest.config.js` - Default TypeScript
- `jest.sparc.config.js` - SPARC methodology tests
- `jest.rtb.config.js` - RTB tests
- `jest.phase5.config.js` - Phase 5 tests

## Architecture

### Multi-Language Stack
- **TypeScript**: Orchestration, coordination, web services (claude-flow, agentic-flow)
- **Rust**: Performance-critical, WASM, neural networks (ARCADIA, FANN, rustbpe)
- **Python**: ML/AI training, evaluation, scripting (nanochat)

### SPARC Methodology
All development follows SPARC phases:
1. **Specification** - Requirements analysis
2. **Pseudocode** - Algorithm design
3. **Architecture** - System design
4. **Refinement** - TDD implementation
5. **Completion** - Integration and deployment

### Agent Execution Pattern
Claude Code's Task tool is primary for spawning agents. MCP tools (claude-flow, ruv-swarm, flow-nexus) coordinate but don't execute.

```bash
# Setup MCP servers
claude mcp add claude-flow npx claude-flow@alpha mcp start
claude mcp add ruv-swarm npx ruv-swarm mcp start
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

### Key Integration Points
- **MCP Protocol**: Standard communication between services
- **AgentDB Memory**: Cross-agent learning (<1ms QUIC sync)
- **Hooks System**: Pre/post operation coordination via `npx claude-flow@alpha hooks`

## File Organization

Never save working files to root. Use appropriate subdirectories:
- `/src` - Source code
- `/tests` - Test files
- `/docs` - Documentation
- `/config` - Configuration
- `/scripts` - Utilities

## Package Managers

- **npm** - TypeScript/JavaScript projects
- **cargo** - Rust projects
- **uv** - Python projects (per CLAUDE.md parent instructions)

## Published Packages

| Package | Version | Registry |
|---------|---------|----------|
| claude-flow | 2.7.47 | npm |
| agentic-flow | 1.10.2 | npm |
| ruvector | 0.1.33 | npm |
| dspy.ts | 2.1.1 | npm |
| agentdb | 1.6.1 | npm |
| ruvector-core | 0.1.22 | crates.io |

## Project-Specific Documentation

Major subprojects have their own CLAUDE.md with detailed instructions:
- `sources/ricable/agentdb/CLAUDE.md` - Ericsson RAN optimization, cognitive consciousness
- `sources/ruvnet/ARCADIA/CLAUDE.md` - Game engine, agent execution patterns
- `sources/ruvnet/claude-flow/CLAUDE.md` - Swarm orchestration

## Concurrent Execution Rules

Batch related operations in single messages:
- Spawn all agents together via Task tool
- Batch all TodoWrite items together
- Batch all file operations together
- Batch all bash commands together
