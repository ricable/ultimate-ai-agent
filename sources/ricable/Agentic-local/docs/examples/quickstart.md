# Quick Start Guide

## Prerequisites

Before starting, ensure you have:

- **macOS** with Apple Silicon (M1/M2/M3) - or Linux
- **Node.js** 18+ and npm 9+
- **Docker Desktop** installed and running
- **16GB+ RAM** (32GB+ recommended for 32B model)

## Installation (5 minutes)

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd Agentic-local

# Install Node.js dependencies
npm install

# Copy environment template
cp .env.example .env
```

### 2. Set Up WasmEdge with MLX Support

```bash
# This will take 15-30 minutes
./scripts/setup-wasmedge-mlx.sh

# Restart your terminal after installation
source ~/.zshrc  # or ~/.bashrc
```

### 3. Install LlamaEdge

```bash
./scripts/setup-llamaedge.sh
```

### 4. Download Qwen 2.5 Coder Model

```bash
./scripts/download-qwen-coder.sh
# Select option based on your RAM:
# 1) 7B for 16GB RAM
# 2) 14B for 24GB RAM
# 3) 32B for 48GB+ RAM
```

### 5. Set Up GaiaNet Node (Optional for Monetization)

```bash
./scripts/setup-gaianet.sh
```

## Configuration

Edit your `.env` file:

```bash
# Use local GaiaNet node
LLM_PROVIDER=local
GAIANET_ENDPOINT=http://localhost:8080/v1
GAIANET_MODEL=Qwen2.5-Coder-32B-Instruct

# Enable Agent Booster for 352x speedup
ENABLE_AGENT_BOOSTER=true

# Sandbox settings
SANDBOX_TYPE=docker
SANDBOX_MEMORY_LIMIT=2g
SANDBOX_CPU_LIMIT=2
```

## Start Your Infrastructure

### Option A: Using LlamaEdge Directly (Local Only)

```bash
# Start LlamaEdge inference server
llamaedge

# In another terminal, run your agent
npm run start:agent
```

### Option B: Using GaiaNet (Local + Monetization)

```bash
# Start GaiaNet node (includes inference + network participation)
gaianet start

# Verify it's running
gaianet info

# In another terminal, run your agent
npm run start:agent
```

## Your First Agent

Create `my-first-agent.js`:

```javascript
import 'dotenv/config';
import { AgenticFlow } from 'agentic-flow';

const agent = new AgenticFlow({
  provider: 'local',
  baseURL: 'http://localhost:8080/v1',
  model: 'Qwen2.5-Coder-32B-Instruct'
});

const result = await agent.run(`
  Create a JavaScript function that:
  1. Takes an array of numbers
  2. Returns the sum, average, min, and max
  3. Include example usage
`);

console.log('Generated Code:\n', result.code);
```

Run it:

```bash
node my-first-agent.js
```

**Expected output:**
```
Generated Code:
 function analyzeNumbers(numbers) {
   const sum = numbers.reduce((a, b) => a + b, 0);
   const average = sum / numbers.length;
   const min = Math.min(...numbers);
   const max = Math.max(...numbers);

   return { sum, average, min, max };
 }

 // Example usage:
 console.log(analyzeNumbers([1, 2, 3, 4, 5]));
 // Output: { sum: 15, average: 3, min: 1, max: 5 }
```

## Run Example Workflows

### Basic Agent Examples

```bash
node src/orchestration/basic-agent.js
```

This demonstrates:
- Simple code generation
- Iterative development with error correction
- Multi-step reasoning

### Swarm Intelligence Examples

```bash
node src/orchestration/swarm-agent.js
```

This demonstrates:
- Hierarchical swarm (project management)
- Mesh network (parallel processing)
- Star network (code review)

## Test the Sandbox

Verify Docker sandbox security:

```bash
npm run sandbox:test
```

This runs 7 security tests including:
- Basic execution
- Network isolation
- Resource limits
- File system protection

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Mac Silicon                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Orchestration Layer (Node.js)                      │   │
│  │  - agentic-flow / claude-flow                       │   │
│  │  - Agent Booster (WASM, 352x faster)                │   │
│  │  - ReasoningBank (memory & learning)                │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼────────────────────────────────────┐   │
│  │  Inference Layer (WasmEdge + MLX)                   │   │
│  │  - GaiaNet Node or LlamaEdge                        │   │
│  │  - Qwen 2.5 Coder (7B/14B/32B)                      │   │
│  │  - GPU Accelerated (Apple Silicon)                  │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                          │
│  ┌────────────────▼────────────────────────────────────┐   │
│  │  Execution Layer (Docker Sandbox)                   │   │
│  │  - Isolated containers                              │   │
│  │  - Network disabled                                 │   │
│  │  - Resource limits enforced                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                            │
         │ Your requests              │ External requests
         │ (FREE)                     │ (EARN CRYPTO)
         │                            │
         ▼                            ▼
    Your apps                  GaiaNet Network
```

## Workflow Examples

### Example 1: Build a REST API

```javascript
import { AgenticFlow } from 'agentic-flow';
import { DockerSandbox } from './src/sandbox/docker-sandbox.js';

const sandbox = new DockerSandbox();

const agent = new AgenticFlow({
  provider: 'local',
  baseURL: 'http://localhost:8080/v1',
  model: 'Qwen2.5-Coder-32B-Instruct',

  onCodeGenerated: async (code) => {
    return await sandbox.execute(code, 'javascript');
  }
});

const result = await agent.run(`
  Create an Express.js REST API with:
  - GET /users - list all users
  - POST /users - create user
  - GET /users/:id - get user by ID
  - PUT /users/:id - update user
  - DELETE /users/:id - delete user

  Use in-memory storage. Include input validation.
`);

console.log('API Code:', result.code);
console.log('Tests:', result.tests);
```

### Example 2: Data Analysis Pipeline

```javascript
const result = await agent.run(`
  Analyze this sales data:
  [
    { date: '2024-01-01', product: 'Widget', quantity: 10, price: 29.99 },
    { date: '2024-01-02', product: 'Gadget', quantity: 5, price: 49.99 },
    { date: '2024-01-03', product: 'Widget', quantity: 7, price: 29.99 }
  ]

  Calculate:
  1. Total revenue
  2. Revenue by product
  3. Best-selling product
  4. Average order value

  Return results as JSON.
`, { enableReasoning: true });
```

### Example 3: Multi-Agent Software Project

```javascript
import { SwarmOrchestrator } from 'ruv-swarm';

const swarm = new SwarmOrchestrator({
  topology: 'hierarchical',
  queen: projectManagerAgent,
  drones: [backendAgent, frontendAgent, qaAgent, devopsAgent]
});

const project = await swarm.execute({
  task: 'Build a todo app with React frontend and Node.js backend',
  requirements: [
    'User authentication',
    'CRUD operations for todos',
    'RESTful API',
    'Unit tests',
    'Docker deployment'
  ]
});
```

## Monitoring

### Check GaiaNet Node Status

```bash
# Node info
gaianet info

# View logs
gaianet log

# Monitor earnings (if registered)
# Visit: https://www.gaianet.ai/dashboard
```

### Monitor Resource Usage

```bash
# GPU/CPU usage
top -o cpu

# Memory pressure
memory_pressure

# Docker containers
docker stats
```

## Troubleshooting

### Agent not connecting to local node

```bash
# Verify node is running
curl http://localhost:8080/v1/models

# Check logs
gaianet log | tail -20

# Restart node
gaianet stop && gaianet start
```

### Sandbox execution failing

```bash
# Verify Docker is running
docker ps

# Test sandbox
npm run sandbox:test

# Check Docker resources in Docker Desktop settings
# Recommended: 4GB RAM, 2 CPUs minimum
```

### Out of memory errors

```bash
# Reduce context size in .env
CONTEXT_WINDOW=16384

# Or use a smaller model
./scripts/download-qwen-coder.sh
# Select option 1 (7B model)
```

## Next Steps

1. **Read the full technical analysis**: `docs/technical-analysis/sovereign-agentic-architectures.md`

2. **Explore monetization**: `docs/setup-guides/gaianet-monetization.md`

3. **Review security**: `docs/setup-guides/sandbox-security.md`

4. **Build your own agents**: Check `src/orchestration/` for examples

5. **Join the community**:
   - GaiaNet: https://www.gaianet.ai
   - Ruvnet packages: https://www.npmjs.com/~ruvnet

## Cost Comparison

### Traditional Cloud API Approach

| Model | Cost per 1M tokens | 100 requests/day | Monthly Cost |
|-------|-------------------|------------------|--------------|
| GPT-4 | $30 | ~300K tokens | $270 |
| Claude Opus | $75 | ~300K tokens | $675 |

### Sovereign Stack (This Repo)

| Component | Setup Cost | Monthly Cost | Notes |
|-----------|-----------|--------------|-------|
| WasmEdge | Free | $0 | Open source |
| LlamaEdge | Free | $0 | Open source |
| Qwen 2.5 Coder | Free | $0 | Open weights |
| **Total** | **$0** | **$0** | Unlimited local usage |
| GaiaNet (optional) | Free | **-$50 to -$200** | You EARN money! |

**ROI**: Infinite (negative cost if monetizing with GaiaNet)

## Summary

You now have a complete sovereign AI stack:

✅ **Local inference** on your own hardware
✅ **Zero API costs** for unlimited usage
✅ **GPU acceleration** via MLX on Apple Silicon
✅ **Secure sandboxing** for code execution
✅ **Agent orchestration** with ruvnet ecosystem
✅ **Optional monetization** via GaiaNet
✅ **Enterprise-grade** performance and features

**You own the entire stack. No vendor lock-in. No usage limits. No recurring costs.**

Start building! 🚀
