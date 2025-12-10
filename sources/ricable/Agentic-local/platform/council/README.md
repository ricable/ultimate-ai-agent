# MLX Deep Council

Distributed Multi-Model Consensus System for Apple Silicon, inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council).

## Overview

MLX Deep Council implements a three-stage consensus mechanism across multiple LLM models running on distributed Mac machines using Apple's [MLX framework](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).

### The Three Stages

1. **Stage 1: Individual Opinions**
   - Query is sent to all council members (LLMs on different Macs)
   - Each model generates its independent response
   - Responses are collected with latency and confidence metrics

2. **Stage 2: Peer Review**
   - Each model receives anonymized responses from other models
   - Models evaluate and rank each other's responses
   - Scores are aggregated using weighted voting

3. **Stage 3: Chairman Synthesis**
   - The designated "chairman" model receives all data
   - Synthesizes a final response based on peer review scores
   - Resolves conflicts and combines best insights

## Quick Start

### Single Mac (Development)

```bash
# Query the council (uses multiple model instances locally)
npx ts-node platform/index.ts council "What is the best sorting algorithm for this use case?"
```

### Distributed Cluster (Production)

```bash
# 1. Setup the council
./scripts/setup-mlx-council.sh --distributed --hosts mac1,mac2,mac3

# 2. Launch the council
npx ts-node platform/council/council-launcher.ts launch --config council.json

# 3. Interactive mode or single queries
Council> What are the tradeoffs between SQL and NoSQL databases?
```

## Configuration

Create a `council.json` file:

```json
{
  "name": "My Mac Cluster Council",
  "nodes": [
    {
      "hostname": "mac-studio",
      "ip": "192.168.1.101",
      "port": 8080,
      "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
      "gpuMemory": 192,
      "chip": "M2 Ultra"
    },
    {
      "hostname": "macbook-pro",
      "ip": "192.168.1.102",
      "port": 8080,
      "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "gpuMemory": 36,
      "chip": "M3 Pro"
    }
  ],
  "defaultModel": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "chairmanModel": "mlx-community/Qwen2.5-32B-Instruct-4bit",
  "backend": "ring",
  "votingStrategy": "weighted"
}
```

## API Usage

### TypeScript/JavaScript

```typescript
import {
  MLXDeepCouncil,
  createLocalCouncil,
  createDistributedCouncil,
  quickCouncil
} from '@edge-ai/hyperscale-platform';

// Quick query (auto-creates local council)
const result = await quickCouncil("Explain quantum computing");
console.log(result.finalResponse);
console.log(`Consensus: ${result.consensusReached}`);

// Custom distributed council
const council = createDistributedCouncil({
  name: 'Production Council',
  nodes: [
    { hostname: 'mac1', ip: '192.168.1.101', model: 'qwen2.5-32b', gpuMemory: 192, chip: 'M2 Ultra' },
    { hostname: 'mac2', ip: '192.168.1.102', model: 'llama-3.2-3b', gpuMemory: 36, chip: 'M3 Pro' },
    { hostname: 'mac3', ip: '192.168.1.103', model: 'mistral-7b', gpuMemory: 32, chip: 'M4' },
  ],
});

await council.initialize();

const session = await council.query({
  content: "What database should I use for a high-write workload?",
  requireConsensus: true,
  minAgreement: 0.7,
});

// Access individual responses
for (const response of session.individualResponses) {
  console.log(`${response.anonymousId}: ${response.content.slice(0, 100)}...`);
  console.log(`  Score: ${session.aggregatedScores.get(response.anonymousId)}/10`);
}

// Chairman's synthesis
console.log(session.chairmanSynthesis?.finalResponse);
```

## MLX Distributed Backends

### Ring Topology (Recommended for Thunderbolt)

Uses a ring network where each node communicates with neighbors. Optimal for Thunderbolt-connected Macs.

```bash
# Auto-detect Thunderbolt topology
mlx.distributed_config --verbose --hosts mac1,mac2,mac3 --auto-setup
```

### MPI Backend (Network)

Uses MPI for communication over standard network connections.

```bash
# Install MPI
conda install conda-forge::openmpi

# Launch with MPI
mlx.launch --backend mpi -n 3 python/mlx_council_worker.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLX Deep Council                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Mac 1     │    │   Mac 2     │    │   Mac 3     │         │
│  │  (Chairman) │◄──►│  (Member)   │◄──►│  (Member)   │         │
│  │             │    │             │    │             │         │
│  │ Qwen2.5-32B │    │ Llama-3.2-3B│    │ Mistral-7B  │         │
│  │ M2 Ultra    │    │ M3 Pro      │    │ M4          │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        ▲                  ▲                  ▲                  │
│        │                  │                  │                  │
│        └──────────────────┼──────────────────┘                  │
│                           │                                      │
│                    Ring Topology                                 │
│                 (MLX Distributed)                               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Individual Responses    ───────────────────────────►  │
│  Stage 2: Peer Review             ◄─────────────────────────►   │
│  Stage 3: Chairman Synthesis      ◄───────────────────────────  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

```bash
# Launch council in interactive mode
council-launcher launch --config council.json

# Single query
council-launcher query "What is the meaning of life?"

# Configure new council
council-launcher configure --hosts mac1,mac2,mac3 --output my-council.json

# Check status
council-launcher status

# Stop all workers
council-launcher stop
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX >= 0.20.0
- mlx-lm >= 0.18.0
- For distributed: SSH key-based authentication between machines

## References

- [Karpathy's LLM Council](https://github.com/karpathy/llm-council) - Original implementation
- [MLX Distributed Documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [LLM Council: When Ensemble Learning Meets LLMs](https://medium.com/@meshuggah22/andrej-karpathys-llm-council-when-ensemble-learning-meets-large-language-models-e3312fd02064)

## License

MIT
