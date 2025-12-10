# Distributed Cluster Setup Guide

## Overview

This guide shows how to set up a distributed agent cluster across your heterogeneous hardware: Raspberry Pis, Intel NUCs, and Mac Studios/MacBooks.

```
┌──────────────────────────────────────────────────────────────┐
│                    YOUR HOME CLUSTER                          │
│                                                               │
│  🖥️ MacBook M3 Max (128GB)        [SUPER-COORDINATOR]       │
│  │  Role: Primary orchestrator, heavy inference              │
│  │  Models: Qwen 2.5 32B, experimental 72B                   │
│  │  Earning: $300-500/month (ultra-premium)                  │
│  └──┬────────────────────────────────────────────────────    │
│     │                                                         │
│  🖥️ Mac Studio M1 (64GB)          [COORDINATOR]             │
│  │  Role: Secondary orchestrator, large models               │
│  │  Models: Qwen 2.5 32B                                     │
│  │  Earning: $120-200/month (premium)                        │
│  └──┬────────────────────────────────────────────────────    │
│     │                                                         │
│  🖥️ Intel NUC #1-10 (16GB each)   [WORKERS]                 │
│  │  Role: General compute, storage, moderate inference       │
│  │  Models: Qwen 2.5 7B/14B                                  │
│  │  Earning: $30-60/month each × 10 = $300-600/month         │
│  └──┬────────────────────────────────────────────────────    │
│     │                                                         │
│  🍓 Raspberry Pi (8GB)             [EDGE WORKER]             │
│  │  Role: Edge processing, monitoring, lightweight tasks     │
│  │  Models: Qwen 2.5 1.5B (local only)                       │
│  │  Earning: Not public-facing                               │
│  └────────────────────────────────────────────────────────   │
│                                                               │
│  💰 TOTAL MONTHLY EARNING: $750-1,300 (estimated)            │
│  ⚡ TOTAL COMPUTE: 200+ CPU cores, 800GB RAM                 │
│  🚀 COORDINATION: QUAD/QDAG distributed orchestration        │
└──────────────────────────────────────────────────────────────┘
```

## Architecture

### Hierarchical Topology

```
                    MacBook M3 Max
                  [Super-Coordinator]
                     AgentDB Master
                          │
                          │
           ┌──────────────┼──────────────┐
           │                             │
      Mac Studio M1                  Redis Cluster
    [Coordinator]                    [Coordination]
     AgentDB Replica
           │
           │
    ┌──────┴──────┬──────┬─────┬──────┐
    │             │      │     │      │
  NUC-1       NUC-2   NUC-3  ...   NUC-10
 [Worker]    [Worker][Worker]    [Worker]
    │
    │
Raspberry Pi
[Edge Worker]
```

### Role Distribution

| Hardware | Role | Primary Tasks | Public Facing |
|----------|------|--------------|---------------|
| MacBook M3 Max | Super-Coordinator | Orchestration, heavy inference, research | Yes (premium) |
| Mac Studio M1 | Coordinator | Large model hosting, failover | Yes (premium) |
| Intel NUC (×10) | Worker | General compute, storage, code generation | Yes (standard) |
| Raspberry Pi | Edge Worker | Monitoring, lightweight tasks | No |

## Hardware-Specific Configurations

All hardware configurations are auto-detected and loaded from `config/hardware/`:

### MacBook M3 Max
- **Model**: Qwen 2.5 Coder 32B (MLX 4bit) or 72B experimental
- **Context**: 128k tokens
- **Concurrent Tasks**: 20
- **GaiaNet Tier**: Ultra-premium (3x multiplier)
- **Special Features**: Master AgentDB, failover primary, real-time streaming

### Mac Studio M1
- **Model**: Qwen 2.5 Coder 32B (MLX 4bit/5bit)
- **Context**: 128k tokens
- **Concurrent Tasks**: 12
- **GaiaNet Tier**: Premium (2x multiplier)
- **Special Features**: AgentDB replica, coordinator

### Intel NUC (each)
- **Model**: Qwen 2.5 Coder 7B or 14B (GGUF Q5_K_M)
- **Context**: 32k tokens
- **Concurrent Tasks**: 4
- **GaiaNet Tier**: Standard (1x multiplier)
- **Special Features**: Database operations, storage-intensive tasks

### Raspberry Pi
- **Model**: Qwen 2.5 Coder 1.5B (GGUF Q4_K_M)
- **Context**: 4k tokens
- **Concurrent Tasks**: 2
- **GaiaNet Tier**: Not public
- **Special Features**: Edge processing, sensor integration

## Step-by-Step Setup

### Prerequisites (All Nodes)

1. **Operating System**
   - Mac: macOS 13+ (Ventura or later)
   - Linux: Ubuntu 20.04+, Debian 11+, or compatible
   - Raspberry Pi: Raspberry Pi OS 64-bit

2. **Software**
   - Node.js 18+
   - npm 9+
   - Docker (for sandboxing)
   - Redis (one instance for cluster, can run on any node)

3. **Network**
   - All nodes on same network
   - Firewall rules allow Redis port (6379)
   - (Optional) Public IP for GaiaNet monetization

### Part 1: Redis Cluster Coordinator

Choose one node to run Redis (recommend: MacBook M3 Max or Mac Studio):

```bash
# Option A: Docker (Recommended)
docker run -d \
  --name redis-cluster \
  --restart unless-stopped \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:alpine redis-server --appendonly yes

# Option B: Native installation
# macOS
brew install redis
brew services start redis

# Linux
sudo apt-get install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

**Verify Redis:**
```bash
redis-cli ping
# Should return: PONG
```

### Part 2: Setup Each Node

**On MacBook M3 Max (Super-Coordinator):**

```bash
# Clone repository
git clone <your-repo-url>
cd Agentic-local

# Run distributed setup
./scripts/setup-distributed.sh

# Follow prompts:
# - Node name: macbook-m3-max
# - Redis host: localhost (if Redis is on this machine)
# - Redis port: 6379

# Setup WasmEdge + MLX
./scripts/setup-wasmedge-mlx.sh

# Setup LlamaEdge
./scripts/setup-llamaedge.sh

# Download Qwen 32B model
./scripts/download-qwen-coder.sh
# Select option 3 (32B)

# Setup GaiaNet (for monetization)
./scripts/setup-gaianet.sh
# Select option 3 (32B)
# Choose domain: ai-research, web3-dev, developer-tools

# Initialize cluster
npm run cluster:init

# Start services
gaianet start

# Start QUAD orchestrator (in another terminal)
npm run start:quad
```

**On Mac Studio M1 (Coordinator):**

```bash
# Clone repository
git clone <your-repo-url>
cd Agentic-local

# Run distributed setup
./scripts/setup-distributed.sh

# Follow prompts:
# - Node name: mac-studio-m1
# - Redis host: <MacBook-M3-IP>
# - Redis port: 6379

# Setup WasmEdge + MLX
./scripts/setup-wasmedge-mlx.sh

# Setup LlamaEdge
./scripts/setup-llamaedge.sh

# Download Qwen 32B model
./scripts/download-qwen-coder.sh
# Select option 3 (32B)

# Setup GaiaNet
./scripts/setup-gaianet.sh
# Select option 3 (32B)
# Choose domain: developer-tools, web3-dev

# Initialize cluster
npm run cluster:init

# Start services
gaianet start

# Start QUAD orchestrator
npm run start:quad
```

**On Each Intel NUC (Workers 1-10):**

```bash
# Clone repository
git clone <your-repo-url>
cd Agentic-local

# Run distributed setup
./scripts/setup-distributed.sh

# Follow prompts:
# - Node name: intel-nuc-1 (increment for each NUC)
# - Redis host: <MacBook-M3-IP>
# - Redis port: 6379

# For Intel NUCs, use pre-built WasmEdge
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash

# Setup LlamaEdge
./scripts/setup-llamaedge.sh

# Download appropriate model (7B or 14B based on RAM)
./scripts/download-qwen-coder.sh
# 16GB RAM → option 1 (7B)
# 32GB RAM → option 2 (14B)

# Setup GaiaNet
./scripts/setup-gaianet.sh
# Choose corresponding model
# Choose domain: developer-tools, data-processing

# Initialize cluster
npm run cluster:init

# Start services
gaianet start

# Start QUAD orchestrator
npm run start:quad
```

**On Raspberry Pi (Edge Worker):**

```bash
# Clone repository
git clone <your-repo-url>
cd Agentic-local

# Run distributed setup
./scripts/setup-distributed.sh

# Follow prompts:
# - Node name: raspberry-pi
# - Redis host: <MacBook-M3-IP>
# - Redis port: 6379

# Install WasmEdge (ARM64)
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash

# Setup LlamaEdge
./scripts/setup-llamaedge.sh

# Download tiny model
./scripts/download-qwen-coder.sh
# Or manually download 1.5B model

# No GaiaNet (insufficient resources)

# Initialize cluster
npm run cluster:init

# Start local inference only
MODEL_PATH=~/.llamaedge/models/qwen-1.5b.gguf \
CONTEXT_SIZE=4096 \
llamaedge

# Start QUAD orchestrator (worker mode)
npm run start:quad
```

### Part 3: Verify Cluster

**Check cluster status (run on any node):**

```bash
npm run cluster:status
```

**Expected output:**
```
════════════════════════════════════════════════════════════════════════════════
                          🌐  DISTRIBUTED CLUSTER STATUS
════════════════════════════════════════════════════════════════════════════════

📊 CLUSTER OVERVIEW
────────────────────────────────────────────────────────────────────────────────
Total Nodes:     13
Online:          🟢 13
Offline:         🔴 0
Total Cores:     200
Total RAM:       840.0 GB

📋 DISTRIBUTION
────────────────────────────────────────────────────────────────────────────────
By Role:
  super-coordinator                    1
  coordinator                          1
  worker                               10
  edge-worker                          1

By Hardware Type:
  macbook-m3-max                       1
  mac-studio-m1                        1
  intel-nuc                            10
  raspberry-pi                         1

🔧 CLUSTER CAPABILITIES
────────────────────────────────────────────────────────────────────────────────
extreme-inference           massive-context             multi-model-hosting
swarm-coordination          research-workloads          heavy-inference
large-model-hosting         code-generation             data-processing
database-operations         edge-processing             monitoring

🖥️  NODES
────────────────────────────────────────────────────────────────────────────────
Status  Name                    Type                Role                Resources
────────────────────────────────────────────────────────────────────────────────
🟢      macbook-m3-max          macbook-m3-max      super-coordinator   16c/128GB
🟢      mac-studio-m1           mac-studio-m1       coordinator         20c/64GB
🟢      intel-nuc-1             intel-nuc           worker              8c/32GB
🟢      intel-nuc-2             intel-nuc           worker              8c/32GB
...
🟢      intel-nuc-10            intel-nuc           worker              8c/32GB
🟢      raspberry-pi            raspberry-pi        edge-worker         4c/8GB
```

## Using the Distributed Cluster

### QUAD: Distributed Task Execution

**Example: Distributed Code Generation**

```javascript
import { QuadOrchestrator } from '@ruv/quad';

const quad = new QuadOrchestrator({
  // Connects to Redis automatically
  cluster: {
    mode: 'distributed',
    discovery: { method: 'redis' }
  }
});

// Create a complex project
const project = await quad.createTask({
  type: 'full-stack-app',
  requirements: [
    'User authentication',
    'REST API',
    'React frontend',
    'Database',
    'Docker deployment'
  ],

  // QUAD automatically distributes across cluster
  execution: {
    mode: 'parallel'
  }
});

const result = await quad.execute(project);

// Task distribution might look like:
// - Auth service → Mac Studio (heavy inference)
// - API → NUC-1, NUC-2 (code generation)
// - Frontend → NUC-3, NUC-4 (React)
// - Database → NUC-5 (storage-intensive)
// - Docker → NUC-6 (devops)
// - Integration tests → MacBook M3 (orchestration)
// - Monitoring → Raspberry Pi (edge)
```

**Run example:**
```bash
npm run start:quad
```

### QDAG: Workflow Pipelines

**Example: Data Science Pipeline**

```javascript
import { QDAGOrchestrator } from '@ruv/qdag';

const qdag = new QDAGOrchestrator({
  execution: { mode: 'distributed' }
});

const pipeline = await qdag.createWorkflow({
  name: 'ml-training-pipeline',

  nodes: [
    { id: 'data-collection', agent: 'scraper' },
    { id: 'data-cleaning', agent: 'preprocessor' },
    { id: 'feature-engineering', agent: 'feature-engineer' },
    { id: 'train-rf', agent: 'trainer' },
    { id: 'train-xgb', agent: 'trainer' },
    { id: 'train-nn', agent: 'trainer' },  // These 3 run in parallel
    { id: 'model-eval', agent: 'evaluator' },
    { id: 'deployment', agent: 'deployer' }
  ],

  edges: [/* dependency graph */]
});

// QDAG schedules optimally:
// - Data tasks → NUCs (I/O intensive)
// - Training → Mac Studio + MacBook (heavy compute)
// - Deployment → NUC with Docker
```

**Run example:**
```bash
npm run start:qdag
```

### AgentDB: Distributed State

All agents share state via AgentDB:

```javascript
import { AgentDB } from 'agentdb';

const db = new AgentDB({
  distributed: {
    enabled: true,
    redis: { host: process.env.REDIS_HOST }
  }
});

// Store agent memory (automatically synced across cluster)
await db.set('agent-123', {
  role: 'backend-developer',
  context: 'Working on auth service',
  memory: [/* past interactions */]
});

// Retrieve from any node
const agent = await db.get('agent-123');

// Vector search across all agent memories
const results = await db.search(
  'authentication implementation',
  { limit: 10 }
);
```

### RuvLLM: Intelligent Load Balancing

RuvLLM routes inference requests across your heterogeneous hardware:

```javascript
import { RuvLLM } from 'ruvllm';

const llm = new RuvLLM({
  providers: [
    {
      name: 'macbook-m3',
      baseURL: 'http://macbook-m3-ip:8080/v1',
      model: 'Qwen2.5-Coder-32B',
      priority: 1,  // Use first
      capabilities: ['heavy-inference', 'large-context']
    },
    {
      name: 'mac-studio',
      baseURL: 'http://mac-studio-ip:8080/v1',
      model: 'Qwen2.5-Coder-32B',
      priority: 2,  // Failover
      capabilities: ['heavy-inference', 'large-context']
    },
    {
      name: 'nuc-cluster',
      baseURL: 'http://nuc-1-ip:8080/v1',
      model: 'Qwen2.5-Coder-7B',
      priority: 3,  // Lightweight tasks
      capabilities: ['code-generation']
    }
  ],

  loadBalancing: {
    strategy: 'capability-aware',
    healthCheck: { enabled: true }
  }
});

// Automatically routes to best available node
const response = await llm.generate({
  prompt: 'Generate a complex microservices architecture',
  context: 50000  // → Routes to MacBook or Mac Studio (large context)
});
```

## Monitoring

### Real-Time Dashboard

**Watch cluster status (updates every 5s):**
```bash
npm run cluster:status -- --watch
```

### Per-Node Monitoring

**On each node, check local metrics:**
```bash
# GaiaNet nodes
gaianet log

# Resource usage
htop  # or top

# GPU usage (Mac)
sudo powermetrics --samplers gpu_power

# Network
iftop
```

### Aggregate Metrics

**Create monitoring dashboard (example with Grafana):**

```bash
# Run Prometheus + Grafana stack
docker-compose up -d
```

**`docker-compose.yml`:**
```yaml
version: '3'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

Access at `http://localhost:3000`

## Failure Handling

### Node Failure

**QUAD automatically handles failures:**

```javascript
// In QUAD config
const quad = new QuadOrchestrator({
  resilience: {
    retries: 3,
    circuitBreaker: {
      enabled: true,
      threshold: 5
    }
  }
});

// If a node fails mid-task, QUAD:
// 1. Detects via missing heartbeat
// 2. Marks node as offline
// 3. Reassigns task to another capable node
// 4. Continues execution
```

### Coordinator Failure

**MacBook M3 is primary coordinator, Mac Studio is failover:**

If MacBook goes offline:
1. Mac Studio detects (no heartbeat)
2. Mac Studio becomes primary coordinator
3. AgentDB replication ensures no data loss
4. Cluster continues operating

### Redis Failure

**If Redis goes down:**
- Nodes switch to standalone mode
- Lose inter-node coordination
- Each node continues local operations
- Restart Redis to restore cluster

**Prevention: Redis persistence**
```bash
docker run -d \
  --name redis-cluster \
  --restart unless-stopped \
  -v redis-data:/data \
  redis:alpine redis-server --appendonly yes
```

## Performance Optimization

### Task Placement Hints

**Guide QUAD to optimal nodes:**

```javascript
const task = await quad.createTask({
  subtasks: [
    {
      name: 'heavy-inference',
      preferredNodes: ['macbook-m3-max'],  // Explicit placement
      requirements: { ram: '48GB' }
    },
    {
      name: 'storage-query',
      preferredNodes: ['intel-nuc-5'],  // NUC with fast SSD
      requirements: { storage: 'high-iops' }
    }
  ]
});
```

### Model Selection

**Use smaller models for simple tasks:**

```javascript
const llm = new RuvLLM({ /* ... */ });

// Simple task → Route to NUC (7B model, faster)
const simpleCode = await llm.generate({
  prompt: 'Create a hello world function',
  preferModel: 'Qwen2.5-Coder-7B'
});

// Complex task → Route to Mac (32B model, better quality)
const complexCode = await llm.generate({
  prompt: 'Design a distributed microservices architecture',
  preferModel: 'Qwen2.5-Coder-32B'
});
```

### Network Optimization

**Reduce inter-node traffic:**

1. **Enable caching** (RuvLLM):
   ```javascript
   cache: {
     enabled: true,
     ttl: 3600
   }
   ```

2. **Batch operations** (AgentDB):
   ```javascript
   await db.batch([
     { op: 'set', key: 'agent-1', value: data1 },
     { op: 'set', key: 'agent-2', value: data2 }
   ]);
   ```

3. **Local-first execution**:
   - QUAD tries to keep related tasks on same node (affinity)
   - Configure: `affinity: { enabled: true }`

## Scaling

### Adding New Nodes

1. **Set up new hardware** (e.g., another NUC)
2. **Run setup:**
   ```bash
   ./scripts/setup-distributed.sh
   npm run cluster:init
   ```
3. **QUAD automatically discovers** and starts assigning tasks

### Removing Nodes

1. **Graceful shutdown:**
   ```bash
   # Stop orchestrator
   # Stop GaiaNet/LlamaEdge
   # The node will be marked offline automatically (heartbeat timeout)
   ```

2. **Cluster adapts** and redistributes tasks

## Cost Analysis

### Hardware Investment

| Hardware | Qty | Cost Each | Total |
|----------|-----|-----------|-------|
| MacBook M3 Max 128GB | 1 | $4,000 | $4,000 |
| Mac Studio M1 64GB | 1 | $2,500 | $2,500 |
| Intel NUC (32GB) | 10 | $800 | $8,000 |
| Raspberry Pi 5 | 1 | $100 | $100 |
| **Total** | **13** | | **$14,600** |

### Monthly Earnings (GaiaNet)

| Node | Estimated Earnings |
|------|-------------------|
| MacBook M3 Max | $300-500 |
| Mac Studio M1 | $120-200 |
| Intel NUCs (×10) | $300-600 |
| Raspberry Pi | $0 |
| **Total** | **$720-1,300/month** |

### ROI

- **Monthly Earnings**: $720-1,300
- **Electricity Cost** (estimated): ~$100/month
- **Net Earnings**: $620-1,200/month
- **Break-Even**: 12-24 months
- **Year 2+**: Pure profit while also having local AI

**Plus**: You retain full ownership and control, unlike cloud rentals.

## Troubleshooting

### Node Not Appearing in Cluster

1. **Check Redis connection:**
   ```bash
   redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
   ```

2. **Verify node registration:**
   ```bash
   redis-cli keys 'cluster:node:*'
   ```

3. **Check firewall:**
   ```bash
   # Allow Redis port
   sudo ufw allow 6379
   ```

### Task Not Distributing

1. **Check node capabilities:**
   ```bash
   npm run cluster:status
   ```

2. **Verify task requirements match:**
   ```javascript
   // Task requires 48GB RAM but all workers have 32GB
   // → Won't distribute, will fail
   ```

3. **Check logs:**
   ```bash
   npm run start:quad  # Look for scheduling errors
   ```

### Performance Issues

1. **Identify bottleneck:**
   ```bash
   npm run cluster:status
   # Look for nodes at 100% load
   ```

2. **Rebalance:**
   - Add more nodes
   - Or reduce concurrent tasks on overloaded node

3. **Network latency:**
   ```bash
   ping <other-node-ip>
   # Should be <5ms on local network
   ```

## Summary

You now have a complete distributed agent cluster:

✅ **13 nodes** working in harmony
✅ **200+ CPU cores** and **840GB RAM** at your command
✅ **Automatic task distribution** via QUAD/QDAG
✅ **Intelligent load balancing** via RuvLLM
✅ **Shared state** via AgentDB
✅ **Fault tolerance** with automatic failover
✅ **$720-1,300/month** passive income via GaiaNet

**All local. All sovereign. All yours.**

Start the cluster:
```bash
npm run cluster:init
npm run start:quad
```

Watch it work:
```bash
npm run cluster:status -- --watch
```

🚀 **Your home just became an AI data center.**
