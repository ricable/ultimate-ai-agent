# MCP Integration Guide

Complete guide for integrating Synaptic Neural Mesh with AI assistants using the Model Context Protocol (MCP). This enables seamless coordination between human-AI teams and distributed neural networks.

## 🤖 What is MCP Integration?

The Model Context Protocol (MCP) is a standardized way for AI assistants to interact with external tools and systems. Synaptic Neural Mesh's MCP integration allows AI assistants like Claude to:

- Spawn and manage neural agents
- Coordinate distributed learning tasks
- Monitor mesh network health
- Orchestrate complex multi-agent workflows
- Provide natural language interfaces to the neural mesh

## 🚀 Quick Setup

### Prerequisites

- **Claude Code** installed globally
- **Synaptic Neural Mesh** node running
- **Node.js 18+** for MCP server

### 1. Install Claude Code

```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Activate with permissions
claude --dangerously-skip-permissions
```

### 2. Configure MCP Server

```bash
# Add Synaptic Neural Mesh MCP server
claude mcp add synaptic-mesh npx synaptic-mesh@alpha mcp start

# Verify installation
claude mcp list
```

### 3. Start MCP-Enabled Mesh

```bash
# Start mesh with MCP support
npx synaptic-mesh start --mcp --mcp-stdio

# Or with specific MCP configuration
npx synaptic-mesh start --mcp --mcp-port 3001 --mcp-host localhost
```

## 🔧 MCP Server Configuration

### Basic Configuration

The MCP server can be configured through the mesh configuration file:

```json
{
  "mcp": {
    "enabled": true,
    "transport": "stdio",
    "port": 3001,
    "host": "localhost",
    "authentication": {
      "type": "api_key",
      "key": "your-api-key"
    },
    "capabilities": [
      "neural_management",
      "mesh_coordination",
      "dag_operations",
      "monitoring"
    ]
  }
}
```

### Advanced Configuration

```json
{
  "mcp": {
    "enabled": true,
    "transport": "websocket",
    "port": 3001,
    "host": "0.0.0.0",
    "ssl": {
      "enabled": true,
      "cert": "/path/to/cert.pem",
      "key": "/path/to/key.pem"
    },
    "authentication": {
      "type": "certificate",
      "cert": "/path/to/client.crt"
    },
    "rateLimiting": {
      "enabled": true,
      "requestsPerMinute": 100,
      "burstSize": 20
    },
    "logging": {
      "level": "info",
      "file": "/var/log/synaptic-mcp.log"
    }
  }
}
```

## 🛠️ Available MCP Tools

### Neural Network Management

#### `synaptic_neural_spawn`
Spawn neural agents with specified configurations.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "enum": ["mlp", "lstm", "cnn", "transformer"],
      "description": "Neural architecture type"
    },
    "task": {
      "type": "string",
      "description": "Task description for the agent"
    },
    "config": {
      "type": "object",
      "properties": {
        "layers": {"type": "array"},
        "activation": {"type": "string"},
        "memoryLimit": {"type": "string"}
      }
    },
    "replicas": {
      "type": "number",
      "default": 1,
      "description": "Number of agent replicas to spawn"
    }
  },
  "required": ["type", "task"]
}
```

**Example Usage in Claude:**
```
Human: Spawn 3 LSTM neural agents for time series forecasting

Claude: I'll spawn 3 LSTM neural agents for time series forecasting.

[Uses synaptic_neural_spawn tool]
✅ Successfully spawned 3 LSTM agents
- Agent IDs: agent_ts001, agent_ts002, agent_ts003
- Task: time series forecasting
- Memory per agent: 64MB
- Status: All agents active and ready
```

#### `synaptic_neural_list`
List and filter neural agents.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["active", "idle", "training", "terminated"]
    },
    "type": {
      "type": "string",
      "enum": ["mlp", "lstm", "cnn", "transformer"]
    },
    "limit": {"type": "number", "default": 100}
  }
}
```

#### `synaptic_neural_inference`
Run inference on neural agents.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "agentId": {
      "type": "string",
      "description": "Target agent ID"
    },
    "input": {
      "type": "array",
      "description": "Input data for inference"
    },
    "timeout": {
      "type": "number",
      "default": 5000,
      "description": "Timeout in milliseconds"
    }
  },
  "required": ["agentId", "input"]
}
```

### Mesh Network Management

#### `synaptic_mesh_status`
Get comprehensive mesh network status.

**Example Response:**
```json
{
  "nodeId": "12D3KooWAbc123...",
  "networkId": "mainnet",
  "connectedPeers": 15,
  "activeAgents": 8,
  "dagHeight": 45678,
  "status": "operational",
  "uptime": "2h 34m 12s"
}
```

#### `synaptic_mesh_connect`
Connect to peers or join mesh networks.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "address": {
      "type": "string",
      "description": "Peer multiaddress or bootstrap address"
    },
    "timeout": {
      "type": "number",
      "default": 30
    }
  },
  "required": ["address"]
}
```

#### `synaptic_mesh_topology`
Analyze and optimize mesh topology.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["analyze", "optimize", "visualize"]
    },
    "strategy": {
      "type": "string",
      "enum": ["adaptive", "performance", "redundancy"]
    }
  }
}
```

### DAG Operations

#### `synaptic_dag_submit`
Submit transactions to the DAG.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "string",
      "description": "Transaction data"
    },
    "type": {
      "type": "string",
      "enum": ["message", "model_update", "consensus", "reward"]
    },
    "priority": {
      "type": "string",
      "enum": ["low", "normal", "high"],
      "default": "normal"
    }
  },
  "required": ["data", "type"]
}
```

#### `synaptic_dag_query`
Query DAG state and history.

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "vertexId": {"type": "string"},
    "height": {"type": "number"},
    "timeRange": {
      "type": "object",
      "properties": {
        "from": {"type": "string"},
        "to": {"type": "string"}
      }
    },
    "type": {"type": "string"},
    "limit": {"type": "number", "default": 100}
  }
}
```

### Monitoring & Analytics

#### `synaptic_monitor_performance`
Get real-time performance metrics.

**Example Response:**
```json
{
  "cpu": {"usage": "15.3%", "cores": 8},
  "memory": {"used": "234MB", "total": "8GB"},
  "network": {"inbound": "2.3MB/s", "outbound": "1.1MB/s"},
  "neural": {
    "activeAgents": 8,
    "averageInferenceTime": "67ms",
    "totalInferences": 1420
  },
  "dag": {
    "height": 45678,
    "pendingTxs": 3,
    "consensusLatency": "450ms"
  }
}
```

## 📝 Workflow Examples

### 1. Distributed Learning Coordination

```
Human: Set up a distributed learning experiment with 5 neural agents for image classification using the CIFAR-10 dataset.

Claude: I'll set up a distributed learning experiment for CIFAR-10 image classification with 5 neural agents.

[Uses multiple MCP tools in sequence]

1. Spawning neural agents...
   ✅ Created 5 CNN agents optimized for image classification
   - Agent IDs: cnn_001 through cnn_005
   - Architecture: 3 conv layers + 2 dense layers
   - Memory: 128MB per agent

2. Configuring distributed training...
   ✅ Initialized federated learning setup
   - Dataset: CIFAR-10 (50,000 training images)
   - Strategy: Federated averaging
   - Epochs: 100 per round
   - Communication rounds: 20

3. Starting training coordination...
   ✅ Training initiated across all agents
   - Round 1/20 in progress
   - Current accuracy: 67.3%
   - Estimated completion: 45 minutes

The experiment is now running. I'll monitor progress and provide updates as the agents learn collaboratively.
```

### 2. Mesh Network Optimization

```
Human: Analyze our mesh network performance and optimize it for better latency.

Claude: I'll analyze your mesh network and optimize it for better latency performance.

[Uses synaptic_mesh_status and synaptic_mesh_topology tools]

📊 Current Network Analysis:
- Connected peers: 23/50
- Average latency: 127ms
- Network diameter: 4 hops
- Clustering coefficient: 0.65
- Identified bottlenecks: 3 congested nodes

🔧 Optimization Recommendations:
1. Topology restructuring to reduce diameter to 3 hops
2. Adding 2 strategic relay nodes
3. Rebalancing connections for high-traffic nodes

Applying optimizations...
✅ Network topology optimized
- New average latency: 89ms (30% improvement)
- Network diameter: 3 hops
- Better load distribution achieved

Your mesh network is now optimized for lower latency!
```

### 3. Real-time Monitoring Dashboard

```
Human: Create a monitoring dashboard for our neural mesh and alert me if any agents fail.

Claude: I'll set up real-time monitoring for your neural mesh with failure alerts.

[Uses synaptic_monitor_performance and sets up event subscriptions]

🖥️ Monitoring Dashboard Created:

Current Status:
┌─────────────────────────────────────┐
│ 🧠 Neural Agents: 12 active        │
│ 🌐 Network Peers: 18 connected     │
│ 📊 DAG Height: 45,842              │
│ ⚡ Avg Latency: 89ms               │
│ 💾 Memory Usage: 1.2GB / 8GB       │
└─────────────────────────────────────┘

Performance Metrics:
- CPU Usage: 23.7% (normal)
- Network I/O: ↓2.8MB/s ↑1.4MB/s
- Inference Rate: 847 req/min
- Error Rate: 0.02% (excellent)

✅ Alert system configured:
- Agent failures: Immediate notification
- High latency: >200ms threshold
- Memory pressure: >90% usage
- Network partitions: Auto-detect

I'll continuously monitor and alert you of any issues!
```

## 🔄 Advanced Integration Patterns

### 1. Automated Swarm Management

```typescript
// MCP tool for intelligent swarm management
export class SwarmManagerTool implements MCPTool {
  name = "synaptic_swarm_manage";
  
  async execute(params: SwarmManagementParams) {
    const { action, strategy, target } = params;
    
    switch (action) {
      case "auto_scale":
        return await this.autoScale(target);
      case "load_balance":
        return await this.loadBalance(strategy);
      case "health_check":
        return await this.healthCheck();
    }
  }
  
  private async autoScale(target: number) {
    const current = await this.mesh.neural.getAgentCount();
    
    if (current < target) {
      // Scale up: spawn more agents
      const needed = target - current;
      const agents = await this.spawnAgents(needed);
      return { action: "scaled_up", agents: needed, ids: agents };
    } else if (current > target) {
      // Scale down: terminate excess agents
      const excess = current - target;
      const terminated = await this.terminateExcessAgents(excess);
      return { action: "scaled_down", terminated: excess };
    }
    
    return { action: "no_scaling_needed", current: current };
  }
}
```

### 2. Intelligent Task Distribution

```typescript
// MCP tool for AI-driven task distribution
export class TaskDistributorTool implements MCPTool {
  name = "synaptic_task_distribute";
  
  async execute(params: TaskDistributionParams) {
    const { tasks, strategy, constraints } = params;
    
    // Analyze agent capabilities and current load
    const agents = await this.mesh.neural.listAgents();
    const capabilities = await this.analyzeCapabilities(agents);
    
    // Use AI to optimize task assignment
    const assignment = await this.optimizeAssignment(
      tasks, 
      capabilities, 
      strategy,
      constraints
    );
    
    // Distribute tasks to agents
    const results = await Promise.all(
      assignment.map(({ agentId, task }) => 
        this.mesh.neural.assignTask(agentId, task)
      )
    );
    
    return {
      distributed: results.length,
      assignment: assignment,
      estimatedCompletion: this.estimateCompletion(assignment)
    };
  }
}
```

### 3. Cross-Agent Learning Coordination

```typescript
// MCP tool for coordinating learning across agents
export class LearningCoordinatorTool implements MCPTool {
  name = "synaptic_learning_coordinate";
  
  async execute(params: LearningCoordinationParams) {
    const { agents, strategy, dataset, config } = params;
    
    // Set up federated learning coordination
    const coordinator = new FederatedCoordinator({
      strategy: strategy, // "fedavg", "fedprox", "scaffold"
      aggregationRounds: config.rounds,
      clientsPerRound: config.clientsPerRound
    });
    
    // Distribute dataset partitions
    const partitions = await this.partitionDataset(dataset, agents.length);
    
    // Initialize learning session
    const session = await coordinator.initializeSession({
      agents: agents,
      dataPartitions: partitions,
      modelConfig: config.model
    });
    
    // Start coordinated learning
    const progress = await coordinator.startLearning(session);
    
    return {
      sessionId: session.id,
      participants: agents.length,
      estimatedDuration: progress.estimatedDuration,
      status: "started"
    };
  }
}
```

## 🛡️ Security & Authentication

### API Key Authentication

```json
{
  "mcp": {
    "authentication": {
      "type": "api_key",
      "key": "sk_synaptic_mesh_...",
      "headers": {
        "Authorization": "Bearer ${key}"
      }
    }
  }
}
```

### Certificate-Based Authentication

```json
{
  "mcp": {
    "authentication": {
      "type": "certificate",
      "clientCert": "/path/to/client.crt",
      "clientKey": "/path/to/client.key",
      "caCert": "/path/to/ca.crt"
    }
  }
}
```

### Role-Based Access Control

```json
{
  "mcp": {
    "rbac": {
      "enabled": true,
      "roles": {
        "operator": [
          "synaptic_neural_spawn",
          "synaptic_neural_list",
          "synaptic_mesh_status"
        ],
        "admin": [
          "*"
        ],
        "readonly": [
          "synaptic_mesh_status",
          "synaptic_monitor_performance",
          "synaptic_neural_list"
        ]
      }
    }
  }
}
```

## 📊 Monitoring & Observability

### MCP Server Metrics

```typescript
// Prometheus metrics for MCP integration
const mcpMetrics = {
  requestsTotal: new Counter({
    name: 'mcp_requests_total',
    help: 'Total MCP requests',
    labelNames: ['tool', 'status']
  }),
  
  requestDuration: new Histogram({
    name: 'mcp_request_duration_seconds',
    help: 'MCP request duration',
    labelNames: ['tool']
  }),
  
  activeConnections: new Gauge({
    name: 'mcp_active_connections',
    help: 'Active MCP connections'
  })
};
```

### Health Checks

```bash
# Check MCP server health
curl http://localhost:3001/health

# Response
{
  "status": "healthy",
  "timestamp": "2025-07-13T12:00:00Z",
  "version": "1.0.0-alpha.1",
  "connections": 3,
  "uptime": "2h 34m 12s"
}
```

## 🚨 Troubleshooting

### Common Issues

#### MCP Server Won't Start
```bash
# Check port availability
netstat -tulpn | grep 3001

# Check MCP configuration
npx synaptic-mesh config validate --section mcp

# Enable debug logging
npx synaptic-mesh start --mcp --debug
```

#### Claude Can't Connect
```bash
# Verify MCP server is registered
claude mcp list

# Re-add MCP server
claude mcp remove synaptic-mesh
claude mcp add synaptic-mesh npx synaptic-mesh@alpha mcp start

# Test connection
claude mcp test synaptic-mesh
```

#### Tool Authorization Failed
```bash
# Check API key
npx synaptic-mesh config get mcp.authentication.key

# Regenerate API key
npx synaptic-mesh auth generate-key --service mcp

# Update Claude configuration
claude mcp update synaptic-mesh --auth-key <new-key>
```

### Debug Mode

```bash
# Start with verbose MCP debugging
npx synaptic-mesh start --mcp --mcp-debug --log-level debug

# Monitor MCP requests
tail -f ~/.synaptic/logs/mcp.log

# Test specific MCP tools
npx synaptic-mesh mcp test-tool synaptic_neural_spawn
```

## 📚 Examples Repository

For complete working examples, check out:

- [Basic MCP Integration](../examples/mcp/basic-integration.md)
- [Advanced Swarm Management](../examples/mcp/swarm-management.md)
- [Real-time Monitoring](../examples/mcp/monitoring.md)
- [Custom MCP Tools](../examples/mcp/custom-tools.md)

---

The MCP integration transforms Synaptic Neural Mesh into a powerful tool that AI assistants can use naturally, enabling unprecedented human-AI collaboration in distributed neural network management.

**Next**: Explore [DAA Integration](daa-integration.md) for advanced swarm intelligence patterns.