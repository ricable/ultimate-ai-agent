# API Reference

Complete API reference for the Synaptic Neural Mesh distributed neural fabric. This documentation covers all programmatic interfaces including REST API, WebSocket API, and JavaScript/TypeScript SDK.

## 🌐 API Overview

The Synaptic Neural Mesh provides multiple API interfaces:

- **REST API**: HTTP-based interface for standard operations
- **WebSocket API**: Real-time bidirectional communication
- **GraphQL API**: Flexible query interface for complex data retrieval
- **MCP Protocol**: Model Context Protocol for AI assistant integration
- **JavaScript SDK**: Type-safe client library

### Base Configuration

```typescript
const config = {
  baseURL: 'http://localhost:8080',
  wsURL: 'ws://localhost:8080/ws',
  graphqlURL: 'http://localhost:8080/graphql',
  mcpURL: 'http://localhost:8080/mcp'
};
```

## 🔧 REST API

### Authentication

All API calls require authentication via API key or node certificate:

```bash
# API Key authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     "http://localhost:8080/api/v1/status"

# Certificate authentication
curl --cert node.crt --key node.key \
     "http://localhost:8080/api/v1/status"
```

### Base Endpoints

```
GET    /api/v1/status           # Node status
GET    /api/v1/health           # Health check
GET    /api/v1/metrics          # Prometheus metrics
GET    /api/v1/info             # Node information
POST   /api/v1/shutdown         # Graceful shutdown
```

---

## 🧠 Neural API

### Agent Management

#### `POST /api/v1/neural/agents`
Spawn a new neural agent.

**Request:**
```json
{
  "type": "mlp",
  "architecture": {
    "layers": [784, 128, 64, 10],
    "activation": "relu",
    "optimizer": "adam"
  },
  "task": "image_classification",
  "memoryLimit": "50MB",
  "timeout": 3600,
  "replicas": 1
}
```

**Response:**
```json
{
  "success": true,
  "agent": {
    "id": "agent_abc123",
    "type": "mlp",
    "status": "active",
    "memoryUsage": "32MB",
    "createdAt": "2025-07-13T12:00:00Z",
    "performance": {
      "inferenceTime": "67ms",
      "accuracy": 0.95,
      "tasksCompleted": 0
    }
  }
}
```

#### `GET /api/v1/neural/agents`
List all neural agents.

**Query Parameters:**
- `status` - Filter by status (`active`, `idle`, `training`, `terminated`)
- `type` - Filter by architecture type
- `limit` - Maximum number of results (default: 100)
- `offset` - Pagination offset

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_abc123",
      "type": "mlp",
      "status": "active",
      "memoryUsage": "32MB",
      "performance": {...}
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

#### `GET /api/v1/neural/agents/{agentId}`
Get specific agent details.

**Response:**
```json
{
  "agent": {
    "id": "agent_abc123",
    "type": "mlp",
    "status": "active",
    "architecture": {
      "layers": [784, 128, 64, 10],
      "activation": "relu",
      "parameters": 101770
    },
    "performance": {
      "inferenceTime": "67ms",
      "memoryUsage": "32MB",
      "accuracy": 0.95,
      "tasksCompleted": 142,
      "uptime": "2h 34m 12s"
    },
    "task": "image_classification",
    "createdAt": "2025-07-13T12:00:00Z"
  }
}
```

#### `POST /api/v1/neural/agents/{agentId}/inference`
Run inference on an agent.

**Request:**
```json
{
  "input": [0.1, 0.2, 0.3, ...],
  "format": "array",
  "timeout": 5000
}
```

**Response:**
```json
{
  "success": true,
  "output": [0.1, 0.8, 0.05, ...],
  "confidence": 0.87,
  "inferenceTime": "45ms",
  "agentId": "agent_abc123"
}
```

#### `DELETE /api/v1/neural/agents/{agentId}`
Terminate a neural agent.

**Query Parameters:**
- `force` - Force termination without graceful shutdown

**Response:**
```json
{
  "success": true,
  "message": "Agent terminated successfully"
}
```

### Training API

#### `POST /api/v1/neural/training/start`
Start distributed training.

**Request:**
```json
{
  "agents": ["agent_abc123", "agent_def456"],
  "dataset": {
    "type": "supervised",
    "source": "/path/to/dataset.json",
    "format": "json"
  },
  "config": {
    "epochs": 100,
    "batchSize": 32,
    "learningRate": 0.001,
    "validationSplit": 0.2
  },
  "strategy": "federated"
}
```

**Response:**
```json
{
  "success": true,
  "trainingId": "training_xyz789",
  "status": "started",
  "estimatedDuration": "45m",
  "participatingAgents": 2
}
```

#### `GET /api/v1/neural/training/{trainingId}`
Get training status.

**Response:**
```json
{
  "training": {
    "id": "training_xyz789",
    "status": "running",
    "progress": 0.34,
    "epoch": 34,
    "totalEpochs": 100,
    "metrics": {
      "loss": 0.23,
      "accuracy": 0.87,
      "validationLoss": 0.31,
      "validationAccuracy": 0.82
    },
    "duration": "23m 45s",
    "estimatedRemaining": "21m 15s"
  }
}
```

---

## 🌐 Mesh API

### Network Management

#### `GET /api/v1/mesh/status`
Get mesh network status.

**Response:**
```json
{
  "mesh": {
    "nodeId": "12D3KooWAbc123...",
    "networkId": "mainnet",
    "connectedPeers": 15,
    "maxPeers": 50,
    "topology": "mesh",
    "uptime": "2h 34m 12s",
    "status": "operational"
  },
  "network": {
    "port": 8080,
    "bindAddress": "0.0.0.0",
    "externalAddress": "192.168.1.100:8080",
    "natStatus": "traversed"
  }
}
```

#### `POST /api/v1/mesh/peers/connect`
Connect to a peer.

**Request:**
```json
{
  "address": "/ip4/192.168.1.100/tcp/8080/p2p/12D3KooW...",
  "timeout": 30
}
```

**Response:**
```json
{
  "success": true,
  "peer": {
    "id": "12D3KooW...",
    "address": "/ip4/192.168.1.100/tcp/8080",
    "latency": "23ms",
    "status": "connected"
  }
}
```

#### `GET /api/v1/mesh/peers`
List connected peers.

**Response:**
```json
{
  "peers": [
    {
      "id": "12D3KooW...",
      "address": "/ip4/192.168.1.100/tcp/8080",
      "latency": "23ms",
      "uptime": "1h 20m",
      "status": "connected",
      "version": "1.0.0-alpha.1"
    }
  ],
  "total": 15
}
```

### Topology Management

#### `GET /api/v1/mesh/topology`
Get current mesh topology.

**Response:**
```json
{
  "topology": {
    "type": "mesh",
    "nodes": 16,
    "edges": 45,
    "diameter": 3,
    "clustering": 0.72,
    "redundancy": "high"
  },
  "visualization": {
    "nodes": [...],
    "links": [...]
  }
}
```

#### `POST /api/v1/mesh/topology/optimize`
Optimize mesh topology.

**Request:**
```json
{
  "strategy": "adaptive",
  "targetLatency": "100ms",
  "targetRedundancy": 3
}
```

---

## 📊 DAG API

### Consensus Management

#### `GET /api/v1/dag/status`
Get DAG consensus status.

**Response:**
```json
{
  "dag": {
    "height": 45678,
    "vertices": 123456,
    "pendingTxs": 3,
    "finality": "450ms",
    "consensus": "avalanche",
    "validators": 12
  },
  "sync": {
    "status": "synced",
    "peers": 8,
    "lastSync": "2025-07-13T12:00:00Z"
  }
}
```

#### `POST /api/v1/dag/submit`
Submit transaction to DAG.

**Request:**
```json
{
  "data": "Hello Neural Mesh!",
  "type": "message",
  "fee": 1000,
  "priority": "normal"
}
```

**Response:**
```json
{
  "success": true,
  "transaction": {
    "id": "tx_abc123",
    "vertex": "vertex_def456",
    "status": "pending",
    "fee": 1000,
    "timestamp": "2025-07-13T12:00:00Z"
  }
}
```

#### `GET /api/v1/dag/vertices/{vertexId}`
Get vertex details.

**Response:**
```json
{
  "vertex": {
    "id": "vertex_def456",
    "parents": ["vertex_abc123", "vertex_ghi789"],
    "timestamp": "2025-07-13T12:00:00Z",
    "data": "...",
    "signature": "...",
    "weight": 42,
    "confirmed": true
  }
}
```

#### `GET /api/v1/dag/query`
Query DAG data.

**Query Parameters:**
- `height` - Query by height
- `from` - Start timestamp
- `to` - End timestamp
- `type` - Transaction type
- `limit` - Maximum results

**Response:**
```json
{
  "vertices": [...],
  "total": 100,
  "query": {...}
}
```

---

## 🔧 Configuration API

#### `GET /api/v1/config`
Get current configuration.

**Response:**
```json
{
  "config": {
    "node": {...},
    "network": {...},
    "neural": {...},
    "dag": {...}
  }
}
```

#### `POST /api/v1/config`
Update configuration.

**Request:**
```json
{
  "network.maxPeers": 100,
  "neural.memoryLimit": "100MB"
}
```

#### `POST /api/v1/config/validate`
Validate configuration.

**Request:**
```json
{
  "config": {...}
}
```

---

## 🔌 WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  // Send authentication
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'YOUR_API_KEY'
  }));
};
```

### Real-time Events

#### Status Updates
```json
{
  "type": "status_update",
  "data": {
    "nodeId": "12D3KooW...",
    "status": "operational",
    "timestamp": "2025-07-13T12:00:00Z"
  }
}
```

#### Neural Agent Events
```json
{
  "type": "agent_spawned",
  "data": {
    "agentId": "agent_abc123",
    "type": "mlp",
    "timestamp": "2025-07-13T12:00:00Z"
  }
}
```

#### Peer Events
```json
{
  "type": "peer_connected",
  "data": {
    "peerId": "12D3KooW...",
    "address": "/ip4/192.168.1.100/tcp/8080",
    "timestamp": "2025-07-13T12:00:00Z"
  }
}
```

#### DAG Events
```json
{
  "type": "vertex_added",
  "data": {
    "vertexId": "vertex_abc123",
    "height": 45679,
    "timestamp": "2025-07-13T12:00:00Z"
  }
}
```

### Subscriptions

```javascript
// Subscribe to all agent events
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'neural.agents',
  filter: { type: 'mlp' }
}));

// Subscribe to mesh events
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'mesh.peers'
}));

// Subscribe to DAG events
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'dag.vertices',
  filter: { type: 'message' }
}));
```

---

## 📱 JavaScript SDK

### Installation

```bash
npm install synaptic-mesh-sdk
```

### Basic Usage

```typescript
import { SynapticMesh } from 'synaptic-mesh-sdk';

const mesh = new SynapticMesh({
  baseURL: 'http://localhost:8080',
  apiKey: 'your-api-key'
});

// Connect to mesh
await mesh.connect();

// Spawn neural agent
const agent = await mesh.neural.spawnAgent({
  type: 'mlp',
  task: 'classification'
});

// Run inference
const result = await agent.inference([0.1, 0.2, 0.3]);

// Monitor status
mesh.on('agent_spawned', (event) => {
  console.log('New agent:', event.agentId);
});
```

### Advanced SDK Usage

```typescript
// Mesh management
const meshStatus = await mesh.mesh.getStatus();
await mesh.mesh.connectPeer('/ip4/192.168.1.100/tcp/8080/p2p/12D3...');

// Neural network operations
const agents = await mesh.neural.listAgents();
const training = await mesh.neural.startTraining({
  agents: ['agent_1', 'agent_2'],
  config: { epochs: 100 }
});

// DAG operations
const vertex = await mesh.dag.getVertex('vertex_abc123');
const tx = await mesh.dag.submitTransaction({
  data: 'Hello World',
  type: 'message'
});

// Configuration
await mesh.config.set('network.maxPeers', 100);
const config = await mesh.config.get();
```

---

## 🤖 MCP Integration

### Model Context Protocol Support

The Synaptic Neural Mesh supports MCP for AI assistant integration:

```typescript
// Enable MCP server
await mesh.mcp.start({
  transport: 'stdio',
  capabilities: [
    'neural_spawn',
    'mesh_management',
    'dag_operations'
  ]
});

// MCP tools for AI assistants
const tools = [
  {
    name: 'synaptic_mesh_spawn_agent',
    description: 'Spawn a neural agent',
    inputSchema: {
      type: 'object',
      properties: {
        type: { type: 'string' },
        task: { type: 'string' }
      }
    }
  }
];
```

### Claude Code Integration

```bash
# Add MCP server
claude mcp add synaptic-mesh npx synaptic-mesh@alpha mcp start

# Use in Claude Code
# "Spawn a neural agent for image classification"
```

---

## 📊 Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "AGENT_SPAWN_FAILED",
    "message": "Failed to spawn neural agent",
    "details": {
      "reason": "Insufficient memory",
      "availableMemory": "20MB",
      "requiredMemory": "50MB"
    },
    "timestamp": "2025-07-13T12:00:00Z"
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_REQUEST` | Malformed request | Check request format |
| `AUTHENTICATION_FAILED` | Invalid credentials | Verify API key |
| `AGENT_SPAWN_FAILED` | Cannot spawn agent | Check memory/resources |
| `PEER_CONNECTION_FAILED` | Cannot connect to peer | Check network connectivity |
| `CONSENSUS_ERROR` | DAG consensus issue | Wait for network sync |
| `RESOURCE_EXHAUSTED` | Insufficient resources | Scale up or optimize |

---

## 🔍 Rate Limiting

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/api/v1/neural/agents` POST | 10/min | 1 minute |
| `/api/v1/neural/agents/{id}/inference` | 1000/min | 1 minute |
| `/api/v1/mesh/peers/connect` | 5/min | 1 minute |
| `/api/v1/dag/submit` | 100/min | 1 minute |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642781234
```

---

## 📚 Examples

### Complete Integration Example

```typescript
import { SynapticMesh, AgentType, TrainingStrategy } from 'synaptic-mesh-sdk';

async function completeExample() {
  // Initialize mesh connection
  const mesh = new SynapticMesh({
    baseURL: 'http://localhost:8080',
    wsURL: 'ws://localhost:8080/ws',
    apiKey: process.env.SYNAPTIC_API_KEY
  });

  try {
    // Connect and verify status
    await mesh.connect();
    const status = await mesh.getStatus();
    console.log('Mesh status:', status);

    // Spawn multiple neural agents
    const agents = await Promise.all([
      mesh.neural.spawnAgent({ type: 'mlp', task: 'classification' }),
      mesh.neural.spawnAgent({ type: 'lstm', task: 'sequence_processing' }),
      mesh.neural.spawnAgent({ type: 'cnn', task: 'image_recognition' })
    ]);

    console.log(`Spawned ${agents.length} agents`);

    // Start distributed training
    const training = await mesh.neural.startTraining({
      agents: agents.map(a => a.id),
      dataset: { source: './training_data.json' },
      config: {
        epochs: 50,
        strategy: TrainingStrategy.Federated
      }
    });

    // Monitor training progress
    const trainingMonitor = setInterval(async () => {
      const progress = await mesh.neural.getTrainingStatus(training.id);
      console.log(`Training progress: ${(progress.progress * 100).toFixed(1)}%`);
      
      if (progress.status === 'completed') {
        clearInterval(trainingMonitor);
        console.log('Training completed!');
      }
    }, 5000);

    // Connect to mesh network
    await mesh.mesh.connectPeer('/ip4/bootstrap.synaptic-mesh.net/tcp/8080/p2p/12D3...');
    
    // Submit some transactions
    await mesh.dag.submitTransaction({
      type: 'agent_result',
      data: JSON.stringify({ accuracy: 0.95 })
    });

  } catch (error) {
    console.error('Error:', error);
  } finally {
    await mesh.disconnect();
  }
}

completeExample();
```

This API reference provides comprehensive coverage of all Synaptic Neural Mesh interfaces, enabling developers to build sophisticated distributed neural applications.

**Next**: Check out our [Integration Examples](../examples/integrations/) for real-world usage patterns.