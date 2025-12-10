# Hello Neural Mesh - First Steps

A complete beginner's example demonstrating the core concepts of Synaptic Neural Mesh through a simple distributed neural network.

## 🎯 What You'll Learn

- Initialize a neural mesh node
- Spawn your first neural agent
- Run distributed inference
- Connect to the global mesh network
- Monitor performance and status

## 📋 Prerequisites

- Node.js 18+ installed
- Basic understanding of neural networks
- 10 minutes of time

## 🚀 Step-by-Step Tutorial

### Step 1: Initialize Your Neural Mesh

```bash
# Create a new project
mkdir hello-neural-mesh
cd hello-neural-mesh

# Initialize the mesh
npx synaptic-mesh@alpha init hello-mesh --template default

# What this creates:
# .synaptic/
# ├── config.json          # Configuration
# ├── keys/                # Cryptographic keys
# └── data/                # Local storage
```

**What happened:**
- Generated quantum-resistant cryptographic keys
- Created default configuration for mainnet
- Set up local data storage directory

### Step 2: Start Your Node

```bash
# Start with web dashboard
npx synaptic-mesh start --ui --port 8080

# You should see:
# ✅ Configuration loaded
# ✅ P2P networking started on port 8080
# ✅ DAG consensus layer active
# ✅ Neural runtime loaded (4 WASM modules)
# ✅ Web UI available at http://localhost:3000
# 🌐 Node ID: 12D3KooWAbc123...
```

**Visit the dashboard:** Open http://localhost:3000 to see your node status.

### Step 3: Spawn Your First Neural Agent

```bash
# Spawn a simple MLP agent for classification
npx synaptic-mesh neural spawn \
  --type mlp \
  --task "image_classification" \
  --memory 64MB

# Expected output:
# ✅ Neural agent spawned successfully
#    ├── Agent ID: agent_abc123
#    ├── Type: mlp (Multi-Layer Perceptron)
#    ├── Task: image_classification
#    ├── Memory: 64MB
#    ├── Architecture: [784, 256, 128, 10]
#    └── Status: Active
```

### Step 4: Check Your Agent

```bash
# List all agents
npx synaptic-mesh neural list

# Get detailed agent info
npx synaptic-mesh neural list --detailed

# Example output:
# 🧠 Active Neural Agents (1)
# ┌─────────────┬─────┬────────────────────┬────────┬────────┐
# │ Agent ID    │ Type│ Task               │ Memory │ Status │
# ├─────────────┼─────┼────────────────────┼────────┼────────┤
# │ agent_abc123│ mlp │ image_classification│ 64MB  │ Active │
# └─────────────┴─────┴────────────────────┴────────┴────────┘
```

### Step 5: Run Your First Inference

```bash
# Test inference with sample data (28x28 image flattened to 784 values)
npx synaptic-mesh neural inference agent_abc123 \
  --input '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]' \
  --format array

# Expected output:
# ✅ Inference completed
#    ├── Agent ID: agent_abc123
#    ├── Input size: 10 (truncated for demo)
#    ├── Output: [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67, 0.89, 0.01, 0.11]
#    ├── Inference time: 45ms
#    ├── Confidence: 0.78
#    └── Predicted class: 3
```

### Step 6: Monitor Performance

```bash
# Check overall status
npx synaptic-mesh status

# Real-time monitoring
npx synaptic-mesh status --watch --refresh 5

# Example status output:
# 🧠 Synaptic Neural Mesh Status
# ═══════════════════════════════════════
# 
# Node Information:
#   ├── Node ID: 12D3KooWAbc123...
#   ├── Version: 1.0.0-alpha.1
#   ├── Network: mainnet
#   ├── Uptime: 5m 23s
#   └── Status: ✅ Operational
# 
# Neural Networks:
#   ├── Active Agents: 1
#   ├── Total Inferences: 1
#   ├── Average Inference Time: 45ms
#   └── Memory Usage: 64MB / 512MB
# 
# Network:
#   ├── Connected Peers: 0 (discovering...)
#   ├── P2P Port: 8080
#   └── Status: ✅ Ready for connections
```

### Step 7: Connect to the Global Mesh

```bash
# Join the global neural mesh
npx synaptic-mesh mesh join \
  --bootstrap /ip4/bootstrap.synaptic-mesh.net/tcp/8080/p2p/12D3KooWBootstrap...

# Or discover peers automatically
npx synaptic-mesh peer discover --timeout 30

# Check connections
npx synaptic-mesh peer list

# Example output:
# 🌐 Connected Peers (3)
# ┌──────────────────┬─────────────────────┬─────────┬────────┐
# │ Peer ID          │ Address             │ Latency │ Uptime │
# ├──────────────────┼─────────────────────┼─────────┼────────┤
# │ 12D3KooWPeer1... │ 192.168.1.100:8080  │ 45ms    │ 2h 15m │
# │ 12D3KooWPeer2... │ 10.0.0.50:8080     │ 67ms    │ 1h 30m │
# │ 12D3KooWPeer3... │ 172.16.0.25:8080   │ 89ms    │ 45m    │
# └──────────────────┴─────────────────────┴─────────┴────────┘
```

### Step 8: Experiment with Multiple Agents

```bash
# Spawn different types of agents
npx synaptic-mesh neural spawn --type lstm --task "sequence_processing"
npx synaptic-mesh neural spawn --type cnn --task "image_processing"

# Check all agents
npx synaptic-mesh neural list

# Run batch inference
npx synaptic-mesh neural batch-inference \
  --agents "agent_abc123,agent_def456" \
  --input-file sample_data.json
```

## 🧪 Practical Examples

### Example 1: Simple Image Classifier

```javascript
// Node.js script: hello-classifier.js
const { SynapticMesh } = require('synaptic-mesh-sdk');

async function imageClassifierExample() {
  // Connect to local mesh
  const mesh = new SynapticMesh({
    baseURL: 'http://localhost:8080'
  });
  
  await mesh.connect();
  
  // Spawn CNN agent for image classification
  const agent = await mesh.neural.spawnAgent({
    type: 'cnn',
    task: 'cifar10_classification',
    architecture: {
      filters: [32, 64, 128],
      kernelSize: 3,
      poolSize: 2,
      denseUnits: 512,
      numClasses: 10
    }
  });
  
  console.log(`✅ Spawned agent: ${agent.id}`);
  
  // Simulate image data (32x32x3 = 3072 values)
  const imageData = Array.from({length: 3072}, () => Math.random());
  
  // Run inference
  const result = await agent.inference(imageData);
  
  console.log('🎯 Classification result:', {
    prediction: result.output.indexOf(Math.max(...result.output)),
    confidence: Math.max(...result.output),
    inferenceTime: result.inferenceTime
  });
  
  await mesh.disconnect();
}

imageClassifierExample().catch(console.error);
```

### Example 2: Time Series Predictor

```javascript
// time-series-predictor.js
async function timeSeriesExample() {
  const mesh = new SynapticMesh({ baseURL: 'http://localhost:8080' });
  await mesh.connect();
  
  // Spawn LSTM for time series
  const agent = await mesh.neural.spawnAgent({
    type: 'lstm',
    task: 'stock_prediction',
    architecture: {
      units: 128,
      layers: 2,
      dropout: 0.2,
      sequenceLength: 60
    }
  });
  
  // Generate sample time series (stock prices)
  const timeSeries = Array.from({length: 60}, (_, i) => 
    100 + 10 * Math.sin(i * 0.1) + Math.random() * 5
  );
  
  const prediction = await agent.inference(timeSeries);
  
  console.log('📈 Stock prediction:', {
    nextPrice: prediction.output[0],
    confidence: prediction.confidence,
    trend: prediction.output[0] > timeSeries[59] ? '↗️ UP' : '↘️ DOWN'
  });
  
  await mesh.disconnect();
}
```

### Example 3: Distributed Learning

```bash
# Start federated learning across multiple agents
npx synaptic-mesh neural train \
  --agents "agent_abc123,agent_def456,agent_ghi789" \
  --strategy federated \
  --dataset ./data/training.json \
  --epochs 50 \
  --batch-size 32

# Monitor training progress
npx synaptic-mesh neural training status --id training_xyz --watch
```

## 🎮 Interactive Experiments

### Experiment 1: Agent Performance Comparison

```bash
# Spawn 3 different agents for the same task
npx synaptic-mesh neural spawn --type mlp --task "benchmark" --id "mlp_agent"
npx synaptic-mesh neural spawn --type lstm --task "benchmark" --id "lstm_agent"  
npx synaptic-mesh neural spawn --type cnn --task "benchmark" --id "cnn_agent"

# Run benchmark comparison
npx synaptic-mesh benchmark compare \
  --agents "mlp_agent,lstm_agent,cnn_agent" \
  --iterations 100 \
  --input-size 784
```

### Experiment 2: Network Effect

```bash
# Test performance with different peer counts
for peers in 0 5 10 15 20; do
  echo "Testing with $peers peers..."
  npx synaptic-mesh benchmark latency --target-peers $peers --duration 60s
done
```

### Experiment 3: Scaling Test

```bash
# Gradually increase agent count and measure performance
for agents in 1 5 10 25 50 100; do
  echo "Scaling to $agents agents..."
  npx synaptic-mesh neural scale --target $agents
  npx synaptic-mesh benchmark throughput --duration 30s
done
```

## 🔧 Configuration Experiments

### Custom Neural Architecture

```json
// .synaptic/custom_config.json
{
  "neural": {
    "customArchitectures": {
      "my_classifier": {
        "type": "mlp",
        "layers": [784, 512, 256, 128, 10],
        "activation": "relu",
        "dropout": 0.3,
        "optimizer": "adam",
        "learningRate": 0.001
      }
    }
  }
}
```

```bash
# Use custom architecture
npx synaptic-mesh neural spawn --architecture my_classifier --task "custom_classification"
```

### Network Topology Experiments

```bash
# Try different mesh topologies
npx synaptic-mesh mesh topology set star --hub-node auto
npx synaptic-mesh mesh topology set ring --clockwise true
npx synaptic-mesh mesh topology set hierarchical --levels 3
npx synaptic-mesh mesh topology set mesh --density 0.7

# Compare performance of each
npx synaptic-mesh benchmark topology --all --duration 120s
```

## 📊 Understanding the Results

### Performance Metrics

- **Inference Time**: How long each neural computation takes
- **Memory Usage**: RAM consumed by each agent
- **Network Latency**: Time for peer-to-peer communication
- **Consensus Time**: How quickly the mesh agrees on state changes
- **Throughput**: Total inferences per second across all agents

### What Good Performance Looks Like

```
✅ Excellent Performance:
- Inference time: <50ms
- Memory per agent: <32MB
- Network latency: <100ms
- Consensus time: <200ms

⚠️ Acceptable Performance:
- Inference time: 50-100ms
- Memory per agent: 32-64MB
- Network latency: 100-200ms
- Consensus time: 200-500ms

❌ Needs Optimization:
- Inference time: >100ms
- Memory per agent: >64MB
- Network latency: >200ms
- Consensus time: >500ms
```

## 🧹 Cleanup

```bash
# Stop all agents
npx synaptic-mesh neural kill --all

# Stop the mesh
npx synaptic-mesh stop

# Clean up data (optional)
rm -rf .synaptic/data/*
```

## 🎯 Next Steps

Now that you've completed your first neural mesh experiment:

1. **Try Advanced Examples**: [Advanced Patterns](../advanced/)
2. **Learn Integration**: [MCP Integration](../integrations/mcp-integration.md)
3. **Optimize Performance**: [Performance Guide](../../guides/performance-optimization.md)
4. **Join the Community**: [Discord](https://discord.gg/synaptic-mesh)

## 🤝 Share Your Results

Share your first neural mesh experience:
- Post your benchmark results in [Discussions](https://github.com/ruvnet/Synaptic-Neural-Mesh/discussions)
- Tweet with #SynapticNeuralMesh
- Contribute improvements via [Pull Requests](https://github.com/ruvnet/Synaptic-Neural-Mesh/pulls)

---

**Congratulations!** 🎉 You've successfully created your first distributed neural network. You're now part of the decentralized AI revolution!