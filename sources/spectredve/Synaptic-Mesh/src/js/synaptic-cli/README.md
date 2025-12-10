# Synaptic Neural Mesh CLI

Revolutionary AI orchestration with neural mesh topology - forked from claude-flow for specialized neural mesh applications.

## 🚀 Quick Start

### Installation

```bash
# Install globally
npm install -g synaptic-mesh

# Or use with npx
npx synaptic-mesh --help
```

### Basic Usage

```bash
# Initialize a new project
synaptic init my-neural-project

# Start the mesh
synaptic start

# Check status
synaptic status

# Add nodes to the mesh
synaptic mesh add --type researcher --role "data-analyst"

# Create and train neural models
synaptic neural create my-model --architecture transformer
synaptic neural train my-model --dataset ./data/training.json

# Create DAG workflows
synaptic dag create workflow1 --file ./workflows/analysis.json
synaptic dag run workflow1 --watch

# Connect to peers
synaptic peer discover
synaptic peer connect /ip4/192.168.1.100/tcp/7073/p2p/QmPeerID
```

## 📋 Commands

### Core Commands

- `synaptic init` - Initialize new neural mesh project
- `synaptic start` - Start the neural mesh
- `synaptic status` - Show mesh status and health
- `synaptic config` - Manage configuration

### Mesh Management

- `synaptic mesh list` - List mesh nodes
- `synaptic mesh add` - Add nodes to mesh
- `synaptic mesh connect` - Connect nodes
- `synaptic mesh topology` - Visualize mesh structure
- `synaptic mesh optimize` - Optimize topology

### Neural Networks

- `synaptic neural list` - List neural models
- `synaptic neural create` - Create new model
- `synaptic neural train` - Train models
- `synaptic neural evaluate` - Evaluate model performance
- `synaptic neural predict` - Make predictions

### DAG Workflows

- `synaptic dag list` - List workflows
- `synaptic dag create` - Create new workflow
- `synaptic dag run` - Execute workflow
- `synaptic dag visualize` - Visualize workflow

### P2P Network

- `synaptic peer list` - List connected peers
- `synaptic peer connect` - Connect to peers
- `synaptic peer discover` - Discover nearby peers
- `synaptic peer share` - Share files
- `synaptic peer download` - Download from network

### Advanced

- `synaptic mcp` - Start MCP server
- `synaptic swarm` - Manage neural swarms
- `synaptic export` - Export configurations
- `synaptic import` - Import configurations

## 🏗️ Architecture

### Neural Mesh Topology

The Synaptic Neural Mesh supports multiple topology types:

- **Mesh**: Full connectivity for maximum redundancy
- **Hierarchical**: Tree-like structure for organized processing
- **Ring**: Circular connections for sequential processing
- **Star**: Hub-and-spoke for centralized coordination

### Core Services

1. **Mesh Coordination** (Port 7070) - Node coordination and topology management
2. **Neural Network** (Port 7071) - Model training and inference
3. **DAG Workflows** (Port 7072) - Workflow execution engine
4. **P2P Network** (Port 7073) - Peer-to-peer communication
5. **MCP Server** (Port 3000) - Model Context Protocol integration

## 📁 Project Structure

```
.synaptic/
├── config.json          # Main configuration
├── agents/              # Agent definitions
├── memory/              # Persistent memory
├── sessions/            # Session data
├── logs/               # Application logs
├── data/               # Datasets and models
├── workflows/          # DAG workflow definitions
└── certificates/       # Security certificates
```

## ⚙️ Configuration

### Basic Configuration

```json
{
  "project": {
    "name": "my-neural-project",
    "template": "advanced"
  },
  "mesh": {
    "topology": "mesh",
    "defaultAgents": 5
  },
  "neural": {
    "enabled": true,
    "defaultModel": "transformer"
  },
  "peer": {
    "autoDiscovery": true,
    "maxPeers": 50
  },
  "features": {
    "mcp": true,
    "webui": true,
    "monitoring": true
  }
}
```

### Environment Variables

- `SYNAPTIC_CONFIG_PATH` - Custom config directory
- `SYNAPTIC_LOG_LEVEL` - Logging level (debug, info, warn, error)
- `SYNAPTIC_PORT` - Override default coordination port
- `SYNAPTIC_HOST` - Override default host binding

## 🧠 Neural Models

### Supported Architectures

- **MLP** - Multi-layer Perceptron
- **CNN** - Convolutional Neural Network
- **RNN** - Recurrent Neural Network
- **Transformer** - Attention-based models

### Model Training

```bash
# Create a classification model
synaptic neural create sentiment-model \
  --type classification \
  --architecture transformer \
  --layers 512,256,128

# Train with custom parameters
synaptic neural train sentiment-model \
  --dataset ./data/sentiment.json \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001

# Evaluate performance
synaptic neural evaluate sentiment-model \
  --dataset ./data/test.json \
  --metrics accuracy,precision,recall,f1
```

## 🔄 DAG Workflows

### Workflow Definition

```json
{
  "name": "data-processing-pipeline",
  "description": "Process and analyze data",
  "nodes": [
    {
      "id": "load-data",
      "type": "loader",
      "config": { "source": "./data/input.csv" }
    },
    {
      "id": "preprocess",
      "type": "transformer",
      "config": { "operations": ["normalize", "encode"] }
    },
    {
      "id": "analyze",
      "type": "neural",
      "config": { "model": "sentiment-model" }
    },
    {
      "id": "save-results",
      "type": "saver",
      "config": { "destination": "./results/" }
    }
  ],
  "edges": [
    { "from": "load-data", "to": "preprocess" },
    { "from": "preprocess", "to": "analyze" },
    { "from": "analyze", "to": "save-results" }
  ]
}
```

## 🌐 P2P Network

### Peer Discovery

The mesh automatically discovers peers using:

- **mDNS** - Local network discovery
- **DHT** - Distributed hash table
- **Bootstrap nodes** - Known peer addresses

### Data Sharing

```bash
# Share a dataset
synaptic peer share ./data/training.json --replicas 3

# Download shared data
synaptic peer download QmHash123... --output ./downloaded-data.json
```

## 🔧 Development

### Building from Source

```bash
git clone https://github.com/synaptic-neural-mesh/synaptic-cli.git
cd synaptic-cli
npm install
npm run build
```

### Running Tests

```bash
npm test
npm run test:coverage
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📊 Monitoring & Debugging

### Status Dashboard

```bash
# Real-time status monitoring
synaptic status --watch

# Performance metrics
synaptic status --metrics
```

### Logging

```bash
# View logs
tail -f .synaptic/logs/synaptic.log

# Debug mode
synaptic start --debug

# Verbose logging
SYNAPTIC_LOG_LEVEL=debug synaptic start
```

## 🔐 Security

### Features

- **End-to-end encryption** for peer communication
- **Certificate-based authentication** for trusted nodes
- **Access control** for sensitive operations
- **Audit logging** for compliance

### Configuration

```json
{
  "security": {
    "encryption": true,
    "authentication": true,
    "certificates": {
      "autoGenerate": true,
      "keySize": 2048
    }
  }
}
```

## 🚀 Deployment

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 7070 7071 7072 7073 3000
CMD ["synaptic", "start"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synaptic-mesh
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synaptic-mesh
  template:
    metadata:
      labels:
        app: synaptic-mesh
    spec:
      containers:
      - name: synaptic
        image: synaptic-mesh:latest
        ports:
        - containerPort: 7070
        env:
        - name: SYNAPTIC_CONFIG_PATH
          value: "/config"
```

## 📚 Examples

See the `examples/` directory for:

- Basic neural mesh setup
- Advanced workflow pipelines
- P2P network configurations
- Custom neural architectures
- Integration patterns

## 🤝 Support

- **Documentation**: [docs.synaptic-mesh.org](https://docs.synaptic-mesh.org)
- **Issues**: [GitHub Issues](https://github.com/synaptic-neural-mesh/synaptic-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/synaptic-neural-mesh/synaptic-cli/discussions)
- **Discord**: [Community Server](https://discord.gg/synaptic-mesh)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the foundation of [claude-flow](https://github.com/ruvnet/claude-flow)
- Inspired by distributed neural network research
- Community contributions and feedback

---

**Synaptic Neural Mesh CLI** - Revolutionizing AI orchestration through neural mesh topology.