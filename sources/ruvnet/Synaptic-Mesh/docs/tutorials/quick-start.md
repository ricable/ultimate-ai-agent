# Quick Start Guide

Get up and running with Synaptic Neural Mesh in under 10 minutes. This guide will walk you through setting up your first distributed neural mesh node and connecting to the global network.

## Prerequisites

Before you begin, ensure you have:

- **Node.js 18+** - [Download here](https://nodejs.org/)
- **NPM 8+** - Comes with Node.js
- **Claude Code** (recommended for AI integration) - [Install guide](https://docs.anthropic.com/claude/docs/claude-code)

### Installing Claude Code (Optional but Recommended)

```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Activate with permissions
claude --dangerously-skip-permissions
```

## Step 1: Initialize Your Neural Mesh

The fastest way to get started is using the alpha release:

```bash
# Initialize with auto-configuration
npx --y synaptic-mesh@alpha init --force

# Or specify a project name
npx --y synaptic-mesh@alpha init my-neural-mesh --template default
```

This command will:
- Create a new project directory
- Generate cryptographic keys
- Set up default configuration
- Install necessary dependencies

### What Gets Created

```
my-neural-mesh/
├── .synaptic/
│   ├── config.json          # Main configuration
│   ├── keys/                # Quantum-resistant keys
│   │   ├── node.key         # Node identity
│   │   └── mesh.key         # Mesh encryption
│   └── data/                # Local storage
├── README.md                # Project documentation
└── package.json             # Dependencies
```

## Step 2: Explore Available Commands

```bash
# Navigate to your project (if you specified a name)
cd my-neural-mesh

# See all available commands
npx synaptic-mesh --help

# Check version
npx synaptic-mesh --version
```

You should see the Synaptic Neural Mesh logo and command overview:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ████████╗██╗   ║
║   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗╚══██╔══╝██║   ║
║   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝   ██║   ██║   ║
║   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝    ██║   ██║   ║
║   ███████║   ██║   ██║ ╚████║██║  ██║██║        ██║   ██║   ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝   ║
║                                                               ║
║              🧠 Neural Mesh - Distributed Intelligence 🧠      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## Step 3: Start Your Neural Mesh Node

Launch your node with the web dashboard enabled:

```bash
# Start with web UI (recommended for beginners)
npx synaptic-mesh start --ui --port 8080

# Or start in the background
npx synaptic-mesh start --daemon --ui
```

You should see output like:
```
🧠 Initializing Synaptic Neural Mesh...
✅ Configuration loaded
✅ Cryptographic keys initialized
✅ P2P networking started on port 8080
✅ DAG consensus layer active
✅ Neural runtime loaded (4 WASM modules)
✅ Web UI available at http://localhost:3000
🌐 Node ID: 12D3KooWAbc123def456...

🚀 Neural Mesh is operational!
```

## Step 4: Access the Web Dashboard

Open your browser and navigate to:
- **Web UI**: http://localhost:3000
- **Metrics**: http://localhost:9090 (if metrics enabled)

The dashboard provides:
- Real-time node status
- Peer connection map
- Neural agent management
- DAG consensus visualization
- Performance metrics

## Step 5: Check Node Status

```bash
# Basic status check
npx synaptic-mesh status

# Detailed status with all components
npx synaptic-mesh status --detailed

# JSON output for scripting
npx synaptic-mesh status --json
```

Example output:
```
🧠 Synaptic Neural Mesh Status
═══════════════════════════════════════

Node Information:
  ├── Node ID: 12D3KooWAbc123...
  ├── Version: 1.0.0-alpha.1
  ├── Network: mainnet
  ├── Uptime: 2m 15s
  └── Status: ✅ Operational

Networking:
  ├── P2P Port: 8080
  ├── Connected Peers: 0/50
  ├── Bootstrap Status: 🔍 Discovering
  └── NAT Status: ✅ Traversed

Neural Networks:
  ├── Active Agents: 0
  ├── Memory Usage: 32MB / 512MB
  └── WASM Modules: 4 loaded

DAG Consensus:
  ├── Network Height: 1
  ├── Pending Transactions: 0
  └── Status: ✅ Synced
```

## Step 6: Spawn Your First Neural Agent

Create a neural agent to start processing tasks:

```bash
# Spawn a basic MLP agent
npx synaptic-mesh neural spawn --type mlp --task "classification"

# Spawn an LSTM for sequence processing
npx synaptic-mesh neural spawn --type lstm --task "time-series-analysis"

# List all active agents
npx synaptic-mesh neural list
```

Example output:
```
✅ Neural agent spawned successfully
   ├── Agent ID: agent_abc123
   ├── Type: mlp
   ├── Task: classification
   ├── Memory: 45MB
   └── Status: Active

Active Neural Agents: 1
```

## Step 7: Join the Global Mesh Network

Connect to other nodes in the global neural mesh:

```bash
# Join via bootstrap peer (example address)
npx synaptic-mesh mesh join /ip4/144.126.223.47/tcp/8080/p2p/12D3KooWBootstrap...

# Or let the node discover peers automatically
npx synaptic-mesh mesh peers --discover
```

Once connected, you'll see:
```
🌐 Joining mesh network...
✅ Connected to bootstrap peer
🔍 Discovering additional peers...
✅ Mesh network joined successfully
   ├── Connected Peers: 3
   ├── Network Height: 45,678
   └── Sync Status: ✅ Synced
```

## Step 8: Explore Advanced Features

### With Claude Flow Integration

If you have Claude Code installed, enable enhanced coordination:

```bash
# Initialize enhanced coordination layer
npx claude-flow@alpha init --force --synaptic-mesh

# Launch coordinated neural swarm
npx claude-flow@alpha hive-mind spawn "distributed learning" --synaptic --agents 8
```

### Monitor Performance

```bash
# Real-time monitoring
npx synaptic-mesh status --watch --refresh 2

# Enable metrics collection
npx synaptic-mesh start --metrics --metrics-port 9090
```

### Distributed Training

```bash
# Start distributed training across the mesh
npx synaptic-mesh neural train --dataset ./data/training.json --epochs 100
```

## Common Tasks

### Managing Configuration

```bash
# View current configuration
npx synaptic-mesh config show

# Change default port
npx synaptic-mesh config set network.port 9090

# Reset to defaults
npx synaptic-mesh config reset
```

### Peer Management

```bash
# List connected peers
npx synaptic-mesh peer list --detailed

# Ping all peers
npx synaptic-mesh peer ping

# Connect to specific peer
npx synaptic-mesh peer connect /ip4/192.168.1.100/tcp/8080/p2p/12D3KooW...
```

### DAG Operations

```bash
# Query recent DAG vertices
npx synaptic-mesh dag query --recent 10

# Submit transaction
npx synaptic-mesh dag submit "Hello Neural Mesh!" --type message

# Check consensus status
npx synaptic-mesh dag status --validators
```

## Stopping Your Node

```bash
# Graceful shutdown
npx synaptic-mesh stop

# Force shutdown
npx synaptic-mesh stop --force

# Quick shutdown with timeout
npx synaptic-mesh stop --timeout 10
```

## Next Steps

🎉 **Congratulations!** You now have a functioning neural mesh node. Here's what to explore next:

### For Developers
1. **[API Reference](../api/api-reference.md)** - Integrate with your applications
2. **[Advanced Patterns](../guides/advanced-patterns.md)** - Complex deployment scenarios
3. **[Integration Examples](../examples/integrations/)** - Real-world integrations

### For Researchers
1. **[Neural Architecture Guide](../guides/neural-architectures.md)** - Custom neural networks
2. **[Distributed Learning](../tutorials/distributed-learning.md)** - Federated training
3. **[Research Templates](../examples/research/)** - Academic use cases

### For Operators
1. **[Production Deployment](../tutorials/production-deployment.md)** - Scale to production
2. **[Monitoring & Observability](../guides/monitoring.md)** - Operational insights
3. **[Security Hardening](../guides/security.md)** - Secure deployments

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
npx synaptic-mesh start --port 8081
```

**Permission denied:**
```bash
sudo npx synaptic-mesh start
# Or change to unprivileged port
npx synaptic-mesh start --port 8080
```

**Network connectivity issues:**
```bash
# Check firewall settings
npx synaptic-mesh peer ping --debug

# Use different discovery method
npx synaptic-mesh start --discovery mdns
```

**For more help:**
- [Common Issues Guide](../troubleshooting/common-issues.md)
- [FAQ](../troubleshooting/faq.md)
- [GitHub Issues](https://github.com/ruvnet/Synaptic-Neural-Mesh/issues)

## Step 9: Explore Synaptic Market (Optional)

If you want to participate in the decentralized Claude-Max marketplace:

⚠️ **IMPORTANT**: You must have your own Claude subscription and credentials

### Market Setup

```bash
# 1. Ensure Claude Code is installed and authenticated
claude login

# 2. Enable market participation
npx synaptic-mesh market init --opt-in

# 3. Set your participation limits
npx synaptic-mesh market config --daily-limit 5 --auto-accept false

# 4. View terms and compliance information
npx synaptic-mesh market --terms
```

### Provider Mode: Offer Your Claude Capacity

```bash
# Advertise available Claude capacity
npx synaptic-mesh market offer --slots 3 --price 5 --min-reputation 0.8

# Monitor your offerings
npx synaptic-mesh market status --provider

# View earnings and statistics
npx synaptic-mesh wallet balance --detailed
```

### Client Mode: Use Distributed Claude Capacity

```bash
# Find available providers
npx synaptic-mesh market browse --max-price 10

# Submit a task bid
npx synaptic-mesh market bid --task "Analyze this code" --max-price 8

# Check bid status
npx synaptic-mesh market bids --active
```

### Market Commands Reference

```bash
# Wallet operations
npx synaptic-mesh wallet balance              # Check ruv token balance
npx synaptic-mesh wallet deposit <amount>     # Add tokens (via faucet/exchange)
npx synaptic-mesh wallet history             # Transaction history

# Market operations  
npx synaptic-mesh market status              # Overall market status
npx synaptic-mesh market reputation         # Your reputation score
npx synaptic-mesh market disputes           # Any active disputes

# Advanced market features
npx synaptic-mesh market escrow list         # View escrowed amounts
npx synaptic-mesh market settle <job-id>     # Manually settle completed job
```

### 🛡️ Security & Compliance

The Synaptic Market operates under strict compliance rules:

- **✅ No account sharing**: Each participant uses their own Claude credentials
- **✅ Local execution**: Claude runs only on your local machine
- **✅ Voluntary participation**: You approve each task individually
- **✅ Full transparency**: Complete audit trail of your Claude usage
- **✅ Privacy preserved**: Task content is encrypted end-to-end

### Market Troubleshooting

**"Market not available":**
```bash
# Check if market crate is properly installed
npx synaptic-mesh market --version

# Verify Claude authentication
claude auth status
```

**"Insufficient tokens":**
```bash
# Get tokens from faucet (testnet)
npx synaptic-mesh wallet faucet

# Check for pending earnings
npx synaptic-mesh market earnings --pending
```

**"Job execution failed":**
```bash
# Check Docker status
docker version

# Verify Claude container access
npx synaptic-mesh market test-execution
```

## What's Next?

- **Join the Community**: [Discord](https://discord.gg/synaptic-mesh)
- **Contribute**: [Contributing Guide](../../CONTRIBUTING.md)
- **Stay Updated**: [GitHub Releases](https://github.com/ruvnet/Synaptic-Neural-Mesh/releases)
- **Market Guide**: [Advanced Market Usage](../examples/advanced/market-strategies.md)

---

**Ready to dive deeper?** Check out our [First Neural Mesh Tutorial](first-neural-mesh.md) to build more complex distributed intelligence systems!