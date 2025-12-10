# Synaptic Neural Mesh CLI Reference

Complete command-line interface reference for the Synaptic Neural Mesh distributed neural fabric.

## Global Options

All commands support these global options:

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --debug` | Enable debug mode with verbose output | `false` |
| `-q, --quiet` | Suppress non-essential output | `false` |
| `--no-color` | Disable colored output | `false` |
| `--config <path>` | Specify custom config file path | `.synaptic/config.json` |
| `-v, --version` | Display version information | - |
| `-h, --help` | Display help information | - |

## Commands Overview

```bash
synaptic-mesh <command> [options]
```

| Command | Description | Status |
|---------|-------------|--------|
| [`init`](#init) | Initialize new neural mesh node | ✅ Active |
| [`start`](#start) | Start neural mesh node | ✅ Active |
| [`stop`](#stop) | Stop neural mesh node | ✅ Active |
| [`status`](#status) | Display mesh status | ✅ Active |
| [`mesh`](#mesh) | Manage mesh topology | ✅ Active |
| [`neural`](#neural) | Manage neural networks | ✅ Active |
| [`dag`](#dag) | Manage DAG consensus | ✅ Active |
| [`peer`](#peer) | Manage peer connections | ✅ Active |
| [`config`](#config) | Manage configuration | ✅ Active |

---

## init

Initialize a new neural mesh node with optional project template.

### Usage
```bash
synaptic-mesh init [project-name] [options]
```

### Arguments
- `project-name` - Name of the project directory (optional, defaults to current directory)

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `-t, --template <type>` | Project template to use | `default` |
| `-p, --port <number>` | Default port for mesh networking | `8080` |
| `-n, --network <id>` | Network ID to join | `mainnet` |
| `--docker` | Include Docker configuration | `false` |
| `--k8s` | Include Kubernetes manifests | `false` |
| `--force` | Overwrite existing files | `false` |
| `--no-deps` | Skip dependency installation | `false` |

### Templates
| Template | Description | Use Case |
|----------|-------------|----------|
| `default` | Basic neural mesh setup | General purpose development |
| `enterprise` | Production-ready configuration | Enterprise deployments |
| `research` | Research-oriented setup | Academic and R&D projects |
| `edge` | Lightweight edge computing | IoT and edge devices |
| `minimal` | Minimal configuration | Testing and experimentation |

### Examples
```bash
# Initialize with default template
synaptic-mesh init my-neural-mesh

# Enterprise setup with Docker and K8s
synaptic-mesh init enterprise-mesh --template enterprise --docker --k8s

# Research setup on custom port
synaptic-mesh init research-project --template research --port 9090

# Edge computing setup
synaptic-mesh init edge-node --template edge --network edge-testnet
```

### Generated Files
```
project-name/
├── .synaptic/
│   ├── config.json          # Main configuration
│   ├── keys/                # Cryptographic keys
│   └── data/                # Local data storage
├── docker-compose.yml       # Docker setup (if --docker)
├── k8s/                     # Kubernetes manifests (if --k8s)
├── README.md                # Project documentation
└── package.json             # Node.js dependencies
```

---

## start

Start the neural mesh node with specified configuration.

### Usage
```bash
synaptic-mesh start [options]
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `-p, --port <number>` | Port for P2P networking | `8080` |
| `-b, --bind <address>` | Bind address | `0.0.0.0` |
| `-n, --network <id>` | Network ID | `mainnet` |
| `--bootstrap <peers>` | Bootstrap peer addresses | - |
| `--ui` | Enable web UI dashboard | `false` |
| `--ui-port <number>` | Web UI port | `3000` |
| `--mcp` | Enable MCP server for AI assistants | `false` |
| `--mcp-stdio` | Use stdio transport for MCP | `false` |
| `--daemon` | Run as background daemon | `false` |
| `--log-level <level>` | Log level (error, warn, info, debug) | `info` |
| `--metrics` | Enable metrics collection | `false` |
| `--metrics-port <number>` | Metrics server port | `9090` |

### Examples
```bash
# Start with default settings
synaptic-mesh start

# Start with web UI on custom port
synaptic-mesh start --port 8080 --ui --ui-port 3000

# Start with bootstrap peers
synaptic-mesh start --bootstrap "/ip4/192.168.1.100/tcp/8080/p2p/12D3KooW..."

# Start with MCP integration for AI assistants
synaptic-mesh start --mcp --mcp-stdio

# Production setup with metrics
synaptic-mesh start --daemon --metrics --log-level warn
```

### Startup Process
1. **Configuration Loading** - Load and validate configuration files
2. **Key Management** - Generate or load cryptographic keys
3. **P2P Initialization** - Initialize libp2p networking stack
4. **DAG Setup** - Initialize QuDAG consensus layer
5. **Neural Runtime** - Load WASM neural network modules
6. **Peer Discovery** - Connect to bootstrap peers or discover via mDNS
7. **Ready State** - Node operational and accepting connections

---

## stop

Stop the running neural mesh node gracefully.

### Usage
```bash
synaptic-mesh stop [options]
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `-f, --force` | Force immediate shutdown | `false` |
| `--timeout <seconds>` | Graceful shutdown timeout | `30` |
| `--save-state` | Save current state before shutdown | `true` |

### Examples
```bash
# Graceful shutdown
synaptic-mesh stop

# Force shutdown
synaptic-mesh stop --force

# Quick shutdown with short timeout
synaptic-mesh stop --timeout 10
```

### Shutdown Process
1. **Signal Handling** - Catch shutdown signals
2. **Connection Cleanup** - Close peer connections gracefully
3. **State Persistence** - Save current DAG state and neural models
4. **Resource Cleanup** - Free WASM modules and memory
5. **Exit** - Clean process termination

---

## status

Display detailed information about the mesh node status.

### Usage
```bash
synaptic-mesh status [options]
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `-j, --json` | Output in JSON format | `false` |
| `-w, --watch` | Continuous monitoring mode | `false` |
| `--refresh <seconds>` | Refresh interval for watch mode | `5` |
| `--detailed` | Show detailed component status | `false` |

### Examples
```bash
# Basic status
synaptic-mesh status

# JSON output for scripting
synaptic-mesh status --json

# Continuous monitoring
synaptic-mesh status --watch --refresh 2

# Detailed status
synaptic-mesh status --detailed
```

### Status Information
```
🧠 Synaptic Neural Mesh Status
═══════════════════════════════════════

Node Information:
  ├── Node ID: 12D3KooWAbc123...
  ├── Version: 1.0.0-alpha.1
  ├── Network: mainnet
  ├── Uptime: 2h 34m 12s
  └── Status: ✅ Operational

Networking:
  ├── P2P Port: 8080
  ├── Connected Peers: 15/50
  ├── Bootstrap Status: ✅ Connected
  └── NAT Status: ✅ Traversed

Neural Networks:
  ├── Active Agents: 8
  ├── Total Agents Spawned: 23
  ├── Memory Usage: 127MB / 512MB
  └── Average Inference Time: 67ms

DAG Consensus:
  ├── Network Height: 45,678
  ├── Pending Transactions: 3
  ├── Consensus Participants: 12
  └── Last Block: 2 seconds ago

Performance:
  ├── CPU Usage: 15.3%
  ├── Memory Usage: 234MB
  ├── Network I/O: ↓ 2.3MB/s ↑ 1.1MB/s
  └── Disk Usage: 1.2GB
```

---

## mesh

Manage mesh topology and network operations.

### Usage
```bash
synaptic-mesh mesh <subcommand> [options]
```

### Subcommands

#### `mesh join`
Join an existing mesh network.

```bash
synaptic-mesh mesh join <peer-address> [options]
```

**Options:**
- `--timeout <seconds>` - Connection timeout (default: `30`)
- `--retry <count>` - Retry attempts (default: `3`)

**Examples:**
```bash
# Join via peer address
synaptic-mesh mesh join /ip4/192.168.1.100/tcp/8080/p2p/12D3KooW...

# Join with timeout
synaptic-mesh mesh join <peer-address> --timeout 60
```

#### `mesh leave`
Leave the current mesh network.

```bash
synaptic-mesh mesh leave [options]
```

**Options:**
- `--graceful` - Perform graceful disconnect (default: `true`)
- `--save-peers` - Save peer list for reconnection (default: `true`)

#### `mesh topology`
Display or modify mesh topology.

```bash
synaptic-mesh mesh topology [type] [options]
```

**Types:**
- `star` - Star topology (central hub)
- `mesh` - Full mesh topology
- `ring` - Ring topology
- `hierarchical` - Hierarchical topology

**Options:**
- `--max-peers <number>` - Maximum peer connections
- `--optimize` - Auto-optimize topology

#### `mesh peers`
List and manage connected peers.

```bash
synaptic-mesh mesh peers [action] [options]
```

**Actions:**
- `list` - List all connected peers (default)
- `add` - Add a peer manually
- `remove` - Remove a peer
- `ping` - Ping all peers

---

## neural

Manage neural networks and AI agents.

### Usage
```bash
synaptic-mesh neural <subcommand> [options]
```

### Subcommands

#### `neural spawn`
Spawn a new neural agent.

```bash
synaptic-mesh neural spawn [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-t, --type <architecture>` | Neural architecture type | `mlp` |
| `--task <description>` | Task description for the agent | - |
| `--memory <size>` | Memory limit for agent | `50MB` |
| `--timeout <seconds>` | Agent timeout | `3600` |
| `--replicas <count>` | Number of replicas to spawn | `1` |

**Architectures:**
- `mlp` - Multi-Layer Perceptron
- `lstm` - Long Short-Term Memory
- `cnn` - Convolutional Neural Network
- `transformer` - Transformer architecture
- `custom` - Custom architecture from file

**Examples:**
```bash
# Spawn basic MLP agent
synaptic-mesh neural spawn --type mlp --task "classification"

# Spawn LSTM for sequence processing
synaptic-mesh neural spawn --type lstm --task "time-series" --memory 100MB

# Spawn multiple replicas
synaptic-mesh neural spawn --type cnn --replicas 5
```

#### `neural list`
List active neural agents.

```bash
synaptic-mesh neural list [options]
```

**Options:**
- `--detailed` - Show detailed agent information
- `--filter <status>` - Filter by status (active, idle, training)

#### `neural kill`
Terminate neural agents.

```bash
synaptic-mesh neural kill <agent-id> [options]
```

**Options:**
- `--force` - Force termination without graceful shutdown
- `--all` - Terminate all agents

#### `neural train`
Initiate distributed training.

```bash
synaptic-mesh neural train [options]
```

**Options:**
- `--dataset <path>` - Training dataset path
- `--epochs <number>` - Number of training epochs
- `--batch-size <size>` - Training batch size
- `--learning-rate <rate>` - Learning rate

---

## dag

Manage DAG consensus and transactions.

### Usage
```bash
synaptic-mesh dag <subcommand> [options]
```

### Subcommands

#### `dag query`
Query DAG state and transactions.

```bash
synaptic-mesh dag query [options]
```

**Options:**
- `--id <vertex-id>` - Query specific vertex
- `--height <number>` - Query by height
- `--tx <hash>` - Query transaction by hash
- `--recent <count>` - Show recent vertices

#### `dag submit`
Submit a new transaction to the DAG.

```bash
synaptic-mesh dag submit <data> [options]
```

**Options:**
- `--type <type>` - Transaction type
- `--fee <amount>` - Transaction fee
- `--priority <level>` - Priority level (low, normal, high)

#### `dag status`
Show DAG consensus status.

```bash
synaptic-mesh dag status [options]
```

**Options:**
- `--validators` - Show validator information
- `--consensus` - Show consensus algorithm details

---

## peer

Manage peer connections and discovery.

### Usage
```bash
synaptic-mesh peer <subcommand> [options]
```

### Subcommands

#### `peer list`
List connected peers.

```bash
synaptic-mesh peer list [options]
```

**Options:**
- `--detailed` - Show detailed peer information
- `--sort <field>` - Sort by field (id, address, uptime, latency)

#### `peer connect`
Connect to a specific peer.

```bash
synaptic-mesh peer connect <address> [options]
```

#### `peer disconnect`
Disconnect from a peer.

```bash
synaptic-mesh peer disconnect <peer-id> [options]
```

#### `peer ping`
Ping peers to check connectivity.

```bash
synaptic-mesh peer ping [peer-id] [options]
```

---

## config

Manage node configuration.

### Usage
```bash
synaptic-mesh config <subcommand> [options]
```

### Subcommands

#### `config show`
Display current configuration.

```bash
synaptic-mesh config show [options]
```

**Options:**
- `--section <name>` - Show specific section
- `--json` - Output in JSON format

#### `config set`
Set configuration values.

```bash
synaptic-mesh config set <key> <value> [options]
```

#### `config get`
Get configuration values.

```bash
synaptic-mesh config get <key> [options]
```

#### `config validate`
Validate configuration file.

```bash
synaptic-mesh config validate [options]
```

#### `config reset`
Reset configuration to defaults.

```bash
synaptic-mesh config reset [options]
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SYNAPTIC_CONFIG` | Configuration file path | `.synaptic/config.json` |
| `SYNAPTIC_DATA_DIR` | Data directory path | `.synaptic/data` |
| `SYNAPTIC_LOG_LEVEL` | Log level | `info` |
| `SYNAPTIC_NETWORK` | Default network | `mainnet` |
| `SYNAPTIC_PORT` | Default P2P port | `8080` |

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error |
| `2` | Configuration error |
| `3` | Network error |
| `4` | Permission error |
| `5` | Resource error |

## Examples

### Basic Workflow
```bash
# 1. Initialize a new mesh node
synaptic-mesh init my-mesh --template default

# 2. Start the node
synaptic-mesh start --ui

# 3. Check status
synaptic-mesh status

# 4. Spawn neural agents
synaptic-mesh neural spawn --type mlp --task "classification"

# 5. Join mesh network
synaptic-mesh mesh join /ip4/192.168.1.100/tcp/8080/p2p/12D3KooW...
```

### Production Deployment
```bash
# Initialize with enterprise template
synaptic-mesh init production-mesh --template enterprise --docker --k8s

# Start with full monitoring
synaptic-mesh start --daemon --metrics --log-level warn --ui

# Configure for high availability
synaptic-mesh config set mesh.maxPeers 100
synaptic-mesh config set neural.maxAgents 1000
```

### Research Setup
```bash
# Initialize research environment
synaptic-mesh init research --template research

# Start with debugging enabled
synaptic-mesh start --debug --ui

# Spawn various neural architectures
synaptic-mesh neural spawn --type lstm --task "sequence-learning"
synaptic-mesh neural spawn --type cnn --task "image-processing"
synaptic-mesh neural spawn --type transformer --task "nlp-research"
```

---

**Need help?** Check our [troubleshooting guide](../troubleshooting/common-issues.md) or visit the [GitHub repository](https://github.com/ruvnet/Synaptic-Neural-Mesh).