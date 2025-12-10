# P2P Network Integration Demo

This example demonstrates the P2P networking capabilities integrated into the Synaptic Neural Mesh CLI.

## Quick Start

### 1. Initialize P2P Network

```bash
# Initialize with all quantum-resistant features
./synaptic-mesh p2p init --quantum --onion --shadow --max-peers 100

# Output:
# Initializing P2P networking...
# ✓ P2P networking initialized!
# Quantum-resistant: enabled
# Onion routing: enabled  
# Shadow addresses: enabled
# Max peers: 100
```

### 2. Discover Peers

```bash
# Discover peers using Kademlia DHT
./synaptic-mesh p2p discover --method kademlia --count 10

# Output:
# Discovering peers using kademlia method...
# Target count: 10
# ✓ Discovered peers:
#   - peer-0: /ip4/192.168.1.100/tcp/9000
#   - peer-1: /ip4/192.168.1.101/tcp/9000
#   - peer-2: /ip4/192.168.1.102/tcp/9000
#   - peer-3: /ip4/192.168.1.103/tcp/9000
#   - peer-4: /ip4/192.168.1.104/tcp/9000
```

### 3. Create Anonymous Circuit

```bash
# Create 3-hop onion circuit for anonymous communication
./synaptic-mesh p2p circuit peer-target-123 --hops 3

# Output:
# Creating onion circuit to peer-target-123 with 3 hops...
# ✓ Circuit established!
# Circuit ID: 550e8400-e29b-41d4-a716-446655440001
# Hops: node-1 → node-2 → node-3 → peer-target-123
```

### 4. Generate Shadow Address

```bash
# Generate privacy-preserving shadow address
./synaptic-mesh p2p shadow --operation generate

# Output:
# Generating shadow address...
# ✓ Shadow address generated:
# shadow-f47ac10b-58cc-4372-a567-0e02b2c3d479
```

### 5. Establish Quantum-Secure Connection

```bash
# Establish quantum-resistant connection
./synaptic-mesh p2p quantum peer-789 --level 3

# Output:
# Establishing quantum-secure connection to peer-789...
# Security level: 3
# ✓ Quantum-secure connection established!
# Protocol: ML-KEM-768
# Shared secret: ***********
```

### 6. Send Neural Messages

```bash
# Send thought data to another agent
./synaptic-mesh p2p message agent-001 \
    --msg-type thought \
    --content "Processing visual cortex input data" \
    --priority 8

# Output:
# Sending neural message to agent-001...
# Message ID: f47ac10b-58cc-4372-a567-0e02b2c3d479
# Type: thought
# Priority: 8
# ✓ Message sent!
```

```bash
# Send coordination message to swarm
./synaptic-mesh p2p message swarm-alpha \
    --msg-type coordination \
    --content "Synchronizing neural patterns across mesh"

# Output:
# Sending neural message to swarm-alpha...
# Message ID: a1b2c3d4-5e6f-7890-abcd-ef1234567890
# Type: coordination
# Priority: 5
# ✓ Message sent!
```

### 7. NAT Traversal

```bash
# Perform automatic NAT traversal
./synaptic-mesh p2p nat peer-behind-firewall --method auto

# Output:
# Performing NAT traversal for peer: peer-behind-firewall
# Method: auto
# ✓ NAT traversal successful!
# Method used: STUN + TURN relay
# Connection type: Direct (hole punched)
```

### 8. Check Network Status

```bash
# View detailed network status
./synaptic-mesh p2p status --detailed

# Output:
# P2P Network Status
# ─────────────────────────────
# Connected peers: 5
#   - peer-001 [Quantum-secure] /ip4/192.168.1.101/tcp/9000
#   - peer-002 [Shadow: shadow-abc123] /ip4/192.168.1.102/tcp/9000
#   - peer-003 [Onion circuit] /ip4/192.168.1.103/tcp/9000
#   - peer-004 [NAT traversed] /ip4/10.0.0.1/tcp/9000
#   - peer-005 [Direct] /ip4/192.168.1.105/tcp/9000
#
# Active circuits: 2
# Shadow addresses: 3
# Quantum connections: 1
# Data transferred: 1.2 GB
# Messages processed: 1,542
```

## Advanced Usage

### Multi-Node Setup

To demonstrate a full mesh network:

```bash
# Node 1 (Bootstrap node)
./synaptic-mesh p2p init --quantum --max-peers 50
./synaptic-mesh start --daemon

# Node 2 (Connect to bootstrap)
./synaptic-mesh p2p init --quantum
./synaptic-mesh network connect /ip4/192.168.1.100/tcp/9000/p2p/12D3KooW...

# Node 3 (With onion routing)
./synaptic-mesh p2p init --quantum --onion
./synaptic-mesh p2p circuit peer-001 --hops 3
```

### Neural Mesh Communication

```bash
# Agent coordination workflow
./synaptic-mesh p2p message coordinator \
    --msg-type coordination \
    --content "Initiate pattern recognition task"

./synaptic-mesh p2p message worker-001 \
    --msg-type command \
    --content "Analyze dataset partition 1" \
    --priority 9

./synaptic-mesh p2p message worker-002 \
    --msg-type command \
    --content "Analyze dataset partition 2" \
    --priority 9

# Check responses
./synaptic-mesh agent list --detailed
```

### Privacy & Security Demo

```bash
# Full privacy stack
./synaptic-mesh p2p shadow --operation generate
./synaptic-mesh p2p circuit target-node --hops 5
./synaptic-mesh p2p quantum target-node --level 5

# Send anonymous message
./synaptic-mesh p2p message target-node \
    --msg-type thought \
    --content "Confidential neural data"
```

## Architecture Overview

The P2P integration provides:

```
┌─────────────────────────────────────┐
│         CLI Commands                │
├─────────────────────────────────────┤
│  P2PIntegration Module              │
│  ├── Quantum Key Exchange          │
│  ├── Onion Router                  │
│  ├── Shadow Address Manager        │
│  ├── Traffic Obfuscator            │
│  └── Network Manager               │
├─────────────────────────────────────┤
│  QuDAG P2P Infrastructure          │
│  ├── libp2p Transport Layer        │
│  ├── Kademlia DHT                  │
│  ├── GossipSub Messaging           │
│  ├── NAT Traversal                 │
│  └── Request/Response Protocol     │
└─────────────────────────────────────┘
```

## Performance Testing

```bash
# Benchmark message throughput
./synaptic-mesh benchmark --suite p2p --iterations 1000

# Monitor network performance
./synaptic-mesh p2p status --detailed
./synaptic-mesh status --watch
```

## Security Features

- **Quantum Resistance**: ML-KEM-768 post-quantum cryptography
- **Anonymous Routing**: Multi-hop onion circuits
- **Privacy Protection**: Rotating shadow addresses
- **Traffic Obfuscation**: Anti-analysis countermeasures
- **NAT Traversal**: Universal connectivity

## Configuration

The P2P system can be configured via environment variables:

```bash
export SYNAPTIC_MAX_PEERS=100
export SYNAPTIC_QUANTUM_LEVEL=3
export SYNAPTIC_ENABLE_ONION=true
export SYNAPTIC_BOOTSTRAP_PEERS="peer1,peer2,peer3"
```

## Troubleshooting

Common issues and solutions:

```bash
# Connection issues
./synaptic-mesh p2p status
./synaptic-mesh network peers --detailed

# NAT problems
./synaptic-mesh p2p nat peer-id --method stun
./synaptic-mesh p2p nat peer-id --method turn

# Discovery issues
./synaptic-mesh p2p discover --method mdns
```

This integration creates a robust P2P backbone for distributed neural coordination!