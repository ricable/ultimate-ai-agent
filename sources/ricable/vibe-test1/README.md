# Ericsson RAN Autonomous Swarm Optimization Platform

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██████╗ ██╗ ██████╗███████╗███████╗ ██████╗ ███╗   ██╗              ║
║   ██╔════╝██╔══██╗██║██╔════╝██╔════╝██╔════╝██╔═══██╗████╗  ██║              ║
║   █████╗  ██████╔╝██║██║     ███████╗███████╗██║   ██║██╔██╗ ██║              ║
║   ██╔══╝  ██╔══██╗██║██║     ╚════██║╚════██║██║   ██║██║╚██╗██║              ║
║   ███████╗██║  ██║██║╚██████╗███████║███████║╚██████╔╝██║ ╚████║              ║
║   ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝              ║
║                                                                               ║
║   ██████╗  █████╗ ███╗   ██╗    ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗ ║
║   ██╔══██╗██╔══██╗████╗  ██║    ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║ ║
║   ██████╔╝███████║██╔██╗ ██║    ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║ ║
║   ██╔══██╗██╔══██║██║╚██╗██║    ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║ ║
║   ██║  ██║██║  ██║██║ ╚████║    ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║ ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Overview

A next-generation, decentralized, autonomous **Federated Swarm Architecture** for Ericsson RAN optimization. This platform deploys intelligent, self-learning agents directly to the network edge—residing on gNodeB and eNodeB compute units—enabling autonomous Configuration Management (CM), Performance Management (PM), and Fault Management (FM).

## Key Features

### Core Engines (Ruvnet Ecosystem)

| Component | Version | Description |
|-----------|---------|-------------|
| **AgentDB** | v1.6.1 | Cognitive Engine with ReasoningBank, Causal Graphs, and Decision Transformer for offline RL |
| **Ruvector** | v0.1.26 | Spatio-Temporal GNN with Flash Attention for interference pattern analysis |
| **Midstreamer** | v0.2.3 | Temporal Engine with DTW, Chaos Analysis, and N-BEATS forecasting |
| **RuvSwarm** | v1.0.20 | Swarm Orchestration with Federated Learning (FedAvg/FedProx/FedNova) |
| **AIDefence** | v0.1.6 | Security layer with adversarial detection and guardrails |

### RAN Optimization Modules

- **Configuration Management (CM)**
  - Uplink Power Control (P0, Alpha, PUSCH, PUCCH)
  - Decision Transformer-based RL optimization
  - Slice-aware parameter tuning (eMBB, URLLC, mMTC)
  - Safety guardrails with rate limiting

- **Performance Management (PM)**
  - Multi-granularity analysis (15-min, hourly, daily, weekly)
  - Attractor-based anomaly detection
  - Capacity planning and forecasting
  - Correlation analysis for correlation-break detection

- **Fault Management (FM)**
  - 9 problem categories (Uplink Interference, Pilot Pollution, Coverage Hole, etc.)
  - Causal Graph-based Root Cause Analysis
  - Automated Self-Healing Workflows
  - Compensation actions for neighbors

### Architecture Highlights

- **Nano-Agent Design**: Each cell operates as an autonomous agent with local memory and reasoning
- **Federated Learning**: Privacy-preserving model aggregation across the swarm
- **Edge-Native**: Designed for WASM deployment on baseband units
- **Zero-Touch Operations**: Autonomous learning and optimization without human intervention

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ericsson-ran-swarm.git
cd ericsson-ran-swarm

# Install dependencies
npm install

# Build the project
npm run build
```

## Quick Start

```typescript
import { NanoAgent, createSwarm, SwarmSimulator } from 'ericsson-ran-swarm';

// Initialize simulator
const simulator = new SwarmSimulator({
  numCells: 19,
  numUsers: 1000,
  scenario: 'urban',
  trafficProfile: 'hotspot'
});
simulator.initialize();

// Create swarm of agents
const cells = simulator.getAllCellKpis().map(k => k.cgi);
const agents = createSwarm(cells, 'cluster-1');

// Process KPIs and run optimization
for (const [cellId, agent] of agents) {
  const kpi = simulator.getCellKpis(cellId);
  agent.processKPI(kpi);
  agent.runOptimization(kpi, neighborKpis, params, 'eMBB');
}
```

## Running the Demo

```bash
npm run simulate
```

This runs a complete simulation demonstrating:
1. Cell grid initialization with realistic RF propagation
2. User mobility and traffic patterns
3. Fault injection and self-healing
4. Federated learning across the swarm
5. Real-time KPI monitoring

## Project Structure

```
ericsson-ran-swarm/
├── src/
│   ├── core/
│   │   ├── agentdb/        # Cognitive engine (ReasoningBank, Decision Transformer)
│   │   ├── ruvector/       # Spatial engine (ST-GNN, attention mechanisms)
│   │   ├── midstreamer/    # Temporal engine (DTW, chaos analysis)
│   │   └── ruv_swarm/      # Orchestration (federation, leader election)
│   ├── ran/
│   │   ├── cm/             # Configuration Management (uplink optimizer)
│   │   ├── pm/             # Performance Management (KPI analysis)
│   │   └── fm/             # Fault Management (RCA, self-healing)
│   ├── security/           # AIDefence layer
│   ├── simulation/         # Network simulator
│   └── types/              # TypeScript type definitions
├── config/                 # Configuration files
├── scripts/                # Demo and utility scripts
└── tests/                  # Unit and integration tests
```

## Technical Details

### Uplink Power Control Physics

The optimizer implements 3GPP TS 38.213 power control:

```
P_PUSCH = min{P_CMAX, P_0 + α·PL + Δ_TF + f(i)}
```

Where:
- **P_0**: Target received power at gNodeB (-126 to -60 dBm)
- **α (Alpha)**: Pathloss compensation factor (0.4 to 1.0)
- **PL**: Measured pathloss
- **Δ_TF**: MCS-based offset

### ST-GNN Architecture

```
Message Passing: h_v^{(k+1)} = φ(h_v^{(k)}, Σ α_vu·ψ(h_u^{(k)}))

Attention Types: GAT, GATv2, Flash Attention, Linear Attention
Temporal: LSTM encoding of historical KPIs
Hypergraph: Interference cluster modeling
```

### Federated Learning

```
FedAvg: w^{t+1} = Σ (n_k/n) · w_k^{t+1}

Privacy: Differential Privacy (ε-δ)
Compression: Top-K sparsification
Byzantine: Trust-score based filtering
```

## Configuration

See `config/swarm-config.json` for full configuration options:

```json
{
  "swarm": {
    "topology": "mesh",
    "consensusProtocol": "strange-loops"
  },
  "federation": {
    "algorithm": "FedAvg",
    "minClientsPerRound": 3,
    "differentialPrivacy": {
      "enabled": true,
      "epsilon": 1.0
    }
  },
  "optimization": {
    "rl": {
      "algorithm": "DecisionTransformer",
      "contextLength": 20
    },
    "safetyGuardrails": {
      "maxPowerDbm": 46,
      "minAlpha": 0.4
    }
  }
}
```

## API Reference

### NanoAgent

```typescript
const agent = new NanoAgent(cellGlobalIdentity);

// Process KPI data
agent.processKPI(kpi: CellKPIs);

// Update spatial context
agent.updateSpatialContext(graph: RANGraph);

// Run optimization cycle
agent.runOptimization(kpi, neighborKpis, params, sliceType);

// Get statistics
const stats = agent.getStats();
```

### SwarmSimulator

```typescript
const sim = new SwarmSimulator(config);
sim.initialize();
sim.start();  // Continuous simulation
sim.step();   // Single step

const graph = sim.getRANGraph();
const kpis = sim.getAllCellKpis();
sim.applyParameterChange(cellId, params);
```

## Problem Categories

| Category | Indicators | Auto-Healing |
|----------|------------|--------------|
| UPLINK_INTERFERENCE | High IoT, Low SINR | Adjust P0/Alpha |
| PILOT_POLLUTION | Low SINR, High RSRP | Tilt adjustment |
| COVERAGE_HOLE | Low RSRP | Power/Tilt increase |
| SLEEPING_CELL | Zero traffic | Cell restart |
| PCI_CONFLICT | HO failures | PCI change |
| CAPACITY_SATURATION | High PRB utilization | Carrier activation |

## Security Features

- **Adversarial Detection**: Statistical and physics-based input validation
- **Safety Guardrails**: Parameter range enforcement, rate limiting
- **Model Provenance**: Blockchain-inspired tracking of federated updates
- **Byzantine Tolerance**: Trust-score based contributor filtering

## Contributing

See CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file.

## References

- 3GPP TS 38.213: NR Physical layer procedures for control
- 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling"
