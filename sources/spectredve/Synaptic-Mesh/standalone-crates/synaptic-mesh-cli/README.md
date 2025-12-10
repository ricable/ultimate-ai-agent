# Synaptic Mesh CLI

**Development prototype** for the Synaptic Neural Mesh project. This CLI provides basic command structure and placeholder implementations for distributed AI concepts.

**⚠️ CURRENT STATUS: Early Development (~20% Functional)**

## Current Features

- **CLI Structure**: Command framework with help system *(✅ working)*
- **Basic Commands**: Command parsing and routing *(✅ working)*  
- **Placeholder Responses**: Mock implementations for testing *(⚠️ not real functionality)*
- **Integration Framework**: Structures for future component integration *(🚧 incomplete)*
- **Kimi-FANN Core**: Fully functional neural inference engine *(✅ working - see crates/kimi-fann-core)*

## Planned Features (Not Yet Implemented)

- **Synaptic Market**: Decentralized Claude-Max marketplace *(placeholder commands only)*
- **P2P Operations**: Launch and manage mesh nodes *(returns hardcoded responses)*
- **Neural Networks**: Train and run WASM neural networks *(mock implementation)*
- **Swarm Management**: Control distributed agent swarms *(not implemented)*
- **QuDAG Networking**: Quantum-resistant DAG operations *(basic structure only)*
- **Token Wallet**: Manage RUV tokens *(returns hardcoded balance of 1000)*

## Kimi-FANN Core (Fully Functional!)

While the main CLI is still in development, the **Kimi-FANN Core** component is fully functional and published on crates.io:

```bash
# Install Kimi neural inference engine
cargo install kimi-fann-core

# Use it immediately
kimi "What is machine learning?"
kimi --expert coding "Write a bubble sort"
kimi --consensus "Design a neural network"
```

See [crates/kimi-fann-core](crates/kimi-fann-core) for full documentation.

## Development Installation

```bash
# Clone the repository
git clone https://github.com/ruvnet/Synaptic-Neural-Mesh
cd Synaptic-Neural-Mesh/standalone-crates/synaptic-mesh-cli

# Build from source (development only)
cargo build

# Run with placeholder functionality
cargo run -- --help
```

**Note**: The main synaptic-mesh-cli crate is not published to crates.io as it's not yet functional. However, kimi-fann-core is fully functional and available on crates.io.

## Current Usage (Placeholder Responses)

### Basic Commands (Return Hardcoded Values)
```bash
# These commands work but return placeholder responses:
cargo run -- node start --port 8080        # Returns "Node started" message
cargo run -- wallet balance                # Always returns balance: 1000
cargo run -- market status                 # Returns mock market data
```

### What Actually Happens
```bash
# Example of placeholder behavior:
$ cargo run -- wallet balance
✅ Wallet Balance: 1000 RUV tokens

# This is hardcoded - there's no real wallet implementation
```

### Development Testing
```bash
# Run tests (mostly structure validation)
cargo test

# Check command parsing
cargo run -- --help

# Explore placeholder implementations
cargo run -- market offer --slots 5 --price 10 --opt-in
# Returns: "Market offer created" (but no real marketplace exists)
```

## Library Usage

```rust
use synaptic_mesh_cli::{MeshCommand, execute_command};

// Start a node
let cmd = MeshCommand::NodeStart { port: 8080 };
execute_command(cmd).await?;

// Initialize market
let cmd = MeshCommand::MarketInit { db_path: None };
execute_command(cmd).await?;

// Check wallet balance
let cmd = MeshCommand::WalletBalance;
let result = execute_command(cmd).await?;
```

## Implementation Status

**Current State**: This is a research prototype with placeholder implementations.

### What's Actually Working:
- ✅ **CLI Structure** - Command parsing and help system
- ✅ **Type Definitions** - Rust structs and enums for future implementation
- ✅ **Error Handling** - Basic error types and handling

### What's Not Working (Placeholders):
- ❌ **Market Operations** - Returns hardcoded responses, no real marketplace
- ❌ **P2P Networking** - Creates QuDAG objects but no actual networking
- ❌ **Neural Networks** - Mock implementations with fake predictions
- ❌ **Wallet Functions** - Always returns balance of 1000, fake transactions
- ❌ **Token Management** - No real token system implemented

### Future Implementation Needed:
1. Real neural network integration with ruv-FANN
2. Actual P2P networking layer implementation  
3. Token economics and wallet system
4. Market mechanism with real compute federation
5. WASM compilation and deployment

## License

MIT OR Apache-2.0