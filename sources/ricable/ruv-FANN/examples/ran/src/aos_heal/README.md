# Agent AOS Heal - Self-Healing Action Networks

## Overview

Agent AOS Heal is a sophisticated self-healing network system that automatically generates and executes healing actions for network anomalies. It uses advanced machine learning techniques including sequence-to-sequence models, beam search optimization, and action validation networks to provide intelligent network remediation.

## Architecture

### Core Components

1. **Sequence-to-Sequence Models** (`Seq2SeqModel`)
   - Transformer-based encoder-decoder architecture
   - Generates AMOS (Ericsson MO Script) scripts from anomaly features
   - Supports attention mechanisms for context-aware action generation

2. **Template Selection Networks** (`TemplateSelector`)
   - Neural network-based template matching
   - Selects appropriate AMOS templates based on anomaly characteristics
   - Supports multiple healing action types

3. **RESTCONF Payload Generators** (`RestconfGenerator`)
   - Generates HTTP payloads for ENM (Ericsson Network Manager) APIs
   - Supports multiple REST operations (GET, POST, PUT, PATCH, DELETE)
   - Handles authentication and request formatting

4. **Action Validation Networks** (`ActionValidator`)
   - Validates healing actions before execution
   - Checks dependencies, conflicts, and business rules
   - Provides confidence scores for action safety

5. **Beam Search Optimization** (`AdvancedBeamSearch`)
   - Finds optimal sequences of healing actions
   - Considers action dependencies and conflicts
   - Balances effectiveness, cost, and diversity

6. **ENM Client Integration** (`EnmClient`)
   - Direct integration with Ericsson Network Manager
   - Executes AMOS scripts and RESTCONF operations
   - Handles authentication and error recovery

## Key Features

### Healing Action Types

- **Process Restart**: Restart failed processes on network nodes
- **Cell Blocking/Unblocking**: Manage cell operational state
- **Parameter Adjustment**: Modify network configuration parameters
- **Load Balancing**: Redistribute network traffic
- **Service Migration**: Move services between nodes
- **Resource Allocation**: Adjust resource assignments
- **Network Reconfiguration**: Modify network topology

### Advanced Capabilities

- **Intelligent Action Generation**: Uses transformer models to generate contextually appropriate healing actions
- **Beam Search Optimization**: Finds optimal action sequences considering multiple constraints
- **Dependency Management**: Ensures actions are executed in correct order
- **Conflict Resolution**: Prevents conflicting actions from being executed simultaneously
- **Rollback Planning**: Maintains rollback strategies for safe action execution
- **Real-time Validation**: Validates actions before execution to prevent harmful operations

## Usage

### Basic Setup

```rust
use ran_opt::{
    AgentAosHeal, ActionGeneratorConfig, default_amos_templates,
    HealingAction, HealingActionType,
};
use candle_core::{Device, Tensor};

// Create configuration
let config = ActionGeneratorConfig {
    beam_width: 5,
    max_depth: 8,
    temperature: 0.7,
    ..ActionGeneratorConfig::default()
};

// Initialize Agent AOS Heal
let agent = AgentAosHeal::new(
    config,
    default_amos_templates(),
    "https://enm.example.com".to_string(),
    "your_auth_token".to_string(),
)?;
```

### Generate Healing Actions

```rust
// Create anomaly features tensor
let device = Device::Cpu;
let anomaly_features = Tensor::randn(0f32, 1f32, (1, 256), &device)?;

// Generate healing actions
let healing_actions = agent.generate_healing_actions(&anomaly_features)?;

// Execute actions
let results = agent.execute_healing_actions(healing_actions).await?;
```

### Beam Search Optimization

```rust
use ran_opt::{AdvancedBeamSearch, NetworkState};

// Create beam search optimizer
let beam_search = AdvancedBeamSearch::new(5, 8);

// Define network states
let mut initial_state = NetworkState::new();
initial_state.performance_score = 0.3;
initial_state.failed_processes = vec!["process_A".to_string()];

let mut target_state = NetworkState::new();
target_state.performance_score = 0.9;
target_state.failed_processes = vec![];

// Find optimal action sequence
let optimized_actions = beam_search.search(&initial_state, &target_state);
```

### ENM Client Integration

```rust
use ran_opt::EnmClient;

// Create ENM client
let client = EnmClient::new(
    "https://enm.example.com".to_string(),
    "your_auth_token".to_string(),
);

// Execute AMOS script
let result = client.execute_amos_script(
    "lt all\nmo node123\nrestart cell_manager",
    "node123"
).await?;

// Restart process
client.restart_process("node123", "cell_manager").await?;

// Block/unblock cell
client.set_cell_state("cell001", CellState::Blocked).await?;
```

## Configuration

### ActionGeneratorConfig

```rust
pub struct ActionGeneratorConfig {
    pub input_dim: usize,           // Input feature dimension (default: 256)
    pub hidden_dim: usize,          // Hidden layer dimension (default: 512)
    pub num_layers: usize,          // Number of transformer layers (default: 6)
    pub vocab_size: usize,          // Vocabulary size (default: 10000)
    pub max_sequence_length: usize, // Maximum sequence length (default: 128)
    pub dropout_prob: f64,          // Dropout probability (default: 0.1)
    pub beam_width: usize,          // Beam search width (default: 5)
    pub temperature: f64,           // Sampling temperature (default: 1.0)
}
```

### AMOS Templates

The system includes default AMOS templates for common healing actions:

- **Process Restart**: `lt all\nmo {node}\nrestart {process}`
- **Cell Blocking**: `lt all\nmo {cell}\nset cellState blocked`
- **Cell Unblocking**: `lt all\nmo {cell}\nset cellState active`
- **Parameter Adjustment**: `lt all\nmo {entity}\nset {parameter} {value}`

## API Reference

### Core Types

#### HealingAction
```rust
pub struct HealingAction {
    pub action_type: HealingActionType,
    pub target_entity: String,
    pub parameters: HashMap<String, String>,
    pub priority: f32,
    pub confidence: f32,
    pub estimated_duration: u64,
    pub rollback_plan: Option<String>,
}
```

#### NetworkState
```rust
pub struct NetworkState {
    pub performance_score: f32,
    pub max_utilization: f32,
    pub failed_processes: Vec<String>,
    pub blocked_cells: Vec<String>,
    pub active_alarms: Vec<String>,
    pub resource_usage: HashMap<String, f32>,
}
```

#### RestconfPayload
```rust
pub struct RestconfPayload {
    pub method: String,
    pub endpoint: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub timeout: u64,
}
```

### Main Methods

#### AgentAosHeal
- `new()`: Initialize the agent with configuration and templates
- `generate_healing_actions()`: Generate actions from anomaly features
- `execute_healing_actions()`: Execute actions via ENM APIs

#### AdvancedBeamSearch
- `new()`: Create beam search optimizer
- `search()`: Find optimal action sequence between network states

#### EnmClient
- `new()`: Create ENM client with authentication
- `execute_amos_script()`: Execute AMOS script on target node
- `restart_process()`: Restart specific process
- `set_cell_state()`: Change cell operational state
- `get_performance_metrics()`: Retrieve network metrics

## Examples

### Complete Workflow Example

```rust
use ran_opt::*;
use candle_core::{Device, Tensor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize Agent AOS Heal
    let config = ActionGeneratorConfig::default();
    let agent = AgentAosHeal::new(
        config,
        default_amos_templates(),
        "https://enm.example.com".to_string(),
        "your_token".to_string(),
    )?;
    
    // 2. Create anomaly features
    let device = Device::Cpu;
    let anomaly_features = Tensor::randn(0f32, 1f32, (1, 256), &device)?;
    
    // 3. Generate healing actions
    let actions = agent.generate_healing_actions(&anomaly_features)?;
    
    // 4. Execute actions
    let results = agent.execute_healing_actions(actions).await?;
    
    // 5. Process results
    for result in results {
        if result.success {
            println!("Action succeeded: {:?}", result.action.action_type);
        } else {
            println!("Action failed: {:?}", result.error);
        }
    }
    
    Ok(())
}
```

### Beam Search Example

```rust
use ran_opt::{AdvancedBeamSearch, NetworkState, HealingActionType};

fn optimize_healing_sequence() {
    let beam_search = AdvancedBeamSearch::new(5, 10);
    
    // Create problematic network state
    let mut initial_state = NetworkState::new();
    initial_state.performance_score = 0.3;
    initial_state.failed_processes = vec!["cell_mgr".to_string()];
    initial_state.blocked_cells = vec!["cell_001".to_string()];
    
    // Define target state
    let mut target_state = NetworkState::new();
    target_state.performance_score = 0.9;
    target_state.failed_processes = vec![];
    target_state.blocked_cells = vec![];
    
    // Find optimal sequence
    let actions = beam_search.search(&initial_state, &target_state);
    
    for action in actions {
        println!("Action: {:?} on {}", action.action_type, action.target_entity);
    }
}
```

## Testing

The module includes comprehensive tests covering:

- Action generation and validation
- Beam search optimization
- ENM client integration
- Template selection
- Serialization/deserialization
- Error handling

Run tests with:
```bash
cargo test aos_heal
```

## Performance Considerations

- **Memory Usage**: Transformer models can be memory-intensive; configure appropriately
- **Beam Search**: Larger beam widths provide better solutions but increase computation time
- **Network Latency**: ENM API calls may have network latency; consider timeout settings
- **Validation Overhead**: Action validation adds safety but increases processing time

## Error Handling

The system provides comprehensive error handling:

- **Network Errors**: Handles ENM API connection issues
- **Validation Errors**: Prevents execution of invalid actions
- **Authentication Errors**: Manages token expiration and renewal
- **Timeout Errors**: Handles long-running operations

## Security

- **Authentication**: Uses bearer tokens for ENM API access
- **Validation**: Validates all actions before execution
- **Rollback**: Maintains rollback plans for safe operation
- **Logging**: Comprehensive logging for audit trails

## Future Enhancements

- **Multi-domain Support**: Extend to other network domains beyond RAN
- **Learning Capabilities**: Implement reinforcement learning for action optimization
- **Real-time Adaptation**: Dynamic model updates based on execution results
- **Advanced Metrics**: Enhanced performance and success metrics
- **Distributed Execution**: Support for distributed action execution

## Contributing

When contributing to Agent AOS Heal:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Consider performance implications of modifications
5. Ensure compatibility with existing ENM integrations

## License

This module is part of the ran-opt project and follows the same licensing terms.