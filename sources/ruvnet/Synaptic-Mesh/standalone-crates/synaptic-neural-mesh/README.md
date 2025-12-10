# Synaptic Neural Mesh

Neural mesh coordination layer for distributed AI agents in the Synaptic Neural Mesh project.

## Features

- **Agent Coordination**: Efficient coordination of distributed neural agents
- **Mesh Topology**: Self-organizing mesh network for AI communication
- **Async Operations**: Built on Tokio for high-performance async I/O
- **Integration**: Seamless integration with QuDAG core networking

## Usage

```rust
use synaptic_neural_mesh::{NeuralMesh, Agent};

#[tokio::main]
async fn main() {
    let mesh = NeuralMesh::new();
    let agent = Agent::new("agent-1");
    mesh.add_agent(agent).await;
}
```

## License

MIT OR Apache-2.0