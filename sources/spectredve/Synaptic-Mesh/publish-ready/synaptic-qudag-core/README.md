# Synaptic QuDAG Core

Core DAG networking and consensus library for the Synaptic Neural Mesh project.

## Features

- **DAG-based consensus**: Efficient directed acyclic graph structure
- **P2P networking**: Built on modern cryptographic primitives
- **Async/await**: Fully asynchronous implementation
- **Production-ready**: Designed for distributed neural mesh networks

## Usage

```rust
use synaptic_qudag_core::{QuDAGNode, QuDAGNetwork};

#[tokio::main]
async fn main() {
    let network = QuDAGNetwork::new();
    let node = QuDAGNode::new("example-data".as_bytes());
    network.add_node(node).await.unwrap();
}
```

## License

MIT OR Apache-2.0