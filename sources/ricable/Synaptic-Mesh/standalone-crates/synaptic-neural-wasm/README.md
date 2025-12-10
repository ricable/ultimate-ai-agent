# Synaptic Neural WASM

WASM-optimized neural network engine for the Synaptic Neural Mesh project.

## Features

- **SIMD Acceleration**: Leverages WebAssembly SIMD for fast computations
- **Lightweight**: Minimal dependencies for small WASM bundle size
- **Browser & Node.js**: Works in any WASM environment
- **Neural Operations**: Matrix operations, activation functions, and more

## Usage

```rust
use synaptic_neural_wasm::{NeuralNetwork, Layer};

let mut network = NeuralNetwork::new();
network.add_layer(Layer::dense(784, 128));
network.add_layer(Layer::dense(128, 10));
```

## License

MIT OR Apache-2.0