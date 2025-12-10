# PFS Core - High-Performance Neural Network Infrastructure

## Overview

PFS Core is a high-performance neural network infrastructure implemented in Rust, designed for maximum efficiency and zero-cost abstractions. It provides SIMD-optimized operations, custom memory management, and advanced profiling capabilities.

## Features

### Core Components

- **Tensor Operations**: SIMD-optimized tensor operations with custom memory layouts
- **Neural Network Layers**: Dense layers with efficient forward/backward propagation
- **Activation Functions**: ReLU, Sigmoid, Tanh, LeakyReLU, and Softmax
- **Optimizers**: SGD and Adam optimizers with customizable parameters
- **Batch Processing**: Parallel batch processing with work-stealing

### Advanced Features

- **Custom Memory Allocator**: Uses mimalloc for efficient memory management
- **SIMD Optimization**: Explicit SIMD operations using packed_simd_2
- **Cache-Friendly Operations**: Block-wise matrix multiplication and cache-oblivious transpose
- **Memory Pools**: Efficient tensor allocation and reuse
- **Profiling**: Comprehensive performance profiling and memory tracking

## Architecture

### Memory Layout

All tensors are aligned to 64-byte boundaries for optimal SIMD performance:

```rust
#[repr(C, align(64))]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}
```

### SIMD Operations

The library uses explicit SIMD instructions for maximum performance:

```rust
// Vectorized addition with 8-wide SIMD
for i in 0..chunks {
    let a_vec = f32x8::from_slice_unaligned(&a[offset..]);
    let b_vec = f32x8::from_slice_unaligned(&b[offset..]);
    let result_vec = a_vec + b_vec;
    result_vec.write_to_slice_unaligned(&mut result[offset..]);
}
```

### Performance Optimizations

1. **Zero-Cost Abstractions**: Leverages Rust's zero-cost abstractions for maximum performance
2. **Unsafe Code**: Uses unsafe Rust where needed for performance-critical operations
3. **Cache-Friendly Algorithms**: Implements blocked matrix multiplication and cache-oblivious transpose
4. **Memory Alignment**: 64-byte alignment for optimal SIMD performance
5. **Parallel Processing**: Uses Rayon for efficient parallel batch processing

## Usage

### Basic Neural Network

```rust
use ran_opt::pfs_core::*;

// Create a neural network
let mut network = NeuralNetwork::new();
network.add_layer(Box::new(DenseLayer::new(784, 128)), Activation::ReLU);
network.add_layer(Box::new(DenseLayer::new(128, 10)), Activation::Softmax);

// Create input data
let input = Tensor::from_vec(
    (0..784).map(|i| i as f32).collect(),
    vec![1, 784]
);

// Forward pass
let output = network.forward(&input);
```

### Tensor Operations

```rust
use ran_opt::pfs_core::*;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

// SIMD-optimized operations
let c = a.add(&b);        // Element-wise addition
let d = a.mul(&b);        // Element-wise multiplication
let e = a.matmul(&b);     // Matrix multiplication
let f = a.transpose();    // Matrix transpose
```

### Advanced Features

```rust
use ran_opt::pfs_core::advanced::*;

// Create aligned tensors for maximum SIMD performance
let mut tensor = AdvancedTensor::new_aligned(vec![1000, 1000]);

// Use explicit SIMD operations
let mut result = AdvancedTensor::new_aligned(vec![1000, 1000]);
simd_ops::simd_add(&tensor_a, &tensor_b, &mut result);

// Use tensor pool for efficient memory management
let mut pool = TensorPool::new();
let tensor = pool.get_tensor(vec![100, 100]);
// ... use tensor ...
pool.return_tensor(tensor);
```

### Profiling

```rust
use ran_opt::pfs_core::profiler::*;

// Profile operations
let result = profile("matrix_multiplication", || {
    tensor_a.matmul(&tensor_b)
});

// Track memory usage
track_memory_allocation("tensors", 1024);

// Print comprehensive report
print_global_memory_report();
```

## Benchmarks

The library includes comprehensive benchmarks in `benches/pfs_core_bench.rs`:

- Tensor creation and operations
- Neural network forward/backward passes
- SIMD operation performance
- Memory allocation efficiency
- Activation function performance

Run benchmarks with:
```bash
cargo bench
```

## Performance Characteristics

### Tensor Operations (1000x1000 matrices)
- Matrix multiplication: ~10-50ms (depending on hardware)
- Element-wise addition: ~1-5ms
- Matrix transpose: ~2-10ms

### Neural Network (784→128→10, batch=32)
- Forward pass: ~5-20ms
- Backward pass: ~10-40ms

### SIMD Operations (1M elements)
- Vectorized addition: ~1-3ms
- Vectorized multiplication: ~1-3ms
- Vectorized ReLU: ~1-2ms

## Memory Usage

The library uses a custom memory allocator (mimalloc) and provides detailed memory tracking:

- Tensor memory is 64-byte aligned for optimal SIMD performance
- Memory pools reduce allocation overhead
- Comprehensive memory usage reporting

## Safety and Correctness

While the library uses unsafe Rust for performance, it maintains safety through:

- Debug assertions for bounds checking
- Comprehensive test suite
- Memory safety guarantees through careful pointer management
- Thread safety for parallel operations

## Testing

The library includes extensive tests:

```bash
cargo test                    # Run all tests
cargo test --release         # Run optimized tests
cargo test integration_tests # Run integration tests
cargo test performance_tests # Run performance tests
```

## Dependencies

- `rayon`: Parallel processing
- `packed_simd_2`: SIMD operations
- `ndarray`: Linear algebra operations
- `mimalloc`: Memory allocator
- `blas`/`lapack`: High-performance linear algebra
- `criterion`: Benchmarking

## Architecture Decisions

### Why Rust?
- Zero-cost abstractions for maximum performance
- Memory safety without garbage collection
- Excellent SIMD support
- Strong type system for correctness

### Why Custom Memory Management?
- Predictable allocation patterns
- Reduced memory fragmentation
- Better cache locality
- Efficient memory pooling

### Why Explicit SIMD?
- Maximum control over vectorization
- Consistent performance across platforms
- Ability to optimize for specific workloads
- Better than relying on auto-vectorization

## Contributing

When contributing to PFS Core:

1. Maintain zero-cost abstractions
2. Use unsafe code judiciously with proper safety comments
3. Include comprehensive benchmarks for new features
4. Ensure thread safety for parallel operations
5. Add profiling hooks for new operations
6. Include both unit and integration tests

## License

This project is part of the RAN-OPT system for radio access network optimization.