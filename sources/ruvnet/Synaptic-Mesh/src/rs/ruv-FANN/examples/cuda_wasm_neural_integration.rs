//! Comprehensive example showing CUDA-WASM integration with ruv-FANN
//!
//! This example demonstrates the seamless integration between CUDA-WASM
//! transpiler and ruv-FANN neural networks for GPU-accelerated neural computation.

use cuda_rust_wasm::{
    init_neural_integration, get_neural_capabilities, NeuralBridge, BridgeConfig, 
    NeuralOperation, NeuralActivationFunction,
};
use ruv_fann::{Network, NetworkBuilder, ActivationFunction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CUDA-WASM + ruv-FANN Neural Integration Demo");
    println!("===========================================");
    
    // Initialize the neural integration system
    init_neural_integration()?;
    
    // Show system capabilities
    let capabilities = get_neural_capabilities();
    println!("System Capabilities:");
    println!("  CUDA Transpilation: {}", capabilities.cuda_transpilation);
    println!("  GPU Acceleration: {}", capabilities.gpu_acceleration);
    println!("  WASM Support: {}", capabilities.wasm_support);
    println!("  Performance Monitoring: {}", capabilities.performance_monitoring);
    println!("  Memory Pooling: {}", capabilities.memory_pooling);
    println!("  Auto Fallback: {}", capabilities.auto_fallback);
    println!("  Batch Processing: {}", capabilities.batch_processing);
    println!();
    
    // Demo 1: Basic neural bridge operations
    demo_basic_neural_bridge()?;
    
    // Demo 2: Integration with ruv-FANN networks
    demo_ruv_fann_integration()?;
    
    // Demo 3: Performance comparison
    demo_performance_comparison()?;
    
    // Demo 4: Custom CUDA kernel integration
    demo_custom_cuda_kernels()?;
    
    // Demo 5: Batch neural network processing
    demo_batch_neural_processing()?;
    
    println!("All demos completed successfully!");
    
    Ok(())
}

/// Demo 1: Basic neural bridge operations
fn demo_basic_neural_bridge() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 1: Basic Neural Bridge Operations ===");
    
    // Create neural bridge with optimized configuration
    let config = BridgeConfig {
        enable_gpu: true,
        memory_pool_size: 512, // 512 MB
        enable_monitoring: true,
        auto_fallback: true,
        batch_size: 64,
        ..Default::default()
    };
    
    let bridge = NeuralBridge::with_config(config)?;
    
    println!("Neural bridge created successfully");
    println!("GPU Available: {}", bridge.is_gpu_available());
    
    if let Some(info) = bridge.get_device_info() {
        println!("GPU Device: {} ({})", info.name, info.vendor);
        println!("Memory: {} MB", info.memory_size / 1024 / 1024);
        println!("Compute Units: {}", info.compute_units);
    }
    
    // Test matrix multiplication
    println!("\nTesting Matrix Multiplication (256x256):");
    let size = 256;
    let matrix_a: Vec<f32> = (0..size * size).map(|i| (i as f32) / 1000.0).collect();
    let matrix_b: Vec<f32> = (0..size * size).map(|i| ((i * 2) as f32) / 1000.0).collect();
    
    let mut input_data = matrix_a;
    input_data.extend(matrix_b);
    
    let operation = NeuralOperation::MatrixMultiply {
        a_rows: size,
        a_cols: size,
        b_cols: size,
    };
    
    let start = std::time::Instant::now();
    let result = bridge.execute_neural_operation(operation, &input_data)?;
    let duration = start.elapsed();
    
    println!("  Matrix multiplication completed in {:.2}ms", duration.as_millis());
    println!("  Result size: {} elements", result.len());
    println!("  First 5 results: {:?}", &result[0..5]);
    
    // Test activation functions
    println!("\nTesting Activation Functions:");
    let test_input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    
    let functions = [
        ("Sigmoid", NeuralActivationFunction::Sigmoid),
        ("ReLU", NeuralActivationFunction::ReLU),
        ("Tanh", NeuralActivationFunction::Tanh),
        ("GELU", NeuralActivationFunction::GELU),
    ];
    
    for (name, function) in &functions {
        let operation = NeuralOperation::ActivationFunction {
            function: *function,
            size: test_input.len(),
        };
        
        let result = bridge.execute_neural_operation(operation, &test_input)?;
        println!("  {}: {:?}", name, result);
    }
    
    // Show performance statistics
    let perf_stats = bridge.get_performance_stats();
    println!("\nPerformance Statistics:");
    println!("  Total Operations: {}", perf_stats.total_operations);
    println!("  Average Execution Time: {:.2}ms", perf_stats.average_execution_time * 1000.0);
    println!("  GPU Utilization: {:.1}%", perf_stats.gpu_utilization * 100.0);
    
    println!();
    Ok(())
}

/// Demo 2: Integration with ruv-FANN networks
fn demo_ruv_fann_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 2: ruv-FANN Integration ===");
    
    // Create a ruv-FANN neural network
    let mut network: Network<f32> = NetworkBuilder::new()
        .input_layer(4)
        .hidden_layer(8)
        .hidden_layer(6)
        .output_layer(2)
        .build();
    
    // Initialize random weights
    network.randomize_weights(-1.0, 1.0);
    
    println!("ruv-FANN Network Created:");
    println!("  Layers: {}", network.num_layers());
    println!("  Inputs: {}", network.num_inputs());
    println!("  Outputs: {}", network.num_outputs());
    println!("  Total Neurons: {}", network.total_neurons());
    println!("  Total Connections: {}", network.total_connections());
    
    // Test input data
    let input_data = vec![0.5, 0.8, 0.2, 0.9];
    
    // Run inference using ruv-FANN (CPU)
    println!("\nRunning inference:");
    let start = std::time::Instant::now();
    let cpu_output = network.run(&input_data);
    let cpu_time = start.elapsed();
    
    println!("  CPU (ruv-FANN): {:?} in {:.3}ms", cpu_output, cpu_time.as_millis());
    
    // Simulate GPU-accelerated inference using neural bridge
    let bridge = NeuralBridge::new()?;
    let layer_sizes = vec![4, 8, 6, 2];
    
    let operation = NeuralOperation::ForwardPropagation {
        layer_sizes,
    };
    
    let start = std::time::Instant::now();
    let gpu_output = bridge.execute_neural_operation(operation, &input_data)?;
    let gpu_time = start.elapsed();
    
    println!("  GPU (CUDA-WASM): {:?} in {:.3}ms", &gpu_output[0..2], gpu_time.as_millis());
    
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
    
    // Batch processing demonstration
    println!("\nBatch Processing Test:");
    let batch_size = 100;
    let batch_inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| vec![
            (i as f32) / 100.0,
            ((i * 2) as f32) / 100.0,
            ((i * 3) as f32) / 100.0,
            ((i * 4) as f32) / 100.0,
        ])
        .collect();
    
    // CPU batch processing
    let start = std::time::Instant::now();
    let _cpu_batch_results: Vec<Vec<f32>> = batch_inputs.iter()
        .map(|input| network.run(input))
        .collect();
    let cpu_batch_time = start.elapsed();
    
    // GPU batch processing
    let batch_processor = bridge.create_batch_processor();
    let operations: Vec<_> = (0..batch_size)
        .map(|_| NeuralOperation::ForwardPropagation {
            layer_sizes: vec![4, 8, 6, 2],
        })
        .collect();
    
    let start = std::time::Instant::now();
    let _gpu_batch_results = batch_processor.process_batch(operations, batch_inputs)?;
    let gpu_batch_time = start.elapsed();
    
    let batch_speedup = cpu_batch_time.as_secs_f64() / gpu_batch_time.as_secs_f64();
    println!("  CPU Batch: {:.2}ms", cpu_batch_time.as_millis());
    println!("  GPU Batch: {:.2}ms", gpu_batch_time.as_millis());
    println!("  Batch Speedup: {:.2}x", batch_speedup);
    
    println!();
    Ok(())
}

/// Demo 3: Performance comparison
fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 3: Performance Comparison ===");
    
    let bridge = NeuralBridge::new()?;
    
    // Test different operation sizes
    let sizes = vec![1000, 10000, 100000];
    
    println!("Vector Addition Performance:");
    for size in &sizes {
        // Generate test data
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();
        let mut input_data = a.clone();
        input_data.extend(b.clone());
        
        // GPU operation
        let operation = NeuralOperation::VectorAdd { size: *size };
        let start = std::time::Instant::now();
        let _gpu_result = bridge.execute_neural_operation(operation, &input_data)?;
        let gpu_time = start.elapsed();
        
        // CPU operation (simulation)
        let start = std::time::Instant::now();
        let _cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let cpu_time = start.elapsed();
        
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        let throughput = *size as f64 / gpu_time.as_secs_f64() / 1e6; // Million ops/sec
        
        println!("  Size {:6}: GPU {:.3}ms, CPU {:.3}ms, Speedup {:.2}x, Throughput {:.1} Mops/s", 
                 size, gpu_time.as_millis(), cpu_time.as_millis(), speedup, throughput);
    }
    
    println!("\nActivation Function Performance:");
    let size = 100000;
    let input_data: Vec<f32> = (0..size).map(|i| (i as f32) / 1000.0 - 50.0).collect();
    
    let functions = [
        ("Sigmoid", NeuralActivationFunction::Sigmoid),
        ("ReLU", NeuralActivationFunction::ReLU),
        ("Tanh", NeuralActivationFunction::Tanh),
        ("GELU", NeuralActivationFunction::GELU),
    ];
    
    for (name, function) in &functions {
        let operation = NeuralOperation::ActivationFunction {
            function: *function,
            size,
        };
        
        let start = std::time::Instant::now();
        let _result = bridge.execute_neural_operation(operation, &input_data)?;
        let gpu_time = start.elapsed();
        
        let throughput = size as f64 / gpu_time.as_secs_f64() / 1e6;
        println!("  {:>8}: {:.3}ms, {:.1} Mops/s", name, gpu_time.as_millis(), throughput);
    }
    
    // Memory usage statistics
    let memory_stats = bridge.get_memory_stats();
    println!("\nMemory Usage:");
    println!("  Total Allocated: {:.1} MB", memory_stats.total_allocated as f64 / 1024.0 / 1024.0);
    println!("  GPU Allocated: {:.1} MB", memory_stats.gpu_allocated as f64 / 1024.0 / 1024.0);
    println!("  CPU Allocated: {:.1} MB", memory_stats.cpu_allocated as f64 / 1024.0 / 1024.0);
    println!("  Peak Usage: {:.1} MB", memory_stats.peak_usage as f64 / 1024.0 / 1024.0);
    
    println!();
    Ok(())
}

/// Demo 4: Custom CUDA kernel integration
fn demo_custom_cuda_kernels() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 4: Custom CUDA Kernel Integration ===");
    
    let bridge = NeuralBridge::new()?;
    
    // Custom CUDA kernel for polynomial evaluation: y = ax^3 + bx^2 + cx + d
    let polynomial_kernel = r#"
        __global__ void polynomial_eval(float* x, float* y, float a, float b, float c, float d, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float val = x[idx];
                y[idx] = a * val * val * val + b * val * val + c * val + d;
            }
        }
    "#;
    
    let input_data: Vec<f32> = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    
    let operation = NeuralOperation::Custom {
        kernel_source: polynomial_kernel.to_string(),
        name: "polynomial_eval".to_string(),
    };
    
    println!("Custom Polynomial Kernel (y = 2x³ - x² + 3x + 1):");
    let result = bridge.execute_neural_operation(operation, &input_data)?;
    
    for (i, (&x, &y)) in input_data.iter().zip(result.iter()).enumerate() {
        let expected = 2.0 * x.powi(3) - x.powi(2) + 3.0 * x + 1.0;
        println!("  x[{}] = {:5.1} -> y = {:8.2} (expected: {:8.2})", i, x, y, expected);
    }
    
    // Custom kernel for matrix transpose
    let transpose_kernel = r#"
        __global__ void matrix_transpose(float* input, float* output, int rows, int cols) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (idx < cols && idy < rows) {
                output[idx * rows + idy] = input[idy * cols + idx];
            }
        }
    "#;
    
    let matrix_size = 4;
    let matrix_data: Vec<f32> = (0..matrix_size * matrix_size).map(|i| i as f32).collect();
    
    println!("\nMatrix Transpose (4x4):");
    println!("Original matrix:");
    for i in 0..matrix_size {
        print!("  ");
        for j in 0..matrix_size {
            print!("{:5.0} ", matrix_data[i * matrix_size + j]);
        }
        println!();
    }
    
    let transpose_operation = NeuralOperation::Custom {
        kernel_source: transpose_kernel.to_string(),
        name: "matrix_transpose".to_string(),
    };
    
    let transposed = bridge.execute_neural_operation(transpose_operation, &matrix_data)?;
    
    println!("Transposed matrix:");
    for i in 0..matrix_size {
        print!("  ");
        for j in 0..matrix_size {
            print!("{:5.0} ", transposed[i * matrix_size + j]);
        }
        println!();
    }
    
    println!();
    Ok(())
}

/// Demo 5: Batch neural network processing
fn demo_batch_neural_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demo 5: Batch Neural Network Processing ===");
    
    let bridge = NeuralBridge::new()?;
    let batch_processor = bridge.create_batch_processor();
    
    // Simulate multiple neural network architectures
    let network_configs = vec![
        ("Small", vec![10, 20, 10]),
        ("Medium", vec![50, 100, 50, 25]),
        ("Large", vec![100, 200, 100, 50, 10]),
    ];
    
    println!("Testing multiple network architectures in batch:");
    
    let mut operations = Vec::new();
    let mut inputs = Vec::new();
    
    for (name, layer_sizes) in &network_configs {
        let input_size = layer_sizes[0];
        let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32) / input_size as f32).collect();
        
        operations.push(NeuralOperation::ForwardPropagation {
            layer_sizes: layer_sizes.clone(),
        });
        inputs.push(input_data);
        
        println!("  {} Network: {:?}", name, layer_sizes);
    }
    
    // Process all networks in batch
    let start = std::time::Instant::now();
    let results = batch_processor.process_batch(operations, inputs)?;
    let batch_time = start.elapsed();
    
    println!("\nBatch processing completed in {:.2}ms", batch_time.as_millis());
    
    for (i, (name, _)) in network_configs.iter().enumerate() {
        if i < results.len() {
            println!("  {} result: {} outputs, first 3: {:?}", 
                     name, results[i].len(), &results[i][0..3.min(results[i].len())]);
        }
    }
    
    // Benchmark different batch sizes
    println!("\nBatch Size Performance:");
    let batch_sizes = vec![1, 4, 16, 64];
    let network_arch = vec![32, 64, 32, 16];
    
    for batch_size in batch_sizes {
        let operations: Vec<_> = (0..batch_size)
            .map(|_| NeuralOperation::ForwardPropagation {
                layer_sizes: network_arch.clone(),
            })
            .collect();
        
        let inputs: Vec<_> = (0..batch_size)
            .map(|i| (0..32).map(|j| ((i * 32 + j) as f32) / 1000.0).collect())
            .collect();
        
        let start = std::time::Instant::now();
        let _batch_results = batch_processor.process_batch(operations, inputs)?;
        let time = start.elapsed();
        
        let throughput = batch_size as f64 / time.as_secs_f64();
        println!("  Batch size {:2}: {:.2}ms ({:.1} networks/sec)", 
                 batch_size, time.as_millis(), throughput);
    }
    
    println!();
    Ok(())
}