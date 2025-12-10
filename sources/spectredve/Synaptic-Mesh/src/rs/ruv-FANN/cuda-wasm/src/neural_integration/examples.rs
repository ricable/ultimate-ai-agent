//! Examples and Demos for Neural Integration
//!
//! This module provides comprehensive examples showing how to use the 
//! CUDA-WASM neural integration with ruv-FANN.

use super::{
    ActivationFunction, BridgeConfig, NeuralBridge, NeuralOperation, NeuralResult,
    GpuDevice, Precision,
};

/// Example: Basic matrix operations with GPU acceleration
pub fn matrix_operations_example() -> NeuralResult<()> {
    println!("=== Matrix Operations Example ===");
    
    // Create bridge with GPU acceleration
    let config = BridgeConfig {
        enable_gpu: true,
        gpu_device: GpuDevice::Auto,
        memory_pool_size: 256, // 256 MB
        enable_monitoring: true,
        auto_fallback: true,
        batch_size: 32,
        precision: Precision::Float32,
    };
    
    let bridge = NeuralBridge::with_config(config)?;
    
    println!("GPU Available: {}", bridge.is_gpu_available());
    if let Some(info) = bridge.get_device_info() {
        println!("Device: {} ({})", info.name, info.vendor);
    }
    
    // Example matrix multiplication: 4x4 * 4x4
    let matrix_a = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    
    let matrix_b = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    
    let mut input_data = matrix_a.clone();
    input_data.extend(matrix_b);
    
    let operation = NeuralOperation::MatrixMultiply {
        a_rows: 4,
        a_cols: 4,
        b_cols: 4,
        _phantom: std::marker::PhantomData,
    };
    
    let result = bridge.execute_neural_operation(operation, &input_data)?;
    
    println!("Matrix A * Identity Matrix =");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:8.2} ", result[i * 4 + j]);
        }
        println!();
    }
    
    let stats = bridge.get_performance_stats();
    println!("Performance: {:.2} ops/sec", stats.throughput);
    
    Ok(())
}

/// Example: Neural network forward propagation
pub fn neural_network_example() -> NeuralResult<()> {
    println!("\n=== Neural Network Example ===");
    
    let bridge = NeuralBridge::new()?;
    
    // Create a simple neural network: 3 inputs -> 4 hidden -> 2 outputs
    let layer_sizes = vec![3, 4, 2];
    
    // Sample input data
    let input_data = vec![0.5, 0.8, 0.2]; // 3 inputs
    
    // Forward propagation
    let operation = NeuralOperation::ForwardPropagation {
        layer_sizes: layer_sizes.clone(),
        _phantom: std::marker::PhantomData,
    };
    
    let result = bridge.execute_neural_operation(operation, &input_data)?;
    
    println!("Network architecture: {layer_sizes:?}");
    println!("Input: {input_data:?}");
    println!("Output: {result:?}");
    
    // Test different activation functions
    let activation_functions = [
        ("Sigmoid", ActivationFunction::Sigmoid),
        ("ReLU", ActivationFunction::ReLU),
        ("Tanh", ActivationFunction::Tanh),
        ("GELU", ActivationFunction::GELU),
    ];
    
    let test_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    
    println!("\nActivation Function Comparison:");
    for (name, function) in &activation_functions {
        let operation = NeuralOperation::ActivationFunction {
            function: *function,
            size: test_input.len(),
            _phantom: std::marker::PhantomData,
        };
        
        let result = bridge.execute_neural_operation(operation, &test_input)?;
        println!("{name:>8}: {result:?}");
    }
    
    Ok(())
}

/// Example: Performance benchmarking
pub fn performance_benchmark_example() -> NeuralResult<()> {
    println!("\n=== Performance Benchmark Example ===");
    
    let bridge = NeuralBridge::new()?;
    
    // Benchmark different operation sizes
    let sizes = vec![100, 1000, 10000, 100000];
    
    println!("Vector Addition Benchmark:");
    for size in &sizes {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();
        
        let mut input_data = a;
        input_data.extend(b);
        
        let operation = NeuralOperation::VectorAdd { size: *size, _phantom: std::marker::PhantomData };
        
        let start = std::time::Instant::now();
        let _result = bridge.execute_neural_operation(operation, &input_data)?;
        let duration = start.elapsed();
        
        let throughput = *size as f64 / duration.as_secs_f64();
        println!("  Size {size:6}: {throughput:8.2} elements/sec");
    }
    
    // Matrix multiplication benchmark
    println!("\nMatrix Multiplication Benchmark:");
    let matrix_sizes = vec![16, 32, 64, 128];
    
    for size in &matrix_sizes {
        let matrix_a: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let matrix_b: Vec<f32> = (0..size * size).map(|i| (i * 2) as f32).collect();
        
        let mut input_data = matrix_a;
        input_data.extend(matrix_b);
        
        let operation = NeuralOperation::MatrixMultiply {
            a_rows: *size,
            a_cols: *size,
            b_cols: *size,
            _phantom: std::marker::PhantomData,
        };
        
        let start = std::time::Instant::now();
        let _result = bridge.execute_neural_operation(operation, &input_data)?;
        let duration = start.elapsed();
        
        let flops = 2.0 * (*size as f64).powi(3); // 2 * n^3 operations
        let gflops = flops / duration.as_secs_f64() / 1e9;
        
        println!("  {size}x{size}: {gflops:8.2} GFLOPS");
    }
    
    let memory_stats = bridge.get_memory_stats();
    println!("\nMemory Usage:");
    println!("  Total: {} bytes", memory_stats.total_allocated);
    println!("  GPU: {} bytes", memory_stats.gpu_allocated);
    println!("  CPU: {} bytes", memory_stats.cpu_allocated);
    
    Ok(())
}

/// Example: Batch processing for efficiency
pub fn batch_processing_example() -> NeuralResult<()> {
    println!("\n=== Batch Processing Example ===");
    
    let bridge = NeuralBridge::new()?;
    let batch_processor = bridge.create_batch_processor();
    
    // Create multiple operations to process in batch
    let operations = vec![
        NeuralOperation::VectorAdd { size: 1000, _phantom: std::marker::PhantomData },
        NeuralOperation::ActivationFunction { 
            function: ActivationFunction::ReLU, 
            size: 1000,
            _phantom: std::marker::PhantomData 
        },
        NeuralOperation::ActivationFunction { 
            function: ActivationFunction::Sigmoid, 
            size: 1000,
            _phantom: std::marker::PhantomData 
        },
    ];
    
    let inputs = vec![
        // Vector addition input (a + b)
        {
            let mut data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            data.extend((0..1000).map(|i| (i * 2) as f32));
            data
        },
        // ReLU input
        (0..1000).map(|i| (i as f32) - 500.0).collect(),
        // Sigmoid input
        (0..1000).map(|i| (i as f32) / 100.0 - 5.0).collect(),
    ];
    
    let start = std::time::Instant::now();
    let results = batch_processor.process_batch(operations, inputs)?;
    let duration = start.elapsed();
    
    println!("Processed {} operations in {:.2}ms", results.len(), duration.as_millis());
    println!("Batch throughput: {:.2} ops/sec", results.len() as f64 / duration.as_secs_f64());
    
    // Show some results
    for (i, result) in results.iter().enumerate() {
        println!("Operation {}: {} outputs", i, result.len());
        if !result.is_empty() {
            println!("  First 5: {:?}", &result[0..5.min(result.len())]);
        }
    }
    
    Ok(())
}

/// Example: Custom CUDA kernel integration
pub fn custom_kernel_example() -> NeuralResult<()> {
    println!("\n=== Custom CUDA Kernel Example ===");
    
    let bridge = NeuralBridge::new()?;
    
    // Custom CUDA kernel for element-wise square
    let kernel_source = r#"
        __global__ void element_square(float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = input[idx] * input[idx];
            }
        }
    "#;
    
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let operation = NeuralOperation::Custom {
        kernel_source: kernel_source.to_string(),
        name: "element_square".to_string(),
        _phantom: std::marker::PhantomData,
    };
    
    let result = bridge.execute_neural_operation(operation, &input_data)?;
    
    println!("Custom Kernel: Element-wise Square");
    println!("Input:  {input_data:?}");
    println!("Output: {result:?}");
    
    // Verify results
    let expected: Vec<f32> = input_data.iter().map(|x| x * x).collect();
    println!("Expected: {expected:?}");
    
    Ok(())
}

/// Example: Error handling and fallback behavior
pub fn error_handling_example() -> NeuralResult<()> {
    println!("\n=== Error Handling Example ===");
    
    // Test with GPU disabled to demonstrate fallback
    let config = BridgeConfig {
        enable_gpu: false, // Force CPU mode
        auto_fallback: true,
        ..Default::default()
    };
    
    let bridge = NeuralBridge::with_config(config)?;
    
    println!("GPU Available: {} (forced CPU mode)", bridge.is_gpu_available());
    
    // This should work even without GPU
    let operation = NeuralOperation::VectorAdd { size: 100, _phantom: std::marker::PhantomData };
    let input_data: Vec<f32> = (0..200).map(|i| i as f32).collect();
    
    match bridge.execute_neural_operation(operation, &input_data) {
        Ok(result) => {
            println!("CPU fallback successful: {} results", result.len());
            println!("First 5 results: {:?}", &result[0..5]);
        }
        Err(e) => {
            println!("Error: {e}");
        }
    }
    
    // Test error conditions
    println!("\nTesting error conditions:");
    
    // Invalid input size
    let operation = NeuralOperation::VectorAdd { size: 100, _phantom: std::marker::PhantomData };
    let invalid_input = vec![1.0, 2.0, 3.0]; // Too small
    
    match bridge.execute_neural_operation(operation, &invalid_input) {
        Ok(_) => println!("Unexpected success with invalid input"),
        Err(e) => println!("Expected error: {e}"),
    }
    
    Ok(())
}

/// Example: Real-world neural network training simulation
pub fn training_simulation_example() -> NeuralResult<()> {
    println!("\n=== Training Simulation Example ===");
    
    let bridge = NeuralBridge::new()?;
    
    // Simulate training a simple neural network
    let layer_sizes = vec![4, 8, 8, 3]; // 4 inputs, 2 hidden layers, 3 outputs
    let learning_rate = 0.01;
    let epochs = 100;
    
    println!("Neural Network: {layer_sizes:?}");
    println!("Learning Rate: {learning_rate}");
    println!("Epochs: {epochs}");
    
    // Generate synthetic training data
    let batch_size = 32;
    let training_samples = 1000;
    
    println!("\nSimulating training...");
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        for batch in 0..(training_samples / batch_size) {
            // Generate random batch
            let batch_input: Vec<f32> = (0..batch_size * layer_sizes[0])
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect();
            
            // Forward propagation
            let forward_op = NeuralOperation::ForwardPropagation {
                layer_sizes: layer_sizes.clone(),
                _phantom: std::marker::PhantomData,
            };
            
            let _outputs = bridge.execute_neural_operation(forward_op, &batch_input)?;
            
            // Simulate loss calculation
            let loss = rand::random::<f32>() * 0.1 + 0.9_f32.powf(epoch as f32);
            total_loss += loss;
            
            // Backward propagation (simulated)
            let backward_op = NeuralOperation::BackwardPropagation {
                layer_sizes: layer_sizes.clone(),
                _phantom: std::marker::PhantomData,
            };
            
            let _gradients = bridge.execute_neural_operation(backward_op, &batch_input)?;
        }
        
        let avg_loss = total_loss / (training_samples / batch_size) as f32;
        
        if epoch % 20 == 0 {
            println!("Epoch {epoch}: Loss = {avg_loss:.6}");
        }
    }
    
    let perf_stats = bridge.get_performance_stats();
    println!("\nTraining completed!");
    println!("Total operations: {}", perf_stats.total_operations);
    println!("Average execution time: {:.2}ms", perf_stats.average_execution_time * 1000.0);
    println!("GPU utilization: {:.1}%", perf_stats.gpu_utilization * 100.0);
    
    Ok(())
}

/// Run all examples
pub fn run_all_examples() -> NeuralResult<()> {
    println!("CUDA-WASM Neural Integration Examples");
    println!("=====================================");
    
    matrix_operations_example()?;
    neural_network_example()?;
    performance_benchmark_example()?;
    batch_processing_example()?;
    custom_kernel_example()?;
    error_handling_example()?;
    training_simulation_example()?;
    
    println!("\n=== All Examples Completed Successfully! ===");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_operations_example() {
        let result = matrix_operations_example();
        assert!(result.is_ok(), "Matrix operations example failed: {result:?}");
    }
    
    #[test]
    fn test_neural_network_example() {
        let result = neural_network_example();
        assert!(result.is_ok(), "Neural network example failed: {result:?}");
    }
    
    #[test]
    fn test_error_handling_example() {
        let result = error_handling_example();
        assert!(result.is_ok(), "Error handling example failed: {result:?}");
    }
}