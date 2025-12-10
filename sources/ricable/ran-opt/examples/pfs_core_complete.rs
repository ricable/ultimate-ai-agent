use ran_opt::pfs_core::*;
use ran_opt::pfs_core::advanced::*;
use ran_opt::pfs_core::profiler::*;
use std::time::Instant;

fn main() {
    println!("PFS Core Complete Demo");
    println!("======================");
    
    // Initialize performance monitor
    let monitor = PerformanceMonitor::new();
    
    // Demo 1: High-Performance Neural Network
    println!("\n1. High-Performance Neural Network Demo");
    println!("---------------------------------------");
    
    let result = monitor.time("network_creation", || {
        create_and_train_network()
    });
    
    println!("Network training completed: {:?}", result);
    
    // Demo 2: Advanced Tensor Operations
    println!("\n2. Advanced Tensor Operations Demo");
    println!("----------------------------------");
    
    monitor.time("advanced_tensor_ops", || {
        demonstrate_advanced_tensors()
    });
    
    // Demo 3: SIMD Operations
    println!("\n3. SIMD Operations Benchmark");
    println!("----------------------------");
    
    monitor.time("simd_operations", || {
        benchmark_simd_operations()
    });
    
    // Demo 4: Memory Pool Management
    println!("\n4. Memory Pool Management");
    println!("-------------------------");
    
    monitor.time("memory_pool", || {
        demonstrate_memory_pool()
    });
    
    // Demo 5: Parallel Batch Processing
    println!("\n5. Parallel Batch Processing");
    println!("-----------------------------");
    
    monitor.time("parallel_processing", || {
        demonstrate_parallel_processing()
    });
    
    // Demo 6: Cache-Oblivious Algorithms
    println!("\n6. Cache-Oblivious Transpose");
    println!("-----------------------------");
    
    monitor.time("cache_oblivious", || {
        demonstrate_cache_oblivious_transpose()
    });
    
    // Print comprehensive performance report
    println!("\n7. Performance Report");
    println!("---------------------");
    monitor.print_full_report();
    
    // Print global memory report
    print_global_memory_report();
    
    println!("\nDemo completed successfully!");
}

fn create_and_train_network() -> f32 {
    println!("Creating neural network for MNIST-like task...");
    
    // Create network architecture: 784 -> 512 -> 256 -> 128 -> 10
    let mut network = NeuralNetwork::new();
    network.add_layer(Box::new(DenseLayer::new(784, 512)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(512, 256)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(256, 128)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(128, 10)), Activation::Softmax);
    
    // Create synthetic training data
    let batch_size = 64;
    let input_size = 784;
    let num_classes = 10;
    
    // Input data (normalized pixel values)
    let input_data: Vec<f32> = (0..batch_size * input_size)
        .map(|i| ((i as f32).sin() * 0.5 + 0.5).clamp(0.0, 1.0))
        .collect();
    
    let input = Tensor::from_vec(input_data, vec![batch_size, input_size]);
    
    // Target data (one-hot encoded)
    let mut target_data = vec![0.0; batch_size * num_classes];
    for i in 0..batch_size {
        let class = i % num_classes;
        target_data[i * num_classes + class] = 1.0;
    }
    let target = Tensor::from_vec(target_data, vec![batch_size, num_classes]);
    
    // Training loop
    let optimizer = Adam::new(0.001);
    let mut total_loss = 0.0;
    let num_epochs = 10;
    
    println!("Training for {} epochs...", num_epochs);
    
    for epoch in 0..num_epochs {
        let start = Instant::now();
        
        // Forward pass
        let output = network.forward(&input);
        
        // Compute loss (cross-entropy)
        let loss = compute_cross_entropy_loss(&output, &target);
        total_loss += loss;
        
        // Backward pass
        let loss_grad = compute_loss_gradient(&output, &target);
        let _input_grad = network.backward(&loss_grad);
        
        // Update weights
        network.update_weights(&optimizer);
        
        let epoch_time = start.elapsed();
        
        if epoch % 2 == 0 {
            println!("Epoch {}: Loss = {:.4}, Time = {:?}", epoch, loss, epoch_time);
        }
    }
    
    let avg_loss = total_loss / num_epochs as f32;
    println!("Training completed. Average loss: {:.4}", avg_loss);
    
    avg_loss
}

fn demonstrate_advanced_tensors() {
    println!("Creating aligned tensors for optimal SIMD performance...");
    
    let size = 1024;
    let mut tensor_a = AdvancedTensor::new_aligned(vec![size, size]);
    let mut tensor_b = AdvancedTensor::new_aligned(vec![size, size]);
    
    // Initialize with test data
    for i in 0..size {
        for j in 0..size {
            tensor_a.set(&[i, j], (i * j) as f32 / 1000.0);
            tensor_b.set(&[i, j], ((i + j) as f32).sin());
        }
    }
    
    // Perform blocked matrix multiplication
    let mut result = AdvancedTensor::new_aligned(vec![size, size]);
    let start = Instant::now();
    blocked_matmul(&tensor_a, &tensor_b, &mut result, 64);
    let blocked_time = start.elapsed();
    
    println!("Blocked matrix multiplication ({}x{}): {:?}", size, size, blocked_time);
    
    // Verify result
    let sample_result = result.get(&[0, 0]);
    println!("Sample result: {:.6}", sample_result);
    
    track_memory_allocation("advanced_tensors", (size * size * 4 * 3) as u64);
}

fn benchmark_simd_operations() {
    println!("Benchmarking SIMD operations...");
    
    let size = 1_000_000;
    let mut tensor_a = AdvancedTensor::new_aligned(vec![size]);
    let mut tensor_b = AdvancedTensor::new_aligned(vec![size]);
    let mut result = AdvancedTensor::new_aligned(vec![size]);
    
    // Initialize with test data
    for i in 0..size {
        tensor_a.set(&[i], (i as f32).sin());
        tensor_b.set(&[i], (i as f32).cos());
    }
    
    // Benchmark SIMD addition
    let start = Instant::now();
    simd_ops::simd_add(&tensor_a, &tensor_b, &mut result);
    let add_time = start.elapsed();
    
    // Benchmark SIMD multiplication
    let start = Instant::now();
    simd_ops::simd_mul(&tensor_a, &tensor_b, &mut result);
    let mul_time = start.elapsed();
    
    // Benchmark SIMD ReLU
    let start = Instant::now();
    simd_ops::simd_relu(&tensor_a, &mut result);
    let relu_time = start.elapsed();
    
    println!("SIMD addition ({} elements): {:?}", size, add_time);
    println!("SIMD multiplication ({} elements): {:?}", size, mul_time);
    println!("SIMD ReLU ({} elements): {:?}", size, relu_time);
    
    // Calculate throughput
    let add_throughput = size as f64 / add_time.as_secs_f64() / 1_000_000.0;
    let mul_throughput = size as f64 / mul_time.as_secs_f64() / 1_000_000.0;
    
    println!("Addition throughput: {:.2} M elements/sec", add_throughput);
    println!("Multiplication throughput: {:.2} M elements/sec", mul_throughput);
    
    track_memory_allocation("simd_operations", (size * 4 * 3) as u64);
}

fn demonstrate_memory_pool() {
    println!("Demonstrating memory pool efficiency...");
    
    let mut pool = TensorPool::with_sizes(vec![1000, 10000, 100000]);
    
    // Allocate and deallocate tensors
    let start = Instant::now();
    
    let mut tensors = Vec::new();
    for i in 0..100 {
        let size = if i % 3 == 0 { 10 } else if i % 3 == 1 { 100 } else { 1000 };
        let tensor = pool.get_tensor(vec![size, size]);
        tensors.push(tensor);
    }
    
    // Return all tensors
    for tensor in tensors {
        pool.return_tensor(tensor);
    }
    
    let pool_time = start.elapsed();
    
    // Compare with direct allocation
    let start = Instant::now();
    let mut direct_tensors = Vec::new();
    for i in 0..100 {
        let size = if i % 3 == 0 { 10 } else if i % 3 == 1 { 100 } else { 1000 };
        let tensor = AdvancedTensor::new_aligned(vec![size, size]);
        direct_tensors.push(tensor);
    }
    let direct_time = start.elapsed();
    
    println!("Pool allocation (100 tensors): {:?}", pool_time);
    println!("Direct allocation (100 tensors): {:?}", direct_time);
    println!("Pool speedup: {:.2}x", direct_time.as_secs_f64() / pool_time.as_secs_f64());
    
    track_memory_allocation("memory_pool", 100 * 1000 * 4);
}

fn demonstrate_parallel_processing() {
    println!("Demonstrating parallel batch processing...");
    
    let num_threads = 4;
    let processor = ParallelBatchProcessor::new(32, num_threads);
    
    // Create large dataset
    let num_samples = 1000;
    let feature_size = 256;
    let mut data = AdvancedTensor::new_aligned(vec![num_samples, feature_size]);
    
    // Initialize with test data
    for i in 0..num_samples {
        for j in 0..feature_size {
            data.set(&[i, j], (i * feature_size + j) as f32 / 1000.0);
        }
    }
    
    // Process in parallel
    let start = Instant::now();
    let results = processor.process_parallel(&data, |batch| {
        // Simulate some computation (e.g., neural network forward pass)
        let mut output = AdvancedTensor::new_aligned(vec![batch.shape[0], 10]);
        
        // Apply some transformations
        for i in 0..batch.shape[0] {
            for j in 0..10 {
                let mut sum = 0.0;
                for k in 0..batch.shape[1] {
                    sum += batch.get(&[i, k]) * ((j * k) as f32 / 100.0).sin();
                }
                output.set(&[i, j], sum / batch.shape[1] as f32);
            }
        }
        
        output
    });
    
    let parallel_time = start.elapsed();
    
    println!("Processed {} samples in {} batches", num_samples, results.len());
    println!("Parallel processing time: {:?}", parallel_time);
    println!("Throughput: {:.2} samples/sec", num_samples as f64 / parallel_time.as_secs_f64());
    
    track_memory_allocation("parallel_processing", (num_samples * feature_size * 4) as u64);
}

fn demonstrate_cache_oblivious_transpose() {
    println!("Demonstrating cache-oblivious transpose...");
    
    let rows = 2048;
    let cols = 1024;
    
    let mut input = AdvancedTensor::new_aligned(vec![rows, cols]);
    let mut output = AdvancedTensor::new_aligned(vec![cols, rows]);
    
    // Initialize input
    for i in 0..rows {
        for j in 0..cols {
            input.set(&[i, j], (i * cols + j) as f32);
        }
    }
    
    // Perform cache-oblivious transpose
    let start = Instant::now();
    cache_oblivious_transpose(&input, &mut output);
    let transpose_time = start.elapsed();
    
    println!("Cache-oblivious transpose ({}x{}): {:?}", rows, cols, transpose_time);
    
    // Verify correctness
    let mut errors = 0;
    for i in 0..std::cmp::min(rows, 100) {
        for j in 0..std::cmp::min(cols, 100) {
            if (input.get(&[i, j]) - output.get(&[j, i])).abs() > 1e-6 {
                errors += 1;
            }
        }
    }
    
    println!("Verification: {} errors found", errors);
    
    // Calculate throughput
    let elements = rows * cols;
    let throughput = elements as f64 / transpose_time.as_secs_f64() / 1_000_000.0;
    println!("Transpose throughput: {:.2} M elements/sec", throughput);
    
    track_memory_allocation("cache_oblivious", (rows * cols * 4 * 2) as u64);
}

fn compute_cross_entropy_loss(output: &Tensor, target: &Tensor) -> f32 {
    assert_eq!(output.shape, target.shape);
    
    let mut loss = 0.0;
    let batch_size = output.shape[0];
    let num_classes = output.shape[1];
    
    for i in 0..batch_size {
        for j in 0..num_classes {
            let pred = output.get(&[i, j]).max(1e-7); // Avoid log(0)
            let true_val = target.get(&[i, j]);
            loss -= true_val * pred.ln();
        }
    }
    
    loss / batch_size as f32
}

fn compute_loss_gradient(output: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(output.shape, target.shape);
    
    let mut grad = output.clone();
    let batch_size = output.shape[0] as f32;
    
    for i in 0..grad.data.len() {
        grad.data[i] = (output.data[i] - target.data[i]) / batch_size;
    }
    
    grad
}