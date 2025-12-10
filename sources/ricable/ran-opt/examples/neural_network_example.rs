use ran_opt::pfs_core::*;
use std::time::Instant;

fn main() {
    println!("PFS Core Neural Network Example");
    println!("===============================");
    
    // Create a simple feedforward neural network
    let mut network = NeuralNetwork::new();
    network.add_layer(Box::new(DenseLayer::new(784, 256)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(256, 128)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(128, 10)), Activation::Softmax);
    
    println!("Created neural network with architecture: 784 -> 256 -> 128 -> 10");
    
    // Create sample input data (simulating MNIST-like data)
    let batch_size = 64;
    let input_size = 784;
    let input_data: Vec<f32> = (0..batch_size * input_size)
        .map(|i| (i as f32 / 1000.0).sin())
        .collect();
    
    let input = Tensor::from_vec(input_data, vec![batch_size, input_size]);
    
    println!("Input tensor shape: {:?}", input.shape);
    
    // Benchmark forward pass
    let start = Instant::now();
    let num_iterations = 100;
    
    for _ in 0..num_iterations {
        let output = network.forward(&input);
        // Prevent optimization from removing the computation
        std::hint::black_box(output);
    }
    
    let duration = start.elapsed();
    println!("Forward pass benchmark:");
    println!("  {} iterations in {:?}", num_iterations, duration);
    println!("  Average time per iteration: {:?}", duration / num_iterations);
    
    // Demonstrate tensor operations
    println!("\nTensor Operations Demo:");
    
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    
    println!("Tensor A: {:?}", a.data);
    println!("Tensor B: {:?}", b.data);
    
    let c = a.add(&b);
    println!("A + B: {:?}", c.data);
    
    let d = a.mul(&b);
    println!("A * B (element-wise): {:?}", d.data);
    
    let e = a.matmul(&b);
    println!("A @ B (matrix multiply): {:?}", e.data);
    
    // Demonstrate activation functions
    println!("\nActivation Functions Demo:");
    
    let test_tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5]);
    println!("Input: {:?}", test_tensor.data);
    
    let relu_output = Activation::ReLU.forward(&test_tensor);
    println!("ReLU: {:?}", relu_output.data);
    
    let sigmoid_output = Activation::Sigmoid.forward(&test_tensor);
    println!("Sigmoid: {:?}", sigmoid_output.data);
    
    let tanh_output = Activation::Tanh.forward(&test_tensor);
    println!("Tanh: {:?}", tanh_output.data);
    
    let leaky_relu_output = Activation::LeakyReLU(0.01).forward(&test_tensor);
    println!("LeakyReLU: {:?}", leaky_relu_output.data);
    
    // Demonstrate batch processing
    println!("\nBatch Processing Demo:");
    
    let processor = BatchProcessor::new(16);
    let large_data = Tensor::from_vec(
        (0..10000).map(|i| (i as f32 / 100.0).sin()).collect(),
        vec![100, 100]
    );
    
    let start = Instant::now();
    let results = processor.process_batches(&large_data, |batch| {
        // Apply a simple transformation
        batch.apply(|x| x * 2.0 + 1.0)
    });
    let duration = start.elapsed();
    
    println!("Processed {} samples in {} batches", large_data.shape[0], results.len());
    println!("Batch processing time: {:?}", duration);
    
    // Memory usage demonstration
    println!("\nMemory Usage Demo:");
    
    let start = Instant::now();
    let large_tensor = Tensor::new(vec![2048, 2048]);
    let alloc_time = start.elapsed();
    
    println!("Allocated tensor of size {}x{} in {:?}", 
             large_tensor.shape[0], large_tensor.shape[1], alloc_time);
    
    // SIMD operations benchmark
    println!("\nSIMD Operations Benchmark:");
    
    let simd_a = Tensor::from_vec(
        (0..1000000).map(|i| i as f32).collect(),
        vec![1000, 1000]
    );
    let simd_b = Tensor::from_vec(
        (0..1000000).map(|i| (i + 1) as f32).collect(),
        vec![1000, 1000]
    );
    
    let start = Instant::now();
    let simd_result = simd_a.add(&simd_b);
    let simd_time = start.elapsed();
    
    println!("SIMD addition of 1M elements: {:?}", simd_time);
    println!("Result sample: {:?}", &simd_result.data[0..5]);
    
    println!("\nDemo completed successfully!");
}