use super::*;
use crate::pfs_core::advanced::*;
use crate::pfs_core::profiler::*;
use std::time::Duration;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_complete_neural_network_pipeline() {
        let mut network = NeuralNetwork::new();
        network.add_layer(Box::new(DenseLayer::new(4, 8)), Activation::ReLU);
        network.add_layer(Box::new(DenseLayer::new(8, 4)), Activation::ReLU);
        network.add_layer(Box::new(DenseLayer::new(4, 2)), Activation::Softmax);
        
        // Create sample input
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4]
        );
        
        // Forward pass
        let output = network.forward(&input);
        assert_eq!(output.shape, vec![2, 2]);
        
        // Check output is valid probabilities (softmax)
        for i in 0..2 {
            let sum = output.get(&[i, 0]) + output.get(&[i, 1]);
            assert!((sum - 1.0).abs() < 1e-5);
        }
        
        // Create dummy gradients for backward pass
        let grad = Tensor::from_vec(
            vec![0.1, 0.2, 0.3, 0.4],
            vec![2, 2]
        );
        
        // Backward pass
        let _input_grad = network.backward(&grad);
        
        // Update weights
        let optimizer = SGD::new(0.01);
        network.update_weights(&optimizer);
    }
    
    #[test]
    fn test_batch_processing() {
        let processor = BatchProcessor::new(2);
        
        let data = Tensor::from_vec(
            (0..20).map(|i| i as f32).collect(),
            vec![5, 4]
        );
        
        let results = processor.process_batches(&data, |batch| {
            batch.apply(|x| x * 2.0)
        });
        
        assert_eq!(results.len(), 3); // 5 samples with batch size 2 = 3 batches
        assert_eq!(results[0].shape, vec![2, 4]);
        assert_eq!(results[1].shape, vec![2, 4]);
        assert_eq!(results[2].shape, vec![1, 4]); // Last batch has only 1 sample
    }
    
    #[test]
    fn test_tensor_operations_correctness() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]);
        
        // Test addition
        let c = a.add(&b);
        assert_eq!(c.data, vec![3.0, 2.0, 4.0, 6.0]);
        
        // Test multiplication
        let d = a.mul(&b);
        assert_eq!(d.data, vec![2.0, 0.0, 3.0, 8.0]);
        
        // Test matrix multiplication
        let e = a.matmul(&b);
        assert_eq!(e.data, vec![4.0, 4.0, 10.0, 8.0]);
        
        // Test transpose
        let f = a.transpose();
        assert_eq!(f.data, vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(f.shape, vec![2, 2]);
    }
    
    #[test]
    fn test_activation_functions() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5]);
        
        // Test ReLU
        let relu_output = Activation::ReLU.forward(&input);
        assert_eq!(relu_output.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        
        // Test Sigmoid
        let sigmoid_output = Activation::Sigmoid.forward(&input);
        for &val in &sigmoid_output.data {
            assert!(val >= 0.0 && val <= 1.0);
        }
        
        // Test Tanh
        let tanh_output = Activation::Tanh.forward(&input);
        for &val in &tanh_output.data {
            assert!(val >= -1.0 && val <= 1.0);
        }
        
        // Test LeakyReLU
        let leaky_relu_output = Activation::LeakyReLU(0.1).forward(&input);
        assert_eq!(leaky_relu_output.data, vec![-0.2, -0.1, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_advanced_tensor_operations() {
        let mut a = AdvancedTensor::new_aligned(vec![100, 100]);
        let mut b = AdvancedTensor::new_aligned(vec![100, 100]);
        let mut result = AdvancedTensor::new_aligned(vec![100, 100]);
        
        // Initialize with test data
        for i in 0..100 {
            for j in 0..100 {
                a.set(&[i, j], (i * 100 + j) as f32);
                b.set(&[i, j], (i * 100 + j + 1) as f32);
            }
        }
        
        // Test SIMD addition
        simd_ops::simd_add(&a, &b, &mut result);
        
        // Verify results
        for i in 0..100 {
            for j in 0..100 {
                let expected = (i * 100 + j) as f32 + (i * 100 + j + 1) as f32;
                assert_eq!(result.get(&[i, j]), expected);
            }
        }
    }
    
    #[test]
    fn test_profiler_integration() {
        let profiler = Profiler::new();
        
        // Test timing
        profiler.time("test_operation", || {
            std::thread::sleep(Duration::from_millis(1));
        });
        
        // Test counters
        profiler.increment_counter("operations");
        profiler.add_to_counter("operations", 5);
        
        let stats = profiler.get_stats();
        assert!(stats.timing_stats.contains_key("test_operation"));
        assert_eq!(stats.counter_stats["operations"], 6);
    }
    
    #[test]
    fn test_memory_tracking() {
        let tracker = MemoryTracker::new();
        
        tracker.track_allocation("test_tensors", 1024);
        tracker.track_allocation("test_tensors", 2048);
        tracker.track_deallocation("test_tensors", 1024);
        
        let usage = tracker.get_memory_usage();
        assert_eq!(usage["test_tensors"], (2048, 1));
    }
    
    #[test]
    fn test_optimizer_behavior() {
        let mut weights = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let grads = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
        
        let sgd = SGD::new(0.1);
        sgd.update(&mut weights, &grads);
        
        // Check that weights were updated
        assert_eq!(weights.data, vec![0.99, 1.98, 2.97, 3.96]);
    }
    
    #[test]
    fn test_large_tensor_operations() {
        let size = 1000;
        let a = Tensor::from_vec(
            (0..size*size).map(|i| (i as f32) / (size as f32)).collect(),
            vec![size, size]
        );
        
        let b = Tensor::from_vec(
            (0..size*size).map(|i| ((i + 1) as f32) / (size as f32)).collect(),
            vec![size, size]
        );
        
        // Test large matrix operations
        let c = a.add(&b);
        assert_eq!(c.shape, vec![size, size]);
        
        let d = a.mul(&b);
        assert_eq!(d.shape, vec![size, size]);
        
        // Test transpose
        let e = a.transpose();
        assert_eq!(e.shape, vec![size, size]);
    }
    
    #[test]
    fn test_parallel_batch_processing() {
        let processor = ParallelBatchProcessor::new(10, 4);
        let data = AdvancedTensor::new_aligned(vec![100, 50]);
        
        // Initialize data
        for i in 0..100 {
            for j in 0..50 {
                data.set(&[i, j], (i * 50 + j) as f32);
            }
        }
        
        let results = processor.process_parallel(&data, |batch| {
            let mut output = AdvancedTensor::new_aligned(batch.shape.clone());
            
            // Apply some transformation
            for i in 0..batch.shape[0] {
                for j in 0..batch.shape[1] {
                    output.set(&[i, j], batch.get(&[i, j]) * 2.0);
                }
            }
            
            output
        });
        
        assert_eq!(results.len(), 10); // 100 samples with batch size 10
        
        // Verify results
        for result in &results {
            assert_eq!(result.shape[1], 50);
        }
    }
    
    #[test]
    fn test_cache_oblivious_transpose() {
        let mut input = AdvancedTensor::new_aligned(vec![64, 32]);
        let mut output = AdvancedTensor::new_aligned(vec![32, 64]);
        
        // Initialize input
        for i in 0..64 {
            for j in 0..32 {
                input.set(&[i, j], (i * 32 + j) as f32);
            }
        }
        
        cache_oblivious_transpose(&input, &mut output);
        
        // Verify transpose
        for i in 0..64 {
            for j in 0..32 {
                assert_eq!(output.get(&[j, i]), input.get(&[i, j]));
            }
        }
    }
    
    #[test]
    fn test_tensor_pool() {
        let mut pool = TensorPool::with_sizes(vec![100, 1000, 10000]);
        
        // Get tensors
        let tensor1 = pool.get_tensor(vec![10, 10]);
        let tensor2 = pool.get_tensor(vec![10, 10]);
        
        // Return tensors
        pool.return_tensor(tensor1);
        pool.return_tensor(tensor2);
        
        // Get tensor again (should reuse)
        let tensor3 = pool.get_tensor(vec![10, 10]);
        assert_eq!(tensor3.shape, vec![10, 10]);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_tensor_operations() {
        let size = 512;
        let a = Tensor::from_vec(
            (0..size*size).map(|i| i as f32).collect(),
            vec![size, size]
        );
        let b = Tensor::from_vec(
            (0..size*size).map(|i| (i + 1) as f32).collect(),
            vec![size, size]
        );
        
        let start = Instant::now();
        let _c = a.matmul(&b);
        let matmul_time = start.elapsed();
        
        let start = Instant::now();
        let _d = a.add(&b);
        let add_time = start.elapsed();
        
        let start = Instant::now();
        let _e = a.transpose();
        let transpose_time = start.elapsed();
        
        println!("Matrix multiplication ({}x{}): {:?}", size, size, matmul_time);
        println!("Matrix addition ({}x{}): {:?}", size, size, add_time);
        println!("Matrix transpose ({}x{}): {:?}", size, size, transpose_time);
        
        // Ensure operations complete in reasonable time
        assert!(matmul_time < Duration::from_secs(1));
        assert!(add_time < Duration::from_millis(100));
        assert!(transpose_time < Duration::from_millis(100));
    }
    
    #[test]
    fn benchmark_neural_network_forward() {
        let mut network = NeuralNetwork::new();
        network.add_layer(Box::new(DenseLayer::new(784, 256)), Activation::ReLU);
        network.add_layer(Box::new(DenseLayer::new(256, 128)), Activation::ReLU);
        network.add_layer(Box::new(DenseLayer::new(128, 10)), Activation::Softmax);
        
        let input = Tensor::from_vec(
            (0..784*32).map(|i| (i as f32) / 1000.0).collect(),
            vec![32, 784]
        );
        
        let start = Instant::now();
        let _output = network.forward(&input);
        let forward_time = start.elapsed();
        
        println!("Neural network forward pass (batch=32): {:?}", forward_time);
        
        // Ensure forward pass completes in reasonable time
        assert!(forward_time < Duration::from_millis(500));
    }
    
    #[test]
    fn benchmark_simd_operations() {
        let mut a = AdvancedTensor::new_aligned(vec![1000, 1000]);
        let mut b = AdvancedTensor::new_aligned(vec![1000, 1000]);
        let mut result = AdvancedTensor::new_aligned(vec![1000, 1000]);
        
        // Initialize
        for i in 0..1000 {
            for j in 0..1000 {
                a.set(&[i, j], i as f32);
                b.set(&[i, j], j as f32);
            }
        }
        
        let start = Instant::now();
        simd_ops::simd_add(&a, &b, &mut result);
        let simd_add_time = start.elapsed();
        
        let start = Instant::now();
        simd_ops::simd_mul(&a, &b, &mut result);
        let simd_mul_time = start.elapsed();
        
        println!("SIMD addition (1M elements): {:?}", simd_add_time);
        println!("SIMD multiplication (1M elements): {:?}", simd_mul_time);
        
        // Ensure SIMD operations are fast
        assert!(simd_add_time < Duration::from_millis(50));
        assert!(simd_mul_time < Duration::from_millis(50));
    }
}