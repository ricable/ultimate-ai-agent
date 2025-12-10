//! SIMD Performance Benchmarks
//! 
//! Comprehensive benchmarks testing SIMD operations vs fallback implementations
//! to validate performance improvements and target achievement.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use kimi_fann_core::simd_operations::*;
use kimi_fann_core::memory_pool::*;
use kimi_fann_core::performance_monitor::*;

fn simd_matrix_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_matrix_operations");
    
    let processor = SIMDProcessor::new();
    let sizes = vec![64, 128, 256, 512, 1024];
    
    for size in sizes {
        let matrix_elements = size * size;
        group.throughput(Throughput::Elements(matrix_elements as u64));
        
        group.bench_with_input(
            BenchmarkId::new("matrix_vector_multiply", size),
            &size,
            |b, &size| {
                let matrix = vec![1.0f32; size * size];
                let vector = vec![1.0f32; size];
                let mut output = vec![0.0f32; size];
                
                b.iter(|| {
                    processor.matrix_vector_mul(
                        black_box(&matrix),
                        black_box(&vector),
                        black_box(&mut output)
                    );
                });
            },
        );
    }
    
    group.finish();
}

fn simd_activation_functions_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_activation_functions");
    
    let processor = SIMDProcessor::new();
    let sizes = vec![1000, 5000, 10000, 50000, 100000];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // ReLU benchmark
        group.bench_with_input(
            BenchmarkId::new("relu", size),
            &size,
            |b, &size| {
                let input = vec![0.5f32; size];
                let mut output = vec![0.0f32; size];
                
                b.iter(|| {
                    processor.relu_activate(
                        black_box(&input),
                        black_box(&mut output)
                    );
                });
            },
        );
        
        // Sigmoid benchmark
        group.bench_with_input(
            BenchmarkId::new("sigmoid", size),
            &size,
            |b, &size| {
                let input = vec![0.5f32; size];
                let mut output = vec![0.0f32; size];
                
                b.iter(|| {
                    processor.sigmoid_activate(
                        black_box(&input),
                        black_box(&mut output)
                    );
                });
            },
        );
    }
    
    group.finish();
}

fn simd_vs_fallback_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_fallback");
    
    let simd_processor = SIMDProcessor::new();
    let fallback_processor = {
        let mut proc = SIMDProcessor::new();
        proc.feature_enabled = false; // Force fallback
        proc
    };
    
    let size = 10000;
    let input = vec![0.5f32; size];
    let mut output_simd = vec![0.0f32; size];
    let mut output_fallback = vec![0.0f32; size];
    
    group.bench_function("relu_simd", |b| {
        b.iter(|| {
            simd_processor.relu_activate(
                black_box(&input),
                black_box(&mut output_simd)
            );
        });
    });
    
    group.bench_function("relu_fallback", |b| {
        b.iter(|| {
            fallback_processor.relu_activate(
                black_box(&input),
                black_box(&mut output_fallback)
            );
        });
    });
    
    // Dot product comparison
    let vector_a = vec![1.0f32; size];
    let vector_b = vec![2.0f32; size];
    
    group.bench_function("dot_product_simd", |b| {
        b.iter(|| {
            black_box(simd_processor.dot_product(
                black_box(&vector_a),
                black_box(&vector_b)
            ));
        });
    });
    
    group.bench_function("dot_product_fallback", |b| {
        b.iter(|| {
            black_box(fallback_processor.dot_product(
                black_box(&vector_a),
                black_box(&vector_b)
            ));
        });
    });
    
    group.finish();
}

fn neural_layer_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_layer_performance");
    
    let layer_configs = vec![
        (64, 32),   // Small layer
        (128, 64),  // Medium layer
        (256, 128), // Large layer
        (512, 256), // Very large layer
    ];
    
    for (input_size, output_size) in layer_configs {
        let layer = SIMDNeuralLayer::new(input_size, output_size);
        let input = vec![0.5f32; input_size];
        let mut output = vec![0.0f32; output_size];
        
        group.throughput(Throughput::Elements((input_size * output_size) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("forward_pass", format!("{}x{}", input_size, output_size)),
            &(input_size, output_size),
            |b, _| {
                b.iter(|| {
                    layer.forward(
                        black_box(&input),
                        black_box(&mut output)
                    );
                });
            },
        );
    }
    
    group.finish();
}

fn memory_pool_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_performance");
    
    // Allocation benchmarks
    let allocation_sizes = vec![1024, 4096, 16384, 65536]; // 1KB to 64KB
    
    for size in allocation_sizes {
        group.bench_with_input(
            BenchmarkId::new("allocate_weights", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut pool = NeuralMemoryPool::new();
                    let allocation = pool.allocate_weights(size).unwrap();
                    pool.deallocate(allocation);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("allocate_activations", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut pool = NeuralMemoryPool::new();
                    let allocation = pool.allocate_activations(size).unwrap();
                    pool.deallocate(allocation);
                });
            },
        );
    }
    
    // Garbage collection benchmark
    group.bench_function("garbage_collection", |b| {
        b.iter(|| {
            let mut pool = NeuralMemoryPool::new();
            
            // Allocate many small blocks
            let mut allocations = Vec::new();
            for _ in 0..100 {
                if let Ok(allocation) = pool.allocate_activations(1024) {
                    allocations.push(allocation);
                }
            }
            
            // Deallocate some
            for allocation in allocations.drain(0..50) {
                pool.deallocate(allocation);
            }
            
            // Run garbage collection
            pool.garbage_collect();
            
            // Clean up remaining
            for allocation in allocations {
                pool.deallocate(allocation);
            }
        });
    });
    
    group.finish();
}

fn performance_monitoring_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_monitoring");
    
    group.bench_function("operation_timing", |b| {
        let monitor = PerformanceMonitor::new();
        
        b.iter(|| {
            let timer = monitor.start_operation(OperationType::NeuralInference);
            
            // Simulate some work
            black_box({
                let sum: f32 = (0..1000).map(|i| i as f32).sum();
                sum
            });
            
            timer.complete();
        });
    });
    
    group.bench_function("memory_recording", |b| {
        let mut monitor = PerformanceMonitor::new();
        
        b.iter(|| {
            monitor.record_memory_usage(black_box(15.5));
        });
    });
    
    group.bench_function("snapshot_generation", |b| {
        let monitor = PerformanceMonitor::new();
        
        b.iter(|| {
            black_box(monitor.get_snapshot());
        });
    });
    
    group.finish();
}

fn batch_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let batch_sizes = vec![1, 8, 16, 32, 64];
    
    for batch_size in batch_sizes {
        let processor = SIMDBatchProcessor::new(batch_size);
        let layer = SIMDNeuralLayer::new(128, 64);
        
        let inputs: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| vec![0.5f32; 128])
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_processing", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(processor.process_batch(
                        black_box(&inputs),
                        black_box(&layer)
                    ));
                });
            },
        );
    }
    
    group.finish();
}

fn inference_latency_target_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_latency_targets");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(1000);
    
    // Target: <50ms neural inference
    group.bench_function("target_inference_50ms", |b| {
        let layer1 = SIMDNeuralLayer::new(256, 128);
        let layer2 = SIMDNeuralLayer::new(128, 64);
        let layer3 = SIMDNeuralLayer::new(64, 32);
        
        let input = vec![0.5f32; 256];
        let mut hidden1 = vec![0.0f32; 128];
        let mut hidden2 = vec![0.0f32; 64];
        let mut output = vec![0.0f32; 32];
        
        b.iter(|| {
            // Simulate full neural network forward pass
            layer1.forward(black_box(&input), black_box(&mut hidden1));
            layer2.forward(black_box(&hidden1), black_box(&mut hidden2));
            layer3.forward(black_box(&hidden2), black_box(&mut output));
            
            black_box(&output);
        });
    });
    
    group.finish();
}

fn throughput_target_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_targets");
    
    // Target: >1000 ops/sec
    let layer = SIMDNeuralLayer::new(64, 32);
    let inputs: Vec<Vec<f32>> = (0..1000)
        .map(|_| vec![0.5f32; 64])
        .collect();
    
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("target_throughput_1000ops", |b| {
        b.iter(|| {
            for input in &inputs {
                let mut output = vec![0.0f32; 32];
                layer.forward(black_box(input), black_box(&mut output));
            }
        });
    });
    
    group.finish();
}

fn memory_efficiency_target_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency_targets");
    
    // Target: <25MB memory usage
    group.bench_function("target_memory_25mb", |b| {
        b.iter(|| {
            let mut pool = NeuralMemoryPool::new();
            let mut allocations = Vec::new();
            
            // Allocate up to memory limit
            let target_bytes = 20 * 1024 * 1024; // 20MB (leaving margin)
            let mut allocated_bytes = 0;
            
            while allocated_bytes < target_bytes {
                if let Ok(allocation) = pool.allocate_weights(1024) {
                    allocated_bytes += allocation.size;
                    allocations.push(allocation);
                } else {
                    break;
                }
            }
            
            let stats = pool.get_memory_stats();
            black_box(stats.is_within_limits());
            
            // Clean up
            for allocation in allocations {
                pool.deallocate(allocation);
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    simd_benches,
    simd_matrix_operations_benchmark,
    simd_activation_functions_benchmark,
    simd_vs_fallback_benchmark,
    neural_layer_performance_benchmark,
    batch_processing_benchmark
);

criterion_group!(
    memory_benches,
    memory_pool_performance_benchmark,
    memory_efficiency_target_benchmark
);

criterion_group!(
    monitoring_benches,
    performance_monitoring_benchmark
);

criterion_group!(
    target_benches,
    inference_latency_target_benchmark,
    throughput_target_benchmark
);

criterion_main!(simd_benches, memory_benches, monitoring_benches, target_benches);