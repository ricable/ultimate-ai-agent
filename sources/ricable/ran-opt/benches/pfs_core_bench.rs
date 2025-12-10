use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use ran_opt::pfs_core::*;
use std::time::Duration;

fn bench_tensor_creation(c: &mut Criterion) {
    c.bench_function("tensor_creation_1000x1000", |b| {
        b.iter(|| {
            let tensor = Tensor::new(vec![1000, 1000]);
            black_box(tensor);
        })
    });
    
    c.bench_function("tensor_creation_10000x100", |b| {
        b.iter(|| {
            let tensor = Tensor::new(vec![10000, 100]);
            black_box(tensor);
        })
    });
}

fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    
    group.bench_function("add_1000x1000", |b| {
        b.iter_batched(
            || {
                let a = Tensor::from_vec(
                    (0..1_000_000).map(|i| i as f32).collect(),
                    vec![1000, 1000]
                );
                let b = Tensor::from_vec(
                    (0..1_000_000).map(|i| (i + 1) as f32).collect(),
                    vec![1000, 1000]
                );
                (a, b)
            },
            |(a, b)| {
                black_box(a.add(&b));
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("mul_1000x1000", |b| {
        b.iter_batched(
            || {
                let a = Tensor::from_vec(
                    (0..1_000_000).map(|i| i as f32).collect(),
                    vec![1000, 1000]
                );
                let b = Tensor::from_vec(
                    (0..1_000_000).map(|i| (i + 1) as f32).collect(),
                    vec![1000, 1000]
                );
                (a, b)
            },
            |(a, b)| {
                black_box(a.mul(&b));
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("matmul_512x512", |b| {
        b.iter_batched(
            || {
                let a = Tensor::from_vec(
                    (0..262144).map(|i| (i as f32 / 1000.0).sin()).collect(),
                    vec![512, 512]
                );
                let b = Tensor::from_vec(
                    (0..262144).map(|i| (i as f32 / 1000.0).cos()).collect(),
                    vec![512, 512]
                );
                (a, b)
            },
            |(a, b)| {
                black_box(a.matmul(&b));
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("transpose_1000x1000", |b| {
        b.iter_batched(
            || {
                Tensor::from_vec(
                    (0..1_000_000).map(|i| i as f32).collect(),
                    vec![1000, 1000]
                )
            },
            |tensor| {
                black_box(tensor.transpose());
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");
    
    let tensor = Tensor::from_vec(
        (0..1_000_000).map(|i| (i as f32 / 100000.0) - 5.0).collect(),
        vec![1000, 1000]
    );
    
    group.bench_function("relu_forward", |b| {
        let activation = Activation::ReLU;
        b.iter(|| {
            black_box(activation.forward(&tensor));
        });
    });
    
    group.bench_function("sigmoid_forward", |b| {
        let activation = Activation::Sigmoid;
        b.iter(|| {
            black_box(activation.forward(&tensor));
        });
    });
    
    group.bench_function("tanh_forward", |b| {
        let activation = Activation::Tanh;
        b.iter(|| {
            black_box(activation.forward(&tensor));
        });
    });
    
    group.bench_function("leaky_relu_forward", |b| {
        let activation = Activation::LeakyReLU(0.01);
        b.iter(|| {
            black_box(activation.forward(&tensor));
        });
    });
    
    let small_tensor = Tensor::from_vec(
        (0..10000).map(|i| (i as f32 / 1000.0) - 5.0).collect(),
        vec![100, 100]
    );
    
    group.bench_function("softmax_forward", |b| {
        let activation = Activation::Softmax;
        b.iter(|| {
            black_box(activation.forward(&small_tensor));
        });
    });
    
    group.finish();
}

fn bench_neural_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network");
    
    // Create a simple 3-layer network
    let mut network = NeuralNetwork::new();
    network.add_layer(Box::new(DenseLayer::new(784, 128)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(128, 64)), Activation::ReLU);
    network.add_layer(Box::new(DenseLayer::new(64, 10)), Activation::Softmax);
    
    group.bench_function("forward_pass_batch_32", |b| {
        let input = Tensor::from_vec(
            (0..25088).map(|i| (i as f32 / 1000.0).sin()).collect(),
            vec![32, 784]
        );
        
        b.iter(|| {
            black_box(network.forward(&input));
        });
    });
    
    group.bench_function("forward_pass_batch_128", |b| {
        let input = Tensor::from_vec(
            (0..100352).map(|i| (i as f32 / 1000.0).sin()).collect(),
            vec![128, 784]
        );
        
        b.iter(|| {
            black_box(network.forward(&input));
        });
    });
    
    group.finish();
}

fn bench_batch_processor(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processor");
    
    let processor = BatchProcessor::new(64);
    let data = Tensor::from_vec(
        (0..784000).map(|i| (i as f32 / 1000.0).sin()).collect(),
        vec![1000, 784]
    );
    
    group.bench_function("process_1000_samples", |b| {
        b.iter(|| {
            let results = processor.process_batches(&data, |batch| {
                // Simple operation for benchmarking
                batch.apply(|x| x * 2.0 + 1.0)
            });
            black_box(results);
        });
    });
    
    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    group.bench_function("allocate_large_tensor", |b| {
        b.iter(|| {
            let tensor = Tensor::new(vec![2048, 2048]);
            black_box(tensor);
        });
    });
    
    group.bench_function("allocate_many_small_tensors", |b| {
        b.iter(|| {
            let tensors: Vec<Tensor> = (0..100)
                .map(|_| Tensor::new(vec![128, 128]))
                .collect();
            black_box(tensors);
        });
    });
    
    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    // Test SIMD efficiency by comparing with scalar operations
    let data_size = 1_000_000;
    let a: Vec<f32> = (0..data_size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..data_size).map(|i| (i + 1) as f32).collect();
    
    group.bench_function("simd_add", |b| {
        let tensor_a = Tensor::from_vec(a.clone(), vec![1000, 1000]);
        let tensor_b = Tensor::from_vec(b.clone(), vec![1000, 1000]);
        
        b.iter(|| {
            black_box(tensor_a.add(&tensor_b));
        });
    });
    
    group.bench_function("simd_mul", |b| {
        let tensor_a = Tensor::from_vec(a.clone(), vec![1000, 1000]);
        let tensor_b = Tensor::from_vec(b.clone(), vec![1000, 1000]);
        
        b.iter(|| {
            black_box(tensor_a.mul(&tensor_b));
        });
    });
    
    group.finish();
}

fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizers");
    
    let mut weights = Tensor::from_vec(
        (0..1000000).map(|i| (i as f32 / 1000.0).sin()).collect(),
        vec![1000, 1000]
    );
    
    let grads = Tensor::from_vec(
        (0..1000000).map(|i| (i as f32 / 100000.0)).collect(),
        vec![1000, 1000]
    );
    
    group.bench_function("sgd_update", |b| {
        let sgd = SGD::new(0.01);
        b.iter(|| {
            sgd.update(&mut weights, &grads);
        });
    });
    
    group.bench_function("adam_update", |b| {
        let adam = Adam::new(0.001);
        b.iter(|| {
            adam.update(&mut weights, &grads);
        });
    });
    
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(50);
    targets = 
        bench_tensor_creation,
        bench_tensor_operations,
        bench_activation_functions,
        bench_neural_network,
        bench_batch_processor,
        bench_memory_allocation,
        bench_simd_operations,
        bench_optimizers
}

criterion_main!(benches);