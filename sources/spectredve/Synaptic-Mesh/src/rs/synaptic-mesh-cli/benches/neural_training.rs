// Neural training performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

// Mock neural network for benchmarking
struct MockNeuralNetwork {
    layers: Vec<Vec<f32>>,
    weights: Vec<Vec<Vec<f32>>>,
}

impl MockNeuralNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut weights = Vec::new();
        
        for &size in layer_sizes {
            layers.push(vec![0.0; size]);
        }
        
        for i in 0..layer_sizes.len() - 1 {
            let mut layer_weights = Vec::new();
            for _ in 0..layer_sizes[i] {
                layer_weights.push(vec![0.1; layer_sizes[i + 1]]);
            }
            weights.push(layer_weights);
        }
        
        Self { layers, weights }
    }
    
    fn forward_pass(&mut self, input: &[f32]) -> Vec<f32> {
        // Copy input to first layer
        for (i, &val) in input.iter().enumerate() {
            if i < self.layers[0].len() {
                self.layers[0][i] = val;
            }
        }
        
        // Forward propagation through layers
        for layer_idx in 0..self.weights.len() {
            let current_layer = &self.layers[layer_idx].clone();
            let next_layer = &mut self.layers[layer_idx + 1];
            
            for (j, next_neuron) in next_layer.iter_mut().enumerate() {
                *next_neuron = 0.0;
                for (i, &current_val) in current_layer.iter().enumerate() {
                    if i < self.weights[layer_idx].len() && j < self.weights[layer_idx][i].len() {
                        *next_neuron += current_val * self.weights[layer_idx][i][j];
                    }
                }
                // Apply ReLU activation
                *next_neuron = next_neuron.max(0.0);
            }
        }
        
        self.layers.last().unwrap().clone()
    }
    
    fn backward_pass(&mut self, target: &[f32], learning_rate: f32) {
        let output_layer_idx = self.layers.len() - 1;
        let output = &self.layers[output_layer_idx].clone();
        
        // Calculate output error (simplified)
        let mut errors: Vec<Vec<f32>> = vec![vec![0.0; layer.len()]; self.layers.len()];
        
        for (i, (&output_val, &target_val)) in output.iter().zip(target.iter()).enumerate() {
            errors[output_layer_idx][i] = target_val - output_val;
        }
        
        // Backpropagate errors (simplified)
        for layer_idx in (0..self.weights.len()).rev() {
            let current_errors = &errors[layer_idx + 1].clone();
            
            for (i, current_neuron) in self.layers[layer_idx].iter().enumerate() {
                for (j, &error) in current_errors.iter().enumerate() {
                    if i < self.weights[layer_idx].len() && j < self.weights[layer_idx][i].len() {
                        // Update weight
                        self.weights[layer_idx][i][j] += learning_rate * error * current_neuron;
                        
                        // Propagate error back
                        if layer_idx > 0 {
                            errors[layer_idx][i] += error * self.weights[layer_idx][i][j];
                        }
                    }
                }
            }
        }
    }
    
    fn train_batch(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], learning_rate: f32) {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            self.forward_pass(input);
            self.backward_pass(target, learning_rate);
        }
    }
}

fn bench_neural_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_forward_pass");
    
    let layer_sizes = vec![
        vec![10, 5, 1],      // Small network
        vec![100, 50, 10],   // Medium network  
        vec![784, 128, 64, 10], // Large network (MNIST-like)
    ];
    
    for sizes in layer_sizes {
        let mut network = MockNeuralNetwork::new(&sizes);
        let input = vec![0.5; sizes[0]];
        
        group.bench_with_input(
            BenchmarkId::new("forward_pass", format!("{:?}", sizes)),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(network.forward_pass(black_box(input)));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_neural_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_training");
    group.measurement_time(Duration::from_secs(10));
    
    let batch_sizes = vec![1, 10, 32, 64];
    let network_size = vec![100, 50, 10];
    
    for batch_size in batch_sizes {
        let mut network = MockNeuralNetwork::new(&network_size);
        
        let inputs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![(i as f32) / batch_size as f32; 100])
            .collect();
        
        let targets: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![(i as f32) / batch_size as f32; 10])
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("train_batch", batch_size),
            &(inputs, targets),
            |b, (inputs, targets)| {
                b.iter(|| {
                    network.train_batch(
                        black_box(inputs),
                        black_box(targets),
                        black_box(0.01),
                    );
                });
            },
        );
    }
    
    group.finish();
}

fn bench_neural_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_memory");
    
    let network_sizes = vec![
        (1000, "1K parameters"),
        (10000, "10K parameters"),
        (100000, "100K parameters"),
        (1000000, "1M parameters"),
    ];
    
    for (size, description) in network_sizes {
        group.bench_function(
            BenchmarkId::new("memory_allocation", description),
            |b| {
                b.iter(|| {
                    let layers = vec![size / 3, size / 3, size / 3];
                    black_box(MockNeuralNetwork::new(black_box(&layers)));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_neural_parallel_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_parallel");
    
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;
    
    let thread_counts = vec![1, 2, 4, 8];
    
    for thread_count in thread_counts {
        group.bench_function(
            BenchmarkId::new("parallel_training", thread_count),
            |b| {
                b.iter(|| {
                    let networks: Vec<Arc<Mutex<MockNeuralNetwork>>> = (0..thread_count)
                        .map(|_| Arc::new(Mutex::new(MockNeuralNetwork::new(&[100, 50, 10]))))
                        .collect();
                    
                    let handles: Vec<_> = networks.into_iter().map(|network| {
                        thread::spawn(move || {
                            let inputs = vec![vec![0.5; 100]; 10];
                            let targets = vec![vec![0.3; 10]; 10];
                            
                            let mut net = network.lock().unwrap();
                            net.train_batch(&inputs, &targets, 0.01);
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_neural_convergence_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_convergence");
    group.measurement_time(Duration::from_secs(15));
    
    let learning_rates = vec![0.001, 0.01, 0.1, 0.5];
    
    for lr in learning_rates {
        group.bench_function(
            BenchmarkId::new("convergence_speed", format!("lr_{}", lr)),
            |b| {
                b.iter(|| {
                    let mut network = MockNeuralNetwork::new(&[10, 5, 1]);
                    let inputs = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]];
                    let targets = vec![vec![0.5]];
                    
                    // Train until convergence or max epochs
                    for _ in 0..100 {
                        network.train_batch(black_box(&inputs), black_box(&targets), black_box(lr));
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    neural_benches,
    bench_neural_forward_pass,
    bench_neural_training,
    bench_neural_memory_usage,
    bench_neural_parallel_training,
    bench_neural_convergence_speed
);
criterion_main!(neural_benches);