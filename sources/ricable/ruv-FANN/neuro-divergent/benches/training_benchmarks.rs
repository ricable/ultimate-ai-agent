//! Performance benchmarks for training algorithms and optimizers
//!
//! This benchmark suite evaluates the performance of different training configurations,
//! optimizers, learning rate schedules, and loss functions specifically for time series
//! forecasting tasks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::models::*;
use neuro_divergent::training::*;
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use chrono::{DateTime, Utc, TimeZone};
use std::time::Duration;
use num_traits::Float;

/// Generate synthetic training data with seasonal patterns
fn generate_training_data<T: Float>(
    num_series: usize,
    length: usize,
    features: usize,
) -> TrainingData<T> {
    let mut inputs = Vec::with_capacity(num_series);
    let mut targets = Vec::with_capacity(num_series);
    let mut metadata = Vec::with_capacity(num_series);
    
    for series_idx in 0..num_series {
        let mut series_inputs = Vec::with_capacity(length);
        let mut series_targets = Vec::with_capacity(length);
        
        for t in 0..length {
            let t_float = T::from(t).unwrap();
            let series_offset = T::from(series_idx).unwrap();
            
            // Generate multi-feature input with different patterns
            let mut input_features = Vec::with_capacity(features);
            for f in 0..features {
                let f_float = T::from(f + 1).unwrap();
                let value = (t_float * T::from(0.1).unwrap() * f_float).sin() * T::from(10.0).unwrap() +
                           (t_float * T::from(0.02).unwrap()).cos() * T::from(5.0).unwrap() +
                           t_float * T::from(0.05).unwrap() +
                           series_offset * T::from(2.0).unwrap();
                input_features.push(vec![value]);
            }
            series_inputs.push(input_features);
            
            // Generate target (next value prediction)
            let target = (t_float * T::from(0.1).unwrap()).sin() * T::from(10.0).unwrap() + 
                        t_float * T::from(0.05).unwrap() +
                        series_offset * T::from(2.0).unwrap();
            series_targets.push(vec![vec![target]]);
        }
        
        inputs.push(series_inputs);
        targets.push(series_targets);
        
        metadata.push(TimeSeriesMetadata {
            id: format!("series_{}", series_idx),
            frequency: "H".to_string(),
            seasonal_periods: vec![24, 168], // Daily and weekly seasonality
            scale: Some(1.0),
        });
    }
    
    TrainingData {
        inputs,
        targets,
        exogenous: None,
        static_features: None,
        metadata,
    }
}

/// Benchmark different optimizers
fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizers");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);
    
    let data_sizes = vec![
        ("small", 10, 100),
        ("medium", 50, 500),
        ("large", 100, 1000),
    ];
    
    let optimizers = vec![
        ("Adam", OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999))),
        ("AdamW", OptimizerType::AdamW(AdamW::new(0.001, 0.9, 0.999, 0.01))),
        ("SGD", OptimizerType::SGD(SGD::new(0.01).with_momentum(0.9))),
        ("RMSprop", OptimizerType::RMSprop(RMSprop::new(0.01).with_alpha(0.99))),
        ("ForecastingAdam", OptimizerType::ForecastingAdam(ForecastingAdam::new(0.001, 0.9, 0.999))),
    ];
    
    for (size_name, num_series, length) in data_sizes {
        let training_data = generate_training_data::<f64>(num_series, length, 3);
        group.throughput(Throughput::Elements((num_series * length) as u64));
        
        for (opt_name, optimizer) in &optimizers {
            group.bench_with_input(
                BenchmarkId::new(*opt_name, size_name),
                &training_data,
                |b, data| {
                    b.iter(|| {
                        // Create a simple model for testing
                        let config = MLPConfig::new(24, 12)
                            .with_hidden_layers(vec![64, 32]);
                        let mut model = MLP::new(config).unwrap();
                        
                        // Create trainer
                        let mut trainer = Trainer::new()
                            .with_optimizer(optimizer.clone())
                            .with_loss_function(MSELoss::new())
                            .build();
                        
                        // Train for a few epochs
                        let config = TrainingConfig {
                            max_epochs: 5,
                            batch_size: 32,
                            validation_frequency: 5,
                            patience: None,
                            gradient_clip: Some(1.0),
                            mixed_precision: false,
                            seed: Some(42),
                            device: DeviceConfig::Cpu { num_threads: None },
                            checkpoint: CheckpointConfig {
                                enabled: false,
                                save_frequency: 10,
                                keep_best_only: true,
                                monitor_metric: "loss".to_string(),
                                mode: CheckpointMode::Min,
                            },
                        };
                        
                        let result = trainer.train(&mut model, data, &config);
                        black_box(result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark loss functions for time series
fn bench_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_functions");
    
    let prediction_sizes = vec![
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
        ("xlarge", 100000),
    ];
    
    for (size_name, size) in prediction_sizes {
        // Generate predictions and targets
        let predictions: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0)
            .collect();
        let targets: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0 + ((i * 7) % 5) as f64 * 0.5)
            .collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark different loss functions
        let loss_functions: Vec<(&str, Box<dyn LossFunction<f64>>)> = vec![
            ("MSE", Box::new(MSELoss::new())),
            ("MAE", Box::new(MAELoss::new())),
            ("RMSE", Box::new(RMSELoss::new())),
            ("MAPE", Box::new(MAPELoss::new())),
            ("SMAPE", Box::new(SMAPELoss::new())),
            ("MASE", Box::new(MASELoss::new(1.0))),
            ("Huber", Box::new(HuberLoss::new(1.0))),
            ("LogCosh", Box::new(LogCoshLoss::new())),
            ("Quantile_0.5", Box::new(QuantileLoss::new(0.5))),
            ("Quantile_0.9", Box::new(QuantileLoss::new(0.9))),
        ];
        
        for (loss_name, loss_fn) in &loss_functions {
            // Benchmark forward pass
            group.bench_with_input(
                BenchmarkId::new(format!("{}_forward", loss_name), size_name),
                &(&predictions, &targets),
                |b, (preds, targs)| {
                    b.iter(|| {
                        let loss = loss_fn.forward(black_box(preds), black_box(targs));
                        black_box(loss);
                    });
                },
            );
            
            // Benchmark backward pass
            group.bench_with_input(
                BenchmarkId::new(format!("{}_backward", loss_name), size_name),
                &(&predictions, &targets),
                |b, (preds, targs)| {
                    b.iter(|| {
                        let gradients = loss_fn.backward(black_box(preds), black_box(targs));
                        black_box(gradients);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark learning rate schedulers
fn bench_schedulers(c: &mut Criterion) {
    let mut group = c.benchmark_group("lr_schedulers");
    
    let epoch_counts = vec![
        ("short", 100),
        ("medium", 1000),
        ("long", 10000),
    ];
    
    for (duration_name, epochs) in epoch_counts {
        group.throughput(Throughput::Elements(epochs as u64));
        
        let schedulers: Vec<(&str, Box<dyn LRScheduler<f64>>)> = vec![
            ("Exponential", Box::new(ExponentialScheduler::new(0.001, 0.95))),
            ("Step", Box::new(StepScheduler::new(0.001, 10, 0.5))),
            ("MultiStep", Box::new(MultiStepScheduler::new(0.001, vec![30, 60, 90], 0.1))),
            ("Cosine", Box::new(CosineScheduler::new(0.001, 0.0001, epochs))),
            ("Plateau", Box::new(PlateauScheduler::new(0.001, 0.5, 10, 0.0001))),
            ("WarmupCosine", Box::new(WarmupCosineScheduler::new(0.0, 0.001, 10, 0.0001, epochs))),
            ("OneCycle", Box::new(OneCycleScheduler::new(0.0001, 0.001, epochs))),
        ];
        
        for (scheduler_name, mut scheduler) in schedulers {
            group.bench_with_input(
                BenchmarkId::new(scheduler_name, duration_name),
                &epochs,
                |b, &epochs| {
                    b.iter(|| {
                        let mut learning_rates = Vec::with_capacity(epochs);
                        let mut metrics = EpochMetrics {
                            epoch: 0,
                            loss: 1.0,
                            learning_rate: 0.001,
                            gradient_norm: Some(0.1),
                            additional_metrics: Default::default(),
                        };
                        
                        for epoch in 0..epochs {
                            metrics.epoch = epoch;
                            metrics.loss = 1.0 / (epoch as f64 + 1.0); // Simulated decreasing loss
                            
                            let lr = scheduler.get_lr(&metrics);
                            scheduler.step(&metrics);
                            learning_rates.push(lr);
                        }
                        
                        black_box(learning_rates);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark training with different batch sizes
fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    let batch_sizes = vec![
        ("tiny", 8),
        ("small", 32),
        ("medium", 128),
        ("large", 512),
    ];
    
    let num_samples = 1000;
    let training_data = generate_training_data::<f64>(num_samples, 100, 3);
    
    for (size_name, batch_size) in batch_sizes {
        group.throughput(Throughput::Elements(num_samples as u64));
        
        group.bench_with_input(
            BenchmarkId::new("training", size_name),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = MLPConfig::new(24, 12)
                        .with_hidden_layers(vec![128, 64]);
                    let mut model = MLP::new(config).unwrap();
                    
                    let mut trainer = Trainer::new()
                        .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
                        .with_loss_function(MSELoss::new())
                        .build();
                    
                    let training_config = TrainingConfig {
                        max_epochs: 2,
                        batch_size,
                        validation_frequency: 10,
                        patience: None,
                        gradient_clip: Some(1.0),
                        mixed_precision: false,
                        seed: Some(42),
                        device: DeviceConfig::Cpu { num_threads: None },
                        checkpoint: CheckpointConfig {
                            enabled: false,
                            save_frequency: 10,
                            keep_best_only: true,
                            monitor_metric: "loss".to_string(),
                            mode: CheckpointMode::Min,
                        },
                    };
                    
                    let result = trainer.train(&mut model, &training_data, &training_config);
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark gradient clipping strategies
fn bench_gradient_clipping(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_clipping");
    
    let gradient_sizes = vec![
        ("small", vec![vec![0.1; 100], vec![0.2; 50]]),
        ("medium", vec![vec![0.1; 1000], vec![0.2; 500], vec![0.3; 250]]),
        ("large", vec![vec![0.1; 10000], vec![0.2; 5000], vec![0.3; 2500], vec![0.4; 1000]]),
    ];
    
    for (size_name, gradients) in gradient_sizes {
        let total_params: usize = gradients.iter().map(|g| g.len()).sum();
        group.throughput(Throughput::Elements(total_params as u64));
        
        // Benchmark norm calculation
        group.bench_with_input(
            BenchmarkId::new("norm_calculation", size_name),
            &gradients,
            |b, grads| {
                b.iter(|| {
                    let norm = utils::gradient_norm(black_box(grads));
                    black_box(norm);
                });
            },
        );
        
        // Benchmark gradient clipping
        let clip_values = vec![0.5, 1.0, 5.0, 10.0];
        for clip_value in clip_values {
            group.bench_with_input(
                BenchmarkId::new(format!("clip_{}", clip_value), size_name),
                &gradients,
                |b, grads| {
                    b.iter_batched(
                        || grads.clone(),
                        |mut grads_copy| {
                            let norm = utils::clip_gradients_by_norm(&mut grads_copy, clip_value);
                            black_box((grads_copy, norm));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark training callbacks
fn bench_callbacks(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_callbacks");
    
    let training_data = generate_training_data::<f64>(100, 200, 3);
    
    let callback_configs = vec![
        ("no_callbacks", vec![]),
        ("early_stopping", vec![
            Box::new(EarlyStopping::new("loss", 10, CheckpointMode::Min)) as Box<dyn Callback<f64>>
        ]),
        ("model_checkpoint", vec![
            Box::new(ModelCheckpoint::new("model.bin", "loss", CheckpointMode::Min, true))
        ]),
        ("progress_logger", vec![
            Box::new(ProgressLogger::new(10))
        ]),
        ("all_callbacks", vec![
            Box::new(EarlyStopping::new("loss", 10, CheckpointMode::Min)),
            Box::new(ModelCheckpoint::new("model.bin", "loss", CheckpointMode::Min, true)),
            Box::new(ProgressLogger::new(10)),
            Box::new(TensorBoardLogger::new("./logs")),
        ]),
    ];
    
    for (callback_name, callbacks) in callback_configs {
        group.bench_with_input(
            BenchmarkId::new("training", callback_name),
            &callbacks,
            |b, callbacks| {
                b.iter(|| {
                    let config = MLPConfig::new(24, 12)
                        .with_hidden_layers(vec![64, 32]);
                    let mut model = MLP::new(config).unwrap();
                    
                    let mut trainer = Trainer::new()
                        .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
                        .with_loss_function(MSELoss::new())
                        .with_callbacks(callbacks.clone())
                        .build();
                    
                    let training_config = TrainingConfig {
                        max_epochs: 5,
                        batch_size: 32,
                        validation_frequency: 2,
                        patience: Some(3),
                        gradient_clip: Some(1.0),
                        mixed_precision: false,
                        seed: Some(42),
                        device: DeviceConfig::Cpu { num_threads: None },
                        checkpoint: CheckpointConfig {
                            enabled: false,
                            save_frequency: 10,
                            keep_best_only: true,
                            monitor_metric: "loss".to_string(),
                            mode: CheckpointMode::Min,
                        },
                    };
                    
                    let result = trainer.train(&mut model, &training_data, &training_config);
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel training configurations
#[cfg(feature = "parallel")]
fn bench_parallel_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_training");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(5);
    
    let training_data = generate_training_data::<f64>(1000, 500, 5);
    
    let thread_counts = vec![1, 2, 4, 8];
    
    for num_threads in thread_counts {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let config = MLPConfig::new(24, 12)
                        .with_hidden_layers(vec![256, 128, 64]);
                    let mut model = MLP::new(config).unwrap();
                    
                    let mut trainer = Trainer::new()
                        .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
                        .with_loss_function(MSELoss::new())
                        .build();
                    
                    let training_config = TrainingConfig {
                        max_epochs: 3,
                        batch_size: 64,
                        validation_frequency: 5,
                        patience: None,
                        gradient_clip: Some(1.0),
                        mixed_precision: false,
                        seed: Some(42),
                        device: DeviceConfig::Cpu { num_threads: Some(num_threads) },
                        checkpoint: CheckpointConfig {
                            enabled: false,
                            save_frequency: 10,
                            keep_best_only: true,
                            monitor_metric: "loss".to_string(),
                            mode: CheckpointMode::Min,
                        },
                    };
                    
                    let result = trainer.train(&mut model, &training_data, &training_config);
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency of different training configurations
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(10);
    
    let data_configs = vec![
        ("small_data", 100, 100, 3),
        ("medium_data", 500, 500, 5),
        ("large_data", 1000, 1000, 10),
    ];
    
    for (config_name, num_series, length, features) in data_configs {
        let training_data = generate_training_data::<f64>(num_series, length, features);
        
        // Benchmark with gradient accumulation
        let accumulation_steps = vec![1, 4, 8, 16];
        
        for acc_steps in accumulation_steps {
            group.bench_with_input(
                BenchmarkId::new(format!("grad_acc_{}", acc_steps), config_name),
                &acc_steps,
                |b, &acc_steps| {
                    b.iter(|| {
                        let config = MLPConfig::new(24, 12)
                            .with_hidden_layers(vec![128, 64]);
                        let mut model = MLP::new(config).unwrap();
                        
                        let mut trainer = Trainer::new()
                            .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
                            .with_loss_function(MSELoss::new())
                            .with_gradient_accumulation_steps(acc_steps)
                            .build();
                        
                        let training_config = TrainingConfig {
                            max_epochs: 2,
                            batch_size: 16, // Small batch size to test accumulation
                            validation_frequency: 10,
                            patience: None,
                            gradient_clip: Some(1.0),
                            mixed_precision: false,
                            seed: Some(42),
                            device: DeviceConfig::Cpu { num_threads: None },
                            checkpoint: CheckpointConfig {
                                enabled: false,
                                save_frequency: 10,
                                keep_best_only: true,
                                monitor_metric: "loss".to_string(),
                                mode: CheckpointMode::Min,
                            },
                        };
                        
                        let result = trainer.train(&mut model, &training_data, &training_config);
                        black_box(result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

// Configure criterion benchmarks
criterion_group!(
    benches,
    bench_optimizers,
    bench_loss_functions,
    bench_schedulers,
    bench_batch_sizes,
    bench_gradient_clipping,
    bench_callbacks,
    bench_memory_efficiency,
);

#[cfg(feature = "parallel")]
criterion_group!(
    parallel_benches,
    bench_parallel_training,
);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);