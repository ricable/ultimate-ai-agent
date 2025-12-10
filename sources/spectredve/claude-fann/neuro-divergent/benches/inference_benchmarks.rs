//! Performance benchmarks for inference and prediction speed
//!
//! This benchmark suite measures the performance of model inference across different
//! scenarios including single predictions, batch predictions, streaming predictions,
//! and multi-step forecasting.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::models::*;
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use chrono::{DateTime, Utc, TimeZone};
use std::time::Duration;
use num_traits::Float;
use rayon::prelude::*;

/// Generate pre-trained model weights for consistent benchmarking
fn initialize_model_weights<M: BaseModel<f64>>(model: &mut M) {
    // In real scenarios, we would load pre-trained weights
    // For benchmarking, we'll use consistent random initialization
    model.randomize_weights(-0.1, 0.1, Some(42));
}

/// Generate test data for inference
fn generate_inference_data(
    batch_size: usize,
    sequence_length: usize,
    features: usize,
) -> Vec<TimeSeriesDataFrame> {
    let mut data_frames = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let start_time = Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
        let mut timestamps = Vec::with_capacity(sequence_length);
        let mut values = Vec::with_capacity(sequence_length);
        
        for t in 0..sequence_length {
            timestamps.push(start_time + chrono::Duration::hours(t as i64));
            
            let mut feature_values = Vec::with_capacity(features);
            for f in 0..features {
                let value = ((t + i * 10) as f64 * 0.1).sin() * 10.0 + 
                           (f as f64 * 2.0) +
                           ((t * f + i) % 7) as f64 * 0.5;
                feature_values.push(value);
            }
            values.push(feature_values);
        }
        
        data_frames.push(TimeSeriesDataFrame::new(
            format!("series_{}", i),
            timestamps,
            values,
            (0..features).map(|f| format!("feature_{}", f)).collect(),
        ));
    }
    
    data_frames
}

/// Benchmark single prediction (one series, one forecast)
fn bench_single_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_prediction");
    
    let model_configs = vec![
        ("MLP", {
            let config = MLPConfig::new(24, 12);
            let mut model = MLP::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("LSTM", {
            let config = LSTMConfig::new(24, 12).with_hidden_size(64);
            let mut model = LSTM::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("DLinear", {
            let config = DLinearConfig::new(24, 12);
            let mut model = DLinear::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("NBEATS", {
            let config = NBEATSConfig::new(24, 12).with_num_stacks(2).with_num_blocks(2);
            let mut model = NBEATS::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
    ];
    
    let sequence_lengths = vec![24, 48, 96, 192];
    
    for (model_name, model) in model_configs {
        for seq_len in &sequence_lengths {
            let data = generate_inference_data(1, *seq_len, 1);
            
            group.bench_with_input(
                BenchmarkId::new(model_name, seq_len),
                &data[0],
                |b, data| {
                    b.iter(|| {
                        let prediction = model.predict(black_box(data));
                        black_box(prediction);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark batch predictions (multiple series at once)
fn bench_batch_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prediction");
    group.measurement_time(Duration::from_secs(30));
    
    let batch_sizes = vec![
        ("small", 10),
        ("medium", 50),
        ("large", 100),
        ("xlarge", 500),
    ];
    
    let model_configs = vec![
        ("MLP", {
            let config = MLPConfig::new(24, 12).with_hidden_layers(vec![64, 32]);
            let mut model = MLP::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("LSTM", {
            let config = LSTMConfig::new(24, 12).with_hidden_size(64);
            let mut model = LSTM::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("GRU", {
            let config = GRUConfig::new(24, 12).with_hidden_size(64);
            let mut model = GRU::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("TCN", {
            let config = TCNConfig::new(24, 12).with_num_channels(vec![32, 64]);
            let mut model = TCN::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
    ];
    
    for (batch_name, batch_size) in batch_sizes {
        let data = generate_inference_data(batch_size, 48, 1);
        group.throughput(Throughput::Elements(batch_size as u64));
        
        for (model_name, model) in &model_configs {
            group.bench_with_input(
                BenchmarkId::new(*model_name, batch_name),
                &data,
                |b, data| {
                    b.iter(|| {
                        let predictions: Vec<_> = data.iter()
                            .map(|series| model.predict(black_box(series)))
                            .collect();
                        black_box(predictions);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark streaming predictions (continuous prediction updates)
fn bench_streaming_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_prediction");
    
    let stream_lengths = vec![
        ("short", 100),
        ("medium", 1000),
        ("long", 10000),
    ];
    
    let window_size = 24;
    let models = vec![
        ("MLP", {
            let config = MLPConfig::new(window_size, 1);
            let mut model = MLP::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("LSTM_stateful", {
            let config = LSTMConfig::new(window_size, 1)
                .with_stateful(true)
                .with_hidden_size(32);
            let mut model = LSTM::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("DLinear", {
            let config = DLinearConfig::new(window_size, 1);
            let mut model = DLinear::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
    ];
    
    for (stream_name, stream_length) in stream_lengths {
        // Generate continuous stream of data
        let full_data = generate_inference_data(1, stream_length + window_size, 1)[0].clone();
        group.throughput(Throughput::Elements(stream_length as u64));
        
        for (model_name, mut model) in models.clone() {
            group.bench_with_input(
                BenchmarkId::new(model_name, stream_name),
                &full_data,
                |b, data| {
                    b.iter(|| {
                        let mut predictions = Vec::with_capacity(stream_length);
                        
                        // Simulate streaming prediction
                        for i in 0..stream_length {
                            // Get window of data
                            let window_data = data.slice(i, i + window_size);
                            
                            // Make prediction
                            let pred = model.predict(black_box(&window_data));
                            predictions.push(pred);
                            
                            // Update model state if stateful
                            if model_name.contains("stateful") {
                                model.update_state(&window_data);
                            }
                        }
                        
                        black_box(predictions);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark multi-step forecasting
fn bench_multistep_forecasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("multistep_forecasting");
    group.measurement_time(Duration::from_secs(45));
    
    let forecast_horizons = vec![
        ("short", 12),
        ("medium", 24),
        ("long", 48),
        ("xlong", 96),
    ];
    
    let models = vec![
        ("MLP_direct", {
            let config = MLPConfig::new(48, 24); // Direct multi-step
            let mut model = MLP::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("LSTM_seq2seq", {
            let config = LSTMConfig::new(48, 24)
                .with_hidden_size(128)
                .with_seq2seq(true);
            let mut model = LSTM::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("NBEATS", {
            let config = NBEATSConfig::new(48, 24)
                .with_num_stacks(3)
                .with_num_blocks(3);
            let mut model = NBEATS::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
        ("TFT", {
            let config = TFTConfig::new(48, 24)
                .with_hidden_size(64)
                .with_num_heads(4);
            let mut model = TFT::new(config).unwrap();
            initialize_model_weights(&mut model);
            model
        }),
    ];
    
    for (horizon_name, horizon) in forecast_horizons {
        let data = generate_inference_data(10, 100, 3); // Multi-variate data
        group.throughput(Throughput::Elements((10 * horizon) as u64));
        
        for (model_name, mut model) in models.clone() {
            // Update model config for the specific horizon
            model.set_forecast_horizon(horizon);
            
            group.bench_with_input(
                BenchmarkId::new(model_name, horizon_name),
                &data,
                |b, data| {
                    b.iter(|| {
                        let predictions: Vec<_> = data.iter()
                            .map(|series| {
                                model.predict_multistep(black_box(series), horizon)
                            })
                            .collect();
                        black_box(predictions);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark parallel inference
#[cfg(feature = "parallel")]
fn bench_parallel_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_inference");
    group.measurement_time(Duration::from_secs(60));
    
    let parallel_configs = vec![
        ("sequential", 1),
        ("parallel_2", 2),
        ("parallel_4", 4),
        ("parallel_8", 8),
    ];
    
    let num_series = 1000;
    let data = generate_inference_data(num_series, 48, 1);
    
    let model_config = MLPConfig::new(24, 12).with_hidden_layers(vec![64, 32]);
    let mut model = MLP::new(model_config).unwrap();
    initialize_model_weights(&mut model);
    
    for (config_name, num_threads) in parallel_configs {
        group.throughput(Throughput::Elements(num_series as u64));
        
        group.bench_with_input(
            BenchmarkId::new("MLP", config_name),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    if num_threads == 1 {
                        // Sequential processing
                        let predictions: Vec<_> = data.iter()
                            .map(|series| model.predict(series))
                            .collect();
                        black_box(predictions);
                    } else {
                        // Parallel processing
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(num_threads)
                            .build()
                            .unwrap();
                        
                        let predictions: Vec<_> = pool.install(|| {
                            data.par_iter()
                                .map(|series| model.predict(series))
                                .collect()
                        });
                        black_box(predictions);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark inference with different data types and precision
fn bench_precision_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_inference");
    
    let data_f32 = generate_inference_data(100, 48, 1)
        .into_iter()
        .map(|df| df.convert::<f32>())
        .collect::<Vec<_>>();
    
    let data_f64 = generate_inference_data(100, 48, 1);
    
    // Benchmark f32 models
    {
        let config = MLPConfig::new(24, 12);
        let mut model_f32 = MLP::<f32>::new(config).unwrap();
        initialize_model_weights(&mut model_f32);
        
        group.bench_with_input(
            BenchmarkId::new("MLP", "f32"),
            &data_f32,
            |b, data| {
                b.iter(|| {
                    let predictions: Vec<_> = data.iter()
                        .map(|series| model_f32.predict(black_box(series)))
                        .collect();
                    black_box(predictions);
                });
            },
        );
    }
    
    // Benchmark f64 models
    {
        let config = MLPConfig::new(24, 12);
        let mut model_f64 = MLP::<f64>::new(config).unwrap();
        initialize_model_weights(&mut model_f64);
        
        group.bench_with_input(
            BenchmarkId::new("MLP", "f64"),
            &data_f64,
            |b, data| {
                b.iter(|| {
                    let predictions: Vec<_> = data.iter()
                        .map(|series| model_f64.predict(black_box(series)))
                        .collect();
                    black_box(predictions);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark model warm-up and first prediction latency
fn bench_first_prediction_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_prediction_latency");
    group.sample_size(50); // More samples for cold start benchmarks
    
    let model_configs = vec![
        ("MLP_small", MLPConfig::new(24, 12).with_hidden_layers(vec![32])),
        ("MLP_large", MLPConfig::new(24, 12).with_hidden_layers(vec![256, 128, 64])),
        ("LSTM", LSTMConfig::new(24, 12).with_hidden_size(128)),
        ("NBEATS", NBEATSConfig::new(24, 12).with_num_stacks(3)),
    ];
    
    let data = generate_inference_data(1, 48, 1)[0].clone();
    
    for (model_name, config) in model_configs {
        group.bench_with_input(
            BenchmarkId::new("cold_start", model_name),
            &config,
            |b, config| {
                b.iter_with_setup(
                    || {
                        // Create and initialize model fresh each time
                        match model_name {
                            "MLP_small" | "MLP_large" => {
                                let mut model = MLP::new(config.clone()).unwrap();
                                initialize_model_weights(&mut model);
                                Box::new(model) as Box<dyn BaseModel<f64>>
                            },
                            "LSTM" => {
                                let mut model = LSTM::new(config.clone()).unwrap();
                                initialize_model_weights(&mut model);
                                Box::new(model)
                            },
                            "NBEATS" => {
                                let mut model = NBEATS::new(config.clone()).unwrap();
                                initialize_model_weights(&mut model);
                                Box::new(model)
                            },
                            _ => panic!("Unknown model"),
                        }
                    },
                    |model| {
                        // First prediction (cold)
                        let prediction = model.predict(black_box(&data));
                        black_box(prediction);
                    },
                );
            },
        );
        
        // Also benchmark warm predictions for comparison
        group.bench_with_input(
            BenchmarkId::new("warm", model_name),
            &config,
            |b, config| {
                let mut model = match model_name {
                    "MLP_small" | "MLP_large" => {
                        let mut m = MLP::new(config.clone()).unwrap();
                        initialize_model_weights(&mut m);
                        Box::new(m) as Box<dyn BaseModel<f64>>
                    },
                    "LSTM" => {
                        let mut m = LSTM::new(config.clone()).unwrap();
                        initialize_model_weights(&mut m);
                        Box::new(m)
                    },
                    "NBEATS" => {
                        let mut m = NBEATS::new(config.clone()).unwrap();
                        initialize_model_weights(&mut m);
                        Box::new(m)
                    },
                    _ => panic!("Unknown model"),
                };
                
                // Warm up the model
                for _ in 0..10 {
                    let _ = model.predict(&data);
                }
                
                b.iter(|| {
                    let prediction = model.predict(black_box(&data));
                    black_box(prediction);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark ensemble inference
fn bench_ensemble_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_inference");
    group.measurement_time(Duration::from_secs(45));
    
    let ensemble_sizes = vec![
        ("small", 3),
        ("medium", 5),
        ("large", 10),
    ];
    
    let data = generate_inference_data(50, 48, 1);
    
    for (size_name, num_models) in ensemble_sizes {
        let mut models: Vec<Box<dyn BaseModel<f64>>> = Vec::with_capacity(num_models);
        
        // Create diverse ensemble
        for i in 0..num_models {
            match i % 4 {
                0 => {
                    let config = MLPConfig::new(24, 12).with_hidden_layers(vec![64, 32]);
                    let mut model = MLP::new(config).unwrap();
                    initialize_model_weights(&mut model);
                    models.push(Box::new(model));
                },
                1 => {
                    let config = LSTMConfig::new(24, 12).with_hidden_size(64);
                    let mut model = LSTM::new(config).unwrap();
                    initialize_model_weights(&mut model);
                    models.push(Box::new(model));
                },
                2 => {
                    let config = DLinearConfig::new(24, 12);
                    let mut model = DLinear::new(config).unwrap();
                    initialize_model_weights(&mut model);
                    models.push(Box::new(model));
                },
                _ => {
                    let config = GRUConfig::new(24, 12).with_hidden_size(64);
                    let mut model = GRU::new(config).unwrap();
                    initialize_model_weights(&mut model);
                    models.push(Box::new(model));
                },
            }
        }
        
        group.throughput(Throughput::Elements((data.len() * num_models) as u64));
        
        // Benchmark sequential ensemble
        group.bench_with_input(
            BenchmarkId::new("sequential", size_name),
            &models,
            |b, models| {
                b.iter(|| {
                    let ensemble_predictions: Vec<Vec<_>> = data.iter()
                        .map(|series| {
                            models.iter()
                                .map(|model| model.predict(black_box(series)))
                                .collect()
                        })
                        .collect();
                    
                    // Average predictions
                    let final_predictions: Vec<_> = ensemble_predictions.iter()
                        .map(|preds| {
                            let sum: f64 = preds.iter()
                                .map(|p| p.point_forecast)
                                .sum();
                            sum / preds.len() as f64
                        })
                        .collect();
                    
                    black_box(final_predictions);
                });
            },
        );
        
        // Benchmark parallel ensemble
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", size_name),
            &models,
            |b, models| {
                b.iter(|| {
                    let ensemble_predictions: Vec<Vec<_>> = data.par_iter()
                        .map(|series| {
                            models.par_iter()
                                .map(|model| model.predict(black_box(series)))
                                .collect()
                        })
                        .collect();
                    
                    // Average predictions in parallel
                    let final_predictions: Vec<_> = ensemble_predictions.par_iter()
                        .map(|preds| {
                            let sum: f64 = preds.iter()
                                .map(|p| p.point_forecast)
                                .sum();
                            sum / preds.len() as f64
                        })
                        .collect();
                    
                    black_box(final_predictions);
                });
            },
        );
    }
    
    group.finish();
}

// Configure criterion benchmarks
criterion_group!(
    benches,
    bench_single_prediction,
    bench_batch_prediction,
    bench_streaming_prediction,
    bench_multistep_forecasting,
    bench_precision_inference,
    bench_first_prediction_latency,
    bench_ensemble_inference,
);

#[cfg(feature = "parallel")]
criterion_group!(
    parallel_benches,
    bench_parallel_inference,
);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);