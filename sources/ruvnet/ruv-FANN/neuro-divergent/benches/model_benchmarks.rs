//! Performance benchmarks for all neuro-divergent model architectures
//!
//! This benchmark suite tests the performance of various model architectures
//! under different configurations and data sizes to identify optimization opportunities.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::models::*;
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use chrono::{DateTime, Utc, TimeZone};
use std::time::Duration;

/// Generate synthetic time series data for benchmarking
fn generate_time_series_data(
    length: usize,
    n_features: usize,
    frequency: &str,
) -> TimeSeriesDataFrame {
    let start_time = Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let mut timestamps = Vec::with_capacity(length);
    let mut values = Vec::with_capacity(length);
    
    // Generate timestamps based on frequency
    let time_delta = match frequency {
        "H" => chrono::Duration::hours(1),
        "D" => chrono::Duration::days(1),
        "M" => chrono::Duration::days(30),
        _ => chrono::Duration::hours(1),
    };
    
    for i in 0..length {
        timestamps.push(start_time + time_delta * i as i32);
        
        // Generate synthetic data with patterns
        let base = (i as f64 * 0.1).sin() * 10.0 + 50.0;
        let trend = i as f64 * 0.05;
        let noise = ((i * 17) % 7) as f64 * 0.1 - 0.35;
        let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 5.0;
        
        let mut features = vec![base + trend + noise + seasonal];
        
        // Add additional features if requested
        for j in 1..n_features {
            let feature = (i as f64 * 0.1 * j as f64).cos() * 5.0 + 
                         ((i * (j + 3)) % 11) as f64 * 0.5;
            features.push(feature);
        }
        
        values.push(features);
    }
    
    TimeSeriesDataFrame::new(
        "benchmark_series".to_string(),
        timestamps,
        values,
        vec!["target".to_string()],
    )
}

/// Benchmark basic model architectures (MLP, DLinear, NLinear)
fn bench_basic_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_models");
    group.measurement_time(Duration::from_secs(30));
    
    let data_sizes = vec![
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
        ("xlarge", 100000),
    ];
    
    for (size_name, size) in data_sizes {
        let data = generate_time_series_data(size, 1, "H");
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark MLP
        group.bench_with_input(
            BenchmarkId::new("MLP", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = MLPConfig::new(24, 12)
                        .with_hidden_layers(vec![128, 64, 32])
                        .with_dropout(0.1);
                    let model = MLP::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark DLinear
        group.bench_with_input(
            BenchmarkId::new("DLinear", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = DLinearConfig::new(24, 12)
                        .with_kernel_size(25)
                        .with_individual(true);
                    let model = DLinear::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark NLinear
        group.bench_with_input(
            BenchmarkId::new("NLinear", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = NLinearConfig::new(24, 12)
                        .with_individual(false);
                    let model = NLinear::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark MLPMultivariate
        group.bench_with_input(
            BenchmarkId::new("MLPMultivariate", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = MLPMultivariateConfig::new(3, 24, 12)
                        .with_hidden_layers(vec![256, 128, 64])
                        .with_batch_norm(true);
                    let model = MLPMultivariate::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark recurrent model architectures (RNN, LSTM, GRU)
fn bench_recurrent_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrent_models");
    group.measurement_time(Duration::from_secs(45));
    group.sample_size(10);
    
    let sequence_lengths = vec![
        ("short", 50),
        ("medium", 200),
        ("long", 1000),
    ];
    
    for (seq_name, seq_len) in sequence_lengths {
        let data = generate_time_series_data(seq_len, 1, "H");
        group.throughput(Throughput::Elements(seq_len as u64));
        
        // Benchmark RNN
        group.bench_with_input(
            BenchmarkId::new("RNN", seq_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = RNNConfig::new(24, 12)
                        .with_hidden_size(128)
                        .with_num_layers(2)
                        .with_dropout(0.1);
                    let model = RNN::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark LSTM
        group.bench_with_input(
            BenchmarkId::new("LSTM", seq_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = LSTMConfig::new(24, 12)
                        .with_hidden_size(128)
                        .with_num_layers(2)
                        .with_bidirectional(false)
                        .with_dropout(0.1);
                    let model = LSTM::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark GRU
        group.bench_with_input(
            BenchmarkId::new("GRU", seq_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = GRUConfig::new(24, 12)
                        .with_hidden_size(128)
                        .with_num_layers(2)
                        .with_dropout(0.1);
                    let model = GRU::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark transformer-based models
fn bench_transformer_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer_models");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(5);
    
    let sequence_lengths = vec![
        ("short", 96),
        ("medium", 192),
        ("long", 336),
    ];
    
    for (seq_name, seq_len) in sequence_lengths {
        let data = generate_time_series_data(seq_len + 100, 1, "H");
        group.throughput(Throughput::Elements(seq_len as u64));
        
        // Benchmark TFT (Temporal Fusion Transformer)
        group.bench_with_input(
            BenchmarkId::new("TFT", seq_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = TFTConfig::new(seq_len, 24)
                        .with_hidden_size(128)
                        .with_num_heads(4)
                        .with_num_layers(2)
                        .with_dropout(0.1);
                    let model = TFT::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark Autoformer
        group.bench_with_input(
            BenchmarkId::new("Autoformer", seq_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = AutoformerConfig::new(seq_len, 24)
                        .with_d_model(256)
                        .with_n_heads(8)
                        .with_e_layers(2)
                        .with_d_layers(1);
                    let model = Autoformer::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark Informer
        group.bench_with_input(
            BenchmarkId::new("Informer", seq_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = InformerConfig::new(seq_len, 24)
                        .with_d_model(256)
                        .with_n_heads(8)
                        .with_e_layers(2)
                        .with_prob_sparse_attn(0.5);
                    let model = Informer::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark advanced architectures (NBEATS, N-HiTS, etc.)
fn bench_advanced_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_models");
    group.measurement_time(Duration::from_secs(45));
    group.sample_size(10);
    
    let horizons = vec![
        ("short_horizon", 12),
        ("medium_horizon", 24),
        ("long_horizon", 48),
    ];
    
    for (horizon_name, horizon) in horizons {
        let data = generate_time_series_data(horizon * 10, 1, "H");
        group.throughput(Throughput::Elements(horizon as u64));
        
        // Benchmark NBEATS
        group.bench_with_input(
            BenchmarkId::new("NBEATS", horizon_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = NBEATSConfig::new(horizon * 2, horizon)
                        .with_num_stacks(3)
                        .with_num_blocks(3)
                        .with_num_layers(4)
                        .with_layer_size(256);
                    let model = NBEATS::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark N-HiTS
        group.bench_with_input(
            BenchmarkId::new("NHiTS", horizon_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = NHiTSConfig::new(horizon * 2, horizon)
                        .with_num_stacks(3)
                        .with_num_blocks(1)
                        .with_num_layers(2)
                        .with_layer_size(512)
                        .with_pool_kernel_sizes(vec![2, 2, 1]);
                    let model = NHiTS::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark NBEATSx
        group.bench_with_input(
            BenchmarkId::new("NBEATSx", horizon_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = NBEATSxConfig::new(horizon * 2, horizon)
                        .with_num_stacks(3)
                        .with_num_blocks(3)
                        .with_exogenous_size(5);
                    let model = NBEATSx::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark specialized models (TCN, BiTCN, TimesNet, etc.)
fn bench_specialized_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("specialized_models");
    group.measurement_time(Duration::from_secs(30));
    
    let input_sizes = vec![
        ("small", 48),
        ("medium", 96),
        ("large", 192),
    ];
    
    for (size_name, input_size) in input_sizes {
        let data = generate_time_series_data(input_size * 5, 1, "H");
        group.throughput(Throughput::Elements(input_size as u64));
        
        // Benchmark TCN
        group.bench_with_input(
            BenchmarkId::new("TCN", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = TCNConfig::new(input_size, 24)
                        .with_num_channels(vec![32, 64, 128])
                        .with_kernel_size(3)
                        .with_dropout(0.1);
                    let model = TCN::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark BiTCN
        group.bench_with_input(
            BenchmarkId::new("BiTCN", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = BiTCNConfig::new(input_size, 24)
                        .with_num_channels(vec![32, 64, 128])
                        .with_kernel_size(3)
                        .with_dropout(0.1);
                    let model = BiTCN::new(config).unwrap();
                    black_box(model);
                });
            },
        );
        
        // Benchmark TimesNet
        group.bench_with_input(
            BenchmarkId::new("TimesNet", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = TimesNetConfig::new(input_size, 24)
                        .with_d_model(64)
                        .with_d_ff(128)
                        .with_num_kernels(6);
                    let model = TimesNet::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark model creation with different complexity levels
fn bench_model_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_complexity");
    
    // Test different model sizes for scalability
    let layer_configs = vec![
        ("tiny", vec![32, 16]),
        ("small", vec![128, 64, 32]),
        ("medium", vec![256, 128, 64, 32]),
        ("large", vec![512, 256, 128, 64, 32]),
        ("xlarge", vec![1024, 512, 256, 128, 64, 32]),
    ];
    
    for (size_name, layers) in layer_configs {
        group.bench_with_input(
            BenchmarkId::new("MLP_layers", size_name),
            &layers,
            |b, layers| {
                b.iter(|| {
                    let config = MLPConfig::new(24, 12)
                        .with_hidden_layers(layers.clone());
                    let model = MLP::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    // Test different sequence lengths for RNNs
    let hidden_sizes = vec![
        ("small", 64),
        ("medium", 128),
        ("large", 256),
        ("xlarge", 512),
    ];
    
    for (size_name, hidden_size) in hidden_sizes {
        group.bench_with_input(
            BenchmarkId::new("LSTM_hidden", size_name),
            &hidden_size,
            |b, &hidden_size| {
                b.iter(|| {
                    let config = LSTMConfig::new(24, 12)
                        .with_hidden_size(hidden_size)
                        .with_num_layers(3);
                    let model = LSTM::new(config).unwrap();
                    black_box(model);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark ensemble model creation
fn bench_ensemble_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_models");
    group.sample_size(10);
    
    let ensemble_sizes = vec![
        ("small", 3),
        ("medium", 5),
        ("large", 10),
    ];
    
    for (size_name, num_models) in ensemble_sizes {
        group.bench_with_input(
            BenchmarkId::new("mixed_ensemble", size_name),
            &num_models,
            |b, &num_models| {
                b.iter(|| {
                    let mut models: Vec<Box<dyn BaseModel<f64>>> = Vec::new();
                    
                    for i in 0..num_models {
                        match i % 4 {
                            0 => {
                                let config = MLPConfig::new(24, 12);
                                models.push(Box::new(MLP::new(config).unwrap()));
                            },
                            1 => {
                                let config = LSTMConfig::new(24, 12);
                                models.push(Box::new(LSTM::new(config).unwrap()));
                            },
                            2 => {
                                let config = DLinearConfig::new(24, 12);
                                models.push(Box::new(DLinear::new(config).unwrap()));
                            },
                            _ => {
                                let config = GRUConfig::new(24, 12);
                                models.push(Box::new(GRU::new(config).unwrap()));
                            },
                        }
                    }
                    
                    black_box(models);
                });
            },
        );
    }
    
    group.finish();
}

// Configure criterion benchmarks
criterion_group!(
    benches,
    bench_basic_models,
    bench_recurrent_models,
    bench_transformer_models,
    bench_advanced_models,
    bench_specialized_models,
    bench_model_complexity,
    bench_ensemble_models,
);

criterion_main!(benches);