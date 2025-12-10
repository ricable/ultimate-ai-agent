//! Performance benchmarks for data processing operations
//!
//! This benchmark suite measures the performance of data loading, preprocessing,
//! feature engineering, and validation operations for time series data.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuro_divergent::data::*;
use neuro_divergent::data::preprocessing::*;
use neuro_divergent::data::features::*;
use neuro_divergent::data::validation::*;
use neuro_divergent::data::transforms::*;
use neuro_divergent::data::loaders::*;
use chrono::{DateTime, Utc, TimeZone};
use std::time::Duration;
use tempfile::NamedTempFile;
use std::io::Write;
use polars::prelude::*;
use ndarray::{Array1, Array2};

/// Generate synthetic CSV data for benchmarking
fn generate_csv_data(num_series: usize, length: usize, features: usize) -> String {
    let mut csv_data = String::new();
    
    // Header
    csv_data.push_str("series_id,timestamp,");
    for i in 0..features {
        csv_data.push_str(&format!("feature_{},", i));
    }
    csv_data.push_str("target\n");
    
    // Data
    let start_time = Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    
    for series in 0..num_series {
        for t in 0..length {
            csv_data.push_str(&format!("series_{},", series));
            
            let timestamp = start_time + chrono::Duration::hours(t as i64);
            csv_data.push_str(&format!("{},", timestamp.to_rfc3339()));
            
            // Feature values
            for f in 0..features {
                let value = (t as f64 * 0.1 + f as f64).sin() * 10.0 + 50.0 + series as f64 * 5.0;
                csv_data.push_str(&format!("{:.2},", value));
            }
            
            // Target value
            let target = (t as f64 * 0.15).sin() * 15.0 + 60.0 + series as f64 * 3.0;
            csv_data.push_str(&format!("{:.2}\n", target));
        }
    }
    
    csv_data
}

/// Benchmark CSV data loading
fn bench_csv_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_loading");
    group.measurement_time(Duration::from_secs(30));
    
    let data_sizes = vec![
        ("small", 10, 100, 5),
        ("medium", 100, 1000, 10),
        ("large", 1000, 1000, 20),
    ];
    
    for (size_name, num_series, length, features) in data_sizes {
        let csv_data = generate_csv_data(num_series, length, features);
        let data_size = csv_data.len();
        group.throughput(Throughput::Bytes(data_size as u64));
        
        // Write to temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(csv_data.as_bytes()).unwrap();
        let file_path = temp_file.path().to_str().unwrap().to_string();
        
        // Benchmark Polars CSV reading
        group.bench_with_input(
            BenchmarkId::new("polars_read", size_name),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let df = LazyFrame::scan_csv(
                        black_box(path),
                        Default::default()
                    )
                    .collect()
                    .unwrap();
                    black_box(df);
                });
            },
        );
        
        // Benchmark parsing into TimeSeriesDataset
        group.bench_with_input(
            BenchmarkId::new("parse_dataset", size_name),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let loader = CSVDataLoader::new(path.clone())
                        .with_series_id_column("series_id")
                        .with_timestamp_column("timestamp")
                        .with_target_column("target")
                        .with_feature_columns((0..features).map(|i| format!("feature_{}", i)).collect());
                    
                    let dataset = loader.load().unwrap();
                    black_box(dataset);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark data preprocessing operations
fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");
    
    let data_sizes = vec![
        ("small", 1000),
        ("medium", 10000),
        ("large", 100000),
    ];
    
    for (size_name, size) in data_sizes {
        let data: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0 + (i % 7) as f64)
            .collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark StandardScaler
        group.bench_with_input(
            BenchmarkId::new("standard_scaler_fit", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut scaler = StandardScaler::default();
                    scaler.fit(black_box(data)).unwrap();
                    black_box(scaler);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("standard_scaler_transform", size_name),
            &data,
            |b, data| {
                let mut scaler = StandardScaler::default();
                scaler.fit(data).unwrap();
                
                b.iter(|| {
                    let scaled = scaler.transform(black_box(data)).unwrap();
                    black_box(scaled);
                });
            },
        );
        
        // Benchmark MinMaxScaler
        group.bench_with_input(
            BenchmarkId::new("minmax_scaler_fit", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut scaler = MinMaxScaler::new(0.0, 1.0);
                    scaler.fit(black_box(data)).unwrap();
                    black_box(scaler);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("minmax_scaler_transform", size_name),
            &data,
            |b, data| {
                let mut scaler = MinMaxScaler::new(0.0, 1.0);
                scaler.fit(data).unwrap();
                
                b.iter(|| {
                    let scaled = scaler.transform(black_box(data)).unwrap();
                    black_box(scaled);
                });
            },
        );
        
        // Benchmark RobustScaler
        group.bench_with_input(
            BenchmarkId::new("robust_scaler_fit", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut scaler = RobustScaler::default();
                    scaler.fit(black_box(data)).unwrap();
                    black_box(scaler);
                });
            },
        );
        
        // Benchmark differencing
        group.bench_with_input(
            BenchmarkId::new("differencing_order1", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let differ = Differencer::new(1);
                    let differenced = differ.transform(black_box(data)).unwrap();
                    black_box(differenced);
                });
            },
        );
        
        // Benchmark seasonal differencing
        group.bench_with_input(
            BenchmarkId::new("seasonal_differencing", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let differ = SeasonalDifferencer::new(24); // Daily seasonality
                    let differenced = differ.transform(black_box(data)).unwrap();
                    black_box(differenced);
                });
            },
        );
        
        // Benchmark log transformation
        group.bench_with_input(
            BenchmarkId::new("log_transform", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let transformer = LogTransformer::new();
                    let transformed = transformer.transform(black_box(data)).unwrap();
                    black_box(transformed);
                });
            },
        );
        
        // Benchmark box-cox transformation
        group.bench_with_input(
            BenchmarkId::new("boxcox_transform", size_name),
            &data,
            |b, data| {
                let mut transformer = BoxCoxTransformer::new();
                transformer.fit(data).unwrap();
                
                b.iter(|| {
                    let transformed = transformer.transform(black_box(data)).unwrap();
                    black_box(transformed);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark feature engineering operations
fn bench_feature_engineering(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_engineering");
    group.measurement_time(Duration::from_secs(45));
    
    let sequence_lengths = vec![
        ("short", 100),
        ("medium", 1000),
        ("long", 10000),
    ];
    
    for (seq_name, seq_len) in sequence_lengths {
        // Create time series data
        let timestamps: Vec<DateTime<Utc>> = (0..seq_len)
            .map(|i| Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::hours(i as i64))
            .collect();
        
        let values: Vec<f64> = (0..seq_len)
            .map(|i| {
                let trend = i as f64 * 0.05;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 10.0;
                let noise = ((i * 13) % 11) as f64 * 0.5 - 2.5;
                50.0 + trend + seasonal + noise
            })
            .collect();
        
        let series = TimeSeriesDatasetBuilder::new("test_series".to_string())
            .with_frequency("H".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        group.throughput(Throughput::Elements(seq_len as u64));
        
        // Benchmark lag features
        group.bench_with_input(
            BenchmarkId::new("lag_features", seq_name),
            &series,
            |b, series| {
                b.iter(|| {
                    let lags = vec![1, 2, 3, 6, 12, 24];
                    let lag_features = generate_lag_features(&series.values(), &lags);
                    black_box(lag_features);
                });
            },
        );
        
        // Benchmark rolling statistics
        group.bench_with_input(
            BenchmarkId::new("rolling_statistics", seq_name),
            &series,
            |b, series| {
                b.iter(|| {
                    let windows = vec![7, 14, 30];
                    let mut features = Vec::new();
                    
                    for window in windows {
                        features.push(rolling_mean(&series.values(), window));
                        features.push(rolling_std(&series.values(), window));
                        features.push(rolling_min(&series.values(), window));
                        features.push(rolling_max(&series.values(), window));
                    }
                    
                    black_box(features);
                });
            },
        );
        
        // Benchmark temporal features
        group.bench_with_input(
            BenchmarkId::new("temporal_features", seq_name),
            &series,
            |b, series| {
                b.iter(|| {
                    let temporal_features = extract_temporal_features(&series.timestamps());
                    black_box(temporal_features);
                });
            },
        );
        
        // Benchmark Fourier features
        group.bench_with_input(
            BenchmarkId::new("fourier_features", seq_name),
            &series,
            |b, series| {
                b.iter(|| {
                    let periods = vec![24, 168]; // Daily and weekly
                    let fourier_features = generate_fourier_features(series.len(), &periods, 3);
                    black_box(fourier_features);
                });
            },
        );
        
        // Benchmark all features combined
        group.bench_with_input(
            BenchmarkId::new("all_features", seq_name),
            &series,
            |b, series| {
                b.iter(|| {
                    let mut feature_engine = FeatureEngine::new()
                        .with_lag_features(vec![1, 2, 3, 6, 12, 24])
                        .with_rolling_features(vec![7, 14, 30])
                        .with_temporal_features(true)
                        .with_fourier_features(vec![24, 168], 3);
                    
                    let features = feature_engine.generate_features(black_box(series)).unwrap();
                    black_box(features);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark data validation operations
fn bench_data_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_validation");
    
    let dataset_sizes = vec![
        ("small", 10, 1000),
        ("medium", 100, 1000),
        ("large", 1000, 1000),
    ];
    
    for (size_name, num_series, length) in dataset_sizes {
        // Create dataset with some data quality issues
        let mut dataset = TimeSeriesDataset::new();
        
        for s in 0..num_series {
            let timestamps: Vec<DateTime<Utc>> = (0..length)
                .map(|i| {
                    let base = Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
                    // Introduce some gaps
                    if i % 50 == 0 && i > 0 {
                        base + chrono::Duration::hours((i + 2) as i64)
                    } else {
                        base + chrono::Duration::hours(i as i64)
                    }
                })
                .collect();
            
            let values: Vec<f64> = (0..length)
                .map(|i| {
                    // Introduce some missing values and outliers
                    if i % 100 == 0 {
                        f64::NAN
                    } else if i % 200 == 0 {
                        1000.0 // Outlier
                    } else {
                        (i as f64 * 0.1).sin() * 10.0 + 50.0
                    }
                })
                .collect();
            
            let series = TimeSeriesDatasetBuilder::new(format!("series_{}", s))
                .with_frequency("H".to_string())
                .with_values(values)
                .with_timestamps(timestamps)
                .build()
                .unwrap();
            
            dataset.add_series(series);
        }
        
        group.throughput(Throughput::Elements((num_series * length) as u64));
        
        // Benchmark missing value detection
        group.bench_with_input(
            BenchmarkId::new("missing_values", size_name),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let validator = DataValidator::new()
                        .with_missing_value_check(true);
                    
                    let report = validator.validate(black_box(dataset)).unwrap();
                    black_box(report);
                });
            },
        );
        
        // Benchmark outlier detection
        group.bench_with_input(
            BenchmarkId::new("outlier_detection", size_name),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let validator = DataValidator::new()
                        .with_outlier_detection(OutlierMethod::IQR, 1.5);
                    
                    let report = validator.validate(black_box(dataset)).unwrap();
                    black_box(report);
                });
            },
        );
        
        // Benchmark temporal consistency
        group.bench_with_input(
            BenchmarkId::new("temporal_consistency", size_name),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let validator = DataValidator::new()
                        .with_temporal_consistency_check(true);
                    
                    let report = validator.validate(black_box(dataset)).unwrap();
                    black_box(report);
                });
            },
        );
        
        // Benchmark full validation
        group.bench_with_input(
            BenchmarkId::new("full_validation", size_name),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let validator = DataValidator::new()
                        .with_missing_value_check(true)
                        .with_outlier_detection(OutlierMethod::IQR, 1.5)
                        .with_temporal_consistency_check(true)
                        .with_stationarity_check(true)
                        .with_duplicate_check(true);
                    
                    let report = validator.validate(black_box(dataset)).unwrap();
                    black_box(report);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch data loading
fn bench_batch_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_loading");
    
    let batch_configs = vec![
        ("small_batches", 100, 32),
        ("medium_batches", 100, 128),
        ("large_batches", 100, 512),
    ];
    
    for (config_name, num_series, batch_size) in batch_configs {
        // Create dataset
        let mut dataset = TimeSeriesDataset::new();
        
        for s in 0..num_series {
            let timestamps: Vec<DateTime<Utc>> = (0..1000)
                .map(|i| Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::hours(i as i64))
                .collect();
            
            let values: Vec<f64> = (0..1000)
                .map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0)
                .collect();
            
            let series = TimeSeriesDatasetBuilder::new(format!("series_{}", s))
                .with_frequency("H".to_string())
                .with_values(values)
                .with_timestamps(timestamps)
                .build()
                .unwrap();
            
            dataset.add_series(series);
        }
        
        group.throughput(Throughput::Elements(num_series as u64));
        
        // Benchmark sequential batch loading
        group.bench_with_input(
            BenchmarkId::new("sequential", config_name),
            &(dataset.clone(), batch_size),
            |b, (dataset, batch_size)| {
                b.iter(|| {
                    let loader = DataLoader::new(dataset.clone())
                        .with_batch_size(*batch_size)
                        .with_shuffle(false);
                    
                    let mut batch_count = 0;
                    for batch in loader {
                        black_box(&batch);
                        batch_count += 1;
                    }
                    black_box(batch_count);
                });
            },
        );
        
        // Benchmark shuffled batch loading
        group.bench_with_input(
            BenchmarkId::new("shuffled", config_name),
            &(dataset.clone(), batch_size),
            |b, (dataset, batch_size)| {
                b.iter(|| {
                    let loader = DataLoader::new(dataset.clone())
                        .with_batch_size(*batch_size)
                        .with_shuffle(true)
                        .with_seed(42);
                    
                    let mut batch_count = 0;
                    for batch in loader {
                        black_box(&batch);
                        batch_count += 1;
                    }
                    black_box(batch_count);
                });
            },
        );
        
        // Benchmark parallel batch loading
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", config_name),
            &(dataset.clone(), batch_size),
            |b, (dataset, batch_size)| {
                b.iter(|| {
                    let loader = ParallelDataLoader::new(dataset.clone())
                        .with_batch_size(*batch_size)
                        .with_num_workers(4)
                        .with_prefetch_factor(2);
                    
                    let mut batch_count = 0;
                    for batch in loader {
                        black_box(&batch);
                        batch_count += 1;
                    }
                    black_box(batch_count);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark missing value imputation
fn bench_imputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("imputation");
    
    let data_sizes = vec![
        ("small", 1000),
        ("medium", 10000),
        ("large", 100000),
    ];
    
    for (size_name, size) in data_sizes {
        // Create data with missing values
        let data: Vec<f64> = (0..size)
            .map(|i| {
                if i % 20 == 0 {
                    f64::NAN
                } else {
                    (i as f64 * 0.1).sin() * 10.0 + 50.0
                }
            })
            .collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark forward fill
        group.bench_with_input(
            BenchmarkId::new("forward_fill", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = ForwardFillImputer::new();
                    let imputed = imputer.impute(black_box(data)).unwrap();
                    black_box(imputed);
                });
            },
        );
        
        // Benchmark backward fill
        group.bench_with_input(
            BenchmarkId::new("backward_fill", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = BackwardFillImputer::new();
                    let imputed = imputer.impute(black_box(data)).unwrap();
                    black_box(imputed);
                });
            },
        );
        
        // Benchmark linear interpolation
        group.bench_with_input(
            BenchmarkId::new("linear_interpolation", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = LinearInterpolationImputer::new();
                    let imputed = imputer.impute(black_box(data)).unwrap();
                    black_box(imputed);
                });
            },
        );
        
        // Benchmark spline interpolation
        group.bench_with_input(
            BenchmarkId::new("spline_interpolation", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = SplineInterpolationImputer::new(3);
                    let imputed = imputer.impute(black_box(data)).unwrap();
                    black_box(imputed);
                });
            },
        );
        
        // Benchmark seasonal decomposition imputation
        group.bench_with_input(
            BenchmarkId::new("seasonal_imputation", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let imputer = SeasonalDecompositionImputer::new(24);
                    let imputed = imputer.impute(black_box(data)).unwrap();
                    black_box(imputed);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark data augmentation techniques
fn bench_augmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("augmentation");
    
    let data_sizes = vec![
        ("small", 1000),
        ("medium", 5000),
        ("large", 10000),
    ];
    
    for (size_name, size) in data_sizes {
        let data: Vec<f64> = (0..size)
            .map(|i| {
                let trend = i as f64 * 0.05;
                let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 24.0).sin() * 10.0;
                50.0 + trend + seasonal
            })
            .collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark jittering
        group.bench_with_input(
            BenchmarkId::new("jittering", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let augmenter = JitteringAugmenter::new(0.1, Some(42));
                    let augmented = augmenter.augment(black_box(data), 5).unwrap();
                    black_box(augmented);
                });
            },
        );
        
        // Benchmark scaling
        group.bench_with_input(
            BenchmarkId::new("scaling", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let augmenter = ScalingAugmenter::new(0.8, 1.2, Some(42));
                    let augmented = augmenter.augment(black_box(data), 5).unwrap();
                    black_box(augmented);
                });
            },
        );
        
        // Benchmark magnitude warping
        group.bench_with_input(
            BenchmarkId::new("magnitude_warping", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let augmenter = MagnitudeWarpingAugmenter::new(0.2, 3, Some(42));
                    let augmented = augmenter.augment(black_box(data), 5).unwrap();
                    black_box(augmented);
                });
            },
        );
        
        // Benchmark time warping
        group.bench_with_input(
            BenchmarkId::new("time_warping", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let augmenter = TimeWarpingAugmenter::new(0.2, 3, Some(42));
                    let augmented = augmenter.augment(black_box(data), 5).unwrap();
                    black_box(augmented);
                });
            },
        );
        
        // Benchmark window slicing
        group.bench_with_input(
            BenchmarkId::new("window_slicing", size_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let augmenter = WindowSlicingAugmenter::new(0.8);
                    let augmented = augmenter.augment(black_box(data), 10).unwrap();
                    black_box(augmented);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark array operations
fn bench_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_operations");
    
    let array_sizes = vec![
        ("small", 100, 50),
        ("medium", 1000, 100),
        ("large", 5000, 200),
    ];
    
    for (size_name, rows, cols) in array_sizes {
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64 * 0.01).sin())
            .collect();
        let array = Array2::from_shape_vec((rows, cols), data).unwrap();
        
        group.throughput(Throughput::Elements((rows * cols) as u64));
        
        // Benchmark array slicing
        group.bench_with_input(
            BenchmarkId::new("array_slicing", size_name),
            &array,
            |b, array| {
                b.iter(|| {
                    let slice = array.slice(s![..rows/2, ..cols/2]);
                    black_box(slice);
                });
            },
        );
        
        // Benchmark array transposition
        group.bench_with_input(
            BenchmarkId::new("array_transpose", size_name),
            &array,
            |b, array| {
                b.iter(|| {
                    let transposed = array.t();
                    black_box(transposed);
                });
            },
        );
        
        // Benchmark array broadcasting
        group.bench_with_input(
            BenchmarkId::new("array_broadcasting", size_name),
            &array,
            |b, array| {
                let row_means = array.mean_axis(Axis(1)).unwrap();
                b.iter(|| {
                    let normalized = array - &row_means.insert_axis(Axis(1));
                    black_box(normalized);
                });
            },
        );
        
        // Benchmark array aggregation
        group.bench_with_input(
            BenchmarkId::new("array_aggregation", size_name),
            &array,
            |b, array| {
                b.iter(|| {
                    let mean = array.mean();
                    let std = array.std(0.0);
                    let min = array.fold(f64::INFINITY, |a, &b| a.min(b));
                    let max = array.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    black_box((mean, std, min, max));
                });
            },
        );
    }
    
    group.finish();
}

// Configure criterion benchmarks
criterion_group!(
    benches,
    bench_csv_loading,
    bench_preprocessing,
    bench_feature_engineering,
    bench_data_validation,
    bench_batch_loading,
    bench_imputation,
    bench_augmentation,
    bench_array_operations,
);

criterion_main!(benches);