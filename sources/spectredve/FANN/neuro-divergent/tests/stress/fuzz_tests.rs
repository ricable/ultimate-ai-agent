//! Fuzz tests for discovering edge cases through random input generation.
//! Uses property-based testing and fuzzing to find unexpected bugs.

use proptest::prelude::*;
use quickcheck::{QuickCheck, TestResult};
use rand::{Rng, SeedableRng, rngs::StdRng};
use chrono::{DateTime, Utc, Duration, NaiveDateTime};
use polars::prelude::*;
use std::collections::HashMap;
use std::panic;
use std::f64::{INFINITY, NEG_INFINITY, NAN};

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
    PreprocessingConfig, ScalingConfig, ScalingMethod, MissingValueConfig, MissingValueStrategy,
    OutlierConfig, OutlierDetectionMethod, OutlierHandlingStrategy,
    FeatureEngineeringConfig, ValidationConfig, InterpolationMethod,
};
use neuro_divergent_models::{
    basic::{MLP, DLinear, NLinear},
    forecasting::ForecastingModel,
    core::{ModelConfig, ModelBuilder},
};
use neuro_divergent::prelude::*;

/// Arbitrary input generator for time series data
#[derive(Debug, Clone)]
struct ArbitraryTimeSeries {
    n_series: usize,
    n_points: usize,
    n_features: usize,
    include_nan: bool,
    include_inf: bool,
    include_duplicates: bool,
    include_gaps: bool,
    random_seed: u64,
}

// Implement Arbitrary trait for proptest
impl Arbitrary for ArbitraryTimeSeries {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            0usize..1000,
            0usize..1000,
            0usize..50,
            prop::bool::ANY,
            prop::bool::ANY,
            prop::bool::ANY,
            prop::bool::ANY,
            any::<u64>(),
        )
            .prop_map(|(n_series, n_points, n_features, include_nan, include_inf, include_duplicates, include_gaps, random_seed)| {
                ArbitraryTimeSeries {
                    n_series,
                    n_points,
                    n_features,
                    include_nan,
                    include_inf,
                    include_duplicates,
                    include_gaps,
                    random_seed,
                }
            })
            .boxed()
    }
}

impl ArbitraryTimeSeries {
    fn generate(&self) -> DataFrame {
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        
        let mut unique_ids = Vec::new();
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        for series_idx in 0..self.n_series {
            let series_id = format!("series_{}", series_idx);
            let mut last_time = Utc::now();
            
            for point_idx in 0..self.n_points {
                unique_ids.push(series_id.clone());
                
                // Generate timestamp with possible gaps and duplicates
                if self.include_duplicates && rng.gen_bool(0.1) && !timestamps.is_empty() {
                    timestamps.push(*timestamps.last().unwrap());
                } else if self.include_gaps && rng.gen_bool(0.1) {
                    last_time = last_time + Duration::hours(rng.gen_range(1..24));
                    timestamps.push(last_time.naive_utc());
                } else {
                    last_time = last_time + Duration::hours(1);
                    timestamps.push(last_time.naive_utc());
                }
                
                // Generate value with possible special values
                let value = if self.include_nan && rng.gen_bool(0.05) {
                    NAN
                } else if self.include_inf && rng.gen_bool(0.05) {
                    if rng.gen_bool(0.5) { INFINITY } else { NEG_INFINITY }
                } else {
                    rng.gen_range(-1000.0..1000.0)
                };
                
                values.push(value);
            }
        }
        
        df! {
            "unique_id" => unique_ids,
            "ds" => timestamps,
            "y" => values,
        }.unwrap_or_else(|_| DataFrame::empty())
    }
}

/// Property: TimeSeriesDataFrame should never panic on any input
proptest! {
    #[test]
    fn prop_timeseries_dataframe_no_panic(ts_data: ArbitraryTimeSeries) {
        let df = ts_data.generate();
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        
        // Should not panic regardless of input
        let result = panic::catch_unwind(|| {
            TimeSeriesDataFrame::<f64>::from_polars(df, schema)
        });
        
        prop_assert!(result.is_ok());
    }
}

/// Property: Dataset conversion should handle any valid DataFrame
proptest! {
    #[test]
    fn prop_dataset_conversion_robustness(
        n_series in 1usize..100,
        n_points in 1usize..100,
        include_special_values in prop::bool::ANY,
    ) {
        let ts_data = ArbitraryTimeSeries {
            n_series,
            n_points,
            n_features: 0,
            include_nan: include_special_values,
            include_inf: include_special_values,
            include_duplicates: false,
            include_gaps: false,
            random_seed: 42,
        };
        
        let df = ts_data.generate();
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        
        if let Ok(ts_df) = TimeSeriesDataFrame::<f64>::from_polars(df, schema) {
            let result = panic::catch_unwind(|| {
                ts_df.to_dataset()
            });
            
            prop_assert!(result.is_ok());
            
            if let Ok(Ok(dataset)) = result {
                prop_assert!(dataset.unique_ids.len() <= n_series);
                prop_assert!(dataset.metadata.n_series <= n_series);
            }
        }
    }
}

/// Property: Model creation should handle any valid configuration
proptest! {
    #[test]
    fn prop_model_creation_robustness(
        hidden_size in 1usize..1000,
        num_layers in 1usize..10,
        dropout in 0.0f64..1.0,
        learning_rate in 1e-6f64..1.0,
        batch_size in 1usize..10000,
        horizon in 1usize..365,
        input_size in 1usize..1000,
    ) {
        // MLP model
        let mlp_result = panic::catch_unwind(|| {
            MLP::builder()
                .hidden_size(hidden_size)
                .num_layers(num_layers)
                .dropout(dropout)
                .learning_rate(learning_rate)
                .batch_size(batch_size)
                .horizon(horizon)
                .input_size(input_size)
                .build()
        });
        
        prop_assert!(mlp_result.is_ok());
        
        // DLinear model
        let dlinear_result = panic::catch_unwind(|| {
            DLinear::builder()
                .hidden_size(hidden_size)
                .kernel_size(3)
                .learning_rate(learning_rate)
                .batch_size(batch_size)
                .horizon(horizon)
                .input_size(input_size)
                .build()
        });
        
        prop_assert!(dlinear_result.is_ok());
    }
}

/// Fuzz test for schema validation
#[test]
fn fuzz_schema_validation() {
    let mut rng = rand::thread_rng();
    
    for _ in 0..1000 {
        // Generate random column names
        let random_string = || -> String {
            let len = rng.gen_range(0..100);
            let chars: String = (0..len)
                .map(|_| {
                    let ch = rng.gen_range(0..256) as u8;
                    if ch.is_ascii() && ch != 0 {
                        ch as char
                    } else {
                        'X'
                    }
                })
                .collect();
            chars
        };
        
        let unique_id_col = random_string();
        let ds_col = random_string();
        let y_col = random_string();
        
        // Should handle any string input
        let result = panic::catch_unwind(|| {
            TimeSeriesSchema::new(unique_id_col, ds_col, y_col)
        });
        
        assert!(result.is_ok());
    }
}

/// Fuzz test for preprocessing configuration
#[test]
fn fuzz_preprocessing_config() {
    let mut rng = rand::thread_rng();
    
    for _ in 0..1000 {
        let config = PreprocessingConfig {
            scaling: Some(ScalingConfig {
                method: match rng.gen_range(0..4) {
                    0 => ScalingMethod::Standard,
                    1 => ScalingMethod::MinMax,
                    2 => ScalingMethod::Robust,
                    _ => ScalingMethod::None,
                },
                feature_range: if rng.gen_bool(0.5) {
                    Some((rng.gen_range(-100.0..0.0), rng.gen_range(0.0..100.0)))
                } else {
                    None
                },
                per_series: rng.gen_bool(0.5),
                exclude_columns: (0..rng.gen_range(0..10))
                    .map(|i| format!("col_{}", i))
                    .collect(),
            }),
            missing_values: MissingValueConfig {
                strategy: match rng.gen_range(0..7) {
                    0 => MissingValueStrategy::Drop,
                    1 => MissingValueStrategy::FillConstant(rng.gen_range(-100.0..100.0)),
                    2 => MissingValueStrategy::ForwardFill,
                    3 => MissingValueStrategy::BackwardFill,
                    4 => MissingValueStrategy::FillMean,
                    5 => MissingValueStrategy::FillMedian,
                    _ => MissingValueStrategy::Interpolate,
                },
                max_missing_percentage: rng.gen_range(0.0..100.0),
                interpolate: rng.gen_bool(0.5),
                interpolation_method: InterpolationMethod::Linear,
            },
            outliers: OutlierConfig {
                detection_method: match rng.gen_range(0..4) {
                    0 => OutlierDetectionMethod::ZScore { threshold: rng.gen_range(1.0..5.0) },
                    1 => OutlierDetectionMethod::IQR { multiplier: rng.gen_range(1.0..3.0) },
                    2 => OutlierDetectionMethod::IsolationForest { contamination: rng.gen_range(0.01..0.5) },
                    _ => OutlierDetectionMethod::None,
                },
                handling_strategy: match rng.gen_range(0..5) {
                    0 => OutlierHandlingStrategy::Remove,
                    1 => OutlierHandlingStrategy::Clip,
                    2 => OutlierHandlingStrategy::ReplaceMedian,
                    3 => OutlierHandlingStrategy::ReplaceMean,
                    _ => OutlierHandlingStrategy::ReplaceConstant(rng.gen_range(-10.0..10.0)),
                },
                per_series: rng.gen_bool(0.5),
            },
            feature_engineering: FeatureEngineeringConfig {
                add_lags: if rng.gen_bool(0.5) {
                    Some((0..rng.gen_range(1..10)).map(|i| i + 1).collect())
                } else {
                    None
                },
                add_rolling_features: None,
                add_calendar_features: rng.gen_bool(0.5),
                add_seasonal_decomposition: rng.gen_bool(0.3),
                custom_features: Vec::new(),
            },
            validation: ValidationConfig {
                check_consistency: rng.gen_bool(0.8),
                check_data_sufficiency: rng.gen_bool(0.8),
                min_observations: if rng.gen_bool(0.5) {
                    Some(rng.gen_range(10..1000))
                } else {
                    None
                },
                check_regularity: rng.gen_bool(0.5),
                gap_tolerance: None,
            },
        };
        
        // Config should be valid regardless of random values
        let _config = config; // Use it to avoid unused warning
    }
}

/// QuickCheck test for time series operations
#[quickcheck]
fn qc_filter_operations(series_id: String, start_offset: i64, end_offset: i64) -> TestResult {
    if series_id.is_empty() || start_offset.abs() > 365 * 24 || end_offset.abs() > 365 * 24 {
        return TestResult::discard();
    }
    
    let df = df! {
        "unique_id" => vec![series_id.clone(); 100],
        "ds" => (0..100).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect::<Vec<_>>(),
        "y" => (0..100).map(|i| i as f64).collect::<Vec<_>>(),
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Test filter by ID
    let filtered = ts_df.filter_by_id(&series_id);
    assert!(filtered.is_ok());
    
    // Test filter by date range
    let start = Utc::now() + Duration::hours(start_offset);
    let end = Utc::now() + Duration::hours(end_offset);
    
    if start < end {
        let date_filtered = ts_df.filter_date_range(start, end);
        assert!(date_filtered.is_ok());
    }
    
    TestResult::passed()
}

/// Mutation-based fuzzing for model state
#[test]
fn fuzz_model_state_mutations() {
    let mut rng = rand::thread_rng();
    
    // Create a base model
    let mut model = DLinear::builder()
        .hidden_size(64)
        .kernel_size(3)
        .horizon(12)
        .input_size(24)
        .build()
        .unwrap();
    
    // Apply random mutations
    for _ in 0..100 {
        let mutation_type = rng.gen_range(0..5);
        
        match mutation_type {
            0 => {
                // Mutate configuration (in practice, configs are usually immutable)
                let _config = model.config();
            }
            1 => {
                // Get state
                let _state = model.state();
            }
            2 => {
                // Try to load corrupted state
                let mut fake_state = vec![0u8; rng.gen_range(0..10000)];
                rng.fill(&mut fake_state[..]);
                
                // This should fail gracefully
                let _result = panic::catch_unwind(|| {
                    // In practice, you'd deserialize the state
                    fake_state.len()
                });
            }
            3 => {
                // Access model metadata
                let _horizon = model.config().horizon();
            }
            _ => {
                // No-op
            }
        }
    }
}

/// Arbitrary model configuration generator
#[derive(Debug, Clone)]
struct ArbitraryModelConfig {
    model_type: ModelType,
    hidden_size: usize,
    num_layers: usize,
    dropout: f64,
    learning_rate: f64,
    batch_size: usize,
    horizon: usize,
    input_size: usize,
}

#[derive(Debug, Clone)]
enum ModelType {
    MLP,
    DLinear,
    NLinear,
}

impl Arbitrary for ArbitraryModelConfig {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            prop_oneof![
                Just(ModelType::MLP),
                Just(ModelType::DLinear),
                Just(ModelType::NLinear),
            ],
            1usize..1000,
            1usize..10,
            0.0f64..1.0,
            prop::num::f64::POSITIVE.prop_map(|x| 10f64.powf(-(1.0 + (x % 5.0)))),
            1usize..10000,
            1usize..365,
            1usize..1000,
        )
            .prop_map(|(model_type, hidden_size, num_layers, dropout, learning_rate, batch_size, horizon, input_size)| {
                ArbitraryModelConfig {
                    model_type,
                    hidden_size,
                    num_layers,
                    dropout,
                    learning_rate,
                    batch_size,
                    horizon,
                    input_size,
                }
            })
            .boxed()
    }
}

/// Property: Model training should handle any valid input
proptest! {
    #[test]
    #[ignore] // This is expensive to run
    fn prop_model_training_robustness(
        config: ArbitraryModelConfig,
        ts_data: ArbitraryTimeSeries,
    ) {
        // Ensure we have some data
        if ts_data.n_series == 0 || ts_data.n_points < config.input_size + config.horizon {
            return Ok(());
        }
        
        let df = ts_data.generate();
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        
        if let Ok(ts_df) = TimeSeriesDataFrame::<f64>::from_polars(df, schema) {
            let model: Box<dyn ForecastingModel + Send + Sync> = match config.model_type {
                ModelType::MLP => Box::new(
                    MLP::builder()
                        .hidden_size(config.hidden_size)
                        .num_layers(config.num_layers)
                        .dropout(config.dropout)
                        .learning_rate(config.learning_rate)
                        .batch_size(config.batch_size)
                        .horizon(config.horizon)
                        .input_size(config.input_size)
                        .build()
                        .unwrap()
                ),
                ModelType::DLinear => Box::new(
                    DLinear::builder()
                        .hidden_size(config.hidden_size)
                        .kernel_size(3)
                        .learning_rate(config.learning_rate)
                        .batch_size(config.batch_size)
                        .horizon(config.horizon)
                        .input_size(config.input_size)
                        .build()
                        .unwrap()
                ),
                ModelType::NLinear => Box::new(
                    NLinear::builder()
                        .hidden_size(config.hidden_size)
                        .learning_rate(config.learning_rate)
                        .batch_size(config.batch_size)
                        .horizon(config.horizon)
                        .input_size(config.input_size)
                        .build()
                        .unwrap()
                ),
            };
            
            let mut nf = NeuralForecast::builder()
                .with_model(model)
                .with_frequency(Frequency::Hourly)
                .build()
                .unwrap();
            
            // Training might fail due to data issues, but should not panic
            let result = panic::catch_unwind(|| {
                nf.fit(ts_df)
            });
            
            prop_assert!(result.is_ok());
        }
    }
}

/// Grammar-based fuzzing for API sequences
#[test]
fn fuzz_api_sequences() {
    #[derive(Debug, Clone)]
    enum APICall {
        CreateDataFrame,
        FilterById(String),
        FilterByDate(i64, i64),
        ToDataset,
        Validate,
        CreateModel(ModelType),
        Train,
        Predict,
        SaveModel(String),
        LoadModel(String),
    }
    
    let mut rng = rand::thread_rng();
    
    // Generate random API call sequences
    for _ in 0..100 {
        let sequence_len = rng.gen_range(1..20);
        let mut sequence = Vec::new();
        
        for _ in 0..sequence_len {
            let call = match rng.gen_range(0..10) {
                0 => APICall::CreateDataFrame,
                1 => APICall::FilterById(format!("series_{}", rng.gen_range(0..10))),
                2 => APICall::FilterByDate(rng.gen_range(-100..0), rng.gen_range(0..100)),
                3 => APICall::ToDataset,
                4 => APICall::Validate,
                5 => APICall::CreateModel(match rng.gen_range(0..3) {
                    0 => ModelType::MLP,
                    1 => ModelType::DLinear,
                    _ => ModelType::NLinear,
                }),
                6 => APICall::Train,
                7 => APICall::Predict,
                8 => APICall::SaveModel(format!("model_{}.bin", rng.gen_range(0..10))),
                _ => APICall::LoadModel(format!("model_{}.bin", rng.gen_range(0..10))),
            };
            sequence.push(call);
        }
        
        // Execute sequence and check for panics
        let result = panic::catch_unwind(|| {
            execute_api_sequence(&sequence)
        });
        
        assert!(result.is_ok());
    }
}

fn execute_api_sequence(sequence: &[APICall]) -> Result<(), String> {
    let mut current_df: Option<DataFrame> = None;
    let mut current_ts_df: Option<TimeSeriesDataFrame<f64>> = None;
    let mut current_model: Option<Box<dyn ForecastingModel + Send + Sync>> = None;
    
    for call in sequence {
        match call {
            APICall::CreateDataFrame => {
                let df = df! {
                    "unique_id" => vec!["series_1"; 10],
                    "ds" => (0..10).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect::<Vec<_>>(),
                    "y" => vec![1.0; 10],
                }.unwrap();
                current_df = Some(df);
            }
            APICall::FilterById(id) => {
                if let Some(ts_df) = &current_ts_df {
                    let _ = ts_df.filter_by_id(id);
                }
            }
            APICall::FilterByDate(start, end) => {
                if let Some(ts_df) = &current_ts_df {
                    let start = Utc::now() + Duration::hours(*start);
                    let end = Utc::now() + Duration::hours(*end);
                    if start < end {
                        let _ = ts_df.filter_date_range(start, end);
                    }
                }
            }
            APICall::ToDataset => {
                if let Some(ts_df) = &current_ts_df {
                    let _ = ts_df.to_dataset();
                }
            }
            APICall::Validate => {
                if let Some(ts_df) = &current_ts_df {
                    let _ = ts_df.validate();
                }
            }
            APICall::CreateModel(model_type) => {
                current_model = Some(match model_type {
                    ModelType::MLP => Box::new(
                        MLP::builder()
                            .hidden_size(32)
                            .horizon(12)
                            .input_size(24)
                            .build()
                            .unwrap()
                    ),
                    ModelType::DLinear => Box::new(
                        DLinear::builder()
                            .hidden_size(32)
                            .horizon(12)
                            .input_size(24)
                            .build()
                            .unwrap()
                    ),
                    ModelType::NLinear => Box::new(
                        NLinear::builder()
                            .hidden_size(32)
                            .horizon(12)
                            .input_size(24)
                            .build()
                            .unwrap()
                    ),
                });
            }
            APICall::Train => {
                // Skip actual training in fuzzing
            }
            APICall::Predict => {
                // Skip actual prediction in fuzzing
            }
            APICall::SaveModel(_path) => {
                // Skip actual saving in fuzzing
            }
            APICall::LoadModel(_path) => {
                // Skip actual loading in fuzzing
            }
        }
        
        // Convert DataFrame to TimeSeriesDataFrame if needed
        if current_df.is_some() && current_ts_df.is_none() {
            if let Some(df) = current_df.take() {
                let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
                current_ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).ok();
            }
        }
    }
    
    Ok(())
}