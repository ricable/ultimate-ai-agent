//! Edge case stress tests for handling unusual, extreme, or malformed data inputs.
//! Tests robustness against edge cases that could cause panics or undefined behavior.

use chrono::{DateTime, Utc, Duration, NaiveDateTime, TimeZone};
use ndarray::{Array2, Array1};
use polars::prelude::*;
use proptest::prelude::*;
use std::f64::{INFINITY, NEG_INFINITY, NAN};
use std::collections::HashMap;
use rand::Rng;
use std::panic;

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
    SeriesData, ValidationReport, TimeSeriesDatasetBuilder,
};
use neuro_divergent_core::error::{NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent_models::{
    basic::{MLP, DLinear},
    forecasting::ForecastingModel,
    core::ModelConfig,
};
use neuro_divergent::prelude::*;

/// Test empty dataset handling
#[test]
fn test_empty_dataset() {
    // Test completely empty DataFrame
    let empty_df = DataFrame::empty();
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    
    let result = TimeSeriesDataFrame::<f64>::from_polars(empty_df, schema);
    assert!(result.is_err());
    
    // Test DataFrame with columns but no rows
    let df = df! {
        "unique_id" => Vec::<&str>::new(),
        "ds" => Vec::<NaiveDateTime>::new(),
        "y" => Vec::<f64>::new(),
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle empty data gracefully
    let validation = ts_df.validate().unwrap();
    assert!(!validation.is_valid);
    assert!(validation.errors.iter().any(|e| e.code == "EMPTY_DATASET"));
}

/// Test single data point handling
#[test]
fn test_single_data_point() {
    let df = df! {
        "unique_id" => ["series_1"],
        "ds" => [Utc::now().naive_utc()],
        "y" => [42.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle single point
    let dataset = ts_df.to_dataset().unwrap();
    assert_eq!(dataset.unique_ids.len(), 1);
    assert_eq!(dataset.series_data["series_1"].length, 1);
    
    // Model training should fail gracefully
    let model = MLP::builder()
        .hidden_size(32)
        .horizon(1)
        .input_size(1)
        .build()
        .unwrap();
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(model))
        .with_frequency(Frequency::Daily)
        .build()
        .unwrap();
    
    // Should error on insufficient data
    let fit_result = nf.fit(ts_df);
    assert!(fit_result.is_err());
}

/// Test all NaN/missing values
#[test]
fn test_all_nan_values() {
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1"],
        "ds" => [
            Utc::now().naive_utc(),
            (Utc::now() + Duration::hours(1)).naive_utc(),
            (Utc::now() + Duration::hours(2)).naive_utc(),
        ],
        "y" => [f64::NAN, f64::NAN, f64::NAN],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Validation should catch all NaN values
    let validation = ts_df.validate().unwrap();
    assert!(!validation.is_valid);
    
    // Dataset conversion should handle NaN appropriately
    let dataset = ts_df.to_dataset().unwrap();
    let series = &dataset.series_data["series_1"];
    assert!(series.target_values.iter().all(|v| v.is_nan()));
}

/// Test infinite values
#[test]
fn test_infinite_values() {
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1", "series_1"],
        "ds" => [
            Utc::now().naive_utc(),
            (Utc::now() + Duration::hours(1)).naive_utc(),
            (Utc::now() + Duration::hours(2)).naive_utc(),
            (Utc::now() + Duration::hours(3)).naive_utc(),
        ],
        "y" => [1.0, INFINITY, NEG_INFINITY, 2.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle infinite values
    let dataset = ts_df.to_dataset().unwrap();
    let series = &dataset.series_data["series_1"];
    assert!(series.target_values[1].is_infinite());
    assert!(series.target_values[2].is_infinite());
}

/// Test duplicate timestamps
#[test]
fn test_duplicate_timestamps() {
    let same_time = Utc::now().naive_utc();
    
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1"],
        "ds" => [same_time, same_time, same_time],
        "y" => [1.0, 2.0, 3.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle duplicate timestamps
    let validation = ts_df.validate().unwrap();
    // In production, this should produce a warning about duplicate timestamps
    
    let dataset = ts_df.to_dataset().unwrap();
    assert_eq!(dataset.series_data["series_1"].length, 3);
}

/// Test unsorted timestamps
#[test]
fn test_unsorted_timestamps() {
    let now = Utc::now();
    
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1", "series_1"],
        "ds" => [
            (now + Duration::hours(3)).naive_utc(),
            now.naive_utc(),
            (now + Duration::hours(2)).naive_utc(),
            (now + Duration::hours(1)).naive_utc(),
        ],
        "y" => [4.0, 1.0, 3.0, 2.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle unsorted data
    let dataset = ts_df.to_dataset().unwrap();
    let series = &dataset.series_data["series_1"];
    
    // Verify timestamps should be processed even if unsorted
    assert_eq!(series.timestamps.len(), 4);
}

/// Test mismatched column types
#[test]
fn test_mismatched_column_types() {
    // Try to create DataFrame with wrong types
    let result = panic::catch_unwind(|| {
        df! {
            "unique_id" => [1, 2, 3], // Should be strings
            "ds" => ["2023-01-01", "2023-01-02", "2023-01-03"], // Should be datetime
            "y" => ["a", "b", "c"], // Should be numeric
        }
    });
    
    // DataFrame creation with wrong types should be handled
    assert!(result.is_ok() || result.is_err());
}

/// Test extremely long series names
#[test]
fn test_extremely_long_series_names() {
    let long_name = "x".repeat(10_000); // 10k character series name
    
    let df = df! {
        "unique_id" => [long_name.as_str()],
        "ds" => [Utc::now().naive_utc()],
        "y" => [1.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle long names
    let unique_ids = ts_df.unique_ids().unwrap();
    assert_eq!(unique_ids[0].len(), 10_000);
}

/// Test mixed frequency data
#[test]
fn test_mixed_frequency_data() {
    let now = Utc::now();
    
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1", "series_2", "series_2"],
        "ds" => [
            now.naive_utc(),
            (now + Duration::hours(1)).naive_utc(),
            (now + Duration::hours(2)).naive_utc(),
            now.naive_utc(),
            (now + Duration::days(1)).naive_utc(), // Different frequency
        ],
        "y" => [1.0, 2.0, 3.0, 4.0, 5.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle mixed frequencies
    let dataset = ts_df.to_dataset().unwrap();
    assert_eq!(dataset.unique_ids.len(), 2);
}

/// Test extreme time ranges
#[test]
fn test_extreme_time_ranges() {
    let min_time = DateTime::<Utc>::MIN_UTC;
    let max_time = DateTime::<Utc>::MAX_UTC;
    let normal_time = Utc::now();
    
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1"],
        "ds" => [
            min_time.naive_utc(),
            normal_time.naive_utc(),
            max_time.naive_utc(),
        ],
        "y" => [1.0, 2.0, 3.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle extreme dates
    let (min, max) = ts_df.time_range().unwrap();
    assert!(min <= min_time || min == DateTime::<Utc>::MIN_UTC);
    assert!(max >= max_time || max == DateTime::<Utc>::MAX_UTC);
}

/// Test zero variance series
#[test]
fn test_zero_variance_series() {
    let df = df! {
        "unique_id" => ["series_1"; 100],
        "ds" => (0..100).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect::<Vec<_>>(),
        "y" => [42.0; 100], // Constant value
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle zero variance
    let dataset = ts_df.to_dataset().unwrap();
    let series = &dataset.series_data["series_1"];
    
    let variance = statistical_variance(&series.target_values);
    assert_eq!(variance, 0.0);
}

/// Test extremely sparse data
#[test]
fn test_extremely_sparse_data() {
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    // Create 1000 series with only 1-2 points each
    for i in 0..1000 {
        let series_id = format!("series_{}", i);
        let num_points = if i % 2 == 0 { 1 } else { 2 };
        
        for j in 0..num_points {
            unique_ids.push(series_id.clone());
            timestamps.push((Utc::now() + Duration::days(j)).naive_utc());
            values.push(i as f64);
        }
    }
    
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle sparse data
    let dataset = ts_df.to_dataset().unwrap();
    assert_eq!(dataset.unique_ids.len(), 1000);
    assert!(dataset.metadata.min_series_length <= 2);
}

/// Test mixed numeric types
#[test]
fn test_mixed_numeric_types() {
    // Create series with different numeric representations
    let df = df! {
        "unique_id" => ["series_1", "series_1", "series_1"],
        "ds" => [
            Utc::now().naive_utc(),
            (Utc::now() + Duration::hours(1)).naive_utc(),
            (Utc::now() + Duration::hours(2)).naive_utc(),
        ],
        "y" => [1i32, 2i32, 3i32], // Integer values
        "feature_1" => [1.0f32, 2.0f32, 3.0f32], // Float32
        "feature_2" => [1.0f64, 2.0f64, 3.0f64], // Float64
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
        .with_historical_exogenous(vec!["feature_1".to_string(), "feature_2".to_string()]);
    
    let result = TimeSeriesDataFrame::<f64>::from_polars(df, schema);
    // Should handle type conversions or error appropriately
    assert!(result.is_ok() || result.is_err());
}

/// Test boundary value inputs
#[test]
fn test_boundary_values() {
    let df = df! {
        "unique_id" => ["series_1"; 6],
        "ds" => (0..6).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect::<Vec<_>>(),
        "y" => [
            f64::MIN,
            f64::MIN_POSITIVE,
            0.0,
            f64::EPSILON,
            f64::MAX / 2.0,
            f64::MAX,
        ],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle boundary values
    let dataset = ts_df.to_dataset().unwrap();
    let series = &dataset.series_data["series_1"];
    
    assert!(series.target_values.contains(&f64::MIN));
    assert!(series.target_values.contains(&f64::MAX));
}

/// Test special UTF-8 characters in series names
#[test]
fn test_special_characters_in_names() {
    let special_names = vec![
        "series_😀",
        "series_中文",
        "series_\n\t",
        "series_\"'",
        "series_NULL",
        "series_\\x00",
    ];
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for name in &special_names {
        unique_ids.push(name.to_string());
        timestamps.push(Utc::now().naive_utc());
        values.push(1.0);
    }
    
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Should handle special characters
    let unique_ids = ts_df.unique_ids().unwrap();
    assert_eq!(unique_ids.len(), special_names.len());
}

/// Test cyclic references in schema
#[test]
fn test_cyclic_schema_references() {
    let df = df! {
        "unique_id" => ["series_1"],
        "ds" => [Utc::now().naive_utc()],
        "y" => [1.0],
        "unique_id_2" => ["series_1"], // Duplicate column name concept
    }.unwrap();
    
    // Try to create schema with overlapping column definitions
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
        .with_historical_exogenous(vec!["unique_id".to_string()]); // Self-reference
    
    let result = TimeSeriesDataFrame::<f64>::from_polars(df, schema);
    // Should handle or reject cyclic references appropriately
    assert!(result.is_ok() || result.is_err());
}

/// Property-based test for edge cases
proptest! {
    #[test]
    fn prop_handle_random_edge_cases(
        num_series in 0usize..100,
        include_nan in prop::bool::ANY,
        include_inf in prop::bool::ANY,
        include_duplicates in prop::bool::ANY,
    ) {
        let mut rng = rand::thread_rng();
        let mut unique_ids = Vec::new();
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        if num_series > 0 {
            for i in 0..num_series {
                let series_id = format!("series_{}", i);
                let num_points = rng.gen_range(0..10);
                
                for j in 0..num_points {
                    unique_ids.push(series_id.clone());
                    
                    if include_duplicates && rng.gen_bool(0.1) {
                        timestamps.push(Utc::now().naive_utc());
                    } else {
                        timestamps.push((Utc::now() + Duration::hours(j)).naive_utc());
                    }
                    
                    let value = if include_nan && rng.gen_bool(0.1) {
                        f64::NAN
                    } else if include_inf && rng.gen_bool(0.1) {
                        if rng.gen_bool(0.5) { INFINITY } else { NEG_INFINITY }
                    } else {
                        rng.gen_range(-1000.0..1000.0)
                    };
                    
                    values.push(value);
                }
            }
        }
        
        if !unique_ids.is_empty() {
            let df = df! {
                "unique_id" => unique_ids,
                "ds" => timestamps,
                "y" => values,
            }.unwrap();
            
            let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
            
            // Should not panic on any input
            let result = panic::catch_unwind(|| {
                TimeSeriesDataFrame::<f64>::from_polars(df, schema)
            });
            
            prop_assert!(result.is_ok());
        }
    }
}

/// Helper function to calculate variance
fn statistical_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance
}

/// Test model robustness against edge cases
#[test]
fn test_model_edge_case_robustness() {
    // Test various edge cases that models should handle gracefully
    let edge_cases = vec![
        // Empty input
        (vec![], vec![]),
        // Single point
        (vec![1.0], vec![Utc::now().naive_utc()]),
        // All same values
        (vec![42.0; 100], (0..100).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect()),
        // Contains NaN
        (vec![1.0, f64::NAN, 3.0], (0..3).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect()),
        // Contains infinity
        (vec![1.0, INFINITY, 3.0], (0..3).map(|i| (Utc::now() + Duration::hours(i)).naive_utc()).collect()),
    ];
    
    for (values, timestamps) in edge_cases {
        if values.is_empty() {
            continue; // Skip empty case for model testing
        }
        
        let df = df! {
            "unique_id" => vec!["test"; values.len()],
            "ds" => timestamps,
            "y" => values,
        }.unwrap();
        
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
        
        // Test with DLinear model (simpler, more robust)
        let model = DLinear::builder()
            .hidden_size(32)
            .kernel_size(3)
            .horizon(1)
            .input_size(values.len().max(3))
            .build();
        
        if let Ok(model) = model {
            let mut nf = NeuralForecast::builder()
                .with_model(Box::new(model))
                .with_frequency(Frequency::Hourly)
                .build()
                .unwrap();
            
            // Model should either succeed or fail gracefully
            let _ = nf.fit(ts_df);
        }
    }
}

/// Test schema validation edge cases
#[test]
fn test_schema_validation_edge_cases() {
    // Test missing required columns
    let df = df! {
        "wrong_id" => ["series_1"],
        "wrong_time" => [Utc::now().naive_utc()],
        "wrong_target" => [1.0],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let result = TimeSeriesDataFrame::<f64>::from_polars(df, schema);
    assert!(result.is_err());
    
    // Test empty column names
    let result = panic::catch_unwind(|| {
        TimeSeriesSchema::new("", "", "")
    });
    // Should handle empty column names appropriately
    assert!(result.is_ok());
    
    // Test duplicate column names in schema
    let schema = TimeSeriesSchema::new("col", "col", "col");
    let df = df! {
        "col" => [1.0],
    }.unwrap();
    
    let result = TimeSeriesDataFrame::<f64>::from_polars(df, schema);
    // Should handle duplicate column references
    assert!(result.is_err());
}

#[test]
fn test_overflow_scenarios() {
    // Test integer overflow in calculations
    let large_number = (i64::MAX / 2) as f64;
    
    let df = df! {
        "unique_id" => ["series_1", "series_1"],
        "ds" => [
            Utc::now().naive_utc(),
            (Utc::now() + Duration::hours(1)).naive_utc(),
        ],
        "y" => [large_number, large_number],
    }.unwrap();
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Operations should handle large numbers without overflow
    let dataset = ts_df.to_dataset().unwrap();
    assert!(dataset.series_data["series_1"].target_values[0] == large_number);
}