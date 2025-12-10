//! Comprehensive test suite for neuro-divergent library
//!
//! This module organizes all unit tests for the neuro-divergent neural forecasting library.
//! Tests are organized by component to ensure comprehensive coverage of all functionality.

#![cfg(test)]

// Core component tests
pub mod unit {
    // Core functionality tests
    pub mod core_tests;
    
    // Model implementation tests
    pub mod models {
        pub mod basic_tests;
        pub mod recurrent_tests;
        pub mod advanced_tests;
        pub mod transformer_tests;
        pub mod specialized_tests;
    }
    
    // Training system tests
    pub mod training_tests;
    
    // Data pipeline tests
    pub mod data_tests;
    
    // Registry and factory tests
    pub mod registry_tests;
}

// Integration tests (if needed in the future)
#[cfg(feature = "integration-tests")]
pub mod integration {
    // Integration test modules would go here
}

// Benchmark tests (if needed in the future)
#[cfg(feature = "bench-tests")]
pub mod benchmarks {
    // Benchmark test modules would go here
}

// Common test utilities
pub mod test_utils {
    use chrono::{DateTime, TimeZone, Utc};
    use ndarray::{Array1, Array2};
    use num_traits::Float;
    use polars::prelude::*;
    use std::collections::HashMap;
    
    /// Generate synthetic time series data for testing
    pub fn generate_synthetic_series<T: Float>(
        length: usize,
        pattern: SeriesPattern,
    ) -> Vec<T> {
        match pattern {
            SeriesPattern::Linear { slope, intercept } => {
                (0..length)
                    .map(|i| T::from(i).unwrap() * slope + intercept)
                    .collect()
            }
            SeriesPattern::Sine { frequency, amplitude, phase } => {
                (0..length)
                    .map(|i| {
                        let x = T::from(i).unwrap() * frequency;
                        (x + phase).sin() * amplitude
                    })
                    .collect()
            }
            SeriesPattern::Random { mean, std_dev } => {
                use rand::{thread_rng, Rng};
                use rand_distr::{Distribution, Normal};
                
                let mut rng = thread_rng();
                let normal = Normal::new(mean.to_f64().unwrap(), std_dev.to_f64().unwrap()).unwrap();
                
                (0..length)
                    .map(|_| T::from(normal.sample(&mut rng)).unwrap())
                    .collect()
            }
            SeriesPattern::Seasonal { period, amplitude } => {
                (0..length)
                    .map(|i| {
                        let seasonal = T::from(i % period).unwrap() / T::from(period).unwrap();
                        seasonal * amplitude
                    })
                    .collect()
            }
            SeriesPattern::Trend { start, end } => {
                let step = (end - start) / T::from(length - 1).unwrap();
                (0..length)
                    .map(|i| start + step * T::from(i).unwrap())
                    .collect()
            }
        }
    }
    
    /// Pattern types for synthetic data generation
    #[derive(Debug, Clone)]
    pub enum SeriesPattern<T: Float> {
        Linear { slope: T, intercept: T },
        Sine { frequency: T, amplitude: T, phase: T },
        Random { mean: T, std_dev: T },
        Seasonal { period: usize, amplitude: T },
        Trend { start: T, end: T },
    }
    
    /// Create a test DataFrame with time series data
    pub fn create_test_dataframe<T: Float>(
        n_series: usize,
        series_length: usize,
        pattern: SeriesPattern<T>,
    ) -> DataFrame {
        let mut unique_ids = Vec::new();
        let mut timestamps = Vec::new();
        let mut values = Vec::new();
        
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        
        for series_idx in 0..n_series {
            let series_id = format!("series_{}", series_idx);
            let series_values = generate_synthetic_series(series_length, pattern.clone());
            
            for (i, value) in series_values.iter().enumerate() {
                unique_ids.push(series_id.clone());
                timestamps.push(base_date + chrono::Duration::days(i as i64));
                values.push(value.to_f64().unwrap());
            }
        }
        
        df! {
            "unique_id" => unique_ids,
            "ds" => timestamps.iter().map(|dt| dt.naive_utc()).collect::<Vec<_>>(),
            "y" => values,
        }.unwrap()
    }
    
    /// Create test exogenous features
    pub fn create_test_exogenous<T: Float>(
        n_features: usize,
        length: usize,
    ) -> Array2<T> {
        Array2::from_shape_fn((length, n_features), |(i, j)| {
            T::from(i * n_features + j).unwrap()
        })
    }
    
    /// Assert arrays are approximately equal
    pub fn assert_array_approx_eq<T: Float>(
        actual: &[T],
        expected: &[T],
        tolerance: T,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Arrays have different lengths: {} vs {}",
            actual.len(),
            expected.len()
        );
        
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (*a - *e).abs();
            assert!(
                diff <= tolerance,
                "Arrays differ at index {}: {} vs {} (diff: {})",
                i,
                a,
                e,
                diff
            );
        }
    }
    
    /// Create a mock model configuration for testing
    pub fn create_mock_config<T: Float>() -> HashMap<String, T> {
        let mut config = HashMap::new();
        config.insert("learning_rate".to_string(), T::from(0.001).unwrap());
        config.insert("batch_size".to_string(), T::from(32).unwrap());
        config.insert("epochs".to_string(), T::from(100).unwrap());
        config
    }
    
    /// Generate timestamps for testing
    pub fn generate_timestamps(start: DateTime<Utc>, count: usize, freq: &str) -> Vec<DateTime<Utc>> {
        let duration = match freq {
            "D" => chrono::Duration::days(1),
            "H" => chrono::Duration::hours(1),
            "M" => chrono::Duration::minutes(1),
            "S" => chrono::Duration::seconds(1),
            _ => chrono::Duration::days(1),
        };
        
        (0..count)
            .map(|i| start + duration * i as i32)
            .collect()
    }
    
    /// Assert that a result contains an expected error type
    #[macro_export]
    macro_rules! assert_error_type {
        ($result:expr, $error_pattern:pat) => {
            match $result {
                Err($error_pattern) => (),
                Err(e) => panic!("Expected error pattern {}, got {:?}", stringify!($error_pattern), e),
                Ok(_) => panic!("Expected error, got Ok"),
            }
        };
    }
    
    /// Assert that a float value is within a percentage of expected
    #[macro_export]
    macro_rules! assert_relative_eq_pct {
        ($actual:expr, $expected:expr, $pct:expr) => {
            let actual = $actual;
            let expected = $expected;
            let pct = $pct;
            let tolerance = expected.abs() * pct / 100.0;
            assert!(
                (actual - expected).abs() <= tolerance,
                "Values not within {}%: {} vs {} (diff: {})",
                pct,
                actual,
                expected,
                (actual - expected).abs()
            );
        };
    }
}

// Re-export common test utilities
pub use test_utils::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_synthetic_data_generation() {
        // Test linear pattern
        let linear_data: Vec<f64> = generate_synthetic_series(
            10,
            SeriesPattern::Linear { slope: 2.0, intercept: 1.0 },
        );
        assert_eq!(linear_data.len(), 10);
        assert_eq!(linear_data[0], 1.0);
        assert_eq!(linear_data[1], 3.0);
        
        // Test sine pattern
        let sine_data: Vec<f64> = generate_synthetic_series(
            100,
            SeriesPattern::Sine { 
                frequency: 0.1, 
                amplitude: 1.0, 
                phase: 0.0 
            },
        );
        assert_eq!(sine_data.len(), 100);
        
        // Test seasonal pattern
        let seasonal_data: Vec<f64> = generate_synthetic_series(
            20,
            SeriesPattern::Seasonal { period: 7, amplitude: 10.0 },
        );
        assert_eq!(seasonal_data.len(), 20);
    }
    
    #[test]
    fn test_dataframe_creation() {
        let df = create_test_dataframe(
            2,
            10,
            SeriesPattern::Linear { slope: 1.0, intercept: 0.0 },
        );
        
        assert_eq!(df.height(), 20); // 2 series * 10 points
        assert_eq!(df.width(), 3); // unique_id, ds, y
        
        // Check column names
        let columns: Vec<&str> = df.get_column_names();
        assert!(columns.contains(&"unique_id"));
        assert!(columns.contains(&"ds"));
        assert!(columns.contains(&"y"));
    }
    
    #[test]
    fn test_array_comparison() {
        let arr1 = vec![1.0, 2.0, 3.0];
        let arr2 = vec![1.01, 2.02, 3.03];
        
        assert_array_approx_eq(&arr1, &arr2, 0.05);
        
        #[should_panic]
        fn test_array_comparison_fail() {
            let arr1 = vec![1.0, 2.0, 3.0];
            let arr2 = vec![1.0, 2.0, 4.0];
            assert_array_approx_eq(&arr1, &arr2, 0.01);
        }
    }
}