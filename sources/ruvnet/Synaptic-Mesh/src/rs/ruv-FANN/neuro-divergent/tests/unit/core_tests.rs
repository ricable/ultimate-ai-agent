//! Comprehensive unit tests for neuro-divergent core functionality
//!
//! This module tests all core functionality including traits, data structures,
//! error handling, and integration components.

use neuro_divergent::prelude::*;
use neuro_divergent::{AccuracyMetrics, NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};
use chrono::{DateTime, TimeZone, Utc};
use num_traits::Float;
use polars::prelude::*;
use proptest::prelude::*;
use std::collections::HashMap;
use approx::assert_relative_eq;
use serde::{Deserialize, Serialize};

// ============================================================================
// Data Structure Tests
// ============================================================================

#[cfg(test)]
mod data_tests {
    use super::*;

    #[test]
    fn test_time_series_schema_creation() {
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
            .with_static_features(vec!["category".to_string()])
            .with_exogenous_features(vec!["temperature".to_string(), "humidity".to_string()]);

        assert_eq!(schema.unique_id_col, "unique_id");
        assert_eq!(schema.ds_col, "ds");
        assert_eq!(schema.y_col, "y");
        assert_eq!(schema.static_features, vec!["category"]);
        assert_eq!(schema.exogenous_features, vec!["temperature", "humidity"]);

        let all_cols = schema.all_columns();
        assert_eq!(all_cols.len(), 5);
        assert!(all_cols.contains(&&"unique_id".to_string()));
        assert!(all_cols.contains(&&"temperature".to_string()));
    }

    #[test]
    fn test_time_series_schema_validation() {
        let schema = TimeSeriesSchema::default();
        
        // Create valid DataFrame
        let valid_df = df! {
            "unique_id" => ["A", "B", "C"],
            "ds" => [1, 2, 3],
            "y" => [10.0, 20.0, 30.0],
        }.unwrap();
        
        assert!(schema.validate_dataframe(&valid_df).is_ok());
        
        // Create invalid DataFrame (missing required column)
        let invalid_df = df! {
            "unique_id" => ["A", "B", "C"],
            "ds" => [1, 2, 3],
            // Missing "y" column
        }.unwrap();
        
        assert!(schema.validate_dataframe(&invalid_df).is_err());
    }

    #[test]
    fn test_time_series_dataframe_creation() {
        let data = df! {
            "unique_id" => ["A", "A", "B", "B"],
            "ds" => [1, 2, 1, 2],
            "y" => [10.0, 11.0, 20.0, 21.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema, None);
        
        assert!(ts_df.is_ok());
        let ts_df = ts_df.unwrap();
        assert_eq!(ts_df.shape(), (4, 3));
        assert_eq!(ts_df.n_series(), 2);
    }

    #[test]
    fn test_time_series_dataframe_filtering() {
        let data = df! {
            "unique_id" => ["A", "A", "B", "B"],
            "ds" => [1, 2, 1, 2],
            "y" => [10.0, 11.0, 20.0, 21.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema, None).unwrap();
        
        // Filter by ID
        let filtered = ts_df.filter_by_id("A");
        assert!(filtered.is_ok());
        let filtered = filtered.unwrap();
        assert_eq!(filtered.shape().0, 2);
    }

    #[test]
    fn test_unique_ids_extraction() {
        let data = df! {
            "unique_id" => ["series_1", "series_1", "series_2", "series_3"],
            "ds" => [1, 2, 1, 1],
            "y" => [1.0, 2.0, 3.0, 4.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema, None).unwrap();
        let unique_ids = ts_df.unique_ids().unwrap();
        
        assert_eq!(unique_ids.len(), 3);
        assert!(unique_ids.contains(&"series_1".to_string()));
        assert!(unique_ids.contains(&"series_2".to_string()));
        assert!(unique_ids.contains(&"series_3".to_string()));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let config_err = NeuroDivergentError::config("Invalid configuration");
        assert!(config_err.is_config_error());
        assert_eq!(config_err.category(), "Configuration");

        let data_err = NeuroDivergentError::data("Invalid data format");
        assert!(data_err.is_data_error());
        assert_eq!(data_err.category(), "Data");

        let training_err = NeuroDivergentError::training("Training failed");
        assert!(training_err.is_training_error());
        assert_eq!(training_err.category(), "Training");

        let prediction_err = NeuroDivergentError::prediction("Prediction failed");
        assert!(prediction_err.is_prediction_error());
        assert_eq!(prediction_err.category(), "Prediction");
    }

    #[test]
    fn test_error_context() {
        let base_error: NeuroDivergentResult<()> = Err(NeuroDivergentError::config("base error"));
        let with_context = base_error.context("additional context");
        
        assert!(with_context.is_err());
        let error_msg = with_context.unwrap_err().to_string();
        assert!(error_msg.contains("additional context"));
        assert!(error_msg.contains("base error"));
    }

    #[test]
    fn test_multiple_errors() {
        let errors = vec![
            NeuroDivergentError::config("Config error"),
            NeuroDivergentError::data("Data error"),
            NeuroDivergentError::training("Training error"),
        ];
        
        let combined = NeuroDivergentError::multiple(errors);
        assert_eq!(combined.category(), "Multiple");
        
        let error_msg = combined.to_string();
        assert!(error_msg.contains("Multiple errors occurred"));
    }

    #[test]
    fn test_error_conversion_from_io() {
        // Test IO error conversion
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let converted: NeuroDivergentError = io_error.into();
        
        match converted {
            NeuroDivergentError::IoError(_) => {}, // Expected
            _ => panic!("Expected IoError variant"),
        }
    }
}

// ============================================================================
// Accuracy Metrics Tests
// ============================================================================

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[test]
    fn test_accuracy_metrics_creation() {
        let metrics = AccuracyMetrics::new(1.5f64, 2.25, 0.15, 0.12)
            .with_mase(0.85)
            .with_custom_metric("r2".to_string(), 0.95);

        assert_relative_eq!(metrics.mae, 1.5);
        assert_relative_eq!(metrics.mse, 2.25);
        assert_relative_eq!(metrics.rmse, 1.5); // sqrt(2.25) = 1.5
        assert_relative_eq!(metrics.mape, 0.15);
        assert_relative_eq!(metrics.smape, 0.12);
        assert_eq!(metrics.mase, Some(0.85));
        assert_eq!(metrics.get_metric("r2"), Some(0.95));
    }

    #[test]
    fn test_metrics_getter() {
        let metrics = AccuracyMetrics::new(1.0f32, 4.0, 0.1, 0.08);

        assert_eq!(metrics.get_metric("mae"), Some(1.0));
        assert_eq!(metrics.get_metric("mse"), Some(4.0));
        assert_eq!(metrics.get_metric("rmse"), Some(2.0)); // sqrt(4.0) = 2.0
        assert_eq!(metrics.get_metric("mape"), Some(0.1));
        assert_eq!(metrics.get_metric("smape"), Some(0.08));
        assert_eq!(metrics.get_metric("nonexistent"), None);
    }

    #[test]
    fn test_metric_names() {
        let metrics = AccuracyMetrics::new(1.0f64, 1.0, 0.1, 0.1)
            .with_mase(0.9)
            .with_custom_metric("custom1".to_string(), 0.5)
            .with_custom_metric("custom2".to_string(), 0.7);

        let names = metrics.metric_names();
        assert!(names.contains(&"mae".to_string()));
        assert!(names.contains(&"mse".to_string()));
        assert!(names.contains(&"rmse".to_string()));
        assert!(names.contains(&"mape".to_string()));
        assert!(names.contains(&"smape".to_string()));
        assert!(names.contains(&"mase".to_string())); // Because we set it
        assert!(names.contains(&"custom1".to_string()));
        assert!(names.contains(&"custom2".to_string()));
    }
}

// ============================================================================
// Forecast DataFrame Tests  
// ============================================================================

#[cfg(test)]
mod forecast_tests {
    use super::*;

    #[test]
    fn test_forecast_dataframe_creation() {
        let data = df! {
            "unique_id" => ["A", "A", "B", "B"],
            "ds" => [1, 2, 1, 2],
            "LSTM" => [10.5, 11.2, 20.1, 21.8],
        }.unwrap();

        let schema = TimeSeriesSchema::default();
        let models = vec!["LSTM".to_string()];
        
        let forecast_df = ForecastDataFrame::<f64>::new(
            data,
            models.clone(),
            2,
            None,
            schema,
        );

        assert_eq!(forecast_df.models, models);
        assert_eq!(forecast_df.forecast_horizon, 2);
        assert_eq!(forecast_df.shape(), (4, 3));
    }

    #[test]
    fn test_model_forecast_extraction() {
        let data = df! {
            "unique_id" => ["A", "A"],
            "ds" => [1, 2],
            "LSTM" => [10.5, 11.2],
            "GRU" => [10.3, 11.0],
        }.unwrap();

        let schema = TimeSeriesSchema::default();
        let models = vec!["LSTM".to_string(), "GRU".to_string()];
        
        let forecast_df = ForecastDataFrame::<f64>::new(
            data,
            models,
            2,
            None,
            schema,
        );

        let lstm_forecasts = forecast_df.get_model_forecasts("LSTM");
        assert!(lstm_forecasts.is_ok());
        
        let lstm_df = lstm_forecasts.unwrap();
        let columns = lstm_df.get_column_names();
        assert!(columns.contains(&"LSTM"));
        assert!(columns.contains(&"unique_id"));
        assert!(columns.contains(&"ds"));
    }

    #[test]
    fn test_point_forecasts_extraction() {
        let data = df! {
            "unique_id" => ["A", "A"],
            "ds" => [1, 2],
            "LSTM" => [10.5, 11.2],
            "LSTM_q_0.1" => [9.0, 9.5],
            "LSTM_q_0.9" => [12.0, 13.0],
        }.unwrap();

        let schema = TimeSeriesSchema::default();
        let forecast_df = ForecastDataFrame::<f64>::new(
            data,
            vec!["LSTM".to_string()],
            2,
            Some(vec![0.1, 0.9]),
            schema,
        );

        let point_forecasts = forecast_df.to_point_forecasts();
        let columns = point_forecasts.get_column_names();
        
        // Should include LSTM but not quantile columns
        assert!(columns.contains(&"LSTM"));
        assert!(!columns.contains(&"LSTM_q_0.1"));
        assert!(!columns.contains(&"LSTM_q_0.9"));
    }
}

// ============================================================================
// Cross-Validation DataFrame Tests
// ============================================================================

#[cfg(test)]
mod cross_validation_tests {
    use super::*;

    #[test]
    fn test_cross_validation_dataframe_creation() {
        let data = df! {
            "unique_id" => ["A", "A", "A", "A"],
            "ds" => [1, 2, 3, 4],
            "cutoff" => [1000000000i64, 1000000000, 1000000001, 1000000001],
            "LSTM" => [10.5, 11.2, 12.1, 13.0],
            "mae" => [0.5, 0.6, 0.4, 0.3],
            "mse" => [0.25, 0.36, 0.16, 0.09],
            "mape" => [0.05, 0.06, 0.04, 0.03],
        }.unwrap();

        let cutoffs = vec![
            DateTime::from_timestamp(1000000000, 0).unwrap(),
            DateTime::from_timestamp(1000000001, 0).unwrap(),
        ];
        let models = vec!["LSTM".to_string()];
        let schema = TimeSeriesSchema::default();
        
        let cv_df = CrossValidationDataFrame::<f64>::new(
            data,
            cutoffs.clone(),
            models.clone(),
            schema,
        );

        assert_eq!(cv_df.cutoffs, cutoffs);
        assert_eq!(cv_df.models, models);
        assert_eq!(cv_df.shape(), (4, 7));
    }

    #[test]
    fn test_cutoff_filtering() {
        let cutoff_ts = 1000000000i64;
        let data = df! {
            "unique_id" => ["A", "A", "A", "A"],
            "ds" => [1, 2, 3, 4],
            "cutoff" => [cutoff_ts, cutoff_ts, 1000000001, 1000000001],
            "LSTM" => [10.5, 11.2, 12.1, 13.0],
        }.unwrap();

        let cutoffs = vec![DateTime::from_timestamp(cutoff_ts, 0).unwrap()];
        let cv_df = CrossValidationDataFrame::<f64>::new(
            data,
            cutoffs.clone(),
            vec!["LSTM".to_string()],
            TimeSeriesSchema::default(),
        );

        let cutoff_results = cv_df.get_cutoff_results(cutoffs[0]);
        assert!(cutoff_results.is_ok());
        
        let filtered_df = cutoff_results.unwrap();
        assert_eq!(filtered_df.shape().0, 2); // Should have 2 rows for the cutoff
    }

    #[test]
    fn test_summary_stats() {
        let data = df! {
            "model" => ["LSTM", "LSTM", "GRU", "GRU"],
            "mae" => [0.5, 0.6, 0.4, 0.5],
            "mse" => [0.25, 0.36, 0.16, 0.25],
            "mape" => [0.05, 0.06, 0.04, 0.05],
        }.unwrap();

        let cv_df = CrossValidationDataFrame::<f64>::new(
            data,
            vec![DateTime::from_timestamp(1000000000, 0).unwrap()],
            vec!["LSTM".to_string(), "GRU".to_string()],
            TimeSeriesSchema::default(),
        );

        let summary = cv_df.summary_stats();
        assert!(summary.is_ok());
        
        let summary_df = summary.unwrap();
        let columns = summary_df.get_column_names();
        assert!(columns.contains(&"model"));
        assert!(columns.contains(&"mean_mae"));
        assert!(columns.contains(&"std_mae"));
        assert!(columns.contains(&"mean_mse"));
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_accuracy_metrics_invariants(
            mae in 0.0f64..1000.0,
            mse in 0.0f64..1000.0,
            mape in 0.0f64..1.0,
            smape in 0.0f64..2.0
        ) {
            let metrics = AccuracyMetrics::new(mae, mse, mape, smape);
            
            // RMSE should be square root of MSE
            let expected_rmse = mse.sqrt();
            assert_relative_eq!(metrics.rmse, expected_rmse, epsilon = 1e-10);
            
            // All metrics should be non-negative
            assert!(metrics.mae >= 0.0);
            assert!(metrics.mse >= 0.0);
            assert!(metrics.rmse >= 0.0);
            assert!(metrics.mape >= 0.0);
            assert!(metrics.smape >= 0.0);
            
            // Metric getters should return correct values
            assert_eq!(metrics.get_metric("mae"), Some(mae));
            assert_eq!(metrics.get_metric("mse"), Some(mse));
            assert_eq!(metrics.get_metric("mape"), Some(mape));
            assert_eq!(metrics.get_metric("smape"), Some(smape));
        }

        #[test]
        fn test_time_series_schema_column_consistency(
            unique_id in "[a-zA-Z_][a-zA-Z0-9_]*",
            ds in "[a-zA-Z_][a-zA-Z0-9_]*",
            y in "[a-zA-Z_][a-zA-Z0-9_]*"
        ) {
            let schema = TimeSeriesSchema::new(&unique_id, &ds, &y);
            
            assert_eq!(schema.unique_id_col, unique_id);
            assert_eq!(schema.ds_col, ds);
            assert_eq!(schema.y_col, y);
            
            let required_cols = schema.required_columns();
            assert_eq!(required_cols.len(), 3);
            assert!(required_cols.contains(&schema.unique_id_col));
            assert!(required_cols.contains(&schema.ds_col));
            assert!(required_cols.contains(&schema.y_col));
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_data_workflow() {
        // Create sample time series data
        let data = df! {
            "unique_id" => ["series_1", "series_1", "series_1", "series_2", "series_2", "series_2"],
            "ds" => [1, 2, 3, 1, 2, 3],
            "y" => [10.0, 12.0, 14.0, 20.0, 22.0, 24.0],
        }.unwrap();

        let schema = TimeSeriesSchema::default();
        
        // Create TimeSeriesDataFrame
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(data, schema.clone(), None);
        assert!(ts_df.is_ok());
        let ts_df = ts_df.unwrap();

        // Verify basic properties
        assert_eq!(ts_df.n_series(), 2);
        assert_eq!(ts_df.shape(), (6, 3));

        // Filter by series
        let series_1 = ts_df.filter_by_id("series_1");
        assert!(series_1.is_ok());
        let series_1 = series_1.unwrap();
        assert_eq!(series_1.shape().0, 3);

        // Create mock forecast results
        let forecast_data = df! {
            "unique_id" => ["series_1", "series_1", "series_2", "series_2"],
            "ds" => [4, 5, 4, 5],
            "MockModel" => [16.0, 18.0, 26.0, 28.0],
        }.unwrap();

        let forecast_df = ForecastDataFrame::<f64>::new(
            forecast_data,
            vec!["MockModel".to_string()],
            2,
            None,
            schema,
        );

        assert_eq!(forecast_df.forecast_horizon, 2);
        assert_eq!(forecast_df.models.len(), 1);

        // Test forecast extraction
        let model_forecasts = forecast_df.get_model_forecasts("MockModel");
        assert!(model_forecasts.is_ok());
    }

    #[test]
    fn test_metrics_calculation_workflow() {
        // Create test accuracy metrics
        let metrics1 = AccuracyMetrics::new(1.0f64, 1.0, 0.1, 0.15);
        let metrics2 = AccuracyMetrics::new(0.8f64, 0.64, 0.08, 0.12);

        // Verify RMSE calculation
        assert_relative_eq!(metrics1.rmse, 1.0, epsilon = 1e-10);
        assert_relative_eq!(metrics2.rmse, 0.8, epsilon = 1e-10);

        // Test custom metrics
        let enhanced_metrics = metrics1
            .with_custom_metric("correlation".to_string(), 0.95)
            .with_custom_metric("bias".to_string(), -0.02);

        assert_eq!(enhanced_metrics.get_metric("correlation"), Some(0.95));
        assert_eq!(enhanced_metrics.get_metric("bias"), Some(-0.02));

        let all_names = enhanced_metrics.metric_names();
        assert!(all_names.contains(&"correlation".to_string()));
        assert!(all_names.contains(&"bias".to_string()));
    }
}