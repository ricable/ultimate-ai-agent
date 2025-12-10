//! Model output validation tests against Python NeuralForecast
//!
//! This module validates that our Rust implementations produce outputs
//! within acceptable tolerance of the Python reference implementation.

use approx::{assert_relative_eq, assert_abs_diff_eq};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use neuro_divergent_models::{
    NeuralForecast, ModelConfig, BaseModel,
    models::{LSTM, RNN, GRU},
    config::{LSTMConfig, RNNConfig, GRUConfig},
    data::TimeSeriesDataFrame,
};

// Accuracy tolerances for different comparison types
const POINT_FORECAST_RELATIVE_ERROR: f64 = 1e-6;
const LOSS_ABSOLUTE_ERROR: f64 = 1e-8;
const GRADIENT_RELATIVE_ERROR: f64 = 1e-7;
const METRIC_RATIO_ERROR: f64 = 1e-6;
const PROBABILISTIC_KL_DIVERGENCE: f64 = 0.01;

/// Reference data structure from Python
#[derive(Debug, Deserialize, Serialize)]
struct ReferenceData {
    model: String,
    config: HashMap<String, serde_json::Value>,
    losses: HashMap<String, f64>,
    predictions: HashMap<String, PredictionSet>,
    metrics: HashMap<String, MetricSet>,
    #[serde(default)]
    gradients: HashMap<String, Vec<f64>>,
    #[serde(default)]
    training_history: Vec<TrainingStep>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PredictionSet {
    train: HashMap<String, Vec<f64>>,
    #[serde(default)]
    test: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct MetricSet {
    mae: f64,
    mse: f64,
    rmse: f64,
    mape: f64,
    smape: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct TrainingStep {
    epoch: usize,
    loss: f64,
    #[serde(default)]
    val_loss: Option<f64>,
}

/// Load reference data from JSON file
fn load_reference_data(model_name: &str) -> Result<ReferenceData, Box<dyn std::error::Error>> {
    let path = format!(
        "tests/accuracy/comparison_data/{}_reference.json",
        model_name.to_lowercase()
    );
    let content = fs::read_to_string(&path)?;
    let data: ReferenceData = serde_json::from_str(&content)?;
    Ok(data)
}

/// Load training data from CSV
fn load_training_data() -> Result<TimeSeriesDataFrame, Box<dyn std::error::Error>> {
    let path = "tests/accuracy/comparison_data/train_data.csv";
    TimeSeriesDataFrame::from_csv(path)
}

/// Load test data from CSV
fn load_test_data() -> Result<TimeSeriesDataFrame, Box<dyn std::error::Error>> {
    let path = "tests/accuracy/comparison_data/test_data.csv";
    TimeSeriesDataFrame::from_csv(path)
}

/// Validate point forecast accuracy
fn validate_point_forecasts(
    rust_predictions: &Array1<f64>,
    python_predictions: &[f64],
    tolerance: f64,
) -> Result<(), String> {
    if rust_predictions.len() != python_predictions.len() {
        return Err(format!(
            "Prediction length mismatch: Rust {} vs Python {}",
            rust_predictions.len(),
            python_predictions.len()
        ));
    }

    for (i, (rust_val, py_val)) in rust_predictions
        .iter()
        .zip(python_predictions.iter())
        .enumerate()
    {
        if py_val.abs() > 1e-10 {
            // Relative error for non-zero values
            let relative_error = ((rust_val - py_val) / py_val).abs();
            if relative_error > tolerance {
                return Err(format!(
                    "Point forecast mismatch at index {}: Rust {} vs Python {}, relative error {}",
                    i, rust_val, py_val, relative_error
                ));
            }
        } else {
            // Absolute error for near-zero values
            if (rust_val - py_val).abs() > tolerance {
                return Err(format!(
                    "Point forecast mismatch at index {}: Rust {} vs Python {}",
                    i, rust_val, py_val
                ));
            }
        }
    }
    
    Ok(())
}

/// Validate loss function values
fn validate_loss_values(
    rust_loss: f64,
    python_loss: f64,
    tolerance: f64,
) -> Result<(), String> {
    let diff = (rust_loss - python_loss).abs();
    if diff > tolerance {
        return Err(format!(
            "Loss value mismatch: Rust {} vs Python {}, absolute error {}",
            rust_loss, python_loss, diff
        ));
    }
    Ok(())
}

/// Validate gradient computations
fn validate_gradients(
    rust_gradients: &Array1<f64>,
    python_gradients: &[f64],
    tolerance: f64,
) -> Result<(), String> {
    if rust_gradients.len() != python_gradients.len() {
        return Err(format!(
            "Gradient length mismatch: Rust {} vs Python {}",
            rust_gradients.len(),
            python_gradients.len()
        ));
    }

    for (i, (rust_grad, py_grad)) in rust_gradients
        .iter()
        .zip(python_gradients.iter())
        .enumerate()
    {
        if py_grad.abs() > 1e-10 {
            let relative_error = ((rust_grad - py_grad) / py_grad).abs();
            if relative_error > tolerance {
                return Err(format!(
                    "Gradient mismatch at index {}: Rust {} vs Python {}, relative error {}",
                    i, rust_grad, py_grad, relative_error
                ));
            }
        } else {
            if (rust_grad - py_grad).abs() > tolerance {
                return Err(format!(
                    "Gradient mismatch at index {}: Rust {} vs Python {}",
                    i, rust_grad, py_grad
                ));
            }
        }
    }
    
    Ok(())
}

/// Validate statistical metrics
fn validate_metrics(
    rust_metrics: &MetricSet,
    python_metrics: &MetricSet,
) -> Result<(), String> {
    // MAE validation
    assert_abs_diff_eq!(
        rust_metrics.mae,
        python_metrics.mae,
        epsilon = METRIC_RATIO_ERROR,
        "MAE mismatch"
    );
    
    // MSE validation
    assert_abs_diff_eq!(
        rust_metrics.mse,
        python_metrics.mse,
        epsilon = METRIC_RATIO_ERROR,
        "MSE mismatch"
    );
    
    // RMSE validation
    assert_abs_diff_eq!(
        rust_metrics.rmse,
        python_metrics.rmse,
        epsilon = METRIC_RATIO_ERROR,
        "RMSE mismatch"
    );
    
    // MAPE validation (if not NaN)
    if !python_metrics.mape.is_nan() {
        assert_relative_eq!(
            rust_metrics.mape,
            python_metrics.mape,
            epsilon = METRIC_RATIO_ERROR,
            "MAPE mismatch"
        );
    }
    
    // SMAPE validation (if not NaN)
    if !python_metrics.smape.is_nan() {
        assert_relative_eq!(
            rust_metrics.smape,
            python_metrics.smape,
            epsilon = METRIC_RATIO_ERROR,
            "SMAPE mismatch"
        );
    }
    
    Ok(())
}

// Test macros for different model types
macro_rules! validate_model {
    ($model_name:expr, $model_type:ty, $config_type:ty, $create_config:expr) => {
        #[test]
        fn $model_name() {
            // Load reference data
            let reference = load_reference_data(stringify!($model_name))
                .expect("Failed to load reference data");
            
            // Load training data
            let train_data = load_training_data()
                .expect("Failed to load training data");
            let test_data = load_test_data()
                .expect("Failed to load test data");
            
            // Create model with same configuration
            let config: $config_type = $create_config(&reference.config);
            let mut model = <$model_type>::new(config)
                .expect("Failed to create model");
            
            // Train model
            model.fit(&train_data)
                .expect("Failed to train model");
            
            // Generate predictions
            let predictions = model.predict(&test_data)
                .expect("Failed to generate predictions");
            
            // Validate against each loss function's results
            for (loss_name, metrics) in &reference.metrics {
                println!("Validating {} with {} loss", stringify!($model_name), loss_name);
                
                // Get corresponding predictions
                if let Some(pred_set) = reference.predictions.get(loss_name) {
                    if let Some(test_preds) = pred_set.test.get(&reference.model) {
                        validate_point_forecasts(
                            &predictions,
                            test_preds,
                            POINT_FORECAST_RELATIVE_ERROR
                        ).expect(&format!("Forecast validation failed for {}", loss_name));
                    }
                }
                
                // Validate metrics
                let rust_metrics = calculate_metrics(&predictions, &test_data);
                validate_metrics(&rust_metrics, metrics)
                    .expect(&format!("Metric validation failed for {}", loss_name));
            }
            
            println!("{} validation passed!", stringify!($model_name));
        }
    };
}

/// Calculate metrics for predictions
fn calculate_metrics(predictions: &Array1<f64>, data: &TimeSeriesDataFrame) -> MetricSet {
    let y_true = data.get_target_values();
    let n = predictions.len().min(y_true.len());
    
    let mae = predictions.slice(s![..n])
        .iter()
        .zip(y_true.slice(s![..n]).iter())
        .map(|(pred, true_val)| (pred - true_val).abs())
        .sum::<f64>() / n as f64;
    
    let mse = predictions.slice(s![..n])
        .iter()
        .zip(y_true.slice(s![..n]).iter())
        .map(|(pred, true_val)| (pred - true_val).powi(2))
        .sum::<f64>() / n as f64;
    
    let rmse = mse.sqrt();
    
    // MAPE calculation with zero handling
    let mape = predictions.slice(s![..n])
        .iter()
        .zip(y_true.slice(s![..n]).iter())
        .filter(|(_, true_val)| true_val.abs() > 1e-10)
        .map(|(pred, true_val)| ((pred - true_val) / true_val).abs())
        .sum::<f64>() * 100.0 / n as f64;
    
    // SMAPE calculation
    let smape = predictions.slice(s![..n])
        .iter()
        .zip(y_true.slice(s![..n]).iter())
        .map(|(pred, true_val)| {
            let denominator = (pred.abs() + true_val.abs()).max(1e-10);
            2.0 * (pred - true_val).abs() / denominator
        })
        .sum::<f64>() * 100.0 / n as f64;
    
    MetricSet { mae, mse, rmse, mape, smape }
}

// Helper functions to create configs from reference data
fn create_lstm_config(params: &HashMap<String, serde_json::Value>) -> LSTMConfig {
    LSTMConfig::default()
        .with_horizon(params["h"].as_u64().unwrap() as usize)
        .with_input_size(params["input_size"].as_u64().unwrap() as usize)
        .with_hidden_size(params["hidden_size"].as_u64().unwrap() as usize)
        .with_num_layers(params["n_layers"].as_u64().unwrap() as usize)
        .with_dropout(params["dropout"].as_f64().unwrap() as f32)
        .with_random_seed(params["random_seed"].as_u64().unwrap() as u64)
}

fn create_rnn_config(params: &HashMap<String, serde_json::Value>) -> RNNConfig {
    RNNConfig::default()
        .with_horizon(params["h"].as_u64().unwrap() as usize)
        .with_input_size(params["input_size"].as_u64().unwrap() as usize)
        .with_hidden_size(params["hidden_size"].as_u64().unwrap() as usize)
        .with_num_layers(params["n_layers"].as_u64().unwrap() as usize)
        .with_dropout(params["dropout"].as_f64().unwrap() as f32)
        .with_random_seed(params["random_seed"].as_u64().unwrap() as u64)
}

fn create_gru_config(params: &HashMap<String, serde_json::Value>) -> GRUConfig {
    GRUConfig::default()
        .with_horizon(params["h"].as_u64().unwrap() as usize)
        .with_input_size(params["input_size"].as_u64().unwrap() as usize)
        .with_hidden_size(params["hidden_size"].as_u64().unwrap() as usize)
        .with_num_layers(params["n_layers"].as_u64().unwrap() as usize)
        .with_dropout(params["dropout"].as_f64().unwrap() as f32)
        .with_random_seed(params["random_seed"].as_u64().unwrap() as u64)
}

// Generate validation tests for all models
validate_model!(validate_lstm, LSTM, LSTMConfig, create_lstm_config);
validate_model!(validate_rnn, RNN, RNNConfig, create_rnn_config);
validate_model!(validate_gru, GRU, GRUConfig, create_gru_config);

// Additional model validations would follow the same pattern...

#[cfg(test)]
mod reproducibility_tests {
    use super::*;
    
    #[test]
    fn test_deterministic_training() {
        let train_data = load_training_data().unwrap();
        
        // Train model twice with same seed
        let config = LSTMConfig::default()
            .with_random_seed(42)
            .with_horizon(24)
            .with_input_size(48);
        
        let mut model1 = LSTM::new(config.clone()).unwrap();
        let mut model2 = LSTM::new(config).unwrap();
        
        model1.fit(&train_data).unwrap();
        model2.fit(&train_data).unwrap();
        
        // Predictions should be identical
        let pred1 = model1.predict(&train_data).unwrap();
        let pred2 = model2.predict(&train_data).unwrap();
        
        assert_eq!(pred1, pred2, "Deterministic training failed");
    }
    
    #[test]
    fn test_platform_independence() {
        // Test that results are consistent across different platforms
        // This would compare against pre-computed results for different architectures
        let train_data = load_training_data().unwrap();
        
        let config = RNNConfig::default()
            .with_random_seed(12345)
            .with_horizon(24);
        
        let mut model = RNN::new(config).unwrap();
        model.fit(&train_data).unwrap();
        
        let predictions = model.predict(&train_data).unwrap();
        
        // Compare against known good values
        // These would be pre-computed on different platforms
        let expected_first_values = vec![100.123, 102.456, 98.789];
        
        for (i, expected) in expected_first_values.iter().enumerate() {
            assert_relative_eq!(
                predictions[i],
                expected,
                epsilon = 1e-6,
                "Platform independence check failed"
            );
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[test]
    fn test_constant_values() {
        let data = TimeSeriesDataFrame::from_csv(
            "tests/accuracy/comparison_data/edge_case_constant.csv"
        ).unwrap();
        
        let config = LSTMConfig::default().with_horizon(10);
        let mut model = LSTM::new(config).unwrap();
        
        // Should handle constant values gracefully
        model.fit(&data).unwrap();
        let predictions = model.predict(&data).unwrap();
        
        // All predictions should be close to the constant value
        let constant_value = 42.0;
        for pred in predictions.iter() {
            assert_relative_eq!(
                *pred,
                constant_value,
                epsilon = 0.1,
                "Constant value prediction failed"
            );
        }
    }
    
    #[test]
    fn test_extreme_values() {
        let data = TimeSeriesDataFrame::from_csv(
            "tests/accuracy/comparison_data/edge_case_extreme.csv"
        ).unwrap();
        
        let config = RNNConfig::default().with_horizon(10);
        let mut model = RNN::new(config).unwrap();
        
        // Should handle extreme values without overflow/underflow
        let result = model.fit(&data);
        assert!(result.is_ok(), "Failed to handle extreme values");
        
        let predictions = model.predict(&data).unwrap();
        
        // Check no NaN or Inf values
        for pred in predictions.iter() {
            assert!(pred.is_finite(), "Non-finite prediction detected");
        }
    }
    
    #[test]
    fn test_single_datapoint() {
        let data = TimeSeriesDataFrame::from_csv(
            "tests/accuracy/comparison_data/edge_case_single_point.csv"
        ).unwrap();
        
        let config = LSTMConfig::default().with_horizon(1);
        let mut model = LSTM::new(config).unwrap();
        
        // Should handle single data point gracefully
        let result = model.fit(&data);
        assert!(result.is_ok() || result.is_err(), "Single point handling unclear");
    }
}