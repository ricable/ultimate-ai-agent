//! Core traits and types for climate prediction models

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::Result;

/// Input data for climate prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateInput {
    /// Time series of climate variables (temperature, pressure, humidity, etc.)
    pub features: Vec<f32>,

    /// Sequence length for temporal models
    pub sequence_length: Option<usize>,

    /// Spatial dimensions (lat, lon) for spatial models
    pub spatial_dims: Option<(usize, usize)>,

    /// Additional metadata
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

impl ClimateInput {
    /// Create a new climate input with features
    pub fn new(features: Vec<f32>) -> Self {
        Self {
            features,
            sequence_length: None,
            spatial_dims: None,
            metadata: None,
        }
    }

    /// Create a temporal sequence input
    pub fn with_sequence(features: Vec<f32>, sequence_length: usize) -> Self {
        Self {
            features,
            sequence_length: Some(sequence_length),
            spatial_dims: None,
            metadata: None,
        }
    }

    /// Create a spatial-temporal input
    pub fn with_spatial(features: Vec<f32>, lat_dim: usize, lon_dim: usize) -> Self {
        Self {
            features,
            sequence_length: None,
            spatial_dims: Some((lat_dim, lon_dim)),
            metadata: None,
        }
    }

    /// Get the shape of the input
    pub fn shape(&self) -> Vec<usize> {
        let mut shape = vec![];

        if let Some(seq_len) = self.sequence_length {
            shape.push(seq_len);
        }

        if let Some((lat, lon)) = self.spatial_dims {
            shape.push(lat);
            shape.push(lon);
        }

        // Calculate feature dimension
        let mut total_elements = self.features.len();
        for dim in &shape {
            total_elements /= dim;
        }
        shape.push(total_elements);

        shape
    }
}

/// Prediction output from climate models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimatePrediction {
    /// Predicted temperature (°C)
    pub temperature: f32,

    /// Predicted pressure (hPa)
    pub pressure: Option<f32>,

    /// Predicted humidity (%)
    pub humidity: Option<f32>,

    /// Prediction confidence/uncertainty
    pub confidence: f32,

    /// Time horizon (hours ahead)
    pub horizon: usize,

    /// Additional predictions
    pub extra: std::collections::HashMap<String, f32>,
}

impl ClimatePrediction {
    /// Create a simple temperature prediction
    pub fn temperature(value: f32, confidence: f32, horizon: usize) -> Self {
        Self {
            temperature: value,
            pressure: None,
            humidity: None,
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Create a full prediction with all variables
    pub fn full(temp: f32, pressure: f32, humidity: f32, confidence: f32, horizon: usize) -> Self {
        Self {
            temperature: temp,
            pressure: Some(pressure),
            humidity: Some(humidity),
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Evaluation metrics for model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Root Mean Squared Error
    pub rmse: f64,

    /// Mean Absolute Error
    pub mae: f64,

    /// R² Score (coefficient of determination)
    pub r2_score: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,

    /// Inference time (milliseconds)
    pub inference_time_ms: f64,

    /// Additional metrics
    pub extra: std::collections::HashMap<String, f64>,
}

impl ModelMetrics {
    /// Calculate metrics from predictions and ground truth
    pub fn calculate(predictions: &[f32], ground_truth: &[f32]) -> Result<Self> {
        if predictions.len() != ground_truth.len() {
            return Err(crate::ModelError::InvalidInputShape {
                expected: format!("{}", ground_truth.len()),
                actual: format!("{}", predictions.len()),
            });
        }

        let n = predictions.len() as f64;

        // Calculate errors
        let mut sum_squared_error = 0.0;
        let mut sum_absolute_error = 0.0;
        let mut sum_percentage_error = 0.0;

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            let error = (*pred - *truth) as f64;
            sum_squared_error += error * error;
            sum_absolute_error += error.abs();

            if truth.abs() > 1e-8 {
                sum_percentage_error += (error.abs() / (*truth as f64).abs()) * 100.0;
            }
        }

        let rmse = (sum_squared_error / n).sqrt();
        let mae = sum_absolute_error / n;
        let mape = sum_percentage_error / n;

        // Calculate R² score
        let mean_truth: f64 = ground_truth.iter().map(|&x| x as f64).sum::<f64>() / n;
        let ss_tot: f64 = ground_truth.iter()
            .map(|&x| (x as f64 - mean_truth).powi(2))
            .sum();
        let ss_res = sum_squared_error;
        let r2_score = 1.0 - (ss_res / ss_tot.max(1e-10));

        Ok(Self {
            rmse,
            mae,
            r2_score,
            mape,
            inference_time_ms: 0.0,
            extra: std::collections::HashMap::new(),
        })
    }
}

/// Core trait for all prediction models
#[async_trait]
pub trait PredictionModel: Send + Sync {
    /// Predict a single input
    async fn predict(&self, input: &ClimateInput) -> Result<ClimatePrediction>;

    /// Predict a batch of inputs
    async fn predict_batch(&self, inputs: &[ClimateInput]) -> Result<Vec<ClimatePrediction>> {
        let mut predictions = Vec::with_capacity(inputs.len());
        for input in inputs {
            predictions.push(self.predict(input).await?);
        }
        Ok(predictions)
    }

    /// Load model weights from a file
    async fn load_weights(&mut self, path: &Path) -> Result<()>;

    /// Save model weights to a file
    async fn save_weights(&self, path: &Path) -> Result<()>;

    /// Get model name/identifier
    fn name(&self) -> &str;

    /// Get model configuration
    fn config(&self) -> &crate::ModelConfig;

    /// Evaluate model on test data
    async fn evaluate(&self, inputs: &[ClimateInput], ground_truth: &[ClimatePrediction]) -> Result<ModelMetrics> {
        let start_time = std::time::Instant::now();
        let predictions = self.predict_batch(inputs).await?;
        let inference_time_ms = start_time.elapsed().as_millis() as f64;

        // Extract temperature values for metric calculation
        let pred_temps: Vec<f32> = predictions.iter().map(|p| p.temperature).collect();
        let truth_temps: Vec<f32> = ground_truth.iter().map(|p| p.temperature).collect();

        let mut metrics = ModelMetrics::calculate(&pred_temps, &truth_temps)?;
        metrics.inference_time_ms = inference_time_ms / inputs.len() as f64;

        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_climate_input_shape() {
        let input = ClimateInput::with_sequence(vec![1.0; 30], 10);
        assert_eq!(input.shape(), vec![10, 3]);
    }

    #[test]
    fn test_metrics_calculation() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ground_truth = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let metrics = ModelMetrics::calculate(&predictions, &ground_truth).unwrap();
        assert!(metrics.rmse < 0.15);
        assert!(metrics.mae < 0.11);
        assert!(metrics.r2_score > 0.99);
    }
}
