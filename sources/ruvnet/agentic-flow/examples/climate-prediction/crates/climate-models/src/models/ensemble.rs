//! Ensemble model combining multiple prediction models

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

use crate::{
    traits::{ClimateInput, ClimatePrediction, PredictionModel},
    ModelConfig, Result, ModelError,
};

/// Strategy for combining ensemble predictions
#[derive(Debug, Clone, Copy)]
pub enum EnsembleStrategy {
    /// Simple average of all predictions
    Average,
    /// Weighted average based on model confidence
    WeightedByConfidence,
    /// Weighted average with custom weights
    WeightedCustom,
    /// Use prediction from most confident model
    MostConfident,
    /// Median of predictions (robust to outliers)
    Median,
}

/// Ensemble model that combines multiple models
pub struct EnsembleModel {
    name: String,
    config: ModelConfig,
    models: Vec<Arc<dyn PredictionModel>>,
    strategy: EnsembleStrategy,
    custom_weights: Option<Vec<f32>>,
}

impl EnsembleModel {
    /// Create a new ensemble model
    pub fn new(name: impl Into<String>, strategy: EnsembleStrategy) -> Self {
        Self {
            name: name.into(),
            config: ModelConfig {
                architecture: "Ensemble".to_string(),
                input_dim: 0,
                output_dim: 0,
                hidden_dims: vec![],
                dropout: 0.0,
                num_layers: 0,
                extra: std::collections::HashMap::new(),
            },
            models: Vec::new(),
            strategy,
            custom_weights: None,
        }
    }

    /// Create an ensemble with custom weights
    pub fn with_weights(name: impl Into<String>, weights: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            config: ModelConfig {
                architecture: "Ensemble".to_string(),
                input_dim: 0,
                output_dim: 0,
                hidden_dims: vec![],
                dropout: 0.0,
                num_layers: 0,
                extra: std::collections::HashMap::new(),
            },
            models: Vec::new(),
            strategy: EnsembleStrategy::WeightedCustom,
            custom_weights: Some(weights),
        }
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: Arc<dyn PredictionModel>) {
        info!("Adding model '{}' to ensemble", model.name());
        self.models.push(model);
    }

    /// Combine predictions using the configured strategy
    fn combine_predictions(&self, predictions: Vec<ClimatePrediction>) -> Result<ClimatePrediction> {
        if predictions.is_empty() {
            return Err(ModelError::InferenceFailed(
                "No predictions to combine".to_string()
            ));
        }

        match self.strategy {
            EnsembleStrategy::Average => self.average_predictions(&predictions),
            EnsembleStrategy::WeightedByConfidence => self.weighted_by_confidence(&predictions),
            EnsembleStrategy::WeightedCustom => self.weighted_custom(&predictions),
            EnsembleStrategy::MostConfident => self.most_confident(&predictions),
            EnsembleStrategy::Median => self.median_predictions(&predictions),
        }
    }

    /// Simple average of predictions
    fn average_predictions(&self, predictions: &[ClimatePrediction]) -> Result<ClimatePrediction> {
        let n = predictions.len() as f32;

        let temperature = predictions.iter().map(|p| p.temperature).sum::<f32>() / n;

        let pressure = if predictions.iter().all(|p| p.pressure.is_some()) {
            Some(predictions.iter().filter_map(|p| p.pressure).sum::<f32>() / n)
        } else {
            None
        };

        let humidity = if predictions.iter().all(|p| p.humidity.is_some()) {
            Some(predictions.iter().filter_map(|p| p.humidity).sum::<f32>() / n)
        } else {
            None
        };

        let confidence = predictions.iter().map(|p| p.confidence).sum::<f32>() / n;
        let horizon = predictions[0].horizon;

        Ok(ClimatePrediction {
            temperature,
            pressure,
            humidity,
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        })
    }

    /// Weighted average by model confidence
    fn weighted_by_confidence(&self, predictions: &[ClimatePrediction]) -> Result<ClimatePrediction> {
        let total_confidence: f32 = predictions.iter().map(|p| p.confidence).sum();

        if total_confidence < 1e-6 {
            return self.average_predictions(predictions);
        }

        let temperature = predictions.iter()
            .map(|p| p.temperature * p.confidence)
            .sum::<f32>() / total_confidence;

        let pressure = if predictions.iter().all(|p| p.pressure.is_some()) {
            Some(predictions.iter()
                .filter_map(|p| p.pressure.map(|pr| pr * p.confidence))
                .sum::<f32>() / total_confidence)
        } else {
            None
        };

        let confidence = predictions.iter().map(|p| p.confidence).max_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }).unwrap_or(0.0);

        let horizon = predictions[0].horizon;

        Ok(ClimatePrediction {
            temperature,
            pressure,
            humidity: None,
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        })
    }

    /// Weighted average with custom weights
    fn weighted_custom(&self, predictions: &[ClimatePrediction]) -> Result<ClimatePrediction> {
        let weights = self.custom_weights.as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("No custom weights set".to_string()))?;

        if weights.len() != predictions.len() {
            return Err(ModelError::InvalidInputShape {
                expected: format!("{} weights", predictions.len()),
                actual: format!("{} weights", weights.len()),
            });
        }

        let total_weight: f32 = weights.iter().sum();

        let temperature = predictions.iter()
            .zip(weights.iter())
            .map(|(p, w)| p.temperature * w)
            .sum::<f32>() / total_weight;

        let confidence = predictions.iter()
            .zip(weights.iter())
            .map(|(p, w)| p.confidence * w)
            .sum::<f32>() / total_weight;

        let horizon = predictions[0].horizon;

        Ok(ClimatePrediction {
            temperature,
            pressure: None,
            humidity: None,
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        })
    }

    /// Use prediction from most confident model
    fn most_confident(&self, predictions: &[ClimatePrediction]) -> Result<ClimatePrediction> {
        predictions.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .ok_or_else(|| ModelError::InferenceFailed("No predictions".to_string()))
    }

    /// Median of predictions
    fn median_predictions(&self, predictions: &[ClimatePrediction]) -> Result<ClimatePrediction> {
        let mut temps: Vec<f32> = predictions.iter().map(|p| p.temperature).collect();
        temps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let temperature = if temps.len() % 2 == 0 {
            (temps[temps.len() / 2 - 1] + temps[temps.len() / 2]) / 2.0
        } else {
            temps[temps.len() / 2]
        };

        let confidence = predictions.iter().map(|p| p.confidence).sum::<f32>() / predictions.len() as f32;
        let horizon = predictions[0].horizon;

        Ok(ClimatePrediction {
            temperature,
            pressure: None,
            humidity: None,
            confidence,
            horizon,
            extra: std::collections::HashMap::new(),
        })
    }
}

#[async_trait]
impl PredictionModel for EnsembleModel {
    async fn predict(&self, input: &ClimateInput) -> Result<ClimatePrediction> {
        if self.models.is_empty() {
            return Err(ModelError::ModelNotFound("No models in ensemble".to_string()));
        }

        debug!("Running ensemble inference with {} models", self.models.len());

        // Get predictions from all models
        let mut predictions = Vec::with_capacity(self.models.len());

        for model in &self.models {
            match model.predict(input).await {
                Ok(pred) => predictions.push(pred),
                Err(e) => {
                    // Log error but continue with other models
                    tracing::warn!("Model '{}' failed: {}", model.name(), e);
                }
            }
        }

        if predictions.is_empty() {
            return Err(ModelError::InferenceFailed(
                "All models failed".to_string()
            ));
        }

        // Combine predictions
        self.combine_predictions(predictions)
    }

    async fn predict_batch(&self, inputs: &[ClimateInput]) -> Result<Vec<ClimatePrediction>> {
        // Parallel batch prediction for better performance
        let mut predictions = Vec::with_capacity(inputs.len());

        for input in inputs {
            predictions.push(self.predict(input).await?);
        }

        Ok(predictions)
    }

    async fn load_weights(&mut self, _path: &Path) -> Result<()> {
        // Ensemble doesn't have its own weights
        // Individual models are loaded separately
        Ok(())
    }

    async fn save_weights(&self, _path: &Path) -> Result<()> {
        // Ensemble doesn't have its own weights
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_strategies() {
        let preds = vec![
            ClimatePrediction::temperature(20.0, 0.9, 1),
            ClimatePrediction::temperature(22.0, 0.8, 1),
            ClimatePrediction::temperature(21.0, 0.85, 1),
        ];

        let ensemble = EnsembleModel::new("test", EnsembleStrategy::Average);

        let result = ensemble.average_predictions(&preds).unwrap();
        assert!((result.temperature - 21.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_by_confidence() {
        let preds = vec![
            ClimatePrediction::temperature(20.0, 0.9, 1),
            ClimatePrediction::temperature(25.0, 0.1, 1),
        ];

        let ensemble = EnsembleModel::new("test", EnsembleStrategy::WeightedByConfidence);
        let result = ensemble.weighted_by_confidence(&preds).unwrap();

        // Should be closer to 20.0 due to higher confidence
        assert!(result.temperature < 21.0);
    }
}
