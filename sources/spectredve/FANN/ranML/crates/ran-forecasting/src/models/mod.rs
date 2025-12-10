//! RAN-specific forecasting models

pub mod traffic;
pub mod capacity;
pub mod performance;
pub mod anomaly;

// Re-export main model types
pub use traffic::TrafficPredictor;
pub use capacity::CapacityPredictor;
pub use performance::PerformancePredictor;
pub use anomaly::AnomalyDetector;

use std::collections::HashMap;
use crate::{
    error::{ForecastError, ForecastResult},
    data::{RanTimeSeries, ModelTrainingData, ModelForecastData, ForecastHorizon},
    adapters::{NeuroAdapter, RanModelAdapter},
    RanForecastingModel,
};

/// Ensemble forecaster that combines multiple models
#[derive(Debug)]
pub struct EnsembleForecaster {
    /// Individual models in the ensemble
    models: Vec<Box<dyn RanForecastingModel>>,
    /// Model weights for combining predictions
    weights: Vec<f64>,
    /// Ensemble method
    method: EnsembleMethod,
    /// Model names for identification
    model_names: Vec<String>,
}

impl EnsembleForecaster {
    /// Create a new ensemble forecaster
    pub fn new(method: EnsembleMethod) -> Self {
        Self {
            models: Vec::new(),
            weights: Vec::new(),
            method,
            model_names: Vec::new(),
        }
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: Box<dyn RanForecastingModel>, weight: f64) -> ForecastResult<()> {
        if weight <= 0.0 {
            return Err(ForecastError::invalid_parameter("weight", "must be positive"));
        }

        self.model_names.push(model.model_name().to_string());
        self.models.push(model);
        self.weights.push(weight);
        
        // Normalize weights
        let total_weight: f64 = self.weights.iter().sum();
        for weight in &mut self.weights {
            *weight /= total_weight;
        }

        Ok(())
    }

    /// Add multiple models with equal weights
    pub fn add_models(&mut self, models: Vec<Box<dyn RanForecastingModel>>) -> ForecastResult<()> {
        let weight = 1.0 / models.len() as f64;
        for model in models {
            self.add_model(model, weight)?;
        }
        Ok(())
    }

    /// Get number of models in ensemble
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Get model names
    pub fn model_names(&self) -> &[String] {
        &self.model_names
    }

    /// Combine predictions from multiple models
    fn combine_predictions(&self, predictions: Vec<ModelForecastData>) -> ForecastResult<ModelForecastData> {
        if predictions.is_empty() {
            return Err(ForecastError::prediction_error("No predictions to combine"));
        }

        let num_predictions = predictions[0].values.len();
        
        // Validate all predictions have same length
        for pred in &predictions {
            if pred.values.len() != num_predictions {
                return Err(ForecastError::prediction_error("Prediction lengths don't match"));
            }
        }

        let combined_values = match self.method {
            EnsembleMethod::WeightedAverage => {
                let mut combined = vec![0.0; num_predictions];
                for (i, pred) in predictions.iter().enumerate() {
                    let weight = self.weights[i];
                    for (j, &value) in pred.values.iter().enumerate() {
                        combined[j] += weight * value;
                    }
                }
                combined
            },
            EnsembleMethod::Median => {
                let mut combined = vec![0.0; num_predictions];
                for j in 0..num_predictions {
                    let mut values: Vec<f64> = predictions.iter().map(|p| p.values[j]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    
                    combined[j] = if values.len() % 2 == 0 {
                        let mid = values.len() / 2;
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                }
                combined
            },
            EnsembleMethod::Max => {
                let mut combined = vec![f64::NEG_INFINITY; num_predictions];
                for pred in &predictions {
                    for (j, &value) in pred.values.iter().enumerate() {
                        combined[j] = combined[j].max(value);
                    }
                }
                combined
            },
            EnsembleMethod::Min => {
                let mut combined = vec![f64::INFINITY; num_predictions];
                for pred in &predictions {
                    for (j, &value) in pred.values.iter().enumerate() {
                        combined[j] = combined[j].min(value);
                    }
                }
                combined
            },
        };

        // Use timestamps from the first prediction
        let timestamps = predictions[0].timestamps.clone();

        // Combine confidence intervals if available
        let confidence_intervals = if predictions.iter().all(|p| p.confidence_intervals.is_some()) {
            let mut combined_ci = vec![(0.0, 0.0); num_predictions];
            
            for j in 0..num_predictions {
                let mut lower_bounds = Vec::new();
                let mut upper_bounds = Vec::new();
                
                for (i, pred) in predictions.iter().enumerate() {
                    if let Some(ref ci) = pred.confidence_intervals {
                        let weight = self.weights[i];
                        lower_bounds.push(weight * ci[j].0);
                        upper_bounds.push(weight * ci[j].1);
                    }
                }
                
                combined_ci[j] = (
                    lower_bounds.iter().sum(),
                    upper_bounds.iter().sum(),
                );
            }
            
            Some(combined_ci)
        } else {
            None
        };

        Ok(ModelForecastData {
            values: combined_values,
            timestamps,
            confidence_intervals,
        })
    }
}

impl RanForecastingModel for EnsembleForecaster {
    fn model_name(&self) -> &str {
        "EnsembleForecaster"
    }

    fn fit(&mut self, data: &ModelTrainingData) -> ForecastResult<()> {
        if self.models.is_empty() {
            return Err(ForecastError::model_error("No models in ensemble"));
        }

        tracing::info!("Training ensemble with {} models", self.models.len());

        let mut failed_models = Vec::new();
        
        // Train each model independently
        for (i, model) in self.models.iter_mut().enumerate() {
            match model.fit(data) {
                Ok(_) => {
                    tracing::debug!("Successfully trained model {}: {}", i, model.model_name());
                },
                Err(e) => {
                    tracing::warn!("Failed to train model {}: {}", i, e);
                    failed_models.push(i);
                }
            }
        }

        if failed_models.len() == self.models.len() {
            return Err(ForecastError::training_error("All models in ensemble failed to train"));
        }

        if !failed_models.is_empty() {
            tracing::warn!(
                "{}/{} models failed training but continuing with partial ensemble",
                failed_models.len(),
                self.models.len()
            );
        }

        Ok(())
    }

    fn predict(&self, data: &ModelTrainingData) -> ForecastResult<ModelForecastData> {
        if self.models.is_empty() {
            return Err(ForecastError::model_error("No models in ensemble"));
        }

        let mut predictions = Vec::new();
        let mut successful_models = 0;

        // Get predictions from each model
        for (i, model) in self.models.iter().enumerate() {
            match model.predict(data) {
                Ok(pred) => {
                    predictions.push(pred);
                    successful_models += 1;
                },
                Err(e) => {
                    tracing::warn!("Model {} prediction failed: {}", i, e);
                }
            }
        }

        if predictions.is_empty() {
            return Err(ForecastError::prediction_error("All models in ensemble failed prediction"));
        }

        if successful_models < self.models.len() {
            tracing::warn!(
                "Only {}/{} models produced predictions",
                successful_models,
                self.models.len()
            );
        }

        self.combine_predictions(predictions)
    }

    fn predict_future(&self, horizon: ForecastHorizon) -> ForecastResult<ModelForecastData> {
        if self.models.is_empty() {
            return Err(ForecastError::model_error("No models in ensemble"));
        }

        let mut predictions = Vec::new();

        // Get future predictions from each model
        for model in &self.models {
            match model.predict_future(horizon) {
                Ok(pred) => predictions.push(pred),
                Err(e) => {
                    tracing::warn!("Model {} future prediction failed: {}", model.model_name(), e);
                }
            }
        }

        if predictions.is_empty() {
            return Err(ForecastError::prediction_error("All models failed future prediction"));
        }

        self.combine_predictions(predictions)
    }

    fn update(&mut self, data: &ModelTrainingData) -> ForecastResult<()> {
        let mut updated_models = 0;

        // Update each model that supports online learning
        for model in &mut self.models {
            if model.supports_online_learning() {
                match model.update(data) {
                    Ok(_) => updated_models += 1,
                    Err(e) => {
                        tracing::warn!("Failed to update model {}: {}", model.model_name(), e);
                    }
                }
            }
        }

        if updated_models == 0 {
            return Err(ForecastError::model_error("No models support online learning"));
        }

        tracing::debug!("Updated {}/{} models", updated_models, self.models.len());
        Ok(())
    }

    fn reset(&mut self) -> ForecastResult<()> {
        let mut reset_errors = Vec::new();

        for model in &mut self.models {
            if let Err(e) = model.reset() {
                reset_errors.push(format!("{}: {}", model.model_name(), e));
            }
        }

        if !reset_errors.is_empty() {
            return Err(ForecastError::model_error(format!(
                "Failed to reset some models: {}",
                reset_errors.join(", ")
            )));
        }

        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("model_type".to_string(), "EnsembleForecaster".to_string());
        params.insert("num_models".to_string(), self.models.len().to_string());
        params.insert("ensemble_method".to_string(), format!("{:?}", self.method));
        
        // Add individual model parameters
        for (i, model) in self.models.iter().enumerate() {
            for (key, value) in model.get_parameters() {
                params.insert(format!("model_{}_{}", i, key), value);
            }
        }

        params
    }

    fn supports_online_learning(&self) -> bool {
        self.models.iter().any(|m| m.supports_online_learning())
    }

    fn supports_multivariate(&self) -> bool {
        self.models.iter().any(|m| m.supports_multivariate())
    }
}

/// Methods for combining ensemble predictions
#[derive(Debug, Clone, Copy)]
pub enum EnsembleMethod {
    /// Weighted average of predictions
    WeightedAverage,
    /// Median of predictions
    Median,
    /// Maximum of predictions
    Max,
    /// Minimum of predictions
    Min,
}

/// Builder for ensemble forecasters
pub struct EnsembleBuilder {
    method: EnsembleMethod,
    models: Vec<(Box<dyn RanForecastingModel>, f64)>,
}

impl EnsembleBuilder {
    /// Create a new ensemble builder
    pub fn new(method: EnsembleMethod) -> Self {
        Self {
            method,
            models: Vec::new(),
        }
    }

    /// Add a model with weight
    pub fn add_model(mut self, model: Box<dyn RanForecastingModel>, weight: f64) -> Self {
        self.models.push((model, weight));
        self
    }

    /// Add a traffic predictor
    pub fn add_traffic_predictor(self, model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let model = TrafficPredictor::from_config(model_type, params)?;
        Ok(self.add_model(Box::new(model), 1.0))
    }

    /// Add a capacity predictor
    pub fn add_capacity_predictor(self, model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let model = CapacityPredictor::from_config(model_type, params)?;
        Ok(self.add_model(Box::new(model), 1.0))
    }

    /// Add a performance predictor
    pub fn add_performance_predictor(self, model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let model = PerformancePredictor::from_config(model_type, params)?;
        Ok(self.add_model(Box::new(model), 1.0))
    }

    /// Build the ensemble
    pub fn build(self) -> ForecastResult<EnsembleForecaster> {
        if self.models.is_empty() {
            return Err(ForecastError::config_error("No models added to ensemble"));
        }

        let mut ensemble = EnsembleForecaster::new(self.method);
        
        for (model, weight) in self.models {
            ensemble.add_model(model, weight)?;
        }

        Ok(ensemble)
    }
}

impl Default for EnsembleBuilder {
    fn default() -> Self {
        Self::new(EnsembleMethod::WeightedAverage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_creation() {
        let ensemble = EnsembleForecaster::new(EnsembleMethod::WeightedAverage);
        assert_eq!(ensemble.num_models(), 0);
        assert_eq!(ensemble.model_name(), "EnsembleForecaster");
    }

    #[test]
    fn test_ensemble_builder() {
        let builder = EnsembleBuilder::new(EnsembleMethod::Median);
        
        // Test that builder can be created
        let models = builder.models;
        assert!(models.is_empty());
    }

    #[test]
    fn test_ensemble_methods() {
        use std::mem::discriminant;
        
        let methods = [
            EnsembleMethod::WeightedAverage,
            EnsembleMethod::Median,
            EnsembleMethod::Max,
            EnsembleMethod::Min,
        ];
        
        // Test that all variants are different
        for i in 0..methods.len() {
            for j in (i + 1)..methods.len() {
                assert_ne!(discriminant(&methods[i]), discriminant(&methods[j]));
            }
        }
    }

    // Note: More comprehensive tests would require actual model implementations
    // These are basic structural tests
}