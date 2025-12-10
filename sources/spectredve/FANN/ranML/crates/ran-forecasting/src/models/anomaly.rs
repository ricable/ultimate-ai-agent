//! Anomaly detection models for RAN forecasting

use std::collections::HashMap;
use crate::{
    error::{ForecastError, ForecastResult},
    data::{ModelTrainingData, ModelForecastData, ForecastHorizon},
    adapters::NeuroAdapter,
    RanForecastingModel,
};

/// Anomaly detector for RAN network behavior
#[derive(Debug)]
pub struct AnomalyDetector {
    model: Box<dyn RanForecastingModel>,
    config: AnomalyDetectorConfig,
}

impl AnomalyDetector {
    pub fn new(model: Box<dyn RanForecastingModel>, config: AnomalyDetectorConfig) -> Self {
        Self { model, config }
    }

    pub fn from_config(model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let input_size = params.get("input_size").and_then(|s| s.parse().ok()).unwrap_or(168);
        let horizon = params.get("horizon").and_then(|s| s.parse().ok()).unwrap_or(24);
        
        let model = NeuroAdapter::from_name(model_type, input_size, horizon, params)?;
        let config = AnomalyDetectorConfig::default();
        
        Ok(Self::new(model, config))
    }

    /// Detect anomalies in the given data
    pub fn detect_anomalies(&self, data: &ModelTrainingData) -> ForecastResult<Vec<AnomalyScore>> {
        let predictions = self.model.predict(data)?;
        let actual_values = &data.values[data.values.len() - predictions.values.len()..];
        
        let mut anomaly_scores = Vec::new();
        for (i, (&actual, &predicted)) in actual_values.iter().zip(predictions.values.iter()).enumerate() {
            let residual = (actual - predicted).abs();
            let score = AnomalyScore {
                timestamp: predictions.timestamps[i],
                actual_value: actual,
                predicted_value: predicted,
                residual,
                anomaly_score: residual / (predicted.abs() + 1e-8), // Normalized residual
                is_anomaly: residual > self.config.threshold,
            };
            anomaly_scores.push(score);
        }
        
        Ok(anomaly_scores)
    }
}

impl RanForecastingModel for AnomalyDetector {
    fn model_name(&self) -> &str { "AnomalyDetector" }
    fn fit(&mut self, data: &ModelTrainingData) -> ForecastResult<()> { self.model.fit(data) }
    fn predict(&self, data: &ModelTrainingData) -> ForecastResult<ModelForecastData> { self.model.predict(data) }
    fn predict_future(&self, horizon: ForecastHorizon) -> ForecastResult<ModelForecastData> { self.model.predict_future(horizon) }
    fn update(&mut self, data: &ModelTrainingData) -> ForecastResult<()> { self.model.update(data) }
    fn reset(&mut self) -> ForecastResult<()> { self.model.reset() }
    fn get_parameters(&self) -> HashMap<String, String> { self.model.get_parameters() }
    fn supports_online_learning(&self) -> bool { self.model.supports_online_learning() }
    fn supports_multivariate(&self) -> bool { self.model.supports_multivariate() }
}

#[derive(Debug, Clone)]
pub struct AnomalyDetectorConfig {
    pub threshold: f64,
    pub detection_method: DetectionMethod,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            threshold: 2.0, // 2 standard deviations
            detection_method: DetectionMethod::Statistical,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DetectionMethod {
    Statistical,
    IsolationForest,
    OneClassSVM,
    AutoEncoder,
}

#[derive(Debug, Clone)]
pub struct AnomalyScore {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub actual_value: f64,
    pub predicted_value: f64,
    pub residual: f64,
    pub anomaly_score: f64,
    pub is_anomaly: bool,
}