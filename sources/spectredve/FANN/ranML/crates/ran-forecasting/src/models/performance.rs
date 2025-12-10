//! Performance prediction models for RAN forecasting

use std::collections::HashMap;
use crate::{
    error::{ForecastError, ForecastResult},
    data::{ModelTrainingData, ModelForecastData, ForecastHorizon},
    adapters::NeuroAdapter,
    RanForecastingModel,
};

/// Performance predictor for RAN network KPI forecasting
#[derive(Debug)]
pub struct PerformancePredictor {
    model: Box<dyn RanForecastingModel>,
    config: PerformancePredictorConfig,
}

impl PerformancePredictor {
    pub fn new(model: Box<dyn RanForecastingModel>, config: PerformancePredictorConfig) -> Self {
        Self { model, config }
    }

    pub fn from_config(model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let input_size = params.get("input_size").and_then(|s| s.parse().ok()).unwrap_or(168);
        let horizon = params.get("horizon").and_then(|s| s.parse().ok()).unwrap_or(24);
        
        let model = NeuroAdapter::from_name(model_type, input_size, horizon, params)?;
        let config = PerformancePredictorConfig::default();
        
        Ok(Self::new(model, config))
    }
}

impl RanForecastingModel for PerformancePredictor {
    fn model_name(&self) -> &str { "PerformancePredictor" }
    fn fit(&mut self, data: &ModelTrainingData) -> ForecastResult<()> { self.model.fit(data) }
    fn predict(&self, data: &ModelTrainingData) -> ForecastResult<ModelForecastData> { self.model.predict(data) }
    fn predict_future(&self, horizon: ForecastHorizon) -> ForecastResult<ModelForecastData> { self.model.predict_future(horizon) }
    fn update(&mut self, data: &ModelTrainingData) -> ForecastResult<()> { self.model.update(data) }
    fn reset(&mut self) -> ForecastResult<()> { self.model.reset() }
    fn get_parameters(&self) -> HashMap<String, String> { self.model.get_parameters() }
    fn supports_online_learning(&self) -> bool { self.model.supports_online_learning() }
    fn supports_multivariate(&self) -> bool { self.model.supports_multivariate() }
}

#[derive(Debug, Clone, Default)]
pub struct PerformancePredictorConfig {
    pub kpi_type: KpiCategory,
    pub performance_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum KpiCategory {
    Throughput,
    Latency,
    Quality,
    Efficiency,
}

impl Default for KpiCategory {
    fn default() -> Self { Self::Throughput }
}