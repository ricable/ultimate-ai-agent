//! Capacity prediction models for RAN forecasting

use std::collections::HashMap;
use crate::{
    error::{ForecastError, ForecastResult},
    data::{ModelTrainingData, ModelForecastData, ForecastHorizon},
    adapters::NeuroAdapter,
    RanForecastingModel,
};

/// Capacity predictor for RAN network capacity planning
#[derive(Debug)]
pub struct CapacityPredictor {
    model: Box<dyn RanForecastingModel>,
    config: CapacityPredictorConfig,
}

impl CapacityPredictor {
    pub fn new(model: Box<dyn RanForecastingModel>, config: CapacityPredictorConfig) -> Self {
        Self { model, config }
    }

    pub fn from_config(model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let input_size = params.get("input_size").and_then(|s| s.parse().ok()).unwrap_or(168);
        let horizon = params.get("horizon").and_then(|s| s.parse().ok()).unwrap_or(24);
        
        let model = NeuroAdapter::from_name(model_type, input_size, horizon, params)?;
        let config = CapacityPredictorConfig::default();
        
        Ok(Self::new(model, config))
    }
}

impl RanForecastingModel for CapacityPredictor {
    fn model_name(&self) -> &str { "CapacityPredictor" }
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
pub struct CapacityPredictorConfig {
    pub capacity_type: CapacityType,
    pub planning_horizon: PlanningHorizon,
}

#[derive(Debug, Clone, Copy)]
pub enum CapacityType {
    Processing,
    Memory,
    Bandwidth,
    Storage,
}

impl Default for CapacityType {
    fn default() -> Self { Self::Processing }
}

#[derive(Debug, Clone, Copy)]
pub enum PlanningHorizon {
    ShortTerm,  // Days to weeks
    MediumTerm, // Weeks to months
    LongTerm,   // Months to years
}

impl Default for PlanningHorizon {
    fn default() -> Self { Self::MediumTerm }
}