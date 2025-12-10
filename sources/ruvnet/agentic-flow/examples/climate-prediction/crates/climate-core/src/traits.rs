//! Core traits for extensibility

use crate::{
    types::*,
    error::Result,
};
use async_trait::async_trait;

/// Trait for data sources that provide climate observations
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Fetch historical observations for a location and time range
    async fn fetch_observations(
        &self,
        location: Location,
        time_range: TimeRange,
        variables: Vec<ClimateVariable>,
    ) -> Result<Vec<Observation>>;

    /// Get the name of this data source
    fn name(&self) -> &str;

    /// Check if this data source is available
    async fn is_available(&self) -> bool;
}

/// Trait for climate prediction models
#[async_trait]
pub trait PredictionModel: Send + Sync {
    /// Generate predictions for a single location
    async fn predict(
        &self,
        request: &PredictionRequest,
    ) -> Result<Vec<Prediction>>;

    /// Generate batch predictions for multiple locations
    async fn predict_batch(
        &self,
        request: &BatchPredictionRequest,
    ) -> Result<Vec<Prediction>>;

    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Validate model inputs
    fn validate_input(&self, request: &PredictionRequest) -> Result<()>;
}

/// Trait for physics-informed constraints
pub trait PhysicsConstraint: Send + Sync {
    /// Apply physics constraint to predictions
    fn apply(&self, predictions: &mut [Prediction]) -> Result<()>;

    /// Validate that predictions satisfy physics laws
    fn validate(&self, predictions: &[Prediction]) -> Result<()>;

    /// Get constraint name
    fn name(&self) -> &str;
}

/// Trait for model post-processing
pub trait PostProcessor: Send + Sync {
    /// Process predictions after model inference
    fn process(&self, predictions: Vec<Prediction>) -> Result<Vec<Prediction>>;

    /// Get processor name
    fn name(&self) -> &str;
}

/// Trait for data preprocessing
pub trait DataPreprocessor: Send + Sync {
    /// Preprocess observations before model input
    fn preprocess(&self, observations: Vec<Observation>) -> Result<Vec<Observation>>;

    /// Get preprocessor name
    fn name(&self) -> &str;
}

/// Trait for model evaluation metrics
pub trait MetricCalculator: Send + Sync {
    /// Calculate metric from predictions and observations
    fn calculate(
        &self,
        predictions: &[Prediction],
        observations: &[Observation],
    ) -> Result<f64>;

    /// Get metric name
    fn name(&self) -> &str;
}
