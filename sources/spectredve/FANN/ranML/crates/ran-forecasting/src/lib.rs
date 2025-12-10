//! # RAN Forecasting
//!
//! Time series forecasting capabilities for Radio Access Network (RAN) optimization
//! using neuro-divergent neural forecasting models. This crate provides RAN-specific
//! adaptations of state-of-the-art forecasting models for traffic prediction,
//! capacity planning, and network optimization.
//!
//! ## Features
//!
//! - **Traffic Forecasting**: Predict network traffic patterns and user demand
//! - **Capacity Planning**: Forecast resource utilization and network load
//! - **Performance Prediction**: Predict KPIs like throughput, latency, and quality
//! - **Anomaly Detection**: Detect unusual patterns in network behavior
//! - **Multi-horizon Forecasting**: Short-term (minutes) to long-term (days) predictions
//! - **Multi-variate Support**: Handle multiple time series simultaneously
//!
//! ## Supported Models
//!
//! - **DLinear**: Direct linear forecasting with decomposition
//! - **NBEATS**: Neural basis expansion analysis for time series
//! - **LSTM/GRU**: Recurrent neural networks for sequential data
//! - **Transformers**: Attention-based models for complex patterns
//! - **Ensemble**: Combine multiple models for robust predictions
//!
//! ## Quick Start
//!
//! ```rust
//! use ran_forecasting::{RanForecaster, models::TrafficPredictor, data::RanTimeSeries};
//! use ran_core::{KpiType, TimeSeries};
//! use chrono::Utc;
//!
//! # fn main() -> ran_forecasting::ForecastResult<()> {
//! // Create time series data
//! let mut timeseries = RanTimeSeries::new("cell_throughput".to_string());
//! timeseries.add_measurement(Utc::now(), 100.5)?;
//! timeseries.add_measurement(Utc::now(), 105.2)?;
//!
//! // Configure traffic predictor
//! let predictor = TrafficPredictor::builder()
//!     .model_type("dlinear")
//!     .horizon(24) // 24-hour forecast
//!     .input_window(168) // 1 week history
//!     .build()?;
//!
//! // Create forecaster
//! let mut forecaster = RanForecaster::new(predictor);
//!
//! // Train and predict
//! forecaster.fit(&timeseries)?;
//! let forecast = forecaster.predict()?;
//!
//! println!("Traffic forecast: {:?}", forecast.values);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//!
//! - **models**: RAN-specific forecasting model implementations
//! - **data**: Time series data structures and adapters for RAN data
//! - **adapters**: Integration adapters for neuro-divergent models
//! - **metrics**: RAN-specific evaluation metrics and accuracy measures
//! - **ensemble**: Multi-model ensemble forecasting capabilities

use std::collections::HashMap;
use std::fmt;

// Re-export core types
pub use ran_core::{
    KpiType, TimeSeries, TimePoint, PerformanceMetrics,
    NetworkTopology, Cell, UE, GeoCoordinate,
};

// Re-export forecasting types
pub use adapters::{RanModelAdapter, NeuroAdapter};
pub use data::{RanTimeSeries, RanTimeSeriesDataset, ForecastHorizon};
pub use models::{
    TrafficPredictor, CapacityPredictor, PerformancePredictor,
    AnomalyDetector, EnsembleForecaster,
};
pub use metrics::{ForecastAccuracy, RanMetrics};
pub use config::{RanForecastingConfig, ConfigPresets};

// Core modules
pub mod error;
pub mod data;
pub mod models;
pub mod adapters;
pub mod metrics;
pub mod ensemble;
pub mod config;

// Re-export error types
pub use error::{ForecastError, ForecastResult};

/// Main forecasting interface for RAN applications
#[derive(Debug)]
pub struct RanForecaster<T> 
where
    T: RanForecastingModel,
{
    /// Underlying forecasting model
    pub model: T,
    /// Configuration
    pub config: ForecastConfig,
    /// Training history
    pub training_history: Vec<TrainingEvent>,
    /// Model performance metrics
    pub performance: Option<ForecastAccuracy>,
    /// Is the model fitted
    pub is_fitted: bool,
}

impl<T> RanForecaster<T>
where
    T: RanForecastingModel,
{
    /// Create a new RAN forecaster with the given model
    pub fn new(model: T) -> Self {
        Self {
            model,
            config: ForecastConfig::default(),
            training_history: Vec::new(),
            performance: None,
            is_fitted: false,
        }
    }

    /// Create a new forecaster with configuration
    pub fn with_config(model: T, config: ForecastConfig) -> Self {
        Self {
            model,
            config,
            training_history: Vec::new(),
            performance: None,
            is_fitted: false,
        }
    }

    /// Fit the forecasting model to RAN time series data
    pub fn fit(&mut self, timeseries: &RanTimeSeries) -> ForecastResult<()> {
        tracing::info!("Training RAN forecasting model: {}", self.model.model_name());
        
        let start_time = std::time::Instant::now();
        
        // Validate input data
        self.validate_training_data(timeseries)?;
        
        // Convert RAN data to model format
        let training_data = self.prepare_training_data(timeseries)?;
        
        // Train the model
        self.model.fit(&training_data)?;
        
        // Calculate performance metrics
        let validation_forecast = self.model.predict(&training_data)?;
        self.performance = Some(self.calculate_performance(timeseries, &validation_forecast)?);
        
        let training_time = start_time.elapsed();
        
        // Record training event
        let training_event = TrainingEvent {
            timestamp: chrono::Utc::now(),
            duration: training_time,
            data_points: timeseries.len(),
            model_type: self.model.model_name().to_string(),
            performance: self.performance.clone(),
        };
        self.training_history.push(training_event);
        
        self.is_fitted = true;
        
        tracing::info!(
            "Model training completed in {:?}. Performance: {:?}",
            training_time,
            self.performance
        );
        
        Ok(())
    }

    /// Generate forecasts for the configured horizon
    pub fn predict(&self) -> ForecastResult<RanForecast> {
        if !self.is_fitted {
            return Err(ForecastError::ModelNotFitted);
        }

        tracing::debug!("Generating forecast with horizon: {:?}", self.config.horizon);
        
        let forecast_data = self.model.predict_future(self.config.horizon)?;
        
        let forecast = RanForecast {
            values: forecast_data.values,
            timestamps: forecast_data.timestamps,
            confidence_intervals: forecast_data.confidence_intervals,
            metadata: self.create_forecast_metadata(),
            model_info: ModelInfo {
                name: self.model.model_name().to_string(),
                version: "1.0.0".to_string(),
                parameters: self.model.get_parameters(),
            },
        };

        Ok(forecast)
    }

    /// Predict with custom input data
    pub fn predict_with_data(&self, input_data: &RanTimeSeries) -> ForecastResult<RanForecast> {
        if !self.is_fitted {
            return Err(ForecastError::ModelNotFitted);
        }

        let prepared_data = self.prepare_prediction_data(input_data)?;
        let forecast_data = self.model.predict(&prepared_data)?;
        
        let forecast = RanForecast {
            values: forecast_data.values,
            timestamps: forecast_data.timestamps,
            confidence_intervals: forecast_data.confidence_intervals,
            metadata: self.create_forecast_metadata(),
            model_info: ModelInfo {
                name: self.model.model_name().to_string(),
                version: "1.0.0".to_string(),
                parameters: self.model.get_parameters(),
            },
        };

        Ok(forecast)
    }

    /// Update the model with new data (online learning)
    pub fn update(&mut self, new_data: &RanTimeSeries) -> ForecastResult<()> {
        if !self.is_fitted {
            return Err(ForecastError::ModelNotFitted);
        }

        tracing::debug!("Updating model with {} new data points", new_data.len());
        
        let update_data = self.prepare_training_data(new_data)?;
        self.model.update(&update_data)?;
        
        // Update performance metrics
        let validation_forecast = self.model.predict(&update_data)?;
        self.performance = Some(self.calculate_performance(new_data, &validation_forecast)?);

        Ok(())
    }

    /// Get model performance metrics
    pub fn get_performance(&self) -> Option<&ForecastAccuracy> {
        self.performance.as_ref()
    }

    /// Get training history
    pub fn get_training_history(&self) -> &[TrainingEvent] {
        &self.training_history
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Reset the model (unfitted state)
    pub fn reset(&mut self) -> ForecastResult<()> {
        self.model.reset()?;
        self.training_history.clear();
        self.performance = None;
        self.is_fitted = false;
        Ok(())
    }

    /// Validate training data
    fn validate_training_data(&self, timeseries: &RanTimeSeries) -> ForecastResult<()> {
        if timeseries.is_empty() {
            return Err(ForecastError::DataError("Empty time series".to_string()));
        }

        let min_data_points = self.config.minimum_training_points();
        if timeseries.len() < min_data_points {
            return Err(ForecastError::DataError(format!(
                "Insufficient data: {} points, need at least {}",
                timeseries.len(),
                min_data_points
            )));
        }

        // Check for data quality issues
        if timeseries.has_missing_values() && !self.config.allow_missing_values {
            return Err(ForecastError::DataError("Missing values not allowed".to_string()));
        }

        if timeseries.has_outliers(self.config.outlier_threshold) && !self.config.allow_outliers {
            return Err(ForecastError::DataError("Outliers detected".to_string()));
        }

        Ok(())
    }

    /// Prepare training data in model format
    fn prepare_training_data(&self, timeseries: &RanTimeSeries) -> ForecastResult<ModelTrainingData> {
        // This would convert RAN time series to the format expected by neuro-divergent models
        let data = ModelTrainingData {
            values: timeseries.values().to_vec(),
            timestamps: timeseries.timestamps().to_vec(),
            features: timeseries.features().clone(),
            target_name: timeseries.name().clone(),
        };
        Ok(data)
    }

    /// Prepare data for prediction
    fn prepare_prediction_data(&self, timeseries: &RanTimeSeries) -> ForecastResult<ModelTrainingData> {
        self.prepare_training_data(timeseries)
    }

    /// Calculate performance metrics
    fn calculate_performance(
        &self,
        actual: &RanTimeSeries,
        forecast: &ModelForecastData,
    ) -> ForecastResult<ForecastAccuracy> {
        let accuracy = ForecastAccuracy::calculate(
            actual.values(),
            &forecast.values,
            &self.config.accuracy_metrics,
        )?;
        Ok(accuracy)
    }

    /// Create forecast metadata
    fn create_forecast_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("model_type".to_string(), self.model.model_name().to_string());
        metadata.insert("horizon".to_string(), format!("{:?}", self.config.horizon));
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        
        if let Some(ref performance) = self.performance {
            metadata.insert("mape".to_string(), format!("{:.4}", performance.mape));
            metadata.insert("rmse".to_string(), format!("{:.4}", performance.rmse));
        }
        
        metadata
    }
}

/// Trait for RAN forecasting models
pub trait RanForecastingModel {
    /// Get the model name
    fn model_name(&self) -> &str;

    /// Fit the model to training data
    fn fit(&mut self, data: &ModelTrainingData) -> ForecastResult<()>;

    /// Make predictions on input data
    fn predict(&self, data: &ModelTrainingData) -> ForecastResult<ModelForecastData>;

    /// Predict future values for a given horizon
    fn predict_future(&self, horizon: ForecastHorizon) -> ForecastResult<ModelForecastData>;

    /// Update model with new data (online learning)
    fn update(&mut self, data: &ModelTrainingData) -> ForecastResult<()>;

    /// Reset the model
    fn reset(&mut self) -> ForecastResult<()>;

    /// Get model parameters
    fn get_parameters(&self) -> HashMap<String, String>;

    /// Check if model supports online learning
    fn supports_online_learning(&self) -> bool {
        false
    }

    /// Check if model supports multivariate input
    fn supports_multivariate(&self) -> bool {
        false
    }
}

/// Configuration for RAN forecasting
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Forecast horizon
    pub horizon: ForecastHorizon,
    /// Minimum training data points required
    pub min_training_points: usize,
    /// Allow missing values in data
    pub allow_missing_values: bool,
    /// Allow outliers in data
    pub allow_outliers: bool,
    /// Outlier detection threshold (standard deviations)
    pub outlier_threshold: f64,
    /// Accuracy metrics to calculate
    pub accuracy_metrics: Vec<AccuracyMetric>,
    /// Enable confidence intervals
    pub confidence_intervals: bool,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            horizon: ForecastHorizon::Hours(24),
            min_training_points: 100,
            allow_missing_values: true,
            allow_outliers: true,
            outlier_threshold: 3.0,
            accuracy_metrics: vec![
                AccuracyMetric::MAPE,
                AccuracyMetric::RMSE,
                AccuracyMetric::MAE,
            ],
            confidence_intervals: true,
            confidence_level: 0.95,
        }
    }
}

impl ForecastConfig {
    /// Get minimum training points based on horizon
    pub fn minimum_training_points(&self) -> usize {
        let horizon_points = match self.horizon {
            ForecastHorizon::Minutes(m) => m,
            ForecastHorizon::Hours(h) => h * 60,
            ForecastHorizon::Days(d) => d * 24 * 60,
        };
        self.min_training_points.max(horizon_points * 2)
    }
}

/// Forecast result from RAN forecasting
#[derive(Debug, Clone)]
pub struct RanForecast {
    /// Predicted values
    pub values: Vec<f64>,
    /// Timestamps for predictions
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Confidence intervals (if available)
    pub confidence_intervals: Option<Vec<(f64, f64)>>,
    /// Forecast metadata
    pub metadata: HashMap<String, String>,
    /// Model information
    pub model_info: ModelInfo,
}

impl RanForecast {
    /// Get forecast length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if forecast is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get forecast value at index
    pub fn get_value(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    /// Get timestamp at index
    pub fn get_timestamp(&self, index: usize) -> Option<chrono::DateTime<chrono::Utc>> {
        self.timestamps.get(index).copied()
    }

    /// Get confidence interval at index
    pub fn get_confidence_interval(&self, index: usize) -> Option<(f64, f64)> {
        self.confidence_intervals.as_ref()?.get(index).copied()
    }

    /// Convert to time series format
    pub fn to_timeseries(&self, name: String) -> RanTimeSeries {
        let mut ts = RanTimeSeries::new(name);
        for (value, timestamp) in self.values.iter().zip(self.timestamps.iter()) {
            ts.add_measurement_at(*timestamp, *value).unwrap();
        }
        ts
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model parameters
    pub parameters: HashMap<String, String>,
}

/// Training event record
#[derive(Debug, Clone)]
pub struct TrainingEvent {
    /// Training timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Training duration
    pub duration: std::time::Duration,
    /// Number of data points used
    pub data_points: usize,
    /// Model type
    pub model_type: String,
    /// Performance achieved
    pub performance: Option<ForecastAccuracy>,
}

/// Model training data structure
#[derive(Debug, Clone)]
pub struct ModelTrainingData {
    /// Target values
    pub values: Vec<f64>,
    /// Timestamps
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Additional features
    pub features: HashMap<String, Vec<f64>>,
    /// Target variable name
    pub target_name: String,
}

/// Model forecast output
#[derive(Debug, Clone)]
pub struct ModelForecastData {
    /// Predicted values
    pub values: Vec<f64>,
    /// Prediction timestamps
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Confidence intervals
    pub confidence_intervals: Option<Vec<(f64, f64)>>,
}

/// Accuracy metrics to calculate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccuracyMetric {
    /// Mean Absolute Percentage Error
    MAPE,
    /// Root Mean Square Error
    RMSE,
    /// Mean Absolute Error
    MAE,
    /// Mean Squared Error
    MSE,
    /// Symmetric Mean Absolute Percentage Error
    SMAPE,
    /// Mean Absolute Scaled Error
    MASE,
}

impl fmt::Display for AccuracyMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccuracyMetric::MAPE => write!(f, "MAPE"),
            AccuracyMetric::RMSE => write!(f, "RMSE"),
            AccuracyMetric::MAE => write!(f, "MAE"),
            AccuracyMetric::MSE => write!(f, "MSE"),
            AccuracyMetric::SMAPE => write!(f, "SMAPE"),
            AccuracyMetric::MASE => write!(f, "MASE"),
        }
    }
}

/// Builder for RAN forecaster
pub struct RanForecasterBuilder<T> 
where 
    T: RanForecastingModel,
{
    model: Option<T>,
    config: ForecastConfig,
}

impl<T> RanForecasterBuilder<T>
where
    T: RanForecastingModel,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            config: ForecastConfig::default(),
        }
    }

    /// Set the forecasting model
    pub fn model(mut self, model: T) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the forecast horizon
    pub fn horizon(mut self, horizon: ForecastHorizon) -> Self {
        self.config.horizon = horizon;
        self
    }

    /// Set minimum training points
    pub fn min_training_points(mut self, points: usize) -> Self {
        self.config.min_training_points = points;
        self
    }

    /// Enable/disable missing values
    pub fn allow_missing_values(mut self, allow: bool) -> Self {
        self.config.allow_missing_values = allow;
        self
    }

    /// Enable/disable outliers
    pub fn allow_outliers(mut self, allow: bool) -> Self {
        self.config.allow_outliers = allow;
        self
    }

    /// Set outlier threshold
    pub fn outlier_threshold(mut self, threshold: f64) -> Self {
        self.config.outlier_threshold = threshold;
        self
    }

    /// Set accuracy metrics
    pub fn accuracy_metrics(mut self, metrics: Vec<AccuracyMetric>) -> Self {
        self.config.accuracy_metrics = metrics;
        self
    }

    /// Enable/disable confidence intervals
    pub fn confidence_intervals(mut self, enable: bool) -> Self {
        self.config.confidence_intervals = enable;
        self
    }

    /// Set confidence level
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    /// Build the forecaster
    pub fn build(self) -> ForecastResult<RanForecaster<T>> {
        let model = self.model.ok_or(ForecastError::Configuration("Model not specified".to_string()))?;
        Ok(RanForecaster::with_config(model, self.config))
    }
}

impl<T> Default for RanForecasterBuilder<T>
where
    T: RanForecastingModel,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_config_default() {
        let config = ForecastConfig::default();
        assert!(matches!(config.horizon, ForecastHorizon::Hours(24)));
        assert_eq!(config.min_training_points, 100);
        assert!(config.allow_missing_values);
        assert!(config.allow_outliers);
        assert_eq!(config.outlier_threshold, 3.0);
        assert!(config.confidence_intervals);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_forecast_config_minimum_training_points() {
        let mut config = ForecastConfig::default();
        config.horizon = ForecastHorizon::Hours(1);
        assert_eq!(config.minimum_training_points(), 120); // max(100, 1*60*2)
        
        config.horizon = ForecastHorizon::Days(1);
        assert_eq!(config.minimum_training_points(), 2880); // max(100, 1*24*60*2)
    }

    #[test]
    fn test_accuracy_metric_display() {
        assert_eq!(format!("{}", AccuracyMetric::MAPE), "MAPE");
        assert_eq!(format!("{}", AccuracyMetric::RMSE), "RMSE");
        assert_eq!(format!("{}", AccuracyMetric::MAE), "MAE");
    }

    #[test]
    fn test_ran_forecast_basic() {
        let forecast = RanForecast {
            values: vec![1.0, 2.0, 3.0],
            timestamps: vec![
                chrono::Utc::now(),
                chrono::Utc::now(),
                chrono::Utc::now(),
            ],
            confidence_intervals: None,
            metadata: HashMap::new(),
            model_info: ModelInfo {
                name: "test".to_string(),
                version: "1.0".to_string(),
                parameters: HashMap::new(),
            },
        };

        assert_eq!(forecast.len(), 3);
        assert!(!forecast.is_empty());
        assert_eq!(forecast.get_value(0), Some(1.0));
        assert!(forecast.get_timestamp(0).is_some());
        assert_eq!(forecast.get_confidence_interval(0), None);
    }
}