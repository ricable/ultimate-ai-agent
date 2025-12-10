//! Configuration utilities for RAN forecasting

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{
    error::{ForecastError, ForecastResult},
    data::ForecastHorizon,
    AccuracyMetric,
};

/// Global configuration for RAN forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RanForecastingConfig {
    /// Default model configurations
    pub models: HashMap<String, ModelConfig>,
    /// Default forecast horizons by application
    pub horizons: HashMap<String, ForecastHorizon>,
    /// Default accuracy thresholds
    pub accuracy_thresholds: HashMap<String, f64>,
    /// Data preprocessing settings
    pub preprocessing: PreprocessingConfig,
    /// Training settings
    pub training: TrainingConfig,
    /// Ensemble settings
    pub ensemble: EnsembleConfig,
}

impl Default for RanForecastingConfig {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert("traffic_dlinear".to_string(), ModelConfig {
            model_type: "dlinear".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("input_size".to_string(), "168".to_string()); // 1 week
                params.insert("horizon".to_string(), "24".to_string()); // 24 hours
                params.insert("moving_avg_window".to_string(), "25".to_string());
                params
            },
        });

        let mut horizons = HashMap::new();
        horizons.insert("traffic".to_string(), ForecastHorizon::Hours(24));
        horizons.insert("capacity".to_string(), ForecastHorizon::Days(7));
        horizons.insert("performance".to_string(), ForecastHorizon::Hours(6));

        let mut thresholds = HashMap::new();
        thresholds.insert("mape_threshold".to_string(), 15.0);
        thresholds.insert("rmse_threshold".to_string(), 1.0);

        Self {
            models,
            horizons,
            accuracy_thresholds: thresholds,
            preprocessing: PreprocessingConfig::default(),
            training: TrainingConfig::default(),
            ensemble: EnsembleConfig::default(),
        }
    }
}

impl RanForecastingConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> ForecastResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ForecastError::IoError(e.to_string()))?;
        
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ForecastError::SerializationError(e.to_string()))?;
        
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> ForecastResult<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ForecastError::SerializationError(e.to_string()))?;
        
        std::fs::write(path, content)
            .map_err(|e| ForecastError::IoError(e.to_string()))?;
        
        Ok(())
    }

    /// Get model configuration by name
    pub fn get_model_config(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }

    /// Add or update model configuration
    pub fn set_model_config(&mut self, name: String, config: ModelConfig) {
        self.models.insert(name, config);
    }

    /// Get forecast horizon for application
    pub fn get_horizon(&self, application: &str) -> ForecastHorizon {
        self.horizons.get(application)
            .copied()
            .unwrap_or(ForecastHorizon::Hours(24))
    }

    /// Set forecast horizon for application
    pub fn set_horizon(&mut self, application: String, horizon: ForecastHorizon) {
        self.horizons.insert(application, horizon);
    }

    /// Get accuracy threshold
    pub fn get_threshold(&self, metric: &str) -> f64 {
        self.accuracy_thresholds.get(metric)
            .copied()
            .unwrap_or(0.0)
    }

    /// Set accuracy threshold
    pub fn set_threshold(&mut self, metric: String, threshold: f64) {
        self.accuracy_thresholds.insert(metric, threshold);
    }
}

/// Model-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type identifier
    pub model_type: String,
    /// Model parameters
    pub parameters: HashMap<String, String>,
}

impl ModelConfig {
    /// Create new model configuration
    pub fn new(model_type: String) -> Self {
        Self {
            model_type,
            parameters: HashMap::new(),
        }
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Get parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&String> {
        self.parameters.get(key)
    }

    /// Get parameter as integer
    pub fn get_int_parameter(&self, key: &str) -> Option<i32> {
        self.parameters.get(key)?.parse().ok()
    }

    /// Get parameter as float
    pub fn get_float_parameter(&self, key: &str) -> Option<f64> {
        self.parameters.get(key)?.parse().ok()
    }

    /// Get parameter as boolean
    pub fn get_bool_parameter(&self, key: &str) -> Option<bool> {
        self.parameters.get(key)?.parse().ok()
    }

    /// Set parameter
    pub fn set_parameter(&mut self, key: String, value: String) {
        self.parameters.insert(key, value);
    }
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Enable data normalization
    pub normalize: bool,
    /// Normalization method
    pub normalization_method: String,
    /// Handle missing values
    pub handle_missing: bool,
    /// Missing value strategy
    pub missing_strategy: String,
    /// Remove outliers
    pub remove_outliers: bool,
    /// Outlier detection threshold (standard deviations)
    pub outlier_threshold: f64,
    /// Enable feature scaling
    pub scale_features: bool,
    /// Feature scaling method
    pub scaling_method: String,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            normalization_method: "standard".to_string(),
            handle_missing: true,
            missing_strategy: "linear".to_string(),
            remove_outliers: false,
            outlier_threshold: 3.0,
            scale_features: true,
            scaling_method: "minmax".to_string(),
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Default training epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Enable cross-validation
    pub cross_validation: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Shuffle training data
    pub shuffle: bool,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            learning_rate: 0.001,
            batch_size: 64,
            validation_split: 0.2,
            early_stopping_patience: Some(50),
            cross_validation: false,
            cv_folds: 5,
            shuffle: true,
            random_seed: None,
        }
    }
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Default ensemble method
    pub method: String,
    /// Model weights
    pub weights: HashMap<String, f64>,
    /// Enable automatic weight tuning
    pub auto_weights: bool,
    /// Weight tuning method
    pub weight_tuning_method: String,
    /// Maximum number of models in ensemble
    pub max_models: usize,
    /// Model selection strategy
    pub selection_strategy: String,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            method: "weighted_average".to_string(),
            weights: HashMap::new(),
            auto_weights: true,
            weight_tuning_method: "accuracy_based".to_string(),
            max_models: 5,
            selection_strategy: "best_performers".to_string(),
        }
    }
}

/// Configuration builder for easier setup
pub struct ConfigBuilder {
    config: RanForecastingConfig,
}

impl ConfigBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: RanForecastingConfig::default(),
        }
    }

    /// Add model configuration
    pub fn add_model(mut self, name: String, model_type: String, params: HashMap<String, String>) -> Self {
        let model_config = ModelConfig {
            model_type,
            parameters: params,
        };
        self.config.models.insert(name, model_config);
        self
    }

    /// Set horizon for application
    pub fn set_horizon(mut self, application: String, horizon: ForecastHorizon) -> Self {
        self.config.horizons.insert(application, horizon);
        self
    }

    /// Set accuracy threshold
    pub fn set_threshold(mut self, metric: String, threshold: f64) -> Self {
        self.config.accuracy_thresholds.insert(metric, threshold);
        self
    }

    /// Configure preprocessing
    pub fn preprocessing(mut self, config: PreprocessingConfig) -> Self {
        self.config.preprocessing = config;
        self
    }

    /// Configure training
    pub fn training(mut self, config: TrainingConfig) -> Self {
        self.config.training = config;
        self
    }

    /// Configure ensemble
    pub fn ensemble(mut self, config: EnsembleConfig) -> Self {
        self.config.ensemble = config;
        self
    }

    /// Build the configuration
    pub fn build(self) -> RanForecastingConfig {
        self.config
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset configurations for common use cases
pub struct ConfigPresets;

impl ConfigPresets {
    /// Traffic forecasting preset
    pub fn traffic_forecasting() -> RanForecastingConfig {
        ConfigBuilder::new()
            .add_model(
                "traffic_dlinear".to_string(),
                "dlinear".to_string(),
                {
                    let mut params = HashMap::new();
                    params.insert("input_size".to_string(), "168".to_string());
                    params.insert("horizon".to_string(), "24".to_string());
                    params
                }
            )
            .set_horizon("traffic".to_string(), ForecastHorizon::Hours(24))
            .set_threshold("mape_threshold".to_string(), 10.0)
            .build()
    }

    /// Capacity planning preset
    pub fn capacity_planning() -> RanForecastingConfig {
        ConfigBuilder::new()
            .add_model(
                "capacity_lstm".to_string(),
                "lstm".to_string(),
                {
                    let mut params = HashMap::new();
                    params.insert("input_size".to_string(), "336".to_string()); // 2 weeks
                    params.insert("horizon".to_string(), "168".to_string()); // 1 week
                    params.insert("hidden_size".to_string(), "128".to_string());
                    params
                }
            )
            .set_horizon("capacity".to_string(), ForecastHorizon::Days(7))
            .set_threshold("mape_threshold".to_string(), 20.0)
            .build()
    }

    /// Performance monitoring preset
    pub fn performance_monitoring() -> RanForecastingConfig {
        ConfigBuilder::new()
            .add_model(
                "performance_mlp".to_string(),
                "mlp".to_string(),
                {
                    let mut params = HashMap::new();
                    params.insert("input_size".to_string(), "24".to_string());
                    params.insert("horizon".to_string(), "6".to_string());
                    params.insert("hidden_layers".to_string(), "64".to_string());
                    params
                }
            )
            .set_horizon("performance".to_string(), ForecastHorizon::Hours(6))
            .set_threshold("mape_threshold".to_string(), 5.0)
            .build()
    }

    /// Anomaly detection preset
    pub fn anomaly_detection() -> RanForecastingConfig {
        ConfigBuilder::new()
            .add_model(
                "anomaly_autoencoder".to_string(),
                "mlp".to_string(),
                {
                    let mut params = HashMap::new();
                    params.insert("input_size".to_string(), "48".to_string());
                    params.insert("horizon".to_string(), "1".to_string());
                    params.insert("hidden_layers".to_string(), "32,16,32".to_string());
                    params
                }
            )
            .set_horizon("anomaly".to_string(), ForecastHorizon::Hours(1))
            .set_threshold("anomaly_threshold".to_string(), 2.0)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = RanForecastingConfig::default();
        
        assert!(!config.models.is_empty());
        assert!(!config.horizons.is_empty());
        assert!(!config.accuracy_thresholds.is_empty());
        assert!(config.preprocessing.normalize);
        assert_eq!(config.training.max_epochs, 1000);
        assert_eq!(config.ensemble.max_models, 5);
    }

    #[test]
    fn test_model_config() {
        let mut config = ModelConfig::new("dlinear".to_string());
        config.set_parameter("input_size".to_string(), "168".to_string());
        config.set_parameter("learning_rate".to_string(), "0.001".to_string());
        
        assert_eq!(config.model_type, "dlinear");
        assert_eq!(config.get_parameter("input_size"), Some(&"168".to_string()));
        assert_eq!(config.get_int_parameter("input_size"), Some(168));
        assert_eq!(config.get_float_parameter("learning_rate"), Some(0.001));
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .set_horizon("test".to_string(), ForecastHorizon::Hours(12))
            .set_threshold("test_metric".to_string(), 5.0)
            .build();
        
        assert_eq!(config.get_horizon("test"), ForecastHorizon::Hours(12));
        assert_eq!(config.get_threshold("test_metric"), 5.0);
    }

    #[test]
    fn test_config_serialization() {
        let config = RanForecastingConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RanForecastingConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.training.max_epochs, deserialized.training.max_epochs);
        assert_eq!(config.preprocessing.normalize, deserialized.preprocessing.normalize);
    }

    #[test]
    fn test_config_file_operations() {
        let config = RanForecastingConfig::default();
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();
        
        // Save config to file
        config.to_file(file_path).unwrap();
        
        // Load config from file
        let loaded_config = RanForecastingConfig::from_file(file_path).unwrap();
        
        assert_eq!(config.training.max_epochs, loaded_config.training.max_epochs);
    }

    #[test]
    fn test_preset_configs() {
        let traffic_config = ConfigPresets::traffic_forecasting();
        assert!(traffic_config.models.contains_key("traffic_dlinear"));
        assert_eq!(traffic_config.get_horizon("traffic"), ForecastHorizon::Hours(24));
        
        let capacity_config = ConfigPresets::capacity_planning();
        assert!(capacity_config.models.contains_key("capacity_lstm"));
        assert_eq!(capacity_config.get_horizon("capacity"), ForecastHorizon::Days(7));
        
        let performance_config = ConfigPresets::performance_monitoring();
        assert!(performance_config.models.contains_key("performance_mlp"));
        assert_eq!(performance_config.get_horizon("performance"), ForecastHorizon::Hours(6));
        
        let anomaly_config = ConfigPresets::anomaly_detection();
        assert!(anomaly_config.models.contains_key("anomaly_autoencoder"));
        assert_eq!(anomaly_config.get_horizon("anomaly"), ForecastHorizon::Hours(1));
    }

    #[test]
    fn test_preprocessing_config() {
        let config = PreprocessingConfig::default();
        
        assert!(config.normalize);
        assert_eq!(config.normalization_method, "standard");
        assert!(config.handle_missing);
        assert_eq!(config.missing_strategy, "linear");
        assert_eq!(config.outlier_threshold, 3.0);
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        
        assert_eq!(config.max_epochs, 1000);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.validation_split, 0.2);
        assert!(config.shuffle);
    }

    #[test]
    fn test_ensemble_config() {
        let config = EnsembleConfig::default();
        
        assert_eq!(config.method, "weighted_average");
        assert!(config.auto_weights);
        assert_eq!(config.max_models, 5);
        assert_eq!(config.selection_strategy, "best_performers");
    }
}