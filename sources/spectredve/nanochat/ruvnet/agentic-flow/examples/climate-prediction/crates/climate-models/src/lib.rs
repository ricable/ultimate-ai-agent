//! # Climate Models
//!
//! Machine learning prediction models for climate forecasting.
//!
//! This crate provides multiple model architectures:
//! - LSTM: Long Short-Term Memory networks for temporal sequences
//! - FNO: Fourier Neural Operator for spatial-temporal patterns
//! - Ensemble: Combination of multiple models for robust predictions
//!
//! ## Example
//!
//! ```rust,no_run
//! use climate_models::{PredictionModel, models::LSTMModel, ClimateInput};
//! # use anyhow::Result;
//!
//! # async fn example() -> Result<()> {
//! let model = LSTMModel::from_onnx("models/lstm_climate.onnx").await?;
//! let input = ClimateInput::new(vec![15.5, 1013.2, 65.0]);
//! let prediction = model.predict(&input).await?;
//! println!("Predicted temperature: {:.2}°C", prediction.temperature);
//! # Ok(())
//! # }
//! ```

pub mod models;
pub mod storage;
pub mod traits;
pub mod training;

// Re-export commonly used types
pub use models::{EnsembleModel, FNOModel, LSTMModel};
pub use traits::{ClimatePrediction, ClimateInput, PredictionModel, ModelMetrics};
pub use training::{ModelTrainer, ModelEvaluator, TrainingConfig};
pub use storage::{ModelStorage, ModelVersion};

use thiserror::Error;

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

/// Errors that can occur during model operations
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidInputShape { expected: String, actual: String },

    #[error("Model inference failed: {0}")]
    InferenceFailed(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("ONNX Runtime error: {0}")]
    OnnxError(#[from] ort::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Model configuration for all architectures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Model architecture type
    pub architecture: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,

    /// Dropout probability
    pub dropout: f64,

    /// Number of layers
    pub num_layers: usize,

    /// Additional model-specific parameters
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Create a new configuration for LSTM model
    pub fn lstm(input_dim: usize, hidden_dim: usize, num_layers: usize, output_dim: usize) -> Self {
        Self {
            architecture: "LSTM".to_string(),
            input_dim,
            output_dim,
            hidden_dims: vec![hidden_dim],
            dropout: 0.2,
            num_layers,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Create a new configuration for FNO model
    pub fn fno(input_dim: usize, modes: usize, width: usize, output_dim: usize) -> Self {
        let mut extra = std::collections::HashMap::new();
        extra.insert("modes".to_string(), serde_json::json!(modes));
        extra.insert("width".to_string(), serde_json::json!(width));

        Self {
            architecture: "FNO".to_string(),
            input_dim,
            output_dim,
            hidden_dims: vec![width],
            dropout: 0.0,
            num_layers: 4,
            extra,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_lstm() {
        let config = ModelConfig::lstm(10, 64, 2, 3);
        assert_eq!(config.architecture, "LSTM");
        assert_eq!(config.input_dim, 10);
        assert_eq!(config.output_dim, 3);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_model_config_fno() {
        let config = ModelConfig::fno(10, 12, 64, 3);
        assert_eq!(config.architecture, "FNO");
        assert_eq!(config.extra.get("modes").unwrap(), &serde_json::json!(12));
    }
}
