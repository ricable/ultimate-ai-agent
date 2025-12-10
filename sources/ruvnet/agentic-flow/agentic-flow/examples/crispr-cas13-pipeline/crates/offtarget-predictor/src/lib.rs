//! # Off-Target Predictor
//!
//! ML-based off-target prediction for CRISPR-Cas13.
//! Uses machine learning models to predict potential off-target cleavage sites.

pub mod error;
pub mod features;
pub mod ml_model;
pub mod scoring;

use async_trait::async_trait;
use data_models::targets::{CrisprTarget, OffTargetPrediction, OffTargetSite};
pub use error::{PredictionError, Result};

/// Main predictor interface
#[async_trait]
pub trait OffTargetPredictor: Send + Sync {
    /// Predict off-targets for a given CRISPR target
    async fn predict(&self, target: &CrisprTarget) -> Result<OffTargetPrediction>;

    /// Batch predict off-targets for multiple targets
    async fn predict_batch(&self, targets: Vec<CrisprTarget>) -> Result<Vec<OffTargetPrediction>>;

    /// Get predictor name
    fn name(&self) -> &str;

    /// Get model version
    fn version(&self) -> &str;
}

/// Off-target prediction configuration
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// Reference genome path
    pub reference_genome: String,
    /// Maximum mismatches to search for
    pub max_mismatches: u8,
    /// Minimum score threshold for reporting
    pub min_score: f64,
    /// Use ML model for scoring
    pub use_ml_model: bool,
    /// ML model path
    pub model_path: Option<String>,
    /// Number of parallel workers
    pub num_workers: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            reference_genome: String::new(),
            max_mismatches: 3,
            min_score: 0.3,
            use_ml_model: true,
            model_path: None,
            num_workers: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_config_default() {
        let config = PredictionConfig::default();
        assert_eq!(config.max_mismatches, 3);
        assert_eq!(config.min_score, 0.3);
        assert!(config.use_ml_model);
    }
}
