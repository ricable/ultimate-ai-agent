//! # ran-training: Unified Neural Network Training for Telecom Data
//!
//! A comprehensive crate for training neural networks on telecom performance data
//! using the ruv-FANN library. Combines core training functionality with swarm 
//! orchestration, advanced evaluation metrics, and multi-deployment support.
//!
//! ## Features
//!
//! - **Multi-algorithm training**: Backpropagation, RProp, QuickProp
//! - **Swarm orchestration**: Parallel training with multiple strategies
//! - **Telecom data preprocessing**: Specialized for network performance metrics
//! - **WebAssembly support**: Train and deploy in browsers/edge devices
//! - **Advanced evaluation**: Comprehensive metrics and model comparison
//! - **Hyperparameter tuning**: Grid search and optimization
//! - **Performance optimization**: SIMD acceleration and memory efficiency
//! - **Real-time prediction**: Low-latency inference for live data

#![allow(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "wasm")]
extern crate alloc;
#[cfg(feature = "wasm")]
use alloc::{string::String, vec::Vec};

// Re-export core types
pub use ruv_fann::{ActivationFunction, Network, NetworkBuilder, TrainingAlgorithm};

// Public modules
pub mod data;
pub mod data_splitter;
pub mod preprocessing;
pub mod error;
pub mod models;
pub mod training;
pub mod evaluation;
pub mod config;

// Swarm orchestration (optional feature)
#[cfg(feature = "swarm")]
pub mod swarm;

// WebAssembly bindings
#[cfg(feature = "wasm")]
pub mod wasm;

// GPU support (optional feature)
#[cfg(feature = "gpu")]
pub mod gpu;

// Binary utilities
pub mod utils;

// Re-exports for convenience
pub use error::{TrainingError, TrainingResult};
pub use data::{TelecomDataset, TelecomRecord, TargetType, FeatureStats};
pub use data_splitter::{DataSplitter, DataSplitConfig, DatasetJson, SplitInfo, create_train_test_split};
pub use models::{NeuralModel, TrainingParameters, ModelArchitecture};
pub use training::{TrainingConfig};
pub use evaluation::{EvaluationMetrics, ModelComparison, CrossValidator};
pub use config::SystemConfig;

#[cfg(feature = "swarm")]
pub use swarm::{SwarmOrchestrator, SwarmConfig, SwarmAgent, AgentType};

/// Training metrics for tracking training progress
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingMetrics {
    pub epoch: u32,
    pub training_error: f32,
    pub validation_error: f32,
    pub training_time: std::time::Duration,
}

/// Prediction result structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PredictionResult {
    pub value: f32,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Telecom trainer (alias for compatibility)
pub type TelecomTrainer = UnifiedTrainingSystem;

/// Core training configuration with comprehensive options
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnifiedTrainingConfig {
    /// Neural network architecture: [inputs, hidden_layers..., outputs]
    pub architecture: Vec<usize>,
    /// Training algorithm to use
    pub algorithm: TrainingAlgorithm,
    /// Learning rate (0.0 - 1.0)
    pub learning_rate: f32,
    /// Maximum training epochs
    pub max_epochs: u32,
    /// Target error for training convergence
    pub target_error: f32,
    /// Validation split ratio (0.0 - 1.0)
    pub validation_split: f32,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: u32,
    /// Batch size for training (0 = full batch)
    pub batch_size: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable parallel training
    pub parallel_training: bool,
    /// Enable swarm orchestration
    #[cfg(feature = "swarm")]
    pub enable_swarm: bool,
    /// Swarm configuration
    #[cfg(feature = "swarm")]
    pub swarm_config: Option<SwarmConfig>,
}

impl Default for UnifiedTrainingConfig {
    fn default() -> Self {
        Self {
            architecture: vec![21, 16, 12, 8, 1], // Updated for full telecom feature set
            algorithm: TrainingAlgorithm::RProp,
            learning_rate: 0.1,
            max_epochs: 1000,
            target_error: 0.001,
            validation_split: 0.2,
            early_stopping_patience: 50,
            batch_size: 0, // Full batch by default
            random_seed: Some(42),
            parallel_training: true,
            #[cfg(feature = "swarm")]
            enable_swarm: false,
            #[cfg(feature = "swarm")]
            swarm_config: None,
        }
    }
}

/// Comprehensive training results with detailed metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnifiedTrainingResults {
    /// Individual model results
    pub model_results: Vec<training::ModelTrainingResult>,
    /// Best performing model name
    pub best_model_name: String,
    /// Best validation error achieved
    pub best_validation_error: f32,
    /// Total training time
    pub total_training_time: std::time::Duration,
    /// Detailed evaluation metrics
    pub evaluation_metrics: evaluation::EvaluationMetrics,
    /// Model comparison analysis
    pub model_comparison: evaluation::ModelComparison,
    /// Cross-validation results (if performed)
    pub cross_validation: Option<evaluation::CrossValidationResults>,
    /// Swarm orchestration results (if used)
    #[cfg(feature = "swarm")]
    pub swarm_results: Option<swarm::SwarmResults>,
}

/// Main unified training system that combines all functionality
pub struct UnifiedTrainingSystem {
    /// Configuration
    pub config: UnifiedTrainingConfig,
    /// Swarm orchestrator (if enabled)
    #[cfg(feature = "swarm")]
    pub swarm: Option<SwarmOrchestrator>,
    /// Training metrics
    pub metrics: TrainingMetrics,
}

impl UnifiedTrainingSystem {
    /// Create a new training system with default configuration
    pub fn new() -> Self {
        Self {
            config: UnifiedTrainingConfig::default(),
            #[cfg(feature = "swarm")]
            swarm: None,
            metrics: TrainingMetrics::default(),
        }
    }

    /// Create a training system with custom configuration
    pub fn with_config(config: UnifiedTrainingConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "swarm")]
            swarm: None,
            metrics: TrainingMetrics::default(),
        }
    }

    /// Load configuration from file
    pub fn load_config<P: AsRef<std::path::Path>>(config_path: P) -> TrainingResult<Self> {
        let content = std::fs::read_to_string(config_path)?;
        
        // Try YAML first, then JSON
        let config = if let Ok(config) = serde_yaml::from_str::<UnifiedTrainingConfig>(&content) {
            config
        } else {
            serde_json::from_str::<UnifiedTrainingConfig>(&content)?
        };
        
        Ok(Self::with_config(config))
    }

    /// Initialize swarm orchestration if enabled
    #[cfg(feature = "swarm")]
    pub async fn initialize_swarm(&mut self) -> TrainingResult<()> {
        if self.config.enable_swarm {
            let swarm_config = self.config.swarm_config.clone()
                .unwrap_or_default();
            self.swarm = Some(SwarmOrchestrator::new(swarm_config));
            
            if let Some(ref mut swarm) = self.swarm {
                swarm.initialize().await?;
            }
        }
        Ok(())
    }

    /// Run the complete training pipeline
    pub async fn run_training_pipeline<P: AsRef<std::path::Path>>(
        &mut self, 
        data_path: P
    ) -> TrainingResult<UnifiedTrainingResults> {
        log::info!("Starting unified neural training pipeline");
        
        // Load and preprocess data
        let mut dataset = TelecomDataset::from_csv(
            data_path,
            TargetType::QualityScore,
            None
        )?;
        
        // Normalize features if needed
        dataset.normalize_features();
        
        // Split data
        let (train_dataset, val_dataset) = dataset.train_val_split(self.config.validation_split)?;
        
        let start_time = std::time::Instant::now();
        
        // Choose training strategy based on configuration
        let results = if self.config.parallel_training {
            #[cfg(feature = "swarm")]
            if self.config.enable_swarm && self.swarm.is_some() {
                // Use swarm orchestration
                self.train_with_swarm(&train_dataset, &val_dataset).await?
            } else {
                // Use parallel training without swarm
                self.train_parallel(&train_dataset, &val_dataset).await?
            }
            
            #[cfg(not(feature = "swarm"))]
            {
                // Use parallel training without swarm
                self.train_parallel(&train_dataset, &val_dataset).await?
            }
        } else {
            // Sequential training
            self.train_sequential(&train_dataset, &val_dataset).await?
        };
        
        let total_time = start_time.elapsed();
        
        // Evaluate results
        let evaluation_metrics = evaluation::EvaluationMetrics::calculate_comprehensive(
            &results.model_results,
            &val_dataset
        )?;
        
        let model_comparison = evaluation::ModelComparison::analyze(&results.model_results)?;
        
        // Cross-validation (optional)
        let cross_validation = if self.config.max_epochs > 100 {
            Some(evaluation::CrossValidator::k_fold_validation(
                &dataset,
                5, // 5-fold CV
                &self.config
            )?)
        } else {
            None
        };
        
        log::info!("Training pipeline completed successfully in {:?}", total_time);
        
        Ok(UnifiedTrainingResults {
            model_results: results.model_results,
            best_model_name: results.best_model_name,
            best_validation_error: results.best_validation_error,
            total_training_time: total_time,
            evaluation_metrics,
            model_comparison,
            cross_validation,
            #[cfg(feature = "swarm")]
            swarm_results: None, // TODO: Implement swarm results collection
        })
    }

    /// Train models in parallel without swarm
    async fn train_parallel(
        &self,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
    ) -> TrainingResult<training::TrainingResults> {
        let trainer = training::NeuralTrainer::new(training::SimpleTrainingConfig {
            parallel_training: true,
            ..Default::default()
        });
        
        // Create multiple model variants
        let models = self.create_model_variants()?;
        
        let results = trainer.train_multiple_models(
            models,
            train_data,
            Some(val_data)
        )?;
        
        Ok(training::TrainingResults::new(results, ()))
    }

    /// Train models sequentially
    async fn train_sequential(
        &self,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
    ) -> TrainingResult<training::TrainingResults> {
        let trainer = training::NeuralTrainer::new(training::SimpleTrainingConfig {
            parallel_training: false,
            ..Default::default()
        });
        
        let models = self.create_model_variants()?;
        
        let results = trainer.train_multiple_models(
            models,
            train_data,
            Some(val_data)
        )?;
        
        Ok(training::TrainingResults::new(results, ()))
    }

    /// Train using swarm orchestration
    #[cfg(feature = "swarm")]
    async fn train_with_swarm(
        &mut self,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
    ) -> TrainingResult<training::TrainingResults> {
        if let Some(ref mut swarm) = self.swarm {
            swarm.train_multiple_models(train_data, val_data, &self.config).await
        } else {
            // Fallback to parallel training
            self.train_parallel(train_data, val_data).await
        }
    }

    /// Create model variants for training
    fn create_model_variants(&self) -> TrainingResult<Vec<models::NeuralModel>> {
        let mut models = Vec::new();
        
        // Base model with default parameters
        models.push(models::NeuralModel::new(
            "base_model".to_string(),
            models::ModelArchitecture::from_layers(&self.config.architecture),
            models::TrainingParameters {
                learning_rate: self.config.learning_rate,
                max_epochs: self.config.max_epochs,
                target_error: self.config.target_error,
                momentum: 0.9,
                algorithm: self.config.algorithm,
            }
        )?);
        
        // Variant with different learning rate
        models.push(models::NeuralModel::new(
            "high_lr_model".to_string(),
            models::ModelArchitecture::from_layers(&self.config.architecture),
            models::TrainingParameters {
                learning_rate: self.config.learning_rate * 2.0,
                max_epochs: self.config.max_epochs,
                target_error: self.config.target_error,
                momentum: 0.9,
                algorithm: self.config.algorithm,
            }
        )?);
        
        // Variant with different architecture
        let mut deep_arch = self.config.architecture.clone();
        if deep_arch.len() > 2 {
            deep_arch.insert(deep_arch.len() - 1, deep_arch[deep_arch.len() - 2] / 2);
        }
        
        models.push(models::NeuralModel::new(
            "deep_model".to_string(),
            models::ModelArchitecture::from_layers(&deep_arch),
            models::TrainingParameters {
                learning_rate: self.config.learning_rate * 0.8,
                max_epochs: self.config.max_epochs,
                target_error: self.config.target_error,
                momentum: 0.95,
                algorithm: self.config.algorithm,
            }
        )?);
        
        Ok(models)
    }

    /// Get current training metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Save configuration to file
    pub fn save_config<P: AsRef<std::path::Path>>(&self, path: P) -> TrainingResult<()> {
        let content = serde_yaml::to_string(&self.config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for UnifiedTrainingSystem {
    fn default() -> Self {
        Self::new()
    }
}

// WebAssembly setup
#[cfg(feature = "wasm")]
mod wasm_setup {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn main() {
        console_error_panic_hook::set_once();
        
        #[cfg(feature = "wee_alloc")]
        {
            #[global_allocator]
            static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_config_default() {
        let config = UnifiedTrainingConfig::default();
        assert_eq!(config.architecture, vec![21, 16, 12, 8, 1]);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.max_epochs, 1000);
        assert!(config.parallel_training);
    }

    #[test]
    fn test_training_system_creation() {
        let system = UnifiedTrainingSystem::new();
        assert_eq!(system.config.architecture.len(), 5);
    }

    #[test]
    fn test_model_variants_creation() {
        let system = UnifiedTrainingSystem::new();
        let models = system.create_model_variants().unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].name, "base_model");
        assert_eq!(models[1].name, "high_lr_model");
        assert_eq!(models[2].name, "deep_model");
    }

    #[cfg(feature = "swarm")]
    #[tokio::test]
    async fn test_swarm_initialization() {
        let mut config = UnifiedTrainingConfig::default();
        config.enable_swarm = true;
        
        let mut system = UnifiedTrainingSystem::with_config(config);
        assert!(system.initialize_swarm().await.is_ok());
        assert!(system.swarm.is_some());
    }
}