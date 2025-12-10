//! GPU-Accelerated Neural Network Training System for Telecom Data
//! 
//! This crate provides a comprehensive neural network training system
//! with Mac GPU acceleration (Metal Performance Shaders), state-of-the-art 
//! optimization algorithms, real-time monitoring, hyperparameter tuning,
//! advanced backpropagation, and evaluation metrics specifically designed 
//! for telecom data analysis and prediction.

pub mod fann_compat;
pub mod data;
pub mod models;
pub mod training;
pub mod evaluation;
pub mod evaluation_dashboard;
// pub mod swarm;
// pub mod config;
// pub mod demo;

// Advanced training modules
pub mod backprop;
pub mod optimizers;
pub mod hyperparameter_tuning;

// Enhanced evaluation modules 
pub mod benchmarks;
// pub mod statistical_analysis;

// GPU-accelerated training modules
pub mod gpu_training;
pub mod gpu_data_loader;
pub mod training_monitor;

pub use data::*;
pub use models::*;
pub use training::{NeuralTrainer, ModelTrainingResult, HyperparameterTuner, ValidationResults, EpochResult, TrainingResults, SimpleTrainingConfig};
pub use models::TrainingParameters;
pub use evaluation::*;
pub use evaluation_dashboard::*;
pub use benchmarks::*;
// pub use swarm::*;
// pub use config::TrainingConfig;

// Export advanced training components
pub use backprop::{
    AdvancedBackpropagationTrainer, TrainingConfig as AdvancedTrainingConfig,
    TrainingMetrics, OptimizerType, LearningRateScheduler, RegularizationType,
    WeightInitialization, EarlyStoppingConfig, ForwardResult, BackwardResult
};
pub use optimizers::{
    Optimizer, OptimizerConfig, OptimizerFactory, SGD, Adam, AdaGrad, 
    RMSprop, AdaDelta, AdamW, RAdam, LAMB
};
pub use hyperparameter_tuning::{
    HyperparameterTuner as AdvancedHyperparameterTuner, SearchStrategy, 
    SearchSpace, SearchSpaceTemplates, TuningResult, HyperparameterConfig,
    ParameterDistribution, ObjectiveFunction, AcquisitionFunction
};
// Additional modules temporarily disabled for compilation
// pub use statistical_analysis::{...};
// pub use benchmarks::{...};

use anyhow::Result;
use std::path::Path;

/// Simplified training system for core functionality
pub struct SimpleNeuralTrainingSystem;

impl SimpleNeuralTrainingSystem {
    /// Create a new training system
    pub fn new() -> Self {
        Self
    }

    /// Basic training example
    pub fn train_basic_model(
        train_data: &TelecomDataset,
        val_data: Option<&TelecomDataset>,
        architecture_name: &str,
    ) -> Result<ModelTrainingResult> {
        let input_size = train_data.features.ncols();
        let output_size = 1;
        
        let architecture = match architecture_name {
            "shallow" => NetworkArchitectures::shallow_network(input_size, output_size),
            "deep" => NetworkArchitectures::deep_network(input_size, output_size),
            "wide" => NetworkArchitectures::wide_network(input_size, output_size),
            _ => NetworkArchitectures::shallow_network(input_size, output_size),
        };
        
        let model = NeuralModel::from_architecture(architecture)?;
        let trainer_config = SimpleTrainingConfig::default();
        let trainer = NeuralTrainer::new(trainer_config);
        
        trainer.train_model(model, train_data, val_data)
    }
}

impl Default for SimpleNeuralTrainingSystem {
    fn default() -> Self {
        Self::new()
    }
}