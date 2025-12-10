//! # ran-training: Unified Neural Network Training for Telecom Data
//!
//! A comprehensive crate for training neural networks on telecom performance data.
//! Combines core training functionality with swarm orchestration, advanced evaluation metrics, and multi-deployment support.

#![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "wasm")]
extern crate alloc;
#[cfg(feature = "wasm")]
use alloc::{string::String, vec::Vec};

// Placeholder for ruv_fann types for testing
pub mod ruv_fann_mock {
    #[derive(Debug, Clone, Copy)]
    pub enum ActivationFunction {
        Sigmoid,
        Linear,
    }
    
    #[derive(Debug, Clone, Copy)]
    pub enum TrainingAlgorithm {
        RProp,
        Backprop,
        QuickProp,
    }
    
    pub struct Network<T> {
        _phantom: std::marker::PhantomData<T>,
    }
    
    impl<T> Network<T> {
        pub fn run(&self, _input: &[T]) -> Vec<T> 
        where T: Default + Clone {
            vec![T::default()]
        }
    }
    
    pub struct NetworkBuilder<T> {
        _phantom: std::marker::PhantomData<T>,
    }
    
    impl<T> NetworkBuilder<T> {
        pub fn new() -> Self {
            Self { _phantom: std::marker::PhantomData }
        }
        
        pub fn input_layer(self, _size: usize) -> Self { self }
        pub fn hidden_layer_with_activation(self, _size: usize, _activation: ActivationFunction, _bias: f32) -> Self { self }
        pub fn output_layer_with_activation(self, _size: usize, _activation: ActivationFunction, _bias: f32) -> Self { self }
        
        pub fn build(self) -> Network<T> {
            Network { _phantom: std::marker::PhantomData }
        }
    }
}

// Re-export core types (using mock for testing)
pub use ruv_fann_mock::{ActivationFunction, Network, NetworkBuilder, TrainingAlgorithm};

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
pub use training::{TelecomTrainer, TrainingConfig, TrainingMetrics, PredictionResult};
pub use evaluation::{EvaluationMetrics, ModelComparison, CrossValidator};
pub use config::SystemConfig;

#[cfg(feature = "swarm")]
pub use swarm::{SwarmOrchestrator, SwarmConfig, SwarmAgent, AgentType};

/// Main unified training system that combines all functionality
pub struct UnifiedTrainingSystem {
    /// Configuration
    pub config: training::TrainingConfig,
    /// Training metrics
    pub metrics: TrainingMetrics,
}

impl UnifiedTrainingSystem {
    /// Create a new training system with default configuration
    pub fn new() -> Self {
        Self {
            config: training::TrainingConfig::default(),
            metrics: TrainingMetrics::default(),
        }
    }
}

impl Default for UnifiedTrainingSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_system_creation() {
        let system = UnifiedTrainingSystem::new();
        assert_eq!(system.config.architecture.len(), 5);
    }
}