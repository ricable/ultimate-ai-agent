//! Training utilities for RAN neural networks

use std::time::Duration;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{NeuralError, NeuralResult, ModelType};
use ruv_fann::{Network, TrainingData, TrainingAlgorithm, TrainingState};

/// RAN-specific neural network trainer
#[derive(Debug)]
pub struct RanTrainer {
    /// Model type being trained
    pub model_type: ModelType,
    /// Training configuration
    pub config: TrainingConfig,
    /// Training metrics
    pub metrics: TrainingMetrics,
}

impl RanTrainer {
    /// Create a new RAN trainer
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            config: TrainingConfig::for_model_type(model_type),
            metrics: TrainingMetrics::default(),
        }
    }

    /// Create trainer with custom configuration
    pub fn with_config(model_type: ModelType, config: TrainingConfig) -> Self {
        Self {
            model_type,
            config,
            metrics: TrainingMetrics::default(),
        }
    }

    /// Train a neural network
    pub fn train(&mut self, network: &mut Network<f64>, training_data: &TrainingData<f64>) -> NeuralResult<TrainingMetrics> {
        let start_time = std::time::Instant::now();
        
        // Validate training data
        self.validate_training_data(training_data)?;
        
        // Configure network for training
        network.set_learning_rate(self.config.learning_rate);
        
        // Training loop
        let mut best_error = f64::INFINITY;
        let mut epochs_without_improvement = 0;
        
        for epoch in 0..self.config.max_epochs {
            // Train one epoch
            let epoch_error = network.train_epoch(training_data)
                .map_err(|e| NeuralError::TrainingError(e.to_string()))?;
            
            // Update metrics
            self.metrics.epoch_errors.push(epoch_error);
            self.metrics.current_epoch = epoch + 1;
            
            // Check for improvement
            if epoch_error < best_error {
                best_error = epoch_error;
                epochs_without_improvement = 0;
                self.metrics.best_error = epoch_error;
                self.metrics.best_epoch = epoch + 1;
            } else {
                epochs_without_improvement += 1;
            }
            
            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if epochs_without_improvement >= patience {
                    tracing::info!("Early stopping at epoch {} (no improvement for {} epochs)", 
                                  epoch + 1, patience);
                    break;
                }
            }
            
            // Target error reached
            if epoch_error <= self.config.target_error {
                tracing::info!("Target error {} reached at epoch {}", 
                              self.config.target_error, epoch + 1);
                break;
            }
            
            // Progress callback
            if epoch % 100 == 0 {
                tracing::debug!("Epoch {}: error = {:.6}", epoch + 1, epoch_error);
            }
        }
        
        let training_time = start_time.elapsed();
        self.metrics.training_time = training_time;
        self.metrics.final_error = self.metrics.epoch_errors.last().copied().unwrap_or(f64::INFINITY);
        self.metrics.converged = self.metrics.final_error <= self.config.target_error;
        
        tracing::info!("Training completed: {} epochs, final error: {:.6}, time: {:?}", 
                      self.metrics.current_epoch, self.metrics.final_error, training_time);
        
        Ok(self.metrics.clone())
    }

    /// Validate training data
    fn validate_training_data(&self, training_data: &TrainingData<f64>) -> NeuralResult<()> {
        if training_data.num_data() == 0 {
            return Err(NeuralError::TrainingError("No training data provided".to_string()));
        }
        
        if training_data.num_input() == 0 {
            return Err(NeuralError::TrainingError("No input features".to_string()));
        }
        
        if training_data.num_output() == 0 {
            return Err(NeuralError::TrainingError("No output targets".to_string()));
        }
        
        // Check for minimum data requirements
        let min_samples = self.config.min_training_samples.unwrap_or(10);
        if training_data.num_data() < min_samples {
            return Err(NeuralError::TrainingError(format!(
                "Insufficient training data: {} samples, need at least {}",
                training_data.num_data(), min_samples
            )));
        }
        
        Ok(())
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Target error for convergence
    pub target_error: f64,
    /// Early stopping patience (epochs without improvement)
    pub early_stopping_patience: Option<usize>,
    /// Minimum number of training samples
    pub min_training_samples: Option<usize>,
    /// Validation split ratio
    pub validation_split: f64,
    /// Training algorithm
    pub algorithm: TrainingAlgorithm,
    /// Batch size for batch training
    pub batch_size: Option<usize>,
    /// Enable data shuffling
    pub shuffle_data: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl TrainingConfig {
    /// Create training configuration for specific model type
    pub fn for_model_type(model_type: ModelType) -> Self {
        match model_type {
            ModelType::ThroughputPredictor => Self {
                max_epochs: 1000,
                learning_rate: 0.001,
                target_error: 0.01,
                early_stopping_patience: Some(50),
                min_training_samples: Some(100),
                validation_split: 0.2,
                algorithm: TrainingAlgorithm::RProp,
                batch_size: Some(64),
                shuffle_data: true,
                random_seed: None,
            },
            ModelType::CellStateClassifier => Self {
                max_epochs: 2000,
                learning_rate: 0.01,
                target_error: 0.05,
                early_stopping_patience: Some(100),
                min_training_samples: Some(200),
                validation_split: 0.2,
                algorithm: TrainingAlgorithm::BatchBackprop,
                batch_size: Some(32),
                shuffle_data: true,
                random_seed: None,
            },
            ModelType::HandoverDecision => Self {
                max_epochs: 1500,
                learning_rate: 0.005,
                target_error: 0.02,
                early_stopping_patience: Some(75),
                min_training_samples: Some(150),
                validation_split: 0.2,
                algorithm: TrainingAlgorithm::RProp,
                batch_size: Some(128),
                shuffle_data: true,
                random_seed: None,
            },
            _ => Self::default(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            learning_rate: 0.01,
            target_error: 0.01,
            early_stopping_patience: Some(50),
            min_training_samples: Some(50),
            validation_split: 0.2,
            algorithm: TrainingAlgorithm::RProp,
            batch_size: Some(64),
            shuffle_data: true,
            random_seed: None,
        }
    }
}

/// Training metrics and statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training session ID
    pub session_id: Uuid,
    /// Current epoch
    pub current_epoch: usize,
    /// Best epoch (lowest error)
    pub best_epoch: usize,
    /// Error for each epoch
    pub epoch_errors: Vec<f64>,
    /// Best error achieved
    pub best_error: f64,
    /// Final error
    pub final_error: f64,
    /// Whether training converged
    pub converged: bool,
    /// Total training time
    pub training_time: Duration,
    /// Training start time
    pub start_time: Option<DateTime<Utc>>,
    /// Training end time
    pub end_time: Option<DateTime<Utc>>,
    /// Validation error (if validation used)
    pub validation_error: Option<f64>,
    /// Training data size
    pub training_samples: usize,
    /// Validation data size
    pub validation_samples: usize,
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            start_time: Some(Utc::now()),
            ..Default::default()
        }
    }

    /// Get convergence rate (error reduction per epoch)
    pub fn convergence_rate(&self) -> f64 {
        if self.epoch_errors.len() < 2 {
            return 0.0;
        }
        
        let initial_error = self.epoch_errors[0];
        let final_error = self.final_error;
        
        if initial_error > 0.0 && self.current_epoch > 0 {
            (initial_error - final_error) / (initial_error * self.current_epoch as f64)
        } else {
            0.0
        }
    }

    /// Get training efficiency (error reduction per second)
    pub fn training_efficiency(&self) -> f64 {
        if self.training_time.as_secs_f64() > 0.0 && !self.epoch_errors.is_empty() {
            let error_reduction = self.epoch_errors[0] - self.final_error;
            error_reduction / self.training_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Check if training was successful
    pub fn is_successful(&self) -> bool {
        self.converged || (self.current_epoch > 10 && self.final_error < 0.1)
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Epochs: {}, Final Error: {:.6}, Best Error: {:.6}, Time: {:?}, Converged: {}",
            self.current_epoch,
            self.final_error,
            self.best_error,
            self.training_time,
            self.converged
        )
    }
}

/// Training data builder for RAN applications
#[derive(Debug)]
pub struct RanTrainingDataBuilder {
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f64>>,
}

impl RanTrainingDataBuilder {
    /// Create new training data builder
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add training sample
    pub fn add_sample(&mut self, input: Vec<f64>, output: Vec<f64>) -> NeuralResult<()> {
        if !self.inputs.is_empty() && input.len() != self.inputs[0].len() {
            return Err(NeuralError::TrainingError(
                "Input dimension mismatch".to_string()
            ));
        }
        
        if !self.outputs.is_empty() && output.len() != self.outputs[0].len() {
            return Err(NeuralError::TrainingError(
                "Output dimension mismatch".to_string()
            ));
        }
        
        self.inputs.push(input);
        self.outputs.push(output);
        Ok(())
    }

    /// Build training data
    pub fn build(&self) -> NeuralResult<TrainingData<f64>> {
        if self.inputs.is_empty() {
            return Err(NeuralError::TrainingError("No training data".to_string()));
        }
        
        let num_inputs = self.inputs[0].len();
        let num_outputs = self.outputs[0].len();
        let num_data = self.inputs.len();
        
        let mut training_data = TrainingData::new(num_data, num_inputs, num_outputs);
        
        for (i, (input, output)) in self.inputs.iter().zip(self.outputs.iter()).enumerate() {
            training_data.set_data(i, input, output)
                .map_err(|e| NeuralError::TrainingError(e.to_string()))?;
        }
        
        Ok(training_data)
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

impl Default for RanTrainingDataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::for_model_type(ModelType::ThroughputPredictor);
        assert_eq!(config.max_epochs, 1000);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.algorithm, TrainingAlgorithm::RProp);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        metrics.epoch_errors = vec![1.0, 0.5, 0.2, 0.1];
        metrics.final_error = 0.1;
        metrics.current_epoch = 4;
        
        assert!(metrics.convergence_rate() > 0.0);
        assert!(!metrics.summary().is_empty());
    }

    #[test]
    fn test_training_data_builder() {
        let mut builder = RanTrainingDataBuilder::new();
        
        builder.add_sample(vec![1.0, 2.0], vec![0.5]).unwrap();
        builder.add_sample(vec![2.0, 3.0], vec![0.8]).unwrap();
        
        assert_eq!(builder.len(), 2);
        
        let training_data = builder.build().unwrap();
        assert_eq!(training_data.num_data(), 2);
        assert_eq!(training_data.num_input(), 2);
        assert_eq!(training_data.num_output(), 1);
    }

    #[test]
    fn test_ran_trainer() {
        let trainer = RanTrainer::new(ModelType::ThroughputPredictor);
        assert_eq!(trainer.model_type, ModelType::ThroughputPredictor);
        assert_eq!(trainer.config.max_epochs, 1000);
    }
}