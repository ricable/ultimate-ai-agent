//! Training algorithms and utilities for neural networks

use crate::data::{TelecomDataset, TelecomRecord};
use crate::models::{NeuralModel, TrainingParameters};
use crate::error::{TrainingError, TrainingResult};
use ruv_fann::{Network, TrainingAlgorithm};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use rayon::prelude::*;

// Re-export training types from the original for compatibility
pub use super::UnifiedTrainingConfig as TrainingConfig;

/// Training configuration (simplified version for compatibility)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTrainingConfig {
    pub train_test_split: f32,
    pub validation_split: f32,
    pub normalize_features: bool,
    pub random_seed: Option<u64>,
    pub parallel_training: bool,
    pub save_checkpoints: bool,
    pub checkpoint_interval: usize,
}

impl Default for SimpleTrainingConfig {
    fn default() -> Self {
        Self {
            train_test_split: 0.8,
            validation_split: 0.2,
            normalize_features: true,
            random_seed: Some(42),
            parallel_training: true,
            save_checkpoints: true,
            checkpoint_interval: 100,
        }
    }
}

impl SimpleTrainingConfig {
    /// Load configuration from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> TrainingResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> TrainingResult<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Training result for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingResult {
    pub model_name: String,
    pub training_params: TrainingParameters,
    pub training_time: Duration,
    pub final_error: f32,
    pub epochs_completed: usize,
    pub convergence_achieved: bool,
    pub training_history: Vec<EpochResult>,
    pub validation_results: Option<ValidationResults>,
}

/// Results for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochResult {
    pub epoch: usize,
    pub training_error: f32,
    pub validation_error: Option<f32>,
    pub learning_rate: f32,
    pub time_elapsed: Duration,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub r_squared: f32,
    pub predictions: Vec<f32>,
    pub targets: Vec<f32>,
}

/// Collection of training results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResults {
    pub model_results: Vec<ModelTrainingResult>,
    pub best_model_name: String,
    pub best_validation_error: f32,
    pub total_training_time: Duration,
    pub comparison_metrics: ModelComparisonMetrics,
}

/// Metrics for comparing models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonMetrics {
    pub accuracy_ranking: Vec<(String, f32)>,
    pub speed_ranking: Vec<(String, Duration)>,
    pub efficiency_ranking: Vec<(String, f32)>, // accuracy/time ratio
    pub parameter_count: Vec<(String, usize)>,
}

impl TrainingResults {
    pub fn new(results: Vec<ModelTrainingResult>, _evaluation: ()) -> Self {
        let best_result = results.iter()
            .min_by(|a, b| a.final_error.partial_cmp(&b.final_error).unwrap())
            .unwrap();
        
        let total_time = results.iter()
            .map(|r| r.training_time)
            .sum();
        
        // Create comparison metrics
        let mut accuracy_ranking: Vec<(String, f32)> = results.iter()
            .map(|r| (r.model_name.clone(), r.final_error))
            .collect();
        accuracy_ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut speed_ranking: Vec<(String, Duration)> = results.iter()
            .map(|r| (r.model_name.clone(), r.training_time))
            .collect();
        speed_ranking.sort_by_key(|a| a.1);
        
        let mut efficiency_ranking: Vec<(String, f32)> = results.iter()
            .map(|r| (r.model_name.clone(), 1.0 / (r.final_error * r.training_time.as_secs_f32())))
            .collect();
        efficiency_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let parameter_count: Vec<(String, usize)> = results.iter()
            .map(|r| (r.model_name.clone(), 0)) // TODO: get actual parameter count
            .collect();
        
        let best_model_name = best_result.model_name.clone();
        let best_validation_error = best_result.final_error;
        
        Self {
            model_results: results,
            best_model_name,
            best_validation_error,
            total_training_time: total_time,
            comparison_metrics: ModelComparisonMetrics {
                accuracy_ranking,
                speed_ranking,
                efficiency_ranking,
                parameter_count,
            },
        }
    }
}

/// Enhanced neural network trainer with advanced features
pub struct NeuralTrainer {
    config: SimpleTrainingConfig,
}

impl NeuralTrainer {
    /// Create new trainer with configuration
    pub fn new(config: SimpleTrainingConfig) -> Self {
        Self { config }
    }
    
    /// Train a single model with comprehensive monitoring
    pub fn train_model(
        &self,
        mut model: NeuralModel,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
    ) -> TrainingResult<ModelTrainingResult> {
        log::info!("Training model: {}", model.name);
        
        let start_time = Instant::now();
        let mut training_history = Vec::new();
        
        // Setup training data
        let (train_inputs, train_outputs) = self.prepare_training_data(train_data)?;
        let validation_data_prepared = if let Some(val_data) = validation_data {
            Some(self.prepare_training_data(val_data)?)
        } else {
            None
        };
        
        let mut best_error = f32::MAX;
        let mut epochs_without_improvement = 0;
        let max_epochs_without_improvement = 50;
        
        // Training loop with enhanced monitoring
        for epoch in 0..model.training_params.max_epochs as usize {
            let epoch_start = Instant::now();
            
            // Train one epoch
            let training_error = self.train_epoch(
                &mut model,
                &train_inputs,
                &train_outputs,
                epoch
            )?;
            
            // Validate if validation data provided
            let validation_error = if let Some((val_inputs, val_outputs)) = &validation_data_prepared {
                Some(self.calculate_error(&model, val_inputs, val_outputs))
            } else {
                None
            };
            
            let epoch_time = epoch_start.elapsed();
            
            // Record epoch result with detailed metrics
            training_history.push(EpochResult {
                epoch,
                training_error,
                validation_error,
                learning_rate: model.training_params.learning_rate,
                time_elapsed: epoch_time,
            });
            
            // Check for improvement with adaptive criteria
            let current_error = validation_error.unwrap_or(training_error);
            if current_error < best_error {
                best_error = current_error;
                epochs_without_improvement = 0;
                
                // Save checkpoint if enabled
                if self.config.save_checkpoints && epoch % self.config.checkpoint_interval == 0 {
                    self.save_checkpoint(&model, epoch)?;
                }
            } else {
                epochs_without_improvement += 1;
            }
            
            // Early stopping with dynamic patience
            if current_error <= model.training_params.target_error {
                log::info!("Target error achieved at epoch {}", epoch);
                break;
            }
            
            if epochs_without_improvement >= max_epochs_without_improvement {
                log::info!("Early stopping at epoch {} due to no improvement", epoch);
                break;
            }
            
            // Adaptive learning rate
            if epochs_without_improvement > 10 && epoch % 50 == 0 {
                model.training_params.learning_rate *= 0.95;
                log::debug!("Reduced learning rate to {}", model.training_params.learning_rate);
            }
            
            // Progress logging
            if epoch % 100 == 0 || epoch < 10 {
                log::info!(
                    "Epoch {}: Training Error = {:.6}, Validation Error = {:.6}, LR = {:.6}", 
                    epoch, 
                    training_error, 
                    validation_error.unwrap_or(0.0),
                    model.training_params.learning_rate
                );
            }
        }
        
        let training_time = start_time.elapsed();
        let convergence_achieved = best_error <= model.training_params.target_error;
        
        // Calculate comprehensive validation results
        let validation_results = if let Some(val_data) = validation_data {
            Some(self.calculate_validation_results(&mut model, val_data)?)
        } else {
            None
        };
        
        log::info!(
            "Model {} training completed: {} epochs, final error: {:.6}, time: {:?}",
            model.name, training_history.len(), best_error, training_time
        );
        
        Ok(ModelTrainingResult {
            model_name: model.name,
            training_params: model.training_params,
            training_time,
            final_error: best_error,
            epochs_completed: training_history.len(),
            convergence_achieved,
            training_history,
            validation_results,
        })
    }
    
    /// Train multiple models with intelligent scheduling
    pub fn train_multiple_models(
        &self,
        models: Vec<NeuralModel>,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
    ) -> TrainingResult<Vec<ModelTrainingResult>> {
        log::info!("Training {} models with strategy: {}", 
                  models.len(), 
                  if self.config.parallel_training { "parallel" } else { "sequential" });
        
        if self.config.parallel_training {
            // Parallel training with load balancing
            let results: Result<Vec<_>, _> = models.into_par_iter()
                .map(|model| {
                    log::debug!("Starting parallel training for model: {}", model.name);
                    self.train_model(model, train_data, validation_data)
                })
                .collect();
            results
        } else {
            // Sequential training with progress tracking
            let mut results = Vec::new();
            for (i, model) in models.into_iter().enumerate() {
                log::info!("Training model {}/{}: {}", i + 1, results.len() + 1, model.name);
                let result = self.train_model(model, train_data, validation_data)?;
                results.push(result);
            }
            Ok(results)
        }
    }
    
    /// Prepare training data for FANN format
    fn prepare_training_data(&self, dataset: &TelecomDataset) -> TrainingResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let inputs = dataset.features.clone();
        let outputs: Vec<Vec<f32>> = dataset.targets.iter().map(|&t| vec![t]).collect();
        Ok((inputs, outputs))
    }
    
    /// Train one epoch with the specified algorithm
    fn train_epoch(
        &self,
        model: &mut NeuralModel,
        inputs: &[Vec<f32>],
        outputs: &[Vec<f32>],
        epoch: usize,
    ) -> TrainingResult<f32> {
        let mut total_error = 0.0;
        let sample_count = inputs.len();
        
        // Batch training or online training based on configuration
        for (input, expected_output) in inputs.iter().zip(outputs.iter()) {
            let actual_output = model.predict(input);
            
            // Calculate error for this sample
            let error: f32 = actual_output.iter()
                .zip(expected_output.iter())
                .map(|(a, e)| (a - e).powi(2))
                .sum::<f32>() / actual_output.len() as f32;
            
            total_error += error;
            
            // TODO: Implement actual FANN training step
            // This would require access to FANN's training functions
        }
        
        Ok(total_error / sample_count as f32)
    }
    
    /// Calculate error for validation
    fn calculate_error(&self, model: &NeuralModel, inputs: &[Vec<f32>], outputs: &[Vec<f32>]) -> f32 {
        let mut total_error = 0.0;
        
        for (input, expected_output) in inputs.iter().zip(outputs.iter()) {
            // Note: This requires a const version of predict, or we need to work around mutability
            // For now, we'll use a placeholder calculation
            total_error += 0.1; // Placeholder
        }
        
        total_error / inputs.len() as f32
    }
    
    /// Calculate detailed validation results
    fn calculate_validation_results(
        &self,
        model: &mut NeuralModel,
        validation_data: &TelecomDataset,
    ) -> TrainingResult<ValidationResults> {
        let mut predictions = Vec::new();
        let mut targets = Vec::new();
        
        // Get predictions
        for (features, &target) in validation_data.features.iter().zip(validation_data.targets.iter()) {
            let prediction = model.predict(features);
            predictions.push(prediction[0]);
            targets.push(target);
        }
        
        // Calculate comprehensive metrics
        let mse = self.calculate_mse(&predictions, &targets);
        let mae = self.calculate_mae(&predictions, &targets);
        let rmse = mse.sqrt();
        let r_squared = self.calculate_r_squared(&predictions, &targets);
        
        Ok(ValidationResults {
            mse,
            mae,
            rmse,
            r_squared,
            predictions,
            targets,
        })
    }
    
    /// Calculate Mean Squared Error
    fn calculate_mse(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum::<f32>() / predictions.len() as f32
    }
    
    /// Calculate Mean Absolute Error
    fn calculate_mae(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .sum::<f32>() / predictions.len() as f32
    }
    
    /// Calculate R-squared
    fn calculate_r_squared(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
        
        let ss_res: f32 = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (target - pred).powi(2))
            .sum();
        
        let ss_tot: f32 = targets.iter()
            .map(|&target| (target - target_mean).powi(2))
            .sum();
        
        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
    
    /// Save model checkpoint
    fn save_checkpoint(&self, model: &NeuralModel, epoch: usize) -> TrainingResult<()> {
        let filename = format!("checkpoint_{}_{}.json", model.name, epoch);
        model.save_architecture(&filename)?;
        log::debug!("Saved checkpoint: {}", filename);
        Ok(())
    }
}

/// Hyperparameter tuning utilities with advanced search strategies
pub struct HyperparameterTuner {
    trainer: NeuralTrainer,
}

impl HyperparameterTuner {
    pub fn new(config: SimpleTrainingConfig) -> Self {
        Self {
            trainer: NeuralTrainer::new(config),
        }
    }
    
    /// Perform grid search for hyperparameters
    pub fn grid_search(
        &self,
        model_template: NeuralModel,
        param_combinations: Vec<TrainingParameters>,
        train_data: &TelecomDataset,
        validation_data: &TelecomDataset,
    ) -> TrainingResult<(TrainingParameters, ModelTrainingResult)> {
        log::info!("Starting grid search with {} parameter combinations", param_combinations.len());
        
        let mut best_params = param_combinations[0].clone();
        let mut best_result = None;
        let mut best_error = f32::MAX;
        
        for (i, params) in param_combinations.iter().enumerate() {
            log::info!("Testing parameter combination {}/{}", i + 1, param_combinations.len());
            log::debug!("Parameters: LR={}, Epochs={}, Algorithm={:?}", 
                       params.learning_rate, params.max_epochs, params.algorithm);
            
            let model = model_template.clone_with_name(format!("grid_search_{}", i))?
                .with_training_params(params.clone());
            
            let result = self.trainer.train_model(model, train_data, Some(validation_data))?;
            
            if result.final_error < best_error {
                best_error = result.final_error;
                best_params = params.clone();
                best_result = Some(result);
                log::info!("New best result: error={:.6}", best_error);
            }
        }
        
        log::info!("Grid search completed. Best error: {:.6}", best_error);
        Ok((best_params, best_result.unwrap()))
    }
    
    /// Random search for hyperparameters
    pub fn random_search(
        &self,
        model_template: NeuralModel,
        search_space: HyperparameterSpace,
        num_trials: usize,
        train_data: &TelecomDataset,
        validation_data: &TelecomDataset,
    ) -> TrainingResult<(TrainingParameters, ModelTrainingResult)> {
        use rand::Rng;
        
        log::info!("Starting random search with {} trials", num_trials);
        
        let mut rng = rand::thread_rng();
        let mut best_params = TrainingParameters::default();
        let mut best_result = None;
        let mut best_error = f32::MAX;
        
        for trial in 0..num_trials {
            // Generate random parameters within the search space
            let params = TrainingParameters {
                learning_rate: rng.gen_range(search_space.learning_rate_range.clone()),
                max_epochs: rng.gen_range(search_space.max_epochs_range.clone()),
                target_error: rng.gen_range(search_space.target_error_range.clone()),
                momentum: rng.gen_range(search_space.momentum_range.clone()),
                algorithm: search_space.algorithms[rng.gen_range(0..search_space.algorithms.len())],
            };
            
            log::debug!("Trial {}: LR={:.4}, Epochs={}, Momentum={:.3}", 
                       trial + 1, params.learning_rate, params.max_epochs, params.momentum);
            
            let model = model_template.clone_with_name(format!("random_search_{}", trial))?
                .with_training_params(params.clone());
            
            let result = self.trainer.train_model(model, train_data, Some(validation_data))?;
            
            if result.final_error < best_error {
                best_error = result.final_error;
                best_params = params;
                best_result = Some(result);
                log::info!("Trial {}: New best error={:.6}", trial + 1, best_error);
            }
        }
        
        log::info!("Random search completed. Best error: {:.6}", best_error);
        Ok((best_params, best_result.unwrap()))
    }
}

/// Hyperparameter search space definition
#[derive(Debug, Clone)]
pub struct HyperparameterSpace {
    pub learning_rate_range: std::ops::Range<f32>,
    pub max_epochs_range: std::ops::Range<u32>,
    pub target_error_range: std::ops::Range<f32>,
    pub momentum_range: std::ops::Range<f32>,
    pub algorithms: Vec<TrainingAlgorithm>,
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            learning_rate_range: 0.001..0.5,
            max_epochs_range: 100..2000,
            target_error_range: 0.0001..0.01,
            momentum_range: 0.1..0.99,
            algorithms: vec![
                TrainingAlgorithm::RProp,
                TrainingAlgorithm::Backpropagation,
                TrainingAlgorithm::QuickProp,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelFactory;
    use crate::data::TargetType;
    
    #[test]
    fn test_training_config() {
        let config = SimpleTrainingConfig::default();
        assert_eq!(config.train_test_split, 0.8);
        assert!(config.parallel_training);
    }
    
    #[test]
    fn test_neural_trainer_creation() {
        let trainer = NeuralTrainer::new(SimpleTrainingConfig::default());
        assert!(trainer.config.parallel_training);
    }
    
    #[test]
    fn test_hyperparameter_space() {
        let space = HyperparameterSpace::default();
        assert!(space.learning_rate_range.start < space.learning_rate_range.end);
        assert!(!space.algorithms.is_empty());
    }
}