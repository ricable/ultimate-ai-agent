//! Training algorithms and utilities for neural networks

use crate::data::{TelecomDataset, DataSplit};
use crate::models::{NeuralModel, TrainingParameters};
use crate::fann_compat::{TrainingData, TrainingAlgorithm, IncrementalBackprop};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use ndarray::Array1;

/// Training configuration (simplified)
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
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read configuration file")?;
        let config: Self = serde_json::from_str(&content)
            .context("Failed to parse configuration")?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize configuration")?;
        std::fs::write(path, content)
            .context("Failed to write configuration file")?;
        Ok(())
    }
}

/// Training result for a single model
#[derive(Debug, Clone, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
pub struct EpochResult {
    pub epoch: usize,
    pub training_error: f32,
    pub validation_error: Option<f32>,
    pub learning_rate: f32,
    pub time_elapsed: Duration,
}

/// Validation results
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResults {
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub r_squared: f32,
    pub predictions: Vec<f32>,
    pub targets: Vec<f32>,
}

/// Collection of training results
#[derive(Debug, Clone, Serialize)]
pub struct TrainingResults {
    pub model_results: Vec<ModelTrainingResult>,
    pub best_model_name: String,
    pub best_validation_error: f32,
    pub total_training_time: Duration,
    pub comparison_metrics: ModelComparisonMetrics,
}

/// Metrics for comparing models
#[derive(Debug, Clone, Serialize)]
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

/// Neural network trainer
pub struct NeuralTrainer {
    config: SimpleTrainingConfig,
}

impl NeuralTrainer {
    /// Create new trainer with configuration
    pub fn new(config: SimpleTrainingConfig) -> Self {
        Self { config }
    }
    
    /// Train a single model
    pub fn train_model(
        &self,
        mut model: NeuralModel,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
    ) -> Result<ModelTrainingResult> {
        log::info!("Training model: {}", model.name);
        
        let start_time = Instant::now();
        let mut training_history = Vec::new();
        
        // Convert data to FANN format
        let fann_data = self.dataset_to_fann_data(train_data)?;
        
        // Create training algorithm
        let mut trainer = IncrementalBackprop::new(model.training_params.learning_rate)
            .with_momentum(model.training_params.momentum);
        
        let mut best_error = f32::MAX;
        let mut epochs_without_improvement = 0;
        let max_epochs_without_improvement = 50;
        
        for epoch in 0..model.training_params.max_epochs {
            let epoch_start = Instant::now();
            
            // Train one epoch
            let training_error = trainer.train_epoch(&mut model.network, &fann_data)
                .context("Failed to train epoch")?;
            
            // Validate if validation data provided
            let validation_error = if let Some(val_data) = validation_data {
                let val_fann_data = self.dataset_to_fann_data(val_data)?;
                Some(trainer.calculate_error(&model.network, &val_fann_data))
            } else {
                None
            };
            
            let epoch_time = epoch_start.elapsed();
            
            // Record epoch result
            training_history.push(EpochResult {
                epoch,
                training_error,
                validation_error,
                learning_rate: model.training_params.learning_rate,
                time_elapsed: epoch_time,
            });
            
            // Check for improvement
            let current_error = validation_error.unwrap_or(training_error);
            if current_error < best_error {
                best_error = current_error;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }
            
            // Early stopping
            if current_error <= model.training_params.target_error {
                log::info!("Target error achieved at epoch {}", epoch);
                break;
            }
            
            if epochs_without_improvement >= max_epochs_without_improvement {
                log::info!("Early stopping at epoch {} due to no improvement", epoch);
                break;
            }
            
            // Log progress
            if epoch % 100 == 0 {
                log::info!("Epoch {}: Training Error = {:.6}, Validation Error = {:.6}", 
                          epoch, training_error, validation_error.unwrap_or(0.0));
            }
        }
        
        let training_time = start_time.elapsed();
        let convergence_achieved = best_error <= model.training_params.target_error;
        
        // Calculate validation results if validation data provided
        let validation_results = if let Some(val_data) = validation_data {
            Some(self.calculate_validation_results(&mut model, val_data)?)
        } else {
            None
        };
        
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
    
    /// Train multiple models in parallel
    pub fn train_multiple_models(
        &self,
        models: Vec<NeuralModel>,
        train_data: &TelecomDataset,
        validation_data: Option<&TelecomDataset>,
    ) -> Result<Vec<ModelTrainingResult>> {
        log::info!("Training {} models", models.len());
        
        if self.config.parallel_training {
            // Parallel training using rayon
            let results: Result<Vec<_>, _> = models.into_par_iter()
                .map(|model| self.train_model(model, train_data, validation_data))
                .collect();
            results
        } else {
            // Sequential training
            let mut results = Vec::new();
            for model in models {
                let result = self.train_model(model, train_data, validation_data)?;
                results.push(result);
            }
            Ok(results)
        }
    }
    
    /// Convert TelecomDataset to FANN TrainingData
    fn dataset_to_fann_data(&self, dataset: &TelecomDataset) -> Result<TrainingData> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        
        for i in 0..dataset.features.nrows() {
            let input: Vec<f32> = dataset.features.row(i).to_vec();
            let output = vec![dataset.targets[i]];
            
            inputs.push(input);
            outputs.push(output);
        }
        
        Ok(TrainingData { inputs, outputs })
    }
    
    /// Calculate detailed validation results
    fn calculate_validation_results(
        &self,
        model: &mut NeuralModel,
        validation_data: &TelecomDataset,
    ) -> Result<ValidationResults> {
        let mut predictions = Vec::new();
        let mut targets = Vec::new();
        
        // Get predictions
        for i in 0..validation_data.features.nrows() {
            let input: Vec<f32> = validation_data.features.row(i).to_vec();
            let prediction = model.predict(&input);
            let target = validation_data.targets[i];
            
            predictions.push(prediction[0]);
            targets.push(target);
        }
        
        // Calculate metrics
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
}

/// Hyperparameter tuning utilities
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
    ) -> Result<(TrainingParameters, ModelTrainingResult)> {
        log::info!("Starting grid search with {} parameter combinations", param_combinations.len());
        
        let mut best_params = param_combinations[0].clone();
        let mut best_result = None;
        let mut best_error = f32::MAX;
        
        for (i, params) in param_combinations.iter().enumerate() {
            log::info!("Testing parameter combination {}/{}", i + 1, param_combinations.len());
            
            let model = model_template.clone().with_training_params(params.clone());
            let result = self.trainer.train_model(model, train_data, Some(validation_data))?;
            
            if result.final_error < best_error {
                best_error = result.final_error;
                best_params = params.clone();
                best_result = Some(result);
            }
        }
        
        Ok((best_params, best_result.unwrap()))
    }
}