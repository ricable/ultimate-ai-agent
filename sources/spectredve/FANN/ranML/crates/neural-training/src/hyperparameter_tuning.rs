//! Advanced Hyperparameter Tuning System
//!
//! This module provides comprehensive hyperparameter optimization strategies
//! including grid search, random search, Bayesian optimization, and population-based methods.

use crate::models::{TrainingParameters, NeuralModel};
use crate::optimizers::{OptimizerConfig, OptimizerFactory};
use crate::backprop::{TrainingConfig, TrainingMetrics};
use crate::training::{NeuralTrainer, ModelTrainingResult};
use crate::data::TelecomDataset;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

/// Hyperparameter search strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Exhaustive grid search over parameter space
    GridSearch,
    /// Random sampling from parameter distributions
    RandomSearch { n_trials: usize },
    /// Bayesian optimization using Gaussian processes
    BayesianOptimization { n_trials: usize, acquisition: AcquisitionFunction },
    /// Population-based training
    PopulationBased { population_size: usize, generations: usize },
    /// Multi-fidelity optimization (successive halving)
    SuccessiveHalving { min_budget: usize, max_budget: usize, eta: f32 },
    /// Hyperband algorithm
    Hyperband { max_budget: usize, eta: f32 },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound { kappa: f32 },
    EntropicRisk { beta: f32 },
}

/// Parameter distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterDistribution {
    /// Uniform distribution over continuous range
    Uniform { low: f32, high: f32 },
    /// Log-uniform distribution for exponential scales
    LogUniform { low: f32, high: f32 },
    /// Normal distribution
    Normal { mean: f32, std: f32 },
    /// Discrete choice from values
    Choice { values: Vec<String> },
    /// Integer uniform distribution
    IntUniform { low: i32, high: i32 },
    /// Boolean choice
    Boolean,
}

/// Hyperparameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterDefinition {
    pub name: String,
    pub distribution: ParameterDistribution,
    pub description: String,
}

/// Search space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    pub parameters: HashMap<String, HyperparameterDefinition>,
    pub constraints: Vec<ParameterConstraint>,
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    /// Conditional constraint: if condition then restriction
    Conditional {
        condition: String,
        condition_value: String,
        restricted_param: String,
        allowed_values: Vec<String>,
    },
    /// Mutual exclusion: only one parameter can be active
    MutualExclusion {
        parameters: Vec<String>,
    },
    /// Range constraint: parameter A must be less than parameter B
    Range {
        param_a: String,
        param_b: String,
        relation: RangeRelation,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RangeRelation {
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}

/// Sampled hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub trial_id: String,
    pub parameters: HashMap<String, ParameterValue>,
    pub score: Option<f32>,
    pub training_time: Option<std::time::Duration>,
    pub status: TrialStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f32),
    Int(i32),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrialStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Pruned,
}

/// Hyperparameter tuning result
#[derive(Debug, Clone, Serialize)]
pub struct TuningResult {
    pub best_config: HyperparameterConfig,
    pub best_score: f32,
    pub all_trials: Vec<HyperparameterConfig>,
    pub convergence_history: Vec<f32>,
    pub total_time: std::time::Duration,
    pub n_completed_trials: usize,
    pub optimization_path: Vec<OptimizationStep>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationStep {
    pub trial_id: String,
    pub score: f32,
    pub parameters: HashMap<String, ParameterValue>,
    pub improvement: f32,
    pub timestamp: std::time::SystemTime,
}

/// Advanced hyperparameter tuner
pub struct HyperparameterTuner {
    search_space: SearchSpace,
    strategy: SearchStrategy,
    objective: ObjectiveFunction,
    early_stopping: Option<EarlyStoppingConfig>,
    n_jobs: usize,
}

/// Objective function configuration
#[derive(Debug, Clone)]
pub enum ObjectiveFunction {
    /// Minimize validation loss
    ValidationLoss,
    /// Maximize validation accuracy
    ValidationAccuracy,
    /// Custom weighted combination
    Weighted {
        loss_weight: f32,
        accuracy_weight: f32,
        training_time_weight: f32,
    },
    /// Multi-objective (Pareto optimization)
    MultiObjective {
        objectives: Vec<String>,
        weights: Vec<f32>,
    },
}

/// Early stopping for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_trials: usize,
    pub improvement_threshold: f32,
}

impl HyperparameterTuner {
    /// Create new hyperparameter tuner
    pub fn new(
        search_space: SearchSpace,
        strategy: SearchStrategy,
        objective: ObjectiveFunction,
    ) -> Self {
        Self {
            search_space,
            strategy,
            objective,
            early_stopping: None,
            n_jobs: 1,
        }
    }

    /// Configure early stopping
    pub fn with_early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Set number of parallel jobs
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Run hyperparameter optimization
    pub fn optimize(
        &self,
        model_template: &NeuralModel,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut all_trials = Vec::new();
        let mut convergence_history = Vec::new();
        let mut optimization_path = Vec::new();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_config = None;
        let mut trials_without_improvement = 0;

        match &self.strategy {
            SearchStrategy::GridSearch => {
                let configs = self.generate_grid_search_configs()?;
                for (i, config) in configs.into_iter().enumerate() {
                    let trial_result = self.evaluate_trial(
                        config, 
                        model_template, 
                        train_data, 
                        val_data
                    )?;
                    
                    let score = trial_result.score.unwrap_or(f32::NEG_INFINITY);
                    convergence_history.push(best_score.max(score));
                    
                    if score > best_score {
                        best_score = score;
                        best_config = Some(trial_result.clone());
                        trials_without_improvement = 0;
                        
                        optimization_path.push(OptimizationStep {
                            trial_id: trial_result.trial_id.clone(),
                            score,
                            parameters: trial_result.parameters.clone(),
                            improvement: score - best_score,
                            timestamp: std::time::SystemTime::now(),
                        });
                    } else {
                        trials_without_improvement += 1;
                    }
                    
                    all_trials.push(trial_result);
                    
                    // Check early stopping
                    if let Some(ref early_stop) = self.early_stopping {
                        if i >= early_stop.min_trials && 
                           trials_without_improvement >= early_stop.patience {
                            break;
                        }
                    }
                }
            },
            SearchStrategy::RandomSearch { n_trials } => {
                for i in 0..*n_trials {
                    let config = self.sample_random_config()?;
                    let trial_result = self.evaluate_trial(
                        config, 
                        model_template, 
                        train_data, 
                        val_data
                    )?;
                    
                    let score = trial_result.score.unwrap_or(f32::NEG_INFINITY);
                    convergence_history.push(best_score.max(score));
                    
                    if score > best_score {
                        best_score = score;
                        best_config = Some(trial_result.clone());
                        trials_without_improvement = 0;
                        
                        optimization_path.push(OptimizationStep {
                            trial_id: trial_result.trial_id.clone(),
                            score,
                            parameters: trial_result.parameters.clone(),
                            improvement: score - best_score,
                            timestamp: std::time::SystemTime::now(),
                        });
                    } else {
                        trials_without_improvement += 1;
                    }
                    
                    all_trials.push(trial_result);
                    
                    // Check early stopping
                    if let Some(ref early_stop) = self.early_stopping {
                        if i >= early_stop.min_trials && 
                           trials_without_improvement >= early_stop.patience {
                            break;
                        }
                    }
                }
            },
            SearchStrategy::BayesianOptimization { n_trials, acquisition } => {
                // Initialize with random samples
                let n_initial = 5.min(*n_trials);
                let mut gp_data = Vec::new();
                
                for i in 0..n_initial {
                    let config = self.sample_random_config()?;
                    let trial_result = self.evaluate_trial(
                        config, 
                        model_template, 
                        train_data, 
                        val_data
                    )?;
                    
                    let score = trial_result.score.unwrap_or(f32::NEG_INFINITY);
                    gp_data.push((trial_result.parameters.clone(), score));
                    convergence_history.push(best_score.max(score));
                    
                    if score > best_score {
                        best_score = score;
                        best_config = Some(trial_result.clone());
                        
                        optimization_path.push(OptimizationStep {
                            trial_id: trial_result.trial_id.clone(),
                            score,
                            parameters: trial_result.parameters.clone(),
                            improvement: score - best_score,
                            timestamp: std::time::SystemTime::now(),
                        });
                    }
                    
                    all_trials.push(trial_result);
                }
                
                // Bayesian optimization loop
                for _ in n_initial..*n_trials {
                    let config = self.acquire_next_config(&gp_data, acquisition)?;
                    let trial_result = self.evaluate_trial(
                        config, 
                        model_template, 
                        train_data, 
                        val_data
                    )?;
                    
                    let score = trial_result.score.unwrap_or(f32::NEG_INFINITY);
                    gp_data.push((trial_result.parameters.clone(), score));
                    convergence_history.push(best_score.max(score));
                    
                    if score > best_score {
                        best_score = score;
                        best_config = Some(trial_result.clone());
                        
                        optimization_path.push(OptimizationStep {
                            trial_id: trial_result.trial_id.clone(),
                            score,
                            parameters: trial_result.parameters.clone(),
                            improvement: score - best_score,
                            timestamp: std::time::SystemTime::now(),
                        });
                    }
                    
                    all_trials.push(trial_result);
                }
            },
            SearchStrategy::PopulationBased { population_size, generations } => {
                let mut population = Vec::new();
                
                // Initialize population
                for _ in 0..*population_size {
                    let config = self.sample_random_config()?;
                    let trial_result = self.evaluate_trial(
                        config, 
                        model_template, 
                        train_data, 
                        val_data
                    )?;
                    population.push(trial_result);
                }
                
                // Evolution loop
                for generation in 0..*generations {
                    // Select and mutate
                    let mut new_population = Vec::new();
                    
                    for _ in 0..*population_size {
                        let parent = self.tournament_selection(&population, 3);
                        let mutated_config = self.mutate_config(&parent.parameters)?;
                        let trial_result = self.evaluate_trial(
                            mutated_config, 
                            model_template, 
                            train_data, 
                            val_data
                        )?;
                        new_population.push(trial_result);
                    }
                    
                    population = new_population;
                    
                    // Update best
                    for trial in &population {
                        let score = trial.score.unwrap_or(f32::NEG_INFINITY);
                        if score > best_score {
                            best_score = score;
                            best_config = Some(trial.clone());
                        }
                        all_trials.push(trial.clone());
                    }
                    
                    convergence_history.push(best_score);
                }
            },
            SearchStrategy::SuccessiveHalving { min_budget, max_budget, eta } => {
                let configs = self.generate_random_configs(64)?; // Start with many configs
                let mut current_configs = configs;
                let mut current_budget = *min_budget;
                
                while current_budget <= *max_budget && !current_configs.is_empty() {
                    let mut evaluated_configs = Vec::new();
                    
                    // Evaluate all configs with current budget
                    for config in current_configs {
                        let trial_result = self.evaluate_trial_with_budget(
                            config, 
                            model_template, 
                            train_data, 
                            val_data,
                            current_budget
                        )?;
                        evaluated_configs.push(trial_result);
                    }
                    
                    // Sort by performance and keep top 1/eta
                    evaluated_configs.sort_by(|a, b| {
                        b.score.unwrap_or(f32::NEG_INFINITY)
                            .partial_cmp(&a.score.unwrap_or(f32::NEG_INFINITY))
                            .unwrap()
                    });
                    
                    let n_keep = (evaluated_configs.len() as f32 / eta).ceil() as usize;
                    current_configs = evaluated_configs[..n_keep.min(evaluated_configs.len())]
                        .iter()
                        .map(|trial| trial.parameters.clone())
                        .map(|params| HyperparameterConfig {
                            trial_id: uuid::Uuid::new_v4().to_string(),
                            parameters: params,
                            score: None,
                            training_time: None,
                            status: TrialStatus::Pending,
                        })
                        .collect();
                    
                    // Update results
                    for trial in evaluated_configs {
                        let score = trial.score.unwrap_or(f32::NEG_INFINITY);
                        if score > best_score {
                            best_score = score;
                            best_config = Some(trial.clone());
                        }
                        all_trials.push(trial);
                    }
                    
                    convergence_history.push(best_score);
                    current_budget = (current_budget as f32 * eta) as usize;
                }
            },
            SearchStrategy::Hyperband { max_budget, eta } => {
                let s_max = ((*max_budget as f32).log(*eta).floor()) as usize;
                
                for s in (0..=s_max).rev() {
                    let n = ((s_max + 1) as f32 / (s + 1) as f32).ceil() as usize * eta.powi(s as i32) as usize;
                    let r = *max_budget / eta.powi(s as i32) as usize;
                    
                    let configs = self.generate_random_configs(n)?;
                    let mut current_configs = configs;
                    let mut current_budget = r;
                    
                    for i in 0..=s {
                        let n_i = (n as f32 / eta.powi(i as i32)).floor() as usize;
                        let r_i = r * eta.powi(i as i32) as usize;
                        
                        let mut evaluated_configs = Vec::new();
                        
                        for config in current_configs.iter().take(n_i) {
                            let trial_result = self.evaluate_trial_with_budget(
                                config.clone(), 
                                model_template, 
                                train_data, 
                                val_data,
                                r_i
                            )?;
                            evaluated_configs.push(trial_result);
                        }
                        
                        // Sort and keep top 1/eta
                        evaluated_configs.sort_by(|a, b| {
                            b.score.unwrap_or(f32::NEG_INFINITY)
                                .partial_cmp(&a.score.unwrap_or(f32::NEG_INFINITY))
                                .unwrap()
                        });
                        
                        let n_keep = (n_i as f32 / eta).floor() as usize;
                        current_configs = evaluated_configs[..n_keep.min(evaluated_configs.len())]
                            .iter()
                            .map(|trial| trial.parameters.clone())
                            .map(|params| HyperparameterConfig {
                                trial_id: uuid::Uuid::new_v4().to_string(),
                                parameters: params,
                                score: None,
                                training_time: None,
                                status: TrialStatus::Pending,
                            })
                            .collect();
                        
                        // Update results
                        for trial in evaluated_configs {
                            let score = trial.score.unwrap_or(f32::NEG_INFINITY);
                            if score > best_score {
                                best_score = score;
                                best_config = Some(trial.clone());
                            }
                            all_trials.push(trial);
                        }
                        
                        convergence_history.push(best_score);
                    }
                }
            },
        }

        let total_time = start_time.elapsed();
        
        let n_completed_trials = all_trials.len();
        Ok(TuningResult {
            best_config: best_config.unwrap_or_else(|| HyperparameterConfig {
                trial_id: "no_trials".to_string(),
                parameters: HashMap::new(),
                score: Some(f32::NEG_INFINITY),
                training_time: None,
                status: TrialStatus::Failed("No successful trials".to_string()),
            }),
            best_score,
            all_trials,
            convergence_history,
            total_time,
            n_completed_trials,
            optimization_path,
        })
    }

    /// Evaluate a single hyperparameter configuration
    fn evaluate_trial(
        &self,
        config: HyperparameterConfig,
        model_template: &NeuralModel,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
    ) -> Result<HyperparameterConfig> {
        self.evaluate_trial_with_budget(config, model_template, train_data, val_data, 1000)
    }

    /// Evaluate with specific training budget (epochs)
    fn evaluate_trial_with_budget(
        &self,
        mut config: HyperparameterConfig,
        model_template: &NeuralModel,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
        max_epochs: usize,
    ) -> Result<HyperparameterConfig> {
        let start_time = std::time::Instant::now();
        
        // Convert hyperparameters to training configuration
        let training_config = self.config_to_training_params(&config.parameters, max_epochs)?;
        let model = model_template.clone().with_training_params(training_config.clone());
        
        // Create trainer and train
        let trainer_config = crate::training::SimpleTrainingConfig::default();
        let trainer = NeuralTrainer::new(trainer_config);
        
        let result = trainer.train_model(model, train_data, Some(val_data));
        
        match result {
            Ok(training_result) => {
                let score = self.calculate_objective_score(&training_result);
                config.score = Some(score);
                config.training_time = Some(start_time.elapsed());
                config.status = TrialStatus::Completed;
            },
            Err(e) => {
                config.status = TrialStatus::Failed(e.to_string());
            }
        }
        
        Ok(config)
    }

    /// Convert hyperparameter config to training parameters
    fn config_to_training_params(
        &self,
        params: &HashMap<String, ParameterValue>,
        max_epochs: usize,
    ) -> Result<TrainingParameters> {
        let mut training_params = TrainingParameters::default();
        training_params.max_epochs = max_epochs;
        
        // Extract parameters
        if let Some(ParameterValue::Float(lr)) = params.get("learning_rate") {
            training_params.learning_rate = *lr;
        }
        
        if let Some(ParameterValue::Float(momentum)) = params.get("momentum") {
            training_params.momentum = *momentum;
        }
        
        if let Some(ParameterValue::Float(weight_decay)) = params.get("weight_decay") {
            training_params.weight_decay = *weight_decay;
        }
        
        if let Some(ParameterValue::Int(batch_size)) = params.get("batch_size") {
            training_params.batch_size = Some(*batch_size as usize);
        }
        
        if let Some(ParameterValue::Float(dropout)) = params.get("dropout_rate") {
            training_params.dropout_rate = *dropout;
        }
        
        Ok(training_params)
    }

    /// Calculate objective score from training result
    fn calculate_objective_score(&self, result: &ModelTrainingResult) -> f32 {
        match &self.objective {
            ObjectiveFunction::ValidationLoss => {
                if let Some(ref val_results) = result.validation_results {
                    -val_results.mse // Negative because we want to minimize loss
                } else {
                    -result.final_error
                }
            },
            ObjectiveFunction::ValidationAccuracy => {
                if let Some(ref val_results) = result.validation_results {
                    val_results.r_squared // R-squared as accuracy proxy
                } else {
                    1.0 / (1.0 + result.final_error) // Convert error to accuracy-like metric
                }
            },
            ObjectiveFunction::Weighted { loss_weight, accuracy_weight, training_time_weight } => {
                let loss_score = if let Some(ref val_results) = result.validation_results {
                    -val_results.mse
                } else {
                    -result.final_error
                };
                
                let accuracy_score = if let Some(ref val_results) = result.validation_results {
                    val_results.r_squared
                } else {
                    1.0 / (1.0 + result.final_error)
                };
                
                let time_score = -result.training_time.as_secs_f32(); // Negative to prefer faster training
                
                loss_weight * loss_score + 
                accuracy_weight * accuracy_score + 
                training_time_weight * time_score
            },
            ObjectiveFunction::MultiObjective { objectives: _, weights } => {
                // Simplified multi-objective scoring
                let loss_score = -result.final_error;
                let accuracy_score = 1.0 / (1.0 + result.final_error);
                
                weights.get(0).unwrap_or(&0.5) * loss_score + 
                weights.get(1).unwrap_or(&0.5) * accuracy_score
            },
        }
    }

    /// Generate grid search configurations
    fn generate_grid_search_configs(&self) -> Result<Vec<HyperparameterConfig>> {
        let mut configs = Vec::new();
        let mut param_values = HashMap::new();
        
        // Generate all possible values for each parameter
        for (name, param_def) in &self.search_space.parameters {
            let values = match &param_def.distribution {
                ParameterDistribution::Choice { values } => {
                    values.iter().map(|v| ParameterValue::String(v.clone())).collect()
                },
                ParameterDistribution::Boolean => {
                    vec![ParameterValue::Bool(true), ParameterValue::Bool(false)]
                },
                ParameterDistribution::IntUniform { low, high } => {
                    (*low..=*high).map(ParameterValue::Int).collect()
                },
                ParameterDistribution::Uniform { low, high } => {
                    // Discretize continuous parameters for grid search
                    let n_points = 5;
                    let step = (high - low) / (n_points - 1) as f32;
                    (0..n_points)
                        .map(|i| ParameterValue::Float(low + i as f32 * step))
                        .collect()
                },
                _ => {
                    // For other distributions, sample a few representative values
                    (0..5).map(|_| self.sample_parameter(&param_def.distribution)).collect()
                }
            };
            param_values.insert(name.clone(), values);
        }
        
        // Generate Cartesian product
        self.cartesian_product(&param_values, &mut configs);
        
        Ok(configs)
    }

    /// Generate Cartesian product of parameter values
    fn cartesian_product(
        &self,
        param_values: &HashMap<String, Vec<ParameterValue>>,
        configs: &mut Vec<HyperparameterConfig>,
    ) {
        if param_values.is_empty() {
            return;
        }
        
        let keys: Vec<_> = param_values.keys().collect();
        let mut indices = vec![0; keys.len()];
        
        loop {
            let mut parameters = HashMap::new();
            for (i, key) in keys.iter().enumerate() {
                parameters.insert(
                    (*key).clone(),
                    param_values[*key][indices[i]].clone()
                );
            }
            
            configs.push(HyperparameterConfig {
                trial_id: uuid::Uuid::new_v4().to_string(),
                parameters,
                score: None,
                training_time: None,
                status: TrialStatus::Pending,
            });
            
            // Increment indices
            let mut carry = 1;
            for i in (0..indices.len()).rev() {
                indices[i] += carry;
                if indices[i] < param_values[keys[i]].len() {
                    carry = 0;
                    break;
                } else {
                    indices[i] = 0;
                }
            }
            
            if carry == 1 {
                break;
            }
        }
    }

    /// Sample a random configuration
    fn sample_random_config(&self) -> Result<HyperparameterConfig> {
        let mut parameters = HashMap::new();
        
        for (name, param_def) in &self.search_space.parameters {
            let value = self.sample_parameter(&param_def.distribution);
            parameters.insert(name.clone(), value);
        }
        
        Ok(HyperparameterConfig {
            trial_id: uuid::Uuid::new_v4().to_string(),
            parameters,
            score: None,
            training_time: None,
            status: TrialStatus::Pending,
        })
    }

    /// Sample from parameter distribution
    fn sample_parameter(&self, distribution: &ParameterDistribution) -> ParameterValue {
        let mut rng = rand::thread_rng();
        
        match distribution {
            ParameterDistribution::Uniform { low, high } => {
                ParameterValue::Float(rng.gen_range(*low..*high))
            },
            ParameterDistribution::LogUniform { low, high } => {
                let log_low = low.ln();
                let log_high = high.ln();
                let log_value = rng.gen_range(log_low..log_high);
                ParameterValue::Float(log_value.exp())
            },
            ParameterDistribution::Normal { mean, std } => {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(*mean, *std).unwrap();
                ParameterValue::Float(normal.sample(&mut rng))
            },
            ParameterDistribution::Choice { values } => {
                let idx = rng.gen_range(0..values.len());
                ParameterValue::String(values[idx].clone())
            },
            ParameterDistribution::IntUniform { low, high } => {
                ParameterValue::Int(rng.gen_range(*low..=*high))
            },
            ParameterDistribution::Boolean => {
                ParameterValue::Bool(rng.gen_bool(0.5))
            },
        }
    }

    /// Generate multiple random configurations
    fn generate_random_configs(&self, n: usize) -> Result<Vec<HyperparameterConfig>> {
        (0..n).map(|_| self.sample_random_config()).collect()
    }

    /// Acquire next configuration for Bayesian optimization
    fn acquire_next_config(
        &self,
        _gp_data: &[(HashMap<String, ParameterValue>, f32)],
        _acquisition: &AcquisitionFunction,
    ) -> Result<HyperparameterConfig> {
        // Simplified implementation - in practice would use Gaussian process
        self.sample_random_config()
    }

    /// Tournament selection for population-based optimization
    fn tournament_selection<'a>(&self, population: &'a [HyperparameterConfig], k: usize) -> &'a HyperparameterConfig {
        let mut rng = rand::thread_rng();
        let mut best = &population[rng.gen_range(0..population.len())];
        
        for _ in 1..k {
            let candidate = &population[rng.gen_range(0..population.len())];
            if candidate.score.unwrap_or(f32::NEG_INFINITY) > best.score.unwrap_or(f32::NEG_INFINITY) {
                best = candidate;
            }
        }
        
        best
    }

    /// Mutate configuration for population-based optimization
    fn mutate_config(&self, parameters: &HashMap<String, ParameterValue>) -> Result<HyperparameterConfig> {
        let mut mutated = parameters.clone();
        let mut rng = rand::thread_rng();
        
        // Mutate 20% of parameters
        let n_mutate = (parameters.len() as f32 * 0.2).ceil() as usize;
        let param_names: Vec<_> = parameters.keys().collect();
        
        for _ in 0..n_mutate {
            let param_name = param_names[rng.gen_range(0..param_names.len())];
            if let Some(param_def) = self.search_space.parameters.get(*param_name) {
                let new_value = self.sample_parameter(&param_def.distribution);
                mutated.insert(param_name.clone(), new_value);
            }
        }
        
        Ok(HyperparameterConfig {
            trial_id: uuid::Uuid::new_v4().to_string(),
            parameters: mutated,
            score: None,
            training_time: None,
            status: TrialStatus::Pending,
        })
    }
}

/// Predefined search spaces for common optimization scenarios
pub struct SearchSpaceTemplates;

impl SearchSpaceTemplates {
    /// Basic search space for neural network training
    pub fn basic_neural_network() -> SearchSpace {
        let mut parameters = HashMap::new();
        
        parameters.insert("learning_rate".to_string(), HyperparameterDefinition {
            name: "learning_rate".to_string(),
            distribution: ParameterDistribution::LogUniform { low: 1e-5, high: 1e-1 },
            description: "Learning rate for optimization".to_string(),
        });
        
        parameters.insert("batch_size".to_string(), HyperparameterDefinition {
            name: "batch_size".to_string(),
            distribution: ParameterDistribution::Choice { 
                values: vec!["16".to_string(), "32".to_string(), "64".to_string(), "128".to_string()] 
            },
            description: "Batch size for training".to_string(),
        });
        
        parameters.insert("momentum".to_string(), HyperparameterDefinition {
            name: "momentum".to_string(),
            distribution: ParameterDistribution::Uniform { low: 0.0, high: 0.999 },
            description: "Momentum for SGD optimizer".to_string(),
        });
        
        parameters.insert("weight_decay".to_string(), HyperparameterDefinition {
            name: "weight_decay".to_string(),
            distribution: ParameterDistribution::LogUniform { low: 1e-6, high: 1e-2 },
            description: "Weight decay for regularization".to_string(),
        });
        
        parameters.insert("dropout_rate".to_string(), HyperparameterDefinition {
            name: "dropout_rate".to_string(),
            distribution: ParameterDistribution::Uniform { low: 0.0, high: 0.5 },
            description: "Dropout rate for regularization".to_string(),
        });
        
        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Extended search space including optimizer choice
    pub fn extended_neural_network() -> SearchSpace {
        let mut basic = Self::basic_neural_network();
        
        basic.parameters.insert("optimizer".to_string(), HyperparameterDefinition {
            name: "optimizer".to_string(),
            distribution: ParameterDistribution::Choice { 
                values: vec!["SGD".to_string(), "Adam".to_string(), "AdamW".to_string(), "RMSprop".to_string()] 
            },
            description: "Optimizer algorithm".to_string(),
        });
        
        basic.parameters.insert("beta1".to_string(), HyperparameterDefinition {
            name: "beta1".to_string(),
            distribution: ParameterDistribution::Uniform { low: 0.8, high: 0.999 },
            description: "Beta1 parameter for Adam optimizer".to_string(),
        });
        
        basic.parameters.insert("beta2".to_string(), HyperparameterDefinition {
            name: "beta2".to_string(),
            distribution: ParameterDistribution::Uniform { low: 0.9, high: 0.9999 },
            description: "Beta2 parameter for Adam optimizer".to_string(),
        });
        
        // Add constraint: beta parameters only relevant for Adam optimizers
        basic.constraints.push(ParameterConstraint::Conditional {
            condition: "optimizer".to_string(),
            condition_value: "SGD".to_string(),
            restricted_param: "beta1".to_string(),
            allowed_values: vec!["0.9".to_string()], // Default value when not applicable
        });
        
        basic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space_creation() {
        let search_space = SearchSpaceTemplates::basic_neural_network();
        assert!(search_space.parameters.contains_key("learning_rate"));
        assert!(search_space.parameters.contains_key("batch_size"));
        assert!(search_space.parameters.contains_key("momentum"));
    }

    #[test]
    fn test_parameter_sampling() {
        let tuner = HyperparameterTuner::new(
            SearchSpaceTemplates::basic_neural_network(),
            SearchStrategy::RandomSearch { n_trials: 10 },
            ObjectiveFunction::ValidationLoss,
        );
        
        let uniform_dist = ParameterDistribution::Uniform { low: 0.0, high: 1.0 };
        let sample = tuner.sample_parameter(&uniform_dist);
        
        if let ParameterValue::Float(value) = sample {
            assert!(value >= 0.0 && value <= 1.0);
        } else {
            panic!("Expected float value");
        }
    }

    #[test]
    fn test_random_config_generation() {
        let tuner = HyperparameterTuner::new(
            SearchSpaceTemplates::basic_neural_network(),
            SearchStrategy::RandomSearch { n_trials: 10 },
            ObjectiveFunction::ValidationLoss,
        );
        
        let config = tuner.sample_random_config().unwrap();
        assert!(!config.parameters.is_empty());
        assert!(config.parameters.contains_key("learning_rate"));
    }

    #[test]
    fn test_grid_search_generation() {
        let mut simple_space = SearchSpace {
            parameters: HashMap::new(),
            constraints: Vec::new(),
        };
        
        simple_space.parameters.insert("param1".to_string(), HyperparameterDefinition {
            name: "param1".to_string(),
            distribution: ParameterDistribution::Choice { 
                values: vec!["a".to_string(), "b".to_string()] 
            },
            description: "Test parameter".to_string(),
        });
        
        simple_space.parameters.insert("param2".to_string(), HyperparameterDefinition {
            name: "param2".to_string(),
            distribution: ParameterDistribution::Boolean,
            description: "Boolean parameter".to_string(),
        });
        
        let tuner = HyperparameterTuner::new(
            simple_space,
            SearchStrategy::GridSearch,
            ObjectiveFunction::ValidationLoss,
        );
        
        let configs = tuner.generate_grid_search_configs().unwrap();
        assert_eq!(configs.len(), 4); // 2 * 2 combinations
    }
}