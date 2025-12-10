//! Advanced Training Engine Example
//!
//! This example demonstrates the comprehensive training system with:
//! - Advanced backpropagation with multiple optimizers
//! - Hyperparameter tuning with different search strategies
//! - Configurable training parameters and early stopping
//! - Performance metrics and visualization

use neural_training::{
    // Data handling
    TelecomDataLoader, TelecomDataset,
    
    // Models and architectures
    NetworkArchitectures, NeuralModel, TrainingParameters,
    
    // Advanced training components
    AdvancedBackpropagationTrainer, AdvancedTrainingConfig,
    OptimizerType, LearningRateScheduler, RegularizationType,
    WeightInitialization, EarlyStoppingConfig,
    
    // Optimizers
    OptimizerFactory, OptimizerConfig,
    
    // Hyperparameter tuning
    AdvancedHyperparameterTuner, SearchStrategy, SearchSpaceTemplates,
    ObjectiveFunction, TuningResult,
    
    // Basic training
    NeuralTrainer, SimpleTrainingConfig,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "advanced_training")]
#[command(about = "Advanced neural network training with optimization and tuning")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train with advanced backpropagation and optimizer comparison
    TrainOptimizers {
        /// Path to training data
        #[arg(long)]
        data_path: PathBuf,
        /// Maximum epochs for training
        #[arg(long, default_value = "1000")]
        max_epochs: usize,
        /// Compare all optimizers
        #[arg(long)]
        compare_all: bool,
    },
    
    /// Hyperparameter tuning with different search strategies
    TuneHyperparameters {
        /// Path to training data
        #[arg(long)]
        data_path: PathBuf,
        /// Search strategy: grid, random, bayesian, population, successive_halving
        #[arg(long, default_value = "random")]
        strategy: String,
        /// Number of trials for search
        #[arg(long, default_value = "50")]
        n_trials: usize,
    },
    
    /// Demonstrate different weight initialization strategies
    WeightInitialization {
        /// Path to training data
        #[arg(long)]
        data_path: PathBuf,
        /// Compare all initialization methods
        #[arg(long)]
        compare_all: bool,
    },
    
    /// Full advanced training pipeline
    FullPipeline {
        /// Path to training data
        #[arg(long)]
        data_path: PathBuf,
        /// Enable hyperparameter tuning
        #[arg(long)]
        tune_hyperparams: bool,
        /// Number of tuning trials
        #[arg(long, default_value = "20")]
        tuning_trials: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::TrainOptimizers { data_path, max_epochs, compare_all } => {
            train_with_optimizers(data_path, max_epochs, compare_all).await
        },
        Commands::TuneHyperparameters { data_path, strategy, n_trials } => {
            tune_hyperparameters(data_path, strategy, n_trials).await
        },
        Commands::WeightInitialization { data_path, compare_all } => {
            test_weight_initialization(data_path, compare_all).await
        },
        Commands::FullPipeline { data_path, tune_hyperparams, tuning_trials } => {
            run_full_pipeline(data_path, tune_hyperparams, tuning_trials).await
        },
    }
}

/// Train with different optimizers and compare performance
async fn train_with_optimizers(data_path: PathBuf, max_epochs: usize, compare_all: bool) -> Result<()> {
    println!("🚀 Advanced Optimizer Comparison Training");
    println!("=========================================");
    
    // Load and prepare data
    let dataset = TelecomDataLoader::load(&data_path)?;
    let split = dataset.split_train_test(0.8)?;
    let val_split = split.train.split_train_test(0.8)?;
    
    println!("📊 Dataset Information:");
    println!("  - Training samples: {}", val_split.train.features.nrows());
    println!("  - Validation samples: {}", val_split.test.features.nrows());
    println!("  - Test samples: {}", split.test.features.nrows());
    println!("  - Features: {}", val_split.train.features.ncols());
    
    // Create base model architecture
    let input_size = val_split.train.features.ncols();
    let output_size = 1;
    let architecture = NetworkArchitectures::deep_network(input_size, output_size);
    let base_model = NeuralModel::from_architecture(architecture)?;
    
    let optimizers = if compare_all {
        vec![
            ("SGD", OptimizerType::SGD { momentum: 0.9 }),
            ("Adam", OptimizerType::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }),
            ("AdaGrad", OptimizerType::AdaGrad { epsilon: 1e-8 }),
            ("RMSprop", OptimizerType::RMSprop { beta: 0.9, epsilon: 1e-8 }),
            ("AdaDelta", OptimizerType::AdaDelta { rho: 0.9, epsilon: 1e-6 }),
            ("AdamW", OptimizerType::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }), // AdamW will be handled separately
        ]
    } else {
        vec![
            ("SGD", OptimizerType::SGD { momentum: 0.9 }),
            ("Adam", OptimizerType::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }),
            ("RMSprop", OptimizerType::RMSprop { beta: 0.9, epsilon: 1e-8 }),
        ]
    };
    
    let mut results = Vec::new();
    
    for (name, optimizer_type) in optimizers {
        println!("\n🏋️ Training with {} optimizer...", name);
        
        let training_config = AdvancedTrainingConfig {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            batch_size: 32,
            optimizer: optimizer_type,
            lr_scheduler: Some(LearningRateScheduler::ReduceOnPlateau {
                patience: 10,
                factor: 0.5,
                threshold: 1e-4,
            }),
            gradient_clipping: Some(1.0),
            early_stopping: Some(EarlyStoppingConfig {
                patience: 20,
                min_delta: 1e-4,
                restore_best_weights: true,
            }),
            regularization: RegularizationType::L2 { lambda: 0.01 },
            initialization: WeightInitialization::Xavier,
        };
        
        let mut model = base_model.clone();
        let mut trainer = AdvancedBackpropagationTrainer::new(training_config);
        
        // Initialize weights
        trainer.initialize_weights(&mut model.network);
        
        let start_time = Instant::now();
        let mut best_loss = f32::INFINITY;
        let mut epoch_losses = Vec::new();
        
        // Convert to FANN data format
        let train_data = dataset_to_fann_data(&val_split.train)?;
        
        // Training loop
        for epoch in 0..max_epochs {
            match trainer.train_epoch(&mut model.network, &train_data) {
                Ok(loss) => {
                    epoch_losses.push(loss);
                    
                    if loss < best_loss {
                        best_loss = loss;
                    }
                    
                    if epoch % 100 == 0 {
                        let metrics = trainer.get_metrics();
                        println!("  Epoch {}: Loss = {:.6}, LR = {:.6}, Grad Norm = {:.6}", 
                                epoch, loss, metrics.learning_rate, metrics.gradient_norm);
                    }
                    
                    if trainer.should_stop_early() {
                        println!("  Early stopping at epoch {}", epoch);
                        trainer.restore_best_weights(&mut model.network);
                        break;
                    }
                },
                Err(e) => {
                    println!("  Training failed: {}", e);
                    break;
                }
            }
        }
        
        let training_time = start_time.elapsed();
        let final_metrics = trainer.get_metrics();
        
        results.push((name, best_loss, training_time, final_metrics.clone()));
        
        println!("  ✅ Training completed:");
        println!("    - Best Loss: {:.6}", best_loss);
        println!("    - Training Time: {:.2}s", training_time.as_secs_f32());
        println!("    - Final LR: {:.6}", final_metrics.learning_rate);
        println!("    - Epochs: {}", final_metrics.epoch);
    }
    
    // Print comparison results
    println!("\n📈 Optimizer Comparison Results");
    println!("==============================");
    
    // Sort by best loss
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    println!("Rank | Optimizer | Best Loss | Time (s) | Epochs | Final LR");
    println!("-----|-----------|-----------|----------|--------|----------");
    for (i, (name, loss, time, metrics)) in results.iter().enumerate() {
        println!("{:4} | {:9} | {:9.6} | {:8.2} | {:6} | {:8.6}", 
                i + 1, name, loss, time.as_secs_f32(), metrics.epoch, metrics.learning_rate);
    }
    
    let best_optimizer = &results[0];
    println!("\n🏆 Best Optimizer: {} with loss {:.6}", best_optimizer.0, best_optimizer.1);
    
    Ok(())
}

/// Demonstrate hyperparameter tuning with different strategies
async fn tune_hyperparameters(data_path: PathBuf, strategy: String, n_trials: usize) -> Result<()> {
    println!("🔍 Hyperparameter Tuning with {} Strategy", strategy.to_uppercase());
    println!("===========================================");
    
    // Load and prepare data
    let dataset = TelecomDataLoader::load(&data_path)?;
    let split = dataset.split_train_test(0.8)?;
    let val_split = split.train.split_train_test(0.8)?;
    
    // Create base model
    let input_size = val_split.train.features.ncols();
    let output_size = 1;
    let architecture = NetworkArchitectures::deep_network(input_size, output_size);
    let base_model = NeuralModel::from_architecture(architecture)?;
    
    // Define search strategy
    let search_strategy = match strategy.to_lowercase().as_str() {
        "grid" => SearchStrategy::GridSearch,
        "random" => SearchStrategy::RandomSearch { n_trials },
        "bayesian" => SearchStrategy::BayesianOptimization { 
            n_trials, 
            acquisition: neural_training::AcquisitionFunction::ExpectedImprovement 
        },
        "population" => SearchStrategy::PopulationBased { 
            population_size: (n_trials / 10).max(10), 
            generations: 10 
        },
        "successive_halving" => SearchStrategy::SuccessiveHalving {
            min_budget: 50,
            max_budget: 500,
            eta: 3.0,
        },
        "hyperband" => SearchStrategy::Hyperband {
            max_budget: 1000,
            eta: 3.0,
        },
        _ => {
            println!("Unknown strategy '{}', using random search", strategy);
            SearchStrategy::RandomSearch { n_trials }
        }
    };
    
    // Create search space
    let search_space = SearchSpaceTemplates::extended_neural_network();
    
    // Create tuner
    let tuner = AdvancedHyperparameterTuner::new(
        search_space,
        search_strategy,
        ObjectiveFunction::ValidationLoss,
    );
    
    println!("📊 Starting hyperparameter optimization...");
    println!("  - Strategy: {}", strategy);
    println!("  - Max trials: {}", n_trials);
    println!("  - Training samples: {}", val_split.train.features.nrows());
    println!("  - Validation samples: {}", val_split.test.features.nrows());
    
    let start_time = Instant::now();
    
    // Run optimization
    let tuning_result = tuner.optimize(&base_model, &val_split.train, &val_split.test)?;
    
    let total_time = start_time.elapsed();
    
    // Print results
    println!("\n🎯 Hyperparameter Tuning Results");
    println!("===============================");
    println!("  - Total Time: {:.2}s", total_time.as_secs_f32());
    println!("  - Completed Trials: {}", tuning_result.n_completed_trials);
    println!("  - Best Score: {:.6}", tuning_result.best_score);
    
    println!("\n🏆 Best Configuration:");
    for (param, value) in &tuning_result.best_config.parameters {
        match value {
            neural_training::ParameterValue::Float(v) => println!("  - {}: {:.6}", param, v),
            neural_training::ParameterValue::Int(v) => println!("  - {}: {}", param, v),
            neural_training::ParameterValue::String(v) => println!("  - {}: {}", param, v),
            neural_training::ParameterValue::Bool(v) => println!("  - {}: {}", param, v),
        }
    }
    
    // Show optimization path
    if !tuning_result.optimization_path.is_empty() {
        println!("\n📈 Optimization Progress (Top 5 Improvements):");
        for (i, step) in tuning_result.optimization_path.iter().take(5).enumerate() {
            println!("  {}. Trial {}: Score = {:.6} (Improvement: {:.6})", 
                    i + 1, step.trial_id, step.score, step.improvement);
        }
    }
    
    // Show convergence
    if tuning_result.convergence_history.len() > 1 {
        let initial_score = tuning_result.convergence_history[0];
        let final_score = *tuning_result.convergence_history.last().unwrap();
        let total_improvement = final_score - initial_score;
        
        println!("\n📊 Convergence Analysis:");
        println!("  - Initial Score: {:.6}", initial_score);
        println!("  - Final Score: {:.6}", final_score);
        println!("  - Total Improvement: {:.6}", total_improvement);
        println!("  - Improvement Rate: {:.2}%", (total_improvement / initial_score.abs()) * 100.0);
    }
    
    Ok(())
}

/// Test different weight initialization strategies
async fn test_weight_initialization(data_path: PathBuf, compare_all: bool) -> Result<()> {
    println!("⚖️ Weight Initialization Strategy Comparison");
    println!("==========================================");
    
    // Load and prepare data
    let dataset = TelecomDataLoader::load(&data_path)?;
    let split = dataset.split_train_test(0.8)?;
    let val_split = split.train.split_train_test(0.8)?;
    
    // Create base model architecture
    let input_size = val_split.train.features.ncols();
    let output_size = 1;
    let architecture = NetworkArchitectures::deep_network(input_size, output_size);
    
    let initialization_methods = if compare_all {
        vec![
            ("Random", WeightInitialization::Random),
            ("Xavier", WeightInitialization::Xavier),
            ("He", WeightInitialization::He),
            ("LeCun", WeightInitialization::LeCun),
        ]
    } else {
        vec![
            ("Xavier", WeightInitialization::Xavier),
            ("He", WeightInitialization::He),
        ]
    };
    
    let mut results = Vec::new();
    
    for (name, init_method) in initialization_methods {
        println!("\n🎲 Testing {} initialization...", name);
        
        let training_config = AdvancedTrainingConfig {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            batch_size: 32,
            optimizer: OptimizerType::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
            lr_scheduler: Some(LearningRateScheduler::ReduceOnPlateau {
                patience: 10,
                factor: 0.5,
                threshold: 1e-4,
            }),
            gradient_clipping: Some(1.0),
            early_stopping: Some(EarlyStoppingConfig {
                patience: 15,
                min_delta: 1e-4,
                restore_best_weights: true,
            }),
            regularization: RegularizationType::L2 { lambda: 0.01 },
            initialization: init_method,
        };
        
        let mut model = NeuralModel::from_architecture(architecture.clone())?;
        let mut trainer = AdvancedBackpropagationTrainer::new(training_config);
        
        // Initialize weights
        trainer.initialize_weights(&mut model.network);
        
        // Quick training to test convergence
        let train_data = dataset_to_fann_data(&val_split.train)?;
        let mut losses = Vec::new();
        let start_time = Instant::now();
        
        for epoch in 0..200 {
            match trainer.train_epoch(&mut model.network, &train_data) {
                Ok(loss) => {
                    losses.push(loss);
                    
                    if epoch % 50 == 0 {
                        println!("  Epoch {}: Loss = {:.6}", epoch, loss);
                    }
                    
                    if trainer.should_stop_early() {
                        break;
                    }
                },
                Err(e) => {
                    println!("  Training failed: {}", e);
                    break;
                }
            }
        }
        
        let training_time = start_time.elapsed();
        let final_loss = losses.last().copied().unwrap_or(f32::INFINITY);
        let convergence_speed = if losses.len() > 10 {
            let initial = losses[0];
            let after_10 = losses[9];
            (initial - after_10) / initial
        } else {
            0.0
        };
        
        results.push((name, final_loss, convergence_speed, training_time, losses.len()));
        
        println!("  ✅ Initialization test completed:");
        println!("    - Final Loss: {:.6}", final_loss);
        println!("    - Convergence Speed: {:.2}%", convergence_speed * 100.0);
        println!("    - Epochs: {}", losses.len());
        println!("    - Time: {:.2}s", training_time.as_secs_f32());
    }
    
    // Print comparison results
    println!("\n📈 Weight Initialization Comparison");
    println!("==================================");
    
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    println!("Rank | Method | Final Loss | Conv. Speed | Epochs | Time (s)");
    println!("-----|--------|------------|-------------|--------|----------");
    for (i, (name, loss, conv_speed, time, epochs)) in results.iter().enumerate() {
        println!("{:4} | {:6} | {:10.6} | {:10.2}% | {:6} | {:8.2}", 
                i + 1, name, loss, conv_speed * 100.0, epochs, time.as_secs_f32());
    }
    
    let best_init = &results[0];
    println!("\n🏆 Best Initialization: {} with final loss {:.6}", best_init.0, best_init.1);
    
    Ok(())
}

/// Run the complete advanced training pipeline
async fn run_full_pipeline(data_path: PathBuf, tune_hyperparams: bool, tuning_trials: usize) -> Result<()> {
    println!("🚀 Full Advanced Training Pipeline");
    println!("=================================");
    
    // Load and prepare data
    println!("📂 Loading dataset...");
    let dataset = TelecomDataLoader::load(&data_path)?;
    let split = dataset.split_train_test(0.8)?;
    let val_split = split.train.split_train_test(0.8)?;
    
    println!("📊 Dataset prepared:");
    println!("  - Training: {} samples", val_split.train.features.nrows());
    println!("  - Validation: {} samples", val_split.test.features.nrows());
    println!("  - Test: {} samples", split.test.features.nrows());
    
    // Create base model
    let input_size = val_split.train.features.ncols();
    let output_size = 1;
    let architecture = NetworkArchitectures::deep_network(input_size, output_size);
    let mut model = NeuralModel::from_architecture(architecture)?;
    
    // Step 1: Find best initialization method
    println!("\n🎲 Step 1: Testing weight initialization methods...");
    let init_result = find_best_initialization(&val_split.train, &val_split.test, &model).await?;
    println!("✅ Best initialization: {:?}", init_result);
    
    // Step 2: Hyperparameter tuning (if enabled)
    let best_config = if tune_hyperparams {
        println!("\n🔍 Step 2: Hyperparameter tuning...");
        let search_space = SearchSpaceTemplates::extended_neural_network();
        let tuner = AdvancedHyperparameterTuner::new(
            search_space,
            SearchStrategy::RandomSearch { n_trials: tuning_trials },
            ObjectiveFunction::ValidationLoss,
        );
        
        let tuning_result = tuner.optimize(&model, &val_split.train, &val_split.test)?;
        println!("✅ Best hyperparameters found with score: {:.6}", tuning_result.best_score);
        
        // Convert to training config
        Some(tuning_result.best_config)
    } else {
        None
    };
    
    // Step 3: Final training with best configuration
    println!("\n🏋️ Step 3: Final training with optimized configuration...");
    
    let training_config = if let Some(config) = best_config {
        // Convert hyperparameter config to training config
        convert_hyperparam_to_training_config(&config, init_result)
    } else {
        AdvancedTrainingConfig {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            batch_size: 32,
            optimizer: OptimizerType::Adam { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
            lr_scheduler: Some(LearningRateScheduler::ReduceOnPlateau {
                patience: 15,
                factor: 0.5,
                threshold: 1e-4,
            }),
            gradient_clipping: Some(1.0),
            early_stopping: Some(EarlyStoppingConfig {
                patience: 30,
                min_delta: 1e-4,
                restore_best_weights: true,
            }),
            regularization: RegularizationType::L2 { lambda: 0.01 },
            initialization: init_result,
        }
    };
    
    let mut trainer = AdvancedBackpropagationTrainer::new(training_config);
    trainer.initialize_weights(&mut model.network);
    
    // Training loop
    let train_data = dataset_to_fann_data(&val_split.train)?;
    let start_time = Instant::now();
    let mut epoch_losses = Vec::new();
    
    for epoch in 0..1000 {
        match trainer.train_epoch(&mut model.network, &train_data) {
            Ok(loss) => {
                epoch_losses.push(loss);
                
                if epoch % 100 == 0 {
                    let metrics = trainer.get_metrics();
                    println!("  Epoch {}: Loss = {:.6}, LR = {:.6}", 
                            epoch, loss, metrics.learning_rate);
                }
                
                if trainer.should_stop_early() {
                    println!("  Early stopping at epoch {}", epoch);
                    trainer.restore_best_weights(&mut model.network);
                    break;
                }
            },
            Err(e) => {
                println!("  Training failed: {}", e);
                break;
            }
        }
    }
    
    let training_time = start_time.elapsed();
    let final_metrics = trainer.get_metrics();
    
    // Step 4: Final evaluation
    println!("\n📊 Step 4: Final evaluation...");
    
    // Evaluate on test set
    let test_data = dataset_to_fann_data(&split.test)?;
    let test_loss = evaluate_model(&mut model.network, &test_data);
    
    println!("\n🎉 Training Pipeline Completed!");
    println!("==============================");
    println!("📈 Training Results:");
    println!("  - Final Training Loss: {:.6}", final_metrics.train_loss);
    println!("  - Test Loss: {:.6}", test_loss);
    println!("  - Training Time: {:.2}s", training_time.as_secs_f32());
    println!("  - Epochs Completed: {}", final_metrics.epoch);
    println!("  - Final Learning Rate: {:.6}", final_metrics.learning_rate);
    
    println!("\n💡 Model Configuration:");
    println!("  - Architecture: Deep Network");
    println!("  - Initialization: {:?}", init_result);
    println!("  - Optimizer: Adam");
    println!("  - Regularization: L2");
    println!("  - Early Stopping: Enabled");
    
    if tune_hyperparams {
        println!("\n🔍 Hyperparameter Tuning:");
        println!("  - Enabled with {} trials", tuning_trials);
        println!("  - Strategy: Random Search");
    }
    
    Ok(())
}

// Helper functions

/// Find the best weight initialization method
async fn find_best_initialization(
    train_data: &TelecomDataset,
    val_data: &TelecomDataset,
    model_template: &NeuralModel,
) -> Result<WeightInitialization> {
    let methods = vec![
        WeightInitialization::Xavier,
        WeightInitialization::He,
        WeightInitialization::LeCun,
    ];
    
    let mut best_init = WeightInitialization::Xavier;
    let mut best_loss = f32::INFINITY;
    
    for init_method in methods {
        let config = AdvancedTrainingConfig {
            learning_rate: 0.001,
            initialization: init_method,
            ..Default::default()
        };
        
        let mut model = model_template.clone();
        let mut trainer = AdvancedBackpropagationTrainer::new(config);
        trainer.initialize_weights(&mut model.network);
        
        // Quick training test
        let train_fann_data = dataset_to_fann_data(train_data)?;
        
        for _ in 0..50 {
            if let Ok(loss) = trainer.train_epoch(&mut model.network, &train_fann_data) {
                if loss < best_loss {
                    best_loss = loss;
                    best_init = init_method;
                }
            }
        }
    }
    
    Ok(best_init)
}

/// Convert hyperparameter config to training config
fn convert_hyperparam_to_training_config(
    hyperparam_config: &neural_training::HyperparameterConfig,
    init_method: WeightInitialization,
) -> AdvancedTrainingConfig {
    let mut config = AdvancedTrainingConfig::default();
    config.initialization = init_method;
    
    // Extract parameters from hyperparameter config
    for (key, value) in &hyperparam_config.parameters {
        match (key.as_str(), value) {
            ("learning_rate", neural_training::ParameterValue::Float(lr)) => {
                config.learning_rate = *lr;
            },
            ("momentum", neural_training::ParameterValue::Float(momentum)) => {
                config.momentum = *momentum;
            },
            ("weight_decay", neural_training::ParameterValue::Float(wd)) => {
                config.weight_decay = *wd;
            },
            ("batch_size", neural_training::ParameterValue::String(bs)) => {
                if let Ok(batch_size) = bs.parse::<usize>() {
                    config.batch_size = batch_size;
                }
            },
            _ => {} // Ignore unknown parameters
        }
    }
    
    config
}

/// Convert TelecomDataset to FANN TrainingData format
fn dataset_to_fann_data(dataset: &TelecomDataset) -> Result<neural_training::fann_compat::TrainingData> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    
    for i in 0..dataset.features.nrows() {
        let input: Vec<f32> = dataset.features.row(i).to_vec();
        let output = vec![dataset.targets[i]];
        
        inputs.push(input);
        outputs.push(output);
    }
    
    Ok(neural_training::fann_compat::TrainingData { inputs, outputs })
}

/// Evaluate model on test data
fn evaluate_model(network: &mut neural_training::fann_compat::Network, test_data: &neural_training::fann_compat::TrainingData) -> f32 {
    let mut total_error = 0.0;
    
    for (input, target) in test_data.inputs.iter().zip(test_data.outputs.iter()) {
        let output = network.run(input);
        let error: f32 = output.iter()
            .zip(target.iter())
            .map(|(&o, &t)| (o - t).powi(2))
            .sum::<f32>() / output.len() as f32;
        total_error += error;
    }
    
    total_error / test_data.inputs.len() as f32
}