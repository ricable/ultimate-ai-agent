//! Neural network training CLI with swarm coordination and WASM support

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use neural_training::*;
use std::path::PathBuf;
use tokio;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "neural-trainer")]
#[command(about = "A swarm-based neural network training system for telecom data")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Number of parallel agents
    #[arg(short = 'j', long, default_value = "5")]
    parallel_agents: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Train neural networks using swarm coordination
    Train {
        /// Input data file (CSV)
        #[arg(short, long, value_name = "FILE")]
        data: PathBuf,
        
        /// Output directory for results
        #[arg(short, long, default_value = "results")]
        output: PathBuf,
        
        /// Disable swarm coordination (single-threaded training)
        #[arg(long)]
        no_swarm: bool,
        
        /// Maximum training epochs
        #[arg(long, default_value = "1000")]
        max_epochs: usize,
        
        /// Learning rate
        #[arg(long, default_value = "0.01")]
        learning_rate: f32,
        
        /// Target error for convergence
        #[arg(long, default_value = "0.001")]
        target_error: f32,
    },
    
    /// Perform hyperparameter tuning
    Tune {
        /// Input data file (CSV)
        #[arg(short, long, value_name = "FILE")]
        data: PathBuf,
        
        /// Output directory for results
        #[arg(short, long, default_value = "tuning_results")]
        output: PathBuf,
        
        /// Model architecture to tune
        #[arg(short, long, default_value = "shallow")]
        architecture: String,
        
        /// Number of hyperparameter combinations to test
        #[arg(long, default_value = "27")]
        max_combinations: usize,
    },
    
    /// Evaluate trained models
    Evaluate {
        /// Input data file (CSV)
        #[arg(short, long, value_name = "FILE")]
        data: PathBuf,
        
        /// Directory containing trained models
        #[arg(short, long, value_name = "DIR")]
        models: PathBuf,
        
        /// Output file for evaluation report
        #[arg(short, long, default_value = "evaluation_report.json")]
        output: PathBuf,
        
        /// Perform cross-validation
        #[arg(long)]
        cross_validate: bool,
        
        /// Number of folds for cross-validation
        #[arg(long, default_value = "5")]
        k_folds: usize,
    },
    
    /// Generate default configuration file
    GenerateConfig {
        /// Output configuration file
        #[arg(short, long, default_value = "training_config.json")]
        output: PathBuf,
        
        /// Configuration format (json or yaml)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    
    /// Data preprocessing and analysis
    Preprocess {
        /// Input data file (CSV)
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,
        
        /// Output directory for processed data
        #[arg(short, long, default_value = "processed_data")]
        output: PathBuf,
        
        /// Train/test split ratio
        #[arg(long, default_value = "0.8")]
        split_ratio: f32,
        
        /// Enable feature normalization
        #[arg(long)]
        normalize: bool,
        
        /// Remove outliers
        #[arg(long)]
        remove_outliers: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Load configuration
    let mut config = if let Some(config_path) = &cli.config {
        TrainingConfig::load(config_path)
            .context("Failed to load configuration file")?
    } else {
        TrainingConfig::default()
    };
    
    // Override config with CLI parameters
    config.swarm.max_parallel_agents = cli.parallel_agents;
    
    // Set log level based on verbose flag
    if cli.verbose {
        config.output.logging.level = "debug".to_string();
    }
    
    // Initialize logging
    config.init_logging()
        .context("Failed to initialize logging")?;
    
    // Validate configuration
    config.validate()
        .context("Configuration validation failed")?;
    
    log::info!("Neural Network Training System v1.0.0");
    log::info!("Swarm coordination: {}", config.swarm.enabled);
    log::info!("Max parallel agents: {}", config.swarm.max_parallel_agents);
    
    // Execute command
    match cli.command {
        Commands::Train {
            data,
            output,
            no_swarm,
            max_epochs,
            learning_rate,
            target_error,
        } => {
            execute_training(
                data,
                output,
                config.with_swarm_enabled(!no_swarm)
                    .with_training_params(learning_rate, config.training.momentum, max_epochs),
                target_error,
            ).await?;
        }
        
        Commands::Tune {
            data,
            output,
            architecture,
            max_combinations,
        } => {
            execute_hyperparameter_tuning(data, output, config, architecture, max_combinations).await?;
        }
        
        Commands::Evaluate {
            data,
            models,
            output,
            cross_validate,
            k_folds,
        } => {
            execute_evaluation(data, models, output, config, cross_validate, k_folds).await?;
        }
        
        Commands::GenerateConfig { output, format } => {
            execute_generate_config(output, format)?;
        }
        
        Commands::Preprocess {
            input,
            output,
            split_ratio,
            normalize,
            remove_outliers,
        } => {
            execute_preprocessing(input, output, split_ratio, normalize, remove_outliers).await?;
        }
    }
    
    log::info!("Operation completed successfully");
    Ok(())
}

/// Execute training command
async fn execute_training(
    data_path: PathBuf,
    output_path: PathBuf,
    config: TrainingConfig,
    target_error: f32,
) -> Result<()> {
    log::info!("Starting neural network training");
    log::info!("Data file: {:?}", data_path);
    log::info!("Output directory: {:?}", output_path);
    
    // Create output directory
    std::fs::create_dir_all(&output_path)
        .context("Failed to create output directory")?;
    
    // Create training system
    let mut system = NeuralTrainingSystem::new();
    system.config = config.with_data_file(&data_path);
    system.config.training.target_error = target_error;
    
    // Run training pipeline
    let results = system.run_training_pipeline(&data_path).await
        .context("Training pipeline failed")?;
    
    // Save results
    let results_file = output_path.join("training_results.json");
    let results_json = serde_json::to_string_pretty(&results)
        .context("Failed to serialize results")?;
    std::fs::write(&results_file, results_json)
        .context("Failed to save results")?;
    
    log::info!("Training completed successfully");
    log::info!("Results saved to: {:?}", results_file);
    log::info!("Best model: {}", results.best_model_name);
    log::info!("Best validation error: {:.6}", results.best_validation_error);
    
    Ok(())
}

/// Execute hyperparameter tuning command
async fn execute_hyperparameter_tuning(
    data_path: PathBuf,
    output_path: PathBuf,
    config: TrainingConfig,
    architecture: String,
    max_combinations: usize,
) -> Result<()> {
    log::info!("Starting hyperparameter tuning");
    log::info!("Architecture: {}", architecture);
    log::info!("Max combinations: {}", max_combinations);
    
    // Create output directory
    std::fs::create_dir_all(&output_path)
        .context("Failed to create output directory")?;
    
    // Load and split data
    let dataset = TelecomDataLoader::load(&data_path)
        .context("Failed to load dataset")?;
    let data_split = dataset.split_train_test(config.data.train_test_split)
        .context("Failed to split dataset")?;
    
    // Find architecture configuration
    let arch_config = config.enabled_architectures()
        .into_iter()
        .find(|arch| arch.name == architecture)
        .ok_or_else(|| anyhow::anyhow!("Architecture '{}' not found", architecture))?;
    
    // Create model template
    let input_size = dataset.features.ncols();
    let model_arch = NetworkArchitecture {
        name: arch_config.name.clone(),
        layer_sizes: arch_config.layer_sizes.clone(),
        activation_functions: arch_config.activations.iter()
            .map(|a| parse_activation_function(a))
            .collect(),
        connection_rate: config.models.connection_rate,
        bias: config.models.use_bias,
    };
    
    let model_template = NeuralModel::from_architecture(model_arch)
        .context("Failed to create model template")?;
    
    // Create hyperparameter configuration
    let hyperparameter_config = HyperparameterConfig::default();
    let mut combinations = hyperparameter_config.generate_combinations();
    combinations.truncate(max_combinations);
    
    // Perform tuning
    let swarm = SwarmOrchestrator::new();
    let tuner = HyperparameterTuner::new(config);
    let (best_params, best_result) = tuner.grid_search(
        model_template,
        combinations,
        &data_split.train,
        &data_split.test,
    ).context("Hyperparameter tuning failed")?;
    
    // Save results
    let tuning_results = serde_json::json!({
        "best_parameters": best_params,
        "best_result": best_result,
        "architecture": architecture,
    });
    
    let results_file = output_path.join("tuning_results.json");
    std::fs::write(&results_file, serde_json::to_string_pretty(&tuning_results)?)
        .context("Failed to save tuning results")?;
    
    log::info!("Hyperparameter tuning completed");
    log::info!("Best parameters saved to: {:?}", results_file);
    
    Ok(())
}

/// Execute evaluation command
async fn execute_evaluation(
    data_path: PathBuf,
    models_path: PathBuf,
    output_path: PathBuf,
    config: TrainingConfig,
    cross_validate: bool,
    k_folds: usize,
) -> Result<()> {
    log::info!("Starting model evaluation");
    
    // Load test data
    let dataset = TelecomDataLoader::load(&data_path)
        .context("Failed to load dataset")?;
    
    // TODO: Load trained models from models_path
    // This would require saving/loading model weights
    
    if cross_validate {
        log::info!("Performing {}-fold cross-validation", k_folds);
        // TODO: Implement cross-validation evaluation
    }
    
    log::info!("Evaluation completed");
    Ok(())
}

/// Execute generate config command
fn execute_generate_config(output_path: PathBuf, format: String) -> Result<()> {
    log::info!("Generating default configuration");
    
    let config = TrainingConfig::default();
    
    if format == "yaml" {
        let yaml_content = serde_yaml::to_string(&config)
            .context("Failed to serialize configuration to YAML")?;
        std::fs::write(&output_path, yaml_content)
            .context("Failed to write configuration file")?;
    } else {
        let json_content = serde_json::to_string_pretty(&config)
            .context("Failed to serialize configuration to JSON")?;
        std::fs::write(&output_path, json_content)
            .context("Failed to write configuration file")?;
    }
    
    log::info!("Configuration saved to: {:?}", output_path);
    Ok(())
}

/// Execute preprocessing command
async fn execute_preprocessing(
    input_path: PathBuf,
    output_path: PathBuf,
    split_ratio: f32,
    normalize: bool,
    remove_outliers: bool,
) -> Result<()> {
    log::info!("Starting data preprocessing");
    
    // Create output directory
    std::fs::create_dir_all(&output_path)
        .context("Failed to create output directory")?;
    
    // Load dataset
    let mut dataset = TelecomDataLoader::load(&input_path)
        .context("Failed to load dataset")?;
    
    // Apply preprocessing
    if normalize {
        dataset.normalize()
            .context("Failed to normalize dataset")?;
        log::info!("Applied feature normalization");
    }
    
    // Split data
    let data_split = dataset.split_train_test(split_ratio)
        .context("Failed to split dataset")?;
    
    // Save processed data
    let train_path = output_path.join("train.json");
    let test_path = output_path.join("test.json");
    
    dataset.save_splits(&train_path, &test_path, &data_split)
        .context("Failed to save processed data")?;
    
    log::info!("Preprocessing completed");
    log::info!("Training data: {:?} ({} samples)", train_path, data_split.train.features.nrows());
    log::info!("Test data: {:?} ({} samples)", test_path, data_split.test.features.nrows());
    
    Ok(())
}

/// Parse activation function from string
fn parse_activation_function(activation: &str) -> neural_training::fann_compat::ActivationFunction {
    use neural_training::fann_compat::ActivationFunction;
    
    match activation.to_lowercase().as_str() {
        "linear" => ActivationFunction::Linear,
        "sigmoid" => ActivationFunction::Sigmoid,
        "tanh" => ActivationFunction::Tanh,
        "relu" => ActivationFunction::ReLU,
        "leaky_relu" => ActivationFunction::ReLULeaky,
        "elliot" => ActivationFunction::Elliot,
        "gaussian" => ActivationFunction::Gaussian,
        _ => {
            log::warn!("Unknown activation function '{}', using ReLU", activation);
            ActivationFunction::ReLU
        }
    }
}