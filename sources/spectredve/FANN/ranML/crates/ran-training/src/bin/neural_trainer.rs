use std::path::PathBuf;
use clap::Parser;
use anyhow::Result;
use log::{info, warn};
use ran_training::*;

#[derive(Parser)]
#[command(name = "neural-trainer")]
#[command(about = "Pure Rust neural network training with Mac GPU acceleration")]
struct Args {
    /// Path to the CSV data file
    #[arg(short, long)]
    data_path: PathBuf,
    
    /// Output directory for models and results
    #[arg(short, long, default_value = "models")]
    output_dir: PathBuf,
    
    /// Number of epochs for training
    #[arg(short, long, default_value = "100")]
    epochs: usize,
    
    /// Batch size for training
    #[arg(short, long, default_value = "32")]
    batch_size: usize,
    
    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,
    
    /// Enable GPU acceleration (Mac Metal)
    #[arg(long)]
    use_gpu: bool,
    
    /// Model type to train (all, cnn, lstm, dense)
    #[arg(short, long, default_value = "all")]
    model_type: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    
    info!("🚀 Starting Pure Rust Neural Network Training Pipeline");
    info!("📊 Data path: {:?}", args.data_path);
    info!("🔧 GPU acceleration: {}", args.use_gpu);
    info!("📈 Model type: {}", args.model_type);
    
    // Initialize swarm coordination
    info!("🐝 Initializing swarm coordination...");
    let swarm_config = SwarmConfig {
        id: "neural-training-swarm".to_string(),
        max_agents: 3,
        use_gpu: args.use_gpu,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        epochs: args.epochs,
    };
    
    // Load and preprocess data
    info!("📂 Loading and preprocessing data...");
    let preprocessor = DataPreprocessor::new()?;
    let dataset = preprocessor.load_csv_data(&args.data_path)?;
    info!("✅ Data loaded: {} samples, {} features", dataset.len(), dataset.feature_count());
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // Train models based on selection
    match args.model_type.as_str() {
        "all" => {
            info!("🎯 Training all 3 models in parallel...");
            train_all_models(&dataset, &swarm_config, &args.output_dir).await?;
        }
        "cnn" => {
            info!("🧠 Training CNN model...");
            train_cnn_model(&dataset, &swarm_config, &args.output_dir).await?;
        }
        "lstm" => {
            info!("🔄 Training LSTM model...");
            train_lstm_model(&dataset, &swarm_config, &args.output_dir).await?;
        }
        "dense" => {
            info!("📊 Training Dense MLP model...");
            train_dense_model(&dataset, &swarm_config, &args.output_dir).await?;
        }
        _ => {
            warn!("⚠️  Unknown model type: {}", args.model_type);
            return Err(anyhow::anyhow!("Invalid model type"));
        }
    }
    
    info!("🎉 Training complete! Models saved to {:?}", args.output_dir);
    Ok(())
}

async fn train_all_models(
    dataset: &TelecomDataset,
    config: &SwarmConfig,
    output_dir: &PathBuf,
) -> Result<()> {
    info!("🐝 Spawning 3 training agents in parallel...");
    
    let (train_data, val_data, test_data) = dataset.split_data(0.7, 0.15, 0.15)?;
    
    // Spawn all 3 agents in parallel using tokio
    let cnn_handle = tokio::spawn(train_cnn_agent(train_data.clone(), val_data.clone(), config.clone(), output_dir.clone()));
    let lstm_handle = tokio::spawn(train_lstm_agent(train_data.clone(), val_data.clone(), config.clone(), output_dir.clone()));
    let dense_handle = tokio::spawn(train_dense_agent(train_data.clone(), val_data.clone(), config.clone(), output_dir.clone()));
    
    // Wait for all agents to complete
    let (cnn_result, lstm_result, dense_result) = tokio::try_join!(cnn_handle, lstm_handle, dense_handle)?;
    
    // Check results
    cnn_result?;
    lstm_result?;
    dense_result?;
    
    // Evaluate all models
    info!("📊 Evaluating all models...");
    let evaluator = ModelEvaluator::new()?;
    evaluator.evaluate_all_models(&test_data, output_dir).await?;
    
    info!("✅ All models trained and evaluated successfully!");
    Ok(())
}

async fn train_cnn_agent(
    train_data: TrainingData,
    val_data: TrainingData,
    config: SwarmConfig,
    output_dir: PathBuf,
) -> Result<()> {
    info!("🧠 CNN Agent: Starting training...");
    
    let mut model = CNNModel::new(
        train_data.feature_count(),
        train_data.target_count(),
        config.use_gpu,
    )?;
    
    let trainer = ModelTrainer::new(config.clone())?;
    let training_results = trainer.train_model(&mut model, &train_data, &val_data).await?;
    
    // Save model
    let model_path = output_dir.join("cnn_model.safetensors");
    model.save(&model_path)?;
    
    info!("✅ CNN Agent: Training complete! Loss: {:.4}", training_results.final_loss);
    Ok(())
}

async fn train_lstm_agent(
    train_data: TrainingData,
    val_data: TrainingData,
    config: SwarmConfig,
    output_dir: PathBuf,
) -> Result<()> {
    info!("🔄 LSTM Agent: Starting training...");
    
    let mut model = LSTMModel::new(
        train_data.feature_count(),
        train_data.target_count(),
        config.use_gpu,
    )?;
    
    let trainer = ModelTrainer::new(config.clone())?;
    let training_results = trainer.train_model(&mut model, &train_data, &val_data).await?;
    
    // Save model
    let model_path = output_dir.join("lstm_model.safetensors");
    model.save(&model_path)?;
    
    info!("✅ LSTM Agent: Training complete! Loss: {:.4}", training_results.final_loss);
    Ok(())
}

async fn train_dense_agent(
    train_data: TrainingData,
    val_data: TrainingData,
    config: SwarmConfig,
    output_dir: PathBuf,
) -> Result<()> {
    info!("📊 Dense Agent: Starting training...");
    
    let mut model = DenseModel::new(
        train_data.feature_count(),
        train_data.target_count(),
        config.use_gpu,
    )?;
    
    let trainer = ModelTrainer::new(config.clone())?;
    let training_results = trainer.train_model(&mut model, &train_data, &val_data).await?;
    
    // Save model
    let model_path = output_dir.join("dense_model.safetensors");
    model.save(&model_path)?;
    
    info!("✅ Dense Agent: Training complete! Loss: {:.4}", training_results.final_loss);
    Ok(())
}

// Single model training functions
async fn train_cnn_model(dataset: &TelecomDataset, config: &SwarmConfig, output_dir: &PathBuf) -> Result<()> {
    let (train_data, val_data, _) = dataset.split_data(0.8, 0.2, 0.0)?;
    train_cnn_agent(train_data, val_data, config.clone(), output_dir.clone()).await
}

async fn train_lstm_model(dataset: &TelecomDataset, config: &SwarmConfig, output_dir: &PathBuf) -> Result<()> {
    let (train_data, val_data, _) = dataset.split_data(0.8, 0.2, 0.0)?;
    train_lstm_agent(train_data, val_data, config.clone(), output_dir.clone()).await
}

async fn train_dense_model(dataset: &TelecomDataset, config: &SwarmConfig, output_dir: &PathBuf) -> Result<()> {
    let (train_data, val_data, _) = dataset.split_data(0.8, 0.2, 0.0)?;
    train_dense_agent(train_data, val_data, config.clone(), output_dir.clone()).await
}