//! Complete workflow: training, evaluation, storage, and prediction

use anyhow::Result;
use climate_models::{
    models::{ensemble::EnsembleStrategy, EnsembleModel, LSTMModel},
    storage::{ModelStorage, ModelVersion},
    training::{ModelEvaluator, ModelTrainer, TrainingConfig},
    traits::{ClimateInput, ClimatePrediction, PredictionModel},
    ModelConfig,
};
use std::{path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Complete Climate Model Workflow ===\n");

    // Step 1: Configure Training
    println!("Step 1: Configure Training");
    println!("----------------------------");

    let mut config = TrainingConfig::default();
    config.epochs = 50;
    config.batch_size = 32;
    config.learning_rate = 0.001;

    println!("Training Configuration:");
    println!("  Epochs: {}", config.epochs);
    println!("  Batch Size: {}", config.batch_size);
    println!("  Learning Rate: {}", config.learning_rate);
    println!("  Optimizer: {}", config.optimizer);
    println!();

    // Step 2: Train Model (requires Python environment and data)
    println!("Step 2: Train Model");
    println!("-------------------");

    println!("To train a model, run:");
    println!("  python3 python/train_lstm.py \\");
    println!("    --data data/climate_train.npy \\");
    println!("    --output models/lstm_v1 \\");
    println!("    --config config/training_config.json");
    println!();

    // Step 3: Model Storage
    println!("Step 3: Model Storage and Versioning");
    println!("-------------------------------------");

    let storage = ModelStorage::new("models/")?;

    let model_config = ModelConfig::lstm(3, 64, 2, 3);
    let version = ModelVersion::new(
        "1.0.0",
        "LSTM",
        model_config,
        PathBuf::from("models/lstm_v1/lstm_climate.onnx"),
    )
    .with_metric("rmse", 0.45)
    .with_metric("mae", 0.32)
    .with_metric("r2_score", 0.94)
    .with_metadata("trained_by", "climate-team")
    .with_metadata("dataset", "climate-2024");

    println!("Model Version: {}", version.version);
    println!("Architecture: {}", version.architecture);
    println!("Metrics:");
    for (name, value) in &version.metrics {
        println!("  {}: {:.4}", name, value);
    }
    println!();

    // Note: Actual save would require the model file to exist
    // storage.save_version(&version)?;

    // Step 4: Load and Evaluate
    println!("Step 4: Model Evaluation");
    println!("------------------------");

    // Create test data
    let test_inputs = vec![
        ClimateInput::with_sequence(vec![15.0, 1013.0, 60.0, 15.5, 1012.5, 61.0], 2),
        ClimateInput::with_sequence(vec![16.0, 1012.0, 65.0, 16.5, 1011.5, 66.0], 2),
        ClimateInput::with_sequence(vec![14.0, 1014.0, 55.0, 14.5, 1013.5, 56.0], 2),
    ];

    let test_targets = vec![
        ClimatePrediction::temperature(16.0, 0.9, 1),
        ClimatePrediction::temperature(17.0, 0.9, 1),
        ClimatePrediction::temperature(15.0, 0.9, 1),
    ];

    println!("Test set size: {} samples", test_inputs.len());
    println!();

    // With a trained model, you would:
    // let model = Arc::new(LSTMModel::from_onnx("models/lstm_v1/lstm_climate.onnx").await?);
    // let evaluator = ModelEvaluator::new(test_inputs.clone(), test_targets.clone());
    // let result = evaluator.evaluate(model).await?;
    //
    // println!("Evaluation Results:");
    // println!("  RMSE: {:.4}", result.metrics.rmse);
    // println!("  MAE: {:.4}", result.metrics.mae);
    // println!("  R² Score: {:.4}", result.metrics.r2_score);
    // println!("  Inference Time: {:.2} ms", result.metrics.inference_time_ms);

    // Step 5: Create Ensemble
    println!("Step 5: Ensemble Model");
    println!("----------------------");

    println!("Creating ensemble with multiple models...");

    // In production, load multiple trained models:
    // let lstm = Arc::new(LSTMModel::from_onnx("models/lstm_v1.onnx").await?) as Arc<dyn PredictionModel>;
    // let fno = Arc::new(FNOModel::from_onnx("models/fno_v1.onnx").await?) as Arc<dyn PredictionModel>;
    //
    // let mut ensemble = EnsembleModel::new("production_ensemble", EnsembleStrategy::WeightedByConfidence);
    // ensemble.add_model(lstm);
    // ensemble.add_model(fno);
    //
    // // Make predictions
    // let input = ClimateInput::new(vec![15.5, 1013.2, 65.0]);
    // let prediction = ensemble.predict(&input).await?;

    println!("Ensemble strategies available:");
    println!("  - Average: Simple average of predictions");
    println!("  - WeightedByConfidence: Weight by model confidence");
    println!("  - WeightedCustom: Custom weights per model");
    println!("  - MostConfident: Use most confident model");
    println!("  - Median: Robust to outliers");
    println!();

    // Step 6: Production Deployment
    println!("Step 6: Production Deployment");
    println!("------------------------------");

    println!("For production deployment:");
    println!("  1. Train models with production data");
    println!("  2. Export to ONNX format");
    println!("  3. Version and store models");
    println!("  4. Load in Rust service for inference");
    println!("  5. Monitor performance metrics");
    println!();

    println!("Example service endpoint:");
    println!("  POST /predict");
    println!("  Body: {{ \"features\": [15.5, 1013.2, 65.0] }}");
    println!("  Response: {{");
    println!("    \"temperature\": 16.2,");
    println!("    \"confidence\": 0.89,");
    println!("    \"horizon\": 1");
    println!("  }}");
    println!();

    // Step 7: Model Comparison
    println!("Step 7: Model Comparison");
    println!("------------------------");

    println!("Compare multiple models:");
    println!();
    println!("| Model         | RMSE  | MAE   | R²    | Inference (ms) |");
    println!("|---------------|-------|-------|-------|----------------|");
    println!("| LSTM v1       | 0.450 | 0.320 | 0.940 |           2.3  |");
    println!("| LSTM v2       | 0.420 | 0.310 | 0.950 |           2.8  |");
    println!("| FNO v1        | 0.520 | 0.380 | 0.920 |           5.1  |");
    println!("| Ensemble      | 0.410 | 0.290 | 0.955 |           7.4  |");
    println!();

    println!("Best model: Ensemble (RMSE: 0.410)");
    println!();

    println!("=== Workflow Complete ===");
    println!();
    println!("Next steps:");
    println!("  1. Prepare your climate dataset");
    println!("  2. Run training with python/train_lstm.py");
    println!("  3. Evaluate models on test set");
    println!("  4. Deploy best model to production");

    Ok(())
}
