//! Ensemble model demonstration

use anyhow::Result;
use climate_models::{
    models::{ensemble::EnsembleStrategy, EnsembleModel, LSTMModel},
    traits::{ClimateInput, ClimatePrediction, PredictionModel},
    ModelConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Ensemble Model Demo ===\n");

    // Create multiple models (in production, these would be trained models)
    let lstm1 = Arc::new(LSTMModel::new(
        "lstm_v1",
        ModelConfig::lstm(3, 64, 2, 3),
    )) as Arc<dyn PredictionModel>;

    let lstm2 = Arc::new(LSTMModel::new(
        "lstm_v2",
        ModelConfig::lstm(3, 128, 3, 3),
    )) as Arc<dyn PredictionModel>;

    // Demonstrate different ensemble strategies
    let strategies = vec![
        ("Average", EnsembleStrategy::Average),
        ("Weighted by Confidence", EnsembleStrategy::WeightedByConfidence),
        ("Most Confident", EnsembleStrategy::MostConfident),
        ("Median", EnsembleStrategy::Median),
    ];

    for (name, strategy) in strategies {
        println!("=== Strategy: {} ===", name);

        let mut ensemble = EnsembleModel::new(format!("ensemble_{}", name), strategy);
        ensemble.add_model(lstm1.clone());
        ensemble.add_model(lstm2.clone());

        println!("Models in ensemble: {}", 2);
        println!();

        // Demonstrate with mock predictions
        let predictions = vec![
            ClimatePrediction::temperature(20.0, 0.9, 1),
            ClimatePrediction::temperature(22.0, 0.8, 1),
            ClimatePrediction::temperature(21.0, 0.85, 1),
        ];

        // This is a simulation - in production you would call ensemble.predict()
        // let input = ClimateInput::new(vec![15.0, 1013.0, 65.0]);
        // let result = ensemble.predict(&input).await?;

        println!("Sample predictions:");
        for (i, pred) in predictions.iter().enumerate() {
            println!(
                "  Model {}: {:.2}°C (confidence: {:.2})",
                i + 1,
                pred.temperature,
                pred.confidence
            );
        }
        println!();
    }

    println!("=== Custom Weighted Ensemble ===");
    let weights = vec![0.6, 0.4]; // Give more weight to first model

    let mut weighted_ensemble = EnsembleModel::with_weights("weighted_ensemble", weights);
    weighted_ensemble.add_model(lstm1);
    weighted_ensemble.add_model(lstm2);

    println!("Custom weights: 60% model 1, 40% model 2");
    println!();

    println!("To use ensemble with trained models:");
    println!("  1. Train multiple models with different architectures");
    println!("  2. Load them as Arc<dyn PredictionModel>");
    println!("  3. Add to ensemble with your chosen strategy");
    println!("  4. Call ensemble.predict() for robust predictions");

    Ok(())
}
