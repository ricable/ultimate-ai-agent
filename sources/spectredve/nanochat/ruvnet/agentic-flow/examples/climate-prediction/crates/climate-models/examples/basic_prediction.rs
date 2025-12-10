//! Basic climate prediction example

use anyhow::Result;
use climate_models::{
    models::LSTMModel,
    traits::{ClimateInput, PredictionModel},
    ModelConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Climate Prediction Example ===\n");

    // For this example, we'll create a mock model since we don't have a trained ONNX file
    // In production, you would use:
    // let model = LSTMModel::from_onnx("models/lstm_climate.onnx").await?;

    let config = ModelConfig::lstm(3, 64, 2, 3);
    let model = LSTMModel::new("example_lstm", config);

    println!("Model: {}", model.name());
    println!("Configuration: {:?}\n", model.config());

    // Create sample input data (temperature, pressure, humidity over time)
    let features = vec![
        // Time step 1
        15.5, 1013.2, 65.0,
        // Time step 2
        16.0, 1012.8, 64.5,
        // Time step 3
        16.5, 1012.5, 64.0,
    ];

    let input = ClimateInput::with_sequence(features, 3);

    println!("Input shape: {:?}", input.shape());
    println!("Input data: {:?}\n", input.features);

    // Note: This would require a trained model to work
    // let prediction = model.predict(&input).await?;
    //
    // println!("=== Prediction ===");
    // println!("Temperature: {:.2}°C", prediction.temperature);
    // if let Some(pressure) = prediction.pressure {
    //     println!("Pressure: {:.2} hPa", pressure);
    // }
    // if let Some(humidity) = prediction.humidity {
    //     println!("Humidity: {:.2}%", humidity);
    // }
    // println!("Confidence: {:.2}%", prediction.confidence * 100.0);
    // println!("Horizon: {} hours", prediction.horizon);

    println!("To run actual predictions, train a model using:");
    println!("  python3 python/train_lstm.py --data data/climate.npy --output models/ --config config.json");

    Ok(())
}
