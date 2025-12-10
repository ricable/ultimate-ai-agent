// Basic Climate Prediction Example
//
// This example demonstrates the simplest way to use the Climate Prediction System
// to get a weather forecast for a specific location.

use climate_prediction::{ClimatePredictor, ModelType, PredictionConfig};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the predictor with the ensemble model (recommended)
    // The ensemble model combines neural networks, ARIMA, and hybrid models
    // for the most accurate predictions
    let predictor = ClimatePredictor::new(ModelType::Ensemble)?;

    println!("🌤️  Climate Prediction System - Basic Example\n");
    println!("Initializing predictor...");

    // Configure the prediction request
    // This example predicts weather for New York City, 7 days ahead
    let config = PredictionConfig {
        latitude: 40.7128,   // New York City latitude
        longitude: -74.0060, // New York City longitude
        days_ahead: 7,       // Predict 7 days into the future
        include_uncertainty: true, // Include confidence intervals
        units: "metric".to_string(), // Use metric units (Celsius, mm, etc.)
    };

    println!("Getting prediction for NYC (40.7128°N, 74.0060°W)...\n");

    // Make the prediction
    // This call will:
    // 1. Check the cache for recent predictions
    // 2. If not cached, run inference on all models in parallel
    // 3. Combine results using ensemble weights
    // 4. Store results in ReasoningBank for learning
    let prediction = predictor.predict(&config).await?;

    // Display the results in a user-friendly format
    println!("📊 Prediction Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Temperature:  {:.1}°C", prediction.temperature);
    println!("   Feels Like:   {:.1}°C", prediction.feels_like);
    println!("   Humidity:     {}%", prediction.humidity);
    println!("   Precipitation: {:.1}mm", prediction.precipitation);
    println!("   Wind Speed:   {:.1} km/h", prediction.wind_speed);
    println!("   Wind Direction: {}° ({})",
        prediction.wind_direction,
        direction_to_cardinal(prediction.wind_direction)
    );
    println!("   Pressure:     {:.1} hPa", prediction.pressure);
    println!("   Cloud Cover:  {}%", prediction.cloud_cover);
    println!("   UV Index:     {}", prediction.uv_index);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Display confidence and model information
    println!("\n📈 Prediction Metadata:");
    println!("   Confidence:   {:.1}%", prediction.confidence * 100.0);
    println!("   Model Used:   {}", prediction.model_name);

    // If uncertainty is included, show the range
    if let Some(uncertainty) = prediction.uncertainty {
        println!("\n🎯 Uncertainty Range:");
        println!("   Temperature:");
        println!("     Most Likely: {:.1}°C", prediction.temperature);
        println!("     Low Estimate: {:.1}°C", prediction.temperature - uncertainty);
        println!("     High Estimate: {:.1}°C", prediction.temperature + uncertainty);
    }

    // Display daily forecast if available
    if !prediction.daily_forecast.is_empty() {
        println!("\n📅 7-Day Forecast:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for (i, day) in prediction.daily_forecast.iter().enumerate() {
            println!("Day {}: {} - High: {:.1}°C, Low: {:.1}°C, Precip: {:.1}mm",
                i + 1,
                day.date,
                day.temp_max,
                day.temp_min,
                day.precipitation
            );
        }
    }

    // Provide weather interpretation
    println!("\n🌡️  Weather Interpretation:");
    interpret_weather(&prediction);

    println!("\n✅ Prediction complete!");

    Ok(())
}

/// Convert wind direction in degrees to cardinal direction (N, S, E, W, etc.)
fn direction_to_cardinal(degrees: i32) -> &'static str {
    let directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ];

    let index = ((degrees as f64 + 11.25) / 22.5) as usize % 16;
    directions[index]
}

/// Provide human-readable interpretation of weather conditions
fn interpret_weather(prediction: &climate_prediction::PredictionResult) {
    // Temperature interpretation
    match prediction.temperature {
        t if t < 0.0 => println!("   🥶 Freezing conditions expected"),
        t if t < 10.0 => println!("   🧊 Cold weather ahead"),
        t if t < 20.0 => println!("   🌤️  Mild temperatures expected"),
        t if t < 30.0 => println!("   ☀️  Warm weather forecast"),
        _ => println!("   🔥 Hot conditions expected"),
    }

    // Precipitation interpretation
    match prediction.precipitation {
        p if p < 1.0 => println!("   ☁️  Dry conditions"),
        p if p < 5.0 => println!("   🌦️  Light rain possible"),
        p if p < 20.0 => println!("   🌧️  Moderate rain expected"),
        _ => println!("   ⛈️  Heavy precipitation likely"),
    }

    // Wind interpretation
    match prediction.wind_speed {
        w if w < 10.0 => println!("   🍃 Calm winds"),
        w if w < 30.0 => println!("   💨 Moderate winds"),
        w if w < 50.0 => println!("   🌬️  Strong winds expected"),
        _ => println!("   🌪️  Very strong winds - take precautions"),
    }

    // UV index interpretation
    match prediction.uv_index {
        0..=2 => println!("   😎 Low UV - minimal protection needed"),
        3..=5 => println!("   🕶️  Moderate UV - use sunscreen"),
        6..=7 => println!("   🧴 High UV - protection essential"),
        8..=10 => println!("   ⚠️  Very high UV - avoid midday sun"),
        _ => println!("   🚨 Extreme UV - stay in shade"),
    }

    // Confidence interpretation
    match prediction.confidence {
        c if c > 0.9 => println!("   ✅ Very high confidence in prediction"),
        c if c > 0.8 => println!("   👍 High confidence"),
        c if c > 0.7 => println!("   ⚖️  Moderate confidence"),
        _ => println!("   ⚠️  Lower confidence - conditions may vary"),
    }
}
