// Advanced Climate Prediction Example
//
// This example demonstrates advanced features including:
// - Batch predictions for multiple locations
// - Model comparison (Neural vs ARIMA vs Ensemble)
// - ReasoningBank integration for pattern learning
// - Custom model weights
// - Historical accuracy tracking

use climate_prediction::{
    ClimatePredictor, ModelType, PredictionConfig,
    reasoningbank::{ReasoningBank, MemoryEntry},
};
use anyhow::Result;
use serde_json::json;
use std::time::Instant;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Climate Prediction System - Advanced Example\n");

    // Initialize ReasoningBank for adaptive learning
    let reasoningbank = ReasoningBank::new("auto").await?;
    println!("✅ ReasoningBank initialized (backend: auto-detected)");

    // Example 1: Compare different models
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 1: Model Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    compare_models().await?;

    // Example 2: Batch predictions
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 2: Batch Predictions");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    batch_predictions(&reasoningbank).await?;

    // Example 3: ReasoningBank pattern learning
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 3: Pattern Learning with ReasoningBank");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    pattern_learning(&reasoningbank).await?;

    // Example 4: Custom ensemble weights
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 4: Custom Ensemble Weights");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    custom_weights(&reasoningbank).await?;

    println!("\n✅ All examples completed successfully!");

    Ok(())
}

/// Example 1: Compare predictions from different models
async fn compare_models() -> Result<()> {
    let config = PredictionConfig {
        latitude: 51.5074,   // London
        longitude: -0.1278,
        days_ahead: 7,
        include_uncertainty: true,
        units: "metric".to_string(),
    };

    println!("Comparing models for London (51.5074°N, 0.1278°W)...\n");

    // Test each model type
    let models = vec![
        (ModelType::Neural, "Neural Network (WASM)"),
        (ModelType::Arima, "ARIMA Time Series"),
        (ModelType::Hybrid, "Hybrid Model"),
        (ModelType::Ensemble, "Ensemble (All Combined)"),
    ];

    for (model_type, model_name) in models {
        let start = Instant::now();
        let predictor = ClimatePredictor::new(model_type)?;
        let prediction = predictor.predict(&config).await?;
        let duration = start.elapsed();

        println!("📊 {}:", model_name);
        println!("   Temperature: {:.1}°C", prediction.temperature);
        println!("   Confidence:  {:.1}%", prediction.confidence * 100.0);
        println!("   Inference Time: {:?}", duration);

        if let Some(uncertainty) = prediction.uncertainty {
            println!("   Uncertainty: ±{:.1}°C", uncertainty);
        }
        println!();
    }

    Ok(())
}

/// Example 2: Batch predictions for multiple locations
async fn batch_predictions(reasoningbank: &ReasoningBank) -> Result<()> {
    let locations = vec![
        ("New York", 40.7128, -74.0060),
        ("London", 51.5074, -0.1278),
        ("Tokyo", 35.6762, 139.6503),
        ("Sydney", -33.8688, 151.2093),
        ("Mumbai", 19.0760, 72.8777),
        ("São Paulo", -23.5505, -46.6333),
    ];

    println!("Running predictions for {} cities...\n", locations.len());

    let predictor = ClimatePredictor::new(ModelType::Ensemble)?;
    let start = Instant::now();

    // Process all locations in parallel
    let mut handles = vec![];
    for (city, lat, lon) in locations {
        let predictor_clone = predictor.clone();
        let config = PredictionConfig {
            latitude: lat,
            longitude: lon,
            days_ahead: 7,
            include_uncertainty: true,
            units: "metric".to_string(),
        };

        let handle = tokio::spawn(async move {
            let result = predictor_clone.predict(&config).await;
            (city, result)
        });
        handles.push(handle);
    }

    // Wait for all predictions
    let mut results = vec![];
    for handle in handles {
        results.push(handle.await?);
    }

    let total_duration = start.elapsed();

    // Display results
    println!("Results (completed in {:?}):", total_duration);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for (city, result) in results {
        match result {
            Ok(prediction) => {
                println!("{:12} {:.1}°C (confidence: {:.1}%)",
                    city,
                    prediction.temperature,
                    prediction.confidence * 100.0
                );

                // Store in ReasoningBank for learning
                let entry = MemoryEntry {
                    key: format!("prediction/{}/latest", city.replace(" ", "_")),
                    value: json!({
                        "temperature": prediction.temperature,
                        "confidence": prediction.confidence,
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                    }),
                    timestamp: chrono::Utc::now(),
                    metadata: Some(json!({
                        "city": city,
                        "model": "ensemble",
                    })),
                };

                reasoningbank.store("climate/predictions", entry).await.ok();
            }
            Err(e) => println!("{:12} Error: {}", city, e),
        }
    }

    Ok(())
}

/// Example 3: Learn patterns from historical predictions
async fn pattern_learning(reasoningbank: &ReasoningBank) -> Result<()> {
    println!("Learning patterns from historical predictions...\n");

    // Simulate storing multiple predictions over time
    let predictor = ClimatePredictor::new(ModelType::Ensemble)?;

    let test_locations = vec![
        (40.7128, -74.0060, "New York"),
        (34.0522, -118.2437, "Los Angeles"),
        (41.8781, -87.6298, "Chicago"),
    ];

    println!("Generating predictions for pattern analysis...");

    for (lat, lon, city) in test_locations {
        let config = PredictionConfig {
            latitude: lat,
            longitude: lon,
            days_ahead: 7,
            include_uncertainty: true,
            units: "metric".to_string(),
        };

        let prediction = predictor.predict(&config).await?;

        // Store prediction with rich metadata for pattern learning
        let entry = MemoryEntry {
            key: format!("training/{}/{}", city.replace(" ", "_"), chrono::Utc::now().timestamp()),
            value: json!({
                "temperature": prediction.temperature,
                "precipitation": prediction.precipitation,
                "humidity": prediction.humidity,
                "wind_speed": prediction.wind_speed,
                "confidence": prediction.confidence,
            }),
            timestamp: chrono::Utc::now(),
            metadata: Some(json!({
                "location": {
                    "city": city,
                    "latitude": lat,
                    "longitude": lon,
                },
                "conditions": categorize_conditions(&prediction),
                "season": current_season(lat),
                "accuracy_expected": prediction.confidence,
            })),
        };

        reasoningbank.store("climate/training", entry).await?;
    }

    // Query learned patterns
    println!("\n🧠 Learned Patterns:");

    // Search for high-confidence predictions
    let high_confidence = reasoningbank
        .query("climate/training", Some("confidence > 0.9"))
        .await?;

    println!("   High-confidence patterns: {} found", high_confidence.len());

    // Search for specific weather conditions
    let warm_weather = reasoningbank
        .search("climate/training", "warm sunny weather", 10)
        .await?;

    println!("   Warm weather patterns: {} found", warm_weather.len());

    // Display pattern statistics
    if !high_confidence.is_empty() {
        println!("\n📊 Pattern Statistics:");
        let avg_temp: f64 = high_confidence.iter()
            .filter_map(|entry| entry.value.get("temperature")?.as_f64())
            .sum::<f64>() / high_confidence.len() as f64;

        println!("   Average temperature: {:.1}°C", avg_temp);
        println!("   Sample size: {}", high_confidence.len());
    }

    Ok(())
}

/// Example 4: Use custom ensemble weights based on learned patterns
async fn custom_weights(reasoningbank: &ReasoningBank) -> Result<()> {
    println!("Using custom ensemble weights optimized by ReasoningBank...\n");

    // Retrieve optimal weights from previous learning
    let weights_entry = reasoningbank
        .retrieve("climate/models", "ensemble_weights")
        .await;

    let weights = match weights_entry {
        Ok(Some(entry)) => {
            println!("✅ Retrieved learned weights from ReasoningBank");
            extract_weights(&entry.value)
        }
        _ => {
            println!("⚠️  No learned weights found, using defaults");
            vec![0.5, 0.3, 0.2] // Neural, ARIMA, Hybrid
        }
    };

    println!("Ensemble weights: Neural={:.2}, ARIMA={:.2}, Hybrid={:.2}",
        weights[0], weights[1], weights[2]
    );

    // Make prediction with custom weights
    let predictor = ClimatePredictor::with_weights(ModelType::Ensemble, &weights)?;

    let config = PredictionConfig {
        latitude: 37.7749, // San Francisco
        longitude: -122.4194,
        days_ahead: 7,
        include_uncertainty: true,
        units: "metric".to_string(),
    };

    let prediction = predictor.predict(&config).await?;

    println!("\n📊 Prediction with custom weights:");
    println!("   Temperature: {:.1}°C", prediction.temperature);
    println!("   Confidence:  {:.1}%", prediction.confidence * 100.0);

    // Store new weights if accuracy improves
    if prediction.confidence > 0.85 {
        println!("\n💾 Storing improved weights to ReasoningBank...");

        let entry = MemoryEntry {
            key: "ensemble_weights".to_string(),
            value: json!({
                "weights": weights,
                "accuracy": prediction.confidence,
                "updated_at": chrono::Utc::now().to_rfc3339(),
            }),
            timestamp: chrono::Utc::now(),
            metadata: Some(json!({
                "optimization_method": "adaptive_learning",
                "sample_size": 100,
            })),
        };

        reasoningbank.store("climate/models", entry).await?;
        println!("✅ Weights stored successfully");
    }

    Ok(())
}

// Helper functions

fn categorize_conditions(prediction: &climate_prediction::PredictionResult) -> String {
    if prediction.precipitation > 10.0 {
        "rainy".to_string()
    } else if prediction.cloud_cover > 70 {
        "cloudy".to_string()
    } else if prediction.temperature > 25.0 {
        "hot_sunny".to_string()
    } else {
        "clear".to_string()
    }
}

fn current_season(latitude: f64) -> &'static str {
    let month = chrono::Utc::now().month();

    if latitude > 0.0 { // Northern hemisphere
        match month {
            3..=5 => "spring",
            6..=8 => "summer",
            9..=11 => "autumn",
            _ => "winter",
        }
    } else { // Southern hemisphere
        match month {
            3..=5 => "autumn",
            6..=8 => "winter",
            9..=11 => "spring",
            _ => "summer",
        }
    }
}

fn extract_weights(value: &serde_json::Value) -> Vec<f64> {
    value.get("weights")
        .and_then(|w| w.as_array())
        .map(|arr| arr.iter()
            .filter_map(|v| v.as_f64())
            .collect())
        .unwrap_or_else(|| vec![0.5, 0.3, 0.2])
}
