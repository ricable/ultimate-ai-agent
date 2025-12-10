//! Climate Prediction with ReasoningBank Integration
//!
//! This example demonstrates how to use ReasoningBank for continuous learning
//! and optimization in a climate prediction system.

use climate_core::{
    ClimateApiClient, ClimateInput, Location, ModelType,
    PredictionCache, ReasoningBankOptimizer,
};
use std::path::PathBuf;
use tracing::{info, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("climate_prediction=debug,climate_core=debug")
        .init();

    info!("🌍 Climate Prediction System with ReasoningBank");
    info!("================================================\n");

    // Initialize hooks for coordination
    run_hooks_pre_task()?;

    // Initialize components
    let db_path = PathBuf::from(".swarm/climate_memory.db");
    let mut optimizer = ReasoningBankOptimizer::new(db_path, "climate_prediction")?;
    let cache = PredictionCache::new();
    let api_client = ClimateApiClient::new(None);

    // Example locations
    let locations = vec![
        Location {
            latitude: 40.7128,
            longitude: -74.0060,
            name: Some("New York, NY".to_string()),
        },
        Location {
            latitude: 34.0522,
            longitude: -118.2437,
            name: Some("Los Angeles, CA".to_string()),
        },
        Location {
            latitude: 41.8781,
            longitude: -87.6298,
            name: Some("Chicago, IL".to_string()),
        },
    ];

    info!("📊 Running predictions with ReasoningBank optimization...\n");

    for location in &locations {
        info!("🔍 Location: {}", location.name.as_ref().unwrap());

        // Check cache first
        if let Some(cached) = cache.get_prediction(location) {
            info!("   ✅ Using cached prediction");
            info!("   Temperature: {:.1}°C", cached.temperature);
            continue;
        }

        // Optimize model selection using ReasoningBank
        let optimization = optimizer.optimize_model_selection((
            location.latitude,
            location.longitude,
        ))?;

        info!("   🧠 Optimized model: {:?}", optimization.recommended_model);
        info!("   Confidence: {:.2}%", optimization.confidence * 100.0);
        info!("   Reason: {}", optimization.reason);

        // Create input
        let input = ClimateInput {
            location: location.clone(),
            forecast_hours: 24,
            model_preference: Some(optimization.recommended_model),
        };

        // Fetch current data
        let current_data = api_client.fetch_current(location).await?;
        info!("   📍 Current: {:.1}°C, {:.0}% humidity",
              current_data.temperature, current_data.humidity);

        // Make prediction (simplified - in production this would use the actual model)
        let prediction = climate_core::ClimatePrediction {
            location: location.clone(),
            predicted_at: chrono::Utc::now(),
            prediction_time: chrono::Utc::now() + chrono::Duration::hours(24),
            temperature: current_data.temperature + 2.0,
            humidity: current_data.humidity - 5.0,
            precipitation_probability: 0.2,
            confidence: optimization.confidence,
            model_used: optimization.recommended_model,
        };

        info!("   🔮 Prediction: {:.1}°C (in 24h)", prediction.temperature);

        // Store in cache
        cache.store_prediction(prediction.clone())?;

        // Record result in ReasoningBank for learning
        optimizer.record_prediction_result(&input, &prediction, Some(&current_data))?;

        info!("");
    }

    // Show optimization statistics
    info!("\n📈 ReasoningBank Optimization Statistics:");
    let stats = optimizer.get_optimization_stats()?;
    info!("   Total predictions recorded: {}", stats.total_predictions);
    info!("   Average confidence: {:.2}%", stats.avg_confidence * 100.0);
    info!("   Cache entries: {}", stats.cache_size);
    info!("\n   Predictions by model:");
    for (model, count) in &stats.patterns_by_model {
        info!("     - {}: {}", model, count);
    }

    // Show cache statistics
    let cache_stats = cache.stats();
    info!("\n💾 Cache Statistics:");
    info!("   Cached predictions: {}", cache_stats.prediction_count);
    info!("   Cached climate data: {}", cache_stats.climate_data_count);

    // Finalize hooks
    run_hooks_post_task()?;

    info!("\n✅ Climate prediction complete!");
    info!("💡 ReasoningBank will continue learning from predictions to improve accuracy\n");

    Ok(())
}

/// Run pre-task hooks for coordination
fn run_hooks_pre_task() -> anyhow::Result<()> {
    info!("🔗 Running pre-task hooks...");

    let commands = vec![
        "npx claude-flow@alpha hooks pre-task --description 'Climate prediction with ReasoningBank'",
        "npx claude-flow@alpha hooks session-restore --session-id 'climate-prediction'",
    ];

    for cmd in commands {
        if let Err(e) = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
        {
            error!("Warning: Hook command failed: {}", e);
        }
    }

    Ok(())
}

/// Run post-task hooks for coordination
fn run_hooks_post_task() -> anyhow::Result<()> {
    info!("🔗 Running post-task hooks...");

    let commands = vec![
        "npx claude-flow@alpha hooks post-task --task-id 'climate-prediction'",
        "npx claude-flow@alpha hooks session-end --export-metrics true",
    ];

    for cmd in commands {
        if let Err(e) = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
        {
            error!("Warning: Hook command failed: {}", e);
        }
    }

    Ok(())
}
