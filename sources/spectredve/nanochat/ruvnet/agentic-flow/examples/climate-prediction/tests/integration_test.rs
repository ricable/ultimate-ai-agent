//! Integration tests for climate prediction with ReasoningBank

use climate_core::{
    ClimateApiClient, ClimateInput, Location, ModelType,
    PredictionCache, ReasoningBankOptimizer,
};
use std::path::PathBuf;
use tempfile::tempdir;

#[tokio::test]
async fn test_end_to_end_prediction() {
    // Setup
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut optimizer = ReasoningBankOptimizer::new(db_path, "test").unwrap();
    let cache = PredictionCache::new();
    let api_client = ClimateApiClient::new(None);

    let location = Location {
        latitude: 40.7128,
        longitude: -74.0060,
        name: Some("New York".to_string()),
    };

    // Optimize model selection
    let optimization = optimizer
        .optimize_model_selection((location.latitude, location.longitude))
        .unwrap();

    assert!(optimization.confidence > 0.0);

    // Fetch current data
    let current_data = api_client.fetch_current(&location).await.unwrap();
    assert!(current_data.temperature != 0.0);

    // Make prediction
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

    // Cache prediction
    cache.store_prediction(prediction.clone()).unwrap();

    // Verify cache
    let cached = cache.get_prediction(&location).unwrap();
    assert_eq!(cached.temperature, prediction.temperature);

    // Record result
    let input = ClimateInput {
        location: location.clone(),
        forecast_hours: 24,
        model_preference: Some(optimization.recommended_model),
    };

    optimizer
        .record_prediction_result(&input, &prediction, Some(&current_data))
        .unwrap();

    // Verify learning
    let stats = optimizer.get_optimization_stats().unwrap();
    assert!(stats.total_predictions > 0);
}

#[tokio::test]
async fn test_reasoningbank_learning() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut optimizer = ReasoningBankOptimizer::new(db_path, "test").unwrap();

    let location = (40.7128, -74.0060);

    // First prediction
    let result1 = optimizer.optimize_model_selection(location).unwrap();
    assert_eq!(result1.patterns_analyzed, 0); // No history yet

    // Simulate recording a prediction
    let location_obj = Location {
        latitude: location.0,
        longitude: location.1,
        name: Some("Test".to_string()),
    };

    let input = ClimateInput {
        location: location_obj.clone(),
        forecast_hours: 24,
        model_preference: Some(ModelType::Ensemble),
    };

    let prediction = climate_core::ClimatePrediction {
        location: location_obj.clone(),
        predicted_at: chrono::Utc::now(),
        prediction_time: chrono::Utc::now() + chrono::Duration::hours(24),
        temperature: 25.0,
        humidity: 60.0,
        precipitation_probability: 0.3,
        confidence: 0.85,
        model_used: ModelType::Ensemble,
    };

    let actual = climate_core::ClimateData {
        location: location_obj,
        timestamp: chrono::Utc::now(),
        temperature: 24.5,
        humidity: 58.0,
        pressure: 1013.25,
        wind_speed: 5.0,
        precipitation: 0.0,
        cloud_cover: 30.0,
    };

    optimizer
        .record_prediction_result(&input, &prediction, Some(&actual))
        .unwrap();

    // Clear cache to force pattern lookup
    optimizer.clear_cache();

    // Second prediction should use learned patterns
    let result2 = optimizer.optimize_model_selection(location).unwrap();
    assert!(result2.patterns_analyzed > 0);
    assert!(result2.confidence > 0.5);
}

#[test]
fn test_cache_expiry() {
    use chrono::Duration;

    let cache = PredictionCache::with_ttl(Duration::milliseconds(100));
    let location = Location {
        latitude: 40.7128,
        longitude: -74.0060,
        name: Some("New York".to_string()),
    };

    let prediction = climate_core::ClimatePrediction {
        location: location.clone(),
        predicted_at: chrono::Utc::now(),
        prediction_time: chrono::Utc::now() + Duration::hours(24),
        temperature: 25.0,
        humidity: 60.0,
        precipitation_probability: 0.3,
        confidence: 0.85,
        model_used: ModelType::Ensemble,
    };

    cache.store_prediction(prediction).unwrap();

    // Should be cached
    assert!(cache.get_prediction(&location).is_some());

    // Wait for expiry
    std::thread::sleep(std::time::Duration::from_millis(150));

    // Should be expired
    assert!(cache.get_prediction(&location).is_none());
}
