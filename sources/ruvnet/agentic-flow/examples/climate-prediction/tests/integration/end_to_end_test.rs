// End-to-end integration tests
use std::time::Duration;
use tokio::time::sleep;

mod common;
use common::test_helpers::*;

#[tokio::test]
async fn test_complete_prediction_workflow() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    println!("=== Starting E2E Test: Complete Prediction Workflow ===");

    // Step 1: Fetch historical data
    println!("Step 1: Fetching historical data...");
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: load_fixture("sample_data.json").to_string(),
            headers: vec![],
        })
        .await;

    let historical_response = mock_client.get_next_response().await;
    assert!(historical_response.is_some());
    assert_eq!(historical_response.unwrap().status, 200);

    // Step 2: Process and validate data
    println!("Step 2: Processing and validating data...");
    let data = generate_sample_climate_data(24);
    for point in &data {
        assert!(point.temperature >= -100.0 && point.temperature <= 100.0);
        assert!(point.humidity >= 0.0 && point.humidity <= 100.0);
    }

    // Step 3: Run model inference
    println!("Step 3: Running model inference...");
    let input_features: Vec<f32> = data
        .iter()
        .flat_map(|point| {
            vec![
                point.temperature as f32,
                point.humidity as f32,
                point.pressure as f32,
            ]
        })
        .collect();

    let prediction = mock_predict(&input_features);
    assert!(prediction.is_some());

    // Step 4: Return prediction via API
    println!("Step 4: Returning prediction...");
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: serde_json::json!({
                "predictions": [{
                    "temperature": prediction.unwrap(),
                    "confidence": 0.85
                }]
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let api_response = mock_client.get_next_response().await;
    assert!(api_response.is_some());

    let resp = api_response.unwrap();
    assert_eq!(resp.status, 200);

    let result: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(result["predictions"].is_array());

    println!("=== E2E Test Complete: All steps passed ===");
}

#[tokio::test]
async fn test_error_recovery_workflow() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    println!("=== Testing Error Recovery ===");

    // Simulate API failure
    mock_client
        .add_response(MockResponse {
            status: 503,
            body: r#"{"error": "Service Unavailable"}"#.to_string(),
            headers: vec![],
        })
        .await;

    // Retry should succeed
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: load_fixture("sample_data.json").to_string(),
            headers: vec![],
        })
        .await;

    let mut attempts = 0;
    let mut success = false;

    for _ in 0..config.max_retries {
        attempts += 1;
        if let Some(resp) = mock_client.get_next_response().await {
            if resp.status == 200 {
                success = true;
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    assert!(success, "Should recover from transient errors");
    assert!(attempts <= config.max_retries);

    println!("=== Error Recovery Test Complete ===");
}

#[tokio::test]
async fn test_high_load_scenario() {
    println!("=== Testing High Load Scenario ===");

    let mock_client = MockHttpClient::new();
    let num_requests = 50;

    // Add responses for all requests
    for _ in 0..num_requests {
        mock_client
            .add_response(MockResponse {
                status: 200,
                body: serde_json::json!({"success": true}).to_string(),
                headers: vec![],
            })
            .await;
    }

    // Simulate concurrent requests
    let mut handles = vec![];
    for i in 0..num_requests {
        let client = mock_client.clone();
        handles.push(tokio::spawn(async move {
            let start = std::time::Instant::now();
            let response = client.get_next_response().await;
            let duration = start.elapsed();

            (i, response.is_some(), duration)
        }));
    }

    let results = futures::future::join_all(handles).await;

    let successful = results.iter().filter(|r| r.as_ref().unwrap().1).count();
    let avg_duration: Duration = results
        .iter()
        .map(|r| r.as_ref().unwrap().2)
        .sum::<Duration>()
        / results.len() as u32;

    assert_eq!(successful, num_requests);
    assert!(avg_duration < Duration::from_millis(100));

    println!("=== High Load Test Complete: {}/{} requests successful ===", successful, num_requests);
}

#[tokio::test]
async fn test_data_pipeline_integrity() {
    println!("=== Testing Data Pipeline Integrity ===");

    // Step 1: Generate source data
    let source_data = generate_sample_climate_data(100);
    assert_eq!(source_data.len(), 100);

    // Step 2: Simulate data transformation
    let transformed_data: Vec<_> = source_data
        .iter()
        .map(|point| TransformedData {
            temp_celsius: point.temperature,
            temp_fahrenheit: point.temperature * 9.0 / 5.0 + 32.0,
            humidity_percent: point.humidity,
        })
        .collect();

    assert_eq!(transformed_data.len(), source_data.len());

    // Step 3: Validate transformations
    for (original, transformed) in source_data.iter().zip(transformed_data.iter()) {
        assert_approx_eq(original.temperature, transformed.temp_celsius, 0.01);

        let expected_f = original.temperature * 9.0 / 5.0 + 32.0;
        assert_approx_eq(expected_f, transformed.temp_fahrenheit, 0.01);
    }

    // Step 4: Aggregate statistics
    let avg_temp: f64 = transformed_data
        .iter()
        .map(|d| d.temp_celsius)
        .sum::<f64>()
        / transformed_data.len() as f64;

    assert!(avg_temp > 0.0);

    println!("=== Data Pipeline Integrity Test Complete ===");
}

#[tokio::test]
async fn test_multi_location_predictions() {
    println!("=== Testing Multi-Location Predictions ===");

    let locations = vec![
        (40.7128, -74.0060),  // New York
        (34.0522, -118.2437), // Los Angeles
        (51.5074, -0.1278),   // London
        (35.6762, 139.6503),  // Tokyo
    ];

    let mock_client = MockHttpClient::new();

    for (lat, lon) in &locations {
        mock_client
            .add_response(MockResponse {
                status: 200,
                body: serde_json::json!({
                    "location": {"latitude": lat, "longitude": lon},
                    "prediction": {
                        "temperature": 22.5,
                        "confidence": 0.85
                    }
                })
                .to_string(),
                headers: vec![],
            })
            .await;
    }

    let mut predictions = vec![];
    for _ in 0..locations.len() {
        if let Some(resp) = mock_client.get_next_response().await {
            assert_eq!(resp.status, 200);
            predictions.push(resp);
        }
    }

    assert_eq!(predictions.len(), locations.len());

    println!("=== Multi-Location Test Complete: {} predictions ===", predictions.len());
}

#[tokio::test]
async fn test_time_series_forecast() {
    println!("=== Testing Time Series Forecast ===");

    let historical_data = generate_sample_climate_data(168); // 1 week of hourly data

    // Forecast next 24 hours
    let forecast_horizon = 24;
    let mut forecasts = vec![];

    for i in 0..forecast_horizon {
        let window = &historical_data[i..i + 24];
        let input_features: Vec<f32> = window
            .iter()
            .map(|p| p.temperature as f32)
            .collect();

        let prediction = mock_predict(&input_features);
        forecasts.push(prediction.unwrap());
    }

    assert_eq!(forecasts.len(), forecast_horizon);

    // Validate forecast consistency
    for i in 1..forecasts.len() {
        let diff = (forecasts[i] - forecasts[i - 1]).abs();
        assert!(diff < 10.0, "Forecast should change gradually");
    }

    println!("=== Time Series Forecast Test Complete ===");
}

#[tokio::test]
async fn test_cache_effectiveness() {
    println!("=== Testing Cache Effectiveness ===");

    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let cache: Arc<Mutex<HashMap<String, (String, std::time::Instant)>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let cache_ttl = Duration::from_secs(300); // 5 minutes

    // First request - cache miss
    let key = "weather_nyc".to_string();
    let start1 = std::time::Instant::now();

    {
        let cache_lock = cache.lock().await;
        assert!(cache_lock.get(&key).is_none());
    }

    // Simulate API call
    sleep(Duration::from_millis(50)).await;
    let value = r#"{"temperature": 22.5}"#.to_string();

    {
        let mut cache_lock = cache.lock().await;
        cache_lock.insert(key.clone(), (value.clone(), std::time::Instant::now()));
    }

    let duration1 = start1.elapsed();

    // Second request - cache hit
    let start2 = std::time::Instant::now();

    {
        let cache_lock = cache.lock().await;
        let cached = cache_lock.get(&key);
        assert!(cached.is_some());

        let (cached_value, cached_time) = cached.unwrap();
        assert_eq!(cached_value, &value);
        assert!(cached_time.elapsed() < cache_ttl);
    }

    let duration2 = start2.elapsed();

    // Cache hit should be much faster
    assert!(duration2 < duration1);

    println!("=== Cache Test Complete: Hit {}x faster ===", duration1.as_micros() / duration2.as_micros());
}

// Helper structs and functions
#[derive(Debug, Clone)]
struct TransformedData {
    temp_celsius: f64,
    temp_fahrenheit: f64,
    humidity_percent: f64,
}

fn mock_predict(features: &[f32]) -> Option<f32> {
    if features.is_empty() {
        return None;
    }

    let avg = features.iter().sum::<f32>() / features.len() as f32;
    Some(avg.clamp(-100.0, 100.0))
}
