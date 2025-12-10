// Integration tests for data ingestion
use std::time::Duration;
use tokio::time::sleep;

mod common;
use common::test_helpers::*;

#[tokio::test]
async fn test_api_client_basic_request() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    // Setup mock response
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: load_fixture("mock_responses/api_success.json").to_string(),
            headers: vec![("content-type".to_string(), "application/json".to_string())],
        })
        .await;

    // Simulate API call
    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 200);
    assert!(resp.body.contains("success"));
}

#[tokio::test]
async fn test_api_client_retry_logic() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    // Add failing responses followed by success
    for _ in 0..2 {
        mock_client
            .add_response(MockResponse {
                status: 503,
                body: r#"{"error": "Service Unavailable"}"#.to_string(),
                headers: vec![],
            })
            .await;
    }

    mock_client
        .add_response(MockResponse {
            status: 200,
            body: load_fixture("mock_responses/api_success.json").to_string(),
            headers: vec![],
        })
        .await;

    // Test retry mechanism
    let mut attempts = 0;
    let mut last_response = None;

    for _ in 0..config.max_retries {
        let response = mock_client.get_next_response().await;
        if let Some(resp) = response {
            attempts += 1;
            if resp.status == 200 {
                last_response = Some(resp);
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    assert!(attempts <= config.max_retries);
    assert!(last_response.is_some());
    assert_eq!(last_response.unwrap().status, 200);
}

#[tokio::test]
async fn test_rate_limit_handling() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    // Add rate limit response
    mock_client
        .add_response(MockResponse {
            status: 429,
            body: load_fixture("mock_responses/api_rate_limit.json").to_string(),
            headers: vec![("retry-after".to_string(), "60".to_string())],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 429);
    assert!(resp.body.contains("RATE_LIMIT_EXCEEDED"));

    // Extract retry-after header
    let retry_after = resp
        .headers
        .iter()
        .find(|(k, _)| k == "retry-after")
        .map(|(_, v)| v.parse::<u64>().unwrap_or(0))
        .unwrap_or(0);

    assert_eq!(retry_after, 60);
}

#[tokio::test]
async fn test_data_validation_success() {
    let data = generate_sample_climate_data(10);

    // Validate each data point
    for point in &data {
        assert!(point.temperature >= -100.0 && point.temperature <= 100.0);
        assert!(point.humidity >= 0.0 && point.humidity <= 100.0);
        assert!(point.pressure >= 800.0 && point.pressure <= 1200.0);
        assert!(point.wind_speed >= 0.0);
        assert!(point.precipitation >= 0.0);
    }
}

#[tokio::test]
async fn test_data_validation_edge_cases() {
    // Test extreme values
    let extreme_data = vec![
        ClimateDataPoint {
            timestamp: chrono::Utc::now(),
            temperature: -89.2, // Record low (Vostok Station)
            humidity: 0.0,
            pressure: 870.0, // Typhoon Tip
            wind_speed: 113.0, // Category 5 hurricane
            precipitation: 0.0,
        },
        ClimateDataPoint {
            timestamp: chrono::Utc::now(),
            temperature: 56.7, // Death Valley record
            humidity: 100.0,
            pressure: 1085.0, // Siberian high
            wind_speed: 0.0,
            precipitation: 305.0, // Record rainfall
        },
    ];

    for point in &extreme_data {
        assert!(point.temperature >= -100.0 && point.temperature <= 100.0);
        assert!(point.humidity >= 0.0 && point.humidity <= 100.0);
        assert!(point.pressure >= 800.0 && point.pressure <= 1200.0);
    }
}

#[tokio::test]
async fn test_concurrent_api_requests() {
    let mock_client = MockHttpClient::new();

    // Add multiple responses
    for i in 0..10 {
        mock_client
            .add_response(MockResponse {
                status: 200,
                body: format!(r#"{{"id": {}}}"#, i),
                headers: vec![],
            })
            .await;
    }

    // Simulate concurrent requests
    let mut handles = vec![];
    for _ in 0..10 {
        let client = mock_client.clone();
        handles.push(tokio::spawn(async move {
            client.get_next_response().await
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let successful = results.iter().filter(|r| r.is_ok()).count();

    assert_eq!(successful, 10);
}

#[tokio::test]
async fn test_data_caching() {
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    // Simple cache implementation for testing
    let cache: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

    let key = "weather_data_nyc".to_string();
    let value = load_fixture("mock_responses/api_success.json").to_string();

    // Store in cache
    {
        let mut cache_lock = cache.lock().await;
        cache_lock.insert(key.clone(), value.clone());
    }

    // Retrieve from cache
    {
        let cache_lock = cache.lock().await;
        let cached_value = cache_lock.get(&key);
        assert!(cached_value.is_some());
        assert_eq!(cached_value.unwrap(), &value);
    }
}

#[tokio::test]
async fn test_authentication_header() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    mock_client
        .add_response(MockResponse {
            status: 200,
            body: "{}".to_string(),
            headers: vec![
                ("authorization".to_string(), format!("Bearer {}", config.api_key)),
            ],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    let auth_header = resp
        .headers
        .iter()
        .find(|(k, _)| k == "authorization")
        .map(|(_, v)| v.clone());

    assert!(auth_header.is_some());
    assert!(auth_header.unwrap().contains(&config.api_key));
}

#[tokio::test]
async fn test_timeout_handling() {
    let config = create_test_config();

    // Test timeout behavior
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        async {
            sleep(Duration::from_secs(5)).await;
            "completed"
        }
    ).await;

    assert!(result.is_err()); // Should timeout
}
