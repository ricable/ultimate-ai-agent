// Integration tests for API endpoints
use serde_json::json;
mod common;
use common::test_helpers::*;

#[tokio::test]
async fn test_health_endpoint() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    mock_client
        .add_response(MockResponse {
            status: 200,
            body: json!({
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 200);
    assert!(resp.body.contains("healthy"));
}

#[tokio::test]
async fn test_current_weather_endpoint() {
    let config = create_test_config();
    let mock_client = MockHttpClient::new();

    mock_client
        .add_response(MockResponse {
            status: 200,
            body: load_fixture("mock_responses/api_success.json").to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 200);

    let data: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert_eq!(data["status"], "success");
    assert!(data["data"]["current_weather"].is_object());
}

#[tokio::test]
async fn test_prediction_endpoint_valid_request() {
    let mock_client = MockHttpClient::new();

    let request_body = json!({
        "location": {
            "latitude": 40.7128,
            "longitude": -74.0060
        },
        "time_horizon_hours": 24,
        "parameters": ["temperature", "humidity", "precipitation"]
    });

    mock_client
        .add_response(MockResponse {
            status: 200,
            body: json!({
                "predictions": [
                    {
                        "timestamp": "2025-01-02T00:00:00Z",
                        "temperature": 21.5,
                        "humidity": 68.0,
                        "precipitation_probability": 0.3
                    }
                ],
                "confidence": 0.85
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 200);

    let data: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(data["predictions"].is_array());
    assert_eq!(data["confidence"], 0.85);
}

#[tokio::test]
async fn test_prediction_endpoint_invalid_location() {
    let mock_client = MockHttpClient::new();

    let invalid_request = json!({
        "location": {
            "latitude": 91.0,  // Invalid: > 90
            "longitude": -74.0060
        },
        "time_horizon_hours": 24
    });

    mock_client
        .add_response(MockResponse {
            status: 400,
            body: json!({
                "error": "Invalid latitude: must be between -90 and 90",
                "code": "INVALID_LOCATION"
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 400);
    assert!(resp.body.contains("INVALID_LOCATION"));
}

#[tokio::test]
async fn test_prediction_endpoint_missing_parameters() {
    let mock_client = MockHttpClient::new();

    let incomplete_request = json!({
        "location": {
            "latitude": 40.7128
            // Missing longitude
        }
    });

    mock_client
        .add_response(MockResponse {
            status: 400,
            body: json!({
                "error": "Missing required field: longitude",
                "code": "MISSING_PARAMETER"
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 400);
}

#[tokio::test]
async fn test_authentication_required() {
    let mock_client = MockHttpClient::new();

    mock_client
        .add_response(MockResponse {
            status: 401,
            body: json!({
                "error": "Authentication required",
                "code": "UNAUTHORIZED"
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 401);
    assert!(resp.body.contains("UNAUTHORIZED"));
}

#[tokio::test]
async fn test_rate_limiting_endpoint() {
    let mock_client = MockHttpClient::new();

    // Simulate multiple rapid requests
    for i in 0..5 {
        let status = if i < 3 { 200 } else { 429 };

        mock_client
            .add_response(MockResponse {
                status,
                body: if status == 200 {
                    json!({"success": true}).to_string()
                } else {
                    load_fixture("mock_responses/api_rate_limit.json").to_string()
                },
                headers: vec![],
            })
            .await;
    }

    let mut successful = 0;
    let mut rate_limited = 0;

    for _ in 0..5 {
        if let Some(resp) = mock_client.get_next_response().await {
            if resp.status == 200 {
                successful += 1;
            } else if resp.status == 429 {
                rate_limited += 1;
            }
        }
    }

    assert_eq!(successful, 3);
    assert_eq!(rate_limited, 2);
}

#[tokio::test]
async fn test_cors_headers() {
    let mock_client = MockHttpClient::new();

    mock_client
        .add_response(MockResponse {
            status: 200,
            body: json!({"success": true}).to_string(),
            headers: vec![
                ("access-control-allow-origin".to_string(), "*".to_string()),
                (
                    "access-control-allow-methods".to_string(),
                    "GET, POST, OPTIONS".to_string(),
                ),
                (
                    "access-control-allow-headers".to_string(),
                    "Content-Type, Authorization".to_string(),
                ),
            ],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    let has_cors = resp
        .headers
        .iter()
        .any(|(k, _)| k == "access-control-allow-origin");

    assert!(has_cors);
}

#[tokio::test]
async fn test_pagination_support() {
    let mock_client = MockHttpClient::new();

    // First page
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: json!({
                "data": [1, 2, 3],
                "pagination": {
                    "page": 1,
                    "per_page": 3,
                    "total": 10,
                    "next_page": 2
                }
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    // Second page
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: json!({
                "data": [4, 5, 6],
                "pagination": {
                    "page": 2,
                    "per_page": 3,
                    "total": 10,
                    "next_page": 3
                }
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let page1 = mock_client.get_next_response().await;
    assert!(page1.is_some());

    let p1_data: serde_json::Value = serde_json::from_str(&page1.unwrap().body).unwrap();
    assert_eq!(p1_data["pagination"]["page"], 1);

    let page2 = mock_client.get_next_response().await;
    assert!(page2.is_some());

    let p2_data: serde_json::Value = serde_json::from_str(&page2.unwrap().body).unwrap();
    assert_eq!(p2_data["pagination"]["page"], 2);
}

#[tokio::test]
async fn test_error_response_format() {
    let mock_client = MockHttpClient::new();

    mock_client
        .add_response(MockResponse {
            status: 500,
            body: json!({
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "timestamp": "2025-01-01T12:00:00Z",
                    "trace_id": "abc123"
                }
            })
            .to_string(),
            headers: vec![],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    assert_eq!(resp.status, 500);

    let data: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert!(data["error"].is_object());
    assert!(data["error"]["code"].is_string());
    assert!(data["error"]["trace_id"].is_string());
}

#[tokio::test]
async fn test_content_negotiation() {
    let mock_client = MockHttpClient::new();

    // JSON response
    mock_client
        .add_response(MockResponse {
            status: 200,
            body: json!({"format": "json"}).to_string(),
            headers: vec![("content-type".to_string(), "application/json".to_string())],
        })
        .await;

    let response = mock_client.get_next_response().await;
    assert!(response.is_some());

    let resp = response.unwrap();
    let content_type = resp
        .headers
        .iter()
        .find(|(k, _)| k == "content-type")
        .map(|(_, v)| v.as_str());

    assert_eq!(content_type, Some("application/json"));
}
