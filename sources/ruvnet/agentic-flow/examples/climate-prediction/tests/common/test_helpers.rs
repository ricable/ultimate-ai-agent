// Common test utilities and helpers
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Mock HTTP client for testing
pub struct MockHttpClient {
    responses: Arc<Mutex<Vec<MockResponse>>>,
}

#[derive(Clone)]
pub struct MockResponse {
    pub status: u16,
    pub body: String,
    pub headers: Vec<(String, String)>,
}

impl MockHttpClient {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn add_response(&self, response: MockResponse) {
        let mut responses = self.responses.lock().await;
        responses.push(response);
    }

    pub async fn get_next_response(&self) -> Option<MockResponse> {
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            None
        } else {
            Some(responses.remove(0))
        }
    }
}

/// Generate sample climate data for testing
pub fn generate_sample_climate_data(num_points: usize) -> Vec<ClimateDataPoint> {
    (0..num_points)
        .map(|i| ClimateDataPoint {
            timestamp: chrono::Utc::now() - chrono::Duration::days(i as i64),
            temperature: 20.0 + (i as f64 * 0.1),
            humidity: 60.0 + (i as f64 * 0.05),
            pressure: 1013.25 + (i as f64 * 0.01),
            wind_speed: 5.0 + (i as f64 * 0.02),
            precipitation: if i % 3 == 0 { 2.5 } else { 0.0 },
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct ClimateDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub temperature: f64,
    pub humidity: f64,
    pub pressure: f64,
    pub wind_speed: f64,
    pub precipitation: f64,
}

/// Create a test configuration
pub fn create_test_config() -> TestConfig {
    TestConfig {
        api_key: "test_api_key_12345".to_string(),
        base_url: "http://localhost:8080".to_string(),
        timeout_seconds: 30,
        max_retries: 3,
        model_path: "./tests/fixtures/test_model.onnx".to_string(),
    }
}

#[derive(Clone)]
pub struct TestConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub model_path: String,
}

/// Assert that two floats are approximately equal
pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
    assert!(
        (a - b).abs() < epsilon,
        "Values not approximately equal: {} vs {} (epsilon: {})",
        a,
        b,
        epsilon
    );
}

/// Wait for a condition to be true with timeout
pub async fn wait_for_condition<F>(mut condition: F, timeout_ms: u64) -> bool
where
    F: FnMut() -> bool,
{
    let start = std::time::Instant::now();
    loop {
        if condition() {
            return true;
        }
        if start.elapsed().as_millis() > timeout_ms as u128 {
            return false;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

/// Create a temporary directory for test files
pub fn create_temp_test_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("Failed to create temp directory")
}

/// Load JSON fixture from file
pub fn load_fixture(name: &str) -> Value {
    let path = format!("./tests/fixtures/{}", name);
    let content = std::fs::read_to_string(&path)
        .expect(&format!("Failed to load fixture: {}", path));
    serde_json::from_str(&content).expect("Failed to parse fixture JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_generation() {
        let data = generate_sample_climate_data(10);
        assert_eq!(data.len(), 10);
        assert!(data[0].temperature > 0.0);
    }

    #[test]
    fn test_approx_eq() {
        assert_approx_eq(1.0, 1.0001, 0.001);
    }

    #[tokio::test]
    async fn test_mock_http_client() {
        let client = MockHttpClient::new();
        client
            .add_response(MockResponse {
                status: 200,
                body: "test".to_string(),
                headers: vec![],
            })
            .await;

        let response = client.get_next_response().await;
        assert!(response.is_some());
        assert_eq!(response.unwrap().status, 200);
    }
}
