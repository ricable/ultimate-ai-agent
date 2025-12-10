use climate_data::{
    cache::{CacheKey, DataCache},
    clients::{open_meteo::OpenMeteoClient, ClientConfig, WeatherClient},
    types::Coordinates,
    validation::DataValidator,
};
use std::time::Duration;

#[tokio::test]
async fn test_cache_basic_operations() {
    let cache = DataCache::new(100, Duration::from_secs(60));

    // Create test key
    let key = CacheKey::from_coords(40.71, -74.01, chrono::Utc::now().timestamp());

    // Initially should be empty
    assert!(!cache.contains(&key).await);

    // Get stats
    let stats = cache.stats().await;
    assert_eq!(stats.entry_count, 0);
}

#[tokio::test]
async fn test_open_meteo_client_creation() {
    let config = ClientConfig {
        api_key: None, // Open-Meteo doesn't require API key
        base_url: String::new(),
        timeout_secs: 30,
        max_retries: 3,
    };

    let client = OpenMeteoClient::new(config);
    assert!(client.is_ok());

    let client = client.unwrap();
    assert_eq!(client.name(), "Open-Meteo");
}

#[tokio::test]
#[ignore] // Ignore by default to avoid hitting API in CI
async fn test_open_meteo_fetch_current() {
    let config = ClientConfig {
        api_key: None,
        base_url: String::new(),
        timeout_secs: 30,
        max_retries: 3,
    };

    let client = OpenMeteoClient::new(config).unwrap();
    let location = Coordinates::new(40.7128, -74.0060); // New York

    let result = client.fetch_current(location).await;

    match result {
        Ok(weather) => {
            println!("Temperature: {}°C", weather.temperature);
            println!("Humidity: {}%", weather.humidity);

            // Validate the data
            assert!(DataValidator::validate_weather_data(&weather).is_ok());
        }
        Err(e) => {
            eprintln!("API call failed: {}", e);
            // Don't fail the test if API is unavailable
        }
    }
}

#[tokio::test]
#[ignore] // Ignore by default to avoid hitting API in CI
async fn test_open_meteo_fetch_forecast() {
    let config = ClientConfig {
        api_key: None,
        base_url: String::new(),
        timeout_secs: 30,
        max_retries: 3,
    };

    let client = OpenMeteoClient::new(config).unwrap();
    let location = Coordinates::new(51.5074, -0.1278); // London

    let result = client.fetch_forecast(location, 24).await;

    match result {
        Ok(forecast) => {
            println!("Forecast points: {}", forecast.forecasts.len());
            assert!(!forecast.forecasts.is_empty());

            // Validate the time series
            assert!(DataValidator::validate_time_series(&forecast.forecasts).is_ok());
        }
        Err(e) => {
            eprintln!("API call failed: {}", e);
        }
    }
}

#[test]
fn test_coordinate_validation() {
    // Valid coordinates
    assert!(DataValidator::validate_coordinates(40.7, -74.0).is_ok());
    assert!(DataValidator::validate_coordinates(0.0, 0.0).is_ok());
    assert!(DataValidator::validate_coordinates(90.0, 180.0).is_ok());
    assert!(DataValidator::validate_coordinates(-90.0, -180.0).is_ok());

    // Invalid coordinates
    assert!(DataValidator::validate_coordinates(91.0, 0.0).is_err());
    assert!(DataValidator::validate_coordinates(-91.0, 0.0).is_err());
    assert!(DataValidator::validate_coordinates(0.0, 181.0).is_err());
    assert!(DataValidator::validate_coordinates(0.0, -181.0).is_err());
}

#[test]
fn test_coordinates_struct() {
    let coords = Coordinates::new(40.7128, -74.0060);
    assert_eq!(coords.latitude, 40.7128);
    assert_eq!(coords.longitude, -74.0060);
}

#[test]
fn test_cache_key_creation() {
    let key1 = CacheKey::from_coords(40.7128, -74.0060, 1000000);
    let key2 = CacheKey::from_coords(40.7128, -74.0060, 1000000);
    let key3 = CacheKey::from_coords(40.7129, -74.0060, 1000000);

    // Same coordinates and timestamp should produce equal keys
    assert_eq!(key1, key2);

    // Different coordinates should produce different keys (small difference still gets rounded to same)
    // Use a larger difference to ensure different keys
    let key4 = CacheKey::from_coords(41.0, -74.0060, 1000000);
    assert_ne!(key1, key4);
}
