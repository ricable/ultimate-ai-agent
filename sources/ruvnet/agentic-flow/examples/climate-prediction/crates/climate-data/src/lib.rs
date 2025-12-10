//! # Climate Data Ingestion Module
//!
//! This crate provides a unified interface for fetching climate and weather data
//! from multiple sources including OpenWeatherMap, Open-Meteo, and ERA5.
//!
//! ## Features
//!
//! - **Multiple Data Sources**: Support for OpenWeatherMap, Open-Meteo, and ERA5
//! - **Caching Layer**: Automatic caching with TTL support
//! - **Rate Limiting**: Built-in rate limiting for API clients
//! - **Data Validation**: Comprehensive validation including range, step, and outlier detection
//! - **Async/Await**: Fully asynchronous API using tokio
//! - **Error Handling**: Rich error types with retry logic
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use climate_data::{
//!     clients::{ClientConfig, openweathermap::OpenWeatherMapClient, WeatherClient},
//!     types::Coordinates,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize client
//!     let config = ClientConfig {
//!         api_key: Some(std::env::var("OPENWEATHERMAP_API_KEY")?),
//!         base_url: String::new(),
//!         timeout_secs: 30,
//!         max_retries: 3,
//!     };
//!
//!     let client = OpenWeatherMapClient::new(config)?;
//!
//!     // Fetch current weather
//!     let location = Coordinates::new(40.7128, -74.0060); // New York
//!     let weather = client.fetch_current(location).await?;
//!
//!     println!("Temperature: {}°C", weather.temperature);
//!     println!("Humidity: {}%", weather.humidity);
//!
//!     Ok(())
//! }
//! ```

pub mod cache;
pub mod clients;
pub mod error;
pub mod types;
pub mod validation;

pub use cache::DataCache;
pub use error::{ClimateDataError, Result};
pub use types::{
    Coordinates, DataSource, ForecastData, HistoricalData, TimeRange, WeatherData, WeatherRequest,
};
pub use validation::DataValidator;

/// Initialize logging for the climate-data crate
pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_target(true)
        .with_level(true)
        .init();
}

/// Load configuration from environment variables
pub fn load_config_from_env() -> std::collections::HashMap<String, String> {
    let mut config = std::collections::HashMap::new();

    if let Ok(key) = std::env::var("OPENWEATHERMAP_API_KEY") {
        config.insert("openweathermap_api_key".to_string(), key);
    }

    if let Ok(url) = std::env::var("ERA5_BUCKET_URL") {
        config.insert("era5_bucket_url".to_string(), url);
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_exports() {
        // Verify all main types are exported
        let _coords = Coordinates::new(0.0, 0.0);
        let _cache = DataCache::default();
    }
}
