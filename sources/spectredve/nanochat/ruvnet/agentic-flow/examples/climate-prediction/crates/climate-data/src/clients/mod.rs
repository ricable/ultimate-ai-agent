pub mod openweathermap;
pub mod open_meteo;
pub mod era5;

use crate::{
    error::Result,
    types::{Coordinates, ForecastData, HistoricalData, WeatherData, TimeRange},
};
use async_trait::async_trait;

/// Trait for weather data clients
#[async_trait]
pub trait WeatherClient: Send + Sync {
    /// Fetch current weather data
    async fn fetch_current(&self, location: Coordinates) -> Result<WeatherData>;

    /// Fetch forecast data
    async fn fetch_forecast(&self, location: Coordinates, hours: u32) -> Result<ForecastData>;

    /// Fetch historical data
    async fn fetch_historical(
        &self,
        location: Coordinates,
        time_range: TimeRange,
    ) -> Result<HistoricalData>;

    /// Get client name
    fn name(&self) -> &str;
}

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}
