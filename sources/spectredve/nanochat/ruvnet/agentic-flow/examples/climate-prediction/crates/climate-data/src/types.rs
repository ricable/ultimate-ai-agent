use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Geographic coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
pub struct Coordinates {
    #[validate(range(min = -90.0, max = 90.0))]
    pub latitude: f64,
    #[validate(range(min = -180.0, max = 180.0))]
    pub longitude: f64,
}

impl Coordinates {
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self {
            latitude,
            longitude,
        }
    }
}

/// Weather data point
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct WeatherData {
    pub timestamp: DateTime<Utc>,
    pub location: Coordinates,

    // Temperature (Celsius)
    #[validate(range(min = -100.0, max = 60.0))]
    pub temperature: f64,

    // Humidity (percentage)
    #[validate(range(min = 0.0, max = 100.0))]
    pub humidity: f64,

    // Pressure (hPa)
    #[validate(range(min = 870.0, max = 1085.0))]
    pub pressure: f64,

    // Wind speed (m/s)
    #[validate(range(min = 0.0, max = 113.0))]
    pub wind_speed: f64,

    // Wind direction (degrees)
    #[validate(range(min = 0.0, max = 360.0))]
    pub wind_direction: f64,

    // Precipitation (mm)
    #[validate(range(min = 0.0))]
    pub precipitation: f64,

    // Cloud coverage (percentage)
    #[validate(range(min = 0.0, max = 100.0))]
    pub cloud_cover: f64,

    // Visibility (meters)
    #[validate(range(min = 0.0))]
    pub visibility: Option<f64>,

    // UV index
    #[validate(range(min = 0.0, max = 15.0))]
    pub uv_index: Option<f64>,

    // Weather condition code
    pub condition_code: Option<i32>,

    // Weather description
    pub description: Option<String>,
}

/// Historical climate data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalData {
    pub location: Coordinates,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub data_points: Vec<WeatherData>,
    pub source: DataSource,
}

/// Forecast data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastData {
    pub location: Coordinates,
    pub generated_at: DateTime<Utc>,
    pub forecasts: Vec<WeatherData>,
    pub source: DataSource,
}

/// Data source enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataSource {
    OpenWeatherMap,
    OpenMeteo,
    ERA5,
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::OpenWeatherMap => write!(f, "OpenWeatherMap"),
            DataSource::OpenMeteo => write!(f, "Open-Meteo"),
            DataSource::ERA5 => write!(f, "ERA5"),
        }
    }
}

/// Time range for data queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeRange {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.end <= self.start {
            return Err("End time must be after start time".to_string());
        }
        Ok(())
    }
}

/// Request parameters for fetching weather data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherRequest {
    pub location: Coordinates,
    pub time_range: Option<TimeRange>,
    pub include_forecast: bool,
}
