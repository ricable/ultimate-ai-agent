# Climate Data Ingestion Module

High-performance climate data fetching and processing module for the Climate Prediction System.

## Features

✅ **Multiple Data Sources**
- OpenWeatherMap API (real-time + forecast)
- Open-Meteo API (free, no API key required)
- ERA5 Climate Reanalysis (historical data)

✅ **Production-Ready**
- Async/await with Tokio runtime
- Built-in rate limiting and retry logic
- Automatic caching with TTL
- Comprehensive error handling
- Data validation (range, step, outlier detection)

✅ **Data Quality**
- Range validation (physical limits)
- Step tests (detect unrealistic jumps)
- Persistence tests (outlier detection)
- Temporal continuity checks

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
climate-data = { path = "../climate-data" }
```

## Quick Start

```rust
use climate_data::{
    clients::{ClientConfig, openweathermap::OpenWeatherMapClient, WeatherClient},
    types::Coordinates,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure client
    let config = ClientConfig {
        api_key: Some(std::env::var("OPENWEATHERMAP_API_KEY")?),
        timeout_secs: 30,
        max_retries: 3,
        ..Default::default()
    };

    let client = OpenWeatherMapClient::new(config)?;

    // Fetch current weather
    let location = Coordinates::new(40.7128, -74.0060); // New York
    let weather = client.fetch_current(location).await?;

    println!("Temperature: {}°C", weather.temperature);
    println!("Humidity: {}%", weather.humidity);
    println!("Pressure: {} hPa", weather.pressure);

    Ok(())
}
```

## API Clients

### OpenWeatherMap

Requires API key (free tier: 60 calls/minute).

```rust
use climate_data::clients::{ClientConfig, openweathermap::OpenWeatherMapClient};

let config = ClientConfig {
    api_key: Some(env::var("OPENWEATHERMAP_API_KEY")?),
    ..Default::default()
};

let client = OpenWeatherMapClient::new(config)?;
```

### Open-Meteo (Recommended for Testing)

Free, no API key required!

```rust
use climate_data::clients::{ClientConfig, open_meteo::OpenMeteoClient};

let config = ClientConfig::default();
let client = OpenMeteoClient::new(config)?;

// Fetch current weather
let weather = client.fetch_current(location).await?;

// Fetch forecast (up to 16 days)
let forecast = client.fetch_forecast(location, 168).await?; // 7 days

// Fetch historical data
let time_range = TimeRange::new(start_date, end_date);
let historical = client.fetch_historical(location, time_range).await?;
```

### ERA5 Climate Reanalysis

For historical climate data (1940-present). Requires cloud storage access.

```rust
use climate_data::clients::{ClientConfig, era5::ERA5Client};

let config = ClientConfig {
    base_url: "s3://era5-bucket/".to_string(),
    ..Default::default()
};

let client = ERA5Client::new(config)?;
```

## Data Validation

All fetched data is automatically validated:

```rust
use climate_data::validation::DataValidator;

// Validate single data point
DataValidator::validate_weather_data(&weather)?;

// Validate time series
DataValidator::validate_time_series(&forecast.forecasts)?;

// Validate coordinates
DataValidator::validate_coordinates(lat, lon)?;
```

## Caching

Built-in caching layer with TTL support:

```rust
use climate_data::cache::DataCache;
use std::time::Duration;

// Custom cache (1000 entries, 15 min TTL)
let cache = DataCache::new(1000, Duration::from_secs(15 * 60));

// Check stats
let stats = cache.stats().await;
println!("Cached entries: {}", stats.entry_count);
```

## Error Handling

Rich error types with retry logic:

```rust
use climate_data::ClimateDataError;

match client.fetch_current(location).await {
    Ok(weather) => println!("Success: {:?}", weather),
    Err(ClimateDataError::RateLimitError(msg)) => {
        eprintln!("Rate limited: {}", msg);
    }
    Err(ClimateDataError::ApiError { status_code, message }) => {
        eprintln!("API error {}: {}", status_code, message);
    }
    Err(e) if e.is_retryable() => {
        eprintln!("Retryable error: {}", e);
    }
    Err(e) => eprintln!("Fatal error: {}", e),
}
```

## Environment Variables

```bash
# OpenWeatherMap
export OPENWEATHERMAP_API_KEY="your-api-key"

# ERA5 (if using)
export ERA5_BUCKET_URL="s3://your-bucket/"
```

## Testing

```bash
# Run unit tests
cargo test --lib

# Run integration tests (requires network)
cargo test --test integration_tests -- --ignored

# Run all tests
cargo test --no-fail-fast
```

## Architecture

```
climate-data/
├── src/
│   ├── lib.rs              # Public API and exports
│   ├── types.rs            # Core data structures
│   ├── error.rs            # Error types
│   ├── validation.rs       # Data quality checks
│   ├── cache.rs            # Caching layer
│   └── clients/
│       ├── mod.rs          # Client trait
│       ├── openweathermap.rs
│       ├── open_meteo.rs
│       └── era5.rs
└── tests/
    └── integration_tests.rs
```

## Data Structures

### WeatherData

```rust
pub struct WeatherData {
    pub timestamp: DateTime<Utc>,
    pub location: Coordinates,
    pub temperature: f64,      // Celsius
    pub humidity: f64,         // Percentage
    pub pressure: f64,         // hPa
    pub wind_speed: f64,       // m/s
    pub wind_direction: f64,   // Degrees
    pub precipitation: f64,    // mm
    pub cloud_cover: f64,      // Percentage
    pub visibility: Option<f64>,
    pub uv_index: Option<f64>,
    pub condition_code: Option<i32>,
    pub description: Option<String>,
}
```

### ForecastData

```rust
pub struct ForecastData {
    pub location: Coordinates,
    pub generated_at: DateTime<Utc>,
    pub forecasts: Vec<WeatherData>,
    pub source: DataSource,
}
```

### HistoricalData

```rust
pub struct HistoricalData {
    pub location: Coordinates,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub data_points: Vec<WeatherData>,
    pub source: DataSource,
}
```

## Performance

- **Caching**: 15-minute TTL reduces API calls by ~90%
- **Rate Limiting**: Automatic backoff prevents throttling
- **Parallel Fetching**: Tokio async runtime enables concurrent requests
- **Validation**: Zero-copy validation using validator crate

## Contributing

1. Add new data sources by implementing the `WeatherClient` trait
2. Extend validation rules in `validation.rs`
3. Add tests for new features
4. Update documentation

## License

MIT

## Support

For issues or questions about this module, please file an issue in the main climate-prediction repository.
