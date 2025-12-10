use crate::{
    cache::{CacheKey, DataCache},
    clients::{ClientConfig, WeatherClient},
    error::{ClimateDataError, Result},
    types::{Coordinates, DataSource, ForecastData, HistoricalData, TimeRange, WeatherData},
    validation::DataValidator,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use governor::{Quota, RateLimiter, state::direct::NotKeyed, state::InMemoryState, clock::DefaultClock};
use nonzero_ext::nonzero;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::sync::Arc;

const BASE_URL: &str = "https://api.openweathermap.org/data/2.5";

/// OpenWeatherMap API client
pub struct OpenWeatherMapClient {
    api_key: String,
    client: Client,
    cache: DataCache,
    rate_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
    max_retries: u32,
}

impl OpenWeatherMapClient {
    /// Create a new OpenWeatherMap client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let api_key = config.api_key.ok_or_else(|| {
            ClimateDataError::MissingApiKey("OpenWeatherMap".to_string())
        })?;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| ClimateDataError::ConfigError(e.to_string()))?;

        // Rate limit: 60 calls per minute (free tier)
        let quota = Quota::per_minute(nonzero!(60u32));
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Ok(Self {
            api_key,
            client,
            cache: DataCache::default(),
            rate_limiter,
            max_retries: config.max_retries,
        })
    }

    /// Make a rate-limited API request with retries
    async fn make_request<T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
    ) -> Result<T> {
        let mut attempts = 0;

        loop {
            // Rate limiting
            self.rate_limiter.until_ready().await;

            let response = self.client.get(url).send().await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        return resp.json().await.map_err(Into::into);
                    } else if status.as_u16() == 429 {
                        // Rate limit hit
                        attempts += 1;
                        if attempts >= self.max_retries {
                            return Err(ClimateDataError::RateLimitError(
                                "Max retries exceeded".to_string(),
                            ));
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempts))).await;
                    } else {
                        let message = resp.text().await.unwrap_or_default();
                        return Err(ClimateDataError::ApiError {
                            status_code: status.as_u16(),
                            message,
                        });
                    }
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_retries || !e.is_timeout() && !e.is_connect() {
                        return Err(e.into());
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempts))).await;
                }
            }
        }
    }

    /// Parse OpenWeatherMap response to WeatherData
    fn parse_current_response(&self, data: OWMCurrentResponse, location: Coordinates) -> WeatherData {
        WeatherData {
            timestamp: DateTime::from_timestamp(data.dt, 0).unwrap_or_else(Utc::now),
            location,
            temperature: data.main.temp,
            humidity: data.main.humidity as f64,
            pressure: data.main.pressure,
            wind_speed: data.wind.speed,
            wind_direction: data.wind.deg.unwrap_or(0.0),
            precipitation: data.rain.map(|r| r.one_hour).unwrap_or(0.0),
            cloud_cover: data.clouds.all as f64,
            visibility: Some(data.visibility as f64),
            uv_index: None, // Requires separate API call
            condition_code: data.weather.first().map(|w| w.id),
            description: data.weather.first().map(|w| w.description.clone()),
        }
    }
}

#[async_trait]
impl WeatherClient for OpenWeatherMapClient {
    async fn fetch_current(&self, location: Coordinates) -> Result<WeatherData> {
        DataValidator::validate_coordinates(location.latitude, location.longitude)?;

        // Check cache first
        let cache_key = CacheKey::from_coords(
            location.latitude,
            location.longitude,
            Utc::now().timestamp(),
        );

        if let Some(cached) = self.cache.get(&cache_key).await {
            tracing::debug!("Cache hit for {:?}", location);
            return Ok(cached);
        }

        let url = format!(
            "{}/weather?lat={}&lon={}&units=metric&appid={}",
            BASE_URL, location.latitude, location.longitude, self.api_key
        );

        let response: OWMCurrentResponse = self.make_request(&url).await?;
        let weather_data = self.parse_current_response(response, location);

        // Validate data
        DataValidator::validate_weather_data(&weather_data)?;

        // Cache the result
        self.cache.set(cache_key, weather_data.clone()).await?;

        Ok(weather_data)
    }

    async fn fetch_forecast(&self, location: Coordinates, hours: u32) -> Result<ForecastData> {
        DataValidator::validate_coordinates(location.latitude, location.longitude)?;

        let url = format!(
            "{}/forecast?lat={}&lon={}&units=metric&appid={}",
            BASE_URL, location.latitude, location.longitude, self.api_key
        );

        let response: OWMForecastResponse = self.make_request(&url).await?;

        let forecasts: Vec<WeatherData> = response
            .list
            .into_iter()
            .take((hours / 3) as usize) // OWM returns 3-hour intervals
            .map(|item| self.parse_forecast_item(item, location))
            .collect();

        // Validate time series
        DataValidator::validate_time_series(&forecasts)?;

        Ok(ForecastData {
            location,
            generated_at: Utc::now(),
            forecasts,
            source: DataSource::OpenWeatherMap,
        })
    }

    async fn fetch_historical(
        &self,
        _location: Coordinates,
        _time_range: TimeRange,
    ) -> Result<HistoricalData> {
        // OpenWeatherMap historical data requires paid subscription
        // This is a placeholder implementation
        Err(ClimateDataError::ConfigError(
            "Historical data requires OpenWeatherMap paid subscription".to_string(),
        ))
    }

    fn name(&self) -> &str {
        "OpenWeatherMap"
    }
}

impl OpenWeatherMapClient {
    fn parse_forecast_item(&self, item: OWMForecastItem, location: Coordinates) -> WeatherData {
        WeatherData {
            timestamp: DateTime::from_timestamp(item.dt, 0).unwrap_or_else(Utc::now),
            location,
            temperature: item.main.temp,
            humidity: item.main.humidity as f64,
            pressure: item.main.pressure,
            wind_speed: item.wind.speed,
            wind_direction: item.wind.deg.unwrap_or(0.0),
            precipitation: item.rain.map(|r| r.three_hour).unwrap_or(0.0),
            cloud_cover: item.clouds.all as f64,
            visibility: Some(item.visibility.unwrap_or(10000) as f64),
            uv_index: None,
            condition_code: item.weather.first().map(|w| w.id),
            description: item.weather.first().map(|w| w.description.clone()),
        }
    }
}

// OpenWeatherMap API response structures
#[derive(Debug, Deserialize)]
struct OWMCurrentResponse {
    dt: i64,
    main: OWMMain,
    wind: OWMWind,
    clouds: OWMClouds,
    rain: Option<OWMRain>,
    visibility: u32,
    weather: Vec<OWMWeather>,
}

#[derive(Debug, Deserialize)]
struct OWMForecastResponse {
    list: Vec<OWMForecastItem>,
}

#[derive(Debug, Deserialize)]
struct OWMForecastItem {
    dt: i64,
    main: OWMMain,
    wind: OWMWind,
    clouds: OWMClouds,
    rain: Option<OWMRain3h>,
    visibility: Option<u32>,
    weather: Vec<OWMWeather>,
}

#[derive(Debug, Deserialize)]
struct OWMMain {
    temp: f64,
    humidity: u32,
    pressure: f64,
}

#[derive(Debug, Deserialize)]
struct OWMWind {
    speed: f64,
    deg: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OWMClouds {
    all: u32,
}

#[derive(Debug, Deserialize)]
struct OWMRain {
    #[serde(rename = "1h")]
    one_hour: f64,
}

#[derive(Debug, Deserialize)]
struct OWMRain3h {
    #[serde(rename = "3h")]
    three_hour: f64,
}

#[derive(Debug, Deserialize)]
struct OWMWeather {
    id: i32,
    description: String,
}
