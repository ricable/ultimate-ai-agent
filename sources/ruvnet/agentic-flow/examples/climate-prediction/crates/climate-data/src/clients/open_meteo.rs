use crate::{
    cache::{CacheKey, DataCache},
    clients::{ClientConfig, WeatherClient},
    error::{ClimateDataError, Result},
    types::{Coordinates, DataSource, ForecastData, HistoricalData, TimeRange, WeatherData},
    validation::DataValidator,
};
use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::Client;
use serde::Deserialize;

const BASE_URL: &str = "https://api.open-meteo.com/v1";

/// Open-Meteo API client (free, no API key required)
pub struct OpenMeteoClient {
    client: Client,
    cache: DataCache,
    max_retries: u32,
}

impl OpenMeteoClient {
    /// Create a new Open-Meteo client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| ClimateDataError::ConfigError(e.to_string()))?;

        Ok(Self {
            client,
            cache: DataCache::default(),
            max_retries: config.max_retries,
        })
    }

    /// Make API request with retries
    async fn make_request<T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
    ) -> Result<T> {
        let mut attempts = 0;

        loop {
            let response = self.client.get(url).send().await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        return resp.json().await.map_err(Into::into);
                    } else if status.as_u16() == 429 {
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
                    if attempts >= self.max_retries {
                        return Err(e.into());
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempts))).await;
                }
            }
        }
    }

    /// Parse Open-Meteo response to WeatherData
    fn parse_hourly_data(
        &self,
        data: &OpenMeteoResponse,
        location: Coordinates,
        index: usize,
    ) -> Result<WeatherData> {
        let time_str = data.hourly.time.get(index).ok_or_else(|| {
            ClimateDataError::ValidationError("Missing time data".to_string())
        })?;

        let timestamp = NaiveDateTime::parse_from_str(time_str, "%Y-%m-%dT%H:%M")
            .map_err(|e| ClimateDataError::ValidationError(e.to_string()))?
            .and_utc();

        Ok(WeatherData {
            timestamp,
            location,
            temperature: data.hourly.temperature_2m[index],
            humidity: data.hourly.relative_humidity_2m[index],
            pressure: data.hourly.surface_pressure[index],
            wind_speed: data.hourly.wind_speed_10m[index],
            wind_direction: data.hourly.wind_direction_10m[index],
            precipitation: data.hourly.precipitation[index],
            cloud_cover: data.hourly.cloud_cover[index],
            visibility: Some(data.hourly.visibility.as_ref().map(|v| v[index]).unwrap_or(10000.0)),
            uv_index: data.hourly.uv_index.as_ref().map(|uv| uv[index]),
            condition_code: data.hourly.weather_code.as_ref().map(|wc| wc[index] as i32),
            description: None,
        })
    }
}

#[async_trait]
impl WeatherClient for OpenMeteoClient {
    async fn fetch_current(&self, location: Coordinates) -> Result<WeatherData> {
        DataValidator::validate_coordinates(location.latitude, location.longitude)?;

        // Check cache
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
            "{}/forecast?latitude={}&longitude={}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,visibility,uv_index,weather_code&current_weather=true&forecast_days=1",
            BASE_URL, location.latitude, location.longitude
        );

        let response: OpenMeteoResponse = self.make_request(&url).await?;

        // Get current hour data (index 0)
        let weather_data = self.parse_hourly_data(&response, location, 0)?;

        // Validate
        DataValidator::validate_weather_data(&weather_data)?;

        // Cache
        self.cache.set(cache_key, weather_data.clone()).await?;

        Ok(weather_data)
    }

    async fn fetch_forecast(&self, location: Coordinates, hours: u32) -> Result<ForecastData> {
        DataValidator::validate_coordinates(location.latitude, location.longitude)?;

        let forecast_days = (hours / 24).max(1).min(16); // Max 16 days

        let url = format!(
            "{}/forecast?latitude={}&longitude={}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,visibility,uv_index,weather_code&forecast_days={}",
            BASE_URL, location.latitude, location.longitude, forecast_days
        );

        let response: OpenMeteoResponse = self.make_request(&url).await?;

        let mut forecasts = Vec::new();
        let count = hours.min(response.hourly.time.len() as u32) as usize;

        for i in 0..count {
            let data = self.parse_hourly_data(&response, location, i)?;
            forecasts.push(data);
        }

        // Validate
        DataValidator::validate_time_series(&forecasts)?;

        Ok(ForecastData {
            location,
            generated_at: Utc::now(),
            forecasts,
            source: DataSource::OpenMeteo,
        })
    }

    async fn fetch_historical(
        &self,
        location: Coordinates,
        time_range: TimeRange,
    ) -> Result<HistoricalData> {
        DataValidator::validate_coordinates(location.latitude, location.longitude)?;
        time_range.validate().map_err(|e| {
            ClimateDataError::ValidationError(e)
        })?;

        let start_date = time_range.start.format("%Y-%m-%d");
        let end_date = time_range.end.format("%Y-%m-%d");

        let url = format!(
            "{}/archive?latitude={}&longitude={}&start_date={}&end_date={}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
            BASE_URL, location.latitude, location.longitude, start_date, end_date
        );

        let response: OpenMeteoResponse = self.make_request(&url).await?;

        let mut data_points = Vec::new();
        for i in 0..response.hourly.time.len() {
            let data = self.parse_hourly_data(&response, location, i)?;
            data_points.push(data);
        }

        // Validate
        DataValidator::validate_time_series(&data_points)?;

        Ok(HistoricalData {
            location,
            start_date: time_range.start,
            end_date: time_range.end,
            data_points,
            source: DataSource::OpenMeteo,
        })
    }

    fn name(&self) -> &str {
        "Open-Meteo"
    }
}

// Open-Meteo API response structures
#[derive(Debug, Deserialize)]
struct OpenMeteoResponse {
    hourly: OpenMeteoHourly,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoHourly {
    time: Vec<String>,
    temperature_2m: Vec<f64>,
    relative_humidity_2m: Vec<f64>,
    surface_pressure: Vec<f64>,
    wind_speed_10m: Vec<f64>,
    wind_direction_10m: Vec<f64>,
    precipitation: Vec<f64>,
    cloud_cover: Vec<f64>,
    visibility: Option<Vec<f64>>,
    uv_index: Option<Vec<f64>>,
    weather_code: Option<Vec<u32>>,
}
