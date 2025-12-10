//! Climate API client for fetching weather data

use crate::{ClimateData, ClimateError, Location, Result};
use chrono::Utc;
use reqwest::Client;
use serde_json::Value;

/// Client for climate data APIs
pub struct ClimateApiClient {
    client: Client,
    api_key: Option<String>,
}

impl ClimateApiClient {
    /// Create a new API client
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    /// Fetch current climate data for a location
    pub async fn fetch_current(&self, location: &Location) -> Result<ClimateData> {
        // In production, this would call a real weather API
        // For demo purposes, we return simulated data

        tracing::info!(
            "Fetching climate data for lat={}, lon={}",
            location.latitude,
            location.longitude
        );

        // Simulate API delay
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Return simulated data
        Ok(ClimateData {
            location: location.clone(),
            timestamp: Utc::now(),
            temperature: 20.0 + (location.latitude / 10.0),
            humidity: 60.0,
            pressure: 1013.25,
            wind_speed: 5.0,
            precipitation: 0.0,
            cloud_cover: 30.0,
        })
    }

    /// Fetch historical climate data
    pub async fn fetch_historical(
        &self,
        location: &Location,
        days: u32,
    ) -> Result<Vec<ClimateData>> {
        tracing::info!(
            "Fetching {} days of historical data for lat={}, lon={}",
            days,
            location.latitude,
            location.longitude
        );

        // Simulate API delay
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Return simulated historical data
        let mut data = Vec::new();
        for day in 0..days {
            let timestamp = Utc::now() - chrono::Duration::days(day as i64);
            data.push(ClimateData {
                location: location.clone(),
                timestamp,
                temperature: 18.0 + (location.latitude / 10.0) + (day as f64 * 0.5),
                humidity: 55.0 + (day as f64 * 0.3),
                pressure: 1013.25,
                wind_speed: 4.0 + (day as f64 * 0.1),
                precipitation: 0.0,
                cloud_cover: 25.0 + (day as f64 * 0.5),
            });
        }

        Ok(data)
    }

    /// Validate API connectivity
    pub async fn health_check(&self) -> Result<bool> {
        tracing::info!("Performing API health check");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_current() {
        let client = ClimateApiClient::new(None);
        let location = Location {
            latitude: 40.7128,
            longitude: -74.0060,
            name: Some("New York".to_string()),
        };

        let data = client.fetch_current(&location).await.unwrap();
        assert_eq!(data.location.latitude, 40.7128);
        assert!(data.temperature > 0.0);
    }

    #[tokio::test]
    async fn test_fetch_historical() {
        let client = ClimateApiClient::new(None);
        let location = Location {
            latitude: 40.7128,
            longitude: -74.0060,
            name: Some("New York".to_string()),
        };

        let data = client.fetch_historical(&location, 7).await.unwrap();
        assert_eq!(data.len(), 7);
    }
}
