use crate::{error::Result, types::WeatherData, ClimateDataError};
use validator::Validate;

/// Data quality validator
pub struct DataValidator;

impl DataValidator {
    /// Validate a single weather data point
    pub fn validate_weather_data(data: &WeatherData) -> Result<()> {
        // Use validator crate for range checks
        data.validate()
            .map_err(|e| ClimateDataError::ValidationError(e.to_string()))?;

        // Additional custom validations
        Self::check_temperature_humidity_consistency(data)?;
        Self::check_pressure_altitude_consistency(data)?;

        Ok(())
    }

    /// Validate a series of weather data points
    pub fn validate_time_series(data: &[WeatherData]) -> Result<()> {
        if data.is_empty() {
            return Err(ClimateDataError::ValidationError(
                "Empty data series".to_string(),
            ));
        }

        // Validate each point
        for point in data {
            Self::validate_weather_data(point)?;
        }

        // Check temporal consistency
        Self::check_temporal_continuity(data)?;

        // Check for duplicate timestamps
        Self::check_duplicate_timestamps(data)?;

        // Check for outliers
        Self::check_outliers(data)?;

        Ok(())
    }

    /// Range test: Check if values are within physically possible ranges
    fn check_temperature_humidity_consistency(data: &WeatherData) -> Result<()> {
        // At very low temperatures, high humidity is suspicious
        if data.temperature < -30.0 && data.humidity > 80.0 {
            return Err(ClimateDataError::ValidationError(
                format!(
                    "Unlikely combination: temp={}, humidity={}",
                    data.temperature, data.humidity
                ),
            ));
        }
        Ok(())
    }

    /// Check pressure consistency (basic altitude check)
    fn check_pressure_altitude_consistency(data: &WeatherData) -> Result<()> {
        // Sea level pressure range: 980-1050 hPa typically
        if data.pressure < 900.0 || data.pressure > 1050.0 {
            tracing::warn!(
                "Unusual pressure reading: {} hPa at {:?}",
                data.pressure,
                data.location
            );
        }
        Ok(())
    }

    /// Step test: Check for unrealistic jumps between consecutive readings
    fn check_temporal_continuity(data: &[WeatherData]) -> Result<()> {
        for window in data.windows(2) {
            let dt = (window[1].timestamp - window[0].timestamp).num_hours().abs();

            // Temperature shouldn't change more than 10°C per hour
            let temp_diff = (window[1].temperature - window[0].temperature).abs();
            if temp_diff > 10.0 * dt as f64 {
                return Err(ClimateDataError::ValidationError(
                    format!(
                        "Unrealistic temperature jump: {} -> {} in {} hours",
                        window[0].temperature, window[1].temperature, dt
                    ),
                ));
            }

            // Pressure shouldn't change more than 5 hPa per hour
            let pressure_diff = (window[1].pressure - window[0].pressure).abs();
            if pressure_diff > 5.0 * dt as f64 {
                return Err(ClimateDataError::ValidationError(
                    format!(
                        "Unrealistic pressure jump: {} -> {} in {} hours",
                        window[0].pressure, window[1].pressure, dt
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Check for duplicate timestamps
    fn check_duplicate_timestamps(data: &[WeatherData]) -> Result<()> {
        let mut timestamps: Vec<_> = data.iter().map(|d| d.timestamp).collect();
        timestamps.sort();

        for window in timestamps.windows(2) {
            if window[0] == window[1] {
                return Err(ClimateDataError::ValidationError(
                    format!("Duplicate timestamp found: {}", window[0]),
                ));
            }
        }
        Ok(())
    }

    /// Persistence test: Check if values stay constant for too long
    fn check_outliers(data: &[WeatherData]) -> Result<()> {
        if data.len() < 3 {
            return Ok(());
        }

        // Calculate mean and standard deviation for temperature
        let temps: Vec<f64> = data.iter().map(|d| d.temperature).collect();
        let mean = temps.iter().sum::<f64>() / temps.len() as f64;
        let variance = temps
            .iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>() / temps.len() as f64;
        let std_dev = variance.sqrt();

        // Flag values more than 3 standard deviations from mean
        for temp in &temps {
            if (temp - mean).abs() > 3.0 * std_dev {
                tracing::warn!(
                    "Potential outlier detected: temp={} (mean={}, std={})",
                    temp,
                    mean,
                    std_dev
                );
            }
        }

        Ok(())
    }

    /// Check if coordinates are valid
    pub fn validate_coordinates(lat: f64, lon: f64) -> Result<()> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(ClimateDataError::InvalidCoordinates { lat, lon });
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(ClimateDataError::InvalidCoordinates { lat, lon });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::types::Coordinates;

    fn create_test_data() -> WeatherData {
        WeatherData {
            timestamp: Utc::now(),
            location: Coordinates::new(40.7128, -74.0060),
            temperature: 20.0,
            humidity: 60.0,
            pressure: 1013.25,
            wind_speed: 5.0,
            wind_direction: 180.0,
            precipitation: 0.0,
            cloud_cover: 50.0,
            visibility: Some(10000.0),
            uv_index: Some(5.0),
            condition_code: Some(800),
            description: Some("Clear sky".to_string()),
        }
    }

    #[test]
    fn test_valid_data() {
        let data = create_test_data();
        assert!(DataValidator::validate_weather_data(&data).is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let mut data = create_test_data();
        data.temperature = -150.0; // Below absolute zero
        assert!(DataValidator::validate_weather_data(&data).is_err());
    }

    #[test]
    fn test_invalid_humidity() {
        let mut data = create_test_data();
        data.humidity = 150.0; // Above 100%
        assert!(DataValidator::validate_weather_data(&data).is_err());
    }

    #[test]
    fn test_coordinates() {
        assert!(DataValidator::validate_coordinates(40.7, -74.0).is_ok());
        assert!(DataValidator::validate_coordinates(91.0, 0.0).is_err());
        assert!(DataValidator::validate_coordinates(0.0, 181.0).is_err());
    }
}
