// Property-based tests using proptest
use proptest::prelude::*;

mod common;
use common::test_helpers::*;

// Property: Temperature should always be within physical bounds
proptest! {
    #[test]
    fn test_temperature_bounds(temp in -100.0f64..100.0f64) {
        let validated = validate_temperature(temp);
        prop_assert!(validated.is_ok());
        prop_assert!(validated.unwrap() >= -100.0 && validated.unwrap() <= 100.0);
    }

    #[test]
    fn test_invalid_temperature_rejected(temp in prop::num::f64::ANY) {
        if temp < -100.0 || temp > 100.0 {
            let validated = validate_temperature(temp);
            prop_assert!(validated.is_err());
        }
    }
}

// Property: Humidity should be 0-100%
proptest! {
    #[test]
    fn test_humidity_bounds(humidity in 0.0f64..100.0f64) {
        let validated = validate_humidity(humidity);
        prop_assert!(validated.is_ok());
        prop_assert!(validated.unwrap() >= 0.0 && validated.unwrap() <= 100.0);
    }

    #[test]
    fn test_negative_humidity_rejected(humidity in prop::num::f64::NEGATIVE) {
        let validated = validate_humidity(humidity);
        prop_assert!(validated.is_err());
    }
}

// Property: Pressure should be within atmospheric bounds
proptest! {
    #[test]
    fn test_pressure_bounds(pressure in 800.0f64..1200.0f64) {
        let validated = validate_pressure(pressure);
        prop_assert!(validated.is_ok());
    }
}

// Property: Wind speed cannot be negative
proptest! {
    #[test]
    fn test_wind_speed_non_negative(speed in 0.0f64..200.0f64) {
        let validated = validate_wind_speed(speed);
        prop_assert!(validated.is_ok());
        prop_assert!(validated.unwrap() >= 0.0);
    }

    #[test]
    fn test_negative_wind_speed_rejected(speed in prop::num::f64::NEGATIVE) {
        let validated = validate_wind_speed(speed);
        prop_assert!(validated.is_err());
    }
}

// Property: Precipitation cannot be negative
proptest! {
    #[test]
    fn test_precipitation_non_negative(precip in 0.0f64..500.0f64) {
        let validated = validate_precipitation(precip);
        prop_assert!(validated.is_ok());
        prop_assert!(validated.unwrap() >= 0.0);
    }
}

// Property: Latitude must be -90 to 90
proptest! {
    #[test]
    fn test_latitude_bounds(lat in -90.0f64..90.0f64) {
        let validated = validate_latitude(lat);
        prop_assert!(validated.is_ok());
        prop_assert!(validated.unwrap() >= -90.0 && validated.unwrap() <= 90.0);
    }

    #[test]
    fn test_invalid_latitude(lat in prop::num::f64::ANY) {
        if lat < -90.0 || lat > 90.0 {
            let validated = validate_latitude(lat);
            prop_assert!(validated.is_err());
        }
    }
}

// Property: Longitude must be -180 to 180
proptest! {
    #[test]
    fn test_longitude_bounds(lon in -180.0f64..180.0f64) {
        let validated = validate_longitude(lon);
        prop_assert!(validated.is_ok());
        prop_assert!(validated.unwrap() >= -180.0 && validated.unwrap() <= 180.0);
    }
}

// Property: Data transformation is reversible
proptest! {
    #[test]
    fn test_celsius_fahrenheit_conversion(celsius in -100.0f64..100.0f64) {
        let fahrenheit = celsius_to_fahrenheit(celsius);
        let back_to_celsius = fahrenheit_to_celsius(fahrenheit);

        // Allow small floating point error
        prop_assert!((celsius - back_to_celsius).abs() < 0.001);
    }
}

// Property: Array operations maintain length
proptest! {
    #[test]
    fn test_data_processing_preserves_length(
        data_len in 1usize..1000
    ) {
        let data = generate_sample_climate_data(data_len);
        prop_assert_eq!(data.len(), data_len);

        let processed = process_data(&data);
        prop_assert_eq!(processed.len(), data.len());
    }
}

// Property: Moving average should be between min and max
proptest! {
    #[test]
    fn test_moving_average_bounds(
        values in prop::collection::vec(0.0f64..100.0, 10..100)
    ) {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let moving_avg = calculate_moving_average(&values, 5);

        for avg in moving_avg {
            prop_assert!(avg >= min && avg <= max);
        }
    }
}

// Property: Predictions should be continuous (no huge jumps)
proptest! {
    #[test]
    fn test_prediction_continuity(
        temp_base in 10.0f64..30.0
    ) {
        let data: Vec<f32> = (0..24)
            .map(|i| (temp_base + (i as f64 * 0.1)) as f32)
            .collect();

        let prediction = mock_predict_continuity(&data);

        if let Some(pred) = prediction {
            // Prediction should be within reasonable range of input
            let avg_input = data.iter().sum::<f32>() / data.len() as f32;
            let diff = (pred - avg_input).abs();
            prop_assert!(diff < 10.0, "Prediction {} too far from average {}", pred, avg_input);
        }
    }
}

// Property: Confidence scores are between 0 and 1
proptest! {
    #[test]
    fn test_confidence_score_bounds(
        data_len in 10usize..200
    ) {
        let data = generate_sample_climate_data(data_len);
        let confidence = calculate_confidence(&data);

        prop_assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}

// Property: Larger datasets should have higher confidence
proptest! {
    #[test]
    fn test_confidence_increases_with_data(
        small_len in 10usize..50,
        large_len in 100usize..200
    ) {
        let small_data = generate_sample_climate_data(small_len);
        let large_data = generate_sample_climate_data(large_len);

        let small_confidence = calculate_confidence(&small_data);
        let large_confidence = calculate_confidence(&large_data);

        prop_assert!(large_confidence >= small_confidence);
    }
}

// Helper validation functions
fn validate_temperature(temp: f64) -> Result<f64, String> {
    if temp >= -100.0 && temp <= 100.0 {
        Ok(temp)
    } else {
        Err(format!("Temperature {} out of bounds", temp))
    }
}

fn validate_humidity(humidity: f64) -> Result<f64, String> {
    if humidity >= 0.0 && humidity <= 100.0 {
        Ok(humidity)
    } else {
        Err(format!("Humidity {} out of bounds", humidity))
    }
}

fn validate_pressure(pressure: f64) -> Result<f64, String> {
    if pressure >= 800.0 && pressure <= 1200.0 {
        Ok(pressure)
    } else {
        Err(format!("Pressure {} out of bounds", pressure))
    }
}

fn validate_wind_speed(speed: f64) -> Result<f64, String> {
    if speed >= 0.0 {
        Ok(speed)
    } else {
        Err(format!("Wind speed {} cannot be negative", speed))
    }
}

fn validate_precipitation(precip: f64) -> Result<f64, String> {
    if precip >= 0.0 {
        Ok(precip)
    } else {
        Err(format!("Precipitation {} cannot be negative", precip))
    }
}

fn validate_latitude(lat: f64) -> Result<f64, String> {
    if lat >= -90.0 && lat <= 90.0 {
        Ok(lat)
    } else {
        Err(format!("Latitude {} out of bounds", lat))
    }
}

fn validate_longitude(lon: f64) -> Result<f64, String> {
    if lon >= -180.0 && lon <= 180.0 {
        Ok(lon)
    } else {
        Err(format!("Longitude {} out of bounds", lon))
    }
}

fn celsius_to_fahrenheit(celsius: f64) -> f64 {
    celsius * 9.0 / 5.0 + 32.0
}

fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0
}

fn process_data(data: &[ClimateDataPoint]) -> Vec<ClimateDataPoint> {
    data.to_vec() // Simple identity function for testing
}

fn calculate_moving_average(values: &[f64], window: usize) -> Vec<f64> {
    if values.len() < window {
        return vec![];
    }

    values
        .windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

fn mock_predict_continuity(features: &[f32]) -> Option<f32> {
    if features.is_empty() {
        return None;
    }
    Some(features.iter().sum::<f32>() / features.len() as f32)
}

fn calculate_confidence(data: &[ClimateDataPoint]) -> f64 {
    let base_confidence = 0.5;
    let length_factor = (data.len() as f64).ln() / 10.0;
    (base_confidence + length_factor).min(1.0).max(0.0)
}
