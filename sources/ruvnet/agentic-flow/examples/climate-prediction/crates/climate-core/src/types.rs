//! Core data types for climate prediction

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Geographic location
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

/// Time range for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Climate variables
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ClimateVariable {
    Temperature,
    Precipitation,
    Humidity,
    WindSpeed,
    WindDirection,
    Pressure,
    CloudCover,
    SeaLevelPressure,
}

/// Measurement unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Unit {
    Celsius,
    Fahrenheit,
    Kelvin,
    Millimeters,
    Inches,
    Percent,
    MetersPerSecond,
    Degrees,
    Hectopascals,
}

/// A single climate observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub location: Location,
    pub timestamp: DateTime<Utc>,
    pub variable: ClimateVariable,
    pub value: f64,
    pub unit: Unit,
    pub confidence: Option<f64>,
}

/// Climate data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateDataSeries {
    pub location: Location,
    pub variable: ClimateVariable,
    pub unit: Unit,
    pub data_points: Vec<DataPoint>,
}

/// Single data point in a time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub quality: Option<f64>,
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub location: Location,
    pub variable: ClimateVariable,
    pub forecast_time: DateTime<Utc>,
    pub predicted_value: f64,
    pub confidence_interval: ConfidenceInterval,
    pub model_version: String,
}

/// Confidence interval for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64, // e.g., 0.95 for 95%
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub trained_on: DateTime<Utc>,
    pub variables: Vec<ClimateVariable>,
    pub spatial_resolution: f64,
    pub temporal_resolution: String,
    pub metrics: HashMap<String, f64>,
}

/// Request for climate predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub location: Location,
    pub time_range: TimeRange,
    pub variables: Vec<ClimateVariable>,
    pub model_name: Option<String>,
}

/// Batch prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionRequest {
    pub locations: Vec<Location>,
    pub time_range: TimeRange,
    pub variables: Vec<ClimateVariable>,
    pub model_name: Option<String>,
}

impl Location {
    /// Create a new location
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude: None,
        }
    }

    /// Create a location with altitude
    pub fn with_altitude(latitude: f64, longitude: f64, altitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude: Some(altitude),
        }
    }

    /// Validate latitude and longitude
    pub fn is_valid(&self) -> bool {
        self.latitude >= -90.0 && self.latitude <= 90.0
            && self.longitude >= -180.0 && self.longitude <= 180.0
    }
}
