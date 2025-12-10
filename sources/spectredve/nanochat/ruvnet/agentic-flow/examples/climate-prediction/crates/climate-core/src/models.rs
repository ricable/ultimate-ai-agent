//! Data models for climate prediction

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Geographic location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub name: Option<String>,
}

/// Climate data from API or historical records
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateData {
    pub location: Location,
    pub timestamp: DateTime<Utc>,
    pub temperature: f64,      // Celsius
    pub humidity: f64,         // Percentage
    pub pressure: f64,         // hPa
    pub wind_speed: f64,       // m/s
    pub precipitation: f64,    // mm
    pub cloud_cover: f64,      // Percentage
}

/// Climate prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimatePrediction {
    pub location: Location,
    pub predicted_at: DateTime<Utc>,
    pub prediction_time: DateTime<Utc>,
    pub temperature: f64,
    pub humidity: f64,
    pub precipitation_probability: f64,
    pub confidence: f64,       // 0.0-1.0
    pub model_used: ModelType,
}

/// Available prediction models
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Simple linear regression
    Linear,
    /// Time-series ARIMA
    Arima,
    /// Neural network
    Neural,
    /// Ensemble of multiple models
    Ensemble,
}

impl ModelType {
    pub fn as_str(&self) -> &str {
        match self {
            ModelType::Linear => "linear",
            ModelType::Arima => "arima",
            ModelType::Neural => "neural",
            ModelType::Ensemble => "ensemble",
        }
    }
}

/// Input for climate prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateInput {
    pub location: Location,
    pub forecast_hours: u32,
    pub model_preference: Option<ModelType>,
}

/// Historical climate pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimatePattern {
    pub location_hash: String,
    pub season: String,
    pub model_type: ModelType,
    pub avg_accuracy: f64,
    pub sample_size: usize,
}
