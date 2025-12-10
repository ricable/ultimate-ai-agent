//! # RAN Core - Radio Access Network Domain Abstractions
//!
//! This crate provides the core domain abstractions for Radio Access Network (RAN) 
//! ML optimization and automation. It defines the fundamental types, traits, and 
//! structures used throughout the ranML system.
//!
//! ## Features
//!
//! - **Cell Management**: Base station and cell definitions
//! - **Network Topology**: Network element hierarchy and relationships
//! - **Performance Metrics**: KPIs and measurement definitions
//! - **Resource Management**: Spectrum, power, and computational resources
//! - **Optimization Targets**: QoS, QoE, and efficiency metrics
//! - **Event System**: Network events and state changes
//!
//! ## Core Concepts
//!
//! ### Network Elements
//! - **gNodeB**: 5G base station representation
//! - **Cell**: Individual cell within a base station
//! - **UE**: User Equipment representation
//! - **Slice**: Network slice abstraction
//!
//! ### Performance Metrics
//! - **KPIs**: Key Performance Indicators
//! - **QoS**: Quality of Service metrics
//! - **QoE**: Quality of Experience metrics
//! - **Resource Utilization**: Efficiency measurements
//!
//! ### Optimization Domains
//! - **Coverage**: Signal coverage optimization
//! - **Capacity**: Traffic capacity management
//! - **Interference**: Signal interference mitigation
//! - **Energy**: Power consumption optimization

#![allow(missing_docs)]
#![warn(rust_2018_idioms)]

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ndarray::Array2;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export commonly used types for public API

/// Core error types for RAN operations
pub mod error;

/// Network element definitions
pub mod network;

/// Performance metrics and KPIs
pub mod metrics;

/// Resource management abstractions
pub mod resources;

/// Optimization target definitions
pub mod optimization;

/// Event system for network state changes
pub mod events;

/// Utility functions and common operations
pub mod utils;

// Re-export essential types
pub use error::{RanError, RanResult};
pub use network::{Cell, CellId, GNodeB, NetworkElement, NetworkTopology, UE};
pub use metrics::{Kpi, KpiType, PerformanceMetrics, QoSMetrics, QoEMetrics};
pub use resources::{ResourcePool, ResourceType, SpectrumResource, PowerResource};
pub use optimization::{OptimizationTarget, OptimizationObjective, OptimizationConstraint};
pub use events::{NetworkEvent, EventType, EventSeverity};

/// Geographic coordinates for network elements
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoCoordinate {
    /// Latitude in degrees
    pub latitude: f64,
    /// Longitude in degrees  
    pub longitude: f64,
    /// Altitude in meters above sea level
    pub altitude: Option<f64>,
}

impl GeoCoordinate {
    /// Create a new geographic coordinate
    pub fn new(latitude: f64, longitude: f64, altitude: Option<f64>) -> Self {
        Self {
            latitude,
            longitude,
            altitude,
        }
    }

    /// Calculate distance to another coordinate in meters
    pub fn distance_to(&self, other: &GeoCoordinate) -> f64 {
        let lat1_rad = self.latitude.to_radians();
        let lat2_rad = other.latitude.to_radians();
        let delta_lat = (other.latitude - self.latitude).to_radians();
        let delta_lon = (other.longitude - self.longitude).to_radians();

        let a = (delta_lat / 2.0).sin() * (delta_lat / 2.0).sin()
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin() * (delta_lon / 2.0).sin();
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        6371000.0 * c // Earth's radius in meters
    }
}

/// Time series data point with timestamp
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesPoint<T> {
    /// Timestamp of the measurement
    pub timestamp: DateTime<Utc>,
    /// Value of the measurement
    pub value: T,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

impl<T> TimeSeriesPoint<T> {
    /// Create a new time series point
    pub fn new(timestamp: DateTime<Utc>, value: T) -> Self {
        Self {
            timestamp,
            value,
            metadata: None,
        }
    }

    /// Create a new time series point with metadata
    pub fn with_metadata(
        timestamp: DateTime<Utc>,
        value: T,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            timestamp,
            value,
            metadata: Some(metadata),
        }
    }
}

/// Time series collection for network measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries<T> {
    /// Series identifier
    pub id: String,
    /// Data points in chronological order
    pub points: Vec<TimeSeriesPoint<T>>,
    /// Series metadata
    pub metadata: HashMap<String, String>,
}

impl<T> TimeSeries<T> {
    /// Create a new time series
    pub fn new(id: String) -> Self {
        Self {
            id,
            points: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a data point to the series
    pub fn add_point(&mut self, point: TimeSeriesPoint<T>) {
        self.points.push(point);
        // Keep points sorted by timestamp
        self.points.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the series is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get the latest data point
    pub fn latest(&self) -> Option<&TimeSeriesPoint<T>> {
        self.points.last()
    }

    /// Get points within a time range
    pub fn range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&TimeSeriesPoint<T>> {
        self.points
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .collect()
    }
}

/// Configuration for ML model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier
    pub model_id: String,
    /// Model type (e.g., "lstm", "nbeats", "transformer")
    pub model_type: String,
    /// Model parameters as key-value pairs
    pub parameters: HashMap<String, serde_json::Value>,
    /// Training configuration
    pub training: TrainingConfig,
    /// Inference configuration
    pub inference: InferenceConfig,
}

/// Training configuration for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}

/// Inference configuration for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Prediction horizon (number of steps ahead)
    pub horizon: usize,
    /// Input window size
    pub input_size: usize,
    /// Confidence interval level
    pub confidence_level: f64,
    /// Enable uncertainty quantification
    pub uncertainty_quantification: bool,
}

/// Trait for network element identification
pub trait NetworkElementId {
    /// Get the unique identifier for this network element
    fn id(&self) -> Uuid;
    
    /// Get the element type name
    fn element_type(&self) -> &'static str;
    
    /// Get the element display name
    fn display_name(&self) -> String;
}

/// Trait for measurable network elements
pub trait Measurable {
    /// The type of measurements this element produces
    type Measurement;
    
    /// Take a measurement from this element
    fn measure(&self) -> RanResult<Self::Measurement>;
    
    /// Get the measurement history
    fn measurement_history(&self) -> &TimeSeries<Self::Measurement>;
}

/// Trait for optimizable network elements
pub trait Optimizable {
    /// The type of optimization parameters
    type Parameters;
    
    /// Get current optimization parameters
    fn get_parameters(&self) -> &Self::Parameters;
    
    /// Set optimization parameters
    fn set_parameters(&mut self, params: Self::Parameters) -> RanResult<()>;
    
    /// Get optimization constraints
    fn get_constraints(&self) -> Vec<OptimizationConstraint>;
}

/// Trait for network elements that can be monitored
pub trait Monitorable {
    /// Get the current monitoring state
    fn is_monitored(&self) -> bool;
    
    /// Enable monitoring
    fn enable_monitoring(&mut self) -> RanResult<()>;
    
    /// Disable monitoring
    fn disable_monitoring(&mut self) -> RanResult<()>;
    
    /// Get monitoring interval in seconds
    fn monitoring_interval(&self) -> u64;
    
    /// Set monitoring interval in seconds
    fn set_monitoring_interval(&mut self, interval: u64) -> RanResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_geo_coordinate_distance() {
        let coord1 = GeoCoordinate::new(52.520008, 13.404954, None); // Berlin
        let coord2 = GeoCoordinate::new(48.856614, 2.3522219, None); // Paris
        
        let distance = coord1.distance_to(&coord2);
        // Distance between Berlin and Paris is approximately 878 km
        assert!(distance > 875000.0 && distance < 885000.0);
    }

    #[test]
    fn test_time_series_operations() {
        let mut series = TimeSeries::new("test_series".to_string());
        
        let now = Utc::now();
        let point1 = TimeSeriesPoint::new(now, 1.0);
        let point2 = TimeSeriesPoint::new(now + chrono::Duration::seconds(1), 2.0);
        
        series.add_point(point1);
        series.add_point(point2);
        
        assert_eq!(series.len(), 2);
        assert_eq!(series.latest().unwrap().value, 2.0);
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig {
            model_id: "test_model".to_string(),
            model_type: "lstm".to_string(),
            parameters: HashMap::new(),
            training: TrainingConfig {
                epochs: 100,
                learning_rate: 0.001,
                batch_size: 32,
                validation_split: 0.2,
                early_stopping_patience: Some(10),
            },
            inference: InferenceConfig {
                horizon: 24,
                input_size: 168,
                confidence_level: 0.95,
                uncertainty_quantification: true,
            },
        };
        
        assert_eq!(config.model_id, "test_model");
        assert_eq!(config.model_type, "lstm");
        assert_eq!(config.training.epochs, 100);
        assert_eq!(config.inference.horizon, 24);
    }
}