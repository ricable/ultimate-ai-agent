//! Ensemble forecasting utilities

pub use crate::models::EnsembleForecaster;
pub use crate::models::EnsembleMethod;
pub use crate::models::EnsembleBuilder;

// Re-export for convenience
pub use crate::models::{
    TrafficPredictor,
    CapacityPredictor, 
    PerformancePredictor,
    AnomalyDetector,
};