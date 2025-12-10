//! ASA-INT-01 - Uplink Interference Classifier
//! 
//! This module implements advanced uplink interference classification using machine learning
//! to achieve >95% classification accuracy. It provides:
//! - Real-time interference detection and classification
//! - Noise floor analysis and pattern recognition
//! - Mitigation strategy recommendations
//! - Performance monitoring and metrics

pub mod config;
pub mod error;
pub mod features;
pub mod models;
pub mod proto;
pub mod service;

pub use config::Config;
pub use error::{Error, Result};
pub use features::InterferenceFeatureExtractor;
pub use models::InterferenceClassifier;
pub use service::InterferenceService;

// Re-export proto types
pub use proto::interference_classifier::{
    ClassifyRequest, ClassifyResponse, ConfidenceRequest, ConfidenceResponse,
    MitigationRequest, MitigationResponse, TrainRequest, TrainResponse,
    MetricsRequest, MetricsResponse, NoiseFloorMeasurement, CellParameters,
    TrainingExample, ModelConfig,
};