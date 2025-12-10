//! ASA-5G-01 - ENDC Setup Failure Predictor
//! 
//! This module implements advanced ENDC (E-UTRAN New Radio - Dual Connectivity) 
//! setup failure prediction using machine learning to achieve >80% prediction accuracy.
//! 
//! Key Features:
//! - Real-time ENDC failure prediction and prevention
//! - Signal quality analysis and bearer configuration optimization
//! - Proactive mitigation strategy recommendations
//! - 5G NSA/SA service health monitoring

pub mod config;
pub mod error;
pub mod models;
pub mod predictor;
pub mod proto;
pub mod service;

pub use config::Config;
pub use error::{Error, Result};
pub use models::EndcFailurePredictor;
pub use predictor::EndcPredictionEngine;
pub use service::EndcService;

// Re-export proto types
pub use proto::ran::service_assurance::{
    PredictEndcFailureRequest, PredictEndcFailureResponse,
    Get5GServiceHealthRequest, Get5GServiceHealthResponse,
    EndcMetrics, FailureAnalysis, FailureMitigation,
    ServiceHealthReport, ServiceHealthSummary,
};