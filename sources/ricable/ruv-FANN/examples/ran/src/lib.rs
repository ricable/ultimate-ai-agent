//! RAN Intelligence Platform
//! 
//! AI-powered RAN Intelligence & Automation Platform using ruv-FANN
//! for 5G network optimization and service assurance.

// Core modules
pub mod common;
pub mod features;
pub mod models;

// Integration layer
pub mod integration;

// Service Assurance modules
pub mod asa_5g;

// AFM (Autonomous Fault Management) modules
pub mod afm_correlate;
pub mod afm_detect;
pub mod afm_rca;

// AOS (Autonomous Operations System) modules
pub mod aos_heal;

// DTM (Dynamic Traffic Management) modules
pub mod dtm_mobility;
pub mod dtm_power;
pub mod dtm_traffic;

// PFS (Platform Foundation Services) modules
pub mod pfs_core;
pub mod pfs_data;
pub mod pfs_logs;
pub mod pfs_twin;

// RIC (RAN Intelligence Controller) modules
pub mod ric_conflict;
pub mod ric_tsa;

use thiserror::Error;

// Re-export commonly used types and traits
pub use common::{RanModel, ModelMetrics, FeatureEngineer, DataIngester, RanConfig};
pub use features::FeatureEngineeringService;
pub use models::{ModelRegistry, RanNeuralNetwork};

// Re-export AFM modules
pub use afm_correlate::{EvidenceItem, EvidenceSource, CorrelationResult, ImpactAssessment};
pub use afm_detect::*;
pub use afm_rca::*;

// Re-export ASA-5G modules
pub use asa_5g::{
    Asa5gConfig, EndcPredictionInput, EndcPredictionOutput, RiskLevel,
    SignalQualityFeatures, MitigationRecommendation, MonitoringDashboard,
    EndcPredictor, SignalAnalyzer, MonitoringService, MitigationService
};

#[derive(Error, Debug)]
pub enum RanError {
    #[error("ML model error: {0}")]
    ModelError(String),
    
    #[error("Feature engineering error: {0}")]
    FeatureError(String),
    
    #[error("Data processing error: {0}")]
    DataError(String),
    
    #[error("Network communication error: {0}")]
    NetworkError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("5G service assurance error: {0}")]
    ServiceAssuranceError(String),
    
    #[error("AFM correlation error: {0}")]
    AfmCorrelationError(String),
    
    #[error("AFM detection error: {0}")]
    AfmDetectionError(String),
    
    #[error("AFM root cause analysis error: {0}")]
    AfmRcaError(String),
    
    #[error("AOS healing error: {0}")]
    AosHealError(String),
    
    #[error("DTM mobility error: {0}")]
    DtmMobilityError(String),
    
    #[error("DTM power error: {0}")]
    DtmPowerError(String),
    
    #[error("DTM traffic error: {0}")]
    DtmTrafficError(String),
    
    #[error("PFS core error: {0}")]
    PfsCoreError(String),
    
    #[error("PFS data error: {0}")]
    PfsDataError(String),
    
    #[error("PFS logs error: {0}")]
    PfsLogsError(String),
    
    #[error("PFS twin error: {0}")]
    PfsTwinError(String),
    
    #[error("RIC conflict error: {0}")]
    RicConflictError(String),
    
    #[error("RIC TSA error: {0}")]
    RicTsaError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, RanError>;

/// Initialize the RAN Intelligence Platform
pub async fn init_platform(config: &RanConfig) -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Initialize components
    tracing::info!("Initializing RAN Intelligence Platform");
    tracing::info!("Configuration: {:?}", config);
    
    // Validate configuration
    if config.database_url.is_empty() {
        return Err(RanError::ConfigError("Database URL cannot be empty".to_string()));
    }
    
    if config.ml_service_endpoint.is_empty() {
        return Err(RanError::ConfigError("ML service endpoint cannot be empty".to_string()));
    }
    
    tracing::info!("RAN Intelligence Platform initialized successfully");
    Ok(())
}

/// Get platform version information
pub fn version_info() -> &'static str {
    "RAN Intelligence Platform v1.0.0 - Powered by ruv-FANN"
}

/// Get available modules information
pub fn available_modules() -> Vec<&'static str> {
    vec![
        "asa_5g - 5G Service Assurance",
        "afm_correlate - AFM Correlation Analysis", 
        "afm_detect - AFM Anomaly Detection",
        "afm_rca - AFM Root Cause Analysis",
        "aos_heal - AOS Healing Module",
        "dtm_mobility - Dynamic Traffic Management Mobility",
        "dtm_power - Dynamic Traffic Management Power",
        "dtm_traffic - Dynamic Traffic Management Traffic",
        "pfs_core - Platform Foundation Services Core",
        "pfs_data - Platform Foundation Services Data",
        "pfs_logs - Platform Foundation Services Logs",
        "pfs_twin - Platform Foundation Services Twin",
        "ric_conflict - RIC Conflict Resolution",
        "ric_tsa - RIC Traffic Steering Automation",
        "common - Common utilities and types",
        "features - Feature engineering",
        "models - ML model management",
    ]
}

/// Common types used across the RAN Intelligence Platform
pub mod types {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UeId(pub String);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CellId(pub String);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelId(pub Uuid);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeSeries {
        pub timestamp: DateTime<Utc>,
        pub values: Vec<f64>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SignalQuality {
        pub timestamp: DateTime<Utc>,
        pub ue_id: UeId,
        pub lte_rsrp: f64,
        pub lte_sinr: f64,
        pub nr_ssb_rsrp: Option<f64>,
        pub endc_setup_success_rate: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PredictionResult {
        pub timestamp: DateTime<Utc>,
        pub confidence: f64,
        pub metadata: serde_json::Value,
    }
}