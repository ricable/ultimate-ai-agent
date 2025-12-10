//! Neural network modules for the standalone swarm demo
//! 
//! Enhanced with real network data processing and advanced anomaly detection

pub mod demand_predictor;
pub mod endc_predictor;
pub mod feature_engineering;
pub mod kpi_predictor;
pub mod latency_optimizer;
pub mod ml_model;
pub mod neural_agent;
pub mod quality_predictor;
pub mod throughput_model;

// New enhanced modules with real data integration
pub mod enhanced_anomaly_detector;
pub mod enhanced_endc_predictor;

// Re-export key types
pub use enhanced_anomaly_detector::{
    EnhancedAnomalyDetector, AnomalyEvent, AnomalyType, AnomalySeverity,
    AnomalyDetectorConfig, AnomalyThresholds
};
pub use enhanced_endc_predictor::{
    EnhancedEndcPredictor, EndcPrediction, RiskLevel, ContributingFactor,
    EndcPredictorConfig
};
pub use ml_model::{MLModel, ModelType, TrainingConfig};
pub use neural_agent::NeuralAgent;