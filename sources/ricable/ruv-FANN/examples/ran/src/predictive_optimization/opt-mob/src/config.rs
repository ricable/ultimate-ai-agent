//! Configuration for OPT-MOB handover prediction service

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptMobConfig {
    /// Neural network model configuration
    pub model: ModelConfig,
    
    /// UE metrics processing configuration
    pub processing: ProcessingConfig,
    
    /// Neighbor cell analysis configuration
    pub neighbor_analysis: NeighborAnalysisConfig,
    
    /// gRPC service configuration
    pub service: ServiceConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Metrics and monitoring configuration
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the trained model file
    pub model_path: PathBuf,
    
    /// Number of input features
    pub input_features: usize,
    
    /// Number of hidden layers
    pub hidden_layers: Vec<usize>,
    
    /// Number of output classes
    pub output_classes: usize,
    
    /// Activation function
    pub activation_function: String,
    
    /// Learning rate for online learning
    pub learning_rate: f64,
    
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    
    /// Handover probability threshold
    pub handover_threshold: f64,
    
    /// Model retraining interval (seconds)
    pub retrain_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Time window for metrics aggregation (seconds)
    pub time_window_seconds: u64,
    
    /// Minimum number of samples for prediction
    pub min_samples: usize,
    
    /// Maximum age of metrics (seconds)
    pub max_age_seconds: u64,
    
    /// Smoothing factor for exponential averaging
    pub smoothing_factor: f64,
    
    /// RSRP normalization range
    pub rsrp_range: (f64, f64),
    
    /// SINR normalization range
    pub sinr_range: (f64, f64),
    
    /// Speed normalization range
    pub speed_range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborAnalysisConfig {
    /// Maximum number of neighbor cells to consider
    pub max_neighbors: usize,
    
    /// Minimum RSRP threshold for neighbor cells
    pub min_rsrp_threshold: f64,
    
    /// Minimum SINR threshold for neighbor cells
    pub min_sinr_threshold: f64,
    
    /// Maximum load factor for target cells
    pub max_load_factor: f64,
    
    /// Distance weight factor
    pub distance_weight: f64,
    
    /// Load balance weight factor
    pub load_balance_weight: f64,
    
    /// Signal quality weight factor
    pub signal_quality_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// gRPC server bind address
    pub bind_address: String,
    
    /// gRPC server port
    pub port: u16,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
    
    /// Request timeout (seconds)
    pub request_timeout_seconds: u64,
    
    /// Enable TLS
    pub enable_tls: bool,
    
    /// TLS certificate path
    pub tls_cert_path: Option<PathBuf>,
    
    /// TLS private key path
    pub tls_key_path: Option<PathBuf>,
    
    /// Maximum batch size for batch predictions
    pub max_batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database connection URL
    pub url: String,
    
    /// Connection pool size
    pub pool_size: u32,
    
    /// Connection timeout (seconds)
    pub connection_timeout_seconds: u64,
    
    /// Query timeout (seconds)
    pub query_timeout_seconds: u64,
    
    /// Enable database logging
    pub enable_logging: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub enable_prometheus: bool,
    
    /// Prometheus metrics port
    pub prometheus_port: u16,
    
    /// Metrics collection interval (seconds)
    pub metrics_interval_seconds: u64,
    
    /// Performance alert thresholds
    pub alert_thresholds: AlertThresholds,
    
    /// Log level
    pub log_level: String,
    
    /// Log file path
    pub log_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Minimum accuracy threshold
    pub min_accuracy: f64,
    
    /// Maximum false positive rate
    pub max_false_positive_rate: f64,
    
    /// Maximum prediction latency (ms)
    pub max_prediction_latency_ms: f64,
    
    /// Minimum throughput (predictions/second)
    pub min_throughput: f64,
}

impl Default for OptMobConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            processing: ProcessingConfig::default(),
            neighbor_analysis: NeighborAnalysisConfig::default(),
            service: ServiceConfig::default(),
            database: DatabaseConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/handover_v1.fann"),
            input_features: 8,
            hidden_layers: vec![16, 12, 8],
            output_classes: 2,
            activation_function: "sigmoid".to_string(),
            learning_rate: 0.001,
            confidence_threshold: 0.8,
            handover_threshold: 0.5,
            retrain_interval_seconds: 3600, // 1 hour
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            time_window_seconds: 30,
            min_samples: 5,
            max_age_seconds: 300,
            smoothing_factor: 0.3,
            rsrp_range: (-140.0, -40.0),
            sinr_range: (-20.0, 30.0),
            speed_range: (0.0, 200.0),
        }
    }
}

impl Default for NeighborAnalysisConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 8,
            min_rsrp_threshold: -110.0,
            min_sinr_threshold: -5.0,
            max_load_factor: 0.9,
            distance_weight: 0.3,
            load_balance_weight: 0.4,
            signal_quality_weight: 0.3,
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 50051,
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            max_batch_size: 100,
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgresql://localhost/ran_optimization".to_string(),
            pool_size: 20,
            connection_timeout_seconds: 30,
            query_timeout_seconds: 10,
            enable_logging: false,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: true,
            prometheus_port: 9090,
            metrics_interval_seconds: 30,
            alert_thresholds: AlertThresholds::default(),
            log_level: "info".to_string(),
            log_file: None,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            min_accuracy: 0.90,
            max_false_positive_rate: 0.05,
            max_prediction_latency_ms: 10.0,
            min_throughput: 1000.0,
        }
    }
}

impl OptMobConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: OptMobConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate model configuration
        if self.model.input_features == 0 {
            return Err(anyhow::anyhow!("Model input features must be greater than 0"));
        }
        
        if self.model.hidden_layers.is_empty() {
            return Err(anyhow::anyhow!("Model must have at least one hidden layer"));
        }
        
        if self.model.output_classes == 0 {
            return Err(anyhow::anyhow!("Model output classes must be greater than 0"));
        }
        
        if self.model.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be positive"));
        }
        
        if self.model.confidence_threshold < 0.0 || self.model.confidence_threshold > 1.0 {
            return Err(anyhow::anyhow!("Confidence threshold must be between 0 and 1"));
        }
        
        if self.model.handover_threshold < 0.0 || self.model.handover_threshold > 1.0 {
            return Err(anyhow::anyhow!("Handover threshold must be between 0 and 1"));
        }
        
        // Validate processing configuration
        if self.processing.time_window_seconds == 0 {
            return Err(anyhow::anyhow!("Time window must be greater than 0"));
        }
        
        if self.processing.min_samples == 0 {
            return Err(anyhow::anyhow!("Minimum samples must be greater than 0"));
        }
        
        if self.processing.smoothing_factor < 0.0 || self.processing.smoothing_factor > 1.0 {
            return Err(anyhow::anyhow!("Smoothing factor must be between 0 and 1"));
        }
        
        // Validate neighbor analysis configuration
        if self.neighbor_analysis.max_neighbors == 0 {
            return Err(anyhow::anyhow!("Maximum neighbors must be greater than 0"));
        }
        
        if self.neighbor_analysis.max_load_factor < 0.0 || self.neighbor_analysis.max_load_factor > 1.0 {
            return Err(anyhow::anyhow!("Maximum load factor must be between 0 and 1"));
        }
        
        // Validate service configuration
        if self.service.port == 0 {
            return Err(anyhow::anyhow!("Service port must be greater than 0"));
        }
        
        if self.service.max_connections == 0 {
            return Err(anyhow::anyhow!("Maximum connections must be greater than 0"));
        }
        
        if self.service.max_batch_size == 0 {
            return Err(anyhow::anyhow!("Maximum batch size must be greater than 0"));
        }
        
        // Validate database configuration
        if self.database.url.is_empty() {
            return Err(anyhow::anyhow!("Database URL cannot be empty"));
        }
        
        if self.database.pool_size == 0 {
            return Err(anyhow::anyhow!("Database pool size must be greater than 0"));
        }
        
        // Validate monitoring configuration
        if self.monitoring.enable_prometheus && self.monitoring.prometheus_port == 0 {
            return Err(anyhow::anyhow!("Prometheus port must be greater than 0"));
        }
        
        if self.monitoring.metrics_interval_seconds == 0 {
            return Err(anyhow::anyhow!("Metrics interval must be greater than 0"));
        }
        
        // Validate alert thresholds
        if self.monitoring.alert_thresholds.min_accuracy < 0.0 || 
           self.monitoring.alert_thresholds.min_accuracy > 1.0 {
            return Err(anyhow::anyhow!("Minimum accuracy must be between 0 and 1"));
        }
        
        if self.monitoring.alert_thresholds.max_false_positive_rate < 0.0 || 
           self.monitoring.alert_thresholds.max_false_positive_rate > 1.0 {
            return Err(anyhow::anyhow!("Maximum false positive rate must be between 0 and 1"));
        }
        
        if self.monitoring.alert_thresholds.max_prediction_latency_ms <= 0.0 {
            return Err(anyhow::anyhow!("Maximum prediction latency must be positive"));
        }
        
        if self.monitoring.alert_thresholds.min_throughput <= 0.0 {
            return Err(anyhow::anyhow!("Minimum throughput must be positive"));
        }
        
        Ok(())
    }
}