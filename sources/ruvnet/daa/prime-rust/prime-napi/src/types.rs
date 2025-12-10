//! Type conversions between Rust and JavaScript
//!
//! Provides mappings and conversion functions for passing data between
//! Rust Prime ML types and JavaScript NAPI types.

use napi::bindgen_prelude::*;
use napi::Result;
use std::collections::HashMap;

use daa_prime_core::types::{
    AggregationStrategy, GradientUpdate, ModelMetadata, NodeId, OptimizerType, TrainingConfig,
    TrainingMetrics,
};

/// Node information for JavaScript
#[napi(object)]
#[derive(Debug, Clone)]
pub struct NodeInfoJs {
    /// Unique node identifier
    pub node_id: String,
    /// Node type (e.g., "trainer", "coordinator")
    pub node_type: String,
    /// Last heartbeat timestamp (Unix epoch milliseconds)
    pub last_heartbeat: f64,
    /// Reliability score (0.0 - 1.0)
    pub reliability_score: f64,
}

/// Optimizer type for JavaScript
#[napi(object)]
#[derive(Debug, Clone)]
pub struct OptimizerTypeJs {
    /// Optimizer name: "sgd", "adam", "adamw"
    pub optimizer_type: String,
    /// Optimizer-specific parameters
    pub params: Option<HashMap<String, f64>>,
}

impl From<OptimizerTypeJs> for OptimizerType {
    fn from(js: OptimizerTypeJs) -> Self {
        match js.optimizer_type.as_str() {
            "sgd" => {
                let momentum = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("momentum"))
                    .copied()
                    .unwrap_or(0.0) as f32;
                OptimizerType::Sgd { momentum }
            }
            "adamw" => {
                let beta1 = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("beta1"))
                    .copied()
                    .unwrap_or(0.9) as f32;
                let beta2 = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("beta2"))
                    .copied()
                    .unwrap_or(0.999) as f32;
                let weight_decay = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("weightDecay"))
                    .copied()
                    .unwrap_or(0.01) as f32;
                OptimizerType::AdamW {
                    beta1,
                    beta2,
                    weight_decay,
                }
            }
            _ => {
                // Default to Adam
                let beta1 = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("beta1"))
                    .copied()
                    .unwrap_or(0.9) as f32;
                let beta2 = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("beta2"))
                    .copied()
                    .unwrap_or(0.999) as f32;
                OptimizerType::Adam { beta1, beta2 }
            }
        }
    }
}

/// Aggregation strategy for JavaScript
#[napi(object)]
#[derive(Debug, Clone)]
pub struct AggregationStrategyJs {
    /// Strategy name: "fedavg", "secure", "trimmed_mean", "krum"
    pub strategy_type: String,
    /// Strategy-specific parameters
    pub params: Option<HashMap<String, f64>>,
}

impl From<AggregationStrategyJs> for AggregationStrategy {
    fn from(js: AggregationStrategyJs) -> Self {
        match js.strategy_type.as_str() {
            "trimmed_mean" => {
                let trim_ratio = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("trimRatio"))
                    .copied()
                    .unwrap_or(0.1) as f32;
                AggregationStrategy::TrimmedMean { trim_ratio }
            }
            "krum" => {
                let selection_count = js
                    .params
                    .as_ref()
                    .and_then(|p| p.get("selectionCount"))
                    .copied()
                    .unwrap_or(1.0) as usize;
                AggregationStrategy::Krum { selection_count }
            }
            "secure" | "secure_aggregation" => AggregationStrategy::SecureAggregation,
            _ => AggregationStrategy::FederatedAveraging,
        }
    }
}

/// Model metadata for JavaScript
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ModelMetadataJs {
    /// Model identifier
    pub id: String,
    /// Model version
    pub version: u32,
    /// Model architecture (e.g., "ResNet50", "BERT")
    pub architecture: String,
    /// Number of parameters in the model
    pub parameters_count: u32,
    /// Creation timestamp (Unix epoch milliseconds)
    pub created_at: f64,
    /// Last update timestamp (Unix epoch milliseconds)
    pub updated_at: f64,
}

impl From<ModelMetadata> for ModelMetadataJs {
    fn from(meta: ModelMetadata) -> Self {
        Self {
            id: meta.id,
            version: meta.version as u32,
            architecture: meta.architecture,
            parameters_count: meta.parameters_count as u32,
            created_at: meta.created_at as f64,
            updated_at: meta.updated_at as f64,
        }
    }
}

/// Convert Rust TrainingMetrics to JavaScript object
pub fn training_metrics_to_js(metrics: &TrainingMetrics) -> crate::trainer::TrainingMetricsJs {
    crate::trainer::TrainingMetricsJs {
        loss: metrics.loss as f64,
        accuracy: metrics.accuracy as f64,
        samples_processed: metrics.samples_processed as u32,
        computation_time_ms: metrics.computation_time_ms as u32,
    }
}

/// Convert Rust GradientUpdate to JavaScript object
pub fn gradient_update_to_js(update: &GradientUpdate) -> crate::trainer::GradientUpdateJs {
    crate::trainer::GradientUpdateJs {
        node_id: update.node_id.0.clone(),
        model_version: update.model_version as u32,
        round: update.round as u32,
        metrics: training_metrics_to_js(&update.metrics),
        timestamp: update.timestamp as f64,
    }
}

/// Create a default training configuration
#[napi]
pub fn create_default_training_config() -> crate::trainer::TrainingConfigJs {
    crate::trainer::TrainingConfigJs {
        batch_size: 32,
        learning_rate: 0.001,
        epochs: 10,
        optimizer: "adam".to_string(),
        optimizer_params: Some({
            let mut params = HashMap::new();
            params.insert("beta1".to_string(), 0.9);
            params.insert("beta2".to_string(), 0.999);
            params
        }),
        aggregation_strategy: "fedavg".to_string(),
    }
}

/// Create a default coordinator configuration
#[napi]
pub fn create_default_coordinator_config() -> crate::coordinator::CoordinatorConfig {
    crate::coordinator::CoordinatorConfig::default()
}

/// Validate node ID format
#[napi]
pub fn validate_node_id(node_id: String) -> Result<bool> {
    // Node IDs should be non-empty and alphanumeric with hyphens/underscores
    let valid = !node_id.is_empty()
        && node_id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_');
    Ok(valid)
}

/// Generate a unique node ID
#[napi]
pub fn generate_node_id(prefix: Option<String>) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    let prefix = prefix.unwrap_or_else(|| "node".to_string());
    format!("{}-{}", prefix, timestamp)
}
