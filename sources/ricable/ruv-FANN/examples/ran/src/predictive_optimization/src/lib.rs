//! # RAN Intelligence Platform - Predictive Optimization Services
//!
//! This crate provides comprehensive predictive optimization services for 5G/6G RAN networks,
//! implementing three core optimization areas:
//!
//! - **OPT-MOB**: Predictive Handover Trigger Model (>90% accuracy)
//! - **OPT-ENG**: Cell Sleep Mode Forecaster (MAPE <10%, >95% detection)
//! - **OPT-RES**: Predictive Carrier Aggregation SCell Manager (>80% accuracy)
//!
//! ## Architecture
//!
//! The platform uses a unified architecture with shared components:
//! - Neural network prediction engines using ruv-FANN
//! - Real-time gRPC services for network integration
//! - Performance monitoring and metrics collection
//! - Continuous learning and model adaptation
//!
//! ## Usage
//!
//! ```rust
//! use predictive_optimization::{OptimizationPlatform, PlatformConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = PlatformConfig::default();
//!     let platform = OptimizationPlatform::new(config).await?;
//!     
//!     // Start all optimization services
//!     platform.start_all_services().await?;
//!     
//!     // Platform is now ready for optimization requests
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;

// Re-export all optimization modules
pub use opt_mob;
pub use opt_eng;
pub use opt_res;

pub mod config;
pub mod service;
pub mod monitoring;
pub mod metrics;

#[derive(Error, Debug)]
pub enum PlatformError {
    #[error("Handover optimization error: {0}")]
    Handover(#[from] opt_mob::OptMobError),
    
    #[error("Energy optimization error: {0}")]
    Energy(#[from] opt_eng::OptEngError),
    
    #[error("Resource optimization error: {0}")]
    Resource(#[from] opt_res::OptResError),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Service error: {0}")]
    Service(String),
    
    #[error("Monitoring error: {0}")]
    Monitoring(String),
}

/// Unified platform configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    pub handover: opt_mob::config::OptMobConfig,
    pub energy: opt_eng::config::OptEngConfig,
    pub resource: opt_res::config::OptResConfig,
    pub platform: config::CorePlatformConfig,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            handover: opt_mob::config::OptMobConfig::default(),
            energy: opt_eng::config::OptEngConfig::default(),
            resource: opt_res::config::OptResConfig::default(),
            platform: config::CorePlatformConfig::default(),
        }
    }
}

/// Platform-wide metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformMetrics {
    pub handover_metrics: opt_mob::HandoverMetrics,
    pub energy_metrics: opt_eng::ForecastingMetrics,
    pub resource_metrics: opt_res::CarrierAggregationMetrics,
    pub platform_health: PlatformHealth,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformHealth {
    pub overall_health: HealthStatus,
    pub handover_health: HealthStatus,
    pub energy_health: HealthStatus,
    pub resource_health: HealthStatus,
    pub system_utilization: f64,
    pub error_rate: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Main optimization platform
pub struct OptimizationPlatform {
    config: Arc<PlatformConfig>,
    handover_optimizer: Arc<opt_mob::HandoverOptimizer>,
    energy_optimizer: Arc<opt_eng::CellSleepOptimizer>,
    resource_optimizer: Arc<opt_res::CarrierAggregationOptimizer>,
    platform_monitor: Arc<monitoring::PlatformMonitor>,
    metrics: Arc<RwLock<PlatformMetrics>>,
    start_time: DateTime<Utc>,
}

impl OptimizationPlatform {
    /// Create new optimization platform
    pub async fn new(config: PlatformConfig) -> Result<Self> {
        let config = Arc::new(config);
        
        // Initialize optimization engines
        let handover_optimizer = Arc::new(
            opt_mob::HandoverOptimizer::new(config.handover.clone()).await?
        );
        
        let energy_optimizer = Arc::new(
            opt_eng::CellSleepOptimizer::new(config.energy.clone()).await?
        );
        
        let resource_optimizer = Arc::new(
            opt_res::CarrierAggregationOptimizer::new(config.resource.clone()).await?
        );
        
        // Initialize platform monitoring
        let platform_monitor = Arc::new(
            monitoring::PlatformMonitor::new(config.clone()).await?
        );
        
        // Initialize metrics
        let metrics = Arc::new(RwLock::new(PlatformMetrics {
            handover_metrics: opt_mob::HandoverMetrics::new(),
            energy_metrics: opt_eng::ForecastingMetrics::new(),
            resource_metrics: opt_res::CarrierAggregationMetrics::new(),
            platform_health: PlatformHealth {
                overall_health: HealthStatus::Unknown,
                handover_health: HealthStatus::Unknown,
                energy_health: HealthStatus::Unknown,
                resource_health: HealthStatus::Unknown,
                system_utilization: 0.0,
                error_rate: 0.0,
                uptime_seconds: 0,
            },
            last_updated: Utc::now(),
        }));
        
        Ok(Self {
            config,
            handover_optimizer,
            energy_optimizer,
            resource_optimizer,
            platform_monitor,
            metrics,
            start_time: Utc::now(),
        })
    }
    
    /// Start all optimization services
    pub async fn start_all_services(&self) -> Result<()> {
        log::info!("Starting RAN Intelligence Platform optimization services...");
        
        // Start individual optimizers
        self.handover_optimizer.start_monitoring().await?;
        self.energy_optimizer.start_monitoring().await?;
        self.resource_optimizer.start_monitoring().await?;
        
        // Start platform monitoring
        self.platform_monitor.start().await?;
        
        // Update health status
        self.update_platform_health().await?;
        
        log::info!("All optimization services started successfully");
        Ok(())
    }
    
    /// Stop all optimization services
    pub async fn stop_all_services(&self) -> Result<()> {
        log::info!("Stopping RAN Intelligence Platform optimization services...");
        
        // Stop individual optimizers
        self.handover_optimizer.stop_monitoring().await?;
        self.energy_optimizer.stop_monitoring().await?;
        self.resource_optimizer.stop_monitoring().await?;
        
        // Stop platform monitoring
        self.platform_monitor.stop().await?;
        
        log::info!("All optimization services stopped successfully");
        Ok(())
    }
    
    /// Get comprehensive platform metrics
    pub async fn get_platform_metrics(&self) -> Result<PlatformMetrics> {
        // Update metrics from individual services
        let handover_metrics = self.handover_optimizer.get_metrics().await?;
        let energy_metrics = self.energy_optimizer.get_metrics().await?;
        let resource_metrics = self.resource_optimizer.get_metrics().await?;
        
        let mut metrics = self.metrics.write().await;
        metrics.handover_metrics = handover_metrics;
        metrics.energy_metrics = energy_metrics;
        metrics.resource_metrics = resource_metrics;
        metrics.last_updated = Utc::now();
        
        // Update platform health
        let platform_health = self.calculate_platform_health(&metrics).await;
        metrics.platform_health = platform_health;
        
        Ok(metrics.clone())
    }
    
    /// Get platform health status
    pub async fn get_health_status(&self) -> Result<PlatformHealth> {
        let metrics = self.get_platform_metrics().await?;
        Ok(metrics.platform_health)
    }
    
    /// Execute comprehensive optimization for a network scenario
    pub async fn optimize_network(&self, scenario: &NetworkOptimizationScenario) -> Result<OptimizationResults> {
        log::info!("Executing comprehensive network optimization for scenario: {}", scenario.name);
        
        let start_time = std::time::Instant::now();
        let mut results = OptimizationResults::new(scenario.name.clone());
        
        // Handover optimization
        if let Some(ref handover_data) = scenario.handover_data {
            let handover_results = self.optimize_handovers(handover_data).await?;
            results.handover_results = Some(handover_results);
        }
        
        // Energy optimization
        if let Some(ref energy_data) = scenario.energy_data {
            let energy_results = self.optimize_energy(energy_data).await?;
            results.energy_results = Some(energy_results);
        }
        
        // Resource optimization
        if let Some(ref resource_data) = scenario.resource_data {
            let resource_results = self.optimize_resources(resource_data).await?;
            results.resource_results = Some(resource_results);
        }
        
        results.execution_time_ms = start_time.elapsed().as_millis() as f64;
        results.timestamp = Utc::now();
        
        log::info!("Network optimization completed in {:.2}ms", results.execution_time_ms);
        Ok(results)
    }
    
    /// Handover optimization
    async fn optimize_handovers(&self, data: &HandoverOptimizationData) -> Result<HandoverOptimizationResults> {
        let mut results = Vec::new();
        
        for ue_data in &data.ue_metrics {
            let neighbors = data.neighbor_cells.get(&ue_data.cell_id).unwrap_or(&vec![]);
            let prediction = self.handover_optimizer.predict_handover(ue_data, neighbors).await?;
            results.push(prediction);
        }
        
        Ok(HandoverOptimizationResults {
            predictions: results,
            total_ues: data.ue_metrics.len(),
            handover_candidates: results.iter().filter(|p| p.handover_probability > 0.5).count(),
        })
    }
    
    /// Energy optimization
    async fn optimize_energy(&self, data: &EnergyOptimizationData) -> Result<EnergyOptimizationResults> {
        let mut cell_results = std::collections::HashMap::new();
        let mut total_savings = 0.0;
        
        for (cell_id, historical_data) in &data.cell_utilization {
            let forecast = self.energy_optimizer.forecast_prb_utilization(cell_id, historical_data).await?;
            let sleep_windows = self.energy_optimizer.detect_sleep_opportunities(cell_id, &forecast).await?;
            let savings = self.energy_optimizer.calculate_energy_savings(&sleep_windows).await?;
            
            total_savings += savings;
            cell_results.insert(cell_id.clone(), EnergyOptimizationCellResult {
                forecast,
                sleep_windows,
                energy_savings_kwh: savings,
            });
        }
        
        Ok(EnergyOptimizationResults {
            cell_results,
            total_energy_savings_kwh: total_savings,
            optimization_horizon_hours: 1,
        })
    }
    
    /// Resource optimization
    async fn optimize_resources(&self, data: &ResourceOptimizationData) -> Result<ResourceOptimizationResults> {
        let recommendations = self.resource_optimizer.optimize_multi_ue_ca(
            &data.ue_demands,
            &data.available_scells,
        ).await?;
        
        let total_throughput_gain: f64 = recommendations.iter()
            .map(|r| r.expected_throughput_gain_mbps)
            .sum();
        
        Ok(ResourceOptimizationResults {
            scell_recommendations: recommendations,
            total_throughput_gain_mbps: total_throughput_gain,
            resource_efficiency_improvement: 0.0, // Would be calculated from actual metrics
        })
    }
    
    async fn update_platform_health(&self) -> Result<()> {
        let handover_metrics = self.handover_optimizer.get_metrics().await?;
        let energy_metrics = self.energy_optimizer.get_metrics().await?;
        let resource_metrics = self.resource_optimizer.get_metrics().await?;
        
        let mut metrics = self.metrics.write().await;
        
        // Determine health status for each service
        metrics.platform_health.handover_health = if handover_metrics.meets_targets() {
            HealthStatus::Healthy
        } else if handover_metrics.accuracy > 0.8 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };
        
        metrics.platform_health.energy_health = if energy_metrics.meets_targets() {
            HealthStatus::Healthy
        } else if energy_metrics.mape < 15.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };
        
        metrics.platform_health.resource_health = if resource_metrics.meets_targets() {
            HealthStatus::Healthy
        } else if resource_metrics.prediction_accuracy > 70.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };
        
        // Determine overall health
        metrics.platform_health.overall_health = match (
            &metrics.platform_health.handover_health,
            &metrics.platform_health.energy_health,
            &metrics.platform_health.resource_health,
        ) {
            (HealthStatus::Healthy, HealthStatus::Healthy, HealthStatus::Healthy) => HealthStatus::Healthy,
            (HealthStatus::Critical, _, _) | (_, HealthStatus::Critical, _) | (_, _, HealthStatus::Critical) => HealthStatus::Critical,
            _ => HealthStatus::Warning,
        };
        
        // Update uptime
        metrics.platform_health.uptime_seconds = (Utc::now() - self.start_time).num_seconds() as u64;
        
        Ok(())
    }
    
    async fn calculate_platform_health(&self, metrics: &PlatformMetrics) -> PlatformHealth {
        // Calculate system utilization (simplified)
        let avg_utilization = (
            metrics.handover_metrics.throughput_per_second / 10000.0 + // Normalize to 10k/sec max
            metrics.energy_metrics.forecast_latency_ms / 1000.0 + // Normalize to 1s max
            metrics.resource_metrics.prediction_latency_ms / 5.0   // Normalize to 5ms max
        ) / 3.0;
        
        // Calculate error rate (simplified)
        let error_rate = (
            metrics.handover_metrics.false_positive_rate +
            metrics.energy_metrics.false_positive_rate +
            metrics.resource_metrics.false_positive_rate
        ) / 3.0;
        
        PlatformHealth {
            overall_health: metrics.platform_health.overall_health.clone(),
            handover_health: metrics.platform_health.handover_health.clone(),
            energy_health: metrics.platform_health.energy_health.clone(),
            resource_health: metrics.platform_health.resource_health.clone(),
            system_utilization: avg_utilization.min(1.0),
            error_rate,
            uptime_seconds: (Utc::now() - self.start_time).num_seconds() as u64,
        }
    }
}

// Network optimization scenario types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimizationScenario {
    pub name: String,
    pub handover_data: Option<HandoverOptimizationData>,
    pub energy_data: Option<EnergyOptimizationData>,
    pub resource_data: Option<ResourceOptimizationData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverOptimizationData {
    pub ue_metrics: Vec<opt_mob::UeMetrics>,
    pub neighbor_cells: std::collections::HashMap<String, Vec<opt_mob::NeighborCell>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyOptimizationData {
    pub cell_utilization: std::collections::HashMap<String, Vec<opt_eng::PrbUtilization>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationData {
    pub ue_demands: Vec<opt_res::UeThroughputDemand>,
    pub available_scells: Vec<opt_res::SCellConfig>,
}

// Optimization results types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResults {
    pub scenario_name: String,
    pub handover_results: Option<HandoverOptimizationResults>,
    pub energy_results: Option<EnergyOptimizationResults>,
    pub resource_results: Option<ResourceOptimizationResults>,
    pub execution_time_ms: f64,
    pub timestamp: DateTime<Utc>,
}

impl OptimizationResults {
    fn new(scenario_name: String) -> Self {
        Self {
            scenario_name,
            handover_results: None,
            energy_results: None,
            resource_results: None,
            execution_time_ms: 0.0,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverOptimizationResults {
    pub predictions: Vec<opt_mob::HandoverPrediction>,
    pub total_ues: usize,
    pub handover_candidates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyOptimizationResults {
    pub cell_results: std::collections::HashMap<String, EnergyOptimizationCellResult>,
    pub total_energy_savings_kwh: f64,
    pub optimization_horizon_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyOptimizationCellResult {
    pub forecast: Vec<opt_eng::PrbUtilization>,
    pub sleep_windows: Vec<opt_eng::SleepWindow>,
    pub energy_savings_kwh: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimizationResults {
    pub scell_recommendations: Vec<opt_res::SCellRecommendation>,
    pub total_throughput_gain_mbps: f64,
    pub resource_efficiency_improvement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_platform_creation() {
        let config = PlatformConfig::default();
        let platform = OptimizationPlatform::new(config).await;
        assert!(platform.is_ok());
    }
    
    #[test]
    fn test_platform_config_default() {
        let config = PlatformConfig::default();
        assert!(config.handover.model.input_features > 0);
        assert!(config.energy.forecasting.min_data_points > 0);
        assert!(config.resource.prediction.confidence_threshold > 0.0);
    }
    
    #[test]
    fn test_health_status() {
        let health = PlatformHealth {
            overall_health: HealthStatus::Healthy,
            handover_health: HealthStatus::Healthy,
            energy_health: HealthStatus::Warning,
            resource_health: HealthStatus::Healthy,
            system_utilization: 0.7,
            error_rate: 0.02,
            uptime_seconds: 3600,
        };
        
        assert_eq!(health.overall_health, HealthStatus::Healthy);
        assert!(health.system_utilization < 1.0);
        assert!(health.error_rate < 0.1);
    }
}