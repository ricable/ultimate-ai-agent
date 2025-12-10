//! PFS (Performance Monitoring System) Module
//! 
//! This module provides comprehensive performance monitoring, profiling, and analytics
//! for the standalone swarm demo with real-time metrics collection and analysis.
//! 
//! ## Features
//! - **Advanced Tensor Operations**: SIMD-optimized tensor operations with swarm-specific enhancements
//! - **Real-time Performance Monitoring**: Comprehensive metrics collection and analysis
//! - **Enhanced Profiling**: Agent-aware profiling with operation tracing
//! - **Metrics Collection**: Real-time metrics aggregation with CSV data integration
//! - **Analytics Engine**: Advanced anomaly detection, trend analysis, and predictive insights
//! 
//! ## Integration with fanndata.csv
//! The PFS system seamlessly integrates with real CSV data processing, providing:
//! - Real-time data quality assessment
//! - Processing performance metrics
//! - Data-driven performance optimization
//! - Validation error tracking and analysis

pub mod advanced;
pub mod performance_monitor;
pub mod profiler;
pub mod metrics_collector;
pub mod real_time_analytics;

// Re-export main types for easy access
pub use advanced::{
    SwarmTensor, TensorStatistics, SwarmTensorPool, SwarmBatchProcessor,
    simd_ops, cache_oblivious_transpose_swarm
};

pub use performance_monitor::{
    NeuralSwarmPerformanceMonitor, SwarmPerformanceMetrics, SwarmPerformanceAlert,
    SwarmPerformanceDashboard, AlertSeverity
};

pub use profiler::{
    SwarmProfiler, AgentProfile, OperationTrace, SwarmProfileStats,
    profile_swarm, profile_swarm_async, increment_swarm_counter,
    SwarmMemoryTracker, track_swarm_memory_allocation, track_swarm_memory_deallocation
};

pub use metrics_collector::{
    SwarmMetricsCollector, MetricPoint, AggregatedMetric, AgentMetrics,
    MetricContext, MetricTrend, CsvProcessingStats, metrics_utils
};

pub use real_time_analytics::{
    SwarmRealTimeAnalytics, AnalyticsConfig, AnomalyResult, TrendAnalysisResult,
    PatternResult, PredictionResult, Insight, AnalyticsResults, InsightPriority
};

/// Re-export commonly used types from the main performance module
pub use crate::performance::PerformanceMetrics;

/// PFS system initialization and management
pub struct PFSSystem {
    performance_monitor: NeuralSwarmPerformanceMonitor,
    metrics_collector: SwarmMetricsCollector,
    analytics_engine: SwarmRealTimeAnalytics,
    profiler: SwarmProfiler,
    tensor_pool: SwarmTensorPool,
}

impl PFSSystem {
    /// Initialize the complete PFS system
    pub fn new() -> Self {
        Self {
            performance_monitor: NeuralSwarmPerformanceMonitor::new(5, 10000), // 5-second intervals, 10k history
            metrics_collector: SwarmMetricsCollector::new(1000, 50000), // 1-second collection, 50k buffer
            analytics_engine: SwarmRealTimeAnalytics::new(AnalyticsConfig::default()),
            profiler: SwarmProfiler::new(Some("pfs-system".to_string())),
            tensor_pool: SwarmTensorPool::new(),
        }
    }
    
    /// Start all PFS subsystems
    pub async fn start(&self) -> Result<(), String> {
        println!("🚀 Starting PFS (Performance Monitoring System)...");
        
        // Start performance monitoring
        self.performance_monitor.start_monitoring().await;
        
        // Start metrics collection
        self.metrics_collector.start_collection().await;
        
        println!("✅ PFS system started successfully");
        println!("   📊 Performance monitoring: Active");
        println!("   📈 Metrics collection: Active");
        println!("   🧠 Analytics engine: Ready");
        println!("   ⏱️ Profiler: Enabled");
        println!("   🧮 Tensor pool: Initialized");
        
        Ok(())
    }
    
    /// Load real data for enhanced monitoring
    pub async fn load_fanndata(&self, csv_path: &str) -> Result<(), String> {
        println!("📂 Loading real data from {}", csv_path);
        self.performance_monitor.load_real_data(csv_path).await?;
        println!("✅ Real data loaded successfully");
        Ok(())
    }
    
    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> PFSSystemStatus {
        let dashboard = self.performance_monitor.get_dashboard().await;
        let metrics = self.metrics_collector.get_aggregated_metrics().await;
        let profiler_stats = self.profiler.get_stats().await;
        let agents = self.metrics_collector.get_active_agents().await;
        
        PFSSystemStatus {
            dashboard,
            active_metrics_count: metrics.len(),
            active_agents_count: agents.len(),
            total_operations: profiler_stats.total_operations,
            average_performance_score: profiler_stats.average_performance_score,
            system_uptime: profiler_stats.session_duration,
            tensor_pool_stats: self.tensor_pool.get_stats(),
        }
    }
    
    /// Generate comprehensive system report
    pub async fn generate_system_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("🐝 PFS System Comprehensive Report\n");
        report.push_str("==================================\n\n");
        
        // Performance monitoring report
        let dashboard = self.performance_monitor.get_dashboard().await;
        report.push_str("📊 Performance Dashboard Summary:\n");
        report.push_str(&format!("├── Total Agents: {}\n", dashboard.system_overview.total_agents));
        report.push_str(&format!("├── Active Neural Networks: {}\n", dashboard.system_overview.active_neural_networks));
        report.push_str(&format!("├── System Health: {:.1}%\n", dashboard.system_overview.average_system_health * 100.0));
        report.push_str(&format!("└── CSV Files Processed: {}\n\n", dashboard.data_pipeline_stats.csv_files_processed));
        
        // Metrics collection report
        report.push_str(&self.metrics_collector.generate_metrics_report().await);
        
        // Profiler report
        report.push_str(&self.profiler.generate_report().await);
        
        report.push_str("=== End PFS System Report ===\n");
        report
    }
    
    /// Export all system data
    pub async fn export_system_data(&self, format: &str) -> Result<String, String> {
        match format {
            "json" => {
                let performance_data = self.performance_monitor.export_enhanced_performance_data("json").await?;
                let metrics_data = self.metrics_collector.export_metrics_json().await
                    .map_err(|e| format!("Failed to export metrics: {}", e))?;
                
                let combined_data = serde_json::json!({
                    "pfs_system_export": {
                        "timestamp": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        "performance_data": serde_json::from_str::<serde_json::Value>(&performance_data)
                            .unwrap_or(serde_json::Value::Null),
                        "metrics_data": serde_json::from_str::<serde_json::Value>(&metrics_data)
                            .unwrap_or(serde_json::Value::Null),
                        "tensor_pool_stats": {
                            "allocations": self.tensor_pool.get_stats().0,
                            "deallocations": self.tensor_pool.get_stats().1
                        }
                    }
                });
                
                Ok(serde_json::to_string_pretty(&combined_data).unwrap())
            },
            _ => Err("Unsupported export format".to_string())
        }
    }
    
    /// Access individual subsystems
    pub fn performance_monitor(&self) -> &NeuralSwarmPerformanceMonitor {
        &self.performance_monitor
    }
    
    pub fn metrics_collector(&self) -> &SwarmMetricsCollector {
        &self.metrics_collector
    }
    
    pub fn analytics_engine(&mut self) -> &mut SwarmRealTimeAnalytics {
        &mut self.analytics_engine
    }
    
    pub fn profiler(&self) -> &SwarmProfiler {
        &self.profiler
    }
    
    pub fn tensor_pool(&mut self) -> &mut SwarmTensorPool {
        &mut self.tensor_pool
    }
}

/// System status summary
#[derive(Debug, Clone)]
pub struct PFSSystemStatus {
    pub dashboard: SwarmPerformanceDashboard,
    pub active_metrics_count: usize,
    pub active_agents_count: usize,
    pub total_operations: u64,
    pub average_performance_score: f64,
    pub system_uptime: std::time::Duration,
    pub tensor_pool_stats: (usize, usize), // (allocations, deallocations)
}

/// Convenience functions for quick access to PFS functionality
pub mod pfs_utils {
    use super::*;
    
    /// Create a tensor from CSV data with automatic error handling
    pub fn create_tensor_from_csv(
        csv_data: &[Vec<f32>], 
        agent_id: &str
    ) -> Result<SwarmTensor, String> {
        SwarmTensor::from_csv_data(csv_data, Some(agent_id.to_string()))
    }
    
    /// Quick performance profiling for operations
    pub async fn quick_profile<F, R>(agent_id: &str, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        profile_swarm(agent_id, operation, f).await
    }
    
    /// Create standardized metric context for neural operations
    pub fn neural_metric_context(model_version: &str, batch_size: usize) -> MetricContext {
        metrics_utils::create_neural_context("neural_inference", model_version, batch_size)
    }
    
    /// Create standardized metric context for CSV processing
    pub fn csv_metric_context(file_path: &str, batch_size: usize) -> MetricContext {
        metrics_utils::create_csv_context(file_path, "processing", batch_size)
    }
}