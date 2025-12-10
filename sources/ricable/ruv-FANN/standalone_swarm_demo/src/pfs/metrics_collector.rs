//! Metrics Collection and Aggregation for Neural Swarm
//! 
//! This module provides real-time metrics collection, aggregation, and analysis
//! for neural swarm operations with integration to fanndata.csv processing.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use crate::utils::csv_data_parser::CsvDataParser;
use super::advanced::{SwarmTensor, TensorStatistics};

/// Real-time metrics collector for swarm operations
pub struct SwarmMetricsCollector {
    metrics_buffer: Arc<RwLock<VecDeque<MetricPoint>>>,
    aggregated_metrics: Arc<RwLock<HashMap<String, AggregatedMetric>>>,
    csv_parser: CsvDataParser,
    collection_interval: std::time::Duration,
    buffer_size_limit: usize,
    active_agents: Arc<RwLock<HashMap<String, AgentMetrics>>>,
}

/// Individual metric point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: u64,
    pub agent_id: String,
    pub metric_name: String,
    pub value: f64,
    pub unit: String,
    pub tags: HashMap<String, String>,
    pub context: MetricContext,
}

/// Context information for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricContext {
    pub operation_type: String,
    pub data_source: String, // e.g., "fanndata.csv", "synthetic", "real-time"
    pub processing_stage: String, // e.g., "ingestion", "training", "inference"
    pub model_version: String,
    pub batch_size: usize,
}

/// Aggregated metric with statistical information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    pub metric_name: String,
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub last_value: f64,
    pub last_updated: u64,
    pub trend: MetricTrend,
    pub anomaly_score: f64,
}

/// Trend information for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Unknown,
}

/// Agent-specific metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub agent_id: String,
    pub agent_type: String,
    pub metrics: HashMap<String, AggregatedMetric>,
    pub health_score: f64,
    pub last_activity: u64,
    pub csv_processing_stats: CsvProcessingStats,
}

/// Statistics for CSV data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvProcessingStats {
    pub files_processed: u64,
    pub records_processed: u64,
    pub processing_time_ms: f64,
    pub validation_errors: u64,
    pub data_quality_score: f64,
    pub feature_extraction_time_ms: f64,
}

impl SwarmMetricsCollector {
    pub fn new(collection_interval_ms: u64, buffer_size_limit: usize) -> Self {
        Self {
            metrics_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size_limit))),
            aggregated_metrics: Arc::new(RwLock::new(HashMap::new())),
            csv_parser: CsvDataParser::new(),
            collection_interval: std::time::Duration::from_millis(collection_interval_ms),
            buffer_size_limit,
            active_agents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Start the metrics collection process
    pub async fn start_collection(&self) {
        let metrics_buffer = Arc::clone(&self.metrics_buffer);
        let aggregated_metrics = Arc::clone(&self.aggregated_metrics);
        let active_agents = Arc::clone(&self.active_agents);
        let interval = self.collection_interval;
        
        tokio::spawn(async move {
            let mut collection_timer = tokio::time::interval(interval);
            
            loop {
                collection_timer.tick().await;
                
                // Process buffered metrics
                Self::process_buffered_metrics(&metrics_buffer, &aggregated_metrics, &active_agents).await;
                
                // Update agent health scores
                Self::update_agent_health_scores(&active_agents).await;
                
                // Clean up old metrics
                Self::cleanup_old_metrics(&metrics_buffer, &aggregated_metrics).await;
            }
        });
    }
    
    /// Record a metric point
    pub async fn record_metric(
        &self,
        agent_id: &str,
        metric_name: &str,
        value: f64,
        unit: &str,
        tags: HashMap<String, String>,
        context: MetricContext,
    ) {
        let metric_point = MetricPoint {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            agent_id: agent_id.to_string(),
            metric_name: metric_name.to_string(),
            value,
            unit: unit.to_string(),
            tags,
            context,
        };
        
        let mut buffer = self.metrics_buffer.write().await;
        if buffer.len() >= self.buffer_size_limit {
            buffer.pop_front(); // Remove oldest metric
        }
        buffer.push_back(metric_point);
    }
    
    /// Record CSV processing metrics
    pub async fn record_csv_processing_metrics(
        &self,
        agent_id: &str,
        csv_file_path: &str,
        processing_time: std::time::Duration,
        records_processed: u64,
        validation_errors: u64,
    ) -> Result<(), String> {
        // Parse the CSV file to get additional statistics
        let csv_data = self.csv_parser.parse_csv_file(csv_file_path)?;
        let tensor = SwarmTensor::from_csv_data(&csv_data, Some(agent_id.to_string()))?;
        let stats = tensor.compute_statistics();
        
        // Calculate data quality score based on various factors
        let data_quality_score = self.calculate_data_quality_score(&stats, validation_errors, records_processed);
        
        // Record individual metrics
        let context = MetricContext {
            operation_type: "csv_processing".to_string(),
            data_source: csv_file_path.to_string(),
            processing_stage: "ingestion".to_string(),
            model_version: "1.0".to_string(),
            batch_size: records_processed as usize,
        };
        
        let mut tags = HashMap::new();
        tags.insert("data_source".to_string(), "fanndata".to_string());
        tags.insert("file_path".to_string(), csv_file_path.to_string());
        
        // Record processing time
        self.record_metric(
            agent_id,
            "csv_processing_time_ms",
            processing_time.as_millis() as f64,
            "milliseconds",
            tags.clone(),
            context.clone(),
        ).await;
        
        // Record data quality metrics
        self.record_metric(
            agent_id,
            "data_quality_score",
            data_quality_score,
            "score",
            tags.clone(),
            context.clone(),
        ).await;
        
        // Record data statistics
        self.record_metric(
            agent_id,
            "data_mean",
            stats.mean as f64,
            "value",
            tags.clone(),
            context.clone(),
        ).await;
        
        self.record_metric(
            agent_id,
            "data_std_dev",
            stats.std_dev as f64,
            "value",
            tags.clone(),
            context.clone(),
        ).await;
        
        self.record_metric(
            agent_id,
            "records_processed",
            records_processed as f64,
            "count",
            tags.clone(),
            context.clone(),
        ).await;
        
        self.record_metric(
            agent_id,
            "validation_errors",
            validation_errors as f64,
            "count",
            tags,
            context,
        ).await;
        
        // Update agent CSV processing stats
        self.update_agent_csv_stats(agent_id, records_processed, processing_time, validation_errors, data_quality_score).await;
        
        Ok(())
    }
    
    /// Calculate data quality score based on various factors
    fn calculate_data_quality_score(&self, stats: &TensorStatistics, validation_errors: u64, total_records: u64) -> f64 {
        let mut quality_score = 1.0;
        
        // Factor in validation errors
        if total_records > 0 {
            let error_rate = validation_errors as f64 / total_records as f64;
            quality_score *= 1.0 - error_rate.min(1.0);
        }
        
        // Factor in data distribution (prefer lower variance relative to mean)
        if stats.mean != 0.0 {
            let coefficient_of_variation = stats.std_dev / stats.mean.abs();
            quality_score *= (1.0 / (1.0 + coefficient_of_variation)).max(0.1);
        }
        
        // Factor in data completeness (no NaN/infinite values assumed in our case)
        // This would be more sophisticated in a real implementation
        
        quality_score.max(0.0).min(1.0)
    }
    
    /// Update agent CSV processing statistics
    async fn update_agent_csv_stats(
        &self,
        agent_id: &str,
        records_processed: u64,
        processing_time: std::time::Duration,
        validation_errors: u64,
        data_quality_score: f64,
    ) {
        let mut agents = self.active_agents.write().await;
        let agent = agents.entry(agent_id.to_string()).or_insert_with(|| AgentMetrics {
            agent_id: agent_id.to_string(),
            agent_type: "data_processor".to_string(),
            metrics: HashMap::new(),
            health_score: 1.0,
            last_activity: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            csv_processing_stats: CsvProcessingStats {
                files_processed: 0,
                records_processed: 0,
                processing_time_ms: 0.0,
                validation_errors: 0,
                data_quality_score: 1.0,
                feature_extraction_time_ms: 0.0,
            },
        });
        
        agent.csv_processing_stats.files_processed += 1;
        agent.csv_processing_stats.records_processed += records_processed;
        agent.csv_processing_stats.processing_time_ms += processing_time.as_millis() as f64;
        agent.csv_processing_stats.validation_errors += validation_errors;
        
        // Update running average of data quality score
        let total_files = agent.csv_processing_stats.files_processed as f64;
        agent.csv_processing_stats.data_quality_score = 
            (agent.csv_processing_stats.data_quality_score * (total_files - 1.0) + data_quality_score) / total_files;
        
        agent.last_activity = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }
    
    /// Process buffered metrics and update aggregations
    async fn process_buffered_metrics(
        metrics_buffer: &Arc<RwLock<VecDeque<MetricPoint>>>,
        aggregated_metrics: &Arc<RwLock<HashMap<String, AggregatedMetric>>>,
        active_agents: &Arc<RwLock<HashMap<String, AgentMetrics>>>,
    ) {
        let mut buffer = metrics_buffer.write().await;
        let mut aggregated = aggregated_metrics.write().await;
        let mut agents = active_agents.write().await;
        
        while let Some(metric_point) = buffer.pop_front() {
            // Update global aggregated metrics
            let key = format!("{}_{}", metric_point.agent_id, metric_point.metric_name);
            let aggregated_metric = aggregated.entry(key).or_insert_with(|| AggregatedMetric {
                metric_name: metric_point.metric_name.clone(),
                count: 0,
                sum: 0.0,
                mean: 0.0,
                min: f64::MAX,
                max: f64::MIN,
                std_dev: 0.0,
                last_value: 0.0,
                last_updated: metric_point.timestamp,
                trend: MetricTrend::Unknown,
                anomaly_score: 0.0,
            });
            
            // Update aggregated metric
            aggregated_metric.count += 1;
            aggregated_metric.sum += metric_point.value;
            aggregated_metric.mean = aggregated_metric.sum / aggregated_metric.count as f64;
            aggregated_metric.min = aggregated_metric.min.min(metric_point.value);
            aggregated_metric.max = aggregated_metric.max.max(metric_point.value);
            aggregated_metric.last_value = metric_point.value;
            aggregated_metric.last_updated = metric_point.timestamp;
            
            // Calculate standard deviation (simplified online algorithm)
            if aggregated_metric.count > 1 {
                let variance = (metric_point.value - aggregated_metric.mean).powi(2) / aggregated_metric.count as f64;
                aggregated_metric.std_dev = variance.sqrt();
            }
            
            // Update trend
            aggregated_metric.trend = Self::calculate_trend(aggregated_metric);
            
            // Update agent-specific metrics
            let agent = agents.entry(metric_point.agent_id.clone()).or_insert_with(|| AgentMetrics {
                agent_id: metric_point.agent_id.clone(),
                agent_type: "unknown".to_string(),
                metrics: HashMap::new(),
                health_score: 1.0,
                last_activity: metric_point.timestamp,
                csv_processing_stats: CsvProcessingStats {
                    files_processed: 0,
                    records_processed: 0,
                    processing_time_ms: 0.0,
                    validation_errors: 0,
                    data_quality_score: 1.0,
                    feature_extraction_time_ms: 0.0,
                },
            });
            
            agent.metrics.insert(metric_point.metric_name.clone(), aggregated_metric.clone());
            agent.last_activity = metric_point.timestamp;
        }
    }
    
    /// Calculate metric trend based on recent values
    fn calculate_trend(metric: &AggregatedMetric) -> MetricTrend {
        if metric.count < 3 {
            return MetricTrend::Unknown;
        }
        
        // Simplified trend calculation based on mean and recent values
        let recent_change = (metric.last_value - metric.mean) / metric.mean.abs();
        
        match recent_change {
            x if x > 0.1 => MetricTrend::Increasing,
            x if x < -0.1 => MetricTrend::Decreasing,
            x if x.abs() > 0.05 => MetricTrend::Volatile,
            _ => MetricTrend::Stable,
        }
    }
    
    /// Update health scores for all agents
    async fn update_agent_health_scores(active_agents: &Arc<RwLock<HashMap<String, AgentMetrics>>>) {
        let mut agents = active_agents.write().await;
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        for (_, agent) in agents.iter_mut() {
            // Calculate health score based on various factors
            let mut health_score = 1.0;
            
            // Factor in recent activity
            let time_since_activity = current_time.saturating_sub(agent.last_activity);
            if time_since_activity > 300 { // 5 minutes
                health_score *= 0.5; // Reduce health if agent inactive
            }
            
            // Factor in error rates
            if agent.csv_processing_stats.records_processed > 0 {
                let error_rate = agent.csv_processing_stats.validation_errors as f64 / 
                                agent.csv_processing_stats.records_processed as f64;
                health_score *= 1.0 - error_rate.min(1.0);
            }
            
            // Factor in data quality
            health_score *= agent.csv_processing_stats.data_quality_score;
            
            agent.health_score = health_score.max(0.0).min(1.0);
        }
    }
    
    /// Clean up old metrics to prevent memory leaks
    async fn cleanup_old_metrics(
        metrics_buffer: &Arc<RwLock<VecDeque<MetricPoint>>>,
        aggregated_metrics: &Arc<RwLock<HashMap<String, AggregatedMetric>>>,
    ) {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let cutoff_time = current_time - 3600; // Keep 1 hour of data
        
        // Clean buffer
        {
            let mut buffer = metrics_buffer.write().await;
            buffer.retain(|metric| metric.timestamp > cutoff_time);
        }
        
        // Clean aggregated metrics (remove very old ones)
        {
            let mut aggregated = aggregated_metrics.write().await;
            aggregated.retain(|_, metric| metric.last_updated > cutoff_time);
        }
    }
    
    /// Get current aggregated metrics
    pub async fn get_aggregated_metrics(&self) -> HashMap<String, AggregatedMetric> {
        self.aggregated_metrics.read().await.clone()
    }
    
    /// Get metrics for a specific agent
    pub async fn get_agent_metrics(&self, agent_id: &str) -> Option<AgentMetrics> {
        let agents = self.active_agents.read().await;
        agents.get(agent_id).cloned()
    }
    
    /// Get all active agents
    pub async fn get_active_agents(&self) -> HashMap<String, AgentMetrics> {
        self.active_agents.read().await.clone()
    }
    
    /// Get recent metric points for a specific metric
    pub async fn get_recent_metrics(&self, metric_name: &str, limit: usize) -> Vec<MetricPoint> {
        let buffer = self.metrics_buffer.read().await;
        buffer.iter()
            .filter(|metric| metric.metric_name == metric_name)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Generate a comprehensive metrics report
    pub async fn generate_metrics_report(&self) -> String {
        let aggregated = self.get_aggregated_metrics().await;
        let agents = self.get_active_agents().await;
        
        let mut report = String::new();
        report.push_str("📊 Swarm Metrics Collection Report\n");
        report.push_str("==================================\n\n");
        
        // Summary statistics
        report.push_str(&format!("Total Metrics: {}\n", aggregated.len()));
        report.push_str(&format!("Active Agents: {}\n", agents.len()));
        
        // Agent health overview
        if !agents.is_empty() {
            let avg_health: f64 = agents.values().map(|a| a.health_score).sum::<f64>() / agents.len() as f64;
            report.push_str(&format!("Average Agent Health: {:.3}\n\n", avg_health));
            
            // Agent details
            report.push_str("🤖 Agent Health Status:\n");
            for (agent_id, agent) in &agents {
                let health_status = match agent.health_score {
                    x if x >= 0.8 => "🟢 Healthy",
                    x if x >= 0.6 => "🟡 Warning",
                    x if x >= 0.4 => "🟠 Degraded",
                    _ => "🔴 Critical",
                };
                report.push_str(&format!("├── {}: {} ({:.3})\n", agent_id, health_status, agent.health_score));
            }
            report.push_str("\n");
        }
        
        // CSV processing statistics
        let csv_agents: Vec<_> = agents.values()
            .filter(|a| a.csv_processing_stats.files_processed > 0)
            .collect();
        
        if !csv_agents.is_empty() {
            report.push_str("📂 CSV Processing Statistics:\n");
            for agent in csv_agents {
                report.push_str(&format!("├── {}: {} files, {} records\n", 
                    agent.agent_id, 
                    agent.csv_processing_stats.files_processed,
                    agent.csv_processing_stats.records_processed
                ));
                report.push_str(&format!("│   ├── Quality Score: {:.3}\n", agent.csv_processing_stats.data_quality_score));
                report.push_str(&format!("│   ├── Avg Processing Time: {:.2}ms\n", 
                    agent.csv_processing_stats.processing_time_ms / agent.csv_processing_stats.files_processed as f64));
                report.push_str(&format!("│   └── Validation Errors: {}\n", agent.csv_processing_stats.validation_errors));
            }
            report.push_str("\n");
        }
        
        // Top metrics by activity
        if !aggregated.is_empty() {
            let mut sorted_metrics: Vec<_> = aggregated.iter().collect();
            sorted_metrics.sort_by(|a, b| b.1.count.cmp(&a.1.count));
            
            report.push_str("📈 Most Active Metrics:\n");
            for (metric_name, metric) in sorted_metrics.iter().take(10) {
                let trend_icon = match metric.trend {
                    MetricTrend::Increasing => "📈",
                    MetricTrend::Decreasing => "📉",
                    MetricTrend::Stable => "➡️",
                    MetricTrend::Volatile => "📊",
                    MetricTrend::Unknown => "❓",
                };
                report.push_str(&format!("├── {}: {} {} (Count: {}, Last: {:.3})\n", 
                    metric_name, trend_icon, metric.metric_name, metric.count, metric.last_value));
            }
        }
        
        report.push_str("\n=== End Report ===\n");
        report
    }
    
    /// Export metrics to JSON format
    pub async fn export_metrics_json(&self) -> Result<String, serde_json::Error> {
        let aggregated = self.get_aggregated_metrics().await;
        let agents = self.get_active_agents().await;
        
        let export_data = serde_json::json!({
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "aggregated_metrics": aggregated,
            "agents": agents,
            "summary": {
                "total_metrics": aggregated.len(),
                "active_agents": agents.len(),
                "average_health": if !agents.is_empty() {
                    agents.values().map(|a| a.health_score).sum::<f64>() / agents.len() as f64
                } else {
                    0.0
                }
            }
        });
        
        serde_json::to_string_pretty(&export_data)
    }
}

/// Utility functions for metric operations
pub mod metrics_utils {
    use super::*;
    
    /// Create a standardized metric context for neural network operations
    pub fn create_neural_context(
        operation_type: &str,
        model_version: &str,
        batch_size: usize,
    ) -> MetricContext {
        MetricContext {
            operation_type: operation_type.to_string(),
            data_source: "neural_network".to_string(),
            processing_stage: "training".to_string(),
            model_version: model_version.to_string(),
            batch_size,
        }
    }
    
    /// Create a standardized metric context for CSV processing
    pub fn create_csv_context(
        file_path: &str,
        processing_stage: &str,
        batch_size: usize,
    ) -> MetricContext {
        MetricContext {
            operation_type: "csv_processing".to_string(),
            data_source: file_path.to_string(),
            processing_stage: processing_stage.to_string(),
            model_version: "1.0".to_string(),
            batch_size,
        }
    }
    
    /// Create standardized tags for swarm operations
    pub fn create_swarm_tags(agent_type: &str, operation_class: &str) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        tags.insert("agent_type".to_string(), agent_type.to_string());
        tags.insert("operation_class".to_string(), operation_class.to_string());
        tags.insert("system".to_string(), "neural_swarm".to_string());
        tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = SwarmMetricsCollector::new(1000, 10000);
        let metrics = collector.get_aggregated_metrics().await;
        assert!(metrics.is_empty());
    }
    
    #[tokio::test]
    async fn test_metric_recording() {
        let collector = SwarmMetricsCollector::new(1000, 10000);
        
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        
        let context = MetricContext {
            operation_type: "test".to_string(),
            data_source: "test".to_string(),
            processing_stage: "test".to_string(),
            model_version: "1.0".to_string(),
            batch_size: 100,
        };
        
        collector.record_metric(
            "test-agent",
            "test_metric",
            42.0,
            "units",
            tags,
            context,
        ).await;
        
        // Process metrics
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let metrics = collector.get_aggregated_metrics().await;
        assert!(metrics.contains_key("test-agent_test_metric"));
    }
    
    #[tokio::test]
    async fn test_agent_metrics() {
        let collector = SwarmMetricsCollector::new(1000, 10000);
        
        // Simulate CSV processing
        collector.update_agent_csv_stats(
            "csv-agent",
            1000,
            Duration::from_millis(500),
            5,
            0.95,
        ).await;
        
        let agent_metrics = collector.get_agent_metrics("csv-agent").await;
        assert!(agent_metrics.is_some());
        
        let agent = agent_metrics.unwrap();
        assert_eq!(agent.csv_processing_stats.records_processed, 1000);
        assert_eq!(agent.csv_processing_stats.validation_errors, 5);
        assert!((agent.csv_processing_stats.data_quality_score - 0.95).abs() < 0.01);
    }
}