//! Metrics Collection and Analysis
//! 
//! This module provides comprehensive metrics collection for monitoring
//! swarm performance and system health.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsPoint {
    pub timestamp: u64,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesMetrics {
    pub name: String,
    pub points: VecDeque<MetricsPoint>,
    pub max_points: usize,
}

impl TimeSeriesMetrics {
    pub fn new(name: String, max_points: usize) -> Self {
        Self {
            name,
            points: VecDeque::new(),
            max_points,
        }
    }
    
    pub fn add_point(&mut self, value: f64, tags: HashMap<String, String>) {
        let point = MetricsPoint {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            value,
            tags,
        };
        
        self.points.push_back(point);
        
        if self.points.len() > self.max_points {
            self.points.pop_front();
        }
    }
    
    pub fn get_latest(&self) -> Option<&MetricsPoint> {
        self.points.back()
    }
    
    pub fn get_average(&self, duration_seconds: u64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let cutoff_time = current_time.saturating_sub(duration_seconds);
        
        let recent_points: Vec<&MetricsPoint> = self.points.iter()
            .filter(|p| p.timestamp >= cutoff_time)
            .collect();
        
        if recent_points.is_empty() {
            return None;
        }
        
        let sum: f64 = recent_points.iter().map(|p| p.value).sum();
        Some(sum / recent_points.len() as f64)
    }
}

pub struct MetricsCollector {
    pub metrics: HashMap<String, TimeSeriesMetrics>,
    pub start_time: Instant,
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            start_time: Instant::now(),
            counters: HashMap::new(),
            gauges: HashMap::new(),
        }
    }
    
    pub fn record_metric(&mut self, name: &str, value: f64, tags: HashMap<String, String>) {
        let metrics = self.metrics.entry(name.to_string())
            .or_insert_with(|| TimeSeriesMetrics::new(name.to_string(), 1000));
        
        metrics.add_point(value, tags);
    }
    
    pub fn record_counter(&mut self, name: &str, increment: u64) {
        let counter = self.counters.entry(name.to_string()).or_insert(0);
        *counter += increment;
    }
    
    pub fn set_gauge(&mut self, name: &str, value: f64) {
        self.gauges.insert(name.to_string(), value);
    }
    
    pub fn record_timing(&mut self, name: &str, duration: Duration) {
        let tags = HashMap::new();
        self.record_metric(name, duration.as_secs_f64(), tags);
    }
    
    pub fn record_fitness(&mut self, agent_id: &str, fitness: f32, iteration: u32) {
        let mut tags = HashMap::new();
        tags.insert("agent_id".to_string(), agent_id.to_string());
        tags.insert("iteration".to_string(), iteration.to_string());
        
        self.record_metric("agent_fitness", fitness as f64, tags);
    }
    
    pub fn record_convergence(&mut self, iteration: u32, best_fitness: f32, diversity: f32) {
        let mut tags = HashMap::new();
        tags.insert("iteration".to_string(), iteration.to_string());
        
        self.record_metric("best_fitness", best_fitness as f64, tags.clone());
        self.record_metric("population_diversity", diversity as f64, tags);
    }
    
    pub fn record_system_metrics(&mut self) {
        // Memory usage
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                self.set_gauge("memory_usage_mb", kb / 1024.0);
                            }
                        }
                    }
                }
            }
        }
        
        // CPU usage (simplified)
        let cpu_time = self.start_time.elapsed().as_secs_f64();
        self.set_gauge("cpu_time_seconds", cpu_time);
        
        // Uptime
        let uptime = self.start_time.elapsed().as_secs_f64();
        self.set_gauge("uptime_seconds", uptime);
    }
    
    pub fn get_metric_summary(&self, name: &str) -> Option<MetricSummary> {
        let metrics = self.metrics.get(name)?;
        
        if metrics.points.is_empty() {
            return None;
        }
        
        let values: Vec<f64> = metrics.points.iter().map(|p| p.value).collect();
        
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = values.iter().sum();
        let count = values.len();
        let mean = sum / count as f64;
        
        // Calculate standard deviation
        let variance: f64 = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        // Calculate percentiles
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50 = percentile(&sorted_values, 0.5);
        let p95 = percentile(&sorted_values, 0.95);
        let p99 = percentile(&sorted_values, 0.99);
        
        Some(MetricSummary {
            name: name.to_string(),
            count,
            min,
            max,
            mean,
            std_dev,
            p50,
            p95,
            p99,
        })
    }
    
    pub fn export_metrics(&self) -> MetricsExport {
        let mut metric_summaries = HashMap::new();
        
        for name in self.metrics.keys() {
            if let Some(summary) = self.get_metric_summary(name) {
                metric_summaries.insert(name.clone(), summary);
            }
        }
        
        MetricsExport {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            metrics: metric_summaries,
            counters: self.counters.clone(),
            gauges: self.gauges.clone(),
        }
    }
    
    pub fn clear_metrics(&mut self) {
        self.metrics.clear();
        self.counters.clear();
        self.gauges.clear();
    }
    
    pub fn get_performance_report(&self) -> PerformanceReport {
        let total_metrics: usize = self.metrics.values()
            .map(|m| m.points.len())
            .sum();
        
        let memory_usage = self.gauges.get("memory_usage_mb")
            .copied()
            .unwrap_or(0.0);
        
        let uptime = self.start_time.elapsed();
        
        PerformanceReport {
            uptime,
            total_metrics_collected: total_metrics,
            active_metric_streams: self.metrics.len(),
            memory_usage_mb: memory_usage,
            counters_total: self.counters.values().sum(),
            gauges_count: self.gauges.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub name: String,
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExport {
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub metrics: HashMap<String, MetricSummary>,
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub uptime: Duration,
    pub total_metrics_collected: usize,
    pub active_metric_streams: usize,
    pub memory_usage_mb: f64,
    pub counters_total: u64,
    pub gauges_count: usize,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    
    let index = (p * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collection() {
        let mut collector = MetricsCollector::new();
        
        let tags = HashMap::new();
        collector.record_metric("test_metric", 10.0, tags);
        collector.record_counter("test_counter", 5);
        collector.set_gauge("test_gauge", 3.14);
        
        assert!(collector.metrics.contains_key("test_metric"));
        assert_eq!(collector.counters.get("test_counter"), Some(&5));
        assert_eq!(collector.gauges.get("test_gauge"), Some(&3.14));
    }
    
    #[test]
    fn test_metric_summary() {
        let mut collector = MetricsCollector::new();
        let tags = HashMap::new();
        
        // Add some test data
        for i in 1..=10 {
            collector.record_metric("test", i as f64, tags.clone());
        }
        
        let summary = collector.get_metric_summary("test").unwrap();
        assert_eq!(summary.count, 10);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 10.0);
        assert_eq!(summary.mean, 5.5);
    }
}