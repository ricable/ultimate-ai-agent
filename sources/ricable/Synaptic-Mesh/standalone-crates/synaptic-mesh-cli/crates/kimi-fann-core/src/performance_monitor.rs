//! Real-time Performance Monitoring System
//! 
//! Comprehensive performance monitoring for neural inference with sub-millisecond
//! tracking, bottleneck detection, and automatic optimization recommendations.

use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use crate::ExpertDomain;

/// Target performance metrics
pub const TARGET_INFERENCE_MS: f64 = 50.0;
pub const TARGET_MEMORY_MB: f64 = 25.0;
pub const TARGET_THROUGHPUT_OPS_SEC: f64 = 1000.0;
pub const TARGET_P2P_LATENCY_MS: f64 = 1.0;

/// Real-time performance monitor with microsecond precision
#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    start_time: Instant,
    sample_buffer: VecDeque<PerformanceSample>,
    max_samples: usize,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
            start_time: Instant::now(),
            sample_buffer: VecDeque::with_capacity(1000),
            max_samples: 1000,
        }
    }
    
    /// Start timing an operation
    pub fn start_operation(&self, operation_type: OperationType) -> OperationTimer {
        OperationTimer::new(operation_type, Arc::clone(&self.metrics))
    }
    
    /// Record memory usage
    pub fn record_memory_usage(&mut self, usage_mb: f64) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.current_memory_mb = usage_mb;
            metrics.peak_memory_mb = metrics.peak_memory_mb.max(usage_mb);
            metrics.memory_samples.push(usage_mb);
            
            // Keep only recent samples
            if metrics.memory_samples.len() > 100 {
                metrics.memory_samples.remove(0);
            }
        }
    }
    
    /// Record throughput measurement
    pub fn record_throughput(&mut self, ops_completed: u64, duration: Duration) {
        let ops_per_sec = ops_completed as f64 / duration.as_secs_f64();
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.current_throughput = ops_per_sec;
            metrics.peak_throughput = metrics.peak_throughput.max(ops_per_sec);
        }
    }
    
    /// Add performance sample to buffer
    pub fn add_sample(&mut self, sample: PerformanceSample) {
        self.sample_buffer.push_back(sample);
        if self.sample_buffer.len() > self.max_samples {
            self.sample_buffer.pop_front();
        }
    }
    
    /// Get current performance snapshot
    pub fn get_snapshot(&self) -> PerformanceSnapshot {
        let metrics = self.metrics.lock().unwrap();
        let uptime = self.start_time.elapsed();
        
        PerformanceSnapshot {
            uptime_ms: uptime.as_millis() as f64,
            current_metrics: metrics.clone(),
            bottlenecks: self.detect_bottlenecks(&metrics),
            recommendations: self.generate_recommendations(&metrics),
            targets_met: self.check_targets(&metrics),
        }
    }
    
    /// Detect performance bottlenecks
    fn detect_bottlenecks(&self, metrics: &PerformanceMetrics) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();
        
        // Check inference latency
        if metrics.avg_inference_ms > TARGET_INFERENCE_MS {
            bottlenecks.push(PerformanceBottleneck {
                component: "Neural Inference".to_string(),
                metric: "Latency".to_string(),
                current_value: metrics.avg_inference_ms,
                target_value: TARGET_INFERENCE_MS,
                severity: if metrics.avg_inference_ms > TARGET_INFERENCE_MS * 2.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::Warning
                },
                suggested_fix: "Enable SIMD optimizations, reduce model complexity".to_string(),
            });
        }
        
        // Check memory usage
        if metrics.current_memory_mb > TARGET_MEMORY_MB {
            bottlenecks.push(PerformanceBottleneck {
                component: "Memory Management".to_string(),
                metric: "Usage".to_string(),
                current_value: metrics.current_memory_mb,
                target_value: TARGET_MEMORY_MB,
                severity: if metrics.current_memory_mb > TARGET_MEMORY_MB * 2.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::Warning
                },
                suggested_fix: "Implement memory pooling, reduce batch sizes".to_string(),
            });
        }
        
        // Check throughput
        if metrics.current_throughput < TARGET_THROUGHPUT_OPS_SEC {
            bottlenecks.push(PerformanceBottleneck {
                component: "Processing Pipeline".to_string(),
                metric: "Throughput".to_string(),
                current_value: metrics.current_throughput,
                target_value: TARGET_THROUGHPUT_OPS_SEC,
                severity: if metrics.current_throughput < TARGET_THROUGHPUT_OPS_SEC * 0.5 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::Warning
                },
                suggested_fix: "Increase parallelism, optimize routing logic".to_string(),
            });
        }
        
        // Check P2P latency
        if metrics.avg_p2p_latency_ms > TARGET_P2P_LATENCY_MS {
            bottlenecks.push(PerformanceBottleneck {
                component: "P2P Communication".to_string(),
                metric: "Latency".to_string(),
                current_value: metrics.avg_p2p_latency_ms,
                target_value: TARGET_P2P_LATENCY_MS,
                severity: BottleneckSeverity::Warning,
                suggested_fix: "Optimize message serialization, use UDP for low-latency".to_string(),
            });
        }
        
        bottlenecks
    }
    
    /// Generate optimization recommendations
    fn generate_recommendations(&self, metrics: &PerformanceMetrics) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // SIMD recommendations
        if metrics.simd_utilization < 0.8 {
            recommendations.push(OptimizationRecommendation {
                category: "SIMD Optimization".to_string(),
                description: "Enable AVX2/FMA instructions for matrix operations".to_string(),
                expected_improvement: "5-8x speedup for neural inference".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                priority: Priority::High,
            });
        }
        
        // Memory optimization
        if metrics.memory_fragmentation > 0.3 {
            recommendations.push(OptimizationRecommendation {
                category: "Memory Management".to_string(),
                description: "Implement memory pooling to reduce fragmentation".to_string(),
                expected_improvement: "30-50% memory efficiency improvement".to_string(),
                implementation_effort: ImplementationEffort::High,
                priority: Priority::Medium,
            });
        }
        
        // GPU acceleration
        if !metrics.gpu_acceleration_enabled {
            recommendations.push(OptimizationRecommendation {
                category: "GPU Acceleration".to_string(),
                description: "Enable WebGPU compute shaders for parallel processing".to_string(),
                expected_improvement: "10-100x speedup for large models".to_string(),
                implementation_effort: ImplementationEffort::High,
                priority: Priority::Medium,
            });
        }
        
        // Caching optimization
        if metrics.cache_hit_rate < 0.7 {
            recommendations.push(OptimizationRecommendation {
                category: "Caching Strategy".to_string(),
                description: "Implement LRU cache for frequent computations".to_string(),
                expected_improvement: "2-3x improvement for repeated queries".to_string(),
                implementation_effort: ImplementationEffort::Low,
                priority: Priority::High,
            });
        }
        
        recommendations
    }
    
    /// Check if performance targets are met
    fn check_targets(&self, metrics: &PerformanceMetrics) -> PerformanceTargets {
        PerformanceTargets {
            inference_latency_met: metrics.avg_inference_ms <= TARGET_INFERENCE_MS,
            memory_usage_met: metrics.current_memory_mb <= TARGET_MEMORY_MB,
            throughput_met: metrics.current_throughput >= TARGET_THROUGHPUT_OPS_SEC,
            p2p_latency_met: metrics.avg_p2p_latency_ms <= TARGET_P2P_LATENCY_MS,
            overall_score: self.calculate_performance_score(metrics),
        }
    }
    
    /// Calculate overall performance score (0-100)
    fn calculate_performance_score(&self, metrics: &PerformanceMetrics) -> f64 {
        let mut score = 100.0;
        
        // Latency penalty
        if metrics.avg_inference_ms > TARGET_INFERENCE_MS {
            score -= (metrics.avg_inference_ms / TARGET_INFERENCE_MS - 1.0) * 30.0;
        }
        
        // Memory penalty
        if metrics.current_memory_mb > TARGET_MEMORY_MB {
            score -= (metrics.current_memory_mb / TARGET_MEMORY_MB - 1.0) * 25.0;
        }
        
        // Throughput penalty
        if metrics.current_throughput < TARGET_THROUGHPUT_OPS_SEC {
            score -= (1.0 - metrics.current_throughput / TARGET_THROUGHPUT_OPS_SEC) * 25.0;
        }
        
        // P2P latency penalty
        if metrics.avg_p2p_latency_ms > TARGET_P2P_LATENCY_MS {
            score -= (metrics.avg_p2p_latency_ms / TARGET_P2P_LATENCY_MS - 1.0) * 20.0;
        }
        
        score.max(0.0).min(100.0)
    }
    
    /// Export performance data for analysis
    pub fn export_metrics(&self) -> String {
        let snapshot = self.get_snapshot();
        serde_json::to_string_pretty(&snapshot).unwrap_or_else(|_| "Export failed".to_string())
    }
    
    /// Get performance trends over time
    pub fn get_trends(&self) -> PerformanceTrends {
        let samples: Vec<_> = self.sample_buffer.iter().cloned().collect();
        
        if samples.len() < 2 {
            return PerformanceTrends::default();
        }
        
        let recent_samples = &samples[samples.len().saturating_sub(10)..];
        let older_samples = &samples[..samples.len().saturating_sub(10).max(1)];
        
        let recent_avg_latency = recent_samples.iter()
            .map(|s| s.latency_ms)
            .sum::<f64>() / recent_samples.len() as f64;
        
        let older_avg_latency = older_samples.iter()
            .map(|s| s.latency_ms)
            .sum::<f64>() / older_samples.len() as f64;
        
        PerformanceTrends {
            latency_trend: if recent_avg_latency > older_avg_latency * 1.1 {
                Trend::Worsening
            } else if recent_avg_latency < older_avg_latency * 0.9 {
                Trend::Improving
            } else {
                Trend::Stable
            },
            memory_trend: Trend::Stable, // Simplified for now
            throughput_trend: Trend::Stable,
            overall_trend: Trend::Stable,
        }
    }
}

/// Timer for individual operations
pub struct OperationTimer {
    operation_type: OperationType,
    start_time: Instant,
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl OperationTimer {
    fn new(operation_type: OperationType, metrics: Arc<Mutex<PerformanceMetrics>>) -> Self {
        Self {
            operation_type,
            start_time: Instant::now(),
            metrics,
        }
    }
    
    /// Complete the operation and record metrics
    pub fn complete(self) -> Duration {
        let duration = self.start_time.elapsed();
        let duration_ms = duration.as_micros() as f64 / 1000.0;
        
        if let Ok(mut metrics) = self.metrics.lock() {
            match self.operation_type {
                OperationType::NeuralInference => {
                    metrics.total_inferences += 1;
                    metrics.total_inference_time_ms += duration_ms;
                    metrics.avg_inference_ms = metrics.total_inference_time_ms / metrics.total_inferences as f64;
                },
                OperationType::FeatureExtraction => {
                    metrics.feature_extraction_count += 1;
                    metrics.avg_feature_extraction_ms = 
                        (metrics.avg_feature_extraction_ms * (metrics.feature_extraction_count - 1) as f64 + duration_ms) 
                        / metrics.feature_extraction_count as f64;
                },
                OperationType::Routing => {
                    metrics.routing_decisions += 1;
                    metrics.avg_routing_ms = 
                        (metrics.avg_routing_ms * (metrics.routing_decisions - 1) as f64 + duration_ms) 
                        / metrics.routing_decisions as f64;
                },
                OperationType::P2PMessage => {
                    metrics.p2p_messages += 1;
                    metrics.total_p2p_latency_ms += duration_ms;
                    metrics.avg_p2p_latency_ms = metrics.total_p2p_latency_ms / metrics.p2p_messages as f64;
                },
            }
        }
        
        duration
    }
}

/// Performance metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_inferences: u64,
    pub total_inference_time_ms: f64,
    pub avg_inference_ms: f64,
    
    pub feature_extraction_count: u64,
    pub avg_feature_extraction_ms: f64,
    
    pub routing_decisions: u64,
    pub avg_routing_ms: f64,
    
    pub p2p_messages: u64,
    pub total_p2p_latency_ms: f64,
    pub avg_p2p_latency_ms: f64,
    
    pub current_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub memory_samples: Vec<f64>,
    pub memory_fragmentation: f64,
    
    pub current_throughput: f64,
    pub peak_throughput: f64,
    
    pub simd_utilization: f64,
    pub gpu_acceleration_enabled: bool,
    pub cache_hit_rate: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_inferences: 0,
            total_inference_time_ms: 0.0,
            avg_inference_ms: 0.0,
            feature_extraction_count: 0,
            avg_feature_extraction_ms: 0.0,
            routing_decisions: 0,
            avg_routing_ms: 0.0,
            p2p_messages: 0,
            total_p2p_latency_ms: 0.0,
            avg_p2p_latency_ms: 0.0,
            current_memory_mb: 0.0,
            peak_memory_mb: 0.0,
            memory_samples: Vec::new(),
            memory_fragmentation: 0.0,
            current_throughput: 0.0,
            peak_throughput: 0.0,
            simd_utilization: 0.0,
            gpu_acceleration_enabled: false,
            cache_hit_rate: 0.0,
        }
    }
}

/// Types of operations to monitor
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    NeuralInference,
    FeatureExtraction,
    Routing,
    P2PMessage,
}

/// Performance sample for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    pub timestamp_ms: u64,
    pub operation_type: String,
    pub latency_ms: f64,
    pub memory_mb: f64,
    pub cpu_usage: f64,
}

/// Complete performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub uptime_ms: f64,
    pub current_metrics: PerformanceMetrics,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub targets_met: PerformanceTargets,
}

/// Performance bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub component: String,
    pub metric: String,
    pub current_value: f64,
    pub target_value: f64,
    pub severity: BottleneckSeverity,
    pub suggested_fix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Warning,
    Critical,
}

/// Optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub description: String,
    pub expected_improvement: String,
    pub implementation_effort: ImplementationEffort,
    pub priority: Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance target tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub inference_latency_met: bool,
    pub memory_usage_met: bool,
    pub throughput_met: bool,
    pub p2p_latency_met: bool,
    pub overall_score: f64,
}

/// Performance trends analysis
#[derive(Debug, Clone, Default)]
pub struct PerformanceTrends {
    pub latency_trend: Trend,
    pub memory_trend: Trend,
    pub throughput_trend: Trend,
    pub overall_trend: Trend,
}

#[derive(Debug, Clone, Default)]
pub enum Trend {
    #[default]
    Stable,
    Improving,
    Worsening,
}

/// Global performance monitor instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_MONITOR: Arc<Mutex<PerformanceMonitor>> = 
        Arc::new(Mutex::new(PerformanceMonitor::new()));
}

/// Convenience macros for performance monitoring
#[macro_export]
macro_rules! monitor_operation {
    ($op_type:expr, $code:block) => {
        {
            let timer = crate::performance_monitor::GLOBAL_MONITOR
                .lock()
                .unwrap()
                .start_operation($op_type);
            let result = $code;
            timer.complete();
            result
        }
    };
}

#[macro_export]
macro_rules! record_memory {
    ($usage_mb:expr) => {
        crate::performance_monitor::GLOBAL_MONITOR
            .lock()
            .unwrap()
            .record_memory_usage($usage_mb);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        let snapshot = monitor.get_snapshot();
        assert!(snapshot.uptime_ms >= 0.0);
    }
    
    #[test]
    fn test_operation_timer() {
        let monitor = PerformanceMonitor::new();
        let timer = monitor.start_operation(OperationType::NeuralInference);
        
        thread::sleep(Duration::from_millis(10));
        let duration = timer.complete();
        
        assert!(duration.as_millis() >= 10);
    }
    
    #[test]
    fn test_bottleneck_detection() {
        let mut monitor = PerformanceMonitor::new();
        monitor.record_memory_usage(100.0); // Above target
        
        let snapshot = monitor.get_snapshot();
        assert!(!snapshot.bottlenecks.is_empty());
    }
    
    #[test]
    fn test_performance_score() {
        let monitor = PerformanceMonitor::new();
        let snapshot = monitor.get_snapshot();
        
        assert!(snapshot.targets_met.overall_score >= 0.0);
        assert!(snapshot.targets_met.overall_score <= 100.0);
    }
}