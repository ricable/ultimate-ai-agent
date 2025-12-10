//! Performance Monitoring for Neural Integration
//!
//! This module provides real-time performance monitoring, bottleneck detection,
//! and automatic optimization suggestions for neural operations.

use super::{
    NeuralIntegrationError, NeuralResult, OperationHandle, OperationStats, 
    PerformanceDegradation, PerformanceMonitorTrait, PerformanceStats,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Real-time performance monitor with adaptive optimization
pub struct RealTimeMonitor {
    operations: Arc<RwLock<HashMap<OperationHandle, OngoingOperation>>>,
    history: Arc<Mutex<PerformanceHistory>>,
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    next_handle: Arc<Mutex<u64>>,
    config: MonitorConfig,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub history_size: usize,
    pub baseline_window: usize,
    pub degradation_threshold: f64, // 1.5 = 50% slower than baseline
    pub enable_auto_optimization: bool,
    pub sample_rate: f64, // 0.0 to 1.0
}

/// Ongoing operation tracking
#[derive(Debug)]
struct OngoingOperation {
    name: String,
    start_time: Instant,
    gpu_start: Option<Instant>,
    memory_start: usize,
    expected_duration: Option<Duration>,
}

/// Performance history tracking
struct PerformanceHistory {
    operations: VecDeque<CompletedOperation>,
    aggregated_stats: HashMap<String, AggregatedStats>,
    total_operations: u64,
}

/// Completed operation record
#[derive(Debug, Clone)]
struct CompletedOperation {
    name: String,
    execution_time: Duration,
    gpu_time: Duration,
    memory_transfer_time: Duration,
    throughput: f64,
    timestamp: Instant,
    memory_usage: usize,
    success: bool,
}

/// Aggregated statistics for an operation type
#[derive(Debug, Clone)]
struct AggregatedStats {
    count: u64,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
    avg_time: Duration,
    std_dev: f64,
    throughput_sum: f64,
    memory_usage_sum: usize,
    failure_count: u64,
}

/// Performance baseline for comparison
#[derive(Debug, Clone)]
struct PerformanceBaseline {
    operation_name: String,
    expected_time: Duration,
    expected_throughput: f64,
    confidence: f64,
    sample_count: u64,
    last_updated: Instant,
}

/// No-op monitor for when monitoring is disabled
pub struct NoOpMonitor;

impl Default for NoOpMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl NoOpMonitor {
    pub fn new() -> Self {
        NoOpMonitor
    }
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            history_size: 10000,
            baseline_window: 100,
            degradation_threshold: 1.5,
            enable_auto_optimization: true,
            sample_rate: 1.0,
        }
    }
}

impl RealTimeMonitor {
    /// Create a new real-time performance monitor
    pub fn new() -> NeuralResult<Self> {
        Self::with_config(MonitorConfig::default())
    }
    
    /// Create a monitor with custom configuration
    pub fn with_config(config: MonitorConfig) -> NeuralResult<Self> {
        Ok(Self {
            operations: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(Mutex::new(PerformanceHistory::new(config.history_size))),
            baselines: Arc::new(RwLock::new(HashMap::new())),
            next_handle: Arc::new(Mutex::new(1)),
            config,
        })
    }
    
    /// Update baseline for an operation
    fn update_baseline(&self, operation: &CompletedOperation) {
        if !operation.success {
            return;
        }
        
        let mut baselines = self.baselines.write().unwrap();
        let baseline = baselines.entry(operation.name.clone())
            .or_insert_with(|| PerformanceBaseline {
                operation_name: operation.name.clone(),
                expected_time: operation.execution_time,
                expected_throughput: operation.throughput,
                confidence: 0.5,
                sample_count: 0,
                last_updated: Instant::now(),
            });
        
        // Update baseline using exponential moving average
        let alpha = 0.1; // Learning rate
        let new_time_ms = operation.execution_time.as_secs_f64() * 1000.0;
        let old_time_ms = baseline.expected_time.as_secs_f64() * 1000.0;
        let updated_time_ms = alpha * new_time_ms + (1.0 - alpha) * old_time_ms;
        
        baseline.expected_time = Duration::from_secs_f64(updated_time_ms / 1000.0);
        baseline.expected_throughput = alpha * operation.throughput + (1.0 - alpha) * baseline.expected_throughput;
        baseline.sample_count += 1;
        baseline.last_updated = Instant::now();
        
        // Increase confidence as we get more samples
        baseline.confidence = (baseline.sample_count as f64 / 100.0).min(1.0);
    }
    
    /// Check for performance degradation
    fn check_degradation(&self, operation: &CompletedOperation) -> Option<PerformanceDegradation> {
        let baselines = self.baselines.read().unwrap();
        
        if let Some(baseline) = baselines.get(&operation.name) {
            if baseline.confidence < 0.3 || baseline.sample_count < 10 {
                return None; // Not enough data for reliable comparison
            }
            
            let actual_time = operation.execution_time.as_secs_f64();
            let expected_time = baseline.expected_time.as_secs_f64();
            let degradation_factor = actual_time / expected_time;
            
            if degradation_factor > self.config.degradation_threshold {
                return Some(PerformanceDegradation {
                    operation: operation.name.clone(),
                    expected_time,
                    actual_time,
                    degradation_factor,
                    suggested_action: self.generate_optimization_suggestion(operation, baseline),
                });
            }
        }
        
        None
    }
    
    /// Generate optimization suggestions
    fn generate_optimization_suggestion(
        &self,
        operation: &CompletedOperation,
        baseline: &PerformanceBaseline,
    ) -> String {
        if operation.memory_transfer_time > operation.execution_time / 2 {
            "Consider using memory pooling or batch operations to reduce transfer overhead".to_string()
        } else if operation.gpu_time < operation.execution_time / 3 {
            "GPU utilization is low, consider increasing batch size or workgroup size".to_string()
        } else if operation.throughput < baseline.expected_throughput * 0.7 {
            "Throughput is significantly below baseline, check for memory pressure or resource contention".to_string()
        } else {
            "Performance degradation detected, consider profiling individual kernels".to_string()
        }
    }
    
    /// Get performance trends for an operation
    pub fn get_trends(&self, operation_name: &str, window_size: usize) -> Option<PerformanceTrend> {
        let history = self.history.lock().unwrap();
        let recent_ops: Vec<&CompletedOperation> = history.operations
            .iter()
            .rev()
            .filter(|op| op.name == operation_name)
            .take(window_size)
            .collect();
        
        if recent_ops.len() < 5 {
            return None;
        }
        
        let times: Vec<f64> = recent_ops.iter()
            .map(|op| op.execution_time.as_secs_f64())
            .collect();
        
        let trend_slope = calculate_trend_slope(&times);
        let volatility = calculate_volatility(&times);
        
        Some(PerformanceTrend {
            operation_name: operation_name.to_string(),
            trend_slope,
            volatility,
            sample_count: recent_ops.len(),
            improving: trend_slope < -0.01, // Negative slope means improving (faster)
        })
    }
    
    /// Get bottleneck analysis
    pub fn get_bottleneck_analysis(&self, operation_name: &str) -> Option<BottleneckAnalysis> {
        let history = self.history.lock().unwrap();
        if let Some(stats) = history.aggregated_stats.get(operation_name) {
            let avg_execution = stats.avg_time.as_secs_f64();
            let avg_memory_transfer = stats.total_time.as_secs_f64() / stats.count as f64;
            
            let memory_ratio = avg_memory_transfer / avg_execution;
            let gpu_ratio = 1.0 - memory_ratio; // Simplified calculation
            
            let bottleneck_type = if memory_ratio > 0.5 {
                BottleneckType::MemoryTransfer
            } else if gpu_ratio < 0.3 {
                BottleneckType::GpuUnderutilization
            } else if stats.failure_count as f64 / stats.count as f64 > 0.1 {
                BottleneckType::ErrorRate
            } else {
                BottleneckType::Computation
            };
            
            Some(BottleneckAnalysis {
                operation_name: operation_name.to_string(),
                bottleneck_type,
                memory_transfer_ratio: memory_ratio,
                gpu_utilization_ratio: gpu_ratio,
                error_rate: stats.failure_count as f64 / stats.count as f64,
                recommendation: generate_bottleneck_recommendation(&bottleneck_type),
            })
        } else {
            None
        }
    }
}

impl PerformanceMonitorTrait for RealTimeMonitor {
    fn start_operation(&self, name: &str) -> OperationHandle {
        // Sample operations based on config
        if self.config.sample_rate < 1.0 && rand::random::<f64>() > self.config.sample_rate {
            return OperationHandle(0); // Skip monitoring for this operation
        }
        
        let mut next_handle = self.next_handle.lock().unwrap();
        let handle = OperationHandle(*next_handle);
        *next_handle += 1;
        
        let operation = OngoingOperation {
            name: name.to_string(),
            start_time: Instant::now(),
            gpu_start: None,
            memory_start: 0, // TODO: Get actual memory usage
            expected_duration: self.get_expected_duration(name),
        };
        
        let mut operations = self.operations.write().unwrap();
        operations.insert(handle, operation);
        
        handle
    }
    
    fn end_operation(&self, handle: OperationHandle) -> NeuralResult<OperationStats> {
        if handle.0 == 0 {
            // This operation was not monitored
            return Ok(OperationStats {
                name: "unmonitored".to_string(),
                execution_time: 0.0,
                gpu_time: 0.0,
                memory_transfer_time: 0.0,
                throughput: 0.0,
            });
        }
        
        let mut operations = self.operations.write().unwrap();
        let ongoing = operations.remove(&handle).ok_or_else(|| {
            NeuralIntegrationError::PerformanceError("Invalid operation handle".to_string())
        })?;
        
        let end_time = Instant::now();
        let execution_time = end_time.duration_since(ongoing.start_time);
        
        // TODO: Get actual GPU and memory transfer times
        let gpu_time = execution_time * 7 / 10; // Assume 70% GPU time
        let memory_transfer_time = execution_time * 2 / 10; // Assume 20% transfer time
        
        let throughput = 1.0 / execution_time.as_secs_f64(); // Operations per second
        
        let completed_op = CompletedOperation {
            name: ongoing.name.clone(),
            execution_time,
            gpu_time,
            memory_transfer_time,
            throughput,
            timestamp: end_time,
            memory_usage: 0, // TODO: Get actual memory usage
            success: true, // TODO: Determine success based on context
        };
        
        // Update history and baselines
        {
            let mut history = self.history.lock().unwrap();
            history.add_operation(completed_op.clone());
        }
        
        self.update_baseline(&completed_op);
        
        Ok(OperationStats {
            name: ongoing.name,
            execution_time: execution_time.as_secs_f64(),
            gpu_time: gpu_time.as_secs_f64(),
            memory_transfer_time: memory_transfer_time.as_secs_f64(),
            throughput,
        })
    }
    
    fn get_performance_summary(&self) -> PerformanceStats {
        let history = self.history.lock().unwrap();
        
        if history.total_operations == 0 {
            return PerformanceStats {
                total_operations: 0,
                average_execution_time: 0.0,
                gpu_utilization: 0.0,
                memory_bandwidth: 0.0,
                throughput: 0.0,
            };
        }
        
        let total_time: Duration = history.operations.iter()
            .map(|op| op.execution_time)
            .sum();
        
        let total_gpu_time: Duration = history.operations.iter()
            .map(|op| op.gpu_time)
            .sum();
        
        let total_throughput: f64 = history.operations.iter()
            .map(|op| op.throughput)
            .sum();
        
        PerformanceStats {
            total_operations: history.total_operations,
            average_execution_time: total_time.as_secs_f64() / history.total_operations as f64,
            gpu_utilization: (total_gpu_time.as_secs_f64() / total_time.as_secs_f64()) as f32,
            memory_bandwidth: 0.0, // TODO: Calculate actual memory bandwidth
            throughput: total_throughput / history.total_operations as f64,
        }
    }
    
    fn detect_degradation(&self) -> Option<PerformanceDegradation> {
        let history = self.history.lock().unwrap();
        
        // Check the most recent operation for degradation
        if let Some(recent_op) = history.operations.back() {
            // Clone the operation to avoid borrow after drop
            let recent_op_clone = recent_op.clone();
            drop(history);
            self.check_degradation(&recent_op_clone)
        } else {
            None
        }
    }
}

impl PerformanceHistory {
    fn new(max_size: usize) -> Self {
        Self {
            operations: VecDeque::with_capacity(max_size),
            aggregated_stats: HashMap::new(),
            total_operations: 0,
        }
    }
    
    fn add_operation(&mut self, operation: CompletedOperation) {
        // Add to history
        if self.operations.len() >= self.operations.capacity() {
            self.operations.pop_front();
        }
        self.operations.push_back(operation.clone());
        self.total_operations += 1;
        
        // Update aggregated stats
        let stats = self.aggregated_stats.entry(operation.name.clone())
            .or_insert_with(|| AggregatedStats {
                count: 0,
                total_time: Duration::ZERO,
                min_time: operation.execution_time,
                max_time: operation.execution_time,
                avg_time: Duration::ZERO,
                std_dev: 0.0,
                throughput_sum: 0.0,
                memory_usage_sum: 0,
                failure_count: 0,
            });
        
        stats.count += 1;
        stats.total_time += operation.execution_time;
        stats.min_time = stats.min_time.min(operation.execution_time);
        stats.max_time = stats.max_time.max(operation.execution_time);
        stats.avg_time = stats.total_time / stats.count as u32;
        stats.throughput_sum += operation.throughput;
        stats.memory_usage_sum += operation.memory_usage;
        
        if !operation.success {
            stats.failure_count += 1;
        }
        
        // Update standard deviation (simplified calculation)
        let times: Vec<f64> = self.operations.iter()
            .filter(|op| op.name == operation.name)
            .map(|op| op.execution_time.as_secs_f64())
            .collect();
        
        if times.len() > 1 {
            stats.std_dev = calculate_std_dev(&times);
        }
    }
}

impl RealTimeMonitor {
    fn get_expected_duration(&self, name: &str) -> Option<Duration> {
        let baselines = self.baselines.read().unwrap();
        baselines.get(name).map(|b| b.expected_time)
    }
}

impl PerformanceMonitorTrait for NoOpMonitor {
    fn start_operation(&self, _name: &str) -> OperationHandle {
        OperationHandle(0)
    }
    
    fn end_operation(&self, _handle: OperationHandle) -> NeuralResult<OperationStats> {
        Ok(OperationStats {
            name: "noop".to_string(),
            execution_time: 0.0,
            gpu_time: 0.0,
            memory_transfer_time: 0.0,
            throughput: 0.0,
        })
    }
    
    fn get_performance_summary(&self) -> PerformanceStats {
        PerformanceStats {
            total_operations: 0,
            average_execution_time: 0.0,
            gpu_utilization: 0.0,
            memory_bandwidth: 0.0,
            throughput: 0.0,
        }
    }
    
    fn detect_degradation(&self) -> Option<PerformanceDegradation> {
        None
    }
}

/// Performance trend information
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub operation_name: String,
    pub trend_slope: f64,
    pub volatility: f64,
    pub sample_count: usize,
    pub improving: bool,
}

/// Bottleneck analysis
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    pub operation_name: String,
    pub bottleneck_type: BottleneckType,
    pub memory_transfer_ratio: f64,
    pub gpu_utilization_ratio: f64,
    pub error_rate: f64,
    pub recommendation: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy)]
pub enum BottleneckType {
    MemoryTransfer,
    Computation,
    GpuUnderutilization,
    ErrorRate,
}

/// Calculate trend slope using linear regression
fn calculate_trend_slope(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
    let y_sum: f64 = values.iter().sum();
    let xy_sum: f64 = values.iter().enumerate()
        .map(|(i, &y)| i as f64 * y)
        .sum();
    let x_sq_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
    
    (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum.powi(2))
}

/// Calculate volatility (standard deviation)
fn calculate_volatility(values: &[f64]) -> f64 {
    calculate_std_dev(values)
}

/// Calculate standard deviation
fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance.sqrt()
}

/// Generate recommendation for bottleneck type
fn generate_bottleneck_recommendation(bottleneck_type: &BottleneckType) -> String {
    match bottleneck_type {
        BottleneckType::MemoryTransfer => {
            "Optimize memory transfers by using larger batch sizes, memory pooling, or reducing data precision".to_string()
        }
        BottleneckType::Computation => {
            "Optimize computation by improving algorithm efficiency, using better GPU kernels, or increasing parallelism".to_string()
        }
        BottleneckType::GpuUnderutilization => {
            "Increase GPU utilization by using larger workgroup sizes, higher occupancy, or more parallel work".to_string()
        }
        BottleneckType::ErrorRate => {
            "Reduce error rate by improving input validation, handling edge cases, or fixing stability issues".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_monitor_creation() {
        let monitor = RealTimeMonitor::new().unwrap();
        let stats = monitor.get_performance_summary();
        assert_eq!(stats.total_operations, 0);
    }
    
    #[test]
    fn test_operation_tracking() {
        let monitor = RealTimeMonitor::new().unwrap();
        
        let handle = monitor.start_operation("test_op");
        std::thread::sleep(Duration::from_millis(10));
        let stats = monitor.end_operation(handle).unwrap();
        
        assert_eq!(stats.name, "test_op");
        assert!(stats.execution_time > 0.0);
    }
    
    #[test]
    fn test_trend_calculation() {
        let values = vec![1.0, 1.1, 1.2, 1.15, 1.3];
        let slope = calculate_trend_slope(&values);
        assert!(slope > 0.0); // Generally increasing
    }
    
    #[test]
    fn test_std_dev_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = calculate_std_dev(&values);
        assert!((std_dev - 1.58).abs() < 0.1); // Approximately sqrt(2.5)
    }
    
    #[test]
    fn test_noop_monitor() {
        let monitor = NoOpMonitor;
        let handle = monitor.start_operation("test");
        let stats = monitor.end_operation(handle).unwrap();
        assert_eq!(stats.name, "noop");
    }
}