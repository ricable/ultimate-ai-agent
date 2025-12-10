//! High-performance monitoring and profiling for WASM optimization
//!
//! This module provides comprehensive performance monitoring with minimal
//! overhead, optimized for WASM environments.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance counter types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CounterType {
    /// Kernel execution time
    KernelExecution,
    /// Memory allocation time
    MemoryAllocation,
    /// Memory transfer time
    MemoryTransfer,
    /// Compilation time
    Compilation,
    /// Total pipeline time
    TotalPipeline,
    /// WebGPU command encoding
    WebGPUEncoding,
    /// Custom counter
    Custom(String),
}

/// Performance measurement
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Duration of the operation
    pub duration: Duration,
    /// Timestamp when measurement was taken
    pub timestamp: Instant,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Operation size/complexity (e.g., data size, thread count)
    pub size: Option<usize>,
}

/// Performance statistics for a counter
#[derive(Debug, Clone)]
pub struct CounterStats {
    /// Total number of measurements
    pub count: u64,
    /// Total time spent
    pub total_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Average time
    pub avg_time: Duration,
    /// 95th percentile time
    pub p95_time: Duration,
    /// 99th percentile time
    pub p99_time: Duration,
    /// Total throughput (operations per second)
    pub throughput: f64,
    /// Total data processed (bytes)
    pub total_bytes: u64,
    /// Data throughput (bytes per second)
    pub data_throughput: f64,
}

/// High-performance monitor with minimal overhead
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Counters organized by type
    counters: Arc<Mutex<HashMap<CounterType, Vec<Measurement>>>>,
    /// Global start time
    start_time: Instant,
    /// Configuration
    config: MonitorConfig,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Maximum measurements to keep per counter
    pub max_measurements: usize,
    /// Enable detailed timing (may have overhead)
    pub detailed_timing: bool,
    /// Enable throughput calculation
    pub calculate_throughput: bool,
    /// Sampling rate (1.0 = all measurements, 0.1 = 10% sampling)
    pub sampling_rate: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_measurements: 1000,
            detailed_timing: cfg!(debug_assertions),
            calculate_throughput: true,
            sampling_rate: 1.0,
        }
    }
}

/// RAII timer for automatic measurement
pub struct Timer<'a> {
    monitor: &'a PerformanceMonitor,
    counter_type: CounterType,
    start_time: Instant,
    metadata: HashMap<String, String>,
    size: Option<usize>,
}

impl<'a> Timer<'a> {
    /// Create a new timer
    fn new(monitor: &'a PerformanceMonitor, counter_type: CounterType) -> Self {
        Self {
            monitor,
            counter_type,
            start_time: Instant::now(),
            metadata: HashMap::new(),
            size: None,
        }
    }

    /// Add metadata to the measurement
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the operation size for throughput calculation
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let measurement = Measurement {
            duration,
            timestamp: self.start_time,
            metadata: std::mem::take(&mut self.metadata),
            size: self.size,
        };
        
        self.monitor.record_measurement(self.counter_type.clone(), measurement);
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create a new performance monitor with custom configuration
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            config,
        }
    }

    /// Start timing an operation
    pub fn time(&self, counter_type: CounterType) -> Timer<'_> {
        Timer::new(self, counter_type)
    }

    /// Record a measurement manually
    pub fn record(&self, counter_type: CounterType, duration: Duration) {
        self.record_with_size(counter_type, duration, None);
    }

    /// Record a measurement with size information
    pub fn record_with_size(&self, counter_type: CounterType, duration: Duration, size: Option<usize>) {
        // Apply sampling
        if self.config.sampling_rate < 1.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            duration.as_nanos().hash(&mut hasher);
            let sample = (hasher.finish() % 1000) as f64 / 1000.0;
            
            if sample > self.config.sampling_rate {
                return;
            }
        }

        let measurement = Measurement {
            duration,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
            size,
        };

        self.record_measurement(counter_type, measurement);
    }

    /// Record a measurement with metadata
    fn record_measurement(&self, counter_type: CounterType, measurement: Measurement) {
        let mut counters = self.counters.lock().unwrap();
        let measurements = counters.entry(counter_type).or_default();
        
        measurements.push(measurement);
        
        // Limit memory usage by keeping only recent measurements
        if measurements.len() > self.config.max_measurements {
            measurements.drain(0..measurements.len() - self.config.max_measurements);
        }
    }

    /// Get statistics for a counter type
    pub fn stats(&self, counter_type: &CounterType) -> Option<CounterStats> {
        let counters = self.counters.lock().unwrap();
        let measurements = counters.get(counter_type)?;
        
        if measurements.is_empty() {
            return None;
        }

        let mut durations: Vec<Duration> = measurements.iter().map(|m| m.duration).collect();
        durations.sort();

        let count = measurements.len() as u64;
        let total_time: Duration = durations.iter().sum();
        let min_time = durations[0];
        let max_time = durations[durations.len() - 1];
        let avg_time = total_time / count as u32;
        
        let p95_index = (durations.len() as f64 * 0.95) as usize;
        let p99_index = (durations.len() as f64 * 0.99) as usize;
        let p95_time = durations.get(p95_index.saturating_sub(1)).copied().unwrap_or(max_time);
        let p99_time = durations.get(p99_index.saturating_sub(1)).copied().unwrap_or(max_time);

        let throughput = if total_time.as_secs_f64() > 0.0 {
            count as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let total_bytes: u64 = measurements.iter()
            .filter_map(|m| m.size)
            .map(|s| s as u64)
            .sum();

        let data_throughput = if total_time.as_secs_f64() > 0.0 {
            total_bytes as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        Some(CounterStats {
            count,
            total_time,
            min_time,
            max_time,
            avg_time,
            p95_time,
            p99_time,
            throughput,
            total_bytes,
            data_throughput,
        })
    }

    /// Get all counter statistics
    pub fn all_stats(&self) -> HashMap<CounterType, CounterStats> {
        let counters = self.counters.lock().unwrap();
        let mut stats = HashMap::new();
        
        for counter_type in counters.keys() {
            if let Some(counter_stats) = self.stats(counter_type) {
                stats.insert(counter_type.clone(), counter_stats);
            }
        }
        
        stats
    }

    /// Clear all measurements
    pub fn clear(&self) {
        self.counters.lock().unwrap().clear();
    }

    /// Get total runtime since monitor creation
    pub fn total_runtime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Generate a performance report
    pub fn report(&self) -> PerformanceReport {
        let all_stats = self.all_stats();
        let total_runtime = self.total_runtime();
        
        PerformanceReport {
            stats: all_stats,
            total_runtime,
            monitor_config: self.config.clone(),
        }
    }

    /// Get memory usage of the monitor itself
    pub fn memory_usage(&self) -> usize {
        let counters = self.counters.lock().unwrap();
        counters.values()
            .map(|measurements| measurements.len() * std::mem::size_of::<Measurement>())
            .sum::<usize>()
            + counters.len() * std::mem::size_of::<Vec<Measurement>>()
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance report with comprehensive metrics
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Statistics for each counter type
    pub stats: HashMap<CounterType, CounterStats>,
    /// Total runtime of the monitor
    pub total_runtime: Duration,
    /// Monitor configuration used
    pub monitor_config: MonitorConfig,
}

impl PerformanceReport {
    /// Generate a human-readable report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Performance Report ===\n");
        report.push_str(&format!("Total Runtime: {:.2}s\n", self.total_runtime.as_secs_f64()));
        report.push_str(&format!("Monitor Config: {:?}\n\n", self.monitor_config));
        
        for (counter_type, stats) in &self.stats {
            report.push_str(&format!("{counter_type:?}:\n"));
            report.push_str(&format!("  Count: {}\n", stats.count));
            report.push_str(&format!("  Total Time: {:.2}ms\n", stats.total_time.as_millis()));
            report.push_str(&format!("  Avg Time: {:.2}ms\n", stats.avg_time.as_millis()));
            report.push_str(&format!("  Min Time: {:.2}ms\n", stats.min_time.as_millis()));
            report.push_str(&format!("  Max Time: {:.2}ms\n", stats.max_time.as_millis()));
            report.push_str(&format!("  P95 Time: {:.2}ms\n", stats.p95_time.as_millis()));
            report.push_str(&format!("  P99 Time: {:.2}ms\n", stats.p99_time.as_millis()));
            report.push_str(&format!("  Throughput: {:.2} ops/s\n", stats.throughput));
            
            if stats.total_bytes > 0 {
                report.push_str(&format!("  Data Processed: {:.2} MB\n", stats.total_bytes as f64 / 1_000_000.0));
                report.push_str(&format!("  Data Throughput: {:.2} MB/s\n", stats.data_throughput / 1_000_000.0));
            }
            
            report.push('\n');
        }
        
        report
    }

    /// Export to JSON format
    pub fn to_json(&self) -> Result<String, String> {
        // For now, just return a simple string representation
        Ok(self.to_string())
    }
}

/// Global performance monitor instance
static GLOBAL_MONITOR: std::sync::OnceLock<PerformanceMonitor> = std::sync::OnceLock::new();

/// Get the global performance monitor
pub fn global_monitor() -> &'static PerformanceMonitor {
    GLOBAL_MONITOR.get_or_init(PerformanceMonitor::new)
}

/// Time an operation using the global monitor
pub fn time_operation(counter_type: CounterType) -> Timer<'static> {
    global_monitor().time(counter_type)
}

/// Record a measurement using the global monitor
pub fn record_measurement(counter_type: CounterType, duration: Duration) {
    global_monitor().record(counter_type, duration);
}

/// Get global performance report
pub fn global_report() -> PerformanceReport {
    global_monitor().report()
}

/// Macro for easy timing of code blocks
#[macro_export]
macro_rules! time_block {
    ($counter_type:expr, $block:block) => {{
        let _timer = $crate::profiling::performance_monitor::time_operation($counter_type);
        $block
    }};
    
    ($counter_type:expr, $size:expr, $block:block) => {{
        let _timer = $crate::profiling::performance_monitor::time_operation($counter_type).with_size($size);
        $block
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        // Test basic timing
        {
            let _timer = monitor.time(CounterType::KernelExecution);
            thread::sleep(Duration::from_millis(10));
        }
        
        let stats = monitor.stats(&CounterType::KernelExecution).unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.avg_time >= Duration::from_millis(9));
    }

    #[test]
    fn test_timer_with_metadata() {
        let monitor = PerformanceMonitor::new();
        
        {
            let _timer = monitor.time(CounterType::MemoryAllocation)
                .with_metadata("size", "1024")
                .with_size(1024);
            thread::sleep(Duration::from_millis(5));
        }
        
        let stats = monitor.stats(&CounterType::MemoryAllocation).unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_bytes, 1024);
    }

    #[test]
    fn test_global_monitor() {
        {
            let _timer = time_operation(CounterType::Compilation);
            thread::sleep(Duration::from_millis(1));
        }
        
        let report = global_report();
        assert!(report.stats.contains_key(&CounterType::Compilation));
    }

    #[test]
    fn test_time_block_macro() {
        time_block!(CounterType::Custom("test".to_string()), {
            thread::sleep(Duration::from_millis(1));
        });
        
        let report = global_report();
        assert!(report.stats.contains_key(&CounterType::Custom("test".to_string())));
    }
}