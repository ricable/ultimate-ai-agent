//! Enhanced Performance Profiler for Neural Swarm Operations
//! 
//! This module provides advanced profiling capabilities specifically designed for
//! swarm intelligence systems with real-time metrics collection and analysis.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

/// Enhanced profiler for neural swarm operations with agent tracking
pub struct SwarmProfiler {
    timings: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
    counters: Arc<RwLock<HashMap<String, AtomicU64>>>,
    agent_profiles: Arc<RwLock<HashMap<String, AgentProfile>>>,
    operation_traces: Arc<RwLock<Vec<OperationTrace>>>,
    enabled: bool,
    agent_id: Option<String>,
    session_start: Instant,
}

/// Profile information for individual agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_id: String,
    pub agent_type: String,
    pub total_operations: u64,
    pub total_execution_time: Duration,
    pub average_operation_time: Duration,
    pub peak_memory_usage: usize,
    pub cache_statistics: CacheStats,
    pub error_count: u64,
    pub performance_score: f64,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_ratio: f64,
    pub evictions: u64,
    pub size_bytes: u64,
}

/// Detailed operation trace for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTrace {
    pub operation_id: String,
    pub agent_id: String,
    pub operation_type: String,
    pub start_timestamp: u64,
    pub duration_ms: f64,
    pub memory_before: usize,
    pub memory_after: usize,
    pub cpu_usage_percent: f32,
    pub success: bool,
    pub metadata: HashMap<String, String>,
}

impl SwarmProfiler {
    pub fn new(agent_id: Option<String>) -> Self {
        Self {
            timings: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            agent_profiles: Arc::new(RwLock::new(HashMap::new())),
            operation_traces: Arc::new(RwLock::new(Vec::new())),
            enabled: true,
            agent_id,
            session_start: Instant::now(),
        }
    }
    
    pub fn disabled() -> Self {
        Self {
            timings: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            agent_profiles: Arc::new(RwLock::new(HashMap::new())),
            operation_traces: Arc::new(RwLock::new(Vec::new())),
            enabled: false,
            agent_id: None,
            session_start: Instant::now(),
        }
    }
    
    /// Time an operation with enhanced tracking
    pub async fn time<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        
        let operation_id = format!("{}-{}-{}", 
            self.agent_id.as_deref().unwrap_or("unknown"),
            name,
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
        );
        
        let start = Instant::now();
        let memory_before = Self::get_memory_usage();
        let cpu_before = Self::get_cpu_usage();
        
        let result = f();
        
        let duration = start.elapsed();
        let memory_after = Self::get_memory_usage();
        let cpu_after = Self::get_cpu_usage();
        
        // Record timing
        {
            let mut timings = self.timings.write().await;
            timings.entry(name.to_string()).or_insert_with(Vec::new).push(duration);
        }
        
        // Create operation trace
        let trace = OperationTrace {
            operation_id: operation_id.clone(),
            agent_id: self.agent_id.clone().unwrap_or_else(|| "unknown".to_string()),
            operation_type: name.to_string(),
            start_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            memory_before,
            memory_after,
            cpu_usage_percent: cpu_after - cpu_before,
            success: true, // Would be determined by actual operation result
            metadata: HashMap::new(),
        };
        
        // Store trace
        {
            let mut traces = self.operation_traces.write().await;
            if traces.len() >= 10000 {
                traces.remove(0); // Keep only recent traces
            }
            traces.push(trace);
        }
        
        // Update agent profile
        self.update_agent_profile(name, duration, memory_after, true).await;
        
        result
    }
    
    /// Time an async operation
    pub async fn time_async<F, Fut, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        if !self.enabled {
            return f().await;
        }
        
        let start = Instant::now();
        let memory_before = Self::get_memory_usage();
        
        let result = f().await;
        
        let duration = start.elapsed();
        let memory_after = Self::get_memory_usage();
        
        // Record timing
        {
            let mut timings = self.timings.write().await;
            timings.entry(name.to_string()).or_insert_with(Vec::new).push(duration);
        }
        
        // Update agent profile
        self.update_agent_profile(name, duration, memory_after, true).await;
        
        result
    }
    
    /// Increment a counter with optional metadata
    pub async fn increment_counter(&self, name: &str) {
        if !self.enabled {
            return;
        }
        
        let mut counters = self.counters.write().await;
        counters.entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add to a counter with value
    pub async fn add_to_counter(&self, name: &str, value: u64) {
        if !self.enabled {
            return;
        }
        
        let mut counters = self.counters.write().await;
        counters.entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(value, Ordering::Relaxed);
    }
    
    /// Update agent profile information
    async fn update_agent_profile(&self, operation: &str, duration: Duration, memory_usage: usize, success: bool) {
        let agent_id = self.agent_id.clone().unwrap_or_else(|| "unknown".to_string());
        
        let mut profiles = self.agent_profiles.write().await;
        let profile = profiles.entry(agent_id.clone()).or_insert_with(|| AgentProfile {
            agent_id: agent_id.clone(),
            agent_type: "unknown".to_string(),
            total_operations: 0,
            total_execution_time: Duration::ZERO,
            average_operation_time: Duration::ZERO,
            peak_memory_usage: 0,
            cache_statistics: CacheStats {
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
                evictions: 0,
                size_bytes: 0,
            },
            error_count: 0,
            performance_score: 1.0,
        });
        
        profile.total_operations += 1;
        profile.total_execution_time += duration;
        profile.average_operation_time = profile.total_execution_time / profile.total_operations as u32;
        profile.peak_memory_usage = profile.peak_memory_usage.max(memory_usage);
        
        if !success {
            profile.error_count += 1;
        }
        
        // Calculate performance score based on efficiency metrics
        let efficiency = 1.0 / (1.0 + profile.average_operation_time.as_secs_f64());
        let reliability = 1.0 - (profile.error_count as f64 / profile.total_operations as f64);
        profile.performance_score = (efficiency + reliability) / 2.0;
    }
    
    /// Get comprehensive statistics
    pub async fn get_stats(&self) -> SwarmProfileStats {
        let timings = self.timings.read().await;
        let counters = self.counters.read().await;
        let profiles = self.agent_profiles.read().await;
        let traces = self.operation_traces.read().await;
        
        let mut timing_stats = HashMap::new();
        for (name, durations) in timings.iter() {
            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = if count > 0 { total / count as u32 } else { Duration::ZERO };
            let min = durations.iter().min().copied().unwrap_or(Duration::ZERO);
            let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);
            
            // Calculate percentiles
            let mut sorted_durations = durations.clone();
            sorted_durations.sort();
            let p50 = if !sorted_durations.is_empty() {
                sorted_durations[sorted_durations.len() / 2]
            } else {
                Duration::ZERO
            };
            let p95 = if !sorted_durations.is_empty() {
                sorted_durations[(sorted_durations.len() * 95) / 100]
            } else {
                Duration::ZERO
            };
            let p99 = if !sorted_durations.is_empty() {
                sorted_durations[(sorted_durations.len() * 99) / 100]
            } else {
                Duration::ZERO
            };
            
            timing_stats.insert(name.clone(), EnhancedTimingStats {
                total,
                count,
                avg,
                min,
                max,
                p50,
                p95,
                p99,
                throughput_per_second: if avg.as_secs_f64() > 0.0 { 1.0 / avg.as_secs_f64() } else { 0.0 },
            });
        }
        
        let mut counter_stats = HashMap::new();
        for (name, counter) in counters.iter() {
            counter_stats.insert(name.clone(), counter.load(Ordering::Relaxed));
        }
        
        let session_duration = self.session_start.elapsed();
        
        SwarmProfileStats {
            timing_stats,
            counter_stats,
            agent_profiles: profiles.clone(),
            recent_traces: traces.iter().rev().take(100).cloned().collect(), // Last 100 traces
            session_duration,
            total_operations: profiles.values().map(|p| p.total_operations).sum(),
            average_performance_score: if !profiles.is_empty() {
                profiles.values().map(|p| p.performance_score).sum::<f64>() / profiles.len() as f64
            } else {
                0.0
            },
        }
    }
    
    /// Get agent-specific statistics
    pub async fn get_agent_stats(&self, agent_id: &str) -> Option<AgentProfile> {
        let profiles = self.agent_profiles.read().await;
        profiles.get(agent_id).cloned()
    }
    
    /// Get operation traces for a specific agent
    pub async fn get_agent_traces(&self, agent_id: &str, limit: usize) -> Vec<OperationTrace> {
        let traces = self.operation_traces.read().await;
        traces.iter()
            .filter(|trace| trace.agent_id == agent_id)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Reset all profiling data
    pub async fn reset(&self) {
        self.timings.write().await.clear();
        self.counters.write().await.clear();
        self.agent_profiles.write().await.clear();
        self.operation_traces.write().await.clear();
    }
    
    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// Get current memory usage (placeholder implementation)
    fn get_memory_usage() -> usize {
        // In a real implementation, this would query actual system memory
        // For now, return a simulated value
        1024 * 1024 * (50 + rand::random::<usize>() % 200) // 50-250 MB
    }
    
    /// Get current CPU usage (placeholder implementation)
    fn get_cpu_usage() -> f32 {
        // In a real implementation, this would query actual CPU usage
        // For now, return a simulated value
        rand::random::<f32>() * 100.0
    }
    
    /// Generate a comprehensive performance report
    pub async fn generate_report(&self) -> String {
        let stats = self.get_stats().await;
        
        let mut report = String::new();
        report.push_str("🐝 Swarm Performance Profile Report\n");
        report.push_str("=====================================\n\n");
        
        // Session Overview
        report.push_str(&format!("📊 Session Overview:\n"));
        report.push_str(&format!("├── Duration: {:?}\n", stats.session_duration));
        report.push_str(&format!("├── Total Operations: {}\n", stats.total_operations));
        report.push_str(&format!("├── Active Agents: {}\n", stats.agent_profiles.len()));
        report.push_str(&format!("└── Average Performance Score: {:.3}\n\n", stats.average_performance_score));
        
        // Timing Statistics
        if !stats.timing_stats.is_empty() {
            report.push_str("⏱️ Operation Timing Statistics:\n");
            report.push_str(&format!("{:<30} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12}\n", 
                     "Operation", "Count", "Avg (μs)", "P50 (μs)", "P95 (μs)", "P99 (μs)", "Tput/sec"));
            report.push_str(&format!("{:-<102}\n", ""));
            
            let mut sorted_timings: Vec<_> = stats.timing_stats.iter().collect();
            sorted_timings.sort_by(|a, b| b.1.total.cmp(&a.1.total));
            
            for (name, timing) in sorted_timings {
                report.push_str(&format!("{:<30} {:>10} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}\n",
                         name,
                         timing.count,
                         timing.avg.as_secs_f64() * 1_000_000.0,
                         timing.p50.as_secs_f64() * 1_000_000.0,
                         timing.p95.as_secs_f64() * 1_000_000.0,
                         timing.p99.as_secs_f64() * 1_000_000.0,
                         timing.throughput_per_second));
            }
            report.push_str("\n");
        }
        
        // Agent Profiles
        if !stats.agent_profiles.is_empty() {
            report.push_str("🤖 Agent Performance Profiles:\n");
            report.push_str(&format!("{:<20} {:>12} {:>15} {:>15} {:>12} {:>10}\n", 
                     "Agent ID", "Operations", "Avg Time (μs)", "Peak Mem (MB)", "Errors", "Score"));
            report.push_str(&format!("{:-<84}\n", ""));
            
            let mut sorted_agents: Vec<_> = stats.agent_profiles.iter().collect();
            sorted_agents.sort_by(|a, b| b.1.performance_score.partial_cmp(&a.1.performance_score).unwrap());
            
            for (agent_id, profile) in sorted_agents {
                report.push_str(&format!("{:<20} {:>12} {:>15.2} {:>15.2} {:>12} {:>10.3}\n",
                         agent_id,
                         profile.total_operations,
                         profile.average_operation_time.as_secs_f64() * 1_000_000.0,
                         profile.peak_memory_usage as f64 / (1024.0 * 1024.0),
                         profile.error_count,
                         profile.performance_score));
            }
            report.push_str("\n");
        }
        
        // Counter Statistics
        if !stats.counter_stats.is_empty() {
            report.push_str("📈 Counter Statistics:\n");
            report.push_str(&format!("{:<30} {:>15}\n", "Counter", "Value"));
            report.push_str(&format!("{:-<45}\n", ""));
            
            let mut sorted_counters: Vec<_> = stats.counter_stats.iter().collect();
            sorted_counters.sort_by(|a, b| b.1.cmp(a.1));
            
            for (name, value) in sorted_counters {
                report.push_str(&format!("{:<30} {:>15}\n", name, value));
            }
            report.push_str("\n");
        }
        
        // Recent Operation Traces
        if !stats.recent_traces.is_empty() {
            report.push_str("🔍 Recent Operation Traces (Last 10):\n");
            for trace in stats.recent_traces.iter().take(10) {
                report.push_str(&format!("├── {} [{}]: {:.2}ms ({})\n",
                    trace.operation_type,
                    trace.agent_id,
                    trace.duration_ms,
                    if trace.success { "✓" } else { "✗" }
                ));
            }
            report.push_str("\n");
        }
        
        report.push_str("=== End Report ===\n");
        report
    }
}

/// Enhanced timing statistics with percentiles
#[derive(Debug, Clone)]
pub struct EnhancedTimingStats {
    pub total: Duration,
    pub count: usize,
    pub avg: Duration,
    pub min: Duration,
    pub max: Duration,
    pub p50: Duration,  // Median
    pub p95: Duration,
    pub p99: Duration,
    pub throughput_per_second: f64,
}

/// Comprehensive profile statistics for the swarm
#[derive(Debug)]
pub struct SwarmProfileStats {
    pub timing_stats: HashMap<String, EnhancedTimingStats>,
    pub counter_stats: HashMap<String, u64>,
    pub agent_profiles: HashMap<String, AgentProfile>,
    pub recent_traces: Vec<OperationTrace>,
    pub session_duration: Duration,
    pub total_operations: u64,
    pub average_performance_score: f64,
}

/// Thread-local profiler optimized for swarm operations
thread_local! {
    static THREAD_SWARM_PROFILER: tokio::sync::OnceCell<Arc<SwarmProfiler>> = tokio::sync::OnceCell::new();
}

/// Profile a function call with thread-local swarm profiler
pub async fn profile_swarm<F, R>(agent_id: &str, operation_name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let profiler = THREAD_SWARM_PROFILER.with(|cell| {
        cell.get_or_init(|| async {
            Arc::new(SwarmProfiler::new(Some(agent_id.to_string())))
        })
    }).await;
    
    profiler.time(operation_name, f).await
}

/// Profile an async function call
pub async fn profile_swarm_async<F, Fut, R>(agent_id: &str, operation_name: &str, f: F) -> R
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let profiler = THREAD_SWARM_PROFILER.with(|cell| {
        cell.get_or_init(|| async {
            Arc::new(SwarmProfiler::new(Some(agent_id.to_string())))
        })
    }).await;
    
    profiler.time_async(operation_name, f).await
}

/// Increment a counter with thread-local profiler
pub async fn increment_swarm_counter(agent_id: &str, counter_name: &str) {
    let profiler = THREAD_SWARM_PROFILER.with(|cell| {
        cell.get_or_init(|| async {
            Arc::new(SwarmProfiler::new(Some(agent_id.to_string())))
        })
    }).await;
    
    profiler.increment_counter(counter_name).await;
}

/// Memory usage tracker with swarm-specific features
pub struct SwarmMemoryTracker {
    allocations: Arc<RwLock<HashMap<String, (u64, u64)>>>, // (size, count)
    agent_allocations: Arc<RwLock<HashMap<String, HashMap<String, (u64, u64)>>>>, // agent_id -> allocations
    peak_usage: Arc<RwLock<u64>>,
    allocation_timeline: Arc<RwLock<Vec<AllocationEvent>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub timestamp: u64,
    pub agent_id: String,
    pub category: String,
    pub size: u64,
    pub operation: String, // "allocate" or "deallocate"
}

impl SwarmMemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            agent_allocations: Arc::new(RwLock::new(HashMap::new())),
            peak_usage: Arc::new(RwLock::new(0)),
            allocation_timeline: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn track_allocation(&self, agent_id: &str, category: &str, size: u64) {
        // Update global allocations
        {
            let mut allocations = self.allocations.write().await;
            let entry = allocations.entry(category.to_string()).or_insert((0, 0));
            entry.0 += size;
            entry.1 += 1;
        }
        
        // Update agent-specific allocations
        {
            let mut agent_allocs = self.agent_allocations.write().await;
            let agent_entry = agent_allocs.entry(agent_id.to_string()).or_insert_with(HashMap::new);
            let entry = agent_entry.entry(category.to_string()).or_insert((0, 0));
            entry.0 += size;
            entry.1 += 1;
        }
        
        // Update peak usage
        {
            let total_usage: u64 = {
                let allocations = self.allocations.read().await;
                allocations.values().map(|(size, _)| size).sum()
            };
            let mut peak = self.peak_usage.write().await;
            if total_usage > *peak {
                *peak = total_usage;
            }
        }
        
        // Record allocation event
        {
            let mut timeline = self.allocation_timeline.write().await;
            if timeline.len() >= 10000 {
                timeline.remove(0); // Keep only recent events
            }
            timeline.push(AllocationEvent {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                agent_id: agent_id.to_string(),
                category: category.to_string(),
                size,
                operation: "allocate".to_string(),
            });
        }
    }
    
    pub async fn track_deallocation(&self, agent_id: &str, category: &str, size: u64) {
        // Update global allocations
        {
            let mut allocations = self.allocations.write().await;
            let entry = allocations.entry(category.to_string()).or_insert((0, 0));
            entry.0 = entry.0.saturating_sub(size);
            entry.1 = entry.1.saturating_sub(1);
        }
        
        // Update agent-specific allocations
        {
            let mut agent_allocs = self.agent_allocations.write().await;
            if let Some(agent_entry) = agent_allocs.get_mut(agent_id) {
                if let Some(entry) = agent_entry.get_mut(category) {
                    entry.0 = entry.0.saturating_sub(size);
                    entry.1 = entry.1.saturating_sub(1);
                }
            }
        }
        
        // Record deallocation event
        {
            let mut timeline = self.allocation_timeline.write().await;
            if timeline.len() >= 10000 {
                timeline.remove(0);
            }
            timeline.push(AllocationEvent {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                agent_id: agent_id.to_string(),
                category: category.to_string(),
                size,
                operation: "deallocate".to_string(),
            });
        }
    }
    
    pub async fn get_memory_usage(&self) -> HashMap<String, (u64, u64)> {
        self.allocations.read().await.clone()
    }
    
    pub async fn get_agent_memory_usage(&self, agent_id: &str) -> HashMap<String, (u64, u64)> {
        let agent_allocs = self.agent_allocations.read().await;
        agent_allocs.get(agent_id).cloned().unwrap_or_default()
    }
    
    pub async fn get_peak_usage(&self) -> u64 {
        *self.peak_usage.read().await
    }
    
    pub async fn get_allocation_timeline(&self, limit: usize) -> Vec<AllocationEvent> {
        let timeline = self.allocation_timeline.read().await;
        timeline.iter().rev().take(limit).cloned().collect()
    }
    
    pub async fn print_swarm_memory_report(&self) {
        let allocations = self.allocations.read().await;
        let peak = self.get_peak_usage().await;
        
        println!("\n🧠 Swarm Memory Usage Report");
        println!("============================");
        println!("Peak Usage: {} MB", peak as f64 / (1024.0 * 1024.0));
        println!();
        println!("{:<30} {:>15} {:>10}", "Category", "Size (MB)", "Count");
        println!("{:-<55}", "");
        
        let mut sorted_allocs: Vec<_> = allocations.iter().collect();
        sorted_allocs.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        
        for (name, (size, count)) in sorted_allocs {
            println!("{:<30} {:>15.2} {:>10}", 
                name, 
                *size as f64 / (1024.0 * 1024.0), 
                count
            );
        }
        
        let total_size: u64 = allocations.values().map(|(size, _)| size).sum();
        let total_count: u64 = allocations.values().map(|(_, count)| count).sum();
        
        println!("{:-<55}", "");
        println!("{:<30} {:>15.2} {:>10}", 
            "TOTAL", 
            total_size as f64 / (1024.0 * 1024.0), 
            total_count
        );
        println!("=== End Memory Report ===\n");
    }
}

/// Global swarm memory tracker
static SWARM_MEMORY_TRACKER: tokio::sync::OnceCell<SwarmMemoryTracker> = tokio::sync::OnceCell::new();

/// Track memory allocation globally for swarm
pub async fn track_swarm_memory_allocation(agent_id: &str, category: &str, size: u64) {
    let tracker = SWARM_MEMORY_TRACKER.get_or_init(|| async {
        SwarmMemoryTracker::new()
    }).await;
    tracker.track_allocation(agent_id, category, size).await;
}

/// Track memory deallocation globally for swarm
pub async fn track_swarm_memory_deallocation(agent_id: &str, category: &str, size: u64) {
    let tracker = SWARM_MEMORY_TRACKER.get_or_init(|| async {
        SwarmMemoryTracker::new()
    }).await;
    tracker.track_deallocation(agent_id, category, size).await;
}

/// Get global swarm memory usage
pub async fn get_global_swarm_memory_usage() -> HashMap<String, (u64, u64)> {
    let tracker = SWARM_MEMORY_TRACKER.get_or_init(|| async {
        SwarmMemoryTracker::new()
    }).await;
    tracker.get_memory_usage().await
}

/// Print global swarm memory report
pub async fn print_global_swarm_memory_report() {
    let tracker = SWARM_MEMORY_TRACKER.get_or_init(|| async {
        SwarmMemoryTracker::new()
    }).await;
    tracker.print_swarm_memory_report().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_swarm_profiler_timing() {
        let profiler = SwarmProfiler::new(Some("test-agent".to_string()));
        
        profiler.time("test_operation", || {
            std::thread::sleep(Duration::from_millis(10));
        }).await;
        
        let stats = profiler.get_stats().await;
        assert!(stats.timing_stats.contains_key("test_operation"));
        assert_eq!(stats.timing_stats["test_operation"].count, 1);
        assert!(stats.timing_stats["test_operation"].total >= Duration::from_millis(10));
    }
    
    #[tokio::test]
    async fn test_swarm_profiler_counters() {
        let profiler = SwarmProfiler::new(Some("test-agent".to_string()));
        
        profiler.increment_counter("test_counter").await;
        profiler.increment_counter("test_counter").await;
        profiler.add_to_counter("test_counter", 5).await;
        
        let stats = profiler.get_stats().await;
        assert_eq!(stats.counter_stats["test_counter"], 7);
    }
    
    #[tokio::test]
    async fn test_swarm_memory_tracker() {
        let tracker = SwarmMemoryTracker::new();
        
        tracker.track_allocation("agent-1", "tensors", 1024).await;
        tracker.track_allocation("agent-1", "tensors", 2048).await;
        tracker.track_deallocation("agent-1", "tensors", 1024).await;
        
        let usage = tracker.get_memory_usage().await;
        assert_eq!(usage["tensors"], (2048, 1));
        
        let agent_usage = tracker.get_agent_memory_usage("agent-1").await;
        assert_eq!(agent_usage["tensors"], (2048, 1));
    }
    
    #[tokio::test]
    async fn test_agent_profile_updates() {
        let profiler = SwarmProfiler::new(Some("test-agent".to_string()));
        
        // Perform several operations
        for i in 0..5 {
            profiler.time(&format!("operation_{}", i), || {
                std::thread::sleep(Duration::from_millis(i * 2));
            }).await;
        }
        
        let agent_stats = profiler.get_agent_stats("test-agent").await;
        assert!(agent_stats.is_some());
        
        let stats = agent_stats.unwrap();
        assert_eq!(stats.total_operations, 5);
        assert!(stats.performance_score > 0.0);
    }
}