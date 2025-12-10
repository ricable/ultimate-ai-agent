use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

/// Performance profiler for neural network operations
pub struct Profiler {
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    counters: Arc<Mutex<HashMap<String, AtomicU64>>>,
    enabled: bool,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
            enabled: true,
        }
    }
    
    pub fn disabled() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
            enabled: false,
        }
    }
    
    pub fn time<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        let mut timings = self.timings.lock().unwrap();
        timings.entry(name.to_string()).or_insert_with(Vec::new).push(duration);
        
        result
    }
    
    pub fn increment_counter(&self, name: &str) {
        if !self.enabled {
            return;
        }
        
        let mut counters = self.counters.lock().unwrap();
        counters.entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn add_to_counter(&self, name: &str, value: u64) {
        if !self.enabled {
            return;
        }
        
        let mut counters = self.counters.lock().unwrap();
        counters.entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(value, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> ProfileStats {
        let timings = self.timings.lock().unwrap();
        let counters = self.counters.lock().unwrap();
        
        let mut timing_stats = HashMap::new();
        for (name, durations) in timings.iter() {
            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = if count > 0 { total / count as u32 } else { Duration::ZERO };
            let min = durations.iter().min().copied().unwrap_or(Duration::ZERO);
            let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);
            
            timing_stats.insert(name.clone(), TimingStats {
                total,
                count,
                avg,
                min,
                max,
            });
        }
        
        let mut counter_stats = HashMap::new();
        for (name, counter) in counters.iter() {
            counter_stats.insert(name.clone(), counter.load(Ordering::Relaxed));
        }
        
        ProfileStats {
            timing_stats,
            counter_stats,
        }
    }
    
    pub fn reset(&self) {
        self.timings.lock().unwrap().clear();
        self.counters.lock().unwrap().clear();
    }
    
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

#[derive(Debug, Clone)]
pub struct TimingStats {
    pub total: Duration,
    pub count: usize,
    pub avg: Duration,
    pub min: Duration,
    pub max: Duration,
}

#[derive(Debug)]
pub struct ProfileStats {
    pub timing_stats: HashMap<String, TimingStats>,
    pub counter_stats: HashMap<String, u64>,
}

impl ProfileStats {
    pub fn print_report(&self) {
        println!("\n=== Performance Profile Report ===");
        
        if !self.timing_stats.is_empty() {
            println!("\nTiming Statistics:");
            println!("{:<30} {:>10} {:>12} {:>12} {:>12} {:>12}", 
                     "Operation", "Count", "Total (ms)", "Avg (μs)", "Min (μs)", "Max (μs)");
            println!("{:-<90}", "");
            
            let mut sorted_timings: Vec<_> = self.timing_stats.iter().collect();
            sorted_timings.sort_by(|a, b| b.1.total.cmp(&a.1.total));
            
            for (name, stats) in sorted_timings {
                println!("{:<30} {:>10} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                         name,
                         stats.count,
                         stats.total.as_secs_f64() * 1000.0,
                         stats.avg.as_secs_f64() * 1_000_000.0,
                         stats.min.as_secs_f64() * 1_000_000.0,
                         stats.max.as_secs_f64() * 1_000_000.0);
            }
        }
        
        if !self.counter_stats.is_empty() {
            println!("\nCounter Statistics:");
            println!("{:<30} {:>15}", "Counter", "Value");
            println!("{:-<45}", "");
            
            let mut sorted_counters: Vec<_> = self.counter_stats.iter().collect();
            sorted_counters.sort_by(|a, b| b.1.cmp(a.1));
            
            for (name, value) in sorted_counters {
                println!("{:<30} {:>15}", name, value);
            }
        }
        
        println!("\n=== End Report ===\n");
    }
}

/// Thread-local profiler for zero-overhead profiling
thread_local! {
    static THREAD_PROFILER: Profiler = Profiler::new();
}

/// Profile a function call with thread-local profiler
pub fn profile<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    THREAD_PROFILER.with(|profiler| profiler.time(name, f))
}

/// Increment a counter with thread-local profiler
pub fn increment_counter(name: &str) {
    THREAD_PROFILER.with(|profiler| profiler.increment_counter(name));
}

/// Add to a counter with thread-local profiler
pub fn add_to_counter(name: &str, value: u64) {
    THREAD_PROFILER.with(|profiler| profiler.add_to_counter(name, value));
}

/// Get stats from thread-local profiler
pub fn get_thread_stats() -> ProfileStats {
    THREAD_PROFILER.with(|profiler| profiler.get_stats())
}

/// Reset thread-local profiler
pub fn reset_thread_profiler() {
    THREAD_PROFILER.with(|profiler| profiler.reset());
}

/// Memory usage tracker
pub struct MemoryTracker {
    allocations: Arc<Mutex<HashMap<String, (u64, u64)>>>, // (size, count)
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn track_allocation(&self, name: &str, size: u64) {
        let mut allocations = self.allocations.lock().unwrap();
        let entry = allocations.entry(name.to_string()).or_insert((0, 0));
        entry.0 += size;
        entry.1 += 1;
    }
    
    pub fn track_deallocation(&self, name: &str, size: u64) {
        let mut allocations = self.allocations.lock().unwrap();
        let entry = allocations.entry(name.to_string()).or_insert((0, 0));
        entry.0 = entry.0.saturating_sub(size);
        entry.1 = entry.1.saturating_sub(1);
    }
    
    pub fn get_memory_usage(&self) -> HashMap<String, (u64, u64)> {
        self.allocations.lock().unwrap().clone()
    }
    
    pub fn print_memory_report(&self) {
        let allocations = self.allocations.lock().unwrap();
        
        println!("\n=== Memory Usage Report ===");
        println!("{:<30} {:>15} {:>10}", "Category", "Size (bytes)", "Count");
        println!("{:-<55}", "");
        
        let mut sorted_allocs: Vec<_> = allocations.iter().collect();
        sorted_allocs.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        
        for (name, (size, count)) in sorted_allocs {
            println!("{:<30} {:>15} {:>10}", name, size, count);
        }
        
        let total_size: u64 = allocations.values().map(|(size, _)| size).sum();
        let total_count: u64 = allocations.values().map(|(_, count)| count).sum();
        
        println!("{:-<55}", "");
        println!("{:<30} {:>15} {:>10}", "TOTAL", total_size, total_count);
        println!("=== End Memory Report ===\n");
    }
}

/// Global memory tracker
static MEMORY_TRACKER: std::sync::OnceLock<MemoryTracker> = std::sync::OnceLock::new();

/// Track memory allocation globally
pub fn track_memory_allocation(name: &str, size: u64) {
    let tracker = MEMORY_TRACKER.get_or_init(|| MemoryTracker::new());
    tracker.track_allocation(name, size);
}

/// Track memory deallocation globally
pub fn track_memory_deallocation(name: &str, size: u64) {
    let tracker = MEMORY_TRACKER.get_or_init(|| MemoryTracker::new());
    tracker.track_deallocation(name, size);
}

/// Get global memory usage
pub fn get_global_memory_usage() -> HashMap<String, (u64, u64)> {
    let tracker = MEMORY_TRACKER.get_or_init(|| MemoryTracker::new());
    tracker.get_memory_usage()
}

/// Print global memory report
pub fn print_global_memory_report() {
    let tracker = MEMORY_TRACKER.get_or_init(|| MemoryTracker::new());
    tracker.print_memory_report();
}

/// Performance monitor for continuous profiling
pub struct PerformanceMonitor {
    profiler: Profiler,
    memory_tracker: MemoryTracker,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            profiler: Profiler::new(),
            memory_tracker: MemoryTracker::new(),
            start_time: Instant::now(),
        }
    }
    
    pub fn time<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.profiler.time(name, f)
    }
    
    pub fn track_allocation(&self, name: &str, size: u64) {
        self.memory_tracker.track_allocation(name, size);
    }
    
    pub fn track_deallocation(&self, name: &str, size: u64) {
        self.memory_tracker.track_deallocation(name, size);
    }
    
    pub fn print_full_report(&self) {
        let uptime = self.start_time.elapsed();
        println!("\n=== Full Performance Report ===");
        println!("Uptime: {:?}", uptime);
        
        self.profiler.get_stats().print_report();
        self.memory_tracker.print_memory_report();
    }
    
    pub fn reset(&self) {
        self.profiler.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_profiler_timing() {
        let profiler = Profiler::new();
        
        profiler.time("test_operation", || {
            thread::sleep(Duration::from_millis(10));
        });
        
        let stats = profiler.get_stats();
        assert!(stats.timing_stats.contains_key("test_operation"));
        assert_eq!(stats.timing_stats["test_operation"].count, 1);
        assert!(stats.timing_stats["test_operation"].total >= Duration::from_millis(10));
    }
    
    #[test]
    fn test_profiler_counters() {
        let profiler = Profiler::new();
        
        profiler.increment_counter("test_counter");
        profiler.increment_counter("test_counter");
        profiler.add_to_counter("test_counter", 5);
        
        let stats = profiler.get_stats();
        assert_eq!(stats.counter_stats["test_counter"], 7);
    }
    
    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();
        
        tracker.track_allocation("tensors", 1024);
        tracker.track_allocation("tensors", 2048);
        tracker.track_deallocation("tensors", 1024);
        
        let usage = tracker.get_memory_usage();
        assert_eq!(usage["tensors"], (2048, 1));
    }
}