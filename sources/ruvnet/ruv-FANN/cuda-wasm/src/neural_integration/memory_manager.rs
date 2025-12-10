//! Memory Management for Neural Integration
//!
//! This module provides efficient memory management for neural operations,
//! including GPU-CPU data transfer, memory pooling, and automatic optimization.

use super::{
    BridgeConfig, BufferHandle, MemoryHandle, MemoryManagerTrait, MemoryStats,
    NeuralIntegrationError, NeuralResult,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Hybrid memory manager that efficiently handles both CPU and GPU memory
pub struct HybridMemoryManager {
    config: BridgeConfig,
    cpu_pool: Arc<Mutex<CpuMemoryPool>>,
    gpu_pool: Arc<Mutex<GpuMemoryPool>>,
    transfer_cache: Arc<RwLock<TransferCache>>,
    stats: Arc<Mutex<MemoryStatsTracker>>,
    pressure_monitor: Arc<Mutex<MemoryPressureMonitor>>,
}

/// CPU memory pool for efficient allocation
struct CpuMemoryPool {
    pools: HashMap<usize, VecDeque<Vec<f32>>>,
    allocated_bytes: usize,
    allocations: u64,
    deallocations: u64,
}

/// GPU memory pool for WebGPU buffers
struct GpuMemoryPool {
    device: Option<Arc<wgpu::Device>>,
    buffers: HashMap<BufferHandle, GpuBuffer>,
    free_buffers: HashMap<usize, VecDeque<BufferHandle>>,
    allocated_bytes: usize,
    allocations: u64,
    deallocations: u64,
    next_handle: u64,
}

/// GPU buffer wrapper
struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,
    last_used: Instant,
    usage_count: u32,
}

/// Transfer cache for frequently used data
struct TransferCache {
    cache: HashMap<u64, CachedTransfer>,
    max_entries: usize,
    total_size: usize,
    max_size: usize,
}

/// Cached transfer data
struct CachedTransfer {
    data: Vec<f32>,
    gpu_buffer: Option<BufferHandle>,
    last_accessed: Instant,
    access_count: u32,
}

/// Memory statistics tracker
struct MemoryStatsTracker {
    cpu_allocated: usize,
    gpu_allocated: usize,
    peak_cpu: usize,
    peak_gpu: usize,
    total_allocations: u64,
    total_deallocations: u64,
    cache_hits: u64,
    cache_misses: u64,
    transfer_bytes: u64,
}

/// Memory pressure monitoring
struct MemoryPressureMonitor {
    cpu_threshold: usize,
    gpu_threshold: usize,
    cleanup_triggered: bool,
    last_cleanup: Instant,
    pressure_events: VecDeque<PressureEvent>,
}

/// Memory pressure event
#[derive(Debug, Clone)]
struct PressureEvent {
    timestamp: Instant,
    pressure_type: PressureType,
    memory_usage: usize,
    threshold: usize,
}

#[derive(Debug, Clone)]
enum PressureType {
    CpuHigh,
    GpuHigh,
    CacheEviction,
}

impl HybridMemoryManager {
    /// Create a new hybrid memory manager
    pub fn new(config: &BridgeConfig) -> NeuralResult<Self> {
        let cpu_pool = Arc::new(Mutex::new(CpuMemoryPool::new()));
        let gpu_pool = Arc::new(Mutex::new(GpuMemoryPool::new()));
        let transfer_cache = Arc::new(RwLock::new(TransferCache::new(
            config.memory_pool_size * 1024 * 1024 / 4, // Convert MB to f32 count
        )));
        let stats = Arc::new(Mutex::new(MemoryStatsTracker::new()));
        let pressure_monitor = Arc::new(Mutex::new(MemoryPressureMonitor::new(
            config.memory_pool_size * 1024 * 1024, // CPU threshold
            config.memory_pool_size * 1024 * 1024 / 2, // GPU threshold (conservative)
        )));
        
        Ok(Self {
            config: config.clone(),
            cpu_pool,
            gpu_pool,
            transfer_cache,
            stats,
            pressure_monitor,
        })
    }
    
    /// Set GPU device for GPU operations
    pub fn set_gpu_device(&self, device: Arc<wgpu::Device>) -> NeuralResult<()> {
        let mut gpu_pool = self.gpu_pool.lock().unwrap();
        gpu_pool.device = Some(device);
        Ok(())
    }
    
    /// Perform memory cleanup when under pressure
    fn cleanup_memory(&self) -> NeuralResult<()> {
        // Clean CPU pool
        {
            let mut cpu_pool = self.cpu_pool.lock().unwrap();
            cpu_pool.cleanup_old_buffers();
        }
        
        // Clean GPU pool
        {
            let mut gpu_pool = self.gpu_pool.lock().unwrap();
            gpu_pool.cleanup_old_buffers();
        }
        
        // Clean transfer cache
        {
            let mut cache = self.transfer_cache.write().unwrap();
            cache.evict_lru();
        }
        
        // Update pressure monitor
        {
            let mut monitor = self.pressure_monitor.lock().unwrap();
            monitor.cleanup_triggered = true;
            monitor.last_cleanup = Instant::now();
        }
        
        log::info!("Memory cleanup completed");
        Ok(())
    }
    
    /// Check memory pressure and trigger cleanup if needed
    fn check_memory_pressure(&self) -> NeuralResult<()> {
        let stats = self.get_memory_stats();
        
        let mut should_cleanup = false;
        
        // Check memory pressure with monitor
        {
            let mut monitor = self.pressure_monitor.lock().unwrap();
            
            // Check CPU pressure
            let cpu_threshold = monitor.cpu_threshold;
            if stats.cpu_allocated > cpu_threshold {
                monitor.pressure_events.push_back(PressureEvent {
                    timestamp: Instant::now(),
                    pressure_type: PressureType::CpuHigh,
                    memory_usage: stats.cpu_allocated,
                    threshold: cpu_threshold,
                });
                
                if !monitor.cleanup_triggered || 
                   monitor.last_cleanup.elapsed() > Duration::from_secs(30) {
                    should_cleanup = true;
                }
            }
            
            // Check GPU pressure
            let gpu_threshold = monitor.gpu_threshold;
            if stats.gpu_allocated > gpu_threshold {
                monitor.pressure_events.push_back(PressureEvent {
                    timestamp: Instant::now(),
                    pressure_type: PressureType::GpuHigh,
                    memory_usage: stats.gpu_allocated,
                    threshold: gpu_threshold,
                });
            
                if !monitor.cleanup_triggered || 
                   monitor.last_cleanup.elapsed() > Duration::from_secs(30) {
                    should_cleanup = true;
                }
            }
        } // monitor lock released here
        
        if should_cleanup {
            self.cleanup_memory()?;
        }
        
        Ok(())
    }
}

impl MemoryManagerTrait for HybridMemoryManager {
    fn allocate(&self, size: usize) -> NeuralResult<MemoryHandle> {
        self.check_memory_pressure()?;
        
        let mut cpu_pool = self.cpu_pool.lock().unwrap();
        let buffer = cpu_pool.allocate(size);
        
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cpu_allocated += size * 4; // f32 = 4 bytes
            stats.total_allocations += 1;
            stats.peak_cpu = stats.peak_cpu.max(stats.cpu_allocated);
        }
        
        Ok(MemoryHandle(buffer.as_ptr() as u64))
    }
    
    fn deallocate(&self, handle: MemoryHandle) -> NeuralResult<()> {
        // In a real implementation, we would track allocations and deallocate properly
        // For now, we'll just update stats
        let mut stats = self.stats.lock().unwrap();
        stats.total_deallocations += 1;
        Ok(())
    }
    
    fn transfer_to_gpu(&self, data: &[f32]) -> NeuralResult<BufferHandle> {
        self.check_memory_pressure()?;
        
        // Check cache first
        let data_hash = calculate_hash(data);
        {
            let mut cache = self.transfer_cache.write().unwrap();
            if let Some(cached) = cache.get_mut(&data_hash) {
                cached.last_accessed = Instant::now();
                cached.access_count += 1;
                
                if let Some(buffer_handle) = cached.gpu_buffer {
                    let mut stats = self.stats.lock().unwrap();
                    stats.cache_hits += 1;
                    return Ok(buffer_handle);
                }
            }
        }
        
        // Cache miss - create new GPU buffer
        let mut gpu_pool = self.gpu_pool.lock().unwrap();
        let buffer_handle = gpu_pool.create_buffer(data)?;
        
        // Cache the transfer
        {
            let mut cache = self.transfer_cache.write().unwrap();
            cache.insert(data_hash, CachedTransfer {
                data: data.to_vec(),
                gpu_buffer: Some(buffer_handle),
                last_accessed: Instant::now(),
                access_count: 1,
            });
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_misses += 1;
            stats.transfer_bytes += data.len() as u64 * 4;
            stats.gpu_allocated += data.len() * 4;
            stats.peak_gpu = stats.peak_gpu.max(stats.gpu_allocated);
        }
        
        Ok(buffer_handle)
    }
    
    fn transfer_from_gpu(&self, buffer: BufferHandle) -> NeuralResult<Vec<f32>> {
        let gpu_pool = self.gpu_pool.lock().unwrap();
        let data = gpu_pool.read_buffer(buffer)?;
        
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.transfer_bytes += data.len() as u64 * 4;
        }
        
        Ok(data)
    }
    
    fn get_memory_stats(&self) -> MemoryStats {
        let stats = self.stats.lock().unwrap();
        MemoryStats {
            total_allocated: stats.cpu_allocated + stats.gpu_allocated,
            gpu_allocated: stats.gpu_allocated,
            cpu_allocated: stats.cpu_allocated,
            peak_usage: stats.peak_cpu.max(stats.peak_gpu),
            allocations: stats.total_allocations,
            deallocations: stats.total_deallocations,
        }
    }
}

impl CpuMemoryPool {
    fn new() -> Self {
        Self {
            pools: HashMap::new(),
            allocated_bytes: 0,
            allocations: 0,
            deallocations: 0,
        }
    }
    
    fn allocate(&mut self, size: usize) -> Vec<f32> {
        // Round up to nearest power of 2 for better pooling
        let pool_size = size.next_power_of_two();
        
        if let Some(pool) = self.pools.get_mut(&pool_size) {
            if let Some(mut buffer) = pool.pop_front() {
                buffer.resize(size, 0.0);
                self.allocations += 1;
                return buffer;
            }
        }
        
        // Create new buffer
        let buffer = vec![0.0f32; size];
        self.allocated_bytes += size * 4;
        self.allocations += 1;
        buffer
    }
    
    fn deallocate(&mut self, mut buffer: Vec<f32>, original_size: usize) {
        let pool_size = original_size.next_power_of_two();
        buffer.clear();
        buffer.resize(pool_size, 0.0);
        
        self.pools.entry(pool_size).or_default().push_back(buffer);
        self.deallocations += 1;
    }
    
    fn cleanup_old_buffers(&mut self) {
        // Keep only recent pools and limit pool sizes
        for (_, pool) in self.pools.iter_mut() {
            while pool.len() > 10 { // Limit pool size
                pool.pop_front();
            }
        }
    }
}

impl GpuMemoryPool {
    fn new() -> Self {
        Self {
            device: None,
            buffers: HashMap::new(),
            free_buffers: HashMap::new(),
            allocated_bytes: 0,
            allocations: 0,
            deallocations: 0,
            next_handle: 1,
        }
    }
    
    fn create_buffer(&mut self, data: &[f32]) -> NeuralResult<BufferHandle> {
        let device = self.device.as_ref().ok_or_else(|| {
            NeuralIntegrationError::GpuInitError("GPU device not set".to_string())
        })?;
        
        let size = data.len() * 4; // f32 = 4 bytes
        
        // Try to reuse existing buffer
        if let Some(pool) = self.free_buffers.get_mut(&size) {
            if let Some(handle) = pool.pop_front() {
                if let Some(gpu_buffer) = self.buffers.get_mut(&handle) {
                    // Write data to existing buffer
                    gpu_buffer.last_used = Instant::now();
                    gpu_buffer.usage_count += 1;
                    // TODO: Write data to buffer using queue.write_buffer
                    return Ok(handle);
                }
            }
        }
        
        // Create new buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neural data buffer"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        
        // Write data to buffer
        {
            let mut buffer_view = buffer.slice(..).get_mapped_range_mut();
            let data_bytes = bytemuck::cast_slice(data);
            buffer_view.copy_from_slice(data_bytes);
        }
        buffer.unmap();
        
        let handle = BufferHandle(self.next_handle);
        self.next_handle += 1;
        
        let gpu_buffer = GpuBuffer {
            buffer,
            size,
            last_used: Instant::now(),
            usage_count: 1,
        };
        
        self.buffers.insert(handle, gpu_buffer);
        self.allocated_bytes += size;
        self.allocations += 1;
        
        Ok(handle)
    }
    
    fn read_buffer(&self, handle: BufferHandle) -> NeuralResult<Vec<f32>> {
        let gpu_buffer = self.buffers.get(&handle).ok_or_else(|| {
            NeuralIntegrationError::OperationError("Invalid buffer handle".to_string())
        })?;
        
        // TODO: Implement actual buffer reading using WebGPU
        // For now, return dummy data
        Ok(vec![0.0f32; gpu_buffer.size / 4])
    }
    
    fn cleanup_old_buffers(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(300); // 5 minutes
        
        let mut to_remove = Vec::new();
        for (handle, gpu_buffer) in &self.buffers {
            if gpu_buffer.last_used < cutoff && gpu_buffer.usage_count < 2 {
                to_remove.push(*handle);
            }
        }
        
        for handle in to_remove {
            if let Some(gpu_buffer) = self.buffers.remove(&handle) {
                self.allocated_bytes -= gpu_buffer.size;
                self.deallocations += 1;
                
                // Add to free pool
                self.free_buffers.entry(gpu_buffer.size)
                    .or_default()
                    .push_back(handle);
            }
        }
    }
}

impl TransferCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries: 1000,
            total_size: 0,
            max_size,
        }
    }
    
    fn get_mut(&mut self, key: &u64) -> Option<&mut CachedTransfer> {
        self.cache.get_mut(key)
    }
    
    fn insert(&mut self, key: u64, transfer: CachedTransfer) {
        self.total_size += transfer.data.len();
        self.cache.insert(key, transfer);
        
        // Evict if necessary
        if self.cache.len() > self.max_entries || self.total_size > self.max_size {
            self.evict_lru();
        }
    }
    
    fn evict_lru(&mut self) {
        if self.cache.is_empty() {
            return;
        }
        
        // Find least recently used entry
        let mut oldest_key = None;
        let mut oldest_time = Instant::now();
        
        for (key, transfer) in &self.cache {
            if transfer.last_accessed < oldest_time {
                oldest_time = transfer.last_accessed;
                oldest_key = Some(*key);
            }
        }
        
        if let Some(key) = oldest_key {
            if let Some(transfer) = self.cache.remove(&key) {
                self.total_size -= transfer.data.len();
            }
        }
    }
}

impl MemoryStatsTracker {
    fn new() -> Self {
        Self {
            cpu_allocated: 0,
            gpu_allocated: 0,
            peak_cpu: 0,
            peak_gpu: 0,
            total_allocations: 0,
            total_deallocations: 0,
            cache_hits: 0,
            cache_misses: 0,
            transfer_bytes: 0,
        }
    }
}

impl MemoryPressureMonitor {
    fn new(cpu_threshold: usize, gpu_threshold: usize) -> Self {
        Self {
            cpu_threshold,
            gpu_threshold,
            cleanup_triggered: false,
            last_cleanup: Instant::now() - Duration::from_secs(3600), // Start with old timestamp
            pressure_events: VecDeque::new(),
        }
    }
}

/// Calculate hash for data caching
fn calculate_hash(data: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    
    // Hash a sample of the data for performance
    let sample_size = (data.len() / 100).max(1).min(1000);
    for i in (0..data.len()).step_by(data.len() / sample_size + 1) {
        data[i].to_bits().hash(&mut hasher);
    }
    data.len().hash(&mut hasher);
    
    hasher.finish()
}

/// No-op memory manager for testing
pub struct NoOpMemoryManager;

impl MemoryManagerTrait for NoOpMemoryManager {
    fn allocate(&self, _size: usize) -> NeuralResult<MemoryHandle> {
        Ok(MemoryHandle(0))
    }
    
    fn deallocate(&self, _handle: MemoryHandle) -> NeuralResult<()> {
        Ok(())
    }
    
    fn transfer_to_gpu(&self, data: &[f32]) -> NeuralResult<BufferHandle> {
        Ok(BufferHandle(data.as_ptr() as u64))
    }
    
    fn transfer_from_gpu(&self, _buffer: BufferHandle) -> NeuralResult<Vec<f32>> {
        Ok(vec![0.0; 100]) // Dummy data
    }
    
    fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: 0,
            gpu_allocated: 0,
            cpu_allocated: 0,
            peak_usage: 0,
            allocations: 0,
            deallocations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_memory_pool() {
        let mut pool = CpuMemoryPool::new();
        
        let buffer1 = pool.allocate(100);
        assert_eq!(buffer1.len(), 100);
        
        let buffer2 = pool.allocate(200);
        assert_eq!(buffer2.len(), 200);
        
        assert_eq!(pool.allocations, 2);
    }
    
    #[test]
    fn test_transfer_cache() {
        let mut cache = TransferCache::new(1000);
        
        let transfer = CachedTransfer {
            data: vec![1.0, 2.0, 3.0],
            gpu_buffer: Some(BufferHandle(1)),
            last_accessed: Instant::now(),
            access_count: 1,
        };
        
        cache.insert(123, transfer);
        assert!(cache.cache.contains_key(&123));
    }
    
    #[test]
    fn test_memory_stats() {
        let config = BridgeConfig::default();
        let manager = HybridMemoryManager::new(&config).unwrap();
        
        let stats = manager.get_memory_stats();
        assert_eq!(stats.total_allocated, 0);
    }
    
    #[test]
    fn test_hash_calculation() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let data3 = vec![1.0, 2.0, 3.0, 5.0];
        
        assert_eq!(calculate_hash(&data1), calculate_hash(&data2));
        assert_ne!(calculate_hash(&data1), calculate_hash(&data3));
    }
}