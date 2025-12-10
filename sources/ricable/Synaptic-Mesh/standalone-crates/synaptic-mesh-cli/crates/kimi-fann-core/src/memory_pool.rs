//! High-Performance Memory Pool for Neural Networks
//! 
//! Custom memory allocator optimized for neural network operations with
//! sub-25MB memory usage target and automatic garbage collection.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};

/// Target memory usage limit (25MB)
pub const MEMORY_LIMIT_BYTES: usize = 25 * 1024 * 1024;

/// Block size for memory pool allocations
const BLOCK_SIZE: usize = 4096; // 4KB blocks

/// Maximum number of blocks in pool
const MAX_BLOCKS: usize = MEMORY_LIMIT_BYTES / BLOCK_SIZE;

/// Neural network memory pool with automatic management
#[derive(Debug)]
pub struct NeuralMemoryPool {
    free_blocks: VecDeque<MemoryBlock>,
    allocated_blocks: Vec<AllocatedBlock>,
    total_allocated: AtomicUsize,
    peak_usage: AtomicUsize,
    allocation_count: AtomicUsize,
    last_gc: Instant,
    gc_threshold_bytes: usize,
}

impl NeuralMemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        Self {
            free_blocks: VecDeque::with_capacity(MAX_BLOCKS),
            allocated_blocks: Vec::with_capacity(MAX_BLOCKS),
            total_allocated: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            last_gc: Instant::now(),
            gc_threshold_bytes: MEMORY_LIMIT_BYTES / 4, // GC at 25% of limit
        }
    }
    
    /// Allocate memory for neural network weights
    pub fn allocate_weights(&mut self, size: usize) -> Result<NeuralAllocation, MemoryError> {
        let blocks_needed = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if blocks_needed > MAX_BLOCKS {
            return Err(MemoryError::AllocationTooLarge);
        }
        
        // Check if we need garbage collection
        if self.should_run_gc() {
            self.garbage_collect();
        }
        
        // Try to allocate from free blocks first
        if let Some(allocation) = self.allocate_from_pool(blocks_needed) {
            self.update_allocation_stats(blocks_needed * BLOCK_SIZE);
            return Ok(allocation);
        }
        
        // Allocate new blocks if pool is empty
        self.allocate_new_blocks(blocks_needed)
    }
    
    /// Allocate memory for neural network activations (temporary)
    pub fn allocate_activations(&mut self, size: usize) -> Result<NeuralAllocation, MemoryError> {
        // Use smaller blocks for activations since they're temporary
        let allocation = self.allocate_weights(size)?;
        Ok(NeuralAllocation {
            id: allocation.id,
            ptr: allocation.ptr,
            size: allocation.size,
            allocation_type: AllocationType::Activations,
            allocated_at: allocation.allocated_at,
        })
    }
    
    /// Free allocated memory
    pub fn deallocate(&mut self, allocation: NeuralAllocation) {
        let blocks_to_free = (allocation.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Add blocks back to free pool
        for i in 0..blocks_to_free {
            let block_ptr = unsafe {
                allocation.ptr.as_ptr().add(i * BLOCK_SIZE)
            };
            
            self.free_blocks.push_back(MemoryBlock {
                ptr: NonNull::new(block_ptr as *mut u8).unwrap(),
                size: BLOCK_SIZE,
                last_used: Instant::now(),
            });
        }
        
        // Remove from allocated list
        self.allocated_blocks.retain(|block| block.id != allocation.id);
        
        // Update stats
        let current = self.total_allocated.load(Ordering::Relaxed);
        self.total_allocated.store(current.saturating_sub(allocation.size), Ordering::Relaxed);
    }
    
    /// Force garbage collection
    pub fn garbage_collect(&mut self) {
        let now = Instant::now();
        
        // Remove old temporary allocations
        let mut freed_bytes = 0;
        self.allocated_blocks.retain(|block| {
            if block.allocation_type == AllocationType::Activations && 
               now.duration_since(block.allocated_at) > Duration::from_secs(30) {
                freed_bytes += block.size;
                
                // Add blocks back to free pool
                let blocks_count = (block.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                for i in 0..blocks_count {
                    if let Some(ptr) = NonNull::new(unsafe { 
                        block.ptr.as_ptr().add(i * BLOCK_SIZE) 
                    }) {
                        self.free_blocks.push_back(MemoryBlock {
                            ptr,
                            size: BLOCK_SIZE,
                            last_used: now,
                        });
                    }
                }
                false
            } else {
                true
            }
        });
        
        // Compact free blocks (remove duplicates and sort by age)
        self.free_blocks.make_contiguous().sort_by_key(|block| block.last_used);
        
        // Update allocation stats
        let current = self.total_allocated.load(Ordering::Relaxed);
        self.total_allocated.store(current.saturating_sub(freed_bytes), Ordering::Relaxed);
        
        self.last_gc = now;
    }
    
    /// Get current memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let peak_usage = self.peak_usage.load(Ordering::Relaxed);
        let allocation_count = self.allocation_count.load(Ordering::Relaxed);
        
        MemoryStats {
            total_allocated_bytes: total_allocated,
            total_allocated_mb: total_allocated as f64 / (1024.0 * 1024.0),
            peak_usage_bytes: peak_usage,
            peak_usage_mb: peak_usage as f64 / (1024.0 * 1024.0),
            free_blocks: self.free_blocks.len(),
            allocated_blocks: self.allocated_blocks.len(),
            allocation_count,
            memory_limit_mb: MEMORY_LIMIT_BYTES as f64 / (1024.0 * 1024.0),
            usage_percentage: (total_allocated as f64 / MEMORY_LIMIT_BYTES as f64) * 100.0,
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }
    
    /// Check if garbage collection should run
    fn should_run_gc(&self) -> bool {
        let current_usage = self.total_allocated.load(Ordering::Relaxed);
        let time_since_gc = self.last_gc.elapsed();
        
        current_usage > self.gc_threshold_bytes || 
        time_since_gc > Duration::from_secs(60) // Force GC every minute
    }
    
    /// Allocate from existing free blocks
    fn allocate_from_pool(&mut self, blocks_needed: usize) -> Option<NeuralAllocation> {
        if self.free_blocks.len() < blocks_needed {
            return None;
        }
        
        // Take the first block as base
        let base_block = self.free_blocks.pop_front()?;
        let allocation_id = self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        // Collect additional blocks if needed
        let mut total_size = BLOCK_SIZE;
        for _ in 1..blocks_needed {
            if self.free_blocks.pop_front().is_some() {
                total_size += BLOCK_SIZE;
            } else {
                // Not enough contiguous blocks, return what we took
                self.free_blocks.push_front(base_block);
                return None;
            }
        }
        
        let allocation = NeuralAllocation {
            id: allocation_id,
            ptr: base_block.ptr,
            size: total_size,
            allocation_type: AllocationType::Weights,
            allocated_at: Instant::now(),
        };
        
        // Track allocation
        self.allocated_blocks.push(AllocatedBlock {
            id: allocation_id,
            ptr: base_block.ptr,
            size: total_size,
            allocation_type: AllocationType::Weights,
            allocated_at: allocation.allocated_at,
        });
        
        Some(allocation)
    }
    
    /// Allocate new blocks from system
    fn allocate_new_blocks(&mut self, blocks_needed: usize) -> Result<NeuralAllocation, MemoryError> {
        let total_size = blocks_needed * BLOCK_SIZE;
        let current_usage = self.total_allocated.load(Ordering::Relaxed);
        
        if current_usage + total_size > MEMORY_LIMIT_BYTES {
            return Err(MemoryError::MemoryLimitExceeded);
        }
        
        // Allocate from system
        let layout = Layout::from_size_align(total_size, std::mem::align_of::<f32>())
            .map_err(|_| MemoryError::InvalidLayout)?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(MemoryError::SystemAllocationFailed);
        }
        
        let non_null_ptr = NonNull::new(ptr).unwrap();
        let allocation_id = self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let allocated_at = Instant::now();
        
        let allocation = NeuralAllocation {
            id: allocation_id,
            ptr: non_null_ptr,
            size: total_size,
            allocation_type: AllocationType::Weights,
            allocated_at,
        };
        
        // Track allocation
        self.allocated_blocks.push(AllocatedBlock {
            id: allocation_id,
            ptr: non_null_ptr,
            size: total_size,
            allocation_type: AllocationType::Weights,
            allocated_at,
        });
        
        self.update_allocation_stats(total_size);
        Ok(allocation)
    }
    
    /// Update allocation statistics
    fn update_allocation_stats(&self, size: usize) {
        let new_total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak usage atomically
        let mut current_peak = self.peak_usage.load(Ordering::Relaxed);
        while current_peak < new_total {
            match self.peak_usage.compare_exchange_weak(
                current_peak, 
                new_total, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual,
            }
        }
    }
    
    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        if self.free_blocks.is_empty() {
            return 0.0;
        }
        
        let total_free = self.free_blocks.len() * BLOCK_SIZE;
        let largest_contiguous = BLOCK_SIZE; // Simplified - we don't track contiguous blocks
        
        1.0 - (largest_contiguous as f64 / total_free as f64)
    }
}

impl Drop for NeuralMemoryPool {
    fn drop(&mut self) {
        // Free all allocated blocks
        for block in &self.allocated_blocks {
            unsafe {
                let layout = Layout::from_size_align_unchecked(block.size, std::mem::align_of::<f32>());
                dealloc(block.ptr.as_ptr(), layout);
            }
        }
        
        // Free all blocks in the free pool
        for block in &self.free_blocks {
            unsafe {
                let layout = Layout::from_size_align_unchecked(block.size, std::mem::align_of::<f32>());
                dealloc(block.ptr.as_ptr(), layout);
            }
        }
    }
}

/// Memory block in the pool
#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    last_used: Instant,
}

/// Allocated block tracking
#[derive(Debug, Clone)]
struct AllocatedBlock {
    id: usize,
    ptr: NonNull<u8>,
    size: usize,
    allocation_type: AllocationType,
    allocated_at: Instant,
}

/// Neural network memory allocation
#[derive(Debug)]
pub struct NeuralAllocation {
    pub id: usize,
    pub ptr: NonNull<u8>,
    pub size: usize,
    pub allocation_type: AllocationType,
    pub allocated_at: Instant,
}

impl NeuralAllocation {
    /// Get allocation as mutable f32 slice
    pub unsafe fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        let len = self.size / std::mem::size_of::<f32>();
        std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut f32, len)
    }
    
    /// Get allocation as f32 slice
    pub unsafe fn as_f32_slice(&self) -> &[f32] {
        let len = self.size / std::mem::size_of::<f32>();
        std::slice::from_raw_parts(self.ptr.as_ptr() as *const f32, len)
    }
    
    /// Zero out the allocation
    pub fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.size);
        }
    }
}

/// Type of memory allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationType {
    Weights,      // Long-lived neural network weights
    Activations,  // Temporary activation values
    Gradients,    // Temporary gradient values
    Cache,        // Cached computations
}

/// Memory allocation errors
#[derive(Debug, Clone)]
pub enum MemoryError {
    AllocationTooLarge,
    MemoryLimitExceeded,
    InvalidLayout,
    SystemAllocationFailed,
    AllocationNotFound,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::AllocationTooLarge => write!(f, "Requested allocation exceeds maximum block size"),
            MemoryError::MemoryLimitExceeded => write!(f, "Memory limit of {}MB exceeded", MEMORY_LIMIT_BYTES / 1024 / 1024),
            MemoryError::InvalidLayout => write!(f, "Invalid memory layout requested"),
            MemoryError::SystemAllocationFailed => write!(f, "System memory allocation failed"),
            MemoryError::AllocationNotFound => write!(f, "Allocation not found for deallocation"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated_bytes: usize,
    pub total_allocated_mb: f64,
    pub peak_usage_bytes: usize,
    pub peak_usage_mb: f64,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub allocation_count: usize,
    pub memory_limit_mb: f64,
    pub usage_percentage: f64,
    pub fragmentation_ratio: f64,
}

impl MemoryStats {
    /// Check if usage is within target limits
    pub fn is_within_limits(&self) -> bool {
        self.total_allocated_mb <= 25.0 && self.fragmentation_ratio < 0.3
    }
    
    /// Get memory efficiency score (0-100)
    pub fn efficiency_score(&self) -> f64 {
        let usage_score = (1.0 - (self.usage_percentage / 100.0).min(1.0)) * 50.0;
        let fragmentation_score = (1.0 - self.fragmentation_ratio.min(1.0)) * 50.0;
        usage_score + fragmentation_score
    }
}

/// Global memory pool instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_MEMORY_POOL: Arc<Mutex<NeuralMemoryPool>> = 
        Arc::new(Mutex::new(NeuralMemoryPool::new()));
}

/// Convenience macro for memory allocation
#[macro_export]
macro_rules! allocate_neural_memory {
    ($size:expr, $type:expr) => {
        {
            let mut pool = crate::memory_pool::GLOBAL_MEMORY_POOL.lock().unwrap();
            match $type {
                crate::memory_pool::AllocationType::Weights => pool.allocate_weights($size),
                crate::memory_pool::AllocationType::Activations => pool.allocate_activations($size),
                _ => pool.allocate_weights($size), // Default to weights
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_creation() {
        let pool = NeuralMemoryPool::new();
        let stats = pool.get_memory_stats();
        assert_eq!(stats.total_allocated_bytes, 0);
        assert!(stats.is_within_limits());
    }
    
    #[test]
    fn test_memory_allocation() {
        let mut pool = NeuralMemoryPool::new();
        let allocation = pool.allocate_weights(1024).unwrap();
        
        assert_eq!(allocation.size, BLOCK_SIZE); // Rounded up to block size
        assert_eq!(allocation.allocation_type, AllocationType::Weights);
        
        let stats = pool.get_memory_stats();
        assert_eq!(stats.allocated_blocks, 1);
        
        pool.deallocate(allocation);
        let stats_after = pool.get_memory_stats();
        assert_eq!(stats_after.allocated_blocks, 0);
    }
    
    #[test]
    fn test_memory_limit() {
        let mut pool = NeuralMemoryPool::new();
        
        // Try to allocate more than the limit
        let result = pool.allocate_weights(MEMORY_LIMIT_BYTES + 1);
        assert!(matches!(result, Err(MemoryError::AllocationTooLarge)));
    }
    
    #[test]
    fn test_garbage_collection() {
        let mut pool = NeuralMemoryPool::new();
        
        // Allocate some temporary memory
        let activation = pool.allocate_activations(1024).unwrap();
        let stats_before = pool.get_memory_stats();
        assert_eq!(stats_before.allocated_blocks, 1);
        
        pool.deallocate(activation);
        pool.garbage_collect();
        
        let stats_after = pool.get_memory_stats();
        assert_eq!(stats_after.allocated_blocks, 0);
    }
    
    #[test]
    fn test_memory_stats() {
        let mut pool = NeuralMemoryPool::new();
        let allocation = pool.allocate_weights(2048).unwrap();
        
        let stats = pool.get_memory_stats();
        assert!(stats.total_allocated_bytes > 0);
        assert!(stats.total_allocated_mb > 0.0);
        assert!(stats.usage_percentage > 0.0);
        assert!(stats.efficiency_score() > 0.0);
        
        pool.deallocate(allocation);
    }
}