//! Memory Pool Management for WASM
//!
//! This module implements efficient memory management for micro-experts
//! in WebAssembly environments with limited heap space.

use crate::*;
use lru::LruCache;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Maximum total memory usage in bytes
    pub max_memory_bytes: usize,
    /// Maximum number of active experts
    pub max_active_experts: usize,
    /// Expert cache size
    pub expert_cache_size: usize,
    /// Enable memory compaction
    pub enable_compaction: bool,
    /// Garbage collection threshold (0.0 to 1.0)
    pub gc_threshold: f32,
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 128 * 1024 * 1024, // 128MB
            max_active_experts: 20,
            expert_cache_size: 50,
            enable_compaction: true,
            gc_threshold: 0.8,
            memory_alignment: 32, // 256-bit alignment for SIMD
        }
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocationStats {
    /// Total allocated memory
    pub total_allocated: usize,
    /// Memory in use
    pub memory_in_use: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Number of cached experts
    pub cached_experts: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f32,
    /// Last garbage collection timestamp
    pub last_gc_time: f64,
    /// Number of garbage collections performed
    pub gc_count: u64,
}

/// Memory block for expert storage
#[derive(Debug, Clone)]
struct MemoryBlock {
    /// Block identifier
    id: u64,
    /// Data storage
    data: Vec<u8>,
    /// Size of the block
    size: usize,
    /// Reference count
    ref_count: u32,
    /// Last access timestamp
    last_access: f64,
    /// Whether this block is aligned for SIMD
    simd_aligned: bool,
}

impl MemoryBlock {
    fn new(id: u64, size: usize, alignment: usize) -> Self {
        let simd_aligned = alignment >= 32;
        let data = if simd_aligned {
            Self::allocate_aligned(size, alignment)
        } else {
            vec![0u8; size]
        };

        Self {
            id,
            data,
            size,
            ref_count: 1,
            last_access: Utils::now(),
            simd_aligned,
        }
    }

    fn allocate_aligned(size: usize, alignment: usize) -> Vec<u8> {
        // Allocate extra bytes to ensure alignment
        let mut vec = vec![0u8; size + alignment - 1];
        let ptr = vec.as_mut_ptr();
        let aligned_ptr = ((ptr as usize + alignment - 1) / alignment) * alignment;
        let offset = aligned_ptr - ptr as usize;
        
        // Adjust the vector to start at the aligned address
        if offset > 0 {
            vec = vec![0u8; size + offset];
        }
        
        vec.resize(size, 0);
        vec
    }

    fn increment_ref(&mut self) {
        self.ref_count += 1;
        self.last_access = Utils::now();
    }

    fn decrement_ref(&mut self) -> bool {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count == 0
    }

    fn is_expired(&self, current_time: f64, ttl_ms: f64) -> bool {
        current_time - self.last_access > ttl_ms
    }
}

/// Memory pool for expert management
#[wasm_bindgen]
pub struct WasmMemoryPool {
    /// Configuration
    config: MemoryPoolConfig,
    /// Active memory blocks
    blocks: HashMap<u64, MemoryBlock>,
    /// Expert cache (LRU)
    expert_cache: LruCache<ExpertId, CompressedExpert>,
    /// Active experts in memory
    active_experts: HashMap<ExpertId, u64>, // Expert ID -> Block ID
    /// Free block queue for reuse
    free_blocks: VecDeque<u64>,
    /// Next block ID
    next_block_id: u64,
    /// Memory allocation statistics
    stats: MemoryAllocationStats,
    /// Expert compressor for memory management
    compressor: ExpertCompressor,
}

#[wasm_bindgen]
impl WasmMemoryPool {
    /// Create a new memory pool
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmMemoryPool, JsValue> {
        let config: MemoryPoolConfig = serde_json::from_str(config_json)
            .unwrap_or_else(|_| MemoryPoolConfig::default());

        let expert_cache_size = NonZeroUsize::new(config.expert_cache_size)
            .unwrap_or_else(|| NonZeroUsize::new(1).unwrap());

        Ok(WasmMemoryPool {
            config: config.clone(),
            blocks: HashMap::new(),
            expert_cache: LruCache::new(expert_cache_size),
            active_experts: HashMap::new(),
            free_blocks: VecDeque::new(),
            next_block_id: 1,
            stats: MemoryAllocationStats {
                total_allocated: 0,
                memory_in_use: 0,
                active_allocations: 0,
                cached_experts: 0,
                fragmentation_ratio: 0.0,
                last_gc_time: Utils::now(),
                gc_count: 0,
            },
            compressor: ExpertCompressor::default(),
        })
    }

    /// Load an expert into memory
    #[wasm_bindgen]
    pub fn load_expert(&mut self, expert_json: &str) -> Result<u64, JsValue> {
        let expert: KimiMicroExpert = serde_json::from_str(expert_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid expert JSON: {}", e)))?;

        let expert_id = expert.id();

        // Check if expert is already active
        if let Some(&block_id) = self.active_experts.get(&expert_id) {
            // Increment reference count
            if let Some(block) = self.blocks.get_mut(&block_id) {
                block.increment_ref();
                return Ok(block_id);
            }
        }

        // Check cache first
        if self.expert_cache.contains(&expert_id) {
            return self.load_from_cache(expert_id);
        }

        // Compress and load expert
        let compressed = self.compressor.compress_expert(&expert)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let block_id = self.allocate_block(compressed.compressed_weights.len())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Store compressed expert data
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.data.clear();
            block.data.extend_from_slice(&compressed.compressed_weights);
        }

        // Cache the compressed expert
        self.expert_cache.put(expert_id, compressed);
        self.active_experts.insert(expert_id, block_id);

        // Update statistics
        self.update_stats();

        // Check if garbage collection is needed
        if self.should_trigger_gc() {
            self.garbage_collect()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(block_id)
    }

    /// Unload an expert from memory
    #[wasm_bindgen]
    pub fn unload_expert(&mut self, expert_id: ExpertId) -> Result<(), JsValue> {
        if let Some(block_id) = self.active_experts.remove(&expert_id) {
            if let Some(block) = self.blocks.get_mut(&block_id) {
                if block.decrement_ref() {
                    // No more references, free the block
                    self.free_block(block_id)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?;
                }
            }
        }

        self.update_stats();
        Ok(())
    }

    /// Get expert data from memory
    #[wasm_bindgen]
    pub fn get_expert_data(&mut self, expert_id: ExpertId) -> Result<String, JsValue> {
        let block_id = self.active_experts.get(&expert_id)
            .ok_or_else(|| JsValue::from_str("Expert not loaded"))?;

        let compressed = self.expert_cache.get(&expert_id)
            .ok_or_else(|| JsValue::from_str("Expert not in cache"))?;

        let expert = self.compressor.decompress_expert(compressed)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        expert.to_json()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get memory statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.stats)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Force garbage collection
    #[wasm_bindgen]
    pub fn garbage_collect(&mut self) -> Result<(), JsValue> {
        self.perform_garbage_collection()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compact memory
    #[wasm_bindgen]
    pub fn compact_memory(&mut self) -> Result<(), JsValue> {
        if !self.config.enable_compaction {
            return Ok(());
        }

        self.perform_memory_compaction()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clear all cached experts
    #[wasm_bindgen]
    pub fn clear_cache(&mut self) {
        self.expert_cache.clear();
        self.active_experts.clear();
        self.blocks.clear();
        self.free_blocks.clear();
        self.next_block_id = 1;
        self.update_stats();
    }

    /// Check if memory pool is healthy
    #[wasm_bindgen]
    pub fn is_healthy(&self) -> bool {
        self.stats.memory_in_use <= self.config.max_memory_bytes &&
        self.stats.cached_experts <= self.config.expert_cache_size &&
        self.stats.fragmentation_ratio < 0.5
    }
}

impl WasmMemoryPool {
    /// Load expert from cache
    fn load_from_cache(&mut self, expert_id: ExpertId) -> Result<u64, JsValue> {
        let compressed = self.expert_cache.get(&expert_id)
            .ok_or_else(|| JsValue::from_str("Expert not in cache"))?
            .clone();

        let block_id = self.allocate_block(compressed.compressed_weights.len())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Store compressed expert data
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.data.clear();
            block.data.extend_from_slice(&compressed.compressed_weights);
        }

        self.active_experts.insert(expert_id, block_id);
        self.update_stats();

        Ok(block_id)
    }

    /// Allocate a memory block
    fn allocate_block(&mut self, size: usize) -> Result<u64> {
        // Check if we can reuse a free block
        while let Some(block_id) = self.free_blocks.pop_front() {
            if let Some(block) = self.blocks.get_mut(&block_id) {
                if block.size >= size {
                    block.ref_count = 1;
                    block.last_access = Utils::now();
                    block.data.clear();
                    block.data.resize(size, 0);
                    return Ok(block_id);
                }
            }
        }

        // Check memory limits
        if self.stats.memory_in_use + size > self.config.max_memory_bytes {
            // Try to free some memory
            self.evict_unused_experts()?;
            
            if self.stats.memory_in_use + size > self.config.max_memory_bytes {
                return Err(KimiError::MemoryError("Out of memory".to_string()));
            }
        }

        // Allocate new block
        let block_id = self.next_block_id;
        self.next_block_id += 1;

        let block = MemoryBlock::new(block_id, size, self.config.memory_alignment);
        self.blocks.insert(block_id, block);

        Ok(block_id)
    }

    /// Free a memory block
    fn free_block(&mut self, block_id: u64) -> Result<()> {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.ref_count = 0;
            block.data.clear(); // Clear data immediately
            self.free_blocks.push_back(block_id);
        }
        Ok(())
    }

    /// Update memory statistics
    fn update_stats(&mut self) {
        let mut total_allocated = 0;
        let mut memory_in_use = 0;
        let mut active_allocations = 0;

        for block in self.blocks.values() {
            total_allocated += block.size;
            if block.ref_count > 0 {
                memory_in_use += block.size;
                active_allocations += 1;
            }
        }

        self.stats.total_allocated = total_allocated;
        self.stats.memory_in_use = memory_in_use;
        self.stats.active_allocations = active_allocations;
        self.stats.cached_experts = self.expert_cache.len();
        
        // Calculate fragmentation ratio
        self.stats.fragmentation_ratio = if total_allocated > 0 {
            1.0 - (memory_in_use as f32 / total_allocated as f32)
        } else {
            0.0
        };
    }

    /// Check if garbage collection should be triggered
    fn should_trigger_gc(&self) -> bool {
        let memory_utilization = self.stats.memory_in_use as f32 / self.config.max_memory_bytes as f32;
        memory_utilization >= self.config.gc_threshold ||
        self.stats.fragmentation_ratio > 0.3
    }

    /// Perform garbage collection
    fn perform_garbage_collection(&mut self) -> Result<()> {
        let current_time = Utils::now();
        let ttl_ms = 300_000.0; // 5 minutes

        // Remove expired blocks
        let mut expired_blocks = Vec::new();
        for (block_id, block) in &self.blocks {
            if block.ref_count == 0 && block.is_expired(current_time, ttl_ms) {
                expired_blocks.push(*block_id);
            }
        }

        for block_id in expired_blocks {
            self.blocks.remove(&block_id);
            self.free_blocks.retain(|&id| id != block_id);
        }

        // Remove expired cache entries
        let mut expired_experts = Vec::new();
        for (&expert_id, _) in self.expert_cache.iter() {
            if !self.active_experts.contains_key(&expert_id) {
                expired_experts.push(expert_id);
            }
        }

        let take_count = expired_experts.len() / 2;
        for expert_id in expired_experts.into_iter().take(take_count) {
            self.expert_cache.pop(&expert_id);
        }

        self.stats.last_gc_time = current_time;
        self.stats.gc_count += 1;
        self.update_stats();

        Ok(())
    }

    /// Evict unused experts to free memory
    fn evict_unused_experts(&mut self) -> Result<()> {
        let mut candidates: Vec<(ExpertId, f64)> = Vec::new();

        // Find experts that can be evicted (reference count == 1)
        for (&expert_id, &block_id) in &self.active_experts {
            if let Some(block) = self.blocks.get(&block_id) {
                if block.ref_count == 1 {
                    candidates.push((expert_id, block.last_access));
                }
            }
        }

        // Sort by last access time (oldest first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Evict up to half of the candidates
        let evict_count = (candidates.len() / 2).max(1);
        for (expert_id, _) in candidates.into_iter().take(evict_count) {
            self.unload_expert(expert_id)
                .map_err(|e| KimiError::MemoryError(format!("Failed to evict expert: {:?}", e)))?;
        }

        Ok(())
    }

    /// Perform memory compaction
    fn perform_memory_compaction(&mut self) -> Result<()> {
        // Remove completely free blocks
        let mut blocks_to_remove = Vec::new();
        for (block_id, block) in &self.blocks {
            if block.ref_count == 0 && self.free_blocks.contains(block_id) {
                blocks_to_remove.push(*block_id);
            }
        }

        for block_id in blocks_to_remove {
            self.blocks.remove(&block_id);
            self.free_blocks.retain(|&id| id != block_id);
        }

        self.update_stats();
        Ok(())
    }

    /// Estimate memory usage for an expert
    pub fn estimate_expert_memory_usage(&self, expert: &KimiMicroExpert) -> usize {
        // Base expert size
        let base_size = std::mem::size_of::<KimiMicroExpert>();
        
        // Parameter memory (assuming f32 weights)
        let param_size = expert.parameter_count() * std::mem::size_of::<f32>();
        
        // Additional overhead (metadata, etc.)
        let overhead = 1024; // 1KB overhead
        
        base_size + param_size + overhead
    }

    /// Get memory utilization ratio
    pub fn get_memory_utilization(&self) -> f32 {
        if self.config.max_memory_bytes > 0 {
            self.stats.memory_in_use as f32 / self.config.max_memory_bytes as f32
        } else {
            0.0
        }
    }

    /// Check if there's enough memory for an expert
    pub fn can_load_expert(&self, estimated_size: usize) -> bool {
        self.stats.memory_in_use + estimated_size <= self.config.max_memory_bytes &&
        self.stats.cached_experts < self.config.expert_cache_size
    }

    /// Get recommendations for memory optimization
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.stats.fragmentation_ratio > 0.4 {
            recommendations.push("Consider running memory compaction".to_string());
        }

        if self.get_memory_utilization() > 0.9 {
            recommendations.push("Memory usage is high, consider evicting unused experts".to_string());
        }

        if self.stats.cached_experts >= self.config.expert_cache_size {
            recommendations.push("Expert cache is full, oldest entries will be evicted".to_string());
        }

        let time_since_gc = Utils::now() - self.stats.last_gc_time;
        if time_since_gc > 600_000.0 { // 10 minutes
            recommendations.push("Consider running garbage collection".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Memory pool is operating efficiently".to_string());
        }

        recommendations
    }
}

/// Memory pool factory for different configurations
pub struct MemoryPoolFactory;

impl MemoryPoolFactory {
    /// Create a memory pool optimized for browser environments
    pub fn create_browser_pool() -> Result<WasmMemoryPool, JsValue> {
        let config = MemoryPoolConfig {
            max_memory_bytes: 64 * 1024 * 1024, // 64MB for browsers
            max_active_experts: 10,
            expert_cache_size: 25,
            enable_compaction: true,
            gc_threshold: 0.7,
            memory_alignment: 16, // Reduced alignment for browsers
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        WasmMemoryPool::new(&config_json)
    }

    /// Create a memory pool optimized for Node.js environments
    pub fn create_nodejs_pool() -> Result<WasmMemoryPool, JsValue> {
        let config = MemoryPoolConfig {
            max_memory_bytes: 256 * 1024 * 1024, // 256MB for Node.js
            max_active_experts: 50,
            expert_cache_size: 100,
            enable_compaction: true,
            gc_threshold: 0.8,
            memory_alignment: 32, // Full SIMD alignment
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        WasmMemoryPool::new(&config_json)
    }

    /// Create a memory pool optimized for edge/embedded environments
    pub fn create_edge_pool() -> Result<WasmMemoryPool, JsValue> {
        let config = MemoryPoolConfig {
            max_memory_bytes: 16 * 1024 * 1024, // 16MB for edge devices
            max_active_experts: 3,
            expert_cache_size: 5,
            enable_compaction: true,
            gc_threshold: 0.6,
            memory_alignment: 8, // Minimal alignment
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        WasmMemoryPool::new(&config_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let pool = WasmMemoryPool::new(&config_json).unwrap();
        
        assert!(pool.is_healthy());
        assert_eq!(pool.stats.memory_in_use, 0);
    }

    #[test]
    fn test_memory_block_alignment() {
        let block = MemoryBlock::new(1, 1000, 32);
        assert!(block.simd_aligned);
        assert_eq!(block.size, 1000);
        assert_eq!(block.ref_count, 1);
    }

    #[test]
    fn test_memory_pool_factory() {
        let browser_pool = MemoryPoolFactory::create_browser_pool().unwrap();
        let nodejs_pool = MemoryPoolFactory::create_nodejs_pool().unwrap();
        let edge_pool = MemoryPoolFactory::create_edge_pool().unwrap();

        // Verify different configurations
        assert!(browser_pool.config.max_memory_bytes < nodejs_pool.config.max_memory_bytes);
        assert!(edge_pool.config.max_memory_bytes < browser_pool.config.max_memory_bytes);
    }

    #[test]
    fn test_garbage_collection_timing() {
        let config = MemoryPoolConfig {
            gc_threshold: 0.5,
            ..Default::default()
        };
        let config_json = serde_json::to_string(&config).unwrap();
        let pool = WasmMemoryPool::new(&config_json).unwrap();

        // Should not trigger GC when memory usage is low
        assert!(!pool.should_trigger_gc());
    }
}