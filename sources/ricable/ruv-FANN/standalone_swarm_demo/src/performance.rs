//! Performance Optimization Module
//! 
//! This module provides performance monitoring, optimization, and memory management utilities.

use std::time::Instant;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Performance metrics collector
#[derive(Debug)]
pub struct PerformanceMetrics {
    pub memory_usage: usize,
    pub cpu_usage: f32,
    pub execution_times: HashMap<String, f64>,
    pub memory_allocations: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub cache_misses: AtomicUsize,
}

impl Clone for PerformanceMetrics {
    fn clone(&self) -> Self {
        Self {
            memory_usage: self.memory_usage,
            cpu_usage: self.cpu_usage,
            execution_times: self.execution_times.clone(),
            memory_allocations: AtomicUsize::new(self.memory_allocations.load(Ordering::Relaxed)),
            cache_hits: AtomicUsize::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicUsize::new(self.cache_misses.load(Ordering::Relaxed)),
        }
    }
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            memory_usage: 0,
            cpu_usage: 0.0,
            execution_times: HashMap::new(),
            memory_allocations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }
    
    pub fn record_execution_time(&mut self, operation: &str, duration: f64) {
        self.execution_times.insert(operation.to_string(), duration);
    }
    
    pub fn increment_memory_allocations(&self) {
        self.memory_allocations.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_cache_hit_ratio(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }
}

/// Memory pool for efficient allocation
pub struct MemoryPool<T> {
    pool: Vec<T>,
    in_use: Vec<bool>,
    capacity: usize,
}

impl<T: Clone + Default> MemoryPool<T> {
    pub fn new(capacity: usize) -> Self {
        let pool = (0..capacity).map(|_| T::default()).collect();
        let in_use = vec![false; capacity];
        
        Self {
            pool,
            in_use,
            capacity,
        }
    }
    
    pub fn acquire(&mut self) -> Option<&mut T> {
        for i in 0..self.capacity {
            if !self.in_use[i] {
                self.in_use[i] = true;
                return Some(&mut self.pool[i]);
            }
        }
        None
    }
    
    pub fn release(&mut self, index: usize) -> bool {
        if index < self.capacity && self.in_use[index] {
            self.in_use[index] = false;
            // Reset the item to default state
            self.pool[index] = T::default();
            true
        } else {
            false
        }
    }
    
    pub fn available_count(&self) -> usize {
        self.in_use.iter().filter(|&&used| !used).count()
    }
}

/// Simple LRU cache for computational results
pub struct LRUCache<K, V> {
    cache: HashMap<K, V>,
    access_order: Vec<K>,
    capacity: usize,
}

impl<K: Clone + Eq + std::hash::Hash, V> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            capacity,
        }
    }
    
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(value) = self.cache.get(key) {
            // Move to end (most recently used)
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                let key = self.access_order.remove(pos);
                self.access_order.push(key);
            }
            Some(value)
        } else {
            None
        }
    }
    
    pub fn insert(&mut self, key: K, value: V) {
        if self.cache.len() >= self.capacity {
            // Remove least recently used
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.cache.remove(&lru_key);
                self.access_order.remove(0);
            }
        }
        
        self.cache.insert(key.clone(), value);
        self.access_order.push(key);
    }
}

/// Performance timer for measuring execution time
pub struct PerformanceTimer {
    start: Instant,
    name: String,
}

impl PerformanceTimer {
    pub fn new(name: String) -> Self {
        Self {
            start: Instant::now(),
            name,
        }
    }
    
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    
    pub fn finish(self) -> (String, f64) {
        let elapsed = self.elapsed();
        (self.name, elapsed)
    }
}

/// SIMD optimized vector operations
pub struct VectorOperations;

impl VectorOperations {
    /// Optimized dot product using SIMD when available
    pub fn dot_product_optimized(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        // Use SIMD for larger vectors
        if a.len() >= 8 {
            Self::dot_product_simd(a, b)
        } else {
            Self::dot_product_scalar(a, b)
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        // This would use actual SIMD instructions in a real implementation
        // For now, we'll use the scalar version
        Self::dot_product_scalar(a, b)
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        Self::dot_product_scalar(a, b)
    }
    
    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    /// Optimized vector addition
    pub fn vector_add_optimized(a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.len() != b.len() {
            return Vec::new();
        }
        
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
    
    /// Optimized vector scaling
    pub fn vector_scale_optimized(vector: &mut [f32], scalar: f32) {
        for element in vector.iter_mut() {
            *element *= scalar;
        }
    }
}

/// Performance optimization utilities
pub struct PerformanceOptimizer;

impl PerformanceOptimizer {
    /// Optimize neural network computations
    pub fn optimize_neural_computation(
        weights: &[Vec<f32>],
        inputs: &[f32],
    ) -> Vec<f32> {
        let mut results = Vec::with_capacity(weights.len());
        
        for weight_row in weights {
            let result = VectorOperations::dot_product_optimized(weight_row, inputs);
            results.push(result);
        }
        
        results
    }
    
    /// Batch process multiple inputs efficiently
    pub fn batch_process<F, T, R>(
        inputs: &[T],
        batch_size: usize,
        processor: F,
    ) -> Vec<R>
    where
        F: Fn(&[T]) -> Vec<R>,
        T: Clone,
        R: Clone,
    {
        let mut results = Vec::new();
        
        for chunk in inputs.chunks(batch_size) {
            let batch_results = processor(chunk);
            results.extend(batch_results);
        }
        
        results
    }
    
    /// Memory-efficient matrix multiplication
    pub fn matrix_multiply_efficient(
        a: &[Vec<f32>],
        b: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
            return Vec::new();
        }
        
        let rows_a = a.len();
        let cols_b = b[0].len();
        let mut result = vec![vec![0.0; cols_b]; rows_a];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..a[0].len() {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        result
    }
}

/// System resource monitor
pub struct ResourceMonitor {
    initial_memory: usize,
    peak_memory: usize,
    start_time: Instant,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            initial_memory: Self::get_memory_usage(),
            peak_memory: 0,
            start_time: Instant::now(),
        }
    }
    
    fn get_memory_usage() -> usize {
        // In a real implementation, this would query actual system memory
        // For now, we'll return a placeholder
        0
    }
    
    pub fn update_peak_memory(&mut self) {
        let current_memory = Self::get_memory_usage();
        if current_memory > self.peak_memory {
            self.peak_memory = current_memory;
        }
    }
    
    pub fn get_runtime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    pub fn get_memory_stats(&self) -> (usize, usize, usize) {
        (self.initial_memory, self.peak_memory, Self::get_memory_usage())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        let mut pool: MemoryPool<i32> = MemoryPool::new(5);
        
        // Acquire some items
        let item1 = pool.acquire();
        let item2 = pool.acquire();
        
        assert!(item1.is_some());
        assert!(item2.is_some());
        assert_eq!(pool.available_count(), 3);
        
        // Release an item
        pool.release(0);
        assert_eq!(pool.available_count(), 4);
    }
    
    #[test]
    fn test_lru_cache() {
        let mut cache = LRUCache::new(2);
        
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        
        // This should evict key2 since key1 was just accessed
        cache.insert("key3", "value3");
        
        assert_eq!(cache.get(&"key2"), None);
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        assert_eq!(cache.get(&"key3"), Some(&"value3"));
    }
    
    #[test]
    fn test_vector_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let dot_product = VectorOperations::dot_product_optimized(&a, &b);
        assert_eq!(dot_product, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 70
        
        let sum = VectorOperations::vector_add_optimized(&a, &b);
        assert_eq!(sum, vec![6.0, 8.0, 10.0, 12.0]);
    }
}