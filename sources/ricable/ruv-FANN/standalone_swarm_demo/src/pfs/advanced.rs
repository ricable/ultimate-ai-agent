//! Advanced Performance Optimization and Tensor Operations
//! 
//! This module provides high-performance tensor operations, memory management,
//! and SIMD optimizations specifically tailored for the swarm demo.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::sync::Arc;
use rayon::prelude::*;
use wide::f32x8;

/// Advanced tensor with custom memory layout optimizations for swarm operations
#[repr(C, align(64))]
pub struct SwarmTensor {
    data: *mut f32,
    shape: Vec<usize>,
    strides: Vec<usize>,
    layout: Layout,
    aligned: bool,
    agent_id: Option<String>,
    operation_id: Option<String>,
}

impl SwarmTensor {
    /// Create a new tensor with aligned memory for optimal SIMD performance
    /// Enhanced for swarm operations with agent tracking
    pub fn new_aligned(shape: Vec<usize>, agent_id: Option<String>) -> Self {
        let size = shape.iter().product::<usize>();
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), 64).unwrap();
        
        unsafe {
            let data = alloc(layout) as *mut f32;
            if data.is_null() {
                panic!("Failed to allocate aligned memory for agent {:?}", agent_id);
            }
            
            // Initialize to zero
            ptr::write_bytes(data, 0, size);
            
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            
            Self {
                data,
                shape,
                strides,
                layout,
                aligned: true,
                agent_id,
                operation_id: None,
            }
        }
    }
    
    /// Create tensor from CSV data with enhanced validation
    pub fn from_csv_data(csv_data: &[Vec<f32>], agent_id: Option<String>) -> Result<Self, String> {
        if csv_data.is_empty() {
            return Err("Cannot create tensor from empty CSV data".to_string());
        }
        
        let rows = csv_data.len();
        let cols = csv_data[0].len();
        
        // Validate data consistency
        for (i, row) in csv_data.iter().enumerate() {
            if row.len() != cols {
                return Err(format!("Inconsistent row length at row {}: expected {}, got {}", i, cols, row.len()));
            }
        }
        
        let mut tensor = Self::new_aligned(vec![rows, cols], agent_id);
        
        // Fill tensor with CSV data
        for (i, row) in csv_data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!("Invalid value at row {}, col {}: {}", i, j, value));
                }
                tensor.set(&[i, j], value);
            }
        }
        
        Ok(tensor)
    }
    
    /// Set operation ID for tracking
    pub fn set_operation_id(&mut self, operation_id: String) {
        self.operation_id = Some(operation_id);
    }
    
    /// Get element at index with bounds checking
    #[inline]
    pub fn get(&self, indices: &[usize]) -> f32 {
        debug_assert_eq!(indices.len(), self.shape.len());
        debug_assert!(indices.iter().zip(&self.shape).all(|(i, s)| i < s));
        
        let idx = self.compute_index(indices);
        unsafe { *self.data.add(idx) }
    }
    
    /// Set element at index with bounds checking
    #[inline]
    pub fn set(&mut self, indices: &[usize], value: f32) {
        debug_assert_eq!(indices.len(), self.shape.len());
        debug_assert!(indices.iter().zip(&self.shape).all(|(i, s)| i < s));
        
        let idx = self.compute_index(indices);
        unsafe { *self.data.add(idx) = value; }
    }
    
    #[inline]
    fn compute_index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum()
    }
    
    /// Get shape information
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get agent ID
    pub fn agent_id(&self) -> Option<&String> {
        self.agent_id.as_ref()
    }
    
    /// Get operation ID
    pub fn operation_id(&self) -> Option<&String> {
        self.operation_id.as_ref()
    }
    
    /// Get slice of data (unsafe but fast)
    pub unsafe fn as_slice(&self) -> &[f32] {
        let size = self.shape.iter().product();
        std::slice::from_raw_parts(self.data, size)
    }
    
    /// Get mutable slice of data (unsafe but fast)
    pub unsafe fn as_mut_slice(&mut self) -> &mut [f32] {
        let size = self.shape.iter().product();
        std::slice::from_raw_parts_mut(self.data, size)
    }
    
    /// Compute statistical metrics for performance analysis
    pub fn compute_statistics(&self) -> TensorStatistics {
        let data = unsafe { self.as_slice() };
        
        if data.is_empty() {
            return TensorStatistics::default();
        }
        
        let mut sum = 0.0;
        let mut min = data[0];
        let mut max = data[0];
        
        for &value in data {
            sum += value;
            min = min.min(value);
            max = max.max(value);
        }
        
        let mean = sum / data.len() as f32;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();
        
        TensorStatistics {
            mean,
            std_dev,
            min,
            max,
            variance,
            sum,
            count: data.len(),
        }
    }
}

impl Drop for SwarmTensor {
    fn drop(&mut self) {
        if self.aligned {
            unsafe {
                dealloc(self.data as *mut u8, self.layout);
            }
        }
    }
}

unsafe impl Send for SwarmTensor {}
unsafe impl Sync for SwarmTensor {}

impl Clone for SwarmTensor {
    fn clone(&self) -> Self {
        let mut new_tensor = Self::new_aligned(self.shape.clone(), self.agent_id.clone());
        new_tensor.operation_id = self.operation_id.clone();
        
        let size = self.shape.iter().product();
        unsafe {
            ptr::copy_nonoverlapping(self.data, new_tensor.data, size);
        }
        
        new_tensor
    }
}

/// Statistical information about tensor data
#[derive(Debug, Clone, Default)]
pub struct TensorStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub variance: f32,
    pub sum: f32,
    pub count: usize,
}

/// Enhanced SIMD operations for swarm tensors
pub mod simd_ops {
    use super::*;
    
    /// Vectorized addition with performance tracking
    pub fn simd_add_tracked(a: &SwarmTensor, b: &SwarmTensor, result: &mut SwarmTensor, track_perf: bool) -> Result<(), String> {
        if a.shape != b.shape || a.shape != result.shape {
            return Err("Tensor shapes must match for addition".to_string());
        }
        
        let start_time = if track_perf { Some(std::time::Instant::now()) } else { None };
        
        let size = a.shape.iter().product();
        let chunks = size / 8;
        let remainder = size % 8;
        
        unsafe {
            let a_ptr = a.as_slice().as_ptr();
            let b_ptr = b.as_slice().as_ptr();
            let result_ptr = result.as_mut_slice().as_mut_ptr();
            
            // Process 8 elements at a time with SIMD
            for i in 0..chunks {
                let offset = i * 8;
                let a_slice = std::slice::from_raw_parts(a_ptr.add(offset), 8);
                let b_slice = std::slice::from_raw_parts(b_ptr.add(offset), 8);
                
                let a_vec = f32x8::new([
                    a_slice[0], a_slice[1], a_slice[2], a_slice[3],
                    a_slice[4], a_slice[5], a_slice[6], a_slice[7]
                ]);
                let b_vec = f32x8::new([
                    b_slice[0], b_slice[1], b_slice[2], b_slice[3],
                    b_slice[4], b_slice[5], b_slice[6], b_slice[7]
                ]);
                
                let result_vec = a_vec + b_vec;
                let result_array = result_vec.to_array();
                
                let result_slice = std::slice::from_raw_parts_mut(result_ptr.add(offset), 8);
                result_slice.copy_from_slice(&result_array);
            }
            
            // Handle remainder
            for i in (chunks * 8)..size {
                *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
            }
        }
        
        if let Some(start) = start_time {
            let duration = start.elapsed();
            eprintln!("SIMD addition took: {:?} for {} elements", duration, size);
        }
        
        Ok(())
    }
    
    /// Vectorized ReLU activation with enhanced performance
    pub fn simd_relu_enhanced(input: &SwarmTensor, output: &mut SwarmTensor) -> Result<(), String> {
        if input.shape != output.shape {
            return Err("Input and output tensor shapes must match".to_string());
        }
        
        let size = input.shape.iter().product();
        let chunks = size / 8;
        let zeros = f32x8::splat(0.0);
        
        unsafe {
            let input_ptr = input.as_slice().as_ptr();
            let output_ptr = output.as_mut_slice().as_mut_ptr();
            
            for i in 0..chunks {
                let offset = i * 8;
                let input_slice = std::slice::from_raw_parts(input_ptr.add(offset), 8);
                
                let input_vec = f32x8::new([
                    input_slice[0], input_slice[1], input_slice[2], input_slice[3],
                    input_slice[4], input_slice[5], input_slice[6], input_slice[7]
                ]);
                
                let result_vec = input_vec.max(zeros);
                let result_array = result_vec.to_array();
                
                let output_slice = std::slice::from_raw_parts_mut(output_ptr.add(offset), 8);
                output_slice.copy_from_slice(&result_array);
            }
            
            // Handle remainder
            for i in (chunks * 8)..size {
                *output_ptr.add(i) = (*input_ptr.add(i)).max(0.0);
            }
        }
        
        Ok(())
    }
}

/// Enhanced tensor pool for swarm operations
pub struct SwarmTensorPool {
    pools: Vec<Vec<SwarmTensor>>,
    sizes: Vec<(Vec<usize>, String)>, // (shape, agent_id)
    allocation_count: std::sync::atomic::AtomicUsize,
    deallocation_count: std::sync::atomic::AtomicUsize,
}

impl SwarmTensorPool {
    pub fn new() -> Self {
        Self {
            pools: Vec::new(),
            sizes: Vec::new(),
            allocation_count: std::sync::atomic::AtomicUsize::new(0),
            deallocation_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    pub fn get_tensor(&mut self, shape: Vec<usize>, agent_id: String) -> SwarmTensor {
        self.allocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Try to find a matching pool
        for (i, (pool_shape, pool_agent)) in self.sizes.iter().enumerate() {
            if *pool_shape == shape && *pool_agent == agent_id {
                if let Some(mut tensor) = self.pools[i].pop() {
                    // Reset tensor data to zero
                    unsafe {
                        let data = tensor.as_mut_slice();
                        data.fill(0.0);
                    }
                    return tensor;
                }
            }
        }
        
        // Create new tensor if no pooled tensor available
        SwarmTensor::new_aligned(shape, Some(agent_id))
    }
    
    pub fn return_tensor(&mut self, tensor: SwarmTensor) {
        self.deallocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let shape = tensor.shape().to_vec();
        let agent_id = tensor.agent_id().cloned().unwrap_or_else(|| "unknown".to_string());
        
        // Find matching pool
        for (i, (pool_shape, pool_agent)) in self.sizes.iter().enumerate() {
            if *pool_shape == shape && *pool_agent == agent_id {
                self.pools[i].push(tensor);
                return;
            }
        }
        
        // Create new pool if size not found
        self.sizes.push((shape, agent_id));
        let mut new_pool = Vec::new();
        new_pool.push(tensor);
        self.pools.push(new_pool);
    }
    
    pub fn get_stats(&self) -> (usize, usize) {
        (
            self.allocation_count.load(std::sync::atomic::Ordering::Relaxed),
            self.deallocation_count.load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}

/// Parallel batch processor enhanced for swarm operations
pub struct SwarmBatchProcessor {
    batch_size: usize,
    thread_pool: rayon::ThreadPool,
    agent_id: String,
}

impl SwarmBatchProcessor {
    pub fn new(batch_size: usize, num_threads: usize, agent_id: String) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("swarm-batch-{}-{}", agent_id, i))
            .build()
            .unwrap();
        
        Self {
            batch_size,
            thread_pool,
            agent_id,
        }
    }
    
    pub fn process_parallel<F>(&self, data: &SwarmTensor, process_fn: F) -> Vec<SwarmTensor>
    where
        F: Fn(&SwarmTensor) -> SwarmTensor + Send + Sync,
    {
        let num_samples = data.shape()[0];
        let num_batches = (num_samples + self.batch_size - 1) / self.batch_size;
        
        self.thread_pool.install(|| {
            (0..num_batches)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.batch_size;
                    let end = ((i + 1) * self.batch_size).min(num_samples);
                    let batch_size = end - start;
                    
                    // Create batch view
                    let mut batch = SwarmTensor::new_aligned(
                        vec![batch_size, data.shape()[1]], 
                        Some(format!("{}-batch-{}", self.agent_id, i))
                    );
                    
                    unsafe {
                        let batch_ptr = batch.as_mut_slice().as_mut_ptr();
                        let data_ptr = data.as_slice().as_ptr();
                        
                        for j in 0..batch_size {
                            for k in 0..data.shape()[1] {
                                *batch_ptr.add(j * data.shape()[1] + k) = 
                                    *data_ptr.add((start + j) * data.shape()[1] + k);
                            }
                        }
                    }
                    
                    process_fn(&batch)
                })
                .collect()
        })
    }
}

/// Cache-oblivious matrix transpose optimized for swarm tensors
pub fn cache_oblivious_transpose_swarm(input: &SwarmTensor, output: &mut SwarmTensor) -> Result<(), String> {
    if input.shape().len() != 2 || output.shape().len() != 2 {
        return Err("Both tensors must be 2D for transpose".to_string());
    }
    
    if input.shape()[0] != output.shape()[1] || input.shape()[1] != output.shape()[0] {
        return Err("Output tensor dimensions must be transposed from input".to_string());
    }
    
    let m = input.shape()[0];
    let n = input.shape()[1];
    
    transpose_recursive_swarm(input, output, 0, 0, m, n, 0, 0);
    Ok(())
}

fn transpose_recursive_swarm(
    input: &SwarmTensor,
    output: &mut SwarmTensor,
    i0: usize, j0: usize,
    m: usize, n: usize,
    ti0: usize, tj0: usize,
) {
    const THRESHOLD: usize = 64;
    
    if m <= THRESHOLD && n <= THRESHOLD {
        // Base case: direct transpose
        for i in 0..m {
            for j in 0..n {
                let val = input.get(&[i0 + i, j0 + j]);
                output.set(&[ti0 + j, tj0 + i], val);
            }
        }
    } else if m >= n {
        // Divide by rows
        let m1 = m / 2;
        let m2 = m - m1;
        
        transpose_recursive_swarm(input, output, i0, j0, m1, n, ti0, tj0);
        transpose_recursive_swarm(input, output, i0 + m1, j0, m2, n, ti0, tj0 + m1);
    } else {
        // Divide by columns
        let n1 = n / 2;
        let n2 = n - n1;
        
        transpose_recursive_swarm(input, output, i0, j0, m, n1, ti0, tj0);
        transpose_recursive_swarm(input, output, i0, j0 + n1, m, n2, ti0 + n1, tj0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_swarm_tensor_creation() {
        let tensor = SwarmTensor::new_aligned(vec![2, 3], Some("test-agent".to_string()));
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.aligned);
        assert_eq!(tensor.agent_id(), Some(&"test-agent".to_string()));
    }
    
    #[test]
    fn test_csv_data_loading() {
        let csv_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let tensor = SwarmTensor::from_csv_data(&csv_data, Some("csv-agent".to_string())).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]), 1.0);
        assert_eq!(tensor.get(&[1, 2]), 6.0);
    }
    
    #[test]
    fn test_simd_operations() {
        let mut a = SwarmTensor::new_aligned(vec![1000], Some("simd-a".to_string()));
        let mut b = SwarmTensor::new_aligned(vec![1000], Some("simd-b".to_string()));
        let mut result = SwarmTensor::new_aligned(vec![1000], Some("simd-result".to_string()));
        
        // Initialize with test data
        for i in 0..1000 {
            a.set(&[i], i as f32);
            b.set(&[i], (i + 1) as f32);
        }
        
        simd_ops::simd_add_tracked(&a, &b, &mut result, false).unwrap();
        
        // Check results
        for i in 0..1000 {
            assert_eq!(result.get(&[i]), (i + i + 1) as f32);
        }
    }
    
    #[test]
    fn test_tensor_statistics() {
        let mut tensor = SwarmTensor::new_aligned(vec![3], Some("stats-test".to_string()));
        tensor.set(&[0], 1.0);
        tensor.set(&[1], 2.0);
        tensor.set(&[2], 3.0);
        
        let stats = tensor.compute_statistics();
        assert_eq!(stats.mean, 2.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.sum, 6.0);
        assert_eq!(stats.count, 3);
    }
    
    #[test]
    fn test_tensor_pool() {
        let mut pool = SwarmTensorPool::new();
        
        let tensor1 = pool.get_tensor(vec![10, 10], "pool-agent".to_string());
        let tensor2 = pool.get_tensor(vec![10, 10], "pool-agent".to_string());
        
        pool.return_tensor(tensor1);
        pool.return_tensor(tensor2);
        
        let tensor3 = pool.get_tensor(vec![10, 10], "pool-agent".to_string());
        assert_eq!(tensor3.shape(), &[10, 10]);
        
        let (allocs, deallocs) = pool.get_stats();
        assert_eq!(allocs, 3);
        assert_eq!(deallocs, 2);
    }
}