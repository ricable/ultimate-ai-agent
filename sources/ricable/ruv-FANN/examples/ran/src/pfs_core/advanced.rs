use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::sync::Arc;
use rayon::prelude::*;
use super::{Tensor, TensorOps};

/// Advanced tensor with custom memory layout optimizations
#[repr(C, align(64))]
pub struct AdvancedTensor {
    data: *mut f32,
    shape: Vec<usize>,
    strides: Vec<usize>,
    layout: Layout,
    aligned: bool,
}

impl AdvancedTensor {
    /// Create a new tensor with aligned memory for optimal SIMD performance
    pub fn new_aligned(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), 64).unwrap();
        
        unsafe {
            let data = alloc(layout) as *mut f32;
            if data.is_null() {
                panic!("Failed to allocate aligned memory");
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
            }
        }
    }
    
    /// Get raw pointer to data (unsafe)
    pub unsafe fn as_ptr(&self) -> *const f32 {
        self.data
    }
    
    /// Get mutable raw pointer to data (unsafe)
    pub unsafe fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data
    }
    
    /// Get element at index (bounds checked in debug mode)
    #[inline]
    pub fn get(&self, indices: &[usize]) -> f32 {
        debug_assert_eq!(indices.len(), self.shape.len());
        debug_assert!(indices.iter().zip(&self.shape).all(|(i, s)| i < s));
        
        let idx = self.compute_index(indices);
        unsafe { *self.data.add(idx) }
    }
    
    /// Set element at index (bounds checked in debug mode)
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
}

impl Drop for AdvancedTensor {
    fn drop(&mut self) {
        if self.aligned {
            unsafe {
                dealloc(self.data as *mut u8, self.layout);
            }
        }
    }
}

unsafe impl Send for AdvancedTensor {}
unsafe impl Sync for AdvancedTensor {}

/// Cache-friendly matrix multiplication with blocking
pub fn blocked_matmul(a: &AdvancedTensor, b: &AdvancedTensor, c: &mut AdvancedTensor, block_size: usize) {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(c.shape.len(), 2);
    assert_eq!(a.shape[1], b.shape[0]);
    assert_eq!(c.shape[0], a.shape[0]);
    assert_eq!(c.shape[1], b.shape[1]);
    
    let m = a.shape[0];
    let n = b.shape[1];
    let k = a.shape[1];
    
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();
        
        // Block-wise multiplication for cache efficiency
        for i_block in (0..m).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                for k_block in (0..k).step_by(block_size) {
                    let i_end = (i_block + block_size).min(m);
                    let j_end = (j_block + block_size).min(n);
                    let k_end = (k_block + block_size).min(k);
                    
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k_idx in k_block..k_end {
                                sum += *a_ptr.add(i * k + k_idx) * *b_ptr.add(k_idx * n + j);
                            }
                            *c_ptr.add(i * n + j) += sum;
                        }
                    }
                }
            }
        }
    }
}

/// SIMD-optimized vector operations
pub mod simd_ops {
    use super::*;
    // use packed_simd_2::*;  // Replaced with wide crate
use wide::f32x8;
    
    /// Vectorized addition with explicit SIMD
    pub fn simd_add(a: &AdvancedTensor, b: &AdvancedTensor, result: &mut AdvancedTensor) {
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.shape, result.shape);
        
        let size = a.shape.iter().product();
        let chunks = size / 8;
        let remainder = size % 8;
        
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let result_ptr = result.as_mut_ptr();
            
            // Process 8 elements at a time with SIMD
            for i in 0..chunks {
                let offset = i * 8;
                let a_slice = std::slice::from_raw_parts(a_ptr.add(offset), 8);
                let b_slice = std::slice::from_raw_parts(b_ptr.add(offset), 8);
                let a_array: [f32; 8] = [a_slice[0], a_slice[1], a_slice[2], a_slice[3], a_slice[4], a_slice[5], a_slice[6], a_slice[7]];
                let b_array: [f32; 8] = [b_slice[0], b_slice[1], b_slice[2], b_slice[3], b_slice[4], b_slice[5], b_slice[6], b_slice[7]];
                let a_vec = f32x8::new(a_array);
                let b_vec = f32x8::new(b_array);
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
    }
    
    /// Vectorized multiplication with explicit SIMD
    pub fn simd_mul(a: &AdvancedTensor, b: &AdvancedTensor, result: &mut AdvancedTensor) {
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.shape, result.shape);
        
        let size = a.shape.iter().product();
        let chunks = size / 8;
        
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let result_ptr = result.as_mut_ptr();
            
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
                let result_vec = a_vec * b_vec;
                let result_array = result_vec.to_array();
                let result_slice = std::slice::from_raw_parts_mut(result_ptr.add(offset), 8);
                result_slice.copy_from_slice(&result_array);
            }
            
            // Handle remainder
            for i in (chunks * 8)..size {
                *result_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
            }
        }
    }
    
    /// Vectorized ReLU activation
    pub fn simd_relu(input: &AdvancedTensor, output: &mut AdvancedTensor) {
        assert_eq!(input.shape, output.shape);
        
        let size = input.shape.iter().product();
        let chunks = size / 8;
        let zeros = f32x8::splat(0.0);
        
        unsafe {
            let input_ptr = input.as_ptr();
            let output_ptr = output.as_mut_ptr();
            
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
    }
}

/// Memory pool for efficient tensor allocation
pub struct TensorPool {
    pools: Vec<Vec<AdvancedTensor>>,
    sizes: Vec<usize>,
}

impl TensorPool {
    pub fn new() -> Self {
        Self {
            pools: Vec::new(),
            sizes: Vec::new(),
        }
    }
    
    pub fn with_sizes(sizes: Vec<usize>) -> Self {
        let mut pools = Vec::new();
        for _ in &sizes {
            pools.push(Vec::new());
        }
        
        Self { pools, sizes }
    }
    
    pub fn get_tensor(&mut self, shape: Vec<usize>) -> AdvancedTensor {
        let size: usize = shape.iter().product();
        
        // Try to find a matching pool
        for (i, pool_size) in self.sizes.iter().enumerate() {
            if *pool_size == size {
                if let Some(mut tensor) = self.pools[i].pop() {
                    tensor.shape = shape;
                    return tensor;
                }
            }
        }
        
        // Create new tensor if no pooled tensor available
        AdvancedTensor::new_aligned(shape)
    }
    
    pub fn return_tensor(&mut self, tensor: AdvancedTensor) {
        let size = tensor.shape.iter().product();
        
        // Find matching pool
        for (i, pool_size) in self.sizes.iter().enumerate() {
            if *pool_size == size {
                self.pools[i].push(tensor);
                return;
            }
        }
        
        // Create new pool if size not found
        self.sizes.push(size);
        let mut new_pool = Vec::new();
        new_pool.push(tensor);
        self.pools.push(new_pool);
    }
}

/// Parallel batch processor with work-stealing
pub struct ParallelBatchProcessor {
    batch_size: usize,
    thread_pool: rayon::ThreadPool,
}

impl ParallelBatchProcessor {
    pub fn new(batch_size: usize, num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        
        Self {
            batch_size,
            thread_pool,
        }
    }
    
    pub fn process_parallel<F>(&self, data: &AdvancedTensor, process_fn: F) -> Vec<AdvancedTensor>
    where
        F: Fn(&AdvancedTensor) -> AdvancedTensor + Send + Sync,
    {
        let num_samples = data.shape[0];
        let num_batches = (num_samples + self.batch_size - 1) / self.batch_size;
        
        self.thread_pool.install(|| {
            (0..num_batches)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.batch_size;
                    let end = ((i + 1) * self.batch_size).min(num_samples);
                    let batch_size = end - start;
                    
                    // Create batch view
                    let mut batch = AdvancedTensor::new_aligned(vec![batch_size, data.shape[1]]);
                    
                    unsafe {
                        let batch_ptr = batch.as_mut_ptr();
                        let data_ptr = data.as_ptr();
                        
                        for j in 0..batch_size {
                            for k in 0..data.shape[1] {
                                *batch_ptr.add(j * data.shape[1] + k) = 
                                    *data_ptr.add((start + j) * data.shape[1] + k);
                            }
                        }
                    }
                    
                    process_fn(&batch)
                })
                .collect()
        })
    }
}

/// Cache-oblivious matrix transpose
pub fn cache_oblivious_transpose(input: &AdvancedTensor, output: &mut AdvancedTensor) {
    assert_eq!(input.shape.len(), 2);
    assert_eq!(output.shape.len(), 2);
    assert_eq!(input.shape[0], output.shape[1]);
    assert_eq!(input.shape[1], output.shape[0]);
    
    let m = input.shape[0];
    let n = input.shape[1];
    
    transpose_recursive(input, output, 0, 0, m, n, 0, 0);
}

fn transpose_recursive(
    input: &AdvancedTensor,
    output: &mut AdvancedTensor,
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
        
        transpose_recursive(input, output, i0, j0, m1, n, ti0, tj0);
        transpose_recursive(input, output, i0 + m1, j0, m2, n, ti0, tj0 + m1);
    } else {
        // Divide by columns
        let n1 = n / 2;
        let n2 = n - n1;
        
        transpose_recursive(input, output, i0, j0, m, n1, ti0, tj0);
        transpose_recursive(input, output, i0, j0 + n1, m, n2, ti0 + n1, tj0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_advanced_tensor_creation() {
        let tensor = AdvancedTensor::new_aligned(vec![2, 3]);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert!(tensor.aligned);
    }
    
    #[test]
    fn test_simd_operations() {
        let mut a = AdvancedTensor::new_aligned(vec![1000]);
        let mut b = AdvancedTensor::new_aligned(vec![1000]);
        let mut result = AdvancedTensor::new_aligned(vec![1000]);
        
        // Initialize with test data
        for i in 0..1000 {
            a.set(&[i], i as f32);
            b.set(&[i], (i + 1) as f32);
        }
        
        simd_ops::simd_add(&a, &b, &mut result);
        
        // Check results
        for i in 0..1000 {
            assert_eq!(result.get(&[i]), (i + i + 1) as f32);
        }
    }
    
    #[test]
    fn test_tensor_pool() {
        let mut pool = TensorPool::new();
        
        let tensor1 = pool.get_tensor(vec![10, 10]);
        let tensor2 = pool.get_tensor(vec![10, 10]);
        
        pool.return_tensor(tensor1);
        pool.return_tensor(tensor2);
        
        let tensor3 = pool.get_tensor(vec![10, 10]);
        assert_eq!(tensor3.shape, vec![10, 10]);
    }
}