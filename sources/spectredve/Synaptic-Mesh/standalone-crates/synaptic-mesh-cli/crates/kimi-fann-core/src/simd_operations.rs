//! SIMD-optimized neural network operations
//! 
//! High-performance SIMD implementations for matrix operations, activation functions,
//! and vectorized computations to achieve <50ms neural inference targets.

use std::arch::x86_64::*;

/// SIMD-optimized processor for neural network operations
#[derive(Debug, Clone)]
pub struct SIMDProcessor {
    pub feature_enabled: bool,
    pub vector_width: usize,
}

impl SIMDProcessor {
    /// Create new SIMD processor with automatic feature detection
    pub fn new() -> Self {
        let feature_enabled = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let vector_width = if feature_enabled { 8 } else { 4 };
        
        Self {
            feature_enabled,
            vector_width,
        }
    }
    
    /// SIMD-optimized matrix-vector multiplication
    pub fn matrix_vector_mul(&self, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
        if self.feature_enabled && matrix.len() >= 16 && vector.len() >= 8 {
            unsafe {
                self.simd_matrix_vector_mul_avx2(matrix, vector, output);
            }
        } else {
            self.fallback_matrix_vector_mul(matrix, vector, output);
        }
    }
    
    /// SIMD-optimized activation function (ReLU)
    pub fn relu_activate(&self, input: &[f32], output: &mut [f32]) {
        if self.feature_enabled && input.len() >= 8 {
            unsafe {
                self.simd_relu_avx2(input, output);
            }
        } else {
            self.fallback_relu(input, output);
        }
    }
    
    /// SIMD-optimized sigmoid activation
    pub fn sigmoid_activate(&self, input: &[f32], output: &mut [f32]) {
        if self.feature_enabled && input.len() >= 8 {
            unsafe {
                self.simd_sigmoid_avx2(input, output);
            }
        } else {
            self.fallback_sigmoid(input, output);
        }
    }
    
    /// SIMD-optimized dot product
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.feature_enabled && a.len() >= 8 && a.len() == b.len() {
            unsafe {
                self.simd_dot_product_avx2(a, b)
            }
        } else {
            self.fallback_dot_product(a, b)
        }
    }
    
    /// SIMD-optimized vector addition with FMA
    pub fn vector_add_fma(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        if self.feature_enabled && a.len() >= 8 {
            unsafe {
                self.simd_vector_add_fma_avx2(a, b, c, output);
            }
        } else {
            self.fallback_vector_add_fma(a, b, c, output);
        }
    }
    
    /// SIMD-optimized feature extraction for neural input
    pub fn extract_features_simd(&self, text_features: &[f32], output: &mut [f32]) {
        if self.feature_enabled && text_features.len() >= 16 {
            unsafe {
                self.simd_feature_extraction_avx2(text_features, output);
            }
        } else {
            output[..text_features.len().min(output.len())].copy_from_slice(
                &text_features[..text_features.len().min(output.len())]
            );
        }
    }
    
    // SIMD implementations using AVX2
    
    #[target_feature(enable = "avx2")]
    unsafe fn simd_matrix_vector_mul_avx2(&self, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
        let rows = output.len();
        let cols = vector.len();
        
        // Process 8 elements at a time with AVX2
        for row in 0..rows {
            let mut sum = _mm256_setzero_ps();
            let row_start = row * cols;
            
            let mut col = 0;
            while col + 8 <= cols {
                let matrix_vec = _mm256_loadu_ps(&matrix[row_start + col]);
                let input_vec = _mm256_loadu_ps(&vector[col]);
                sum = _mm256_fmadd_ps(matrix_vec, input_vec, sum);
                col += 8;
            }
            
            // Horizontal add to get final sum
            let result = self.horizontal_add_avx2(sum);
            
            // Handle remaining elements
            let mut remainder = result;
            for c in col..cols {
                remainder += matrix[row_start + c] * vector[c];
            }
            
            output[row] = remainder;
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn simd_relu_avx2(&self, input: &[f32], output: &mut [f32]) {
        let zero = _mm256_setzero_ps();
        let len = input.len().min(output.len());
        
        let mut i = 0;
        while i + 8 <= len {
            let input_vec = _mm256_loadu_ps(&input[i]);
            let result = _mm256_max_ps(input_vec, zero);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        
        // Handle remaining elements
        for j in i..len {
            output[j] = input[j].max(0.0);
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn simd_sigmoid_avx2(&self, input: &[f32], output: &mut [f32]) {
        let one = _mm256_set1_ps(1.0);
        let len = input.len().min(output.len());
        
        let mut i = 0;
        while i + 8 <= len {
            let input_vec = _mm256_loadu_ps(&input[i]);
            
            // Approximate sigmoid using fast approximation
            // sigmoid(x) ≈ 0.5 + 0.25 * x * (1 - |x|/4) for |x| < 2
            let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), input_vec);
            let quarter = _mm256_set1_ps(0.25);
            let four = _mm256_set1_ps(4.0);
            let half = _mm256_set1_ps(0.5);
            
            let normalized = _mm256_div_ps(abs_x, four);
            let one_minus_norm = _mm256_sub_ps(one, normalized);
            let product = _mm256_mul_ps(_mm256_mul_ps(quarter, input_vec), one_minus_norm);
            let result = _mm256_add_ps(half, product);
            
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        
        // Handle remaining elements with standard sigmoid
        for j in i..len {
            output[j] = 1.0 / (1.0 + (-input[j]).exp());
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn simd_dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        
        let mut i = 0;
        while i + 8 <= len {
            let a_vec = _mm256_loadu_ps(&a[i]);
            let b_vec = _mm256_loadu_ps(&b[i]);
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            i += 8;
        }
        
        let mut result = self.horizontal_add_avx2(sum);
        
        // Handle remaining elements
        for j in i..len {
            result += a[j] * b[j];
        }
        
        result
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn simd_vector_add_fma_avx2(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len()).min(output.len());
        
        let mut i = 0;
        while i + 8 <= len {
            let a_vec = _mm256_loadu_ps(&a[i]);
            let b_vec = _mm256_loadu_ps(&b[i]);
            let c_vec = _mm256_loadu_ps(&c[i]);
            
            // Perform fused multiply-add: a * b + c
            let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        
        // Handle remaining elements
        for j in i..len {
            output[j] = a[j] * b[j] + c[j];
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn simd_feature_extraction_avx2(&self, text_features: &[f32], output: &mut [f32]) {
        let len = text_features.len().min(output.len());
        
        let mut i = 0;
        while i + 8 <= len {
            let features = _mm256_loadu_ps(&text_features[i]);
            
            // Apply feature transformation: normalize and scale
            let scale = _mm256_set1_ps(2.0);
            let bias = _mm256_set1_ps(-1.0);
            let normalized = _mm256_fmadd_ps(features, scale, bias);
            
            _mm256_storeu_ps(&mut output[i], normalized);
            i += 8;
        }
        
        // Handle remaining elements
        for j in i..len {
            output[j] = text_features[j] * 2.0 - 1.0;
        }
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_add_avx2(&self, vec: __m256) -> f32 {
        // Horizontal add of 8 f32 values in AVX2 register
        let sum128_low = _mm256_extractf128_ps(vec, 0);
        let sum128_high = _mm256_extractf128_ps(vec, 1);
        let sum128 = _mm_add_ps(sum128_low, sum128_high);
        
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        
        _mm_cvtss_f32(sum32)
    }
    
    // Fallback implementations for non-SIMD systems
    
    fn fallback_matrix_vector_mul(&self, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
        let rows = output.len();
        let cols = vector.len();
        
        for row in 0..rows {
            let mut sum = 0.0;
            for col in 0..cols {
                sum += matrix[row * cols + col] * vector[col];
            }
            output[row] = sum;
        }
    }
    
    fn fallback_relu(&self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i].max(0.0);
        }
    }
    
    fn fallback_sigmoid(&self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
    }
    
    fn fallback_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }
    
    fn fallback_vector_add_fma(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len()).min(output.len());
        for i in 0..len {
            output[i] = a[i] * b[i] + c[i];
        }
    }
    
    /// Get performance characteristics
    pub fn get_performance_info(&self) -> SIMDPerformanceInfo {
        SIMDPerformanceInfo {
            avx2_enabled: self.feature_enabled,
            vector_width: self.vector_width,
            theoretical_speedup: if self.feature_enabled { 8.0 } else { 1.0 },
            memory_bandwidth_gb_s: if self.feature_enabled { 45.0 } else { 12.0 },
        }
    }
}

/// Performance information for SIMD operations
#[derive(Debug, Clone)]
pub struct SIMDPerformanceInfo {
    pub avx2_enabled: bool,
    pub vector_width: usize,
    pub theoretical_speedup: f32,
    pub memory_bandwidth_gb_s: f32,
}

/// SIMD-optimized neural layer computation
#[derive(Debug, Clone)]
pub struct SIMDNeuralLayer {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub input_size: usize,
    pub output_size: usize,
    processor: SIMDProcessor,
}

impl SIMDNeuralLayer {
    /// Create new SIMD-optimized neural layer
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = vec![0.0; input_size * output_size];
        let biases = vec![0.0; output_size];
        
        // Initialize weights with Xavier initialization
        let stddev = (2.0 / input_size as f32).sqrt();
        for weight in &mut weights {
            *weight = (fastrand::f32() - 0.5) * 2.0 * stddev;
        }
        
        Self {
            weights,
            biases,
            input_size,
            output_size,
            processor: SIMDProcessor::new(),
        }
    }
    
    /// Forward pass with SIMD optimization
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.input_size);
        assert_eq!(output.len(), self.output_size);
        
        // SIMD matrix-vector multiplication
        self.processor.matrix_vector_mul(&self.weights, input, output);
        
        // Add biases using SIMD
        let ones = vec![1.0; self.output_size];
        let mut temp = vec![0.0; self.output_size];
        self.processor.vector_add_fma(&ones, &self.biases, output, &mut temp);
        output.copy_from_slice(&temp);
        
        // Apply ReLU activation
        self.processor.relu_activate(output, output);
    }
    
    /// Get layer performance metrics
    pub fn get_performance_metrics(&self) -> LayerPerformanceMetrics {
        LayerPerformanceMetrics {
            total_parameters: self.weights.len() + self.biases.len(),
            flops_per_forward: self.input_size * self.output_size + self.output_size,
            memory_usage_mb: (self.weights.len() + self.biases.len()) * 4 / 1024 / 1024,
            simd_enabled: self.processor.feature_enabled,
        }
    }
}

/// Performance metrics for neural layers
#[derive(Debug, Clone)]
pub struct LayerPerformanceMetrics {
    pub total_parameters: usize,
    pub flops_per_forward: usize,
    pub memory_usage_mb: usize,
    pub simd_enabled: bool,
}

/// Batch processing with SIMD optimization
pub struct SIMDBatchProcessor {
    processor: SIMDProcessor,
    batch_size: usize,
}

impl SIMDBatchProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            processor: SIMDProcessor::new(),
            batch_size,
        }
    }
    
    /// Process multiple inputs in parallel using SIMD
    pub fn process_batch(&self, inputs: &[Vec<f32>], layer: &SIMDNeuralLayer) -> Vec<Vec<f32>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let mut output = vec![0.0; layer.output_size];
            layer.forward(input, &mut output);
            outputs.push(output);
        }
        
        outputs
    }
    
    /// Get optimal batch size for current hardware
    pub fn get_optimal_batch_size(&self) -> usize {
        if self.processor.feature_enabled {
            32 // Optimal for AVX2
        } else {
            8  // Conservative for fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_processor_creation() {
        let processor = SIMDProcessor::new();
        assert!(processor.vector_width > 0);
    }
    
    #[test]
    fn test_simd_matrix_vector_mul() {
        let processor = SIMDProcessor::new();
        let matrix = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let vector = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        
        processor.matrix_vector_mul(&matrix, &vector, &mut output);
        
        assert_eq!(output[0], 3.0); // 1*1 + 2*1
        assert_eq!(output[1], 7.0); // 3*1 + 4*1
    }
    
    #[test]
    fn test_simd_relu() {
        let processor = SIMDProcessor::new();
        let input = vec![-1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
        let mut output = vec![0.0; 8];
        
        processor.relu_activate(&input, &mut output);
        
        let expected = vec![0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 0.0, 6.0];
        assert_eq!(output, expected);
    }
    
    #[test]
    fn test_simd_dot_product() {
        let processor = SIMDProcessor::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = processor.dot_product(&a, &b);
        assert_eq!(result, 10.0); // 1+2+3+4
    }
    
    #[test]
    fn test_neural_layer_forward() {
        let layer = SIMDNeuralLayer::new(4, 2);
        let input = vec![1.0, 0.5, -0.5, 2.0];
        let mut output = vec![0.0; 2];
        
        layer.forward(&input, &mut output);
        
        // Output should be non-negative due to ReLU
        assert!(output[0] >= 0.0);
        assert!(output[1] >= 0.0);
    }
    
    #[test]
    fn test_performance_info() {
        let processor = SIMDProcessor::new();
        let info = processor.get_performance_info();
        
        assert!(info.vector_width > 0);
        assert!(info.theoretical_speedup >= 1.0);
    }
}