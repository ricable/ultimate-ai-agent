use ndarray::{Array2, Array3, Array4, s, Axis};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Sliding window attention mechanism optimized for log sequences
#[derive(Debug)]
pub struct SlidingWindowAttention {
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
    dropout_rate: f32,
    scale: f32,
    kernel: CustomAttentionKernel,
}

#[derive(Debug)]
pub struct CustomAttentionKernel {
    cache: HashMap<String, Array3<f32>>,
    use_flash_attention: bool,
    block_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub dropout_rate: f32,
    pub use_flash_attention: bool,
    pub block_size: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        AttentionConfig {
            num_heads: 8,
            head_dim: 32,
            window_size: 64,
            dropout_rate: 0.1,
            use_flash_attention: true,
            block_size: 16,
        }
    }
}

impl SlidingWindowAttention {
    pub fn new(embedding_dim: usize, num_heads: usize, window_size: usize) -> Self {
        let head_dim = embedding_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        SlidingWindowAttention {
            num_heads,
            head_dim,
            window_size,
            w_q: Array2::from_shape_fn((embedding_dim, embedding_dim), |(_, _)| {
                rand::random::<f32>() * 0.02 - 0.01
            }),
            w_k: Array2::from_shape_fn((embedding_dim, embedding_dim), |(_, _)| {
                rand::random::<f32>() * 0.02 - 0.01
            }),
            w_v: Array2::from_shape_fn((embedding_dim, embedding_dim), |(_, _)| {
                rand::random::<f32>() * 0.02 - 0.01
            }),
            w_o: Array2::from_shape_fn((embedding_dim, embedding_dim), |(_, _)| {
                rand::random::<f32>() * 0.02 - 0.01
            }),
            dropout_rate: 0.1,
            scale,
            kernel: CustomAttentionKernel::new(AttentionConfig::default()),
        }
    }

    pub fn forward(&self, query: &Array3<f32>, key: &Array3<f32>, value: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, embedding_dim) = query.dim();
        
        // Linear projections
        let q = self.project_tensor(query, &self.w_q);
        let k = self.project_tensor(key, &self.w_k);
        let v = self.project_tensor(value, &self.w_v);
        
        // Reshape for multi-head attention
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_len);
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_len);
        let v_heads = self.reshape_for_heads(&v, batch_size, seq_len);
        
        // Apply sliding window attention
        let attention_output = self.sliding_window_attention(&q_heads, &k_heads, &v_heads);
        
        // Reshape back and apply output projection
        let output = self.reshape_from_heads(&attention_output, batch_size, seq_len);
        self.project_tensor(&output, &self.w_o)
    }

    fn project_tensor(&self, input: &Array3<f32>, weight: &Array2<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = input.dim();
        let input_2d = input.to_shape((batch_size * seq_len, input.shape()[2])).unwrap();
        let output_2d = input_2d.dot(weight);
        output_2d.to_shape((batch_size, seq_len, weight.shape()[1])).unwrap().to_owned()
    }

    fn reshape_for_heads(&self, input: &Array3<f32>, batch_size: usize, seq_len: usize) -> Array4<f32> {
        let mut output = Array4::zeros((batch_size, self.num_heads, seq_len, self.head_dim));
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        output[[b, h, s, d]] = input[[b, s, h * self.head_dim + d]];
                    }
                }
            }
        }
        
        output
    }

    fn reshape_from_heads(&self, input: &Array4<f32>, batch_size: usize, seq_len: usize) -> Array3<f32> {
        let mut output = Array3::zeros((batch_size, seq_len, self.num_heads * self.head_dim));
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        output[[b, s, h * self.head_dim + d]] = input[[b, h, s, d]];
                    }
                }
            }
        }
        
        output
    }

    fn sliding_window_attention(&self, query: &Array4<f32>, key: &Array4<f32>, value: &Array4<f32>) -> Array4<f32> {
        let (batch_size, num_heads, seq_len, head_dim) = query.dim();
        let mut output = Array4::zeros((batch_size, num_heads, seq_len, head_dim));
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                let q_head = query.slice(s![b, h, .., ..]);
                let k_head = key.slice(s![b, h, .., ..]);
                let v_head = value.slice(s![b, h, .., ..]);
                
                let attention_head = self.kernel.compute_attention(
                    &q_head.to_owned(),
                    &k_head.to_owned(),
                    &v_head.to_owned(),
                    self.window_size,
                    self.scale,
                );
                
                output.slice_mut(s![b, h, .., ..]).assign(&attention_head);
            }
        }
        
        output
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(&self, query: &Array3<f32>, key: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = query.dim();
        
        let q = self.project_tensor(query, &self.w_q);
        let k = self.project_tensor(key, &self.w_k);
        
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_len);
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_len);
        
        let mut attention_weights = Array3::zeros((batch_size, self.num_heads, seq_len));
        
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let q_head = q_heads.slice(s![b, h, .., ..]);
                let k_head = k_heads.slice(s![b, h, .., ..]);
                
                let weights = self.compute_attention_weights(&q_head.to_owned(), &k_head.to_owned());
                attention_weights.slice_mut(s![b, h, ..]).assign(&weights);
            }
        }
        
        attention_weights
    }

    fn compute_attention_weights(&self, query: &Array2<f32>, key: &Array2<f32>) -> Array1<f32> {
        let seq_len = query.shape()[0];
        let mut weights = Array1::zeros(seq_len);
        
        for i in 0..seq_len {
            let q_i = query.slice(s![i, ..]);
            let mut max_score = f32::NEG_INFINITY;
            
            let start = i.saturating_sub(self.window_size / 2);
            let end = (i + self.window_size / 2 + 1).min(seq_len);
            
            for j in start..end {
                let k_j = key.slice(s![j, ..]);
                let score = q_i.dot(&k_j) * self.scale;
                max_score = max_score.max(score);
            }
            
            weights[i] = max_score;
        }
        
        // Apply softmax
        let weights_exp = weights.mapv(f32::exp);
        let sum_exp = weights_exp.sum();
        weights_exp / sum_exp
    }

    pub fn get_weights(&self) -> Vec<Array2<f32>> {
        vec![
            self.w_q.clone(),
            self.w_k.clone(),
            self.w_v.clone(),
            self.w_o.clone(),
        ]
    }
}

impl CustomAttentionKernel {
    pub fn new(config: AttentionConfig) -> Self {
        CustomAttentionKernel {
            cache: HashMap::new(),
            use_flash_attention: config.use_flash_attention,
            block_size: config.block_size,
        }
    }

    /// Compute attention with optimized kernels
    pub fn compute_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        window_size: usize,
        scale: f32,
    ) -> Array2<f32> {
        if self.use_flash_attention {
            self.flash_attention(query, key, value, window_size, scale)
        } else {
            self.standard_attention(query, key, value, window_size, scale)
        }
    }

    /// Flash attention implementation for memory efficiency
    fn flash_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        window_size: usize,
        scale: f32,
    ) -> Array2<f32> {
        let (seq_len, head_dim) = query.dim();
        let mut output = Array2::zeros((seq_len, head_dim));
        
        // Process in blocks to reduce memory usage
        for block_start in (0..seq_len).step_by(self.block_size) {
            let block_end = (block_start + self.block_size).min(seq_len);
            
            for i in block_start..block_end {
                let q_i = query.slice(s![i, ..]);
                let mut output_i = Array1::zeros(head_dim);
                let mut denominator = 0.0f32;
                
                let start = i.saturating_sub(window_size / 2);
                let end = (i + window_size / 2 + 1).min(seq_len);
                
                // Compute attention for this position
                for j in start..end {
                    let k_j = key.slice(s![j, ..]);
                    let v_j = value.slice(s![j, ..]);
                    
                    let score = q_i.dot(&k_j) * scale;
                    let exp_score = score.exp();
                    
                    output_i = output_i + &(v_j * exp_score);
                    denominator += exp_score;
                }
                
                if denominator > 0.0 {
                    output_i = output_i / denominator;
                }
                
                output.slice_mut(s![i, ..]).assign(&output_i);
            }
        }
        
        output
    }

    /// Standard attention implementation
    fn standard_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        window_size: usize,
        scale: f32,
    ) -> Array2<f32> {
        let (seq_len, head_dim) = query.dim();
        let mut output = Array2::zeros((seq_len, head_dim));
        
        for i in 0..seq_len {
            let q_i = query.slice(s![i, ..]);
            let mut attention_weights = Vec::new();
            let mut values = Vec::new();
            
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);
            
            // Compute attention scores
            for j in start..end {
                let k_j = key.slice(s![j, ..]);
                let v_j = value.slice(s![j, ..]);
                
                let score = q_i.dot(&k_j) * scale;
                attention_weights.push(score);
                values.push(v_j.to_owned());
            }
            
            // Apply softmax
            let max_score = attention_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Vec<f32> = attention_weights.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            
            let normalized_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
            
            // Compute weighted sum
            let mut weighted_value = Array1::zeros(head_dim);
            for (weight, value) in normalized_weights.iter().zip(values.iter()) {
                weighted_value = weighted_value + &(value * *weight);
            }
            
            output.slice_mut(s![i, ..]).assign(&weighted_value);
        }
        
        output
    }

    /// Sparse attention pattern for long sequences
    pub fn sparse_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        pattern: &SparsePattern,
    ) -> Array2<f32> {
        let (seq_len, head_dim) = query.dim();
        let mut output = Array2::zeros((seq_len, head_dim));
        
        for i in 0..seq_len {
            let q_i = query.slice(s![i, ..]);
            let indices = pattern.get_indices(i, seq_len);
            
            let mut attention_weights = Vec::new();
            let mut values = Vec::new();
            
            for &j in &indices {
                let k_j = key.slice(s![j, ..]);
                let v_j = value.slice(s![j, ..]);
                
                let score = q_i.dot(&k_j);
                attention_weights.push(score);
                values.push(v_j.to_owned());
            }
            
            // Apply softmax
            let max_score = attention_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Vec<f32> = attention_weights.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            
            let normalized_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
            
            // Compute weighted sum
            let mut weighted_value = Array1::zeros(head_dim);
            for (weight, value) in normalized_weights.iter().zip(values.iter()) {
                weighted_value = weighted_value + &(value * *weight);
            }
            
            output.slice_mut(s![i, ..]).assign(&weighted_value);
        }
        
        output
    }
}

/// Sparse attention patterns for efficient processing
#[derive(Debug, Clone)]
pub enum SparsePattern {
    Local(usize),           // Local window
    Strided(usize, usize),  // Strided attention
    Random(usize),          // Random sampling
    Fixed(Vec<usize>),      // Fixed pattern
}

impl SparsePattern {
    pub fn get_indices(&self, pos: usize, seq_len: usize) -> Vec<usize> {
        match self {
            SparsePattern::Local(window) => {
                let start = pos.saturating_sub(window / 2);
                let end = (pos + window / 2 + 1).min(seq_len);
                (start..end).collect()
            }
            SparsePattern::Strided(stride, window) => {
                let mut indices = Vec::new();
                let start = pos.saturating_sub(window / 2);
                
                for i in (start..seq_len).step_by(*stride) {
                    if indices.len() >= *window {
                        break;
                    }
                    indices.push(i);
                }
                
                indices
            }
            SparsePattern::Random(count) => {
                let mut indices: Vec<usize> = (0..seq_len).collect();
                
                // Simple random sampling (Fisher-Yates shuffle)
                for i in 0..*count.min(&seq_len) {
                    let j = i + (rand::random::<usize>() % (seq_len - i));
                    indices.swap(i, j);
                }
                
                indices.into_iter().take(*count).collect()
            }
            SparsePattern::Fixed(pattern) => {
                pattern.iter()
                    .map(|&offset| (pos + offset) % seq_len)
                    .collect()
            }
        }
    }
}

/// Log-specific attention patterns
pub struct LogAttentionPattern {
    patterns: HashMap<String, SparsePattern>,
}

impl LogAttentionPattern {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Error logs need wider context
        patterns.insert("ERROR".to_string(), SparsePattern::Local(128));
        
        // Normal logs can use smaller windows
        patterns.insert("INFO".to_string(), SparsePattern::Local(32));
        
        // Debug logs use strided attention
        patterns.insert("DEBUG".to_string(), SparsePattern::Strided(4, 64));
        
        LogAttentionPattern { patterns }
    }

    pub fn get_pattern(&self, log_type: &str) -> &SparsePattern {
        self.patterns.get(log_type).unwrap_or(&SparsePattern::Local(64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_attention() {
        let attention = SlidingWindowAttention::new(256, 8, 32);
        
        let batch_size = 2;
        let seq_len = 64;
        let embedding_dim = 256;
        
        let query = Array3::from_shape_fn((batch_size, seq_len, embedding_dim), |(_, _, _)| {
            rand::random::<f32>()
        });
        let key = query.clone();
        let value = query.clone();
        
        let output = attention.forward(&query, &key, &value);
        
        assert_eq!(output.shape(), &[batch_size, seq_len, embedding_dim]);
    }

    #[test]
    fn test_flash_attention() {
        let config = AttentionConfig::default();
        let kernel = CustomAttentionKernel::new(config);
        
        let seq_len = 32;
        let head_dim = 64;
        
        let query = Array2::from_shape_fn((seq_len, head_dim), |(_, _)| rand::random::<f32>());
        let key = query.clone();
        let value = query.clone();
        
        let output = kernel.compute_attention(&query, &key, &value, 16, 0.125);
        
        assert_eq!(output.shape(), &[seq_len, head_dim]);
    }

    #[test]
    fn test_sparse_patterns() {
        let pattern = SparsePattern::Local(8);
        let indices = pattern.get_indices(10, 20);
        
        assert!(indices.len() <= 8);
        assert!(indices.contains(&10));
    }
}