//! # MLP-Based Attention Mechanisms
//!
//! This module implements attention mechanisms using MLP networks to simulate
//! the query-key-value attention patterns, since ruv-FANN doesn't have native
//! attention support.
//!
//! ## Key Innovation: Attention Without Native Support
//!
//! Traditional attention computes:
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
//! ```
//!
//! Our MLP-based approach:
//! 1. **Query/Key/Value Projections**: Separate MLPs for Q, K, V transformations
//! 2. **Attention Scoring**: MLP that takes concatenated [Q, K] and outputs scalar scores
//! 3. **Multi-Head**: Parallel MLPs for each attention head
//! 4. **Output Projection**: Final MLP to combine multi-head outputs
//!
//! ## Performance Considerations
//!
//! - **Memory Efficient**: No need to store large attention matrices
//! - **Parallelizable**: Each head can be computed independently
//! - **Flexible**: Can learn non-linear attention patterns beyond dot-product

use crate::{ModelError, ActivationFunction, tensor_ops};
use num_traits::Float;
use std::fmt::Debug;

/// Configuration for attention mechanisms
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttentionConfig {
    /// Model dimension
    pub d_model: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Dimension of keys
    pub d_k: usize,
    
    /// Dimension of values
    pub d_v: usize,
    
    /// Dropout rate
    pub dropout: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            num_heads: 8,
            d_k: 32,
            d_v: 32,
            dropout: 0.1,
        }
    }
}

/// Single attention head implemented using MLPs
pub struct AttentionHead<T: Float> {
    /// MLP for query projection
    query_network: ruv_fann::Network<T>,
    
    /// MLP for key projection
    key_network: ruv_fann::Network<T>,
    
    /// MLP for value projection
    value_network: ruv_fann::Network<T>,
    
    /// MLP for attention score computation
    attention_scorer: ruv_fann::Network<T>,
    
    d_k: usize,
    d_v: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> AttentionHead<T> {
    /// Create a new attention head
    pub fn new(d_model: usize, d_k: usize, d_v: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        // Query projection: input -> d_k dimensions
        let query_network = NetworkBuilder::new()
            .input_layer(d_model)
            .hidden_layer(d_k * 2)
            .output_layer_with_activation(d_k, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create query network: {}", e) })?;
        
        // Key projection: input -> d_k dimensions
        let key_network = NetworkBuilder::new()
            .input_layer(d_model)
            .hidden_layer(d_k * 2)
            .output_layer_with_activation(d_k, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create key network: {}", e) })?;
        
        // Value projection: input -> d_v dimensions
        let value_network = NetworkBuilder::new()
            .input_layer(d_model)
            .hidden_layer(d_v * 2)
            .output_layer_with_activation(d_v, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create value network: {}", e) })?;
        
        // Attention scorer: takes concatenated [query, key] -> scalar score
        let attention_scorer = NetworkBuilder::new()
            .input_layer(d_k + d_k) // Concatenated query and key
            .hidden_layer(d_k)
            .hidden_layer(d_k / 2)
            .output_layer_with_activation(1, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create attention scorer: {}", e) })?;
        
        Ok(Self {
            query_network,
            key_network,
            value_network,
            attention_scorer,
            d_k,
            d_v,
        })
    }
    
    /// Compute attention for a single head
    pub fn forward(&mut self, queries: &[Vec<T>], keys: &[Vec<T>], values: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        let seq_len = queries.len();
        
        // Project inputs to Q, K, V spaces
        let mut q_projected = Vec::with_capacity(seq_len);
        let mut k_projected = Vec::with_capacity(seq_len);
        let mut v_projected = Vec::with_capacity(seq_len);
        
        for (i, (query, key, value)) in queries.iter().zip(keys.iter()).zip(values.iter()).enumerate() {
            q_projected.push(self.query_network.run(query));
            k_projected.push(self.key_network.run(key));
            v_projected.push(self.value_network.run(value));
        }
        
        // Compute attention scores and apply attention
        let mut attention_output = Vec::with_capacity(seq_len);
        
        for i in 0..seq_len {
            let mut attention_scores = Vec::with_capacity(seq_len);
            
            // Compute attention scores for position i against all positions
            for j in 0..seq_len {
                let mut combined_input = q_projected[i].clone();
                combined_input.extend_from_slice(&k_projected[j]);
                
                let score = self.attention_scorer.run(&combined_input);
                attention_scores.push(score[0]); // Single output score
            }
            
            // Apply softmax to attention scores
            let attention_weights = tensor_ops::softmax(&attention_scores);
            
            // Compute weighted sum of values
            let mut attended_value = vec![T::zero(); self.d_v];
            for (weight, value) in attention_weights.iter().zip(v_projected.iter()) {
                for k in 0..self.d_v {
                    attended_value[k] = attended_value[k] + *weight * value[k];
                }
            }
            
            attention_output.push(attended_value);
        }
        
        Ok(attention_output)
    }
    
    /// Get the output dimension of this attention head
    pub fn output_dim(&self) -> usize {
        self.d_v
    }
}

/// Multi-head attention mechanism using multiple MLPs
pub struct MultiHeadAttention<T: Float> {
    /// Individual attention heads
    heads: Vec<AttentionHead<T>>,
    
    /// Output projection network
    output_projection: ruv_fann::Network<T>,
    
    config: AttentionConfig,
}

impl<T: Float + Debug + Clone + Send + Sync> MultiHeadAttention<T> {
    /// Create a new multi-head attention mechanism
    pub fn new(config: AttentionConfig) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        // Create individual attention heads
        let mut heads = Vec::with_capacity(config.num_heads);
        for _ in 0..config.num_heads {
            heads.push(AttentionHead::new(config.d_model, config.d_k, config.d_v)?);
        }
        
        // Output projection: concatenated heads -> d_model
        let output_projection = NetworkBuilder::new()
            .input_layer(config.num_heads * config.d_v)
            .hidden_layer(config.d_model)
            .output_layer_with_activation(config.d_model, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create output projection: {}", e) })?;
        
        Ok(Self {
            heads,
            output_projection,
            config,
        })
    }
    
    /// Forward pass through multi-head attention
    pub fn forward(&mut self, queries: &[Vec<T>], keys: &[Vec<T>], values: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        let seq_len = queries.len();
        
        // Compute attention for each head in parallel (conceptually)
        let mut head_outputs = Vec::with_capacity(self.config.num_heads);
        for head in &mut self.heads {
            head_outputs.push(head.forward(queries, keys, values)?);
        }
        
        // Concatenate head outputs and apply output projection
        let mut final_output = Vec::with_capacity(seq_len);
        
        for i in 0..seq_len {
            // Concatenate outputs from all heads for position i
            let mut concatenated = Vec::new();
            for head_output in &head_outputs {
                concatenated.extend_from_slice(&head_output[i]);
            }
            
            // Apply output projection
            let projected = self.output_projection.run(&concatenated);
            final_output.push(projected);
        }
        
        Ok(final_output)
    }
    
    /// Get configuration
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }
    
    /// Create self-attention (queries, keys, values are the same)
    pub fn self_attention(&mut self, input: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        self.forward(input, input, input)
    }
    
    /// Create cross-attention (queries from one sequence, keys/values from another)
    pub fn cross_attention(&mut self, queries: &[Vec<T>], context: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        self.forward(queries, context, context)
    }
}

/// Causal (masked) attention for decoder layers
pub struct CausalAttention<T: Float> {
    attention: MultiHeadAttention<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> CausalAttention<T> {
    /// Create causal attention mechanism
    pub fn new(config: AttentionConfig) -> Result<Self, ModelError> {
        Ok(Self {
            attention: MultiHeadAttention::new(config)?,
        })
    }
    
    /// Forward pass with causal masking
    pub fn forward(&mut self, queries: &[Vec<T>], keys: &[Vec<T>], values: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        // For now, we implement this the same as regular attention
        // In a full implementation, we would modify the attention heads to apply causal masking
        self.attention.forward(queries, keys, values)
    }
}

/// Efficient attention approximation for long sequences (used in Informer)
pub struct SparseAttention<T: Float> {
    attention: MultiHeadAttention<T>,
    sparsity_factor: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> SparseAttention<T> {
    /// Create sparse attention mechanism
    pub fn new(config: AttentionConfig, sparsity_factor: usize) -> Result<Self, ModelError> {
        Ok(Self {
            attention: MultiHeadAttention::new(config)?,
            sparsity_factor,
        })
    }
    
    /// Forward pass with sparse sampling
    pub fn forward(&mut self, queries: &[Vec<T>], keys: &[Vec<T>], values: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        // Sample indices for sparse attention
        let seq_len = queries.len();
        let step = (seq_len / self.sparsity_factor).max(1);
        
        let mut sparse_queries = Vec::new();
        let mut sparse_keys = Vec::new();
        let mut sparse_values = Vec::new();
        
        for i in (0..seq_len).step_by(step) {
            sparse_queries.push(queries[i].clone());
            sparse_keys.push(keys[i].clone());
            sparse_values.push(values[i].clone());
        }
        
        // Apply regular attention to sparse sequence
        let sparse_output = self.attention.forward(&sparse_queries, &sparse_keys, &sparse_values)?;
        
        // Interpolate back to full sequence length
        self.interpolate_output(&sparse_output, seq_len)
    }
    
    /// Interpolate sparse output back to full sequence length
    fn interpolate_output(&self, sparse_output: &[Vec<T>], target_len: usize) -> Result<Vec<Vec<T>>, ModelError> {
        if sparse_output.is_empty() {
            return Err(ModelError::PredictionError { message: "Empty sparse output".to_string() });
        }
        
        let d_model = sparse_output[0].len();
        let mut interpolated = Vec::with_capacity(target_len);
        
        let scale = sparse_output.len() as f64 / target_len as f64;
        
        for i in 0..target_len {
            let sparse_index = (i as f64 * scale) as usize;
            let clamped_index = sparse_index.min(sparse_output.len() - 1);
            interpolated.push(sparse_output[clamped_index].clone());
        }
        
        Ok(interpolated)
    }
}

/// Auto-correlation attention mechanism for AutoFormer
pub struct AutoCorrelationAttention<T: Float> {
    value_network: ruv_fann::Network<T>,
    period_detector: ruv_fann::Network<T>,
    d_model: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> AutoCorrelationAttention<T> {
    /// Create auto-correlation attention
    pub fn new(d_model: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        let value_network = NetworkBuilder::new()
            .input_layer(d_model)
            .hidden_layer(d_model)
            .output_layer_with_activation(d_model, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create value network: {}", e) })?;
        
        let period_detector = NetworkBuilder::new()
            .input_layer(d_model)
            .hidden_layer(d_model / 2)
            .output_layer_with_activation(1, ruv_fann::activation::ActivationFunction::Sigmoid, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create period detector: {}", e) })?;
        
        Ok(Self {
            value_network,
            period_detector,
            d_model,
        })
    }
    
    /// Forward pass using auto-correlation
    pub fn forward(&mut self, input: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        let seq_len = input.len();
        let mut output = Vec::with_capacity(seq_len);
        
        // Project values
        let mut values = Vec::with_capacity(seq_len);
        for inp in input {
            values.push(self.value_network.run(inp));
        }
        
        // Detect periods and apply auto-correlation
        for i in 0..seq_len {
            let period_score = self.period_detector.run(&input[i])[0];
            
            // Simple auto-correlation: weighted average of nearby values
            let mut correlated_value = vec![T::zero(); self.d_model];
            let mut total_weight = T::zero();
            
            for j in 0..seq_len {
                let distance = if i >= j { i - j } else { j - i };
                let weight = period_score * T::from((-distance as f64 / 10.0).exp()).unwrap();
                total_weight = total_weight + weight;
                
                for k in 0..self.d_model {
                    correlated_value[k] = correlated_value[k] + weight * values[j][k];
                }
            }
            
            // Normalize by total weight
            if total_weight > T::zero() {
                for k in 0..self.d_model {
                    correlated_value[k] = correlated_value[k] / total_weight;
                }
            }
            
            output.push(correlated_value);
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_head_creation() {
        let head = AttentionHead::<f32>::new(64, 16, 16).unwrap();
        assert_eq!(head.d_k, 16);
        assert_eq!(head.d_v, 16);
        assert_eq!(head.output_dim(), 16);
    }
    
    #[test]
    fn test_multi_head_attention_creation() {
        let config = AttentionConfig {
            d_model: 64,
            num_heads: 4,
            d_k: 16,
            d_v: 16,
            dropout: 0.1,
        };
        
        let attention = MultiHeadAttention::<f32>::new(config.clone()).unwrap();
        assert_eq!(attention.config.num_heads, 4);
        assert_eq!(attention.config.d_model, 64);
    }
    
    #[test]
    fn test_sparse_attention_creation() {
        let config = AttentionConfig::default();
        let sparse_attention = SparseAttention::<f32>::new(config, 4).unwrap();
        assert_eq!(sparse_attention.sparsity_factor, 4);
    }
    
    #[test]
    fn test_auto_correlation_attention() {
        let auto_corr = AutoCorrelationAttention::<f32>::new(64).unwrap();
        assert_eq!(auto_corr.d_model, 64);
    }
}