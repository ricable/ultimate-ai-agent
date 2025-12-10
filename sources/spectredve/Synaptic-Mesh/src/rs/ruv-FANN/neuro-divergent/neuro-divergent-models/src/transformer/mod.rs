//! # Transformer Models Module
//!
//! This module contains transformer-based forecasting models implemented using ruv-FANN's
//! neural network foundation. Since ruv-FANN doesn't have native attention mechanisms,
//! all attention operations are approximated using MLP networks.
//!
//! ## Architecture Overview
//!
//! ### MLP-Based Attention Simulation
//! Instead of native attention, we use:
//! - **Query Networks**: MLP that transforms input to query space
//! - **Key Networks**: MLP that transforms input to key space  
//! - **Value Networks**: MLP that transforms input to value space
//! - **Attention Scoring**: MLP that computes attention weights
//! - **Output Projection**: MLP that projects attended values
//!
//! ### Multi-Head Attention
//! Implemented as parallel MLP networks, each representing one attention head.
//! The outputs are concatenated and passed through a final projection layer.
//!
//! ## Model Implementations
//!
//! Each transformer model follows a similar pattern:
//! 1. **Input Processing**: Embedding and positional encoding
//! 2. **Attention Layers**: Multi-head attention using MLPs
//! 3. **Feed-Forward Networks**: Standard MLPs for transformation
//! 4. **Output Generation**: Final layers for forecasting

pub mod attention;
pub mod tft;
pub mod informer;
pub mod autoformer;
pub mod fedformer;
pub mod patchtst;
pub mod itransformer;

// Re-export commonly used components
pub use attention::{AttentionHead, MultiHeadAttention, AttentionConfig};

use crate::{BaseModel, ModelError, ActivationFunction};
use num_traits::Float;
use std::fmt::Debug;

/// Common configuration parameters shared across transformer models
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerCommonConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Feed-forward network hidden dimension
    pub d_ff: usize,
    
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    
    /// Number of decoder layers (if applicable)
    pub num_decoder_layers: usize,
    
    /// Dropout rate
    pub dropout: f64,
    
    /// Activation function
    pub activation: ActivationFunction,
    
    /// Maximum sequence length
    pub max_seq_len: usize,
    
    /// Input dimension
    pub input_dim: usize,
    
    /// Output dimension (forecast horizon)
    pub output_dim: usize,
}

impl Default for TransformerCommonConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            num_heads: 8,
            d_ff: 1024,
            num_encoder_layers: 3,
            num_decoder_layers: 2,
            dropout: 0.1,
            activation: ActivationFunction::ReLU,
            max_seq_len: 512,
            input_dim: 6,
            output_dim: 24,
        }
    }
}

/// Feed-forward network component used in transformer models
pub struct FeedForwardNetwork<T: Float> {
    linear1: ruv_fann::Network<T>,
    linear2: ruv_fann::Network<T>,
    activation: ActivationFunction,
}

impl<T: Float + Debug + Clone + Send + Sync> FeedForwardNetwork<T> {
    /// Create a new feed-forward network
    pub fn new(input_dim: usize, hidden_dim: usize, activation: ActivationFunction) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        let linear1 = NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer_with_activation(hidden_dim, activation.to_ruv_fann_activation(), T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create linear1: {}", e) })?;
            
        let linear2 = NetworkBuilder::new()
            .input_layer(hidden_dim)
            .output_layer_with_activation(input_dim, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create linear2: {}", e) })?;
        
        Ok(Self {
            linear1,
            linear2,
            activation,
        })
    }
    
    /// Forward pass through the feed-forward network
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        let hidden = self.linear1.run(input);
        let output = self.linear2.run(&hidden);
        Ok(output)
    }
}

/// Layer normalization component
pub struct LayerNorm<T: Float> {
    epsilon: T,
    scale: Vec<T>,
    bias: Vec<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> LayerNorm<T> {
    /// Create a new layer normalization layer
    pub fn new(d_model: usize) -> Self {
        Self {
            epsilon: T::from(1e-5).unwrap(),
            scale: vec![T::one(); d_model],
            bias: vec![T::zero(); d_model],
        }
    }
    
    /// Apply layer normalization
    pub fn forward(&self, input: &[T]) -> Vec<T> {
        crate::tensor_ops::layer_norm(input, self.epsilon)
    }
}

/// Residual connection wrapper
pub struct ResidualConnection<T: Float> {
    layer_norm: LayerNorm<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> ResidualConnection<T> {
    /// Create a new residual connection
    pub fn new(d_model: usize) -> Self {
        Self {
            layer_norm: LayerNorm::new(d_model),
        }
    }
    
    /// Apply residual connection with layer normalization
    pub fn forward(&self, input: &[T], sublayer_output: &[T]) -> Vec<T> {
        let normalized = self.layer_norm.forward(sublayer_output);
        input.iter()
            .zip(normalized.iter())
            .map(|(&x, &norm)| x + norm)
            .collect()
    }
}

/// Positional encoding generator
pub struct PositionalEncoding<T: Float> {
    encodings: Vec<Vec<T>>,
    d_model: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> PositionalEncoding<T> {
    /// Create positional encodings for the given parameters
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let encodings = crate::tensor_ops::positional_encoding(max_seq_len, d_model);
        Self { encodings, d_model }
    }
    
    /// Get positional encoding for a specific position
    pub fn get_encoding(&self, position: usize) -> Option<&[T]> {
        self.encodings.get(position).map(|v| v.as_slice())
    }
    
    /// Add positional encoding to input embeddings
    pub fn add_position(&self, embeddings: &[Vec<T>]) -> Vec<Vec<T>> {
        embeddings.iter()
            .enumerate()
            .map(|(pos, emb)| {
                if let Some(pos_enc) = self.get_encoding(pos) {
                    emb.iter()
                        .zip(pos_enc.iter())
                        .map(|(&e, &p)| e + p)
                        .collect()
                } else {
                    emb.clone()
                }
            })
            .collect()
    }
}

/// Input embedding layer for transformers
pub struct InputEmbedding<T: Float> {
    embedding_network: ruv_fann::Network<T>,
    d_model: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> InputEmbedding<T> {
    /// Create a new input embedding layer
    pub fn new(input_dim: usize, d_model: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        let embedding_network = NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(d_model * 2)
            .output_layer_with_activation(d_model, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create embedding network: {}", e) })?;
        
        Ok(Self {
            embedding_network,
            d_model,
        })
    }
    
    /// Generate embeddings for input sequence
    pub fn forward(&mut self, input_sequence: &[Vec<T>]) -> Vec<Vec<T>> {
        input_sequence.iter()
            .map(|input| self.embedding_network.run(input))
            .collect()
    }
}

/// Transformer encoder layer
pub struct TransformerEncoderLayer<T: Float> {
    self_attention: MultiHeadAttention<T>,
    feed_forward: FeedForwardNetwork<T>,
    residual1: ResidualConnection<T>,
    residual2: ResidualConnection<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> TransformerEncoderLayer<T> {
    /// Create a new transformer encoder layer
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, activation: ActivationFunction) -> Result<Self, ModelError> {
        let self_attention = MultiHeadAttention::new(AttentionConfig {
            d_model,
            num_heads,
            d_k: d_model / num_heads,
            d_v: d_model / num_heads,
            dropout: 0.1,
        })?;
        
        let feed_forward = FeedForwardNetwork::new(d_model, d_ff, activation)?;
        let residual1 = ResidualConnection::new(d_model);
        let residual2 = ResidualConnection::new(d_model);
        
        Ok(Self {
            self_attention,
            feed_forward,
            residual1,
            residual2,
        })
    }
    
    /// Forward pass through encoder layer
    pub fn forward(&mut self, input: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        // Self-attention with residual connection
        let attention_output = self.self_attention.forward(input, input, input)?;
        let residual1_output: Vec<Vec<T>> = input.iter()
            .zip(attention_output.iter())
            .map(|(inp, att)| self.residual1.forward(inp, att))
            .collect();
        
        // Feed-forward with residual connection
        let mut ff_output = Vec::new();
        for (inp, res1) in input.iter().zip(residual1_output.iter()) {
            let ff_result = self.feed_forward.forward(res1)?;
            let res2_result = self.residual2.forward(inp, &ff_result);
            ff_output.push(res2_result);
        }
        
        Ok(ff_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::<f32>::new(10, 8);
        assert_eq!(pe.encodings.len(), 10);
        assert_eq!(pe.encodings[0].len(), 8);
    }
    
    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::<f32>::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input);
        assert_eq!(output.len(), 4);
    }
    
    #[test]
    fn test_feed_forward_network() {
        let mut ffn = FeedForwardNetwork::<f32>::new(4, 8, ActivationFunction::ReLU).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.len(), 4);
    }
}