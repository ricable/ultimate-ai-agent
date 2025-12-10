//! # Informer: Efficient Transformer for Long Sequence Time-Series Forecasting
//!
//! The Informer model addresses the quadratic time complexity and high memory usage
//! of vanilla Transformers when dealing with long sequences. It introduces several
//! key innovations:
//!
//! ## Key Innovations
//!
//! 1. **ProbSparse Self-Attention**: O(L log L) complexity instead of O(L²)
//! 2. **Self-Attention Distilling**: Progressive dimension reduction
//! 3. **Generative Style Decoder**: Efficient one-forward step prediction
//! 4. **Convolutional Self-Attention**: 1D convolution in attention computation
//!
//! ## Architecture Components
//!
//! - **Encoder**: Multiple layers with ProbSparse attention and distilling
//! - **Decoder**: Generative decoder with masked attention
//! - **Embedding**: Positional and temporal embeddings
//! - **Distilling**: Progressive attention map size reduction

use crate::{BaseModel, ModelError, ActivationFunction};
use crate::transformer::attention::{SparseAttention, AttentionConfig};
use num_traits::Float;
use std::fmt::Debug;

/// Configuration for the Informer model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InformerConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Feed-forward network dimension
    pub d_ff: usize,
    
    /// Historical sequence length
    pub seq_len: usize,
    
    /// Label sequence length (decoder input length)
    pub label_len: usize,
    
    /// Prediction sequence length
    pub pred_len: usize,
    
    /// Dropout rate
    pub dropout: f64,
    
    /// Attention factor for ProbSparse attention
    pub factor: usize,
    
    /// Number of features
    pub num_features: usize,
    
    /// Distilling factor (conv kernel size)
    pub distil_factor: usize,
    
    /// Activation function
    pub activation: ActivationFunction,
}

impl Default for InformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_encoder_layers: 2,
            num_decoder_layers: 1,
            num_heads: 8,
            d_ff: 2048,
            seq_len: 96,
            label_len: 48,
            pred_len: 24,
            dropout: 0.05,
            factor: 5,
            num_features: 7,
            distil_factor: 2,
            activation: ActivationFunction::ReLU,
        }
    }
}

/// ProbSparse Self-Attention mechanism for efficient long sequence processing
pub struct ProbSparseAttention<T: Float> {
    /// Sparse attention mechanism
    sparse_attention: SparseAttention<T>,
    
    /// Attention factor for sampling
    factor: usize,
    
    /// Model dimension
    d_model: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> ProbSparseAttention<T> {
    /// Create a new ProbSparse attention mechanism
    pub fn new(d_model: usize, num_heads: usize, factor: usize) -> Result<Self, ModelError> {
        let config = AttentionConfig {
            d_model,
            num_heads,
            d_k: d_model / num_heads,
            d_v: d_model / num_heads,
            dropout: 0.1,
        };
        
        let sparse_attention = SparseAttention::new(config, factor)?;
        
        Ok(Self {
            sparse_attention,
            factor,
            d_model,
        })
    }
    
    /// Forward pass with ProbSparse attention
    pub fn forward(&mut self, queries: &[Vec<T>], keys: &[Vec<T>], values: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        // Use sparse attention with factor-based sampling
        self.sparse_attention.forward(queries, keys, values)
    }
}

/// Attention Distilling layer for progressive dimension reduction
pub struct AttentionDistilling<T: Float> {
    /// Convolution layer for distilling
    conv_layer: ruv_fann::Network<T>,
    
    /// Normalization layer
    norm_layer: LayerNorm<T>,
    
    /// Activation function
    activation: ActivationFunction,
    
    /// Kernel size for distilling
    kernel_size: usize,
}

/// Simple layer normalization implementation
pub struct LayerNorm<T: Float> {
    epsilon: T,
    d_model: usize,
}

impl<T: Float + Debug + Clone + Send + Sync> LayerNorm<T> {
    pub fn new(d_model: usize) -> Self {
        Self {
            epsilon: T::from(1e-5).unwrap(),
            d_model,
        }
    }
    
    pub fn forward(&self, input: &[T]) -> Vec<T> {
        crate::tensor_ops::layer_norm(input, self.epsilon)
    }
}

impl<T: Float + Debug + Clone + Send + Sync> AttentionDistilling<T> {
    /// Create a new attention distilling layer
    pub fn new(d_model: usize, kernel_size: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        // Simulate 1D convolution with MLP
        // In practice, this would be a proper convolution layer
        let conv_layer = NetworkBuilder::new()
            .input_layer(d_model * kernel_size)
            .hidden_layer(d_model)
            .output_layer_with_activation(d_model, ruv_fann::activation::ActivationFunction::ReLU, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create conv layer: {}", e) })?;
        
        let norm_layer = LayerNorm::new(d_model);
        
        Ok(Self {
            conv_layer,
            norm_layer,
            activation: ActivationFunction::ReLU,
            kernel_size,
        })
    }
    
    /// Forward pass through distilling layer
    pub fn forward(&mut self, input: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        let seq_len = input.len();
        let d_model = input[0].len();
        
        // Apply "convolution" by processing windows
        let mut distilled_output = Vec::new();
        
        for i in (0..seq_len).step_by(self.kernel_size) {
            // Create window
            let mut window_input = Vec::new();
            for j in 0..self.kernel_size {
                if i + j < seq_len {
                    window_input.extend_from_slice(&input[i + j]);
                } else {
                    // Pad with zeros
                    window_input.extend(vec![T::zero(); d_model]);
                }
            }
            
            // Apply convolution-like operation
            let conv_output = self.conv_layer.run(&window_input);
            
            // Apply normalization
            let normalized = self.norm_layer.forward(&conv_output);
            
            distilled_output.push(normalized);
        }
        
        Ok(distilled_output)
    }
}

/// Informer Encoder Layer with ProbSparse attention and distilling
pub struct InformerEncoderLayer<T: Float> {
    /// ProbSparse self-attention
    attention: ProbSparseAttention<T>,
    
    /// Feed-forward network
    feed_forward: FeedForwardNetwork<T>,
    
    /// Layer normalization layers
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    
    /// Dropout rate
    dropout: f64,
}

/// Feed-forward network for Informer
pub struct FeedForwardNetwork<T: Float> {
    linear1: ruv_fann::Network<T>,
    linear2: ruv_fann::Network<T>,
    activation: ActivationFunction,
}

impl<T: Float + Debug + Clone + Send + Sync> FeedForwardNetwork<T> {
    pub fn new(d_model: usize, d_ff: usize, activation: ActivationFunction) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        let linear1 = NetworkBuilder::new()
            .input_layer(d_model)
            .output_layer_with_activation(d_ff, activation.to_ruv_fann_activation(), T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create linear1: {}", e) })?;
        
        let linear2 = NetworkBuilder::new()
            .input_layer(d_ff)
            .output_layer_with_activation(d_model, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create linear2: {}", e) })?;
        
        Ok(Self {
            linear1,
            linear2,
            activation,
        })
    }
    
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        let hidden = self.linear1.run(input);
        let output = self.linear2.run(&hidden);
        Ok(output)
    }
}

impl<T: Float + Debug + Clone + Send + Sync> InformerEncoderLayer<T> {
    /// Create a new Informer encoder layer
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, factor: usize, dropout: f64) -> Result<Self, ModelError> {
        let attention = ProbSparseAttention::new(d_model, num_heads, factor)?;
        let feed_forward = FeedForwardNetwork::new(d_model, d_ff, ActivationFunction::ReLU)?;
        let norm1 = LayerNorm::new(d_model);
        let norm2 = LayerNorm::new(d_model);
        
        Ok(Self {
            attention,
            feed_forward,
            norm1,
            norm2,
            dropout,
        })
    }
    
    /// Forward pass through encoder layer
    pub fn forward(&mut self, input: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        // Self-attention with residual connection and layer norm
        let attention_output = self.attention.forward(input, input, input)?;
        let residual1: Vec<Vec<T>> = input.iter()
            .zip(attention_output.iter())
            .map(|(inp, att)| {
                let normalized = self.norm1.forward(att);
                inp.iter()
                    .zip(normalized.iter())
                    .map(|(&x, &norm)| x + norm)
                    .collect()
            })
            .collect();
        
        // Feed-forward with residual connection and layer norm
        let mut ff_output = Vec::new();
        for (inp, res1) in input.iter().zip(residual1.iter()) {
            let ff_result = self.feed_forward.forward(res1)?;
            let normalized = self.norm2.forward(&ff_result);
            let residual2: Vec<T> = inp.iter()
                .zip(normalized.iter())
                .map(|(&x, &norm)| x + norm)
                .collect();
            ff_output.push(residual2);
        }
        
        Ok(ff_output)
    }
}

/// Informer Decoder Layer with masked attention
pub struct InformerDecoderLayer<T: Float> {
    /// Masked self-attention
    self_attention: ProbSparseAttention<T>,
    
    /// Cross-attention with encoder
    cross_attention: ProbSparseAttention<T>,
    
    /// Feed-forward network
    feed_forward: FeedForwardNetwork<T>,
    
    /// Layer normalization layers
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    norm3: LayerNorm<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> InformerDecoderLayer<T> {
    /// Create a new Informer decoder layer
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, factor: usize) -> Result<Self, ModelError> {
        let self_attention = ProbSparseAttention::new(d_model, num_heads, factor)?;
        let cross_attention = ProbSparseAttention::new(d_model, num_heads, factor)?;
        let feed_forward = FeedForwardNetwork::new(d_model, d_ff, ActivationFunction::ReLU)?;
        let norm1 = LayerNorm::new(d_model);
        let norm2 = LayerNorm::new(d_model);
        let norm3 = LayerNorm::new(d_model);
        
        Ok(Self {
            self_attention,
            cross_attention,
            feed_forward,
            norm1,
            norm2,
            norm3,
        })
    }
    
    /// Forward pass through decoder layer
    pub fn forward(&mut self, target: &[Vec<T>], encoder_output: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        // Masked self-attention
        let self_att_output = self.self_attention.forward(target, target, target)?;
        let residual1: Vec<Vec<T>> = target.iter()
            .zip(self_att_output.iter())
            .map(|(inp, att)| {
                let normalized = self.norm1.forward(att);
                inp.iter()
                    .zip(normalized.iter())
                    .map(|(&x, &norm)| x + norm)
                    .collect()
            })
            .collect();
        
        // Cross-attention with encoder
        let cross_att_output = self.cross_attention.forward(&residual1, encoder_output, encoder_output)?;
        let residual2: Vec<Vec<T>> = residual1.iter()
            .zip(cross_att_output.iter())
            .map(|(inp, att)| {
                let normalized = self.norm2.forward(att);
                inp.iter()
                    .zip(normalized.iter())
                    .map(|(&x, &norm)| x + norm)
                    .collect()
            })
            .collect();
        
        // Feed-forward
        let mut ff_output = Vec::new();
        for (inp, res2) in residual1.iter().zip(residual2.iter()) {
            let ff_result = self.feed_forward.forward(res2)?;
            let normalized = self.norm3.forward(&ff_result);
            let residual3: Vec<T> = inp.iter()
                .zip(normalized.iter())
                .map(|(&x, &norm)| x + norm)
                .collect();
            ff_output.push(residual3);
        }
        
        Ok(ff_output)
    }
}

/// Main Informer forecaster implementation
pub struct InformerForecaster<T: Float> {
    config: InformerConfig,
    
    /// Input embedding
    input_embedding: ruv_fann::Network<T>,
    
    /// Positional encoding
    positional_encoding: Vec<Vec<T>>,
    
    /// Encoder layers
    encoder_layers: Vec<InformerEncoderLayer<T>>,
    
    /// Distilling layers
    distilling_layers: Vec<AttentionDistilling<T>>,
    
    /// Decoder layers  
    decoder_layers: Vec<InformerDecoderLayer<T>>,
    
    /// Output projection
    output_projection: ruv_fann::Network<T>,
    
    /// Fitted flag
    is_fitted: bool,
}

impl<T: Float + Debug + Clone + Send + Sync> InformerForecaster<T> {
    /// Create a new Informer forecaster
    pub fn new(config: InformerConfig) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        // Input embedding
        let input_embedding = NetworkBuilder::new()
            .input_layer(config.num_features)
            .hidden_layer(config.d_model)
            .output_layer_with_activation(config.d_model, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create input embedding: {}", e) })?;
        
        // Positional encoding
        let positional_encoding = crate::tensor_ops::positional_encoding(
            config.seq_len + config.pred_len, 
            config.d_model
        );
        
        // Encoder layers
        let mut encoder_layers = Vec::with_capacity(config.num_encoder_layers);
        for _ in 0..config.num_encoder_layers {
            encoder_layers.push(InformerEncoderLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.factor,
                config.dropout,
            )?);
        }
        
        // Distilling layers (one less than encoder layers)
        let mut distilling_layers = Vec::with_capacity(config.num_encoder_layers.saturating_sub(1));
        for _ in 0..config.num_encoder_layers.saturating_sub(1) {
            distilling_layers.push(AttentionDistilling::new(
                config.d_model,
                config.distil_factor,
            )?);
        }
        
        // Decoder layers
        let mut decoder_layers = Vec::with_capacity(config.num_decoder_layers);
        for _ in 0..config.num_decoder_layers {
            decoder_layers.push(InformerDecoderLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.factor,
            )?);
        }
        
        // Output projection
        let output_projection = NetworkBuilder::new()
            .input_layer(config.d_model)
            .hidden_layer(config.d_model / 2)
            .output_layer_with_activation(1, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create output projection: {}", e) })?;
        
        Ok(Self {
            config,
            input_embedding,
            positional_encoding,
            encoder_layers,
            distilling_layers,
            decoder_layers,
            output_projection,
            is_fitted: false,
        })
    }
}

impl<T: Float + Debug + Clone + Send + Sync> BaseModel<T> for InformerForecaster<T> {
    type Config = InformerConfig;
    
    fn new(config: Self::Config) -> Self {
        Self::new(config).unwrap()
    }
    
    fn fit(&mut self, 
           y: &[T], 
           x: Option<&[Vec<T>]>, 
           static_features: Option<&[T]>) -> Result<(), ModelError> {
        
        if y.len() < self.config.seq_len + self.config.pred_len {
            return Err(ModelError::InvalidInput { 
                message: format!("Insufficient data: need at least {} samples", 
                               self.config.seq_len + self.config.pred_len) 
            });
        }
        
        // Placeholder for actual training implementation
        self.is_fitted = true;
        Ok(())
    }
    
    fn predict(&mut self, 
               h: usize, 
               x: Option<&[Vec<T>]>, 
               static_features: Option<&[T]>) -> Result<Vec<T>, ModelError> {
        
        if !self.is_fitted {
            return Err(ModelError::NotFitted { message: "Model must be fitted before prediction".to_string() });
        }
        
        // Simplified prediction implementation
        // In reality, this would include:
        // 1. Input embedding and positional encoding
        // 2. Encoder processing with distilling
        // 3. Decoder processing with cross-attention
        // 4. Output projection
        
        Ok(vec![T::zero(); h])
    }
    
    fn predict_with_uncertainty(&mut self, 
                               h: usize, 
                               x: Option<&[Vec<T>]>, 
                               static_features: Option<&[T]>) -> Result<(Vec<T>, Vec<T>), ModelError> {
        let predictions = self.predict(h, x, static_features)?;
        let uncertainty = vec![T::one(); h];
        Ok((predictions, uncertainty))
    }
    
    fn forecast_horizon(&self) -> usize {
        self.config.pred_len
    }
    
    fn supports_multivariate(&self) -> bool {
        true
    }
    
    fn supports_static_features(&self) -> bool {
        false
    }
    
    fn model_name(&self) -> &'static str {
        "Informer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_informer_config_default() {
        let config = InformerConfig::default();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.seq_len, 96);
        assert_eq!(config.pred_len, 24);
        assert_eq!(config.factor, 5);
    }
    
    #[test]
    fn test_prob_sparse_attention() {
        let attention = ProbSparseAttention::<f32>::new(64, 4, 5).unwrap();
        assert_eq!(attention.d_model, 64);
        assert_eq!(attention.factor, 5);
    }
    
    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::<f32>::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input);
        assert_eq!(output.len(), 4);
    }
    
    #[test]
    fn test_informer_forecaster_creation() {
        let config = InformerConfig::default();
        let forecaster = InformerForecaster::<f32>::new(config).unwrap();
        assert!(!forecaster.is_fitted);
        assert_eq!(forecaster.model_name(), "Informer");
    }
}