//! # Temporal Fusion Transformer (TFT)
//!
//! Implementation of the Temporal Fusion Transformer architecture for interpretable
//! multi-horizon forecasting. TFT combines high-performance deep learning with
//! interpretable insights into temporal dynamics.
//!
//! ## Architecture Overview
//!
//! TFT consists of several key components:
//! 1. **Variable Selection Networks**: Learn importance weights for input variables
//! 2. **Static Covariate Encoders**: Process time-invariant features
//! 3. **Temporal Processing**: LSTM-based sequence modeling
//! 4. **Self-Attention**: Multi-head attention for capturing long-range dependencies
//! 5. **Gating Mechanisms**: GLU-based gating throughout the architecture
//! 6. **Multi-Horizon Decoding**: Generate forecasts for multiple future steps
//!
//! ## Key Features
//!
//! - **Interpretability**: Variable importance weights and attention visualization
//! - **Multi-horizon**: Simultaneous forecasting for multiple future time steps
//! - **Heterogeneous inputs**: Handles static, known future, and observed inputs
//! - **Gating**: Extensive use of gating for feature selection and suppression

use crate::{BaseModel, ModelError, ActivationFunction, TimeSeriesInput, ForecastOutput};
use crate::transformer::attention::{MultiHeadAttention, AttentionConfig};
use num_traits::Float;
use std::fmt::Debug;

/// Configuration for the TFT model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TFTConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    
    /// Number of LSTM layers
    pub lstm_layers: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Attention dimension
    pub attention_dim: usize,
    
    /// Number of quantiles for probabilistic forecasting
    pub num_quantiles: usize,
    
    /// Dropout rate
    pub dropout: f64,
    
    /// Historical sequence length
    pub encoder_length: usize,
    
    /// Prediction horizon length
    pub decoder_length: usize,
    
    /// Number of static features
    pub num_static_features: usize,
    
    /// Number of time-varying known features
    pub num_known_features: usize,
    
    /// Number of time-varying observed features
    pub num_observed_features: usize,
    
    /// Number of categorical static variables
    pub num_categorical_static: usize,
    
    /// Number of categorical known variables
    pub num_categorical_known: usize,
}

impl Default for TFTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            lstm_layers: 2,
            num_heads: 4,
            attention_dim: 256,
            num_quantiles: 3,
            dropout: 0.1,
            encoder_length: 168,
            decoder_length: 24,
            num_static_features: 0,
            num_known_features: 0,
            num_observed_features: 1,
            num_categorical_static: 0,
            num_categorical_known: 0,
        }
    }
}

/// Variable Selection Network for learning input importance
pub struct VariableSelectionNetwork<T: Float> {
    /// Input dimension
    input_dim: usize,
    
    /// Hidden dimension
    hidden_dim: usize,
    
    /// Number of variables
    num_variables: usize,
    
    /// Context network for learning variable importance
    context_network: ruv_fann::Network<T>,
    
    /// Variable processing networks
    variable_networks: Vec<ruv_fann::Network<T>>,
    
    /// Softmax selection weights network
    selection_network: ruv_fann::Network<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> VariableSelectionNetwork<T> {
    /// Create a new variable selection network
    pub fn new(input_dim: usize, hidden_dim: usize, num_variables: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        // Context network processes all variables together
        let context_network = NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(hidden_dim)
            .hidden_layer(hidden_dim)
            .output_layer_with_activation(hidden_dim, ruv_fann::activation::ActivationFunction::ReLU, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create context network: {}", e) })?;
        
        // Individual variable processing networks
        let mut variable_networks = Vec::with_capacity(num_variables);
        let var_input_dim = input_dim / num_variables;
        
        for _ in 0..num_variables {
            let network = NetworkBuilder::new()
                .input_layer(var_input_dim)
                .hidden_layer(hidden_dim)
                .output_layer_with_activation(hidden_dim, ruv_fann::activation::ActivationFunction::ReLU, T::one())
                .build()
                .map_err(|e| ModelError::ConfigError { message: format!("Failed to create variable network: {}", e) })?;
            variable_networks.push(network);
        }
        
        // Selection network outputs variable importance weights
        let selection_network = NetworkBuilder::new()
            .input_layer(hidden_dim)
            .hidden_layer(hidden_dim / 2)
            .output_layer_with_activation(num_variables, ruv_fann::activation::ActivationFunction::Sigmoid, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create selection network: {}", e) })?;
        
        Ok(Self {
            input_dim,
            hidden_dim,
            num_variables,
            context_network,
            variable_networks,
            selection_network,
        })
    }
    
    /// Forward pass through variable selection network
    pub fn forward(&mut self, input: &[T]) -> Result<(Vec<T>, Vec<T>), ModelError> {
        // Get context representation
        let context = self.context_network.run(input);
        
        // Get variable importance weights
        let importance_weights = self.selection_network.run(&context);
        
        // Process each variable individually
        let var_size = self.input_dim / self.num_variables;
        let mut processed_variables = Vec::new();
        
        for (i, network) in self.variable_networks.iter_mut().enumerate() {
            let start_idx = i * var_size;
            let end_idx = ((i + 1) * var_size).min(input.len());
            
            if start_idx < input.len() {
                let var_input = &input[start_idx..end_idx];
                let processed = network.run(var_input);
                
                // Weight by importance
                let weighted: Vec<T> = processed.iter()
                    .map(|&x| x * importance_weights[i])
                    .collect();
                processed_variables.extend(weighted);
            }
        }
        
        Ok((processed_variables, importance_weights))
    }
}

/// Gated Linear Unit (GLU) for feature gating
pub struct GatedLinearUnit<T: Float> {
    linear_layer: ruv_fann::Network<T>,
    gate_layer: ruv_fann::Network<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> GatedLinearUnit<T> {
    /// Create a new GLU
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        let linear_layer = NetworkBuilder::new()
            .input_layer(input_dim)
            .output_layer_with_activation(output_dim, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create linear layer: {}", e) })?;
        
        let gate_layer = NetworkBuilder::new()
            .input_layer(input_dim)
            .output_layer_with_activation(output_dim, ruv_fann::activation::ActivationFunction::Sigmoid, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create gate layer: {}", e) })?;
        
        Ok(Self {
            linear_layer,
            gate_layer,
        })
    }
    
    /// Forward pass through GLU
    pub fn forward(&mut self, input: &[T]) -> Vec<T> {
        let linear_output = self.linear_layer.run(input);
        let gate_output = self.gate_layer.run(input);
        
        linear_output.iter()
            .zip(gate_output.iter())
            .map(|(&lin, &gate)| lin * gate)
            .collect()
    }
}

/// LSTM Encoder for temporal processing
pub struct LSTMEncoder<T: Float> {
    lstm_layers: Vec<LSTMLayer<T>>,
    num_layers: usize,
}

/// Simplified LSTM layer using MLPs for gates
pub struct LSTMLayer<T: Float> {
    input_size: usize,
    hidden_size: usize,
    forget_gate: ruv_fann::Network<T>,
    input_gate: ruv_fann::Network<T>,
    candidate_gate: ruv_fann::Network<T>,
    output_gate: ruv_fann::Network<T>,
    cell_state: Vec<T>,
    hidden_state: Vec<T>,
}

impl<T: Float + Debug + Clone + Send + Sync> LSTMLayer<T> {
    /// Create a new LSTM layer
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        let combined_input_size = input_size + hidden_size;
        
        let forget_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .hidden_layer(hidden_size)
            .output_layer_with_activation(hidden_size, ruv_fann::activation::ActivationFunction::Sigmoid, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create forget gate: {}", e) })?;
        
        let input_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .hidden_layer(hidden_size)
            .output_layer_with_activation(hidden_size, ruv_fann::activation::ActivationFunction::Sigmoid, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create input gate: {}", e) })?;
        
        let candidate_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .hidden_layer(hidden_size)
            .output_layer_with_activation(hidden_size, ruv_fann::activation::ActivationFunction::Tanh, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create candidate gate: {}", e) })?;
        
        let output_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .hidden_layer(hidden_size)
            .output_layer_with_activation(hidden_size, ruv_fann::activation::ActivationFunction::Sigmoid, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create output gate: {}", e) })?;
        
        Ok(Self {
            input_size,
            hidden_size,
            forget_gate,
            input_gate,
            candidate_gate,
            output_gate,
            cell_state: vec![T::zero(); hidden_size],
            hidden_state: vec![T::zero(); hidden_size],
        })
    }
    
    /// Forward pass through LSTM layer
    pub fn forward(&mut self, input: &[T]) -> Vec<T> {
        // Combine input and previous hidden state
        let mut combined_input = input.to_vec();
        combined_input.extend_from_slice(&self.hidden_state);
        
        // Compute gates
        let forget_values = self.forget_gate.run(&combined_input);
        let input_values = self.input_gate.run(&combined_input);
        let candidate_values = self.candidate_gate.run(&combined_input);
        let output_values = self.output_gate.run(&combined_input);
        
        // Update cell state: C_t = f_t * C_{t-1} + i_t * C̃_t
        for i in 0..self.hidden_size {
            self.cell_state[i] = forget_values[i] * self.cell_state[i] + 
                                input_values[i] * candidate_values[i];
        }
        
        // Update hidden state: h_t = o_t * tanh(C_t)
        for i in 0..self.hidden_size {
            self.hidden_state[i] = output_values[i] * self.cell_state[i].tanh();
        }
        
        self.hidden_state.clone()
    }
    
    /// Reset LSTM states
    pub fn reset_states(&mut self) {
        self.cell_state.fill(T::zero());
        self.hidden_state.fill(T::zero());
    }
}

impl<T: Float + Debug + Clone + Send + Sync> LSTMEncoder<T> {
    /// Create a new LSTM encoder
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self, ModelError> {
        let mut lstm_layers = Vec::with_capacity(num_layers);
        
        // First layer
        lstm_layers.push(LSTMLayer::new(input_size, hidden_size)?);
        
        // Subsequent layers
        for _ in 1..num_layers {
            lstm_layers.push(LSTMLayer::new(hidden_size, hidden_size)?);
        }
        
        Ok(Self {
            lstm_layers,
            num_layers,
        })
    }
    
    /// Process sequence through LSTM encoder
    pub fn forward(&mut self, input_sequence: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut outputs = Vec::with_capacity(input_sequence.len());
        
        // Reset all LSTM states
        for layer in &mut self.lstm_layers {
            layer.reset_states();
        }
        
        // Process each time step
        for input in input_sequence {
            let mut layer_output = input.clone();
            
            // Pass through each LSTM layer
            for layer in &mut self.lstm_layers {
                layer_output = layer.forward(&layer_output);
            }
            
            outputs.push(layer_output);
        }
        
        outputs
    }
}

/// Main TFT forecaster implementation
pub struct TFTForecaster<T: Float> {
    config: TFTConfig,
    
    // Variable selection networks
    static_variable_selection: Option<VariableSelectionNetwork<T>>,
    temporal_known_variable_selection: VariableSelectionNetwork<T>,
    temporal_observed_variable_selection: VariableSelectionNetwork<T>,
    
    // Static covariate processing
    static_encoder: Option<ruv_fann::Network<T>>,
    
    // Temporal processing
    lstm_encoder: LSTMEncoder<T>,
    lstm_decoder: LSTMEncoder<T>,
    
    // Attention mechanism
    attention: MultiHeadAttention<T>,
    
    // Gating mechanisms
    encoder_glu: GatedLinearUnit<T>,
    decoder_glu: GatedLinearUnit<T>,
    
    // Output projection
    output_projection: ruv_fann::Network<T>,
    
    // Quantile projection for probabilistic forecasting
    quantile_projection: ruv_fann::Network<T>,
    
    // State flags
    is_fitted: bool,
}

impl<T: Float + Debug + Clone + Send + Sync> TFTForecaster<T> {
    /// Create a new TFT forecaster
    pub fn new(config: TFTConfig) -> Result<Self, ModelError> {
        use ruv_fann::NetworkBuilder;
        
        // Static variable selection (if static features exist)
        let static_variable_selection = if config.num_static_features > 0 {
            Some(VariableSelectionNetwork::new(
                config.num_static_features,
                config.hidden_size,
                config.num_static_features,
            )?)
        } else {
            None
        };
        
        // Temporal variable selection networks
        let temporal_known_variable_selection = VariableSelectionNetwork::new(
            config.num_known_features,
            config.hidden_size,
            config.num_known_features.max(1),
        )?;
        
        let temporal_observed_variable_selection = VariableSelectionNetwork::new(
            config.num_observed_features,
            config.hidden_size,
            config.num_observed_features.max(1),
        )?;
        
        // Static encoder
        let static_encoder = if config.num_static_features > 0 {
            Some(NetworkBuilder::new()
                .input_layer(config.hidden_size)
                .hidden_layer(config.hidden_size)
                .output_layer_with_activation(config.hidden_size, ruv_fann::activation::ActivationFunction::ReLU, T::one())
                .build()
                .map_err(|e| ModelError::ConfigError { message: format!("Failed to create static encoder: {}", e) })?)
        } else {
            None
        };
        
        // LSTM encoder and decoder
        let lstm_encoder = LSTMEncoder::new(config.hidden_size, config.hidden_size, config.lstm_layers)?;
        let lstm_decoder = LSTMEncoder::new(config.hidden_size, config.hidden_size, config.lstm_layers)?;
        
        // Multi-head attention
        let attention_config = AttentionConfig {
            d_model: config.hidden_size,
            num_heads: config.num_heads,
            d_k: config.attention_dim / config.num_heads,
            d_v: config.attention_dim / config.num_heads,
            dropout: config.dropout,
        };
        let attention = MultiHeadAttention::new(attention_config)?;
        
        // Gating mechanisms
        let encoder_glu = GatedLinearUnit::new(config.hidden_size, config.hidden_size)?;
        let decoder_glu = GatedLinearUnit::new(config.hidden_size, config.hidden_size)?;
        
        // Output projection
        let output_projection = NetworkBuilder::new()
            .input_layer(config.hidden_size)
            .hidden_layer(config.hidden_size / 2)
            .output_layer_with_activation(1, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create output projection: {}", e) })?;
        
        // Quantile projection for probabilistic forecasting
        let quantile_projection = NetworkBuilder::new()
            .input_layer(config.hidden_size)
            .hidden_layer(config.hidden_size / 2)
            .output_layer_with_activation(config.num_quantiles, ruv_fann::activation::ActivationFunction::Linear, T::one())
            .build()
            .map_err(|e| ModelError::ConfigError { message: format!("Failed to create quantile projection: {}", e) })?;
        
        Ok(Self {
            config,
            static_variable_selection,
            temporal_known_variable_selection,
            temporal_observed_variable_selection,
            static_encoder,
            lstm_encoder,
            lstm_decoder,
            attention,
            encoder_glu,
            decoder_glu,
            output_projection,
            quantile_projection,
            is_fitted: false,
        })
    }
}

impl<T: Float + Debug + Clone + Send + Sync> BaseModel<T> for TFTForecaster<T> {
    type Config = TFTConfig;
    
    fn new(config: Self::Config) -> Self {
        Self::new(config).unwrap()
    }
    
    fn fit(&mut self, 
           y: &[T], 
           x: Option<&[Vec<T>]>, 
           static_features: Option<&[T]>) -> Result<(), ModelError> {
        
        if y.len() < self.config.encoder_length + self.config.decoder_length {
            return Err(ModelError::InvalidInput { 
                message: format!("Insufficient data: need at least {} samples, got {}", 
                               self.config.encoder_length + self.config.decoder_length, y.len()) 
            });
        }
        
        // For now, mark as fitted without actual training implementation
        // In a complete implementation, this would include:
        // 1. Data preprocessing and windowing
        // 2. Batch creation
        // 3. Forward/backward pass training loops
        // 4. Optimization with learning rate scheduling
        // 5. Early stopping and validation
        
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
        
        // Simplified prediction - in reality this would involve:
        // 1. Variable selection for all input types
        // 2. Static context encoding
        // 3. LSTM encoding of historical sequence
        // 4. Self-attention application
        // 5. LSTM decoding for future steps
        // 6. Output projection
        
        // For now, return zeros as placeholder
        Ok(vec![T::zero(); h])
    }
    
    fn predict_with_uncertainty(&mut self, 
                               h: usize, 
                               x: Option<&[Vec<T>]>, 
                               static_features: Option<&[T]>) -> Result<(Vec<T>, Vec<T>), ModelError> {
        let predictions = self.predict(h, x, static_features)?;
        let uncertainty = vec![T::one(); h]; // Placeholder uncertainty
        Ok((predictions, uncertainty))
    }
    
    fn forecast_horizon(&self) -> usize {
        self.config.decoder_length
    }
    
    fn supports_multivariate(&self) -> bool {
        true
    }
    
    fn supports_static_features(&self) -> bool {
        true
    }
    
    fn model_name(&self) -> &'static str {
        "TFT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tft_config_default() {
        let config = TFTConfig::default();
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.encoder_length, 168);
        assert_eq!(config.decoder_length, 24);
    }
    
    #[test]
    fn test_variable_selection_network() {
        let vsn = VariableSelectionNetwork::<f32>::new(12, 64, 3).unwrap();
        assert_eq!(vsn.input_dim, 12);
        assert_eq!(vsn.hidden_dim, 64);
        assert_eq!(vsn.num_variables, 3);
    }
    
    #[test]
    fn test_gated_linear_unit() {
        let mut glu = GatedLinearUnit::<f32>::new(10, 10).unwrap();
        let input = vec![1.0; 10];
        let output = glu.forward(&input);
        assert_eq!(output.len(), 10);
    }
    
    #[test]
    fn test_lstm_layer() {
        let mut lstm = LSTMLayer::<f32>::new(5, 10).unwrap();
        let input = vec![1.0; 5];
        let output = lstm.forward(&input);
        assert_eq!(output.len(), 10);
    }
    
    #[test]
    fn test_tft_forecaster_creation() {
        let config = TFTConfig::default();
        let forecaster = TFTForecaster::<f32>::new(config).unwrap();
        assert!(!forecaster.is_fitted);
        assert_eq!(forecaster.model_name(), "TFT");
    }
}