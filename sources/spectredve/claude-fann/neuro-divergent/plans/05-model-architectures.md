# Neural Forecasting Model Architectures Analysis

## Overview
This document provides a comprehensive analysis of all 27 neural forecasting models from NeuralForecast library, examining their architectures, mathematical foundations, and algorithmic approaches. The goal is to understand how each model can be implemented using ruv-FANN's building blocks.

## 1. Recurrent Neural Network Models

### 1.1 Basic RNN Models

#### RNN (Recurrent Neural Network)
- **Architecture**: Simple recurrent structure with hidden state
- **Mathematical Foundation**: h_t = tanh(W_hh * h_{t-1} + W_ih * x_t + b_h)
- **Implementation Requirements**: 
  - Hidden state management
  - Time-wise weight sharing
  - Gradient flow through time
- **ruv-FANN Mapping**: Use Layer with feedback connections and temporal state management

#### LSTM (Long Short-Term Memory)
- **Architecture**: Gate-based recurrent architecture
- **Mathematical Foundation**: 
  - Forget gate: f_t = σ(W_f * [h_{t-1}, x_t] + b_f)
  - Input gate: i_t = σ(W_i * [h_{t-1}, x_t] + b_i)
  - Candidate values: C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
  - Cell state: C_t = f_t * C_{t-1} + i_t * C̃_t
  - Output gate: o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
  - Hidden state: h_t = o_t * tanh(C_t)
- **Implementation Requirements**: 
  - Four gate mechanisms (forget, input, output, candidate)
  - Cell state and hidden state management
  - Element-wise multiplication and addition operations
- **ruv-FANN Mapping**: Create specialized LSTM layer with 4 sub-networks for gates

#### GRU (Gated Recurrent Unit)
- **Architecture**: Simplified gated recurrent structure
- **Mathematical Foundation**:
  - Reset gate: r_t = σ(W_r * [h_{t-1}, x_t])
  - Update gate: z_t = σ(W_z * [h_{t-1}, x_t])
  - New gate: n_t = tanh(W_n * [r_t * h_{t-1}, x_t])
  - Hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
- **Implementation Requirements**:
  - Three gate mechanisms (reset, update, new)
  - Element-wise operations
- **ruv-FANN Mapping**: Create GRU layer with 3 sub-networks for gates

### 1.2 Simple Feedforward Models

#### MLP (Multilayer Perceptron)
- **Architecture**: Standard feedforward neural network
- **Mathematical Foundation**: y = f(W_n * f(W_{n-1} * ... * f(W_1 * x + b_1) + b_{n-1}) + b_n)
- **Implementation Requirements**:
  - Multiple fully connected layers
  - Configurable activation functions
  - Input concatenation for different variable types
- **ruv-FANN Mapping**: Direct implementation using NetworkBuilder with hidden layers

#### MLPMultivariate
- **Architecture**: MLP adapted for multivariate time series
- **Mathematical Foundation**: Similar to MLP but with multivariate input handling
- **Implementation Requirements**:
  - Channel-wise processing
  - Multivariate input normalization
  - Shared or separate processing paths
- **ruv-FANN Mapping**: Multiple MLP networks or single MLP with expanded input layer

## 2. Convolutional Models

### 2.1 Temporal Convolutional Networks

#### TCN (Temporal Convolutional Network)
- **Architecture**: 1D convolutional layers with dilated convolutions
- **Mathematical Foundation**: y_t = Σ(w_i * x_{t-i*d}) where d is dilation
- **Implementation Requirements**:
  - 1D convolutional operations
  - Dilated convolutions
  - Residual connections
  - Causal convolutions
- **ruv-FANN Mapping**: Implement as specialized layers with sliding window operations

#### BiTCN (Bidirectional TCN)
- **Architecture**: Bidirectional temporal convolutions
- **Mathematical Foundation**: Combines forward and backward TCN outputs
- **Implementation Requirements**:
  - Forward and backward TCN processing
  - Bidirectional information fusion
- **ruv-FANN Mapping**: Two TCN networks with output combination

### 2.2 Advanced Convolutional Models

#### TimesNet
- **Architecture**: 2D vision-inspired temporal modeling
- **Mathematical Foundation**: 
  - FFT-based period detection
  - 2D convolution on period-transformed data
  - Inception blocks for multi-scale features
- **Implementation Requirements**:
  - FFT operations for period detection
  - 2D convolution layers (can be simulated with 1D)
  - Inception block architecture
  - Period-based data transformation
- **ruv-FANN Mapping**: Complex multi-stage network with FFT preprocessing and specialized convolution layers

## 3. Transformer-Based Models

### 3.1 Core Transformer Models

#### TFT (Temporal Fusion Transformer)
- **Architecture**: Specialized transformer for time series
- **Mathematical Foundation**:
  - Multi-head attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - Variable selection networks
  - Static and temporal encoding
- **Implementation Requirements**:
  - Multi-head attention mechanism
  - Variable selection networks
  - Static covariate encoding
  - Temporal feature processing
  - Gating mechanisms
- **ruv-FANN Mapping**: Complex multi-component network with attention simulation using MLPs

#### Informer
- **Architecture**: Efficient transformer with sparse attention
- **Mathematical Foundation**: ProbSparse attention mechanism
- **Implementation Requirements**:
  - Sparse attention computation
  - Efficient attention pattern selection
  - Long sequence processing capability
- **ruv-FANN Mapping**: Attention layers implemented as specialized MLP networks with sparse connections

#### AutoFormer
- **Architecture**: Auto-correlation based transformer
- **Mathematical Foundation**: Auto-correlation mechanism instead of self-attention
- **Implementation Requirements**:
  - Auto-correlation computation
  - Series decomposition
  - Trend and seasonal modeling
- **ruv-FANN Mapping**: Specialized correlation layers with decomposition preprocessing

#### FedFormer
- **Architecture**: Frequency enhanced transformer
- **Mathematical Foundation**: Fourier transforms with attention in frequency domain
- **Implementation Requirements**:
  - Fourier transform operations
  - Frequency domain attention
  - Mixed domain processing
- **ruv-FANN Mapping**: FFT preprocessing with transformer-like MLP attention

### 3.2 Patch-Based Models

#### PatchTST
- **Architecture**: Patch-based transformer for time series
- **Mathematical Foundation**: 
  - Patch embedding: divide series into patches
  - Channel-independent processing
  - Position encoding for patches
- **Implementation Requirements**:
  - Patch creation and embedding
  - Position encoding
  - Channel-independent transformer layers
- **ruv-FANN Mapping**: Sliding window preprocessing with independent MLP processing

#### iTransformer
- **Architecture**: Inverted transformer architecture
- **Mathematical Foundation**: Attention across time series variables instead of time
- **Implementation Requirements**:
  - Variable-wise attention
  - Inverted attention patterns
  - Cross-series dependencies
- **ruv-FANN Mapping**: Transposed input processing with MLP-based attention

## 4. Specialized Architecture Models

### 4.1 N-BEATS Family

#### NBEATS
- **Architecture**: Neural basis expansion with interpretable stacks
- **Mathematical Foundation**:
  - Basis function expansion: f(x) = Σ(θ_i * b_i(x))
  - Stack decomposition: trend, seasonality, identity
  - Residual learning: x_l = x_{l-1} - backcast_l
- **Implementation Requirements**:
  - Multiple basis function types (polynomial, Fourier, etc.)
  - Stack-based architecture
  - Backcast and forecast branches
  - Residual connections between stacks
- **ruv-FANN Mapping**: Multi-stack network with specialized basis function layers

#### NBEATSx
- **Architecture**: Extended NBEATS with exogenous variables
- **Mathematical Foundation**: NBEATS + exogenous variable handling
- **Implementation Requirements**:
  - All NBEATS components
  - Exogenous variable integration
  - Enhanced basis functions
- **ruv-FANN Mapping**: Extended NBEATS architecture with additional input processing

#### NHITS
- **Architecture**: Neural hierarchical interpolation for time series
- **Mathematical Foundation**: Multi-rate hierarchical processing
- **Implementation Requirements**:
  - Hierarchical sampling rates
  - Interpolation mechanisms
  - Multi-scale processing
- **ruv-FANN Mapping**: Multi-scale network with interpolation layers

### 4.2 Linear Models

#### DLinear
- **Architecture**: Decomposition + Linear
- **Mathematical Foundation**:
  - Series decomposition: x = trend + seasonal
  - Linear mapping: y_trend = Linear(trend), y_seasonal = Linear(seasonal)
  - Final output: y = y_trend + y_seasonal
- **Implementation Requirements**:
  - Moving average decomposition
  - Separate linear transformations
  - Component recombination
- **ruv-FANN Mapping**: Preprocessing decomposition + two linear networks

#### NLinear
- **Architecture**: Normalized Linear
- **Mathematical Foundation**: 
  - Normalization: x_norm = (x - μ) / σ
  - Linear transformation: y = Linear(x_norm)
  - Denormalization: final_y = y * σ + μ
- **Implementation Requirements**:
  - Input normalization
  - Linear transformation
  - Output denormalization
- **ruv-FANN Mapping**: Simple linear network with normalization preprocessing

### 4.3 Advanced Specialized Models

#### TiDE (Time-series Dense Encoder)
- **Architecture**: Dense encoder-decoder for time series
- **Mathematical Foundation**: Dense feature extraction with residual connections
- **Implementation Requirements**:
  - Dense feature encoding
  - Residual connections
  - Multi-scale temporal processing
- **ruv-FANN Mapping**: Dense network with residual connection simulation

#### DeepAR
- **Architecture**: Autoregressive RNN with probabilistic outputs
- **Mathematical Foundation**: 
  - Autoregressive: y_t ~ P(y_t | y_{1:t-1}, x_{1:T})
  - Probabilistic outputs with distribution parameters
- **Implementation Requirements**:
  - RNN backbone
  - Probabilistic output layers
  - Distribution parameter estimation
- **ruv-FANN Mapping**: RNN network with specialized probabilistic output layers

#### DeepNPTS
- **Architecture**: Deep Neural Point Time Series
- **Mathematical Foundation**: Point process modeling for irregular time series
- **Implementation Requirements**:
  - Point process handling
  - Irregular time interval processing
  - Event-based modeling
- **ruv-FANN Mapping**: Specialized event-based network architecture

### 4.4 Mixing Models

#### TSMixer
- **Architecture**: Time Series Mixer with MLP-based mixing
- **Mathematical Foundation**: 
  - Time mixing: MLP across time dimension
  - Feature mixing: MLP across feature dimension
  - Alternating mixing operations
- **Implementation Requirements**:
  - Time-wise MLP operations
  - Feature-wise MLP operations
  - Alternating processing stages
- **ruv-FANN Mapping**: Alternating MLP networks with different input orientations

#### TSMixerx
- **Architecture**: Extended TSMixer with additional components
- **Mathematical Foundation**: TSMixer + enhanced mixing strategies
- **Implementation Requirements**:
  - All TSMixer components
  - Enhanced mixing mechanisms
  - Additional processing stages
- **ruv-FANN Mapping**: Extended TSMixer architecture

### 4.5 Graph-Based Models

#### StemGNN
- **Architecture**: Spectral Temporal Graph Neural Network
- **Mathematical Foundation**:
  - Graph convolution: H^(l+1) = σ(D^(-1/2)AD^(-1/2)H^(l)W^(l))
  - Spectral analysis with graph Fourier transform
  - Temporal dynamics on graph structure
- **Implementation Requirements**:
  - Graph convolution operations
  - Spectral graph analysis
  - Dynamic graph structure handling
- **ruv-FANN Mapping**: Specialized graph layers with adjacency matrix operations

### 4.6 Large Language Model Integration

#### TimeLLM
- **Architecture**: Time series forecasting with Large Language Models
- **Mathematical Foundation**: 
  - Time series tokenization
  - LLM-based sequence modeling
  - Cross-modal understanding
- **Implementation Requirements**:
  - Time series to text conversion
  - Large-scale transformer architecture
  - Cross-modal processing
- **ruv-FANN Mapping**: Extremely large MLP networks simulating transformer behavior (computationally intensive)

## 5. Commonalities and Shared Components

### 5.1 Common Building Blocks

1. **Linear Layers**: Used in almost all models for final projection
2. **Activation Functions**: ReLU, Sigmoid, Tanh are most common
3. **Normalization**: Layer normalization, batch normalization
4. **Dropout**: Regularization technique used across models
5. **Residual Connections**: Skip connections for better gradient flow
6. **Attention Mechanisms**: Can be approximated with MLP networks

### 5.2 Shared Preprocessing Components

1. **Input Normalization**: Standardization, min-max scaling
2. **Feature Engineering**: Lag features, rolling statistics
3. **Time Series Decomposition**: Trend, seasonal, residual
4. **Embedding Layers**: For categorical variables

### 5.3 Common Output Processing

1. **Linear Projection**: Final forecast generation
2. **Probabilistic Outputs**: Distribution parameter estimation
3. **Multi-horizon Forecasting**: Direct or recursive approaches
4. **Uncertainty Quantification**: Confidence intervals

## 6. Mathematical Foundations Summary

### 6.1 Core Mathematical Concepts

1. **Convolution**: f * g(t) = ∫ f(τ)g(t-τ)dτ
2. **Attention**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
3. **Recurrence**: h_t = f(h_{t-1}, x_t)
4. **Basis Expansion**: f(x) = Σ(θ_i * b_i(x))
5. **Fourier Transform**: F(ω) = ∫ f(t)e^(-iωt)dt

### 6.2 Time Series Specific Operations

1. **Autoregression**: y_t = Σ(φ_i * y_{t-i}) + ε_t
2. **Moving Average**: MA(q) = Σ(θ_i * ε_{t-i})
3. **Seasonal Decomposition**: x_t = trend_t + seasonal_t + residual_t
4. **Differencing**: ∇y_t = y_t - y_{t-1}

## 7. Implementation Complexity Assessment

### 7.1 Low Complexity (Direct ruv-FANN Implementation)
- MLP, MLPMultivariate
- DLinear, NLinear
- Basic RNN structures

### 7.2 Medium Complexity (Extended ruv-FANN Implementation)
- LSTM, GRU
- NBEATS family
- TSMixer family
- Basic transformer models

### 7.3 High Complexity (Significant Extensions Required)
- TimesNet (requires FFT)
- StemGNN (requires graph operations)
- Advanced transformers (Informer, AutoFormer)
- TimeLLM (requires massive scale)

## 8. Architecture Innovation Insights

### 8.1 Key Innovations Across Models

1. **Attention Mechanisms**: From self-attention to specialized patterns
2. **Decomposition Approaches**: Breaking time series into components
3. **Multi-scale Processing**: Capturing different temporal resolutions
4. **Probabilistic Modeling**: Uncertainty quantification
5. **Domain-specific Adaptations**: Time series specific modifications

### 8.2 Evolution Patterns

1. **From RNNs to Transformers**: Capturing long-range dependencies
2. **From Complex to Simple**: Linear models proving competitive
3. **From Single-scale to Multi-scale**: Hierarchical processing
4. **From Deterministic to Probabilistic**: Uncertainty modeling

This comprehensive analysis provides the foundation for implementing these 27 neural forecasting models using ruv-FANN's building blocks, with clear understanding of architectural requirements and mathematical foundations for each approach.