# Neural Forecasting Model Implementation Plans

## Overview
This document provides specific implementation strategies for all 27 neural forecasting models using ruv-FANN's building blocks. Each model includes detailed implementation plans, code structure, and training/inference pipelines.

## 1. Implementation Strategy Framework

### 1.1 ruv-FANN Building Blocks Available
- **Network**: Main neural network container with layers
- **Layer**: Collection of neurons with connections
- **Neuron**: Individual processing unit with activation functions
- **Connection**: Weighted connections between neurons
- **Activation Functions**: Linear, Sigmoid, Tanh, ReLU, etc.
- **Training Algorithms**: Backprop, RPROP, Quickprop
- **Error Functions**: MSE, MAE, Tanh Error

### 1.2 Extension Requirements
For complex models, we need to extend ruv-FANN with:
- **Temporal Layers**: For RNN/LSTM/GRU functionality
- **Attention Layers**: For transformer-based models
- **Convolution Layers**: For TCN and TimesNet
- **Specialized Processors**: For decomposition, embedding, etc.

## 2. Implementation Plans by Model Category

### 2.1 Simple Feedforward Models

#### 2.1.1 MLP (Multilayer Perceptron)
```rust
// Implementation Structure
struct MLPForecaster<T: Float> {
    network: Network<T>,
    input_size: usize,
    output_size: usize,
    hidden_sizes: Vec<usize>,
}

impl<T: Float> MLPForecaster<T> {
    fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut builder = NetworkBuilder::new()
            .input_layer(input_size);
        
        for &hidden_size in &hidden_sizes {
            builder = builder.hidden_layer(hidden_size);
        }
        
        let network = builder
            .output_layer(output_size)
            .build();
            
        Self { network, input_size, output_size, hidden_sizes }
    }
    
    fn forecast(&mut self, inputs: &[T]) -> Vec<T> {
        // Concatenate historical values + exogenous features
        let processed_inputs = self.preprocess_inputs(inputs);
        self.network.run(&processed_inputs)
    }
    
    fn preprocess_inputs(&self, raw_inputs: &[T]) -> Vec<T> {
        // Handle concatenation of different input types:
        // - Historical target values
        // - Static exogenous features  
        // - Future exogenous features
        raw_inputs.to_vec() // Simplified
    }
}

// Training Pipeline
struct MLPTrainer<T: Float> {
    trainer: IncrementalBackprop<T>,
    error_function: MseError,
}

impl<T: Float> MLPTrainer<T> {
    fn train(&mut self, 
             forecaster: &mut MLPForecaster<T>, 
             data: &TrainingData<T>) -> Result<T, TrainingError> {
        // Standard backpropagation training
        self.trainer.train_epoch(&mut forecaster.network, data)
    }
}
```

**Implementation Complexity**: Low
**Required Extensions**: None (direct ruv-FANN implementation)
**Key Features**:
- Direct NetworkBuilder usage
- Standard feedforward architecture
- Configurable hidden layer sizes

#### 2.1.2 MLPMultivariate
```rust
struct MLPMultivariateForecaster<T: Float> {
    networks: Vec<Network<T>>, // One network per series or shared
    n_series: usize,
    shared_weights: bool,
}

impl<T: Float> MLPMultivariateForecaster<T> {
    fn new(n_series: usize, 
           input_size: usize, 
           hidden_sizes: Vec<usize>, 
           output_size: usize,
           shared_weights: bool) -> Self {
        
        let networks = if shared_weights {
            // Single shared network
            vec![Self::build_network(input_size, &hidden_sizes, output_size)]
        } else {
            // Separate network per series
            (0..n_series)
                .map(|_| Self::build_network(input_size, &hidden_sizes, output_size))
                .collect()
        };
        
        Self { networks, n_series, shared_weights }
    }
    
    fn forecast(&mut self, multivariate_inputs: &[Vec<T>]) -> Vec<Vec<T>> {
        if self.shared_weights {
            // Process all series through shared network
            multivariate_inputs.iter()
                .map(|series_input| self.networks[0].run(series_input))
                .collect()
        } else {
            // Process each series through its own network
            multivariate_inputs.iter()
                .zip(self.networks.iter_mut())
                .map(|(series_input, network)| network.run(series_input))
                .collect()
        }
    }
}
```

**Implementation Complexity**: Low
**Required Extensions**: None
**Key Features**:
- Multiple networks or shared network approach
- Channel-wise processing capability

### 2.2 Linear Models

#### 2.2.1 DLinear
```rust
struct DLinearForecaster<T: Float> {
    trend_network: Network<T>,
    seasonal_network: Network<T>,
    moving_average_window: usize,
}

impl<T: Float> DLinearForecaster<T> {
    fn new(input_size: usize, output_size: usize, ma_window: usize) -> Self {
        // Simple linear networks (single layer)
        let trend_network = NetworkBuilder::new()
            .input_layer(input_size)
            .output_layer_with_activation(output_size, ActivationFunction::Linear, T::one())
            .build();
            
        let seasonal_network = NetworkBuilder::new()
            .input_layer(input_size)
            .output_layer_with_activation(output_size, ActivationFunction::Linear, T::one())
            .build();
            
        Self { trend_network, seasonal_network, moving_average_window: ma_window }
    }
    
    fn forecast(&mut self, inputs: &[T]) -> Vec<T> {
        // 1. Decompose input series
        let (trend, seasonal) = self.series_decomposition(inputs);
        
        // 2. Apply linear transformations
        let trend_forecast = self.trend_network.run(&trend);
        let seasonal_forecast = self.seasonal_network.run(&seasonal);
        
        // 3. Combine forecasts
        trend_forecast.iter()
            .zip(seasonal_forecast.iter())
            .map(|(&t, &s)| t + s)
            .collect()
    }
    
    fn series_decomposition(&self, series: &[T]) -> (Vec<T>, Vec<T>) {
        // Moving average for trend extraction
        let trend = self.moving_average(series);
        let seasonal: Vec<T> = series.iter()
            .zip(trend.iter())
            .map(|(&s, &t)| s - t)
            .collect();
        (trend, seasonal)
    }
    
    fn moving_average(&self, series: &[T]) -> Vec<T> {
        let window = self.moving_average_window;
        let mut result = Vec::new();
        
        for i in 0..series.len() {
            let start = if i >= window { i - window + 1 } else { 0 };
            let window_sum: T = series[start..=i].iter().copied().sum();
            let window_len = T::from(i - start + 1).unwrap();
            result.push(window_sum / window_len);
        }
        
        result
    }
}
```

**Implementation Complexity**: Low
**Required Extensions**: Preprocessing utilities for decomposition
**Key Features**:
- Series decomposition preprocessing
- Separate linear networks for trend/seasonal
- Component recombination

#### 2.2.2 NLinear
```rust
struct NLinearForecaster<T: Float> {
    network: Network<T>,
    normalization_stats: Option<(T, T)>, // (mean, std)
}

impl<T: Float> NLinearForecaster<T> {
    fn new(input_size: usize, output_size: usize) -> Self {
        let network = NetworkBuilder::new()
            .input_layer(input_size)
            .output_layer_with_activation(output_size, ActivationFunction::Linear, T::one())
            .build();
            
        Self { network, normalization_stats: None }
    }
    
    fn fit_normalization(&mut self, training_data: &[T]) {
        let mean = training_data.iter().copied().sum::<T>() / T::from(training_data.len()).unwrap();
        let variance = training_data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<T>() / T::from(training_data.len()).unwrap();
        let std = variance.sqrt();
        
        self.normalization_stats = Some((mean, std));
    }
    
    fn forecast(&mut self, inputs: &[T]) -> Vec<T> {
        // 1. Normalize inputs
        let normalized_inputs = self.normalize(inputs);
        
        // 2. Apply linear transformation
        let normalized_output = self.network.run(&normalized_inputs);
        
        // 3. Denormalize outputs
        self.denormalize(&normalized_output)
    }
    
    fn normalize(&self, inputs: &[T]) -> Vec<T> {
        if let Some((mean, std)) = self.normalization_stats {
            inputs.iter()
                .map(|&x| (x - mean) / std)
                .collect()
        } else {
            inputs.to_vec()
        }
    }
    
    fn denormalize(&self, outputs: &[T]) -> Vec<T> {
        if let Some((mean, std)) = self.normalization_stats {
            outputs.iter()
                .map(|&y| y * std + mean)
                .collect()
        } else {
            outputs.to_vec()
        }
    }
}
```

**Implementation Complexity**: Low
**Required Extensions**: Normalization utilities
**Key Features**:
- Input/output normalization
- Simple linear transformation
- Statistical preprocessing

### 2.3 Recurrent Models

#### 2.3.1 Basic RNN
```rust
// Requires extension to ruv-FANN
struct RecurrentLayer<T: Float> {
    hidden_size: usize,
    input_weights: Vec<Vec<T>>,
    hidden_weights: Vec<Vec<T>>,
    biases: Vec<T>,
    hidden_state: Vec<T>,
    activation: ActivationFunction,
}

impl<T: Float> RecurrentLayer<T> {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            hidden_size,
            input_weights: vec![vec![T::zero(); input_size]; hidden_size],
            hidden_weights: vec![vec![T::zero(); hidden_size]; hidden_size],
            biases: vec![T::zero(); hidden_size],
            hidden_state: vec![T::zero(); hidden_size],
            activation: ActivationFunction::Tanh,
        }
    }
    
    fn forward(&mut self, input: &[T]) -> Vec<T> {
        // h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
        let mut new_hidden = Vec::with_capacity(self.hidden_size);
        
        for i in 0..self.hidden_size {
            let mut sum = self.biases[i];
            
            // Input contribution
            for (j, &x) in input.iter().enumerate() {
                sum = sum + self.input_weights[i][j] * x;
            }
            
            // Hidden state contribution
            for (j, &h) in self.hidden_state.iter().enumerate() {
                sum = sum + self.hidden_weights[i][j] * h;
            }
            
            new_hidden.push(sum.tanh()); // Apply activation
        }
        
        self.hidden_state = new_hidden.clone();
        new_hidden
    }
    
    fn reset_state(&mut self) {
        self.hidden_state.fill(T::zero());
    }
}

struct RNNForecaster<T: Float> {
    rnn_layers: Vec<RecurrentLayer<T>>,
    output_layer: Network<T>,
}

impl<T: Float> RNNForecaster<T> {
    fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut rnn_layers = Vec::new();
        let mut layer_input_size = input_size;
        
        for &hidden_size in &hidden_sizes {
            rnn_layers.push(RecurrentLayer::new(layer_input_size, hidden_size));
            layer_input_size = hidden_size;
        }
        
        let output_layer = NetworkBuilder::new()
            .input_layer(layer_input_size)
            .output_layer(output_size)
            .build();
            
        Self { rnn_layers, output_layer }
    }
    
    fn forecast_sequence(&mut self, input_sequence: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut outputs = Vec::new();
        
        for input in input_sequence {
            let mut layer_output = input.clone();
            
            // Pass through RNN layers
            for rnn_layer in &mut self.rnn_layers {
                layer_output = rnn_layer.forward(&layer_output);
            }
            
            // Final output layer
            let forecast = self.output_layer.run(&layer_output);
            outputs.push(forecast);
        }
        
        outputs
    }
}
```

**Implementation Complexity**: Medium
**Required Extensions**: RecurrentLayer implementation
**Key Features**:
- Hidden state management
- Sequential processing
- Multi-layer RNN support

#### 2.3.2 LSTM
```rust
struct LSTMCell<T: Float> {
    input_size: usize,
    hidden_size: usize,
    
    // Gate weight matrices
    forget_gate: Network<T>,
    input_gate: Network<T>,
    candidate_gate: Network<T>,
    output_gate: Network<T>,
    
    // States
    cell_state: Vec<T>,
    hidden_state: Vec<T>,
}

impl<T: Float> LSTMCell<T> {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let combined_input_size = input_size + hidden_size;
        
        // Create gate networks
        let forget_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();
            
        let input_gate = NetworkBuilder::new()
            .input_layer(combined_input_size) 
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();
            
        let candidate_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Tanh, T::one())
            .build();
            
        let output_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();
        
        Self {
            input_size,
            hidden_size,
            forget_gate,
            input_gate, 
            candidate_gate,
            output_gate,
            cell_state: vec![T::zero(); hidden_size],
            hidden_state: vec![T::zero(); hidden_size],
        }
    }
    
    fn forward(&mut self, input: &[T]) -> Vec<T> {
        // Combine input and previous hidden state
        let mut combined_input = input.to_vec();
        combined_input.extend_from_slice(&self.hidden_state);
        
        // Compute gates
        let forget_values = self.forget_gate.run(&combined_input);
        let input_values = self.input_gate.run(&combined_input);
        let candidate_values = self.candidate_gate.run(&combined_input);
        let output_values = self.output_gate.run(&combined_input);
        
        // Update cell state
        // C_t = f_t * C_{t-1} + i_t * C̃_t
        for i in 0..self.hidden_size {
            self.cell_state[i] = forget_values[i] * self.cell_state[i] + 
                                input_values[i] * candidate_values[i];
        }
        
        // Update hidden state
        // h_t = o_t * tanh(C_t)
        for i in 0..self.hidden_size {
            self.hidden_state[i] = output_values[i] * self.cell_state[i].tanh();
        }
        
        self.hidden_state.clone()
    }
    
    fn reset_states(&mut self) {
        self.cell_state.fill(T::zero());
        self.hidden_state.fill(T::zero());
    }
}

struct LSTMForecaster<T: Float> {
    lstm_layers: Vec<LSTMCell<T>>,
    output_layer: Network<T>,
}
```

**Implementation Complexity**: Medium-High
**Required Extensions**: LSTM cell implementation with gate networks
**Key Features**:
- Four gate mechanisms using separate networks
- Cell state and hidden state management
- Element-wise operations

#### 2.3.3 GRU
```rust
struct GRUCell<T: Float> {
    input_size: usize,
    hidden_size: usize,
    
    reset_gate: Network<T>,
    update_gate: Network<T>,
    new_gate: Network<T>,
    
    hidden_state: Vec<T>,
}

impl<T: Float> GRUCell<T> {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let combined_input_size = input_size + hidden_size;
        
        let reset_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();
            
        let update_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Sigmoid, T::one())
            .build();
            
        let new_gate = NetworkBuilder::new()
            .input_layer(combined_input_size)
            .output_layer_with_activation(hidden_size, ActivationFunction::Tanh, T::one())
            .build();
        
        Self {
            input_size,
            hidden_size,
            reset_gate,
            update_gate,
            new_gate,
            hidden_state: vec![T::zero(); hidden_size],
        }
    }
    
    fn forward(&mut self, input: &[T]) -> Vec<T> {
        // Combine input and hidden state for reset and update gates
        let mut combined_input = input.to_vec();
        combined_input.extend_from_slice(&self.hidden_state);
        
        let reset_values = self.reset_gate.run(&combined_input);
        let update_values = self.update_gate.run(&combined_input);
        
        // Create reset hidden state for new gate
        let reset_hidden: Vec<T> = reset_values.iter()
            .zip(self.hidden_state.iter())
            .map(|(&r, &h)| r * h)
            .collect();
            
        let mut new_input = input.to_vec();
        new_input.extend_from_slice(&reset_hidden);
        
        let new_values = self.new_gate.run(&new_input);
        
        // Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        for i in 0..self.hidden_size {
            self.hidden_state[i] = (T::one() - update_values[i]) * new_values[i] + 
                                  update_values[i] * self.hidden_state[i];
        }
        
        self.hidden_state.clone()
    }
}
```

**Implementation Complexity**: Medium
**Required Extensions**: GRU cell implementation
**Key Features**:
- Three gate mechanisms
- Simplified compared to LSTM
- Element-wise state updates

### 2.4 Attention-Based Models (Transformer Family)

#### 2.4.1 Basic Transformer Components
```rust
// Attention mechanism using MLP approximation
struct MultiHeadAttention<T: Float> {
    num_heads: usize,
    head_dim: usize,
    query_networks: Vec<Network<T>>,
    key_networks: Vec<Network<T>>,
    value_networks: Vec<Network<T>>,
    output_network: Network<T>,
}

impl<T: Float> MultiHeadAttention<T> {
    fn new(input_dim: usize, num_heads: usize, head_dim: usize) -> Self {
        let mut query_networks = Vec::new();
        let mut key_networks = Vec::new();
        let mut value_networks = Vec::new();
        
        for _ in 0..num_heads {
            query_networks.push(
                NetworkBuilder::new()
                    .input_layer(input_dim)
                    .output_layer_with_activation(head_dim, ActivationFunction::Linear, T::one())
                    .build()
            );
            
            key_networks.push(
                NetworkBuilder::new()
                    .input_layer(input_dim)
                    .output_layer_with_activation(head_dim, ActivationFunction::Linear, T::one())
                    .build()
            );
            
            value_networks.push(
                NetworkBuilder::new()
                    .input_layer(input_dim)
                    .output_layer_with_activation(head_dim, ActivationFunction::Linear, T::one())
                    .build()
            );
        }
        
        let output_network = NetworkBuilder::new()
            .input_layer(num_heads * head_dim)
            .output_layer_with_activation(input_dim, ActivationFunction::Linear, T::one())
            .build();
        
        Self {
            num_heads,
            head_dim,
            query_networks,
            key_networks,
            value_networks,
            output_network,
        }
    }
    
    fn attention(&mut self, inputs: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut head_outputs = Vec::new();
        
        for head_idx in 0..self.num_heads {
            let mut queries = Vec::new();
            let mut keys = Vec::new();
            let mut values = Vec::new();
            
            // Generate Q, K, V for this head
            for input in inputs {
                queries.push(self.query_networks[head_idx].run(input));
                keys.push(self.key_networks[head_idx].run(input));
                values.push(self.value_networks[head_idx].run(input));
            }
            
            // Compute attention scores (simplified)
            let attention_output = self.compute_scaled_dot_product_attention(&queries, &keys, &values);
            head_outputs.extend(attention_output);
        }
        
        // Combine heads and apply output projection
        inputs.iter()
            .enumerate()
            .map(|(i, _)| {
                let combined_heads: Vec<T> = (0..self.num_heads)
                    .flat_map(|h| head_outputs[h * inputs.len() + i].clone())
                    .collect();
                self.output_network.run(&combined_heads)
            })
            .collect()
    }
    
    fn compute_scaled_dot_product_attention(&self, 
                                          queries: &[Vec<T>], 
                                          keys: &[Vec<T>], 
                                          values: &[Vec<T>]) -> Vec<Vec<T>> {
        // Simplified attention computation
        // In practice, this would involve matrix operations
        let seq_len = queries.len();
        let mut outputs = Vec::with_capacity(seq_len);
        
        for i in 0..seq_len {
            let mut attended_value = vec![T::zero(); self.head_dim];
            let mut attention_weights = Vec::with_capacity(seq_len);
            
            // Compute attention scores
            for j in 0..seq_len {
                let score = self.dot_product(&queries[i], &keys[j]) / 
                           T::from(self.head_dim as f64).unwrap().sqrt();
                attention_weights.push(score);
            }
            
            // Softmax
            let max_score = attention_weights.iter().fold(attention_weights[0], |max, &x| max.max(x));
            let exp_scores: Vec<T> = attention_weights.iter()
                .map(|&score| (score - max_score).exp())
                .collect();
            let sum_exp: T = exp_scores.iter().copied().sum();
            let attention_probs: Vec<T> = exp_scores.iter()
                .map(|&exp_score| exp_score / sum_exp)
                .collect();
            
            // Weighted sum of values
            for (prob, value) in attention_probs.iter().zip(values.iter()) {
                for k in 0..self.head_dim {
                    attended_value[k] = attended_value[k] + *prob * value[k];
                }
            }
            
            outputs.push(attended_value);
        }
        
        outputs
    }
    
    fn dot_product(&self, a: &[T], b: &[T]) -> T {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}
```

**Implementation Complexity**: High
**Required Extensions**: Complex attention mechanism simulation
**Key Features**:
- Multi-head attention using multiple networks
- Scaled dot-product attention approximation
- Sequence-to-sequence processing

#### 2.4.2 TFT (Temporal Fusion Transformer)
```rust
struct VariableSelectionNetwork<T: Float> {
    selection_network: Network<T>,
    processing_networks: Vec<Network<T>>,
}

impl<T: Float> VariableSelectionNetwork<T> {
    fn new(input_dim: usize, num_variables: usize, hidden_dim: usize) -> Self {
        // Network to learn variable importance weights
        let selection_network = NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(hidden_dim)
            .output_layer_with_activation(num_variables, ActivationFunction::Sigmoid, T::one())
            .build();
        
        // Networks to process each variable
        let processing_networks = (0..num_variables)
            .map(|_| {
                NetworkBuilder::new()
                    .input_layer(input_dim / num_variables)
                    .hidden_layer(hidden_dim)
                    .output_layer(hidden_dim)
                    .build()
            })
            .collect();
        
        Self { selection_network, processing_networks }
    }
    
    fn forward(&mut self, inputs: &[Vec<T>]) -> Vec<T> {
        // Learn variable importance
        let combined_input: Vec<T> = inputs.iter().flatten().cloned().collect();
        let importance_weights = self.selection_network.run(&combined_input);
        
        // Process each variable
        let mut processed_variables = Vec::new();
        for (i, variable_input) in inputs.iter().enumerate() {
            let processed = self.processing_networks[i].run(variable_input);
            // Weight by importance
            let weighted: Vec<T> = processed.iter()
                .map(|&x| x * importance_weights[i])
                .collect();
            processed_variables.extend(weighted);
        }
        
        processed_variables
    }
}

struct TFTForecaster<T: Float> {
    // Input processing
    static_variable_selection: VariableSelectionNetwork<T>,
    temporal_variable_selection: VariableSelectionNetwork<T>,
    
    // Encoders
    static_encoder: Network<T>,
    temporal_encoder: LSTMCell<T>,
    
    // Attention
    attention: MultiHeadAttention<T>,
    
    // Output
    output_layer: Network<T>,
}

impl<T: Float> TFTForecaster<T> {
    fn new(config: TFTConfig) -> Self {
        // Initialize all components based on configuration
        // This is a simplified structure
        todo!("Full TFT implementation")
    }
    
    fn forecast(&mut self, 
               static_inputs: &[Vec<T>],
               temporal_inputs: &[Vec<Vec<T>>]) -> Vec<T> {
        // 1. Variable selection
        let selected_static = self.static_variable_selection.forward(static_inputs);
        let selected_temporal: Vec<Vec<T>> = temporal_inputs.iter()
            .map(|seq| self.temporal_variable_selection.forward(&[seq.clone()]))
            .collect();
        
        // 2. Encoding
        let static_context = self.static_encoder.run(&selected_static);
        
        let mut temporal_contexts = Vec::new();
        for temporal_seq in selected_temporal {
            for step in temporal_seq {
                temporal_contexts.push(self.temporal_encoder.forward(&step));
            }
        }
        
        // 3. Attention
        let attended = self.attention.attention(&temporal_contexts);
        
        // 4. Output generation
        let final_context: Vec<T> = attended.last().unwrap().clone();
        self.output_layer.run(&final_context)
    }
}
```

**Implementation Complexity**: Very High
**Required Extensions**: Variable selection, complex attention, multiple encoders
**Key Features**:
- Variable selection networks
- Multi-component architecture
- Static and temporal processing

### 2.5 Specialized Models

#### 2.5.1 NBEATS
```rust
struct NBEATSBlock<T: Float> {
    basis_type: BasisType,
    backcast_network: Network<T>,
    forecast_network: Network<T>,
    theta_backcast_network: Network<T>,
    theta_forecast_network: Network<T>,
    basis_parameters: BasisParameters<T>,
}

#[derive(Clone)]
enum BasisType {
    Generic,
    Trend,
    Seasonality,
}

struct BasisParameters<T: Float> {
    polynomial_degree: Option<usize>,
    harmonics: Option<Vec<T>>,
    trend_coefficients: Option<Vec<T>>,
}

impl<T: Float> NBEATSBlock<T> {
    fn new(input_size: usize, 
           output_size: usize, 
           hidden_size: usize,
           basis_type: BasisType,
           basis_params: BasisParameters<T>) -> Self {
        
        // Main processing networks
        let backcast_network = NetworkBuilder::new()
            .input_layer(input_size)
            .hidden_layer(hidden_size)
            .hidden_layer(hidden_size)
            .output_layer(input_size)
            .build();
            
        let forecast_network = NetworkBuilder::new()
            .input_layer(input_size)
            .hidden_layer(hidden_size)  
            .hidden_layer(hidden_size)
            .output_layer(output_size)
            .build();
        
        // Theta networks for basis coefficients
        let theta_dim = Self::get_theta_dimension(&basis_type, &basis_params);
        
        let theta_backcast_network = NetworkBuilder::new()
            .input_layer(input_size)
            .hidden_layer(hidden_size)
            .output_layer(theta_dim)
            .build();
            
        let theta_forecast_network = NetworkBuilder::new()
            .input_layer(input_size)
            .hidden_layer(hidden_size)
            .output_layer(theta_dim)
            .build();
        
        Self {
            basis_type,
            backcast_network,
            forecast_network,
            theta_backcast_network,
            theta_forecast_network,
            basis_parameters: basis_params,
        }
    }
    
    fn forward(&mut self, input: &[T]) -> (Vec<T>, Vec<T>) {
        // Generate theta coefficients
        let theta_backcast = self.theta_backcast_network.run(input);
        let theta_forecast = self.theta_forecast_network.run(input);
        
        // Apply basis functions
        let backcast = self.apply_basis_functions(&theta_backcast, input.len(), true);
        let forecast = self.apply_basis_functions(&theta_forecast, 
                                                self.get_forecast_length(), false);
        
        (backcast, forecast)
    }
    
    fn apply_basis_functions(&self, theta: &[T], length: usize, is_backcast: bool) -> Vec<T> {
        match self.basis_type {
            BasisType::Generic => {
                // Generic basis: use theta directly as coefficients
                if theta.len() >= length {
                    theta[..length].to_vec()
                } else {
                    let mut result = theta.to_vec();
                    result.resize(length, T::zero());
                    result
                }
            },
            BasisType::Trend => {
                self.apply_trend_basis(theta, length)
            },
            BasisType::Seasonality => {
                self.apply_seasonality_basis(theta, length)
            }
        }
    }
    
    fn apply_trend_basis(&self, theta: &[T], length: usize) -> Vec<T> {
        // Polynomial trend basis
        let degree = self.basis_parameters.polynomial_degree.unwrap_or(3);
        let mut result = vec![T::zero(); length];
        
        for t in 0..length {
            let t_norm = T::from(t as f64 / length as f64).unwrap();
            let mut value = T::zero();
            
            for (i, &coeff) in theta.iter().take(degree + 1).enumerate() {
                value = value + coeff * t_norm.powi(i as i32);
            }
            
            result[t] = value;
        }
        
        result
    }
    
    fn apply_seasonality_basis(&self, theta: &[T], length: usize) -> Vec<T> {
        // Fourier basis for seasonality
        let harmonics = self.basis_parameters.harmonics.as_ref()
            .map(|h| h.clone())
            .unwrap_or_else(|| vec![T::one(); theta.len() / 2]);
        
        let mut result = vec![T::zero(); length];
        
        for t in 0..length {
            let t_norm = T::from(2.0 * std::f64::consts::PI * t as f64 / length as f64).unwrap();
            let mut value = T::zero();
            
            for (i, &harmonic) in harmonics.iter().enumerate() {
                if 2 * i + 1 < theta.len() {
                    let cos_coeff = theta[2 * i];
                    let sin_coeff = theta[2 * i + 1];
                    value = value + cos_coeff * (harmonic * t_norm).cos() + 
                           sin_coeff * (harmonic * t_norm).sin();
                }
            }
            
            result[t] = value;
        }
        
        result
    }
    
    fn get_theta_dimension(basis_type: &BasisType, params: &BasisParameters<T>) -> usize {
        match basis_type {
            BasisType::Generic => 64, // Default generic basis size
            BasisType::Trend => params.polynomial_degree.unwrap_or(3) + 1,
            BasisType::Seasonality => 2 * params.harmonics.as_ref()
                .map(|h| h.len())
                .unwrap_or(4), // Default 4 harmonics
        }
    }
    
    fn get_forecast_length(&self) -> usize {
        // This should be configured based on forecasting horizon
        24 // Default forecast horizon
    }
}

struct NBEATSStack<T: Float> {
    blocks: Vec<NBEATSBlock<T>>,
    stack_type: BasisType,
}

impl<T: Float> NBEATSStack<T> {
    fn new(num_blocks: usize, 
           input_size: usize, 
           output_size: usize,
           hidden_size: usize,
           stack_type: BasisType) -> Self {
        
        let basis_params = BasisParameters {
            polynomial_degree: Some(3),
            harmonics: Some(vec![T::one(); 4]),
            trend_coefficients: None,
        };
        
        let blocks = (0..num_blocks)
            .map(|_| NBEATSBlock::new(input_size, output_size, hidden_size, 
                                    stack_type.clone(), basis_params.clone()))
            .collect();
        
        Self { blocks, stack_type }
    }
    
    fn forward(&mut self, mut input: Vec<T>) -> (Vec<T>, Vec<T>) {
        let mut stack_forecast = vec![T::zero(); self.get_forecast_length()];
        
        for block in &mut self.blocks {
            let (backcast, forecast) = block.forward(&input);
            
            // Update residual input for next block
            input = input.iter()
                .zip(backcast.iter())
                .map(|(&inp, &back)| inp - back)
                .collect();
            
            // Accumulate forecasts
            for (i, &f) in forecast.iter().enumerate() {
                if i < stack_forecast.len() {
                    stack_forecast[i] = stack_forecast[i] + f;
                }
            }
        }
        
        (input, stack_forecast) // Return residual and forecast
    }
    
    fn get_forecast_length(&self) -> usize {
        24 // Default forecast horizon
    }
}

struct NBEATSForecaster<T: Float> {
    stacks: Vec<NBEATSStack<T>>,
}

impl<T: Float> NBEATSForecaster<T> {
    fn new(config: NBEATSConfig) -> Self {
        let mut stacks = Vec::new();
        
        // Generic stacks
        for _ in 0..config.num_generic_stacks {
            stacks.push(NBEATSStack::new(
                config.num_blocks_per_stack,
                config.input_size,
                config.output_size,
                config.hidden_size,
                BasisType::Generic,
            ));
        }
        
        // Interpretable stacks
        if config.use_trend_stack {
            stacks.push(NBEATSStack::new(
                config.num_blocks_per_stack,
                config.input_size,
                config.output_size,
                config.hidden_size,
                BasisType::Trend,
            ));
        }
        
        if config.use_seasonality_stack {
            stacks.push(NBEATSStack::new(
                config.num_blocks_per_stack,
                config.input_size,
                config.output_size,
                config.hidden_size,
                BasisType::Seasonality,
            ));
        }
        
        Self { stacks }
    }
    
    fn forecast(&mut self, input: &[T]) -> Vec<T> {
        let mut residual = input.to_vec();
        let mut total_forecast = vec![T::zero(); self.get_forecast_length()];
        
        for stack in &mut self.stacks {
            let (new_residual, stack_forecast) = stack.forward(residual);
            residual = new_residual;
            
            // Accumulate forecasts from all stacks
            for (i, &f) in stack_forecast.iter().enumerate() {
                if i < total_forecast.len() {
                    total_forecast[i] = total_forecast[i] + f;
                }
            }
        }
        
        total_forecast
    }
    
    fn get_forecast_length(&self) -> usize {
        24 // Default forecast horizon
    }
}

struct NBEATSConfig {
    num_generic_stacks: usize,
    num_blocks_per_stack: usize,
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    use_trend_stack: bool,
    use_seasonality_stack: bool,
}
```

**Implementation Complexity**: High
**Required Extensions**: Basis function implementations, residual learning
**Key Features**:
- Multiple basis function types
- Stack-based architecture
- Residual learning between blocks
- Interpretable trend/seasonality stacks

### 2.6 Advanced Models

#### 2.6.1 TimesNet (Simplified)
```rust
// This requires complex FFT operations - simplified version
struct TimesNetForecaster<T: Float> {
    inception_blocks: Vec<InceptionBlock<T>>,
    embedding_layer: Network<T>,
    projection_layer: Network<T>,
    top_k_periods: usize,
}

struct InceptionBlock<T: Float> {
    conv_layers: Vec<Network<T>>, // Simulated convolution with MLPs
    activation: ActivationFunction,
}

impl<T: Float> InceptionBlock<T> {
    fn new(input_size: usize, output_size: usize) -> Self {
        // Multiple "convolution" layers with different kernel sizes
        // Simulated as MLPs with different window processing
        let conv_layers = vec![
            // 1x1 "convolution"
            NetworkBuilder::new()
                .input_layer(input_size)
                .output_layer(output_size / 4)
                .build(),
            // 3x1 "convolution" - process 3 consecutive time steps
            NetworkBuilder::new()
                .input_layer(input_size * 3)
                .output_layer(output_size / 4)
                .build(),
            // 5x1 "convolution"
            NetworkBuilder::new()
                .input_layer(input_size * 5)
                .output_layer(output_size / 4)
                .build(),
            // MaxPool + 1x1
            NetworkBuilder::new()
                .input_layer(input_size)
                .output_layer(output_size / 4)
                .build(),
        ];
        
        Self {
            conv_layers,
            activation: ActivationFunction::ReLU,
        }
    }
    
    fn forward(&mut self, input_sequence: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut outputs = Vec::new();
        
        // Process with different "kernel sizes"
        for (i, layer) in self.conv_layers.iter_mut().enumerate() {
            match i {
                0 => {
                    // 1x1 - process each timestep individually
                    for input in input_sequence {
                        outputs.push(layer.run(input));
                    }
                },
                1 => {
                    // 3x1 - process 3 consecutive timesteps
                    for window in input_sequence.windows(3) {
                        let combined: Vec<T> = window.iter().flatten().cloned().collect();
                        outputs.push(layer.run(&combined));
                    }
                },
                2 => {
                    // 5x1 - process 5 consecutive timesteps
                    for window in input_sequence.windows(5) {
                        let combined: Vec<T> = window.iter().flatten().cloned().collect();
                        outputs.push(layer.run(&combined));
                    }
                },
                3 => {
                    // MaxPool simulation - take max over windows
                    for window in input_sequence.windows(2) {
                        let pooled = self.max_pool(window);
                        outputs.push(layer.run(&pooled));
                    }
                },
                _ => {}
            }
        }
        
        outputs
    }
    
    fn max_pool(&self, window: &[Vec<T>]) -> Vec<T> {
        if window.is_empty() {
            return Vec::new();
        }
        
        let feature_dim = window[0].len();
        let mut pooled = vec![T::neg_infinity(); feature_dim];
        
        for timestep in window {
            for (i, &value) in timestep.iter().enumerate() {
                if value > pooled[i] {
                    pooled[i] = value;
                }
            }
        }
        
        pooled
    }
}

impl<T: Float> TimesNetForecaster<T> {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let embedding_layer = NetworkBuilder::new()
            .input_layer(input_dim)
            .hidden_layer(hidden_dim)
            .output_layer(hidden_dim)
            .build();
        
        let inception_blocks = vec![
            InceptionBlock::new(hidden_dim, hidden_dim),
            InceptionBlock::new(hidden_dim, hidden_dim),
        ];
        
        let projection_layer = NetworkBuilder::new()
            .input_layer(hidden_dim)
            .output_layer(output_dim)
            .build();
        
        Self {
            inception_blocks,
            embedding_layer,
            projection_layer,
            top_k_periods: 5,
        }
    }
    
    fn forecast(&mut self, input_sequence: &[Vec<T>]) -> Vec<T> {
        // 1. Embedding
        let mut embedded_sequence = Vec::new();
        for input in input_sequence {
            embedded_sequence.push(self.embedding_layer.run(input));
        }
        
        // 2. Period detection (simplified - skip FFT)
        let periods = self.detect_periods(&embedded_sequence);
        
        // 3. Process through inception blocks
        let mut processed = embedded_sequence;
        for inception_block in &mut self.inception_blocks {
            processed = inception_block.forward(&processed);
        }
        
        // 4. Final projection
        let final_representation = processed.last().unwrap();
        self.projection_layer.run(final_representation)
    }
    
    fn detect_periods(&self, sequence: &[Vec<T>]) -> Vec<usize> {
        // Simplified period detection without FFT
        // In practice, this would use FFT to find dominant frequencies
        vec![7, 30, 365] // Common periods: weekly, monthly, yearly
    }
}
```

**Implementation Complexity**: Very High
**Required Extensions**: FFT operations, complex 2D convolution simulation
**Key Features**:
- Period detection (requires FFT)
- Inception block architecture
- Multi-scale temporal processing

## 3. Training and Inference Pipeline Framework

### 3.1 Universal Training Pipeline
```rust
trait TimeSeriesForecaster<T: Float> {
    fn forecast(&mut self, input: &TimeSeriesInput<T>) -> Vec<T>;
    fn train_step(&mut self, data: &TrainingData<T>) -> Result<T, TrainingError>;
    fn validate(&mut self, data: &TrainingData<T>) -> T;
    fn save_state(&self) -> Vec<u8>;
    fn load_state(&mut self, state: &[u8]) -> Result<(), Box<dyn std::error::Error>>;
}

struct TimeSeriesInput<T: Float> {
    historical_targets: Vec<T>,
    static_features: Option<Vec<T>>,
    historical_features: Option<Vec<Vec<T>>>,
    future_features: Option<Vec<Vec<T>>>,
}

struct ForecastingPipeline<T: Float> {
    preprocessor: TimeSeriesPreprocessor<T>,
    model: Box<dyn TimeSeriesForecaster<T>>,
    postprocessor: TimeSeriesPostprocessor<T>,
}

impl<T: Float> ForecastingPipeline<T> {
    fn new(model: Box<dyn TimeSeriesForecaster<T>>) -> Self {
        Self {
            preprocessor: TimeSeriesPreprocessor::new(),
            model,
            postprocessor: TimeSeriesPostprocessor::new(),
        }
    }
    
    fn train(&mut self, 
             raw_data: &RawTimeSeriesData<T>, 
             config: &TrainingConfig) -> Result<TrainingHistory<T>, TrainingError> {
        let processed_data = self.preprocessor.fit_transform(raw_data)?;
        let training_data = self.create_training_data(&processed_data, config)?;
        
        let mut history = TrainingHistory::new();
        
        for epoch in 0..config.max_epochs {
            let train_error = self.model.train_step(&training_data)?;
            let val_error = if let Some(val_data) = &config.validation_data {
                Some(self.model.validate(val_data))
            } else {
                None
            };
            
            history.add_epoch(epoch, train_error, val_error);
            
            // Early stopping check
            if let Some(early_stopping) = &config.early_stopping {
                if early_stopping.should_stop(&history) {
                    break;
                }
            }
        }
        
        Ok(history)
    }
    
    fn forecast(&mut self, raw_input: &RawTimeSeriesData<T>) -> Result<Vec<T>, Box<dyn std::error::Error>> {
        let processed_input = self.preprocessor.transform(raw_input)?;
        let ts_input = self.create_forecasting_input(&processed_input);
        let raw_forecast = self.model.forecast(&ts_input);
        let final_forecast = self.postprocessor.inverse_transform(&raw_forecast)?;
        Ok(final_forecast)
    }
}

struct TimeSeriesPreprocessor<T: Float> {
    scalers: HashMap<String, MinMaxScaler<T>>,
    differencing_orders: HashMap<String, usize>,
    seasonal_decomposition: Option<SeasonalDecomposer<T>>,
}

struct TrainingConfig {
    max_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    validation_data: Option<TrainingData<f64>>,
    early_stopping: Option<EarlyStoppingConfig>,
}

struct TrainingHistory<T: Float> {
    epochs: Vec<usize>,
    train_errors: Vec<T>,
    validation_errors: Vec<Option<T>>,
}
```

### 3.2 Model-Specific Training Strategies

```rust
// Specific training implementations for different model types

impl<T: Float> TrainingAlgorithm<T> for MLPForecaster<T> {
    fn train_epoch(&mut self, network: &mut Network<T>, data: &TrainingData<T>) -> Result<T, TrainingError> {
        // Standard backpropagation
        let mut total_error = T::zero();
        let batch_size = data.inputs.len();
        
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            let output = network.run(input);
            let error = self.calculate_sample_error(&output, target);
            total_error = total_error + error;
            
            // Backpropagate error
            self.backpropagate(network, input, target, &output)?;
        }
        
        Ok(total_error / T::from(batch_size).unwrap())
    }
}

impl<T: Float> TrainingAlgorithm<T> for LSTMForecaster<T> {
    fn train_epoch(&mut self, network: &mut Network<T>, data: &TrainingData<T>) -> Result<T, TrainingError> {
        // Backpropagation through time (BPTT)
        let mut total_error = T::zero();
        
        for sequence_batch in data.sequence_batches() {
            // Reset LSTM states
            self.reset_states();
            
            // Forward pass through sequence
            let mut sequence_outputs = Vec::new();
            for input in &sequence_batch.inputs {
                let output = self.forward_step(input);
                sequence_outputs.push(output);
            }
            
            // Calculate sequence error
            let sequence_error = self.calculate_sequence_error(&sequence_outputs, &sequence_batch.targets);
            total_error = total_error + sequence_error;
            
            // BPTT
            self.backpropagate_through_time(&sequence_batch)?;
        }
        
        Ok(total_error / T::from(data.num_sequences()).unwrap())
    }
    
    fn backpropagate_through_time(&mut self, sequence: &SequenceBatch<T>) -> Result<(), TrainingError> {
        // Implement BPTT algorithm
        // This involves computing gradients backward through time
        // and updating LSTM cell parameters
        todo!("Implement BPTT")
    }
}

impl<T: Float> TrainingAlgorithm<T> for NBEATSForecaster<T> {
    fn train_epoch(&mut self, network: &mut Network<T>, data: &TrainingData<T>) -> Result<T, TrainingError> {
        // NBEATS uses residual learning approach
        let mut total_error = T::zero();
        
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            let mut residual = input.clone();
            let mut forecast_accumulator = vec![T::zero(); target.len()];
            
            // Forward through each stack
            for stack in &mut self.stacks {
                let (new_residual, stack_forecast) = stack.forward(residual);
                residual = new_residual;
                
                // Accumulate forecasts
                for (i, &f) in stack_forecast.iter().enumerate() {
                    if i < forecast_accumulator.len() {
                        forecast_accumulator[i] = forecast_accumulator[i] + f;
                    }
                }
            }
            
            let error = self.calculate_sample_error(&forecast_accumulator, target);
            total_error = total_error + error;
            
            // Specialized NBEATS backpropagation
            self.backpropagate_nbeats(input, target, &forecast_accumulator)?;
        }
        
        Ok(total_error / T::from(data.inputs.len()).unwrap())
    }
}
```

## 4. Implementation Priority and Roadmap

### 4.1 Phase 1: Basic Models (Low Complexity)
1. **MLP & MLPMultivariate**: Direct implementation using NetworkBuilder
2. **DLinear & NLinear**: Simple linear models with preprocessing
3. **Basic RNN**: Simple recurrent layer implementation

### 4.2 Phase 2: Intermediate Models (Medium Complexity)
1. **LSTM & GRU**: Gate-based recurrent models
2. **NBEATS family**: Basis function expansion models
3. **TSMixer family**: MLP-based mixing models

### 4.3 Phase 3: Advanced Models (High Complexity)
1. **Basic Transformer components**: Attention mechanism simulation
2. **TFT**: Temporal fusion transformer
3. **TimesNet**: 2D temporal modeling (simplified)

### 4.4 Phase 4: Specialized Models (Very High Complexity)
1. **Advanced Transformers**: Informer, AutoFormer, FedFormer
2. **Graph-based models**: StemGNN
3. **LLM integration**: TimeLLM (extremely resource-intensive)

## 5. Code Organization Structure

```
src/
├── forecasting/
│   ├── mod.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── mlp.rs
│   │   ├── linear.rs
│   │   ├── rnn.rs
│   │   ├── lstm.rs
│   │   ├── gru.rs
│   │   ├── nbeats.rs
│   │   ├── transformer.rs
│   │   └── specialized/
│   │       ├── mod.rs
│   │       ├── timesnet.rs
│   │       ├── stemgnn.rs
│   │       └── timellm.rs
│   ├── layers/
│   │   ├── mod.rs
│   │   ├── recurrent.rs
│   │   ├── attention.rs
│   │   ├── convolution.rs
│   │   └── basis_functions.rs
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── normalization.rs
│   │   ├── decomposition.rs
│   │   └── feature_engineering.rs
│   ├── training/
│   │   ├── mod.rs
│   │   ├── schedulers.rs
│   │   ├── early_stopping.rs
│   │   └── metrics.rs
│   └── utils/
│       ├── mod.rs
│       ├── time_series.rs
│       └── math_ops.rs
```

## 6. Testing and Validation Strategy

### 6.1 Unit Tests for Each Model
- Test individual model components
- Verify mathematical correctness
- Test edge cases and error handling

### 6.2 Integration Tests
- End-to-end forecasting pipeline tests
- Cross-model comparison tests
- Performance benchmarking

### 6.3 Validation on Standard Datasets
- M4 Competition dataset
- ETT (Electricity Transforming Temperature) dataset
- Various domain-specific datasets

This comprehensive implementation plan provides a roadmap for implementing all 27 neural forecasting models using ruv-FANN's building blocks, with clear priorities, architectural strategies, and code organization principles.