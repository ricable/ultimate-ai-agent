# Recurrent Models

Recurrent models excel at capturing temporal dependencies and sequential patterns in time series data. These models maintain internal state and can theoretically capture arbitrarily long dependencies, making them ideal for complex temporal forecasting tasks.

## LSTM (Long Short-Term Memory)

LSTM networks are the most popular recurrent architecture for time series forecasting, designed to capture long-term dependencies while avoiding the vanishing gradient problem.

### Architecture

LSTM cells contain three gates that control information flow:
- **Forget Gate**: Decides what information to discard from cell state
- **Input Gate**: Determines what new information to store in cell state  
- **Output Gate**: Controls what parts of cell state to output

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct LSTMConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub bidirectional: bool,
    pub use_bias: bool,
    pub batch_first: bool,
}
```

### Usage Examples

#### Basic LSTM

```rust
use neuro_divergent::models::{LSTM, LSTMConfig};

let lstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .dropout(0.1)
    .build()?;
```

#### Deep LSTM for Complex Patterns

```rust
let deep_lstm = LSTM::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(128)
    .num_layers(4)       // Deep network
    .dropout(0.2)        // More regularization
    .bidirectional(false) // Unidirectional for forecasting
    .build()?;
```

#### Bidirectional LSTM for Analysis

```rust
// For feature extraction or analysis (not direct forecasting)
let bilstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .bidirectional(true)  // Access to future context
    .dropout(0.15)
    .build()?;
```

#### LSTM for Financial Data

```rust
// Configuration optimized for financial time series
let financial_lstm = LSTM::builder()
    .horizon(5)           // 5-day forecast
    .input_size(60)       // 3 months of daily data
    .hidden_size(128)     // Sufficient capacity
    .num_layers(3)        // Capture complex patterns
    .dropout(0.3)         // High regularization for noisy data
    .use_bias(true)
    .build()?;
```

### Configuration Parameters

- **`horizon`** (required): Number of future steps to forecast
- **`input_size`** (required): Length of input sequence
- **`hidden_size`** (default: 64): Size of hidden state vector
- **`num_layers`** (default: 2): Number of LSTM layers
- **`dropout`** (default: 0.1): Dropout rate between layers
- **`bidirectional`** (default: false): Whether to use bidirectional LSTM
- **`use_bias`** (default: true): Whether to use bias parameters
- **`batch_first`** (default: true): Input tensor format (batch, seq, features)

### Performance Characteristics

- **Training Time**: O(n·h²) - Quadratic in hidden size
- **Inference Time**: O(h²) - Sequential processing
- **Memory Usage**: High - Stores cell states and hidden states
- **Best For**: Long sequences with complex temporal patterns

### Strengths

- **Long-Term Memory**: Can capture dependencies across long sequences
- **Proven Architecture**: Well-studied and reliable
- **Flexible**: Works well across many domains
- **Interpretable States**: Hidden states can provide insights

### Limitations

- **Sequential Processing**: Cannot be parallelized across time steps
- **Computational Cost**: Expensive for long sequences
- **Vanishing Gradients**: Still possible with very long sequences

## RNN (Recurrent Neural Network)

Basic recurrent neural network, simpler than LSTM but faster to train and suitable for shorter sequences.

### Architecture

Simple recurrent structure:
- Hidden state updated at each time step
- Output depends on current input and previous hidden state
- Single recurrent connection

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct RNNConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub activation: ActivationType,
    pub use_bias: bool,
}
```

### Usage Examples

#### Basic RNN

```rust
let rnn = RNN::builder()
    .horizon(3)          // Short horizon
    .input_size(12)      // Short input sequence  
    .hidden_size(32)     // Modest hidden size
    .num_layers(1)       // Single layer
    .activation(ActivationType::Tanh)
    .build()?;
```

#### Stacked RNN

```rust
let stacked_rnn = RNN::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(3)       // Multiple layers
    .dropout(0.2)        // Regularization
    .activation(ActivationType::ReLU)
    .build()?;
```

### Configuration Parameters

- **`activation`** (default: Tanh): Activation function (Tanh, ReLU, Sigmoid)
- Other parameters similar to LSTM

### Performance Characteristics

- **Training Time**: O(n·h²) - Faster than LSTM
- **Memory Usage**: Lower than LSTM
- **Best For**: Short sequences, quick experimentation

### Strengths

- **Simple**: Easy to understand and implement
- **Fast**: Fewer parameters than LSTM
- **Memory Efficient**: Lower memory requirements

### Limitations

- **Vanishing Gradients**: Severe problems with long sequences
- **Limited Memory**: Cannot capture long-term dependencies
- **Less Powerful**: Generally inferior to LSTM for complex tasks

## GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM with fewer parameters, offering a good balance between performance and computational efficiency.

### Architecture

GRU has two gates:
- **Reset Gate**: Controls how much past information to forget
- **Update Gate**: Controls how much of the new state to be added

### Configuration

```rust
#[derive(Debug, Clone)]  
pub struct GRUConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub bidirectional: bool,
    pub use_bias: bool,
}
```

### Usage Examples

#### Standard GRU

```rust
let gru = GRU::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .dropout(0.1)
    .build()?;
```

#### GRU for Fast Prototyping

```rust
// Faster alternative to LSTM
let fast_gru = GRU::builder()
    .horizon(14)
    .input_size(56)
    .hidden_size(96)     // Fewer parameters than equivalent LSTM
    .num_layers(2)
    .dropout(0.15)
    .build()?;
```

### Performance Characteristics

- **Training Time**: Faster than LSTM (fewer parameters)
- **Memory Usage**: Lower than LSTM
- **Performance**: Often comparable to LSTM
- **Best For**: When computational efficiency is important

### Strengths

- **Efficient**: Fewer parameters than LSTM
- **Good Performance**: Often matches LSTM performance
- **Simpler**: Easier to tune than LSTM

### Limitations

- **Less Expressive**: Fewer parameters than LSTM
- **Recent Architecture**: Less extensively studied than LSTM

## BiLSTM (Bidirectional LSTM)

Bidirectional LSTM processes sequences in both forward and backward directions, providing access to future context.

### Architecture

- **Forward LSTM**: Processes sequence from start to end
- **Backward LSTM**: Processes sequence from end to start
- **Concatenation**: Combines forward and backward hidden states

### Configuration

```rust
// BiLSTM is configured using LSTM with bidirectional=true
let bilstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .bidirectional(true)  // Enable bidirectional processing
    .dropout(0.1)
    .build()?;
```

### Usage Examples

#### Feature Extraction

```rust
// For extracting features from time series
let feature_extractor = LSTM::builder()
    .horizon(1)           // Single output
    .input_size(100)      // Long sequence
    .hidden_size(128)
    .num_layers(2)
    .bidirectional(true)  // Full context
    .dropout(0.2)
    .build()?;
```

#### Sequence Labeling

```rust
// For tasks that require full sequence context
let sequence_model = LSTM::builder()
    .horizon(28)          // Same length as input
    .input_size(28)
    .hidden_size(64)
    .bidirectional(true)
    .build()?;
```

### Performance Characteristics

- **Memory Usage**: Double that of unidirectional LSTM
- **Training Time**: Longer than unidirectional
- **Best For**: Analysis tasks where future context is available

### Strengths

- **Full Context**: Access to both past and future information
- **Better Representations**: Often produces better feature representations
- **Versatile**: Useful for many sequence tasks

### Limitations

- **Not Suitable for Forecasting**: Requires future data
- **Higher Cost**: Double the computational requirements
- **Complex Training**: More parameters to optimize

## Training and Optimization

### Training Configuration for Recurrent Models

```rust
use neuro_divergent::training::{TrainingConfig, OptimizerConfig, SchedulerConfig};

let training_config = TrainingConfig::builder()
    .max_epochs(200)     // RNNs often need more epochs
    .batch_size(32)      // Moderate batch size
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::Adam {
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01  // L2 regularization
    })
    .scheduler(SchedulerConfig::StepLR {
        step_size: 50,
        gamma: 0.5         // Reduce LR during training
    })
    .gradient_clipping(1.0)  // Important for RNNs
    .build()?;
```

### Gradient Clipping

```rust
// Essential for stable RNN training
let training_config = TrainingConfig::builder()
    .gradient_clipping(1.0)      // Clip gradients to norm of 1.0
    .gradient_clipping_type(ClippingType::Norm)  // L2 norm clipping
    .build()?;
```

### Learning Rate Scheduling

```rust
// Adaptive learning rate for RNNs
let scheduler_config = SchedulerConfig::ReduceLROnPlateau {
    patience: 10,
    factor: 0.5,
    min_lr: 1e-6,
    threshold: 0.01,
};
```

## Advanced Techniques

### Attention Mechanisms

```rust
// LSTM with attention for long sequences
let lstm_attention = LSTM::builder()
    .horizon(12)
    .input_size(168)     // Weekly data
    .hidden_size(128)
    .num_layers(2)
    .use_attention(true) // Enable attention mechanism
    .attention_heads(8)  // Multi-head attention
    .build()?;
```

### Residual Connections

```rust
// Deep LSTM with residual connections
let residual_lstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(6)       // Deep network
    .use_residual(true)  // Skip connections
    .dropout(0.2)
    .build()?;
```

### Sequence-to-Sequence

```rust
// Encoder-decoder architecture
let seq2seq_config = LSTMSeq2SeqConfig::builder()
    .encoder_hidden_size(128)
    .decoder_hidden_size(128)
    .encoder_layers(2)
    .decoder_layers(2)
    .horizon(12)
    .input_size(36)
    .build()?;

let seq2seq_model = LSTMSeq2Seq::new(seq2seq_config)?;
```

## Model Comparison

### Performance Characteristics

| Model | Training Speed | Memory Usage | Long-term Memory | Parameters |
|-------|---------------|--------------|------------------|------------|
| RNN | Fast | Low | Poor | Fewest |
| GRU | Medium | Medium | Good | Medium |
| LSTM | Slow | High | Excellent | Most |
| BiLSTM | Slowest | Highest | Excellent | Double |

### Accuracy Benchmarks

Typical performance on forecasting tasks:

#### M4 Competition (Average sMAPE)
- **LSTM**: 13.1
- **GRU**: 13.3  
- **RNN**: 14.2
- **BiLSTM**: 12.9 (analysis tasks)

#### Financial Data (Daily Returns MAE)
- **LSTM**: 0.081
- **GRU**: 0.083
- **RNN**: 0.089

### Selection Guidelines

```rust
fn select_recurrent_model(data_characteristics: &DataInfo) -> Box<dyn BaseModel<f64>> {
    match data_characteristics {
        // Long sequences with complex patterns
        DataInfo { sequence_length: len, complexity: high, .. } if *len > 100 && *high => {
            Box::new(LSTM::builder()
                .horizon(data_characteristics.horizon)
                .input_size(*len)
                .hidden_size(128)
                .num_layers(3)
                .dropout(0.2)
                .build().unwrap())
        },
        
        // Medium sequences, efficiency important
        DataInfo { sequence_length: len, .. } if *len > 50 => {
            Box::new(GRU::builder()
                .horizon(data_characteristics.horizon)
                .input_size(*len)
                .hidden_size(96)
                .num_layers(2)
                .build().unwrap())
        },
        
        // Short sequences, simple patterns
        DataInfo { sequence_length: len, .. } if *len <= 50 => {
            Box::new(RNN::builder()
                .horizon(data_characteristics.horizon)
                .input_size(*len)
                .hidden_size(64)
                .num_layers(1)
                .build().unwrap())
        },
        
        // Default to LSTM
        _ => {
            Box::new(LSTM::builder()
                .horizon(data_characteristics.horizon)
                .hidden_size(64)
                .num_layers(2)
                .build().unwrap())
        }
    }
}
```

## Best Practices

### Architecture Design

```rust
// Guidelines for LSTM architecture
fn design_lstm_architecture(data_size: usize, sequence_length: usize) -> LSTMConfig {
    let hidden_size = match data_size {
        n if n < 1000 => 32,      // Small dataset
        n if n < 10000 => 64,     // Medium dataset  
        _ => 128,                 // Large dataset
    };
    
    let num_layers = match sequence_length {
        n if n < 50 => 1,         // Short sequences
        n if n < 200 => 2,        // Medium sequences
        _ => 3,                   // Long sequences
    };
    
    let dropout = match data_size {
        n if n < 5000 => 0.1,     // Less regularization
        _ => 0.2,                 // More regularization
    };
    
    LSTMConfig::builder()
        .hidden_size(hidden_size)
        .num_layers(num_layers)
        .dropout(dropout)
        .build()
        .unwrap()
}
```

### Hyperparameter Tuning

```rust
// Systematic hyperparameter search for LSTM
let hidden_sizes = vec![32, 64, 128, 256];
let num_layers_options = vec![1, 2, 3, 4];
let dropout_rates = vec![0.0, 0.1, 0.2, 0.3];

let mut best_config = None;
let mut best_score = f64::INFINITY;

for &hidden_size in &hidden_sizes {
    for &num_layers in &num_layers_options {
        for &dropout in &dropout_rates {
            let config = LSTMConfig::builder()
                .horizon(7)
                .input_size(28)
                .hidden_size(hidden_size)
                .num_layers(num_layers)
                .dropout(dropout)
                .build()?;
                
            let score = evaluate_config(&config, &data)?;
            
            if score < best_score {
                best_score = score;
                best_config = Some(config);
            }
        }
    }
}
```

### Data Preprocessing

```rust
// Preprocessing for recurrent models
use neuro_divergent::preprocessing::{StandardScaler, SequenceNormalizer};

// Standardize features
let mut scaler = StandardScaler::new();
let scaled_data = scaler.fit_transform(&data)?;

// Normalize sequences (important for RNNs)
let mut seq_normalizer = SequenceNormalizer::new();
let normalized_data = seq_normalizer.fit_transform(&scaled_data)?;

// Create sequences with proper windowing
let sequences = create_sequences(&normalized_data, input_size, horizon)?;
```

### Regularization Strategies

```rust
// Multiple regularization techniques
let lstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .dropout(0.2)              // Dropout regularization
    .weight_decay(0.01)        // L2 regularization
    .use_layer_norm(true)      // Layer normalization
    .use_gradient_clipping(1.0) // Gradient clipping
    .build()?;
```

## Integration Examples

### Time Series Forecasting Pipeline

```rust
use neuro_divergent::prelude::*;

// Complete LSTM forecasting pipeline
let data = TimeSeriesDataFrame::from_csv("timeseries.csv")?;

// Preprocess data
let preprocessor = StandardScaler::new();
let processed_data = preprocessor.fit_transform(&data)?;

// Create LSTM model
let lstm = LSTM::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(128)
    .num_layers(3)
    .dropout(0.2)
    .build()?;

// Train model
let mut nf = NeuralForecast::builder()
    .with_model(Box::new(lstm))
    .with_frequency(Frequency::Monthly)
    .build()?;

nf.fit(processed_data)?;

// Generate forecasts
let forecasts = nf.predict()?;
```

### Multi-Model Ensemble

```rust
// Ensemble of different recurrent models
let models = vec![
    Box::new(LSTM::builder()
        .horizon(7)
        .hidden_size(64)
        .num_layers(2)
        .build()?) as Box<dyn BaseModel<f64>>,
        
    Box::new(GRU::builder()
        .horizon(7)
        .hidden_size(96)
        .num_layers(2)
        .build()?) as Box<dyn BaseModel<f64>>,
        
    Box::new(RNN::builder()
        .horizon(7)
        .hidden_size(64)
        .num_layers(1)
        .activation(ActivationType::ReLU)
        .build()?) as Box<dyn BaseModel<f64>>,
];

let ensemble = NeuralForecast::builder()
    .with_models(models)
    .build()?;
```

### Production Deployment

```rust
// Optimized LSTM for production
let production_lstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)          // Balanced size
    .num_layers(2)            // Not too deep
    .dropout(0.0)             // No dropout for inference
    .use_optimized_inference(true)  // Optimize for speed
    .device(Device::GPU(0))   // GPU acceleration
    .build()?;

// Save model
production_lstm.save("production_lstm.json")?;

// Load and use
let loaded_lstm = LSTM::load("production_lstm.json")?;
let forecasts = loaded_lstm.predict(&new_data)?;
```

Recurrent models provide powerful capabilities for capturing temporal patterns in time series data, with LSTM being the most versatile choice for most applications, GRU offering efficiency benefits, and RNN suitable for simple tasks or prototyping.