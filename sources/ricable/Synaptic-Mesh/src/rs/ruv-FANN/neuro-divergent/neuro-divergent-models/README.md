# Neuro-Divergent Models

A comprehensive neural forecasting library built on the ruv-FANN foundation, providing state-of-the-art time series forecasting models optimized for Rust's performance and safety guarantees.

## Technical Introduction

Neuro-Divergent Models is a neural forecasting model library that implements 27+ neural network architectures specifically designed for time series forecasting. Built on top of the ruv-FANN (Fast Artificial Neural Network) library, it provides a unified, type-safe, and performant interface for deploying neural forecasting models in production environments.

The library bridges the gap between research-focused implementations and production-ready systems by offering:

- **Memory-efficient implementations** with zero-copy data operations where possible
- **Type-safe generic architecture** supporting f32 and f64 precision
- **Unified trait-based API** for consistent model interfaces
- **Built-in data preprocessing** and validation pipelines
- **Configurable training algorithms** from ruv-FANN's optimization suite
- **Production-ready serialization** for model persistence and deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Neuro-Divergent Models                   │
├─────────────────────────────────────────────────────────────┤
│  BaseModel Trait │ ModelConfig Trait │ NetworkAdapter     │
├─────────────────────────────────────────────────────────────┤
│               ruv-FANN Foundation                          │
│  Network Builder │ Training Algorithms │ Activation Funcs  │
├─────────────────────────────────────────────────────────────┤
│  Memory Management │ Error Handling │ Parallel Processing │
└─────────────────────────────────────────────────────────────┘
```

## Model Categories

### Basic Models (4 implementations)

#### 1. MLP (Multi-Layer Perceptron)
**Mathematical Foundation**: Standard feedforward neural network with configurable hidden layers.

```
Input → [Linear → Activation] × N → Linear → Output
```

**Configuration**:
```rust
MLPConfig::new(input_size: 24, horizon: 12)
    .with_hidden_layers(vec![128, 64, 32])
    .with_activation(ActivationFunction::ReLU)
    .with_learning_rate(0.001)
```

**Best For**: General-purpose forecasting, non-linear patterns, feature-rich datasets.

#### 2. DLinear (Direct Linear)
**Mathematical Foundation**: Direct linear mapping with seasonal-trend decomposition.

```
Y = Linear(Trend) + Linear(Seasonal) + Linear(Residual)
```

**Configuration**:
```rust
DLinearConfig::new(input_size: 168, horizon: 24)
    .with_kernel_size(25)  // Moving average kernel
    .with_individual_weights(true)
```

**Best For**: Long sequences, linear trends, computationally constrained environments.

#### 3. NLinear (Normalized Linear)
**Mathematical Foundation**: Normalized linear model with subtraction normalization.

```
Y = Linear(X - Last(X)) + Last(X)
```

**Configuration**:
```rust
NLinearConfig::new(input_size: 96, horizon: 24)
    .with_normalization_type(NormalizationType::LastValue)
```

**Best For**: Data with distribution shifts, simple baselines, interpretable models.

#### 4. MLPMultivariate
**Mathematical Foundation**: Multi-layer perceptron with multivariate input handling.

```
Input[N_vars, T] → Flatten → [Linear → Activation] × N → Output[H]
```

**Configuration**:
```rust
MLPMultivariateConfig::new(n_variables: 7, input_size: 24, horizon: 12)
    .with_shared_weights(false)
    .with_variable_selection(true)
```

**Best For**: Multiple time series, cross-series dependencies, feature engineering.

### Recurrent Models (3 implementations)

#### 1. RNN (Recurrent Neural Network)
**Mathematical Foundation**: Basic recurrent cell with hidden state memory.

```
h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
```

**Configuration**:
```rust
RNNConfig::default_with_horizon(24)
    .with_architecture(hidden_size: 128, num_layers: 2, dropout: 0.1)
    .with_training(max_steps: 1000, learning_rate: 0.001)
```

**Best For**: Simple sequential patterns, educational purposes, baseline comparisons.

#### 2. LSTM (Long Short-Term Memory)
**Mathematical Foundation**: LSTM cell with forget, input, and output gates.

```
f_t = σ(W_f * [h_{t-1}, x_t] + b_f)  // Forget gate
i_t = σ(W_i * [h_{t-1}, x_t] + b_i)  // Input gate
C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)  // Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t  // Cell state
o_t = σ(W_o * [h_{t-1}, x_t] + b_o)  // Output gate
h_t = o_t * tanh(C_t)  // Hidden state
```

**Configuration**:
```rust
LSTMConfig::default_with_horizon(48)
    .with_architecture(hidden_size: 256, num_layers: 3, dropout: 0.2)
    .with_training(max_steps: 2000, learning_rate: 0.0005)
    .with_gradient_clipping(1.0)
```

**Best For**: Long sequences, complex temporal dependencies, vanishing gradient problems.

#### 3. GRU (Gated Recurrent Unit)
**Mathematical Foundation**: Simplified gating mechanism with reset and update gates.

```
z_t = σ(W_z * [h_{t-1}, x_t])  // Update gate
r_t = σ(W_r * [h_{t-1}, x_t])  // Reset gate
h̃_t = tanh(W * [r_t * h_{t-1}, x_t])  // New gate
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  // Hidden state
```

**Configuration**:
```rust
GRUConfig::default_with_horizon(24)
    .with_architecture(hidden_size: 128, num_layers: 2, dropout: 0.15)
    .with_bidirectional(true)
```

**Best For**: Medium-length sequences, computational efficiency, comparable to LSTM performance.

### Advanced Models (4 implementations)

#### 1. NBEATS (Neural Basis Expansion Analysis)
**Mathematical Foundation**: Hierarchical doubly residual architecture with interpretable basis functions.

```
Stack: [Block₁, Block₂, ..., Blockₙ] → Forecast + Backcast
Block: FC → Basis_expansion → [θ_f, θ_b] → [Forecast, Backcast]
```

**Configuration**:
```rust
NBEATSConfig::new(input_size: 168, horizon: 24)
    .with_stacks(vec![
        StackConfig::trend(4),      // 4 trend blocks
        StackConfig::seasonal(4),   // 4 seasonality blocks
        StackConfig::generic(4)     // 4 generic blocks
    ])
    .with_hidden_size(512)
```

**Best For**: Interpretable forecasting, trend/seasonal decomposition, long horizons.

#### 2. NBEATSx (Extended NBEATS)
**Mathematical Foundation**: NBEATS with exogenous variable support.

```
NBEATSx = NBEATS(target) + ExogEncoder(exog_features)
```

**Configuration**:
```rust
NBEATSxConfig::new(input_size: 168, horizon: 24)
    .with_exog_features(vec!["temperature", "humidity", "day_of_week"])
    .with_static_features(vec!["location_id", "category"])
```

**Best For**: Multi-variate forecasting, external regressors, calendar effects.

#### 3. NHITS (Neural Hierarchical Interpolation)
**Mathematical Foundation**: Multi-rate neural network with hierarchical interpolation.

```
NHITS: [MaxPool → MLP → Interpolate] × N_stacks → Sum
```

**Configuration**:
```rust
NHITSConfig::new(input_size: 168, horizon: 24)
    .with_pooling_sizes(vec![1, 2, 4])  // Multi-rate pooling
    .with_interpolation_mode(InterpolationMode::Linear)
```

**Best For**: Long sequences, computational efficiency, hierarchical patterns.

#### 4. TiDE (Time-series Dense Encoder)
**Mathematical Foundation**: Dense encoder-decoder with residual connections.

```
TiDE: Encoder(past) + Decoder(future_features) → Residual → Output
```

**Configuration**:
```rust
TiDEConfig::new(input_size: 96, horizon: 24)
    .with_encoder_layers(vec![512, 512])
    .with_decoder_layers(vec![256, 128])
    .with_temporal_decoder_layers(2)
```

**Best For**: Dense feature representations, encoder-decoder patterns, residual learning.

### Transformer Models (6 implementations)

#### 1. TFT (Temporal Fusion Transformers)
**Mathematical Foundation**: Multi-horizon forecasting with attention-based feature selection.

```
TFT: [VSN → LSTM] + [VSN → Attention] + [GLN] → Multi-Head Attention → Output
```

**Configuration**:
```rust
TFTConfig::new(input_size: 168, horizon: 24)
    .with_hidden_size(256)
    .with_num_heads(8)
    .with_dropout(0.1)
    .with_static_features(vec!["category", "location"])
    .with_known_regressors(vec!["day_of_week", "hour_of_day"])
```

**Best For**: Complex temporal patterns, feature importance, multi-horizon forecasting.

#### 2. Informer (Efficient Transformer)
**Mathematical Foundation**: ProbSparse self-attention for long sequence efficiency.

```
ProbSparse Attention: O(L log L) instead of O(L²)
```

**Configuration**:
```rust
InformerConfig::new(input_size: 720, horizon: 168)  // Weekly → Daily
    .with_factor(5)  // ProbSparse factor
    .with_num_heads(8)
    .with_num_layers(6)
```

**Best For**: Very long sequences, memory efficiency, sparse attention patterns.

#### 3. AutoFormer (Auto-correlation Transformer)
**Mathematical Foundation**: Auto-correlation mechanism replacing self-attention.

```
AutoCorrelation: Series-wise connections discovered by auto-correlation
```

**Configuration**:
```rust
AutoFormerConfig::new(input_size: 336, horizon: 96)
    .with_autocorr_factor(1)
    .with_moving_avg_kernel(25)
    .with_num_heads(8)
```

**Best For**: Seasonal patterns, auto-correlation discovery, periodic time series.

#### 4. FedFormer (Frequency Domain Transformer)
**Mathematical Foundation**: Fourier/Wavelet transforms in frequency domain.

```
FedFormer: FFT → Frequency Attention → iFFT
```

**Configuration**:
```rust
FedFormerConfig::new(input_size: 336, horizon: 96)
    .with_mode(FrequencyMode::Fourier)
    .with_modes(32)  // Top frequency modes
    .with_wavelet("db4")
```

**Best For**: Frequency domain patterns, spectral analysis, signal processing.

#### 5. PatchTST (Patch-based Transformer)
**Mathematical Foundation**: Patching mechanism for computational efficiency.

```
PatchTST: Patch(X) → Transformer → Reconstruct
```

**Configuration**:
```rust
PatchTSTConfig::new(input_size: 336, horizon: 96)
    .with_patch_length(16)
    .with_stride(8)
    .with_num_heads(16)
```

**Best For**: Long sequences, patch-based processing, computational efficiency.

#### 6. iTransformer (Inverted Transformer)
**Mathematical Foundation**: Inverted dimensions - attention across variates instead of time.

```
iTransformer: Transpose → Attention(Variables) → Transpose
```

**Configuration**:
```rust
iTransformerConfig::new(n_variables: 7, input_size: 96, horizon: 24)
    .with_num_heads(8)
    .with_cross_variate_attention(true)
```

**Best For**: Multivariate series, cross-variable dependencies, variable selection.

### Specialized Models (10+ implementations)

#### 1. DeepAR (Deep Autoregressive)
**Mathematical Foundation**: Probabilistic forecasting with autoregressive likelihood.

```
DeepAR: LSTM → MLP → Distribution Parameters → Probabilistic Output
```

**Configuration**:
```rust
DeepARConfig::new(input_size: 168, horizon: 24)
    .with_distribution(DistributionType::StudentT)
    .with_likelihood_weights(true)
```

**Best For**: Probabilistic forecasting, uncertainty quantification, irregular patterns.

#### 2. DeepNPTS (Deep Non-Parametric Time Series)
**Mathematical Foundation**: Non-parametric approach with neural estimation.

```
DeepNPTS: Non-parametric density estimation with neural networks
```

**Best For**: Complex distributions, non-parametric inference, flexible modeling.

#### 3. TCN (Temporal Convolutional Networks)
**Mathematical Foundation**: 1D convolutions with dilated causal structure.

```
TCN: [Dilated Conv → Activation → Dropout] × N → Residual
```

**Configuration**:
```rust
TCNConfig::new(input_size: 168, horizon: 24)
    .with_kernel_size(3)
    .with_dilations(vec![1, 2, 4, 8, 16])
    .with_num_channels(vec![128, 128, 128, 128])
```

**Best For**: Long sequences, parallel processing, CNN advantages.

#### 4. BiTCN (Bidirectional TCN)
**Mathematical Foundation**: Bidirectional temporal convolutions.

```
BiTCN: TCN(forward) + TCN(backward) → Concat → Output
```

**Best For**: Bidirectional patterns, enhanced context modeling.

#### 5. TimesNet (Time-2D Variation)
**Mathematical Foundation**: 2D representation of time series with vision backbones.

```
TimesNet: 1D → 2D(Period, Time) → Vision Backbone → 2D → 1D
```

**Best For**: Complex temporal patterns, vision-inspired processing.

#### 6. StemGNN (Spectral Temporal Graph)
**Mathematical Foundation**: Graph neural networks for time series with spectral analysis.

```
StemGNN: GFT → Spectral Conv → Temporal Conv → iGFT
```

**Best For**: Graph-structured time series, spectral analysis, multivariate dependencies.

#### 7-10. TSMixer, TSMixerx, TimeLLM
Advanced mixing and language model approaches for time series forecasting.

## Usage Examples

### Basic Usage - Single Model

```rust
use neuro_divergent_models::{
    NeuralForecast, models::LSTM, LSTMConfig,
    data::{TimeSeriesDataFrame, TimeSeriesSchema}
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load time series data
    let schema = TimeSeriesSchema::new()
        .with_time_column("timestamp")
        .with_target_column("value")
        .with_identifier_column("series_id");
    
    let train_data = TimeSeriesDataFrame::from_csv("train.csv", schema)?;
    
    // Configure LSTM model
    let lstm_config = LSTMConfig::default_with_horizon(24)
        .with_architecture(hidden_size: 128, num_layers: 2, dropout: 0.1)
        .with_training(max_steps: 1000, learning_rate: 0.001);
    
    // Create and train model
    let mut nf = NeuralForecast::new()
        .with_model("lstm", Box::new(LSTM::new(lstm_config)?))
        .build()?;
    
    // Fit model
    let training_metrics = nf.fit(&train_data).await?;
    println!("Training completed: {:.4} MSE", training_metrics.mse);
    
    // Generate forecasts
    let forecasts = nf.predict().await?;
    
    // Evaluate
    let test_data = TimeSeriesDataFrame::from_csv("test.csv", schema)?;
    let metrics = nf.evaluate(&test_data).await?;
    println!("Test MAPE: {:.2}%", metrics.mape * 100.0);
    
    Ok(())
}
```

### Multi-Model Ensemble

```rust
use neuro_divergent_models::{
    NeuralForecast, 
    models::{LSTM, NBEATS, TFT},
    ensemble::WeightedEnsemble
};

async fn ensemble_forecasting() -> Result<(), Box<dyn std::error::Error>> {
    // Configure multiple models
    let lstm = LSTM::new(LSTMConfig::default_with_horizon(24))?;
    let nbeats = NBEATS::new(NBEATSConfig::new(168, 24))?;
    let tft = TFT::new(TFTConfig::new(168, 24))?;
    
    // Create ensemble
    let mut ensemble = NeuralForecast::new()
        .with_model("lstm", Box::new(lstm))
        .with_model("nbeats", Box::new(nbeats))
        .with_model("tft", Box::new(tft))
        .with_ensemble(WeightedEnsemble::new(vec![0.3, 0.4, 0.3]))
        .build()?;
    
    // Train ensemble
    ensemble.fit(&train_data).await?;
    
    // Generate ensemble forecasts
    let forecasts = ensemble.predict().await?;
    
    Ok(())
}
```

### Custom Model Implementation

```rust
use neuro_divergent_models::{
    foundation::{BaseModel, ModelConfig},
    data::{TimeSeriesInput, ForecastOutput},
    errors::{NeuroDivergentResult, NeuroDivergentError}
};

struct CustomMLP<T: Float> {
    config: CustomMLPConfig<T>,
    network: Option<Network<T>>,
    is_trained: bool,
}

impl<T: Float + Send + Sync> BaseModel<T> for CustomMLP<T> {
    fn name(&self) -> &str { "CustomMLP" }
    
    fn config(&self) -> &dyn ModelConfig<T> { &self.config }
    
    fn is_trained(&self) -> bool { self.is_trained }
    
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingMetrics<T>> {
        // Custom training logic
        self.network = Some(self.build_network()?);
        self.is_trained = true;
        
        Ok(TrainingMetrics::default())
    }
    
    fn predict(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<ForecastOutput<T>> {
        let network = self.network.as_ref()
            .ok_or(NeuroDivergentError::ModelNotTrained)?;
        
        // Custom prediction logic
        let predictions = network.run(&input.values)?;
        
        Ok(ForecastOutput::new(predictions))
    }
    
    // ... implement other required methods
}
```

### Advanced Configuration

```rust
use neuro_divergent_models::{
    config::{TrainingConfig, ValidationConfig, CrossValidationConfig},
    optimization::LearningRateSchedule,
    callbacks::{EarlyStopping, ModelCheckpoint}
};

async fn advanced_training() -> Result<(), Box<dyn std::error::Error>> {
    // Advanced training configuration
    let training_config = TrainingConfig::new()
        .with_max_epochs(1000)
        .with_batch_size(32)
        .with_learning_rate_schedule(
            LearningRateSchedule::exponential_decay(0.001, 0.95, 100)
        )
        .with_gradient_clipping(1.0)
        .with_weight_decay(0.0001);
    
    // Cross-validation setup
    let cv_config = CrossValidationConfig::new()
        .with_n_folds(5)
        .with_strategy(CVStrategy::TimeSeriesSplit)
        .with_gap(24); // 24-hour gap between train/validation
    
    // Callbacks
    let early_stopping = EarlyStopping::new()
        .with_patience(20)
        .with_min_delta(0.001)
        .with_monitor_metric("val_mse");
    
    let checkpoint = ModelCheckpoint::new("best_model.json")
        .with_monitor_metric("val_mse")
        .with_save_best_only(true);
    
    // Create model with advanced configuration
    let mut nf = NeuralForecast::new()
        .with_model("lstm", Box::new(LSTM::new(lstm_config)?))
        .with_training_config(training_config)
        .with_cross_validation(cv_config)
        .with_callbacks(vec![
            Box::new(early_stopping),
            Box::new(checkpoint)
        ])
        .build()?;
    
    // Train with cross-validation
    let cv_results = nf.cross_validate(&data).await?;
    println!("CV Results: {:.4} ± {:.4}", cv_results.mean_score, cv_results.std_score);
    
    Ok(())
}
```

## Model Selection Guide

### Decision Tree

```
Start Here
├── **Sequence Length**
│   ├── Short (< 100 points)
│   │   ├── Linear patterns → **NLinear, DLinear**
│   │   ├── Non-linear patterns → **MLP, GRU**
│   │   └── Multiple series → **MLPMultivariate**
│   ├── Medium (100-1000 points)
│   │   ├── Sequential patterns → **LSTM, GRU**
│   │   ├── Seasonal patterns → **NBEATS, AutoFormer**
│   │   └── Complex patterns → **TFT, TCN**
│   └── Long (> 1000 points)
│       ├── Computational efficiency → **PatchTST, NHITS**
│       ├── Frequency patterns → **FedFormer, TimesNet**
│       └── Maximum accuracy → **Informer, TFT**
├── **Data Characteristics**
│   ├── Univariate → **NBEATS, NHITS, AutoFormer**
│   ├── Multivariate → **iTransformer, TFT, MLPMultivariate**
│   ├── Irregular → **DeepAR, RNN**
│   └── With exogenous features → **NBEATSx, TFT**
├── **Requirements**
│   ├── Interpretability → **NBEATS, DLinear, NLinear**
│   ├── Uncertainty quantification → **DeepAR, DeepNPTS**
│   ├── Real-time inference → **DLinear, NLinear, MLP**
│   └── Maximum accuracy → **TFT, Informer, Ensemble**
└── **Domain-Specific**
    ├── Finance → **DeepAR, TFT, LSTM**
    ├── Energy → **NBEATS, AutoFormer, TFT**
    ├── Retail → **TFT, NHITS, DeepAR**
    └── IoT/Sensors → **TCN, LSTM, TimesNet**
```

### Performance Characteristics

| Model Category | Training Speed | Inference Speed | Memory Usage | Accuracy | Interpretability |
|----------------|---------------|-----------------|--------------|----------|------------------|
| **Basic**      |               |                 |              |          |                  |
| MLP            | Fast          | Fast            | Low          | Medium   | Low              |
| DLinear        | Very Fast     | Very Fast       | Very Low     | Medium   | High             |
| NLinear        | Very Fast     | Very Fast       | Very Low     | Low      | High             |
| **Recurrent**  |               |                 |              |          |                  |
| RNN            | Medium        | Fast            | Low          | Low      | Medium           |
| LSTM           | Slow          | Medium          | Medium       | High     | Low              |
| GRU            | Medium        | Medium          | Medium       | High     | Low              |
| **Advanced**   |               |                 |              |          |                  |
| NBEATS         | Slow          | Medium          | Medium       | High     | High             |
| NHITS          | Medium        | Fast            | Medium       | High     | Medium           |
| TiDE           | Medium        | Medium          | Medium       | High     | Low              |
| **Transformer**|               |                 |              |          |                  |
| TFT            | Very Slow     | Slow            | High         | Very High| Medium           |
| Informer       | Slow          | Medium          | Medium       | High     | Low              |
| PatchTST       | Medium        | Fast            | Medium       | High     | Low              |
| **Specialized**|               |                 |              |          |                  |
| DeepAR         | Slow          | Medium          | Medium       | High     | Medium           |
| TCN            | Medium        | Fast            | Medium       | High     | Low              |

## API Documentation

### Core Traits

#### BaseModel<T: Float + Send + Sync>

```rust
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync {
    fn name(&self) -> &str;
    fn config(&self) -> &dyn ModelConfig<T>;
    fn is_trained(&self) -> bool;
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingMetrics<T>>;
    fn predict(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<ForecastOutput<T>>;
    fn predict_batch(&self, inputs: &[TimeSeriesInput<T>]) -> NeuroDivergentResult<Vec<ForecastOutput<T>>>;
    fn validate(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ValidationMetrics<T>>;
    fn save_state(&self) -> NeuroDivergentResult<Vec<u8>>;
    fn load_state(&mut self, state: &[u8]) -> NeuroDivergentResult<()>;
    fn input_size(&self) -> usize;
    fn horizon(&self) -> usize;
    fn reset(&mut self);
}
```

#### ModelConfig<T: Float>

```rust
pub trait ModelConfig<T: Float>: Send + Sync {
    fn model_type(&self) -> &str;
    fn input_size(&self) -> usize;
    fn horizon(&self) -> usize;
    fn validate(&self) -> NeuroDivergentResult<()>;
    fn to_json(&self) -> NeuroDivergentResult<String>;
    fn from_json(json: &str) -> NeuroDivergentResult<Self> where Self: Sized;
}
```

### Data Structures

#### TimeSeriesDataFrame

```rust
pub struct TimeSeriesDataFrame<T: Float> {
    // Core data storage
    data: HashMap<String, Vec<T>>,
    schema: TimeSeriesSchema,
    
    // Metadata
    time_column: String,
    target_column: String,
    identifier_column: Option<String>,
}

impl<T: Float> TimeSeriesDataFrame<T> {
    pub fn new(schema: TimeSeriesSchema) -> Self;
    pub fn from_csv<P: AsRef<Path>>(path: P, schema: TimeSeriesSchema) -> NeuroDivergentResult<Self>;
    pub fn add_column(&mut self, name: String, values: Vec<T>);
    pub fn get_column(&self, name: &str) -> Option<&Vec<T>>;
    pub fn filter_by_time_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> NeuroDivergentResult<Self>;
    pub fn group_by_identifier(&self) -> HashMap<String, Self>;
    pub fn train_test_split(&self, test_size: f64) -> (Self, Self);
}
```

#### ForecastOutput

```rust
pub struct ForecastOutput<T: Float> {
    pub predictions: Vec<T>,
    pub prediction_intervals: Option<Vec<(T, T)>>,
    pub timestamps: Option<Vec<DateTime<Utc>>>,
    pub model_name: String,
    pub metadata: HashMap<String, String>,
}
```

### Training Configuration

#### TrainingConfig

```rust
pub struct TrainingConfig<T: Float> {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: T,
    pub learning_rate_schedule: Option<LearningRateSchedule<T>>,
    pub gradient_clipping: Option<T>,
    pub weight_decay: T,
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    pub validation_split: T,
    pub shuffle: bool,
    pub seed: Option<u64>,
}
```

### Ensemble Methods

```rust
pub enum EnsembleMethod {
    Simple,           // Simple averaging
    Weighted(Vec<f64>), // Weighted averaging
    Stacking(Box<dyn BaseModel<T>>), // Meta-model stacking
    BayesianAveraging, // Bayesian model averaging
}

pub struct EnsembleConfig<T: Float> {
    pub method: EnsembleMethod,
    pub cv_folds: usize,
    pub diversity_penalty: T,
}
```

## Advanced Usage

### Custom Model Development

```rust
use neuro_divergent_models::{
    foundation::{BaseModel, ModelConfig, NetworkAdapter},
    core::ModelBuilder,
    layers::{DenseLayer, DropoutLayer},
    utils::WindowGenerator
};

/// Custom Wavelet Neural Network for time series forecasting
pub struct WaveletNN<T: Float> {
    config: WaveletNNConfig<T>,
    network: Option<Network<T>>,
    wavelet_transform: WaveletTransform<T>,
    is_trained: bool,
}

impl<T: Float> WaveletNN<T> {
    pub fn new(config: WaveletNNConfig<T>) -> NeuroDivergentResult<Self> {
        let wavelet_transform = WaveletTransform::new(
            config.wavelet_type.clone(),
            config.decomposition_levels
        )?;
        
        Ok(Self {
            config,
            network: None,
            wavelet_transform,
            is_trained: false,
        })
    }
    
    fn build_network(&self) -> NeuroDivergentResult<Network<T>> {
        let mut builder = NetworkBuilder::new();
        
        // Input layer (wavelet coefficients)
        builder.add_layer(DenseLayer::new(
            self.config.wavelet_coeffs_size,
            self.config.hidden_size
        ));
        
        // Hidden layers with residual connections
        for i in 0..self.config.num_layers {
            let layer = DenseLayer::new(
                self.config.hidden_size,
                self.config.hidden_size
            ).with_activation(self.config.activation);
            
            builder.add_layer(layer);
            
            if self.config.dropout > T::zero() {
                builder.add_layer(DropoutLayer::new(self.config.dropout));
            }
        }
        
        // Output layer
        builder.add_layer(DenseLayer::new(
            self.config.hidden_size,
            self.config.horizon
        ));
        
        builder.build()
    }
}

impl<T: Float + Send + Sync> BaseModel<T> for WaveletNN<T> {
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingMetrics<T>> {
        // 1. Apply wavelet transform to input data
        let wavelet_data = self.wavelet_transform.transform(&data.target_values())?;
        
        // 2. Create training pairs
        let window_gen = WindowGenerator::new(
            self.config.input_size,
            self.config.horizon
        );
        let training_pairs = window_gen.generate_pairs(&wavelet_data)?;
        
        // 3. Build and train network
        self.network = Some(self.build_network()?);
        let network = self.network.as_mut().unwrap();
        
        // 4. Training loop with custom loss function
        let mut best_loss = T::infinity();
        let mut patience_counter = 0;
        
        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = T::zero();
            
            for (input, target) in &training_pairs {
                let prediction = network.forward(input)?;
                let loss = self.wavelet_loss(&prediction, target)?;
                
                network.backward(loss)?;
                epoch_loss = epoch_loss + loss;
            }
            
            // Early stopping
            if epoch_loss < best_loss {
                best_loss = epoch_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    break;
                }
            }
        }
        
        self.is_trained = true;
        Ok(TrainingMetrics::new(best_loss))
    }
    
    fn predict(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<ForecastOutput<T>> {
        let network = self.network.as_ref()
            .ok_or(NeuroDivergentError::ModelNotTrained)?;
        
        // 1. Transform input to wavelet domain
        let wavelet_input = self.wavelet_transform.transform(&input.values)?;
        
        // 2. Neural network prediction
        let wavelet_prediction = network.forward(&wavelet_input)?;
        
        // 3. Inverse wavelet transform
        let time_domain_prediction = self.wavelet_transform
            .inverse_transform(&wavelet_prediction)?;
        
        Ok(ForecastOutput::new(time_domain_prediction))
    }
}
```

### Ensemble Techniques

```rust
use neuro_divergent_models::ensemble::{
    StackingEnsemble, BayesianEnsemble, DiversityEnsemble
};

/// Advanced ensemble with model selection and weighting
pub struct AdaptiveEnsemble<T: Float> {
    base_models: Vec<Box<dyn BaseModel<T>>>,
    meta_model: Box<dyn BaseModel<T>>,
    diversity_weights: Vec<T>,
    performance_history: Vec<Vec<T>>,
}

impl<T: Float> AdaptiveEnsemble<T> {
    pub fn new() -> Self {
        Self {
            base_models: Vec::new(),
            meta_model: Box::new(MLP::new(MLPConfig::default())),
            diversity_weights: Vec::new(),
            performance_history: Vec::new(),
        }
    }
    
    pub fn add_model(&mut self, model: Box<dyn BaseModel<T>>) -> &mut Self {
        self.base_models.push(model);
        self.diversity_weights.push(T::one());
        self.performance_history.push(Vec::new());
        self
    }
    
    /// Dynamic weight adjustment based on recent performance
    pub fn update_weights(&mut self, recent_errors: &[Vec<T>]) {
        for (i, errors) in recent_errors.iter().enumerate() {
            let recent_mse = errors.iter()
                .fold(T::zero(), |acc, &x| acc + x * x) / T::from(errors.len()).unwrap();
            
            // Inverse relationship: better performance = higher weight
            self.diversity_weights[i] = T::one() / (T::one() + recent_mse);
        }
        
        // Normalize weights
        let total_weight: T = self.diversity_weights.iter().sum();
        for weight in &mut self.diversity_weights {
            *weight = *weight / total_weight;
        }
    }
    
    /// Ensemble prediction with adaptive weighting
    pub fn predict_adaptive(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<ForecastOutput<T>> {
        let mut predictions = Vec::new();
        
        // Get predictions from all models
        for model in &self.base_models {
            let pred = model.predict(input)?;
            predictions.push(pred.predictions);
        }
        
        // Weighted combination
        let horizon = predictions[0].len();
        let mut ensemble_pred = vec![T::zero(); horizon];
        
        for (i, pred) in predictions.iter().enumerate() {
            let weight = self.diversity_weights[i];
            for (j, &value) in pred.iter().enumerate() {
                ensemble_pred[j] = ensemble_pred[j] + weight * value;
            }
        }
        
        Ok(ForecastOutput::new(ensemble_pred))
    }
}
```

### Optimization Strategies

```rust
use neuro_divergent_models::{
    optimization::{
        AdamOptimizer, RMSpropOptimizer, LearningRateSchedule,
        GradientClipping, WeightDecay
    },
    callbacks::{
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoardLogger, MetricsLogger
    }
};

/// Advanced training with multiple optimization strategies
pub async fn optimized_training<T: Float + Send + Sync>(
    model: &mut dyn BaseModel<T>,
    data: &TimeSeriesDataset<T>
) -> NeuroDivergentResult<TrainingMetrics<T>> {
    
    // Advanced optimizer with adaptive learning rate
    let optimizer = AdamOptimizer::new(T::from(0.001).unwrap())
        .with_beta1(T::from(0.9).unwrap())
        .with_beta2(T::from(0.999).unwrap())
        .with_epsilon(T::from(1e-8).unwrap())
        .with_weight_decay(T::from(0.0001).unwrap());
    
    // Learning rate schedule
    let lr_schedule = LearningRateSchedule::cosine_annealing(
        T::from(0.001).unwrap(),  // initial_lr
        T::from(0.00001).unwrap(), // min_lr
        100                       // T_max
    );
    
    // Gradient clipping
    let grad_clip = GradientClipping::by_norm(T::from(1.0).unwrap());
    
    // Callbacks
    let early_stopping = EarlyStopping::new()
        .with_patience(20)
        .with_min_delta(T::from(0.0001).unwrap())
        .with_restore_best_weights(true);
    
    let checkpoint = ModelCheckpoint::new("checkpoints/")
        .with_save_best_only(true)
        .with_save_weights_only(false);
    
    let reduce_lr = ReduceLROnPlateau::new()
        .with_factor(T::from(0.5).unwrap())
        .with_patience(10);
    
    // Training configuration
    let training_config = TrainingConfig::new()
        .with_optimizer(optimizer)
        .with_lr_schedule(lr_schedule)
        .with_gradient_clipping(grad_clip)
        .with_callbacks(vec![
            Box::new(early_stopping),
            Box::new(checkpoint),
            Box::new(reduce_lr)
        ]);
    
    // Execute training
    let trainer = ModelTrainer::new(training_config);
    trainer.fit(model, data).await
}
```

### Memory Optimization

```rust
use neuro_divergent_models::{
    memory::{MemoryPool, LazyLoading, GradientCheckpointing},
    streaming::{StreamingDataset, BatchProcessor}
};

/// Memory-efficient training for large datasets
pub struct MemoryEfficientTrainer<T: Float> {
    memory_pool: MemoryPool<T>,
    batch_processor: BatchProcessor<T>,
    gradient_checkpointing: bool,
}

impl<T: Float> MemoryEfficientTrainer<T> {
    pub fn new(memory_limit_gb: usize) -> Self {
        Self {
            memory_pool: MemoryPool::new(memory_limit_gb * 1024 * 1024 * 1024),
            batch_processor: BatchProcessor::new(),
            gradient_checkpointing: true,
        }
    }
    
    /// Train model with streaming data and memory management
    pub async fn train_streaming<M: BaseModel<T>>(
        &mut self,
        model: &mut M,
        dataset: StreamingDataset<T>
    ) -> NeuroDivergentResult<TrainingMetrics<T>> {
        
        let mut metrics = TrainingMetrics::new();
        
        // Process data in chunks
        let chunk_size = self.calculate_optimal_chunk_size(model)?;
        
        for chunk in dataset.chunks(chunk_size) {
            // Load chunk into memory pool
            let batch = self.memory_pool.load_batch(chunk)?;
            
            // Process batch with gradient checkpointing
            let batch_metrics = if self.gradient_checkpointing {
                self.train_batch_checkpointed(model, &batch).await?
            } else {
                self.train_batch_standard(model, &batch).await?
            };
            
            metrics.update(&batch_metrics);
            
            // Release memory
            self.memory_pool.release_batch(&batch)?;
        }
        
        Ok(metrics)
    }
}
```

## Contributing

We welcome contributions to the Neuro-Divergent Models library! Please see our [Contributing Guide](../CONTRIBUTING.md) for details on:

- Code style and conventions
- Testing requirements
- Documentation standards
- Pull request process
- Model implementation guidelines

### Adding New Models

To add a new model:

1. Create the model module in the appropriate category directory
2. Implement the `BaseModel` and `ModelConfig` traits
3. Add comprehensive tests
4. Update documentation
5. Add usage examples

### Performance Benchmarks

Run benchmarks with:

```bash
cargo bench --features=testing
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

This library is inspired by and builds upon:

- [NeuralForecast](https://github.com/Nixtla/neuralforecast) - Python neural forecasting library
- [ruv-FANN](https://github.com/ruvnet/ruv-FANN) - Rust Fast Artificial Neural Network library
- Various research papers and implementations in the time series forecasting community

## Citation

If you use Neuro-Divergent Models in your research, please cite:

```bibtex
@software{neuro_divergent_models,
  title={Neuro-Divergent Models: Neural Forecasting Library for Rust},
  author={rUv Contributors},
  year={2024},
  url={https://github.com/ruvnet/ruv-FANN}
}
```