# Basic Models

Basic models provide simple yet effective forecasting capabilities with minimal computational requirements. These models are ideal for establishing baselines, handling simple patterns, and situations where interpretability and speed are important.

## DLinear (Decomposition Linear)

DLinear performs linear forecasting through decomposition-based modeling, separating trend and seasonal components for better interpretability and performance.

### Architecture

DLinear decomposes time series into trend and seasonal components, then applies separate linear transformations:

1. **Decomposition**: Series = Trend + Seasonal + Residual
2. **Linear Mapping**: Each component is linearly mapped to future values
3. **Aggregation**: Final forecast combines all component predictions

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct DLinearConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub kernel_size: usize,
    pub individual: bool,
    pub use_decomposition: bool,
}
```

### Usage Examples

#### Basic Usage

```rust
use neuro_divergent::models::{DLinear, DLinearConfig};

let config = DLinearConfig::builder()
    .horizon(7)           // Forecast 7 steps ahead
    .input_size(28)       // Use 28 historical points
    .kernel_size(25)      // Moving average kernel size
    .individual(false)    // Shared parameters across series
    .use_decomposition(true)  // Enable trend/seasonal decomposition
    .build()?;

let model = DLinear::new(config)?;
```

#### Builder Pattern

```rust
let dlinear = DLinear::builder()
    .horizon(12)
    .input_size(36)
    .kernel_size(25)
    .individual(true)     // Individual parameters per series
    .build()?;
```

#### Advanced Configuration

```rust
let config = DLinearConfig {
    horizon: 24,
    input_size: 168,      // Weekly data (24 * 7)
    kernel_size: 25,      // Smooth decomposition
    individual: true,     // Series-specific parameters
    use_decomposition: true,
};

let model = DLinear::new(config)?;
```

### Configuration Parameters

- **`horizon`** (required): Number of future time steps to forecast
- **`input_size`** (required): Number of historical time steps to use as input
- **`kernel_size`** (default: 25): Size of moving average kernel for decomposition
- **`individual`** (default: false): Whether to use individual linear layers for each series
- **`use_decomposition`** (default: true): Whether to use trend/seasonal decomposition

### Performance Characteristics

- **Training Time**: O(n) - Very fast
- **Inference Time**: O(1) - Constant time
- **Memory Usage**: Low
- **Interpretability**: High
- **Best For**: Datasets with clear trend/seasonal patterns

### Strengths

- **Extremely Fast**: Linear operations only
- **Interpretable**: Clear decomposition into trend/seasonal components  
- **Robust**: Minimal parameters, less prone to overfitting
- **Efficient**: Low memory and computational requirements
- **Baseline**: Excellent starting point for any forecasting task

### Limitations

- **Linear Assumption**: Cannot capture complex non-linear patterns
- **Fixed Decomposition**: Uses simple moving average decomposition
- **Limited Capacity**: May underfit complex datasets

## NLinear (Normalized Linear)

NLinear applies normalization before linear transformation, improving performance on non-stationary time series.

### Architecture

1. **Normalization**: Subtract last value and normalize by standard deviation
2. **Linear Mapping**: Apply linear transformation to normalized data
3. **Denormalization**: Add back last value to get final forecast

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct NLinearConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub individual: bool,
    pub use_normalization: bool,
}
```

### Usage Examples

#### Basic Usage

```rust
let nlinear = NLinear::builder()
    .horizon(7)
    .input_size(28)
    .individual(false)
    .use_normalization(true)
    .build()?;
```

#### For Non-Stationary Data

```rust
// Configuration for non-stationary financial data
let config = NLinearConfig::builder()
    .horizon(5)
    .input_size(60)      // 2-3 months of daily data
    .individual(true)    // Each stock has different characteristics
    .use_normalization(true)  // Handle different price levels
    .build()?;

let model = NLinear::new(config)?;
```

### Configuration Parameters

- **`horizon`** (required): Number of future time steps to forecast
- **`input_size`** (required): Number of historical time steps to use
- **`individual`** (default: false): Series-specific linear layers
- **`use_normalization`** (default: true): Apply normalization before modeling

### Performance Characteristics

- **Training Time**: O(n) - Very fast
- **Inference Time**: O(1) - Constant time  
- **Memory Usage**: Low
- **Best For**: Non-stationary time series with varying scales

### Strengths

- **Handles Non-Stationarity**: Normalization helps with varying scales
- **Simple and Fast**: Minimal computational overhead
- **Good Baseline**: Often surprisingly competitive
- **Stable Training**: Normalization improves numerical stability

## MLP (Multi-Layer Perceptron)

Standard multi-layer perceptron for univariate time series forecasting.

### Architecture

A feed-forward neural network with:
- Input layer (input_size)
- Hidden layers with ReLU activation
- Output layer (horizon)
- Optional dropout for regularization

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct MLPConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub dropout: f64,
    pub activation: ActivationType,
}
```

### Usage Examples

#### Basic MLP

```rust
let mlp = MLP::builder()
    .horizon(7)
    .input_size(28)
    .hidden_sizes(vec![64, 32])  // Two hidden layers
    .dropout(0.1)
    .activation(ActivationType::ReLU)
    .build()?;
```

#### Deep MLP for Complex Patterns

```rust
let deep_mlp = MLP::builder()
    .horizon(12)
    .input_size(48)
    .hidden_sizes(vec![256, 128, 64, 32])  // 4 hidden layers
    .dropout(0.2)  // More regularization for deeper network
    .activation(ActivationType::ReLU)
    .build()?;
```

#### Shallow MLP for Simple Data

```rust
let shallow_mlp = MLP::builder()
    .horizon(3)
    .input_size(12)
    .hidden_sizes(vec![32])  // Single hidden layer
    .dropout(0.05)  // Light regularization
    .build()?;
```

### Configuration Parameters

- **`horizon`** (required): Forecast horizon
- **`input_size`** (required): Input sequence length
- **`hidden_sizes`** (default: [64]): Sizes of hidden layers
- **`dropout`** (default: 0.1): Dropout rate for regularization
- **`activation`** (default: ReLU): Activation function

### Performance Characteristics

- **Training Time**: O(n·h) where h is hidden size
- **Inference Time**: O(h) - Proportional to network size
- **Memory Usage**: Medium
- **Best For**: Non-linear patterns in univariate series

### Strengths

- **Non-Linear**: Can capture complex patterns
- **Flexible**: Easily adjustable architecture
- **Universal Approximator**: Theoretically can model any function
- **Fast Training**: No recurrent connections

### Limitations

- **No Temporal Structure**: Treats input as fixed-size vector
- **Limited Context**: Cannot handle variable-length sequences
- **Overfitting**: May overfit with small datasets

## MLP Multivariate

Multi-layer perceptron specifically designed for multivariate time series forecasting.

### Architecture

Similar to MLP but with modifications for multivariate data:
- Input layer handles multiple variables
- Shared or separate processing for each variable
- Output layer produces forecasts for all variables

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct MLPMultivariateConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub num_variables: usize,
    pub hidden_sizes: Vec<usize>,
    pub dropout: f64,
    pub shared_weights: bool,
    pub activation: ActivationType,
}
```

### Usage Examples

#### Multivariate Economic Data

```rust
let mv_mlp = MLPMultivariate::builder()
    .horizon(12)
    .input_size(36)
    .num_variables(5)    // GDP, inflation, unemployment, etc.
    .hidden_sizes(vec![128, 64])
    .shared_weights(false)  // Different processing per variable
    .dropout(0.15)
    .build()?;
```

#### Shared Processing

```rust
// When variables have similar characteristics
let shared_mlp = MLPMultivariate::builder()
    .horizon(7)
    .input_size(28)
    .num_variables(10)   // Multiple similar sensors
    .hidden_sizes(vec![64, 32])
    .shared_weights(true)  // Shared processing
    .dropout(0.1)
    .build()?;
```

### Configuration Parameters

- **`num_variables`** (required): Number of variables to forecast
- **`shared_weights`** (default: false): Share weights across variables
- All other parameters same as MLP

### Performance Characteristics

- **Training Time**: O(n·h·v) where v is number of variables
- **Memory Usage**: Medium to High (depends on num_variables)
- **Best For**: Multivariate forecasting with cross-variable dependencies

## Training and Optimization

### Training Configuration

```rust
use neuro_divergent::training::{TrainingConfig, OptimizerConfig, LossFunctionConfig};

let training_config = TrainingConfig::builder()
    .max_epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::Adam { 
        beta1: 0.9, 
        beta2: 0.999 
    })
    .loss_function(LossFunctionConfig::MSE)
    .early_stopping_patience(10)
    .build()?;
```

### Loss Functions for Basic Models

```rust
// Different loss functions for different scenarios
let mse_config = LossFunctionConfig::MSE;           // Standard squared loss
let mae_config = LossFunctionConfig::MAE;           // Robust to outliers  
let huber_config = LossFunctionConfig::Huber {      // Combines MSE + MAE
    delta: 1.0
};
let mape_config = LossFunctionConfig::MAPE;         // Percentage error
```

### Regularization Techniques

```rust
// L2 regularization
let config = MLPConfig::builder()
    .hidden_sizes(vec![128, 64])
    .dropout(0.2)
    .l2_regularization(0.01)  // Weight decay
    .build()?;

// Early stopping
let training_config = TrainingConfig::builder()
    .early_stopping_patience(15)
    .validation_split(0.2)
    .build()?;
```

## Performance Comparison

### Computational Complexity

| Model | Parameters | Training Time | Inference Time | Memory |
|-------|-----------|---------------|----------------|---------|
| DLinear | O(I) | O(n) | O(1) | Low |
| NLinear | O(I) | O(n) | O(1) | Low |  
| MLP | O(I×H + H²) | O(n×H) | O(H) | Medium |
| MLP-MV | O(V×I×H + H²) | O(n×V×H) | O(V×H) | High |

*Where I=input_size, H=hidden_size, V=num_variables, n=dataset_size*

### Accuracy Benchmarks

Typical performance on standard datasets:

#### M4 Monthly Data
- **DLinear**: 13.2 sMAPE
- **NLinear**: 13.5 sMAPE  
- **MLP**: 13.8 sMAPE
- **Linear Baseline**: 14.1 sMAPE

#### Financial Returns (Daily)
- **NLinear**: 0.084 MAE (best for non-stationary)
- **DLinear**: 0.087 MAE
- **MLP**: 0.089 MAE
- **Random Walk**: 0.095 MAE

## Best Practices

### Model Selection Guidelines

```rust
// Choose based on data characteristics
fn select_basic_model(data_info: &DataInfo) -> Box<dyn BaseModel<f64>> {
    match data_info {
        // Stationary data with clear trend/seasonality
        DataInfo { stationary: true, seasonal: true, .. } => {
            Box::new(DLinear::builder()
                .horizon(data_info.horizon)
                .use_decomposition(true)
                .build().unwrap())
        },
        
        // Non-stationary data
        DataInfo { stationary: false, .. } => {
            Box::new(NLinear::builder()
                .horizon(data_info.horizon)
                .use_normalization(true)
                .build().unwrap())
        },
        
        // Complex non-linear patterns
        DataInfo { complex_patterns: true, .. } => {
            Box::new(MLP::builder()
                .horizon(data_info.horizon)
                .hidden_sizes(vec![64, 32])
                .dropout(0.15)
                .build().unwrap())
        },
        
        // Default to simple linear
        _ => {
            Box::new(DLinear::builder()
                .horizon(data_info.horizon)
                .build().unwrap())
        }
    }
}
```

### Hyperparameter Tuning

```rust
// Grid search for MLP
let hidden_sizes_options = vec![
    vec![32],
    vec![64],
    vec![64, 32],
    vec![128, 64],
    vec![128, 64, 32],
];

let dropout_options = vec![0.0, 0.1, 0.2, 0.3];

let mut best_config = None;
let mut best_score = f64::INFINITY;

for hidden_sizes in hidden_sizes_options {
    for &dropout in &dropout_options {
        let config = MLPConfig::builder()
            .horizon(7)
            .input_size(28)
            .hidden_sizes(hidden_sizes.clone())
            .dropout(dropout)
            .build()?;
            
        let model = MLP::new(config.clone())?;
        let cv_score = evaluate_cross_validation(&model, &data)?;
        
        if cv_score < best_score {
            best_score = cv_score;
            best_config = Some(config);
        }
    }
}
```

### Data Preprocessing

```rust
// Preprocessing recommendations for basic models
use neuro_divergent::preprocessing::{StandardScaler, RobustScaler};

// For DLinear - minimal preprocessing needed
let dlinear_data = data.clone();  // Raw data often works well

// For NLinear - built-in normalization, but scaling can help
let mut scaler = RobustScaler::new();  // Robust to outliers
let nlinear_data = scaler.fit_transform(&data)?;

// For MLP - standardization usually helps
let mut scaler = StandardScaler::new();
let mlp_data = scaler.fit_transform(&data)?;
```

### Ensemble Usage

```rust
// Combine basic models for robustness
let ensemble_models = vec![
    Box::new(DLinear::builder()
        .horizon(7)
        .use_decomposition(true)
        .build()?) as Box<dyn BaseModel<f64>>,
        
    Box::new(NLinear::builder()
        .horizon(7)
        .use_normalization(true)
        .build()?) as Box<dyn BaseModel<f64>>,
        
    Box::new(MLP::builder()
        .horizon(7)
        .hidden_sizes(vec![64, 32])
        .dropout(0.1)
        .build()?) as Box<dyn BaseModel<f64>>,
];

let ensemble = NeuralForecast::builder()
    .with_models(ensemble_models)
    .build()?;
```

## Integration Examples

### Simple Forecasting Pipeline

```rust
use neuro_divergent::prelude::*;

// Load data
let data = TimeSeriesDataFrame::from_csv("simple_data.csv")?;

// Create simple model
let model = DLinear::builder()
    .horizon(7)
    .input_size(28)
    .build()?;

// Create forecaster
let mut nf = NeuralForecast::builder()
    .with_model(Box::new(model))
    .with_frequency(Frequency::Daily)
    .build()?;

// Train and predict
nf.fit(data.clone())?;
let forecasts = nf.predict()?;
```

### Rapid Prototyping

```rust
// Quick baseline establishment
fn establish_baseline(data: &TimeSeriesDataFrame<f64>) -> NeuroDivergentResult<Vec<f64>> {
    // Try multiple simple models quickly
    let models = vec![
        ("DLinear", DLinear::builder().horizon(7).build()?),
        ("NLinear", NLinear::builder().horizon(7).build()?),
        ("MLP", MLP::builder().horizon(7).hidden_sizes(vec![32]).build()?),
    ];
    
    let mut results = Vec::new();
    
    for (name, mut model) in models {
        let dataset = TimeSeriesDataset::from_dataframe(data)?;
        model.fit(&dataset)?;
        let forecast = model.predict(&dataset)?;
        
        let mae = calculate_mae(&forecast.forecasts, &actual_values)?;
        results.push(mae);
        println!("{}: MAE = {:.4}", name, mae);
    }
    
    Ok(results)
}
```

Basic models provide an excellent foundation for time series forecasting, offering simplicity, speed, and interpretability while often achieving competitive performance on many real-world datasets.