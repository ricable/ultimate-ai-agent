# Basic Models

Basic models provide simple yet effective approaches to neural forecasting. They're perfect for getting started, establishing baselines, and handling straightforward forecasting tasks. These models are fast to train, memory-efficient, and often surprisingly competitive.

## Overview

| Model | Strengths | Best For | Complexity |
|-------|-----------|----------|------------|
| MLP | Universal approximator, fast | Non-linear patterns, baselines | Low |
| DLinear | Excellent for trends, interpretable | Trended data, simple seasonality | Very Low |
| NLinear | Minimal parameters, very fast | Stationary data, real-time | Very Low |
| MLPMultivariate | Handles multiple variables | Cross-variable relationships | Low-Medium |

## Multi-Layer Perceptron (MLP)

The Multi-Layer Perceptron is a feed-forward neural network that learns non-linear mappings between input windows and forecast horizons. It's the foundation of neural forecasting and often serves as a strong baseline.

### When to Use MLP

✅ **Good for:**
- Establishing performance baselines
- Non-linear relationships in data
- Fast training and prediction requirements
- Simple deployment scenarios
- When data has complex but short-term patterns

❌ **Avoid when:**
- Strong temporal dependencies exist
- Long-term memory is required
- Data has clear sequential structure
- You need probabilistic forecasts

### Architecture

```
Input Window     Hidden Layers      Output
[t-n,...,t-1,t] → [128] → [64] → [t+1,...,t+h]
     │               │      │         │
  Features      Non-linear  Non-linear  Forecasts
               Transform   Transform
```

### Configuration

```rust
use neuro_divergent::models::MLP;

// Basic configuration
let config = MLPConfig::new()
    .with_input_size(24)                    // 24 time steps lookback
    .with_hidden_layers(vec![128, 64])      // Two hidden layers
    .with_horizon(12)                       // 12 steps ahead
    .with_activation(ActivationFunction::ReLU)
    .with_dropout(0.1);                     // 10% dropout

let mlp = MLP::new(config)?;

// Advanced configuration
let advanced_config = MLPConfig::new()
    .with_input_size(48)
    .with_hidden_layers(vec![256, 128, 64])  // Deeper network
    .with_horizon(24)
    .with_activation(ActivationFunction::Swish)  // Modern activation
    .with_dropout(0.2)
    .with_batch_norm(true)                   // Batch normalization
    .with_residual_connections(true)         // Skip connections
    .with_output_activation(ActivationFunction::Linear);

let advanced_mlp = MLP::new(advanced_config)?;
```

### Best Practices

#### 1. Architecture Design
```rust
// Start simple, add complexity gradually
let simple_config = MLPConfig::new()
    .with_hidden_layers(vec![64]);     // Single hidden layer

let medium_config = MLPConfig::new()
    .with_hidden_layers(vec![128, 64]); // Two layers

let complex_config = MLPConfig::new()
    .with_hidden_layers(vec![256, 128, 64, 32]); // Four layers
```

#### 2. Regularization
```rust
let regularized_config = MLPConfig::new()
    .with_hidden_layers(vec![128, 64])
    .with_dropout(0.2)                  // Prevent overfitting
    .with_weight_decay(1e-4)            // L2 regularization
    .with_early_stopping(patience: 15); // Stop when validation stops improving
```

#### 3. Feature Engineering
```rust
// Enhance input with engineered features
let enhanced_data = data
    .add_lags(vec![1, 7, 30])           // Add lagged values
    .add_rolling_means(vec![7, 30])     // Add moving averages
    .add_seasonal_features()            // Add day/month indicators
    .add_trend_features();              // Add trend components

let model = MLP::new(config)?;
model.fit(&enhanced_data)?;
```

### Example: Sales Forecasting

```rust
use neuro_divergent::prelude::*;

async fn sales_forecasting_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load daily sales data
    let data = TimeSeriesDataFrame::from_csv("sales.csv")?
        .with_time_column("date")
        .with_target_column("sales")
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Configure MLP for sales patterns
    let config = MLPConfig::new()
        .with_input_size(30)             // 30 days lookback
        .with_hidden_layers(vec![128, 64, 32])
        .with_horizon(7)                 // 1 week ahead
        .with_dropout(0.15)
        .with_activation(ActivationFunction::ReLU);
    
    let mut mlp = MLP::new(config)?;
    
    // Split data for training/validation
    let (train_data, val_data) = data.time_split(0.8)?;
    
    // Train with validation monitoring
    let training_config = TrainingConfig::new()
        .with_max_epochs(200)
        .with_learning_rate(0.001)
        .with_batch_size(32)
        .with_validation_data(&val_data)
        .with_early_stopping(patience: 20);
    
    mlp.fit_with_config(&train_data, &training_config).await?;
    
    // Generate forecasts
    let forecasts = mlp.predict(&val_data).await?;
    
    // Evaluate performance
    let metrics = ForecastMetrics::calculate(&forecasts, &val_data.targets())?;
    println!("MLP Sales Forecast - MAE: {:.2}, MAPE: {:.2}%", 
             metrics.mae, metrics.mape);
    
    Ok(())
}
```

## DLinear (Direct Linear)

DLinear decomposes time series into trend and seasonal components using linear operations. Despite its simplicity, it's remarkably effective for many real-world forecasting tasks, especially those with clear trends.

### When to Use DLinear

✅ **Good for:**
- Data with clear trends
- Seasonal patterns
- When interpretability is important
- Fast training requirements
- Baseline comparisons
- Production deployment with strict latency requirements

❌ **Avoid when:**
- Complex non-linear relationships exist
- Multiple seasonalities overlap
- You need probabilistic forecasts
- Irregular or chaotic patterns

### Architecture

```
Input Sequence
     │
  Decomposition
     ├─── Trend Component ──── Linear Layer ──┐
     │                                        │
     └─── Seasonal Component ── Linear Layer ──┼── Sum ── Output
                                              │
                                         Residual
```

### Configuration

```rust
use neuro_divergent::models::DLinear;

// Basic configuration
let config = DLinearConfig::new()
    .with_input_size(96)                // 96 time steps lookback
    .with_horizon(24)                   // 24 steps ahead
    .with_decomposition_kernel(25);     // Moving average kernel size

let dlinear = DLinear::new(config)?;

// Advanced configuration with custom decomposition
let advanced_config = DLinearConfig::new()
    .with_input_size(168)               // 1 week of hourly data
    .with_horizon(24)                   // 1 day ahead
    .with_decomposition_method(DecompositionMethod::STL)  // STL decomposition
    .with_trend_order(2)                // Quadratic trend
    .with_seasonal_periods(vec![24, 168]) // Daily and weekly seasonality
    .with_regularization(0.01);         // Light regularization

let advanced_dlinear = DLinear::new(advanced_config)?;
```

### Decomposition Methods

#### 1. Moving Average Decomposition (Default)
```rust
let ma_config = DLinearConfig::new()
    .with_decomposition_method(DecompositionMethod::MovingAverage)
    .with_decomposition_kernel(25);     // Kernel size for smoothing
```

#### 2. STL Decomposition
```rust
let stl_config = DLinearConfig::new()
    .with_decomposition_method(DecompositionMethod::STL)
    .with_seasonal_periods(vec![7, 365]); // Weekly and yearly patterns
```

#### 3. X13-ARIMA-SEATS
```rust
let x13_config = DLinearConfig::new()
    .with_decomposition_method(DecompositionMethod::X13)
    .with_trading_day_adjustment(true)
    .with_easter_adjustment(true);
```

### Example: Energy Load Forecasting

```rust
async fn energy_load_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load hourly energy consumption data
    let data = TimeSeriesDataFrame::from_csv("energy_load.csv")?
        .with_time_column("timestamp")
        .with_target_column("load_mw")
        .with_frequency(Frequency::Hourly)
        .build()?;
    
    // Configure DLinear for energy patterns
    let config = DLinearConfig::new()
        .with_input_size(168)            // 1 week lookback
        .with_horizon(24)                // 1 day ahead
        .with_decomposition_kernel(25)    // Smooth trends
        .with_seasonal_periods(vec![24, 168]); // Daily/weekly patterns
    
    let mut dlinear = DLinear::new(config)?;
    
    // Quick training (DLinear is very fast)
    let training_config = TrainingConfig::new()
        .with_max_epochs(50)             // Few epochs needed
        .with_learning_rate(0.01)        // Higher LR for linear models
        .with_batch_size(64);
    
    dlinear.fit_with_config(&data, &training_config).await?;
    
    // Get decomposed components for analysis
    let decomposition = dlinear.decompose(&data)?;
    println!("Trend component variance: {:.2}", 
             decomposition.trend.variance());
    println!("Seasonal component variance: {:.2}", 
             decomposition.seasonal.variance());
    
    // Generate interpretable forecasts
    let forecasts = dlinear.predict(&data).await?;
    
    Ok(())
}
```

## NLinear (Normalized Linear)

NLinear normalizes input sequences by subtracting the last value, then applies a linear transformation. This simple approach is surprisingly effective for stationary time series and provides excellent computational efficiency.

### When to Use NLinear

✅ **Good for:**
- Stationary time series
- Real-time prediction scenarios
- Minimal computational resources
- When you need the fastest possible model
- Baseline comparisons
- Embedded systems

❌ **Avoid when:**
- Strong trends exist
- Complex seasonality patterns
- Non-stationary data
- You need sophisticated modeling

### Architecture

```
Input: [x₁, x₂, ..., xₙ]
         │
    Normalize: [x₁-xₙ, x₂-xₙ, ..., 0]
         │
    Linear Layer
         │
    Output: [y₁, y₂, ..., yₕ]
         │
    Denormalize: [y₁+xₙ, y₂+xₙ, ..., yₕ+xₙ]
```

### Configuration

```rust
use neuro_divergent::models::NLinear;

// Basic configuration
let config = NLinearConfig::new()
    .with_input_size(48)                // 48 time steps
    .with_horizon(12);                  // 12 steps ahead

let nlinear = NLinear::new(config)?;

// Configuration with regularization
let regularized_config = NLinearConfig::new()
    .with_input_size(96)
    .with_horizon(24)
    .with_weight_decay(1e-5)            // Light regularization
    .with_bias(false);                  // Remove bias term

let regularized_nlinear = NLinear::new(regularized_config)?;
```

### Example: High-Frequency Trading

```rust
async fn high_frequency_trading_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load minute-by-minute price data
    let data = TimeSeriesDataFrame::from_csv("price_data.csv")?
        .with_time_column("timestamp")
        .with_target_column("price")
        .with_frequency(Frequency::Minutes(1))
        .build()?;
    
    // Configure NLinear for ultra-fast predictions
    let config = NLinearConfig::new()
        .with_input_size(60)             // 1 hour lookback
        .with_horizon(5)                 // 5 minutes ahead
        .with_bias(false);               // Simplest possible model
    
    let mut nlinear = NLinear::new(config)?;
    
    // Very fast training
    let training_config = TrainingConfig::new()
        .with_max_epochs(20)             // Minimal training
        .with_learning_rate(0.01)
        .with_batch_size(128);           // Large batches for speed
    
    nlinear.fit_with_config(&data, &training_config).await?;
    
    // Ultra-fast prediction
    let start = std::time::Instant::now();
    let forecasts = nlinear.predict(&data).await?;
    let prediction_time = start.elapsed();
    
    println!("Prediction time: {:.2}ms", prediction_time.as_secs_f64() * 1000.0);
    
    Ok(())
}
```

## MLPMultivariate

MLPMultivariate extends the basic MLP to handle multiple time series simultaneously, learning cross-variable relationships and shared patterns across different but related series.

### When to Use MLPMultivariate

✅ **Good for:**
- Multiple related time series
- Cross-variable relationships
- Shared seasonality patterns
- Portfolio forecasting
- Multi-product demand planning

❌ **Avoid when:**
- Series are completely independent
- You have only one time series
- Series have very different scales
- Strong temporal dependencies exist

### Architecture

```
Series 1: [x₁₁, x₁₂, ..., x₁ₙ] ──┐
Series 2: [x₂₁, x₂₂, ..., x₂ₙ] ──┼── Concatenate ── MLP ──┬── Output 1
    ...                           │                        ├── Output 2
Series k: [xₖ₁, xₖ₂, ..., xₖₙ] ──┘                        └── Output k
```

### Configuration

```rust
use neuro_divergent::models::MLPMultivariate;

// Basic multivariate configuration
let config = MLPMultivariateConfig::new()
    .with_num_series(5)                 // 5 time series
    .with_input_size(24)                // 24 time steps each
    .with_hidden_layers(vec![256, 128])  // Larger networks for complexity
    .with_horizon(12)                   // 12 steps ahead
    .with_shared_layers(true);          // Share lower layers

let mlp_mv = MLPMultivariate::new(config)?;

// Advanced configuration with series-specific layers
let advanced_config = MLPMultivariateConfig::new()
    .with_num_series(10)
    .with_input_size(48)
    .with_shared_layers_config(vec![512, 256])    // Shared lower layers
    .with_series_specific_layers(vec![64, 32])    // Series-specific upper layers
    .with_cross_attention(true)                   // Cross-series attention
    .with_horizon(24);

let advanced_mlp_mv = MLPMultivariate::new(advanced_config)?;
```

### Example: Retail Chain Forecasting

```rust
async fn retail_chain_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load sales data for multiple stores
    let data = TimeSeriesDataFrame::from_csv("store_sales.csv")?
        .with_time_column("date")
        .with_target_column("sales")
        .with_series_id_column("store_id")  // Multiple stores
        .with_static_features(vec!["store_size", "location_type"])
        .with_frequency(Frequency::Daily)
        .build()?;
    
    let num_stores = data.num_series();
    
    // Configure for retail chain patterns
    let config = MLPMultivariateConfig::new()
        .with_num_series(num_stores)
        .with_input_size(56)             // 8 weeks lookback
        .with_hidden_layers(vec![512, 256, 128])
        .with_horizon(14)                // 2 weeks ahead
        .with_shared_layers(true)        // Learn common patterns
        .with_cross_series_attention(true); // Store interactions
    
    let mut mlp_mv = MLPMultivariate::new(config)?;
    
    // Train on all stores simultaneously
    let training_config = TrainingConfig::new()
        .with_max_epochs(100)
        .with_learning_rate(0.001)
        .with_batch_size(16)             // Smaller batches for multivariate
        .with_gradient_clipping(1.0);    // Prevent exploding gradients
    
    mlp_mv.fit_with_config(&data, &training_config).await?;
    
    // Get forecasts for all stores
    let forecasts = mlp_mv.predict(&data).await?;
    
    // Analyze cross-store correlations
    let correlations = mlp_mv.get_cross_series_correlations()?;
    println!("Average cross-store correlation: {:.3}", 
             correlations.mean());
    
    Ok(())
}
```

## Model Comparison and Selection

### Performance Comparison

```rust
async fn compare_basic_models() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_sample_data()?;
    
    // Configure all models with similar capacity
    let mlp = MLP::new(MLPConfig::new()
        .with_input_size(48)
        .with_hidden_layers(vec![128, 64])
        .with_horizon(12))?;
    
    let dlinear = DLinear::new(DLinearConfig::new()
        .with_input_size(48)
        .with_horizon(12))?;
    
    let nlinear = NLinear::new(NLinearConfig::new()
        .with_input_size(48)
        .with_horizon(12))?;
    
    // Compare performance
    let comparison = ModelComparison::new()
        .add_model("MLP", mlp)
        .add_model("DLinear", dlinear)
        .add_model("NLinear", nlinear)
        .with_cross_validation(5)
        .with_metrics(vec![Metric::MAE, Metric::MAPE, Metric::RMSE]);
    
    let results = comparison.run(&data).await?;
    results.print_summary();
    
    Ok(())
}
```

### When to Choose Each Model

#### Choose MLP when:
- You need a reliable baseline
- Data has non-linear patterns but no strong temporal structure
- Training time is not critical
- You want good general-purpose performance

#### Choose DLinear when:
- Data has clear trends and seasonality
- Interpretability is important
- You need fast training and prediction
- Linear relationships dominate

#### Choose NLinear when:
- You need the fastest possible model
- Data is stationary or nearly stationary
- Computational resources are extremely limited
- You want the simplest effective model

#### Choose MLPMultivariate when:
- You have multiple related time series
- Cross-series relationships exist
- You want to leverage shared patterns
- Series have similar characteristics

## Optimization Tips

### 1. Hyperparameter Tuning
```rust
let tuner = GridSearchCV::new()
    .add_param("hidden_size", vec![32, 64, 128, 256])
    .add_param("num_layers", vec![1, 2, 3])
    .add_param("dropout", vec![0.0, 0.1, 0.2, 0.3])
    .add_param("learning_rate", vec![0.001, 0.01, 0.1])
    .with_cv_folds(5)
    .with_metric(Metric::MAE);

let best_config = tuner.search(&data).await?;
```

### 2. Feature Engineering
```rust
// Enhance basic models with better features
let enhanced_data = data
    .add_fourier_features(frequencies: vec![1, 2, 3])  // Capture seasonality
    .add_polynomial_trends(degree: 2)                  // Quadratic trends
    .add_interaction_features()                        // Feature interactions
    .add_statistical_features(window: 7);              // Rolling statistics
```

### 3. Ensemble Methods
```rust
// Combine basic models for better performance
let ensemble = WeightedEnsemble::new()
    .add_model(mlp, weight: 0.4)
    .add_model(dlinear, weight: 0.3)
    .add_model(nlinear, weight: 0.3)
    .build()?;

let ensemble_forecasts = ensemble.predict(&data).await?;
```

## Next Steps

Basic models provide an excellent foundation for neural forecasting. Once you've mastered these:

1. **Add Temporal Structure**: Explore [Recurrent Models](recurrent-models.md) for sequential dependencies
2. **Handle Complex Patterns**: Try [Advanced Models](advanced-models.md) for sophisticated univariate forecasting
3. **Scale to Long Sequences**: Learn [Transformer Models](transformer-models.md) for attention-based approaches
4. **Specialized Use Cases**: Check [Specialized Models](specialized-models.md) for domain-specific solutions

Remember: often the simplest model that works is the best choice. Basic models excel at many real-world forecasting tasks and should always be part of your model comparison.