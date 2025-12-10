# Basic Concepts

This guide introduces the fundamental concepts behind neural forecasting and how Neuro-Divergent implements them. Whether you're new to time series forecasting or neural networks, this guide will help you understand the core principles.

## Time Series Forecasting Fundamentals

### What is Time Series Forecasting?

Time series forecasting predicts future values based on historical patterns in sequential data. Unlike traditional regression, time series data has:

- **Temporal Dependencies**: Past values influence future values
- **Seasonality**: Recurring patterns (daily, weekly, yearly)
- **Trends**: Long-term directional changes
- **Irregularities**: Random noise and anomalies

### Traditional vs Neural Approaches

| Traditional Methods | Neural Methods |
|-------------------|----------------|
| ARIMA, ETS, Prophet | LSTM, Transformer, NBEATS |
| Linear assumptions | Non-linear pattern learning |
| Manual feature engineering | Automatic feature learning |
| Single series focus | Multi-series learning |
| Statistical foundations | Data-driven approach |

## Neural Forecasting Concepts

### Why Neural Networks for Forecasting?

Neural networks excel at forecasting because they can:

1. **Learn Complex Patterns**: Non-linear relationships, interactions
2. **Handle Multiple Variables**: Multivariate time series, external features
3. **Scale to Large Datasets**: Global models across many series
4. **Adapt Automatically**: No manual parameter tuning
5. **Provide Uncertainty**: Probabilistic predictions

### Key Neural Forecasting Principles

#### 1. Supervised Learning for Sequences

Neural forecasting transforms time series into supervised learning:

```
Input Window:  [t-n, t-n+1, ..., t-1, t]
Output Window: [t+1, t+2, ..., t+h]
```

Where:
- `n` = input window size (lookback period)
- `h` = forecast horizon (prediction steps)

#### 2. Feature Engineering

Raw time series is enhanced with:

```rust
// Time-based features
let features = FeatureBuilder::new()
    .add_lags(vec![1, 7, 30])          // Historical lags
    .add_rolling_stats(vec![7, 30])     // Moving averages
    .add_calendar_features()            // Day of week, month
    .add_seasonal_decomposition()       // Trend, seasonal components
    .build();
```

#### 3. Global vs Local Models

**Local Models**: One model per time series
- Traditional approach
- Requires sufficient data per series
- No cross-series learning

**Global Models**: One model for all time series
- Modern neural approach
- Learns from all series simultaneously
- Better for sparse data

```rust
// Local model - train separately for each series
for series in time_series_collection {
    let model = LSTM::new(config.clone())?;
    model.fit(&series)?;
}

// Global model - train on all series together
let global_model = LSTM::new(config)?;
global_model.fit(&combined_series)?;  // All series together
```

## Neuro-Divergent Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │────│   Model Layer    │────│  Training Layer │
│                 │    │                  │    │                 │
│ • Preprocessing │    │ • Model Registry │    │ • Optimizers    │
│ • Validation    │    │ • 27+ Models     │    │ • Loss Functions│
│ • Transformers  │    │ • Configuration  │    │ • Schedulers    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Raw Data** → Time series in various formats
2. **Preprocessing** → Cleaning, validation, feature engineering
3. **Training** → Model learns patterns from historical data
4. **Prediction** → Generate forecasts with uncertainty estimates
5. **Evaluation** → Assess model performance and accuracy

### Key Abstractions

#### BaseModel Trait
All models implement the `BaseModel` trait:

```rust
pub trait BaseModel<T: Float> {
    type Config: ModelConfig<T>;
    
    fn new(config: Self::Config) -> Result<Self, ModelError>;
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<(), ModelError>;
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<PredictionResult<T>, ModelError>;
    fn is_trained(&self) -> bool;
}
```

#### Configuration Pattern
Models use builder pattern for configuration:

```rust
let config = LSTMConfig::new()
    .with_input_size(24)
    .with_hidden_size(128)
    .with_num_layers(2)
    .with_dropout(0.1)
    .with_horizon(12);
```

## Model Categories

### 1. Basic Models

**Multi-Layer Perceptron (MLP)**
- Feed-forward neural network
- Good baseline for simple patterns
- Fast training and prediction

**Linear Models (DLinear, NLinear)**
- Linear transformations with neural enhancements
- Excellent for trended data
- Interpretable and efficient

```rust
// MLP for non-linear patterns
let mlp = MLP::new(MLPConfig::new()
    .with_input_size(24)
    .with_hidden_layers(vec![128, 64])
    .with_horizon(12))?;

// DLinear for trended data
let dlinear = DLinear::new(DLinearConfig::new()
    .with_input_size(24)
    .with_horizon(12))?;
```

### 2. Recurrent Models

**Long Short-Term Memory (LSTM)**
- Designed for sequential data
- Handles long-term dependencies
- Industry standard for time series

**Gated Recurrent Unit (GRU)**
- Simplified LSTM variant
- Faster training
- Good for shorter sequences

```rust
// LSTM for complex temporal patterns
let lstm = LSTM::new(LSTMConfig::new()
    .with_input_size(24)
    .with_hidden_size(128)
    .with_num_layers(2))?;

// GRU for efficiency
let gru = GRU::new(GRUConfig::new()
    .with_input_size(24)
    .with_hidden_size(64))?;
```

### 3. Transformer Models

**Temporal Fusion Transformer (TFT)**
- State-of-the-art attention-based model
- Handles multiple input types
- Provides interpretability

**Informer**
- Efficient transformer for long sequences
- Sparse attention mechanism
- Good for high-frequency data

```rust
// TFT for complex multivariate forecasting
let tft = TFT::new(TFTConfig::new()
    .with_d_model(128)
    .with_num_heads(8)
    .with_num_layers(3))?;
```

### 4. Advanced Models

**NBEATS**
- Pure neural basis expansion
- Excellent for univariate forecasting
- Interpretable decomposition

**DeepAR**
- Probabilistic forecasting
- Handles intermittent demand
- Provides uncertainty estimates

```rust
// NBEATS for interpretable forecasting
let nbeats = NBEATS::new(NBEATSConfig::new()
    .with_stacks(vec![
        NBEATSStack::trend_stack(3, 32),
        NBEATSStack::seasonal_stack(3, 32),
    ]))?;

// DeepAR for probabilistic forecasting
let deepar = DeepAR::new(DeepARConfig::new()
    .with_context_length(24)
    .with_prediction_length(12))?;
```

## Data Concepts

### Time Series Structure

```rust
pub struct TimeSeriesData<T: Float> {
    pub timestamps: Vec<DateTime<Utc>>,     // Time indices
    pub values: Vec<T>,                     // Target variable
    pub static_features: Option<Vec<T>>,    // Time-invariant features
    pub exogenous_historical: Option<Vec<Vec<T>>>,  // Historical covariates
    pub exogenous_future: Option<Vec<Vec<T>>>,      // Future known covariates
    pub series_id: String,                  // Series identifier
}
```

### Feature Types

#### 1. Target Variable
The main variable to forecast:
```rust
let target_values = vec![100.0, 105.0, 103.0, 108.0, ...];
```

#### 2. Static Features
Constants for each series:
```rust
// Store size, category, location - never change
let static_features = vec![2500.0, 1.0, 37.7749];  // [size, category_encoded, latitude]
```

#### 3. Historical Covariates
Known variables that change over time:
```rust
// Temperature, promotions - known for historical period
let historical_covariates = vec![
    vec![20.1, 21.5, 19.8, ...],  // Temperature
    vec![0.0, 1.0, 0.0, ...],     // Promotion (0/1)
];
```

#### 4. Future Covariates
Known variables for the forecast period:
```rust
// Calendar features, planned promotions - known in advance
let future_covariates = vec![
    vec![1.0, 2.0, 3.0, ...],     // Day of week
    vec![0.0, 0.0, 1.0, ...],     // Planned promotions
];
```

### Data Preprocessing

#### 1. Scaling and Normalization
```rust
let scaler = StandardScaler::new();
let scaled_data = scaler.fit_transform(&raw_data)?;

// Later, inverse transform predictions
let original_scale_predictions = scaler.inverse_transform(&predictions)?;
```

#### 2. Missing Value Handling
```rust
let imputer = TimeSeriesImputer::new()
    .with_method(ImputationMethod::Linear)
    .with_max_gap(7);  // Maximum gap to fill

let clean_data = imputer.transform(&data_with_gaps)?;
```

#### 3. Outlier Detection
```rust
let outlier_detector = StatisticalOutlierDetector::new()
    .with_method(OutlierMethod::IQR)
    .with_threshold(3.0);

let outliers = outlier_detector.detect(&data)?;
let clean_data = outlier_detector.remove_outliers(&data, &outliers)?;
```

## Training Concepts

### Loss Functions

Different loss functions optimize for different objectives:

```rust
// Mean Absolute Error - robust to outliers
let mae_loss = MAE::new();

// Mean Squared Error - penalizes large errors more
let mse_loss = MSE::new();

// Huber Loss - combination of MAE and MSE
let huber_loss = HuberLoss::new(delta: 1.0);

// Quantile Loss - for probabilistic forecasting
let quantile_loss = QuantileLoss::new(quantiles: vec![0.1, 0.5, 0.9]);
```

### Optimization

```rust
// Adam optimizer - most common choice
let optimizer = Adam::new()
    .with_learning_rate(0.001)
    .with_weight_decay(1e-5);

// Training configuration
let training_config = TrainingConfig::new()
    .with_optimizer(optimizer)
    .with_max_epochs(100)
    .with_batch_size(32)
    .with_early_stopping(patience: 10);
```

### Validation Strategies

#### 1. Time Series Split
```rust
// Respect temporal order
let (train_data, val_data) = data.time_split(0.8)?;  // 80% train, 20% validation
```

#### 2. Cross-Validation
```rust
// Time series cross-validation
let cv_results = TimeSeriesCrossValidator::new()
    .with_n_splits(5)
    .with_gap(7)      // Gap between train and test
    .validate(&model, &data)?;
```

## Prediction Concepts

### Point vs Probabilistic Forecasts

#### Point Forecasts
Single value predictions:
```rust
let point_forecasts = model.predict(&data)?;
// Returns: [105.2, 103.4, 107.8, ...]
```

#### Probabilistic Forecasts
Multiple quantiles or full distributions:
```rust
let prob_forecasts = model.predict_with_intervals(&data, &[0.8, 0.9])?;
// Returns: Point forecasts + confidence intervals
```

### Uncertainty Sources

1. **Aleatoric Uncertainty**: Inherent data noise
2. **Epistemic Uncertainty**: Model uncertainty
3. **Distributional Uncertainty**: Distribution shape uncertainty

### Multi-Step Forecasting

#### Direct Strategy
Train separate models for each horizon:
```rust
// Train h different models
for h in 1..=12 {
    let model_h = LSTM::new(config.with_horizon(h))?;
    model_h.fit(&data)?;
}
```

#### Recursive Strategy
Use 1-step model recursively:
```rust
// Use model prediction as next input
let mut current_input = last_window;
let mut forecasts = Vec::new();

for _ in 0..horizon {
    let next_pred = model.predict_one_step(&current_input)?;
    forecasts.push(next_pred);
    current_input = update_window(current_input, next_pred);
}
```

#### Direct Multi-Output
Train model to predict entire horizon:
```rust
// Single model predicts all h steps
let config = LSTMConfig::new().with_horizon(12);
let model = LSTM::new(config)?;
let forecasts = model.predict(&data)?;  // Returns 12 predictions
```

## Evaluation Concepts

### Forecast Accuracy Metrics

```rust
// Mean Absolute Error
let mae = forecasts.iter()
    .zip(actuals.iter())
    .map(|(f, a)| (f - a).abs())
    .sum::<f64>() / forecasts.len() as f64;

// Mean Absolute Percentage Error
let mape = forecasts.iter()
    .zip(actuals.iter())
    .map(|(f, a)| ((f - a) / a).abs())
    .sum::<f64>() / forecasts.len() as f64 * 100.0;

// Root Mean Squared Error
let rmse = (forecasts.iter()
    .zip(actuals.iter())
    .map(|(f, a)| (f - a).powi(2))
    .sum::<f64>() / forecasts.len() as f64).sqrt();
```

### Model Selection Criteria

1. **Accuracy**: How close are predictions to actual values?
2. **Computational Efficiency**: Training and prediction speed
3. **Interpretability**: Can you understand model decisions?
4. **Robustness**: Performance across different conditions
5. **Uncertainty Quantification**: Quality of confidence intervals

## Best Practices

### Model Selection Guidelines

1. **Start Simple**: Begin with MLP or Linear models
2. **Consider Data Size**: Global models for many short series
3. **Check Seasonality**: Transformer/NBEATS for strong seasonality
4. **Need Uncertainty**: Use DeepAR or probabilistic models
5. **Computational Constraints**: Consider model complexity

### Data Quality Checklist

- [ ] No significant gaps in data
- [ ] Outliers identified and handled
- [ ] Appropriate frequency (not too sparse/dense)
- [ ] Sufficient history (at least 2-3 seasonal cycles)
- [ ] Static features are truly static
- [ ] Future covariates are actually known in advance

### Training Best Practices

- [ ] Use time-aware train/validation splits
- [ ] Monitor for overfitting with early stopping
- [ ] Scale/normalize features appropriately
- [ ] Start with default hyperparameters
- [ ] Use cross-validation for model selection
- [ ] Save models for reproducibility

## Next Steps

Now that you understand the fundamentals:

1. **Try Different Models**: Explore [Model Overview](models/index.md)
2. **Work with Your Data**: See [Data Handling](data-handling.md)
3. **Optimize Training**: Read [Training Guide](training.md)
4. **Improve Predictions**: Check [Prediction Guide](prediction.md)
5. **Deploy Models**: See [Best Practices](best-practices.md)

Understanding these concepts will help you make informed decisions about model selection, data preparation, and evaluation strategies for your specific forecasting challenges.