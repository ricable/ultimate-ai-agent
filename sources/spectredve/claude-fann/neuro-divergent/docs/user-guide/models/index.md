# Model Overview

Neuro-Divergent provides 27+ state-of-the-art neural forecasting models, organized into five categories. This overview helps you understand each model's strengths, use cases, and when to choose them for your forecasting tasks.

## Model Categories

| Category | Models | Best For | Complexity |
|----------|--------|----------|-----------|
| [Basic Models](basic-models.md) | MLP, DLinear, NLinear, MLPMultivariate | Simple patterns, baselines | Low |
| [Recurrent Models](recurrent-models.md) | RNN, LSTM, GRU | Sequential dependencies | Medium |
| [Advanced Models](advanced-models.md) | NBEATS, NBEATSx, NHiTS | Complex univariate patterns | High |
| [Transformer Models](transformer-models.md) | TFT, Informer, Autoformer, FEDformer, PatchTST, iTransformer | Long sequences, multivariate | High |
| [Specialized Models](specialized-models.md) | DeepAR, TCN, BiTCN, TimesNet, StemGNN, TSMixer, TSMixerx, TimeLLM | Specific use cases | Varies |

## Quick Model Selection Guide

### "I'm new to neural forecasting"
**Start with**: MLP → LSTM → NBEATS
- MLP: Simple and fast baseline
- LSTM: Industry standard for sequences  
- NBEATS: Advanced univariate forecasting

### "I have simple, clean data"
**Try**: DLinear → NLinear → MLP
- DLinear: Excellent for trended data
- NLinear: Good for stationary data
- MLP: Non-linear patterns

### "I need probabilistic forecasts"
**Use**: DeepAR → NBEATS → TFT
- DeepAR: Built for uncertainty quantification
- NBEATS: Supports prediction intervals
- TFT: Advanced probabilistic capabilities

### "I have multiple related time series"
**Choose**: LSTM → TFT → TimesNet
- LSTM: Global model across series
- TFT: Handles static and dynamic features
- TimesNet: Cross-series relationships

### "I have very long sequences"
**Consider**: Informer → PatchTST → iTransformer
- Informer: Efficient for long sequences
- PatchTST: Patch-based processing
- iTransformer: Inverted transformer design

### "I need interpretability"
**Pick**: NBEATS → TFT → DLinear
- NBEATS: Trend/seasonal decomposition
- TFT: Attention visualization
- DLinear: Linear interpretability

## Detailed Model Comparison

### Performance vs Complexity

```
High Performance    │ TFT ●              ● TimesNet
                   │     ● Informer  ● NBEATS
                   │ ● DeepAR          ● NBEATSx
Medium Performance │         ● LSTM      ● NHiTS
                   │     ● GRU    ● TCN
                   │ ● RNN           ● BiTCN
Low Performance    │     ● MLP ● DLinear ● NLinear
                   └─────────────────────────────────
                     Low      Medium      High
                              Complexity
```

### Training Speed vs Accuracy

```
High Accuracy      │           ● TFT
                   │       ● NBEATS    ● TimesNet
                   │   ● LSTM     ● DeepAR
Medium Accuracy    │       ● GRU ● TCN
                   │   ● RNN         ● Informer
                   │ ● MLP     ● DLinear
Low Accuracy       │     ● NLinear
                   └─────────────────────────────────
                     Fast              Slow
                           Training Speed
```

## Model Characteristics

| Model | Univariate | Multivariate | Probabilistic | Interpretable | Global | Memory Efficient |
|-------|------------|--------------|---------------|---------------|--------|------------------|
| **Basic Models** |
| MLP | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| DLinear | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| NLinear | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| MLPMultivariate | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Recurrent Models** |
| RNN | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| LSTM | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| GRU | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Advanced Models** |
| NBEATS | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| NBEATSx | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NHiTS | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Transformer Models** |
| TFT | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Informer | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Autoformer | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| FEDformer | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| PatchTST | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| iTransformer | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Specialized Models** |
| DeepAR | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| TCN | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| BiTCN | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| TimesNet | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| StemGNN | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| TSMixer | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| TSMixerx | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| TimeLLM | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |

## Use Case Recommendations

### Business Forecasting
- **Sales/Revenue**: LSTM, TFT, DeepAR
- **Demand Planning**: NBEATS, DeepAR, TimesNet
- **Financial Markets**: Informer, TFT, LSTM
- **Web Traffic**: MLP, LSTM, TCN

### Operational Forecasting  
- **Energy Load**: NBEATS, TFT, LSTM
- **Resource Planning**: DeepAR, LSTM, TFT
- **Inventory Management**: DeepAR, NBEATS, LSTM
- **Maintenance Scheduling**: TCN, LSTM, MLP

### Scientific Applications
- **Weather Prediction**: TFT, Informer, TimesNet
- **Environmental Monitoring**: LSTM, TCN, TFT
- **Medical Time Series**: DeepAR, LSTM, NBEATS
- **Sensor Data**: TCN, LSTM, MLP

### Special Requirements
- **Real-time Prediction**: MLP, DLinear, TCN
- **Interpretable Results**: NBEATS, DLinear, TFT
- **Uncertain/Sparse Data**: DeepAR, NBEATS
- **Very Long Sequences**: Informer, PatchTST
- **Multiple Related Series**: TFT, TimesNet, LSTM

## Data Size Guidelines

### Small Datasets (< 1K observations)
**Recommended**: MLP, DLinear, NLinear
```rust
let mlp = MLP::new(MLPConfig::new()
    .with_input_size(12)
    .with_hidden_layers(vec![32, 16])
    .with_regularization(0.1))?;  // High regularization
```

### Medium Datasets (1K - 10K observations)
**Recommended**: LSTM, GRU, NBEATS, TCN
```rust
let lstm = LSTM::new(LSTMConfig::new()
    .with_input_size(24)
    .with_hidden_size(64)
    .with_num_layers(1))?;  // Conservative sizing
```

### Large Datasets (10K - 100K observations)
**Recommended**: TFT, Informer, DeepAR, TimesNet
```rust
let tft = TFT::new(TFTConfig::new()
    .with_d_model(128)
    .with_num_heads(8)
    .with_num_layers(3))?;  // Full power
```

### Very Large Datasets (> 100K observations)
**Recommended**: PatchTST, iTransformer, distributed training
```rust
let patch_tst = PatchTST::new(PatchTSTConfig::new()
    .with_patch_size(16)
    .with_d_model(256)
    .with_num_layers(6))?;  // Maximum capacity
```

## Performance Benchmarks

### Training Time (relative to MLP baseline)

| Model | Small Data | Medium Data | Large Data |
|-------|------------|-------------|------------|
| MLP | 1.0x | 1.0x | 1.0x |
| DLinear | 0.8x | 0.8x | 0.8x |
| NLinear | 0.6x | 0.6x | 0.6x |
| LSTM | 3.0x | 4.0x | 5.0x |
| GRU | 2.5x | 3.5x | 4.5x |
| NBEATS | 2.0x | 3.0x | 4.0x |
| TFT | 8.0x | 12.0x | 15.0x |
| Informer | 6.0x | 8.0x | 10.0x |
| DeepAR | 5.0x | 7.0x | 9.0x |

### Memory Usage (relative to MLP baseline)

| Model | Parameters | Peak Memory |
|-------|------------|-------------|
| MLP | 1.0x | 1.0x |
| DLinear | 0.3x | 0.5x |
| NLinear | 0.2x | 0.4x |
| LSTM | 4.0x | 6.0x |
| GRU | 3.0x | 4.5x |
| NBEATS | 2.0x | 3.0x |
| TFT | 8.0x | 12.0x |
| Informer | 6.0x | 9.0x |
| DeepAR | 3.0x | 5.0x |

## Model Selection Workflow

### Step 1: Define Requirements
```rust
// Define your constraints and requirements
let requirements = ForecastingRequirements {
    horizon: 12,
    frequency: Frequency::Daily,
    interpretability_needed: true,
    probabilistic_forecasts: false,
    training_time_limit: Duration::from_secs(300),  // 5 minutes
    memory_limit: 8_000_000_000,  // 8GB
    accuracy_priority: Priority::High,
};
```

### Step 2: Automated Model Selection
```rust
let selector = ModelSelector::new()
    .with_requirements(requirements)
    .with_data_info(&data.info())
    .with_cross_validation(5);

let recommendations = selector.recommend_models()?;
// Returns: [NBEATS, DLinear, TFT] with scores
```

### Step 3: Model Comparison
```rust
let comparison = ModelComparison::new()
    .add_models(recommendations)
    .with_metrics(vec![Metric::MAE, Metric::MAPE, Metric::RMSE])
    .with_cross_validation(5);

let results = comparison.run(&data).await?;
results.print_summary();
results.plot_comparison("model_comparison.png")?;
```

## Advanced Model Combinations

### Ensemble Methods
```rust
// Simple ensemble
let ensemble = EnsembleModel::new()
    .add_model(LSTM::new(lstm_config)?)
    .add_model(NBEATS::new(nbeats_config)?)
    .add_model(TFT::new(tft_config)?)
    .with_weights(vec![0.4, 0.3, 0.3])  // Weighted average
    .build()?;

// Stacked ensemble
let stacker = StackedEnsemble::new()
    .add_base_models(vec![lstm, nbeats, tft])
    .with_meta_model(MLP::new(mlp_config)?)  // Meta-learner
    .build()?;
```

### Hierarchical Forecasting
```rust
// Forecast at multiple aggregation levels
let hierarchical = HierarchicalModel::new()
    .with_bottom_level_model(LSTM::new(config)?)
    .with_reconciliation_method(ReconciliationMethod::OLS)
    .with_hierarchy_structure(hierarchy_matrix)
    .build()?;
```

### Multi-Objective Optimization
```rust
// Optimize for both accuracy and uncertainty
let multi_objective = MultiObjectiveModel::new()
    .add_objective(Objective::Accuracy, weight: 0.7)
    .add_objective(Objective::UncertaintyQuality, weight: 0.3)
    .with_base_model(TFT::new(config)?)
    .build()?;
```

## Migration Guide

### From Scikit-learn
```python
# Python scikit-learn
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

```rust
// Rust Neuro-Divergent
let model = MLP::new(MLPConfig::new()
    .with_input_size(X_train.ncols())
    .with_hidden_layers(vec![100, 50]))?;
model.fit(&train_data)?;
let predictions = model.predict(&test_data)?;
```

### From Prophet
```python
# Python Prophet
from prophet import Prophet
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

```rust
// Rust Neuro-Divergent (similar decomposition)
let model = NBEATS::new(NBEATSConfig::new()
    .with_stacks(vec![
        NBEATSStack::trend_stack(3, 32),
        NBEATSStack::seasonal_stack(3, 32),
    ]))?;
model.fit(&data)?;
let forecast = model.predict()?;
```

### From TensorFlow/PyTorch
```python
# Python TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
```

```rust
// Rust Neuro-Divergent
let model = LSTM::new(LSTMConfig::new()
    .with_input_size(1)
    .with_hidden_size(50)
    .with_num_layers(2)
    .with_output_size(1))?;
```

## Next Steps

Now explore specific model categories:

1. **Start Simple**: [Basic Models](basic-models.md) - MLP, DLinear, NLinear
2. **Learn Sequences**: [Recurrent Models](recurrent-models.md) - LSTM, GRU, RNN  
3. **Advanced Patterns**: [Advanced Models](advanced-models.md) - NBEATS, NHiTS
4. **Modern Approaches**: [Transformer Models](transformer-models.md) - TFT, Informer
5. **Special Cases**: [Specialized Models](specialized-models.md) - DeepAR, TCN

Each category guide provides detailed explanations, configuration examples, and practical tips for getting the best results from each model type.