# Quick Start Guide

Get up and running with Neuro-Divergent in under 5 minutes! This guide will walk you through creating your first neural forecast.

## Prerequisites

- Rust 1.75.0 or later installed
- Neuro-Divergent added to your project (see [Installation](installation.md))

## Your First Forecast in 3 Steps

### Step 1: Create a New Project

```bash
cargo new my_forecast_project
cd my_forecast_project
```

Add Neuro-Divergent to `Cargo.toml`:
```toml
[dependencies]
neuro-divergent = "0.1.0"
tokio = { version = "1.0", features = ["full"] }  # For async support
```

### Step 2: Create Sample Data

Replace `src/main.rs` with:

```rust
use neuro_divergent::prelude::*;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Generate sample time series data
    let data = create_sample_data();
    
    // Create and configure LSTM model
    let lstm_config = LSTMConfig::new()
        .with_input_size(24)    // Look back 24 time steps
        .with_hidden_size(128)  // 128 hidden units
        .with_num_layers(2)     // 2 LSTM layers
        .with_horizon(12)       // Forecast 12 steps ahead
        .with_dropout(0.1);     // 10% dropout for regularization
    
    let lstm = LSTM::new(lstm_config)?;
    
    // Create NeuralForecast instance
    let mut nf = NeuralForecast::new()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Train the model
    println!("Training model...");
    nf.fit(&data).await?;
    
    // Generate forecasts
    println!("Generating forecasts...");
    let forecasts = nf.predict().await?;
    
    // Display results
    print_results(&forecasts);
    
    Ok(())
}

fn create_sample_data() -> TimeSeriesDataFrame {
    use chrono::{DateTime, Utc, Duration};
    
    // Create 100 days of sample data with trend and seasonality
    let start_date = Utc::now() - Duration::days(100);
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..100 {
        let date = start_date + Duration::days(i);
        timestamps.push(date);
        
        // Simple trend + seasonal pattern + noise
        let trend = i as f64 * 0.1;
        let seasonal = (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin() * 5.0;
        let noise = rand::random::<f64>() * 2.0 - 1.0;
        let value = 100.0 + trend + seasonal + noise;
        
        values.push(value);
    }
    
    TimeSeriesDataFrame::new()
        .with_time_column("timestamp", timestamps)
        .with_target_column("y", values)
        .with_series_id("series_1")
        .build()
        .expect("Failed to create sample data")
}

fn print_results(forecasts: &ForecastDataFrame) {
    println!("\n📊 Forecast Results:");
    println!("==================");
    
    for (i, forecast) in forecasts.forecasts().iter().enumerate() {
        println!("Day {}: {:.2}", i + 1, forecast);
    }
    
    if let Some(intervals) = forecasts.prediction_intervals() {
        println!("\n📈 Prediction Intervals (80%):");
        for (i, (lower, upper)) in intervals.bounds_80().iter().enumerate() {
            println!("Day {}: [{:.2}, {:.2}]", i + 1, lower, upper);
        }
    }
}
```

### Step 3: Run Your First Forecast

```bash
cargo run
```

You should see output like:
```
Training model...
✅ Training completed in 15.3s
Generating forecasts...
✅ Forecasts generated in 0.2s

📊 Forecast Results:
==================
Day 1: 105.23
Day 2: 103.45
Day 3: 107.89
...
Day 12: 104.56

📈 Prediction Intervals (80%):
Day 1: [102.45, 108.01]
Day 2: [100.23, 106.67]
...
```

🎉 **Congratulations!** You've just created your first neural forecast with Neuro-Divergent!

## Understanding the Code

Let's break down what happened:

### Data Creation
```rust
let data = create_sample_data();
```
We created synthetic time series data with:
- **Trend**: Gradual increase over time
- **Seasonality**: Weekly pattern (7-day cycle)
- **Noise**: Random variations

### Model Configuration
```rust
let lstm_config = LSTMConfig::new()
    .with_input_size(24)    // Use 24 historical points
    .with_hidden_size(128)  // LSTM internal state size
    .with_num_layers(2)     // Stack 2 LSTM layers
    .with_horizon(12)       // Predict 12 points ahead
    .with_dropout(0.1);     // Prevent overfitting
```

### Training
```rust
nf.fit(&data).await?;
```
The model learns patterns from historical data using:
- Backpropagation through time
- Adam optimizer (default)
- Early stopping for optimal performance

### Forecasting
```rust
let forecasts = nf.predict().await?;
```
Generates 12-step-ahead forecasts with uncertainty estimates.

## Exploring Different Models

Try different models by changing the configuration:

### Simple MLP
```rust
let mlp_config = MLPConfig::new()
    .with_input_size(24)
    .with_hidden_layers(vec![128, 64])
    .with_horizon(12);

let mlp = MLP::new(mlp_config)?;
```

### Advanced NBEATS
```rust
let nbeats_config = NBEATSConfig::new()
    .with_stacks(vec![
        NBEATSStack::trend_stack(3, 32),
        NBEATSStack::seasonal_stack(3, 32),
    ])
    .with_horizon(12);

let nbeats = NBEATS::new(nbeats_config)?;
```

### Transformer (TFT)
```rust
let tft_config = TFTConfig::new()
    .with_d_model(128)
    .with_num_heads(8)
    .with_num_layers(3)
    .with_horizon(12);

let tft = TFT::new(tft_config)?;
```

## Working with Real Data

### Loading CSV Data
```rust
use neuro_divergent::data::loaders::CSVLoader;

let data = CSVLoader::new()
    .with_time_column("date")
    .with_target_column("sales")
    .with_date_format("%Y-%m-%d")
    .load("data/sales.csv")?;
```

### Expected CSV Format
```csv
date,sales,category
2023-01-01,1250.5,electronics
2023-01-02,1340.2,electronics
2023-01-03,1180.7,electronics
...
```

### Multiple Time Series
```rust
let data = CSVLoader::new()
    .with_time_column("date")
    .with_target_column("sales")
    .with_series_id_column("category")  // Group by category
    .load("data/multi_series.csv")?;

// Models will be trained on all series simultaneously
nf.fit(&data).await?;
```

## Adding External Features

### Static Features
```rust
let data = TimeSeriesDataFrame::new()
    .with_time_column("timestamp", timestamps)
    .with_target_column("y", values)
    .with_static_features(vec![
        ("category".to_string(), "electronics".to_string()),
        ("store_size".to_string(), "large".to_string()),
    ])
    .build()?;
```

### Dynamic Features
```rust
let data = TimeSeriesDataFrame::new()
    .with_time_column("timestamp", timestamps)
    .with_target_column("y", values)
    .with_exogenous_features(vec![
        ("temperature".to_string(), temperature_data),
        ("promotion".to_string(), promotion_data),
    ])
    .build()?;
```

## Hyperparameter Tuning

### Manual Tuning
```rust
let training_config = TrainingConfig::new()
    .with_max_epochs(200)
    .with_learning_rate(0.001)
    .with_batch_size(64)
    .with_early_stopping(patience: 20)
    .with_validation_split(0.2);

nf.fit_with_config(&data, &training_config).await?;
```

### Automated Tuning
```rust
use neuro_divergent::tuning::GridSearch;

let grid_search = GridSearch::new()
    .add_param("hidden_size", vec![64, 128, 256])
    .add_param("num_layers", vec![1, 2, 3])
    .add_param("learning_rate", vec![0.001, 0.01, 0.1])
    .with_cv_folds(5)
    .with_metric(Metric::MAE);

let best_config = grid_search.search(&data).await?;
```

## Model Evaluation

### Cross-Validation
```rust
let cv_results = nf.cross_validate(&data)
    .with_folds(5)
    .with_metrics(vec![Metric::MAE, Metric::MAPE, Metric::RMSE])
    .run().await?;

println!("Cross-validation MAE: {:.2}", cv_results.mean_mae());
```

### Backtesting
```rust
let backtest_results = nf.backtest(&data)
    .with_window_size(30)
    .with_step_size(7)
    .with_horizon(14)
    .run().await?;

backtest_results.plot_results("backtest.png")?;
```

## Saving and Loading Models

### Save Trained Model
```rust
// Save to file
nf.save("my_model.nd")?;

// Save to bytes
let model_bytes = nf.to_bytes()?;
```

### Load Trained Model
```rust
// Load from file
let nf = NeuralForecast::load("my_model.nd")?;

// Load from bytes
let nf = NeuralForecast::from_bytes(&model_bytes)?;
```

## Production Deployment

### Async Batch Prediction
```rust
use futures::future::join_all;

let series_data = vec![data1, data2, data3];
let futures: Vec<_> = series_data.iter()
    .map(|data| nf.predict_async(data))
    .collect();

let all_forecasts = join_all(futures).await;
```

### Model Serving
```rust
use warp::Filter;

let model = Arc::new(nf);

let predict_route = warp::path("predict")
    .and(warp::post())
    .and(warp::body::json())
    .and(with_model(model.clone()))
    .and_then(handle_prediction);

warp::serve(predict_route)
    .run(([127, 0, 0, 1], 3030))
    .await;
```

## Next Steps

Now that you've created your first forecast, explore more advanced features:

1. **Learn the Fundamentals**: Read [Basic Concepts](basic-concepts.md)
2. **Choose the Right Model**: Explore [Model Overview](models/index.md)
3. **Handle Your Data**: See [Data Handling](data-handling.md)
4. **Optimize Performance**: Check [Performance Guide](performance.md)
5. **Production Deployment**: Read [Best Practices](best-practices.md)

## Common Next Steps

### "My forecasts aren't accurate"
- Try different models ([Model Overview](models/index.md))
- Tune hyperparameters ([Training Guide](training.md))
- Improve data quality ([Data Handling](data-handling.md))

### "Training is too slow"
- Enable GPU acceleration ([Performance](performance.md))
- Use distributed training ([Advanced Usage](advanced-usage.md))
- Optimize data pipeline ([Data Handling](data-handling.md))

### "I need probabilistic forecasts"
- Use models with uncertainty: DeepAR, NBEATS
- Enable prediction intervals in configuration
- See [Prediction Guide](prediction.md)

### "I have multiple time series"
- Use global models (LSTM, Transformer)
- Implement cross-learning
- Check [Advanced Usage](advanced-usage.md)

Happy forecasting! 🎯