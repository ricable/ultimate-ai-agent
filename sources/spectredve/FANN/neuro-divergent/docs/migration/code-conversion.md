# Code Conversion Guide: Python to Rust Examples

This guide provides comprehensive side-by-side examples of converting Python NeuralForecast code to Rust neuro-divergent, covering all major workflows and patterns.

## Table of Contents

1. [Basic Workflow Conversion](#basic-workflow-conversion)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing) 
3. [Model Training and Prediction](#model-training-and-prediction)
4. [Cross-Validation Workflows](#cross-validation-workflows)
5. [Ensemble Methods](#ensemble-methods)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Custom Models](#custom-models)
8. [Error Handling Patterns](#error-handling-patterns)
9. [Performance Optimization](#performance-optimization)
10. [Integration Patterns](#integration-patterns)

## Basic Workflow Conversion

### Simple Forecasting Pipeline

**Python NeuralForecast**:
```python
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.utils import AirPassengersDF

# Load data
df = AirPassengersDF

# Create and configure model
model = LSTM(h=12, input_size=24, hidden_size=128)

# Initialize NeuralForecast
nf = NeuralForecast(models=[model], freq='M')

# Train model
nf.fit(df)

# Generate predictions
forecasts = nf.predict()
print(forecasts.head())
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::{NeuralForecast, models::LSTM, Frequency};
use polars::prelude::*;
use anyhow::Result;

fn main() -> Result<()> {
    // Load data
    let df = LazyFrame::scan_csv("air_passengers.csv", Default::default())?
        .collect()?;

    // Create and configure model
    let model = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .hidden_size(128)
        .build()?;

    // Initialize NeuralForecast
    let mut nf = NeuralForecast::builder()
        .with_models(vec![Box::new(model)])
        .with_frequency(Frequency::Monthly)
        .build()?;

    // Train model
    nf.fit(df.clone())?;

    // Generate predictions
    let forecasts = nf.predict()?;
    println!("{}", forecasts.head(Some(5)));

    Ok(())
}
```

### Multiple Models Pipeline

**Python NeuralForecast**:
```python
from neuralforecast.models import LSTM, NBEATS, TFT

# Multiple models
models = [
    LSTM(h=12, input_size=24, hidden_size=64),
    NBEATS(h=12, input_size=24, stack_types=['trend', 'seasonality']),
    TFT(h=12, input_size=24, hidden_size=128)
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df)
forecasts = nf.predict()

# Access individual model predictions
lstm_forecast = forecasts[['ds', 'unique_id', 'LSTM']]
nbeats_forecast = forecasts[['ds', 'unique_id', 'NBEATS']]
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::models::{LSTM, NBEATS, TFT};

fn multi_model_pipeline(df: DataFrame) -> Result<DataFrame> {
    // Multiple models
    let models: Vec<Box<dyn Model>> = vec![
        Box::new(LSTM::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(64)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(12)
            .input_size(24)
            .stack_types(vec!["trend", "seasonality"])
            .build()?),
        Box::new(TFT::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(128)
            .build()?),
    ];

    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;

    nf.fit(df.clone())?;
    let forecasts = nf.predict()?;

    // Access individual model predictions
    let lstm_forecast = forecasts.select([
        col("ds"), col("unique_id"), col("LSTM")
    ])?;
    
    Ok(forecasts)
}
```

## Data Loading and Preprocessing

### CSV Data Loading

**Python NeuralForecast**:
```python
import pandas as pd

# Load CSV data
df = pd.read_csv('data.csv')

# Basic preprocessing
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values(['unique_id', 'ds'])
df = df.dropna()

# Feature engineering
df['month'] = df['ds'].dt.month
df['year'] = df['ds'].dt.year
df['dayofweek'] = df['ds'].dt.dayofweek

# Groupby operations
df_processed = df.groupby('unique_id').apply(
    lambda x: x.assign(y_lag1=x['y'].shift(1))
).reset_index(drop=True)
```

**Rust neuro-divergent**:
```rust
use polars::prelude::*;
use chrono::{Datelike, Weekday};

fn load_and_preprocess_data(file_path: &str) -> Result<DataFrame> {
    let df = LazyFrame::scan_csv(file_path, Default::default())?
        // Parse datetime column
        .with_columns([
            col("ds").str().strptime(StrptimeOptions::default()),
        ])
        // Sort by unique_id and ds
        .sort_by_exprs(
            [col("unique_id"), col("ds")],
            SortMultipleOptions::default(),
        )
        // Drop null values
        .drop_nulls(None)
        // Feature engineering
        .with_columns([
            col("ds").dt().month().alias("month"),
            col("ds").dt().year().alias("year"),
            col("ds").dt().weekday().alias("dayofweek"),
        ])
        // Groupby operations
        .group_by(["unique_id"])
        .agg([
            col("ds"),
            col("y"),
            col("y").shift(1).alias("y_lag1"),
            col("month"),
            col("year"),
            col("dayofweek"),
        ])
        .explode([col("ds"), col("y"), col("y_lag1"), col("month"), col("year"), col("dayofweek")])
        .collect()?;

    Ok(df)
}
```

### Complex Data Transformations

**Python NeuralForecast**:
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Outlier detection and handling
    Q1 = df.groupby('unique_id')['y'].quantile(0.25)
    Q3 = df.groupby('unique_id')['y'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Scale features
    scaler = StandardScaler()
    df['y_scaled'] = scaler.fit_transform(df[['y']])
    
    # Add statistical features
    df['rolling_mean'] = df.groupby('unique_id')['y'].transform(
        lambda x: x.rolling(window=7).mean()
    )
    df['rolling_std'] = df.groupby('unique_id')['y'].transform(
        lambda x: x.rolling(window=7).std()
    )
    
    return df
```

**Rust neuro-divergent**:
```rust
use polars::prelude::*;

fn preprocess_data(df: DataFrame) -> Result<DataFrame> {
    let processed_df = df
        .lazy()
        // Handle missing values (forward fill then backward fill)
        .with_columns([
            col("y").forward_fill(None).backward_fill(None)
        ])
        // Outlier detection and handling per group
        .with_columns([
            col("y").quantile(lit(0.25), QuantileInterpolOptions::default())
                .over([col("unique_id")]).alias("q1"),
            col("y").quantile(lit(0.75), QuantileInterpolOptions::default())
                .over([col("unique_id")]).alias("q3"),
        ])
        .with_columns([
            (col("q3") - col("q1")).alias("iqr")
        ])
        // Scale features (standardization per group)
        .with_columns([
            ((col("y") - col("y").mean().over([col("unique_id")])) / 
             col("y").std(1).over([col("unique_id")])).alias("y_scaled")
        ])
        // Add statistical features
        .with_columns([
            col("y").rolling_mean(RollingOptions::default().window_size(Duration::parse("7i")))
                .over([col("unique_id")]).alias("rolling_mean"),
            col("y").rolling_std(RollingOptions::default().window_size(Duration::parse("7i")))
                .over([col("unique_id")]).alias("rolling_std"),
        ])
        // Clean up temporary columns
        .drop(["q1", "q3", "iqr"])
        .collect()?;

    Ok(processed_df)
}
```

## Model Training and Prediction

### Training with Validation

**Python NeuralForecast**:
```python
from sklearn.model_selection import train_test_split

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Configure model with validation
model = LSTM(
    h=12,
    input_size=24,
    hidden_size=128,
    max_steps=1000,
    learning_rate=0.001,
    early_stop_patience_steps=50,
    val_check_steps=10
)

# Train with validation
nf = NeuralForecast(models=[model], freq='D')
nf.fit(train_df, val_size=len(val_df))

# Evaluate on validation set
val_forecasts = nf.predict(val_df)
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::training::{TrainingConfig, EarlyStoppingConfig};

fn train_with_validation(df: DataFrame) -> Result<NeuralForecast> {
    // Split data
    let n_train = (df.height() as f64 * 0.8) as usize;
    let train_df = df.slice(0, n_train);
    let val_df = df.slice(n_train, df.height() - n_train);

    // Configure model with validation
    let model = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .hidden_size(128)
        .training_config(TrainingConfig {
            max_steps: 1000,
            learning_rate: 0.001,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 50,
                validation_check_steps: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .build()?;

    // Train with validation
    let mut nf = NeuralForecast::builder()
        .with_models(vec![Box::new(model)])
        .with_frequency(Frequency::Daily)
        .build()?;

    nf.fit_with_validation(train_df, Some(val_df))?;

    Ok(nf)
}
```

### Custom Training Loop

**Python NeuralForecast**:
```python
import torch
import torch.nn as nn
from torch.optim import Adam

class CustomTrainingLoop:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_step(self, batch):
        self.optimizer.zero_grad()
        predictions = self.model(batch['x'])
        loss = self.criterion(predictions, batch['y'])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, dataloader, epochs=100):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
            losses.append(epoch_loss / len(dataloader))
        return losses
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::training::{Trainer, Loss, Optimizer};
use neuro_divergent::data::DataLoader;

struct CustomTrainingLoop {
    model: Box<dyn Model>,
    optimizer: Box<dyn Optimizer>,
    criterion: Box<dyn Loss>,
}

impl CustomTrainingLoop {
    fn new(model: Box<dyn Model>, learning_rate: f64) -> Self {
        Self {
            model,
            optimizer: Box::new(Adam::new(learning_rate)),
            criterion: Box::new(MSELoss::new()),
        }
    }
    
    fn train_step(&mut self, batch: &Batch) -> Result<f64> {
        self.optimizer.zero_grad();
        let predictions = self.model.forward(&batch.x)?;
        let loss = self.criterion.compute(&predictions, &batch.y)?;
        let gradients = self.criterion.backward(&predictions, &batch.y)?;
        self.model.backward(&gradients)?;
        self.optimizer.step(self.model.parameters_mut())?;
        Ok(loss)
    }
    
    fn train(&mut self, dataloader: &mut DataLoader, epochs: usize) -> Result<Vec<f64>> {
        let mut losses = Vec::new();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            
            for batch in dataloader.iter() {
                let loss = self.train_step(&batch?)?;
                epoch_loss += loss;
                batch_count += 1;
            }
            
            losses.push(epoch_loss / batch_count as f64);
        }
        
        Ok(losses)
    }
}
```

## Cross-Validation Workflows

### Time Series Cross-Validation

**Python NeuralForecast**:
```python
from neuralforecast.models import LSTM
from neuralforecast.cross_validation import cross_validation

# Configure cross-validation
cv_results = cross_validation(
    df=df,
    models=[LSTM(h=12, input_size=24)],
    freq='D',
    h=12,
    n_windows=5,
    step_size=12,
    fitted=True
)

# Analyze results
mae_scores = cv_results.groupby(['unique_id', 'cutoff'])['LSTM'].apply(
    lambda x: np.mean(np.abs(x - cv_results.loc[x.index, 'y']))
)
print(f"Mean MAE: {mae_scores.mean():.4f}")
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::validation::{CrossValidationConfig, TimeSeriesSplit};
use neuro_divergent::metrics::MAE;

fn time_series_cross_validation(df: DataFrame) -> Result<DataFrame> {
    let model = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .build()?;

    let cv_config = CrossValidationConfig {
        n_windows: 5,
        horizon: 12,
        step_size: 12,
        fitted: true,
        ..Default::default()
    };

    let mut nf = NeuralForecast::builder()
        .with_models(vec![Box::new(model)])
        .with_frequency(Frequency::Daily)
        .build()?;

    let cv_results = nf.cross_validation(df, cv_config)?;

    // Analyze results
    let mae = MAE::new();
    let mae_scores = cv_results
        .group_by(["unique_id", "cutoff"])
        .agg([
            mae.compute(col("LSTM"), col("y")).alias("mae")
        ])
        .collect()?;

    let mean_mae = mae_scores
        .column("mae")?
        .mean()
        .unwrap_or(f64::NAN);

    println!("Mean MAE: {:.4}", mean_mae);

    Ok(cv_results)
}
```

## Ensemble Methods

### Model Averaging

**Python NeuralForecast**:
```python
from neuralforecast.models import LSTM, NBEATS, TFT
import numpy as np

# Individual models
models = [
    LSTM(h=12, input_size=24, hidden_size=64),
    NBEATS(h=12, input_size=24),
    TFT(h=12, input_size=24, hidden_size=128)
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df)
forecasts = nf.predict()

# Ensemble averaging
ensemble_forecast = forecasts[['ds', 'unique_id']].copy()
model_cols = ['LSTM', 'NBEATS', 'TFT']
ensemble_forecast['ensemble'] = forecasts[model_cols].mean(axis=1)

# Weighted averaging
weights = [0.4, 0.3, 0.3]  # Based on validation performance
ensemble_forecast['weighted_ensemble'] = np.average(
    forecasts[model_cols].values, 
    axis=1, 
    weights=weights
)
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::ensemble::{EnsembleMethod, WeightedAverage};

fn ensemble_forecasting(df: DataFrame) -> Result<DataFrame> {
    // Individual models
    let models: Vec<Box<dyn Model>> = vec![
        Box::new(LSTM::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(64)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(12)
            .input_size(24)
            .build()?),
        Box::new(TFT::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(128)
            .build()?),
    ];

    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;

    nf.fit(df.clone())?;
    let forecasts = nf.predict()?;

    // Ensemble averaging
    let ensemble_forecast = forecasts
        .with_columns([
            // Simple average
            (col("LSTM") + col("NBEATS") + col("TFT")) / lit(3.0)
                .alias("ensemble"),
            
            // Weighted average
            (col("LSTM") * lit(0.4) + 
             col("NBEATS") * lit(0.3) + 
             col("TFT") * lit(0.3))
                .alias("weighted_ensemble"),
        ])
        .select([
            col("ds"), col("unique_id"), 
            col("ensemble"), col("weighted_ensemble")
        ]);

    Ok(ensemble_forecast)
}
```

## Hyperparameter Tuning

### Grid Search

**Python NeuralForecast**:
```python
from itertools import product
from sklearn.model_selection import ParameterGrid

# Define parameter grid
param_grid = {
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout': [0.0, 0.1, 0.2]
}

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    model = LSTM(h=12, input_size=24, **params)
    nf = NeuralForecast(models=[model], freq='D')
    
    # Cross-validation
    cv_results = cross_validation(
        df=train_df, models=[model], freq='D', 
        h=12, n_windows=3
    )
    
    score = np.mean((cv_results['LSTM'] - cv_results['y']) ** 2)
    
    if score < best_score:
        best_score = score
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

**Rust neuro-divergent**:
```rust
use neuro_divergent::tuning::{GridSearch, ParameterGrid, HyperparameterTuner};
use neuro_divergent::metrics::MSE;

fn hyperparameter_tuning(train_df: DataFrame) -> Result<LSTM> {
    // Define parameter grid
    let param_grid = ParameterGrid::new()
        .add_param("hidden_size", vec![64, 128, 256])
        .add_param("num_layers", vec![1, 2, 3])
        .add_param("learning_rate", vec![0.001, 0.01, 0.1])
        .add_param("dropout", vec![0.0, 0.1, 0.2]);

    let mut best_score = f64::INFINITY;
    let mut best_params = None;

    for params in param_grid.iter() {
        let model = LSTM::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(params.get("hidden_size").unwrap())
            .num_layers(params.get("num_layers").unwrap())
            .learning_rate(params.get("learning_rate").unwrap())
            .dropout(params.get("dropout").unwrap())
            .build()?;

        let mut nf = NeuralForecast::builder()
            .with_models(vec![Box::new(model)])
            .with_frequency(Frequency::Daily)
            .build()?;

        // Cross-validation
        let cv_config = CrossValidationConfig {
            n_windows: 3,
            horizon: 12,
            ..Default::default()
        };
        
        let cv_results = nf.cross_validation(train_df.clone(), cv_config)?;
        
        let mse = MSE::new();
        let score = mse.compute_aggregated(&cv_results, "LSTM", "y")?;

        if score < best_score {
            best_score = score;
            best_params = Some(params.clone());
        }
    }

    println!("Best parameters: {:?}", best_params);
    println!("Best score: {:.6}", best_score);

    // Build final model with best parameters
    let best_params = best_params.unwrap();
    let best_model = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .hidden_size(best_params.get("hidden_size").unwrap())
        .num_layers(best_params.get("num_layers").unwrap())
        .learning_rate(best_params.get("learning_rate").unwrap())
        .dropout(best_params.get("dropout").unwrap())
        .build()?;

    Ok(best_model)
}
```

## Error Handling Patterns

### Comprehensive Error Handling

**Python NeuralForecast**:
```python
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingPipeline:
    def __init__(self):
        self.model = None
        self.is_fitted = False
    
    def run_pipeline(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            # Validate input data
            self._validate_data(df)
            
            # Train model
            self._train_model(df)
            
            # Generate forecasts
            forecasts = self._generate_forecasts()
            
            logger.info("Pipeline completed successfully")
            return forecasts
            
        except ValueError as e:
            logger.error(f"Data validation error: {e}")
            return None
        except RuntimeError as e:
            logger.error(f"Training error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def _validate_data(self, df: pd.DataFrame):
        required_cols = ['unique_id', 'ds', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        if df['y'].isna().any():
            raise ValueError("Target column contains null values")
    
    def _train_model(self, df: pd.DataFrame):
        try:
            self.model = LSTM(h=12, input_size=24)
            nf = NeuralForecast(models=[self.model], freq='D')
            nf.fit(df)
            self.is_fitted = True
        except Exception as e:
            raise RuntimeError(f"Model training failed: {e}")
    
    def _generate_forecasts(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict()
```

**Rust neuro-divergent**:
```rust
use anyhow::{Context, Result, bail};
use tracing::{info, error, warn};

pub struct ForecastingPipeline {
    model: Option<Box<dyn Model>>,
    is_fitted: bool,
}

impl ForecastingPipeline {
    pub fn new() -> Self {
        Self {
            model: None,
            is_fitted: false,
        }
    }
    
    pub fn run_pipeline(&mut self, df: DataFrame) -> Result<DataFrame> {
        // Validate input data
        self.validate_data(&df)
            .context("Data validation failed")?;
        
        // Train model
        self.train_model(df.clone())
            .context("Model training failed")?;
        
        // Generate forecasts
        let forecasts = self.generate_forecasts()
            .context("Forecast generation failed")?;
        
        info!("Pipeline completed successfully");
        Ok(forecasts)
    }
    
    fn validate_data(&self, df: &DataFrame) -> Result<()> {
        let required_cols = ["unique_id", "ds", "y"];
        let df_cols = df.get_column_names();
        
        for col in required_cols {
            if !df_cols.contains(&col) {
                bail!("Missing required column: {}", col);
            }
        }
        
        let y_null_count = df.column("y")?.null_count();
        if y_null_count > 0 {
            bail!("Target column contains {} null values", y_null_count);
        }
        
        Ok(())
    }
    
    fn train_model(&mut self, df: DataFrame) -> Result<()> {
        let model = LSTM::builder()
            .horizon(12)
            .input_size(24)
            .build()
            .context("Failed to build LSTM model")?;
        
        let mut nf = NeuralForecast::builder()
            .with_models(vec![Box::new(model)])
            .with_frequency(Frequency::Daily)
            .build()
            .context("Failed to build NeuralForecast")?;
        
        nf.fit(df)
            .context("Model fitting failed")?;
        
        self.model = Some(Box::new(nf));
        self.is_fitted = true;
        
        Ok(())
    }
    
    fn generate_forecasts(&self) -> Result<DataFrame> {
        match &self.model {
            Some(model) if self.is_fitted => {
                model.predict()
                    .context("Prediction failed")
            }
            _ => bail!("Model must be fitted before prediction"),
        }
    }
}

// Usage with comprehensive error handling
fn main() -> Result<()> {
    tracing_subscriber::init();
    
    let df = LazyFrame::scan_csv("data.csv", Default::default())?
        .collect()
        .context("Failed to load data")?;
    
    let mut pipeline = ForecastingPipeline::new();
    
    match pipeline.run_pipeline(df) {
        Ok(forecasts) => {
            info!("Forecasting completed successfully");
            println!("{}", forecasts.head(Some(10)));
        }
        Err(e) => {
            error!("Pipeline failed: {:?}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}
```

---

**Next**: Continue to [Data Formats](data-formats.md) for pandas to polars migration guide.