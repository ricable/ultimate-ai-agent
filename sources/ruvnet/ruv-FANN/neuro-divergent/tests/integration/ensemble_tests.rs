//! Integration tests for multi-model ensembles
//!
//! Tests ensemble forecasting scenarios including:
//! - Multiple model types working together
//! - Model averaging and weighting
//! - Ensemble prediction aggregation
//! - Performance comparisons

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType,
    models::{LSTM, NBEATS, DeepAR, RNN, Transformer, MLP},
    data::{TimeSeriesDataset, TimeSeriesDataFrame},
    training::{AccuracyMetrics, ForecastingMetrics},
};
use std::collections::HashMap;
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration};
use num_traits::Float;
use rand::Rng;

/// Helper to generate synthetic data with multiple patterns
fn generate_complex_synthetic_data<T: Float>(
    n_series: usize,
    n_points: usize,
    patterns: Vec<SeriesPattern>,
) -> Result<TimeSeriesDataFrame<T>, Box<dyn std::error::Error>>
where
    T: From<f64> + Into<f64>,
{
    let mut rng = rand::thread_rng();
    let start_date = Utc::now() - Duration::days(n_points as i64);
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for series_id in 0..n_series {
        let series_name = format!("series_{}", series_id);
        let pattern = &patterns[series_id % patterns.len()];
        
        for i in 0..n_points {
            let date = start_date + Duration::days(i as i64);
            unique_ids.push(series_name.clone());
            timestamps.push(date.timestamp());
            
            // Generate value based on pattern
            let value = pattern.generate_value::<T>(i, &mut rng);
            values.push(value.into());
        }
    }
    
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

/// Pattern generator for synthetic series
struct SeriesPattern {
    trend_slope: f64,
    seasonal_amplitude: f64,
    seasonal_period: f64,
    noise_level: f64,
    base_value: f64,
}

impl SeriesPattern {
    fn generate_value<T: Float>(&self, time_index: usize, rng: &mut impl Rng) -> T 
    where
        T: From<f64>,
    {
        let t = time_index as f64;
        
        // Base value
        let mut value = self.base_value;
        
        // Add trend
        value += self.trend_slope * t;
        
        // Add seasonality
        value += self.seasonal_amplitude * (2.0 * std::f64::consts::PI * t / self.seasonal_period).sin();
        
        // Add noise
        value += rng.gen_range(-self.noise_level..self.noise_level);
        
        T::from(value)
    }
}

/// Test basic ensemble with different model types
#[test]
fn test_basic_ensemble() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with different patterns
    let patterns = vec![
        SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        },
        SeriesPattern {
            trend_slope: -0.05,
            seasonal_amplitude: 3.0,
            seasonal_period: 30.0,
            noise_level: 0.5,
            base_value: 15.0,
        },
    ];
    
    let data = generate_complex_synthetic_data::<f32>(4, 200, patterns)?;
    
    // Create diverse ensemble
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .num_layers(2)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .interpretable()
            .build()?),
        Box::new(RNN::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(48)
            .build()?),
        Box::new(MLP::builder()
            .horizon(14)
            .input_size(28)
            .hidden_layers(vec![64, 32])
            .build()?),
    ];
    
    // Create ensemble
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .with_num_threads(2)
        .build()?;
    
    // Fit ensemble
    nf.fit(data.clone())?;
    
    // Generate forecasts
    let forecasts = nf.predict()?;
    
    // Verify all models produced forecasts
    assert_eq!(forecasts.model_names().len(), 4);
    assert!(forecasts.model_names().contains(&"LSTM".to_string()));
    assert!(forecasts.model_names().contains(&"NBEATS".to_string()));
    assert!(forecasts.model_names().contains(&"RNN".to_string()));
    assert!(forecasts.model_names().contains(&"MLP".to_string()));
    
    Ok(())
}

/// Test ensemble with model weighting
#[test]
fn test_weighted_ensemble() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_complex_synthetic_data::<f32>(
        2, 
        150,
        vec![SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        }],
    )?;
    
    // Split data for validation
    let (train_data, val_data) = data.train_test_split(120)?;
    
    // Create models
    let models = vec![
        ("LSTM", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .build()?) as Box<dyn BaseModel<f32>>),
        ("NBEATS", Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .build()?)),
        ("RNN", Box::new(RNN::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .build()?)),
    ];
    
    // Train individual models and calculate validation performance
    let mut model_weights = HashMap::new();
    let mut individual_forecasts = Vec::new();
    
    for (name, mut model) in models {
        // Train model
        let mut nf = NeuralForecast::builder()
            .with_model(model)
            .with_frequency(Frequency::Daily)
            .build()?;
        
        nf.fit(train_data.clone())?;
        
        // Validate
        let val_forecast = nf.predict_on(val_data.clone())?;
        individual_forecasts.push((name.to_string(), val_forecast.clone()));
        
        // Calculate validation metrics
        let metrics = AccuracyMetrics::calculate(&val_data, &val_forecast)?;
        
        // Weight inversely proportional to error
        let weight = 1.0 / (metrics.mae() + 1e-6);
        model_weights.insert(name.to_string(), weight);
    }
    
    // Normalize weights
    let total_weight: f32 = model_weights.values().sum();
    for weight in model_weights.values_mut() {
        *weight /= total_weight;
    }
    
    println!("Model weights based on validation performance:");
    for (name, weight) in &model_weights {
        println!("{}: {:.3}", name, weight);
    }
    
    // Create weighted ensemble forecast
    let weighted_forecast = create_weighted_ensemble_forecast(
        individual_forecasts,
        model_weights,
    )?;
    
    Ok(())
}

/// Test ensemble with different architectures for different series
#[test]
fn test_heterogeneous_ensemble() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with very different patterns
    let patterns = vec![
        // Linear trend with weekly seasonality
        SeriesPattern {
            trend_slope: 0.2,
            seasonal_amplitude: 10.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 50.0,
        },
        // No trend, strong monthly seasonality
        SeriesPattern {
            trend_slope: 0.0,
            seasonal_amplitude: 20.0,
            seasonal_period: 30.0,
            noise_level: 2.0,
            base_value: 100.0,
        },
        // Negative trend, no seasonality
        SeriesPattern {
            trend_slope: -0.3,
            seasonal_amplitude: 0.0,
            seasonal_period: 1.0,
            noise_level: 3.0,
            base_value: 80.0,
        },
    ];
    
    let data = generate_complex_synthetic_data::<f32>(6, 300, patterns)?;
    
    // Create specialized models for different patterns
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        // LSTM for complex patterns
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(42)
            .hidden_size(128)
            .num_layers(3)
            .bidirectional(true)
            .build()?),
        // NBEATS for interpretable decomposition
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(42)
            .interpretable()
            .with_trend()
            .with_seasonality()
            .seasonality_period(7)
            .build()?),
        // Transformer for long-range dependencies
        Box::new(Transformer::builder()
            .horizon(14)
            .input_size(42)
            .d_model(64)
            .num_heads(4)
            .num_layers(2)
            .build()?),
    ];
    
    // Train ensemble
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .with_num_threads(3)
        .build()?;
    
    nf.fit(data)?;
    
    // Test on each type of series
    let test_data = generate_complex_synthetic_data::<f32>(3, 50, patterns)?;
    let forecasts = nf.predict_on(test_data)?;
    
    // Analyze which models perform best on which patterns
    // This would involve comparing individual model forecasts
    
    Ok(())
}

/// Test ensemble robustness to outliers
#[test]
fn test_ensemble_robustness() -> Result<(), Box<dyn std::error::Error>> {
    // Generate clean data
    let mut data = generate_complex_synthetic_data::<f32>(
        2,
        200,
        vec![SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        }],
    )?;
    
    // Inject outliers
    let mut df = data.to_polars()?;
    let y_values = df.column("y")?.f64()?.to_vec();
    let mut y_with_outliers = y_values.clone();
    
    // Add random outliers
    let mut rng = rand::thread_rng();
    for i in 0..y_with_outliers.len() {
        if rng.gen_bool(0.05) { // 5% outlier rate
            y_with_outliers[i] = Some(y_with_outliers[i].unwrap_or(0.0) * rng.gen_range(2.0..5.0));
        }
    }
    
    df = df.with_column(Series::new("y", y_with_outliers))?;
    data = TimeSeriesDataFrame::from_polars(df, data.schema.clone(), data.frequency)?;
    
    // Create ensemble with robust and non-robust models
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        // LSTM with Huber loss (robust)
        Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .loss_function(LossFunction::Huber)
            .build()?),
        // MLP with MSE loss (sensitive to outliers)
        Box::new(MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![32, 16])
            .loss_function(LossFunction::MSE)
            .build()?),
        // NBEATS with MAE loss (somewhat robust)
        Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .loss_function(LossFunction::MAE)
            .build()?),
    ];
    
    // Train ensemble
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // This should handle outliers gracefully
    nf.fit(data)?;
    
    // Generate forecasts
    let forecasts = nf.predict()?;
    
    // Ensemble should produce reasonable forecasts despite outliers
    assert_eq!(forecasts.model_names().len(), 3);
    
    Ok(())
}

/// Test ensemble with cross-validation
#[test]
fn test_ensemble_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_complex_synthetic_data::<f32>(
        3,
        300,
        vec![SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        }],
    )?;
    
    // Create ensemble
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(32)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Perform cross-validation
    let cv_config = CrossValidationConfig::new(3, 14)
        .with_step_size(14)
        .with_refit(true);
    
    let cv_results = nf.cross_validation(data, cv_config)?;
    
    // Analyze CV results
    assert_eq!(cv_results.n_windows(), 3);
    assert_eq!(cv_results.model_names(), vec!["LSTM", "NBEATS"]);
    
    // Calculate metrics for each model
    let model_metrics = cv_results.calculate_metrics()?;
    
    println!("Cross-validation results:");
    for (model_name, metrics) in model_metrics {
        println!("{}: MAE={:.3}, RMSE={:.3}", 
                 model_name, metrics.mae(), metrics.rmse());
    }
    
    Ok(())
}

/// Test ensemble with dynamic model selection
#[test]
fn test_dynamic_model_selection() -> Result<(), Box<dyn std::error::Error>> {
    // Generate diverse dataset
    let patterns = vec![
        SeriesPattern {
            trend_slope: 0.2,
            seasonal_amplitude: 10.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 50.0,
        },
        SeriesPattern {
            trend_slope: -0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 30.0,
            noise_level: 2.0,
            base_value: 100.0,
        },
    ];
    
    let data = generate_complex_synthetic_data::<f32>(10, 200, patterns)?;
    
    // Create model pool
    let model_pool = vec![
        ("LSTM_small", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(16)
            .build()?) as Box<dyn BaseModel<f32>>),
        ("LSTM_large", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(64)
            .num_layers(2)
            .build()?)),
        ("NBEATS_generic", Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .build()?)),
        ("NBEATS_interpretable", Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .interpretable()
            .build()?)),
        ("MLP", Box::new(MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![32, 16])
            .build()?)),
    ];
    
    // Select best models for each series using validation
    let series_ids = data.unique_series_ids()?;
    let mut selected_models = HashMap::new();
    
    for series_id in &series_ids {
        // Get data for this series
        let series_data = data.filter_series(series_id)?;
        let (train, val) = series_data.train_test_split(150)?;
        
        // Evaluate each model
        let mut best_model = String::new();
        let mut best_score = f32::INFINITY;
        
        for (name, model) in &model_pool {
            let mut model_clone = model.clone();
            
            // Train on series
            model_clone.fit(&train)?;
            
            // Validate
            let forecast = model_clone.predict(&val)?;
            let metrics = AccuracyMetrics::calculate(&val, &forecast)?;
            
            if metrics.mae() < best_score {
                best_score = metrics.mae();
                best_model = name.to_string();
            }
        }
        
        selected_models.insert(series_id.clone(), best_model);
        println!("Series {}: selected model {}", series_id, selected_models[series_id]);
    }
    
    Ok(())
}

/// Test ensemble forecasting speed
#[test]
fn test_ensemble_performance() -> Result<(), Box<dyn std::error::Error>> {
    // Generate moderate dataset
    let data = generate_complex_synthetic_data::<f32>(
        50,
        200,
        vec![SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        }],
    )?;
    
    // Create ensemble of different sizes
    let ensemble_sizes = vec![1, 3, 5];
    
    for size in ensemble_sizes {
        let models: Vec<Box<dyn BaseModel<f32>>> = (0..size)
            .map(|i| {
                Box::new(LSTM::builder()
                    .horizon(7)
                    .input_size(14)
                    .hidden_size(32)
                    .max_steps(10)
                    .build()
                    .unwrap()) as Box<dyn BaseModel<f32>>
            })
            .collect();
        
        let mut nf = NeuralForecast::builder()
            .with_models(models)
            .with_frequency(Frequency::Daily)
            .with_num_threads(4)
            .build()?;
        
        // Measure fitting time
        let fit_start = std::time::Instant::now();
        nf.fit(data.clone())?;
        let fit_time = fit_start.elapsed();
        
        // Measure prediction time
        let pred_start = std::time::Instant::now();
        let _ = nf.predict()?;
        let pred_time = pred_start.elapsed();
        
        println!("Ensemble size {}: fit={:?}, predict={:?}", 
                 size, fit_time, pred_time);
    }
    
    Ok(())
}

/// Helper function to create weighted ensemble forecast
fn create_weighted_ensemble_forecast(
    individual_forecasts: Vec<(String, ForecastDataFrame<f32>)>,
    weights: HashMap<String, f32>,
) -> Result<ForecastDataFrame<f32>, Box<dyn std::error::Error>> {
    // This is a placeholder - actual implementation would aggregate forecasts
    // based on weights
    Ok(individual_forecasts[0].1.clone())
}

/// Test ensemble with probabilistic models
#[test]
fn test_probabilistic_ensemble() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_complex_synthetic_data::<f32>(
        2,
        150,
        vec![SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        }],
    )?;
    
    // Create ensemble with probabilistic models
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        // DeepAR for probabilistic forecasts
        Box::new(DeepAR::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .num_layers(2)
            .num_samples(100)
            .build()?),
        // LSTM with prediction intervals
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .prediction_intervals(PredictionIntervals::default())
            .num_samples(100)
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .with_prediction_intervals(PredictionIntervals::default())
        .build()?;
    
    // Fit ensemble
    nf.fit(data)?;
    
    // Generate probabilistic forecasts
    let forecasts = nf.predict_with_config(
        PredictionConfig::new()
            .with_intervals()
            .with_num_samples(200)
    )?;
    
    // Verify prediction intervals
    assert!(forecasts.has_intervals());
    assert_eq!(forecasts.confidence_levels(), Some(vec![0.8, 0.9, 0.95]));
    
    Ok(())
}

/// Test ensemble memory efficiency
#[test]
fn test_ensemble_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    // Generate large dataset
    let data = generate_complex_synthetic_data::<f32>(
        100,
        500,
        vec![SeriesPattern {
            trend_slope: 0.1,
            seasonal_amplitude: 5.0,
            seasonal_period: 7.0,
            noise_level: 1.0,
            base_value: 20.0,
        }],
    )?;
    
    // Create memory-efficient ensemble
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        // Small models with parameter sharing
        Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(16)
            .num_layers(1)
            .build()?),
        Box::new(MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![16])
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .with_num_threads(2)
        .build()?;
    
    // Train with batching to reduce memory usage
    nf.fit_with_validation(
        data,
        ValidationConfig::new()
            .with_validation_split(0.2)
            .with_shuffle(false)
    )?;
    
    // Generate forecasts in batches
    let forecasts = nf.predict()?;
    
    Ok(())
}