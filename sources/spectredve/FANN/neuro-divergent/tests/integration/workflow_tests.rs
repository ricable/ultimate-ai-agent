//! End-to-end integration tests for complete forecasting workflows
//!
//! Tests basic workflows including:
//! - Loading CSV data
//! - Creating multiple models (LSTM, NBEATS, MLP)
//! - Fitting models
//! - Generating forecasts
//! - Evaluating accuracy

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType, PredictionIntervals,
    models::{LSTM, NBEATS, RNN, Transformer},
    data::{TimeSeriesDataset, StandardScaler, MinMaxScaler},
    training::{AccuracyMetrics, ForecastingMetrics},
};
use std::path::Path;
use std::fs;
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration};
use num_traits::Float;
use rand::Rng;

/// Generate synthetic time series data for testing
fn generate_synthetic_data<T: Float>(
    n_series: usize,
    n_points: usize,
    frequency: Frequency,
    with_trend: bool,
    with_seasonality: bool,
) -> Result<TimeSeriesDataFrame<T>, Box<dyn std::error::Error>>
where
    T: From<f64> + Into<f64>,
{
    let mut rng = rand::thread_rng();
    
    // Generate time index
    let start_date = Utc::now() - Duration::days(n_points as i64);
    let dates: Vec<DateTime<Utc>> = (0..n_points)
        .map(|i| start_date + Duration::days(i as i64))
        .collect();
    
    // Generate synthetic series
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for series_id in 0..n_series {
        let series_name = format!("series_{}", series_id);
        
        for (i, &date) in dates.iter().enumerate() {
            unique_ids.push(series_name.clone());
            timestamps.push(date.timestamp());
            
            // Generate synthetic value
            let mut value = T::from(rng.gen_range(10.0..20.0));
            
            // Add trend
            if with_trend {
                value = value + T::from(i as f64 * 0.1);
            }
            
            // Add seasonality (weekly pattern)
            if with_seasonality {
                let seasonal_component = T::from((i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin() * 5.0);
                value = value + seasonal_component;
            }
            
            // Add noise
            let noise = T::from(rng.gen_range(-1.0..1.0));
            value = value + noise;
            
            values.push(value.into());
        }
    }
    
    // Create DataFrame
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema {
            unique_id_col: "unique_id".to_string(),
            ds_col: "ds".to_string(),
            y_col: "y".to_string(),
            static_cols: vec![],
            exog_cols: vec![],
        },
        Some(frequency),
    ))
}

/// Test basic workflow with single model
#[test]
fn test_basic_single_model_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic data
    let data = generate_synthetic_data::<f32>(
        5,    // 5 time series
        100,  // 100 time points each
        Frequency::Daily,
        true, // with trend
        true, // with seasonality
    )?;
    
    // Create LSTM model
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .num_layers(1)
        .dropout(0.1)
        .learning_rate(0.001)
        .max_steps(50)
        .build()?;
    
    // Create NeuralForecast instance
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Fit the model
    nf.fit(data.clone())?;
    assert!(nf.is_fitted());
    
    // Generate forecasts
    let forecasts = nf.predict()?;
    
    // Validate forecast structure
    assert_eq!(forecasts.horizon(), 7);
    assert_eq!(forecasts.model_names(), vec!["LSTM"]);
    
    Ok(())
}

/// Test workflow with multiple models
#[test]
fn test_multi_model_ensemble_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_synthetic_data::<f32>(
        3,    // 3 time series
        150,  // 150 time points
        Frequency::Daily,
        true,
        true,
    )?;
    
    // Create multiple models
    let lstm = LSTM::builder()
        .horizon(14)
        .input_size(28)
        .hidden_size(32)
        .num_layers(1)
        .build()?;
    
    let nbeats = NBEATS::builder()
        .horizon(14)
        .input_size(28)
        .interpretable()
        .build()?;
    
    let rnn = RNN::builder()
        .horizon(14)
        .input_size(28)
        .hidden_size(32)
        .build()?;
    
    // Create NeuralForecast with ensemble
    let mut nf = NeuralForecast::builder()
        .with_models(vec![
            Box::new(lstm),
            Box::new(nbeats),
            Box::new(rnn),
        ])
        .with_frequency(Frequency::Daily)
        .with_num_threads(2)
        .build()?;
    
    // Fit all models
    nf.fit(data.clone())?;
    
    // Generate forecasts
    let forecasts = nf.predict()?;
    
    // Validate
    assert_eq!(forecasts.model_names().len(), 3);
    assert!(forecasts.model_names().contains(&"LSTM".to_string()));
    assert!(forecasts.model_names().contains(&"NBEATS".to_string()));
    assert!(forecasts.model_names().contains(&"RNN".to_string()));
    
    Ok(())
}

/// Test workflow with data preprocessing
#[test]
fn test_workflow_with_preprocessing() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data with different scales
    let mut data = generate_synthetic_data::<f32>(
        2,
        100,
        Frequency::Hourly,
        true,
        false,
    )?;
    
    // Apply standard scaling
    let scaler = StandardScaler::new();
    let scaled_data = scaler.fit_transform(&data)?;
    
    // Create model with scaler configuration
    let lstm = LSTM::builder()
        .horizon(24)
        .input_size(48)
        .hidden_size(64)
        .scaler_type(ScalerType::StandardScaler)
        .build()?;
    
    // Train model
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Hourly)
        .with_local_scaler(ScalerType::StandardScaler)
        .build()?;
    
    nf.fit(scaled_data)?;
    
    // Generate forecasts
    let forecasts = nf.predict()?;
    
    // Inverse transform forecasts
    let original_scale_forecasts = scaler.inverse_transform(&forecasts)?;
    
    Ok(())
}

/// Test workflow with exogenous variables
#[test]
fn test_workflow_with_exogenous_variables() -> Result<(), Box<dyn std::error::Error>> {
    // Generate base data
    let mut data = generate_synthetic_data::<f32>(
        2,
        100,
        Frequency::Daily,
        true,
        true,
    )?;
    
    // Add exogenous variables
    let mut df = data.to_polars()?;
    
    // Add temperature as exogenous variable
    let n_rows = df.height();
    let mut rng = rand::thread_rng();
    let temperatures: Vec<f32> = (0..n_rows)
        .map(|_| rng.gen_range(15.0..30.0))
        .collect();
    
    df = df.with_column(Series::new("temperature", temperatures))?;
    
    // Add holiday indicator
    let holidays: Vec<i32> = (0..n_rows)
        .map(|i| if i % 7 == 0 { 1 } else { 0 })
        .collect();
    
    df = df.with_column(Series::new("holiday", holidays))?;
    
    // Create schema with exogenous variables
    let schema = TimeSeriesSchema {
        unique_id_col: "unique_id".to_string(),
        ds_col: "ds".to_string(),
        y_col: "y".to_string(),
        static_cols: vec![],
        exog_cols: vec!["temperature".to_string(), "holiday".to_string()],
    };
    
    let data_with_exog = TimeSeriesDataFrame::new(df, schema, Some(Frequency::Daily));
    
    // Create model with exogenous features
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .hist_exog_features(vec!["temperature".to_string(), "holiday".to_string()])
        .build()?;
    
    // Train and predict
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(data_with_exog)?;
    
    Ok(())
}

/// Test workflow with prediction intervals
#[test]
fn test_workflow_with_prediction_intervals() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_synthetic_data::<f32>(
        1,
        100,
        Frequency::Daily,
        true,
        true,
    )?;
    
    // Create prediction intervals
    let intervals = PredictionIntervals::new(
        vec![0.80, 0.95],
        IntervalMethod::Quantile,
    )?;
    
    // Create model with prediction intervals
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .prediction_intervals(intervals.clone())
        .num_samples(100)
        .build()?;
    
    // Train model
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .with_prediction_intervals(intervals)
        .build()?;
    
    nf.fit(data)?;
    
    // Generate forecasts with intervals
    let forecasts = nf.predict_with_config(
        PredictionConfig::new()
            .with_intervals()
            .with_num_samples(100)
    )?;
    
    // Validate intervals are present
    assert!(forecasts.has_intervals());
    assert_eq!(forecasts.confidence_levels(), Some(vec![0.80, 0.95]));
    
    Ok(())
}

/// Test workflow with large dataset
#[test]
#[ignore] // This test is slow, run with --ignored flag
fn test_large_dataset_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Generate large dataset
    let data = generate_synthetic_data::<f32>(
        1000,  // 1000 time series
        500,   // 500 time points each
        Frequency::Hourly,
        true,
        true,
    )?;
    
    println!("Generated dataset with {} rows", data.shape().0);
    
    // Create efficient model configuration
    let lstm = LSTM::builder()
        .horizon(24)
        .input_size(48)
        .hidden_size(32)
        .num_layers(1)
        .max_steps(10) // Fewer steps for testing
        .build()?;
    
    // Train with multiple threads
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Hourly)
        .with_num_threads(4)
        .build()?;
    
    let start = std::time::Instant::now();
    nf.fit(data)?;
    let fit_duration = start.elapsed();
    
    println!("Fitting completed in {:?}", fit_duration);
    
    // Generate forecasts
    let start = std::time::Instant::now();
    let forecasts = nf.predict()?;
    let predict_duration = start.elapsed();
    
    println!("Prediction completed in {:?}", predict_duration);
    
    Ok(())
}

/// Test workflow with model persistence
#[test]
fn test_workflow_with_persistence() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_synthetic_data::<f32>(
        2,
        100,
        Frequency::Daily,
        true,
        true,
    )?;
    
    // Create and train model
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(data.clone())?;
    
    // Save model
    let temp_dir = tempfile::tempdir()?;
    let model_path = temp_dir.path().join("neural_forecast.model");
    nf.save(&model_path)?;
    
    // Load model
    let loaded_nf = NeuralForecast::<f32>::load(&model_path)?;
    
    // Verify loaded model works
    assert!(loaded_nf.is_fitted());
    let forecasts = loaded_nf.predict()?;
    assert_eq!(forecasts.horizon(), 7);
    
    Ok(())
}

/// Test workflow error handling
#[test]
fn test_workflow_error_handling() {
    // Test prediction before fitting
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .build()
        .unwrap();
    
    let nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()
        .unwrap();
    
    let result = nf.predict();
    assert!(result.is_err());
    assert!(result.unwrap_err().is_prediction_error());
    
    // Test empty data
    let empty_df = df! {
        "unique_id" => Vec::<String>::new(),
        "ds" => Vec::<i64>::new(),
        "y" => Vec::<f32>::new(),
    }.unwrap();
    
    let empty_data = TimeSeriesDataFrame::new(
        empty_df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    );
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(LSTM::default()))
        .with_frequency(Frequency::Daily)
        .build()
        .unwrap();
    
    let result = nf.fit(empty_data);
    assert!(result.is_err());
    assert!(result.unwrap_err().is_data_error());
}

/// Test workflow with custom loss functions
#[test]
fn test_workflow_with_custom_loss() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let data = generate_synthetic_data::<f32>(
        2,
        100,
        Frequency::Daily,
        true,
        false,
    )?;
    
    // Test different loss functions
    let loss_functions = vec![
        LossFunction::MSE,
        LossFunction::MAE,
        LossFunction::MAPE,
        LossFunction::Huber,
    ];
    
    for loss_fn in loss_functions {
        let lstm = LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .loss_function(loss_fn)
            .max_steps(10)
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(lstm))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        nf.fit(data.clone())?;
        let forecasts = nf.predict()?;
        
        // Validate forecasts were generated
        assert_eq!(forecasts.horizon(), 7);
    }
    
    Ok(())
}

/// Test complete workflow with evaluation metrics
#[test]
fn test_workflow_with_evaluation() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data
    let full_data = generate_synthetic_data::<f32>(
        2,
        120,
        Frequency::Daily,
        true,
        true,
    )?;
    
    // Split into train and test
    let train_size = 100;
    let (train_data, test_data) = full_data.train_test_split(train_size)?;
    
    // Create and train model
    let lstm = LSTM::builder()
        .horizon(20)
        .input_size(30)
        .hidden_size(64)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(train_data)?;
    
    // Generate forecasts on test set
    let forecasts = nf.predict_on(test_data.clone())?;
    
    // Calculate accuracy metrics
    let metrics = AccuracyMetrics::calculate(&test_data, &forecasts)?;
    
    // Validate metrics exist
    assert!(metrics.mae() > 0.0);
    assert!(metrics.mse() > 0.0);
    assert!(metrics.rmse() > 0.0);
    assert!(metrics.mape() > 0.0);
    
    println!("Model evaluation metrics:");
    println!("MAE: {:.4}", metrics.mae());
    println!("MSE: {:.4}", metrics.mse());
    println!("RMSE: {:.4}", metrics.rmse());
    println!("MAPE: {:.2}%", metrics.mape() * 100.0);
    
    Ok(())
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    
    /// Benchmark model fitting performance
    #[test]
    #[ignore]
    fn bench_model_fitting() -> Result<(), Box<dyn std::error::Error>> {
        let sizes = vec![10, 50, 100, 500];
        let n_series = 10;
        
        for size in sizes {
            let data = generate_synthetic_data::<f32>(
                n_series,
                size,
                Frequency::Daily,
                true,
                true,
            )?;
            
            let lstm = LSTM::builder()
                .horizon(7)
                .input_size(14)
                .hidden_size(32)
                .max_steps(10)
                .build()?;
            
            let mut nf = NeuralForecast::builder()
                .with_model(Box::new(lstm))
                .with_frequency(Frequency::Daily)
                .build()?;
            
            let start = std::time::Instant::now();
            nf.fit(data)?;
            let duration = start.elapsed();
            
            println!(
                "Fitting {} series with {} points took {:?}",
                n_series, size, duration
            );
        }
        
        Ok(())
    }
    
    /// Benchmark prediction performance
    #[test]
    #[ignore]
    fn bench_prediction_performance() -> Result<(), Box<dyn std::error::Error>> {
        let data = generate_synthetic_data::<f32>(
            100,
            200,
            Frequency::Daily,
            true,
            true,
        )?;
        
        let lstm = LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(64)
            .max_steps(20)
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(lstm))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        nf.fit(data)?;
        
        // Benchmark prediction
        let n_runs = 10;
        let start = std::time::Instant::now();
        
        for _ in 0..n_runs {
            let _ = nf.predict()?;
        }
        
        let total_duration = start.elapsed();
        let avg_duration = total_duration / n_runs as u32;
        
        println!(
            "Average prediction time: {:?} (over {} runs)",
            avg_duration, n_runs
        );
        
        Ok(())
    }
}