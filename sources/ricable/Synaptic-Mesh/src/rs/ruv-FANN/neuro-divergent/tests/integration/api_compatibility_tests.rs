//! Integration tests for API compatibility with Python NeuralForecast
//!
//! Tests API compatibility scenarios including:
//! - Replicating Python NeuralForecast examples
//! - Ensuring identical API surface
//! - Testing all configuration options
//! - Validating output formats

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType, PredictionIntervals,
    models::{LSTM, NBEATS, DeepAR, RNN, Transformer, MLP, NHITS, TFT},
    data::{TimeSeriesDataFrame, TimeSeriesSchema},
    config::{
        LossFunction, OptimizerType, Device, CrossValidationConfig,
        EarlyStoppingConfig, SchedulerConfig, SchedulerType,
    },
};
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration};
use num_traits::Float;
use serde_json;
use std::collections::HashMap;

/// Test basic Python NeuralForecast example
#[test]
fn test_python_basic_example() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate Python example:
    // ```python
    // from neuralforecast import NeuralForecast
    // from neuralforecast.models import LSTM
    // 
    // models = [LSTM(h=12, input_size=24, hidden_size=64)]
    // nf = NeuralForecast(models=models, freq='D')
    // nf.fit(df)
    // forecasts = nf.predict()
    // ```
    
    // Create sample data similar to Python
    let df = create_sample_dataframe()?;
    
    // Create LSTM model with same parameters
    let lstm = LSTM::builder()
        .horizon(12)          // h=12
        .input_size(24)       // input_size=24
        .hidden_size(64)      // hidden_size=64
        .build()?;
    
    // Create NeuralForecast instance
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)  // freq='D'
        .build()?;
    
    // Fit model
    nf.fit(df)?;
    
    // Generate predictions
    let forecasts = nf.predict()?;
    
    // Validate output format
    assert_eq!(forecasts.horizon(), 12);
    assert!(forecasts.has_column("LSTM"));
    
    Ok(())
}

/// Test Python auto model example
#[test]
fn test_python_auto_model_example() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate Python example:
    // ```python
    // from neuralforecast.models import AutoLSTM, AutoNBEATS
    // 
    // models = [
    //     AutoLSTM(h=12, loss=MAE(), num_samples=5),
    //     AutoNBEATS(h=12, loss=MAE(), num_samples=5)
    // ]
    // ```
    
    let df = create_sample_dataframe()?;
    
    // Create auto-tuning models
    let auto_lstm = LSTM::auto()
        .horizon(12)
        .loss_function(LossFunction::MAE)
        .num_samples(5)
        .search_space(SearchSpace::default())
        .build()?;
    
    let auto_nbeats = NBEATS::auto()
        .horizon(12)
        .loss_function(LossFunction::MAE)
        .num_samples(5)
        .search_space(SearchSpace::default())
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_models(vec![
            Box::new(auto_lstm),
            Box::new(auto_nbeats),
        ])
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df)?;
    
    Ok(())
}

/// Test Python prediction intervals example
#[test]
fn test_python_prediction_intervals() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate Python example:
    // ```python
    // from neuralforecast.models import LSTM
    // from neuralforecast.losses.pytorch import DistributionLoss
    // 
    // models = [
    //     LSTM(
    //         h=12,
    //         input_size=24,
    //         hidden_size=64,
    //         loss=DistributionLoss(distribution='Normal'),
    //         predict_likelihood=True,
    //         quantiles=[0.1, 0.5, 0.9]
    //     )
    // ]
    // ```
    
    let df = create_sample_dataframe()?;
    
    // Create prediction intervals
    let intervals = PredictionIntervals::new(
        vec![0.80, 0.90],  // 80% and 90% intervals
        IntervalMethod::Quantile,
    )?;
    
    let lstm = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .hidden_size(64)
        .loss_function(LossFunction::Quantile)
        .prediction_intervals(intervals)
        .num_samples(100)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df)?;
    
    // Generate predictions with intervals
    let forecasts = nf.predict_with_config(
        PredictionConfig::new()
            .with_intervals()
            .with_num_samples(100)
    )?;
    
    // Verify intervals are present
    assert!(forecasts.has_intervals());
    assert!(forecasts.has_column("LSTM-lo-80"));
    assert!(forecasts.has_column("LSTM-hi-80"));
    assert!(forecasts.has_column("LSTM-lo-90"));
    assert!(forecasts.has_column("LSTM-hi-90"));
    
    Ok(())
}

/// Test Python cross-validation example
#[test]
fn test_python_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate Python example:
    // ```python
    // from neuralforecast import NeuralForecast
    // from neuralforecast.models import LSTM, NBEATS
    // 
    // models = [LSTM(h=12), NBEATS(h=12)]
    // nf = NeuralForecast(models=models, freq='D')
    // 
    // cv_df = nf.cross_validation(
    //     df=df,
    //     n_windows=3,
    //     step_size=1,
    //     refit=True
    // )
    // ```
    
    let df = create_sample_dataframe()?;
    
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder().horizon(12).input_size(24).build()?),
        Box::new(NBEATS::builder().horizon(12).input_size(24).build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Cross-validation configuration
    let cv_config = CrossValidationConfig::new(3, 12)  // n_windows=3, h=12
        .with_step_size(1)
        .with_refit(true);
    
    let cv_results = nf.cross_validation(df, cv_config)?;
    
    // Validate CV output format
    assert_eq!(cv_results.n_windows(), 3);
    assert!(cv_results.has_column("unique_id"));
    assert!(cv_results.has_column("ds"));
    assert!(cv_results.has_column("cutoff"));
    assert!(cv_results.has_column("y"));
    assert!(cv_results.has_column("LSTM"));
    assert!(cv_results.has_column("NBEATS"));
    
    Ok(())
}

/// Test Python exogenous variables example
#[test]
fn test_python_exogenous_variables() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate Python example:
    // ```python
    // models = [
    //     LSTM(
    //         h=12,
    //         input_size=24,
    //         hidden_size=64,
    //         futr_exog_list=['temperature', 'holiday'],
    //         hist_exog_list=['temperature', 'holiday'],
    //         stat_exog_list=['location']
    //     )
    // ]
    // ```
    
    let df = create_dataframe_with_exogenous()?;
    
    let lstm = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .hidden_size(64)
        .futr_exog_features(vec!["temperature".to_string(), "holiday".to_string()])
        .hist_exog_features(vec!["temperature".to_string(), "holiday".to_string()])
        .static_features(vec!["location".to_string()])
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df)?;
    
    // For prediction with future exogenous, we need future values
    let future_df = create_future_exogenous()?;
    let forecasts = nf.predict_with_exogenous(future_df)?;
    
    Ok(())
}

/// Test all Python model configurations
#[test]
fn test_all_python_models() -> Result<(), Box<dyn std::error::Error>> {
    let df = create_sample_dataframe()?;
    
    // Test all models available in Python NeuralForecast
    let models: Vec<(&str, Box<dyn BaseModel<f32>>)> = vec![
        // Basic models
        ("MLP", Box::new(MLP::builder()
            .horizon(12)
            .input_size(24)
            .hidden_layers(vec![512, 512])
            .build()?)),
        
        ("LSTM", Box::new(LSTM::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(128)
            .num_layers(2)
            .build()?)),
        
        ("RNN", Box::new(RNN::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(128)
            .cell_type(RNNCellType::GRU)
            .build()?)),
        
        // Advanced models
        ("NBEATS", Box::new(NBEATS::builder()
            .horizon(12)
            .input_size(24)
            .interpretable()
            .build()?)),
        
        ("NHITS", Box::new(NHITS::builder()
            .horizon(12)
            .input_size(24)
            .n_blocks(vec![1, 1, 1])
            .mlp_units(vec![vec![512, 512]; 3])
            .n_pool_kernel_size(vec![16, 8, 1])
            .build()?)),
        
        ("DeepAR", Box::new(DeepAR::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(128)
            .num_layers(2)
            .build()?)),
        
        ("TFT", Box::new(TFT::builder()
            .horizon(12)
            .input_size(24)
            .hidden_size(128)
            .num_attention_heads(4)
            .build()?)),
        
        ("Transformer", Box::new(Transformer::builder()
            .horizon(12)
            .input_size(24)
            .d_model(128)
            .num_heads(4)
            .num_layers(2)
            .build()?)),
    ];
    
    // Test each model individually
    for (model_name, model) in models {
        println!("Testing {} model", model_name);
        
        let mut nf = NeuralForecast::builder()
            .with_model(model)
            .with_frequency(Frequency::Daily)
            .build()?;
        
        nf.fit(df.clone())?;
        let forecasts = nf.predict()?;
        
        assert!(forecasts.has_column(model_name));
        assert_eq!(forecasts.horizon(), 12);
    }
    
    Ok(())
}

/// Test Python training configuration options
#[test]
fn test_python_training_configs() -> Result<(), Box<dyn std::error::Error>> {
    // Test various training configurations from Python API
    let df = create_sample_dataframe()?;
    
    // Test with different optimizers
    let optimizers = vec![
        OptimizerType::Adam,
        OptimizerType::AdamW,
        OptimizerType::SGD,
        OptimizerType::RMSprop,
    ];
    
    for optimizer in optimizers {
        let lstm = LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .optimizer(optimizer)
            .learning_rate(0.001)
            .max_steps(10) // Small for testing
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(lstm))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        nf.fit(df.clone())?;
    }
    
    // Test with learning rate schedulers
    let lstm_with_scheduler = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .scheduler(SchedulerConfig {
            scheduler_type: SchedulerType::StepLR,
            step_size: Some(10),
            gamma: Some(0.1),
            milestones: None,
            patience: None,
            factor: None,
        })
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm_with_scheduler))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df.clone())?;
    
    // Test with early stopping
    let lstm_with_early_stop = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .early_stopping(EarlyStoppingConfig::new(
            "val_loss".to_string(),
            5,  // patience
            0.001,  // min_delta
            EarlyStoppingMode::Min,
        ))
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm_with_early_stop))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df)?;
    
    Ok(())
}

/// Test Python save/load functionality
#[test]
fn test_python_save_load() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate Python example:
    // ```python
    // nf = NeuralForecast(models=[LSTM(h=12)], freq='D')
    // nf.fit(df)
    // nf.save('path/to/model')
    // 
    // nf_loaded = NeuralForecast.load('path/to/model')
    // forecasts = nf_loaded.predict()
    // ```
    
    let df = create_sample_dataframe()?;
    let temp_dir = tempfile::tempdir()?;
    let model_path = temp_dir.path().join("neuralforecast_model");
    
    // Create and train model
    let lstm = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df)?;
    
    // Save model
    nf.save(&model_path)?;
    
    // Load model
    let nf_loaded = NeuralForecast::<f32>::load(&model_path)?;
    
    // Generate predictions from loaded model
    let forecasts = nf_loaded.predict()?;
    
    assert_eq!(forecasts.horizon(), 12);
    assert!(forecasts.has_column("LSTM"));
    
    Ok(())
}

/// Test Python multiple time series functionality
#[test]
fn test_python_multiple_series() -> Result<(), Box<dyn std::error::Error>> {
    // Create data with multiple series like Python example
    let df = create_multiple_series_dataframe(10)?; // 10 different series
    
    let lstm = LSTM::builder()
        .horizon(12)
        .input_size(24)
        .hidden_size(64)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Fit on all series
    nf.fit(df)?;
    
    // Predict for all series
    let forecasts = nf.predict()?;
    
    // Verify forecasts contain all series
    let unique_ids = forecasts.unique_values("unique_id")?;
    assert_eq!(unique_ids.len(), 10);
    
    Ok(())
}

/// Test Python loss functions
#[test]
fn test_python_loss_functions() -> Result<(), Box<dyn std::error::Error>> {
    let df = create_sample_dataframe()?;
    
    // Test all loss functions available in Python
    let loss_functions = vec![
        ("MAE", LossFunction::MAE),
        ("MSE", LossFunction::MSE),
        ("MAPE", LossFunction::MAPE),
        ("SMAPE", LossFunction::SMAPE),
        ("Huber", LossFunction::Huber),
        ("Quantile", LossFunction::Quantile),
    ];
    
    for (loss_name, loss_fn) in loss_functions {
        println!("Testing {} loss function", loss_name);
        
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
        
        nf.fit(df.clone())?;
        let _ = nf.predict()?;
    }
    
    Ok(())
}

/// Test Python batch prediction
#[test]
fn test_python_batch_prediction() -> Result<(), Box<dyn std::error::Error>> {
    // Test batch prediction functionality
    let train_df = create_sample_dataframe()?;
    let test_df = create_test_dataframe()?;
    
    let nbeats = NBEATS::builder()
        .horizon(14)
        .input_size(28)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(nbeats))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Fit on training data
    nf.fit(train_df)?;
    
    // Batch predict on test data
    let forecasts = nf.predict_on(test_df)?;
    
    assert_eq!(forecasts.horizon(), 14);
    
    Ok(())
}

/// Test output format compatibility
#[test]
fn test_output_format_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    let df = create_sample_dataframe()?;
    
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(df)?;
    let forecasts = nf.predict()?;
    
    // Verify output format matches Python
    assert!(forecasts.has_column("unique_id"));
    assert!(forecasts.has_column("ds"));
    assert!(forecasts.has_column("LSTM"));
    
    // Export to pandas-compatible format
    let pandas_df = forecasts.to_pandas_format()?;
    
    // Verify JSON serialization matches Python format
    let json_output = forecasts.to_json()?;
    let parsed: serde_json::Value = serde_json::from_str(&json_output)?;
    
    assert!(parsed.is_object());
    assert!(parsed["columns"].is_array());
    assert!(parsed["data"].is_array());
    
    Ok(())
}

/// Helper functions to create test data

fn create_sample_dataframe() -> Result<TimeSeriesDataFrame<f32>, Box<dyn std::error::Error>> {
    let start_date = Utc::now() - Duration::days(200);
    let mut dates = Vec::new();
    let mut values = Vec::new();
    let mut ids = Vec::new();
    
    // Single series for simplicity
    for i in 0..200 {
        let date = start_date + Duration::days(i);
        dates.push(date.timestamp());
        values.push(100.0 + 10.0 * (i as f32 * 0.1).sin() + rand::random::<f32>() * 5.0);
        ids.push("series_1".to_string());
    }
    
    let df = df! {
        "unique_id" => ids,
        "ds" => dates,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

fn create_dataframe_with_exogenous() -> Result<TimeSeriesDataFrame<f32>, Box<dyn std::error::Error>> {
    let base_df = create_sample_dataframe()?;
    let mut df = base_df.to_polars()?;
    
    let n_rows = df.height();
    let temperatures: Vec<f32> = (0..n_rows)
        .map(|i| 20.0 + 10.0 * (i as f32 * 0.05).sin())
        .collect();
    
    let holidays: Vec<i32> = (0..n_rows)
        .map(|i| if i % 7 == 0 { 1 } else { 0 })
        .collect();
    
    let locations: Vec<&str> = vec!["NYC"; n_rows];
    
    df = df.with_column(Series::new("temperature", temperatures))?;
    df = df.with_column(Series::new("holiday", holidays))?;
    df = df.with_column(Series::new("location", locations))?;
    
    let mut schema = base_df.schema.clone();
    schema.exog_cols = vec!["temperature".to_string(), "holiday".to_string()];
    schema.static_cols = vec!["location".to_string()];
    
    Ok(TimeSeriesDataFrame::new(df, schema, base_df.frequency))
}

fn create_future_exogenous() -> Result<TimeSeriesDataFrame<f32>, Box<dyn std::error::Error>> {
    // Create future exogenous variables for prediction
    let last_date = Utc::now();
    let mut dates = Vec::new();
    let mut ids = Vec::new();
    let mut temperatures = Vec::new();
    let mut holidays = Vec::new();
    
    for i in 0..12 {
        let date = last_date + Duration::days(i);
        dates.push(date.timestamp());
        ids.push("series_1".to_string());
        temperatures.push(20.0 + 10.0 * (i as f32 * 0.05).sin());
        holidays.push(if i % 7 == 0 { 1 } else { 0 });
    }
    
    let df = df! {
        "unique_id" => ids,
        "ds" => dates,
        "temperature" => temperatures,
        "holiday" => holidays,
    }?;
    
    let schema = TimeSeriesSchema {
        unique_id_col: "unique_id".to_string(),
        ds_col: "ds".to_string(),
        y_col: "".to_string(), // No y values for future
        exog_cols: vec!["temperature".to_string(), "holiday".to_string()],
        static_cols: vec![],
    };
    
    Ok(TimeSeriesDataFrame::new(df, schema, Some(Frequency::Daily)))
}

fn create_multiple_series_dataframe(n_series: usize) -> Result<TimeSeriesDataFrame<f32>, Box<dyn std::error::Error>> {
    let start_date = Utc::now() - Duration::days(300);
    let mut dates = Vec::new();
    let mut values = Vec::new();
    let mut ids = Vec::new();
    
    for series_idx in 0..n_series {
        let base_value = 50.0 + series_idx as f32 * 20.0;
        
        for i in 0..300 {
            let date = start_date + Duration::days(i);
            dates.push(date.timestamp());
            values.push(base_value + 10.0 * (i as f32 * 0.1).sin() + rand::random::<f32>() * 5.0);
            ids.push(format!("series_{}", series_idx));
        }
    }
    
    let df = df! {
        "unique_id" => ids,
        "ds" => dates,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

fn create_test_dataframe() -> Result<TimeSeriesDataFrame<f32>, Box<dyn std::error::Error>> {
    // Create test data for batch prediction
    let start_date = Utc::now() - Duration::days(50);
    let mut dates = Vec::new();
    let mut values = Vec::new();
    let mut ids = Vec::new();
    
    for i in 0..50 {
        let date = start_date + Duration::days(i);
        dates.push(date.timestamp());
        values.push(100.0 + 10.0 * (i as f32 * 0.1).sin() + rand::random::<f32>() * 5.0);
        ids.push("series_1".to_string());
    }
    
    let df = df! {
        "unique_id" => ids,
        "ds" => dates,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

// Placeholder structs for API compatibility
#[derive(Default)]
struct SearchSpace;

#[derive(Clone)]
enum RNNCellType {
    LSTM,
    GRU,
    RNN,
}

#[derive(Clone)]
enum EarlyStoppingMode {
    Min,
    Max,
}

struct PredictionConfig;
impl PredictionConfig {
    fn new() -> Self { Self }
    fn with_intervals(self) -> Self { self }
    fn with_num_samples(self, _: usize) -> Self { self }
}

enum IntervalMethod {
    Quantile,
    ConformalPrediction,
    Bootstrap,
}