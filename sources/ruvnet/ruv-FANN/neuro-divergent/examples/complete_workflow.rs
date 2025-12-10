//! Complete end-to-end forecasting workflow example
//! 
//! This example demonstrates a complete workflow including:
//! - Data loading and preprocessing
//! - Model creation and configuration
//! - Training with validation
//! - Forecasting and evaluation
//! - Model persistence

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType, PredictionIntervals,
    models::{LSTM, NBEATS, DeepAR, MLP},
    data::{
        TimeSeriesDataFrame, TimeSeriesSchema, DataPreprocessor,
        StandardScaler, DataValidator, ValidationReport,
    },
    config::{
        LossFunction, OptimizerType, Device, CrossValidationConfig,
        EarlyStoppingConfig, EarlyStoppingMode, IntervalMethod,
    },
    training::{AccuracyMetrics, ForecastingMetrics},
};
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration, NaiveDateTime};
use std::path::Path;
use std::fs;
use anyhow::{Result, Context};

fn main() -> Result<()> {
    println!("🚀 Neuro-Divergent Complete Workflow Example");
    println!("=" .repeat(50));
    
    // Step 1: Load and prepare data
    println!("\n1️⃣ Loading and preparing data...");
    let data = load_time_series_data("data/sales_data.csv")
        .unwrap_or_else(|_| {
            println!("   ⚠️  Could not load CSV file, generating synthetic data instead");
            generate_synthetic_sales_data().expect("Failed to generate synthetic data")
        });
    
    println!("   ✅ Loaded {} series with {} total observations", 
             data.n_series()?, data.shape().0);
    
    // Step 2: Data validation and preprocessing
    println!("\n2️⃣ Validating and preprocessing data...");
    let validation_report = validate_data(&data)?;
    print_validation_report(&validation_report);
    
    let preprocessed_data = preprocess_data(data)?;
    println!("   ✅ Data preprocessing complete");
    
    // Step 3: Split data for evaluation
    println!("\n3️⃣ Splitting data for training and testing...");
    let (train_data, test_data) = split_data(preprocessed_data, 0.8)?;
    println!("   ✅ Train size: {} rows", train_data.shape().0);
    println!("   ✅ Test size: {} rows", test_data.shape().0);
    
    // Step 4: Create and configure models
    println!("\n4️⃣ Creating forecasting models...");
    let models = create_model_ensemble()?;
    println!("   ✅ Created {} models for ensemble", models.len());
    
    // Step 5: Train models
    println!("\n5️⃣ Training models...");
    let mut neural_forecast = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .with_num_threads(4)
        .with_prediction_intervals(PredictionIntervals::new(
            vec![0.80, 0.95],
            IntervalMethod::Quantile,
        )?)
        .build()?;
    
    let start_time = std::time::Instant::now();
    neural_forecast.fit(train_data.clone())?;
    let training_time = start_time.elapsed();
    
    println!("   ✅ Training completed in {:.2} seconds", training_time.as_secs_f64());
    
    // Step 6: Cross-validation
    println!("\n6️⃣ Running cross-validation...");
    let cv_results = run_cross_validation(&mut neural_forecast, train_data.clone())?;
    print_cv_results(&cv_results);
    
    // Step 7: Generate forecasts
    println!("\n7️⃣ Generating forecasts...");
    let forecasts = neural_forecast.predict_on(test_data.clone())?;
    println!("   ✅ Generated forecasts for {} periods", forecasts.horizon());
    
    // Step 8: Evaluate performance
    println!("\n8️⃣ Evaluating model performance...");
    let metrics = evaluate_forecasts(&test_data, &forecasts)?;
    print_evaluation_metrics(&metrics);
    
    // Step 9: Generate future predictions
    println!("\n9️⃣ Generating future predictions...");
    let future_forecasts = neural_forecast.predict()?;
    
    // Save forecasts to CSV
    save_forecasts(&future_forecasts, "output/forecasts.csv")?;
    println!("   ✅ Saved forecasts to output/forecasts.csv");
    
    // Step 10: Save trained models
    println!("\n🔟 Saving trained models...");
    save_models(&neural_forecast, "models/trained_ensemble")?;
    println!("   ✅ Models saved successfully");
    
    // Bonus: Visualize results (if plotting is available)
    #[cfg(feature = "plotting")]
    {
        println!("\n📊 Generating visualizations...");
        visualize_results(&test_data, &forecasts)?;
    }
    
    println!("\n✨ Workflow completed successfully!");
    
    Ok(())
}

/// Load time series data from CSV file
fn load_time_series_data(path: &str) -> Result<TimeSeriesDataFrame<f32>> {
    // Read CSV file
    let df = CsvReader::from_path(path)?
        .has_header(true)
        .finish()?;
    
    // Define schema
    let schema = TimeSeriesSchema {
        unique_id_col: "store_id".to_string(),
        ds_col: "date".to_string(),
        y_col: "sales".to_string(),
        exog_cols: vec!["promotion".to_string(), "holiday".to_string()],
        static_cols: vec!["store_type".to_string()],
    };
    
    Ok(TimeSeriesDataFrame::new(df, schema, Some(Frequency::Daily)))
}

/// Generate synthetic sales data for demonstration
fn generate_synthetic_sales_data() -> Result<TimeSeriesDataFrame<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let n_stores = 5;
    let n_days = 730; // 2 years
    let start_date = Utc::now() - Duration::days(n_days);
    
    let mut store_ids = Vec::new();
    let mut dates = Vec::new();
    let mut sales = Vec::new();
    let mut promotions = Vec::new();
    let mut holidays = Vec::new();
    let mut store_types = Vec::new();
    
    for store_id in 0..n_stores {
        let store_name = format!("store_{}", store_id);
        let store_type = if store_id < 2 { "urban" } else { "suburban" };
        let base_sales = 1000.0 + store_id as f64 * 500.0;
        
        for day in 0..n_days {
            let date = start_date + Duration::days(day);
            
            // Generate sales with trend, seasonality, and noise
            let trend = day as f64 * 0.5;
            let weekly_seasonality = 200.0 * (2.0 * std::f64::consts::PI * day as f64 / 7.0).sin();
            let yearly_seasonality = 500.0 * (2.0 * std::f64::consts::PI * day as f64 / 365.0).sin();
            let noise = rng.gen_range(-100.0..100.0);
            
            let is_holiday = day % 30 == 0;
            let has_promotion = rng.gen_bool(0.2);
            
            let holiday_effect = if is_holiday { 300.0 } else { 0.0 };
            let promotion_effect = if has_promotion { 200.0 } else { 0.0 };
            
            let total_sales = base_sales + trend + weekly_seasonality + yearly_seasonality + 
                            holiday_effect + promotion_effect + noise;
            
            store_ids.push(store_name.clone());
            dates.push(date.timestamp());
            sales.push(total_sales.max(0.0) as f32);
            promotions.push(has_promotion as i32);
            holidays.push(is_holiday as i32);
            store_types.push(store_type);
        }
    }
    
    let df = df! {
        "store_id" => store_ids,
        "date" => dates,
        "sales" => sales,
        "promotion" => promotions,
        "holiday" => holidays,
        "store_type" => store_types,
    }?;
    
    let schema = TimeSeriesSchema {
        unique_id_col: "store_id".to_string(),
        ds_col: "date".to_string(),
        y_col: "sales".to_string(),
        exog_cols: vec!["promotion".to_string(), "holiday".to_string()],
        static_cols: vec!["store_type".to_string()],
    };
    
    Ok(TimeSeriesDataFrame::new(df, schema, Some(Frequency::Daily)))
}

/// Validate time series data
fn validate_data(data: &TimeSeriesDataFrame<f32>) -> Result<ValidationReport> {
    let validator = DataValidator::new()
        .with_check_missing(true)
        .with_check_duplicates(true)
        .with_check_outliers(true)
        .with_check_frequency(true);
    
    validator.validate(data).context("Data validation failed")
}

/// Print validation report
fn print_validation_report(report: &ValidationReport) {
    println!("   📋 Validation Report:");
    println!("      - Valid: {}", if report.is_valid { "✅" } else { "❌" });
    
    if !report.issues.is_empty() {
        println!("      - Issues found:");
        for issue in &report.issues {
            println!("        • {}: {}", issue.issue_type, issue.description);
        }
    }
}

/// Preprocess data
fn preprocess_data(data: TimeSeriesDataFrame<f32>) -> Result<TimeSeriesDataFrame<f32>> {
    let preprocessor = DataPreprocessor::builder()
        .with_missing_value_strategy(MissingValueStrategy::Interpolate)
        .with_outlier_detection(OutlierMethod::IQR(1.5))
        .with_outlier_treatment(OutlierTreatment::Cap)
        .with_scaler(ScalerType::StandardScaler)
        .build()?;
    
    preprocessor.fit_transform(&data)
        .context("Preprocessing failed")
}

/// Split data into train and test sets
fn split_data(
    data: TimeSeriesDataFrame<f32>, 
    train_ratio: f64
) -> Result<(TimeSeriesDataFrame<f32>, TimeSeriesDataFrame<f32>)> {
    let total_rows = data.shape().0;
    let train_size = (total_rows as f64 * train_ratio) as usize;
    
    data.train_test_split(train_size)
        .context("Failed to split data")
}

/// Create ensemble of models
fn create_model_ensemble() -> Result<Vec<Box<dyn BaseModel<f32>>>> {
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        // LSTM for capturing sequential patterns
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(128)
            .num_layers(2)
            .dropout(0.1)
            .learning_rate(0.001)
            .loss_function(LossFunction::MAE)
            .early_stopping_patience(10)
            .hist_exog_features(vec!["promotion".to_string(), "holiday".to_string()])
            .static_features(vec!["store_type".to_string()])
            .build()?),
        
        // NBEATS for interpretable decomposition
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .interpretable()
            .with_trend()
            .with_seasonality()
            .seasonality_period(7)
            .learning_rate(0.001)
            .build()?),
        
        // DeepAR for probabilistic forecasting
        Box::new(DeepAR::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(64)
            .num_layers(2)
            .num_samples(100)
            .learning_rate(0.001)
            .hist_exog_features(vec!["promotion".to_string(), "holiday".to_string()])
            .build()?),
        
        // Simple MLP as baseline
        Box::new(MLP::builder()
            .horizon(14)
            .input_size(28)
            .hidden_layers(vec![128, 64, 32])
            .activation("relu")
            .learning_rate(0.001)
            .build()?),
    ];
    
    Ok(models)
}

/// Run cross-validation
fn run_cross_validation(
    nf: &mut NeuralForecast<f32>,
    data: TimeSeriesDataFrame<f32>
) -> Result<CrossValidationResults> {
    let cv_config = CrossValidationConfig::new(3, 14)
        .with_step_size(14)
        .with_refit(false); // Don't refit to save time
    
    let cv_results = nf.cross_validation(data, cv_config)?;
    
    Ok(cv_results)
}

/// Print cross-validation results
fn print_cv_results(results: &CrossValidationResults) {
    println!("   📊 Cross-validation Results:");
    
    let metrics = results.calculate_metrics()
        .expect("Failed to calculate CV metrics");
    
    for (model_name, model_metrics) in metrics {
        println!("      {}:", model_name);
        println!("        - MAE: {:.2}", model_metrics.mae());
        println!("        - RMSE: {:.2}", model_metrics.rmse());
        println!("        - MAPE: {:.2}%", model_metrics.mape() * 100.0);
    }
}

/// Evaluate forecast performance
fn evaluate_forecasts(
    actual: &TimeSeriesDataFrame<f32>,
    forecasts: &ForecastDataFrame<f32>
) -> Result<HashMap<String, AccuracyMetrics>> {
    let mut metrics_map = HashMap::new();
    
    for model_name in forecasts.model_names() {
        let metrics = AccuracyMetrics::calculate(actual, forecasts)?;
        metrics_map.insert(model_name, metrics);
    }
    
    Ok(metrics_map)
}

/// Print evaluation metrics
fn print_evaluation_metrics(metrics: &HashMap<String, AccuracyMetrics>) {
    println!("   📈 Model Performance Metrics:");
    
    for (model_name, model_metrics) in metrics {
        println!("      {}:", model_name);
        println!("        - MAE: {:.2}", model_metrics.mae());
        println!("        - MSE: {:.2}", model_metrics.mse());
        println!("        - RMSE: {:.2}", model_metrics.rmse());
        println!("        - MAPE: {:.2}%", model_metrics.mape() * 100.0);
        println!("        - SMAPE: {:.2}%", model_metrics.smape() * 100.0);
    }
}

/// Save forecasts to CSV
fn save_forecasts(forecasts: &ForecastDataFrame<f32>, path: &str) -> Result<()> {
    // Create output directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Convert to DataFrame and save
    let df = forecasts.to_polars()?;
    let mut file = std::fs::File::create(path)?;
    CsvWriter::new(&mut file)
        .has_header(true)
        .finish(&mut df.clone())?;
    
    Ok(())
}

/// Save trained models
fn save_models(nf: &NeuralForecast<f32>, path: &str) -> Result<()> {
    // Create model directory if it doesn't exist
    fs::create_dir_all(path)?;
    
    let model_file = Path::new(path).join("neural_forecast.ndf");
    nf.save(&model_file)?;
    
    // Also save metadata
    let metadata = serde_json::json!({
        "version": "1.0.0",
        "created_at": Utc::now().to_rfc3339(),
        "models": nf.model_names(),
        "frequency": format!("{:?}", nf.frequency()),
        "is_fitted": nf.is_fitted(),
    });
    
    let metadata_file = Path::new(path).join("metadata.json");
    fs::write(metadata_file, serde_json::to_string_pretty(&metadata)?)?;
    
    Ok(())
}

/// Visualize results (feature-gated)
#[cfg(feature = "plotting")]
fn visualize_results(
    actual: &TimeSeriesDataFrame<f32>,
    forecasts: &ForecastDataFrame<f32>
) -> Result<()> {
    use plotly::{Plot, Scatter};
    
    // Create plots for each series
    for series_id in actual.unique_series_ids()? {
        let mut plot = Plot::new();
        
        // Add actual values
        let actual_trace = Scatter::new(
            actual.get_timestamps(&series_id)?,
            actual.get_values(&series_id)?
        ).name("Actual");
        
        plot.add_trace(actual_trace);
        
        // Add forecasts for each model
        for model_name in forecasts.model_names() {
            let forecast_trace = Scatter::new(
                forecasts.get_timestamps(&series_id)?,
                forecasts.get_model_forecasts(&series_id, &model_name)?
            ).name(&model_name);
            
            plot.add_trace(forecast_trace);
        }
        
        // Save plot
        let plot_file = format!("output/plots/{}_forecast.html", series_id);
        plot.write_html(&plot_file);
    }
    
    println!("   ✅ Plots saved to output/plots/");
    Ok(())
}

// Type definitions for the example
use std::collections::HashMap;

#[derive(Debug)]
struct CrossValidationResults {
    // Implementation details
}

impl CrossValidationResults {
    fn calculate_metrics(&self) -> Result<HashMap<String, AccuracyMetrics>> {
        // Placeholder implementation
        Ok(HashMap::new())
    }
}

// Placeholder enums for the example
#[derive(Debug, Clone)]
enum MissingValueStrategy {
    Interpolate,
}

#[derive(Debug, Clone)]
enum OutlierMethod {
    IQR(f32),
}

#[derive(Debug, Clone)]
enum OutlierTreatment {
    Cap,
}