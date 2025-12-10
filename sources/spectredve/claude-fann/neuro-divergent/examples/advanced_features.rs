//! Advanced features example
//!
//! This example demonstrates:
//! - Exogenous variables
//! - Multiple time series
//! - Custom scalers
//! - Probabilistic forecasting
//! - Hyperparameter tuning
//! - Advanced model configurations

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType, PredictionIntervals,
    models::{LSTM, NBEATS, DeepAR, TFT, Transformer},
    data::{
        TimeSeriesDataFrame, TimeSeriesSchema, DataPreprocessor,
        StandardScaler, RobustScaler, FeatureEngineer,
    },
    config::{
        LossFunction, OptimizerType, Device, IntervalMethod,
        SchedulerConfig, SchedulerType, EarlyStoppingConfig,
    },
    training::{HyperparameterTuner, SearchSpace, AccuracyMetrics},
};
use polars::prelude::*;
use chrono::{Utc, Duration, Datelike};
use std::collections::HashMap;
use anyhow::{Result, Context};

fn main() -> Result<()> {
    println!("🚀 Neuro-Divergent Advanced Features Example");
    println!("=" .repeat(60));
    
    // 1. Advanced Data Generation
    println!("\n1️⃣ Generating advanced time series data...");
    let data = generate_advanced_dataset()?;
    println!("   ✅ Generated {} series with exogenous variables", data.n_series()?);
    print_data_summary(&data)?;
    
    // 2. Feature Engineering
    println!("\n2️⃣ Performing feature engineering...");
    let engineered_data = perform_feature_engineering(data)?;
    println!("   ✅ Added {} engineered features", 
             engineered_data.schema.all_columns().len() - data.schema.all_columns().len());
    
    // 3. Custom Preprocessing Pipeline
    println!("\n3️⃣ Applying custom preprocessing pipeline...");
    let preprocessed_data = apply_custom_preprocessing(engineered_data)?;
    println!("   ✅ Data preprocessed with custom scalers");
    
    // Split data
    let (train_data, test_data) = preprocessed_data.train_test_split_by_date(
        Utc::now() - Duration::days(30)
    )?;
    
    // 4. Hyperparameter Tuning
    println!("\n4️⃣ Performing hyperparameter tuning...");
    let best_models = tune_hyperparameters(&train_data)?;
    println!("   ✅ Found optimal hyperparameters for {} models", best_models.len());
    
    // 5. Advanced Model Configurations
    println!("\n5️⃣ Creating advanced model configurations...");
    let advanced_models = create_advanced_models()?;
    
    // 6. Multi-Series Training
    println!("\n6️⃣ Training on multiple time series...");
    let mut multi_series_nf = train_multi_series_models(advanced_models, &train_data)?;
    
    // 7. Probabilistic Forecasting
    println!("\n7️⃣ Generating probabilistic forecasts...");
    let probabilistic_forecasts = generate_probabilistic_forecasts(&multi_series_nf, &test_data)?;
    analyze_prediction_intervals(&probabilistic_forecasts)?;
    
    // 8. Custom Loss Functions
    println!("\n8️⃣ Testing custom loss functions...");
    test_custom_loss_functions(&train_data)?;
    
    // 9. Transfer Learning
    println!("\n9️⃣ Demonstrating transfer learning...");
    demonstrate_transfer_learning(&train_data)?;
    
    // 10. Real-time Forecasting
    println!("\n🔟 Setting up real-time forecasting...");
    setup_realtime_forecasting(&multi_series_nf)?;
    
    // 11. Model Interpretability
    println!("\n🔍 Analyzing model interpretability...");
    analyze_model_interpretability(&multi_series_nf)?;
    
    // 12. Advanced Evaluation
    println!("\n📊 Performing advanced evaluation...");
    perform_advanced_evaluation(&multi_series_nf, &test_data)?;
    
    println!("\n✨ Advanced features demonstration completed!");
    
    Ok(())
}

/// Generate advanced dataset with multiple features
fn generate_advanced_dataset() -> Result<TimeSeriesDataFrame<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Configuration
    let n_stores = 10;
    let n_products = 3;
    let n_days = 730; // 2 years
    let start_date = Utc::now() - Duration::days(n_days);
    
    let mut data_rows = Vec::new();
    
    for store_id in 0..n_stores {
        let store_type = if store_id < 3 { "flagship" } 
                        else if store_id < 7 { "standard" } 
                        else { "express" };
        
        let store_location = match store_id % 3 {
            0 => "urban",
            1 => "suburban",
            _ => "rural",
        };
        
        for product_id in 0..n_products {
            let product_category = match product_id {
                0 => "electronics",
                1 => "clothing",
                _ => "groceries",
            };
            
            let base_demand = match product_category {
                "electronics" => 50.0 + store_id as f64 * 10.0,
                "clothing" => 100.0 + store_id as f64 * 15.0,
                _ => 200.0 + store_id as f64 * 20.0,
            };
            
            for day in 0..n_days {
                let date = start_date + Duration::days(day);
                
                // Time features
                let day_of_week = date.weekday().num_days_from_monday();
                let day_of_month = date.day();
                let month = date.month();
                let is_weekend = day_of_week >= 5;
                
                // Generate sales with complex patterns
                let weekly_pattern = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * day_of_week as f64 / 7.0).sin();
                let monthly_pattern = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * day_of_month as f64 / 30.0).sin();
                let yearly_pattern = 1.0 + 0.4 * (2.0 * std::f64::consts::PI * day as f64 / 365.0).sin();
                
                // External factors
                let temperature = 20.0 + 15.0 * (2.0 * std::f64::consts::PI * day as f64 / 365.0).cos() 
                                + rng.gen_range(-5.0..5.0);
                let precipitation = rng.gen_range(0.0..50.0).max(0.0);
                let holiday = (day % 30 == 0) || (day % 365 == 0) || (day % 365 == 180);
                
                // Marketing
                let promotion = rng.gen_bool(0.15);
                let discount_pct = if promotion { rng.gen_range(0.1..0.3) } else { 0.0 };
                let advertising_spend = if promotion { rng.gen_range(100.0..500.0) } else { 0.0 };
                
                // Competition
                let competitor_price = base_demand * rng.gen_range(0.9..1.1);
                let n_competitors = (store_id % 3 + 1) as f64;
                
                // Calculate sales
                let seasonal_factor = weekly_pattern * monthly_pattern * yearly_pattern;
                let weather_factor = 1.0 - precipitation / 100.0 + (temperature - 20.0) / 50.0;
                let promotion_factor = 1.0 + discount_pct * 2.0;
                let competition_factor = 1.0 - n_competitors * 0.1;
                let holiday_factor = if holiday { 1.5 } else { 1.0 };
                
                let sales = base_demand * seasonal_factor * weather_factor * 
                           promotion_factor * competition_factor * holiday_factor + 
                           rng.gen_range(-10.0..10.0);
                
                // Build row
                data_rows.push(DataRow {
                    unique_id: format!("store_{}_product_{}", store_id, product_id),
                    timestamp: date.timestamp(),
                    value: sales.max(0.0) as f32,
                    // Exogenous variables
                    temperature: temperature as f32,
                    precipitation: precipitation as f32,
                    holiday: holiday as i32,
                    promotion: promotion as i32,
                    discount_pct: discount_pct as f32,
                    advertising_spend: advertising_spend as f32,
                    competitor_price: competitor_price as f32,
                    n_competitors: n_competitors as f32,
                    is_weekend: is_weekend as i32,
                    day_of_week: day_of_week as i32,
                    day_of_month: day_of_month as i32,
                    month: month as i32,
                    // Static features
                    store_type: store_type.to_string(),
                    store_location: store_location.to_string(),
                    product_category: product_category.to_string(),
                });
            }
        }
    }
    
    // Convert to DataFrame
    let df = create_dataframe_from_rows(data_rows)?;
    
    let schema = TimeSeriesSchema {
        unique_id_col: "unique_id".to_string(),
        ds_col: "ds".to_string(),
        y_col: "y".to_string(),
        exog_cols: vec![
            "temperature", "precipitation", "holiday", "promotion",
            "discount_pct", "advertising_spend", "competitor_price",
            "n_competitors", "is_weekend", "day_of_week", 
            "day_of_month", "month"
        ].into_iter().map(String::from).collect(),
        static_cols: vec![
            "store_type", "store_location", "product_category"
        ].into_iter().map(String::from).collect(),
    };
    
    Ok(TimeSeriesDataFrame::new(df, schema, Some(Frequency::Daily)))
}

/// Perform feature engineering
fn perform_feature_engineering(data: TimeSeriesDataFrame<f32>) -> Result<TimeSeriesDataFrame<f32>> {
    let engineer = FeatureEngineer::builder()
        // Lag features
        .add_lag_features(vec![1, 7, 14, 28])
        // Rolling window features
        .add_rolling_mean(vec![7, 14, 28])
        .add_rolling_std(vec![7, 14])
        .add_rolling_min_max(vec![7, 28])
        // Date features
        .add_date_features(vec![
            DateFeature::Year,
            DateFeature::Quarter,
            DateFeature::WeekOfYear,
            DateFeature::IsMonthStart,
            DateFeature::IsMonthEnd,
            DateFeature::IsQuarterStart,
            DateFeature::IsYearStart,
        ])
        // Interaction features
        .add_interaction("temperature", "is_weekend")
        .add_interaction("promotion", "holiday")
        .add_interaction("discount_pct", "advertising_spend")
        // Fourier features for seasonality
        .add_fourier_features(vec![
            FourierOrder::new(7, 3),   // Weekly seasonality
            FourierOrder::new(30, 5),  // Monthly seasonality
            FourierOrder::new(365, 10), // Yearly seasonality
        ])
        .build()?;
    
    engineer.transform(data)
}

/// Apply custom preprocessing
fn apply_custom_preprocessing(data: TimeSeriesDataFrame<f32>) -> Result<TimeSeriesDataFrame<f32>> {
    // Create custom preprocessing pipeline
    let pipeline = PreprocessingPipeline::builder()
        // Handle missing values
        .add_step(MissingValueHandler::new()
            .with_numeric_strategy(NumericStrategy::Interpolate)
            .with_categorical_strategy(CategoricalStrategy::Mode))
        // Remove outliers
        .add_step(OutlierRemover::new()
            .with_method(OutlierMethod::IsolationForest)
            .with_contamination(0.05))
        // Custom scaling for different feature groups
        .add_step(GroupScaler::new()
            .add_group("sales_features", vec!["y", "lag_1", "lag_7"], StandardScaler::new())
            .add_group("weather_features", vec!["temperature", "precipitation"], RobustScaler::new())
            .add_group("marketing_features", vec!["discount_pct", "advertising_spend"], MinMaxScaler::new(0.0, 1.0)))
        // Encode categorical variables
        .add_step(CategoricalEncoder::new()
            .with_one_hot(vec!["store_type", "product_category"])
            .with_target_encoding("store_location", "y"))
        .build()?;
    
    pipeline.fit_transform(data)
}

/// Tune hyperparameters
fn tune_hyperparameters(train_data: &TimeSeriesDataFrame<f32>) -> Result<Vec<OptimizedModel>> {
    let tuner = HyperparameterTuner::new()
        .with_search_method(SearchMethod::BayesianOptimization)
        .with_n_trials(20)
        .with_cv_folds(3)
        .with_metric(OptimizationMetric::MAE);
    
    // Define search spaces
    let lstm_space = SearchSpace::new()
        .add_int("hidden_size", 32, 256)
        .add_int("num_layers", 1, 4)
        .add_float("dropout", 0.0, 0.5)
        .add_float("learning_rate", 1e-4, 1e-2, LogScale)
        .add_categorical("optimizer", vec!["adam", "adamw", "sgd"]);
    
    let nbeats_space = SearchSpace::new()
        .add_int("n_blocks", 2, 8)
        .add_int("mlp_units", 128, 1024)
        .add_bool("shared_weights")
        .add_float("learning_rate", 1e-4, 1e-2, LogScale);
    
    // Tune models
    let mut optimized_models = Vec::new();
    
    // Tune LSTM
    println!("   Tuning LSTM...");
    let best_lstm = tuner.optimize(
        || LSTM::builder(),
        lstm_space,
        train_data,
    )?;
    optimized_models.push(best_lstm);
    
    // Tune NBEATS
    println!("   Tuning NBEATS...");
    let best_nbeats = tuner.optimize(
        || NBEATS::builder(),
        nbeats_space,
        train_data,
    )?;
    optimized_models.push(best_nbeats);
    
    Ok(optimized_models)
}

/// Create advanced model configurations
fn create_advanced_models() -> Result<Vec<Box<dyn BaseModel<f32>>>> {
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        // Advanced LSTM with attention
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(60)
            .hidden_size(256)
            .num_layers(3)
            .bidirectional(true)
            .attention(AttentionType::Bahdanau)
            .dropout(0.2)
            .recurrent_dropout(0.2)
            .learning_rate(0.001)
            .gradient_clip_val(1.0)
            .scheduler(SchedulerConfig {
                scheduler_type: SchedulerType::CosineAnnealingLR,
                step_size: None,
                gamma: None,
                milestones: None,
                patience: None,
                factor: None,
            })
            .early_stopping(EarlyStoppingConfig::new(
                "val_loss".to_string(),
                10,
                0.0001,
                EarlyStoppingMode::Min,
            ))
            .hist_exog_features(vec![
                "temperature", "precipitation", "promotion",
                "discount_pct", "advertising_spend"
            ].into_iter().map(String::from).collect())
            .static_features(vec![
                "store_type", "store_location", "product_category"
            ].into_iter().map(String::from).collect())
            .build()?),
        
        // TFT for interpretable forecasting
        Box::new(TFT::builder()
            .horizon(14)
            .input_size(60)
            .hidden_size(128)
            .num_attention_heads(8)
            .num_lstm_layers(2)
            .dropout(0.1)
            .static_categoricals(vec!["store_type", "product_category"])
            .static_reals(vec![])
            .time_varying_categoricals_encoder(vec!["holiday", "promotion"])
            .time_varying_categoricals_decoder(vec!["holiday", "promotion"])
            .time_varying_reals_encoder(vec![
                "temperature", "precipitation", "competitor_price"
            ])
            .time_varying_reals_decoder(vec!["temperature", "precipitation"])
            .variable_selection_dropout(0.1)
            .build()?),
        
        // DeepAR with custom distribution
        Box::new(DeepAR::builder()
            .horizon(14)
            .input_size(60)
            .hidden_size(128)
            .num_layers(3)
            .distribution(Distribution::NegativeBinomial)
            .num_samples(200)
            .learning_rate(0.001)
            .hist_exog_features(vec![
                "temperature", "promotion", "advertising_spend"
            ].into_iter().map(String::from).collect())
            .build()?),
        
        // Advanced Transformer
        Box::new(Transformer::builder()
            .horizon(14)
            .input_size(60)
            .d_model(256)
            .num_heads(8)
            .num_encoder_layers(4)
            .num_decoder_layers(4)
            .feed_forward_dim(1024)
            .dropout(0.1)
            .activation("gelu")
            .positional_encoding(PositionalEncodingType::Learnable)
            .build()?),
    ];
    
    Ok(models)
}

/// Train multi-series models
fn train_multi_series_models(
    models: Vec<Box<dyn BaseModel<f32>>>,
    train_data: &TimeSeriesDataFrame<f32>
) -> Result<NeuralForecast<f32>> {
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .with_num_threads(8)
        .with_device(Device::GPU(0)) // Use GPU if available
        .with_batch_size(64)
        .with_prediction_intervals(PredictionIntervals::new(
            vec![0.10, 0.50, 0.90],
            IntervalMethod::ConformalPrediction,
        )?)
        .build()?;
    
    // Configure training callbacks
    let callbacks = vec![
        Box::new(ProgressBar::new()),
        Box::new(TensorBoard::new("./logs")),
        Box::new(ModelCheckpoint::new("./checkpoints")
            .monitor("val_loss")
            .save_best_only(true)),
        Box::new(LearningRateMonitor::new()),
        Box::new(EarlyStopping::new()
            .monitor("val_loss")
            .patience(15)
            .restore_best_weights(true)),
    ];
    
    // Train with callbacks
    nf.fit_with_callbacks(train_data.clone(), callbacks)?;
    
    Ok(nf)
}

/// Generate probabilistic forecasts
fn generate_probabilistic_forecasts(
    nf: &NeuralForecast<f32>,
    test_data: &TimeSeriesDataFrame<f32>
) -> Result<ProbabilisticForecast> {
    let config = PredictionConfig::new()
        .with_intervals()
        .with_num_samples(1000)
        .with_return_samples(true);
    
    let forecasts = nf.predict_probabilistic(test_data, config)?;
    
    Ok(forecasts)
}

/// Analyze prediction intervals
fn analyze_prediction_intervals(forecasts: &ProbabilisticForecast) -> Result<()> {
    println!("   📊 Prediction Interval Analysis:");
    
    // Calculate coverage
    let coverage_10 = forecasts.calculate_coverage(0.10)?;
    let coverage_50 = forecasts.calculate_coverage(0.50)?;
    let coverage_90 = forecasts.calculate_coverage(0.90)?;
    
    println!("      - 10% interval coverage: {:.1}%", coverage_10 * 100.0);
    println!("      - 50% interval coverage: {:.1}%", coverage_50 * 100.0);
    println!("      - 90% interval coverage: {:.1}%", coverage_90 * 100.0);
    
    // Calculate interval widths
    let width_stats = forecasts.calculate_interval_width_statistics()?;
    println!("      - Mean interval width: {:.2}", width_stats.mean);
    println!("      - Std interval width: {:.2}", width_stats.std);
    
    // Calculate sharpness (narrowness of intervals)
    let sharpness = forecasts.calculate_sharpness()?;
    println!("      - Interval sharpness: {:.3}", sharpness);
    
    Ok(())
}

/// Test custom loss functions
fn test_custom_loss_functions(train_data: &TimeSeriesDataFrame<f32>) -> Result<()> {
    // Define custom loss functions
    let loss_functions = vec![
        ("Quantile Loss (0.1)", LossFunction::Quantile(0.1)),
        ("Quantile Loss (0.5)", LossFunction::Quantile(0.5)),
        ("Quantile Loss (0.9)", LossFunction::Quantile(0.9)),
        ("SMAPE", LossFunction::SMAPE),
        ("Tweedie", LossFunction::Tweedie(1.5)),
        ("Focal Loss", LossFunction::Focal(2.0)),
    ];
    
    for (loss_name, loss_fn) in loss_functions {
        println!("   Testing {}...", loss_name);
        
        let model = MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![64, 32])
            .loss_function(loss_fn)
            .max_steps(50)
            .build()?;
        
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(model))
            .with_frequency(Frequency::Daily)
            .build()?;
        
        let subset = train_data.sample(0.1)?; // Use 10% for quick testing
        nf.fit(subset)?;
        
        println!("      ✅ Training completed");
    }
    
    Ok(())
}

/// Demonstrate transfer learning
fn demonstrate_transfer_learning(train_data: &TimeSeriesDataFrame<f32>) -> Result<()> {
    // Step 1: Train base model on all data
    println!("   Training base model on all series...");
    let base_model = LSTM::builder()
        .horizon(14)
        .input_size(28)
        .hidden_size(128)
        .num_layers(2)
        .build()?;
    
    let mut base_nf = NeuralForecast::builder()
        .with_model(Box::new(base_model))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    base_nf.fit(train_data.clone())?;
    
    // Step 2: Fine-tune on specific series
    println!("   Fine-tuning on specific series...");
    let target_series = train_data.filter_series("store_0_product_0")?;
    
    // Load pre-trained weights and fine-tune
    let fine_tuned_model = LSTM::from_pretrained(&base_nf, "LSTM")?
        .freeze_layers(vec![0, 1]) // Freeze first two layers
        .with_learning_rate(0.0001) // Lower learning rate
        .with_max_steps(20);
    
    let mut fine_tuned_nf = NeuralForecast::builder()
        .with_model(Box::new(fine_tuned_model))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    fine_tuned_nf.fit(target_series)?;
    
    println!("   ✅ Transfer learning completed");
    
    Ok(())
}

/// Setup real-time forecasting
fn setup_realtime_forecasting(nf: &NeuralForecast<f32>) -> Result<()> {
    // Create real-time forecasting service
    let realtime_service = RealtimeForecaster::new(nf.clone())
        .with_update_frequency(Duration::minutes(5))
        .with_sliding_window(100)
        .with_incremental_learning(true)
        .with_anomaly_detection(true);
    
    // Simulate real-time updates
    println!("   Simulating real-time updates...");
    
    for i in 0..5 {
        // Generate new observation
        let new_obs = generate_realtime_observation(i)?;
        
        // Update model
        realtime_service.update(new_obs)?;
        
        // Generate updated forecast
        let forecast = realtime_service.forecast()?;
        
        println!("      Update {}: Forecast next value = {:.2}", 
                 i + 1, forecast.next_value());
    }
    
    Ok(())
}

/// Analyze model interpretability
fn analyze_model_interpretability(nf: &NeuralForecast<f32>) -> Result<()> {
    // For NBEATS interpretable model
    if let Some(nbeats) = nf.get_model("NBEATS_interpretable") {
        let components = nbeats.get_components()?;
        
        println!("   NBEATS Components:");
        println!("      - Trend: {:?}", components.trend);
        println!("      - Seasonality: {:?}", components.seasonality);
    }
    
    // For TFT model
    if let Some(tft) = nf.get_model("TFT") {
        let attention_weights = tft.get_attention_weights()?;
        let variable_importance = tft.get_variable_importance()?;
        
        println!("   TFT Interpretability:");
        println!("      - Top 5 important variables:");
        for (var, importance) in variable_importance.top_k(5) {
            println!("        • {}: {:.3}", var, importance);
        }
    }
    
    Ok(())
}

/// Perform advanced evaluation
fn perform_advanced_evaluation(
    nf: &NeuralForecast<f32>,
    test_data: &TimeSeriesDataFrame<f32>
) -> Result<()> {
    let forecasts = nf.predict_on(test_data.clone())?;
    
    // Standard metrics
    let metrics = AccuracyMetrics::calculate(test_data, &forecasts)?;
    println!("   Standard Metrics:");
    println!("      - MAE: {:.2}", metrics.mae());
    println!("      - RMSE: {:.2}", metrics.rmse());
    println!("      - MAPE: {:.2}%", metrics.mape() * 100.0);
    
    // Advanced metrics
    let advanced_metrics = AdvancedMetrics::calculate(test_data, &forecasts)?;
    println!("\n   Advanced Metrics:");
    println!("      - MASE: {:.3}", advanced_metrics.mase());
    println!("      - sMAPE: {:.2}%", advanced_metrics.smape() * 100.0);
    println!("      - CRPS: {:.3}", advanced_metrics.crps());
    println!("      - Quantile Loss: {:.3}", advanced_metrics.quantile_loss());
    
    // Per-series metrics
    println!("\n   Per-Series Performance:");
    let per_series = metrics.calculate_per_series()?;
    for (series_id, series_metrics) in per_series.iter().take(5) {
        println!("      {}: MAE={:.2}, RMSE={:.2}", 
                 series_id, series_metrics.mae(), series_metrics.rmse());
    }
    
    // Residual analysis
    let residuals = calculate_residuals(test_data, &forecasts)?;
    let residual_stats = analyze_residuals(&residuals)?;
    println!("\n   Residual Analysis:");
    println!("      - Mean: {:.3}", residual_stats.mean);
    println!("      - Std: {:.3}", residual_stats.std);
    println!("      - Skewness: {:.3}", residual_stats.skewness);
    println!("      - Kurtosis: {:.3}", residual_stats.kurtosis);
    println!("      - Autocorrelation (lag 1): {:.3}", residual_stats.acf_lag1);
    
    Ok(())
}

// Helper functions and types

struct DataRow {
    unique_id: String,
    timestamp: i64,
    value: f32,
    // Exogenous
    temperature: f32,
    precipitation: f32,
    holiday: i32,
    promotion: i32,
    discount_pct: f32,
    advertising_spend: f32,
    competitor_price: f32,
    n_competitors: f32,
    is_weekend: i32,
    day_of_week: i32,
    day_of_month: i32,
    month: i32,
    // Static
    store_type: String,
    store_location: String,
    product_category: String,
}

fn create_dataframe_from_rows(rows: Vec<DataRow>) -> Result<DataFrame> {
    let n = rows.len();
    
    // Extract columns
    let unique_ids: Vec<_> = rows.iter().map(|r| r.unique_id.clone()).collect();
    let timestamps: Vec<_> = rows.iter().map(|r| r.timestamp).collect();
    let values: Vec<_> = rows.iter().map(|r| r.value).collect();
    
    // Build DataFrame
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
        "temperature" => rows.iter().map(|r| r.temperature).collect::<Vec<_>>(),
        "precipitation" => rows.iter().map(|r| r.precipitation).collect::<Vec<_>>(),
        "holiday" => rows.iter().map(|r| r.holiday).collect::<Vec<_>>(),
        "promotion" => rows.iter().map(|r| r.promotion).collect::<Vec<_>>(),
        "discount_pct" => rows.iter().map(|r| r.discount_pct).collect::<Vec<_>>(),
        "advertising_spend" => rows.iter().map(|r| r.advertising_spend).collect::<Vec<_>>(),
        "competitor_price" => rows.iter().map(|r| r.competitor_price).collect::<Vec<_>>(),
        "n_competitors" => rows.iter().map(|r| r.n_competitors).collect::<Vec<_>>(),
        "is_weekend" => rows.iter().map(|r| r.is_weekend).collect::<Vec<_>>(),
        "day_of_week" => rows.iter().map(|r| r.day_of_week).collect::<Vec<_>>(),
        "day_of_month" => rows.iter().map(|r| r.day_of_month).collect::<Vec<_>>(),
        "month" => rows.iter().map(|r| r.month).collect::<Vec<_>>(),
        "store_type" => rows.iter().map(|r| r.store_type.clone()).collect::<Vec<_>>(),
        "store_location" => rows.iter().map(|r| r.store_location.clone()).collect::<Vec<_>>(),
        "product_category" => rows.iter().map(|r| r.product_category.clone()).collect::<Vec<_>>(),
    }?;
    
    Ok(df)
}

fn print_data_summary(data: &TimeSeriesDataFrame<f32>) -> Result<()> {
    println!("   📊 Data Summary:");
    println!("      - Total rows: {}", data.shape().0);
    println!("      - Number of series: {}", data.n_series()?);
    println!("      - Time range: {} days", data.time_range_days()?);
    println!("      - Exogenous variables: {}", data.schema.exog_cols.len());
    println!("      - Static features: {}", data.schema.static_cols.len());
    
    Ok(())
}

fn generate_realtime_observation(index: usize) -> Result<TimeSeriesObservation> {
    // Simulate real-time data point
    Ok(TimeSeriesObservation {
        unique_id: "store_0_product_0".to_string(),
        timestamp: Utc::now(),
        value: 100.0 + index as f32 * 10.0,
        exog_values: HashMap::new(),
    })
}

fn calculate_residuals(
    actual: &TimeSeriesDataFrame<f32>,
    forecasts: &ForecastDataFrame<f32>
) -> Result<Vec<f32>> {
    // Placeholder
    Ok(vec![0.0; 100])
}

fn analyze_residuals(residuals: &[f32]) -> Result<ResidualStatistics> {
    // Placeholder
    Ok(ResidualStatistics {
        mean: 0.0,
        std: 1.0,
        skewness: 0.0,
        kurtosis: 3.0,
        acf_lag1: 0.1,
    })
}

// Placeholder types
struct OptimizedModel;
struct FeatureEngineer;
struct PreprocessingPipeline;
struct MissingValueHandler;
struct OutlierRemover;
struct GroupScaler;
struct CategoricalEncoder;
struct HyperparameterTuner;
struct SearchSpace;
struct ProbabilisticForecast;
struct RealtimeForecaster;
struct AdvancedMetrics;
struct TimeSeriesObservation;
struct ResidualStatistics;

// Placeholder enums
enum DateFeature { Year, Quarter, WeekOfYear, IsMonthStart, IsMonthEnd, IsQuarterStart, IsYearStart }
enum NumericStrategy { Interpolate }
enum CategoricalStrategy { Mode }
enum OutlierMethod { IsolationForest }
enum SearchMethod { BayesianOptimization }
enum OptimizationMetric { MAE }
enum AttentionType { Bahdanau }
enum Distribution { NegativeBinomial }
enum PositionalEncodingType { Learnable }

struct FourierOrder;
impl FourierOrder {
    fn new(_: i32, _: i32) -> Self { Self }
}

struct LogScale;