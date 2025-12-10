# Data Handling

Effective data handling is crucial for successful neural forecasting. This guide covers everything from loading and preprocessing time series data to feature engineering and validation strategies.

## Overview

Data handling in Neuro-Divergent follows a structured pipeline:

```
Raw Data → Loading → Validation → Preprocessing → Feature Engineering → Model Input
```

Each step ensures your data is properly formatted, cleaned, and optimized for neural forecasting models.

## Data Structures

### TimeSeriesDataFrame

The core data structure for time series in Neuro-Divergent:

```rust
use neuro_divergent::data::TimeSeriesDataFrame;
use chrono::{DateTime, Utc};

// Create from vectors
let timestamps: Vec<DateTime<Utc>> = generate_timestamps();
let values: Vec<f64> = vec![100.0, 105.0, 103.0, 108.0];

let data = TimeSeriesDataFrame::new()
    .with_time_column("timestamp", timestamps)
    .with_target_column("sales", values)
    .with_series_id("store_001")
    .with_frequency(Frequency::Daily)
    .build()?;
```

### Supported Data Types

| Type | Description | Example |
|------|-------------|---------|
| `DateTime<Utc>` | UTC timestamps | `2023-01-01T00:00:00Z` |
| `f32`, `f64` | Floating point values | `123.45` |
| `i32`, `i64` | Integer values | `1000` |
| `String` | Categorical features | `"electronics"` |
| `Vec<T>` | Multi-dimensional features | `[1.0, 2.0, 3.0]` |

## Loading Data

### From CSV Files

```rust
use neuro_divergent::data::loaders::CSVLoader;

// Basic CSV loading
let data = CSVLoader::new()
    .with_file_path("data/sales.csv")
    .with_time_column("date")
    .with_target_column("revenue")
    .load()?;

// Advanced CSV loading with multiple features
let advanced_data = CSVLoader::new()
    .with_file_path("data/multivariate.csv")
    .with_time_column("timestamp")
    .with_target_column("y")
    .with_series_id_column("series_id")
    .with_static_features(vec!["category", "region"])
    .with_dynamic_features(vec!["temperature", "promotion"])
    .with_date_format("%Y-%m-%d %H:%M:%S")
    .with_delimiter(',')
    .with_has_header(true)
    .load()?;
```

#### Expected CSV Format

```csv
timestamp,y,series_id,category,temperature,promotion
2023-01-01 00:00:00,100.5,series_A,electronics,20.1,0
2023-01-01 01:00:00,105.2,series_A,electronics,19.8,0
2023-01-01 02:00:00,103.7,series_A,electronics,19.5,1
2023-01-01 00:00:00,89.3,series_B,clothing,20.1,0
```

### From Parquet Files

```rust
use neuro_divergent::data::loaders::ParquetLoader;

let data = ParquetLoader::new()
    .with_file_path("data/timeseries.parquet")
    .with_time_column("timestamp")
    .with_target_column("value")
    .with_filters(vec![("region", "US")])  // Pre-filter data
    .load()?;
```

### From Databases

```rust
use neuro_divergent::data::loaders::DatabaseLoader;

let data = DatabaseLoader::new()
    .with_connection_string("postgresql://user:pass@localhost/db")
    .with_query(r#"
        SELECT timestamp, sales, store_id, category 
        FROM sales_data 
        WHERE timestamp >= '2023-01-01'
        ORDER BY store_id, timestamp
    "#)
    .with_time_column("timestamp")
    .with_target_column("sales")
    .with_series_id_column("store_id")
    .load()?;
```

### From APIs

```rust
use neuro_divergent::data::loaders::APILoader;

let data = APILoader::new()
    .with_endpoint("https://api.example.com/timeseries")
    .with_authentication(AuthMethod::Bearer("token".to_string()))
    .with_pagination(true)
    .with_rate_limit(10, Duration::from_secs(1))  // 10 requests per second
    .with_retry_policy(RetryPolicy::exponential_backoff(3))
    .load()?;
```

### From Memory/Streaming

```rust
// From vectors in memory
let data = TimeSeriesDataFrame::from_vectors(
    timestamps,
    values,
    Some("series_id".to_string())
)?;

// From streaming data
let stream_loader = StreamLoader::new()
    .with_buffer_size(1000)
    .with_flush_interval(Duration::from_secs(30));

// Process streaming data
for batch in stream_loader.batches() {
    let processed_batch = preprocess_batch(batch)?;
    model.partial_fit(&processed_batch)?;
}
```

## Data Validation

### Automatic Validation

```rust
use neuro_divergent::data::validation::{DataValidator, ValidationConfig};

let validator = DataValidator::new()
    .with_config(ValidationConfig {
        check_missing_values: true,
        check_duplicates: true,
        check_frequency: true,
        check_outliers: true,
        check_stationarity: true,
        min_observations: 100,
        max_gap_size: 7,
    });

let validation_report = validator.validate(&data)?;

if !validation_report.is_valid() {
    println!("Data validation issues found:");
    for issue in validation_report.issues() {
        println!("- {}: {}", issue.severity(), issue.description());
    }
}
```

### Custom Validation Rules

```rust
use neuro_divergent::data::validation::ValidationRule;

// Custom validation rule
struct MinimumLengthRule {
    min_length: usize,
}

impl ValidationRule for MinimumLengthRule {
    fn validate(&self, data: &TimeSeriesDataFrame) -> ValidationResult {
        if data.len() < self.min_length {
            ValidationResult::error(format!(
                "Series too short: {} < {}", 
                data.len(), 
                self.min_length
            ))
        } else {
            ValidationResult::ok()
        }
    }
}

let validator = DataValidator::new()
    .add_rule(Box::new(MinimumLengthRule { min_length: 200 }))
    .add_rule(Box::new(FrequencyConsistencyRule::new()))
    .add_rule(Box::new(OutlierDetectionRule::new(3.0)));

let report = validator.validate(&data)?;
```

### Validation Report

```rust
// Detailed validation report
println!("Validation Summary:");
println!("==================");
println!("Series count: {}", report.series_count());
println!("Total observations: {}", report.total_observations());
println!("Date range: {} to {}", report.start_date(), report.end_date());
println!("Frequency: {:?}", report.detected_frequency());
println!("Missing values: {}", report.missing_count());
println!("Duplicate timestamps: {}", report.duplicate_count());
println!("Outliers detected: {}", report.outlier_count());

// Export detailed report
report.export_to_file("validation_report.json")?;
```

## Data Preprocessing

### Missing Value Handling

```rust
use neuro_divergent::data::preprocessing::MissingValueHandler;

// Linear interpolation for small gaps
let linear_imputer = MissingValueHandler::new()
    .with_method(ImputationMethod::Linear)
    .with_max_gap_size(3);  // Fill gaps up to 3 observations

let clean_data = linear_imputer.transform(&data)?;

// Forward fill for irregular patterns
let ffill_imputer = MissingValueHandler::new()
    .with_method(ImputationMethod::ForwardFill)
    .with_limit(5);  // Maximum 5 consecutive fills

// Seasonal interpolation for periodic data
let seasonal_imputer = MissingValueHandler::new()
    .with_method(ImputationMethod::Seasonal)
    .with_period(24);  // Daily pattern for hourly data

// Advanced model-based imputation
let model_imputer = MissingValueHandler::new()
    .with_method(ImputationMethod::ModelBased)
    .with_model(Box::new(LSTM::new(lstm_config)?));
```

### Outlier Detection and Treatment

```rust
use neuro_divergent::data::preprocessing::OutlierDetector;

// Statistical outlier detection
let outlier_detector = OutlierDetector::new()
    .with_method(OutlierMethod::IQR)
    .with_threshold(3.0)
    .with_seasonal_adjustment(true);

let outliers = outlier_detector.detect(&data)?;
println!("Found {} outliers", outliers.len());

// Treatment options
let treated_data = outlier_detector
    .remove_outliers(&data, &outliers)?;           // Remove outliers

let capped_data = outlier_detector
    .cap_outliers(&data, &outliers)?;              // Cap to threshold

let imputed_data = outlier_detector
    .interpolate_outliers(&data, &outliers)?;      // Interpolate outliers

// Model-based outlier detection
let isolation_forest = OutlierDetector::new()
    .with_method(OutlierMethod::IsolationForest)
    .with_contamination(0.1);  // Expect 10% outliers
```

### Data Transformation

```rust
use neuro_divergent::data::preprocessing::{Scaler, Transformer};

// Scaling
let standard_scaler = StandardScaler::new();
let scaled_data = standard_scaler.fit_transform(&data)?;

let minmax_scaler = MinMaxScaler::new()
    .with_range(0.0, 1.0);
let normalized_data = minmax_scaler.fit_transform(&data)?;

let robust_scaler = RobustScaler::new();  // Robust to outliers
let robust_scaled_data = robust_scaler.fit_transform(&data)?;

// Log transformation for skewed data
let log_transformer = LogTransformer::new()
    .with_offset(1.0);  // log(x + 1) to handle zeros
let log_data = log_transformer.transform(&data)?;

// Box-Cox transformation for normality
let boxcox_transformer = BoxCoxTransformer::new()
    .with_auto_lambda(true);  // Automatically find best lambda
let transformed_data = boxcox_transformer.fit_transform(&data)?;

// Differencing for stationarity
let diff_transformer = DifferencingTransformer::new()
    .with_order(1)           // First difference
    .with_seasonal_order(1)  // Seasonal difference
    .with_seasonal_period(24); // Daily seasonality for hourly data
```

## Feature Engineering

### Time-Based Features

```rust
use neuro_divergent::data::features::TimeFeatureEngineer;

let time_features = TimeFeatureEngineer::new()
    .add_calendar_features()      // year, month, day, hour, etc.
    .add_cyclical_features()      // sin/cos encoding of time
    .add_holiday_features("US")   // Holiday indicators
    .add_business_day_features()  // Weekday/weekend indicators
    .add_seasonal_features(vec![24, 168, 8760]); // Multiple seasonalities

let enhanced_data = time_features.transform(&data)?;
```

### Lag Features

```rust
use neuro_divergent::data::features::LagFeatureEngineer;

let lag_features = LagFeatureEngineer::new()
    .add_lags(vec![1, 2, 3, 7, 14, 30])      // Individual lags
    .add_lag_range(1..=7)                     // Range of lags
    .add_seasonal_lags(vec![24, 168])         // Seasonal lags
    .with_missing_value_strategy(MissingStrategy::Drop);

let lagged_data = lag_features.transform(&data)?;
```

### Rolling Window Features

```rust
use neuro_divergent::data::features::RollingFeatureEngineer;

let rolling_features = RollingFeatureEngineer::new()
    .add_rolling_mean(vec![3, 7, 30])        // Moving averages
    .add_rolling_std(vec![7, 30])            // Rolling standard deviation
    .add_rolling_min_max(vec![7, 30])        // Rolling min/max
    .add_rolling_median(vec![7, 30])         // Rolling median
    .add_rolling_quantiles(vec![7], vec![0.25, 0.75]) // Quartiles
    .add_exponential_smoothing(vec![0.1, 0.3]); // EMA with different alphas

let windowed_data = rolling_features.transform(&data)?;
```

### Fourier Features

```rust
use neuro_divergent::data::features::FourierFeatureEngineer;

// Capture seasonality with Fourier terms
let fourier_features = FourierFeatureEngineer::new()
    .add_fourier_terms(24, 5)    // 5 Fourier terms for daily pattern
    .add_fourier_terms(168, 3)   // 3 Fourier terms for weekly pattern
    .add_fourier_terms(8760, 2); // 2 Fourier terms for yearly pattern

let fourier_data = fourier_features.transform(&data)?;
```

### Statistical Features

```rust
use neuro_divergent::data::features::StatisticalFeatureEngineer;

let stat_features = StatisticalFeatureEngineer::new()
    .add_autocorrelation_features(vec![1, 7, 30])  // ACF at different lags
    .add_trend_features(window: 30)                 // Local trend estimation
    .add_volatility_features(window: 7)             // Rolling volatility
    .add_entropy_features(window: 14)               // Local entropy
    .add_fractal_dimension_features(window: 30);    // Complexity measures

let statistical_data = stat_features.transform(&data)?;
```

### Domain-Specific Features

```rust
// E-commerce specific features
let ecommerce_features = EcommerceFeatureEngineer::new()
    .add_price_features()          // Price changes, relative pricing
    .add_promotion_features()      // Promotion indicators and intensity
    .add_inventory_features()      // Stock levels, out-of-stock events
    .add_competition_features();   // Competitive pricing, events

// Energy forecasting features
let energy_features = EnergyFeatureEngineer::new()
    .add_weather_features()        // Temperature, humidity, wind
    .add_calendar_features()       // Peak/off-peak hours
    .add_economic_features()       // Economic indicators
    .add_capacity_features();      // System capacity constraints

// Financial features
let financial_features = FinancialFeatureEngineer::new()
    .add_technical_indicators()    // RSI, MACD, Bollinger Bands
    .add_volatility_measures()     // GARCH, realized volatility
    .add_market_features()         // VIX, market sentiment
    .add_fundamental_features();   // P/E ratios, earnings data
```

## Feature Selection

### Automated Feature Selection

```rust
use neuro_divergent::data::features::FeatureSelector;

let selector = FeatureSelector::new()
    .with_method(SelectionMethod::Correlation)
    .with_threshold(0.1)           // Minimum correlation with target
    .with_max_features(50)         // Maximum number of features
    .with_remove_collinear(true)   // Remove highly correlated features
    .with_variance_threshold(0.01); // Remove low-variance features

let selected_features = selector.fit_transform(&enhanced_data)?;

// Report feature importance
let importance_scores = selector.get_feature_importance()?;
for (feature, score) in importance_scores.iter().take(10) {
    println!("{}: {:.4}", feature, score);
}
```

### Custom Feature Selection

```rust
// Mutual information based selection
let mi_selector = FeatureSelector::new()
    .with_method(SelectionMethod::MutualInformation)
    .with_k_best(30);

// Recursive feature elimination
let rfe_selector = FeatureSelector::new()
    .with_method(SelectionMethod::RecursiveElimination)
    .with_estimator(Box::new(LSTM::new(config)?))
    .with_n_features(20);

// LASSO regularization based selection
let lasso_selector = FeatureSelector::new()
    .with_method(SelectionMethod::LassoRegularization)
    .with_alpha(0.01);
```

## Data Splitting

### Time-Aware Splitting

```rust
use neuro_divergent::data::splitting::{TimeSeriesSplitter, SplitStrategy};

// Simple time split
let (train_data, test_data) = data.time_split(0.8)?;  // 80% train, 20% test

// Advanced time series splitting
let splitter = TimeSeriesSplitter::new()
    .with_strategy(SplitStrategy::TimeBasedSplit)
    .with_train_ratio(0.7)
    .with_validation_ratio(0.15)
    .with_test_ratio(0.15)
    .with_gap_size(7);  // 7-day gap between splits

let splits = splitter.split(&data)?;
let train_data = splits.train;
let val_data = splits.validation;
let test_data = splits.test;
```

### Cross-Validation Strategies

```rust
use neuro_divergent::data::splitting::TimeSeriesCV;

// Time series cross-validation
let cv = TimeSeriesCV::new()
    .with_n_splits(5)
    .with_test_size(0.2)
    .with_gap_size(7)      // Gap between train and test
    .with_expanding_window(true); // Expanding vs sliding window

let cv_splits = cv.split(&data)?;

for (fold, split) in cv_splits.iter().enumerate() {
    println!("Fold {}: Train size: {}, Test size: {}", 
             fold + 1, split.train.len(), split.test.len());
}
```

### Blocked Cross-Validation

```rust
// For multiple time series
let blocked_cv = BlockedTimeSeriesCV::new()
    .with_n_splits(5)
    .with_block_size(30)   // 30-day blocks
    .with_series_splits(true); // Split by series, not time

let blocked_splits = blocked_cv.split(&multivariate_data)?;
```

## Multi-Series Data Handling

### Global vs Local Approaches

```rust
// Global approach - all series together
let global_data = TimeSeriesDataFrame::from_multiple_series(series_collection)?
    .with_global_scaling(true)      // Scale across all series
    .with_series_encoding(true)     // Add series ID features
    .build()?;

// Local approach - each series separately
for (series_id, series_data) in series_collection.iter() {
    let local_model = LSTM::new(config.clone())?;
    local_model.fit(series_data)?;
    local_models.insert(series_id, local_model);
}
```

### Hierarchical Time Series

```rust
use neuro_divergent::data::hierarchical::HierarchicalData;

// Define hierarchy structure
let hierarchy = HierarchicalStructure::new()
    .add_level("Country", vec!["US", "CA", "MX"])
    .add_level("State", vec!["CA", "TX", "NY", "ON", "BC"])
    .add_level("City", vec!["LA", "SF", "NYC", "TOR"]);

let hierarchical_data = HierarchicalData::new()
    .with_structure(hierarchy)
    .with_bottom_level_data(city_level_data)
    .with_aggregation_method(AggregationMethod::Sum)
    .build()?;

// Forecast at all levels simultaneously
let hierarchical_forecasts = hierarchical_model.predict(&hierarchical_data)?;
```

## Data Quality Monitoring

### Drift Detection

```rust
use neuro_divergent::data::monitoring::DriftDetector;

let drift_detector = DriftDetector::new()
    .with_method(DriftMethod::KolmogorovSmirnov)
    .with_window_size(1000)
    .with_threshold(0.05);

// Monitor for distribution drift
let drift_result = drift_detector.detect_drift(&reference_data, &new_data)?;
if drift_result.drift_detected {
    println!("Data drift detected! P-value: {:.4}", drift_result.p_value);
    println!("Consider retraining the model");
}
```

### Data Quality Metrics

```rust
use neuro_divergent::data::quality::DataQualityMetrics;

let quality_metrics = DataQualityMetrics::calculate(&data)?;

println!("Data Quality Report:");
println!("===================");
println!("Completeness: {:.2}%", quality_metrics.completeness() * 100.0);
println!("Consistency: {:.2}%", quality_metrics.consistency() * 100.0);
println!("Accuracy: {:.2}%", quality_metrics.accuracy() * 100.0);
println!("Timeliness: {:.2}%", quality_metrics.timeliness() * 100.0);
println!("Validity: {:.2}%", quality_metrics.validity() * 100.0);
```

## Performance Optimization

### Memory-Efficient Processing

```rust
use neuro_divergent::data::processing::ChunkedProcessor;

// Process large datasets in chunks
let processor = ChunkedProcessor::new()
    .with_chunk_size(10000)
    .with_overlap(100)      // Overlap between chunks
    .with_parallel(true)    // Parallel processing
    .with_num_threads(8);

let processed_data = processor.process_large_dataset("large_file.csv", |chunk| {
    // Process each chunk
    let cleaned_chunk = clean_data(chunk)?;
    let features_chunk = engineer_features(cleaned_chunk)?;
    Ok(features_chunk)
})?;
```

### Lazy Loading

```rust
use neuro_divergent::data::lazy::LazyDataFrame;

// Lazy loading for very large datasets
let lazy_data = LazyDataFrame::new()
    .with_file_path("huge_dataset.parquet")
    .with_time_column("timestamp")
    .with_target_column("value")
    .with_chunk_size(1000);

// Operations are computed only when needed
let filtered_data = lazy_data
    .filter("timestamp > '2023-01-01'")
    .select(vec!["timestamp", "value", "feature1"])
    .collect()?;  // Triggers computation
```

### Caching

```rust
use neuro_divergent::data::cache::DataCache;

// Cache processed data
let cache = DataCache::new()
    .with_cache_dir("./data_cache")
    .with_compression(true)
    .with_ttl(Duration::from_secs(3600)); // 1 hour TTL

// Check cache first
if let Some(cached_data) = cache.get("processed_features_v1.0")? {
    return Ok(cached_data);
}

// Process and cache
let processed_data = expensive_feature_engineering(&raw_data)?;
cache.put("processed_features_v1.0", &processed_data)?;
```

## Best Practices

### Data Preparation Checklist

- [ ] **Validate data quality** before processing
- [ ] **Handle missing values** appropriately for your domain
- [ ] **Detect and treat outliers** based on business logic
- [ ] **Engineer relevant features** for your problem
- [ ] **Scale/normalize features** consistently
- [ ] **Use time-aware splitting** for evaluation
- [ ] **Monitor for data drift** in production
- [ ] **Document preprocessing steps** for reproducibility

### Common Pitfalls to Avoid

1. **Future Leakage**: Don't use future information in features
2. **Data Snooping**: Don't peek at test data during preprocessing
3. **Inconsistent Scaling**: Apply same scaling to train/test
4. **Ignoring Temporal Order**: Respect time sequence in splits
5. **Over-Engineering**: More features ≠ better performance
6. **Static Preprocessing**: Recompute preprocessing for new data

### Production Considerations

```rust
// Robust preprocessing pipeline for production
let preprocessing_pipeline = PreprocessingPipeline::new()
    .add_step(Box::new(MissingValueHandler::new()))
    .add_step(Box::new(OutlierDetector::new()))
    .add_step(Box::new(FeatureEngineer::new()))
    .add_step(Box::new(StandardScaler::new()))
    .with_error_handling(ErrorHandling::SkipBatch)
    .with_monitoring(true)
    .with_logging(true);

// Save pipeline for consistent preprocessing
preprocessing_pipeline.save("preprocessing_pipeline.json")?;

// Load and apply in production
let production_pipeline = PreprocessingPipeline::load("preprocessing_pipeline.json")?;
let processed_data = production_pipeline.transform(&new_data)?;
```

## Next Steps

Now that you understand data handling fundamentals:

1. **Explore Model Training**: See [Training Guide](training.md) for optimization strategies
2. **Learn Prediction Techniques**: Check [Prediction Guide](prediction.md) for forecasting approaches
3. **Understand Evaluation**: Read [Evaluation Guide](evaluation.md) for performance assessment
4. **Optimize Performance**: Review [Performance Guide](performance.md) for speed and memory optimization

Remember: quality data is the foundation of accurate forecasts. Invest time in understanding and properly preparing your data for the best results.