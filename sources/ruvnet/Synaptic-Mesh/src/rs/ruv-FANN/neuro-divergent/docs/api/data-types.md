# Data Types

The data types module provides comprehensive data structures and operations for time series forecasting. Built on Polars for high performance, these types offer efficient data manipulation with compile-time safety guarantees.

## Overview

The data type system consists of several key components:

- **`TimeSeriesDataFrame<T>`** - Main data container (pandas-compatible)
- **`ForecastDataFrame<T>`** - Forecast results and predictions
- **`TimeSeriesSchema`** - Schema definition and validation
- **`TimeSeriesDataset<T>`** - Low-level dataset for model training
- **Cross-validation results** - Specialized result containers

All data types are generic over floating-point types (`T: Float`) and support both `f32` and `f64` precision.

## TimeSeriesDataFrame<T>

The primary data structure for time series data, providing pandas-like functionality with Polars performance.

### Definition

```rust
pub struct TimeSeriesDataFrame<T: Float> {
    pub data: DataFrame,
    pub schema: TimeSeriesSchema,
    pub frequency: Option<Frequency>,
}
```

### Construction

#### From CSV Files

```rust
use neuro_divergent::{TimeSeriesDataFrame, TimeSeriesSchema, Frequency};

// Define schema
let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
    .with_static_features(vec!["category".to_string()])
    .with_exogenous_features(vec!["price".to_string(), "promotion".to_string()]);

// Load from CSV
let df = TimeSeriesDataFrame::from_csv(
    "data.csv",
    schema,
    Some(Frequency::Daily)
)?;

println!("Loaded {} rows, {} series", df.shape().0, df.n_series());
```

#### From Parquet Files

```rust
let df = TimeSeriesDataFrame::from_parquet(
    "data.parquet", 
    schema, 
    Some(Frequency::Hourly)
)?;
```

#### From Polars DataFrame

```rust
use polars::prelude::*;

// Create Polars DataFrame
let polars_df = df! {
    "unique_id" => vec!["A", "A", "B", "B"],
    "ds" => vec![1, 2, 1, 2],
    "y" => vec![10.0, 15.0, 20.0, 25.0],
}?;

// Convert to TimeSeriesDataFrame
let ts_df = TimeSeriesDataFrame::from_polars(
    polars_df,
    TimeSeriesSchema::default(),
    Some(Frequency::Daily)
)?;
```

### Data Access and Filtering

#### Basic Information

```rust
// Get shape (rows, columns)
let (n_rows, n_cols) = df.shape();

// Get column names
let columns = df.columns();

// Get unique time series IDs
let series_ids = df.unique_ids()?;

// Get time range
let (start_time, end_time) = df.time_range()?;

// Number of time series
let n_series = df.n_series();

println!("Dataset: {} rows, {} columns, {} series", n_rows, n_cols, n_series);
println!("Time range: {} to {}", start_time, end_time);
```

#### Filtering Operations

```rust
use chrono::{DateTime, Utc};

// Filter by date range
let start = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
let end = DateTime::parse_from_rfc3339("2023-12-31T23:59:59Z")?.with_timezone(&Utc);
let filtered_df = df.filter_date_range(start, end)?;

// Filter by unique ID
let series_a = df.filter_by_id("series_A")?;

// Filter by multiple IDs
let important_series = ["series_A", "series_B", "series_C"];
let filtered = df.data
    .clone()
    .lazy()
    .filter(col("unique_id").is_in(lit(Series::new("ids", important_series))))
    .collect()?;
```

#### Adding Exogenous Variables

```rust
use polars::prelude::*;

// Create exogenous data
let exog_data = df! {
    "unique_id" => vec!["A", "A", "B", "B"],
    "ds" => vec![1, 2, 1, 2],
    "weather" => vec![25.0, 27.0, 22.0, 24.0],
    "holiday" => vec![0, 1, 0, 0],
}?;

// Join with main dataset
let enhanced_df = df.with_exogenous(
    exog_data, 
    vec!["unique_id".to_string(), "ds".to_string()]
)?;
```

### Data Export

#### Save to CSV

```rust
df.to_csv("output.csv")?;
```

#### Save to Parquet

```rust
df.to_parquet("output.parquet")?;
```

### Advanced Operations

#### Lazy Operations for Large Datasets

```rust
// Work with lazy frames for memory efficiency
let lazy_result = df.data
    .clone()
    .lazy()
    .filter(col("y").gt(lit(0.0)))
    .group_by([col("unique_id")])
    .agg([
        col("y").mean().alias("mean_y"),
        col("y").std(1).alias("std_y"),
        col("y").count().alias("count")
    ])
    .collect()?;
```

#### Memory-Efficient Streaming

```rust
// For very large datasets
use polars::prelude::*;

let lazy_df = LazyFrame::scan_csv("large_file.csv", ScanArgsCSV::default())?
    .filter(col("y").is_not_null())
    .select([
        col("unique_id"),
        col("ds"),
        col("y"),
        col("y").shift(lit(1)).alias("y_lag1")
    ]);

// Process in chunks
let result = lazy_df.collect()?;
```

## ForecastDataFrame<T>

Container for forecast results from one or more models.

### Definition

```rust
pub struct ForecastDataFrame<T: Float> {
    pub data: DataFrame,
    pub models: Vec<String>,
    pub forecast_horizon: usize,
    pub confidence_levels: Option<Vec<f64>>,
    pub schema: TimeSeriesSchema,
}
```

### Usage

#### Creating Forecast DataFrames

```rust
use polars::prelude::*;

// Create forecast data
let forecast_data = df! {
    "unique_id" => vec!["A", "A", "B", "B"],
    "ds" => vec![3, 4, 3, 4],
    "LSTM" => vec![12.5, 13.2, 26.1, 27.3],
    "NBEATS" => vec![12.1, 13.0, 25.8, 26.9],
}?;

let forecast_df = ForecastDataFrame::new(
    forecast_data,
    vec!["LSTM".to_string(), "NBEATS".to_string()],
    2, // horizon
    None, // no confidence intervals
    schema
);
```

#### Accessing Model Forecasts

```rust
// Get forecasts for specific model
let lstm_forecasts = forecast_df.get_model_forecasts("LSTM")?;

// Get all model forecasts
let all_forecasts = forecast_df.get_all_forecasts();
for (model_name, forecasts) in all_forecasts {
    println!("Model {}: {} forecasts", model_name, forecasts.height());
}

// Convert to point forecasts only (remove intervals)
let point_forecasts = forecast_df.to_point_forecasts();
```

#### Working with Prediction Intervals

```rust
// Forecast with confidence intervals
let forecast_data_with_intervals = df! {
    "unique_id" => vec!["A", "A"],
    "ds" => vec![3, 4],
    "LSTM" => vec![12.5, 13.2],
    "LSTM_q0.1" => vec![10.2, 10.8],
    "LSTM_q0.9" => vec![14.8, 15.6],
}?;

let forecast_df = ForecastDataFrame::new(
    forecast_data_with_intervals,
    vec!["LSTM".to_string()],
    2,
    Some(vec![0.8]), // 80% confidence intervals
    schema
);

// Extract prediction intervals
let intervals = forecast_df.prediction_intervals()?;
```

## TimeSeriesSchema

Defines the structure and meaning of columns in time series data.

### Definition

```rust
pub struct TimeSeriesSchema {
    pub unique_id_col: String,
    pub ds_col: String,
    pub y_col: String,
    pub static_features: Vec<String>,
    pub exogenous_features: Vec<String>,
}
```

### Usage

#### Basic Schema Creation

```rust
// Minimal schema with required columns
let schema = TimeSeriesSchema::new("unique_id", "ds", "y");

// Schema with features
let schema = TimeSeriesSchema::new("series_id", "timestamp", "value")
    .with_static_features(vec![
        "category".to_string(),
        "region".to_string()
    ])
    .with_exogenous_features(vec![
        "temperature".to_string(),
        "humidity".to_string(),
        "holiday".to_string()
    ]);
```

#### Schema Validation

```rust
use polars::prelude::*;

let df = df! {
    "series_id" => vec!["A", "B"],
    "timestamp" => vec![1, 2],
    "value" => vec![10.0, 20.0],
    "category" => vec!["X", "Y"],
    "temperature" => vec![25.0, 22.0],
}?;

// Validate DataFrame against schema
schema.validate_dataframe(&df)?;

// Get column information
let all_cols = schema.all_columns();
let required_cols = schema.required_columns();

println!("All columns: {:?}", all_cols);
println!("Required columns: {:?}", required_cols);
```

#### Default Schema

```rust
// Uses standard NeuralForecast column names
let default_schema = TimeSeriesSchema::default();
// equivalent to: TimeSeriesSchema::new("unique_id", "ds", "y")
```

## TimeSeriesDataset<T>

Low-level dataset interface used by models for training and prediction.

### Definition

```rust
pub struct TimeSeriesDataset<T: Float> {
    pub values: Array2<T>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub series_ids: Vec<String>,
    pub static_features: Option<Array2<T>>,
    pub exogenous_features: Option<Array2<T>>,
    pub schema: TimeSeriesSchema,
}
```

### Usage

#### Creating from TimeSeriesDataFrame

```rust
// Convert TimeSeriesDataFrame to TimeSeriesDataset
let dataset = TimeSeriesDataset::from_dataframe(&ts_df)?;

// Access underlying arrays
println!("Values shape: {:?}", dataset.values.shape());
println!("Number of series: {}", dataset.series_ids.len());

if let Some(static_features) = &dataset.static_features {
    println!("Static features shape: {:?}", static_features.shape());
}
```

#### Manual Construction

```rust
use ndarray::{Array2, Array1};
use chrono::{DateTime, Utc};

let values = Array2::from_shape_vec((100, 1), (0..100).map(|x| x as f64).collect())?;
let timestamps: Vec<DateTime<Utc>> = (0..100)
    .map(|i| Utc::now() + chrono::Duration::days(i))
    .collect();
let series_ids = vec!["series_1".to_string(); 100];

let dataset = TimeSeriesDataset::new()
    .with_values(values)
    .with_timestamps(timestamps)
    .with_series_ids(series_ids)
    .with_schema(schema)
    .build()?;
```

#### Working with Features

```rust
// Add static features (constant per series)
let static_features = Array2::from_shape_vec((10, 2), vec![1.0; 20])?;
let dataset = dataset.with_static_features(static_features);

// Add exogenous features (time-varying)
let exog_features = Array2::from_shape_vec((100, 3), vec![0.5; 300])?;
let dataset = dataset.with_exogenous_features(exog_features);

// Validate dataset
dataset.validate()?;
```

## CrossValidationDataFrame<T>

Container for cross-validation results.

### Definition

```rust
pub struct CrossValidationDataFrame<T: Float> {
    pub data: DataFrame,
    pub cutoffs: Vec<DateTime<Utc>>,
    pub models: Vec<String>,
    pub schema: TimeSeriesSchema,
}
```

### Usage

#### Analyzing CV Results

```rust
// Access cutoff dates
let cutoffs = cv_df.cutoffs();
println!("CV cutoffs: {:?}", cutoffs);

// Get CV metrics by fold
let fold_metrics = cv_df.get_fold_metrics()?;
for (fold, metrics) in fold_metrics.iter().enumerate() {
    println!("Fold {}: MAE = {:.4}", fold, metrics.get("MAE").unwrap_or(&T::from(0.0).unwrap()));
}

// Get overall metrics
let overall_metrics = cv_df.compute_overall_metrics()?;
println!("Overall MAE: {:.4}", overall_metrics.get("MAE").unwrap_or(&T::from(0.0).unwrap()));

// Get best model by metric
let best_model = cv_df.get_best_model("MAE")?;
println!("Best model: {}", best_model);
```

#### Exporting CV Results

```rust
// Save detailed CV results
cv_df.to_csv("cv_results.csv")?;

// Export summary statistics
let summary = cv_df.summary_statistics()?;
summary.to_csv("cv_summary.csv")?;
```

## Data Validation and Quality Checks

### Automatic Validation

```rust
// Validate data quality
let validation_report = ts_df.validate_data_quality()?;

if !validation_report.issues.is_empty() {
    println!("Data quality issues found:");
    for issue in &validation_report.issues {
        println!("- {}: {}", issue.severity, issue.message);
    }
}
```

### Custom Validation Rules

```rust
use neuro_divergent::validation::{ValidationRule, ValidationSeverity};

// Define custom validation rules
let rules = vec![
    ValidationRule::new("no_negatives")
        .with_condition(|df| df.column("y").unwrap().min().unwrap() >= 0.0)
        .with_message("Target variable contains negative values")
        .with_severity(ValidationSeverity::Warning),
        
    ValidationRule::new("sufficient_data")
        .with_condition(|df| df.height() >= 100)
        .with_message("Insufficient data points for reliable training")
        .with_severity(ValidationSeverity::Error),
];

// Apply validation
let report = ts_df.validate_with_rules(&rules)?;
```

## Memory Management and Performance

### Memory-Efficient Operations

```rust
// Use lazy evaluation for large datasets
let result = ts_df.data
    .clone()
    .lazy()
    .filter(col("y").is_not_null())
    .group_by([col("unique_id")])
    .agg([
        col("y").mean().alias("mean_y"),
        col("y").count().alias("count")
    ])
    .collect()?;
```

### Streaming Processing

```rust
// Process data in chunks for memory efficiency
use polars::prelude::*;

fn process_large_dataset(file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let lazy_df = LazyFrame::scan_csv(file_path, ScanArgsCSV::default())?;
    
    // Process in chunks
    let chunk_size = 10000;
    let mut offset = 0;
    
    loop {
        let chunk = lazy_df
            .clone()
            .slice(offset, chunk_size)
            .collect()?;
            
        if chunk.is_empty() {
            break;
        }
        
        // Process chunk
        process_chunk(&chunk)?;
        
        offset += chunk_size;
    }
    
    Ok(())
}
```

### Zero-Copy Operations

```rust
// Work with views to avoid copying data
use ndarray::ArrayView2;

fn analyze_values(dataset: &TimeSeriesDataset<f64>) {
    let values_view: ArrayView2<f64> = dataset.values.view();
    
    // Perform analysis without copying data
    let mean = values_view.mean().unwrap();
    let std = values_view.std(1.0);
    
    println!("Mean: {:.4}, Std: {:.4}", mean, std);
}
```

## Type Safety and Generics

### Working with Different Precisions

```rust
// f32 for memory efficiency
let dataset_f32: TimeSeriesDataset<f32> = TimeSeriesDataset::from_dataframe(&ts_df)?;

// f64 for numerical precision
let dataset_f64: TimeSeriesDataset<f64> = TimeSeriesDataset::from_dataframe(&ts_df)?;

// Convert between precisions
let converted: TimeSeriesDataset<f32> = dataset_f64.into_precision()?;
```

### Trait Bounds and Constraints

```rust
use num_traits::Float;

fn process_dataset<T>(dataset: &TimeSeriesDataset<T>) -> Result<T, Box<dyn std::error::Error>>
where
    T: Float + Send + Sync + std::fmt::Debug,
{
    let mean = dataset.values.mean().unwrap();
    Ok(mean)
}
```

## Error Handling

All data operations return `NeuroDivergentResult<T>` for comprehensive error handling:

```rust
match ts_df.filter_by_id("nonexistent_id") {
    Ok(filtered) => {
        println!("Filtered to {} rows", filtered.shape().0);
    },
    Err(NeuroDivergentError::DataError(msg)) => {
        eprintln!("Data filtering failed: {}", msg);
    },
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

## Integration Examples

### Complete Data Pipeline

```rust
use neuro_divergent::prelude::*;

// 1. Load and validate data
let mut ts_df = TimeSeriesDataFrame::from_csv(
    "raw_data.csv",
    TimeSeriesSchema::default(),
    Some(Frequency::Daily)
)?;

// 2. Data quality checks
let validation_report = ts_df.validate_data_quality()?;
if validation_report.has_errors() {
    return Err("Data quality issues found".into());
}

// 3. Add exogenous variables
let weather_data = TimeSeriesDataFrame::from_csv("weather.csv", weather_schema, None)?;
ts_df = ts_df.with_exogenous(weather_data.data, vec!["unique_id".to_string(), "ds".to_string()])?;

// 4. Filter and preprocess
let recent_data = ts_df.filter_date_range(
    Utc::now() - chrono::Duration::days(365),
    Utc::now()
)?;

// 5. Convert to dataset for model training
let dataset = TimeSeriesDataset::from_dataframe(&recent_data)?;

// 6. Train model
let mut model = LSTM::new(config)?;
model.fit(&dataset)?;

// 7. Generate forecasts
let forecasts = model.predict(&dataset)?;
```

### Multi-Series Processing

```rust
// Process multiple time series efficiently
let series_ids = ts_df.unique_ids()?;

let mut all_forecasts = Vec::new();

for series_id in series_ids {
    let series_data = ts_df.filter_by_id(&series_id)?;
    let dataset = TimeSeriesDataset::from_dataframe(&series_data)?;
    
    let forecast = model.predict(&dataset)?;
    all_forecasts.push((series_id, forecast));
}

// Combine results
let combined_forecasts = combine_forecast_results(all_forecasts)?;
```

The data types provide a robust, efficient foundation for time series forecasting while maintaining type safety and memory efficiency throughout the pipeline.