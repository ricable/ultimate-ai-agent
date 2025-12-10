# Data Formats Migration Guide: pandas to polars

This guide covers the complete migration from pandas DataFrames to polars DataFrames, including data loading, manipulation, feature engineering, and format conversions required for neuro-divergent.

## Table of Contents

1. [Overview: pandas vs polars](#overview-pandas-vs-polars)
2. [Data Loading and I/O](#data-loading-and-io)
3. [Basic DataFrame Operations](#basic-dataframe-operations)
4. [Data Manipulation and Transformation](#data-manipulation-and-transformation)
5. [Time Series Operations](#time-series-operations)
6. [Groupby and Aggregation](#groupby-and-aggregation)
7. [Feature Engineering](#feature-engineering)
8. [Performance Optimizations](#performance-optimizations)
9. [Type System and Schema](#type-system-and-schema)
10. [Migration Utilities](#migration-utilities)

## Overview: pandas vs polars

### Key Differences

| Aspect | pandas | polars | Migration Notes |
|--------|--------|--------|-----------------|
| **Memory Model** | Copy-heavy | Zero-copy | Much better memory efficiency |
| **Execution** | Eager | Lazy (default) | Query optimization opportunities |
| **Type System** | Dynamic | Static schema | Better type safety |
| **Performance** | Single-threaded | Multi-threaded | Automatic parallelization |
| **API Style** | Method chaining | Expression-based | More functional approach |
| **Null Handling** | NaN/None mix | Explicit null | Cleaner null semantics |

### Performance Benefits

- **4-6x faster** data loading and processing
- **2-3x less memory usage** for large datasets
- **Automatic parallelization** across CPU cores
- **Query optimization** with lazy evaluation
- **Better cache locality** with columnar storage

## Data Loading and I/O

### CSV Loading

**pandas**:
```python
import pandas as pd

# Basic CSV loading
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv(
    'data.csv',
    parse_dates=['ds'],
    dtype={'unique_id': 'str', 'y': 'float64'},
    index_col=None
)

# Large file handling
chunks = pd.read_csv('large_file.csv', chunksize=10000)
df = pd.concat(chunks, ignore_index=True)
```

**polars**:
```rust
use polars::prelude::*;

// Basic CSV loading (lazy)
let df = LazyFrame::scan_csv("data.csv", Default::default())?
    .collect()?;

// With schema and options
let schema = Schema::from_iter([
    ("unique_id", DataType::String),
    ("ds", DataType::Datetime(TimeUnit::Nanoseconds, None)),
    ("y", DataType::Float64),
]);

let df = LazyFrame::scan_csv("data.csv", ScanArgsCSV {
    has_header: true,
    schema: Some(Arc::new(schema)),
    ..Default::default()
})?
.with_columns([
    col("ds").str().strptime(StrptimeOptions::default())
])
.collect()?;

// Large file handling (automatic streaming)
let df = LazyFrame::scan_csv("large_file.csv", Default::default())?
    .collect()?; // polars handles memory automatically
```

### Other File Formats

**pandas**:
```python
# Parquet
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet')

# JSON
df = pd.read_json('data.json')
df.to_json('output.json')

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

**polars**:
```rust
// Parquet (preferred format)
let df = LazyFrame::scan_parquet("data.parquet", Default::default())?
    .collect()?;
df.write_parquet("output.parquet", ParquetWriteOptions::default())?;

// JSON
let df = LazyFrame::scan_ndjson("data.ndjson", ScanArgsNdJson::default())?
    .collect()?;
df.write_ndjson("output.ndjson")?;

// Note: polars doesn't support Excel directly, convert via pandas first
```

### Data Format Conversion

**Python Helper Function**:
```python
import pandas as pd
import polars as pl

def pandas_to_polars(df_pandas: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to polars DataFrame"""
    return pl.from_pandas(df_pandas)

def polars_to_pandas(df_polars: pl.DataFrame) -> pd.DataFrame:
    """Convert polars DataFrame to pandas DataFrame"""
    return df_polars.to_pandas()

# Usage
pandas_df = pd.read_csv('data.csv')
polars_df = pandas_to_polars(pandas_df)
```

**Rust Conversion Functions**:
```rust
use polars::prelude::*;
use pyo3::prelude::*;

// Convert from Python pandas DataFrame
fn from_pandas_dataframe(py_df: &PyAny) -> PyResult<DataFrame> {
    // This requires pyo3-polars crate
    let df = py_df.extract::<DataFrame>()?;
    Ok(df)
}

// Save for Python compatibility
fn save_for_python(df: &DataFrame, path: &str) -> PolarsResult<()> {
    // Use parquet for best compatibility and performance
    df.write_parquet(path, ParquetWriteOptions::default())
}
```

## Basic DataFrame Operations

### Data Inspection

**pandas**:
```python
# Basic info
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.info())
print(df.describe())

# Column operations
print(df.columns.tolist())
df_subset = df[['unique_id', 'ds', 'y']]
```

**polars**:
```rust
// Basic info
println!("Shape: {:?}", df.shape());
println!("Schema: {}", df.schema());
println!("{}", df.head(Some(5)));
println!("{}", df.describe(None)?);

// Column operations
let columns: Vec<&str> = df.get_column_names();
let df_subset = df.select([col("unique_id"), col("ds"), col("y")])?;
```

### Filtering and Selection

**pandas**:
```python
# Filtering
df_filtered = df[df['y'] > 100]
df_filtered = df[(df['y'] > 100) & (df['unique_id'] == 'series1')]

# Row selection
df_sample = df.sample(n=1000, random_state=42)
df_head = df.head(100)

# Column selection
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_cols]
```

**polars**:
```rust
// Filtering
let df_filtered = df.filter(col("y").gt(100))?;
let df_filtered = df.filter(
    col("y").gt(100).and(col("unique_id").eq(lit("series1")))
)?;

// Row selection
let df_sample = df.sample_n(
    &Series::new("", &[1000_u32]), 
    false, 
    Some(42),
    None
)?;
let df_head = df.head(Some(100));

// Column selection by type
let numeric_cols: Vec<_> = df.schema()
    .iter()
    .filter(|(_, dtype)| matches!(dtype, DataType::Float64 | DataType::Int64))
    .map(|(name, _)| col(name))
    .collect();
let df_numeric = df.select(numeric_cols)?;
```

## Data Manipulation and Transformation

### Adding and Modifying Columns

**pandas**:
```python
# Add new columns
df['y_log'] = np.log(df['y'])
df['y_lag1'] = df.groupby('unique_id')['y'].shift(1)
df['y_diff'] = df.groupby('unique_id')['y'].diff()

# Conditional columns
df['y_category'] = df['y'].apply(
    lambda x: 'high' if x > df['y'].median() else 'low'
)

# Multiple transformations
df = df.assign(
    y_scaled=(df['y'] - df['y'].mean()) / df['y'].std(),
    y_rank=df.groupby('unique_id')['y'].rank(),
    y_pct_change=df.groupby('unique_id')['y'].pct_change()
)
```

**polars**:
```rust
// Add new columns
let df = df.with_columns([
    col("y").log(2.71828).alias("y_log"),
    col("y").shift(1).over([col("unique_id")]).alias("y_lag1"),
    col("y").diff(1, None).over([col("unique_id")]).alias("y_diff"),
])?;

// Conditional columns
let median_y = df.column("y")?.median().unwrap();
let df = df.with_columns([
    when(col("y").gt(median_y))
        .then(lit("high"))
        .otherwise(lit("low"))
        .alias("y_category")
])?;

// Multiple transformations
let df = df.with_columns([
    ((col("y") - col("y").mean()) / col("y").std(1)).alias("y_scaled"),
    col("y").rank(RankOptions::default(), None)
        .over([col("unique_id")]).alias("y_rank"),
    (col("y").pct_change(None) * lit(100))
        .over([col("unique_id")]).alias("y_pct_change"),
])?;
```

### Handling Missing Values

**pandas**:
```python
# Check for missing values
print(df.isnull().sum())
print(df.isnull().any())

# Handle missing values
df_filled = df.fillna(method='ffill')  # Forward fill
df_filled = df.fillna(df.mean())      # Fill with mean
df_dropped = df.dropna()              # Drop missing

# Group-wise filling
df['y_filled'] = df.groupby('unique_id')['y'].transform(
    lambda x: x.fillna(x.mean())
)
```

**polars**:
```rust
// Check for missing values
let null_counts = df.null_count();
println!("Null counts: {}", null_counts);

// Handle missing values
let df_filled = df.fill_null(AnyValue::Null)?;                    // Forward fill
let df_filled = df.fill_null(col("y").mean())?;                   // Fill with mean
let df_dropped = df.drop_nulls(None)?;                            // Drop missing

// Group-wise filling
let df = df.with_columns([
    col("y").fill_null(col("y").mean().over([col("unique_id")]))
        .alias("y_filled")
])?;
```

## Time Series Operations

### Date and Time Handling

**pandas**:
```python
# Date parsing and formatting
df['ds'] = pd.to_datetime(df['ds'])
df['year'] = df['ds'].dt.year
df['month'] = df['ds'].dt.month
df['dayofweek'] = df['ds'].dt.dayofweek
df['quarter'] = df['ds'].dt.quarter

# Date ranges
date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

# Resampling
df_monthly = df.set_index('ds').groupby('unique_id').resample('M')['y'].mean()
```

**polars**:
```rust
// Date parsing and formatting
let df = df.with_columns([
    col("ds").str().strptime(StrptimeOptions::default()),
    col("ds").dt().year().alias("year"),
    col("ds").dt().month().alias("month"),
    col("ds").dt().weekday().alias("dayofweek"),
    col("ds").dt().quarter().alias("quarter"),
])?;

// Date ranges
let date_range = date_range(
    NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
    NaiveDate::from_ymd_opt(2023, 12, 31).unwrap(),
    Duration::parse("1d"),
    ClosedWindow::Both,
    TimeUnit::Nanoseconds,
    None,
)?;

// Resampling (group by time periods)
let df_monthly = df
    .sort(["unique_id", "ds"], SortMultipleOptions::default())
    .group_by_dynamic(
        col("ds"),
        [],
        DynamicGroupOptions {
            every: Duration::parse("1mo"),
            period: Duration::parse("1mo"),
            ..Default::default()
        },
    )
    .agg([col("y").mean()])?;
```

### Rolling Windows and Lagged Features

**pandas**:
```python
# Rolling statistics
df['y_rolling_mean'] = df.groupby('unique_id')['y'].transform(
    lambda x: x.rolling(window=7).mean()
)
df['y_rolling_std'] = df.groupby('unique_id')['y'].transform(
    lambda x: x.rolling(window=7).std()
)

# Expanding windows
df['y_expanding_mean'] = df.groupby('unique_id')['y'].transform(
    lambda x: x.expanding().mean()
)

# Multiple lags
for lag in [1, 7, 30]:
    df[f'y_lag_{lag}'] = df.groupby('unique_id')['y'].shift(lag)
```

**polars**:
```rust
// Rolling statistics
let df = df.with_columns([
    col("y").rolling_mean(RollingOptions::default().window_size(Duration::parse("7i")))
        .over([col("unique_id")]).alias("y_rolling_mean"),
    col("y").rolling_std(RollingOptions::default().window_size(Duration::parse("7i")))
        .over([col("unique_id")]).alias("y_rolling_std"),
])?;

// Expanding windows
let df = df.with_columns([
    col("y").sum().over([col("unique_id")]) / 
    col("y").count().over([col("unique_id")]).alias("y_expanding_mean"),
])?;

// Multiple lags
let lag_cols: Vec<Expr> = [1, 7, 30].iter().map(|&lag| {
    col("y").shift(lag).over([col("unique_id")]).alias(&format!("y_lag_{}", lag))
}).collect();

let df = df.with_columns(lag_cols)?;
```

## Groupby and Aggregation

### Basic Groupby Operations

**pandas**:
```python
# Basic aggregation
grouped = df.groupby('unique_id').agg({
    'y': ['mean', 'std', 'min', 'max'],
    'ds': ['count']
})

# Multiple grouping columns
grouped = df.groupby(['unique_id', 'year']).agg({
    'y': 'mean'
})

# Custom aggregation functions
def custom_agg(x):
    return x.quantile(0.9) - x.quantile(0.1)

grouped = df.groupby('unique_id')['y'].agg(custom_agg)
```

**polars**:
```rust
// Basic aggregation
let grouped = df.group_by([col("unique_id")])
    .agg([
        col("y").mean().alias("y_mean"),
        col("y").std(1).alias("y_std"),
        col("y").min().alias("y_min"),
        col("y").max().alias("y_max"),
        col("ds").count().alias("ds_count"),
    ])?;

// Multiple grouping columns
let grouped = df.group_by([col("unique_id"), col("year")])
    .agg([col("y").mean().alias("y_mean")])?;

// Custom aggregation functions
let grouped = df.group_by([col("unique_id")])
    .agg([
        (col("y").quantile(lit(0.9), QuantileInterpolOptions::default()) - 
         col("y").quantile(lit(0.1), QuantileInterpolOptions::default()))
        .alias("y_iqr_90_10")
    ])?;
```

### Advanced Groupby Patterns

**pandas**:
```python
# Transform (broadcast back to original size)
df['y_group_mean'] = df.groupby('unique_id')['y'].transform('mean')
df['y_group_rank'] = df.groupby('unique_id')['y'].transform('rank')

# Apply custom functions
def group_statistics(group):
    return pd.Series({
        'mean': group['y'].mean(),
        'trend': np.polyfit(range(len(group)), group['y'], 1)[0],
        'seasonality': group['y'].std() / group['y'].mean()
    })

stats = df.groupby('unique_id').apply(group_statistics)
```

**polars**:
```rust
// Transform (broadcast back to original size)
let df = df.with_columns([
    col("y").mean().over([col("unique_id")]).alias("y_group_mean"),
    col("y").rank(RankOptions::default(), None)
        .over([col("unique_id")]).alias("y_group_rank"),
])?;

// Custom aggregation functions (using expressions)
let stats = df.group_by([col("unique_id")])
    .agg([
        col("y").mean().alias("mean"),
        // Trend calculation using linear regression
        col("y").pearson_corr(
            (col("y").cumcount().cast(DataType::Float64)), 
            1
        ).alias("trend"),
        // Coefficient of variation
        (col("y").std(1) / col("y").mean()).alias("seasonality"),
    ])?;
```

## Feature Engineering

### Time-Based Features

**pandas**:
```python
# Comprehensive time features
df['hour'] = df['ds'].dt.hour
df['day'] = df['ds'].dt.day
df['week'] = df['ds'].dt.isocalendar().week
df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6])
df['is_month_start'] = df['ds'].dt.is_month_start
df['is_month_end'] = df['ds'].dt.is_month_end

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['ds'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['ds'].dt.month / 12)
df['hour_sin'] = np.sin(2 * np.pi * df['ds'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['ds'].dt.hour / 24)
```

**polars**:
```rust
use std::f64::consts::PI;

// Comprehensive time features
let df = df.with_columns([
    col("ds").dt().hour().alias("hour"),
    col("ds").dt().day().alias("day"),
    col("ds").dt().week().alias("week"),
    col("ds").dt().weekday().is_in([lit(6), lit(7)]).alias("is_weekend"),
    col("ds").dt().month_start().alias("is_month_start"),
    col("ds").dt().month_end().alias("is_month_end"),
])?;

// Cyclical encoding
let df = df.with_columns([
    (col("ds").dt().month().cast(DataType::Float64) * lit(2.0 * PI / 12.0))
        .sin().alias("month_sin"),
    (col("ds").dt().month().cast(DataType::Float64) * lit(2.0 * PI / 12.0))
        .cos().alias("month_cos"),
    (col("ds").dt().hour().cast(DataType::Float64) * lit(2.0 * PI / 24.0))
        .sin().alias("hour_sin"),
    (col("ds").dt().hour().cast(DataType::Float64) * lit(2.0 * PI / 24.0))
        .cos().alias("hour_cos"),
])?;
```

### Statistical Features

**pandas**:
```python
# Statistical transformations
df['y_log'] = np.log1p(df['y'])  # log(1 + x)
df['y_sqrt'] = np.sqrt(df['y'])
df['y_reciprocal'] = 1 / (df['y'] + 1e-8)

# Normalization per group
df['y_zscore'] = df.groupby('unique_id')['y'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['y_minmax'] = df.groupby('unique_id')['y'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# Percentile-based features
df['y_percentile'] = df.groupby('unique_id')['y'].transform(
    lambda x: x.rank(pct=True)
)
```

**polars**:
```rust
// Statistical transformations
let df = df.with_columns([
    (col("y") + lit(1.0)).log(2.71828).alias("y_log"),
    col("y").sqrt().alias("y_sqrt"),
    (lit(1.0) / (col("y") + lit(1e-8))).alias("y_reciprocal"),
])?;

// Normalization per group
let df = df.with_columns([
    ((col("y") - col("y").mean().over([col("unique_id")])) / 
     col("y").std(1).over([col("unique_id")])).alias("y_zscore"),
    ((col("y") - col("y").min().over([col("unique_id")])) / 
     (col("y").max().over([col("unique_id")]) - 
      col("y").min().over([col("unique_id")]))).alias("y_minmax"),
])?;

// Percentile-based features
let df = df.with_columns([
    col("y").rank(RankOptions::default(), None)
        .over([col("unique_id")])
        .cast(DataType::Float64) / 
    col("y").count().over([col("unique_id")]).cast(DataType::Float64)
        .alias("y_percentile"),
])?;
```

## Performance Optimizations

### Memory-Efficient Operations

**pandas**:
```python
# Memory optimization
df = df.astype({
    'unique_id': 'category',
    'y': 'float32'  # Reduce precision if acceptable
})

# Chunked processing for large datasets
def process_chunk(chunk):
    # Process each chunk
    return chunk.groupby('unique_id')['y'].mean()

results = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    results.append(process_chunk(chunk))

final_result = pd.concat(results)
```

**polars**:
```rust
// Memory optimization (automatic in polars)
let schema = Schema::from_iter([
    ("unique_id", DataType::String),  // Categorical if many repeats
    ("ds", DataType::Datetime(TimeUnit::Nanoseconds, None)),
    ("y", DataType::Float32),  // Reduce precision if acceptable
]);

// Lazy evaluation for large datasets (automatic memory management)
let result = LazyFrame::scan_csv("large_file.csv", ScanArgsCSV {
    schema: Some(Arc::new(schema)),
    ..Default::default()
})?
.group_by([col("unique_id")])
.agg([col("y").mean()])
.collect()?;  // Only materialize when needed
```

### Query Optimization

**polars Lazy Evaluation**:
```rust
// Complex pipeline with automatic optimization
let optimized_result = LazyFrame::scan_csv("data.csv", Default::default())?
    .filter(col("y").gt(0))                    // Filter early
    .select([col("unique_id"), col("ds"), col("y")])  // Select needed columns
    .with_columns([
        col("y").log(2.71828).alias("y_log"),
        col("y").shift(1).over([col("unique_id")]).alias("y_lag1"),
    ])
    .group_by([col("unique_id")])
    .agg([
        col("y").mean().alias("y_mean"),
        col("y_log").std(1).alias("y_log_std"),
    ])
    .collect()?;  // polars optimizes the entire pipeline
```

## Type System and Schema

### Schema Definition

**polars Schema Management**:
```rust
use polars::prelude::*;

// Define schema upfront
let schema = Schema::from_iter([
    ("unique_id", DataType::String),
    ("ds", DataType::Datetime(TimeUnit::Nanoseconds, None)),
    ("y", DataType::Float64),
    ("static_0", DataType::Float64),
    ("historic_exog_1", DataType::Float64),
    ("future_exog_1", DataType::Float64),
]);

// Validate data against schema
fn validate_neuroforecast_schema(df: &DataFrame) -> PolarsResult<()> {
    let required_columns = ["unique_id", "ds", "y"];
    
    for col_name in required_columns {
        if !df.get_column_names().contains(&col_name) {
            polars_bail!(ColumnNotFound: "Required column '{}' not found", col_name);
        }
    }
    
    // Validate types
    let y_dtype = df.column("y")?.dtype();
    if !matches!(y_dtype, DataType::Float64 | DataType::Float32) {
        polars_bail!(SchemaMismatch: "Column 'y' must be float type, got {:?}", y_dtype);
    }
    
    Ok(())
}
```

## Migration Utilities

### Conversion Helper Functions

```python
# Python helper for pandas → polars conversion
import pandas as pd
import polars as pl
from typing import Dict, Any

def convert_pandas_to_polars_for_neuroforecast(
    df_pandas: pd.DataFrame,
    ensure_schema: bool = True
) -> pl.DataFrame:
    """
    Convert pandas DataFrame to polars with NeuralForecast schema validation.
    """
    # Convert to polars
    df_polars = pl.from_pandas(df_pandas)
    
    if ensure_schema:
        # Ensure required columns exist
        required_cols = ['unique_id', 'ds', 'y']
        missing_cols = [col for col in required_cols if col not in df_polars.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure proper data types
        df_polars = df_polars.with_columns([
            pl.col("unique_id").cast(pl.String),
            pl.col("ds").cast(pl.Datetime),
            pl.col("y").cast(pl.Float64),
        ])
    
    return df_polars

def save_for_rust_consumption(df: pl.DataFrame, path: str) -> None:
    """Save polars DataFrame in format optimized for Rust consumption."""
    # Parquet is the best format for Rust/polars interop
    df.write_parquet(f"{path}.parquet")
    
    # Also save schema for validation
    schema_dict = {name: str(dtype) for name, dtype in df.schema.items()}
    
    import json
    with open(f"{path}_schema.json", 'w') as f:
        json.dump(schema_dict, f, indent=2)
```

```rust
// Rust utilities for data validation and conversion
use polars::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

pub fn load_with_schema_validation(
    data_path: &str,
    schema_path: Option<&str>,
) -> PolarsResult<DataFrame> {
    let mut df = LazyFrame::scan_parquet(data_path, Default::default())?;
    
    if let Some(schema_file) = schema_path {
        let schema_content = std::fs::read_to_string(schema_file)
            .map_err(|e| PolarsError::Io(Arc::new(e)))?;
        let expected_schema: HashMap<String, String> = serde_json::from_str(&schema_content)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        
        // Validate schema matches expectations
        let df_collected = df.collect()?;
        validate_schema(&df_collected, &expected_schema)?;
        return Ok(df_collected);
    }
    
    df.collect()
}

fn validate_schema(
    df: &DataFrame, 
    expected_schema: &HashMap<String, String>
) -> PolarsResult<()> {
    for (col_name, expected_type) in expected_schema {
        match df.column(col_name) {
            Ok(series) => {
                let actual_type = format!("{:?}", series.dtype());
                if actual_type != *expected_type {
                    polars_bail!(SchemaMismatch: 
                        "Column '{}' type mismatch: expected {}, got {}", 
                        col_name, expected_type, actual_type
                    );
                }
            }
            Err(_) => {
                polars_bail!(ColumnNotFound: "Expected column '{}' not found", col_name);
            }
        }
    }
    Ok(())
}

pub fn prepare_for_neuroforecast(df: DataFrame) -> PolarsResult<DataFrame> {
    // Ensure data is properly sorted and formatted for time series
    df.lazy()
        .sort([col("unique_id"), col("ds")], SortMultipleOptions::default())
        .with_columns([
            // Ensure no nulls in critical columns
            col("unique_id").fill_null(lit("unknown")),
            col("y").fill_null(col("y").median()),
        ])
        .collect()
}
```

---

**Next**: Continue to [Model Equivalents](model-equivalents.md) for detailed model mapping documentation.