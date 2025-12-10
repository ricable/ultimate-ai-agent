//! Time series data structures and processing utilities.
//!
//! This module provides comprehensive support for time series data handling,
//! including data validation, preprocessing, and transformation operations.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;

use chrono::{DateTime, TimeZone, Utc};
use ndarray::{Array1, Array2};
use num_traits::Float;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{data_error, error::{ErrorBuilder, NeuroDivergentError, NeuroDivergentResult}};

/// Main data structure for time series data, equivalent to pandas DataFrame
#[derive(Debug, Clone)]
pub struct TimeSeriesDataFrame<T: Float> {
    /// The underlying Polars DataFrame
    pub data: DataFrame,
    /// Schema definition for the time series
    pub schema: TimeSeriesSchema,
    /// Phantom data for type parameter
    phantom: PhantomData<T>,
}

/// Schema definition for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesSchema {
    /// Unique identifier column name
    pub unique_id_col: String,
    /// Date/time column name
    pub ds_col: String,
    /// Target variable column name  
    pub y_col: String,
    /// Static feature column names
    pub static_features: Vec<String>,
    /// Historical exogenous feature column names
    pub historical_exogenous: Vec<String>,
    /// Future exogenous feature column names
    pub future_exogenous: Vec<String>,
    /// Data type constraints
    pub data_types: HashMap<String, DataType>,
}

/// Internal time series dataset for model training and prediction
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset<T: Float> {
    /// Series identifiers
    pub unique_ids: Vec<String>,
    /// Time series values organized by series
    pub series_data: HashMap<String, SeriesData<T>>,
    /// Static features per series
    pub static_features: Option<HashMap<String, Vec<T>>>,
    /// Schema information
    pub schema: TimeSeriesSchema,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Data for a single time series
#[derive(Debug, Clone)]
pub struct SeriesData<T: Float> {
    /// Timestamps for this series
    pub timestamps: Vec<DateTime<Utc>>,
    /// Target values
    pub target_values: Vec<T>,
    /// Historical exogenous variables (if any)
    pub historical_exogenous: Option<Array2<T>>, // shape: (time_steps, n_features)
    /// Future exogenous variables (if any)
    pub future_exogenous: Option<Array2<T>>, // shape: (future_steps, n_features)
    /// Series length
    pub length: usize,
    /// Series frequency (inferred)
    pub frequency: Option<String>,
}

/// Dataset metadata and statistics
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Total number of series
    pub n_series: usize,
    /// Total number of observations
    pub n_observations: usize,
    /// Average series length
    pub avg_series_length: f64,
    /// Minimum series length
    pub min_series_length: usize,
    /// Maximum series length
    pub max_series_length: usize,
    /// Data frequency (if consistent across series)
    pub frequency: Option<String>,
    /// Time range
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    /// Missing value statistics
    pub missing_stats: MissingValueStats,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Missing value statistics
#[derive(Debug, Clone)]
pub struct MissingValueStats {
    /// Total missing values
    pub total_missing: usize,
    /// Missing percentage
    pub missing_percentage: f64,
    /// Missing values per column
    pub missing_per_column: HashMap<String, usize>,
    /// Series with missing values
    pub series_with_missing: Vec<String>,
}

/// Data validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Validation status
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Data quality score (0-100)
    pub quality_score: f64,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Data validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Affected column (if applicable)
    pub column: Option<String>,
    /// Affected series (if applicable)
    pub series_id: Option<String>,
    /// Error severity
    pub severity: ErrorSeverity,
}

/// Data validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Affected column (if applicable)
    pub column: Option<String>,
    /// Affected series (if applicable)
    pub series_id: Option<String>,
}

/// Error severity levels
#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    /// Critical error that prevents processing
    Critical,
    /// High severity error
    High,
    /// Medium severity error
    Medium,
    /// Low severity error
    Low,
}

/// Data preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig<T: Float> {
    /// Scaling configuration
    pub scaling: Option<ScalingConfig<T>>,
    /// Missing value handling
    pub missing_values: MissingValueConfig<T>,
    /// Outlier detection and handling
    pub outliers: OutlierConfig<T>,
    /// Feature engineering options
    pub feature_engineering: FeatureEngineeringConfig,
    /// Validation options
    pub validation: ValidationConfig,
}

/// Scaling configuration options
#[derive(Debug, Clone)]
pub struct ScalingConfig<T: Float> {
    /// Scaling method
    pub method: ScalingMethod,
    /// Feature range for MinMax scaling
    pub feature_range: Option<(T, T)>,
    /// Whether to scale per series individually
    pub per_series: bool,
    /// Columns to exclude from scaling
    pub exclude_columns: Vec<String>,
}

/// Available scaling methods
#[derive(Debug, Clone)]
pub enum ScalingMethod {
    /// Standard (z-score) normalization
    Standard,
    /// Min-max scaling
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// No scaling
    None,
}

/// Missing value handling configuration
#[derive(Debug, Clone)]
pub struct MissingValueConfig<T: Float> {
    /// Strategy for handling missing values
    pub strategy: MissingValueStrategy<T>,
    /// Maximum allowed missing percentage
    pub max_missing_percentage: f64,
    /// Whether to interpolate missing values
    pub interpolate: bool,
    /// Interpolation method
    pub interpolation_method: InterpolationMethod,
}

/// Missing value strategies
#[derive(Debug, Clone)]
pub enum MissingValueStrategy<T: Float> {
    /// Drop rows with missing values
    Drop,
    /// Fill with a constant value
    FillConstant(T),
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// Fill with mean
    FillMean,
    /// Fill with median
    FillMedian,
    /// Linear interpolation
    Interpolate,
}

/// Interpolation methods
#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    Cubic,
    /// Polynomial interpolation
    Polynomial {
        /// The degree of the polynomial
        degree: usize
    },
}

/// Outlier detection and handling configuration
#[derive(Debug, Clone)]
pub struct OutlierConfig<T: Float> {
    /// Detection method
    pub detection_method: OutlierDetectionMethod<T>,
    /// Handling strategy
    pub handling_strategy: OutlierHandlingStrategy<T>,
    /// Whether to apply per series
    pub per_series: bool,
}

/// Outlier detection methods
#[derive(Debug, Clone)]
pub enum OutlierDetectionMethod<T: Float> {
    /// Z-score based detection
    ZScore {
        /// The z-score threshold for outlier detection
        threshold: T
    },
    /// IQR based detection
    IQR {
        /// The IQR multiplier for outlier detection
        multiplier: T
    },
    /// Isolation Forest
    IsolationForest {
        /// The expected proportion of outliers in the data
        contamination: T
    },
    /// No outlier detection
    None,
}

/// Outlier handling strategies
#[derive(Debug, Clone)]
pub enum OutlierHandlingStrategy<T: Float> {
    /// Remove outliers
    Remove,
    /// Clip to bounds
    Clip,
    /// Replace with median
    ReplaceMedian,
    /// Replace with mean
    ReplaceMean,
    /// Replace with constant
    ReplaceConstant(T),
}

/// Feature engineering configuration
#[derive(Debug, Clone)]
pub struct FeatureEngineeringConfig {
    /// Add lag features
    pub add_lags: Option<Vec<usize>>,
    /// Add rolling window features
    pub add_rolling_features: Option<RollingFeaturesConfig>,
    /// Add calendar features
    pub add_calendar_features: bool,
    /// Add seasonal decomposition
    pub add_seasonal_decomposition: bool,
    /// Custom feature transformations
    pub custom_features: Vec<String>,
}

/// Rolling window features configuration
#[derive(Debug, Clone)]
pub struct RollingFeaturesConfig {
    /// Window sizes
    pub window_sizes: Vec<usize>,
    /// Statistics to compute
    pub statistics: Vec<RollingStatistic>,
    /// Minimum periods required
    pub min_periods: Option<usize>,
}

/// Available rolling statistics
#[derive(Debug, Clone)]
pub enum RollingStatistic {
    /// Mean
    Mean,
    /// Standard deviation
    Std,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Median
    Median,
    /// Quantile
    Quantile(f64),
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check for data consistency
    pub check_consistency: bool,
    /// Check for sufficient data
    pub check_data_sufficiency: bool,
    /// Minimum required observations per series
    pub min_observations: Option<usize>,
    /// Check time series regularity
    pub check_regularity: bool,
    /// Allowed gap tolerance for regularity check
    pub gap_tolerance: Option<chrono::Duration>,
}

impl TimeSeriesSchema {
    /// Create a new schema with required columns
    pub fn new(
        unique_id_col: impl Into<String>,
        ds_col: impl Into<String>,
        y_col: impl Into<String>,
    ) -> Self {
        Self {
            unique_id_col: unique_id_col.into(),
            ds_col: ds_col.into(),
            y_col: y_col.into(),
            static_features: Vec::new(),
            historical_exogenous: Vec::new(),
            future_exogenous: Vec::new(),
            data_types: HashMap::new(),
        }
    }

    /// Add static feature columns
    pub fn with_static_features(mut self, features: Vec<String>) -> Self {
        self.static_features = features;
        self
    }

    /// Add historical exogenous feature columns
    pub fn with_historical_exogenous(mut self, features: Vec<String>) -> Self {
        self.historical_exogenous = features;
        self
    }

    /// Add future exogenous feature columns
    pub fn with_future_exogenous(mut self, features: Vec<String>) -> Self {
        self.future_exogenous = features;
        self
    }

    /// Validate schema against a DataFrame
    pub fn validate_dataframe(&self, df: &DataFrame) -> NeuroDivergentResult<()> {
        let columns: Vec<String> = df.get_column_names().into_iter().map(|s| s.to_string()).collect();
        
        // Check required columns
        if !columns.contains(&self.unique_id_col) {
            return Err(data_error!(
                format!("Required unique_id column '{}' not found", self.unique_id_col),
                field = "unique_id_col"
            ));
        }
        
        if !columns.contains(&self.ds_col) {
            return Err(data_error!(
                format!("Required timestamp column '{}' not found", self.ds_col),
                field = "ds_col"
            ));
        }
        
        if !columns.contains(&self.y_col) {
            return Err(data_error!(
                format!("Required target column '{}' not found", self.y_col),
                field = "y_col"
            ));
        }

        // Check optional columns
        for feature in &self.static_features {
            if !columns.contains(feature) {
                return Err(data_error!(
                    format!("Static feature column '{}' not found", feature),
                    field = "static_features"
                ));
            }
        }

        for feature in &self.historical_exogenous {
            if !columns.contains(feature) {
                return Err(data_error!(
                    format!("Historical exogenous column '{}' not found", feature),
                    field = "historical_exogenous"
                ));
            }
        }

        for feature in &self.future_exogenous {
            if !columns.contains(feature) {
                return Err(data_error!(
                    format!("Future exogenous column '{}' not found", feature),
                    field = "future_exogenous"
                ));
            }
        }

        Ok(())
    }

    /// Get all column names defined in this schema
    pub fn all_columns(&self) -> Vec<String> {
        let mut columns = vec![
            self.unique_id_col.clone(),
            self.ds_col.clone(),
            self.y_col.clone(),
        ];
        columns.extend(self.static_features.clone());
        columns.extend(self.historical_exogenous.clone());
        columns.extend(self.future_exogenous.clone());
        columns
    }
}

impl<T: Float> TimeSeriesDataFrame<T> {
    /// Create from Polars DataFrame with schema
    pub fn from_polars(df: DataFrame, schema: TimeSeriesSchema) -> NeuroDivergentResult<Self> {
        // Validate schema against DataFrame
        schema.validate_dataframe(&df)?;
        
        Ok(Self {
            data: df,
            schema,
            phantom: PhantomData,
        })
    }

    /// Create from CSV file
    /// TODO: Fix polars CSV reading API compatibility
    #[allow(dead_code)]
    pub fn from_csv<P: AsRef<Path>>(
        _path: P,
        _schema: TimeSeriesSchema,
    ) -> NeuroDivergentResult<Self> {
        todo!("CSV reading needs to be updated for current polars version")
    }

    /// Create from Parquet file
    pub fn from_parquet<P: AsRef<Path>>(
        path: P,
        schema: TimeSeriesSchema,
    ) -> NeuroDivergentResult<Self> {
        let df = LazyFrame::scan_parquet(path, ScanArgsParquet::default())
            .map_err(|e| ErrorBuilder::data(format!("Failed to read Parquet: {}", e)).build())?
            .collect()
            .map_err(|e| ErrorBuilder::data(format!("Failed to collect DataFrame: {}", e)).build())?;

        Self::from_polars(df, schema)
    }

    /// Get unique time series identifiers
    pub fn unique_ids(&self) -> NeuroDivergentResult<Vec<String>> {
        let ids = self.data
            .column(&self.schema.unique_id_col)
            .map_err(|e| ErrorBuilder::data(format!("Failed to get unique IDs: {}", e)).build())?
            .unique()
            .map_err(|e| ErrorBuilder::data(format!("Failed to get unique values: {}", e)).build())?
            .utf8()
            .map_err(|e| ErrorBuilder::data(format!("Unique ID column is not string type: {}", e)).build())?
            .into_iter()
            .filter_map(|opt| opt.map(|s| s.to_string()))
            .collect();

        Ok(ids)
    }

    /// Filter by date range
    pub fn filter_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> NeuroDivergentResult<Self> {
        let filtered_df = self.data
            .clone()
            .lazy()
            .filter(
                col(&self.schema.ds_col)
                    .gt_eq(lit(start.naive_utc()))
                    .and(col(&self.schema.ds_col).lt_eq(lit(end.naive_utc())))
            )
            .collect()
            .map_err(|e| ErrorBuilder::data(format!("Failed to filter by date range: {}", e)).build())?;

        Ok(Self {
            data: filtered_df,
            schema: self.schema.clone(),
            phantom: PhantomData,
        })
    }

    /// Filter by unique ID
    pub fn filter_by_id(&self, id: &str) -> NeuroDivergentResult<Self> {
        let filtered_df = self.data
            .clone()
            .lazy()
            .filter(col(&self.schema.unique_id_col).eq(lit(id)))
            .collect()
            .map_err(|e| ErrorBuilder::data(format!("Failed to filter by ID: {}", e)).build())?;

        Ok(Self {
            data: filtered_df,
            schema: self.schema.clone(),
            phantom: PhantomData,
        })
    }

    /// Convert to internal dataset format
    pub fn to_dataset(&self) -> NeuroDivergentResult<TimeSeriesDataset<T>> {
        let unique_ids = self.unique_ids()?;
        let mut series_data = HashMap::new();
        let mut static_features = HashMap::new();

        for id in &unique_ids {
            let series_df = self.filter_by_id(id)?;
            let series = self.extract_series_data(&series_df.data, id)?;
            series_data.insert(id.clone(), series);

            // Extract static features if any
            if !self.schema.static_features.is_empty() {
                let static_vals = self.extract_static_features(&series_df.data, id)?;
                static_features.insert(id.clone(), static_vals);
            }
        }

        let metadata = self.compute_metadata(&series_data)?;

        Ok(TimeSeriesDataset {
            unique_ids,
            series_data,
            static_features: if static_features.is_empty() { None } else { Some(static_features) },
            schema: self.schema.clone(),
            metadata,
        })
    }

    /// Extract series data for a single time series
    fn extract_series_data(&self, df: &DataFrame, _id: &str) -> NeuroDivergentResult<SeriesData<T>> {
        // Extract timestamps
        let timestamps: Vec<DateTime<Utc>> = df
            .column(&self.schema.ds_col)
            .map_err(|e| ErrorBuilder::data(format!("Failed to get timestamp column: {}", e)).build())?
            .datetime()
            .map_err(|e| ErrorBuilder::data(format!("Timestamp column is not datetime type: {}", e)).build())?
            .into_iter()
            .filter_map(|opt| opt.map(|ts| Utc.timestamp_nanos(ts * 1000)))
            .collect();

        // Extract target values
        let target_values: Vec<T> = df
            .column(&self.schema.y_col)
            .map_err(|e| ErrorBuilder::data(format!("Failed to get target column: {}", e)).build())?
            .f64()
            .map_err(|e| ErrorBuilder::data(format!("Target column is not numeric: {}", e)).build())?
            .into_iter()
            .filter_map(|opt| opt.and_then(|val| T::from(val)))
            .collect();

        if timestamps.len() != target_values.len() {
            return Err(data_error!(
                "Timestamp and target value lengths don't match"
            ));
        }

        let length = timestamps.len();

        // TODO: Extract exogenous features
        let historical_exogenous = None;
        let future_exogenous = None;

        // TODO: Infer frequency
        let frequency = None;

        Ok(SeriesData {
            timestamps,
            target_values,
            historical_exogenous,
            future_exogenous,
            length,
            frequency,
        })
    }

    /// Extract static features for a series
    fn extract_static_features(&self, df: &DataFrame, _id: &str) -> NeuroDivergentResult<Vec<T>> {
        if self.schema.static_features.is_empty() {
            return Ok(Vec::new());
        }

        let mut features = Vec::new();
        for feature_name in &self.schema.static_features {
            let column = df
                .column(feature_name)
                .map_err(|e| ErrorBuilder::data(format!("Failed to get static feature column '{}': {}", feature_name, e)).build())?;

            // Get first value (static features should be constant within a series)
            let value = match column.dtype() {
                DataType::Float64 | DataType::Float32 => {
                    let float_val = column.f64()
                        .map_err(|e| ErrorBuilder::data(format!("Failed to convert static feature to float: {}", e)).build())?
                        .get(0)
                        .unwrap_or(0.0);
                    T::from(float_val).unwrap_or_else(T::zero)
                }
                DataType::Int64 | DataType::Int32 => {
                    let int_val = column.i64()
                        .map_err(|e| ErrorBuilder::data(format!("Failed to convert static feature to int: {}", e)).build())?
                        .get(0)
                        .unwrap_or(0);
                    T::from(int_val).unwrap_or_else(T::zero)
                }
                _ => T::zero(),
            };
            features.push(value);
        }

        Ok(features)
    }

    /// Compute dataset metadata
    fn compute_metadata(&self, series_data: &HashMap<String, SeriesData<T>>) -> NeuroDivergentResult<DatasetMetadata> {
        let n_series = series_data.len();
        let mut total_observations = 0;
        let mut min_length = usize::MAX;
        let mut max_length = 0;
        let mut min_time = DateTime::<Utc>::MAX_UTC;
        let mut max_time = DateTime::<Utc>::MIN_UTC;

        for series in series_data.values() {
            total_observations += series.length;
            min_length = min_length.min(series.length);
            max_length = max_length.max(series.length);

            if let (Some(first), Some(last)) = (series.timestamps.first(), series.timestamps.last()) {
                min_time = min_time.min(*first);
                max_time = max_time.max(*last);
            }
        }

        let avg_series_length = if n_series > 0 {
            total_observations as f64 / n_series as f64
        } else {
            0.0
        };

        // TODO: Compute missing value statistics
        let missing_stats = MissingValueStats {
            total_missing: 0,
            missing_percentage: 0.0,
            missing_per_column: HashMap::new(),
            series_with_missing: Vec::new(),
        };

        Ok(DatasetMetadata {
            n_series,
            n_observations: total_observations,
            avg_series_length,
            min_series_length: if min_length == usize::MAX { 0 } else { min_length },
            max_series_length: max_length,
            frequency: None, // TODO: Infer common frequency
            time_range: (min_time, max_time),
            missing_stats,
            created_at: Utc::now(),
        })
    }

    /// Validate data integrity
    pub fn validate(&self) -> NeuroDivergentResult<ValidationReport> {
        let mut errors = Vec::new();
        let warnings = Vec::new();
        let mut quality_score = 100.0;

        // Basic schema validation
        if let Err(e) = self.schema.validate_dataframe(&self.data) {
            errors.push(ValidationError {
                code: "SCHEMA_MISMATCH".to_string(),
                message: e.to_string(),
                column: None,
                series_id: None,
                severity: ErrorSeverity::Critical,
            });
            quality_score -= 20.0;
        }

        // Check for empty data
        if self.data.height() == 0 {
            errors.push(ValidationError {
                code: "EMPTY_DATASET".to_string(),
                message: "Dataset contains no rows".to_string(),
                column: None,
                series_id: None,
                severity: ErrorSeverity::Critical,
            });
            quality_score -= 30.0;
        }

        // TODO: Add more validation checks
        // - Check for duplicate timestamps within series
        // - Check for missing values
        // - Check data types
        // - Check time series regularity

        let is_valid = errors.is_empty();

        Ok(ValidationReport {
            is_valid,
            errors,
            warnings,
            quality_score: quality_score.max(0.0),
            validated_at: Utc::now(),
        })
    }

    /// Get number of time series
    pub fn n_series(&self) -> NeuroDivergentResult<usize> {
        Ok(self.unique_ids()?.len())
    }

    /// Get time range
    pub fn time_range(&self) -> NeuroDivergentResult<(DateTime<Utc>, DateTime<Utc>)> {
        // TODO: Fix timestamp extraction with proper polars API
        let min_time = DateTime::<Utc>::MIN_UTC;
        let max_time = DateTime::<Utc>::MAX_UTC;

        Ok((min_time, max_time))
    }

    /// Export to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| ErrorBuilder::data(format!("Failed to create CSV file: {}", e)).build())?;

        CsvWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| ErrorBuilder::data(format!("Failed to write CSV: {}", e)).build())?;

        Ok(())
    }

    /// Export to Parquet
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| ErrorBuilder::data(format!("Failed to create Parquet file: {}", e)).build())?;

        ParquetWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| ErrorBuilder::data(format!("Failed to write Parquet: {}", e)).build())?;

        Ok(())
    }
}

/// Builder for TimeSeriesDataset
pub struct TimeSeriesDatasetBuilder<T: Float> {
    unique_id_col: Option<String>,
    ds_col: Option<String>,
    y_col: Option<String>,
    static_features: Vec<String>,
    historical_exogenous: Vec<String>,
    future_exogenous: Vec<String>,
    phantom: PhantomData<T>,
}

impl<T: Float> TimeSeriesDatasetBuilder<T> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            unique_id_col: None,
            ds_col: None,
            y_col: None,
            static_features: Vec::new(),
            historical_exogenous: Vec::new(),
            future_exogenous: Vec::new(),
            phantom: PhantomData,
        }
    }

    /// Set the unique ID column name
    pub fn with_unique_id_column(mut self, col: impl Into<String>) -> Self {
        self.unique_id_col = Some(col.into());
        self
    }

    /// Set the timestamp column name
    pub fn with_time_column(mut self, col: impl Into<String>) -> Self {
        self.ds_col = Some(col.into());
        self
    }

    /// Set the target column name
    pub fn with_target_column(mut self, col: impl Into<String>) -> Self {
        self.y_col = Some(col.into());
        self
    }

    /// Add static feature columns
    pub fn with_static_features(mut self, features: Vec<String>) -> Self {
        self.static_features = features;
        self
    }

    /// Add historical exogenous feature columns
    pub fn with_historical_exogenous(mut self, features: Vec<String>) -> Self {
        self.historical_exogenous = features;
        self
    }

    /// Add future exogenous feature columns
    pub fn with_future_exogenous(mut self, features: Vec<String>) -> Self {
        self.future_exogenous = features;
        self
    }

    /// Build the schema
    pub fn build(self) -> NeuroDivergentResult<TimeSeriesSchema> {
        let unique_id_col = self.unique_id_col.ok_or_else(|| {
            data_error!("Unique ID column must be specified")
        })?;

        let ds_col = self.ds_col.ok_or_else(|| {
            data_error!("Timestamp column must be specified")
        })?;

        let y_col = self.y_col.ok_or_else(|| {
            data_error!("Target column must be specified")
        })?;

        Ok(TimeSeriesSchema {
            unique_id_col,
            ds_col,
            y_col,
            static_features: self.static_features,
            historical_exogenous: self.historical_exogenous,
            future_exogenous: self.future_exogenous,
            data_types: HashMap::new(),
        })
    }
}

impl<T: Float> Default for TimeSeriesDatasetBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_dataframe() -> DataFrame {
        df! {
            "unique_id" => ["series_1", "series_1", "series_1", "series_2", "series_2"],
            "ds" => [
                NaiveDate::from_ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                NaiveDate::from_ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                NaiveDate::from_ymd_opt(2023, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                NaiveDate::from_ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                NaiveDate::from_ymd_opt(2023, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            ],
            "y" => [1.0, 2.0, 3.0, 4.0, 5.0],
        }.unwrap()
    }

    #[test]
    fn test_schema_creation() {
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y")
            .with_static_features(vec!["static_1".to_string()])
            .with_historical_exogenous(vec!["hist_1".to_string()]);

        assert_eq!(schema.unique_id_col, "unique_id");
        assert_eq!(schema.ds_col, "ds");
        assert_eq!(schema.y_col, "y");
        assert_eq!(schema.static_features, vec!["static_1"]);
        assert_eq!(schema.historical_exogenous, vec!["hist_1"]);
    }

    #[test]
    fn test_schema_validation() {
        let df = create_test_dataframe();
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        
        assert!(schema.validate_dataframe(&df).is_ok());

        let invalid_schema = TimeSeriesSchema::new("nonexistent", "ds", "y");
        assert!(invalid_schema.validate_dataframe(&df).is_err());
    }

    #[test]
    fn test_time_series_dataframe_creation() {
        let df = create_test_dataframe();
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema);
        assert!(ts_df.is_ok());
    }

    #[test]
    fn test_unique_ids_extraction() {
        let df = create_test_dataframe();
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
        
        let unique_ids = ts_df.unique_ids().unwrap();
        assert_eq!(unique_ids.len(), 2);
        assert!(unique_ids.contains(&"series_1".to_string()));
        assert!(unique_ids.contains(&"series_2".to_string()));
    }

    #[test]
    fn test_builder_pattern() {
        let schema = TimeSeriesDatasetBuilder::<f64>::new()
            .with_unique_id_column("unique_id")
            .with_time_column("ds")
            .with_target_column("y")
            .with_static_features(vec!["static_1".to_string()])
            .build();

        assert!(schema.is_ok());
        let schema = schema.unwrap();
        assert_eq!(schema.unique_id_col, "unique_id");
        assert_eq!(schema.static_features, vec!["static_1"]);
    }

    #[test]
    fn test_builder_validation() {
        // Missing required fields should fail
        let result = TimeSeriesDatasetBuilder::<f64>::new()
            .with_unique_id_column("unique_id")
            .build();
        assert!(result.is_err());
    }
}