//! Result types and data structures for forecasting operations

use std::collections::HashMap;
use std::path::Path;
use std::marker::PhantomData;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use serde::{Serialize, Deserialize};
use num_traits::Float;

use crate::config::Frequency;
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Schema definition for time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesSchema {
    /// Unique identifier column name
    pub unique_id_col: String,
    /// Date/time column name  
    pub ds_col: String,
    /// Target variable column name
    pub y_col: String,
    /// Static feature column names
    pub static_features: Vec<String>,
    /// Exogenous feature column names
    pub exogenous_features: Vec<String>,
}

impl TimeSeriesSchema {
    /// Create new time series schema with required columns
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
            exogenous_features: Vec::new(),
        }
    }
    
    /// Add static features to the schema
    pub fn with_static_features(mut self, features: Vec<String>) -> Self {
        self.static_features = features;
        self
    }
    
    /// Add exogenous features to the schema
    pub fn with_exogenous_features(mut self, features: Vec<String>) -> Self {
        self.exogenous_features = features;
        self
    }
    
    /// Get all column names
    pub fn all_columns(&self) -> Vec<&String> {
        let mut cols = vec![&self.unique_id_col, &self.ds_col, &self.y_col];
        cols.extend(&self.static_features);
        cols.extend(&self.exogenous_features);
        cols
    }
    
    /// Get required column names
    pub fn required_columns(&self) -> Vec<&String> {
        vec![&self.unique_id_col, &self.ds_col, &self.y_col]
    }
    
    /// Validate schema against DataFrame
    pub fn validate_dataframe(&self, df: &DataFrame) -> NeuroDivergentResult<()> {
        let df_columns: Vec<&str> = df.get_column_names();
        
        // Check required columns
        for col in self.required_columns() {
            if !df_columns.contains(&col.as_str()) {
                return Err(NeuroDivergentError::data(
                    format!("Required column '{}' not found in DataFrame", col)
                ));
            }
        }
        
        // Check static features
        for col in &self.static_features {
            if !df_columns.contains(&col.as_str()) {
                return Err(NeuroDivergentError::data(
                    format!("Static feature column '{}' not found in DataFrame", col)
                ));
            }
        }
        
        // Check exogenous features
        for col in &self.exogenous_features {
            if !df_columns.contains(&col.as_str()) {
                return Err(NeuroDivergentError::data(
                    format!("Exogenous feature column '{}' not found in DataFrame", col)
                ));
            }
        }
        
        Ok(())
    }
}

impl Default for TimeSeriesSchema {
    fn default() -> Self {
        Self::new("unique_id", "ds", "y")
    }
}

/// Main data structure for time series data (equivalent to pandas DataFrame)
#[derive(Debug, Clone)]
pub struct TimeSeriesDataFrame<T: Float> {
    /// Underlying Polars DataFrame
    pub data: DataFrame,
    /// Schema defining column meanings
    pub schema: TimeSeriesSchema,
    /// Frequency of the time series
    pub frequency: Option<Frequency>,
    /// Phantom data for type parameter
    phantom: PhantomData<T>,
}

impl<T: Float> TimeSeriesDataFrame<T> {
    /// Create from Polars DataFrame with schema
    pub fn from_polars(
        df: DataFrame, 
        schema: TimeSeriesSchema,
        frequency: Option<Frequency>,
    ) -> NeuroDivergentResult<Self> {
        // Validate schema against DataFrame
        schema.validate_dataframe(&df)?;
        
        Ok(Self {
            data: df,
            schema,
            frequency,
            phantom: PhantomData,
        })
    }
    
    /// Create from CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        schema: TimeSeriesSchema,
        frequency: Option<Frequency>,
    ) -> NeuroDivergentResult<Self> {
        let df = CsvReader::from_path(path.as_ref())
            .map_err(|e| NeuroDivergentError::data(format!("CSV reader error: {}", e)))?
            .finish()
            .map_err(|e| NeuroDivergentError::data(format!("CSV read error: {}", e)))?;
            
        Self::from_polars(df, schema, frequency)
    }
    
    /// Create from Parquet file
    pub fn from_parquet<P: AsRef<Path>>(
        path: P,
        schema: TimeSeriesSchema,
        frequency: Option<Frequency>,
    ) -> NeuroDivergentResult<Self> {
        let df = LazyFrame::scan_parquet(path.as_ref(), ScanArgsParquet::default())
            .map_err(|e| NeuroDivergentError::data(format!("Parquet read error: {}", e)))?
            .collect()
            .map_err(|e| NeuroDivergentError::data(format!("Parquet collect error: {}", e)))?;
            
        Self::from_polars(df, schema, frequency)
    }
    
    /// Get unique time series identifiers
    pub fn unique_ids(&self) -> NeuroDivergentResult<Vec<String>> {
        let ids = self.data
            .column(&self.schema.unique_id_col)
            .map_err(|e| NeuroDivergentError::data(format!("Column error: {}", e)))?
            .unique()
            .map_err(|e| NeuroDivergentError::data(format!("Unique error: {}", e)))?
            .cast(&DataType::Utf8)
            .map_err(|e| NeuroDivergentError::data(format!("Cast error: {}", e)))?
            .utf8()
            .map_err(|_| NeuroDivergentError::data("Failed to convert to string iterator"))?
            .into_iter()
            .map(|s| s.unwrap_or_default().to_string())
            .collect();
            
        Ok(ids)
    }
    
    /// Filter by date range
    pub fn filter_date_range(
        &self, 
        start: DateTime<Utc>, 
        end: DateTime<Utc>
    ) -> NeuroDivergentResult<Self> {
        let filtered = self.data
            .clone()
            .lazy()
            .filter(
                col(&self.schema.ds_col)
                    .gt_eq(lit(start.timestamp_millis()))
                    .and(
                        col(&self.schema.ds_col)
                            .lt_eq(lit(end.timestamp_millis()))
                    )
            )
            .collect()
            .map_err(|e| NeuroDivergentError::data(format!("Date filter error: {}", e)))?;
            
        Self::from_polars(filtered, self.schema.clone(), self.frequency.clone())
    }
    
    /// Filter by unique ID
    pub fn filter_by_id(&self, id: &str) -> NeuroDivergentResult<Self> {
        let filtered = self.data
            .clone()
            .lazy()
            .filter(col(&self.schema.unique_id_col).eq(lit(id)))
            .collect()
            .map_err(|e| NeuroDivergentError::data(format!("ID filter error: {}", e)))?;
            
        Self::from_polars(filtered, self.schema.clone(), self.frequency.clone())
    }
    
    /// Add exogenous variables
    pub fn with_exogenous(
        mut self, 
        exogenous_data: DataFrame,
        join_columns: Vec<String>,
    ) -> NeuroDivergentResult<Self> {
        // Join exogenous data
        let joined = self.data
            .join(
                &exogenous_data,
                join_columns.clone(),
                join_columns,
                JoinArgs::new(JoinType::Left),
            )
            .map_err(|e| NeuroDivergentError::data(format!("Join error: {}", e)))?;
            
        self.data = joined;
        Ok(self)
    }
    
    /// Get number of time series
    pub fn n_series(&self) -> usize {
        self.unique_ids().unwrap_or_default().len()
    }
    
    /// Get time range
    pub fn time_range(&self) -> NeuroDivergentResult<(DateTime<Utc>, DateTime<Utc>)> {
        let ds_col = self.data
            .column(&self.schema.ds_col)
            .map_err(|e| NeuroDivergentError::data(format!("Column error: {}", e)))?;
            
        let min_ts = ds_col
            .min::<i64>()
            .ok_or_else(|| NeuroDivergentError::data("Could not calculate min timestamp".to_string()))?;
            
        let max_ts = ds_col
            .max::<i64>()
            .ok_or_else(|| NeuroDivergentError::data("Could not calculate max timestamp".to_string()))?;
            
        let start = DateTime::from_timestamp_millis(min_ts)
            .ok_or_else(|| NeuroDivergentError::data("Invalid start timestamp"))?;
        let end = DateTime::from_timestamp_millis(max_ts)
            .ok_or_else(|| NeuroDivergentError::data("Invalid end timestamp"))?;
            
        Ok((start, end))
    }
    
    /// Get shape (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        self.data.shape()
    }
    
    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.data.get_column_names()
    }
    
    /// Save to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)?;
        CsvWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| NeuroDivergentError::data(format!("CSV write error: {}", e)))?;
        Ok(())
    }
    
    /// Save to Parquet
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)?;
        ParquetWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| NeuroDivergentError::data(format!("Parquet write error: {}", e)))?;
        Ok(())
    }
}

/// Results from forecasting operations
#[derive(Debug, Clone)]
pub struct ForecastDataFrame<T: Float> {
    /// Underlying forecast data
    pub data: DataFrame,
    /// Model names that generated forecasts
    pub models: Vec<String>,
    /// Forecast horizon length
    pub forecast_horizon: usize,
    /// Confidence levels if prediction intervals included
    pub confidence_levels: Option<Vec<f64>>,
    /// Schema for the forecast data
    pub schema: TimeSeriesSchema,
    /// Phantom data for type parameter
    phantom: PhantomData<T>,
}

impl<T: Float> ForecastDataFrame<T> {
    /// Create new forecast dataframe
    pub fn new(
        data: DataFrame,
        models: Vec<String>,
        forecast_horizon: usize,
        confidence_levels: Option<Vec<f64>>,
        schema: TimeSeriesSchema,
    ) -> Self {
        Self {
            data,
            models,
            forecast_horizon,
            confidence_levels,
            schema,
            phantom: PhantomData,
        }
    }
    
    /// Get forecasts for specific model
    pub fn get_model_forecasts(&self, model_name: &str) -> NeuroDivergentResult<DataFrame> {
        // Filter for specific model columns
        let model_cols: Vec<String> = self.data
            .get_column_names()
            .iter()
            .filter(|col| col.contains(model_name))
            .map(|s| s.to_string())
            .collect();
            
        if model_cols.is_empty() {
            return Err(NeuroDivergentError::prediction(
                format!("No forecasts found for model '{}'", model_name)
            ));
        }
        
        // Include identifier columns
        let mut select_cols = vec![
            self.schema.unique_id_col.clone(),
            self.schema.ds_col.clone(),
        ];
        select_cols.extend(model_cols);
        
        let filtered = self.data
            .clone()
            .lazy()
            .select(select_cols.iter().map(|s| col(s)).collect::<Vec<_>>())
            .collect()
            .map_err(|e| NeuroDivergentError::prediction(format!("Model filter error: {}", e)))?;
            
        Ok(filtered)
    }
    
    /// Get all model forecasts as separate DataFrames
    pub fn get_all_forecasts(&self) -> Vec<(String, DataFrame)> {
        let mut results = Vec::new();
        
        for model in &self.models {
            if let Ok(df) = self.get_model_forecasts(model) {
                results.push((model.clone(), df));
            }
        }
        
        results
    }
    
    /// Convert to point forecasts only (remove prediction intervals)
    pub fn to_point_forecasts(&self) -> DataFrame {
        // Select only point forecast columns (not quantiles/intervals)
        let point_cols: Vec<String> = self.data
            .get_column_names()
            .iter()
            .filter(|col| !col.contains("_q") && !col.contains("_lower") && !col.contains("_upper"))
            .map(|s| s.to_string())
            .collect();
            
        self.data
            .clone()
            .lazy()
            .select(point_cols.iter().map(|s| col(s)).collect::<Vec<_>>())
            .collect()
            .unwrap_or_else(|_| self.data.clone())
    }
    
    /// Get prediction intervals if available
    pub fn get_prediction_intervals(&self, model_name: &str) -> Option<DataFrame> {
        // Look for interval columns for the model
        let interval_cols: Vec<String> = self.data
            .get_column_names()
            .iter()
            .filter(|col| {
                col.contains(model_name) && (col.contains("_lower") || col.contains("_upper") || col.contains("_q"))
            })
            .map(|s| s.to_string())
            .collect();
            
        if interval_cols.is_empty() {
            return None;
        }
        
        // Include identifier columns
        let mut select_cols = vec![
            self.schema.unique_id_col.clone(),
            self.schema.ds_col.clone(),
        ];
        select_cols.extend(interval_cols);
        
        self.data
            .clone()
            .lazy()
            .select(select_cols.iter().map(|s| col(s)).collect::<Vec<_>>())
            .collect()
            .ok()
    }
    
    /// Export to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)?;
        CsvWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| NeuroDivergentError::data(format!("CSV write error: {}", e)))?;
        Ok(())
    }
    
    /// Export to Parquet
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)?;
        ParquetWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| NeuroDivergentError::data(format!("Parquet write error: {}", e)))?;
        Ok(())
    }
    
    /// Get forecast shape (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        self.data.shape()
    }
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationDataFrame<T: Float> {
    /// Cross-validation results data
    pub data: DataFrame,
    /// Cutoff dates for each validation window
    pub cutoffs: Vec<DateTime<Utc>>,
    /// Model names included in CV
    pub models: Vec<String>,
    /// Accuracy metrics computed across windows
    pub metrics: Option<AccuracyMetrics<T>>,
    /// Schema for the CV data
    pub schema: TimeSeriesSchema,
    /// Phantom data for type parameter
    phantom: PhantomData<T>,
}

impl<T: Float> CrossValidationDataFrame<T> {
    /// Create new cross-validation dataframe
    pub fn new(
        data: DataFrame,
        cutoffs: Vec<DateTime<Utc>>,
        models: Vec<String>,
        schema: TimeSeriesSchema,
    ) -> Self {
        Self {
            data,
            cutoffs,
            models,
            metrics: None,
            schema,
            phantom: PhantomData,
        }
    }
    
    /// Set accuracy metrics
    pub fn with_metrics(mut self, metrics: AccuracyMetrics<T>) -> Self {
        self.metrics = Some(metrics);
        self
    }
    
    /// Get results for specific cutoff
    pub fn get_cutoff_results(&self, cutoff: DateTime<Utc>) -> NeuroDivergentResult<DataFrame> {
        let cutoff_ts = cutoff.timestamp_millis();
        
        let filtered = self.data
            .clone()
            .lazy()
            .filter(col("cutoff").eq(lit(cutoff_ts)))
            .collect()
            .map_err(|e| NeuroDivergentError::data(format!("Cutoff filter error: {}", e)))?;
            
        Ok(filtered)
    }
    
    /// Get results for specific model
    pub fn get_model_results(&self, model_name: &str) -> NeuroDivergentResult<DataFrame> {
        // Filter for model-specific columns
        let model_cols: Vec<String> = self.data
            .get_column_names()
            .iter()
            .filter(|col| col.contains(model_name) || col == &&self.schema.unique_id_col || col == &&self.schema.ds_col || **col == "cutoff")
            .map(|s| s.to_string())
            .collect();
            
        let filtered = self.data
            .clone()
            .lazy()
            .select(model_cols.iter().map(|s| col(s)).collect::<Vec<_>>())
            .collect()
            .map_err(|e| NeuroDivergentError::data(format!("Model filter error: {}", e)))?;
            
        Ok(filtered)
    }
    
    /// Calculate summary statistics across CV windows
    pub fn summary_stats(&self) -> NeuroDivergentResult<DataFrame> {
        // Group by model and calculate mean/std of metrics
        let summary = self.data
            .clone()
            .lazy()
            .group_by([col("model")])
            .agg([
                col("mae").mean().alias("mean_mae"),
                col("mae").std(1).alias("std_mae"),
                col("mse").mean().alias("mean_mse"),
                col("mse").std(1).alias("std_mse"),
                col("mape").mean().alias("mean_mape"),
                col("mape").std(1).alias("std_mape"),
            ])
            .collect()
            .map_err(|e| NeuroDivergentError::data(format!("Summary stats error: {}", e)))?;
            
        Ok(summary)
    }
    
    /// Export to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()> {
        let mut file = std::fs::File::create(path)?;
        CsvWriter::new(&mut file)
            .finish(&mut self.data.clone())
            .map_err(|e| NeuroDivergentError::data(format!("CSV write error: {}", e)))?;
        Ok(())
    }
}

/// Accuracy metrics for forecast evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics<T: Float> {
    /// Mean Absolute Error
    pub mae: T,
    /// Mean Squared Error
    pub mse: T,
    /// Root Mean Squared Error
    pub rmse: T,
    /// Mean Absolute Percentage Error
    pub mape: T,
    /// Symmetric Mean Absolute Percentage Error  
    pub smape: T,
    /// Mean Absolute Scaled Error
    pub mase: Option<T>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, T>,
}

impl<T: Float> AccuracyMetrics<T> {
    /// Create new accuracy metrics
    pub fn new(mae: T, mse: T, mape: T, smape: T) -> Self {
        let rmse = mse.sqrt();
        Self {
            mae,
            mse,
            rmse,
            mape,
            smape,
            mase: None,
            custom_metrics: HashMap::new(),
        }
    }
    
    /// Set MASE metric
    pub fn with_mase(mut self, mase: T) -> Self {
        self.mase = Some(mase);
        self
    }
    
    /// Add custom metric
    pub fn with_custom_metric(mut self, name: String, value: T) -> Self {
        self.custom_metrics.insert(name, value);
        self
    }
    
    /// Get all metric names
    pub fn metric_names(&self) -> Vec<String> {
        let mut names = vec!["mae", "mse", "rmse", "mape", "smape"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            
        if self.mase.is_some() {
            names.push("mase".to_string());
        }
        
        names.extend(self.custom_metrics.keys().cloned());
        names
    }
    
    /// Get metric value by name
    pub fn get_metric(&self, name: &str) -> Option<T> {
        match name {
            "mae" => Some(self.mae),
            "mse" => Some(self.mse),
            "rmse" => Some(self.rmse),
            "mape" => Some(self.mape),
            "smape" => Some(self.smape),
            "mase" => self.mase,
            _ => self.custom_metrics.get(name).copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    
    #[test]
    fn test_time_series_schema() {
        let schema = TimeSeriesSchema::new("id", "date", "value")
            .with_static_features(vec!["category".to_string()])
            .with_exogenous_features(vec!["temperature".to_string()]);
            
        assert_eq!(schema.unique_id_col, "id");
        assert_eq!(schema.ds_col, "date");
        assert_eq!(schema.y_col, "value");
        assert_eq!(schema.static_features, vec!["category"]);
        assert_eq!(schema.exogenous_features, vec!["temperature"]);
    }
    
    #[test]
    fn test_accuracy_metrics() {
        let metrics = AccuracyMetrics::new(1.0f32, 2.0, 0.1, 0.15)
            .with_mase(0.8)
            .with_custom_metric("custom".to_string(), 0.5);
            
        assert_eq!(metrics.mae, 1.0);
        assert_eq!(metrics.mse, 2.0);
        assert!((metrics.rmse - 2.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(metrics.mase, Some(0.8));
        assert_eq!(metrics.get_metric("custom"), Some(0.5));
    }
    
    #[test]
    fn test_forecast_dataframe_creation() {
        let data = df! {
            "unique_id" => ["A", "A", "B", "B"],
            "ds" => [1, 2, 1, 2],
            "LSTM" => [10.0, 11.0, 20.0, 21.0],
        }.unwrap();
        
        let schema = TimeSeriesSchema::default();
        let models = vec!["LSTM".to_string()];
        
        let forecast_df = ForecastDataFrame::<f32>::new(
            data,
            models,
            2,
            None,
            schema,
        );
        
        assert_eq!(forecast_df.models.len(), 1);
        assert_eq!(forecast_df.forecast_horizon, 2);
        assert_eq!(forecast_df.shape().0, 4);
    }
}