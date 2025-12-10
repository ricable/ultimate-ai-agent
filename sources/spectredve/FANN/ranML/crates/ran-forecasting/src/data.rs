//! Data structures and utilities for RAN time series forecasting

use std::collections::HashMap;
use std::fmt;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ForecastError, ForecastResult};
use ran_core::{KpiType, TimeSeries, TimePoint, PerformanceMetrics};

/// RAN-specific time series data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RanTimeSeries {
    /// Unique identifier for this time series
    pub id: Uuid,
    /// Name/identifier of the time series (e.g., "cell_throughput", "ue_count")
    pub name: String,
    /// KPI type if applicable
    pub kpi_type: Option<KpiType>,
    /// Time series data points
    pub data: TimeSeries<f64>,
    /// Additional features/metadata
    pub features: HashMap<String, Vec<f64>>,
    /// Unit of measurement
    pub unit: Option<String>,
    /// Data source information
    pub source: DataSource,
    /// Data quality metrics
    pub quality: DataQuality,
    /// Timestamp when series was created
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl RanTimeSeries {
    /// Create a new RAN time series
    pub fn new(name: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            kpi_type: None,
            data: TimeSeries::new(),
            features: HashMap::new(),
            unit: None,
            source: DataSource::Unknown,
            quality: DataQuality::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new RAN time series with KPI type
    pub fn with_kpi(name: String, kpi_type: KpiType) -> Self {
        let mut ts = Self::new(name);
        ts.kpi_type = Some(kpi_type);
        ts.unit = Some(kpi_type.unit().to_string());
        ts
    }

    /// Add a measurement at the current time
    pub fn add_measurement(&mut self, value: f64) -> ForecastResult<()> {
        let timestamp = Utc::now();
        self.add_measurement_at(timestamp, value)
    }

    /// Add a measurement at a specific time
    pub fn add_measurement_at(&mut self, timestamp: DateTime<Utc>, value: f64) -> ForecastResult<()> {
        if value.is_nan() || value.is_infinite() {
            return Err(ForecastError::data_error("Invalid measurement value"));
        }

        let time_point = TimePoint::new(timestamp, value);
        self.data.add_point(time_point)
            .map_err(|e| ForecastError::data_error(format!("Failed to add measurement: {}", e)))?;
        
        self.updated_at = Utc::now();
        self.update_quality_metrics();
        
        Ok(())
    }

    /// Add multiple measurements
    pub fn add_measurements(&mut self, measurements: Vec<(DateTime<Utc>, f64)>) -> ForecastResult<()> {
        for (timestamp, value) in measurements {
            self.add_measurement_at(timestamp, value)?;
        }
        Ok(())
    }

    /// Add a feature time series
    pub fn add_feature(&mut self, feature_name: String, values: Vec<f64>) -> ForecastResult<()> {
        if values.len() != self.data.points.len() {
            return Err(ForecastError::dimension_mismatch(self.data.points.len(), values.len()));
        }
        
        self.features.insert(feature_name, values);
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Get time series values
    pub fn values(&self) -> Vec<f64> {
        self.data.points.iter().map(|p| p.value).collect()
    }

    /// Get timestamps
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.data.points.iter().map(|p| p.timestamp).collect()
    }

    /// Get data points
    pub fn points(&self) -> &[TimePoint<f64>] {
        &self.data.points
    }

    /// Get feature values
    pub fn features(&self) -> &HashMap<String, Vec<f64>> {
        &self.features
    }

    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<&Vec<f64>> {
        self.features.get(name)
    }

    /// Get number of data points
    pub fn len(&self) -> usize {
        self.data.points.len()
    }

    /// Check if time series is empty
    pub fn is_empty(&self) -> bool {
        self.data.points.is_empty()
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get time range
    pub fn time_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        if self.is_empty() {
            None
        } else {
            let start = self.data.points.first().unwrap().timestamp;
            let end = self.data.points.last().unwrap().timestamp;
            Some((start, end))
        }
    }

    /// Get sampling interval (average time between points)
    pub fn sampling_interval(&self) -> Option<Duration> {
        if self.len() < 2 {
            return None;
        }

        let total_duration = self.data.points.last().unwrap().timestamp 
            - self.data.points.first().unwrap().timestamp;
        let intervals = self.len() - 1;
        
        Some(total_duration / intervals as i32)
    }

    /// Check for missing values (NaN or infinite)
    pub fn has_missing_values(&self) -> bool {
        self.values().iter().any(|&v| v.is_nan() || v.is_infinite())
    }

    /// Check for outliers using z-score
    pub fn has_outliers(&self, threshold: f64) -> bool {
        let values = self.values();
        if values.len() < 3 {
            return false;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return false;
        }

        values.iter().any(|&x| ((x - mean) / std_dev).abs() > threshold)
    }

    /// Get basic statistics
    pub fn statistics(&self) -> TimeSeriesStatistics {
        let values = self.values();
        if values.is_empty() {
            return TimeSeriesStatistics::default();
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();
        
        let mut sorted_values = values;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        TimeSeriesStatistics {
            count: values.len(),
            mean,
            median,
            std_dev,
            variance,
            min,
            max,
            missing_count: values.iter().filter(|&&v| v.is_nan()).count(),
            outlier_count: self.count_outliers(3.0),
        }
    }

    /// Count outliers using z-score threshold
    fn count_outliers(&self, threshold: f64) -> usize {
        let values = self.values();
        if values.len() < 3 {
            return 0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0;
        }

        values.iter()
            .filter(|&&x| ((x - mean) / std_dev).abs() > threshold)
            .count()
    }

    /// Update data quality metrics
    fn update_quality_metrics(&mut self) {
        let stats = self.statistics();
        
        self.quality.completeness = if stats.count > 0 {
            1.0 - (stats.missing_count as f64 / stats.count as f64)
        } else {
            0.0
        };

        self.quality.consistency = if stats.std_dev == 0.0 {
            1.0
        } else {
            1.0 - (stats.outlier_count as f64 / stats.count as f64)
        };

        // Simple validity check - are values in reasonable range for this KPI type
        self.quality.validity = if let Some(kpi_type) = &self.kpi_type {
            let (min_expected, max_expected) = kpi_type.expected_range();
            let in_range_count = self.values().iter()
                .filter(|&&v| v >= min_expected && v <= max_expected)
                .count();
            in_range_count as f64 / stats.count as f64
        } else {
            1.0 // Unknown KPI type, assume valid
        };

        // Timeliness based on sampling regularity
        self.quality.timeliness = if let Some(interval) = self.sampling_interval() {
            // Check if timestamps are roughly regular
            let expected_interval = interval.num_seconds() as f64;
            let actual_intervals: Vec<f64> = self.data.points.windows(2)
                .map(|w| (w[1].timestamp - w[0].timestamp).num_seconds() as f64)
                .collect();
            
            if actual_intervals.is_empty() {
                1.0
            } else {
                let avg_deviation = actual_intervals.iter()
                    .map(|&i| (i - expected_interval).abs() / expected_interval)
                    .sum::<f64>() / actual_intervals.len() as f64;
                (1.0 - avg_deviation).max(0.0)
            }
        } else {
            1.0
        };

        // Overall quality score
        self.quality.overall_score = (
            self.quality.completeness +
            self.quality.consistency +
            self.quality.validity +
            self.quality.timeliness
        ) / 4.0;
    }

    /// Resample time series to a different frequency
    pub fn resample(&self, interval: Duration) -> ForecastResult<RanTimeSeries> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut resampled = self.clone();
        resampled.id = Uuid::new_v4();
        resampled.data.points.clear();
        resampled.features.clear();

        let start_time = self.data.points.first().unwrap().timestamp;
        let end_time = self.data.points.last().unwrap().timestamp;
        
        let mut current_time = start_time;
        while current_time <= end_time {
            // Find values within the current interval
            let next_time = current_time + interval;
            let values_in_interval: Vec<f64> = self.data.points.iter()
                .filter(|p| p.timestamp >= current_time && p.timestamp < next_time)
                .map(|p| p.value)
                .collect();

            if !values_in_interval.is_empty() {
                // Use mean value for the interval
                let mean_value = values_in_interval.iter().sum::<f64>() / values_in_interval.len() as f64;
                resampled.add_measurement_at(current_time, mean_value)?;
            }

            current_time = next_time;
        }

        Ok(resampled)
    }

    /// Filter time series by time range
    pub fn filter_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ForecastResult<RanTimeSeries> {
        let mut filtered = self.clone();
        filtered.id = Uuid::new_v4();
        filtered.data.points = self.data.points.iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .cloned()
            .collect();

        // Update features to match filtered data
        let filtered_indices: Vec<usize> = self.data.points.iter()
            .enumerate()
            .filter(|(_, p)| p.timestamp >= start && p.timestamp <= end)
            .map(|(i, _)| i)
            .collect();

        for (feature_name, feature_values) in &self.features {
            let filtered_feature: Vec<f64> = filtered_indices.iter()
                .filter_map(|&i| feature_values.get(i).copied())
                .collect();
            filtered.features.insert(feature_name.clone(), filtered_feature);
        }

        filtered.update_quality_metrics();
        Ok(filtered)
    }

    /// Convert to windowed dataset for training
    pub fn to_windows(&self, window_size: usize, horizon: usize, step: usize) -> ForecastResult<WindowedDataset> {
        if self.len() < window_size + horizon {
            return Err(ForecastError::data_error(format!(
                "Insufficient data for windowing: {} points needed, {} available",
                window_size + horizon,
                self.len()
            )));
        }

        let values = self.values();
        let mut windows = Vec::new();
        let mut targets = Vec::new();

        let mut start = 0;
        while start + window_size + horizon <= values.len() {
            let window = values[start..start + window_size].to_vec();
            let target = values[start + window_size..start + window_size + horizon].to_vec();
            
            windows.push(window);
            targets.push(target);
            
            start += step;
        }

        Ok(WindowedDataset {
            windows,
            targets,
            window_size,
            horizon,
            step,
            feature_names: vec![self.name.clone()],
        })
    }
}

/// Forecast horizon specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastHorizon {
    /// Forecast horizon in minutes
    Minutes(usize),
    /// Forecast horizon in hours
    Hours(usize),
    /// Forecast horizon in days
    Days(usize),
}

impl ForecastHorizon {
    /// Convert to number of time steps (assuming 1-minute resolution)
    pub fn to_steps(&self) -> usize {
        match self {
            ForecastHorizon::Minutes(m) => *m,
            ForecastHorizon::Hours(h) => h * 60,
            ForecastHorizon::Days(d) => d * 24 * 60,
        }
    }

    /// Convert to duration
    pub fn to_duration(&self) -> Duration {
        match self {
            ForecastHorizon::Minutes(m) => Duration::minutes(*m as i64),
            ForecastHorizon::Hours(h) => Duration::hours(*h as i64),
            ForecastHorizon::Days(d) => Duration::days(*d as i64),
        }
    }
}

impl fmt::Display for ForecastHorizon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForecastHorizon::Minutes(m) => write!(f, "{}m", m),
            ForecastHorizon::Hours(h) => write!(f, "{}h", h),
            ForecastHorizon::Days(d) => write!(f, "{}d", d),
        }
    }
}

/// Data source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// Unknown source
    Unknown,
    /// Real-time telemetry
    Telemetry {
        node_id: String,
        collector: String,
    },
    /// Historical database
    Database {
        connection: String,
        query: String,
    },
    /// File import
    File {
        path: String,
        format: String,
    },
    /// Simulation
    Simulation {
        model: String,
        parameters: HashMap<String, String>,
    },
    /// External API
    Api {
        endpoint: String,
        version: String,
    },
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    /// Completeness score (0-1)
    pub completeness: f64,
    /// Consistency score (0-1)
    pub consistency: f64,
    /// Validity score (0-1)
    pub validity: f64,
    /// Timeliness score (0-1)
    pub timeliness: f64,
    /// Overall quality score (0-1)
    pub overall_score: f64,
}

impl Default for DataQuality {
    fn default() -> Self {
        Self {
            completeness: 1.0,
            consistency: 1.0,
            validity: 1.0,
            timeliness: 1.0,
            overall_score: 1.0,
        }
    }
}

/// Time series statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeSeriesStatistics {
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of missing values
    pub missing_count: usize,
    /// Number of outliers
    pub outlier_count: usize,
}

/// Collection of RAN time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RanTimeSeriesDataset {
    /// Dataset identifier
    pub id: Uuid,
    /// Dataset name
    pub name: String,
    /// Time series in the dataset
    pub series: HashMap<String, RanTimeSeries>,
    /// Dataset metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl RanTimeSeriesDataset {
    /// Create a new dataset
    pub fn new(name: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            series: HashMap::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a time series to the dataset
    pub fn add_series(&mut self, series: RanTimeSeries) {
        self.series.insert(series.name.clone(), series);
        self.updated_at = Utc::now();
    }

    /// Get a time series by name
    pub fn get_series(&self, name: &str) -> Option<&RanTimeSeries> {
        self.series.get(name)
    }

    /// Get mutable reference to a time series
    pub fn get_series_mut(&mut self, name: &str) -> Option<&mut RanTimeSeries> {
        self.series.get_mut(name)
    }

    /// Remove a time series
    pub fn remove_series(&mut self, name: &str) -> Option<RanTimeSeries> {
        let result = self.series.remove(name);
        if result.is_some() {
            self.updated_at = Utc::now();
        }
        result
    }

    /// Get all series names
    pub fn series_names(&self) -> Vec<&String> {
        self.series.keys().collect()
    }

    /// Get number of series
    pub fn len(&self) -> usize {
        self.series.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// Add metadata
    pub fn add_metadata<K, V>(&mut self, key: K, value: V) 
    where 
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata.insert(key.into(), value.into());
        self.updated_at = Utc::now();
    }

    /// Get common time range across all series
    pub fn common_time_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        if self.series.is_empty() {
            return None;
        }

        let mut earliest: Option<DateTime<Utc>> = None;
        let mut latest: Option<DateTime<Utc>> = None;

        for series in self.series.values() {
            if let Some((start, end)) = series.time_range() {
                earliest = Some(earliest.map_or(start, |e| e.max(start)));
                latest = Some(latest.map_or(end, |l| l.min(end)));
            }
        }

        if let (Some(start), Some(end)) = (earliest, latest) {
            if start <= end {
                Some((start, end))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Align all series to common timestamps
    pub fn align_series(&mut self, interval: Duration) -> ForecastResult<()> {
        if let Some((start, end)) = self.common_time_range() {
            let mut aligned_series = HashMap::new();
            
            for (name, series) in &self.series {
                let filtered = series.filter_by_time_range(start, end)?;
                let resampled = filtered.resample(interval)?;
                aligned_series.insert(name.clone(), resampled);
            }
            
            self.series = aligned_series;
            self.updated_at = Utc::now();
        }
        Ok(())
    }
}

/// Windowed dataset for training
#[derive(Debug, Clone)]
pub struct WindowedDataset {
    /// Input windows
    pub windows: Vec<Vec<f64>>,
    /// Target values
    pub targets: Vec<Vec<f64>>,
    /// Window size
    pub window_size: usize,
    /// Forecast horizon
    pub horizon: usize,
    /// Step size between windows
    pub step: usize,
    /// Feature names
    pub feature_names: Vec<String>,
}

impl WindowedDataset {
    /// Get number of windows
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Split into train and validation sets
    pub fn train_test_split(&self, train_ratio: f64) -> ForecastResult<(WindowedDataset, WindowedDataset)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(ForecastError::invalid_parameter("train_ratio", "must be between 0 and 1"));
        }

        let split_idx = (self.len() as f64 * train_ratio) as usize;
        
        let train_dataset = WindowedDataset {
            windows: self.windows[0..split_idx].to_vec(),
            targets: self.targets[0..split_idx].to_vec(),
            window_size: self.window_size,
            horizon: self.horizon,
            step: self.step,
            feature_names: self.feature_names.clone(),
        };

        let test_dataset = WindowedDataset {
            windows: self.windows[split_idx..].to_vec(),
            targets: self.targets[split_idx..].to_vec(),
            window_size: self.window_size,
            horizon: self.horizon,
            step: self.step,
            feature_names: self.feature_names.clone(),
        };

        Ok((train_dataset, test_dataset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ran_timeseries_creation() {
        let ts = RanTimeSeries::new("test_series".to_string());
        assert_eq!(ts.name, "test_series");
        assert!(ts.is_empty());
        assert_eq!(ts.len(), 0);
    }

    #[test]
    fn test_ran_timeseries_measurements() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        let timestamp = Utc::now();
        
        assert!(ts.add_measurement_at(timestamp, 10.5).is_ok());
        assert_eq!(ts.len(), 1);
        assert!(!ts.is_empty());
        
        let values = ts.values();
        assert_eq!(values[0], 10.5);
        
        let timestamps = ts.timestamps();
        assert_eq!(timestamps[0], timestamp);
    }

    #[test]
    fn test_ran_timeseries_invalid_measurements() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        
        assert!(ts.add_measurement(f64::NAN).is_err());
        assert!(ts.add_measurement(f64::INFINITY).is_err());
        assert!(ts.add_measurement(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn test_ran_timeseries_features() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        ts.add_measurement(10.0).unwrap();
        ts.add_measurement(20.0).unwrap();
        
        let feature_values = vec![1.0, 2.0];
        assert!(ts.add_feature("temperature".to_string(), feature_values.clone()).is_ok());
        
        assert_eq!(ts.get_feature("temperature"), Some(&feature_values));
        
        // Test dimension mismatch
        let wrong_size_features = vec![1.0];
        assert!(ts.add_feature("humidity".to_string(), wrong_size_features).is_err());
    }

    #[test]
    fn test_ran_timeseries_statistics() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        for i in 1..=10 {
            ts.add_measurement(i as f64).unwrap();
        }
        
        let stats = ts.statistics();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.mean, 5.5);
        assert_eq!(stats.missing_count, 0);
    }

    #[test]
    fn test_ran_timeseries_outliers() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        // Add normal values
        for i in 1..=10 {
            ts.add_measurement(i as f64).unwrap();
        }
        // Add outlier
        ts.add_measurement(1000.0).unwrap();
        
        assert!(ts.has_outliers(3.0));
        let stats = ts.statistics();
        assert!(stats.outlier_count > 0);
    }

    #[test]
    fn test_forecast_horizon() {
        let horizon_mins = ForecastHorizon::Minutes(30);
        let horizon_hours = ForecastHorizon::Hours(2);
        let horizon_days = ForecastHorizon::Days(1);
        
        assert_eq!(horizon_mins.to_steps(), 30);
        assert_eq!(horizon_hours.to_steps(), 120);
        assert_eq!(horizon_days.to_steps(), 1440);
        
        assert_eq!(format!("{}", horizon_mins), "30m");
        assert_eq!(format!("{}", horizon_hours), "2h");
        assert_eq!(format!("{}", horizon_days), "1d");
    }

    #[test]
    fn test_ran_timeseries_dataset() {
        let mut dataset = RanTimeSeriesDataset::new("test_dataset".to_string());
        assert!(dataset.is_empty());
        
        let ts1 = RanTimeSeries::new("series1".to_string());
        let ts2 = RanTimeSeries::new("series2".to_string());
        
        dataset.add_series(ts1);
        dataset.add_series(ts2);
        
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        assert!(dataset.get_series("series1").is_some());
        assert!(dataset.get_series("nonexistent").is_none());
        
        let names = dataset.series_names();
        assert!(names.contains(&&"series1".to_string()));
        assert!(names.contains(&&"series2".to_string()));
    }

    #[test]
    fn test_windowed_dataset() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        for i in 1..=20 {
            ts.add_measurement(i as f64).unwrap();
        }
        
        let windowed = ts.to_windows(5, 2, 1).unwrap();
        assert_eq!(windowed.window_size, 5);
        assert_eq!(windowed.horizon, 2);
        assert_eq!(windowed.step, 1);
        assert!(!windowed.is_empty());
        
        // Check first window and target
        assert_eq!(windowed.windows[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(windowed.targets[0], vec![6.0, 7.0]);
    }

    #[test]
    fn test_windowed_dataset_split() {
        let mut ts = RanTimeSeries::new("test_series".to_string());
        for i in 1..=20 {
            ts.add_measurement(i as f64).unwrap();
        }
        
        let windowed = ts.to_windows(5, 2, 1).unwrap();
        let (train, test) = windowed.train_test_split(0.8).unwrap();
        
        assert!(train.len() > 0);
        assert!(test.len() > 0);
        assert_eq!(train.len() + test.len(), windowed.len());
        
        // Test invalid split ratio
        assert!(windowed.train_test_split(0.0).is_err());
        assert!(windowed.train_test_split(1.0).is_err());
    }

    #[test]
    fn test_data_quality() {
        let quality = DataQuality::default();
        assert_eq!(quality.completeness, 1.0);
        assert_eq!(quality.consistency, 1.0);
        assert_eq!(quality.validity, 1.0);
        assert_eq!(quality.timeliness, 1.0);
        assert_eq!(quality.overall_score, 1.0);
    }

    #[test]
    fn test_data_source() {
        let source = DataSource::Telemetry {
            node_id: "cell_001".to_string(),
            collector: "prometheus".to_string(),
        };
        
        match source {
            DataSource::Telemetry { node_id, collector } => {
                assert_eq!(node_id, "cell_001");
                assert_eq!(collector, "prometheus");
            }
            _ => panic!("Wrong data source type"),
        }
    }

    #[test]
    fn test_timeseries_with_kpi() {
        let ts = RanTimeSeries::with_kpi("throughput".to_string(), KpiType::AverageCellThroughput);
        assert_eq!(ts.kpi_type, Some(KpiType::AverageCellThroughput));
        assert_eq!(ts.unit, Some("Mbps".to_string()));
    }
}