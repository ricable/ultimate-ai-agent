//! Utility functions and helper modules for the neuro-divergent library

use std::collections::HashMap;
use num_traits::Float;
use chrono::{DateTime, Utc};

use crate::config::Frequency;
use crate::results::TimeSeriesDataFrame;
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Mathematical utility functions
pub mod math {
    use super::*;
    use num_traits::Float;
    
    /// Calculate Mean Absolute Error (MAE)
    pub fn mae<T: Float>(y_true: &[T], y_pred: &[T]) -> NeuroDivergentResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(NeuroDivergentError::math("Arrays must have the same length"));
        }
        
        if y_true.is_empty() {
            return Err(NeuroDivergentError::math("Arrays cannot be empty"));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(true_val, pred_val)| (*true_val - *pred_val).abs())
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(y_true.len()).unwrap_or(T::one()))
    }
    
    /// Calculate Mean Squared Error (MSE)
    pub fn mse<T: Float>(y_true: &[T], y_pred: &[T]) -> NeuroDivergentResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(NeuroDivergentError::math("Arrays must have the same length"));
        }
        
        if y_true.is_empty() {
            return Err(NeuroDivergentError::math("Arrays cannot be empty"));
        }
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(true_val, pred_val)| {
                let diff = *true_val - *pred_val;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(y_true.len()).unwrap_or(T::one()))
    }
    
    /// Calculate Root Mean Squared Error (RMSE)
    pub fn rmse<T: Float>(y_true: &[T], y_pred: &[T]) -> NeuroDivergentResult<T> {
        let mse_val = mse(y_true, y_pred)?;
        Ok(mse_val.sqrt())
    }
    
    /// Calculate Mean Absolute Percentage Error (MAPE)
    pub fn mape<T: Float>(y_true: &[T], y_pred: &[T]) -> NeuroDivergentResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(NeuroDivergentError::math("Arrays must have the same length"));
        }
        
        if y_true.is_empty() {
            return Err(NeuroDivergentError::math("Arrays cannot be empty"));
        }
        
        let hundred = T::from(100.0).unwrap_or(T::one());
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(true_val, pred_val)| {
                if *true_val == T::zero() {
                    T::zero() // Skip division by zero
                } else {
                    ((*true_val - *pred_val) / *true_val).abs() * hundred
                }
            })
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(y_true.len()).unwrap_or(T::one()))
    }
    
    /// Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    pub fn smape<T: Float>(y_true: &[T], y_pred: &[T]) -> NeuroDivergentResult<T> {
        if y_true.len() != y_pred.len() {
            return Err(NeuroDivergentError::math("Arrays must have the same length"));
        }
        
        if y_true.is_empty() {
            return Err(NeuroDivergentError::math("Arrays cannot be empty"));
        }
        
        let two = T::from(2.0).unwrap_or(T::one() + T::one());
        let hundred = T::from(100.0).unwrap_or(T::one());
        
        let sum = y_true.iter()
            .zip(y_pred.iter())
            .map(|(true_val, pred_val)| {
                let numerator = (*true_val - *pred_val).abs();
                let denominator = (true_val.abs() + pred_val.abs()) / two;
                if denominator == T::zero() {
                    T::zero()
                } else {
                    (numerator / denominator) * hundred
                }
            })
            .fold(T::zero(), |acc, x| acc + x);
            
        Ok(sum / T::from(y_true.len()).unwrap_or(T::one()))
    }
    
    /// Calculate quantile of a sorted array
    pub fn quantile<T: Float + PartialOrd>(sorted_data: &[T], q: f64) -> Option<T> {
        if sorted_data.is_empty() || q < 0.0 || q > 1.0 {
            return None;
        }
        
        if q == 0.0 {
            return sorted_data.first().copied();
        } else if q == 1.0 {
            return sorted_data.last().copied();
        }
        
        let n = sorted_data.len() as f64;
        let index = q * (n - 1.0);
        let lower_index = index.floor() as usize;
        let upper_index = (lower_index + 1).min(sorted_data.len() - 1);
        
        if lower_index == upper_index {
            return Some(sorted_data[lower_index]);
        }
        
        let weight = T::from(index - lower_index as f64)?;
        let lower_val = sorted_data[lower_index];
        let upper_val = sorted_data[upper_index];
        
        Some(lower_val + weight * (upper_val - lower_val))
    }
    
    /// Calculate standard deviation
    pub fn std_dev<T: Float>(data: &[T]) -> NeuroDivergentResult<T> {
        if data.len() < 2 {
            return Err(NeuroDivergentError::math("Need at least 2 data points for standard deviation"));
        }
        
        let n = T::from(data.len()).unwrap_or(T::one());
        let mean = data.iter().fold(T::zero(), |acc, &x| acc + x) / n;
        
        let variance = data.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x) / (n - T::one());
            
        Ok(variance.sqrt())
    }
    
    /// Calculate correlation coefficient
    pub fn correlation<T: Float>(x: &[T], y: &[T]) -> NeuroDivergentResult<T> {
        if x.len() != y.len() {
            return Err(NeuroDivergentError::math("Arrays must have the same length"));
        }
        
        if x.len() < 2 {
            return Err(NeuroDivergentError::math("Need at least 2 data points"));
        }
        
        let n = T::from(x.len()).unwrap();
        let mean_x = x.iter().fold(T::zero(), |acc, &val| acc + val) / n;
        let mean_y = y.iter().fold(T::zero(), |acc, &val| acc + val) / n;
        
        let mut sum_xy = T::zero();
        let mut sum_x2 = T::zero();
        let mut sum_y2 = T::zero();
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            sum_xy = sum_xy + dx * dy;
            sum_x2 = sum_x2 + dx * dx;
            sum_y2 = sum_y2 + dy * dy;
        }
        
        let denominator = (sum_x2 * sum_y2).sqrt();
        if denominator == T::zero() {
            Ok(T::zero())
        } else {
            Ok(sum_xy / denominator)
        }
    }
}

/// Time series utility functions
pub mod time {
    use super::*;
    
    /// Generate date range with given frequency
    pub fn date_range(
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        frequency: Frequency,
    ) -> Vec<DateTime<Utc>> {
        let mut dates = Vec::new();
        let mut current = start;
        let duration = frequency.duration();
        
        while current <= end {
            dates.push(current);
            current = current + duration;
        }
        
        dates
    }
    
    /// Infer frequency from timestamps
    pub fn infer_frequency(timestamps: &[DateTime<Utc>]) -> Option<Frequency> {
        if timestamps.len() < 2 {
            return None;
        }
        
        // Calculate differences between consecutive timestamps
        let mut diffs = Vec::new();
        for window in timestamps.windows(2) {
            if let Some(diff) = window[1].signed_duration_since(window[0]).to_std().ok() {
                diffs.push(diff);
            }
        }
        
        if diffs.is_empty() {
            return None;
        }
        
        // Find the most common difference
        let mut diff_counts: HashMap<std::time::Duration, usize> = HashMap::new();
        for diff in diffs {
            *diff_counts.entry(diff).or_insert(0) += 1;
        }
        
        let most_common_diff = diff_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(diff, _)| diff)?;
        
        // Map duration to frequency
        match most_common_diff.as_secs() {
            1 => Some(Frequency::Second),
            60 => Some(Frequency::Minute),
            3600 => Some(Frequency::Hourly),
            86400 => Some(Frequency::Daily),
            604800 => Some(Frequency::Weekly),
            _ => {
                // Try to infer monthly/quarterly/yearly
                let days = most_common_diff.as_secs() / 86400;
                match days {
                    28..=31 => Some(Frequency::Monthly),
                    90..=93 => Some(Frequency::Quarterly),
                    365..=366 => Some(Frequency::Yearly),
                    _ => Some(Frequency::Custom(format!("{}D", days))),
                }
            }
        }
    }
    
    /// Check if time series has regular frequency
    pub fn is_regular_frequency(timestamps: &[DateTime<Utc>], tolerance_pct: f64) -> bool {
        if timestamps.len() < 3 {
            return true; // Too few points to determine irregularity
        }
        
        let mut diffs = Vec::new();
        for window in timestamps.windows(2) {
            if let Ok(diff) = window[1].signed_duration_since(window[0]).to_std() {
                diffs.push(diff.as_secs() as f64);
            }
        }
        
        if diffs.is_empty() {
            return false;
        }
        
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let tolerance = mean_diff * tolerance_pct / 100.0;
        
        diffs.iter().all(|&diff| (diff - mean_diff).abs() <= tolerance)
    }
    
    /// Create lag features for time series
    pub fn create_lags<T: Float>(data: &[T], lags: &[usize]) -> Vec<Vec<Option<T>>> {
        let mut lag_features = Vec::new();
        
        for &lag in lags {
            let mut lag_values = Vec::new();
            for i in 0..data.len() {
                if i >= lag {
                    lag_values.push(Some(data[i - lag]));
                } else {
                    lag_values.push(None);
                }
            }
            lag_features.push(lag_values);
        }
        
        lag_features
    }
    
    /// Create rolling window features
    pub fn rolling_windows<T: Float + Copy>(
        data: &[T], 
        window_size: usize
    ) -> Vec<Vec<T>> {
        let mut windows = Vec::new();
        
        for i in 0..data.len() {
            if i + 1 >= window_size {
                let start = i + 1 - window_size;
                let window = data[start..=i].to_vec();
                windows.push(window);
            }
        }
        
        windows
    }
    
    /// Calculate seasonal naive forecast (using seasonal lag)
    pub fn seasonal_naive<T: Float + Copy>(
        data: &[T],
        seasonal_period: usize,
        horizon: usize,
    ) -> Vec<T> {
        let mut forecasts = Vec::new();
        let data_len = data.len();
        
        for h in 1..=horizon {
            let seasonal_index = (data_len - seasonal_period + h - 1) % seasonal_period;
            let seasonal_lag_index = data_len - seasonal_period + seasonal_index;
            
            if seasonal_lag_index < data_len {
                forecasts.push(data[seasonal_lag_index]);
            } else {
                // Fallback to last observation if seasonal lag not available
                forecasts.push(data[data_len - 1]);
            }
        }
        
        forecasts
    }
}

/// Data preprocessing utilities
pub mod preprocessing {
    use super::*;
    
    /// Remove outliers using IQR method
    pub fn remove_outliers_iqr<T: Float + PartialOrd + Copy>(
        data: &mut Vec<T>,
        multiplier: f64,
    ) -> NeuroDivergentResult<usize> {
        if data.len() < 4 {
            return Ok(0); // Need at least 4 points for quartiles
        }
        
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1 = super::math::quantile(&sorted_data, 0.25)
            .ok_or_else(|| NeuroDivergentError::math("Could not calculate Q1"))?;
        let q3 = super::math::quantile(&sorted_data, 0.75)
            .ok_or_else(|| NeuroDivergentError::math("Could not calculate Q3"))?;
        
        let iqr = q3 - q1;
        let mult = T::from(multiplier).unwrap_or(T::from(1.5).unwrap());
        let lower_bound = q1 - mult * iqr;
        let upper_bound = q3 + mult * iqr;
        
        let original_len = data.len();
        data.retain(|&x| x >= lower_bound && x <= upper_bound);
        
        Ok(original_len - data.len())
    }
    
    /// Fill missing values using forward fill
    pub fn forward_fill<T: Float + Copy>(data: &mut Vec<Option<T>>) {
        let mut last_valid = None;
        
        for value in data.iter_mut() {
            if value.is_some() {
                last_valid = *value;
            } else if let Some(last) = last_valid {
                *value = Some(last);
            }
        }
    }
    
    /// Fill missing values using backward fill
    pub fn backward_fill<T: Float + Copy>(data: &mut Vec<Option<T>>) {
        let mut next_valid = None;
        
        for value in data.iter_mut().rev() {
            if value.is_some() {
                next_valid = *value;
            } else if let Some(next) = next_valid {
                *value = Some(next);
            }
        }
    }
    
    /// Interpolate missing values linearly
    pub fn linear_interpolate<T: Float + Copy>(data: &mut Vec<Option<T>>) {
        let mut changes = Vec::new();
        let mut start_idx = None;
        
        for (i, value) in data.iter().enumerate() {
            if value.is_some() {
                if let Some(start) = start_idx {
                    // Found end of missing sequence, interpolate
                    if let (Some(start_val), Some(end_val)) = (data[start], data[i]) {
                        let steps = i - start - 1;
                        if steps > 0 {
                            let step_size = (end_val - start_val) / T::from(steps + 1).unwrap();
                            for j in 1..=steps {
                                let new_val = start_val + step_size * T::from(j).unwrap();
                                changes.push((start + j, Some(new_val)));
                            }
                        }
                    }
                    start_idx = None;
                }
            } else if start_idx.is_none() && i > 0 && data[i - 1].is_some() {
                start_idx = Some(i - 1);
            }
        }
        
        // Apply all changes
        for (idx, value) in changes {
            data[idx] = value;
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate time series data frame
    pub fn validate_time_series<T: Float>(
        df: &TimeSeriesDataFrame<T>
    ) -> NeuroDivergentResult<ValidationReport> {
        let mut report = ValidationReport::new();
        
        // Check for empty data
        if df.shape().0 == 0 {
            report.add_error("Time series data is empty".to_string());
            return Ok(report);
        }
        
        // Check for required columns
        let columns = df.columns();
        if !columns.contains(&df.schema.unique_id_col.as_str()) {
            report.add_error(format!("Missing unique ID column: {}", df.schema.unique_id_col));
        }
        if !columns.contains(&df.schema.ds_col.as_str()) {
            report.add_error(format!("Missing date/time column: {}", df.schema.ds_col));
        }
        if !columns.contains(&df.schema.y_col.as_str()) {
            report.add_error(format!("Missing target column: {}", df.schema.y_col));
        }
        
        // Check for missing values in key columns
        if let Ok(unique_id_col) = df.data.column(&df.schema.unique_id_col) {
            let null_count = unique_id_col.null_count();
            if null_count > 0 {
                report.add_warning(format!(
                    "Found {} null values in unique ID column", null_count
                ));
            }
        }
        
        if let Ok(ds_col) = df.data.column(&df.schema.ds_col) {
            let null_count = ds_col.null_count();
            if null_count > 0 {
                report.add_warning(format!(
                    "Found {} null values in date/time column", null_count
                ));
            }
        }
        
        if let Ok(y_col) = df.data.column(&df.schema.y_col) {
            let null_count = y_col.null_count();
            if null_count > 0 {
                report.add_warning(format!(
                    "Found {} null values in target column", null_count
                ));
            }
        }
        
        // Check time series length
        if let Ok(unique_ids) = df.unique_ids() {
            let mut length_issues = 0;
            for id in &unique_ids {
                if let Ok(series_data) = df.filter_by_id(id) {
                    if series_data.shape().0 < 10 {
                        length_issues += 1;
                    }
                }
            }
            
            if length_issues > 0 {
                report.add_warning(format!(
                    "{} time series have fewer than 10 observations", length_issues
                ));
            }
        }
        
        Ok(report)
    }
    
    /// Validation report structure
    #[derive(Debug, Clone)]
    pub struct ValidationReport {
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
        pub info: Vec<String>,
    }
    
    impl ValidationReport {
        pub fn new() -> Self {
            Self {
                errors: Vec::new(),
                warnings: Vec::new(),
                info: Vec::new(),
            }
        }
        
        pub fn add_error(&mut self, error: String) {
            self.errors.push(error);
        }
        
        pub fn add_warning(&mut self, warning: String) {
            self.warnings.push(warning);
        }
        
        pub fn add_info(&mut self, info: String) {
            self.info.push(info);
        }
        
        pub fn has_errors(&self) -> bool {
            !self.errors.is_empty()
        }
        
        pub fn has_warnings(&self) -> bool {
            !self.warnings.is_empty()
        }
        
        pub fn is_valid(&self) -> bool {
            !self.has_errors()
        }
    }
}

/// Performance monitoring utilities
pub mod monitoring {
    use std::time::{Duration, Instant};
    
    /// Simple timer for performance monitoring
    #[derive(Debug)]
    pub struct Timer {
        start: Instant,
        label: String,
    }
    
    impl Timer {
        pub fn new(label: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                label: label.into(),
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        pub fn elapsed_ms(&self) -> u128 {
            self.elapsed().as_millis()
        }
        
        pub fn elapsed_secs(&self) -> f64 {
            self.elapsed().as_secs_f64()
        }
        
        pub fn log_elapsed(&self) {
            log::info!("{}: {:.3}s", self.label, self.elapsed_secs());
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            self.log_elapsed();
        }
    }
    
    /// Memory usage tracker
    #[derive(Debug, Default)]
    pub struct MemoryTracker {
        peak_usage: usize,
        current_usage: usize,
    }
    
    impl MemoryTracker {
        pub fn new() -> Self {
            Self::default()
        }
        
        pub fn track_allocation(&mut self, size: usize) {
            self.current_usage += size;
            if self.current_usage > self.peak_usage {
                self.peak_usage = self.current_usage;
            }
        }
        
        pub fn track_deallocation(&mut self, size: usize) {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
        
        pub fn current_usage(&self) -> usize {
            self.current_usage
        }
        
        pub fn peak_usage(&self) -> usize {
            self.peak_usage
        }
        
        pub fn reset(&mut self) {
            self.current_usage = 0;
            self.peak_usage = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mae_calculation() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9];
        
        let mae = math::mae(&y_true, &y_pred).unwrap();
        assert!((mae - 0.1).abs() < 1e-6);
    }
    
    #[test]
    fn test_mse_calculation() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        
        let mse = math::mse(&y_true, &y_pred).unwrap();
        assert_eq!(mse, 0.0);
    }
    
    #[test]
    fn test_quantile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(math::quantile(&data, 0.0), Some(1.0));
        assert_eq!(math::quantile(&data, 0.5), Some(3.0));
        assert_eq!(math::quantile(&data, 1.0), Some(5.0));
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = math::correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation
    }
    
    #[test]
    fn test_create_lags() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lags = vec![1, 2];
        
        let lag_features = time::create_lags(&data, &lags);
        
        assert_eq!(lag_features.len(), 2);
        assert_eq!(lag_features[0][0], None); // First lag-1 value
        assert_eq!(lag_features[0][1], Some(1.0)); // Second lag-1 value
        assert_eq!(lag_features[1][0], None); // First lag-2 value
        assert_eq!(lag_features[1][1], None); // Second lag-2 value
        assert_eq!(lag_features[1][2], Some(1.0)); // Third lag-2 value
    }
    
    #[test]
    fn test_outlier_removal() {
        let mut data = vec![1.0, 2.0, 3.0, 100.0, 4.0, 5.0]; // 100.0 is outlier
        let removed = preprocessing::remove_outliers_iqr(&mut data, 1.5).unwrap();
        
        assert_eq!(removed, 1);
        assert!(!data.contains(&100.0));
    }
    
    #[test]
    fn test_forward_fill() {
        let mut data = vec![Some(1.0), None, None, Some(4.0), None];
        preprocessing::forward_fill(&mut data);
        
        assert_eq!(data, vec![Some(1.0), Some(1.0), Some(1.0), Some(4.0), Some(4.0)]);
    }
    
    #[test]
    fn test_timer() {
        let timer = monitoring::Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }
}