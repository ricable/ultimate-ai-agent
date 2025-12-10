// Feature engineering for RAN time series
// Neurodivergent approach: creating features from multiple perspectives simultaneously

use crate::MultiVariatePoint;
use anyhow::Result;
use std::collections::HashMap;

pub struct FeatureEngine {
    lookback_periods: Vec<usize>,
}

impl FeatureEngine {
    pub fn new(lookback_periods: Vec<usize>) -> Self {
        Self { lookback_periods }
    }

    /// Generate comprehensive features from time series
    pub fn generate_features(&self, data: &[MultiVariatePoint]) -> Result<Vec<MultiVariatePoint>> {
        let mut enriched_data = Vec::new();

        for (idx, point) in data.iter().enumerate() {
            let mut new_features = point.features.clone();

            // Add time-based features
            self.add_temporal_features(point.timestamp, &mut new_features);

            // Add lag features
            for &lag in &self.lookback_periods {
                if idx >= lag {
                    self.add_lag_features(&data[idx - lag], lag, &mut new_features);
                }
            }

            // Add rolling statistics
            if idx >= 24 {
                self.add_rolling_features(&data[idx - 24..idx], &mut new_features);
            }

            // Add rate of change
            if idx > 0 {
                self.add_rate_of_change(&data[idx - 1], point, &mut new_features);
            }

            enriched_data.push(MultiVariatePoint {
                timestamp: point.timestamp,
                features: new_features,
                labels: point.labels.clone(),
            });
        }

        Ok(enriched_data)
    }

    fn add_temporal_features(&self, timestamp: i64, features: &mut HashMap<String, f64>) {
        use chrono::prelude::*;

        let dt = chrono::DateTime::from_timestamp(timestamp, 0)
            .unwrap_or_else(|| Utc::now());

        features.insert("hour_of_day".to_string(), dt.hour() as f64);
        features.insert("day_of_week".to_string(), dt.weekday().num_days_from_monday() as f64);
        features.insert("day_of_month".to_string(), dt.day() as f64);
        features.insert("is_weekend".to_string(), if dt.weekday().num_days_from_monday() >= 5 { 1.0 } else { 0.0 });

        // Cyclical encoding
        let hour_rad = 2.0 * std::f64::consts::PI * dt.hour() as f64 / 24.0;
        features.insert("hour_sin".to_string(), hour_rad.sin());
        features.insert("hour_cos".to_string(), hour_rad.cos());

        let day_rad = 2.0 * std::f64::consts::PI * dt.weekday().num_days_from_monday() as f64 / 7.0;
        features.insert("day_sin".to_string(), day_rad.sin());
        features.insert("day_cos".to_string(), day_rad.cos());
    }

    fn add_lag_features(&self, lag_point: &MultiVariatePoint, lag: usize, features: &mut HashMap<String, f64>) {
        for (key, &value) in &lag_point.features {
            let lag_key = format!("{}_lag_{}", key, lag);
            features.insert(lag_key, value);
        }
    }

    fn add_rolling_features(&self, window_data: &[MultiVariatePoint], features: &mut HashMap<String, f64>) {
        if window_data.is_empty() {
            return;
        }

        // Get all feature names from first point
        let feature_names: Vec<String> = window_data[0].features.keys().cloned().collect();

        for feature_name in feature_names {
            let values: Vec<f64> = window_data
                .iter()
                .filter_map(|p| p.features.get(&feature_name).copied())
                .collect();

            if values.is_empty() {
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            features.insert(format!("{}_rolling_mean", feature_name), mean);
            features.insert(format!("{}_rolling_std", feature_name), std_dev);
            features.insert(format!("{}_rolling_min", feature_name), min);
            features.insert(format!("{}_rolling_max", feature_name), max);
        }
    }

    fn add_rate_of_change(&self, prev_point: &MultiVariatePoint, curr_point: &MultiVariatePoint, features: &mut HashMap<String, f64>) {
        for (key, &curr_value) in &curr_point.features {
            if let Some(&prev_value) = prev_point.features.get(key) {
                let rate = if prev_value != 0.0 {
                    (curr_value - prev_value) / prev_value
                } else {
                    0.0
                };
                features.insert(format!("{}_rate_of_change", key), rate);
            }
        }
    }

    /// Extract domain-specific RAN features
    pub fn extract_ran_features(&self, data: &[MultiVariatePoint]) -> Result<Vec<MultiVariatePoint>> {
        let mut enriched_data = Vec::new();

        for point in data {
            let mut new_features = point.features.clone();

            // Traffic patterns
            if let (Some(&dl_throughput), Some(&ul_throughput)) = (
                point.features.get("dl_throughput_mbps"),
                point.features.get("ul_throughput_mbps"),
            ) {
                new_features.insert("total_throughput".to_string(), dl_throughput + ul_throughput);
                new_features.insert("dl_ul_ratio".to_string(), if ul_throughput > 0.0 {
                    dl_throughput / ul_throughput
                } else {
                    0.0
                });
            }

            // Resource utilization
            if let (Some(&prb_used), Some(&prb_total)) = (
                point.features.get("prb_used"),
                point.features.get("prb_total"),
            ) {
                let utilization = if prb_total > 0.0 {
                    prb_used / prb_total
                } else {
                    0.0
                };
                new_features.insert("prb_utilization".to_string(), utilization);
                new_features.insert("prb_utilization_category".to_string(),
                    self.categorize_utilization(utilization));
            }

            // User connectivity
            if let (Some(&active_users), Some(&max_users)) = (
                point.features.get("active_users"),
                point.features.get("max_users"),
            ) {
                let user_ratio = if max_users > 0.0 {
                    active_users / max_users
                } else {
                    0.0
                };
                new_features.insert("user_load_ratio".to_string(), user_ratio);
            }

            enriched_data.push(MultiVariatePoint {
                timestamp: point.timestamp,
                features: new_features,
                labels: point.labels.clone(),
            });
        }

        Ok(enriched_data)
    }

    fn categorize_utilization(&self, utilization: f64) -> f64 {
        match utilization {
            u if u < 0.3 => 0.0,    // Low
            u if u < 0.6 => 1.0,    // Medium
            u if u < 0.8 => 2.0,    // High
            _ => 3.0,               // Critical
        }
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new(vec![1, 6, 24, 168]) // 1h, 6h, 24h, 1 week
    }
}
