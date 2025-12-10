// Anomaly detection for RAN metrics
// Neurodivergent approach: pattern recognition through multiple lenses

use crate::MultiVariatePoint;
use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct AnomalyDetector {
    threshold_multiplier: f64,
    baseline_stats: HashMap<String, FeatureStats>,
}

#[derive(Debug, Clone)]
struct FeatureStats {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    percentile_95: f64,
}

impl AnomalyDetector {
    pub fn new(threshold_multiplier: f64) -> Self {
        Self {
            threshold_multiplier,
            baseline_stats: HashMap::new(),
        }
    }

    pub fn fit(&mut self, data: &[MultiVariatePoint]) -> Result<()> {
        if data.is_empty() {
            anyhow::bail!("No data to fit");
        }

        // Extract all feature names
        let feature_names: Vec<String> = data[0].features.keys().cloned().collect();

        for feature_name in feature_names {
            let mut values: Vec<f64> = data
                .iter()
                .filter_map(|p| p.features.get(&feature_name).copied())
                .collect();

            if values.is_empty() {
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            let min = values.first().copied().unwrap_or(0.0);
            let max = values.last().copied().unwrap_or(0.0);
            let percentile_95_idx = (values.len() as f64 * 0.95) as usize;
            let percentile_95 = values.get(percentile_95_idx).copied().unwrap_or(max);

            self.baseline_stats.insert(
                feature_name,
                FeatureStats {
                    mean,
                    std_dev,
                    min,
                    max,
                    percentile_95,
                },
            );
        }

        Ok(())
    }

    pub fn detect(&self, point: &MultiVariatePoint) -> Result<AnomalyResult> {
        let mut anomalies = Vec::new();
        let mut total_score = 0.0;

        for (feature_name, &value) in &point.features {
            if let Some(stats) = self.baseline_stats.get(feature_name) {
                let z_score = (value - stats.mean).abs() / stats.std_dev;

                if z_score > self.threshold_multiplier {
                    anomalies.push(FeatureAnomaly {
                        feature_name: feature_name.clone(),
                        value,
                        expected_mean: stats.mean,
                        z_score,
                        deviation_type: if value > stats.mean {
                            DeviationType::High
                        } else {
                            DeviationType::Low
                        },
                    });
                }

                total_score += z_score;
            }
        }

        let avg_score = if !point.features.is_empty() {
            total_score / point.features.len() as f64
        } else {
            0.0
        };

        Ok(AnomalyResult {
            timestamp: point.timestamp,
            is_anomaly: !anomalies.is_empty(),
            anomaly_score: avg_score,
            feature_anomalies: anomalies,
        })
    }

    /// Detect anomalies using Isolation Forest approach
    pub fn detect_isolation_forest(&self, point: &MultiVariatePoint) -> Result<f64> {
        // Simplified isolation forest score
        // In production, use proper isolation forest implementation
        let mut score = 0.0;
        let mut count = 0;

        for (feature_name, &value) in &point.features {
            if let Some(stats) = self.baseline_stats.get(feature_name) {
                let normalized = (value - stats.min) / (stats.max - stats.min + 1e-10);
                let isolation_score = (normalized - 0.5).abs() * 2.0;
                score += isolation_score;
                count += 1;
            }
        }

        Ok(if count > 0 { score / count as f64 } else { 0.0 })
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub timestamp: i64,
    pub is_anomaly: bool,
    pub anomaly_score: f64,
    pub feature_anomalies: Vec<FeatureAnomaly>,
}

#[derive(Debug, Clone)]
pub struct FeatureAnomaly {
    pub feature_name: String,
    pub value: f64,
    pub expected_mean: f64,
    pub z_score: f64,
    pub deviation_type: DeviationType,
}

#[derive(Debug, Clone)]
pub enum DeviationType {
    High,
    Low,
}

/// Multi-strategy anomaly detection
/// Neurodivergent pattern: simultaneous evaluation through different detection strategies
pub struct MultiStrategyDetector {
    statistical_detector: AnomalyDetector,
    threshold: f64,
}

impl MultiStrategyDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            statistical_detector: AnomalyDetector::new(3.0),
            threshold,
        }
    }

    pub fn fit(&mut self, data: &[MultiVariatePoint]) -> Result<()> {
        self.statistical_detector.fit(data)
    }

    pub fn detect_comprehensive(&self, point: &MultiVariatePoint) -> Result<AnomalyResult> {
        // Statistical detection
        let stat_result = self.statistical_detector.detect(point)?;

        // Isolation forest score
        let iso_score = self.statistical_detector.detect_isolation_forest(point)?;

        // Combined score
        let combined_score = (stat_result.anomaly_score + iso_score) / 2.0;

        Ok(AnomalyResult {
            timestamp: point.timestamp,
            is_anomaly: combined_score > self.threshold,
            anomaly_score: combined_score,
            feature_anomalies: stat_result.feature_anomalies,
        })
    }
}
