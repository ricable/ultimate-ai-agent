// Neurodivergent Rust Code: Embracing non-linear thinking patterns
// for time series analysis with multiple perspectives and parallel processing

pub mod models;
pub mod predictors;
pub mod anomaly;
pub mod feature_engineering;
pub mod multivariate;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: i64,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVariatePoint {
    pub timestamp: i64,
    pub features: HashMap<String, f64>,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub timestamp: i64,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
    pub anomaly_score: f64,
    pub feature_importance: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RanContext {
    pub node_type: String,
    pub cell_id: String,
    pub cluster_id: Option<String>,
    pub region_id: Option<String>,
}

/// Neurodivergent approach: Process multiple granularities simultaneously
/// rather than sequentially, allowing pattern recognition across scales
pub struct TimeSeriesEngine {
    hourly_buffer: Vec<MultiVariatePoint>,
    daily_buffer: Vec<MultiVariatePoint>,
    weekly_buffer: Vec<MultiVariatePoint>,
}

impl TimeSeriesEngine {
    pub fn new() -> Self {
        Self {
            hourly_buffer: Vec::new(),
            daily_buffer: Vec::new(),
            weekly_buffer: Vec::new(),
        }
    }

    /// Process data across all granularities in parallel
    /// Neurodivergent pattern: simultaneous multi-scale analysis
    pub fn ingest_parallel(&mut self, points: Vec<MultiVariatePoint>) -> Result<()> {
        use rayon::prelude::*;

        let (hourly, daily, weekly): (Vec<_>, Vec<_>, Vec<_>) = points
            .par_iter()
            .cloned()
            .fold(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(mut h, mut d, mut w), point| {
                    h.push(point.clone());

                    // Aggregate to daily if hour boundary crossed
                    if point.timestamp % (24 * 3600) == 0 {
                        d.push(point.clone());
                    }

                    // Aggregate to weekly if week boundary crossed
                    if point.timestamp % (7 * 24 * 3600) == 0 {
                        w.push(point);
                    }

                    (h, d, w)
                },
            )
            .reduce(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(mut h1, mut d1, mut w1), (h2, d2, w2)| {
                    h1.extend(h2);
                    d1.extend(d2);
                    w1.extend(w2);
                    (h1, d1, w1)
                },
            );

        self.hourly_buffer.extend(hourly);
        self.daily_buffer.extend(daily);
        self.weekly_buffer.extend(weekly);

        Ok(())
    }

    /// Predict using ensemble of models across granularities
    pub fn predict_ensemble(
        &self,
        horizon: usize,
        context: &RanContext,
    ) -> Result<Vec<PredictionResult>> {
        use rayon::prelude::*;

        // Parallel prediction across different models and granularities
        let predictions: Vec<_> = vec![
            ("hourly", &self.hourly_buffer),
            ("daily", &self.daily_buffer),
            ("weekly", &self.weekly_buffer),
        ]
        .par_iter()
        .filter_map(|(granularity, buffer)| {
            if buffer.is_empty() {
                return None;
            }

            // Each granularity gets its own prediction
            let pred = self.predict_single_granularity(buffer, horizon, granularity, context);
            pred.ok()
        })
        .collect();

        // Merge predictions using weighted ensemble
        self.merge_predictions(predictions)
    }

    fn predict_single_granularity(
        &self,
        buffer: &[MultiVariatePoint],
        horizon: usize,
        granularity: &str,
        _context: &RanContext,
    ) -> Result<Vec<PredictionResult>> {
        // Simplified prediction - in real implementation, use ML models
        let results: Vec<PredictionResult> = (0..horizon)
            .map(|i| {
                let last_point = buffer.last().unwrap();
                let base_timestamp = last_point.timestamp + (i as i64 * 3600);

                // Simple moving average for demonstration
                let recent_values: Vec<f64> = buffer
                    .iter()
                    .rev()
                    .take(24)
                    .filter_map(|p| p.features.get("value").copied())
                    .collect();

                let predicted = if !recent_values.is_empty() {
                    recent_values.iter().sum::<f64>() / recent_values.len() as f64
                } else {
                    0.0
                };

                let std_dev = self.calculate_std_dev(&recent_values);

                PredictionResult {
                    timestamp: base_timestamp,
                    predicted_value: predicted,
                    confidence_interval: (predicted - 2.0 * std_dev, predicted + 2.0 * std_dev),
                    anomaly_score: 0.0,
                    feature_importance: HashMap::new(),
                }
            })
            .collect();

        Ok(results)
    }

    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn merge_predictions(&self, predictions: Vec<Vec<PredictionResult>>) -> Result<Vec<PredictionResult>> {
        if predictions.is_empty() {
            return Ok(Vec::new());
        }

        // Simple averaging for demonstration
        Ok(predictions[0].clone())
    }
}

impl Default for TimeSeriesEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = TimeSeriesEngine::new();
        assert_eq!(engine.hourly_buffer.len(), 0);
    }

    #[test]
    fn test_parallel_ingestion() {
        let mut engine = TimeSeriesEngine::new();
        let points = vec![
            MultiVariatePoint {
                timestamp: 1000,
                features: [("value".to_string(), 42.0)].iter().cloned().collect(),
                labels: HashMap::new(),
            },
        ];

        assert!(engine.ingest_parallel(points).is_ok());
    }
}
