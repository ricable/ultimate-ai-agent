//! Forecast accuracy metrics and evaluation

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{
    error::{ForecastError, ForecastResult},
    AccuracyMetric,
};

/// Forecast accuracy metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForecastAccuracy {
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Root Mean Square Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
    /// Mean Absolute Scaled Error
    pub mase: f64,
    /// Number of data points evaluated
    pub n_samples: usize,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl ForecastAccuracy {
    /// Calculate accuracy metrics from actual and predicted values
    pub fn calculate(
        actual: &[f64],
        predicted: &[f64],
        metrics: &[AccuracyMetric],
    ) -> ForecastResult<Self> {
        if actual.len() != predicted.len() {
            return Err(ForecastError::dimension_mismatch(actual.len(), predicted.len()));
        }

        if actual.is_empty() {
            return Err(ForecastError::data_error("No data points to evaluate"));
        }

        let mut accuracy = Self {
            n_samples: actual.len(),
            ..Default::default()
        };

        for &metric in metrics {
            match metric {
                AccuracyMetric::MAPE => {
                    accuracy.mape = Self::calculate_mape(actual, predicted)?;
                },
                AccuracyMetric::RMSE => {
                    accuracy.rmse = Self::calculate_rmse(actual, predicted)?;
                },
                AccuracyMetric::MAE => {
                    accuracy.mae = Self::calculate_mae(actual, predicted)?;
                },
                AccuracyMetric::MSE => {
                    accuracy.mse = Self::calculate_mse(actual, predicted)?;
                },
                AccuracyMetric::SMAPE => {
                    accuracy.smape = Self::calculate_smape(actual, predicted)?;
                },
                AccuracyMetric::MASE => {
                    accuracy.mase = Self::calculate_mase(actual, predicted)?;
                },
            }
        }

        Ok(accuracy)
    }

    /// Mean Absolute Percentage Error
    fn calculate_mape(actual: &[f64], predicted: &[f64]) -> ForecastResult<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            if a != 0.0 {
                sum += ((a - p) / a).abs() * 100.0;
                count += 1;
            }
        }

        if count == 0 {
            Err(ForecastError::computation_error("Cannot calculate MAPE: all actual values are zero"))
        } else {
            Ok(sum / count as f64)
        }
    }

    /// Root Mean Square Error
    fn calculate_rmse(actual: &[f64], predicted: &[f64]) -> ForecastResult<f64> {
        let mse = Self::calculate_mse(actual, predicted)?;
        Ok(mse.sqrt())
    }

    /// Mean Absolute Error
    fn calculate_mae(actual: &[f64], predicted: &[f64]) -> ForecastResult<f64> {
        let sum: f64 = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).abs())
            .sum();
        Ok(sum / actual.len() as f64)
    }

    /// Mean Squared Error
    fn calculate_mse(actual: &[f64], predicted: &[f64]) -> ForecastResult<f64> {
        let sum: f64 = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum();
        Ok(sum / actual.len() as f64)
    }

    /// Symmetric Mean Absolute Percentage Error
    fn calculate_smape(actual: &[f64], predicted: &[f64]) -> ForecastResult<f64> {
        let sum: f64 = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| {
                let denominator = (a.abs() + p.abs()) / 2.0;
                if denominator != 0.0 {
                    (a - p).abs() / denominator * 100.0
                } else {
                    0.0
                }
            })
            .sum();
        Ok(sum / actual.len() as f64)
    }

    /// Mean Absolute Scaled Error
    fn calculate_mase(actual: &[f64], predicted: &[f64]) -> ForecastResult<f64> {
        // Calculate naive forecast error (seasonal naive with period 1)
        if actual.len() < 2 {
            return Ok(0.0);
        }

        let naive_error: f64 = actual.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / (actual.len() - 1) as f64;

        if naive_error == 0.0 {
            return Ok(0.0);
        }

        let mae = Self::calculate_mae(actual, predicted)?;
        Ok(mae / naive_error)
    }

    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    /// Get metric by name
    pub fn get_metric(&self, metric: AccuracyMetric) -> f64 {
        match metric {
            AccuracyMetric::MAPE => self.mape,
            AccuracyMetric::RMSE => self.rmse,
            AccuracyMetric::MAE => self.mae,
            AccuracyMetric::MSE => self.mse,
            AccuracyMetric::SMAPE => self.smape,
            AccuracyMetric::MASE => self.mase,
        }
    }

    /// Check if accuracy is acceptable based on thresholds
    pub fn is_acceptable(&self, thresholds: &AccuracyThresholds) -> bool {
        self.mape <= thresholds.max_mape &&
        self.rmse <= thresholds.max_rmse &&
        self.mae <= thresholds.max_mae
    }

    /// Get overall score (lower is better)
    pub fn overall_score(&self) -> f64 {
        // Weighted combination of normalized metrics
        let weights = [0.3, 0.3, 0.2, 0.2]; // MAPE, RMSE, MAE, SMAPE
        let normalized_metrics = [
            self.mape / 100.0,                    // Normalize MAPE to [0,1]
            self.rmse / (self.rmse + 1.0),        // Normalize RMSE
            self.mae / (self.mae + 1.0),          // Normalize MAE
            self.smape / 100.0,                   // Normalize SMAPE to [0,1]
        ];

        weights.iter()
            .zip(normalized_metrics.iter())
            .map(|(&w, &m)| w * m)
            .sum()
    }

    /// Generate accuracy report
    pub fn report(&self) -> String {
        format!(
            "Forecast Accuracy Report:\n\
             - MAPE: {:.2}%\n\
             - RMSE: {:.4}\n\
             - MAE: {:.4}\n\
             - MSE: {:.4}\n\
             - SMAPE: {:.2}%\n\
             - MASE: {:.4}\n\
             - Samples: {}\n\
             - Overall Score: {:.4}",
            self.mape, self.rmse, self.mae, self.mse, 
            self.smape, self.mase, self.n_samples, self.overall_score()
        )
    }
}

/// Accuracy thresholds for model validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyThresholds {
    /// Maximum acceptable MAPE (%)
    pub max_mape: f64,
    /// Maximum acceptable RMSE
    pub max_rmse: f64,
    /// Maximum acceptable MAE
    pub max_mae: f64,
    /// Maximum acceptable SMAPE (%)
    pub max_smape: f64,
}

impl Default for AccuracyThresholds {
    fn default() -> Self {
        Self {
            max_mape: 15.0,   // 15% MAPE
            max_rmse: 1.0,    // Depends on data scale
            max_mae: 0.5,     // Depends on data scale
            max_smape: 20.0,  // 20% SMAPE
        }
    }
}

/// RAN-specific metrics
pub struct RanMetrics;

impl RanMetrics {
    /// Calculate coverage accuracy (for coverage prediction models)
    pub fn coverage_accuracy(actual_coverage: &[f64], predicted_coverage: &[f64]) -> ForecastResult<f64> {
        if actual_coverage.len() != predicted_coverage.len() {
            return Err(ForecastError::dimension_mismatch(actual_coverage.len(), predicted_coverage.len()));
        }

        let accuracy: f64 = actual_coverage.iter()
            .zip(predicted_coverage.iter())
            .map(|(&actual, &predicted)| {
                let error = (actual - predicted).abs();
                let max_possible_error = actual.max(predicted).max(100.0); // Coverage is typically 0-100%
                1.0 - (error / max_possible_error)
            })
            .sum::<f64>() / actual_coverage.len() as f64;

        Ok(accuracy * 100.0) // Return as percentage
    }

    /// Calculate handover success rate accuracy
    pub fn handover_accuracy(actual_rate: &[f64], predicted_rate: &[f64]) -> ForecastResult<f64> {
        ForecastAccuracy::calculate_mape(actual_rate, predicted_rate)
    }

    /// Calculate throughput prediction accuracy with RAN-specific considerations
    pub fn throughput_accuracy(actual: &[f64], predicted: &[f64]) -> ForecastResult<ThroughputAccuracy> {
        let base_accuracy = ForecastAccuracy::calculate(
            actual, 
            predicted, 
            &[AccuracyMetric::MAPE, AccuracyMetric::RMSE, AccuracyMetric::MAE]
        )?;

        // Calculate peak hour accuracy (assuming last 25% of data represents peak hours)
        let peak_start = (actual.len() * 3) / 4;
        let peak_actual = &actual[peak_start..];
        let peak_predicted = &predicted[peak_start..];
        
        let peak_accuracy = if !peak_actual.is_empty() {
            ForecastAccuracy::calculate_mape(peak_actual, peak_predicted).unwrap_or(0.0)
        } else {
            0.0
        };

        // Calculate underestimation bias (negative bias means underestimation)
        let bias: f64 = predicted.iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a) / a.max(1.0) * 100.0)
            .sum::<f64>() / actual.len() as f64;

        Ok(ThroughputAccuracy {
            overall: base_accuracy,
            peak_hour_mape: peak_accuracy,
            bias_percentage: bias,
            underestimation_rate: predicted.iter()
                .zip(actual.iter())
                .filter(|(&p, &a)| p < a)
                .count() as f64 / actual.len() as f64 * 100.0,
        })
    }

    /// Calculate latency prediction accuracy
    pub fn latency_accuracy(actual: &[f64], predicted: &[f64]) -> ForecastResult<LatencyAccuracy> {
        let base_accuracy = ForecastAccuracy::calculate(
            actual, 
            predicted, 
            &[AccuracyMetric::MAPE, AccuracyMetric::RMSE, AccuracyMetric::MAE]
        )?;

        // Calculate percentage of predictions within acceptable latency bounds
        let acceptable_threshold = 10.0; // 10ms threshold for acceptable latency
        let acceptable_predictions = predicted.iter()
            .filter(|&&p| p <= acceptable_threshold)
            .count() as f64 / predicted.len() as f64 * 100.0;

        // Calculate SLA compliance accuracy (assuming SLA is 5ms)
        let sla_threshold = 5.0;
        let sla_compliance_actual = actual.iter().filter(|&&a| a <= sla_threshold).count() as f64 / actual.len() as f64;
        let sla_compliance_predicted = predicted.iter().filter(|&&p| p <= sla_threshold).count() as f64 / predicted.len() as f64;
        let sla_accuracy = 100.0 - ((sla_compliance_actual - sla_compliance_predicted).abs() * 100.0);

        Ok(LatencyAccuracy {
            overall: base_accuracy,
            acceptable_prediction_rate: acceptable_predictions,
            sla_compliance_accuracy: sla_accuracy,
        })
    }
}

/// Throughput-specific accuracy metrics
#[derive(Debug, Clone)]
pub struct ThroughputAccuracy {
    /// Overall forecast accuracy
    pub overall: ForecastAccuracy,
    /// Accuracy during peak hours (MAPE)
    pub peak_hour_mape: f64,
    /// Prediction bias (positive = overestimation, negative = underestimation)
    pub bias_percentage: f64,
    /// Rate of underestimation (%)
    pub underestimation_rate: f64,
}

/// Latency-specific accuracy metrics
#[derive(Debug, Clone)]
pub struct LatencyAccuracy {
    /// Overall forecast accuracy
    pub overall: ForecastAccuracy,
    /// Percentage of predictions within acceptable bounds
    pub acceptable_prediction_rate: f64,
    /// SLA compliance prediction accuracy
    pub sla_compliance_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_accuracy_calculation() {
        let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let predicted = vec![9.0, 22.0, 28.0, 42.0, 48.0];
        let metrics = vec![AccuracyMetric::MAPE, AccuracyMetric::RMSE, AccuracyMetric::MAE];

        let accuracy = ForecastAccuracy::calculate(&actual, &predicted, &metrics).unwrap();

        assert!(accuracy.mape > 0.0);
        assert!(accuracy.rmse > 0.0);
        assert!(accuracy.mae > 0.0);
        assert_eq!(accuracy.n_samples, 5);
    }

    #[test]
    fn test_mape_calculation() {
        let actual = vec![10.0, 20.0, 30.0];
        let predicted = vec![9.0, 22.0, 27.0];

        let mape = ForecastAccuracy::calculate_mape(&actual, &predicted).unwrap();
        
        // Expected MAPE: ((1/10 + 2/20 + 3/30) * 100) / 3 = (0.1 + 0.1 + 0.1) * 100 / 3 = 10%
        assert!((mape - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_rmse_calculation() {
        let actual = vec![10.0, 20.0, 30.0];
        let predicted = vec![12.0, 18.0, 32.0];

        let rmse = ForecastAccuracy::calculate_rmse(&actual, &predicted).unwrap();
        
        // Expected RMSE: sqrt((4 + 4 + 4) / 3) = sqrt(4) = 2.0
        assert!((rmse - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_mae_calculation() {
        let actual = vec![10.0, 20.0, 30.0];
        let predicted = vec![12.0, 18.0, 32.0];

        let mae = ForecastAccuracy::calculate_mae(&actual, &predicted).unwrap();
        
        // Expected MAE: (2 + 2 + 2) / 3 = 2.0
        assert!((mae - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_dimension_mismatch() {
        let actual = vec![10.0, 20.0];
        let predicted = vec![9.0, 22.0, 28.0];
        let metrics = vec![AccuracyMetric::MAPE];

        let result = ForecastAccuracy::calculate(&actual, &predicted, &metrics);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_actual_values_mape() {
        let actual = vec![0.0, 0.0, 0.0];
        let predicted = vec![1.0, 2.0, 3.0];

        let result = ForecastAccuracy::calculate_mape(&actual, &predicted);
        assert!(result.is_err());
    }

    #[test]
    fn test_accuracy_thresholds() {
        let accuracy = ForecastAccuracy {
            mape: 10.0,
            rmse: 0.5,
            mae: 0.3,
            ..Default::default()
        };

        let thresholds = AccuracyThresholds::default();
        assert!(accuracy.is_acceptable(&thresholds));

        let strict_thresholds = AccuracyThresholds {
            max_mape: 5.0,
            max_rmse: 0.2,
            max_mae: 0.1,
            max_smape: 10.0,
        };
        assert!(!accuracy.is_acceptable(&strict_thresholds));
    }

    #[test]
    fn test_overall_score() {
        let accuracy = ForecastAccuracy {
            mape: 5.0,
            rmse: 0.1,
            mae: 0.05,
            smape: 6.0,
            ..Default::default()
        };

        let score = accuracy.overall_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_ran_metrics_coverage() {
        let actual = vec![95.0, 98.0, 92.0, 96.0];
        let predicted = vec![93.0, 97.0, 94.0, 95.0];

        let accuracy = RanMetrics::coverage_accuracy(&actual, &predicted).unwrap();
        assert!(accuracy > 90.0); // Should be quite accurate
    }

    #[test]
    fn test_throughput_accuracy() {
        let actual = vec![100.0, 120.0, 80.0, 150.0, 110.0];
        let predicted = vec![95.0, 125.0, 85.0, 145.0, 105.0];

        let accuracy = RanMetrics::throughput_accuracy(&actual, &predicted).unwrap();
        
        assert!(accuracy.overall.mape > 0.0);
        assert!(accuracy.bias_percentage.abs() < 50.0); // Reasonable bias
        assert!(accuracy.underestimation_rate >= 0.0 && accuracy.underestimation_rate <= 100.0);
    }
}