//! Ensemble forecasting methods
//!
//! This module provides ensemble methods for combining multiple forecasting
//! models to improve prediction accuracy and robustness.

use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cmp::Ordering;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::models::ModelType;

/// Ensemble forecaster for combining multiple models
pub struct EnsembleForecaster {
    models: Vec<EnsembleModel>,
    ensemble_strategy: EnsembleStrategy,
    weights: Option<Vec<f32>>,
}

/// Individual model in the ensemble
#[derive(Clone, Debug)]
pub struct EnsembleModel {
    pub name: String,
    pub model_type: ModelType,
    pub weight: f32,
    pub performance_metrics: ModelPerformanceMetrics,
}

/// Model performance metrics
#[derive(Clone, Debug)]
pub struct ModelPerformanceMetrics {
    pub mae: f32,
    pub mse: f32,
    pub mape: f32,
    pub smape: f32,
    pub coverage: f32, // For prediction intervals
}

/// Ensemble strategy
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnsembleStrategy {
    SimpleAverage,
    WeightedAverage,
    Median,
    TrimmedMean(f32), // Trim percentage
    Voting,
    Stacking,
    BayesianModelAveraging,
}

/// Ensemble configuration
#[derive(Clone, Debug)]
pub struct EnsembleConfig {
    pub strategy: EnsembleStrategy,
    pub models: Vec<String>,
    pub weights: Option<Vec<f32>>,
    pub meta_learner: Option<String>,
    pub optimization_metric: OptimizationMetric,
}

/// Optimization metric for ensemble weights
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizationMetric {
    MAE,
    MSE,
    MAPE,
    SMAPE,
    CombinedScore,
}

impl EnsembleForecaster {
    /// Create a new ensemble forecaster
    pub fn new(config: EnsembleConfig) -> Result<Self, String> {
        if let Some(ref weights) = config.weights {
            if weights.len() != config.models.len() {
                return Err("Number of weights must match number of models".to_string());
            }

            // Validate weights sum to 1.0 for weighted average
            if config.strategy == EnsembleStrategy::WeightedAverage {
                let weight_sum: f32 = weights.iter().sum();
                if (weight_sum - 1.0).abs() > 1e-6 {
                    return Err("Weights must sum to 1.0 for weighted average".to_string());
                }
            }
        }

        Ok(Self {
            models: Vec::new(),
            ensemble_strategy: config.strategy,
            weights: config.weights,
        })
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: EnsembleModel) {
        self.models.push(model);
    }

    /// Generate ensemble forecast
    pub fn ensemble_predict(&self, predictions: &[Vec<f32>]) -> Result<EnsembleForecast, String> {
        if predictions.is_empty() {
            return Err("No predictions provided".to_string());
        }

        // Validate all predictions have the same length
        let horizon = predictions[0].len();
        if !predictions.iter().all(|p| p.len() == horizon) {
            return Err("All predictions must have the same horizon".to_string());
        }

        let point_forecast = match self.ensemble_strategy {
            EnsembleStrategy::SimpleAverage => self.simple_average(predictions)?,
            EnsembleStrategy::WeightedAverage => self.weighted_average(predictions)?,
            EnsembleStrategy::Median => self.median_ensemble(predictions)?,
            EnsembleStrategy::TrimmedMean(trim_pct) => self.trimmed_mean(predictions, trim_pct)?,
            EnsembleStrategy::Voting => self.voting_ensemble(predictions)?,
            EnsembleStrategy::Stacking => {
                return Err("Stacking requires trained meta-learner".to_string());
            }
            EnsembleStrategy::BayesianModelAveraging => {
                self.bayesian_model_averaging(predictions)?
            }
        };

        // Calculate prediction intervals
        let intervals = self.calculate_prediction_intervals(predictions, &point_forecast);

        // Calculate ensemble metrics
        let metrics = self.calculate_ensemble_metrics(predictions, &point_forecast);

        Ok(EnsembleForecast {
            point_forecast,
            prediction_intervals: intervals,
            ensemble_metrics: metrics,
            models_used: self.models.len(),
            strategy: self.ensemble_strategy,
        })
    }

    /// Simple average of all predictions
    fn simple_average(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for pred in predictions {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value;
            }
        }

        for value in &mut result {
            *value /= predictions.len() as f32;
        }

        Ok(result)
    }

    /// Weighted average of predictions
    fn weighted_average(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| "Weights not provided for weighted average".to_string())?;

        if weights.len() != predictions.len() {
            return Err("Number of weights must match number of predictions".to_string());
        }

        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for (pred, &weight) in predictions.iter().zip(weights.iter()) {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value * weight;
            }
        }

        Ok(result)
    }

    /// Median ensemble
    fn median_ensemble(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            result[i] = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
        }

        Ok(result)
    }

    /// Trimmed mean ensemble
    fn trimmed_mean(
        &self,
        predictions: &[Vec<f32>],
        trim_percent: f32,
    ) -> Result<Vec<f32>, String> {
        if !(0.0..0.5).contains(&trim_percent) {
            return Err("Trim percentage must be between 0 and 0.5".to_string());
        }

        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];
        let trim_count = ((predictions.len() as f32) * trim_percent).floor() as usize;

        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Remove extreme values
            if trim_count > 0 && values.len() > 2 * trim_count {
                values = values[trim_count..values.len() - trim_count].to_vec();
            }

            result[i] = values.iter().sum::<f32>() / values.len() as f32;
        }

        Ok(result)
    }

    /// Voting ensemble (for classification-like problems)
    fn voting_ensemble(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        // For regression, we can use a threshold-based voting
        // This is a simplified implementation
        self.median_ensemble(predictions)
    }

    /// Bayesian model averaging
    fn bayesian_model_averaging(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        // Calculate model weights based on historical performance
        let model_weights = self.calculate_bayesian_weights();

        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];

        for (pred, &weight) in predictions.iter().zip(model_weights.iter()) {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value * weight;
            }
        }

        Ok(result)
    }

    /// Calculate Bayesian weights based on model performance
    fn calculate_bayesian_weights(&self) -> Vec<f32> {
        if self.models.is_empty() {
            return vec![1.0];
        }

        // Use inverse MSE as weight basis
        let mse_values: Vec<f32> = self
            .models
            .iter()
            .map(|m| m.performance_metrics.mse)
            .collect();

        let min_mse = mse_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Calculate weights proportional to inverse MSE
        let raw_weights: Vec<f32> = mse_values
            .iter()
            .map(|&mse| if mse > 0.0 { min_mse / mse } else { 1.0 })
            .collect();

        // Normalize weights
        let weight_sum: f32 = raw_weights.iter().sum();
        raw_weights.iter().map(|&w| w / weight_sum).collect()
    }

    /// Calculate prediction intervals
    fn calculate_prediction_intervals(
        &self,
        predictions: &[Vec<f32>],
        point_forecast: &[f32],
    ) -> PredictionIntervals {
        let horizon = point_forecast.len();
        let mut lower_50 = vec![0.0; horizon];
        let mut upper_50 = vec![0.0; horizon];
        let mut lower_80 = vec![0.0; horizon];
        let mut upper_80 = vec![0.0; horizon];
        let mut lower_95 = vec![0.0; horizon];
        let mut upper_95 = vec![0.0; horizon];

        for i in 0..horizon {
            let values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();

            let mean = point_forecast[i];
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt();

            // Calculate intervals assuming normal distribution
            lower_50[i] = mean - 0.674 * std_dev;
            upper_50[i] = mean + 0.674 * std_dev;

            lower_80[i] = mean - 1.282 * std_dev;
            upper_80[i] = mean + 1.282 * std_dev;

            lower_95[i] = mean - 1.96 * std_dev;
            upper_95[i] = mean + 1.96 * std_dev;
        }

        PredictionIntervals {
            level_50: (lower_50, upper_50),
            level_80: (lower_80, upper_80),
            level_95: (lower_95, upper_95),
        }
    }

    /// Calculate ensemble performance metrics
    fn calculate_ensemble_metrics(
        &self,
        predictions: &[Vec<f32>],
        ensemble_forecast: &[f32],
    ) -> EnsembleMetrics {
        let horizon = ensemble_forecast.len();

        // Calculate diversity metrics
        let mut pairwise_correlations = Vec::new();
        for i in 0..predictions.len() {
            for j in (i + 1)..predictions.len() {
                let corr = self.calculate_correlation(&predictions[i], &predictions[j]);
                pairwise_correlations.push(corr);
            }
        }

        let avg_correlation = if pairwise_correlations.is_empty() {
            0.0
        } else {
            pairwise_correlations.iter().sum::<f32>() / pairwise_correlations.len() as f32
        };

        // Calculate prediction variance
        let mut prediction_variance = 0.0;
        for i in 0..horizon {
            let values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = ensemble_forecast[i];
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            prediction_variance += variance;
        }
        prediction_variance /= horizon as f32;

        EnsembleMetrics {
            diversity_score: 1.0 - avg_correlation,
            average_model_weight: 1.0 / self.models.len() as f32,
            prediction_variance,
            effective_models: self.calculate_effective_models(),
        }
    }

    /// Calculate correlation between two prediction series
    fn calculate_correlation(&self, pred1: &[f32], pred2: &[f32]) -> f32 {
        let n = pred1.len() as f32;
        let mean1 = pred1.iter().sum::<f32>() / n;
        let mean2 = pred2.iter().sum::<f32>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..pred1.len() {
            let diff1 = pred1[i] - mean1;
            let diff2 = pred2[i] - mean2;
            cov += diff1 * diff2;
            var1 += diff1.powi(2);
            var2 += diff2.powi(2);
        }

        if var1 == 0.0 || var2 == 0.0 {
            return 0.0;
        }

        cov / (var1.sqrt() * var2.sqrt())
    }

    /// Calculate effective number of models (based on weight entropy)
    fn calculate_effective_models(&self) -> f32 {
        let weights = match &self.weights {
            Some(w) => w.clone(),
            None => vec![1.0 / self.models.len() as f32; self.models.len()],
        };

        // Calculate entropy-based effective number
        let entropy: f32 = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum();

        entropy.exp()
    }

    /// Optimize ensemble weights using validation data
    pub fn optimize_weights(
        &mut self,
        validation_predictions: &[Vec<f32>],
        validation_actuals: &[f32],
        metric: OptimizationMetric,
    ) -> Result<Vec<f32>, String> {
        if validation_predictions.len() != self.models.len() {
            return Err("Number of predictions must match number of models".to_string());
        }

        // Simple grid search for weights (can be replaced with more sophisticated optimization)
        let mut best_weights = vec![1.0 / self.models.len() as f32; self.models.len()];
        let mut best_score = f32::INFINITY;

        // Generate weight combinations
        let weight_options = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        // This is a simplified version - real implementation would use optimization algorithms
        for i in 0..self.models.len() {
            let mut test_weights = best_weights.clone();
            for &w in &weight_options {
                test_weights[i] = w;

                // Normalize weights
                let sum: f32 = test_weights.iter().sum();
                if sum > 0.0 {
                    for weight in &mut test_weights {
                        *weight /= sum;
                    }

                    // Calculate ensemble forecast with test weights
                    self.weights = Some(test_weights.clone());
                    if let Ok(forecast) = self.weighted_average(validation_predictions) {
                        let score = self.calculate_optimization_score(
                            &forecast,
                            validation_actuals,
                            metric,
                        );

                        if score < best_score {
                            best_score = score;
                            best_weights = test_weights.clone();
                        }
                    }
                }
            }
        }

        self.weights = Some(best_weights.clone());
        Ok(best_weights)
    }

    /// Calculate optimization score
    fn calculate_optimization_score(
        &self,
        forecast: &[f32],
        actuals: &[f32],
        metric: OptimizationMetric,
    ) -> f32 {
        match metric {
            OptimizationMetric::MAE => calculate_mae(forecast, actuals),
            OptimizationMetric::MSE => calculate_mse(forecast, actuals),
            OptimizationMetric::MAPE => calculate_mape(forecast, actuals),
            OptimizationMetric::SMAPE => calculate_smape(forecast, actuals),
            OptimizationMetric::CombinedScore => {
                // Weighted combination of metrics
                let mae = calculate_mae(forecast, actuals);
                let mape = calculate_mape(forecast, actuals);
                0.5 * mae + 0.5 * mape
            }
        }
    }
}

/// Ensemble forecast result
#[derive(Clone, Debug)]
pub struct EnsembleForecast {
    pub point_forecast: Vec<f32>,
    pub prediction_intervals: PredictionIntervals,
    pub ensemble_metrics: EnsembleMetrics,
    pub models_used: usize,
    pub strategy: EnsembleStrategy,
}

/// Prediction intervals at different confidence levels
#[derive(Clone, Debug)]
pub struct PredictionIntervals {
    pub level_50: (Vec<f32>, Vec<f32>), // (lower, upper)
    pub level_80: (Vec<f32>, Vec<f32>),
    pub level_95: (Vec<f32>, Vec<f32>),
}

/// Ensemble performance metrics
#[derive(Clone, Debug)]
pub struct EnsembleMetrics {
    pub diversity_score: f32, // 0-1, higher is more diverse
    pub average_model_weight: f32,
    pub prediction_variance: f32,
    pub effective_models: f32, // Effective number of models based on weights
}

/// Calculate Mean Absolute Error
fn calculate_mae(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .map(|(&f, &a)| (f - a).abs())
        .sum::<f32>()
        / forecast.len() as f32
}

/// Calculate Mean Squared Error
fn calculate_mse(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .map(|(&f, &a)| (f - a).powi(2))
        .sum::<f32>()
        / forecast.len() as f32
}

/// Calculate Mean Absolute Percentage Error
fn calculate_mape(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .filter(|(_, &a)| a != 0.0)
        .map(|(&f, &a)| ((f - a) / a).abs())
        .sum::<f32>()
        / forecast.len() as f32
        * 100.0
}

/// Calculate Symmetric Mean Absolute Percentage Error
fn calculate_smape(forecast: &[f32], actuals: &[f32]) -> f32 {
    forecast
        .iter()
        .zip(actuals.iter())
        .map(|(&f, &a)| {
            let denominator = (f.abs() + a.abs()) / 2.0;
            if denominator == 0.0 {
                0.0
            } else {
                (f - a).abs() / denominator
            }
        })
        .sum::<f32>()
        / forecast.len() as f32
        * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_average() {
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::SimpleAverage,
            models: vec!["model1".to_string(), "model2".to_string()],
            weights: None,
            meta_learner: None,
            optimization_metric: OptimizationMetric::MAE,
        };

        let forecaster = EnsembleForecaster::new(config).unwrap();

        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let result = forecaster.ensemble_predict(&predictions).unwrap();

        assert_eq!(result.point_forecast, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_weighted_average() {
        let config = EnsembleConfig {
            strategy: EnsembleStrategy::WeightedAverage,
            models: vec!["model1".to_string(), "model2".to_string()],
            weights: Some(vec![0.3, 0.7]),
            meta_learner: None,
            optimization_metric: OptimizationMetric::MAE,
        };

        let forecaster = EnsembleForecaster::new(config).unwrap();

        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let result = forecaster.ensemble_predict(&predictions).unwrap();

        assert_eq!(result.point_forecast[0], 1.0 * 0.3 + 2.0 * 0.7);
        assert_eq!(result.point_forecast[1], 2.0 * 0.3 + 3.0 * 0.7);
    }
}
