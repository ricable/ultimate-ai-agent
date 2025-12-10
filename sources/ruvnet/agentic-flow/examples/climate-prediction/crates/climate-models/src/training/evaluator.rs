//! Model evaluation and performance metrics

use std::sync::Arc;
use tracing::info;

use crate::{
    traits::{ClimateInput, ClimatePrediction, ModelMetrics, PredictionModel},
    Result,
};

/// Evaluation result for a model
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationResult {
    pub model_name: String,
    pub metrics: ModelMetrics,
    pub per_sample_errors: Vec<f32>,
    pub predictions: Vec<ClimatePrediction>,
}

/// Model evaluator for comprehensive testing
pub struct ModelEvaluator {
    test_inputs: Vec<ClimateInput>,
    test_targets: Vec<ClimatePrediction>,
}

impl ModelEvaluator {
    /// Create a new evaluator with test data
    pub fn new(test_inputs: Vec<ClimateInput>, test_targets: Vec<ClimatePrediction>) -> Self {
        Self {
            test_inputs,
            test_targets,
        }
    }

    /// Evaluate a single model
    pub async fn evaluate(&self, model: Arc<dyn PredictionModel>) -> Result<EvaluationResult> {
        info!("Evaluating model: {}", model.name());

        let start_time = std::time::Instant::now();

        // Get predictions
        let predictions = model.predict_batch(&self.test_inputs).await?;

        let inference_time = start_time.elapsed();

        // Calculate metrics
        let pred_temps: Vec<f32> = predictions.iter().map(|p| p.temperature).collect();
        let truth_temps: Vec<f32> = self.test_targets.iter().map(|p| p.temperature).collect();

        let mut metrics = ModelMetrics::calculate(&pred_temps, &truth_temps)?;
        metrics.inference_time_ms = inference_time.as_millis() as f64 / self.test_inputs.len() as f64;

        // Calculate per-sample errors
        let per_sample_errors: Vec<f32> = pred_temps
            .iter()
            .zip(truth_temps.iter())
            .map(|(pred, truth)| (pred - truth).abs())
            .collect();

        Ok(EvaluationResult {
            model_name: model.name().to_string(),
            metrics,
            per_sample_errors,
            predictions,
        })
    }

    /// Evaluate multiple models and compare
    pub async fn evaluate_multiple(
        &self,
        models: Vec<Arc<dyn PredictionModel>>,
    ) -> Result<Vec<EvaluationResult>> {
        let mut results = Vec::with_capacity(models.len());

        for model in models {
            let result = self.evaluate(model).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Generate a comparison report
    pub fn comparison_report(&self, results: &[EvaluationResult]) -> String {
        let mut report = String::new();

        report.push_str("=== Model Comparison Report ===\n\n");

        // Sort by RMSE (best first)
        let mut sorted_results = results.to_vec();
        sorted_results.sort_by(|a, b| {
            a.metrics.rmse.partial_cmp(&b.metrics.rmse)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Table header
        report.push_str(&format!(
            "{:<20} {:<10} {:<10} {:<10} {:<15}\n",
            "Model", "RMSE", "MAE", "R²", "Inference (ms)"
        ));
        report.push_str(&"-".repeat(75));
        report.push('\n');

        // Table rows
        for result in sorted_results {
            report.push_str(&format!(
                "{:<20} {:<10.4} {:<10.4} {:<10.4} {:<15.2}\n",
                result.model_name,
                result.metrics.rmse,
                result.metrics.mae,
                result.metrics.r2_score,
                result.metrics.inference_time_ms
            ));
        }

        report.push('\n');
        report.push_str("=== Best Model ===\n");
        if let Some(best) = sorted_results.first() {
            report.push_str(&format!("Model: {}\n", best.model_name));
            report.push_str(&format!("RMSE: {:.4}\n", best.metrics.rmse));
            report.push_str(&format!("MAE: {:.4}\n", best.metrics.mae));
            report.push_str(&format!("R² Score: {:.4}\n", best.metrics.r2_score));
        }

        report
    }

    /// Calculate statistical significance between two models
    pub fn statistical_comparison(
        &self,
        result_a: &EvaluationResult,
        result_b: &EvaluationResult,
    ) -> StatisticalComparison {
        // Paired t-test on errors
        let errors_a = &result_a.per_sample_errors;
        let errors_b = &result_b.per_sample_errors;

        let n = errors_a.len() as f64;
        let mean_diff: f64 = errors_a.iter()
            .zip(errors_b.iter())
            .map(|(a, b)| (*a - *b) as f64)
            .sum::<f64>() / n;

        let std_diff: f64 = errors_a.iter()
            .zip(errors_b.iter())
            .map(|(a, b)| {
                let diff = (*a - *b) as f64 - mean_diff;
                diff * diff
            })
            .sum::<f64>() / (n - 1.0);

        let std_diff = std_diff.sqrt();
        let t_statistic = mean_diff / (std_diff / n.sqrt());

        StatisticalComparison {
            model_a: result_a.model_name.clone(),
            model_b: result_b.model_name.clone(),
            mean_difference: mean_diff,
            t_statistic,
            significant: t_statistic.abs() > 1.96, // 95% confidence
        }
    }
}

/// Statistical comparison between two models
#[derive(Debug, Clone, serde::Serialize)]
pub struct StatisticalComparison {
    pub model_a: String,
    pub model_b: String,
    pub mean_difference: f64,
    pub t_statistic: f64,
    pub significant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Vec<ClimateInput>, Vec<ClimatePrediction>) {
        let inputs = vec![
            ClimateInput::new(vec![15.0, 1013.0, 60.0]),
            ClimateInput::new(vec![16.0, 1012.0, 65.0]),
            ClimateInput::new(vec![14.0, 1014.0, 55.0]),
        ];

        let targets = vec![
            ClimatePrediction::temperature(15.5, 0.9, 1),
            ClimatePrediction::temperature(16.5, 0.9, 1),
            ClimatePrediction::temperature(14.5, 0.9, 1),
        ];

        (inputs, targets)
    }

    #[test]
    fn test_evaluator_creation() {
        let (inputs, targets) = create_test_data();
        let evaluator = ModelEvaluator::new(inputs, targets);
        assert_eq!(evaluator.test_inputs.len(), 3);
    }

    #[test]
    fn test_comparison_report() {
        let (inputs, targets) = create_test_data();
        let evaluator = ModelEvaluator::new(inputs, targets);

        let results = vec![
            EvaluationResult {
                model_name: "Model A".to_string(),
                metrics: ModelMetrics {
                    rmse: 0.5,
                    mae: 0.4,
                    r2_score: 0.95,
                    mape: 2.0,
                    inference_time_ms: 10.0,
                    extra: std::collections::HashMap::new(),
                },
                per_sample_errors: vec![0.5, 0.3, 0.6],
                predictions: vec![],
            },
        ];

        let report = evaluator.comparison_report(&results);
        assert!(report.contains("Model A"));
        assert!(report.contains("RMSE"));
    }
}
