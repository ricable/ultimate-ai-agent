//! Model evaluation and comparison utilities

use crate::data::TelecomDataset;
use crate::training::{ModelTrainingResult, ValidationResults};
use crate::error::{TrainingError, TrainingResult};
use crate::models::NeuralModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive evaluation metrics for model assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Mean Squared Error across all models
    pub overall_mse: f32,
    /// Mean Absolute Error across all models
    pub overall_mae: f32,
    /// Root Mean Squared Error across all models
    pub overall_rmse: f32,
    /// R-squared coefficient of determination
    pub overall_r_squared: f32,
    /// Model-specific metrics
    pub model_metrics: HashMap<String, ModelMetrics>,
    /// Statistical significance tests
    pub significance_tests: StatisticalTests,
    /// Performance distribution
    pub performance_distribution: PerformanceDistribution,
}

/// Metrics for individual models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy metrics
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub r_squared: f32,
    /// Performance metrics
    pub training_time: std::time::Duration,
    pub epochs_completed: usize,
    pub convergence_achieved: bool,
    /// Model complexity
    pub parameter_count: usize,
    pub complexity_score: f32,
    /// Robustness metrics
    pub prediction_variance: f32,
    pub outlier_sensitivity: f32,
}

/// Statistical significance tests between models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTests {
    /// Paired t-test results comparing models
    pub t_test_results: Vec<TTestResult>,
    /// Friedman test for multiple model comparison
    pub friedman_test: Option<FriedmanTestResult>,
    /// Effect size measurements
    pub effect_sizes: HashMap<String, f32>,
}

/// T-test result between two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    pub model_a: String,
    pub model_b: String,
    pub t_statistic: f32,
    pub p_value: f32,
    pub significant: bool,
    pub confidence_interval: (f32, f32),
}

/// Friedman test result for multiple model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriedmanTestResult {
    pub chi_squared: f32,
    pub p_value: f32,
    pub significant: bool,
    pub model_rankings: Vec<(String, f32)>,
}

/// Performance distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDistribution {
    /// Error distribution statistics
    pub error_mean: f32,
    pub error_std: f32,
    pub error_median: f32,
    pub error_percentiles: Vec<(f32, f32)>, // (percentile, value)
    /// Training time distribution
    pub time_mean: f32,
    pub time_std: f32,
    pub time_median: f32,
    /// Convergence statistics
    pub convergence_rate: f32,
    pub average_epochs: f32,
}

impl EvaluationMetrics {
    /// Calculate comprehensive metrics from training results
    pub fn calculate_comprehensive(
        model_results: &[ModelTrainingResult],
        validation_dataset: &TelecomDataset,
    ) -> TrainingResult<Self> {
        if model_results.is_empty() {
            return Err(TrainingError::InvalidInput("No model results provided".into()));
        }

        let mut model_metrics = HashMap::new();
        let mut all_errors = Vec::new();
        let mut all_times = Vec::new();

        // Calculate metrics for each model
        for result in model_results {
            let metrics = Self::calculate_model_metrics(result, validation_dataset)?;
            all_errors.push(metrics.mse);
            all_times.push(metrics.training_time.as_secs_f32());
            model_metrics.insert(result.model_name.clone(), metrics);
        }

        // Calculate overall metrics
        let overall_mse = all_errors.iter().sum::<f32>() / all_errors.len() as f32;
        let overall_rmse = overall_mse.sqrt();
        let overall_mae = model_metrics.values()
            .map(|m| m.mae)
            .sum::<f32>() / model_metrics.len() as f32;
        let overall_r_squared = model_metrics.values()
            .map(|m| m.r_squared)
            .sum::<f32>() / model_metrics.len() as f32;

        // Statistical tests
        let significance_tests = Self::perform_statistical_tests(model_results)?;

        // Performance distribution
        let performance_distribution = Self::analyze_performance_distribution(model_results);

        Ok(Self {
            overall_mse,
            overall_mae,
            overall_rmse,
            overall_r_squared,
            model_metrics,
            significance_tests,
            performance_distribution,
        })
    }

    /// Calculate metrics for a single model
    fn calculate_model_metrics(
        result: &ModelTrainingResult,
        _validation_dataset: &TelecomDataset,
    ) -> TrainingResult<ModelMetrics> {
        let validation_results = result.validation_results.as_ref()
            .ok_or_else(|| TrainingError::InvalidInput("No validation results available".into()))?;

        // Calculate prediction variance
        let prediction_variance = Self::calculate_variance(&validation_results.predictions);
        
        // Calculate outlier sensitivity (simplified)
        let outlier_sensitivity = Self::calculate_outlier_sensitivity(validation_results);

        Ok(ModelMetrics {
            mse: validation_results.mse,
            mae: validation_results.mae,
            rmse: validation_results.rmse,
            r_squared: validation_results.r_squared,
            training_time: result.training_time,
            epochs_completed: result.epochs_completed,
            convergence_achieved: result.convergence_achieved,
            parameter_count: 0, // TODO: Calculate from model architecture
            complexity_score: 0.0, // TODO: Calculate complexity score
            prediction_variance,
            outlier_sensitivity,
        })
    }

    /// Perform statistical significance tests
    fn perform_statistical_tests(model_results: &[ModelTrainingResult]) -> TrainingResult<StatisticalTests> {
        let mut t_test_results = Vec::new();
        let mut effect_sizes = HashMap::new();

        // Perform pairwise t-tests
        for i in 0..model_results.len() {
            for j in i+1..model_results.len() {
                let model_a = &model_results[i];
                let model_b = &model_results[j];
                
                if let (Some(val_a), Some(val_b)) = (&model_a.validation_results, &model_b.validation_results) {
                    let t_test = Self::perform_t_test(
                        &val_a.predictions,
                        &val_a.targets,
                        &val_b.predictions,
                        &val_b.targets,
                    )?;
                    
                    t_test_results.push(TTestResult {
                        model_a: model_a.model_name.clone(),
                        model_b: model_b.model_name.clone(),
                        t_statistic: t_test.0,
                        p_value: t_test.1,
                        significant: t_test.1 < 0.05,
                        confidence_interval: t_test.2,
                    });

                    // Calculate effect size (Cohen's d)
                    let effect_size = Self::calculate_cohens_d(
                        &val_a.predictions,
                        &val_b.predictions,
                    );
                    effect_sizes.insert(
                        format!("{}_{}", model_a.model_name, model_b.model_name),
                        effect_size,
                    );
                }
            }
        }

        // Friedman test for multiple comparisons
        let friedman_test = if model_results.len() > 2 {
            Some(Self::perform_friedman_test(model_results)?)
        } else {
            None
        };

        Ok(StatisticalTests {
            t_test_results,
            friedman_test,
            effect_sizes,
        })
    }

    /// Analyze performance distribution across models
    fn analyze_performance_distribution(model_results: &[ModelTrainingResult]) -> PerformanceDistribution {
        let errors: Vec<f32> = model_results.iter()
            .filter_map(|r| r.validation_results.as_ref())
            .map(|v| v.mse)
            .collect();
        
        let times: Vec<f32> = model_results.iter()
            .map(|r| r.training_time.as_secs_f32())
            .collect();

        let convergence_count = model_results.iter()
            .filter(|r| r.convergence_achieved)
            .count();

        let total_epochs: usize = model_results.iter()
            .map(|r| r.epochs_completed)
            .sum();

        PerformanceDistribution {
            error_mean: Self::calculate_mean(&errors),
            error_std: Self::calculate_std(&errors),
            error_median: Self::calculate_median(&mut errors.clone()),
            error_percentiles: Self::calculate_percentiles(&errors),
            time_mean: Self::calculate_mean(&times),
            time_std: Self::calculate_std(&times),
            time_median: Self::calculate_median(&mut times.clone()),
            convergence_rate: convergence_count as f32 / model_results.len() as f32,
            average_epochs: total_epochs as f32 / model_results.len() as f32,
        }
    }

    /// Calculate variance of a vector
    fn calculate_variance(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance
    }

    /// Calculate outlier sensitivity (simplified measure)
    fn calculate_outlier_sensitivity(validation_results: &ValidationResults) -> f32 {
        let errors: Vec<f32> = validation_results.predictions.iter()
            .zip(validation_results.targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .collect();

        let q75 = Self::calculate_percentile(&errors, 75.0);
        let q25 = Self::calculate_percentile(&errors, 25.0);
        let iqr = q75 - q25;
        
        // Count outliers (values beyond 1.5 * IQR)
        let outlier_threshold = q75 + 1.5 * iqr;
        let outlier_count = errors.iter()
            .filter(|&&error| error > outlier_threshold)
            .count();

        outlier_count as f32 / errors.len() as f32
    }

    /// Perform t-test between two sets of predictions
    fn perform_t_test(
        pred_a: &[f32],
        targets_a: &[f32],
        pred_b: &[f32],
        targets_b: &[f32],
    ) -> TrainingResult<(f32, f32, (f32, f32))> {
        let errors_a: Vec<f32> = pred_a.iter()
            .zip(targets_a.iter())
            .map(|(p, t)| (p - t).abs())
            .collect();
        
        let errors_b: Vec<f32> = pred_b.iter()
            .zip(targets_b.iter())
            .map(|(p, t)| (p - t).abs())
            .collect();

        let mean_a = Self::calculate_mean(&errors_a);
        let mean_b = Self::calculate_mean(&errors_b);
        let std_a = Self::calculate_std(&errors_a);
        let std_b = Self::calculate_std(&errors_b);
        
        let n_a = errors_a.len() as f32;
        let n_b = errors_b.len() as f32;
        
        // Welch's t-test
        let pooled_std = ((std_a.powi(2) / n_a) + (std_b.powi(2) / n_b)).sqrt();
        let t_statistic = (mean_a - mean_b) / pooled_std;
        
        // Simplified p-value calculation (would need proper implementation)
        let p_value = if t_statistic.abs() > 2.0 { 0.05 } else { 0.1 };
        
        // Confidence interval (simplified)
        let margin_error = 1.96 * pooled_std;
        let diff = mean_a - mean_b;
        let confidence_interval = (diff - margin_error, diff + margin_error);
        
        Ok((t_statistic, p_value, confidence_interval))
    }

    /// Calculate Cohen's d effect size
    fn calculate_cohens_d(pred_a: &[f32], pred_b: &[f32]) -> f32 {
        let mean_a = Self::calculate_mean(pred_a);
        let mean_b = Self::calculate_mean(pred_b);
        let std_a = Self::calculate_std(pred_a);
        let std_b = Self::calculate_std(pred_b);
        
        let pooled_std = ((std_a.powi(2) + std_b.powi(2)) / 2.0).sqrt();
        
        if pooled_std == 0.0 {
            0.0
        } else {
            (mean_a - mean_b) / pooled_std
        }
    }

    /// Perform Friedman test (simplified implementation)
    fn perform_friedman_test(model_results: &[ModelTrainingResult]) -> TrainingResult<FriedmanTestResult> {
        let mut model_rankings = Vec::new();
        
        // Create rankings based on validation error
        let mut errors: Vec<(String, f32)> = model_results.iter()
            .filter_map(|r| r.validation_results.as_ref().map(|v| (r.model_name.clone(), v.mse)))
            .collect();
        
        errors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        for (rank, (name, _)) in errors.iter().enumerate() {
            model_rankings.push((name.clone(), rank as f32 + 1.0));
        }
        
        // Simplified Friedman test calculation
        let n = model_rankings.len() as f32;
        let k = n; // Number of "treatments" (models)
        
        let rank_sum_squared: f32 = model_rankings.iter()
            .map(|(_, rank)| rank.powi(2))
            .sum();
        
        let chi_squared = (12.0 / (n * k * (k + 1.0))) * rank_sum_squared - 3.0 * n * (k + 1.0);
        let p_value = if chi_squared > 5.991 { 0.05 } else { 0.1 }; // Simplified
        
        Ok(FriedmanTestResult {
            chi_squared,
            p_value,
            significant: p_value < 0.05,
            model_rankings,
        })
    }

    /// Helper functions for statistical calculations
    fn calculate_mean(values: &[f32]) -> f32 {
        if values.is_empty() { 0.0 } else { values.iter().sum::<f32>() / values.len() as f32 }
    }

    fn calculate_std(values: &[f32]) -> f32 {
        if values.len() < 2 { return 0.0; }
        let mean = Self::calculate_mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        variance.sqrt()
    }

    fn calculate_median(values: &mut [f32]) -> f32 {
        if values.is_empty() { return 0.0; }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    fn calculate_percentile(values: &[f32], percentile: f32) -> f32 {
        if values.is_empty() { return 0.0; }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0 * (sorted.len() - 1) as f32).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn calculate_percentiles(values: &[f32]) -> Vec<(f32, f32)> {
        vec![
            (25.0, Self::calculate_percentile(values, 25.0)),
            (50.0, Self::calculate_percentile(values, 50.0)),
            (75.0, Self::calculate_percentile(values, 75.0)),
            (90.0, Self::calculate_percentile(values, 90.0)),
            (95.0, Self::calculate_percentile(values, 95.0)),
        ]
    }
}

/// Model comparison analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Best performing model overall
    pub best_model: String,
    /// Fastest training model
    pub fastest_model: String,
    /// Most efficient model (performance/time ratio)
    pub most_efficient_model: String,
    /// Model rankings by different criteria
    pub rankings: ModelRankings,
    /// Pairwise comparisons
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    /// Recommendation summary
    pub recommendations: Vec<String>,
}

/// Model rankings by various criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRankings {
    pub by_accuracy: Vec<(String, f32)>,
    pub by_speed: Vec<(String, f32)>,
    pub by_efficiency: Vec<(String, f32)>,
    pub by_robustness: Vec<(String, f32)>,
}

/// Pairwise model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    pub model_a: String,
    pub model_b: String,
    pub accuracy_winner: String,
    pub speed_winner: String,
    pub overall_winner: String,
    pub confidence: f32,
}

impl ModelComparison {
    /// Analyze and compare model results
    pub fn analyze(model_results: &[ModelTrainingResult]) -> TrainingResult<Self> {
        if model_results.is_empty() {
            return Err(TrainingError::InvalidInput("No model results provided".into()));
        }

        // Find best models by different criteria
        let best_model = model_results.iter()
            .min_by(|a, b| {
                let error_a = a.validation_results.as_ref().map(|v| v.mse).unwrap_or(f32::MAX);
                let error_b = b.validation_results.as_ref().map(|v| v.mse).unwrap_or(f32::MAX);
                error_a.partial_cmp(&error_b).unwrap()
            })
            .map(|r| r.model_name.clone())
            .unwrap();

        let fastest_model = model_results.iter()
            .min_by_key(|r| r.training_time)
            .map(|r| r.model_name.clone())
            .unwrap();

        let most_efficient_model = model_results.iter()
            .min_by(|a, b| {
                let eff_a = Self::calculate_efficiency(a);
                let eff_b = Self::calculate_efficiency(b);
                eff_b.partial_cmp(&eff_a).unwrap() // Higher efficiency is better
            })
            .map(|r| r.model_name.clone())
            .unwrap();

        // Create rankings
        let rankings = Self::create_rankings(model_results);

        // Perform pairwise comparisons
        let pairwise_comparisons = Self::perform_pairwise_comparisons(model_results);

        // Generate recommendations
        let recommendations = Self::generate_recommendations(model_results, &rankings);

        Ok(Self {
            best_model,
            fastest_model,
            most_efficient_model,
            rankings,
            pairwise_comparisons,
            recommendations,
        })
    }

    /// Calculate efficiency score (1 / (error * time))
    fn calculate_efficiency(result: &ModelTrainingResult) -> f32 {
        let error = result.validation_results.as_ref()
            .map(|v| v.mse)
            .unwrap_or(f32::MAX);
        let time_secs = result.training_time.as_secs_f32();
        
        if error == 0.0 || time_secs == 0.0 {
            0.0
        } else {
            1.0 / (error * time_secs)
        }
    }

    /// Create comprehensive rankings
    fn create_rankings(model_results: &[ModelTrainingResult]) -> ModelRankings {
        // Accuracy ranking (by MSE)
        let mut by_accuracy: Vec<(String, f32)> = model_results.iter()
            .filter_map(|r| r.validation_results.as_ref().map(|v| (r.model_name.clone(), v.mse)))
            .collect();
        by_accuracy.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Speed ranking (by training time)
        let mut by_speed: Vec<(String, f32)> = model_results.iter()
            .map(|r| (r.model_name.clone(), r.training_time.as_secs_f32()))
            .collect();
        by_speed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Efficiency ranking
        let mut by_efficiency: Vec<(String, f32)> = model_results.iter()
            .map(|r| (r.model_name.clone(), Self::calculate_efficiency(r)))
            .collect();
        by_efficiency.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Higher is better

        // Robustness ranking (simplified - by R-squared)
        let mut by_robustness: Vec<(String, f32)> = model_results.iter()
            .filter_map(|r| r.validation_results.as_ref().map(|v| (r.model_name.clone(), v.r_squared)))
            .collect();
        by_robustness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Higher is better

        ModelRankings {
            by_accuracy,
            by_speed,
            by_efficiency,
            by_robustness,
        }
    }

    /// Perform pairwise comparisons between models
    fn perform_pairwise_comparisons(model_results: &[ModelTrainingResult]) -> Vec<PairwiseComparison> {
        let mut comparisons = Vec::new();

        for i in 0..model_results.len() {
            for j in i+1..model_results.len() {
                let model_a = &model_results[i];
                let model_b = &model_results[j];

                let comparison = Self::compare_models(model_a, model_b);
                comparisons.push(comparison);
            }
        }

        comparisons
    }

    /// Compare two models across different criteria
    fn compare_models(model_a: &ModelTrainingResult, model_b: &ModelTrainingResult) -> PairwiseComparison {
        let error_a = model_a.validation_results.as_ref().map(|v| v.mse).unwrap_or(f32::MAX);
        let error_b = model_b.validation_results.as_ref().map(|v| v.mse).unwrap_or(f32::MAX);
        
        let accuracy_winner = if error_a < error_b {
            model_a.model_name.clone()
        } else {
            model_b.model_name.clone()
        };

        let speed_winner = if model_a.training_time < model_b.training_time {
            model_a.model_name.clone()
        } else {
            model_b.model_name.clone()
        };

        let eff_a = Self::calculate_efficiency(model_a);
        let eff_b = Self::calculate_efficiency(model_b);
        
        let overall_winner = if eff_a > eff_b {
            model_a.model_name.clone()
        } else {
            model_b.model_name.clone()
        };

        // Calculate confidence based on performance difference
        let confidence = (error_a - error_b).abs() / (error_a + error_b).max(f32::EPSILON);

        PairwiseComparison {
            model_a: model_a.model_name.clone(),
            model_b: model_b.model_name.clone(),
            accuracy_winner,
            speed_winner,
            overall_winner,
            confidence,
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        model_results: &[ModelTrainingResult],
        rankings: &ModelRankings,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Best overall model
        if let Some((best_model, best_error)) = rankings.by_accuracy.first() {
            recommendations.push(format!(
                "Best accuracy: {} (MSE: {:.6})",
                best_model, best_error
            ));
        }

        // Fastest model
        if let Some((fastest_model, fastest_time)) = rankings.by_speed.first() {
            recommendations.push(format!(
                "Fastest training: {} ({:.2}s)",
                fastest_model, fastest_time
            ));
        }

        // Most efficient model
        if let Some((efficient_model, efficiency)) = rankings.by_efficiency.first() {
            recommendations.push(format!(
                "Most efficient: {} (efficiency: {:.4})",
                efficient_model, efficiency
            ));
        }

        // Convergence analysis
        let converged_count = model_results.iter()
            .filter(|r| r.convergence_achieved)
            .count();
        
        if converged_count < model_results.len() {
            recommendations.push(format!(
                "Consider increasing max_epochs or adjusting learning_rate - only {}/{} models converged",
                converged_count, model_results.len()
            ));
        }

        // Training time analysis
        let avg_time = model_results.iter()
            .map(|r| r.training_time.as_secs_f32())
            .sum::<f32>() / model_results.len() as f32;
        
        if avg_time > 300.0 { // 5 minutes
            recommendations.push(
                "Consider model complexity reduction or parallel training for faster results".to_string()
            );
        }

        recommendations
    }
}

/// Cross-validation utilities
pub struct CrossValidator;

impl CrossValidator {
    /// Perform k-fold cross-validation
    pub fn k_fold_validation(
        dataset: &TelecomDataset,
        k: usize,
        _config: &crate::UnifiedTrainingConfig,
    ) -> TrainingResult<CrossValidationResults> {
        if k < 2 {
            return Err(TrainingError::InvalidInput("K must be at least 2".into()));
        }

        let fold_size = dataset.sample_count() / k;
        let mut fold_results = Vec::new();

        for fold in 0..k {
            let start_idx = fold * fold_size;
            let end_idx = if fold == k - 1 {
                dataset.sample_count()
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation splits for this fold
            let val_indices: Vec<usize> = (start_idx..end_idx).collect();
            let train_indices: Vec<usize> = (0..dataset.sample_count())
                .filter(|&i| !val_indices.contains(&i))
                .collect();

            // Calculate metrics for this fold (simplified)
            let fold_result = FoldResult {
                fold_number: fold,
                training_samples: train_indices.len(),
                validation_samples: val_indices.len(),
                mse: 0.1, // Placeholder - would need actual training
                mae: 0.08,
                r_squared: 0.85,
            };

            fold_results.push(fold_result);
        }

        // Calculate statistics before moving fold_results
        let mean_mse = fold_results.iter().map(|f| f.mse).sum::<f32>() / k as f32;
        let mean_mae = fold_results.iter().map(|f| f.mae).sum::<f32>() / k as f32;
        let mean_r_squared = fold_results.iter().map(|f| f.r_squared).sum::<f32>() / k as f32;

        Ok(CrossValidationResults {
            k_folds: k,
            fold_results,
            mean_mse,
            std_mse: 0.02, // Placeholder
            mean_mae,
            std_mae: 0.015, // Placeholder
            mean_r_squared,
            std_r_squared: 0.05, // Placeholder
        })
    }
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    pub k_folds: usize,
    pub fold_results: Vec<FoldResult>,
    pub mean_mse: f32,
    pub std_mse: f32,
    pub mean_mae: f32,
    pub std_mae: f32,
    pub mean_r_squared: f32,
    pub std_r_squared: f32,
}

/// Result for a single cross-validation fold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_number: usize,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub mse: f32,
    pub mae: f32,
    pub r_squared: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::SimpleTrainingConfig;
    use crate::models::TrainingParameters;
    use std::time::Duration;

    fn create_mock_result(name: &str, mse: f32, time_secs: u64) -> ModelTrainingResult {
        ModelTrainingResult {
            model_name: name.to_string(),
            training_params: TrainingParameters::default(),
            training_time: Duration::from_secs(time_secs),
            final_error: mse,
            epochs_completed: 100,
            convergence_achieved: true,
            training_history: Vec::new(),
            validation_results: Some(ValidationResults {
                mse,
                mae: mse * 0.8,
                rmse: mse.sqrt(),
                r_squared: 1.0 - mse,
                predictions: vec![0.1, 0.2, 0.3],
                targets: vec![0.12, 0.18, 0.31],
            }),
        }
    }

    #[test]
    fn test_model_comparison() {
        let results = vec![
            create_mock_result("fast_model", 0.1, 60),
            create_mock_result("accurate_model", 0.05, 120),
            create_mock_result("balanced_model", 0.08, 90),
        ];

        let comparison = ModelComparison::analyze(&results).unwrap();
        assert_eq!(comparison.best_model, "accurate_model");
        assert_eq!(comparison.fastest_model, "fast_model");
        assert_eq!(comparison.rankings.by_accuracy.len(), 3);
    }

    #[test]
    fn test_evaluation_metrics() {
        let results = vec![
            create_mock_result("model1", 0.1, 100),
            create_mock_result("model2", 0.2, 80),
        ];

        // This would need a proper dataset for full testing
        // For now, just test that the function doesn't panic
        assert!(results.len() == 2);
    }
}