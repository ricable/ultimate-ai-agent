//! Model evaluation and comparison utilities

use crate::data::TelecomDataset;
use crate::models::NeuralModel;
use crate::training::{ModelTrainingResult, ValidationResults};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::Mutex;
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

/// Comprehensive evaluation metrics
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationMetrics {
    pub mse: f32,
    pub mae: f32,
    pub rmse: f32,
    pub r_squared: f32,
    pub mape: f32, // Mean Absolute Percentage Error
    pub correlation: f32,
    pub std_residuals: f32,
    pub mean_residuals: f32,
    // Enhanced metrics
    pub adjusted_r_squared: f32,
    pub aic: f32,  // Akaike Information Criterion
    pub bic: f32,  // Bayesian Information Criterion
    pub explained_variance: f32,
    pub median_absolute_error: f32,
    pub max_error: f32,
    pub smape: f32,  // Symmetric Mean Absolute Percentage Error
    pub directional_accuracy: f32,  // Percentage of correct direction predictions
    pub theil_u: f32,  // Theil's U statistic
    pub prediction_interval_coverage: f32,
}

/// Model comparison report
#[derive(Debug, Clone, Serialize)]
pub struct ModelComparisonReport {
    pub best_overall_model: String,
    pub best_accuracy_model: String,
    pub fastest_model: String,
    pub most_efficient_model: String,
    pub model_rankings: ModelRankings,
    pub detailed_metrics: HashMap<String, EvaluationMetrics>,
    pub performance_summary: PerformanceSummary,
}

/// Model rankings across different criteria
#[derive(Debug, Clone, Serialize)]
pub struct ModelRankings {
    pub accuracy_ranking: Vec<RankingEntry>,
    pub speed_ranking: Vec<RankingEntry>,
    pub efficiency_ranking: Vec<RankingEntry>,
    pub robustness_ranking: Vec<RankingEntry>,
}

/// Individual ranking entry
#[derive(Debug, Clone, Serialize)]
pub struct RankingEntry {
    pub model_name: String,
    pub score: f32,
    pub rank: usize,
}

/// Performance summary statistics
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSummary {
    pub average_mse: f32,
    pub average_r_squared: f32,
    pub average_training_time: Duration,
    pub convergence_rate: f32, // Percentage of models that converged
    pub best_mse: f32,
    pub worst_mse: f32,
    pub mse_std_dev: f32,
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize)]
pub struct CrossValidationResults {
    pub model_name: String,
    pub fold_results: Vec<FoldResult>,
    pub mean_metrics: EvaluationMetrics,
    pub std_metrics: EvaluationMetrics,
    pub confidence_intervals: ConfidenceIntervals,
    pub cv_score: f32,
    pub cv_score_std: f32,
    pub best_fold: usize,
    pub worst_fold: usize,
    pub statistical_significance: StatisticalTests,
}

/// Results for a single cross-validation fold
#[derive(Debug, Clone, Serialize)]
pub struct FoldResult {
    pub fold_index: usize,
    pub train_size: usize,
    pub test_size: usize,
    pub metrics: EvaluationMetrics,
    pub training_time: Duration,
}

/// Confidence intervals for metrics
#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceIntervals {
    pub mse_ci: (f32, f32),
    pub r_squared_ci: (f32, f32),
    pub mae_ci: (f32, f32),
    pub rmse_ci: (f32, f32),
    pub correlation_ci: (f32, f32),
    pub confidence_level: f32,
}

/// Statistical significance tests
#[derive(Debug, Clone, Serialize)]
pub struct StatisticalTests {
    pub t_test_p_value: f32,
    pub wilcoxon_p_value: f32,
    pub normality_test_p_value: f32,
    pub heteroscedasticity_test_p_value: f32,
    pub durbin_watson_statistic: f32,
    pub ljung_box_p_value: f32,
}

/// Advanced validation techniques
#[derive(Debug, Clone, Serialize)]
pub struct AdvancedValidation {
    pub nested_cv_results: Vec<CrossValidationResults>,
    pub time_series_cv_results: Option<TimeSeriesValidation>,
    pub bootstrap_results: BootstrapResults,
    pub permutation_test_results: PermutationTestResults,
    pub learning_curve_analysis: LearningCurveAnalysis,
}

/// Time series specific validation
#[derive(Debug, Clone, Serialize)]
pub struct TimeSeriesValidation {
    pub walk_forward_results: Vec<FoldResult>,
    pub expanding_window_results: Vec<FoldResult>,
    pub sliding_window_results: Vec<FoldResult>,
    pub seasonal_decomposition_metrics: SeasonalMetrics,
}

/// Bootstrap validation results
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapResults {
    pub n_bootstraps: usize,
    pub bootstrap_scores: Vec<f32>,
    pub bias_estimate: f32,
    pub optimism_estimate: f32,
    pub confidence_intervals: HashMap<String, (f32, f32)>,
}

/// Permutation test results
#[derive(Debug, Clone, Serialize)]
pub struct PermutationTestResults {
    pub n_permutations: usize,
    pub null_distribution: Vec<f32>,
    pub p_value: f32,
    pub effect_size: f32,
    pub critical_value: f32,
}

/// Learning curve analysis
#[derive(Debug, Clone, Serialize)]
pub struct LearningCurveAnalysis {
    pub training_sizes: Vec<usize>,
    pub train_scores: Vec<f32>,
    pub validation_scores: Vec<f32>,
    pub train_scores_std: Vec<f32>,
    pub validation_scores_std: Vec<f32>,
    pub optimal_training_size: usize,
    pub bias_variance_decomposition: BiasVarianceDecomposition,
}

/// Bias-variance decomposition
#[derive(Debug, Clone, Serialize)]
pub struct BiasVarianceDecomposition {
    pub bias_squared: f32,
    pub variance: f32,
    pub noise: f32,
    pub total_error: f32,
}

/// Seasonal metrics for time series
#[derive(Debug, Clone, Serialize)]
pub struct SeasonalMetrics {
    pub seasonal_strength: f32,
    pub trend_strength: f32,
    pub spike_strength: f32,
    pub linearity: f32,
    pub curvature: f32,
    pub e_acf1: f32,
    pub e_acf10: f32,
}

/// Enhanced model evaluator with comprehensive statistical analysis
pub struct ModelEvaluator {
    pub enable_parallel: bool,
    pub confidence_level: f32,
    pub random_seed: Option<u64>,
    pub performance_tracking: Arc<Mutex<PerformanceTracker>>,
}

/// Performance tracker for evaluation operations
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    pub evaluation_times: Vec<Duration>,
    pub memory_usage: Vec<usize>,
    pub operation_counts: HashMap<String, usize>,
    pub bottlenecks: Vec<String>,
}

/// Model evaluation report with detailed analysis
#[derive(Debug, Clone, Serialize)]
pub struct ModelEvaluationReport {
    pub model_name: String,
    pub evaluation_metrics: EvaluationMetrics,
    pub cross_validation: CrossValidationResults,
    pub advanced_validation: AdvancedValidation,
    pub residual_analysis: ResidualAnalysis,
    pub feature_importance: FeatureImportanceAnalysis,
    pub performance_profile: PerformanceProfile,
    pub recommendations: Vec<String>,
}

/// Residual analysis results
#[derive(Debug, Clone, Serialize)]
pub struct ResidualAnalysis {
    pub residuals: Vec<f32>,
    pub standardized_residuals: Vec<f32>,
    pub studentized_residuals: Vec<f32>,
    pub leverage_values: Vec<f32>,
    pub cooks_distance: Vec<f32>,
    pub outlier_indices: Vec<usize>,
    pub influential_points: Vec<usize>,
    pub normality_test: StatisticalTest,
    pub homoscedasticity_test: StatisticalTest,
    pub independence_test: StatisticalTest,
}

/// Feature importance analysis
#[derive(Debug, Clone, Serialize)]
pub struct FeatureImportanceAnalysis {
    pub feature_names: Vec<String>,
    pub importance_scores: Vec<f32>,
    pub permutation_importance: Vec<f32>,
    pub correlation_matrix: Array2<f32>,
    pub multicollinearity_indices: Vec<f32>,
    pub feature_selection_recommendations: Vec<String>,
}

/// Performance profiling results
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceProfile {
    pub evaluation_time: Duration,
    pub memory_usage: usize,
    pub prediction_time_per_sample: Duration,
    pub throughput_samples_per_second: f32,
    pub cpu_utilization: f32,
    pub memory_efficiency: f32,
    pub bottleneck_analysis: Vec<String>,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize)]
pub struct StatisticalTest {
    pub test_name: String,
    pub test_statistic: f32,
    pub p_value: f32,
    pub critical_value: f32,
    pub is_significant: bool,
    pub interpretation: String,
}

/// Ensemble evaluation results
#[derive(Debug, Clone, Serialize)]
pub struct EnsembleEvaluationResults {
    pub individual_models: Vec<ModelEvaluationReport>,
    pub ensemble_metrics: EvaluationMetrics,
    pub diversity_metrics: DiversityMetrics,
    pub combination_weights: Vec<f32>,
    pub ensemble_improvement: f32,
}

/// Diversity metrics for ensemble models
#[derive(Debug, Clone, Serialize)]
pub struct DiversityMetrics {
    pub disagreement_measure: f32,
    pub double_fault_measure: f32,
    pub q_statistic: f32,
    pub correlation_coefficient: f32,
    pub entropy_measure: f32,
}

impl ModelEvaluator {
    /// Evaluate a single model on test data
    pub fn evaluate_model(
        model: &mut NeuralModel,
        test_data: &TelecomDataset,
    ) -> Result<EvaluationMetrics> {
        let mut predictions = Vec::new();
        let mut targets = Vec::new();
        
        // Get predictions for all test samples
        for i in 0..test_data.features.nrows() {
            let input: Vec<f32> = test_data.features.row(i).to_vec();
            let prediction = model.predict(&input);
            let target = test_data.targets[i];
            
            predictions.push(prediction[0]);
            targets.push(target);
        }
        
        // Calculate all metrics
        let metrics = Self::calculate_metrics(&predictions, &targets);
        Ok(metrics)
    }
    
    /// Calculate comprehensive evaluation metrics
    pub fn calculate_metrics(predictions: &[f32], targets: &[f32]) -> EvaluationMetrics {
        let n = predictions.len() as f32;
        
        // Basic metrics
        let mse = Self::calculate_mse(predictions, targets);
        let mae = Self::calculate_mae(predictions, targets);
        let rmse = mse.sqrt();
        let r_squared = Self::calculate_r_squared(predictions, targets);
        let mape = Self::calculate_mape(predictions, targets);
        let correlation = Self::calculate_correlation(predictions, targets);
        
        // Residual analysis
        let residuals: Vec<f32> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| target - pred)
            .collect();
        
        let mean_residuals = residuals.iter().sum::<f32>() / n;
        let std_residuals = {
            let variance = residuals.iter()
                .map(|&r| (r - mean_residuals).powi(2))
                .sum::<f32>() / n;
            variance.sqrt()
        };
        
        // Calculate enhanced metrics
        let adjusted_r_squared = Self::calculate_adjusted_r_squared(r_squared, n as usize, 22); // 22 features
        let aic = Self::calculate_aic(mse, n as usize, 22);
        let bic = Self::calculate_bic(mse, n as usize, 22);
        let explained_variance = Self::calculate_explained_variance(predictions, targets);
        let median_absolute_error = Self::calculate_median_absolute_error(predictions, targets);
        let max_error = Self::calculate_max_error(predictions, targets);
        let smape = Self::calculate_smape(predictions, targets);
        let directional_accuracy = Self::calculate_directional_accuracy(predictions, targets);
        let theil_u = Self::calculate_theil_u(predictions, targets);
        let prediction_interval_coverage = 95.0; // Placeholder
        
        EvaluationMetrics {
            mse,
            mae,
            rmse,
            r_squared,
            mape,
            correlation,
            std_residuals,
            mean_residuals,
            adjusted_r_squared,
            aic,
            bic,
            explained_variance,
            median_absolute_error,
            max_error,
            smape,
            directional_accuracy,
            theil_u,
            prediction_interval_coverage,
        }
    }
    
    /// Compare multiple models and generate report
    pub fn compare_models(
        training_results: &[ModelTrainingResult],
    ) -> Result<ModelComparisonReport> {
        let mut detailed_metrics = HashMap::new();
        let mut accuracy_scores = Vec::new();
        let mut speed_scores = Vec::new();
        let mut efficiency_scores = Vec::new();
        
        // Calculate metrics for each model
        for result in training_results {
            if let Some(ref validation) = result.validation_results {
                let metrics = Self::calculate_metrics(&validation.predictions, &validation.targets);
                detailed_metrics.insert(result.model_name.clone(), metrics.clone());
                
                // Collect scores for ranking
                accuracy_scores.push((result.model_name.clone(), metrics.r_squared));
                speed_scores.push((result.model_name.clone(), result.training_time.as_secs_f32()));
                
                // Efficiency = accuracy / time (normalized)
                let efficiency = metrics.r_squared / result.training_time.as_secs_f32();
                efficiency_scores.push((result.model_name.clone(), efficiency));
            }
        }
        
        // Create rankings
        let model_rankings = Self::create_rankings(
            &accuracy_scores,
            &speed_scores,
            &efficiency_scores,
            training_results,
        );
        
        // Find best models
        let best_accuracy_model = accuracy_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|x| x.0.clone())
            .unwrap_or_default();
        
        let fastest_model = speed_scores.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|x| x.0.clone())
            .unwrap_or_default();
        
        let most_efficient_model = efficiency_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|x| x.0.clone())
            .unwrap_or_default();
        
        // Calculate performance summary
        let performance_summary = Self::calculate_performance_summary(training_results, &detailed_metrics);
        
        Ok(ModelComparisonReport {
            best_overall_model: best_accuracy_model.clone(),
            best_accuracy_model,
            fastest_model,
            most_efficient_model,
            model_rankings,
            detailed_metrics,
            performance_summary,
        })
    }
    
    /// Perform k-fold cross-validation
    pub fn cross_validate(
        model_template: NeuralModel,
        dataset: &TelecomDataset,
        k_folds: usize,
    ) -> Result<CrossValidationResults> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let n_samples = dataset.features.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());
        
        let fold_size = n_samples / k_folds;
        let mut fold_results = Vec::new();
        
        for fold in 0..k_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == k_folds - 1 { n_samples } else { (fold + 1) * fold_size };
            
            // Split indices into train and test
            let test_indices = &indices[start_idx..end_idx];
            let train_indices: Vec<usize> = indices.iter()
                .filter(|&&idx| !test_indices.contains(&idx))
                .copied()
                .collect();
            
            // Create train and test datasets
            let train_features = dataset.features.select(ndarray::Axis(0), &train_indices);
            let test_features = dataset.features.select(ndarray::Axis(0), test_indices);
            let train_targets = dataset.targets.select(ndarray::Axis(0), &train_indices);
            let test_targets = dataset.targets.select(ndarray::Axis(0), test_indices);
            
            let train_dataset = TelecomDataset {
                features: train_features,
                targets: train_targets,
                feature_names: dataset.feature_names.clone(),
                target_name: dataset.target_name.clone(),
                normalization_stats: dataset.normalization_stats.clone(),
            };
            
            let test_dataset = TelecomDataset {
                features: test_features,
                targets: test_targets,
                feature_names: dataset.feature_names.clone(),
                target_name: dataset.target_name.clone(),
                normalization_stats: dataset.normalization_stats.clone(),
            };
            
            // Train and evaluate model
            let start_time = std::time::Instant::now();
            let mut model = model_template.clone();
            
            // Note: Simplified training for cross-validation
            // In practice, you would use the full training pipeline
            let metrics = Self::evaluate_model(&mut model, &test_dataset)?;
            let training_time = start_time.elapsed();
            
            fold_results.push(FoldResult {
                fold_index: fold,
                train_size: train_indices.len(),
                test_size: test_indices.len(),
                metrics,
                training_time,
            });
        }
        
        // Calculate mean and std metrics
        let mean_metrics = Self::calculate_mean_metrics(&fold_results);
        let std_metrics = Self::calculate_std_metrics(&fold_results, &mean_metrics);
        let confidence_intervals = Self::calculate_confidence_intervals(&fold_results);
        
        Ok(CrossValidationResults {
            model_name: model_template.name,
            fold_results,
            mean_metrics,
            std_metrics,
            confidence_intervals,
        })
    }
    
    // Helper methods for metric calculations
    fn calculate_mse(predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum::<f32>() / predictions.len() as f32
    }
    
    fn calculate_mae(predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .sum::<f32>() / predictions.len() as f32
    }
    
    fn calculate_r_squared(predictions: &[f32], targets: &[f32]) -> f32 {
        let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
        
        let ss_res: f32 = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (target - pred).powi(2))
            .sum();
        
        let ss_tot: f32 = targets.iter()
            .map(|&target| (target - target_mean).powi(2))
            .sum();
        
        if ss_tot == 0.0 { 0.0 } else { 1.0 - (ss_res / ss_tot) }
    }
    
    fn calculate_mape(predictions: &[f32], targets: &[f32]) -> f32 {
        let mut mape_sum = 0.0;
        let mut valid_count = 0;
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if target != 0.0 {
                mape_sum += ((target - pred) / target).abs();
                valid_count += 1;
            }
        }
        
        if valid_count > 0 {
            (mape_sum / valid_count as f32) * 100.0
        } else {
            0.0
        }
    }
    
    fn calculate_correlation(predictions: &[f32], targets: &[f32]) -> f32 {
        let n = predictions.len() as f32;
        let pred_mean = predictions.iter().sum::<f32>() / n;
        let target_mean = targets.iter().sum::<f32>() / n;
        
        let numerator: f32 = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - pred_mean) * (target - target_mean))
            .sum();
        
        let pred_var: f32 = predictions.iter()
            .map(|&pred| (pred - pred_mean).powi(2))
            .sum();
        
        let target_var: f32 = targets.iter()
            .map(|&target| (target - target_mean).powi(2))
            .sum();
        
        let denominator = (pred_var * target_var).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn create_rankings(
        accuracy_scores: &[(String, f32)],
        speed_scores: &[(String, f32)],
        efficiency_scores: &[(String, f32)],
        _training_results: &[ModelTrainingResult],
    ) -> ModelRankings {
        // Create accuracy ranking (higher is better)
        let mut accuracy_ranking: Vec<_> = accuracy_scores.iter()
            .enumerate()
            .map(|(i, (name, score))| RankingEntry {
                model_name: name.clone(),
                score: *score,
                rank: i + 1,
            })
            .collect();
        accuracy_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Create speed ranking (lower is better)
        let mut speed_ranking: Vec<_> = speed_scores.iter()
            .enumerate()
            .map(|(i, (name, score))| RankingEntry {
                model_name: name.clone(),
                score: *score,
                rank: i + 1,
            })
            .collect();
        speed_ranking.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        
        // Create efficiency ranking (higher is better)
        let mut efficiency_ranking: Vec<_> = efficiency_scores.iter()
            .enumerate()
            .map(|(i, (name, score))| RankingEntry {
                model_name: name.clone(),
                score: *score,
                rank: i + 1,
            })
            .collect();
        efficiency_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        ModelRankings {
            accuracy_ranking,
            speed_ranking,
            efficiency_ranking,
            robustness_ranking: Vec::new(), // TODO: Implement robustness ranking
        }
    }
    
    fn calculate_performance_summary(
        training_results: &[ModelTrainingResult],
        detailed_metrics: &HashMap<String, EvaluationMetrics>,
    ) -> PerformanceSummary {
        let n = training_results.len() as f32;
        
        let mse_values: Vec<f32> = detailed_metrics.values().map(|m| m.mse).collect();
        let r_squared_values: Vec<f32> = detailed_metrics.values().map(|m| m.r_squared).collect();
        
        let average_mse = mse_values.iter().sum::<f32>() / n;
        let average_r_squared = r_squared_values.iter().sum::<f32>() / n;
        
        let average_training_time = Duration::from_secs_f32(
            training_results.iter()
                .map(|r| r.training_time.as_secs_f32())
                .sum::<f32>() / n
        );
        
        let convergence_rate = training_results.iter()
            .filter(|r| r.convergence_achieved)
            .count() as f32 / n * 100.0;
        
        let best_mse = mse_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let worst_mse = mse_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let mse_variance = mse_values.iter()
            .map(|&mse| (mse - average_mse).powi(2))
            .sum::<f32>() / n;
        let mse_std_dev = mse_variance.sqrt();
        
        PerformanceSummary {
            average_mse,
            average_r_squared,
            average_training_time,
            convergence_rate,
            best_mse,
            worst_mse,
            mse_std_dev,
        }
    }
    
    fn calculate_mean_metrics(fold_results: &[FoldResult]) -> EvaluationMetrics {
        let n = fold_results.len() as f32;
        
        EvaluationMetrics {
            mse: fold_results.iter().map(|f| f.metrics.mse).sum::<f32>() / n,
            mae: fold_results.iter().map(|f| f.metrics.mae).sum::<f32>() / n,
            rmse: fold_results.iter().map(|f| f.metrics.rmse).sum::<f32>() / n,
            r_squared: fold_results.iter().map(|f| f.metrics.r_squared).sum::<f32>() / n,
            mape: fold_results.iter().map(|f| f.metrics.mape).sum::<f32>() / n,
            correlation: fold_results.iter().map(|f| f.metrics.correlation).sum::<f32>() / n,
            std_residuals: fold_results.iter().map(|f| f.metrics.std_residuals).sum::<f32>() / n,
            mean_residuals: fold_results.iter().map(|f| f.metrics.mean_residuals).sum::<f32>() / n,
        }
    }
    
    fn calculate_std_metrics(fold_results: &[FoldResult], mean_metrics: &EvaluationMetrics) -> EvaluationMetrics {
        let n = fold_results.len() as f32;
        
        let mse_var = fold_results.iter()
            .map(|f| (f.metrics.mse - mean_metrics.mse).powi(2))
            .sum::<f32>() / n;
        
        let r_squared_var = fold_results.iter()
            .map(|f| (f.metrics.r_squared - mean_metrics.r_squared).powi(2))
            .sum::<f32>() / n;
        
        EvaluationMetrics {
            mse: mse_var.sqrt(),
            mae: 0.0, // TODO: Calculate all std metrics
            rmse: 0.0,
            r_squared: r_squared_var.sqrt(),
            mape: 0.0,
            correlation: 0.0,
            std_residuals: 0.0,
            mean_residuals: 0.0,
        }
    }
    
    fn calculate_confidence_intervals(fold_results: &[FoldResult]) -> ConfidenceIntervals {
        // Simplified 95% confidence intervals
        let mse_values: Vec<f32> = fold_results.iter().map(|f| f.metrics.mse).collect();
        let r_squared_values: Vec<f32> = fold_results.iter().map(|f| f.metrics.r_squared).collect();
        let mae_values: Vec<f32> = fold_results.iter().map(|f| f.metrics.mae).collect();
        
        let mse_ci = Self::calculate_ci(&mse_values);
        let r_squared_ci = Self::calculate_ci(&r_squared_values);
        let mae_ci = Self::calculate_ci(&mae_values);
        
        ConfidenceIntervals {
            mse_ci,
            r_squared_ci,
            mae_ci,
        }
    }
    
    fn calculate_ci(values: &[f32]) -> (f32, f32) {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_values.len();
        let lower_idx = (n as f32 * 0.025) as usize;
        let upper_idx = (n as f32 * 0.975) as usize;
        
        (sorted_values[lower_idx], sorted_values[upper_idx.min(n - 1)])
    }
    
    // Enhanced metric calculation methods
    fn calculate_adjusted_r_squared(r_squared: f32, n: usize, p: usize) -> f32 {
        if n <= p + 1 {
            r_squared
        } else {
            1.0 - ((1.0 - r_squared) * (n - 1) as f32) / (n - p - 1) as f32
        }
    }
    
    fn calculate_aic(mse: f32, n: usize, p: usize) -> f32 {
        n as f32 * mse.ln() + 2.0 * p as f32
    }
    
    fn calculate_bic(mse: f32, n: usize, p: usize) -> f32 {
        n as f32 * mse.ln() + (n as f32).ln() * p as f32
    }
    
    fn calculate_explained_variance(predictions: &[f32], targets: &[f32]) -> f32 {
        let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
        let target_variance = targets.iter()
            .map(|&t| (t - target_mean).powi(2))
            .sum::<f32>() / targets.len() as f32;
        
        let residual_variance = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (target - pred).powi(2))
            .sum::<f32>() / predictions.len() as f32;
        
        if target_variance > 0.0 {
            1.0 - (residual_variance / target_variance)
        } else {
            0.0
        }
    }
    
    fn calculate_median_absolute_error(predictions: &[f32], targets: &[f32]) -> f32 {
        let mut absolute_errors: Vec<f32> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (target - pred).abs())
            .collect();
        
        absolute_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = absolute_errors.len();
        
        if n % 2 == 0 {
            (absolute_errors[n / 2 - 1] + absolute_errors[n / 2]) / 2.0
        } else {
            absolute_errors[n / 2]
        }
    }
    
    fn calculate_max_error(predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (target - pred).abs())
            .fold(0.0, |a, b| a.max(b))
    }
    
    fn calculate_smape(predictions: &[f32], targets: &[f32]) -> f32 {
        let mut smape_sum = 0.0;
        let mut valid_count = 0;
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let denominator = (target.abs() + pred.abs()) / 2.0;
            if denominator > 0.0 {
                smape_sum += (target - pred).abs() / denominator;
                valid_count += 1;
            }
        }
        
        if valid_count > 0 {
            (smape_sum / valid_count as f32) * 100.0
        } else {
            0.0
        }
    }
    
    fn calculate_directional_accuracy(predictions: &[f32], targets: &[f32]) -> f32 {
        if predictions.len() < 2 {
            return 0.0;
        }
        
        let mut correct_directions = 0;
        let mut total_directions = 0;
        
        for i in 1..predictions.len() {
            let pred_direction = predictions[i] - predictions[i - 1];
            let target_direction = targets[i] - targets[i - 1];
            
            if pred_direction.signum() == target_direction.signum() {
                correct_directions += 1;
            }
            total_directions += 1;
        }
        
        if total_directions > 0 {
            (correct_directions as f32 / total_directions as f32) * 100.0
        } else {
            0.0
        }
    }
    
    fn calculate_theil_u(predictions: &[f32], targets: &[f32]) -> f32 {
        let mse = Self::calculate_mse(predictions, targets);
        let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
        
        let naive_mse = targets.iter()
            .map(|&target| (target - target_mean).powi(2))
            .sum::<f32>() / targets.len() as f32;
        
        if naive_mse > 0.0 {
            (mse / naive_mse).sqrt()
        } else {
            0.0
        }
    }
}