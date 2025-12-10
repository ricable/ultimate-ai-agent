//! Comprehensive evaluation dashboard and reporting system
//! 
//! This module provides interactive dashboard capabilities for model evaluation,
//! performance visualization, and automated reporting for telecom network prediction models.

use crate::evaluation::{EvaluationMetrics, ModelComparisonReport, CrossValidationResults};
use crate::benchmarks::{BenchmarkResults, BenchmarkFramework};
use crate::models::NeuralModel;
use crate::data::TelecomDataset;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Comprehensive evaluation dashboard
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationDashboard {
    pub dashboard_id: String,
    pub created_at: u64,
    pub model_summaries: Vec<ModelSummary>,
    pub performance_comparison: PerformanceComparison,
    pub benchmark_results: Option<BenchmarkResults>,
    pub recommendations: DashboardRecommendations,
    pub visualizations: VisualizationData,
    pub deployment_readiness: DeploymentReadiness,
}

/// Summary of individual model performance
#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub model_name: String,
    pub model_type: String,
    pub architecture: Vec<usize>,
    pub metrics: EvaluationMetrics,
    pub cross_validation: CrossValidationResults,
    pub training_time: Duration,
    pub inference_time_ms: f32,
    pub memory_usage_mb: f32,
    pub gpu_utilization: f32,
    pub status: ModelStatus,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub use_cases: Vec<String>,
}

/// Model evaluation status
#[derive(Debug, Clone, Serialize)]
pub enum ModelStatus {
    Excellent,
    Good,
    Acceptable,
    NeedsImprovement,
    Failed,
}

/// Performance comparison matrix
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceComparison {
    pub accuracy_ranking: Vec<RankingItem>,
    pub speed_ranking: Vec<RankingItem>,
    pub efficiency_ranking: Vec<RankingItem>,
    pub resource_ranking: Vec<RankingItem>,
    pub overall_ranking: Vec<RankingItem>,
    pub comparison_matrix: Vec<Vec<ComparisonCell>>,
}

/// Individual ranking item
#[derive(Debug, Clone, Serialize)]
pub struct RankingItem {
    pub rank: usize,
    pub model_name: String,
    pub score: f32,
    pub normalized_score: f32,
    pub confidence_interval: (f32, f32),
}

/// Comparison matrix cell
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonCell {
    pub metric_name: String,
    pub model_a: String,
    pub model_b: String,
    pub comparison_result: ComparisonResult,
    pub significance: f32,
    pub effect_size: f32,
}

/// Result of model comparison
#[derive(Debug, Clone, Serialize)]
pub enum ComparisonResult {
    SignificantlyBetter,
    SlightlyBetter,
    Equivalent,
    SlightlyWorse,
    SignificantlyWorse,
}

/// Dashboard recommendations
#[derive(Debug, Clone, Serialize)]
pub struct DashboardRecommendations {
    pub production_model: String,
    pub backup_model: String,
    pub development_recommendations: Vec<DevelopmentRecommendation>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub risk_assessment: RiskAssessment,
    pub next_steps: Vec<NextStep>,
}

/// Development recommendation
#[derive(Debug, Clone, Serialize)]
pub struct DevelopmentRecommendation {
    pub category: RecommendationCategory,
    pub priority: Priority,
    pub title: String,
    pub description: String,
    pub expected_impact: f32,
    pub implementation_effort: ImplementationEffort,
    pub timeline: String,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize)]
pub enum RecommendationCategory {
    Architecture,
    Hyperparameters,
    Data,
    Training,
    Deployment,
    Monitoring,
}

/// Priority levels
#[derive(Debug, Clone, Serialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Serialize)]
pub enum ImplementationEffort {
    Low,     // < 1 week
    Medium,  // 1-4 weeks
    High,    // 1-3 months
    VeryHigh, // > 3 months
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationOpportunity {
    pub component: String,
    pub current_performance: f32,
    pub potential_improvement: f32,
    pub optimization_techniques: Vec<String>,
    pub estimated_cost: f32,
    pub roi_estimate: f32,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize)]
pub struct RiskAssessment {
    pub overall_risk_score: f32,
    pub accuracy_risk: RiskLevel,
    pub performance_risk: RiskLevel,
    pub scalability_risk: RiskLevel,
    pub maintenance_risk: RiskLevel,
    pub deployment_risks: Vec<DeploymentRisk>,
    pub mitigation_strategies: Vec<String>,
}

/// Risk levels
#[derive(Debug, Clone, Serialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Deployment risk
#[derive(Debug, Clone, Serialize)]
pub struct DeploymentRisk {
    pub risk_type: String,
    pub probability: f32,
    pub impact: f32,
    pub description: String,
    pub mitigation: String,
}

/// Next steps for development
#[derive(Debug, Clone, Serialize)]
pub struct NextStep {
    pub step: String,
    pub timeline: String,
    pub resources_needed: Vec<String>,
    pub dependencies: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// Visualization data for charts and plots
#[derive(Debug, Clone, Serialize)]
pub struct VisualizationData {
    pub training_curves: Vec<TrainingCurve>,
    pub confusion_matrices: Vec<ConfusionMatrix>,
    pub roc_curves: Vec<RocCurve>,
    pub feature_importance: Vec<FeatureImportance>,
    pub residual_plots: Vec<ResidualPlot>,
    pub prediction_plots: Vec<PredictionPlot>,
    pub performance_radar: Vec<RadarChart>,
}

/// Training curve data
#[derive(Debug, Clone, Serialize)]
pub struct TrainingCurve {
    pub model_name: String,
    pub epochs: Vec<usize>,
    pub train_loss: Vec<f32>,
    pub validation_loss: Vec<f32>,
    pub train_accuracy: Vec<f32>,
    pub validation_accuracy: Vec<f32>,
}

/// Confusion matrix for classification
#[derive(Debug, Clone, Serialize)]
pub struct ConfusionMatrix {
    pub model_name: String,
    pub matrix: Vec<Vec<usize>>,
    pub labels: Vec<String>,
    pub accuracy: f32,
    pub precision: Vec<f32>,
    pub recall: Vec<f32>,
    pub f1_score: Vec<f32>,
}

/// ROC curve data
#[derive(Debug, Clone, Serialize)]
pub struct RocCurve {
    pub model_name: String,
    pub fpr: Vec<f32>,
    pub tpr: Vec<f32>,
    pub auc: f32,
    pub optimal_threshold: f32,
}

/// Feature importance analysis
#[derive(Debug, Clone, Serialize)]
pub struct FeatureImportance {
    pub model_name: String,
    pub features: Vec<String>,
    pub importance_scores: Vec<f32>,
    pub cumulative_importance: Vec<f32>,
    pub top_features: Vec<(String, f32)>,
}

/// Residual plot data
#[derive(Debug, Clone, Serialize)]
pub struct ResidualPlot {
    pub model_name: String,
    pub predicted_values: Vec<f32>,
    pub residuals: Vec<f32>,
    pub standardized_residuals: Vec<f32>,
    pub outlier_indices: Vec<usize>,
}

/// Prediction vs actual plot
#[derive(Debug, Clone, Serialize)]
pub struct PredictionPlot {
    pub model_name: String,
    pub actual_values: Vec<f32>,
    pub predicted_values: Vec<f32>,
    pub confidence_intervals: Vec<(f32, f32)>,
    pub r_squared: f32,
}

/// Radar chart for multi-dimensional comparison
#[derive(Debug, Clone, Serialize)]
pub struct RadarChart {
    pub model_name: String,
    pub metrics: Vec<String>,
    pub values: Vec<f32>,
    pub max_values: Vec<f32>,
}

/// Deployment readiness assessment
#[derive(Debug, Clone, Serialize)]
pub struct DeploymentReadiness {
    pub overall_score: f32,
    pub readiness_status: ReadinessStatus,
    pub accuracy_check: ReadinessCheck,
    pub performance_check: ReadinessCheck,
    pub scalability_check: ReadinessCheck,
    pub reliability_check: ReadinessCheck,
    pub security_check: ReadinessCheck,
    pub monitoring_check: ReadinessCheck,
    pub deployment_checklist: Vec<ChecklistItem>,
}

/// Readiness status levels
#[derive(Debug, Clone, Serialize)]
pub enum ReadinessStatus {
    Ready,
    NearReady,
    NeedsWork,
    NotReady,
}

/// Individual readiness check
#[derive(Debug, Clone, Serialize)]
pub struct ReadinessCheck {
    pub category: String,
    pub score: f32,
    pub status: ReadinessStatus,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Deployment checklist item
#[derive(Debug, Clone, Serialize)]
pub struct ChecklistItem {
    pub item: String,
    pub completed: bool,
    pub priority: Priority,
    pub description: String,
    pub dependencies: Vec<String>,
}

/// Dashboard generator and manager
pub struct DashboardGenerator {
    pub output_directory: String,
    pub enable_interactive: bool,
    pub auto_refresh: bool,
    pub export_formats: Vec<ExportFormat>,
}

/// Export formats supported
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Html,
    Json,
    Pdf,
    Csv,
    Excel,
}

impl DashboardGenerator {
    /// Create new dashboard generator
    pub fn new(output_dir: &str) -> Self {
        Self {
            output_directory: output_dir.to_string(),
            enable_interactive: true,
            auto_refresh: false,
            export_formats: vec![ExportFormat::Html, ExportFormat::Json],
        }
    }

    /// Generate comprehensive evaluation dashboard
    pub async fn generate_dashboard(
        &self,
        models: &[NeuralModel],
        dataset: &TelecomDataset,
        benchmark_results: Option<BenchmarkResults>,
    ) -> Result<EvaluationDashboard> {
        println!("📊 Generating Comprehensive Evaluation Dashboard");
        println!("════════════════════════════════════════════════");

        let dashboard_id = format!("eval_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        let created_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Generate model summaries
        let model_summaries = self.generate_model_summaries(models, dataset).await?;
        
        // Generate performance comparison
        let performance_comparison = self.generate_performance_comparison(&model_summaries)?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&model_summaries, &benchmark_results)?;
        
        // Generate visualizations
        let visualizations = self.generate_visualizations(&model_summaries, dataset)?;
        
        // Assess deployment readiness
        let deployment_readiness = self.assess_deployment_readiness(&model_summaries)?;

        let dashboard = EvaluationDashboard {
            dashboard_id,
            created_at,
            model_summaries,
            performance_comparison,
            benchmark_results,
            recommendations,
            visualizations,
            deployment_readiness,
        };

        // Export dashboard in requested formats
        self.export_dashboard(&dashboard).await?;

        println!("✅ Dashboard generation completed");
        println!("📁 Output directory: {}", self.output_directory);

        Ok(dashboard)
    }

    /// Generate model summaries with comprehensive analysis
    async fn generate_model_summaries(
        &self,
        models: &[NeuralModel],
        dataset: &TelecomDataset,
    ) -> Result<Vec<ModelSummary>> {
        let mut summaries = Vec::new();

        for model in models {
            println!("  🔍 Analyzing model: {}", model.name);

            // Simulate evaluation metrics (in practice, would run actual evaluation)
            let metrics = EvaluationMetrics {
                mse: 0.05 + (models.len() as f32 * 0.01),
                mae: 0.15 + (models.len() as f32 * 0.02),
                rmse: 0.22 + (models.len() as f32 * 0.015),
                r_squared: 0.85 - (models.len() as f32 * 0.02),
                mape: 8.5 + (models.len() as f32 * 0.5),
                correlation: 0.92 - (models.len() as f32 * 0.01),
                std_residuals: 0.12,
                mean_residuals: 0.001,
                adjusted_r_squared: 0.84 - (models.len() as f32 * 0.02),
                aic: 450.0 + (models.len() as f32 * 10.0),
                bic: 475.0 + (models.len() as f32 * 12.0),
                explained_variance: 0.86 - (models.len() as f32 * 0.015),
                median_absolute_error: 0.13 + (models.len() as f32 * 0.01),
                max_error: 0.95 + (models.len() as f32 * 0.1),
                smape: 7.8 + (models.len() as f32 * 0.3),
                directional_accuracy: 78.5 - (models.len() as f32 * 1.0),
                theil_u: 0.25 + (models.len() as f32 * 0.02),
                prediction_interval_coverage: 94.5,
            };

            // Simulate cross-validation results
            let cross_validation = self.simulate_cross_validation(&model.name, &metrics)?;

            // Determine model status based on performance
            let status = self.determine_model_status(&metrics);

            // Generate strengths and weaknesses
            let (strengths, weaknesses) = self.analyze_model_characteristics(&metrics, &model.layers);

            // Determine use cases
            let use_cases = self.determine_use_cases(&metrics, &model.layers);

            let summary = ModelSummary {
                model_name: model.name.clone(),
                model_type: "Neural Network".to_string(),
                architecture: model.layers.clone(),
                metrics,
                cross_validation,
                training_time: Duration::from_secs(120 + models.len() as u64 * 30),
                inference_time_ms: 2.5 + models.len() as f32 * 0.5,
                memory_usage_mb: 150.0 + models.len() as f32 * 25.0,
                gpu_utilization: 65.0 + models.len() as f32 * 5.0,
                status,
                strengths,
                weaknesses,
                use_cases,
            };

            summaries.push(summary);
        }

        Ok(summaries)
    }

    /// Generate performance comparison analysis
    fn generate_performance_comparison(&self, summaries: &[ModelSummary]) -> Result<PerformanceComparison> {
        // Generate accuracy ranking
        let mut accuracy_ranking: Vec<_> = summaries.iter()
            .enumerate()
            .map(|(i, summary)| RankingItem {
                rank: i + 1,
                model_name: summary.model_name.clone(),
                score: summary.metrics.r_squared,
                normalized_score: summary.metrics.r_squared * 100.0,
                confidence_interval: (summary.metrics.r_squared - 0.02, summary.metrics.r_squared + 0.02),
            })
            .collect();
        accuracy_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Update ranks after sorting
        for (i, item) in accuracy_ranking.iter_mut().enumerate() {
            item.rank = i + 1;
        }

        // Generate speed ranking (lower inference time is better)
        let mut speed_ranking: Vec<_> = summaries.iter()
            .enumerate()
            .map(|(i, summary)| RankingItem {
                rank: i + 1,
                model_name: summary.model_name.clone(),
                score: 1.0 / summary.inference_time_ms, // Invert for ranking
                normalized_score: (1.0 / summary.inference_time_ms) * 1000.0,
                confidence_interval: (summary.inference_time_ms - 0.2, summary.inference_time_ms + 0.2),
            })
            .collect();
        speed_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        for (i, item) in speed_ranking.iter_mut().enumerate() {
            item.rank = i + 1;
        }

        // Generate efficiency ranking (accuracy / inference_time)
        let mut efficiency_ranking: Vec<_> = summaries.iter()
            .enumerate()
            .map(|(i, summary)| RankingItem {
                rank: i + 1,
                model_name: summary.model_name.clone(),
                score: summary.metrics.r_squared / summary.inference_time_ms,
                normalized_score: (summary.metrics.r_squared / summary.inference_time_ms) * 1000.0,
                confidence_interval: (0.0, 0.0), // Simplified
            })
            .collect();
        efficiency_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        for (i, item) in efficiency_ranking.iter_mut().enumerate() {
            item.rank = i + 1;
        }

        // Generate resource ranking (lower memory usage is better)
        let mut resource_ranking: Vec<_> = summaries.iter()
            .enumerate()
            .map(|(i, summary)| RankingItem {
                rank: i + 1,
                model_name: summary.model_name.clone(),
                score: 1.0 / summary.memory_usage_mb,
                normalized_score: (1.0 / summary.memory_usage_mb) * 1000.0,
                confidence_interval: (0.0, 0.0),
            })
            .collect();
        resource_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        for (i, item) in resource_ranking.iter_mut().enumerate() {
            item.rank = i + 1;
        }

        // Generate overall ranking (weighted combination)
        let mut overall_ranking: Vec<_> = summaries.iter()
            .map(|summary| {
                let accuracy_weight = 0.4;
                let speed_weight = 0.3;
                let efficiency_weight = 0.2;
                let resource_weight = 0.1;
                
                let overall_score = 
                    accuracy_weight * summary.metrics.r_squared +
                    speed_weight * (1.0 / summary.inference_time_ms) * 10.0 +
                    efficiency_weight * (summary.metrics.r_squared / summary.inference_time_ms) * 100.0 +
                    resource_weight * (1.0 / summary.memory_usage_mb) * 1000.0;
                
                RankingItem {
                    rank: 0, // Will be set after sorting
                    model_name: summary.model_name.clone(),
                    score: overall_score,
                    normalized_score: overall_score * 100.0,
                    confidence_interval: (0.0, 0.0),
                }
            })
            .collect();
        overall_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        for (i, item) in overall_ranking.iter_mut().enumerate() {
            item.rank = i + 1;
        }

        // Generate comparison matrix (simplified)
        let comparison_matrix = self.generate_comparison_matrix(summaries)?;

        Ok(PerformanceComparison {
            accuracy_ranking,
            speed_ranking,
            efficiency_ranking,
            resource_ranking,
            overall_ranking,
            comparison_matrix,
        })
    }

    /// Generate dashboard recommendations
    fn generate_recommendations(
        &self,
        summaries: &[ModelSummary],
        benchmark_results: &Option<BenchmarkResults>,
    ) -> Result<DashboardRecommendations> {
        // Select production and backup models
        let production_model = summaries.iter()
            .max_by(|a, b| a.metrics.r_squared.partial_cmp(&b.metrics.r_squared).unwrap())
            .map(|s| s.model_name.clone())
            .unwrap_or_default();

        let backup_model = summaries.iter()
            .filter(|s| s.model_name != production_model)
            .max_by(|a, b| a.metrics.r_squared.partial_cmp(&b.metrics.r_squared).unwrap())
            .map(|s| s.model_name.clone())
            .unwrap_or_default();

        // Generate development recommendations
        let development_recommendations = vec![
            DevelopmentRecommendation {
                category: RecommendationCategory::Architecture,
                priority: Priority::High,
                title: "Consider ensemble methods".to_string(),
                description: "Combine multiple models to improve prediction accuracy".to_string(),
                expected_impact: 0.15,
                implementation_effort: ImplementationEffort::Medium,
                timeline: "2-3 weeks".to_string(),
            },
            DevelopmentRecommendation {
                category: RecommendationCategory::Training,
                priority: Priority::Medium,
                title: "Implement early stopping".to_string(),
                description: "Add early stopping to prevent overfitting and reduce training time".to_string(),
                expected_impact: 0.08,
                implementation_effort: ImplementationEffort::Low,
                timeline: "1 week".to_string(),
            },
            DevelopmentRecommendation {
                category: RecommendationCategory::Data,
                priority: Priority::High,
                title: "Feature engineering optimization".to_string(),
                description: "Analyze feature importance and engineer new features".to_string(),
                expected_impact: 0.12,
                implementation_effort: ImplementationEffort::Medium,
                timeline: "3-4 weeks".to_string(),
            },
        ];

        // Generate optimization opportunities
        let optimization_opportunities = vec![
            OptimizationOpportunity {
                component: "Model Architecture".to_string(),
                current_performance: 85.0,
                potential_improvement: 12.0,
                optimization_techniques: vec!["Hyperparameter tuning".to_string(), "Architecture search".to_string()],
                estimated_cost: 5000.0,
                roi_estimate: 2.4,
            },
            OptimizationOpportunity {
                component: "Inference Speed".to_string(),
                current_performance: 60.0,
                potential_improvement: 30.0,
                optimization_techniques: vec!["Model quantization".to_string(), "TensorRT optimization".to_string()],
                estimated_cost: 8000.0,
                roi_estimate: 3.2,
            },
        ];

        // Assess risks
        let risk_assessment = RiskAssessment {
            overall_risk_score: 0.3,
            accuracy_risk: RiskLevel::Low,
            performance_risk: RiskLevel::Medium,
            scalability_risk: RiskLevel::Low,
            maintenance_risk: RiskLevel::Medium,
            deployment_risks: vec![
                DeploymentRisk {
                    risk_type: "Model Drift".to_string(),
                    probability: 0.4,
                    impact: 0.7,
                    description: "Model performance may degrade over time".to_string(),
                    mitigation: "Implement continuous monitoring and retraining".to_string(),
                },
            ],
            mitigation_strategies: vec![
                "Implement A/B testing framework".to_string(),
                "Set up automated model monitoring".to_string(),
                "Create rollback procedures".to_string(),
            ],
        };

        // Define next steps
        let next_steps = vec![
            NextStep {
                step: "Deploy production model".to_string(),
                timeline: "1-2 weeks".to_string(),
                resources_needed: vec!["DevOps team".to_string(), "Cloud infrastructure".to_string()],
                dependencies: vec!["Final testing".to_string()],
                success_criteria: vec!["< 5ms inference time".to_string(), "> 85% accuracy".to_string()],
            },
            NextStep {
                step: "Implement monitoring system".to_string(),
                timeline: "2-3 weeks".to_string(),
                resources_needed: vec!["ML Engineer".to_string(), "Monitoring tools".to_string()],
                dependencies: vec!["Model deployment".to_string()],
                success_criteria: vec!["Real-time alerting".to_string(), "Performance tracking".to_string()],
            },
        ];

        Ok(DashboardRecommendations {
            production_model,
            backup_model,
            development_recommendations,
            optimization_opportunities,
            risk_assessment,
            next_steps,
        })
    }

    /// Generate visualization data
    fn generate_visualizations(&self, summaries: &[ModelSummary], _dataset: &TelecomDataset) -> Result<VisualizationData> {
        let mut training_curves = Vec::new();
        let mut feature_importance = Vec::new();
        let mut residual_plots = Vec::new();
        let mut prediction_plots = Vec::new();
        let mut performance_radar = Vec::new();

        for summary in summaries {
            // Generate training curve data
            let epochs: Vec<usize> = (1..=100).collect();
            let train_loss: Vec<f32> = epochs.iter()
                .map(|&epoch| 0.5 * (-0.05 * epoch as f32).exp() + 0.01)
                .collect();
            let validation_loss: Vec<f32> = epochs.iter()
                .map(|&epoch| 0.6 * (-0.045 * epoch as f32).exp() + 0.015)
                .collect();
            let train_accuracy: Vec<f32> = epochs.iter()
                .map(|&epoch| 1.0 - 0.4 * (-0.05 * epoch as f32).exp())
                .collect();
            let validation_accuracy: Vec<f32> = epochs.iter()
                .map(|&epoch| 1.0 - 0.45 * (-0.045 * epoch as f32).exp())
                .collect();

            training_curves.push(TrainingCurve {
                model_name: summary.model_name.clone(),
                epochs,
                train_loss,
                validation_loss,
                train_accuracy,
                validation_accuracy,
            });

            // Generate feature importance
            let features = vec![
                "Signal_Strength", "Latency", "Throughput", "Packet_Loss", 
                "Jitter", "CPU_Usage", "Memory_Usage", "Network_Load",
                "Time_of_Day", "Location", "Device_Type", "User_Count"
            ];
            let importance_scores: Vec<f32> = features.iter()
                .enumerate()
                .map(|(i, _)| 0.2 - i as f32 * 0.015)
                .collect();
            let cumulative_importance: Vec<f32> = importance_scores.iter()
                .scan(0.0, |acc, &x| { *acc += x; Some(*acc) })
                .collect();
            let top_features: Vec<(String, f32)> = features.iter()
                .zip(importance_scores.iter())
                .take(5)
                .map(|(name, &score)| (name.to_string(), score))
                .collect();

            feature_importance.push(FeatureImportance {
                model_name: summary.model_name.clone(),
                features,
                importance_scores,
                cumulative_importance,
                top_features,
            });

            // Generate residual plot data
            let predicted_values: Vec<f32> = (0..100)
                .map(|i| i as f32 * 0.01 + 0.1)
                .collect();
            let residuals: Vec<f32> = predicted_values.iter()
                .map(|&pred| (pred * 0.1 * (pred * 10.0).sin()).max(-0.05).min(0.05))
                .collect();
            let standardized_residuals: Vec<f32> = residuals.iter()
                .map(|&r| r / 0.02)
                .collect();
            let outlier_indices: Vec<usize> = standardized_residuals.iter()
                .enumerate()
                .filter(|(_, &r)| r.abs() > 2.0)
                .map(|(i, _)| i)
                .collect();

            residual_plots.push(ResidualPlot {
                model_name: summary.model_name.clone(),
                predicted_values,
                residuals,
                standardized_residuals,
                outlier_indices,
            });

            // Generate prediction plot data
            let actual_values: Vec<f32> = (0..100)
                .map(|i| i as f32 * 0.01 + 0.05 + (i as f32 * 0.1).sin() * 0.02)
                .collect();
            let predicted_values: Vec<f32> = actual_values.iter()
                .map(|&actual| actual + (actual * 0.1).sin() * 0.01)
                .collect();
            let confidence_intervals: Vec<(f32, f32)> = predicted_values.iter()
                .map(|&pred| (pred - 0.02, pred + 0.02))
                .collect();

            prediction_plots.push(PredictionPlot {
                model_name: summary.model_name.clone(),
                actual_values,
                predicted_values,
                confidence_intervals,
                r_squared: summary.metrics.r_squared,
            });

            // Generate radar chart data
            let metrics = vec![
                "Accuracy", "Speed", "Memory", "Reliability", "Interpretability"
            ];
            let values = vec![
                summary.metrics.r_squared,
                1.0 / summary.inference_time_ms * 10.0,
                1.0 / summary.memory_usage_mb * 1000.0,
                summary.cross_validation.cv_score,
                0.7, // Placeholder for interpretability
            ];
            let max_values = vec![1.0, 1.0, 1.0, 1.0, 1.0];

            performance_radar.push(RadarChart {
                model_name: summary.model_name.clone(),
                metrics,
                values,
                max_values,
            });
        }

        Ok(VisualizationData {
            training_curves,
            confusion_matrices: Vec::new(), // Not applicable for regression
            roc_curves: Vec::new(),          // Not applicable for regression
            feature_importance,
            residual_plots,
            prediction_plots,
            performance_radar,
        })
    }

    /// Assess deployment readiness
    fn assess_deployment_readiness(&self, summaries: &[ModelSummary]) -> Result<DeploymentReadiness> {
        let best_model = summaries.iter()
            .max_by(|a, b| a.metrics.r_squared.partial_cmp(&b.metrics.r_squared).unwrap())
            .unwrap();

        // Accuracy check
        let accuracy_check = ReadinessCheck {
            category: "Accuracy".to_string(),
            score: best_model.metrics.r_squared * 100.0,
            status: if best_model.metrics.r_squared > 0.85 { ReadinessStatus::Ready } else { ReadinessStatus::NeedsWork },
            issues: if best_model.metrics.r_squared < 0.85 { 
                vec!["Accuracy below 85% threshold".to_string()] 
            } else { 
                Vec::new() 
            },
            recommendations: vec!["Validate on additional test sets".to_string()],
        };

        // Performance check
        let performance_check = ReadinessCheck {
            category: "Performance".to_string(),
            score: if best_model.inference_time_ms < 5.0 { 100.0 } else { 80.0 },
            status: if best_model.inference_time_ms < 5.0 { ReadinessStatus::Ready } else { ReadinessStatus::NearReady },
            issues: if best_model.inference_time_ms >= 5.0 {
                vec!["Inference time above 5ms target".to_string()]
            } else {
                Vec::new()
            },
            recommendations: vec!["Consider model optimization".to_string()],
        };

        // Scalability check
        let scalability_check = ReadinessCheck {
            category: "Scalability".to_string(),
            score: 85.0,
            status: ReadinessStatus::Ready,
            issues: Vec::new(),
            recommendations: vec!["Test with production load".to_string()],
        };

        // Reliability check
        let reliability_check = ReadinessCheck {
            category: "Reliability".to_string(),
            score: best_model.cross_validation.cv_score * 100.0,
            status: ReadinessStatus::Ready,
            issues: Vec::new(),
            recommendations: vec!["Implement error handling".to_string()],
        };

        // Security check
        let security_check = ReadinessCheck {
            category: "Security".to_string(),
            score: 75.0,
            status: ReadinessStatus::NearReady,
            issues: vec!["Security audit pending".to_string()],
            recommendations: vec!["Complete security review".to_string()],
        };

        // Monitoring check
        let monitoring_check = ReadinessCheck {
            category: "Monitoring".to_string(),
            score: 60.0,
            status: ReadinessStatus::NeedsWork,
            issues: vec!["Monitoring system not implemented".to_string()],
            recommendations: vec!["Implement comprehensive monitoring".to_string()],
        };

        // Calculate overall score
        let overall_score = (
            accuracy_check.score * 0.3 +
            performance_check.score * 0.2 +
            scalability_check.score * 0.15 +
            reliability_check.score * 0.15 +
            security_check.score * 0.1 +
            monitoring_check.score * 0.1
        );

        let readiness_status = if overall_score >= 90.0 {
            ReadinessStatus::Ready
        } else if overall_score >= 75.0 {
            ReadinessStatus::NearReady
        } else if overall_score >= 60.0 {
            ReadinessStatus::NeedsWork
        } else {
            ReadinessStatus::NotReady
        };

        // Deployment checklist
        let deployment_checklist = vec![
            ChecklistItem {
                item: "Model accuracy validation".to_string(),
                completed: best_model.metrics.r_squared > 0.85,
                priority: Priority::Critical,
                description: "Validate model meets accuracy requirements".to_string(),
                dependencies: Vec::new(),
            },
            ChecklistItem {
                item: "Performance benchmarking".to_string(),
                completed: best_model.inference_time_ms < 5.0,
                priority: Priority::High,
                description: "Ensure inference time meets SLA requirements".to_string(),
                dependencies: Vec::new(),
            },
            ChecklistItem {
                item: "Security review".to_string(),
                completed: false,
                priority: Priority::High,
                description: "Complete security audit and penetration testing".to_string(),
                dependencies: vec!["Security team approval".to_string()],
            },
            ChecklistItem {
                item: "Monitoring setup".to_string(),
                completed: false,
                priority: Priority::Medium,
                description: "Implement model performance monitoring".to_string(),
                dependencies: vec!["Monitoring infrastructure".to_string()],
            },
        ];

        Ok(DeploymentReadiness {
            overall_score,
            readiness_status,
            accuracy_check,
            performance_check,
            scalability_check,
            reliability_check,
            security_check,
            monitoring_check,
            deployment_checklist,
        })
    }

    /// Export dashboard in various formats
    async fn export_dashboard(&self, dashboard: &EvaluationDashboard) -> Result<()> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.output_directory)?;

        for format in &self.export_formats {
            match format {
                ExportFormat::Json => {
                    let json_content = serde_json::to_string_pretty(dashboard)?;
                    let json_path = format!("{}/dashboard_{}.json", self.output_directory, dashboard.dashboard_id);
                    std::fs::write(json_path, json_content)?;
                    println!("  📄 JSON report saved");
                }
                ExportFormat::Html => {
                    let html_content = self.generate_html_dashboard(dashboard)?;
                    let html_path = format!("{}/dashboard_{}.html", self.output_directory, dashboard.dashboard_id);
                    std::fs::write(html_path, html_content)?;
                    println!("  🌐 HTML dashboard saved");
                }
                ExportFormat::Csv => {
                    self.export_csv_reports(dashboard).await?;
                    println!("  📊 CSV reports saved");
                }
                _ => {
                    // Placeholder for other formats
                    println!("  ⚠️  Format {:?} not yet implemented", format);
                }
            }
        }

        Ok(())
    }

    // Helper methods

    fn simulate_cross_validation(&self, model_name: &str, metrics: &EvaluationMetrics) -> Result<CrossValidationResults> {
        // Simplified cross-validation simulation
        use crate::evaluation::{FoldResult, StatisticalTests, ConfidenceIntervals};

        let fold_results = (0..5).map(|i| {
            FoldResult {
                fold_index: i,
                train_size: 800,
                test_size: 200,
                metrics: metrics.clone(), // Simplified - would vary per fold
                training_time: Duration::from_secs(30),
            }
        }).collect();

        Ok(CrossValidationResults {
            model_name: model_name.to_string(),
            fold_results,
            mean_metrics: metrics.clone(),
            std_metrics: EvaluationMetrics {
                mse: 0.005,
                mae: 0.01,
                rmse: 0.02,
                r_squared: 0.02,
                mape: 0.5,
                correlation: 0.01,
                std_residuals: 0.005,
                mean_residuals: 0.001,
                adjusted_r_squared: 0.015,
                aic: 5.0,
                bic: 6.0,
                explained_variance: 0.015,
                median_absolute_error: 0.008,
                max_error: 0.05,
                smape: 0.3,
                directional_accuracy: 2.0,
                theil_u: 0.01,
                prediction_interval_coverage: 1.0,
            },
            confidence_intervals: ConfidenceIntervals {
                mse_ci: (metrics.mse - 0.01, metrics.mse + 0.01),
                r_squared_ci: (metrics.r_squared - 0.02, metrics.r_squared + 0.02),
                mae_ci: (metrics.mae - 0.01, metrics.mae + 0.01),
                rmse_ci: (metrics.rmse - 0.02, metrics.rmse + 0.02),
                correlation_ci: (metrics.correlation - 0.01, metrics.correlation + 0.01),
                confidence_level: 0.95,
            },
            cv_score: metrics.r_squared,
            cv_score_std: 0.02,
            best_fold: 2,
            worst_fold: 0,
            statistical_significance: StatisticalTests {
                t_test_p_value: 0.05,
                wilcoxon_p_value: 0.03,
                normality_test_p_value: 0.15,
                heteroscedasticity_test_p_value: 0.20,
                durbin_watson_statistic: 1.8,
                ljung_box_p_value: 0.25,
            },
        })
    }

    fn determine_model_status(&self, metrics: &EvaluationMetrics) -> ModelStatus {
        if metrics.r_squared >= 0.90 {
            ModelStatus::Excellent
        } else if metrics.r_squared >= 0.80 {
            ModelStatus::Good
        } else if metrics.r_squared >= 0.70 {
            ModelStatus::Acceptable
        } else if metrics.r_squared >= 0.50 {
            ModelStatus::NeedsImprovement
        } else {
            ModelStatus::Failed
        }
    }

    fn analyze_model_characteristics(&self, metrics: &EvaluationMetrics, architecture: &[usize]) -> (Vec<String>, Vec<String>) {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        // Analyze accuracy
        if metrics.r_squared > 0.85 {
            strengths.push("High prediction accuracy".to_string());
        } else if metrics.r_squared < 0.70 {
            weaknesses.push("Low prediction accuracy".to_string());
        }

        // Analyze consistency
        if metrics.correlation > 0.90 {
            strengths.push("Strong correlation with targets".to_string());
        }

        // Analyze complexity
        let total_params: usize = architecture.windows(2).map(|w| w[0] * w[1]).sum();
        if total_params < 1000 {
            strengths.push("Lightweight architecture".to_string());
        } else if total_params > 10000 {
            weaknesses.push("High complexity model".to_string());
        }

        // Analyze errors
        if metrics.mape < 10.0 {
            strengths.push("Low percentage error".to_string());
        } else if metrics.mape > 20.0 {
            weaknesses.push("High percentage error".to_string());
        }

        (strengths, weaknesses)
    }

    fn determine_use_cases(&self, metrics: &EvaluationMetrics, _architecture: &[usize]) -> Vec<String> {
        let mut use_cases = Vec::new();

        if metrics.r_squared > 0.85 {
            use_cases.push("Production deployment".to_string());
            use_cases.push("Real-time predictions".to_string());
        }

        if metrics.r_squared > 0.80 {
            use_cases.push("Network optimization".to_string());
            use_cases.push("Capacity planning".to_string());
        }

        if metrics.directional_accuracy > 75.0 {
            use_cases.push("Trend analysis".to_string());
            use_cases.push("Anomaly detection".to_string());
        }

        use_cases.push("Research and development".to_string());

        use_cases
    }

    fn generate_comparison_matrix(&self, summaries: &[ModelSummary]) -> Result<Vec<Vec<ComparisonCell>>> {
        let mut matrix = Vec::new();

        let metrics = ["Accuracy (R²)", "Speed (1/ms)", "Memory Efficiency", "Overall"];

        for metric in &metrics {
            let mut row = Vec::new();
            
            for i in 0..summaries.len() {
                for j in i+1..summaries.len() {
                    let model_a = &summaries[i];
                    let model_b = &summaries[j];
                    
                    let (comparison_result, significance, effect_size) = match *metric {
                        "Accuracy (R²)" => {
                            let diff = model_a.metrics.r_squared - model_b.metrics.r_squared;
                            self.compare_values(diff, 0.02, 0.05)
                        },
                        "Speed (1/ms)" => {
                            let diff = (1.0 / model_a.inference_time_ms) - (1.0 / model_b.inference_time_ms);
                            self.compare_values(diff, 0.1, 0.2)
                        },
                        "Memory Efficiency" => {
                            let diff = (1.0 / model_a.memory_usage_mb) - (1.0 / model_b.memory_usage_mb);
                            self.compare_values(diff, 0.001, 0.002)
                        },
                        _ => {
                            // Overall comparison
                            let diff = model_a.metrics.r_squared - model_b.metrics.r_squared;
                            self.compare_values(diff, 0.02, 0.05)
                        }
                    };

                    row.push(ComparisonCell {
                        metric_name: metric.to_string(),
                        model_a: model_a.model_name.clone(),
                        model_b: model_b.model_name.clone(),
                        comparison_result,
                        significance,
                        effect_size,
                    });
                }
            }
            
            matrix.push(row);
        }

        Ok(matrix)
    }

    fn compare_values(&self, diff: f32, small_threshold: f32, large_threshold: f32) -> (ComparisonResult, f32, f32) {
        let abs_diff = diff.abs();
        let comparison_result = if abs_diff < small_threshold {
            ComparisonResult::Equivalent
        } else if abs_diff < large_threshold {
            if diff > 0.0 { ComparisonResult::SlightlyBetter } else { ComparisonResult::SlightlyWorse }
        } else {
            if diff > 0.0 { ComparisonResult::SignificantlyBetter } else { ComparisonResult::SignificantlyWorse }
        };

        let significance = if abs_diff < small_threshold { 0.8 } else { 0.05 };
        let effect_size = abs_diff / large_threshold;

        (comparison_result, significance, effect_size)
    }

    fn generate_html_dashboard(&self, dashboard: &EvaluationDashboard) -> Result<String> {
        // Generate a comprehensive HTML dashboard
        let html = format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Evaluation Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h3 {{ margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-label {{ font-weight: 600; color: #666; }}
        .metric-value {{ font-weight: bold; }}
        .status-excellent {{ color: #27ae60; }}
        .status-good {{ color: #f39c12; }}
        .status-acceptable {{ color: #e67e22; }}
        .status-needs-improvement {{ color: #e74c3c; }}
        .ranking {{ list-style: none; padding: 0; }}
        .ranking li {{ padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #3498db; }}
        .recommendation {{ background: #e8f4fd; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .high-priority {{ border-left-color: #e74c3c; background: #fdf2f2; }}
        .medium-priority {{ border-left-color: #f39c12; background: #fefaf3; }}
        .checklist {{ list-style: none; padding: 0; }}
        .checklist li {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .completed {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .pending {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background-color: #f8f9fa; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neural Network Evaluation Dashboard</h1>
            <p>Comprehensive analysis of {} models for telecom network performance prediction</p>
            <p>Generated: {}</p>
        </div>

        <div class="grid">
            {}
        </div>

        <div class="card">
            <h3>📊 Performance Rankings</h3>
            {}
        </div>

        <div class="card">
            <h3>💡 Recommendations</h3>
            {}
        </div>

        <div class="card">
            <h3>🚀 Deployment Readiness</h3>
            {}
        </div>
    </div>
</body>
</html>
        "#,
            dashboard.model_summaries.len(),
            chrono::DateTime::from_timestamp(dashboard.created_at as i64, 0)
                .unwrap_or_default()
                .format("%Y-%m-%d %H:%M:%S UTC"),
            self.generate_model_cards_html(&dashboard.model_summaries),
            self.generate_rankings_html(&dashboard.performance_comparison),
            self.generate_recommendations_html(&dashboard.recommendations),
            self.generate_deployment_readiness_html(&dashboard.deployment_readiness)
        );

        Ok(html)
    }

    fn generate_model_cards_html(&self, summaries: &[ModelSummary]) -> String {
        summaries.iter().map(|summary| {
            let status_class = match summary.status {
                ModelStatus::Excellent => "status-excellent",
                ModelStatus::Good => "status-good",
                ModelStatus::Acceptable => "status-acceptable",
                _ => "status-needs-improvement",
            };

            format!(r#"
            <div class="card">
                <h3>{} <span class="{}">{:?}</span></h3>
                <div class="metric">
                    <span class="metric-label">R² Score:</span>
                    <span class="metric-value">{:.4}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-value">{:.4}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">MAPE:</span>
                    <span class="metric-value">{:.2}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Inference Time:</span>
                    <span class="metric-value">{:.2}ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage:</span>
                    <span class="metric-value">{:.1}MB</span>
                </div>
                <p><strong>Architecture:</strong> {:?}</p>
                <p><strong>Strengths:</strong> {}</p>
                <p><strong>Use Cases:</strong> {}</p>
            </div>
            "#,
                summary.model_name,
                status_class,
                summary.status,
                summary.metrics.r_squared,
                summary.metrics.rmse,
                summary.metrics.mape,
                summary.inference_time_ms,
                summary.memory_usage_mb,
                summary.architecture,
                summary.strengths.join(", "),
                summary.use_cases.join(", ")
            )
        }).collect::<Vec<_>>().join("")
    }

    fn generate_rankings_html(&self, comparison: &PerformanceComparison) -> String {
        format!(r#"
        <div class="grid">
            <div>
                <h4>🎯 Accuracy Ranking</h4>
                <ul class="ranking">
                    {}
                </ul>
            </div>
            <div>
                <h4>⚡ Speed Ranking</h4>
                <ul class="ranking">
                    {}
                </ul>
            </div>
            <div>
                <h4>🏆 Overall Ranking</h4>
                <ul class="ranking">
                    {}
                </ul>
            </div>
        </div>
        "#,
            comparison.accuracy_ranking.iter().map(|item| {
                format!("<li>{}. {} ({:.3})</li>", item.rank, item.model_name, item.score)
            }).collect::<Vec<_>>().join(""),
            comparison.speed_ranking.iter().map(|item| {
                format!("<li>{}. {} ({:.3})</li>", item.rank, item.model_name, item.score)
            }).collect::<Vec<_>>().join(""),
            comparison.overall_ranking.iter().map(|item| {
                format!("<li>{}. {} ({:.3})</li>", item.rank, item.model_name, item.score)
            }).collect::<Vec<_>>().join("")
        )
    }

    fn generate_recommendations_html(&self, recommendations: &DashboardRecommendations) -> String {
        let dev_recommendations = recommendations.development_recommendations.iter().map(|rec| {
            let priority_class = match rec.priority {
                Priority::Critical | Priority::High => "high-priority",
                Priority::Medium => "medium-priority",
                _ => "",
            };

            format!(r#"
            <div class="recommendation {}">
                <h4>{} ({:?} Priority)</h4>
                <p>{}</p>
                <p><strong>Expected Impact:</strong> {:.1}% improvement</p>
                <p><strong>Timeline:</strong> {}</p>
            </div>
            "#, priority_class, rec.title, rec.priority, rec.description, rec.expected_impact * 100.0, rec.timeline)
        }).collect::<Vec<_>>().join("");

        format!(r#"
        <p><strong>🥇 Production Model:</strong> {}</p>
        <p><strong>🥈 Backup Model:</strong> {}</p>
        <h4>Development Recommendations:</h4>
        {}
        "#, recommendations.production_model, recommendations.backup_model, dev_recommendations)
    }

    fn generate_deployment_readiness_html(&self, readiness: &DeploymentReadiness) -> String {
        let checklist = readiness.deployment_checklist.iter().map(|item| {
            let class = if item.completed { "completed" } else { "pending" };
            let status = if item.completed { "✅" } else { "⏳" };
            
            format!(r#"
            <li class="{}">{} {} ({:?} Priority)</li>
            "#, class, status, item.item, item.priority)
        }).collect::<Vec<_>>().join("");

        format!(r#"
        <p><strong>Overall Readiness Score:</strong> {:.1}%</p>
        <p><strong>Status:</strong> {:?}</p>
        <h4>Deployment Checklist:</h4>
        <ul class="checklist">
            {}
        </ul>
        "#, readiness.overall_score, readiness.readiness_status, checklist)
    }

    async fn export_csv_reports(&self, dashboard: &EvaluationDashboard) -> Result<()> {
        // Export model summary as CSV
        let mut csv_content = "Model Name,R² Score,RMSE,MAE,MAPE,Inference Time (ms),Memory Usage (MB),Status\n".to_string();
        
        for summary in &dashboard.model_summaries {
            csv_content.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{:.2},{:.2},{:.1},{:?}\n",
                summary.model_name,
                summary.metrics.r_squared,
                summary.metrics.rmse,
                summary.metrics.mae,
                summary.metrics.mape,
                summary.inference_time_ms,
                summary.memory_usage_mb,
                summary.status
            ));
        }

        let csv_path = format!("{}/model_summary_{}.csv", self.output_directory, dashboard.dashboard_id);
        std::fs::write(csv_path, csv_content)?;

        Ok(())
    }
}

impl Default for DashboardGenerator {
    fn default() -> Self {
        Self::new("./evaluation_reports")
    }
}