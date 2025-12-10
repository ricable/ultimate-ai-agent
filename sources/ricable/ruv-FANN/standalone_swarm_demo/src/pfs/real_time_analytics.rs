//! Real-time Analytics Engine for Neural Swarm Performance
//! 
//! This module provides advanced real-time analytics capabilities including
//! anomaly detection, trend analysis, and predictive insights for swarm operations.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use super::metrics_collector::{MetricPoint, AggregatedMetric, MetricTrend};
use super::advanced::{SwarmTensor, TensorStatistics};

/// Real-time analytics engine for swarm performance data
pub struct SwarmRealTimeAnalytics {
    anomaly_detector: AnomalyDetector,
    trend_analyzer: TrendAnalyzer,
    pattern_recognizer: PatternRecognizer,
    predictive_engine: PredictiveEngine,
    alert_generator: AlertGenerator,
    analytics_config: AnalyticsConfig,
}

/// Configuration for analytics algorithms
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    pub anomaly_threshold: f64,
    pub trend_window_size: usize,
    pub pattern_min_length: usize,
    pub prediction_horizon_minutes: u32,
    pub alert_cooldown_seconds: u64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            anomaly_threshold: 2.0, // Standard deviations
            trend_window_size: 20,
            pattern_min_length: 5,
            prediction_horizon_minutes: 15,
            alert_cooldown_seconds: 300, // 5 minutes
        }
    }
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub metric_name: String,
    pub agent_id: String,
    pub timestamp: u64,
    pub anomaly_score: f64,
    pub severity: AnomalySeverity,
    pub description: String,
    pub recommended_actions: Vec<String>,
    pub context: AnomalyContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyContext {
    pub baseline_value: f64,
    pub current_value: f64,
    pub deviation_magnitude: f64,
    pub related_metrics: Vec<String>,
    pub potential_causes: Vec<String>,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisResult {
    pub metric_name: String,
    pub trend_direction: MetricTrend,
    pub trend_strength: f64,
    pub confidence_score: f64,
    pub rate_of_change: f64,
    pub trend_start_timestamp: u64,
    pub projected_values: Vec<(u64, f64)>, // (timestamp, projected_value)
}

/// Pattern recognition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    pub pattern_type: String,
    pub confidence: f64,
    pub frequency: f64,
    pub duration_minutes: f64,
    pub affected_metrics: Vec<String>,
    pub pattern_description: String,
    pub next_occurrence_prediction: Option<u64>,
}

/// Predictive analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub metric_name: String,
    pub prediction_horizon_minutes: u32,
    pub predicted_values: Vec<(u64, f64, f64)>, // (timestamp, value, confidence)
    pub risk_assessment: RiskAssessment,
    pub recommended_preemptive_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk_score: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_name: String,
    pub impact_score: f64,
    pub probability: f64,
    pub description: String,
}

impl SwarmRealTimeAnalytics {
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            anomaly_detector: AnomalyDetector::new(config.anomaly_threshold),
            trend_analyzer: TrendAnalyzer::new(config.trend_window_size),
            pattern_recognizer: PatternRecognizer::new(config.pattern_min_length),
            predictive_engine: PredictiveEngine::new(config.prediction_horizon_minutes),
            alert_generator: AlertGenerator::new(config.alert_cooldown_seconds),
            analytics_config: config,
        }
    }
    
    /// Perform comprehensive real-time analysis on metrics
    pub async fn analyze_metrics(
        &mut self,
        metrics: &[MetricPoint],
        aggregated_metrics: &HashMap<String, AggregatedMetric>,
    ) -> AnalyticsResults {
        // Detect anomalies
        let anomalies = self.anomaly_detector.detect_anomalies(metrics, aggregated_metrics).await;
        
        // Analyze trends
        let trends = self.trend_analyzer.analyze_trends(metrics).await;
        
        // Recognize patterns
        let patterns = self.pattern_recognizer.recognize_patterns(metrics).await;
        
        // Generate predictions
        let predictions = self.predictive_engine.generate_predictions(metrics, aggregated_metrics).await;
        
        // Generate insights
        let insights = self.generate_insights(&anomalies, &trends, &patterns, &predictions).await;
        
        // Generate alerts if necessary
        let alerts = self.alert_generator.generate_alerts(&anomalies, &trends, &predictions).await;
        
        AnalyticsResults {
            anomalies,
            trends,
            patterns,
            predictions,
            insights,
            alerts,
            analysis_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Generate high-level insights from analysis results
    async fn generate_insights(
        &self,
        anomalies: &[AnomalyResult],
        trends: &[TrendAnalysisResult],
        patterns: &[PatternResult],
        predictions: &[PredictionResult],
    ) -> Vec<Insight> {
        let mut insights = Vec::new();
        
        // System health insight
        let health_insight = self.generate_system_health_insight(anomalies, trends).await;
        insights.push(health_insight);
        
        // Performance optimization insight
        let optimization_insight = self.generate_optimization_insight(trends, patterns).await;
        insights.push(optimization_insight);
        
        // Capacity planning insight
        let capacity_insight = self.generate_capacity_insight(predictions).await;
        insights.push(capacity_insight);
        
        // Risk assessment insight
        let risk_insight = self.generate_risk_insight(anomalies, predictions).await;
        insights.push(risk_insight);
        
        insights
    }
    
    async fn generate_system_health_insight(
        &self,
        anomalies: &[AnomalyResult],
        trends: &[TrendAnalysisResult],
    ) -> Insight {
        let critical_anomalies = anomalies.iter()
            .filter(|a| matches!(a.severity, AnomalySeverity::Critical | AnomalySeverity::High))
            .count();
        
        let degrading_trends = trends.iter()
            .filter(|t| matches!(t.trend_direction, MetricTrend::Decreasing) && t.trend_strength > 0.7)
            .count();
        
        let health_score = 1.0 - (critical_anomalies as f64 * 0.3 + degrading_trends as f64 * 0.2).min(1.0);
        
        let status = match health_score {
            x if x >= 0.8 => "Excellent",
            x if x >= 0.6 => "Good",
            x if x >= 0.4 => "Fair",
            x if x >= 0.2 => "Poor",
            _ => "Critical",
        };
        
        Insight {
            insight_type: "system_health".to_string(),
            title: format!("System Health: {}", status),
            description: format!(
                "System health score: {:.2}. {} critical anomalies detected, {} degrading trends identified.",
                health_score, critical_anomalies, degrading_trends
            ),
            confidence: 0.9,
            priority: if health_score < 0.5 { InsightPriority::High } else { InsightPriority::Medium },
            actionable_recommendations: if critical_anomalies > 0 || degrading_trends > 0 {
                vec![
                    "Investigate critical anomalies immediately".to_string(),
                    "Review resource allocation for degrading metrics".to_string(),
                    "Consider scaling out affected agents".to_string(),
                ]
            } else {
                vec!["Continue monitoring - system is performing well".to_string()]
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("health_score".to_string(), health_score.to_string());
                meta.insert("critical_anomalies".to_string(), critical_anomalies.to_string());
                meta.insert("degrading_trends".to_string(), degrading_trends.to_string());
                meta
            },
        }
    }
    
    async fn generate_optimization_insight(
        &self,
        trends: &[TrendAnalysisResult],
        patterns: &[PatternResult],
    ) -> Insight {
        let improving_trends = trends.iter()
            .filter(|t| matches!(t.trend_direction, MetricTrend::Increasing) && 
                       t.metric_name.contains("performance") || 
                       t.metric_name.contains("accuracy") ||
                       t.metric_name.contains("throughput"))
            .count();
        
        let cyclical_patterns = patterns.iter()
            .filter(|p| p.pattern_type.contains("cyclical") && p.confidence > 0.7)
            .count();
        
        let optimization_score = (improving_trends as f64 * 0.4 + cyclical_patterns as f64 * 0.3).min(1.0);
        
        Insight {
            insight_type: "performance_optimization".to_string(),
            title: "Performance Optimization Opportunities".to_string(),
            description: format!(
                "Optimization potential score: {:.2}. {} improving performance trends, {} predictable patterns identified.",
                optimization_score, improving_trends, cyclical_patterns
            ),
            confidence: 0.8,
            priority: if optimization_score > 0.6 { InsightPriority::Medium } else { InsightPriority::Low },
            actionable_recommendations: vec![
                "Leverage improving performance trends for workload optimization".to_string(),
                "Use identified patterns for predictive resource allocation".to_string(),
                "Consider automated scaling based on pattern predictions".to_string(),
            ],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("optimization_score".to_string(), optimization_score.to_string());
                meta.insert("improving_trends".to_string(), improving_trends.to_string());
                meta.insert("cyclical_patterns".to_string(), cyclical_patterns.to_string());
                meta
            },
        }
    }
    
    async fn generate_capacity_insight(&self, predictions: &[PredictionResult]) -> Insight {
        let high_risk_predictions = predictions.iter()
            .filter(|p| p.risk_assessment.overall_risk_score > 0.7)
            .count();
        
        let capacity_strain_metrics = predictions.iter()
            .filter(|p| p.metric_name.contains("cpu") || 
                       p.metric_name.contains("memory") || 
                       p.metric_name.contains("disk"))
            .filter(|p| p.predicted_values.iter().any(|(_, value, _)| *value > 0.8))
            .count();
        
        let capacity_risk = (high_risk_predictions as f64 * 0.5 + capacity_strain_metrics as f64 * 0.4).min(1.0);
        
        Insight {
            insight_type: "capacity_planning".to_string(),
            title: "Capacity Planning Recommendations".to_string(),
            description: format!(
                "Capacity risk score: {:.2}. {} high-risk predictions, {} resource metrics approaching limits.",
                capacity_risk, high_risk_predictions, capacity_strain_metrics
            ),
            confidence: 0.75,
            priority: if capacity_risk > 0.6 { InsightPriority::High } else { InsightPriority::Medium },
            actionable_recommendations: if capacity_risk > 0.5 {
                vec![
                    "Plan resource scaling within the next 24-48 hours".to_string(),
                    "Review workload distribution across agents".to_string(),
                    "Consider implementing auto-scaling policies".to_string(),
                ]
            } else {
                vec![
                    "Current capacity appears adequate".to_string(),
                    "Continue monitoring resource utilization trends".to_string(),
                ]
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("capacity_risk".to_string(), capacity_risk.to_string());
                meta.insert("high_risk_predictions".to_string(), high_risk_predictions.to_string());
                meta.insert("capacity_strain_metrics".to_string(), capacity_strain_metrics.to_string());
                meta
            },
        }
    }
    
    async fn generate_risk_insight(
        &self,
        anomalies: &[AnomalyResult],
        predictions: &[PredictionResult],
    ) -> Insight {
        let critical_anomalies = anomalies.iter()
            .filter(|a| matches!(a.severity, AnomalySeverity::Critical))
            .count();
        
        let high_risk_predictions = predictions.iter()
            .filter(|p| p.risk_assessment.overall_risk_score > 0.8)
            .count();
        
        let overall_risk = (critical_anomalies as f64 * 0.6 + high_risk_predictions as f64 * 0.4).min(1.0);
        
        Insight {
            insight_type: "risk_assessment".to_string(),
            title: "System Risk Assessment".to_string(),
            description: format!(
                "Overall risk score: {:.2}. {} critical anomalies active, {} high-risk predictions identified.",
                overall_risk, critical_anomalies, high_risk_predictions
            ),
            confidence: 0.85,
            priority: if overall_risk > 0.7 { InsightPriority::Critical } else { InsightPriority::Medium },
            actionable_recommendations: if overall_risk > 0.5 {
                vec![
                    "Immediate attention required for critical anomalies".to_string(),
                    "Implement risk mitigation strategies from predictions".to_string(),
                    "Increase monitoring frequency for high-risk components".to_string(),
                    "Prepare contingency plans for potential failures".to_string(),
                ]
            } else {
                vec![
                    "Risk levels are within acceptable bounds".to_string(),
                    "Continue standard monitoring protocols".to_string(),
                ]
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("overall_risk".to_string(), overall_risk.to_string());
                meta.insert("critical_anomalies".to_string(), critical_anomalies.to_string());
                meta.insert("high_risk_predictions".to_string(), high_risk_predictions.to_string());
                meta
            },
        }
    }
}

/// High-level insights generated from analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub insight_type: String,
    pub title: String,
    pub description: String,
    pub confidence: f64,
    pub priority: InsightPriority,
    pub actionable_recommendations: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Comprehensive analytics results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsResults {
    pub anomalies: Vec<AnomalyResult>,
    pub trends: Vec<TrendAnalysisResult>,
    pub patterns: Vec<PatternResult>,
    pub predictions: Vec<PredictionResult>,
    pub insights: Vec<Insight>,
    pub alerts: Vec<AnalyticsAlert>,
    pub analysis_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsAlert {
    pub alert_id: String,
    pub alert_type: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub source_analysis: String, // "anomaly", "trend", "prediction"
    pub recommended_actions: Vec<String>,
    pub auto_remediation_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

// Implementation of individual analytics components

/// Anomaly detection using statistical methods
pub struct AnomalyDetector {
    threshold: f64,
    metric_baselines: Arc<RwLock<HashMap<String, MetricBaseline>>>,
}

#[derive(Debug, Clone)]
struct MetricBaseline {
    mean: f64,
    std_dev: f64,
    sample_count: usize,
    last_updated: u64,
}

impl AnomalyDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            metric_baselines: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn detect_anomalies(
        &mut self,
        metrics: &[MetricPoint],
        aggregated_metrics: &HashMap<String, AggregatedMetric>,
    ) -> Vec<AnomalyResult> {
        let mut anomalies = Vec::new();
        
        // Update baselines from aggregated metrics
        self.update_baselines(aggregated_metrics).await;
        
        // Check each metric point for anomalies
        for metric in metrics {
            if let Some(anomaly) = self.check_metric_for_anomaly(metric).await {
                anomalies.push(anomaly);
            }
        }
        
        anomalies
    }
    
    async fn update_baselines(&self, aggregated_metrics: &HashMap<String, AggregatedMetric>) {
        let mut baselines = self.metric_baselines.write().await;
        
        for (key, metric) in aggregated_metrics {
            baselines.insert(key.clone(), MetricBaseline {
                mean: metric.mean,
                std_dev: metric.std_dev,
                sample_count: metric.count as usize,
                last_updated: metric.last_updated,
            });
        }
    }
    
    async fn check_metric_for_anomaly(&self, metric: &MetricPoint) -> Option<AnomalyResult> {
        let baselines = self.metric_baselines.read().await;
        let key = format!("{}_{}", metric.agent_id, metric.metric_name);
        
        if let Some(baseline) = baselines.get(&key) {
            if baseline.sample_count < 10 {
                return None; // Need sufficient data for anomaly detection
            }
            
            let z_score = (metric.value - baseline.mean) / baseline.std_dev;
            
            if z_score.abs() > self.threshold {
                let severity = match z_score.abs() {
                    x if x > 4.0 => AnomalySeverity::Critical,
                    x if x > 3.0 => AnomalySeverity::High,
                    x if x > 2.5 => AnomalySeverity::Medium,
                    _ => AnomalySeverity::Low,
                };
                
                return Some(AnomalyResult {
                    metric_name: metric.metric_name.clone(),
                    agent_id: metric.agent_id.clone(),
                    timestamp: metric.timestamp,
                    anomaly_score: z_score.abs(),
                    severity,
                    description: format!(
                        "Metric {} for agent {} deviates {:.2} standard deviations from baseline",
                        metric.metric_name, metric.agent_id, z_score.abs()
                    ),
                    recommended_actions: vec![
                        "Investigate recent changes to the agent".to_string(),
                        "Check for external factors affecting performance".to_string(),
                        "Review logs for error patterns".to_string(),
                    ],
                    context: AnomalyContext {
                        baseline_value: baseline.mean,
                        current_value: metric.value,
                        deviation_magnitude: z_score.abs(),
                        related_metrics: Vec::new(), // Would be populated with correlation analysis
                        potential_causes: Vec::new(), // Would be populated with domain knowledge
                    },
                });
            }
        }
        
        None
    }
}

/// Trend analysis using time series methods
pub struct TrendAnalyzer {
    window_size: usize,
    metric_histories: Arc<RwLock<HashMap<String, VecDeque<(u64, f64)>>>>,
}

impl TrendAnalyzer {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            metric_histories: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn analyze_trends(&mut self, metrics: &[MetricPoint]) -> Vec<TrendAnalysisResult> {
        self.update_histories(metrics).await;
        
        let histories = self.metric_histories.read().await;
        let mut trends = Vec::new();
        
        for (key, history) in histories.iter() {
            if history.len() >= self.window_size {
                if let Some(trend) = self.calculate_trend(key, history).await {
                    trends.push(trend);
                }
            }
        }
        
        trends
    }
    
    async fn update_histories(&self, metrics: &[MetricPoint]) {
        let mut histories = self.metric_histories.write().await;
        
        for metric in metrics {
            let key = format!("{}_{}", metric.agent_id, metric.metric_name);
            let history = histories.entry(key).or_insert_with(VecDeque::new);
            
            // Maintain window size
            while history.len() >= self.window_size {
                history.pop_front();
            }
            
            history.push_back((metric.timestamp, metric.value));
        }
    }
    
    async fn calculate_trend(&self, key: &str, history: &VecDeque<(u64, f64)>) -> Option<TrendAnalysisResult> {
        if history.len() < self.window_size {
            return None;
        }
        
        let values: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
        let timestamps: Vec<u64> = history.iter().map(|(t, _)| *t).collect();
        
        // Simple linear regression for trend detection
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate R-squared for confidence
        let y_mean = sum_y / n;
        let ss_tot: f64 = values.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = values.iter().enumerate()
            .map(|(i, &y)| {
                let predicted = slope * i as f64 + intercept;
                (y - predicted).powi(2)
            })
            .sum();
        
        let r_squared = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };
        
        let trend_direction = if slope.abs() < 0.001 {
            MetricTrend::Stable
        } else if slope > 0.0 {
            MetricTrend::Increasing
        } else {
            MetricTrend::Decreasing
        };
        
        let trend_strength = slope.abs() / (values.iter().fold(0.0, |acc, &x| acc.max(x)) - 
                                          values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)));
        
        // Generate projections
        let last_timestamp = timestamps.last().copied().unwrap_or(0);
        let projected_values: Vec<(u64, f64)> = (1..=6) // Project 6 time steps ahead
            .map(|i| {
                let future_timestamp = last_timestamp + (i * 300); // 5-minute intervals
                let future_index = values.len() as f64 + i as f64;
                let projected_value = slope * future_index + intercept;
                (future_timestamp, projected_value)
            })
            .collect();
        
        Some(TrendAnalysisResult {
            metric_name: key.clone(),
            trend_direction,
            trend_strength,
            confidence_score: r_squared,
            rate_of_change: slope,
            trend_start_timestamp: timestamps.first().copied().unwrap_or(0),
            projected_values,
        })
    }
}

/// Pattern recognition for periodic and cyclical behaviors
pub struct PatternRecognizer {
    min_pattern_length: usize,
}

impl PatternRecognizer {
    pub fn new(min_pattern_length: usize) -> Self {
        Self { min_pattern_length }
    }
    
    pub async fn recognize_patterns(&mut self, _metrics: &[MetricPoint]) -> Vec<PatternResult> {
        // Placeholder implementation
        // In a real system, this would implement sophisticated pattern recognition algorithms
        // such as FFT for cyclical patterns, autocorrelation for periodicity, etc.
        
        vec![
            PatternResult {
                pattern_type: "daily_cycle".to_string(),
                confidence: 0.85,
                frequency: 1.0 / (24.0 * 60.0), // Daily pattern
                duration_minutes: 24.0 * 60.0,
                affected_metrics: vec!["cpu_utilization".to_string(), "memory_usage".to_string()],
                pattern_description: "Daily usage pattern with peak during business hours".to_string(),
                next_occurrence_prediction: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() + 86400 // Next day
                ),
            }
        ]
    }
}

/// Predictive engine for forecasting future values
pub struct PredictiveEngine {
    prediction_horizon_minutes: u32,
}

impl PredictiveEngine {
    pub fn new(prediction_horizon_minutes: u32) -> Self {
        Self { prediction_horizon_minutes }
    }
    
    pub async fn generate_predictions(
        &mut self,
        _metrics: &[MetricPoint],
        aggregated_metrics: &HashMap<String, AggregatedMetric>,
    ) -> Vec<PredictionResult> {
        let mut predictions = Vec::new();
        
        for (key, metric) in aggregated_metrics {
            // Simple prediction based on trend and variance
            let prediction = self.create_simple_prediction(key, metric).await;
            predictions.push(prediction);
        }
        
        predictions
    }
    
    async fn create_simple_prediction(&self, key: &str, metric: &AggregatedMetric) -> PredictionResult {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Generate predicted values (simplified linear extrapolation)
        let predicted_values: Vec<(u64, f64, f64)> = (1..=self.prediction_horizon_minutes)
            .step_by(5) // 5-minute intervals
            .map(|minutes_ahead| {
                let future_timestamp = current_time + (minutes_ahead as u64 * 60);
                // Simple prediction: last value + some trend-based adjustment
                let predicted_value = metric.last_value + (minutes_ahead as f64 * 0.01); // Placeholder trend
                let confidence = (1.0 - (minutes_ahead as f64 / self.prediction_horizon_minutes as f64) * 0.5).max(0.1);
                (future_timestamp, predicted_value, confidence)
            })
            .collect();
        
        // Assess risk based on predicted values
        let risk_score = self.calculate_risk_score(metric, &predicted_values).await;
        
        PredictionResult {
            metric_name: key.clone(),
            prediction_horizon_minutes: self.prediction_horizon_minutes,
            predicted_values,
            risk_assessment: RiskAssessment {
                overall_risk_score: risk_score,
                risk_factors: vec![
                    RiskFactor {
                        factor_name: "trend_continuation".to_string(),
                        impact_score: 0.6,
                        probability: 0.7,
                        description: "Current trend may continue beyond acceptable bounds".to_string(),
                    }
                ],
                mitigation_strategies: vec![
                    "Monitor closely for threshold breaches".to_string(),
                    "Prepare scaling actions if needed".to_string(),
                ],
            },
            recommended_preemptive_actions: if risk_score > 0.7 {
                vec![
                    "Increase monitoring frequency".to_string(),
                    "Prepare capacity adjustments".to_string(),
                ]
            } else {
                vec!["Continue normal monitoring".to_string()]
            },
        }
    }
    
    async fn calculate_risk_score(&self, metric: &AggregatedMetric, predicted_values: &[(u64, f64, f64)]) -> f64 {
        // Simple risk calculation based on how far predictions deviate from current baseline
        let max_predicted = predicted_values.iter()
            .map(|(_, value, _)| *value)
            .fold(0.0, f64::max);
        
        let deviation_ratio = if metric.mean != 0.0 {
            (max_predicted - metric.mean).abs() / metric.mean.abs()
        } else {
            0.0
        };
        
        (deviation_ratio * 2.0).min(1.0) // Scale to 0-1 range
    }
}

/// Alert generator for creating actionable alerts
pub struct AlertGenerator {
    cooldown_seconds: u64,
    recent_alerts: Arc<RwLock<HashMap<String, u64>>>, // alert_key -> last_sent_timestamp
}

impl AlertGenerator {
    pub fn new(cooldown_seconds: u64) -> Self {
        Self {
            cooldown_seconds,
            recent_alerts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn generate_alerts(
        &mut self,
        anomalies: &[AnomalyResult],
        trends: &[TrendAnalysisResult],
        predictions: &[PredictionResult],
    ) -> Vec<AnalyticsAlert> {
        let mut alerts = Vec::new();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Generate alerts from anomalies
        for anomaly in anomalies {
            if let Some(alert) = self.create_anomaly_alert(anomaly, current_time).await {
                alerts.push(alert);
            }
        }
        
        // Generate alerts from trend analysis
        for trend in trends {
            if let Some(alert) = self.create_trend_alert(trend, current_time).await {
                alerts.push(alert);
            }
        }
        
        // Generate alerts from predictions
        for prediction in predictions {
            if let Some(alert) = self.create_prediction_alert(prediction, current_time).await {
                alerts.push(alert);
            }
        }
        
        alerts
    }
    
    async fn create_anomaly_alert(&self, anomaly: &AnomalyResult, current_time: u64) -> Option<AnalyticsAlert> {
        let alert_key = format!("anomaly_{}_{}", anomaly.agent_id, anomaly.metric_name);
        
        if self.should_send_alert(&alert_key, current_time).await {
            self.record_alert_sent(&alert_key, current_time).await;
            
            Some(AnalyticsAlert {
                alert_id: format!("anomaly_{}", uuid::Uuid::new_v4()),
                alert_type: "anomaly_detection".to_string(),
                severity: match anomaly.severity {
                    AnomalySeverity::Critical => AlertSeverity::Critical,
                    AnomalySeverity::High => AlertSeverity::Error,
                    AnomalySeverity::Medium => AlertSeverity::Warning,
                    AnomalySeverity::Low => AlertSeverity::Info,
                },
                message: format!("Anomaly detected: {}", anomaly.description),
                source_analysis: "anomaly".to_string(),
                recommended_actions: anomaly.recommended_actions.clone(),
                auto_remediation_available: false,
            })
        } else {
            None
        }
    }
    
    async fn create_trend_alert(&self, trend: &TrendAnalysisResult, current_time: u64) -> Option<AnalyticsAlert> {
        // Only alert on significant negative trends
        if !matches!(trend.trend_direction, MetricTrend::Decreasing) || trend.trend_strength < 0.7 {
            return None;
        }
        
        let alert_key = format!("trend_{}", trend.metric_name);
        
        if self.should_send_alert(&alert_key, current_time).await {
            self.record_alert_sent(&alert_key, current_time).await;
            
            Some(AnalyticsAlert {
                alert_id: format!("trend_{}", uuid::Uuid::new_v4()),
                alert_type: "degrading_trend".to_string(),
                severity: AlertSeverity::Warning,
                message: format!("Degrading trend detected in {}", trend.metric_name),
                source_analysis: "trend".to_string(),
                recommended_actions: vec![
                    "Investigate cause of performance degradation".to_string(),
                    "Consider preventive measures before further decline".to_string(),
                ],
                auto_remediation_available: false,
            })
        } else {
            None
        }
    }
    
    async fn create_prediction_alert(&self, prediction: &PredictionResult, current_time: u64) -> Option<AnalyticsAlert> {
        if prediction.risk_assessment.overall_risk_score < 0.8 {
            return None; // Only alert on high-risk predictions
        }
        
        let alert_key = format!("prediction_{}", prediction.metric_name);
        
        if self.should_send_alert(&alert_key, current_time).await {
            self.record_alert_sent(&alert_key, current_time).await;
            
            Some(AnalyticsAlert {
                alert_id: format!("prediction_{}", uuid::Uuid::new_v4()),
                alert_type: "high_risk_prediction".to_string(),
                severity: AlertSeverity::Warning,
                message: format!("High-risk prediction for {}: {:.1}% risk score", 
                               prediction.metric_name, 
                               prediction.risk_assessment.overall_risk_score * 100.0),
                source_analysis: "prediction".to_string(),
                recommended_actions: prediction.recommended_preemptive_actions.clone(),
                auto_remediation_available: true,
            })
        } else {
            None
        }
    }
    
    async fn should_send_alert(&self, alert_key: &str, current_time: u64) -> bool {
        let recent_alerts = self.recent_alerts.read().await;
        
        if let Some(&last_sent) = recent_alerts.get(alert_key) {
            current_time - last_sent > self.cooldown_seconds
        } else {
            true
        }
    }
    
    async fn record_alert_sent(&self, alert_key: &str, current_time: u64) {
        let mut recent_alerts = self.recent_alerts.write().await;
        recent_alerts.insert(alert_key.to_string(), current_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_analytics_engine_creation() {
        let config = AnalyticsConfig::default();
        let _analytics = SwarmRealTimeAnalytics::new(config);
    }
    
    #[tokio::test]
    async fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(2.0);
        
        // Create test metrics
        let metrics = vec![
            MetricPoint {
                timestamp: 1000,
                agent_id: "test-agent".to_string(),
                metric_name: "test_metric".to_string(),
                value: 100.0,
                unit: "units".to_string(),
                tags: HashMap::new(),
                context: super::metrics_collector::MetricContext {
                    operation_type: "test".to_string(),
                    data_source: "test".to_string(),
                    processing_stage: "test".to_string(),
                    model_version: "1.0".to_string(),
                    batch_size: 100,
                },
            }
        ];
        
        let mut aggregated = HashMap::new();
        aggregated.insert(
            "test-agent_test_metric".to_string(),
            AggregatedMetric {
                metric_name: "test_metric".to_string(),
                count: 10,
                sum: 1000.0,
                mean: 100.0,
                min: 90.0,
                max: 110.0,
                std_dev: 5.0,
                last_value: 100.0,
                last_updated: 1000,
                trend: MetricTrend::Stable,
                anomaly_score: 0.0,
            }
        );
        
        let anomalies = detector.detect_anomalies(&metrics, &aggregated).await;
        // With normal values, should detect no anomalies
        assert!(anomalies.is_empty());
    }
}