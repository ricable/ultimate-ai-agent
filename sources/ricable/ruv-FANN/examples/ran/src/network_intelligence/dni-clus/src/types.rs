//! Data types and structures for DNI-CLUS-01
//!
//! This module defines all the core data types used throughout the cell profiling system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// PRB utilization data for a cell over a 24-hour period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrbUtilizationData {
    pub cell_id: String,
    pub timestamp: DateTime<Utc>,
    pub hourly_utilization: Vec<f64>, // 24 values, 0.0-1.0 range
    pub metadata: CellMetadata,
}

/// Cell metadata for contextual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMetadata {
    pub site_id: String,
    pub technology: String,
    pub frequency_band: String,
    pub cell_type: String,
    pub environment: String,
    pub location: GeographicLocation,
    pub additional_attributes: HashMap<String, String>,
}

/// Geographic location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    pub algorithm: ClusteringAlgorithm,
    pub num_clusters: usize,
    pub eps: f64,
    pub min_samples: usize,
    pub auto_tune: bool,
    pub feature_selection: Vec<String>,
    pub distance_metric: DistanceMetric,
    pub normalization: NormalizationMethod,
    pub convergence_threshold: f64,
    pub max_iterations: usize,
    pub random_seed: Option<u64>,
}

/// Supported clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    KMeans,
    DBSCAN,
    HierarchicalClustering,
    GaussianMixture,
    SpectralClustering,
    AffinityPropagation,
    MeanShift,
    OPTICS,
    Hybrid,
}

/// Distance metrics for clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Hamming,
    Jaccard,
    Correlation,
}

/// Normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    ZScore,
    MinMax,
    RobustScaling,
    UnitVector,
    None,
}

/// Feature vector for clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub cell_id: String,
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<HashMap<String, String>>,
}

/// Clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    pub cluster_id: usize,
    pub cluster_name: String,
    pub cell_ids: Vec<String>,
    pub centroid: Vec<f64>,
    pub characteristics: ClusterCharacteristics,
    pub quality_metrics: ClusterQualityMetrics,
    pub behavioral_pattern: BehavioralPattern,
}

/// Cluster characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCharacteristics {
    pub size: usize,
    pub density: f64,
    pub cohesion: f64,
    pub separation: f64,
    pub stability: f64,
    pub dominant_features: Vec<String>,
    pub utilization_statistics: UtilizationStatistics,
    pub temporal_patterns: TemporalPatterns,
    pub geographical_distribution: GeographicalDistribution,
}

/// Utilization statistics for a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<String, f64>, // "p25", "p75", "p90", "p95", "p99"
    pub coefficient_of_variation: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Temporal patterns in utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    pub peak_hours: Vec<usize>,
    pub low_hours: Vec<usize>,
    pub pattern_type: PatternType,
    pub seasonality_strength: f64,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub autocorrelation: f64,
    pub cyclic_patterns: Vec<CyclicPattern>,
}

/// Pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    MorningPeak,
    EveningPeak,
    DoublePeak,
    Flat,
    Irregular,
    NightPeak,
    BusinessHours,
    WeekendPattern,
    Unknown,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Fluctuating,
    Unknown,
}

/// Cyclic patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclicPattern {
    pub period_hours: usize,
    pub amplitude: f64,
    pub phase: f64,
    pub confidence: f64,
}

/// Geographical distribution of cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicalDistribution {
    pub centroid_location: GeographicLocation,
    pub bounding_box: BoundingBox,
    pub spread_radius_km: f64,
    pub density_per_km2: f64,
    pub environment_distribution: HashMap<String, usize>,
    pub technology_distribution: HashMap<String, usize>,
}

/// Bounding box for geographical area
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_latitude: f64,
    pub max_latitude: f64,
    pub min_longitude: f64,
    pub max_longitude: f64,
}

/// Cluster quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterQualityMetrics {
    pub silhouette_score: f64,
    pub calinski_harabasz_score: f64,
    pub davies_bouldin_score: f64,
    pub within_cluster_sum_of_squares: f64,
    pub between_cluster_sum_of_squares: f64,
    pub dunn_index: f64,
    pub xie_beni_index: f64,
    pub partition_coefficient: f64,
}

/// Behavioral pattern of a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    pub primary_pattern: PatternType,
    pub secondary_patterns: Vec<PatternType>,
    pub confidence: f64,
    pub predictability: f64,
    pub anomaly_likelihood: f64,
    pub seasonal_consistency: f64,
    pub load_variability: f64,
    pub usage_efficiency: f64,
}

/// Cell profile generated from clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellProfile {
    pub cell_id: String,
    pub cluster_assignment: ClusterAssignment,
    pub behavioral_profile: BehavioralProfile,
    pub performance_indicators: PerformanceIndicators,
    pub anomaly_detection: AnomalyDetection,
    pub recommendations: Vec<ProfileRecommendation>,
    pub last_updated: DateTime<Utc>,
    pub confidence_level: f64,
}

/// Cluster assignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAssignment {
    pub cluster_id: usize,
    pub cluster_name: String,
    pub assignment_confidence: f64,
    pub distance_to_centroid: f64,
    pub membership_probability: f64,
    pub stability_score: f64,
}

/// Behavioral profile of a cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralProfile {
    pub typical_pattern: Vec<f64>, // 24-hour pattern
    pub pattern_type: PatternType,
    pub load_characteristics: LoadCharacteristics,
    pub usage_patterns: UsagePatterns,
    pub performance_trends: PerformanceTrends,
    pub comparative_analysis: ComparativeAnalysis,
}

/// Load characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCharacteristics {
    pub peak_load: f64,
    pub average_load: f64,
    pub minimum_load: f64,
    pub load_factor: f64,
    pub utilization_efficiency: f64,
    pub capacity_headroom: f64,
    pub overload_risk: f64,
}

/// Usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePatterns {
    pub busy_hour_patterns: Vec<BusyHourPattern>,
    pub day_of_week_patterns: Vec<f64>,
    pub monthly_trends: Vec<f64>,
    pub seasonal_variations: Vec<f64>,
    pub special_event_indicators: Vec<SpecialEventIndicator>,
}

/// Busy hour pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusyHourPattern {
    pub hour: usize,
    pub utilization: f64,
    pub consistency: f64,
    pub duration_minutes: usize,
}

/// Special event indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialEventIndicator {
    pub event_type: String,
    pub likelihood: f64,
    pub impact_factor: f64,
    pub duration_estimate: String,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub utilization_trend: TrendDirection,
    pub trend_strength: f64,
    pub forecast_horizon_days: usize,
    pub predicted_growth_rate: f64,
    pub saturation_forecast: Option<DateTime<Utc>>,
}

/// Comparative analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub peer_comparison: PeerComparison,
    pub cluster_deviation: f64,
    pub performance_ranking: PerformanceRanking,
    pub optimization_potential: OptimizationPotential,
}

/// Peer comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerComparison {
    pub similar_cells: Vec<String>,
    pub performance_percentile: f64,
    pub efficiency_ranking: usize,
    pub outlier_status: OutlierStatus,
}

/// Outlier status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierStatus {
    Normal,
    MildOutlier,
    ModerateOutlier,
    ExtremeOutlier,
}

/// Performance ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRanking {
    pub overall_rank: usize,
    pub efficiency_rank: usize,
    pub utilization_rank: usize,
    pub stability_rank: usize,
    pub total_cells: usize,
}

/// Optimization potential
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPotential {
    pub capacity_optimization: f64,
    pub load_balancing_benefit: f64,
    pub energy_efficiency_gain: f64,
    pub quality_improvement: f64,
    pub cost_reduction_potential: f64,
}

/// Performance indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIndicators {
    pub utilization_efficiency: f64,
    pub load_balance_score: f64,
    pub predictability_score: f64,
    pub stability_score: f64,
    pub quality_score: f64,
    pub anomaly_score: f64,
    pub optimization_score: f64,
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub anomaly_score: f64,
    pub anomaly_type: Option<AnomalyType>,
    pub anomaly_severity: AnomallySeverity,
    pub detected_anomalies: Vec<DetectedAnomaly>,
    pub root_cause_analysis: Option<RootCauseAnalysis>,
}

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    UtilizationSpike,
    UtilizationDrop,
    PatternDeviation,
    PerformanceDegradation,
    CapacityIssue,
    InterfereceDetected,
    ConfigurationAnomaly,
    HardwareIssue,
    Unknown,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomallySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    pub timestamp: DateTime<Utc>,
    pub anomaly_type: AnomalyType,
    pub severity: AnomallySeverity,
    pub confidence: f64,
    pub description: String,
    pub impact_assessment: String,
    pub suggested_actions: Vec<String>,
}

/// Root cause analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseAnalysis {
    pub primary_cause: String,
    pub contributing_factors: Vec<String>,
    pub confidence: f64,
    pub analysis_method: String,
    pub supporting_evidence: Vec<String>,
}

/// Profile recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRecommendation {
    pub recommendation_id: Uuid,
    pub category: RecommendationCategory,
    pub priority: Priority,
    pub title: String,
    pub description: String,
    pub rationale: String,
    pub expected_impact: ExpectedImpact,
    pub implementation_complexity: Complexity,
    pub resource_requirements: Vec<String>,
    pub estimated_timeframe: String,
    pub success_metrics: Vec<String>,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    CapacityOptimization,
    LoadBalancing,
    EnergyEfficiency,
    QualityImprovement,
    CostReduction,
    MaintenanceOptimization,
    ConfigurationTuning,
    NetworkPlanning,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Expected impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImpact {
    pub performance_improvement: f64,
    pub cost_savings: f64,
    pub efficiency_gain: f64,
    pub quality_enhancement: f64,
    pub risk_reduction: f64,
}

/// Implementation complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Complexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Strategic insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicInsight {
    pub insight_id: Uuid,
    pub category: String,
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub impact: ExpectedImpact,
    pub confidence: f64,
    pub affected_clusters: Vec<usize>,
    pub affected_cells: Vec<String>,
    pub key_findings: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub estimated_benefit: EstimatedBenefit,
    pub implementation_roadmap: ImplementationRoadmap,
    pub generated_at: DateTime<Utc>,
}

/// Estimated benefit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedBenefit {
    pub performance_improvement_percent: f64,
    pub cost_savings_percent: f64,
    pub efficiency_gain_percent: f64,
    pub capacity_increase_percent: f64,
    pub energy_savings_percent: f64,
    pub roi_months: f64,
}

/// Implementation roadmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationRoadmap {
    pub phases: Vec<ImplementationPhase>,
    pub total_duration_weeks: usize,
    pub dependencies: Vec<String>,
    pub risks: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// Implementation phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationPhase {
    pub phase_name: String,
    pub duration_weeks: usize,
    pub activities: Vec<String>,
    pub deliverables: Vec<String>,
    pub resources_required: Vec<String>,
}

/// Update strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStrategy {
    Incremental,
    FullRetrain,
    Adaptive,
}

/// Update result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    pub strategy: UpdateStrategy,
    pub cells_updated: usize,
    pub clusters_modified: usize,
    pub processing_time_ms: f64,
    pub success: bool,
    pub message: String,
}

/// Clustering analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringAnalysisResult {
    pub session_id: String,
    pub agent_id: Uuid,
    pub clusters: Vec<ClusteringResult>,
    pub profiles: Vec<CellProfile>,
    pub features: Vec<FeatureVector>,
    pub metrics: ClusteringMetrics,
    pub config: ClusteringConfig,
    pub prb_data_summary: PrbDataSummary,
    pub analysis_metadata: AnalysisMetadata,
}

/// Clustering metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringMetrics {
    pub silhouette_score: f64,
    pub calinski_harabasz_score: f64,
    pub davies_bouldin_score: f64,
    pub adjusted_rand_index: f64,
    pub normalized_mutual_info: f64,
    pub inertia: f64,
    pub num_clusters: usize,
    pub num_cells: usize,
    pub convergence_iterations: usize,
    pub processing_time_ms: f64,
}

/// PRB data summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrbDataSummary {
    pub total_cells: usize,
    pub total_hours: usize,
    pub avg_utilization: f64,
    pub max_utilization: f64,
    pub min_utilization: f64,
    pub date_range: DateRange,
}

/// Date range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub cells_analyzed: usize,
    pub features_extracted: usize,
    pub processing_time_ms: f64,
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    Summary,
    Detailed,
    Strategic,
    Operational,
    Technical,
    Executive,
}

/// Profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    pub report_type: ReportType,
    pub generated_at: DateTime<Utc>,
    pub agent_id: Uuid,
    pub session_id: String,
    pub analysis_results: ClusteringAnalysisResult,
    pub strategic_insights: Vec<StrategicInsight>,
    pub recommendations: Vec<Recommendation>,
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_analyses: usize,
    pub avg_processing_time_ms: f64,
    pub avg_clustering_quality: f64,
    pub success_rate: f64,
    pub uptime_percentage: f64,
    pub last_updated: DateTime<Utc>,
}

/// Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub impact: ExpectedImpact,
    pub actions: Vec<String>,
    pub estimated_benefit: EstimatedBenefit,
    pub implementation_timeframe: String,
    pub resource_requirements: Vec<String>,
}

// Implementation of common traits and utility functions
impl ClusteringConfig {
    pub fn algorithm_name(&self) -> &str {
        match self.algorithm {
            ClusteringAlgorithm::KMeans => "K-Means",
            ClusteringAlgorithm::DBSCAN => "DBSCAN",
            ClusteringAlgorithm::HierarchicalClustering => "Hierarchical",
            ClusteringAlgorithm::GaussianMixture => "Gaussian Mixture",
            ClusteringAlgorithm::SpectralClustering => "Spectral",
            ClusteringAlgorithm::AffinityPropagation => "Affinity Propagation",
            ClusteringAlgorithm::MeanShift => "Mean Shift",
            ClusteringAlgorithm::OPTICS => "OPTICS",
            ClusteringAlgorithm::Hybrid => "Hybrid",
        }
    }
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: 5,
            eps: 0.5,
            min_samples: 3,
            auto_tune: true,
            feature_selection: vec![
                "mean_utilization".to_string(),
                "std_utilization".to_string(),
                "peak_hours".to_string(),
                "temporal_pattern".to_string(),
                "variability".to_string(),
            ],
            distance_metric: DistanceMetric::Euclidean,
            normalization: NormalizationMethod::ZScore,
            convergence_threshold: 1e-4,
            max_iterations: 300,
            random_seed: Some(42),
        }
    }
}

impl std::fmt::Display for PatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternType::MorningPeak => write!(f, "Morning Peak"),
            PatternType::EveningPeak => write!(f, "Evening Peak"),
            PatternType::DoublePeak => write!(f, "Double Peak"),
            PatternType::Flat => write!(f, "Flat"),
            PatternType::Irregular => write!(f, "Irregular"),
            PatternType::NightPeak => write!(f, "Night Peak"),
            PatternType::BusinessHours => write!(f, "Business Hours"),
            PatternType::WeekendPattern => write!(f, "Weekend Pattern"),
            PatternType::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Priority::Low => write!(f, "Low"),
            Priority::Medium => write!(f, "Medium"),
            Priority::High => write!(f, "High"),
            Priority::Critical => write!(f, "Critical"),
            Priority::Emergency => write!(f, "Emergency"),
        }
    }
}