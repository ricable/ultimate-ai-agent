//! Optimization targets and constraint definitions for RAN systems

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    error::{RanError, RanResult},
    metrics::KpiType,
    GeoCoordinate,
};

/// Optimization objective types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Maximize network throughput
    MaximizeThroughput,
    /// Minimize latency
    MinimizeLatency,
    /// Maximize coverage
    MaximizeCoverage,
    /// Minimize interference
    MinimizeInterference,
    /// Maximize energy efficiency
    MaximizeEnergyEfficiency,
    /// Maximize spectral efficiency
    MaximizeSpectralEfficiency,
    /// Minimize operational costs
    MinimizeOperationalCosts,
    /// Maximize Quality of Experience
    MaximizeQoE,
    /// Minimize handover failures
    MinimizeHandoverFailures,
    /// Maximize resource utilization
    MaximizeResourceUtilization,
    /// Minimize network congestion
    MinimizeNetworkCongestion,
    /// Maximize user satisfaction
    MaximizeUserSatisfaction,
}

impl OptimizationObjective {
    /// Get the primary KPI associated with this objective
    pub fn primary_kpi(&self) -> KpiType {
        match self {
            OptimizationObjective::MaximizeThroughput => KpiType::AverageCellThroughput,
            OptimizationObjective::MinimizeLatency => KpiType::UserPlaneLatency,
            OptimizationObjective::MaximizeCoverage => KpiType::CoverageProbability,
            OptimizationObjective::MinimizeInterference => KpiType::SignalQuality,
            OptimizationObjective::MaximizeEnergyEfficiency => KpiType::EnergyEfficiency,
            OptimizationObjective::MaximizeSpectralEfficiency => KpiType::SpectralEfficiency,
            OptimizationObjective::MinimizeOperationalCosts => KpiType::ResourceUtilization,
            OptimizationObjective::MaximizeQoE => KpiType::MeanOpinionScore,
            OptimizationObjective::MinimizeHandoverFailures => KpiType::HandoverSuccessRate,
            OptimizationObjective::MaximizeResourceUtilization => KpiType::ResourceUtilization,
            OptimizationObjective::MinimizeNetworkCongestion => KpiType::CellLoad,
            OptimizationObjective::MaximizeUserSatisfaction => KpiType::MeanOpinionScore,
        }
    }

    /// Check if this is a maximization objective
    pub fn is_maximization(&self) -> bool {
        match self {
            OptimizationObjective::MaximizeThroughput |
            OptimizationObjective::MaximizeCoverage |
            OptimizationObjective::MaximizeEnergyEfficiency |
            OptimizationObjective::MaximizeSpectralEfficiency |
            OptimizationObjective::MaximizeQoE |
            OptimizationObjective::MaximizeResourceUtilization |
            OptimizationObjective::MaximizeUserSatisfaction => true,
            
            OptimizationObjective::MinimizeLatency |
            OptimizationObjective::MinimizeInterference |
            OptimizationObjective::MinimizeOperationalCosts |
            OptimizationObjective::MinimizeHandoverFailures |
            OptimizationObjective::MinimizeNetworkCongestion => false,
        }
    }

    /// Get typical target value for this objective
    pub fn typical_target(&self) -> f64 {
        match self {
            OptimizationObjective::MaximizeThroughput => 500.0, // Mbps
            OptimizationObjective::MinimizeLatency => 10.0, // ms
            OptimizationObjective::MaximizeCoverage => 95.0, // %
            OptimizationObjective::MinimizeInterference => 20.0, // dB SINR
            OptimizationObjective::MaximizeEnergyEfficiency => 1e8, // bit/J
            OptimizationObjective::MaximizeSpectralEfficiency => 5.0, // bit/s/Hz
            OptimizationObjective::MinimizeOperationalCosts => 70.0, // % utilization
            OptimizationObjective::MaximizeQoE => 4.0, // MOS
            OptimizationObjective::MinimizeHandoverFailures => 98.0, // % success rate
            OptimizationObjective::MaximizeResourceUtilization => 80.0, // %
            OptimizationObjective::MinimizeNetworkCongestion => 60.0, // % load
            OptimizationObjective::MaximizeUserSatisfaction => 4.2, // MOS
        }
    }
}

/// Optimization constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationConstraint {
    /// Numerical range constraint
    Range {
        name: String,
        description: String,
        min_value: f64,
        max_value: f64,
        current_value: Option<f64>,
    },
    /// Threshold constraint
    Threshold {
        name: String,
        description: String,
        threshold: f64,
        operator: ComparisonOperator,
        current_value: Option<f64>,
    },
    /// Resource constraint
    ResourceLimit {
        resource_type: String,
        max_consumption: f64,
        current_consumption: f64,
    },
    /// Quality of Service constraint
    QoSRequirement {
        service_type: String,
        min_quality: f64,
        current_quality: Option<f64>,
    },
    /// Geographic constraint
    Geographic {
        description: String,
        bounds: GeographicBounds,
    },
    /// Temporal constraint
    Temporal {
        description: String,
        time_window: TimeWindow,
    },
    /// Regulatory constraint
    Regulatory {
        regulation: String,
        requirement: String,
        compliance_level: f64,
    },
    /// Interference constraint
    Interference {
        max_interference_level: f64,
        current_level: Option<f64>,
        protection_ratio: f64,
    },
}

impl OptimizationConstraint {
    /// Create a new range constraint
    pub fn new(name: String, description: String, min_value: f64, max_value: f64) -> Self {
        OptimizationConstraint::Range {
            name,
            description,
            min_value,
            max_value,
            current_value: None,
        }
    }

    /// Create a new threshold constraint
    pub fn threshold(
        name: String,
        description: String,
        threshold: f64,
        operator: ComparisonOperator,
    ) -> Self {
        OptimizationConstraint::Threshold {
            name,
            description,
            threshold,
            operator,
            current_value: None,
        }
    }

    /// Check if constraint is satisfied
    pub fn is_satisfied(&self) -> Option<bool> {
        match self {
            OptimizationConstraint::Range { min_value, max_value, current_value, .. } => {
                current_value.map(|val| val >= *min_value && val <= *max_value)
            }
            OptimizationConstraint::Threshold { threshold, operator, current_value, .. } => {
                current_value.map(|val| operator.evaluate(val, *threshold))
            }
            OptimizationConstraint::ResourceLimit { max_consumption, current_consumption, .. } => {
                Some(*current_consumption <= *max_consumption)
            }
            OptimizationConstraint::QoSRequirement { min_quality, current_quality, .. } => {
                current_quality.map(|qual| qual >= *min_quality)
            }
            _ => None, // Other constraint types need additional context
        }
    }

    /// Get constraint violation amount (if any)
    pub fn violation_amount(&self) -> Option<f64> {
        match self {
            OptimizationConstraint::Range { min_value, max_value, current_value, .. } => {
                current_value.and_then(|val| {
                    if val < *min_value {
                        Some(*min_value - val)
                    } else if val > *max_value {
                        Some(val - *max_value)
                    } else {
                        None
                    }
                })
            }
            OptimizationConstraint::Threshold { threshold, operator, current_value, .. } => {
                current_value.and_then(|val| {
                    if !operator.evaluate(val, *threshold) {
                        Some((val - *threshold).abs())
                    } else {
                        None
                    }
                })
            }
            OptimizationConstraint::ResourceLimit { max_consumption, current_consumption, .. } => {
                if *current_consumption > *max_consumption {
                    Some(*current_consumption - *max_consumption)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Update current value for the constraint
    pub fn update_current_value(&mut self, value: f64) {
        match self {
            OptimizationConstraint::Range { current_value, .. } |
            OptimizationConstraint::Threshold { current_value, .. } => {
                *current_value = Some(value);
            }
            OptimizationConstraint::ResourceLimit { current_consumption, .. } => {
                *current_consumption = value;
            }
            OptimizationConstraint::QoSRequirement { current_quality, .. } => {
                *current_quality = Some(value);
            }
            OptimizationConstraint::Interference { current_level, .. } => {
                *current_level = Some(value);
            }
            _ => {} // Other types don't have updatable current values
        }
    }
}

/// Comparison operators for constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Equal to (with tolerance)
    Equal,
    /// Not equal to
    NotEqual,
}

impl ComparisonOperator {
    /// Evaluate the comparison
    pub fn evaluate(&self, left: f64, right: f64) -> bool {
        const EPSILON: f64 = 1e-10;
        
        match self {
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::LessThanOrEqual => left <= right,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::Equal => (left - right).abs() < EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= EPSILON,
        }
    }
}

/// Geographic bounds for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicBounds {
    /// Minimum latitude
    pub min_latitude: f64,
    /// Maximum latitude
    pub max_latitude: f64,
    /// Minimum longitude
    pub min_longitude: f64,
    /// Maximum longitude
    pub max_longitude: f64,
    /// Optional altitude constraints
    pub altitude_range: Option<(f64, f64)>,
}

impl GeographicBounds {
    /// Check if a coordinate is within bounds
    pub fn contains(&self, coord: &GeoCoordinate) -> bool {
        coord.latitude >= self.min_latitude &&
        coord.latitude <= self.max_latitude &&
        coord.longitude >= self.min_longitude &&
        coord.longitude <= self.max_longitude &&
        self.altitude_range.map_or(true, |(min_alt, max_alt)| {
            coord.altitude.map_or(true, |alt| alt >= min_alt && alt <= max_alt)
        })
    }

    /// Calculate area in square kilometers
    pub fn area_km2(&self) -> f64 {
        let lat_diff = self.max_latitude - self.min_latitude;
        let lon_diff = self.max_longitude - self.min_longitude;
        
        // Approximate calculation (more accurate methods would use ellipsoid)
        let lat_km = lat_diff * 111.32; // km per degree latitude
        let avg_lat = (self.min_latitude + self.max_latitude) / 2.0;
        let lon_km = lon_diff * 111.32 * avg_lat.to_radians().cos();
        
        lat_km * lon_km
    }
}

/// Time window for temporal constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
    /// Recurrence pattern (if applicable)
    pub recurrence: Option<RecurrencePattern>,
}

impl TimeWindow {
    /// Check if current time is within the window
    pub fn is_active(&self) -> bool {
        let now = Utc::now();
        now >= self.start && now <= self.end
    }

    /// Get duration in seconds
    pub fn duration_seconds(&self) -> i64 {
        (self.end - self.start).num_seconds()
    }
}

/// Recurrence patterns for time windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecurrencePattern {
    /// Daily recurrence
    Daily,
    /// Weekly recurrence
    Weekly,
    /// Monthly recurrence
    Monthly,
    /// Custom interval in seconds
    Custom { interval_seconds: u64 },
}

/// Optimization target combining objectives and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTarget {
    /// Target identifier
    pub id: Uuid,
    /// Target name
    pub name: String,
    /// Primary optimization objective
    pub primary_objective: OptimizationObjective,
    /// Secondary objectives with weights
    pub secondary_objectives: Vec<(OptimizationObjective, f64)>,
    /// Constraints that must be satisfied
    pub constraints: Vec<OptimizationConstraint>,
    /// Target scope (cells, areas, etc.)
    pub scope: OptimizationScope,
    /// Priority level (1-10, higher is more important)
    pub priority: u8,
    /// Target weight in multi-objective optimization
    pub weight: f64,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Target status
    pub status: OptimizationStatus,
    /// Performance metrics
    pub metrics: OptimizationMetrics,
}

impl OptimizationTarget {
    /// Create a new optimization target
    pub fn new(
        name: String,
        primary_objective: OptimizationObjective,
        scope: OptimizationScope,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            primary_objective,
            secondary_objectives: Vec::new(),
            constraints: Vec::new(),
            scope,
            priority: 5, // Medium priority
            weight: 1.0,
            created_at: now,
            last_updated: now,
            status: OptimizationStatus::Active,
            metrics: OptimizationMetrics::default(),
        }
    }

    /// Add a secondary objective
    pub fn add_secondary_objective(&mut self, objective: OptimizationObjective, weight: f64) {
        self.secondary_objectives.push((objective, weight));
        self.last_updated = Utc::now();
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: OptimizationConstraint) {
        self.constraints.push(constraint);
        self.last_updated = Utc::now();
    }

    /// Check if all constraints are satisfied
    pub fn are_constraints_satisfied(&self) -> bool {
        self.constraints.iter()
            .all(|constraint| constraint.is_satisfied().unwrap_or(true))
    }

    /// Get constraint violations
    pub fn get_constraint_violations(&self) -> Vec<&OptimizationConstraint> {
        self.constraints.iter()
            .filter(|constraint| constraint.is_satisfied() == Some(false))
            .collect()
    }

    /// Calculate total violation penalty
    pub fn violation_penalty(&self) -> f64 {
        self.constraints.iter()
            .filter_map(|constraint| constraint.violation_amount())
            .sum()
    }

    /// Calculate multi-objective score
    pub fn calculate_objective_score(&self, kpi_values: &HashMap<KpiType, f64>) -> RanResult<f64> {
        let primary_kpi = self.primary_objective.primary_kpi();
        let primary_value = kpi_values.get(&primary_kpi)
            .ok_or_else(|| RanError::optimization(
                format!("Missing KPI value for primary objective: {:?}", primary_kpi)
            ))?;

        // Normalize primary objective value
        let primary_score = self.normalize_objective_value(
            self.primary_objective,
            *primary_value,
        );

        // Calculate secondary objective scores
        let secondary_score: f64 = self.secondary_objectives.iter()
            .filter_map(|(objective, weight)| {
                let kpi = objective.primary_kpi();
                kpi_values.get(&kpi).map(|value| {
                    weight * self.normalize_objective_value(*objective, *value)
                })
            })
            .sum();

        // Combine scores with primary objective having implicit weight of 1.0
        let total_weight = 1.0 + self.secondary_objectives.iter().map(|(_, w)| w).sum::<f64>();
        let combined_score = (primary_score + secondary_score) / total_weight;

        // Apply constraint penalty
        let penalty = self.violation_penalty();
        Ok((combined_score - penalty).max(0.0))
    }

    /// Normalize objective value to 0-1 scale
    fn normalize_objective_value(&self, objective: OptimizationObjective, value: f64) -> f64 {
        let target = objective.typical_target();
        
        if objective.is_maximization() {
            (value / target).min(1.0)
        } else {
            (target / value.max(0.001)).min(1.0) // Avoid division by zero
        }
    }

    /// Update optimization metrics
    pub fn update_metrics(&mut self, score: f64, improvement: f64) {
        self.metrics.current_score = score;
        self.metrics.improvement_rate = improvement;
        self.metrics.last_evaluation = Utc::now();
        self.metrics.evaluation_count += 1;
        
        if score > self.metrics.best_score {
            self.metrics.best_score = score;
            self.metrics.best_score_time = Some(Utc::now());
        }
        
        self.last_updated = Utc::now();
    }
}

/// Optimization scope definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationScope {
    /// Single cell optimization
    Cell { cell_id: Uuid },
    /// Multiple cells optimization
    Cells { cell_ids: Vec<Uuid> },
    /// Geographic area optimization
    Geographic { bounds: GeographicBounds },
    /// Network slice optimization
    NetworkSlice { slice_id: String },
    /// Service type optimization
    ServiceType { service_type: String },
    /// User group optimization
    UserGroup { group_id: String },
    /// Entire network optimization
    Network,
}

/// Optimization status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStatus {
    /// Target is active and being optimized
    Active,
    /// Target is paused temporarily
    Paused,
    /// Target is completed successfully
    Completed,
    /// Target failed during optimization
    Failed,
    /// Target is pending activation
    Pending,
    /// Target is archived
    Archived,
}

/// Optimization performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    /// Current objective score
    pub current_score: f64,
    /// Best score achieved
    pub best_score: f64,
    /// Timestamp of best score
    pub best_score_time: Option<DateTime<Utc>>,
    /// Rate of improvement
    pub improvement_rate: f64,
    /// Number of evaluations performed
    pub evaluation_count: u64,
    /// Last evaluation timestamp
    pub last_evaluation: DateTime<Utc>,
    /// Convergence status
    pub converged: bool,
    /// Number of constraint violations
    pub constraint_violations: u32,
}

impl Default for OptimizationMetrics {
    fn default() -> Self {
        Self {
            current_score: 0.0,
            best_score: 0.0,
            best_score_time: None,
            improvement_rate: 0.0,
            evaluation_count: 0,
            last_evaluation: Utc::now(),
            converged: false,
            constraint_violations: 0,
        }
    }
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveConfig {
    /// List of optimization targets
    pub targets: Vec<OptimizationTarget>,
    /// Optimization algorithm to use
    pub algorithm: OptimizationAlgorithm,
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Population size for evolutionary algorithms
    pub population_size: Option<u32>,
    /// Learning rate for gradient-based algorithms
    pub learning_rate: Option<f64>,
    /// Constraint handling method
    pub constraint_handling: ConstraintHandling,
}

/// Optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Genetic Algorithm
    GeneticAlgorithm,
    /// Particle Swarm Optimization
    ParticleSwarm,
    /// Simulated Annealing
    SimulatedAnnealing,
    /// Gradient Descent
    GradientDescent,
    /// Bayesian Optimization
    BayesianOptimization,
    /// Multi-objective Genetic Algorithm (NSGA-II)
    NSGAII,
    /// Multi-objective Particle Swarm Optimization
    MOPSO,
}

/// Constraint handling methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintHandling {
    /// Penalty function method
    PenaltyFunction,
    /// Barrier method
    BarrierMethod,
    /// Augmented Lagrangian
    AugmentedLagrangian,
    /// Feasibility rules
    FeasibilityRules,
    /// Constraint domination
    ConstraintDomination,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_objective() {
        let objective = OptimizationObjective::MaximizeThroughput;
        assert_eq!(objective.primary_kpi(), KpiType::AverageCellThroughput);
        assert!(objective.is_maximization());
        assert!(objective.typical_target() > 0.0);
    }

    #[test]
    fn test_optimization_constraint() {
        let mut constraint = OptimizationConstraint::new(
            "throughput".to_string(),
            "Minimum throughput requirement".to_string(),
            100.0,
            1000.0,
        );

        // Initially no current value
        assert_eq!(constraint.is_satisfied(), None);

        // Update with valid value
        constraint.update_current_value(500.0);
        assert_eq!(constraint.is_satisfied(), Some(true));

        // Update with invalid value
        constraint.update_current_value(50.0);
        assert_eq!(constraint.is_satisfied(), Some(false));
        assert_eq!(constraint.violation_amount(), Some(50.0));
    }

    #[test]
    fn test_comparison_operator() {
        assert!(ComparisonOperator::LessThan.evaluate(5.0, 10.0));
        assert!(!ComparisonOperator::LessThan.evaluate(10.0, 5.0));
        assert!(ComparisonOperator::Equal.evaluate(5.0, 5.0));
        assert!(ComparisonOperator::GreaterThanOrEqual.evaluate(10.0, 10.0));
    }

    #[test]
    fn test_geographic_bounds() {
        let bounds = GeographicBounds {
            min_latitude: 52.0,
            max_latitude: 53.0,
            min_longitude: 13.0,
            max_longitude: 14.0,
            altitude_range: None,
        };

        let coord_inside = GeoCoordinate::new(52.5, 13.5, None);
        let coord_outside = GeoCoordinate::new(54.0, 13.5, None);

        assert!(bounds.contains(&coord_inside));
        assert!(!bounds.contains(&coord_outside));
        assert!(bounds.area_km2() > 0.0);
    }

    #[test]
    fn test_optimization_target() {
        let scope = OptimizationScope::Cell { cell_id: Uuid::new_v4() };
        let mut target = OptimizationTarget::new(
            "Test Target".to_string(),
            OptimizationObjective::MaximizeThroughput,
            scope,
        );

        target.add_secondary_objective(OptimizationObjective::MinimizeLatency, 0.5);
        target.add_constraint(OptimizationConstraint::new(
            "min_throughput".to_string(),
            "Minimum throughput".to_string(),
            100.0,
            1000.0,
        ));

        assert_eq!(target.secondary_objectives.len(), 1);
        assert_eq!(target.constraints.len(), 1);
        assert!(target.are_constraints_satisfied()); // No current values set
    }

    #[test]
    fn test_time_window() {
        let start = Utc::now();
        let end = start + chrono::Duration::hours(1);
        
        let window = TimeWindow {
            start,
            end,
            recurrence: None,
        };

        assert!(window.is_active());
        assert_eq!(window.duration_seconds(), 3600);
    }

    #[test]
    fn test_multi_objective_score() {
        let scope = OptimizationScope::Network;
        let mut target = OptimizationTarget::new(
            "Multi-objective Test".to_string(),
            OptimizationObjective::MaximizeThroughput,
            scope,
        );

        target.add_secondary_objective(OptimizationObjective::MinimizeLatency, 0.3);

        let mut kpi_values = HashMap::new();
        kpi_values.insert(KpiType::AverageCellThroughput, 400.0);
        kpi_values.insert(KpiType::UserPlaneLatency, 8.0);

        let score = target.calculate_objective_score(&kpi_values);
        assert!(score.is_ok());
        assert!(score.unwrap() > 0.0);
    }
}