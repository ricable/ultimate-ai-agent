//! RAN Optimization Recommendations
//!
//! Structured recommendation types and action generators.

use serde::{Deserialize, Serialize};

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Unique recommendation ID
    pub id: String,
    /// Priority (1 = highest)
    pub priority: u8,
    /// Category of optimization
    pub category: OptimizationCategory,
    /// Severity/urgency level
    pub severity: Severity,
    /// Affected cell ID
    pub cell_id: String,
    /// Human-readable title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Specific actions to take
    pub actions: Vec<Action>,
    /// Expected improvement
    pub expected_impact: ExpectedImpact,
    /// Risk assessment
    pub risk: Risk,
    /// Validation criteria
    pub validation: ValidationCriteria,
    /// Rollback procedure
    pub rollback: Option<RollbackProcedure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Coverage,
    Capacity,
    Interference,
    Mobility,
    Energy,
    QoS,
    Accessibility,
    Retainability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Optimization,
}

/// Specific action/parameter change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Action type
    pub action_type: ActionType,
    /// ENM parameter name
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Recommended value
    pub new_value: String,
    /// Unit of measurement
    pub unit: Option<String>,
    /// ENM MO path
    pub mo_path: Option<String>,
    /// Requires service impact
    pub service_impact: bool,
    /// Execution order
    pub order: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    ParameterChange,
    AntennaAdjustment,
    FeatureActivation,
    FeatureDeactivation,
    RelationAdd,
    RelationRemove,
    CellLock,
    CellUnlock,
    SoftwareUpgrade,
    HardwareReplacement,
}

/// Expected impact from implementing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImpact {
    /// KPI improvements
    pub kpi_improvements: Vec<KpiImprovement>,
    /// User experience improvement
    pub user_experience: Option<String>,
    /// Capacity gain
    pub capacity_gain: Option<String>,
    /// Energy savings
    pub energy_savings: Option<String>,
    /// Confidence level (%)
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiImprovement {
    pub kpi_name: String,
    pub current_value: String,
    pub expected_value: String,
    pub improvement_percent: Option<f32>,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Risk {
    /// Overall risk level
    pub level: RiskLevel,
    /// Risk factors
    pub factors: Vec<RiskFactor>,
    /// Mitigation steps
    pub mitigations: Vec<String>,
    /// Cells potentially affected
    pub affected_cells: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor: String,
    pub probability: f32,
    pub impact: String,
}

/// Validation criteria to verify improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriteria {
    /// KPIs to monitor
    pub monitor_kpis: Vec<MonitorKpi>,
    /// Monitoring duration (hours)
    pub monitoring_duration_hours: u32,
    /// Success thresholds
    pub success_thresholds: Vec<Threshold>,
    /// Failure conditions that trigger rollback
    pub failure_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorKpi {
    pub kpi_name: String,
    pub baseline_value: String,
    pub target_value: String,
    pub check_interval_minutes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threshold {
    pub kpi_name: String,
    pub operator: ThresholdOperator,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdOperator {
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    NotEqual,
}

/// Rollback procedure if changes cause issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackProcedure {
    /// Trigger conditions for rollback
    pub triggers: Vec<String>,
    /// Rollback steps
    pub steps: Vec<RollbackStep>,
    /// Time window for automatic rollback
    pub auto_rollback_hours: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStep {
    pub order: u8,
    pub action: String,
    pub parameter: String,
    pub restore_value: String,
}

/// Recommendation builder for common scenarios
pub struct RecommendationBuilder {
    rec: Recommendation,
}

impl RecommendationBuilder {
    pub fn new(cell_id: &str, category: OptimizationCategory) -> Self {
        Self {
            rec: Recommendation {
                id: uuid_v4(),
                priority: 3,
                category,
                severity: Severity::Medium,
                cell_id: cell_id.to_string(),
                title: String::new(),
                description: String::new(),
                actions: vec![],
                expected_impact: ExpectedImpact {
                    kpi_improvements: vec![],
                    user_experience: None,
                    capacity_gain: None,
                    energy_savings: None,
                    confidence: 80.0,
                },
                risk: Risk {
                    level: RiskLevel::Low,
                    factors: vec![],
                    mitigations: vec![],
                    affected_cells: vec![],
                },
                validation: ValidationCriteria {
                    monitor_kpis: vec![],
                    monitoring_duration_hours: 24,
                    success_thresholds: vec![],
                    failure_conditions: vec![],
                },
                rollback: None,
            },
        }
    }

    pub fn title(mut self, title: &str) -> Self {
        self.rec.title = title.to_string();
        self
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.rec.description = desc.to_string();
        self
    }

    pub fn priority(mut self, priority: u8) -> Self {
        self.rec.priority = priority;
        self
    }

    pub fn severity(mut self, severity: Severity) -> Self {
        self.rec.severity = severity;
        self
    }

    pub fn add_action(mut self, action: Action) -> Self {
        self.rec.actions.push(action);
        self
    }

    pub fn parameter_change(
        mut self,
        parameter: &str,
        current: &str,
        new: &str,
        unit: Option<&str>,
    ) -> Self {
        self.rec.actions.push(Action {
            action_type: ActionType::ParameterChange,
            parameter: parameter.to_string(),
            current_value: current.to_string(),
            new_value: new.to_string(),
            unit: unit.map(|s| s.to_string()),
            mo_path: None,
            service_impact: false,
            order: self.rec.actions.len() as u8 + 1,
        });
        self
    }

    pub fn expected_kpi_improvement(
        mut self,
        kpi: &str,
        current: &str,
        expected: &str,
        improvement_pct: Option<f32>,
    ) -> Self {
        self.rec.expected_impact.kpi_improvements.push(KpiImprovement {
            kpi_name: kpi.to_string(),
            current_value: current.to_string(),
            expected_value: expected.to_string(),
            improvement_percent: improvement_pct,
        });
        self
    }

    pub fn risk_level(mut self, level: RiskLevel) -> Self {
        self.rec.risk.level = level;
        self
    }

    pub fn affected_cell(mut self, cell_id: &str) -> Self {
        self.rec.risk.affected_cells.push(cell_id.to_string());
        self
    }

    pub fn monitoring_duration(mut self, hours: u32) -> Self {
        self.rec.validation.monitoring_duration_hours = hours;
        self
    }

    pub fn with_rollback(mut self, rollback: RollbackProcedure) -> Self {
        self.rec.rollback = Some(rollback);
        self
    }

    pub fn build(self) -> Recommendation {
        self.rec
    }
}

/// Generate simple UUID v4-like string
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("rec-{:x}", ts)
}

/// Common recommendation templates
pub mod templates {
    use super::*;

    /// Coverage hole remediation
    pub fn coverage_hole(cell_id: &str, rsrp: f32, target_tilt: f32, current_tilt: f32) -> Recommendation {
        RecommendationBuilder::new(cell_id, OptimizationCategory::Coverage)
            .title("Coverage Hole Remediation")
            .description(&format!(
                "Poor coverage detected with RSRP {}dBm. Recommend antenna tilt adjustment to improve coverage.",
                rsrp
            ))
            .priority(2)
            .severity(Severity::High)
            .parameter_change(
                "antennaElectricalTilt",
                &format!("{}", current_tilt),
                &format!("{}", target_tilt),
                Some("degrees"),
            )
            .expected_kpi_improvement(
                "RSRP",
                &format!("{} dBm", rsrp),
                &format!("{} dBm", rsrp + 5.0),
                Some(6.0),
            )
            .risk_level(RiskLevel::Low)
            .monitoring_duration(24)
            .build()
    }

    /// High PRB utilization
    pub fn high_prb_utilization(cell_id: &str, prb_util: f32, neighbor_cells: &[&str]) -> Recommendation {
        let mut builder = RecommendationBuilder::new(cell_id, OptimizationCategory::Capacity)
            .title("High PRB Utilization - Load Balancing Required")
            .description(&format!(
                "PRB utilization at {}% exceeds threshold. Recommend load balancing to neighboring cells.",
                prb_util
            ))
            .priority(2)
            .severity(Severity::High)
            .parameter_change(
                "loadBalancingActive",
                "false",
                "true",
                None,
            )
            .parameter_change(
                "loadBalancingTargetPrb",
                "70",
                "65",
                Some("%"),
            )
            .expected_kpi_improvement(
                "PRB Utilization",
                &format!("{}%", prb_util),
                "70%",
                Some(((prb_util - 70.0) / prb_util * 100.0) as f32),
            )
            .risk_level(RiskLevel::Medium)
            .monitoring_duration(48);

        for neighbor in neighbor_cells {
            builder = builder.affected_cell(neighbor);
        }

        builder.build()
    }

    /// Handover optimization
    pub fn handover_optimization(
        cell_id: &str,
        ho_sr: f32,
        current_ttt: u32,
        new_ttt: u32,
    ) -> Recommendation {
        RecommendationBuilder::new(cell_id, OptimizationCategory::Mobility)
            .title("Handover Success Rate Improvement")
            .description(&format!(
                "Handover success rate at {}% is below target. Recommend timeToTrigger adjustment.",
                ho_sr
            ))
            .priority(2)
            .severity(Severity::Medium)
            .parameter_change(
                "timeToTrigger",
                &format!("{}", current_ttt),
                &format!("{}", new_ttt),
                Some("ms"),
            )
            .expected_kpi_improvement(
                "HO Success Rate",
                &format!("{}%", ho_sr),
                "98%",
                Some((98.0 - ho_sr) / ho_sr * 100.0),
            )
            .risk_level(RiskLevel::Low)
            .monitoring_duration(24)
            .build()
    }

    /// Energy saving activation
    pub fn energy_saving(cell_id: &str, low_traffic_hours: &[u8]) -> Recommendation {
        let hours_str = low_traffic_hours
            .iter()
            .map(|h| h.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        RecommendationBuilder::new(cell_id, OptimizationCategory::Energy)
            .title("Energy Saving Mode Activation")
            .description(&format!(
                "Low traffic detected during hours: {}. Recommend enabling energy saving features.",
                hours_str
            ))
            .priority(3)
            .severity(Severity::Low)
            .parameter_change(
                "energySavingState",
                "DISABLED",
                "ENABLED",
                None,
            )
            .parameter_change(
                "cellSleepModeActive",
                "false",
                "true",
                None,
            )
            .expected_kpi_improvement(
                "Power Consumption",
                "100%",
                "75%",
                Some(25.0),
            )
            .risk_level(RiskLevel::Low)
            .monitoring_duration(168)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_hole_recommendation() {
        let rec = templates::coverage_hole("CELL001", -115.0, 6.0, 4.0);
        assert_eq!(rec.cell_id, "CELL001");
        assert!(matches!(rec.category, OptimizationCategory::Coverage));
        assert!(!rec.actions.is_empty());
    }

    #[test]
    fn test_recommendation_builder() {
        let rec = RecommendationBuilder::new("CELL002", OptimizationCategory::Capacity)
            .title("Test Recommendation")
            .description("Test description")
            .priority(1)
            .severity(Severity::Critical)
            .parameter_change("testParam", "old", "new", Some("unit"))
            .risk_level(RiskLevel::High)
            .build();

        assert_eq!(rec.priority, 1);
        assert!(matches!(rec.severity, Severity::Critical));
        assert!(matches!(rec.risk.level, RiskLevel::High));
        assert_eq!(rec.actions.len(), 1);
    }
}
