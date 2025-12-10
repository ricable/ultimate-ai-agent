//! Data Validation and Cleansing for Network KPI Processing
//! 
//! Comprehensive data validation, cleansing, and quality assurance system
//! for network KPI data processing with intelligent error detection and correction.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::time::SystemTime;

/// Data validation engine for network KPIs
#[derive(Debug)]
pub struct DataValidationEngine {
    pub validation_rules: ValidationRuleSet,
    pub cleansing_rules: CleansingRuleSet,
    pub validation_stats: ValidationStatistics,
    pub quality_thresholds: QualityThresholds,
}

/// Complete set of validation rules
#[derive(Debug, Clone)]
pub struct ValidationRuleSet {
    pub field_rules: HashMap<String, FieldValidationRule>,
    pub business_rules: Vec<BusinessRule>,
    pub consistency_rules: Vec<ConsistencyRule>,
    pub temporal_rules: Vec<TemporalRule>,
}

/// Individual field validation rule
#[derive(Debug, Clone)]
pub struct FieldValidationRule {
    pub field_name: String,
    pub data_type: DataType,
    pub constraints: Vec<ValidationConstraint>,
    pub required: bool,
    pub default_value: Option<f64>,
    pub unit: Option<String>,
    pub description: String,
}

/// Data types for validation
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Integer,
    Float,
    Percentage,
    String,
    Enum(Vec<String>),
    Timestamp,
    Identifier,
}

/// Validation constraints
#[derive(Debug, Clone)]
pub enum ValidationConstraint {
    Range { min: f64, max: f64 },
    MinValue(f64),
    MaxValue(f64),
    Pattern(String),
    Length { min: usize, max: usize },
    NotEmpty,
    UniqueIdentifier,
    EnumValue(Vec<String>),
}

/// Business logic rules
#[derive(Debug, Clone)]
pub struct BusinessRule {
    pub rule_id: String,
    pub description: String,
    pub rule_type: BusinessRuleType,
    pub severity: ValidationSeverity,
    pub applicable_metrics: Vec<String>,
}

/// Business rule types
#[derive(Debug, Clone)]
pub enum BusinessRuleType {
    /// Availability cannot exceed 100%
    AvailabilityLimit,
    /// Throughput must correlate with traffic load
    ThroughputCorrelation,
    /// Error rates must be inversely correlated with quality metrics
    ErrorRateConsistency,
    /// Handover success rates must be within expected ranges
    HandoverLogicalBounds,
    /// ENDC metrics must be consistent with LTE metrics
    EndcConsistency,
    /// Signal quality must correlate with performance metrics
    SignalQualityCorrelation,
    /// Custom business logic
    Custom(String),
}

/// Consistency rules between related fields
#[derive(Debug, Clone)]
pub struct ConsistencyRule {
    pub rule_id: String,
    pub description: String,
    pub primary_field: String,
    pub related_fields: Vec<String>,
    pub consistency_check: ConsistencyCheck,
}

/// Consistency check types
#[derive(Debug, Clone)]
pub enum ConsistencyCheck {
    /// Sum of parts must equal total
    SumEquals { parts: Vec<String>, total: String },
    /// Ratio must be within expected bounds
    RatioCheck { numerator: String, denominator: String, min_ratio: f64, max_ratio: f64 },
    /// Values must be monotonic (increasing/decreasing)
    MonotonicSequence { fields: Vec<String>, increasing: bool },
    /// Custom consistency logic
    Custom(String),
}

/// Temporal validation rules
#[derive(Debug, Clone)]
pub struct TemporalRule {
    pub rule_id: String,
    pub description: String,
    pub field_name: String,
    pub temporal_check: TemporalCheck,
}

/// Temporal check types
#[derive(Debug, Clone)]
pub enum TemporalCheck {
    /// Value change rate must be within bounds
    ChangeRate { max_change_per_hour: f64 },
    /// Value must be within seasonal bounds
    SeasonalBounds { expected_range: (f64, f64) },
    /// Detect sudden spikes or drops
    AnomalyDetection { sensitivity: f32 },
    /// Trending validation
    TrendValidation { max_trend_change: f64 },
}

/// Data cleansing rules
#[derive(Debug, Clone)]
pub struct CleansingRuleSet {
    pub outlier_detection: OutlierDetectionConfig,
    pub missing_value_handling: MissingValueConfig,
    pub normalization_rules: NormalizationConfig,
    pub smoothing_rules: SmoothingConfig,
}

/// Outlier detection configuration
#[derive(Debug, Clone)]
pub struct OutlierDetectionConfig {
    pub method: OutlierDetectionMethod,
    pub threshold: f64,
    pub action: OutlierAction,
}

/// Outlier detection methods
#[derive(Debug, Clone)]
pub enum OutlierDetectionMethod {
    /// Z-score based detection
    ZScore { threshold: f64 },
    /// Interquartile range based
    IQR { multiplier: f64 },
    /// Modified Z-score using median
    ModifiedZScore { threshold: f64 },
    /// Isolation Forest
    IsolationForest { contamination: f64 },
    /// Statistical process control
    StatisticalProcessControl { sigma_level: f64 },
}

/// Actions to take for outliers
#[derive(Debug, Clone)]
pub enum OutlierAction {
    Remove,
    Cap(f64), // Cap at this percentile
    Replace(f64), // Replace with this value
    Interpolate,
    Flag, // Keep but mark as outlier
}

/// Missing value handling configuration
#[derive(Debug, Clone)]
pub struct MissingValueConfig {
    pub strategy: MissingValueStrategy,
    pub max_missing_percentage: f64,
    pub interpolation_method: InterpolationMethod,
}

/// Missing value handling strategies
#[derive(Debug, Clone)]
pub enum MissingValueStrategy {
    Remove,
    Forward fill,
    Backward fill,
    Interpolate,
    UseDefault(f64),
    UseMedian,
    UseMean,
    UseMode,
}

/// Interpolation methods
#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    Linear,
    Polynomial { degree: u32 },
    Spline,
    Moving average { window: usize },
}

/// Normalization configuration
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    pub method: NormalizationMethod,
    pub apply_to_fields: Vec<String>,
}

/// Normalization methods
#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    MinMax { min: f64, max: f64 },
    ZScore,
    Robust, // Using median and IQR
    UnitVector,
    None,
}

/// Smoothing configuration
#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    pub method: SmoothingMethod,
    pub window_size: usize,
    pub apply_to_fields: Vec<String>,
}

/// Smoothing methods
#[derive(Debug, Clone)]
pub enum SmoothingMethod {
    MovingAverage,
    ExponentialSmoothing { alpha: f64 },
    SavitzkyGolay { polynomial_order: usize },
    None,
}

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ValidationSeverity {
    Critical,
    Warning,
    Info,
}

/// Quality thresholds for data acceptance
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub minimum_completeness: f64,    // 0.0 - 1.0
    pub minimum_accuracy: f64,        // 0.0 - 1.0
    pub maximum_error_rate: f64,      // 0.0 - 1.0
    pub minimum_consistency: f64,     // 0.0 - 1.0
    pub maximum_outlier_percentage: f64, // 0.0 - 1.0
}

/// Validation statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub total_records_validated: u64,
    pub valid_records: u64,
    pub invalid_records: u64,
    pub records_with_warnings: u64,
    pub critical_violations: u64,
    pub warning_violations: u64,
    pub info_violations: u64,
    pub outliers_detected: u64,
    pub missing_values_handled: u64,
    pub data_quality_score: f64,
    pub validation_time_ms: f64,
}

/// Validation result for a single record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub quality_score: f64,
    pub violations: Vec<ValidationViolation>,
    pub cleansed_data: HashMap<String, f64>,
    pub metadata: ValidationMetadata,
}

/// Individual validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    pub violation_id: String,
    pub field_name: String,
    pub rule_id: String,
    pub severity: ValidationSeverity,
    pub description: String,
    pub original_value: Option<f64>,
    pub suggested_value: Option<f64>,
    pub violation_type: ViolationType,
}

/// Types of validation violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    FieldValidation,
    BusinessRule,
    ConsistencyCheck,
    TemporalAnomaly,
    OutlierDetection,
    MissingValue,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    pub validation_timestamp: SystemTime,
    pub validator_version: String,
    pub rules_applied: Vec<String>,
    pub processing_time_ms: f64,
    pub data_source: String,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            minimum_completeness: 0.8,     // 80% completeness required
            minimum_accuracy: 0.9,         // 90% accuracy required
            maximum_error_rate: 0.1,       // Max 10% error rate
            minimum_consistency: 0.85,     // 85% consistency required
            maximum_outlier_percentage: 0.05, // Max 5% outliers
        }
    }
}

impl DataValidationEngine {
    /// Create new data validation engine with default rules
    pub fn new() -> Self {
        Self {
            validation_rules: ValidationRuleSet::create_default_rules(),
            cleansing_rules: CleansingRuleSet::default(),
            validation_stats: ValidationStatistics::default(),
            quality_thresholds: QualityThresholds::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(rules: ValidationRuleSet, thresholds: QualityThresholds) -> Self {
        Self {
            validation_rules: rules,
            cleansing_rules: CleansingRuleSet::default(),
            validation_stats: ValidationStatistics::default(),
            quality_thresholds: thresholds,
        }
    }

    /// Validate and cleanse a single record
    pub fn validate_record(&mut self, data: &HashMap<String, f64>, source: &str) -> ValidationResult {
        let start_time = std::time::Instant::now();
        let mut violations = Vec::new();
        let mut cleansed_data = data.clone();
        let mut quality_scores = Vec::new();

        // Step 1: Field-level validation
        for (field_name, rule) in &self.validation_rules.field_rules {
            if let Some(&value) = data.get(field_name) {
                let field_violations = self.validate_field(field_name, value, rule);
                violations.extend(field_violations);
                quality_scores.push(if violations.is_empty() { 1.0 } else { 0.5 });
            } else if rule.required {
                violations.push(ValidationViolation {
                    violation_id: uuid::Uuid::new_v4().to_string(),
                    field_name: field_name.clone(),
                    rule_id: "REQUIRED_FIELD".to_string(),
                    severity: ValidationSeverity::Critical,
                    description: format!("Required field {} is missing", field_name),
                    original_value: None,
                    suggested_value: rule.default_value,
                    violation_type: ViolationType::FieldValidation,
                });
                quality_scores.push(0.0);

                // Apply default value if available
                if let Some(default) = rule.default_value {
                    cleansed_data.insert(field_name.clone(), default);
                }
            }
        }

        // Step 2: Business rules validation
        for business_rule in &self.validation_rules.business_rules {
            let business_violations = self.validate_business_rule(data, business_rule);
            violations.extend(business_violations);
        }

        // Step 3: Consistency checks
        for consistency_rule in &self.validation_rules.consistency_rules {
            let consistency_violations = self.validate_consistency(data, consistency_rule);
            violations.extend(consistency_violations);
        }

        // Step 4: Data cleansing
        self.apply_data_cleansing(&mut cleansed_data);

        // Step 5: Calculate quality score
        let quality_score = if quality_scores.is_empty() {
            0.0
        } else {
            quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
        };

        // Step 6: Determine overall validity
        let critical_violations = violations.iter()
            .filter(|v| v.severity == ValidationSeverity::Critical)
            .count();
        let is_valid = critical_violations == 0 && quality_score >= self.quality_thresholds.minimum_accuracy;

        // Update statistics
        self.update_validation_statistics(&violations, is_valid, quality_score);

        let processing_time = start_time.elapsed().as_millis() as f64;

        ValidationResult {
            is_valid,
            quality_score,
            violations,
            cleansed_data,
            metadata: ValidationMetadata {
                validation_timestamp: SystemTime::now(),
                validator_version: "1.0.0".to_string(),
                rules_applied: self.get_applied_rules(),
                processing_time_ms: processing_time,
                data_source: source.to_string(),
            },
        }
    }

    /// Validate individual field
    fn validate_field(&self, field_name: &str, value: f64, rule: &FieldValidationRule) -> Vec<ValidationViolation> {
        let mut violations = Vec::new();

        // Type validation
        if !self.is_valid_type(value, &rule.data_type) {
            violations.push(ValidationViolation {
                violation_id: uuid::Uuid::new_v4().to_string(),
                field_name: field_name.to_string(),
                rule_id: "TYPE_VALIDATION".to_string(),
                severity: ValidationSeverity::Critical,
                description: format!("Value {} is not valid for type {:?}", value, rule.data_type),
                original_value: Some(value),
                suggested_value: None,
                violation_type: ViolationType::FieldValidation,
            });
        }

        // Constraint validation
        for constraint in &rule.constraints {
            if let Some(violation) = self.validate_constraint(field_name, value, constraint) {
                violations.push(violation);
            }
        }

        violations
    }

    /// Validate data type
    fn is_valid_type(&self, value: f64, data_type: &DataType) -> bool {
        match data_type {
            DataType::Integer => value.fract() == 0.0,
            DataType::Float => value.is_finite(),
            DataType::Percentage => value >= 0.0 && value <= 100.0,
            DataType::String => true, // Can't validate string as f64
            DataType::Enum(_) => true, // Would need string value to validate
            DataType::Timestamp => value > 0.0, // Simplified timestamp check
            DataType::Identifier => value > 0.0 && value.fract() == 0.0,
        }
    }

    /// Validate individual constraint
    fn validate_constraint(&self, field_name: &str, value: f64, constraint: &ValidationConstraint) -> Option<ValidationViolation> {
        match constraint {
            ValidationConstraint::Range { min, max } => {
                if value < *min || value > *max {
                    Some(ValidationViolation {
                        violation_id: uuid::Uuid::new_v4().to_string(),
                        field_name: field_name.to_string(),
                        rule_id: "RANGE_CHECK".to_string(),
                        severity: ValidationSeverity::Warning,
                        description: format!("Value {} is outside range [{}, {}]", value, min, max),
                        original_value: Some(value),
                        suggested_value: Some(value.clamp(*min, *max)),
                        violation_type: ViolationType::FieldValidation,
                    })
                } else {
                    None
                }
            }
            ValidationConstraint::MinValue(min) => {
                if value < *min {
                    Some(ValidationViolation {
                        violation_id: uuid::Uuid::new_v4().to_string(),
                        field_name: field_name.to_string(),
                        rule_id: "MIN_VALUE_CHECK".to_string(),
                        severity: ValidationSeverity::Warning,
                        description: format!("Value {} is below minimum {}", value, min),
                        original_value: Some(value),
                        suggested_value: Some(*min),
                        violation_type: ViolationType::FieldValidation,
                    })
                } else {
                    None
                }
            }
            ValidationConstraint::MaxValue(max) => {
                if value > *max {
                    Some(ValidationViolation {
                        violation_id: uuid::Uuid::new_v4().to_string(),
                        field_name: field_name.to_string(),
                        rule_id: "MAX_VALUE_CHECK".to_string(),
                        severity: ValidationSeverity::Warning,
                        description: format!("Value {} is above maximum {}", value, max),
                        original_value: Some(value),
                        suggested_value: Some(*max),
                        violation_type: ViolationType::FieldValidation,
                    })
                } else {
                    None
                }
            }
            _ => None, // Other constraint types not applicable to f64
        }
    }

    /// Validate business rules
    fn validate_business_rule(&self, data: &HashMap<String, f64>, rule: &BusinessRule) -> Vec<ValidationViolation> {
        let mut violations = Vec::new();

        match &rule.rule_type {
            BusinessRuleType::AvailabilityLimit => {
                if let Some(&availability) = data.get("CELL_AVAILABILITY_%") {
                    if availability > 100.0 {
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: "CELL_AVAILABILITY_%".to_string(),
                            rule_id: rule.rule_id.clone(),
                            severity: rule.severity.clone(),
                            description: "Availability cannot exceed 100%".to_string(),
                            original_value: Some(availability),
                            suggested_value: Some(100.0),
                            violation_type: ViolationType::BusinessRule,
                        });
                    }
                }
            }
            BusinessRuleType::ThroughputCorrelation => {
                if let (Some(&dl_throughput), Some(&traffic)) = 
                    (data.get("&_AVE_4G_LTE_DL_USER_THRPUT"), data.get("ERIC_TRAFF_ERAB_ERL")) {
                    // Expect some correlation between traffic and throughput
                    if traffic > 50.0 && dl_throughput < 5.0 {
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: "&_AVE_4G_LTE_DL_USER_THRPUT".to_string(),
                            rule_id: rule.rule_id.clone(),
                            severity: rule.severity.clone(),
                            description: "Low throughput with high traffic load".to_string(),
                            original_value: Some(dl_throughput),
                            suggested_value: None,
                            violation_type: ViolationType::BusinessRule,
                        });
                    }
                }
            }
            BusinessRuleType::ErrorRateConsistency => {
                if let (Some(&availability), Some(&drop_rate)) = 
                    (data.get("CELL_AVAILABILITY_%"), data.get("ERAB_DROP_RATE_QCI_5")) {
                    // High availability should correlate with low drop rates
                    if availability > 95.0 && drop_rate > 5.0 {
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: "ERAB_DROP_RATE_QCI_5".to_string(),
                            rule_id: rule.rule_id.clone(),
                            severity: rule.severity.clone(),
                            description: "High error rate with high availability is inconsistent".to_string(),
                            original_value: Some(drop_rate),
                            suggested_value: None,
                            violation_type: ViolationType::BusinessRule,
                        });
                    }
                }
            }
            BusinessRuleType::HandoverLogicalBounds => {
                if let Some(&ho_success_rate) = data.get("LTE_INTRA_FREQ_HO_SR") {
                    if ho_success_rate > 100.0 {
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: "LTE_INTRA_FREQ_HO_SR".to_string(),
                            rule_id: rule.rule_id.clone(),
                            severity: rule.severity.clone(),
                            description: "Handover success rate cannot exceed 100%".to_string(),
                            original_value: Some(ho_success_rate),
                            suggested_value: Some(100.0),
                            violation_type: ViolationType::BusinessRule,
                        });
                    }
                }
            }
            BusinessRuleType::EndcConsistency => {
                if let (Some(&endc_setup_sr), Some(&lte_availability)) = 
                    (data.get("ENDC_SETUP_SR"), data.get("CELL_AVAILABILITY_%")) {
                    // ENDC success rate shouldn't be higher than LTE availability
                    if endc_setup_sr > lte_availability {
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: "ENDC_SETUP_SR".to_string(),
                            rule_id: rule.rule_id.clone(),
                            severity: rule.severity.clone(),
                            description: "ENDC success rate higher than LTE availability".to_string(),
                            original_value: Some(endc_setup_sr),
                            suggested_value: Some(lte_availability),
                            violation_type: ViolationType::BusinessRule,
                        });
                    }
                }
            }
            BusinessRuleType::SignalQualityCorrelation => {
                if let (Some(&sinr), Some(&throughput)) = 
                    (data.get("SINR_PUSCH_AVG"), data.get("&_AVE_4G_LTE_DL_USER_THRPUT")) {
                    // Very low SINR should correlate with low throughput
                    if sinr < 0.0 && throughput > 30.0 {
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: "SINR_PUSCH_AVG".to_string(),
                            rule_id: rule.rule_id.clone(),
                            severity: rule.severity.clone(),
                            description: "High throughput with very low SINR is suspicious".to_string(),
                            original_value: Some(sinr),
                            suggested_value: None,
                            violation_type: ViolationType::BusinessRule,
                        });
                    }
                }
            }
            _ => {} // Other business rules not implemented
        }

        violations
    }

    /// Validate consistency rules
    fn validate_consistency(&self, data: &HashMap<String, f64>, rule: &ConsistencyRule) -> Vec<ValidationViolation> {
        let mut violations = Vec::new();

        match &rule.consistency_check {
            ConsistencyCheck::SumEquals { parts, total } => {
                let sum: f64 = parts.iter()
                    .filter_map(|part| data.get(part))
                    .sum();
                
                if let Some(&total_value) = data.get(total) {
                    let difference = (sum - total_value).abs();
                    if difference > 0.01 * total_value.max(1.0) { // 1% tolerance
                        violations.push(ValidationViolation {
                            violation_id: uuid::Uuid::new_v4().to_string(),
                            field_name: total.clone(),
                            rule_id: rule.rule_id.clone(),
                            severity: ValidationSeverity::Warning,
                            description: format!("Sum of parts ({:.2}) doesn't equal total ({:.2})", sum, total_value),
                            original_value: Some(total_value),
                            suggested_value: Some(sum),
                            violation_type: ViolationType::ConsistencyCheck,
                        });
                    }
                }
            }
            ConsistencyCheck::RatioCheck { numerator, denominator, min_ratio, max_ratio } => {
                if let (Some(&num), Some(&den)) = (data.get(numerator), data.get(denominator)) {
                    if den != 0.0 {
                        let ratio = num / den;
                        if ratio < *min_ratio || ratio > *max_ratio {
                            violations.push(ValidationViolation {
                                violation_id: uuid::Uuid::new_v4().to_string(),
                                field_name: numerator.clone(),
                                rule_id: rule.rule_id.clone(),
                                severity: ValidationSeverity::Warning,
                                description: format!("Ratio {:.2} is outside expected range [{:.2}, {:.2}]", 
                                    ratio, min_ratio, max_ratio),
                                original_value: Some(num),
                                suggested_value: None,
                                violation_type: ViolationType::ConsistencyCheck,
                            });
                        }
                    }
                }
            }
            _ => {} // Other consistency checks not implemented
        }

        violations
    }

    /// Apply data cleansing
    fn apply_data_cleansing(&self, data: &mut HashMap<String, f64>) {
        // Outlier detection and handling
        for (field_name, &value) in data.clone().iter() {
            if self.is_outlier(field_name, value) {
                match self.cleansing_rules.outlier_detection.action {
                    OutlierAction::Cap(percentile) => {
                        let capped_value = self.get_percentile_value(field_name, percentile);
                        data.insert(field_name.clone(), capped_value);
                    }
                    OutlierAction::Replace(replacement) => {
                        data.insert(field_name.clone(), replacement);
                    }
                    OutlierAction::Remove => {
                        data.remove(field_name);
                    }
                    _ => {} // Other actions
                }
            }
        }

        // Apply normalization if configured
        self.apply_normalization(data);

        // Apply smoothing if configured
        self.apply_smoothing(data);
    }

    /// Check if value is an outlier
    fn is_outlier(&self, _field_name: &str, value: f64) -> bool {
        // Simplified outlier detection using Z-score
        match &self.cleansing_rules.outlier_detection.method {
            OutlierDetectionMethod::ZScore { threshold } => {
                let z_score = self.calculate_z_score(value);
                z_score.abs() > *threshold
            }
            _ => false, // Other methods not implemented
        }
    }

    /// Calculate Z-score for outlier detection
    fn calculate_z_score(&self, value: f64) -> f64 {
        // Simplified calculation - in production would use historical data
        let mean = 50.0; // Placeholder mean
        let std_dev = 20.0; // Placeholder standard deviation
        (value - mean) / std_dev
    }

    /// Get percentile value for capping
    fn get_percentile_value(&self, _field_name: &str, percentile: f64) -> f64 {
        // Simplified percentile calculation - in production would use historical data
        percentile * 100.0 // Placeholder calculation
    }

    /// Apply normalization
    fn apply_normalization(&self, data: &mut HashMap<String, f64>) {
        match &self.cleansing_rules.normalization_rules.method {
            NormalizationMethod::MinMax { min, max } => {
                for field_name in &self.cleansing_rules.normalization_rules.apply_to_fields {
                    if let Some(&value) = data.get(field_name) {
                        let normalized = (value - min) / (max - min);
                        data.insert(field_name.clone(), normalized.clamp(0.0, 1.0));
                    }
                }
            }
            NormalizationMethod::ZScore => {
                for field_name in &self.cleansing_rules.normalization_rules.apply_to_fields {
                    if let Some(&value) = data.get(field_name) {
                        let z_score = self.calculate_z_score(value);
                        data.insert(field_name.clone(), z_score);
                    }
                }
            }
            _ => {} // Other normalization methods
        }
    }

    /// Apply smoothing
    fn apply_smoothing(&self, _data: &mut HashMap<String, f64>) {
        // Smoothing would require historical data - placeholder implementation
        // In production, would apply moving averages or exponential smoothing
    }

    /// Update validation statistics
    fn update_validation_statistics(&mut self, violations: &[ValidationViolation], is_valid: bool, quality_score: f64) {
        self.validation_stats.total_records_validated += 1;
        
        if is_valid {
            self.validation_stats.valid_records += 1;
        } else {
            self.validation_stats.invalid_records += 1;
        }

        if !violations.is_empty() {
            self.validation_stats.records_with_warnings += 1;
        }

        for violation in violations {
            match violation.severity {
                ValidationSeverity::Critical => self.validation_stats.critical_violations += 1,
                ValidationSeverity::Warning => self.validation_stats.warning_violations += 1,
                ValidationSeverity::Info => self.validation_stats.info_violations += 1,
            }
        }

        // Update running average of data quality score
        let total_records = self.validation_stats.total_records_validated as f64;
        self.validation_stats.data_quality_score = 
            (self.validation_stats.data_quality_score * (total_records - 1.0) + quality_score) / total_records;
    }

    /// Get list of applied rules
    fn get_applied_rules(&self) -> Vec<String> {
        let mut rules = Vec::new();
        
        // Add field validation rules
        for field_name in self.validation_rules.field_rules.keys() {
            rules.push(format!("FIELD_VALIDATION_{}", field_name));
        }
        
        // Add business rules
        for business_rule in &self.validation_rules.business_rules {
            rules.push(business_rule.rule_id.clone());
        }
        
        // Add consistency rules
        for consistency_rule in &self.validation_rules.consistency_rules {
            rules.push(consistency_rule.rule_id.clone());
        }
        
        rules
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> &ValidationStatistics {
        &self.validation_stats
    }

    /// Reset validation statistics
    pub fn reset_statistics(&mut self) {
        self.validation_stats = ValidationStatistics::default();
    }
}

impl ValidationRuleSet {
    /// Create default validation rules for network KPIs
    pub fn create_default_rules() -> Self {
        let mut field_rules = HashMap::new();

        // Cell availability percentage
        field_rules.insert("CELL_AVAILABILITY_%".to_string(), FieldValidationRule {
            field_name: "CELL_AVAILABILITY_%".to_string(),
            data_type: DataType::Percentage,
            constraints: vec![
                ValidationConstraint::Range { min: 0.0, max: 100.0 },
            ],
            required: true,
            default_value: Some(100.0),
            unit: Some("%".to_string()),
            description: "Cell availability percentage".to_string(),
        });

        // VoLTE traffic
        field_rules.insert("VOLTE_TRAFFIC (ERL)".to_string(), FieldValidationRule {
            field_name: "VOLTE_TRAFFIC (ERL)".to_string(),
            data_type: DataType::Float,
            constraints: vec![
                ValidationConstraint::MinValue(0.0),
                ValidationConstraint::MaxValue(1000.0),
            ],
            required: false,
            default_value: Some(0.0),
            unit: Some("Erlang".to_string()),
            description: "VoLTE traffic in Erlangs".to_string(),
        });

        // SINR values
        field_rules.insert("SINR_PUSCH_AVG".to_string(), FieldValidationRule {
            field_name: "SINR_PUSCH_AVG".to_string(),
            data_type: DataType::Float,
            constraints: vec![
                ValidationConstraint::Range { min: -20.0, max: 50.0 },
            ],
            required: false,
            default_value: None,
            unit: Some("dB".to_string()),
            description: "Average PUSCH SINR".to_string(),
        });

        // Add more field rules...

        // Business rules
        let business_rules = vec![
            BusinessRule {
                rule_id: "AVAILABILITY_LIMIT".to_string(),
                description: "Cell availability cannot exceed 100%".to_string(),
                rule_type: BusinessRuleType::AvailabilityLimit,
                severity: ValidationSeverity::Critical,
                applicable_metrics: vec!["CELL_AVAILABILITY_%".to_string()],
            },
            BusinessRule {
                rule_id: "THROUGHPUT_CORRELATION".to_string(),
                description: "Throughput should correlate with traffic load".to_string(),
                rule_type: BusinessRuleType::ThroughputCorrelation,
                severity: ValidationSeverity::Warning,
                applicable_metrics: vec!["&_AVE_4G_LTE_DL_USER_THRPUT".to_string(), "ERIC_TRAFF_ERAB_ERL".to_string()],
            },
            BusinessRule {
                rule_id: "ERROR_RATE_CONSISTENCY".to_string(),
                description: "Error rates should be consistent with quality metrics".to_string(),
                rule_type: BusinessRuleType::ErrorRateConsistency,
                severity: ValidationSeverity::Warning,
                applicable_metrics: vec!["CELL_AVAILABILITY_%".to_string(), "ERAB_DROP_RATE_QCI_5".to_string()],
            },
        ];

        // Consistency rules
        let consistency_rules = vec![
            ConsistencyRule {
                rule_id: "HANDOVER_RATIO_CONSISTENCY".to_string(),
                description: "Handover success ratio should be consistent".to_string(),
                primary_field: "LTE_INTRA_FREQ_HO_SR".to_string(),
                related_fields: vec!["INTRA FREQ HO ATTEMPTS".to_string()],
                consistency_check: ConsistencyCheck::RatioCheck {
                    numerator: "LTE_INTRA_FREQ_HO_SR".to_string(),
                    denominator: "INTRA FREQ HO ATTEMPTS".to_string(),
                    min_ratio: 0.0,
                    max_ratio: 1.0,
                },
            },
        ];

        // Temporal rules
        let temporal_rules = vec![
            TemporalRule {
                rule_id: "AVAILABILITY_CHANGE_RATE".to_string(),
                description: "Availability shouldn't change too rapidly".to_string(),
                field_name: "CELL_AVAILABILITY_%".to_string(),
                temporal_check: TemporalCheck::ChangeRate { max_change_per_hour: 10.0 },
            },
        ];

        Self {
            field_rules,
            business_rules,
            consistency_rules,
            temporal_rules,
        }
    }
}

impl Default for CleansingRuleSet {
    fn default() -> Self {
        Self {
            outlier_detection: OutlierDetectionConfig {
                method: OutlierDetectionMethod::ZScore { threshold: 3.0 },
                threshold: 3.0,
                action: OutlierAction::Flag,
            },
            missing_value_handling: MissingValueConfig {
                strategy: MissingValueStrategy::UseMedian,
                max_missing_percentage: 0.2, // 20% max missing
                interpolation_method: InterpolationMethod::Linear,
            },
            normalization_rules: NormalizationConfig {
                method: NormalizationMethod::None,
                apply_to_fields: Vec::new(),
            },
            smoothing_rules: SmoothingConfig {
                method: SmoothingMethod::None,
                window_size: 5,
                apply_to_fields: Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_engine_creation() {
        let engine = DataValidationEngine::new();
        assert!(engine.validation_rules.field_rules.contains_key("CELL_AVAILABILITY_%"));
        assert!(!engine.validation_rules.business_rules.is_empty());
    }

    #[test]
    fn test_valid_record_validation() {
        let mut engine = DataValidationEngine::new();
        let mut data = HashMap::new();
        data.insert("CELL_AVAILABILITY_%".to_string(), 98.5);
        data.insert("VOLTE_TRAFFIC (ERL)".to_string(), 15.2);
        data.insert("SINR_PUSCH_AVG".to_string(), 12.5);

        let result = engine.validate_record(&data, "test_source");
        assert!(result.is_valid);
        assert!(result.quality_score > 0.8);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_invalid_record_validation() {
        let mut engine = DataValidationEngine::new();
        let mut data = HashMap::new();
        data.insert("CELL_AVAILABILITY_%".to_string(), 105.0); // Invalid - over 100%
        data.insert("SINR_PUSCH_AVG".to_string(), 100.0); // Invalid - too high

        let result = engine.validate_record(&data, "test_source");
        assert!(!result.is_valid);
        assert!(!result.violations.is_empty());
        
        // Should have violations for both fields
        assert!(result.violations.iter().any(|v| v.field_name == "CELL_AVAILABILITY_%"));
    }

    #[test]
    fn test_business_rule_validation() {
        let mut engine = DataValidationEngine::new();
        let mut data = HashMap::new();
        data.insert("CELL_AVAILABILITY_%".to_string(), 110.0); // Violates availability limit

        let result = engine.validate_record(&data, "test_source");
        assert!(!result.is_valid);
        
        let business_violations: Vec<_> = result.violations.iter()
            .filter(|v| matches!(v.violation_type, ViolationType::BusinessRule))
            .collect();
        assert!(!business_violations.is_empty());
    }

    #[test]
    fn test_missing_required_field() {
        let mut engine = DataValidationEngine::new();
        let data = HashMap::new(); // Missing required CELL_AVAILABILITY_%

        let result = engine.validate_record(&data, "test_source");
        assert!(!result.is_valid);
        
        let missing_field_violations: Vec<_> = result.violations.iter()
            .filter(|v| v.rule_id == "REQUIRED_FIELD")
            .collect();
        assert!(!missing_field_violations.is_empty());
    }

    #[test]
    fn test_data_cleansing() {
        let mut engine = DataValidationEngine::new();
        let mut data = HashMap::new();
        data.insert("CELL_AVAILABILITY_%".to_string(), 98.5);
        // Add outlier value that should be detected and handled
        data.insert("test_field".to_string(), 999999.0); // Extreme outlier

        let result = engine.validate_record(&data, "test_source");
        // Cleansed data should handle the outlier
        assert!(result.cleansed_data.contains_key("CELL_AVAILABILITY_%"));
    }

    #[test]
    fn test_validation_statistics() {
        let mut engine = DataValidationEngine::new();
        
        // Validate a few records
        for i in 0..5 {
            let mut data = HashMap::new();
            data.insert("CELL_AVAILABILITY_%".to_string(), 95.0 + i as f64);
            engine.validate_record(&data, "test_source");
        }

        let stats = engine.get_statistics();
        assert_eq!(stats.total_records_validated, 5);
        assert!(stats.valid_records > 0);
    }

    #[test]
    fn test_constraint_validation() {
        let engine = DataValidationEngine::new();
        let constraint = ValidationConstraint::Range { min: 0.0, max: 100.0 };
        
        // Test valid value
        let violation = engine.validate_constraint("test_field", 50.0, &constraint);
        assert!(violation.is_none());
        
        // Test invalid value
        let violation = engine.validate_constraint("test_field", 150.0, &constraint);
        assert!(violation.is_some());
    }
}