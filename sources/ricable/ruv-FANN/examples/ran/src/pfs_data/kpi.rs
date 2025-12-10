//! KPI (Key Performance Indicator) mapping for Ericsson ENM counters
//! 
//! Provides optimized mappings and calculations for common LTE/5G KPIs

use std::collections::HashMap;

/// Simple data type enumeration to replace Arrow DataType
#[derive(Debug, Clone)]
pub enum SimpleDataType {
    Float32,
    Int64,
    String,
}

/// KPI information structure
#[derive(Debug, Clone)]
pub struct KpiInfo {
    pub name: String,
    pub description: String,
    pub data_type: SimpleDataType,
    pub unit: String,
    pub category: KpiCategory,
    pub formula: Option<KpiFormula>,
}

/// KPI categories for organization
#[derive(Debug, Clone, PartialEq)]
pub enum KpiCategory {
    Accessibility,
    Retainability,
    Mobility,
    Integrity,
    Availability,
    Utilization,
    Throughput,
    Latency,
}

/// KPI formula for calculated metrics
#[derive(Debug, Clone)]
pub enum KpiFormula {
    /// Simple ratio: numerator / denominator
    Ratio { numerator: String, denominator: String },
    /// Weighted average
    WeightedAverage { values: Vec<String>, weights: Vec<String> },
    /// Sum of multiple counters
    Sum(Vec<String>),
    /// Average of multiple counters
    Average(Vec<String>),
    /// Custom formula with expression
    Custom(String),
}

/// Main KPI mappings container
#[derive(Debug, Clone)]
pub struct KpiMappings {
    pub mappings: HashMap<String, KpiInfo>,
    pub counter_to_kpi: HashMap<String, Vec<String>>,
}

impl Default for KpiMappings {
    fn default() -> Self {
        Self::new()
    }
}

impl KpiMappings {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();
        let mut counter_to_kpi = HashMap::new();

        // RRC Connection Success Rate
        mappings.insert("rrc_conn_success_rate".to_string(), KpiInfo {
            name: "RRC Connection Success Rate".to_string(),
            description: "Percentage of successful RRC connection establishments".to_string(),
            data_type: SimpleDataType::Float32,
            unit: "percentage".to_string(),
            category: KpiCategory::Accessibility,
            formula: Some(KpiFormula::Ratio {
                numerator: "pmRrcConnEstabSucc".to_string(),
                denominator: "pmRrcConnEstabAtt".to_string(),
            }),
        });

        // SCell Addition Success Rate
        mappings.insert("scell_add_success_rate".to_string(), KpiInfo {
            name: "SCell Addition Success Rate".to_string(),
            description: "Percentage of successful secondary cell additions".to_string(),
            data_type: SimpleDataType::Float32,
            unit: "percentage".to_string(),
            category: KpiCategory::Accessibility,
            formula: Some(KpiFormula::Ratio {
                numerator: "pmLteScellAddSucc".to_string(),
                denominator: "pmLteScellAddAtt".to_string(),
            }),
        });

        // Handover Success Rate
        mappings.insert("handover_success_rate".to_string(), KpiInfo {
            name: "Handover Success Rate".to_string(),
            description: "Percentage of successful handovers".to_string(),
            data_type: SimpleDataType::Float32,
            unit: "percentage".to_string(),
            category: KpiCategory::Mobility,
            formula: Some(KpiFormula::Ratio {
                numerator: "pmHoExeSucc".to_string(),
                denominator: "pmHoExeAtt".to_string(),
            }),
        });

        // Throughput KPIs
        mappings.insert("dl_throughput".to_string(), KpiInfo {
            name: "Downlink Throughput".to_string(),
            description: "Average downlink throughput".to_string(),
            data_type: SimpleDataType::Float32,
            unit: "Mbps".to_string(),
            category: KpiCategory::Throughput,
            formula: Some(KpiFormula::Custom("pmPdcpVolDlDrb * 8 / 1000000 / 900".to_string())),
        });

        mappings.insert("ul_throughput".to_string(), KpiInfo {
            name: "Uplink Throughput".to_string(),
            description: "Average uplink throughput".to_string(),
            data_type: SimpleDataType::Float32,
            unit: "Mbps".to_string(),
            category: KpiCategory::Throughput,
            formula: Some(KpiFormula::Custom("pmPdcpVolUlDrb * 8 / 1000000 / 900".to_string())),
        });

        // Raw counters
        for counter in &[
            "pmRrcConnEstabSucc",
            "pmRrcConnEstabAtt",
            "pmLteScellAddSucc",
            "pmLteScellAddAtt",
            "pmHoExeSucc",
            "pmHoExeAtt",
            "pmPdcpVolDlDrb",
            "pmPdcpVolUlDrb",
            "pmRrcConnEstabFailMmeOvlMod",
            "pmRrcConnEstabFailMmeOvlMos",
            "pmRrcConnEstabFailLic",
            "pmActiveUeDlMax",
            "pmActiveUeUlMax",
            "pmCaScellActDeactSucc",
            "pmCaScellActDeactAtt",
            "pmRrcConnReestSucc",
            "pmRrcConnReestAtt",
            "pmPdcpPktDiscDlHo",
            "pmPdcpPktDiscUlHo",
            "pmRadioThpVolDl",
            "pmRadioThpVolUl",
            "pmPrbAvailDl",
            "pmPrbAvailUl",
            "pmPrbUsedDl",
            "pmPrbUsedUl",
        ] {
            mappings.insert(counter.to_string(), KpiInfo {
                name: counter.to_string(),
                description: format!("Raw counter: {}", counter),
                data_type: if counter.contains("Vol") || counter.contains("Thp") {
                    SimpleDataType::Int64
                } else {
                    SimpleDataType::Int64
                },
                unit: if counter.contains("Vol") {
                    "bytes".to_string()
                } else {
                    "count".to_string()
                },
                category: KpiCategory::Utilization,
                formula: None,
            });
        }

        // Build reverse mapping
        for (kpi_name, kpi_info) in &mappings {
            if let Some(formula) = &kpi_info.formula {
                let counters = Self::extract_counters_from_formula(formula);
                for counter in counters {
                    counter_to_kpi.entry(counter)
                        .or_insert_with(Vec::new)
                        .push(kpi_name.clone());
                }
            }
        }

        Self {
            mappings,
            counter_to_kpi,
        }
    }

    /// Extract counter names from a formula
    fn extract_counters_from_formula(formula: &KpiFormula) -> Vec<String> {
        match formula {
            KpiFormula::Ratio { numerator, denominator } => {
                vec![numerator.clone(), denominator.clone()]
            }
            KpiFormula::WeightedAverage { values, weights } => {
                let mut counters = values.clone();
                counters.extend(weights.clone());
                counters
            }
            KpiFormula::Sum(counters) | KpiFormula::Average(counters) => {
                counters.clone()
            }
            KpiFormula::Custom(expr) => {
                // Simple regex-like extraction for counter names
                let mut counters = Vec::new();
                let words: Vec<&str> = expr.split_whitespace().collect();
                for word in words {
                    if word.starts_with("pm") {
                        counters.push(word.to_string());
                    }
                }
                counters
            }
        }
    }

    /// Calculate KPI value from counter values
    pub fn calculate_kpi(&self, kpi_name: &str, counters: &HashMap<String, f64>) -> Option<f64> {
        let kpi_info = self.mappings.get(kpi_name)?;
        
        if let Some(formula) = &kpi_info.formula {
            match formula {
                KpiFormula::Ratio { numerator, denominator } => {
                    let num = counters.get(numerator)?;
                    let den = counters.get(denominator)?;
                    if *den != 0.0 {
                        Some(num / den * 100.0) // Convert to percentage
                    } else {
                        None
                    }
                }
                KpiFormula::Sum(counter_names) => {
                    let sum: f64 = counter_names.iter()
                        .filter_map(|name| counters.get(name))
                        .sum();
                    Some(sum)
                }
                KpiFormula::Average(counter_names) => {
                    let values: Vec<f64> = counter_names.iter()
                        .filter_map(|name| counters.get(name))
                        .cloned()
                        .collect();
                    if !values.is_empty() {
                        Some(values.iter().sum::<f64>() / values.len() as f64)
                    } else {
                        None
                    }
                }
                KpiFormula::Custom(expr) => {
                    // Simple expression evaluator for throughput calculations
                    self.evaluate_custom_expression(expr, counters)
                }
                _ => None,
            }
        } else {
            // Direct counter value
            counters.get(kpi_name).copied()
        }
    }

    /// Simple expression evaluator for custom formulas
    fn evaluate_custom_expression(&self, expr: &str, counters: &HashMap<String, f64>) -> Option<f64> {
        // Handle throughput calculation: "pmPdcpVolDlDrb * 8 / 1000000 / 900"
        if expr.contains("pmPdcpVolDlDrb") {
            let volume = counters.get("pmPdcpVolDlDrb")?;
            Some(volume * 8.0 / 1_000_000.0 / 900.0) // Convert to Mbps
        } else if expr.contains("pmPdcpVolUlDrb") {
            let volume = counters.get("pmPdcpVolUlDrb")?;
            Some(volume * 8.0 / 1_000_000.0 / 900.0) // Convert to Mbps
        } else {
            None
        }
    }

    /// Get KPIs by category
    pub fn get_kpis_by_category(&self, category: &KpiCategory) -> Vec<&KpiInfo> {
        self.mappings.values()
            .filter(|kpi| &kpi.category == category)
            .collect()
    }

    /// Get all counters needed for a KPI
    pub fn get_required_counters(&self, kpi_name: &str) -> Vec<String> {
        if let Some(kpi_info) = self.mappings.get(kpi_name) {
            if let Some(formula) = &kpi_info.formula {
                Self::extract_counters_from_formula(formula)
            } else {
                vec![kpi_name.to_string()]
            }
        } else {
            vec![]
        }
    }
}

/// KPI calculator for batch processing
pub struct KpiCalculator {
    mappings: KpiMappings,
}

impl KpiCalculator {
    pub fn new() -> Self {
        Self {
            mappings: KpiMappings::new(),
        }
    }

    /// Calculate all KPIs for a set of counter values
    pub fn calculate_all_kpis(&self, counters: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut results = HashMap::new();
        
        for kpi_name in self.mappings.mappings.keys() {
            if let Some(value) = self.mappings.calculate_kpi(kpi_name, counters) {
                results.insert(kpi_name.clone(), value);
            }
        }
        
        results
    }

    /// Calculate KPIs for multiple time periods
    pub fn calculate_time_series(&self, time_series: &[HashMap<String, f64>]) -> Vec<HashMap<String, f64>> {
        time_series.iter()
            .map(|counters| self.calculate_all_kpis(counters))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kpi_mappings() {
        let mappings = KpiMappings::new();
        assert!(mappings.mappings.contains_key("rrc_conn_success_rate"));
        assert!(mappings.mappings.contains_key("pmRrcConnEstabSucc"));
    }

    #[test]
    fn test_kpi_calculation() {
        let mappings = KpiMappings::new();
        let mut counters = HashMap::new();
        counters.insert("pmRrcConnEstabSucc".to_string(), 90.0);
        counters.insert("pmRrcConnEstabAtt".to_string(), 100.0);
        
        let success_rate = mappings.calculate_kpi("rrc_conn_success_rate", &counters);
        assert_eq!(success_rate, Some(90.0));
    }

    #[test]
    fn test_kpi_calculator() {
        let calculator = KpiCalculator::new();
        let mut counters = HashMap::new();
        counters.insert("pmRrcConnEstabSucc".to_string(), 90.0);
        counters.insert("pmRrcConnEstabAtt".to_string(), 100.0);
        
        let results = calculator.calculate_all_kpis(&counters);
        assert!(results.contains_key("rrc_conn_success_rate"));
    }
}