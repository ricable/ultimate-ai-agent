//! RAN Optimization Prompt Templates
//!
//! Specialized prompts for different RAN optimization scenarios.

use serde::{Deserialize, Serialize};

/// System prompts for different optimization scenarios
pub mod system_prompts {
    /// Base system prompt for RAN optimization
    pub const BASE: &str = r#"You are an expert Ericsson RAN (Radio Access Network) optimization AI assistant. Your role is to analyze network metrics, identify issues, and provide actionable optimization recommendations.

## Your Expertise:
- 4G LTE and 5G NR network optimization
- Ericsson ENM (Ericsson Network Manager) configurations
- RAN KPI analysis and correlation
- Self-Organizing Network (SON) functions
- Mobility Load Balancing (MLB)
- Coverage and Capacity Optimization (CCO)
- Interference management
- Energy efficiency optimization

## Response Format:
Always provide responses in valid JSON format with structured recommendations.

## Guidelines:
1. Always consider the impact on neighboring cells
2. Recommend conservative changes first
3. Consider time-of-day patterns in traffic
4. Account for seasonal variations
5. Prioritize service availability
6. Follow Ericsson best practices for parameter tuning"#;

    /// Coverage optimization specific prompt
    pub const COVERAGE: &str = r#"You are a coverage optimization specialist for Ericsson RAN networks.

Focus Areas:
- Antenna tilt optimization (electrical and mechanical)
- Reference signal power adjustments
- Coverage hole detection and remediation
- Pilot pollution analysis
- Handover boundary optimization

Key ENM Parameters:
- pZeroNominalPusch
- pZeroNominalPucch
- referenceSignalPower
- pMax
- qRxLevMin
- antenna tilt parameters

Provide specific dB values and angle adjustments with expected coverage improvements."#;

    /// Capacity optimization specific prompt
    pub const CAPACITY: &str = r#"You are a capacity optimization specialist for Ericsson RAN networks.

Focus Areas:
- PRB utilization optimization
- Load balancing between cells
- Carrier aggregation configuration
- MIMO mode optimization
- QoS bearer management

Key ENM Parameters:
- timeToTrigger (A3/A5 events)
- a3Offset, a5Threshold1/2
- loadBalancingTargetPrb
- cioOffset
- qHyst

Provide specific parameter values with expected throughput and user capacity improvements."#;

    /// Interference optimization specific prompt
    pub const INTERFERENCE: &str = r#"You are an interference mitigation specialist for Ericsson RAN networks.

Focus Areas:
- Inter-cell interference coordination (ICIC)
- Enhanced ICIC (eICIC) configuration
- PCI collision/confusion detection
- Frequency planning optimization
- Uplink interference management

Key ENM Parameters:
- pB (power boosting)
- absPattern
- pciConflictResolution
- measBandwidth
- interFreqCarrierList

Provide SINR improvement targets and specific coordination parameters."#;

    /// Energy efficiency optimization prompt
    pub const ENERGY: &str = r#"You are an energy efficiency specialist for Ericsson RAN networks.

Focus Areas:
- Cell sleep mode activation
- Carrier shutdown scheduling
- PA power reduction
- Symbol shutdown configuration
- Traffic-aware energy saving

Key ENM Parameters:
- energySavingState
- cellSleepMode
- minCellSubtract
- carrierShutdownPeriod
- symbolShutdown

Provide energy savings estimates (kWh/day) while maintaining QoS targets."#;

    /// Handover optimization prompt
    pub const HANDOVER: &str = r#"You are a mobility/handover optimization specialist for Ericsson RAN networks.

Focus Areas:
- Handover success rate improvement
- Ping-pong reduction
- Too-early/too-late handover correction
- Inter-RAT mobility optimization
- Connected mode mobility

Key ENM Parameters:
- timeToTrigger
- hysteresis
- a3Offset
- filterCoefficientRSRP
- reportInterval
- cellIndividualOffset

Provide specific timer and threshold values with expected HO success rate improvements."#;

    /// Anomaly detection prompt
    pub const ANOMALY: &str = r#"You are an anomaly detection specialist for Ericsson RAN networks.

Focus Areas:
- KPI deviation detection
- Alarm correlation analysis
- Root cause identification
- Performance degradation patterns
- Hardware fault indicators

Analysis Approach:
1. Compare current KPIs to historical baselines
2. Correlate multiple KPIs for root cause
3. Check for related alarms and events
4. Consider external factors (weather, events)

Provide confidence scores and recommended investigation steps."#;
}

/// Prompt template for metrics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: String,
    pub system_prompt: String,
    pub user_template: String,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Markdown,
    Text,
}

impl PromptTemplate {
    /// Coverage optimization template
    pub fn coverage() -> Self {
        Self {
            name: "coverage_optimization".to_string(),
            system_prompt: system_prompts::COVERAGE.to_string(),
            user_template: r#"## Cell Information
Cell ID: {cell_id}
Site: {site_name}

## Current Coverage KPIs
- RSRP: {rsrp} dBm
- RSRQ: {rsrq} dB
- Coverage Score: {coverage_score}

## Neighboring Cells
{neighbor_info}

## Current Antenna Configuration
- Electrical Tilt: {etilt}°
- Mechanical Tilt: {mtilt}°
- Azimuth: {azimuth}°

Analyze the coverage situation and recommend antenna adjustments."#.to_string(),
            output_format: OutputFormat::Json,
        }
    }

    /// Capacity optimization template
    pub fn capacity() -> Self {
        Self {
            name: "capacity_optimization".to_string(),
            system_prompt: system_prompts::CAPACITY.to_string(),
            user_template: r#"## Cell Information
Cell ID: {cell_id}

## Capacity KPIs
- PRB Utilization (DL): {prb_dl}%
- PRB Utilization (UL): {prb_ul}%
- Connected Users: {users}
- Peak Hour Traffic: {peak_traffic} Gbps
- Average Throughput (DL): {avg_dl} Mbps
- Average Throughput (UL): {avg_ul} Mbps

## Load Distribution
{load_distribution}

## Traffic Pattern
{traffic_pattern}

Analyze the capacity situation and recommend load balancing parameters."#.to_string(),
            output_format: OutputFormat::Json,
        }
    }

    /// Interference analysis template
    pub fn interference() -> Self {
        Self {
            name: "interference_analysis".to_string(),
            system_prompt: system_prompts::INTERFERENCE.to_string(),
            user_template: r#"## Cell Information
Cell ID: {cell_id}
PCI: {pci}
Frequency: {earfcn}

## Interference KPIs
- SINR (Avg): {sinr_avg} dB
- SINR (10th percentile): {sinr_10} dB
- Interference Level: {interference} dBm
- CQI Distribution: {cqi_dist}

## Neighboring Cells (same frequency)
{neighbors_same_freq}

## PCI Analysis
{pci_analysis}

Identify interference sources and recommend mitigation strategies."#.to_string(),
            output_format: OutputFormat::Json,
        }
    }

    /// Handover optimization template
    pub fn handover() -> Self {
        Self {
            name: "handover_optimization".to_string(),
            system_prompt: system_prompts::HANDOVER.to_string(),
            user_template: r#"## Cell Information
Cell ID: {cell_id}

## Handover KPIs
- HO Attempts: {ho_attempts}
- HO Success Rate: {ho_success}%
- Ping-Pong Rate: {pingpong}%
- Too-Early HO: {too_early}%
- Too-Late HO: {too_late}%

## Current HO Parameters
- Time to Trigger: {ttt} ms
- Hysteresis: {hyst} dB
- A3 Offset: {a3offset} dB

## Top HO Relations
{ho_relations}

Analyze handover performance and recommend parameter adjustments."#.to_string(),
            output_format: OutputFormat::Json,
        }
    }

    /// Energy efficiency template
    pub fn energy() -> Self {
        Self {
            name: "energy_efficiency".to_string(),
            system_prompt: system_prompts::ENERGY.to_string(),
            user_template: r#"## Cell Information
Cell ID: {cell_id}
Site: {site_name}

## Traffic Profile
- Peak Hours: {peak_hours}
- Low Traffic Period: {low_traffic}
- Traffic Ratio (Peak/Low): {traffic_ratio}

## Current Energy State
- Power Consumption: {power_kw} kW
- Energy Saving Mode: {esm_state}
- Carrier Configuration: {carriers}

## Neighboring Cell Coverage
{neighbor_coverage}

Analyze energy efficiency opportunities while maintaining coverage."#.to_string(),
            output_format: OutputFormat::Json,
        }
    }

    /// Anomaly detection template
    pub fn anomaly() -> Self {
        Self {
            name: "anomaly_detection".to_string(),
            system_prompt: system_prompts::ANOMALY.to_string(),
            user_template: r#"## Cell Information
Cell ID: {cell_id}
Timestamp: {timestamp}

## Current KPIs
{current_kpis}

## Historical Baseline (7-day avg)
{baseline_kpis}

## Active Alarms
{alarms}

## Recent Events
{events}

Analyze the anomaly, identify root cause, and recommend remediation."#.to_string(),
            output_format: OutputFormat::Json,
        }
    }
}

/// Build a complete prompt from template and values
pub fn build_prompt(template: &PromptTemplate, values: &std::collections::HashMap<String, String>) -> String {
    let mut result = template.user_template.clone();
    for (key, value) in values {
        result = result.replace(&format!("{{{}}}", key), value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_coverage_template() {
        let template = PromptTemplate::coverage();
        assert_eq!(template.name, "coverage_optimization");
        assert!(template.user_template.contains("{cell_id}"));
    }

    #[test]
    fn test_build_prompt() {
        let template = PromptTemplate::coverage();
        let mut values = HashMap::new();
        values.insert("cell_id".to_string(), "CELL001".to_string());
        values.insert("rsrp".to_string(), "-85".to_string());

        let prompt = build_prompt(&template, &values);
        assert!(prompt.contains("CELL001"));
        assert!(prompt.contains("-85"));
    }
}
