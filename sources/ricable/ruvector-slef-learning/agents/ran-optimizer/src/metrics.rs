//! RAN Metrics Data Structures
//!
//! Comprehensive data structures for RAN KPIs and metrics.

use serde::{Deserialize, Serialize};

/// Complete cell metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMetrics {
    /// Cell identifier (e.g., "eNB123_Cell1")
    pub cell_id: String,
    /// Site name or location
    pub site_name: Option<String>,
    /// Timestamp of measurement
    pub timestamp: String,
    /// Technology type
    pub technology: Technology,
    /// Radio KPIs
    pub radio_kpis: RadioKpis,
    /// Throughput KPIs
    pub throughput_kpis: ThroughputKpis,
    /// Mobility KPIs
    pub mobility_kpis: MobilityKpis,
    /// Accessibility KPIs
    pub accessibility_kpis: AccessibilityKpis,
    /// Retainability KPIs
    pub retainability_kpis: RetainabilityKpis,
    /// Active alarms
    pub alarms: Vec<Alarm>,
    /// Cell configuration
    pub config: CellConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Technology {
    #[serde(rename = "LTE")]
    Lte,
    #[serde(rename = "5G_NSA")]
    Nr5gNsa,
    #[serde(rename = "5G_SA")]
    Nr5gSa,
    #[serde(rename = "NB_IOT")]
    NbIot,
}

/// Radio performance KPIs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RadioKpis {
    /// Reference Signal Received Power (dBm)
    pub rsrp_avg: Option<f32>,
    pub rsrp_p10: Option<f32>,
    pub rsrp_p90: Option<f32>,
    /// Reference Signal Received Quality (dB)
    pub rsrq_avg: Option<f32>,
    pub rsrq_p10: Option<f32>,
    /// Signal to Interference plus Noise Ratio (dB)
    pub sinr_avg: Option<f32>,
    pub sinr_p10: Option<f32>,
    pub sinr_p90: Option<f32>,
    /// Channel Quality Indicator (0-15)
    pub cqi_avg: Option<f32>,
    /// Timing Advance distribution
    pub ta_avg: Option<f32>,
    pub ta_max: Option<f32>,
    /// Physical Resource Block utilization (%)
    pub prb_dl_utilization: Option<f32>,
    pub prb_ul_utilization: Option<f32>,
    /// Interference level
    pub interference_level: Option<f32>,
    /// PUSCH BLER (%)
    pub pusch_bler: Option<f32>,
    /// PDSCH BLER (%)
    pub pdsch_bler: Option<f32>,
}

/// Throughput KPIs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThroughputKpis {
    /// Downlink throughput (Mbps)
    pub dl_throughput_avg: Option<f32>,
    pub dl_throughput_max: Option<f32>,
    pub dl_throughput_p10: Option<f32>,
    /// Uplink throughput (Mbps)
    pub ul_throughput_avg: Option<f32>,
    pub ul_throughput_max: Option<f32>,
    /// Cell throughput (Gbps)
    pub cell_throughput_dl: Option<f32>,
    pub cell_throughput_ul: Option<f32>,
    /// Data volume (GB)
    pub dl_volume_gb: Option<f32>,
    pub ul_volume_gb: Option<f32>,
    /// Spectral efficiency
    pub spectral_efficiency_dl: Option<f32>,
    pub spectral_efficiency_ul: Option<f32>,
}

/// Mobility KPIs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MobilityKpis {
    /// Handover attempts
    pub ho_attempts: Option<u32>,
    /// Handover success rate (%)
    pub ho_success_rate: Option<f32>,
    /// Intra-frequency HO success (%)
    pub intra_freq_ho_sr: Option<f32>,
    /// Inter-frequency HO success (%)
    pub inter_freq_ho_sr: Option<f32>,
    /// Inter-RAT HO success (%)
    pub inter_rat_ho_sr: Option<f32>,
    /// Ping-pong rate (%)
    pub pingpong_rate: Option<f32>,
    /// Too-early HO rate (%)
    pub too_early_ho_rate: Option<f32>,
    /// Too-late HO rate (%)
    pub too_late_ho_rate: Option<f32>,
    /// RRC re-establishment rate (%)
    pub rrc_reest_rate: Option<f32>,
}

/// Accessibility KPIs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessibilityKpis {
    /// RRC setup success rate (%)
    pub rrc_setup_sr: Option<f32>,
    /// E-RAB setup success rate (%)
    pub erab_setup_sr: Option<f32>,
    /// Initial E-RAB setup success rate (%)
    pub initial_erab_sr: Option<f32>,
    /// RACH success rate (%)
    pub rach_sr: Option<f32>,
    /// Connected users
    pub connected_users: Option<u32>,
    /// Max connected users
    pub max_connected_users: Option<u32>,
    /// Active users
    pub active_users: Option<u32>,
}

/// Retainability KPIs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetainabilityKpis {
    /// Call drop rate (%)
    pub call_drop_rate: Option<f32>,
    /// E-RAB drop rate (%)
    pub erab_drop_rate: Option<f32>,
    /// RRC abnormal release rate (%)
    pub rrc_abnormal_release: Option<f32>,
    /// Context release - radio failure (%)
    pub context_release_radio: Option<f32>,
    /// Context release - transport failure (%)
    pub context_release_transport: Option<f32>,
}

/// Cell alarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alarm {
    pub alarm_id: String,
    pub severity: AlarmSeverity,
    pub alarm_text: String,
    pub specific_problem: Option<String>,
    pub raised_time: String,
    pub additional_info: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlarmSeverity {
    Critical,
    Major,
    Minor,
    Warning,
}

/// Cell configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CellConfig {
    /// Physical Cell ID
    pub pci: Option<u16>,
    /// EARFCN/NR-ARFCN
    pub earfcn: Option<u32>,
    /// Bandwidth (MHz)
    pub bandwidth: Option<u8>,
    /// Transmit power (dBm)
    pub tx_power: Option<f32>,
    /// Reference signal power (dBm)
    pub rs_power: Option<f32>,
    /// Antenna configuration
    pub antenna: AntennaConfig,
    /// Handover parameters
    pub ho_params: HoParams,
    /// MIMO mode
    pub mimo_mode: Option<String>,
}

/// Antenna configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AntennaConfig {
    pub electrical_tilt: Option<f32>,
    pub mechanical_tilt: Option<f32>,
    pub azimuth: Option<f32>,
    pub height: Option<f32>,
    pub antenna_type: Option<String>,
    pub beamwidth: Option<f32>,
}

/// Handover parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HoParams {
    /// Time to trigger (ms)
    pub time_to_trigger: Option<u32>,
    /// Hysteresis (dB)
    pub hysteresis: Option<f32>,
    /// A3 offset (dB)
    pub a3_offset: Option<f32>,
    /// Filter coefficient
    pub filter_coefficient: Option<u8>,
    /// Cell individual offset (dB)
    pub cio: Option<f32>,
}

/// Neighboring cell relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborRelation {
    pub neighbor_cell_id: String,
    pub ho_attempts: u32,
    pub ho_success: u32,
    pub ho_success_rate: f32,
    pub cio: f32,
    pub is_configured: bool,
}

/// Site-level aggregated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteMetrics {
    pub site_id: String,
    pub site_name: String,
    pub location: Option<GeoLocation>,
    pub cells: Vec<CellMetrics>,
    pub total_users: u32,
    pub total_throughput_dl: f32,
    pub total_throughput_ul: f32,
    pub avg_availability: f32,
    pub power_consumption_kw: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f32>,
}

/// Traffic profile for time-based analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficProfile {
    pub cell_id: String,
    pub date: String,
    /// Hourly traffic volumes (24 entries)
    pub hourly_volume_dl: Vec<f32>,
    pub hourly_volume_ul: Vec<f32>,
    /// Hourly user counts
    pub hourly_users: Vec<u32>,
    /// Peak hour (0-23)
    pub peak_hour: u8,
    /// Low traffic hours
    pub low_traffic_hours: Vec<u8>,
}

impl CellMetrics {
    /// Calculate overall cell health score (0-100)
    pub fn health_score(&self) -> f32 {
        let mut score = 100.0;
        let mut penalties = 0.0;

        // Radio quality penalties
        if let Some(sinr) = self.radio_kpis.sinr_avg {
            if sinr < 0.0 { penalties += 20.0; }
            else if sinr < 5.0 { penalties += 10.0; }
            else if sinr < 10.0 { penalties += 5.0; }
        }

        // PRB utilization penalties
        if let Some(prb) = self.radio_kpis.prb_dl_utilization {
            if prb > 90.0 { penalties += 15.0; }
            else if prb > 80.0 { penalties += 10.0; }
            else if prb > 70.0 { penalties += 5.0; }
        }

        // Mobility penalties
        if let Some(ho_sr) = self.mobility_kpis.ho_success_rate {
            if ho_sr < 90.0 { penalties += 15.0; }
            else if ho_sr < 95.0 { penalties += 10.0; }
            else if ho_sr < 98.0 { penalties += 5.0; }
        }

        // Drop rate penalties
        if let Some(drop) = self.retainability_kpis.call_drop_rate {
            if drop > 2.0 { penalties += 20.0; }
            else if drop > 1.0 { penalties += 10.0; }
            else if drop > 0.5 { penalties += 5.0; }
        }

        // Alarm penalties
        for alarm in &self.alarms {
            match alarm.severity {
                AlarmSeverity::Critical => penalties += 15.0,
                AlarmSeverity::Major => penalties += 10.0,
                AlarmSeverity::Minor => penalties += 5.0,
                AlarmSeverity::Warning => penalties += 2.0,
            }
        }

        score -= penalties;
        score.max(0.0)
    }

    /// Identify primary issue category
    pub fn primary_issue(&self) -> Option<IssueCategory> {
        // Check for critical issues first
        if let Some(drop) = self.retainability_kpis.call_drop_rate {
            if drop > 2.0 { return Some(IssueCategory::Retainability); }
        }

        if let Some(ho_sr) = self.mobility_kpis.ho_success_rate {
            if ho_sr < 90.0 { return Some(IssueCategory::Mobility); }
        }

        if let Some(prb) = self.radio_kpis.prb_dl_utilization {
            if prb > 85.0 { return Some(IssueCategory::Capacity); }
        }

        if let Some(sinr) = self.radio_kpis.sinr_avg {
            if sinr < 5.0 { return Some(IssueCategory::Interference); }
        }

        if let Some(rsrp) = self.radio_kpis.rsrp_avg {
            if rsrp < -110.0 { return Some(IssueCategory::Coverage); }
        }

        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueCategory {
    Coverage,
    Capacity,
    Interference,
    Mobility,
    Accessibility,
    Retainability,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_score_good_cell() {
        let metrics = CellMetrics {
            cell_id: "TEST001".to_string(),
            site_name: None,
            timestamp: "2024-01-15T10:00:00Z".to_string(),
            technology: Technology::Lte,
            radio_kpis: RadioKpis {
                sinr_avg: Some(15.0),
                prb_dl_utilization: Some(50.0),
                ..Default::default()
            },
            throughput_kpis: Default::default(),
            mobility_kpis: MobilityKpis {
                ho_success_rate: Some(99.0),
                ..Default::default()
            },
            accessibility_kpis: Default::default(),
            retainability_kpis: RetainabilityKpis {
                call_drop_rate: Some(0.2),
                ..Default::default()
            },
            alarms: vec![],
            config: Default::default(),
        };

        let score = metrics.health_score();
        assert!(score >= 90.0);
    }

    #[test]
    fn test_health_score_problematic_cell() {
        let metrics = CellMetrics {
            cell_id: "TEST002".to_string(),
            site_name: None,
            timestamp: "2024-01-15T10:00:00Z".to_string(),
            technology: Technology::Lte,
            radio_kpis: RadioKpis {
                sinr_avg: Some(-2.0),  // Poor SINR
                prb_dl_utilization: Some(92.0),  // High utilization
                ..Default::default()
            },
            throughput_kpis: Default::default(),
            mobility_kpis: MobilityKpis {
                ho_success_rate: Some(88.0),  // Low HO SR
                ..Default::default()
            },
            accessibility_kpis: Default::default(),
            retainability_kpis: RetainabilityKpis {
                call_drop_rate: Some(2.5),  // High drop rate
                ..Default::default()
            },
            alarms: vec![
                Alarm {
                    alarm_id: "ALM001".to_string(),
                    severity: AlarmSeverity::Critical,
                    alarm_text: "High interference".to_string(),
                    specific_problem: None,
                    raised_time: "2024-01-15T09:00:00Z".to_string(),
                    additional_info: None,
                }
            ],
            config: Default::default(),
        };

        let score = metrics.health_score();
        assert!(score < 50.0);
    }

    #[test]
    fn test_primary_issue_detection() {
        let metrics = CellMetrics {
            cell_id: "TEST003".to_string(),
            site_name: None,
            timestamp: "2024-01-15T10:00:00Z".to_string(),
            technology: Technology::Lte,
            radio_kpis: RadioKpis {
                sinr_avg: Some(2.0),  // Poor SINR - interference
                ..Default::default()
            },
            throughput_kpis: Default::default(),
            mobility_kpis: Default::default(),
            accessibility_kpis: Default::default(),
            retainability_kpis: Default::default(),
            alarms: vec![],
            config: Default::default(),
        };

        assert!(matches!(metrics.primary_issue(), Some(IssueCategory::Interference)));
    }
}
