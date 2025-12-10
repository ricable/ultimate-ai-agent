//! Performance metrics and KPI definitions for RAN optimization

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{error::{RanError, RanResult}, TimeSeries, TimeSeriesPoint};

/// Key Performance Indicator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KpiType {
    // Throughput metrics
    /// Average cell throughput
    AverageCellThroughput,
    /// Peak cell throughput
    PeakCellThroughput,
    /// User experienced throughput
    UserExperiencedThroughput,
    
    // Latency metrics
    /// User plane latency
    UserPlaneLatency,
    /// Control plane latency
    ControlPlaneLatency,
    /// Round trip time
    RoundTripTime,
    
    // Reliability metrics
    /// Block Error Rate
    BlockErrorRate,
    /// Packet loss rate
    PacketLossRate,
    /// Service availability
    ServiceAvailability,
    
    // Efficiency metrics
    /// Spectral efficiency
    SpectralEfficiency,
    /// Energy efficiency
    EnergyEfficiency,
    /// Resource utilization
    ResourceUtilization,
    
    // Mobility metrics
    /// Handover success rate
    HandoverSuccessRate,
    /// Handover failure rate
    HandoverFailureRate,
    /// Inter-RAT handover rate
    InterRatHandoverRate,
    
    // Coverage metrics
    /// Coverage probability
    CoverageProbability,
    /// Signal strength
    SignalStrength,
    /// Signal quality
    SignalQuality,
    
    // Capacity metrics
    /// Cell load
    CellLoad,
    /// User density
    UserDensity,
    /// Traffic volume
    TrafficVolume,
    
    // Quality metrics
    /// Mean Opinion Score
    MeanOpinionScore,
    /// Video streaming quality
    VideoStreamingQuality,
    /// Voice quality
    VoiceQuality,
}

impl KpiType {
    /// Get the unit of measurement for this KPI
    pub fn unit(&self) -> &'static str {
        match self {
            KpiType::AverageCellThroughput | KpiType::PeakCellThroughput | 
            KpiType::UserExperiencedThroughput => "Mbps",
            
            KpiType::UserPlaneLatency | KpiType::ControlPlaneLatency | 
            KpiType::RoundTripTime => "ms",
            
            KpiType::BlockErrorRate | KpiType::PacketLossRate | 
            KpiType::ServiceAvailability | KpiType::HandoverSuccessRate | 
            KpiType::HandoverFailureRate | KpiType::InterRatHandoverRate | 
            KpiType::CoverageProbability | KpiType::CellLoad | 
            KpiType::ResourceUtilization => "%",
            
            KpiType::SpectralEfficiency => "bit/s/Hz",
            KpiType::EnergyEfficiency => "bit/J",
            KpiType::SignalStrength => "dBm",
            KpiType::SignalQuality => "dB",
            KpiType::UserDensity => "users/km²",
            KpiType::TrafficVolume => "GB",
            KpiType::MeanOpinionScore | KpiType::VideoStreamingQuality | 
            KpiType::VoiceQuality => "MOS",
        }
    }

    /// Get the typical range for this KPI
    pub fn typical_range(&self) -> (f64, f64) {
        match self {
            KpiType::AverageCellThroughput => (10.0, 1000.0),
            KpiType::PeakCellThroughput => (100.0, 5000.0),
            KpiType::UserExperiencedThroughput => (1.0, 100.0),
            
            KpiType::UserPlaneLatency => (1.0, 50.0),
            KpiType::ControlPlaneLatency => (10.0, 500.0),
            KpiType::RoundTripTime => (1.0, 100.0),
            
            KpiType::BlockErrorRate => (0.0, 10.0),
            KpiType::PacketLossRate => (0.0, 5.0),
            KpiType::ServiceAvailability => (95.0, 100.0),
            
            KpiType::SpectralEfficiency => (1.0, 10.0),
            KpiType::EnergyEfficiency => (1e6, 1e9),
            KpiType::ResourceUtilization => (0.0, 100.0),
            
            KpiType::HandoverSuccessRate => (90.0, 100.0),
            KpiType::HandoverFailureRate => (0.0, 10.0),
            KpiType::InterRatHandoverRate => (0.0, 20.0),
            
            KpiType::CoverageProbability => (80.0, 100.0),
            KpiType::SignalStrength => (-120.0, -50.0),
            KpiType::SignalQuality => (-10.0, 30.0),
            
            KpiType::CellLoad => (0.0, 100.0),
            KpiType::UserDensity => (10.0, 10000.0),
            KpiType::TrafficVolume => (0.1, 1000.0),
            
            KpiType::MeanOpinionScore | KpiType::VideoStreamingQuality | 
            KpiType::VoiceQuality => (1.0, 5.0),
        }
    }

    /// Check if higher values are better for this KPI
    pub fn higher_is_better(&self) -> bool {
        match self {
            KpiType::AverageCellThroughput | KpiType::PeakCellThroughput | 
            KpiType::UserExperiencedThroughput | KpiType::ServiceAvailability |
            KpiType::SpectralEfficiency | KpiType::EnergyEfficiency |
            KpiType::HandoverSuccessRate | KpiType::CoverageProbability |
            KpiType::SignalStrength | KpiType::SignalQuality |
            KpiType::MeanOpinionScore | KpiType::VideoStreamingQuality |
            KpiType::VoiceQuality => true,
            
            KpiType::UserPlaneLatency | KpiType::ControlPlaneLatency |
            KpiType::RoundTripTime | KpiType::BlockErrorRate |
            KpiType::PacketLossRate | KpiType::HandoverFailureRate |
            KpiType::InterRatHandoverRate => false,
            
            KpiType::ResourceUtilization | KpiType::CellLoad |
            KpiType::UserDensity | KpiType::TrafficVolume => true, // Context dependent
        }
    }
}

/// Individual Key Performance Indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kpi {
    /// KPI type
    pub kpi_type: KpiType,
    /// Current value
    pub value: f64,
    /// Target value (if applicable)
    pub target: Option<f64>,
    /// Threshold for alerts
    pub threshold: Option<f64>,
    /// Timestamp of last measurement
    pub timestamp: DateTime<Utc>,
    /// Historical values
    pub history: TimeSeries<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Kpi {
    /// Create a new KPI
    pub fn new(kpi_type: KpiType, value: f64) -> Self {
        let timestamp = Utc::now();
        let mut history = TimeSeries::new(format!("{:?}", kpi_type));
        history.add_point(TimeSeriesPoint::new(timestamp, value));
        
        Self {
            kpi_type,
            value,
            target: None,
            threshold: None,
            timestamp,
            history,
            metadata: HashMap::new(),
        }
    }

    /// Update the KPI value
    pub fn update_value(&mut self, value: f64) {
        self.value = value;
        self.timestamp = Utc::now();
        self.history.add_point(TimeSeriesPoint::new(self.timestamp, value));
    }

    /// Set target value
    pub fn set_target(&mut self, target: f64) {
        self.target = Some(target);
    }

    /// Set threshold for alerts
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = Some(threshold);
    }

    /// Check if KPI is meeting target
    pub fn is_meeting_target(&self) -> Option<bool> {
        self.target.map(|target| {
            if self.kpi_type.higher_is_better() {
                self.value >= target
            } else {
                self.value <= target
            }
        })
    }

    /// Check if KPI exceeds threshold
    pub fn exceeds_threshold(&self) -> Option<bool> {
        self.threshold.map(|threshold| {
            if self.kpi_type.higher_is_better() {
                self.value < threshold
            } else {
                self.value > threshold
            }
        })
    }

    /// Get KPI deviation from target as percentage
    pub fn target_deviation_percentage(&self) -> Option<f64> {
        self.target.map(|target| {
            if target == 0.0 {
                return 0.0;
            }
            ((self.value - target) / target) * 100.0
        })
    }

    /// Calculate trend over last N measurements
    pub fn calculate_trend(&self, n: usize) -> Option<f64> {
        if self.history.points.len() < 2 {
            return None;
        }
        
        let recent_points: Vec<_> = self.history.points
            .iter()
            .rev()
            .take(n.min(self.history.points.len()))
            .collect();
            
        if recent_points.len() < 2 {
            return None;
        }
        
        let first = recent_points.last().unwrap().value;
        let last = recent_points.first().unwrap().value;
        
        Some(((last - first) / first) * 100.0)
    }
}

/// Collection of performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Individual KPIs
    pub kpis: HashMap<KpiType, Kpi>,
    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,
    /// Metrics source identifier
    pub source_id: String,
}

impl PerformanceMetrics {
    /// Create new performance metrics collection
    pub fn new(source_id: String) -> Self {
        Self {
            kpis: HashMap::new(),
            last_updated: Utc::now(),
            source_id,
        }
    }

    /// Add or update a KPI
    pub fn update_kpi(&mut self, kpi_type: KpiType, value: f64) {
        match self.kpis.get_mut(&kpi_type) {
            Some(kpi) => kpi.update_value(value),
            None => {
                self.kpis.insert(kpi_type, Kpi::new(kpi_type, value));
            }
        }
        self.last_updated = Utc::now();
    }

    /// Get KPI value
    pub fn get_kpi_value(&self, kpi_type: KpiType) -> Option<f64> {
        self.kpis.get(&kpi_type).map(|kpi| kpi.value)
    }

    /// Get all KPI values as a map
    pub fn get_all_kpi_values(&self) -> HashMap<KpiType, f64> {
        self.kpis.iter()
            .map(|(kpi_type, kpi)| (*kpi_type, kpi.value))
            .collect()
    }

    /// Get KPIs that are not meeting targets
    pub fn get_underperforming_kpis(&self) -> Vec<&Kpi> {
        self.kpis.values()
            .filter(|kpi| kpi.is_meeting_target() == Some(false))
            .collect()
    }

    /// Get KPIs that exceed thresholds
    pub fn get_threshold_violations(&self) -> Vec<&Kpi> {
        self.kpis.values()
            .filter(|kpi| kpi.exceeds_threshold() == Some(true))
            .collect()
    }

    /// Calculate overall performance score (0-100)
    pub fn calculate_performance_score(&self) -> f64 {
        if self.kpis.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        let mut count = 0;

        for kpi in self.kpis.values() {
            if let Some(target) = kpi.target {
                let (min_val, max_val) = kpi.kpi_type.typical_range();
                let normalized_value = (kpi.value - min_val) / (max_val - min_val);
                let normalized_target = (target - min_val) / (max_val - min_val);
                
                let score = if kpi.kpi_type.higher_is_better() {
                    (normalized_value / normalized_target).min(1.0)
                } else {
                    (normalized_target / normalized_value).min(1.0)
                };
                
                total_score += score * 100.0;
                count += 1;
            }
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
}

/// Quality of Service metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSMetrics {
    /// Guaranteed Bit Rate in Mbps
    pub gbr: f64,
    /// Maximum Bit Rate in Mbps
    pub mbr: f64,
    /// Packet Delay Budget in ms
    pub pdb: f64,
    /// Packet Error Rate as percentage
    pub per: f64,
    /// Priority level (1-9, lower is higher priority)
    pub priority: u8,
    /// QoS Class Identifier
    pub qci: u8,
}

impl QoSMetrics {
    /// Create QoS metrics for a specific service type
    pub fn for_service_type(service_type: ServiceType) -> Self {
        match service_type {
            ServiceType::Voice => Self {
                gbr: 0.064, // 64 kbps
                mbr: 0.128, // 128 kbps
                pdb: 150.0, // 150 ms
                per: 1.0,   // 1%
                priority: 2,
                qci: 1,
            },
            ServiceType::Video => Self {
                gbr: 2.0,   // 2 Mbps
                mbr: 10.0,  // 10 Mbps
                pdb: 300.0, // 300 ms
                per: 0.1,   // 0.1%
                priority: 4,
                qci: 2,
            },
            ServiceType::Data => Self {
                gbr: 0.0,   // Non-GBR
                mbr: 100.0, // 100 Mbps
                pdb: 500.0, // 500 ms
                per: 0.01,  // 0.01%
                priority: 6,
                qci: 9,
            },
            ServiceType::IoT => Self {
                gbr: 0.001, // 1 kbps
                mbr: 0.1,   // 100 kbps
                pdb: 1000.0, // 1 second
                per: 5.0,   // 5%
                priority: 8,
                qci: 8,
            },
        }
    }

    /// Check if current performance meets QoS requirements
    pub fn meets_requirements(&self, current_metrics: &PerformanceMetrics) -> QoSCompliance {
        let mut violations = Vec::new();

        // Check throughput requirements
        if let Some(throughput) = current_metrics.get_kpi_value(KpiType::UserExperiencedThroughput) {
            if throughput < self.gbr {
                violations.push(format!("Throughput {} < GBR {}", throughput, self.gbr));
            }
        }

        // Check latency requirements
        if let Some(latency) = current_metrics.get_kpi_value(KpiType::UserPlaneLatency) {
            if latency > self.pdb {
                violations.push(format!("Latency {} > PDB {}", latency, self.pdb));
            }
        }

        // Check packet error rate
        if let Some(ber) = current_metrics.get_kpi_value(KpiType::BlockErrorRate) {
            if ber > self.per {
                violations.push(format!("BER {} > PER {}", ber, self.per));
            }
        }

        QoSCompliance {
            is_compliant: violations.is_empty(),
            violations,
        }
    }
}

/// QoS compliance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSCompliance {
    /// Whether QoS requirements are met
    pub is_compliant: bool,
    /// List of violations if any
    pub violations: Vec<String>,
}

/// Quality of Experience metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoEMetrics {
    /// Mean Opinion Score (1-5)
    pub mos: f64,
    /// Video quality score
    pub video_quality: f64,
    /// Audio quality score
    pub audio_quality: f64,
    /// Service accessibility (0-100%)
    pub accessibility: f64,
    /// Service retainability (0-100%)
    pub retainability: f64,
    /// Service integrity (0-100%)
    pub integrity: f64,
}

impl QoEMetrics {
    /// Create default QoE metrics
    pub fn default() -> Self {
        Self {
            mos: 3.0,
            video_quality: 3.0,
            audio_quality: 3.0,
            accessibility: 95.0,
            retainability: 98.0,
            integrity: 99.0,
        }
    }

    /// Calculate overall QoE score
    pub fn overall_score(&self) -> f64 {
        // Weighted average of different quality aspects
        let weights = [0.3, 0.25, 0.25, 0.1, 0.05, 0.05];
        let scores = [
            self.mos / 5.0,
            self.video_quality / 5.0,
            self.audio_quality / 5.0,
            self.accessibility / 100.0,
            self.retainability / 100.0,
            self.integrity / 100.0,
        ];

        weights.iter()
            .zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>() * 100.0
    }

    /// Determine service quality level
    pub fn quality_level(&self) -> QualityLevel {
        let score = self.overall_score();
        match score {
            s if s >= 90.0 => QualityLevel::Excellent,
            s if s >= 80.0 => QualityLevel::Good,
            s if s >= 60.0 => QualityLevel::Fair,
            s if s >= 40.0 => QualityLevel::Poor,
            _ => QualityLevel::Bad,
        }
    }
}

/// Service quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Excellent quality (90-100%)
    Excellent,
    /// Good quality (80-89%)
    Good,
    /// Fair quality (60-79%)
    Fair,
    /// Poor quality (40-59%)
    Poor,
    /// Bad quality (0-39%)
    Bad,
}

/// Service types for QoS classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceType {
    /// Voice communication
    Voice,
    /// Video streaming/calling
    Video,
    /// General data services
    Data,
    /// Internet of Things devices
    IoT,
}

/// Metric aggregation functions
pub struct MetricAggregator;

impl MetricAggregator {
    /// Calculate average of a metric over time
    pub fn average(timeseries: &TimeSeries<f64>) -> Option<f64> {
        if timeseries.points.is_empty() {
            return None;
        }
        
        let sum: f64 = timeseries.points.iter().map(|p| p.value).sum();
        Some(sum / timeseries.points.len() as f64)
    }

    /// Calculate percentile of a metric
    pub fn percentile(timeseries: &TimeSeries<f64>, percentile: f64) -> Option<f64> {
        if timeseries.points.is_empty() {
            return None;
        }
        
        let mut values: Vec<f64> = timeseries.points.iter().map(|p| p.value).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0 * (values.len() - 1) as f64).round() as usize;
        Some(values[index.min(values.len() - 1)])
    }

    /// Calculate standard deviation
    pub fn standard_deviation(timeseries: &TimeSeries<f64>) -> Option<f64> {
        let mean = Self::average(timeseries)?;
        if timeseries.points.len() < 2 {
            return Some(0.0);
        }
        
        let variance: f64 = timeseries.points
            .iter()
            .map(|p| (p.value - mean).powi(2))
            .sum::<f64>() / (timeseries.points.len() - 1) as f64;
            
        Some(variance.sqrt())
    }

    /// Calculate maximum value
    pub fn maximum(timeseries: &TimeSeries<f64>) -> Option<f64> {
        timeseries.points.iter().map(|p| p.value).fold(None, |acc, x| {
            Some(acc.map_or(x, |acc| acc.max(x)))
        })
    }

    /// Calculate minimum value
    pub fn minimum(timeseries: &TimeSeries<f64>) -> Option<f64> {
        timeseries.points.iter().map(|p| p.value).fold(None, |acc, x| {
            Some(acc.map_or(x, |acc| acc.min(x)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kpi_creation_and_update() {
        let mut kpi = Kpi::new(KpiType::AverageCellThroughput, 100.0);
        kpi.set_target(150.0);
        kpi.set_threshold(80.0);
        
        assert_eq!(kpi.value, 100.0);
        assert_eq!(kpi.target, Some(150.0));
        assert_eq!(kpi.is_meeting_target(), Some(false));
        
        kpi.update_value(160.0);
        assert_eq!(kpi.value, 160.0);
        assert_eq!(kpi.is_meeting_target(), Some(true));
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new("test_cell".to_string());
        
        metrics.update_kpi(KpiType::AverageCellThroughput, 100.0);
        metrics.update_kpi(KpiType::UserPlaneLatency, 20.0);
        
        assert_eq!(metrics.get_kpi_value(KpiType::AverageCellThroughput), Some(100.0));
        assert_eq!(metrics.get_kpi_value(KpiType::UserPlaneLatency), Some(20.0));
        assert_eq!(metrics.kpis.len(), 2);
    }

    #[test]
    fn test_qos_metrics() {
        let qos = QoSMetrics::for_service_type(ServiceType::Voice);
        assert_eq!(qos.qci, 1);
        assert_eq!(qos.priority, 2);
        assert!(qos.pdb <= 150.0);
    }

    #[test]
    fn test_qoe_metrics() {
        let qoe = QoEMetrics {
            mos: 4.0,
            video_quality: 4.5,
            audio_quality: 4.2,
            accessibility: 98.0,
            retainability: 99.0,
            integrity: 99.5,
        };
        
        assert!(qoe.overall_score() > 80.0);
        assert_eq!(qoe.quality_level(), QualityLevel::Good);
    }

    #[test]
    fn test_metric_aggregation() {
        let mut timeseries = TimeSeries::new("test".to_string());
        let now = Utc::now();
        
        for i in 0..10 {
            timeseries.add_point(TimeSeriesPoint::new(
                now + chrono::Duration::seconds(i),
                (i + 1) as f64,
            ));
        }
        
        assert_eq!(MetricAggregator::average(&timeseries), Some(5.5));
        assert_eq!(MetricAggregator::minimum(&timeseries), Some(1.0));
        assert_eq!(MetricAggregator::maximum(&timeseries), Some(10.0));
    }
}