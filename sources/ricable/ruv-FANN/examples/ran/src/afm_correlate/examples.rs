use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use tokio;

use super::{
    CorrelationEngine, CorrelationConfig, EvidenceItem, EvidenceSource,
    CorrelationResult, EvidenceBundle, ImpactAssessment, SynthesisReport,
};

/// Example usage of the AFM-Correlate correlation engine
pub struct CorrelationEngineExample {
    engine: CorrelationEngine,
}

impl CorrelationEngineExample {
    pub fn new() -> Self {
        let config = CorrelationConfig {
            temporal_window: Duration::minutes(15),
            min_correlation_score: 0.7,
            max_evidence_items: 100,
            attention_heads: 8,
            hidden_dim: 256,
            dropout_rate: 0.1,
        };
        
        Self {
            engine: CorrelationEngine::new(config),
        }
    }

    /// Demonstrate KPI deviation correlation
    pub async fn demonstrate_kpi_correlation(&self) -> Vec<EvidenceBundle> {
        println!("=== KPI Deviation Correlation Demo ===");
        
        // Simulate KPI deviation events
        let base_time = Utc::now();
        let kpi_evidence = vec![
            EvidenceItem {
                id: "kpi_cpu_spike_001".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: base_time,
                severity: 0.85,
                confidence: 0.92,
                features: vec![0.85, 0.12, 0.67, 0.34, 0.89],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("metric".to_string(), "cpu_utilization".to_string());
                    meta.insert("component".to_string(), "compute_node_1".to_string());
                    meta.insert("threshold".to_string(), "80%".to_string());
                    meta.insert("actual".to_string(), "94%".to_string());
                    meta
                },
            },
            EvidenceItem {
                id: "kpi_memory_pressure_001".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: base_time + Duration::seconds(45),
                severity: 0.78,
                confidence: 0.89,
                features: vec![0.78, 0.34, 0.56, 0.23, 0.91],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("metric".to_string(), "memory_utilization".to_string());
                    meta.insert("component".to_string(), "compute_node_1".to_string());
                    meta.insert("threshold".to_string(), "85%".to_string());
                    meta.insert("actual".to_string(), "93%".to_string());
                    meta
                },
            },
            EvidenceItem {
                id: "kpi_network_latency_001".to_string(),
                source: EvidenceSource::PerformanceMetric,
                timestamp: base_time + Duration::seconds(120),
                severity: 0.72,
                confidence: 0.87,
                features: vec![0.72, 0.45, 0.61, 0.38, 0.84],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("metric".to_string(), "network_latency".to_string());
                    meta.insert("component".to_string(), "network_interface_1".to_string());
                    meta.insert("threshold".to_string(), "10ms".to_string());
                    meta.insert("actual".to_string(), "28ms".to_string());
                    meta
                },
            },
        ];
        
        // Add evidence to correlation engine
        for evidence in kpi_evidence {
            self.engine.add_evidence(evidence).await;
        }
        
        // Wait for correlation window
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        // Perform correlation
        let bundles = self.engine.correlate().await;
        
        // Print results
        for bundle in &bundles {
            println!("Bundle ID: {}", bundle.bundle_id);
            println!("Hierarchical Score: {:.3}", bundle.hierarchical_score);
            println!("Correlations: {}", bundle.correlations.len());
            
            for correlation in &bundle.correlations {
                println!("  - Correlation: {} (score: {:.3})", 
                    correlation.correlation_id, correlation.correlation_score);
                println!("    Evidence items: {}", correlation.evidence_items.len());
                println!("    Impact severity: {:.3}", correlation.impact_assessment.severity);
                println!("    Propagation risk: {:.3}", correlation.impact_assessment.propagation_risk);
            }
        }
        
        bundles
    }

    /// Demonstrate alarm sequence correlation
    pub async fn demonstrate_alarm_correlation(&self) -> Vec<EvidenceBundle> {
        println!("\n=== Alarm Sequence Correlation Demo ===");
        
        let base_time = Utc::now();
        let alarm_evidence = vec![
            EvidenceItem {
                id: "alarm_hardware_failure_001".to_string(),
                source: EvidenceSource::AlarmSequence,
                timestamp: base_time,
                severity: 0.95,
                confidence: 0.98,
                features: vec![0.95, 0.23, 0.78, 0.56, 0.89],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("alarm_type".to_string(), "hardware_failure".to_string());
                    meta.insert("component".to_string(), "baseband_unit_3".to_string());
                    meta.insert("alarm_code".to_string(), "HW_FAIL_001".to_string());
                    meta.insert("description".to_string(), "Baseband processing unit failure detected".to_string());
                    meta
                },
            },
            EvidenceItem {
                id: "alarm_service_degradation_001".to_string(),
                source: EvidenceSource::AlarmSequence,
                timestamp: base_time + Duration::seconds(30),
                severity: 0.82,
                confidence: 0.91,
                features: vec![0.82, 0.41, 0.65, 0.47, 0.76],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("alarm_type".to_string(), "service_degradation".to_string());
                    meta.insert("component".to_string(), "cell_sector_7".to_string());
                    meta.insert("alarm_code".to_string(), "SVC_DEG_002".to_string());
                    meta.insert("description".to_string(), "Service quality degradation in sector 7".to_string());
                    meta
                },
            },
            EvidenceItem {
                id: "config_rollback_001".to_string(),
                source: EvidenceSource::ConfigurationChange,
                timestamp: base_time + Duration::seconds(180),
                severity: 0.65,
                confidence: 0.85,
                features: vec![0.65, 0.32, 0.54, 0.28, 0.73],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("change_type".to_string(), "automatic_rollback".to_string());
                    meta.insert("component".to_string(), "baseband_unit_3".to_string());
                    meta.insert("reason".to_string(), "hardware_failure_recovery".to_string());
                    meta.insert("description".to_string(), "Automatic configuration rollback due to hardware failure".to_string());
                    meta
                },
            },
        ];
        
        // Add evidence
        for evidence in alarm_evidence {
            self.engine.add_evidence(evidence).await;
        }
        
        // Wait and correlate
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        let bundles = self.engine.correlate().await;
        
        // Print results
        for bundle in &bundles {
            println!("Bundle ID: {}", bundle.bundle_id);
            println!("Synthesis Report:");
            if let Some(primary_cause) = &bundle.synthesis_report.primary_cause {
                println!("  Primary Cause: {}", primary_cause);
            }
            println!("  Contributing Factors: {}", bundle.synthesis_report.contributing_factors.len());
            for factor in &bundle.synthesis_report.contributing_factors {
                println!("    - {}", factor);
            }
            println!("  Recommendations: {}", bundle.synthesis_report.recommendations.len());
            for rec in &bundle.synthesis_report.recommendations {
                println!("    - {}", rec);
            }
        }
        
        bundles
    }

    /// Demonstrate cross-domain correlation
    pub async fn demonstrate_cross_domain_correlation(&self) -> Vec<EvidenceBundle> {
        println!("\n=== Cross-Domain Correlation Demo ===");
        
        let base_time = Utc::now();
        let cross_domain_evidence = vec![
            // Topology change
            EvidenceItem {
                id: "topology_change_001".to_string(),
                source: EvidenceSource::TopologyImpact,
                timestamp: base_time,
                severity: 0.75,
                confidence: 0.88,
                features: vec![0.75, 0.28, 0.62, 0.44, 0.81],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("change_type".to_string(), "neighbor_relation_update".to_string());
                    meta.insert("component".to_string(), "cell_cluster_A".to_string());
                    meta.insert("impact".to_string(), "handover_parameters_modified".to_string());
                    meta.insert("description".to_string(), "Neighbor cell relations updated for optimization".to_string());
                    meta
                },
            },
            // KPI deviation following topology change
            EvidenceItem {
                id: "kpi_handover_failure_001".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: base_time + Duration::seconds(90),
                severity: 0.83,
                confidence: 0.91,
                features: vec![0.83, 0.35, 0.71, 0.52, 0.87],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("metric".to_string(), "handover_failure_rate".to_string());
                    meta.insert("component".to_string(), "cell_cluster_A".to_string());
                    meta.insert("threshold".to_string(), "2%".to_string());
                    meta.insert("actual".to_string(), "7.3%".to_string());
                    meta
                },
            },
            // Performance metric degradation
            EvidenceItem {
                id: "perf_throughput_drop_001".to_string(),
                source: EvidenceSource::PerformanceMetric,
                timestamp: base_time + Duration::seconds(150),
                severity: 0.79,
                confidence: 0.86,
                features: vec![0.79, 0.42, 0.68, 0.39, 0.82],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("metric".to_string(), "cell_throughput".to_string());
                    meta.insert("component".to_string(), "cell_cluster_A".to_string());
                    meta.insert("baseline".to_string(), "150 Mbps".to_string());
                    meta.insert("current".to_string(), "98 Mbps".to_string());
                    meta
                },
            },
            // Log pattern indicating issues
            EvidenceItem {
                id: "log_pattern_handover_001".to_string(),
                source: EvidenceSource::LogPattern,
                timestamp: base_time + Duration::seconds(200),
                severity: 0.68,
                confidence: 0.82,
                features: vec![0.68, 0.31, 0.59, 0.36, 0.74],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("pattern".to_string(), "handover_timeout_errors".to_string());
                    meta.insert("component".to_string(), "cell_cluster_A".to_string());
                    meta.insert("frequency".to_string(), "high".to_string());
                    meta.insert("description".to_string(), "Increased handover timeout errors in affected cells".to_string());
                    meta
                },
            },
        ];
        
        // Add evidence
        for evidence in cross_domain_evidence {
            self.engine.add_evidence(evidence).await;
        }
        
        // Wait and correlate
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        let bundles = self.engine.correlate().await;
        
        // Print results
        for bundle in &bundles {
            println!("Bundle ID: {}", bundle.bundle_id);
            println!("Multi-scale features: {}", bundle.multi_scale_features.len());
            
            for correlation in &bundle.correlations {
                println!("  Cross-domain correlation: {}", correlation.correlation_id);
                println!("    Cross-domain score: {:.3}", correlation.cross_domain_score);
                println!("    Temporal alignment: {:.3}", correlation.temporal_alignment);
                println!("    Affected components: {:?}", correlation.impact_assessment.affected_components);
                
                // Show evidence source diversity
                let mut sources = std::collections::HashSet::new();
                for item in &correlation.evidence_items {
                    sources.insert(item.source);
                }
                println!("    Evidence sources: {:?}", sources);
            }
        }
        
        bundles
    }

    /// Demonstrate temporal alignment
    pub async fn demonstrate_temporal_alignment(&self) -> Vec<EvidenceBundle> {
        println!("\n=== Temporal Alignment Demo ===");
        
        let base_time = Utc::now();
        let temporal_evidence = vec![
            // Burst pattern
            EvidenceItem {
                id: "burst_001".to_string(),
                source: EvidenceSource::AlarmSequence,
                timestamp: base_time,
                severity: 0.85,
                confidence: 0.91,
                features: vec![0.85, 0.23, 0.67, 0.45, 0.82],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "burst_002".to_string(),
                source: EvidenceSource::AlarmSequence,
                timestamp: base_time + Duration::seconds(15),
                severity: 0.88,
                confidence: 0.93,
                features: vec![0.88, 0.26, 0.71, 0.48, 0.85],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "burst_003".to_string(),
                source: EvidenceSource::AlarmSequence,
                timestamp: base_time + Duration::seconds(30),
                severity: 0.91,
                confidence: 0.95,
                features: vec![0.91, 0.29, 0.75, 0.52, 0.88],
                metadata: HashMap::new(),
            },
            // Cascade pattern
            EvidenceItem {
                id: "cascade_001".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: base_time + Duration::seconds(120),
                severity: 0.72,
                confidence: 0.87,
                features: vec![0.72, 0.35, 0.61, 0.38, 0.79],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "cascade_002".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: base_time + Duration::seconds(180),
                severity: 0.79,
                confidence: 0.89,
                features: vec![0.79, 0.41, 0.68, 0.42, 0.83],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "cascade_003".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: base_time + Duration::seconds(240),
                severity: 0.86,
                confidence: 0.92,
                features: vec![0.86, 0.47, 0.74, 0.49, 0.87],
                metadata: HashMap::new(),
            },
        ];
        
        // Add evidence
        for evidence in temporal_evidence {
            self.engine.add_evidence(evidence).await;
        }
        
        // Wait and correlate
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        let bundles = self.engine.correlate().await;
        
        // Print temporal analysis results
        for bundle in &bundles {
            println!("Bundle ID: {}", bundle.bundle_id);
            println!("Timeline events: {}", bundle.synthesis_report.timeline.len());
            
            for event in &bundle.synthesis_report.timeline {
                println!("  {:?} - {}: {}", 
                    event.timestamp.format("%H:%M:%S"), 
                    event.event_type, 
                    event.description);
            }
        }
        
        bundles
    }

    /// Run comprehensive correlation demonstration
    pub async fn run_comprehensive_demo(&self) {
        println!("🔍 AFM-Correlate Comprehensive Demonstration");
        println!("============================================");
        
        let kpi_bundles = self.demonstrate_kpi_correlation().await;
        println!("✓ KPI correlation demo completed: {} bundles", kpi_bundles.len());
        
        let alarm_bundles = self.demonstrate_alarm_correlation().await;
        println!("✓ Alarm correlation demo completed: {} bundles", alarm_bundles.len());
        
        let cross_domain_bundles = self.demonstrate_cross_domain_correlation().await;
        println!("✓ Cross-domain correlation demo completed: {} bundles", cross_domain_bundles.len());
        
        let temporal_bundles = self.demonstrate_temporal_alignment().await;
        println!("✓ Temporal alignment demo completed: {} bundles", temporal_bundles.len());
        
        let total_bundles = kpi_bundles.len() + alarm_bundles.len() + 
                           cross_domain_bundles.len() + temporal_bundles.len();
        
        println!("\n📊 Summary:");
        println!("Total evidence bundles generated: {}", total_bundles);
        println!("Correlation engine successfully demonstrated hierarchical attention");
        println!("for multi-scale evidence correlation across domains.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_correlation_examples() {
        let example = CorrelationEngineExample::new();
        
        // Test KPI correlation
        let kpi_bundles = example.demonstrate_kpi_correlation().await;
        assert!(!kpi_bundles.is_empty(), "KPI correlation should produce bundles");
        
        // Test alarm correlation
        let alarm_bundles = example.demonstrate_alarm_correlation().await;
        assert!(!alarm_bundles.is_empty(), "Alarm correlation should produce bundles");
        
        // Test cross-domain correlation
        let cross_bundles = example.demonstrate_cross_domain_correlation().await;
        assert!(!cross_bundles.is_empty(), "Cross-domain correlation should produce bundles");
        
        // Test temporal alignment
        let temporal_bundles = example.demonstrate_temporal_alignment().await;
        assert!(!temporal_bundles.is_empty(), "Temporal alignment should produce bundles");
    }
    
    #[tokio::test]
    async fn test_comprehensive_demo() {
        let example = CorrelationEngineExample::new();
        example.run_comprehensive_demo().await;
    }
}