use crate::{Config, Result, Error, EndcPredictionEngine, EndcMetrics, FailureAnalysis, FailureMitigation};
use crate::proto::{
    PredictEndcFailureRequest, PredictEndcFailureResponse,
    Get5GServiceHealthRequest, Get5GServiceHealthResponse,
    ServiceHealthReport, ServiceHealthSummary,
    HealthCheckRequest, HealthCheckResponse,
};
use tonic::{Request, Response, Status};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use std::collections::HashMap;

pub struct EndcService {
    config: Config,
    prediction_engine: Arc<RwLock<EndcPredictionEngine>>,
    health_monitor: Arc<RwLock<HealthMonitor>>,
}

#[derive(Debug)]
struct HealthMonitor {
    cell_health_reports: HashMap<String, ServiceHealthReport>,
    last_update: chrono::DateTime<chrono::Utc>,
}

impl EndcService {
    pub async fn new(config: Config) -> Result<Self> {
        config.validate()?;
        
        let prediction_engine = EndcPredictionEngine::new(config.clone())?;
        let health_monitor = HealthMonitor {
            cell_health_reports: HashMap::new(),
            last_update: chrono::Utc::now(),
        };
        
        Ok(Self {
            config,
            prediction_engine: Arc::new(RwLock::new(prediction_engine)),
            health_monitor: Arc::new(RwLock::new(health_monitor)),
        })
    }
    
    fn build_failure_analysis(&self, metrics: &EndcMetrics, prediction_result: &crate::models::FailurePredictionResult) -> FailureAnalysis {
        let signal_quality_score = self.calculate_signal_quality_score(
            metrics.lte_rsrp, 
            metrics.lte_sinr, 
            metrics.nr_ssb_rsrp, 
            metrics.nr_ssb_sinr
        );
        
        let network_congestion_score = (metrics.data_volume_mbps / 100.0).clamp(0.0, 1.0);
        let ue_capability_score = 0.8; // Simplified
        
        FailureAnalysis {
            ue_id: metrics.ue_id.clone(),
            root_cause: self.determine_root_cause(&prediction_result.predicted_failure_type),
            contributing_factors: prediction_result.contributing_factors.clone(),
            signal_quality_score,
            network_congestion_score,
            ue_capability_score,
            risk_level: prediction_result.risk_level.clone(),
        }
    }
    
    fn determine_root_cause(&self, failure_type: &str) -> String {
        match failure_type {
            "INITIAL_SETUP" => "Poor signal conditions or network congestion preventing initial ENDC establishment".to_string(),
            "BEARER_SETUP" => "Bearer configuration mismatch or resource unavailability".to_string(),
            "BEARER_MODIFICATION" => "Dynamic bearer modification failure due to changing network conditions".to_string(),
            "RELEASE" => "Unexpected connection release due to signaling issues or interference".to_string(),
            _ => "Unknown failure type requiring manual investigation".to_string(),
        }
    }
    
    fn build_mitigation_strategies(&self, failure_type: &str, confidence: f64) -> Vec<FailureMitigation> {
        let mut mitigations = Vec::new();
        
        if confidence < self.config.prediction.confidence_threshold {
            mitigations.push(FailureMitigation {
                strategy: "MONITORING".to_string(),
                effectiveness: 0.6,
                implementation: "Increase monitoring frequency and data collection".to_string(),
                estimated_improvement: 0.3,
                priority: "MEDIUM".to_string(),
            });
            return mitigations;
        }
        
        match failure_type {
            "INITIAL_SETUP" => {
                mitigations.extend([
                    FailureMitigation {
                        strategy: "CELL_RESELECTION".to_string(),
                        effectiveness: 0.78,
                        implementation: "Force UE to select cells with better ENDC capability".to_string(),
                        estimated_improvement: 0.70,
                        priority: "HIGH".to_string(),
                    },
                    FailureMitigation {
                        strategy: "PARAMETER_OPTIMIZATION".to_string(),
                        effectiveness: 0.82,
                        implementation: "Optimize ENDC setup timers and thresholds".to_string(),
                        estimated_improvement: 0.68,
                        priority: "MEDIUM".to_string(),
                    },
                ]);
            }
            "BEARER_SETUP" => {
                mitigations.extend([
                    FailureMitigation {
                        strategy: "BEARER_RECONFIGURATION".to_string(),
                        effectiveness: 0.85,
                        implementation: "Reconfigure bearer types and QoS parameters".to_string(),
                        estimated_improvement: 0.75,
                        priority: "HIGH".to_string(),
                    },
                    FailureMitigation {
                        strategy: "LOAD_BALANCING".to_string(),
                        effectiveness: 0.75,
                        implementation: "Redistribute load to reduce resource contention".to_string(),
                        estimated_improvement: 0.65,
                        priority: "MEDIUM".to_string(),
                    },
                ]);
            }
            "BEARER_MODIFICATION" => {
                mitigations.push(FailureMitigation {
                    strategy: "BEARER_RECONFIGURATION".to_string(),
                    effectiveness: 0.80,
                    implementation: "Stabilize bearer configuration to prevent modifications".to_string(),
                    estimated_improvement: 0.70,
                    priority: "HIGH".to_string(),
                });
            }
            "RELEASE" => {
                mitigations.push(FailureMitigation {
                    strategy: "PARAMETER_OPTIMIZATION".to_string(),
                    effectiveness: 0.77,
                    implementation: "Adjust release timers and retry mechanisms".to_string(),
                    estimated_improvement: 0.60,
                    priority: "MEDIUM".to_string(),
                });
            }
            _ => {
                mitigations.push(FailureMitigation {
                    strategy: "MANUAL_INVESTIGATION".to_string(),
                    effectiveness: 0.50,
                    implementation: "Manual analysis and troubleshooting required".to_string(),
                    estimated_improvement: 0.40,
                    priority: "LOW".to_string(),
                });
            }
        }
        
        mitigations
    }
    
    fn calculate_signal_quality_score(&self, lte_rsrp: f64, lte_sinr: f64, nr_rsrp: f64, nr_sinr: f64) -> f64 {
        let lte_rsrp_norm = ((lte_rsrp + 140.0) / 90.0).clamp(0.0, 1.0);
        let lte_sinr_norm = (lte_sinr / 30.0).clamp(0.0, 1.0);
        let nr_rsrp_norm = ((nr_rsrp + 140.0) / 90.0).clamp(0.0, 1.0);
        let nr_sinr_norm = (nr_sinr / 40.0).clamp(0.0, 1.0);
        
        (lte_rsrp_norm * 0.25 + lte_sinr_norm * 0.25 + nr_rsrp_norm * 0.25 + nr_sinr_norm * 0.25)
    }
}

#[tonic::async_trait]
impl crate::proto::service_assurance_service_server::ServiceAssuranceService for EndcService {
    async fn predict_endc_failure(
        &self,
        request: Request<PredictEndcFailureRequest>,
    ) -> std::result::Result<Response<PredictEndcFailureResponse>, Status> {
        let req = request.into_inner();
        
        info!("Processing ENDC failure prediction request for UE: {}", req.ue_id);
        
        // Validate request
        if req.metrics.is_none() {
            return Err(Status::invalid_argument("ENDC metrics required"));
        }
        
        let metrics = req.metrics.unwrap();
        
        // Make prediction
        let engine = self.prediction_engine.read().await;
        let prediction_result = engine.predict_failure(&metrics)
            .map_err(|e| Status::internal(format!("Prediction failed: {}", e)))?;
        
        // Build analysis and mitigations
        let analysis = self.build_failure_analysis(&metrics, &prediction_result);
        let mitigations = self.build_mitigation_strategies(
            &prediction_result.predicted_failure_type, 
            prediction_result.confidence
        );
        
        let response = PredictEndcFailureResponse {
            ue_id: req.ue_id.clone(),
            failure_probability: prediction_result.failure_probability,
            predicted_failure_type: prediction_result.predicted_failure_type,
            confidence: prediction_result.confidence,
            analysis: Some(analysis),
            mitigations,
            metadata: req.metadata,
        };
        
        info!(
            "ENDC prediction completed for UE {}: {} (probability: {:.3}, confidence: {:.3})",
            req.ue_id, response.predicted_failure_type, response.failure_probability, response.confidence
        );
        
        Ok(Response::new(response))
    }
    
    async fn get5_g_service_health(
        &self,
        request: Request<Get5GServiceHealthRequest>,
    ) -> std::result::Result<Response<Get5GServiceHealthResponse>, Status> {
        let req = request.into_inner();
        
        // Get health reports for requested cells
        let monitor = self.health_monitor.read().await;
        let mut reports = Vec::new();
        
        for cell_id in &req.cell_ids {
            if let Some(report) = monitor.cell_health_reports.get(cell_id) {
                reports.push(report.clone());
            } else {
                // Create default report for unknown cells
                reports.push(ServiceHealthReport {
                    cell_id: cell_id.clone(),
                    endc_success_rate: 0.95,
                    bearer_setup_success_rate: 0.93,
                    handover_success_rate: 0.90,
                    throughput_performance: 0.85,
                    latency_performance: 0.88,
                    health_status: "HEALTHY".to_string(),
                    issues: vec![],
                    last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                });
            }
        }
        
        // Calculate summary
        let total_cells = reports.len() as i32;
        let healthy_cells = reports.iter()
            .filter(|r| r.health_status == "HEALTHY")
            .count() as i32;
        let degraded_cells = reports.iter()
            .filter(|r| r.health_status == "DEGRADED")
            .count() as i32;
        let critical_cells = reports.iter()
            .filter(|r| r.health_status == "CRITICAL")
            .count() as i32;
        
        let avg_endc_success_rate = if !reports.is_empty() {
            reports.iter().map(|r| r.endc_success_rate).sum::<f64>() / reports.len() as f64
        } else {
            0.0
        };
        
        let avg_throughput_performance = if !reports.is_empty() {
            reports.iter().map(|r| r.throughput_performance).sum::<f64>() / reports.len() as f64
        } else {
            0.0
        };
        
        let network_health_score = (avg_endc_success_rate + avg_throughput_performance) / 2.0;
        
        let summary = ServiceHealthSummary {
            total_cells,
            healthy_cells,
            degraded_cells,
            critical_cells,
            average_endc_success_rate: avg_endc_success_rate,
            average_throughput_performance: avg_throughput_performance,
            network_health_score,
        };
        
        let response = Get5GServiceHealthResponse {
            reports,
            summary: Some(summary),
            metadata: req.metadata,
        };
        
        Ok(Response::new(response))
    }
    
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        let health = ran_common::HealthCheck {
            status: ran_common::HealthStatus::Serving as i32,
            message: "ASA-5G-01 ENDC Failure Predictor is healthy".to_string(),
            timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            details: std::collections::HashMap::new(),
        };
        
        let response = HealthCheckResponse {
            health: Some(health),
        };
        
        Ok(Response::new(response))
    }
}