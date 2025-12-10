//! gRPC Services
//!
//! Provides inter-service communication via gRPC with health checks and service discovery.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum GrpcError {
    #[error("Service not available: {0}")]
    ServiceUnavailable(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub service_name: String,
    pub host: String,
    pub port: u16,
    pub health_check_interval_secs: u64,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: ServiceStatus,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub uptime_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisRequest {
    pub request_id: String,
    pub target_sequence: String,
    pub analysis_type: AnalysisType,
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnalysisType {
    OffTargetPrediction,
    ImmuneResponse,
    ComprehensiveAnalysis,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisResponse {
    pub request_id: String,
    pub status: AnalysisStatus,
    pub results: serde_json::Value,
    pub processing_time_ms: u64,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnalysisStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// gRPC service registry
pub struct ServiceRegistry {
    services: Arc<RwLock<std::collections::HashMap<String, RegisteredService>>>,
}

#[derive(Clone, Debug)]
struct RegisteredService {
    config: ServiceConfig,
    status: ServiceStatus,
    last_health_check: chrono::DateTime<chrono::Utc>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Register a new service
    pub async fn register_service(&self, config: ServiceConfig) -> anyhow::Result<()> {
        info!(
            "Registering service: {} at {}:{}",
            config.service_name, config.host, config.port
        );

        let service = RegisteredService {
            config: config.clone(),
            status: ServiceStatus::Healthy,
            last_health_check: chrono::Utc::now(),
        };

        self.services
            .write()
            .await
            .insert(config.service_name.clone(), service);

        Ok(())
    }

    /// Deregister a service
    pub async fn deregister_service(&self, service_name: &str) -> anyhow::Result<()> {
        info!("Deregistering service: {}", service_name);
        self.services.write().await.remove(service_name);
        Ok(())
    }

    /// Get service by name
    pub async fn get_service(&self, service_name: &str) -> Option<ServiceConfig> {
        self.services
            .read()
            .await
            .get(service_name)
            .map(|s| s.config.clone())
    }

    /// List all healthy services
    pub async fn list_healthy_services(&self) -> Vec<ServiceConfig> {
        self.services
            .read()
            .await
            .values()
            .filter(|s| matches!(s.status, ServiceStatus::Healthy))
            .map(|s| s.config.clone())
            .collect()
    }

    /// Update service health status
    pub async fn update_health_status(
        &self,
        service_name: &str,
        status: ServiceStatus,
    ) -> anyhow::Result<()> {
        if let Some(service) = self.services.write().await.get_mut(service_name) {
            service.status = status;
            service.last_health_check = chrono::Utc::now();
        }
        Ok(())
    }
}

/// gRPC Analysis Service
pub struct AnalysisService {
    registry: Arc<ServiceRegistry>,
    start_time: chrono::DateTime<chrono::Utc>,
}

impl AnalysisService {
    pub fn new(registry: Arc<ServiceRegistry>) -> Self {
        Self {
            registry,
            start_time: chrono::Utc::now(),
        }
    }

    /// Health check endpoint
    pub async fn health_check(&self) -> HealthCheckResponse {
        let uptime = chrono::Utc::now()
            .signed_duration_since(self.start_time)
            .num_seconds() as u64;

        HealthCheckResponse {
            status: ServiceStatus::Healthy,
            message: "Service is running".to_string(),
            timestamp: chrono::Utc::now(),
            uptime_secs: uptime,
        }
    }

    /// Process analysis request
    pub async fn process_analysis(
        &self,
        request: AnalysisRequest,
    ) -> Result<AnalysisResponse, GrpcError> {
        info!(
            "Processing analysis request: {} (type: {:?})",
            request.request_id, request.analysis_type
        );

        let start = std::time::Instant::now();

        // Validate request
        if request.target_sequence.is_empty() {
            return Err(GrpcError::InvalidRequest(
                "Target sequence cannot be empty".to_string(),
            ));
        }

        // Process based on analysis type
        let results = match request.analysis_type {
            AnalysisType::OffTargetPrediction => {
                self.predict_offtargets(&request.target_sequence).await?
            }
            AnalysisType::ImmuneResponse => {
                self.analyze_immune_response(&request.target_sequence).await?
            }
            AnalysisType::ComprehensiveAnalysis => {
                self.comprehensive_analysis(&request.target_sequence).await?
            }
        };

        let processing_time = start.elapsed().as_millis() as u64;

        Ok(AnalysisResponse {
            request_id: request.request_id,
            status: AnalysisStatus::Completed,
            results,
            processing_time_ms: processing_time,
            error_message: None,
        })
    }

    /// Predict off-target sites
    async fn predict_offtargets(&self, sequence: &str) -> Result<serde_json::Value, GrpcError> {
        debug!("Running off-target prediction for sequence: {}", sequence);

        // This would integrate with the actual off-target predictor
        Ok(serde_json::json!({
            "off_targets": [
                {"position": "chr1:1000", "score": 0.95, "mismatches": 1},
                {"position": "chr2:2000", "score": 0.87, "mismatches": 2}
            ],
            "total_sites": 2
        }))
    }

    /// Analyze immune response
    async fn analyze_immune_response(
        &self,
        sequence: &str,
    ) -> Result<serde_json::Value, GrpcError> {
        debug!("Running immune response analysis for sequence: {}", sequence);

        // This would integrate with the immune analyzer
        Ok(serde_json::json!({
            "immune_genes": [
                {"gene": "IRF3", "log2fc": 2.5, "pvalue": 0.001},
                {"gene": "STAT1", "log2fc": 3.2, "pvalue": 0.0001}
            ],
            "pathways": ["interferon_response", "inflammatory_response"]
        }))
    }

    /// Comprehensive analysis
    async fn comprehensive_analysis(
        &self,
        sequence: &str,
    ) -> Result<serde_json::Value, GrpcError> {
        debug!("Running comprehensive analysis for sequence: {}", sequence);

        let offtargets = self.predict_offtargets(sequence).await?;
        let immune = self.analyze_immune_response(sequence).await?;

        Ok(serde_json::json!({
            "off_target_analysis": offtargets,
            "immune_analysis": immune,
            "risk_score": 0.65,
            "recommendation": "Further validation required"
        }))
    }

    /// Call external service
    pub async fn call_service(
        &self,
        service_name: &str,
        request: serde_json::Value,
    ) -> Result<serde_json::Value, GrpcError> {
        let service = self
            .registry
            .get_service(service_name)
            .await
            .ok_or_else(|| GrpcError::ServiceUnavailable(service_name.to_string()))?;

        debug!(
            "Calling service {} at {}:{}",
            service_name, service.host, service.port
        );

        // In production, this would make actual gRPC call
        // For now, return mock response
        Ok(serde_json::json!({
            "status": "success",
            "service": service_name,
            "response": "mock_data"
        }))
    }
}

/// Start gRPC server
pub async fn start_grpc_server(
    addr: std::net::SocketAddr,
    service: Arc<AnalysisService>,
) -> anyhow::Result<()> {
    info!("🚀 Starting gRPC server on {}", addr);

    // In production, this would start actual gRPC server
    // For now, log the startup
    info!("gRPC server initialized (mock mode)");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_registry() {
        let registry = ServiceRegistry::new();

        let config = ServiceConfig {
            service_name: "test-service".to_string(),
            host: "localhost".to_string(),
            port: 50051,
            health_check_interval_secs: 30,
            timeout_secs: 10,
        };

        registry.register_service(config.clone()).await.unwrap();

        let service = registry.get_service("test-service").await;
        assert!(service.is_some());
    }

    #[tokio::test]
    async fn test_analysis_request() {
        let registry = Arc::new(ServiceRegistry::new());
        let service = AnalysisService::new(registry);

        let request = AnalysisRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            target_sequence: "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            analysis_type: AnalysisType::OffTargetPrediction,
            parameters: serde_json::json!({}),
        };

        let response = service.process_analysis(request).await.unwrap();
        assert!(matches!(response.status, AnalysisStatus::Completed));
    }
}
