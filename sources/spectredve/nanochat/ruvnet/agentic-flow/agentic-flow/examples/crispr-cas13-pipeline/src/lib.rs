//! # CRISPR-Cas13 Rights-Preserving Bioinformatics Platform
//!
//! High-performance Rust pipeline for CRISPR-Cas13 off-target analysis & immune response profiling
//! with integrated privacy protection, policy enforcement, and immutable audit trails.
//!
//! ## Features
//!
//! - **Off-Target Prediction**: ML-enhanced CFD scoring with position-weighted mismatch analysis
//! - **Immune Response Profiling**: DESeq2-style differential expression analysis
//! - **Privacy Protection**: Differential privacy with budget tracking and data anonymization
//! - **Policy Enforcement**: OPA-based access control and governance
//! - **Audit Logging**: Cryptographically signed, tamper-proof audit trails
//! - **Multi-Variant Support**: All 4 Cas13 variants (a, b, c, d)
//! - **Production-Ready**: Kubernetes deployments, auto-scaling, monitoring
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use crispr_cas13_pipeline::prelude::*;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create CRISPR target
//! let location = data_models::sequencing::GenomicCoordinate::new(0, 1000, 1023, true)?;
//! let target = data_models::targets::CrisprTarget::new(
//!     "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
//!     "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
//!     location,
//! )?;
//!
//! // Run off-target prediction
//! let config = offtarget_predictor::PredictionConfig::default();
//! let predictor = offtarget_predictor::scoring::DefaultPredictor::new(config)?;
//! let prediction = predictor.predict(&target).await?;
//!
//! println!("Found {} off-target sites", prediction.off_targets.len());
//! # Ok(())
//! # }
//! ```

// Rights-Preserving Platform Modules
pub mod api;
pub mod audit;
pub mod governance;
pub mod privacy;
pub mod services;

use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PlatformError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// Platform configuration
#[derive(Clone, Debug)]
pub struct PlatformConfig {
    pub api_host: String,
    pub api_port: u16,
    pub grpc_host: String,
    pub grpc_port: u16,
    pub privacy_config: privacy::differential::PrivacyConfig,
    pub audit_signing_key: String,
    pub enable_policy_enforcement: bool,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            api_host: "0.0.0.0".to_string(),
            api_port: 8080,
            grpc_host: "0.0.0.0".to_string(),
            grpc_port: 50051,
            privacy_config: privacy::differential::PrivacyConfig::default(),
            audit_signing_key: "default-signing-key".to_string(),
            enable_policy_enforcement: true,
        }
    }
}

/// Main platform instance
pub struct Platform {
    config: PlatformConfig,
    gateway_state: api::gateway::GatewayState,
    service_registry: Arc<services::grpc::ServiceRegistry>,
    analysis_service: Arc<services::grpc::AnalysisService>,
}

impl Platform {
    /// Initialize the platform
    pub async fn new(config: PlatformConfig) -> Result<Self, PlatformError> {
        tracing::info!("🧬 Initializing CRISPR-Cas13 Rights-Preserving Platform");

        // Initialize core components
        let privacy_engine = Arc::new(privacy::differential::PrivacyEngine::new(config.privacy_config.clone()));
        let policy_enforcer = Arc::new(governance::policy::PolicyEnforcer::new(
            governance::policy::PolicyConfig::default(),
        ));
        let audit_logger = Arc::new(audit::logger::AuditLogger::new(config.audit_signing_key.clone()));

        // Add default policy if enabled
        if config.enable_policy_enforcement {
            let default_policy = governance::policy::PolicyEnforcer::create_default_policy();
            policy_enforcer
                .add_policy(default_policy)
                .await
                .map_err(|e| PlatformError::InitError(e.to_string()))?;
        }

        // Create gateway state
        let gateway_state = api::gateway::GatewayState {
            auth_tokens: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            privacy_engine: privacy_engine.clone(),
            policy_enforcer: policy_enforcer.clone(),
            audit_logger: audit_logger.clone(),
        };

        // Initialize service registry
        let service_registry = Arc::new(services::grpc::ServiceRegistry::new());
        let analysis_service = Arc::new(services::grpc::AnalysisService::new(service_registry.clone()));

        // Register core services
        service_registry
            .register_service(services::grpc::ServiceConfig {
                service_name: "analysis-service".to_string(),
                host: config.grpc_host.clone(),
                port: config.grpc_port,
                health_check_interval_secs: 30,
                timeout_secs: 10,
            })
            .await
            .map_err(|e| PlatformError::InitError(e.to_string()))?;

        tracing::info!("✅ Platform initialized successfully");

        Ok(Self {
            config,
            gateway_state,
            service_registry,
            analysis_service,
        })
    }

    /// Get platform status
    pub async fn status(&self) -> PlatformStatus {
        let services = self.service_registry.list_healthy_services().await;

        PlatformStatus {
            api_endpoint: format!("{}:{}", self.config.api_host, self.config.api_port),
            grpc_endpoint: format!("{}:{}", self.config.grpc_host, self.config.grpc_port),
            registered_services: services.len(),
            privacy_enabled: true,
            policy_enforcement_enabled: self.config.enable_policy_enforcement,
            audit_enabled: true,
        }
    }

    /// Shutdown platform gracefully
    pub async fn shutdown(&self) -> Result<(), PlatformError> {
        tracing::info!("🛑 Shutting down platform...");

        // Verify audit chain before shutdown
        self.gateway_state
            .audit_logger
            .verify_chain()
            .await
            .map_err(|e| PlatformError::RuntimeError(e.to_string()))?;

        // Export audit chain
        let audit_export = self
            .gateway_state
            .audit_logger
            .export_chain()
            .await
            .map_err(|e| PlatformError::RuntimeError(e.to_string()))?;

        tracing::info!("📝 Audit chain exported ({} bytes)", audit_export.len());
        tracing::info!("✅ Platform shutdown complete");

        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
pub struct PlatformStatus {
    pub api_endpoint: String,
    pub grpc_endpoint: String,
    pub registered_services: usize,
    pub privacy_enabled: bool,
    pub policy_enforcement_enabled: bool,
    pub audit_enabled: bool,
}

// Re-export commonly used types
pub use api::gateway::{GatewayState, start_server as start_api_server};
pub use audit::logger::{AuditLogger, AuditEntry, AuditReport};
pub use governance::policy::{PolicyEnforcer, Policy, PolicyDecision};
pub use privacy::differential::{PrivacyEngine, PrivacyConfig, PrivateData};
pub use services::grpc::{
    AnalysisService, ServiceRegistry, AnalysisRequest, AnalysisResponse,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use data_models::expression::*;
    pub use data_models::metadata::*;
    pub use data_models::sequencing::*;
    pub use data_models::targets::*;
    pub use immune_analyzer::*;
    pub use offtarget_predictor::*;

    // Rights-preserving platform
    pub use crate::{
        Platform, PlatformConfig, PlatformStatus,
        AuditLogger, PolicyEnforcer, PrivacyEngine,
    };
}

// Re-export workspace crates
pub use data_models;
pub use immune_analyzer;
pub use offtarget_predictor;
