//! Main integration module for agentic-flow components
//!
//! Coordinates all agentic-flow features:
//! - Agent Booster for conflict resolution
//! - AgentDB for pattern learning
//! - Swarm coordination
//! - QUIC transport

use crate::{
    error::{JJError, Result},
    wrapper::JJWrapper,
    types::{JJOperation, JJConflict},
    operations::OperationType,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{
    ast_integration::ASTConflictResolver,
    agentdb_learning::SwarmLearningLoop,
    swarm_coordinator::SwarmCoordinator,
    quic_transport::QUICOperationSync,
};

/// Main integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable Agent Booster for conflict resolution
    pub enable_agent_booster: bool,

    /// Enable AgentDB for pattern learning
    pub enable_agentdb: bool,

    /// Enable QUIC transport for operation sync
    pub enable_quic: bool,

    /// Swarm topology (mesh, hierarchical, adaptive)
    pub swarm_topology: String,

    /// Maximum number of agents in swarm
    pub max_agents: usize,

    /// AgentDB API endpoint (if remote)
    pub agentdb_url: Option<String>,

    /// Agent Booster WASM path
    pub agent_booster_wasm: String,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_agent_booster: true,
            enable_agentdb: true,
            enable_quic: true,
            swarm_topology: "adaptive".to_string(),
            max_agents: 20,
            agentdb_url: None,
            agent_booster_wasm: "/usr/local/lib/agent_booster_wasm.wasm".to_string(),
        }
    }
}

/// Main integration struct coordinating all agentic-flow features
pub struct AgenticFlowIntegration {
    jj_wrapper: Arc<JJWrapper>,
    ast_resolver: Option<Arc<ASTConflictResolver>>,
    learning_loop: Option<Arc<SwarmLearningLoop>>,
    swarm_coordinator: Arc<SwarmCoordinator>,
    quic_sync: Option<Arc<QUICOperationSync>>,
    config: IntegrationConfig,
}

impl AgenticFlowIntegration {
    /// Create new integration with configuration
    pub async fn new(config: IntegrationConfig) -> Result<Self> {
        // Initialize JJWrapper
        let jj_wrapper = Arc::new(JJWrapper::new()?);

        // Initialize AST resolver if enabled
        let ast_resolver = if config.enable_agent_booster {
            Some(Arc::new(
                ASTConflictResolver::new(&config.agent_booster_wasm).await?
            ))
        } else {
            None
        };

        // Initialize learning loop if AgentDB enabled
        let learning_loop = if config.enable_agentdb {
            Some(Arc::new(
                SwarmLearningLoop::new(
                    config.agentdb_url.clone()
                ).await?
            ))
        } else {
            None
        };

        // Initialize swarm coordinator
        let swarm_coordinator = Arc::new(
            SwarmCoordinator::new(&config.swarm_topology, config.max_agents)?
        );

        // Initialize QUIC sync if enabled
        let quic_sync = if config.enable_quic {
            Some(Arc::new(
                QUICOperationSync::new().await?
            ))
        } else {
            None
        };

        Ok(Self {
            jj_wrapper,
            ast_resolver,
            learning_loop,
            swarm_coordinator,
            quic_sync,
            config,
        })
    }

    /// Execute agent operation with full pipeline
    pub async fn execute_agent_operation(
        &self,
        agent_id: &str,
        operation: AgentOperation,
    ) -> Result<OperationResult> {
        // Execute operation via JJWrapper
        let jj_operation = match operation.op_type {
            OperationType::Describe => {
                self.jj_wrapper.describe(&operation.message).await?
            }
            OperationType::New => {
                let result = self.jj_wrapper.new_commit(Some(&operation.message)).await?;
                // Convert JJResult to JJOperation
                return Ok(OperationResult {
                    success: result.success(),
                    output: result.stdout,
                    latency_ms: result.execution_time_ms,
                });
            }
            _ => {
                return Err(JJError::OperationNotSupported(
                    format!("Operation type {:?} not yet supported", operation.op_type)
                ));
            }
        };

        // Sync operation log if QUIC enabled
        if let Some(quic_sync) = &self.quic_sync {
            quic_sync.sync_operation(&jj_operation, &[agent_id.to_string()]).await?;
        }

        // Learn from operation if AgentDB enabled
        if let Some(learning_loop) = &self.learning_loop {
            learning_loop.learn_from_operation(&jj_operation).await?;
        }

        Ok(OperationResult {
            success: jj_operation.success,
            output: jj_operation.command.clone(),
            latency_ms: jj_operation.duration_ms,
        })
    }

    /// Resolve conflicts using AST + Agent Booster
    pub async fn resolve_conflicts(
        &self,
        conflicts: Vec<JJConflict>,
    ) -> Result<Vec<Resolution>> {
        if let Some(ast_resolver) = &self.ast_resolver {
            let mut resolutions = Vec::new();

            for conflict in conflicts {
                match ast_resolver.resolve_conflict(&conflict).await {
                    Ok(resolution) => {
                        // Learn from successful resolution
                        if let Some(learning_loop) = &self.learning_loop {
                            learning_loop.learn_from_resolution(
                                &conflict,
                                &resolution
                            ).await?;
                        }

                        resolutions.push(resolution);
                    }
                    Err(e) => {
                        log::warn!("Failed to resolve conflict in {}: {}", conflict.path, e);
                    }
                }
            }

            Ok(resolutions)
        } else {
            Err(JJError::FeatureNotEnabled("Agent Booster".to_string()))
        }
    }

    /// Sync operation log across agents using QUIC
    pub async fn sync_operation_log(
        &self,
        agent_ids: Vec<String>,
    ) -> Result<SyncStatus> {
        if let Some(quic_sync) = &self.quic_sync {
            // Get recent operations
            let operations = self.jj_wrapper.get_operations(50)?;

            let start = std::time::Instant::now();

            // Sync to all agents in parallel
            for operation in operations {
                quic_sync.sync_operation(&operation, &agent_ids).await?;
            }

            let elapsed = start.elapsed();

            Ok(SyncStatus {
                operations_synced: operations.len(),
                agents_synced: agent_ids.len(),
                latency_ms: elapsed.as_millis() as u64,
            })
        } else {
            Err(JJError::FeatureNotEnabled("QUIC transport".to_string()))
        }
    }
}

/// Agent operation to execute
#[derive(Debug, Clone)]
pub struct AgentOperation {
    pub op_type: OperationType,
    pub message: String,
}

/// Result of agent operation
#[derive(Debug, Clone)]
pub struct OperationResult {
    pub success: bool,
    pub output: String,
    pub latency_ms: u64,
}

/// Conflict resolution result
#[derive(Debug, Clone)]
pub struct Resolution {
    pub method: String,
    pub content: String,
    pub confidence: f64,
    pub latency_ms: u64,
}

/// Operation log sync status
#[derive(Debug, Clone)]
pub struct SyncStatus {
    pub operations_synced: usize,
    pub agents_synced: usize,
    pub latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integration_creation() {
        let config = IntegrationConfig::default();
        let integration = AgenticFlowIntegration::new(config).await;

        // Should succeed even if some features unavailable
        assert!(integration.is_ok());
    }
}
