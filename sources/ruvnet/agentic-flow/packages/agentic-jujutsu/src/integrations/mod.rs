// ! Agentic-Flow Integration Modules
//!
//! This module provides integration with agentic-flow components:
//! - Agent Booster (352x faster AST editing)
//! - AgentDB (150x faster vector search)
//! - Claude Flow MCP (213 coordination tools)
//! - QUIC transport (50-70% latency reduction)

pub mod agentic_flow;
pub mod ast_integration;
pub mod agentdb_learning;
pub mod swarm_coordinator;
pub mod quic_transport;

pub use agentic_flow::AgenticFlowIntegration;
pub use ast_integration::{ASTConflictResolver, TemplateResolution, RegexResolution};
pub use agentdb_learning::{SwarmLearningLoop, ConflictPredictor};
pub use swarm_coordinator::{SwarmCoordinator, TopologySelector, MeshSwarm, HierarchicalSwarm};
pub use quic_transport::{QUICOperationSync, QUICConnectionPool};
