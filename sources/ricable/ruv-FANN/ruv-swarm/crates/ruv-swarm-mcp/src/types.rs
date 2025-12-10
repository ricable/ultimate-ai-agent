//! Types specific to MCP server implementation

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Agent type for MCP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    Researcher,
    Coder,
    Analyst,
    Tester,
    Reviewer,
    Documenter,
}

/// Agent capabilities for MCP
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub languages: Vec<String>,
    pub frameworks: Vec<String>,
    pub tools: Vec<String>,
    pub specializations: Vec<String>,
    pub max_concurrent_tasks: usize,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Swarm strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmStrategy {
    Research,
    Development,
    Analysis,
    Testing,
    Optimization,
    Maintenance,
}

/// Coordination mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationMode {
    Centralized,
    Distributed,
    Hierarchical,
    Mesh,
    Hybrid,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub strategy: SwarmStrategy,
    pub mode: CoordinationMode,
    pub max_agents: usize,
    pub parallel: bool,
    pub timeout: std::time::Duration,
}

/// Swarm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
    pub agents: Vec<AgentInfo>,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub total_agents: usize,
}

/// Agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: Uuid,
    pub agent_type: AgentType,
    pub name: Option<String>,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub current_tasks: Vec<Uuid>,
}

/// Swarm metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    pub total_tasks_processed: u64,
    pub average_task_duration_ms: u64,
    pub success_rate: f64,
    pub agent_utilization: f64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
}

/// Swarm status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub is_running: bool,
    pub uptime_secs: u64,
    pub version: String,
    pub config: serde_json::Value,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub impact: String,
    pub priority: TaskPriority,
    pub estimated_improvement: f64,
}

/// Workflow result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub success: bool,
    pub steps_completed: usize,
    pub total_steps: usize,
    pub outputs: serde_json::Value,
    pub errors: Vec<String>,
    pub duration_ms: u64,
}

/// Task creation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCreationResult {
    pub task_id: Uuid,
    pub assigned_agent: Option<Uuid>,
    pub estimated_completion_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// Orchestration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResult {
    pub task_id: Uuid,
    pub success: bool,
    pub agents_used: Vec<Uuid>,
    pub duration_ms: u64,
    pub outputs: serde_json::Value,
}
