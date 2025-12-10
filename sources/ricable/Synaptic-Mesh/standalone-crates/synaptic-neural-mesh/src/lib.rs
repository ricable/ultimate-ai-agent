//! Synaptic Neural Mesh - Coordination layer for distributed AI agents
//!
//! This crate provides the mesh coordination infrastructure for managing
//! distributed neural agents in a self-organizing network.

use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use synaptic_qudag_core::{QuDAGNode, QuDAGNetwork};

/// Neural mesh errors
#[derive(Error, Debug)]
pub enum MeshError {
    #[error("Agent error: {0}")]
    AgentError(String),
    
    #[error("Coordination error: {0}")]
    CoordinationError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Synchronization error: {0}")]
    SynchronizationError(String),
}

pub type Result<T> = std::result::Result<T, MeshError>;

/// Agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub compute_power: f64,
    pub memory_available: usize,
    pub specializations: Vec<String>,
    pub latency_ms: f64,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            compute_power: 1.0,
            memory_available: 1024 * 1024 * 1024, // 1GB
            specializations: vec!["general".to_string()],
            latency_ms: 10.0,
        }
    }
}

/// Neural agent in the mesh
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: Uuid,
    pub name: String,
    pub capabilities: Capabilities,
    pub status: AgentStatus,
    pub created_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
}

/// Agent status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Idle,
    Busy,
    Offline,
}

impl Agent {
    /// Create a new agent
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            capabilities: Capabilities::default(),
            status: AgentStatus::Idle,
            created_at: now,
            last_heartbeat: now,
        }
    }
    
    /// Update heartbeat
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Utc::now();
    }
    
    /// Check if agent is healthy
    pub fn is_healthy(&self) -> bool {
        let elapsed = Utc::now().signed_duration_since(self.last_heartbeat);
        elapsed.num_seconds() < 30 // 30 second timeout
    }
}

/// Coordination task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub name: String,
    pub requirements: TaskRequirements,
    pub status: TaskStatus,
    pub assigned_agent: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Task requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    pub min_compute_power: f64,
    pub min_memory: usize,
    pub required_specializations: Vec<String>,
    pub max_latency_ms: f64,
}

/// Task status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Assigned,
    InProgress,
    Completed,
    Failed,
}

/// Neural mesh coordinator
pub struct NeuralMesh {
    agents: Arc<DashMap<Uuid, Agent>>,
    tasks: Arc<DashMap<Uuid, Task>>,
    dag_network: Arc<QuDAGNetwork>,
    topology: Arc<RwLock<MeshTopology>>,
}

/// Mesh topology
#[derive(Debug, Clone)]
pub struct MeshTopology {
    connections: HashMap<Uuid, Vec<Uuid>>,
}

impl NeuralMesh {
    /// Create a new neural mesh
    pub fn new() -> Self {
        Self {
            agents: Arc::new(DashMap::new()),
            tasks: Arc::new(DashMap::new()),
            dag_network: Arc::new(QuDAGNetwork::new()),
            topology: Arc::new(RwLock::new(MeshTopology {
                connections: HashMap::new(),
            })),
        }
    }
    
    /// Add an agent to the mesh
    pub async fn add_agent(&self, agent: Agent) -> Result<Uuid> {
        let agent_id = agent.id;
        
        // Create DAG node for agent registration
        let node_data = serde_json::to_vec(&agent)
            .map_err(|e| MeshError::AgentError(format!("Serialization failed: {}", e)))?;
        let dag_node = QuDAGNode::new(&node_data);
        
        self.dag_network.add_node(dag_node).await
            .map_err(|e| MeshError::NetworkError(e.to_string()))?;
        
        self.agents.insert(agent_id, agent);
        
        // Update topology
        self.update_topology(agent_id).await?;
        
        Ok(agent_id)
    }
    
    /// Remove an agent from the mesh
    pub async fn remove_agent(&self, agent_id: &Uuid) -> Result<()> {
        self.agents.remove(agent_id);
        
        // Clean up topology
        let mut topology = self.topology.write();
        topology.connections.remove(agent_id);
        for connections in topology.connections.values_mut() {
            connections.retain(|id| id != agent_id);
        }
        
        Ok(())
    }
    
    /// Get an agent by ID
    pub fn get_agent(&self, agent_id: &Uuid) -> Option<Agent> {
        self.agents.get(agent_id).map(|a| a.clone())
    }
    
    /// List all agents
    pub fn list_agents(&self) -> Vec<Agent> {
        self.agents.iter().map(|a| a.clone()).collect()
    }
    
    /// Submit a task to the mesh
    pub async fn submit_task(&self, name: impl Into<String>, requirements: TaskRequirements) -> Result<Uuid> {
        let task = Task {
            id: Uuid::new_v4(),
            name: name.into(),
            requirements,
            status: TaskStatus::Pending,
            assigned_agent: None,
            created_at: Utc::now(),
            completed_at: None,
        };
        
        let task_id = task.id;
        self.tasks.insert(task_id, task);
        
        // Try to assign the task
        self.assign_task(task_id).await?;
        
        Ok(task_id)
    }
    
    /// Assign a task to an appropriate agent
    async fn assign_task(&self, task_id: Uuid) -> Result<()> {
        let task = self.tasks.get(&task_id)
            .ok_or_else(|| MeshError::CoordinationError("Task not found".to_string()))?;
        
        // Find suitable agent
        let suitable_agent = self.find_suitable_agent(&task.requirements);
        
        if let Some(agent_id) = suitable_agent {
            drop(task); // Release the read lock
            
            // Update task assignment
            if let Some(mut task) = self.tasks.get_mut(&task_id) {
                task.assigned_agent = Some(agent_id);
                task.status = TaskStatus::Assigned;
            }
            
            // Update agent status
            if let Some(mut agent) = self.agents.get_mut(&agent_id) {
                agent.status = AgentStatus::Busy;
            }
        }
        
        Ok(())
    }
    
    /// Find a suitable agent for task requirements
    fn find_suitable_agent(&self, requirements: &TaskRequirements) -> Option<Uuid> {
        self.agents.iter()
            .filter(|entry| {
                let agent = entry.value();
                agent.status == AgentStatus::Idle &&
                agent.is_healthy() &&
                agent.capabilities.compute_power >= requirements.min_compute_power &&
                agent.capabilities.memory_available >= requirements.min_memory &&
                agent.capabilities.latency_ms <= requirements.max_latency_ms &&
                requirements.required_specializations.iter().all(|spec| {
                    agent.capabilities.specializations.contains(spec)
                })
            })
            .min_by_key(|entry| {
                // Prefer agents with lower latency
                (entry.value().capabilities.latency_ms * 1000.0) as i64
            })
            .map(|entry| entry.key().clone())
    }
    
    /// Update mesh topology
    async fn update_topology(&self, new_agent_id: Uuid) -> Result<()> {
        let mut topology = self.topology.write();
        
        // Simple mesh topology: connect to 3 nearest agents
        let agents: Vec<Uuid> = self.agents.iter()
            .map(|entry| entry.key().clone())
            .filter(|id| id != &new_agent_id)
            .take(3)
            .collect();
        
        topology.connections.insert(new_agent_id, agents.clone());
        
        // Add bidirectional connections
        for agent_id in agents {
            topology.connections.entry(agent_id)
                .or_insert_with(Vec::new)
                .push(new_agent_id);
        }
        
        Ok(())
    }
    
    /// Get mesh statistics
    pub fn get_stats(&self) -> MeshStats {
        let total_agents = self.agents.len();
        let active_agents = self.agents.iter()
            .filter(|a| a.status == AgentStatus::Active || a.status == AgentStatus::Busy)
            .count();
        let total_tasks = self.tasks.len();
        let completed_tasks = self.tasks.iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        
        MeshStats {
            total_agents,
            active_agents,
            total_tasks,
            completed_tasks,
            dag_nodes: self.dag_network.node_count(),
        }
    }
}

impl Default for NeuralMesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Mesh statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStats {
    pub total_agents: usize,
    pub active_agents: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub dag_nodes: usize,
}

/// Coordination protocol trait
#[async_trait]
pub trait CoordinationProtocol {
    /// Negotiate task assignment
    async fn negotiate_task(&self, task: &Task, agents: &[Agent]) -> Result<Option<Uuid>>;
    
    /// Synchronize state across mesh
    async fn synchronize(&self) -> Result<()>;
    
    /// Handle agent failure
    async fn handle_failure(&self, agent_id: Uuid) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mesh_creation() {
        let mesh = NeuralMesh::new();
        let stats = mesh.get_stats();
        assert_eq!(stats.total_agents, 0);
        assert_eq!(stats.total_tasks, 0);
    }
    
    #[tokio::test]
    async fn test_agent_management() {
        let mesh = NeuralMesh::new();
        
        let agent = Agent::new("test-agent");
        let agent_id = mesh.add_agent(agent.clone()).await.unwrap();
        
        assert_eq!(mesh.get_stats().total_agents, 1);
        
        let retrieved = mesh.get_agent(&agent_id).unwrap();
        assert_eq!(retrieved.name, "test-agent");
        
        mesh.remove_agent(&agent_id).await.unwrap();
        assert_eq!(mesh.get_stats().total_agents, 0);
    }
    
    #[tokio::test]
    async fn test_task_submission() {
        let mesh = NeuralMesh::new();
        
        // Add an agent
        let mut agent = Agent::new("worker");
        agent.capabilities.compute_power = 2.0;
        mesh.add_agent(agent).await.unwrap();
        
        // Submit a task
        let requirements = TaskRequirements {
            min_compute_power: 1.0,
            min_memory: 1024,
            required_specializations: vec!["general".to_string()],
            max_latency_ms: 100.0,
        };
        
        let task_id = mesh.submit_task("test-task", requirements).await.unwrap();
        
        assert_eq!(mesh.get_stats().total_tasks, 1);
        
        // Check task was assigned
        let task = mesh.tasks.get(&task_id).unwrap();
        assert!(task.assigned_agent.is_some());
        assert_eq!(task.status, TaskStatus::Assigned);
    }
}