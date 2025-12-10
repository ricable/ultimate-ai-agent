//! DAA Swarm - Dynamic Agent Architecture for swarm intelligence
//! 
//! This crate implements distributed swarm intelligence with autonomous agents
//! that can coordinate, learn, and adapt in real-time environments.

pub mod agent;
pub mod coordination;
pub mod economics;
pub mod error;
pub mod fault_tolerance;
pub mod lifecycle;
pub mod messaging;
pub mod swarm;
pub mod tasks;
pub mod swarm_intelligence;
pub mod evolutionary_mesh;
pub mod self_organizing;

pub use agent::{SwarmAgent, AgentType, AgentCapabilities, AgentBehavior};
pub use coordination::{CoordinationProtocol, ConsensusStrategy, DecisionEngine};
pub use economics::{EconomicEngine, ResourceAllocation, IncentiveSystem, MarketMechanisms};
pub use error::{SwarmError, Result};
pub use fault_tolerance::{FaultTolerance, RecoveryStrategy, HealthMonitor};
pub use lifecycle::{AgentLifecycle, SpawnStrategy, TerminationPolicy};
pub use messaging::{SwarmMessage, MessageBus, MessageRouter};
pub use swarm::{DAASwarm, SwarmConfig, SwarmTopology, SwarmStats};
pub use tasks::{TaskDistribution, TaskScheduler, WorkloadBalancer};
pub use swarm_intelligence::{SwarmIntelligence, EvolutionaryParams, OptimizationStrategy, FitnessMetrics, AgentGenome};
pub use evolutionary_mesh::{EvolutionaryMesh, MeshTopology, MeshNode, MeshConnection, AdaptationEngine};
pub use self_organizing::{SelfOrganizingSystem, OrganizationPattern, NodeCluster, EmergenceRule, Stigmergy};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Main DAA Swarm instance managing distributed agents with swarm intelligence
#[derive(Debug)]
pub struct DynamicAgentArchitecture {
    swarm: Arc<DAASwarm>,
    agents: Arc<RwLock<HashMap<Uuid, SwarmAgent>>>,
    coordination: Arc<CoordinationProtocol>,
    economics: Arc<EconomicEngine>,
    fault_tolerance: Arc<FaultTolerance>,
    lifecycle: Arc<AgentLifecycle>,
    message_bus: Arc<MessageBus>,
    config: ArchitectureConfig,
    // New swarm intelligence components
    swarm_intelligence: Arc<SwarmIntelligence>,
    evolutionary_mesh: Arc<EvolutionaryMesh>,
    self_organizing: Arc<SelfOrganizingSystem>,
}

impl DynamicAgentArchitecture {
    /// Create a new Dynamic Agent Architecture with swarm intelligence
    pub async fn new(config: ArchitectureConfig) -> Result<Self> {
        let swarm = Arc::new(DAASwarm::new(config.swarm_config.clone()).await?);
        let agents = Arc::new(RwLock::new(HashMap::new()));
        let coordination = Arc::new(CoordinationProtocol::new(config.coordination_config.clone()).await?);
        let economics = Arc::new(EconomicEngine::new(config.economic_config.clone()).await?);
        let fault_tolerance = Arc::new(FaultTolerance::new(config.fault_tolerance_config.clone()).await?);
        let lifecycle = Arc::new(AgentLifecycle::new(config.lifecycle_config.clone()).await?);
        let message_bus = Arc::new(MessageBus::new(config.messaging_config.clone()).await?);
        
        // Initialize swarm intelligence components
        let swarm_intelligence = Arc::new(SwarmIntelligence::new(
            config.swarm_intelligence_config.optimization_strategy.clone()
        ));
        let evolutionary_mesh = Arc::new(EvolutionaryMesh::new(
            config.swarm_intelligence_config.mesh_topology.clone(),
            config.swarm_intelligence_config.optimization_strategy.clone()
        ));
        let self_organizing = Arc::new(SelfOrganizingSystem::new(
            config.swarm_intelligence_config.organization_pattern.clone()
        ));

        Ok(Self {
            swarm,
            agents,
            coordination,
            economics,
            fault_tolerance,
            lifecycle,
            message_bus,
            config,
            swarm_intelligence,
            evolutionary_mesh,
            self_organizing,
        })
    }

    /// Start the DAA system with swarm intelligence
    pub async fn start(&self) -> Result<()> {
        // Start core subsystems
        self.swarm.start().await?;
        self.coordination.start().await?;
        self.economics.start().await?;
        self.fault_tolerance.start().await?;
        self.lifecycle.start().await?;
        self.message_bus.start().await?;

        // Initialize swarm intelligence systems
        let initial_population_size = self.config.swarm_intelligence_config.initial_population_size;
        self.swarm_intelligence.initialize_population(initial_population_size).await;
        self.evolutionary_mesh.initialize(initial_population_size).await;
        self.self_organizing.initialize_rules().await;

        // Spawn initial agents
        self.spawn_initial_agents().await?;

        // Start monitoring and coordination loops
        self.start_coordination_loop().await?;
        self.start_health_monitoring().await?;
        self.start_economic_engine().await?;
        
        // Start swarm intelligence loops
        self.start_swarm_evolution_loop().await?;
        self.start_self_organization_loop().await?;

        tracing::info!("Dynamic Agent Architecture with Swarm Intelligence started successfully");
        Ok(())
    }

    /// Trigger swarm evolution manually
    pub async fn evolve_swarm(&self) -> Result<()> {
        let agents = self.agents.read().await;
        let swarm_context = swarm_intelligence::SwarmContext {
            agent_count: agents.len(),
            network_load: 0.5, // TODO: Get actual network metrics
            error_rate: 0.1,   // TODO: Get actual error metrics
            resource_usage: std::collections::HashMap::new(),
            environment_state: std::collections::HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        self.swarm_intelligence.evolve(&swarm_context).await;
        Ok(())
    }

    /// Trigger self-organization
    pub async fn organize_swarm(&self) -> Result<()> {
        let agents = self.agents.read().await;
        let node_infos: Vec<self_organizing::NodeInfo> = agents.iter()
            .map(|(id, agent)| self_organizing::NodeInfo {
                id: id.to_string(),
                position: vec![0.0, 0.0], // TODO: Get actual positions
                capabilities: std::collections::HashMap::new(), // TODO: Get capabilities
                connections: Vec::new(), // TODO: Get connections
                performance: 0.5, // TODO: Get actual performance
            })
            .collect();
        
        let context = self_organizing::OrganizationContext {
            metrics: std::collections::HashMap::new(),
            detected_patterns: std::collections::HashMap::new(),
            event_counts: std::collections::HashMap::new(),
            last_organization_time: 0,
            significant_change_detected: false,
            desired_clusters: agents.len() / 5,
            use_geographic_clustering: false,
        };
        
        self.self_organizing.organize(node_infos, &context).await;
        Ok(())
    }

    /// Apply self-healing to failed agents
    pub async fn self_heal(&self, failed_agent_ids: Vec<String>) -> Result<()> {
        self.swarm_intelligence.self_heal(failed_agent_ids).await
            .map_err(|e| SwarmError::AdaptationError(e))?;
        Ok(())
    }

    /// Get swarm intelligence metrics
    pub async fn get_swarm_intelligence_metrics(&self) -> SwarmIntelligenceMetrics {
        let mesh_stats = self.evolutionary_mesh.get_stats().await;
        let organization_stats = self.self_organizing.get_metrics().await;
        
        SwarmIntelligenceMetrics {
            mesh_metrics: mesh_stats,
            organization_metrics: organization_stats,
            evolution_generation: 0, // TODO: Get from swarm intelligence
            adaptation_rate: 0.5,    // TODO: Calculate adaptation rate
        }
    }

    /// Get current organization pattern
    pub async fn get_organization_pattern(&self) -> OrganizationPattern {
        self.self_organizing.get_pattern().await
    }

    /// Get node clusters
    pub async fn get_clusters(&self) -> Vec<NodeCluster> {
        self.self_organizing.get_clusters().await
    }

    /// Stop the DAA system
    pub async fn stop(&self) -> Result<()> {
        // Stop all agents
        {
            let mut agents = self.agents.write().await;
            for agent in agents.values_mut() {
                agent.terminate().await?;
            }
            agents.clear();
        }

        // Stop subsystems
        self.message_bus.stop().await?;
        self.lifecycle.stop().await?;
        self.fault_tolerance.stop().await?;
        self.economics.stop().await?;
        self.coordination.stop().await?;
        self.swarm.stop().await?;

        tracing::info!("Dynamic Agent Architecture stopped");
        Ok(())
    }

    /// Spawn a new agent with specified type and capabilities
    pub async fn spawn_agent(
        &self,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
        initial_resources: f64,
    ) -> Result<Uuid> {
        let agent = SwarmAgent::new(
            agent_type,
            capabilities,
            initial_resources,
            Arc::clone(&self.message_bus),
        ).await?;

        let agent_id = agent.id();

        // Register with lifecycle manager
        self.lifecycle.register_agent(agent_id, agent.clone()).await?;

        // Add to swarm
        self.swarm.add_agent(agent.clone()).await?;

        // Allocate economic resources
        self.economics.allocate_resources(agent_id, initial_resources).await?;

        // Start agent
        agent.start().await?;

        // Store agent
        {
            let mut agents = self.agents.write().await;
            agents.insert(agent_id, agent);
        }

        tracing::info!("Spawned agent {} of type {:?}", agent_id, agent_type);
        Ok(agent_id)
    }

    /// Terminate an agent
    pub async fn terminate_agent(&self, agent_id: Uuid) -> Result<bool> {
        let mut agents = self.agents.write().await;
        if let Some(mut agent) = agents.remove(&agent_id) {
            // Terminate agent
            agent.terminate().await?;

            // Remove from swarm
            self.swarm.remove_agent(agent_id).await?;

            // Deallocate resources
            self.economics.deallocate_resources(agent_id).await?;

            // Unregister from lifecycle
            self.lifecycle.unregister_agent(agent_id).await?;

            tracing::info!("Terminated agent {}", agent_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Distribute a task across the swarm
    pub async fn distribute_task(&self, task: tasks::SwarmTask) -> Result<Uuid> {
        // Use coordination protocol to determine optimal task distribution
        let distribution = self.coordination.plan_task_distribution(&task).await?;
        
        // Execute distribution
        let task_id = self.swarm.execute_task(task, distribution).await?;
        
        // Update economic incentives
        self.economics.record_task_distribution(task_id).await?;
        
        Ok(task_id)
    }

    /// Get swarm statistics including swarm intelligence metrics
    pub async fn get_stats(&self) -> ArchitectureStats {
        let agents = self.agents.read().await;
        let swarm_stats = self.swarm.get_stats().await;
        let coordination_stats = self.coordination.get_stats().await;
        let economic_stats = self.economics.get_stats().await;
        let fault_stats = self.fault_tolerance.get_stats().await;
        let swarm_intelligence_metrics = self.get_swarm_intelligence_metrics().await;

        ArchitectureStats {
            total_agents: agents.len(),
            active_agents: agents.values().filter(|a| a.is_active()).count(),
            swarm_efficiency: swarm_stats.efficiency,
            coordination_overhead: coordination_stats.overhead,
            economic_health: economic_stats.health_score,
            fault_rate: fault_stats.fault_rate,
            uptime: swarm_stats.uptime,
            total_tasks_completed: swarm_stats.completed_tasks,
            swarm_intelligence: swarm_intelligence_metrics,
        }
    }

    /// Get agent by ID
    pub async fn get_agent(&self, agent_id: Uuid) -> Option<SwarmAgent> {
        let agents = self.agents.read().await;
        agents.get(&agent_id).cloned()
    }

    /// List all agents
    pub async fn list_agents(&self) -> Vec<Uuid> {
        let agents = self.agents.read().await;
        agents.keys().cloned().collect()
    }

    /// Get agents by type
    pub async fn get_agents_by_type(&self, agent_type: AgentType) -> Vec<Uuid> {
        let agents = self.agents.read().await;
        agents
            .values()
            .filter(|agent| agent.agent_type() == agent_type)
            .map(|agent| agent.id())
            .collect()
    }

    /// Spawn initial agents based on configuration
    async fn spawn_initial_agents(&self) -> Result<()> {
        for &(agent_type, count) in &self.config.initial_agents {
            for _ in 0..count {
                let capabilities = match agent_type {
                    AgentType::Coordinator => AgentCapabilities::coordinator(),
                    AgentType::Worker => AgentCapabilities::worker(),
                    AgentType::Monitor => AgentCapabilities::monitor(),
                    AgentType::Researcher => AgentCapabilities::researcher(),
                    AgentType::Optimizer => AgentCapabilities::optimizer(),
                };

                self.spawn_agent(agent_type, capabilities, 100.0).await?;
            }
        }

        Ok(())
    }

    /// Start the coordination loop
    async fn start_coordination_loop(&self) -> Result<()> {
        let coordination = Arc::clone(&self.coordination);
        let agents = Arc::clone(&self.agents);
        let interval = self.config.coordination_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = coordination.coordinate_agents(&agents).await {
                    tracing::error!("Coordination error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) -> Result<()> {
        let fault_tolerance = Arc::clone(&self.fault_tolerance);
        let agents = Arc::clone(&self.agents);
        let lifecycle = Arc::clone(&self.lifecycle);
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = fault_tolerance.monitor_health(&agents, &lifecycle).await {
                    tracing::error!("Health monitoring error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start economic engine
    async fn start_economic_engine(&self) -> Result<()> {
        let economics = Arc::clone(&self.economics);
        let agents = Arc::clone(&self.agents);
        let interval = self.config.economic_update_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = economics.update_markets(&agents).await {
                    tracing::error!("Economic engine error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start swarm evolution loop
    async fn start_swarm_evolution_loop(&self) -> Result<()> {
        let swarm_intelligence = Arc::clone(&self.swarm_intelligence);
        let agents = Arc::clone(&self.agents);
        let interval = self.config.swarm_intelligence_config.evolution_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                
                let agents_guard = agents.read().await;
                let swarm_context = swarm_intelligence::SwarmContext {
                    agent_count: agents_guard.len(),
                    network_load: 0.5,
                    error_rate: 0.1,
                    resource_usage: std::collections::HashMap::new(),
                    environment_state: std::collections::HashMap::new(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                };
                drop(agents_guard);
                
                swarm_intelligence.evolve(&swarm_context).await;
                tracing::debug!("Swarm evolution cycle completed");
            }
        });

        Ok(())
    }

    /// Start self-organization loop
    async fn start_self_organization_loop(&self) -> Result<()> {
        let self_organizing = Arc::clone(&self.self_organizing);
        let agents = Arc::clone(&self.agents);
        let interval = self.config.swarm_intelligence_config.organization_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                
                let agents_guard = agents.read().await;
                let node_infos: Vec<self_organizing::NodeInfo> = agents_guard.iter()
                    .map(|(id, _agent)| self_organizing::NodeInfo {
                        id: id.to_string(),
                        position: vec![0.0, 0.0],
                        capabilities: std::collections::HashMap::new(),
                        connections: Vec::new(),
                        performance: 0.5,
                    })
                    .collect();
                
                let context = self_organizing::OrganizationContext {
                    metrics: std::collections::HashMap::new(),
                    detected_patterns: std::collections::HashMap::new(),
                    event_counts: std::collections::HashMap::new(),
                    last_organization_time: 0,
                    significant_change_detected: false,
                    desired_clusters: agents_guard.len() / 5,
                    use_geographic_clustering: false,
                };
                drop(agents_guard);
                
                self_organizing.organize(node_infos, &context).await;
                tracing::debug!("Self-organization cycle completed");
            }
        });

        Ok(())
    }
}

/// Configuration for the Dynamic Agent Architecture
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    pub swarm_config: SwarmConfig,
    pub coordination_config: coordination::CoordinationConfig,
    pub economic_config: economics::EconomicConfig,
    pub fault_tolerance_config: fault_tolerance::FaultToleranceConfig,
    pub lifecycle_config: lifecycle::LifecycleConfig,
    pub messaging_config: messaging::MessagingConfig,
    pub swarm_intelligence_config: SwarmIntelligenceConfig,
    pub initial_agents: Vec<(AgentType, usize)>,
    pub coordination_interval: Duration,
    pub health_check_interval: Duration,
    pub economic_update_interval: Duration,
}

/// Configuration for swarm intelligence features
#[derive(Debug, Clone)]
pub struct SwarmIntelligenceConfig {
    pub optimization_strategy: OptimizationStrategy,
    pub mesh_topology: MeshTopology,
    pub organization_pattern: OrganizationPattern,
    pub initial_population_size: usize,
    pub evolution_interval: Duration,
    pub organization_interval: Duration,
    pub evolutionary_params: EvolutionaryParams,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            swarm_config: SwarmConfig::default(),
            coordination_config: coordination::CoordinationConfig::default(),
            economic_config: economics::EconomicConfig::default(),
            fault_tolerance_config: fault_tolerance::FaultToleranceConfig::default(),
            lifecycle_config: lifecycle::LifecycleConfig::default(),
            messaging_config: messaging::MessagingConfig::default(),
            swarm_intelligence_config: SwarmIntelligenceConfig::default(),
            initial_agents: vec![
                (AgentType::Coordinator, 1),
                (AgentType::Worker, 3),
                (AgentType::Monitor, 1),
            ],
            coordination_interval: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(30),
            economic_update_interval: Duration::from_secs(60),
        }
    }
}

impl Default for SwarmIntelligenceConfig {
    fn default() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::HybridAdaptive,
            mesh_topology: MeshTopology::Adaptive,
            organization_pattern: OrganizationPattern::Dynamic,
            initial_population_size: 50,
            evolution_interval: Duration::from_secs(30),
            organization_interval: Duration::from_secs(60),
            evolutionary_params: EvolutionaryParams::default(),
        }
    }
}

/// Statistics about the DAA system with swarm intelligence
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArchitectureStats {
    pub total_agents: usize,
    pub active_agents: usize,
    pub swarm_efficiency: f64,
    pub coordination_overhead: f64,
    pub economic_health: f64,
    pub fault_rate: f64,
    pub uptime: Duration,
    pub total_tasks_completed: u64,
    pub swarm_intelligence: SwarmIntelligenceMetrics,
}

/// Metrics for swarm intelligence features
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwarmIntelligenceMetrics {
    pub mesh_metrics: evolutionary_mesh::MeshMetrics,
    pub organization_metrics: self_organizing::SelfOrganizationMetrics,
    pub evolution_generation: u64,
    pub adaptation_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_daa_creation() {
        let config = ArchitectureConfig::default();
        let daa = DynamicAgentArchitecture::new(config).await;
        assert!(daa.is_ok());
    }

    #[tokio::test]
    async fn test_agent_spawning() {
        let config = ArchitectureConfig::default();
        let daa = DynamicAgentArchitecture::new(config).await.unwrap();
        
        let agent_id = daa.spawn_agent(
            AgentType::Worker,
            AgentCapabilities::worker(),
            100.0,
        ).await.unwrap();
        
        assert!(daa.get_agent(agent_id).await.is_some());
        assert!(daa.terminate_agent(agent_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_architecture_lifecycle() {
        let config = ArchitectureConfig::default();
        let daa = DynamicAgentArchitecture::new(config).await.unwrap();
        
        assert!(daa.start().await.is_ok());
        assert!(daa.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_swarm_intelligence_features() {
        let config = ArchitectureConfig::default();
        let daa = DynamicAgentArchitecture::new(config).await.unwrap();
        
        // Test swarm evolution
        assert!(daa.evolve_swarm().await.is_ok());
        
        // Test self-organization
        assert!(daa.organize_swarm().await.is_ok());
        
        // Test self-healing
        assert!(daa.self_heal(vec!["failed_agent_1".to_string()]).await.is_ok());
        
        // Test metrics retrieval
        let metrics = daa.get_swarm_intelligence_metrics().await;
        assert!(metrics.adaptation_rate >= 0.0);
        
        // Test organization pattern
        let pattern = daa.get_organization_pattern().await;
        assert!(matches!(pattern, OrganizationPattern::Dynamic));
        
        // Test clusters
        let clusters = daa.get_clusters().await;
        assert!(clusters.is_empty()); // Initially empty
    }
}