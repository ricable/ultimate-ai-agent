//! Swarm orchestration for distributed neural network training

use crate::data::TelecomDataset;
use crate::models::NeuralModel;
use crate::training::{TrainingResults, ModelTrainingResult};
use crate::error::{TrainingError, TrainingResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Swarm orchestrator for coordinating multiple training agents
pub struct SwarmOrchestrator {
    config: SwarmConfig,
    agents: Vec<SwarmAgent>,
    coordinator: Option<SwarmCoordinator>,
    results: Arc<Mutex<Vec<ModelTrainingResult>>>,
}

/// Configuration for swarm orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Maximum number of agents in the swarm
    pub max_agents: usize,
    /// Coordination strategy
    pub strategy: CoordinationStrategy,
    /// Communication settings
    pub communication: CommunicationConfig,
    /// Load balancing settings
    pub load_balancing: LoadBalancingConfig,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
}

/// Coordination strategies for the swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Independent agents working in parallel
    Independent,
    /// Hierarchical coordination with a master agent
    Hierarchical,
    /// Collaborative coordination with shared state
    Collaborative,
    /// Adaptive coordination that changes based on performance
    Adaptive,
}

/// Communication configuration between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Enable inter-agent communication
    pub enable_communication: bool,
    /// Communication protocol
    pub protocol: CommunicationProtocol,
    /// Message buffer size
    pub buffer_size: usize,
    /// Communication timeout in milliseconds
    pub timeout_ms: u64,
}

/// Communication protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    /// In-memory channels
    InMemory,
    /// TCP-based communication
    Tcp,
    /// Message queue
    MessageQueue,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Enable dynamic load balancing
    pub dynamic: bool,
    /// Load monitoring interval in seconds
    pub monitoring_interval: u32,
    /// Maximum load threshold
    pub max_load_threshold: f32,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least loaded agent
    LeastLoaded,
    /// Random assignment
    Random,
    /// Capability-based assignment
    CapabilityBased,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable fault tolerance
    pub enable: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Health check interval in seconds
    pub health_check_interval: u32,
    /// Agent failure threshold
    pub failure_threshold: u32,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            max_agents: 4,
            strategy: CoordinationStrategy::Independent,
            communication: CommunicationConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            enable_communication: true,
            protocol: CommunicationProtocol::InMemory,
            buffer_size: 1024,
            timeout_ms: 5000,
        }
    }
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::LeastLoaded,
            dynamic: true,
            monitoring_interval: 30,
            max_load_threshold: 0.8,
        }
    }
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_retries: 3,
            retry_delay_ms: 1000,
            health_check_interval: 60,
            failure_threshold: 3,
        }
    }
}

/// Individual training agent in the swarm
#[derive(Debug, Clone)]
pub struct SwarmAgent {
    /// Agent identifier
    pub id: String,
    /// Agent type/specialization
    pub agent_type: AgentType,
    /// Current status
    pub status: AgentStatus,
    /// Performance metrics
    pub metrics: AgentMetrics,
    /// Capabilities
    pub capabilities: Vec<AgentCapability>,
    /// Current workload
    pub current_workload: f32,
}

/// Types of agents in the swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    /// General-purpose training agent
    GeneralTrainer,
    /// Hyperparameter optimization specialist
    HyperparameterOptimizer,
    /// Architecture search specialist
    ArchitectureSearcher,
    /// Data preprocessing specialist
    DataProcessor,
    /// Model evaluation specialist
    Evaluator,
    /// Coordinator agent
    Coordinator,
}

/// Agent status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is idle and ready for work
    Idle,
    /// Agent is currently training
    Training,
    /// Agent is evaluating a model
    Evaluating,
    /// Agent is communicating with other agents
    Communicating,
    /// Agent has failed
    Failed,
    /// Agent is being restarted
    Restarting,
}

/// Agent performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Number of models trained
    pub models_trained: u32,
    /// Average training time per model
    pub avg_training_time: f32,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
    /// Current load (0.0 - 1.0)
    pub current_load: f32,
    /// Total training time
    pub total_training_time: f32,
    /// Last activity timestamp
    pub last_activity: std::time::SystemTime,
}

/// Agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentCapability {
    /// Can train specific model types
    ModelTraining(Vec<String>),
    /// Can perform hyperparameter optimization
    HyperparameterOptimization,
    /// Can search neural architectures
    ArchitectureSearch,
    /// Can preprocess data
    DataPreprocessing,
    /// Can evaluate models
    ModelEvaluation,
    /// Can coordinate other agents
    Coordination,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            models_trained: 0,
            avg_training_time: 0.0,
            success_rate: 1.0,
            current_load: 0.0,
            total_training_time: 0.0,
            last_activity: std::time::SystemTime::now(),
        }
    }
}

/// Swarm coordinator for managing agent communication and coordination
pub struct SwarmCoordinator {
    agents: HashMap<String, SwarmAgent>,
    message_channels: HashMap<String, mpsc::Sender<CoordinationMessage>>,
    config: SwarmConfig,
}

/// Messages exchanged between agents
#[derive(Debug, Clone)]
pub enum CoordinationMessage {
    /// Task assignment
    TaskAssignment {
        task_id: String,
        model: NeuralModel,
        data_subset: String, // Reference to data subset
    },
    /// Task completion notification
    TaskComplete {
        task_id: String,
        agent_id: String,
        result: ModelTrainingResult,
    },
    /// Status update
    StatusUpdate {
        agent_id: String,
        status: AgentStatus,
        metrics: AgentMetrics,
    },
    /// Request for assistance
    AssistanceRequest {
        requesting_agent: String,
        assistance_type: AssistanceType,
    },
    /// Coordination directive
    CoordinationDirective {
        directive_type: DirectiveType,
        parameters: HashMap<String, String>,
    },
}

/// Types of assistance that can be requested
#[derive(Debug, Clone)]
pub enum AssistanceType {
    /// Load balancing help
    LoadBalancing,
    /// Hyperparameter suggestions
    HyperparameterSuggestion,
    /// Architecture recommendations
    ArchitectureRecommendation,
    /// Data preprocessing help
    DataProcessing,
}

/// Types of coordination directives
#[derive(Debug, Clone)]
pub enum DirectiveType {
    /// Start training
    StartTraining,
    /// Pause training
    PauseTraining,
    /// Resume training
    ResumeTraining,
    /// Terminate training
    TerminateTraining,
    /// Redistribute workload
    RedistributeWorkload,
    /// Update configuration
    UpdateConfiguration,
}

/// Results from swarm orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResults {
    /// Individual agent results
    pub agent_results: HashMap<String, Vec<ModelTrainingResult>>,
    /// Coordination statistics
    pub coordination_stats: CoordinationStats,
    /// Performance analysis
    pub performance_analysis: SwarmPerformanceAnalysis,
    /// Communication metrics
    pub communication_metrics: CommunicationMetrics,
}

/// Coordination statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStats {
    /// Total number of agents used
    pub total_agents: usize,
    /// Total coordination messages sent
    pub total_messages: u32,
    /// Average response time
    pub avg_response_time: f32,
    /// Coordination efficiency (0.0 - 1.0)
    pub coordination_efficiency: f32,
    /// Load balancing effectiveness
    pub load_balancing_effectiveness: f32,
}

/// Swarm performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceAnalysis {
    /// Overall training speedup compared to single-threaded
    pub training_speedup: f32,
    /// Resource utilization efficiency
    pub resource_utilization: f32,
    /// Fault tolerance effectiveness
    pub fault_tolerance_effectiveness: f32,
    /// Scalability metrics
    pub scalability_metrics: ScalabilityMetrics,
}

/// Scalability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// Speedup ratio per additional agent
    pub speedup_per_agent: Vec<f32>,
    /// Efficiency degradation with scale
    pub efficiency_degradation: f32,
    /// Optimal number of agents for current workload
    pub optimal_agent_count: usize,
}

/// Communication metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    /// Total messages exchanged
    pub total_messages: u32,
    /// Average message latency
    pub avg_message_latency: f32,
    /// Message success rate
    pub message_success_rate: f32,
    /// Bandwidth utilization
    pub bandwidth_utilization: f32,
}

impl SwarmOrchestrator {
    /// Create a new swarm orchestrator
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            config,
            agents: Vec::new(),
            coordinator: None,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Initialize the swarm with agents
    pub async fn initialize(&mut self) -> TrainingResult<()> {
        log::info!("Initializing swarm with {} agents", self.config.max_agents);

        // Create agents based on configuration
        for i in 0..self.config.max_agents {
            let agent = SwarmAgent {
                id: format!("agent_{}", i),
                agent_type: self.determine_agent_type(i),
                status: AgentStatus::Idle,
                metrics: AgentMetrics::default(),
                capabilities: self.determine_agent_capabilities(i),
                current_workload: 0.0,
            };
            self.agents.push(agent);
        }

        // Initialize coordinator if using coordinated strategies
        if matches!(self.config.strategy, CoordinationStrategy::Hierarchical | CoordinationStrategy::Collaborative | CoordinationStrategy::Adaptive) {
            self.coordinator = Some(SwarmCoordinator::new(self.config.clone()));
        }

        log::info!("Swarm initialized successfully");
        Ok(())
    }

    /// Train multiple models using swarm coordination
    pub async fn train_multiple_models(
        &mut self,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
        _config: &crate::UnifiedTrainingConfig,
    ) -> TrainingResult<TrainingResults> {
        log::info!("Starting swarm-coordinated training");

        // Create model variants to train
        let models = self.create_model_variants()?;
        
        // Distribute models among agents
        let task_assignments = self.distribute_tasks(models).await?;
        
        // Execute training tasks
        let results = self.execute_training_tasks(task_assignments, train_data, val_data).await?;
        
        log::info!("Swarm training completed with {} results", results.len());
        Ok(TrainingResults::new(results, ()))
    }

    /// Determine agent type based on index and strategy
    fn determine_agent_type(&self, index: usize) -> AgentType {
        match self.config.strategy {
            CoordinationStrategy::Independent => AgentType::GeneralTrainer,
            CoordinationStrategy::Hierarchical => {
                if index == 0 {
                    AgentType::Coordinator
                } else {
                    match index % 4 {
                        1 => AgentType::GeneralTrainer,
                        2 => AgentType::HyperparameterOptimizer,
                        3 => AgentType::Evaluator,
                        _ => AgentType::DataProcessor,
                    }
                }
            },
            CoordinationStrategy::Collaborative | CoordinationStrategy::Adaptive => {
                match index % 3 {
                    0 => AgentType::GeneralTrainer,
                    1 => AgentType::HyperparameterOptimizer,
                    _ => AgentType::Evaluator,
                }
            }
        }
    }

    /// Determine agent capabilities based on type
    fn determine_agent_capabilities(&self, index: usize) -> Vec<AgentCapability> {
        let agent_type = self.determine_agent_type(index);
        match agent_type {
            AgentType::GeneralTrainer => vec![
                AgentCapability::ModelTraining(vec!["neural".to_string()]),
            ],
            AgentType::HyperparameterOptimizer => vec![
                AgentCapability::HyperparameterOptimization,
                AgentCapability::ModelTraining(vec!["neural".to_string()]),
            ],
            AgentType::ArchitectureSearcher => vec![
                AgentCapability::ArchitectureSearch,
                AgentCapability::ModelTraining(vec!["neural".to_string()]),
            ],
            AgentType::DataProcessor => vec![
                AgentCapability::DataPreprocessing,
            ],
            AgentType::Evaluator => vec![
                AgentCapability::ModelEvaluation,
            ],
            AgentType::Coordinator => vec![
                AgentCapability::Coordination,
                AgentCapability::ModelTraining(vec!["neural".to_string()]),
            ],
        }
    }

    /// Create model variants for training
    fn create_model_variants(&self) -> TrainingResult<Vec<NeuralModel>> {
        use crate::models::{ModelFactory, TrainingParameters};
        use ruv_fann::TrainingAlgorithm;

        let mut models = Vec::new();

        // Create different model architectures
        models.push(ModelFactory::create_telecom_optimized()?);
        models.push(ModelFactory::create_shallow(21, 1)?);
        models.push(ModelFactory::create_deep(21, 1)?);
        models.push(ModelFactory::create_wide(21, 1)?);

        // Create variants with different hyperparameters
        let base_model = ModelFactory::create_telecom_optimized()?;
        
        // Learning rate variants
        let mut lr_variant_1 = base_model.clone_with_name("lr_high".to_string())?;
        lr_variant_1.training_params.learning_rate = 0.2;
        models.push(lr_variant_1);

        let mut lr_variant_2 = base_model.clone_with_name("lr_low".to_string())?;
        lr_variant_2.training_params.learning_rate = 0.01;
        models.push(lr_variant_2);

        // Algorithm variants
        let mut algo_variant = base_model.clone_with_name("backprop".to_string())?;
        algo_variant.training_params.algorithm = TrainingAlgorithm::Backprop;
        models.push(algo_variant);

        Ok(models)
    }

    /// Distribute training tasks among agents
    async fn distribute_tasks(&self, models: Vec<NeuralModel>) -> TrainingResult<Vec<TaskAssignment>> {
        let mut assignments = Vec::new();
        
        match self.config.load_balancing.strategy {
            LoadBalancingStrategy::RoundRobin => {
                for (i, model) in models.into_iter().enumerate() {
                    let agent_index = i % self.agents.len();
                    assignments.push(TaskAssignment {
                        task_id: format!("task_{}", i),
                        agent_id: self.agents[agent_index].id.clone(),
                        model,
                        data_subset: "full".to_string(),
                    });
                }
            },
            LoadBalancingStrategy::LeastLoaded => {
                for (i, model) in models.into_iter().enumerate() {
                    let agent = self.agents.iter()
                        .min_by(|a, b| a.current_workload.partial_cmp(&b.current_workload).unwrap())
                        .unwrap();
                    
                    assignments.push(TaskAssignment {
                        task_id: format!("task_{}", i),
                        agent_id: agent.id.clone(),
                        model,
                        data_subset: "full".to_string(),
                    });
                }
            },
            _ => {
                // Default to round-robin for other strategies
                for (i, model) in models.into_iter().enumerate() {
                    let agent_index = i % self.agents.len();
                    assignments.push(TaskAssignment {
                        task_id: format!("task_{}", i),
                        agent_id: self.agents[agent_index].id.clone(),
                        model,
                        data_subset: "full".to_string(),
                    });
                }
            }
        }
        
        Ok(assignments)
    }

    /// Execute training tasks
    async fn execute_training_tasks(
        &self,
        assignments: Vec<TaskAssignment>,
        train_data: &TelecomDataset,
        val_data: &TelecomDataset,
    ) -> TrainingResult<Vec<ModelTrainingResult>> {
        use crate::training::{NeuralTrainer, SimpleTrainingConfig};
        
        let trainer = NeuralTrainer::new(SimpleTrainingConfig {
            parallel_training: true,
            ..Default::default()
        });

        let mut results = Vec::new();

        // Execute tasks in parallel (simplified implementation)
        for assignment in assignments {
            log::info!("Executing task {} on agent {}", assignment.task_id, assignment.agent_id);
            
            let result = trainer.train_model(
                assignment.model,
                train_data,
                Some(val_data)
            )?;
            
            results.push(result);
        }

        Ok(results)
    }
}

/// Task assignment for an agent
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    pub task_id: String,
    pub agent_id: String,
    pub model: NeuralModel,
    pub data_subset: String,
}

impl SwarmCoordinator {
    /// Create a new swarm coordinator
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            agents: HashMap::new(),
            message_channels: HashMap::new(),
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_config_default() {
        let config = SwarmConfig::default();
        assert_eq!(config.max_agents, 4);
        assert!(matches!(config.strategy, CoordinationStrategy::Independent));
    }

    #[test]
    fn test_agent_creation() {
        let agent = SwarmAgent {
            id: "test_agent".to_string(),
            agent_type: AgentType::GeneralTrainer,
            status: AgentStatus::Idle,
            metrics: AgentMetrics::default(),
            capabilities: vec![AgentCapability::ModelTraining(vec!["neural".to_string()])],
            current_workload: 0.0,
        };

        assert_eq!(agent.id, "test_agent");
        assert!(matches!(agent.status, AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_swarm_initialization() {
        let config = SwarmConfig::default();
        let mut orchestrator = SwarmOrchestrator::new(config);
        
        assert!(orchestrator.initialize().await.is_ok());
        assert_eq!(orchestrator.agents.len(), 4);
    }
}