//! Swarm orchestration for parallel neural network training

use crate::data::{TelecomDataset, DataSplit};
use crate::models::{NeuralModel, ModelRegistry, NetworkArchitectures, TrainingParameters, HyperparameterConfig};
use crate::training::{SimpleTrainingConfig, NeuralTrainer, ModelTrainingResult, HyperparameterTuner};
use crate::evaluation::{ModelEvaluator, ModelComparisonReport};
use crate::config::TrainingConfig;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use rayon::prelude::*;

/// Swarm agent types for specialized tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    DataProcessor,
    ArchitectureDesigner,
    TrainingSpecialist,
    ActivationExpert,
    EvaluationAnalyst,
}

/// Swarm agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_type: AgentType,
    pub name: String,
    pub capabilities: Vec<String>,
    pub priority: u8,
    pub max_parallel_tasks: usize,
}

/// Swarm orchestrator for managing neural network training
#[derive(Debug)]
pub struct SwarmOrchestrator {
    agents: HashMap<AgentType, AgentConfig>,
    model_registry: Arc<Mutex<ModelRegistry>>,
    training_config: SimpleTrainingConfig,
    active_tasks: Arc<Mutex<Vec<TrainingTask>>>,
}

/// Training task for swarm execution
#[derive(Debug, Clone)]
pub struct TrainingTask {
    pub id: String,
    pub model_name: String,
    pub agent_type: AgentType,
    pub status: TaskStatus,
    pub priority: u8,
    pub created_at: std::time::Instant,
}

/// Task execution status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Swarm training results
#[derive(Debug, Clone, Serialize)]
pub struct SwarmTrainingResults {
    pub agent_results: HashMap<AgentType, Vec<ModelTrainingResult>>,
    pub best_models_per_agent: HashMap<AgentType, String>,
    pub overall_best_model: String,
    pub coordination_metrics: CoordinationMetrics,
    pub parallel_efficiency: f32,
}

/// Metrics for swarm coordination efficiency
#[derive(Debug, Clone, Serialize)]
pub struct CoordinationMetrics {
    pub total_agents_used: usize,
    pub avg_task_completion_time: std::time::Duration,
    pub resource_utilization: f32,
    pub load_balance_score: f32,
    pub communication_overhead: std::time::Duration,
}

impl SwarmOrchestrator {
    /// Create new swarm orchestrator
    pub fn new() -> Self {
        let mut agents = HashMap::new();
        
        // Initialize specialized agents
        agents.insert(AgentType::DataProcessor, AgentConfig {
            agent_type: AgentType::DataProcessor,
            name: "Data_Processing_Expert".to_string(),
            capabilities: vec![
                "data_preprocessing".to_string(),
                "feature_engineering".to_string(),
                "data_splitting".to_string(),
                "normalization".to_string(),
            ],
            priority: 10,
            max_parallel_tasks: 2,
        });
        
        agents.insert(AgentType::ArchitectureDesigner, AgentConfig {
            agent_type: AgentType::ArchitectureDesigner,
            name: "Neural_Architecture_Designer".to_string(),
            capabilities: vec![
                "network_design".to_string(),
                "architecture_optimization".to_string(),
                "layer_configuration".to_string(),
            ],
            priority: 9,
            max_parallel_tasks: 3,
        });
        
        agents.insert(AgentType::TrainingSpecialist, AgentConfig {
            agent_type: AgentType::TrainingSpecialist,
            name: "Training_Implementation_Specialist".to_string(),
            capabilities: vec![
                "backpropagation".to_string(),
                "gradient_optimization".to_string(),
                "convergence_monitoring".to_string(),
            ],
            priority: 10,
            max_parallel_tasks: 4,
        });
        
        agents.insert(AgentType::ActivationExpert, AgentConfig {
            agent_type: AgentType::ActivationExpert,
            name: "Activation_Function_Researcher".to_string(),
            capabilities: vec![
                "activation_analysis".to_string(),
                "function_optimization".to_string(),
                "gradient_computation".to_string(),
            ],
            priority: 7,
            max_parallel_tasks: 2,
        });
        
        agents.insert(AgentType::EvaluationAnalyst, AgentConfig {
            agent_type: AgentType::EvaluationAnalyst,
            name: "Evaluation_Metrics_Specialist".to_string(),
            capabilities: vec![
                "model_evaluation".to_string(),
                "performance_analysis".to_string(),
                "statistical_validation".to_string(),
            ],
            priority: 8,
            max_parallel_tasks: 3,
        });
        
        Self {
            agents,
            model_registry: Arc::new(Mutex::new(ModelRegistry::new())),
            training_config: SimpleTrainingConfig::default(),
            active_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Initialize training swarm with specialized agents
    pub async fn initialize_training_swarm(&mut self) -> Result<()> {
        log::info!("Initializing training swarm with {} agents", self.agents.len());
        
        // Log agent capabilities
        for (agent_type, config) in &self.agents {
            log::info!("Agent {:?}: {} (Priority: {}, Max Tasks: {})", 
                      agent_type, config.name, config.priority, config.max_parallel_tasks);
            log::debug!("Capabilities: {:?}", config.capabilities);
        }
        
        Ok(())
    }
    
    /// Train multiple models using swarm coordination
    pub async fn train_multiple_models(
        &mut self,
        train_data: &TelecomDataset,
        test_data: &TelecomDataset,
        config: &TrainingConfig,
    ) -> Result<Vec<ModelTrainingResult>> {
        log::info!("Starting swarm-coordinated training");
        
        self.training_config = config.clone();
        
        // Step 1: Data Processing Agent - Prepare data
        let processed_data = self.execute_data_processing(train_data, test_data).await?;
        
        // Step 2: Architecture Designer Agent - Create model architectures
        let architectures = self.execute_architecture_design(&processed_data.train).await?;
        
        // Step 3: Training Specialist Agent - Train models in parallel
        let training_results = self.execute_parallel_training(architectures, &processed_data).await?;
        
        // Step 4: Evaluation Analyst Agent - Analyze results
        let _evaluation_report = self.execute_evaluation_analysis(&training_results).await?;
        
        log::info!("Swarm training completed with {} models", training_results.len());
        Ok(training_results)
    }
    
    /// Execute data processing tasks
    async fn execute_data_processing(
        &self,
        train_data: &TelecomDataset,
        test_data: &TelecomDataset,
    ) -> Result<DataSplit> {
        log::info!("🔄 Data Processing Agent: Preparing training data");
        
        let mut processed_train = train_data.clone();
        let mut processed_test = test_data.clone();
        
        // Apply normalization if configured
        if self.training_config.normalize_features {
            processed_train.normalize()
                .context("Failed to normalize training features")?;
            processed_test.normalize()
                .context("Failed to normalize test features")?;
        }
        
        log::info!("✅ Data processing completed");
        Ok(DataSplit {
            train: processed_train,
            test: processed_test,
        })
    }
    
    /// Execute architecture design tasks
    async fn execute_architecture_design(&self, train_data: &TelecomDataset) -> Result<Vec<NeuralModel>> {
        log::info!("🏗️ Architecture Designer Agent: Creating neural network architectures");
        
        let input_size = train_data.features.ncols();
        let output_size = 1; // Regression task
        
        // Get all predefined architectures
        let architectures = NetworkArchitectures::get_all_architectures(input_size, output_size);
        
        let mut models = Vec::new();
        for architecture in architectures {
            let model = NeuralModel::from_architecture(architecture)
                .context("Failed to create model from architecture")?;
            models.push(model);
        }
        
        log::info!("✅ Created {} neural network architectures", models.len());
        for model in &models {
            log::info!("  - {}: {} parameters", model.name, model.summary().total_parameters);
        }
        
        Ok(models)
    }
    
    /// Execute parallel training with multiple specialists
    async fn execute_parallel_training(
        &self,
        models: Vec<NeuralModel>,
        data: &DataSplit,
    ) -> Result<Vec<ModelTrainingResult>> {
        log::info!("🚀 Training Specialist Agent: Starting parallel model training");
        
        let trainer = NeuralTrainer::new(self.training_config.clone());
        
        // Train models in parallel using rayon
        let results: Result<Vec<_>, _> = models.into_par_iter()
            .map(|model| {
                log::info!("Training model: {}", model.name);
                trainer.train_model(model, &data.train, Some(&data.test))
            })
            .collect();
        
        let training_results = results.context("Failed to train models")?;
        
        log::info!("✅ Parallel training completed");
        for result in &training_results {
            log::info!("  - {}: Error = {:.6}, Epochs = {}, Time = {:?}", 
                      result.model_name, result.final_error, 
                      result.epochs_completed, result.training_time);
        }
        
        Ok(training_results)
    }
    
    /// Execute evaluation and analysis
    async fn execute_evaluation_analysis(
        &self,
        results: &[ModelTrainingResult],
    ) -> Result<ModelComparisonReport> {
        log::info!("📊 Evaluation Analyst Agent: Analyzing model performance");
        
        let report = ModelEvaluator::compare_models(results)
            .context("Failed to generate comparison report")?;
        
        log::info!("✅ Evaluation analysis completed");
        log::info!("Best overall model: {}", report.best_overall_model);
        log::info!("Best accuracy model: {}", report.best_accuracy_model);
        log::info!("Fastest model: {}", report.fastest_model);
        log::info!("Most efficient model: {}", report.most_efficient_model);
        
        Ok(report)
    }
    
    /// Evaluate models using swarm coordination
    pub async fn evaluate_models(
        &self,
        _results: &[ModelTrainingResult],
        _test_data: &TelecomDataset,
    ) -> Result<()> {
        log::info!("🔍 Evaluation Agent: Performing comprehensive model evaluation");
        
        // TODO: Implement comprehensive evaluation
        // This would include cross-validation, statistical tests, etc.
        
        log::info!("✅ Model evaluation completed");
        Ok(())
    }
    
    /// Perform hyperparameter tuning with swarm coordination
    pub async fn hyperparameter_tuning(
        &self,
        model_template: NeuralModel,
        data: &DataSplit,
        hyperparameter_config: &HyperparameterConfig,
    ) -> Result<(TrainingParameters, ModelTrainingResult)> {
        log::info!("🔧 Starting swarm-coordinated hyperparameter tuning");
        
        let tuner = HyperparameterTuner::new(self.training_config.clone());
        let combinations = hyperparameter_config.generate_combinations();
        
        log::info!("Testing {} hyperparameter combinations", combinations.len());
        
        let (best_params, best_result) = tuner.grid_search(
            model_template,
            combinations,
            &data.train,
            &data.test,
        )?;
        
        log::info!("✅ Hyperparameter tuning completed");
        log::info!("Best parameters: LR={}, Momentum={}, Batch Size={:?}", 
                  best_params.learning_rate, best_params.momentum, best_params.batch_size);
        log::info!("Best validation error: {:.6}", best_result.final_error);
        
        Ok((best_params, best_result))
    }
    
    /// Get swarm status and metrics
    pub async fn get_swarm_status(&self) -> SwarmStatus {
        let active_tasks = self.active_tasks.lock().await;
        let registry = self.model_registry.lock().await;
        
        SwarmStatus {
            total_agents: self.agents.len(),
            active_tasks: active_tasks.len(),
            registered_models: registry.model_names().len(),
            agent_utilization: self.calculate_agent_utilization(&active_tasks),
        }
    }
    
    /// Calculate agent utilization metrics
    fn calculate_agent_utilization(&self, active_tasks: &[TrainingTask]) -> HashMap<AgentType, f32> {
        let mut utilization = HashMap::new();
        
        for (agent_type, config) in &self.agents {
            let agent_tasks = active_tasks.iter()
                .filter(|task| task.agent_type == *agent_type)
                .count();
            
            let utilization_rate = agent_tasks as f32 / config.max_parallel_tasks as f32;
            utilization.insert(*agent_type, utilization_rate.min(1.0));
        }
        
        utilization
    }
}

impl Default for SwarmOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Swarm status information
#[derive(Debug, Clone, Serialize)]
pub struct SwarmStatus {
    pub total_agents: usize,
    pub active_tasks: usize,
    pub registered_models: usize,
    pub agent_utilization: HashMap<AgentType, f32>,
}

/// Agent task execution context
#[derive(Debug)]
pub struct AgentContext {
    pub agent_config: AgentConfig,
    pub shared_memory: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    pub communication_channel: tokio::sync::mpsc::Sender<AgentMessage>,
}

/// Inter-agent communication message
#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub from_agent: AgentType,
    pub to_agent: Option<AgentType>,
    pub message_type: MessageType,
    pub payload: serde_json::Value,
    pub timestamp: std::time::Instant,
}

/// Types of messages between agents
#[derive(Debug, Clone, Copy)]
pub enum MessageType {
    DataReady,
    ModelCreated,
    TrainingComplete,
    EvaluationResult,
    CoordinationRequest,
}

/// Swarm coordination utilities
pub struct SwarmCoordinator;

impl SwarmCoordinator {
    /// Coordinate task execution across agents
    pub async fn coordinate_tasks(
        tasks: Vec<TrainingTask>,
        agents: &HashMap<AgentType, AgentConfig>,
    ) -> Result<Vec<TrainingTask>> {
        log::info!("Coordinating {} tasks across {} agents", tasks.len(), agents.len());
        
        // Sort tasks by priority
        let mut sorted_tasks = tasks;
        sorted_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        // Assign tasks to agents based on capabilities and availability
        for task in &mut sorted_tasks {
            if let Some(agent) = agents.get(&task.agent_type) {
                log::debug!("Assigned task {} to agent {}", task.id, agent.name);
            }
        }
        
        Ok(sorted_tasks)
    }
    
    /// Monitor agent performance and adjust workload
    pub async fn monitor_and_balance(
        active_tasks: &[TrainingTask],
        utilization: &HashMap<AgentType, f32>,
    ) -> Result<Vec<LoadBalanceAction>> {
        let mut actions = Vec::new();
        
        // Check for overloaded agents
        for (agent_type, &util) in utilization {
            if util > 0.9 {
                actions.push(LoadBalanceAction::ReduceLoad(*agent_type));
            } else if util < 0.3 {
                actions.push(LoadBalanceAction::IncreaseLoad(*agent_type));
            }
        }
        
        Ok(actions)
    }
}

/// Load balancing actions
#[derive(Debug, Clone)]
pub enum LoadBalanceAction {
    ReduceLoad(AgentType),
    IncreaseLoad(AgentType),
    RedistributeTasks,
    SpawnAdditionalAgent(AgentType),
}