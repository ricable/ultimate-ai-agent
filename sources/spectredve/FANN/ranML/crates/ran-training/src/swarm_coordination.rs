use anyhow::Result;
use log::{info, debug, warn};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SwarmAgent {
    pub id: String,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    pub capabilities: Vec<String>,
    pub performance_metrics: AgentMetrics,
    pub created_at: Instant,
}

#[derive(Debug, Clone)]
pub enum AgentType {
    DataPreprocessor,
    ModelTrainer,
    Evaluator,
    Coordinator,
}

#[derive(Debug, Clone)]
pub enum AgentStatus {
    Idle,
    Working,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct AgentMetrics {
    pub tasks_completed: usize,
    pub average_task_time: f64,
    pub success_rate: f64,
    pub last_activity: Instant,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            tasks_completed: 0,
            average_task_time: 0.0,
            success_rate: 1.0,
            last_activity: Instant::now(),
        }
    }
}

pub struct SwarmCoordinator {
    pub agents: HashMap<String, SwarmAgent>,
    pub task_queue: Vec<SwarmTask>,
    pub completed_tasks: Vec<SwarmTask>,
    pub swarm_id: String,
    pub max_agents: usize,
}

#[derive(Debug, Clone)]
pub struct SwarmTask {
    pub id: String,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub assigned_agent: Option<String>,
    pub status: TaskStatus,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub result: Option<String>,
}

#[derive(Debug, Clone)]
pub enum TaskType {
    DataPreprocessing,
    ModelTraining { model_type: String },
    ModelEvaluation,
    Coordination,
}

#[derive(Debug, Clone)]
pub enum TaskPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl SwarmCoordinator {
    pub fn new(swarm_id: String, max_agents: usize) -> Self {
        info!("🐝 Initializing swarm coordinator: {} (max agents: {})", swarm_id, max_agents);
        
        Self {
            agents: HashMap::new(),
            task_queue: Vec::new(),
            completed_tasks: Vec::new(),
            swarm_id,
            max_agents,
        }
    }
    
    pub fn spawn_agent(&mut self, agent_type: AgentType, capabilities: Vec<String>) -> Result<String> {
        if self.agents.len() >= self.max_agents {
            return Err(anyhow::anyhow!("Maximum number of agents reached"));
        }
        
        let agent_id = format!("agent-{}-{}", 
                              match agent_type {
                                  AgentType::DataPreprocessor => "preprocessor",
                                  AgentType::ModelTrainer => "trainer",
                                  AgentType::Evaluator => "evaluator",
                                  AgentType::Coordinator => "coordinator",
                              },
                              self.agents.len() + 1);
        
        let agent = SwarmAgent {
            id: agent_id.clone(),
            agent_type: agent_type.clone(),
            status: AgentStatus::Idle,
            capabilities,
            performance_metrics: AgentMetrics::default(),
            created_at: Instant::now(),
        };
        
        self.agents.insert(agent_id.clone(), agent);
        
        info!("✅ Spawned agent: {} ({:?})", agent_id, agent_type);
        Ok(agent_id)
    }
    
    pub fn add_task(&mut self, task_type: TaskType, priority: TaskPriority) -> String {
        let task_id = format!("task-{}", self.task_queue.len() + self.completed_tasks.len() + 1);
        
        let task = SwarmTask {
            id: task_id.clone(),
            task_type,
            priority,
            assigned_agent: None,
            status: TaskStatus::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            result: None,
        };
        
        self.task_queue.push(task);
        
        debug!("📋 Added task: {} ({:?})", task_id, task.task_type);
        task_id
    }
    
    pub fn assign_tasks(&mut self) -> Result<()> {
        // Sort tasks by priority
        self.task_queue.sort_by(|a, b| {
            match (&a.priority, &b.priority) {
                (TaskPriority::High, TaskPriority::High) => std::cmp::Ordering::Equal,
                (TaskPriority::High, _) => std::cmp::Ordering::Less,
                (TaskPriority::Medium, TaskPriority::High) => std::cmp::Ordering::Greater,
                (TaskPriority::Medium, TaskPriority::Medium) => std::cmp::Ordering::Equal,
                (TaskPriority::Medium, TaskPriority::Low) => std::cmp::Ordering::Less,
                (TaskPriority::Low, _) => std::cmp::Ordering::Greater,
            }
        });
        
        // Assign tasks to available agents
        for task in &mut self.task_queue {
            if task.status == TaskStatus::Pending {
                if let Some(agent_id) = self.find_suitable_agent(&task.task_type) {
                    task.assigned_agent = Some(agent_id.clone());
                    task.status = TaskStatus::InProgress;
                    task.started_at = Some(Instant::now());
                    
                    if let Some(agent) = self.agents.get_mut(&agent_id) {
                        agent.status = AgentStatus::Working;
                        agent.performance_metrics.last_activity = Instant::now();
                    }
                    
                    info!("🎯 Assigned task {} to agent {}", task.id, agent_id);
                }
            }
        }
        
        Ok(())
    }
    
    fn find_suitable_agent(&self, task_type: &TaskType) -> Option<String> {
        for (agent_id, agent) in &self.agents {
            if agent.status == AgentStatus::Idle {
                let is_suitable = match (task_type, &agent.agent_type) {
                    (TaskType::DataPreprocessing, AgentType::DataPreprocessor) => true,
                    (TaskType::ModelTraining { .. }, AgentType::ModelTrainer) => true,
                    (TaskType::ModelEvaluation, AgentType::Evaluator) => true,
                    (TaskType::Coordination, AgentType::Coordinator) => true,
                    _ => false,
                };
                
                if is_suitable {
                    return Some(agent_id.clone());
                }
            }
        }
        None
    }
    
    pub fn complete_task(&mut self, task_id: &str, result: String) -> Result<()> {
        if let Some(task_idx) = self.task_queue.iter().position(|t| t.id == task_id) {
            let mut task = self.task_queue.remove(task_idx);
            
            task.status = TaskStatus::Completed;
            task.completed_at = Some(Instant::now());
            task.result = Some(result);
            
            // Update agent status
            if let Some(agent_id) = &task.assigned_agent {
                if let Some(agent) = self.agents.get_mut(agent_id) {
                    agent.status = AgentStatus::Idle;
                    agent.performance_metrics.tasks_completed += 1;
                    
                    // Calculate average task time
                    if let (Some(started), Some(completed)) = (task.started_at, task.completed_at) {
                        let task_time = completed.duration_since(started).as_secs_f64();
                        let total_tasks = agent.performance_metrics.tasks_completed as f64;
                        let current_avg = agent.performance_metrics.average_task_time;
                        agent.performance_metrics.average_task_time = 
                            (current_avg * (total_tasks - 1.0) + task_time) / total_tasks;
                    }
                }
            }
            
            self.completed_tasks.push(task);
            info!("✅ Task completed: {}", task_id);
        }
        
        Ok(())
    }
    
    pub fn get_swarm_status(&self) -> SwarmStatus {
        let total_agents = self.agents.len();
        let active_agents = self.agents.values()
            .filter(|a| matches!(a.status, AgentStatus::Working))
            .count();
        
        let pending_tasks = self.task_queue.iter()
            .filter(|t| matches!(t.status, TaskStatus::Pending))
            .count();
        
        let in_progress_tasks = self.task_queue.iter()
            .filter(|t| matches!(t.status, TaskStatus::InProgress))
            .count();
        
        SwarmStatus {
            swarm_id: self.swarm_id.clone(),
            total_agents,
            active_agents,
            pending_tasks,
            in_progress_tasks,
            completed_tasks: self.completed_tasks.len(),
            overall_efficiency: self.calculate_efficiency(),
        }
    }
    
    fn calculate_efficiency(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        
        let total_success_rate: f64 = self.agents.values()
            .map(|a| a.performance_metrics.success_rate)
            .sum();
        
        total_success_rate / self.agents.len() as f64
    }
    
    pub fn print_status(&self) {
        let status = self.get_swarm_status();
        
        println!("\n🐝 Swarm Status: {}", status.swarm_id);
        println!("═══════════════════════════");
        println!("👥 Agents: {}/{} active", status.active_agents, status.total_agents);
        println!("📋 Tasks: {} pending, {} in progress, {} completed", 
                 status.pending_tasks, status.in_progress_tasks, status.completed_tasks);
        println!("⚡ Efficiency: {:.1}%", status.overall_efficiency * 100.0);
        
        println!("\nAgent Details:");
        for (agent_id, agent) in &self.agents {
            println!("  🤖 {}: {:?} - {:?} ({} tasks, {:.2}s avg)", 
                     agent_id, agent.agent_type, agent.status,
                     agent.performance_metrics.tasks_completed,
                     agent.performance_metrics.average_task_time);
        }
    }
}

#[derive(Debug)]
pub struct SwarmStatus {
    pub swarm_id: String,
    pub total_agents: usize,
    pub active_agents: usize,
    pub pending_tasks: usize,
    pub in_progress_tasks: usize,
    pub completed_tasks: usize,
    pub overall_efficiency: f64,
}