//! Complete Workflow Demo
//!
//! This example demonstrates a complete end-to-end workflow using the Synaptic Mesh system:
//! - Distributed AI coordination
//! - Market-based compute trading
//! - Multi-domain expert collaboration
//! - Performance optimization
//! - Real-world use cases

use kimi_fann_core::{
    MicroExpert, ExpertRouter, KimiRuntime, ProcessingConfig, 
    ExpertDomain, NetworkStats, VERSION
};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Represents a complete workflow task
#[derive(Debug, Clone)]
pub struct WorkflowTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub required_domains: Vec<ExpertDomain>,
    pub complexity: TaskComplexity,
    pub priority: TaskPriority,
    pub estimated_time_ms: u64,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskComplexity {
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Workflow execution engine
pub struct WorkflowEngine {
    experts: HashMap<ExpertDomain, MicroExpert>,
    router: ExpertRouter,
    runtime: KimiRuntime,
    task_queue: Vec<WorkflowTask>,
    completed_tasks: Vec<WorkflowResult>,
    performance_metrics: Arc<Mutex<PerformanceMetrics>>,
}

#[derive(Debug, Clone)]
pub struct WorkflowResult {
    pub task_id: String,
    pub success: bool,
    pub output: String,
    pub execution_time_ms: u64,
    pub domains_used: Vec<ExpertDomain>,
    pub cost: u64,
    pub quality_score: f64,
    pub error_message: Option<String>,
}

#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub total_tasks: u64,
    pub successful_tasks: u64,
    pub total_execution_time_ms: u64,
    pub total_cost: u64,
    pub average_quality_score: f64,
    pub domain_utilization: HashMap<ExpertDomain, u64>,
}

impl WorkflowEngine {
    pub fn new() -> Self {
        let config = ProcessingConfig::new_neural_optimized();
        let mut runtime = KimiRuntime::new(config);
        let mut router = ExpertRouter::new();
        let mut experts = HashMap::new();

        // Initialize all expert domains
        for domain in [
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Mathematics,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ] {
            let expert = MicroExpert::new(domain);
            router.add_expert(MicroExpert::new(domain));
            experts.insert(domain, expert);
        }

        Self {
            experts,
            router,
            runtime,
            task_queue: Vec::new(),
            completed_tasks: Vec::new(),
            performance_metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        }
    }

    pub fn add_task(&mut self, task: WorkflowTask) {
        println!("📋 Adding task: {} (Priority: {:?}, Complexity: {:?})", 
                task.title, task.priority, task.complexity);
        self.task_queue.push(task);
    }

    pub fn execute_workflow(&mut self) -> Result<Vec<WorkflowResult>, Box<dyn std::error::Error>> {
        println!("\n🚀 Starting workflow execution with {} tasks", self.task_queue.len());
        
        // Sort tasks by priority and dependencies
        self.sort_tasks_by_priority();
        
        let mut results = Vec::new();
        
        while !self.task_queue.is_empty() {
            // Find next executable task (no pending dependencies)
            if let Some(task_idx) = self.find_next_executable_task() {
                let task = self.task_queue.remove(task_idx);
                println!("\n🔄 Executing task: {}", task.title);
                
                let result = self.execute_task(&task)?;
                results.push(result.clone());
                self.completed_tasks.push(result);
                
                // Update performance metrics
                self.update_metrics(&task, &self.completed_tasks.last().unwrap());
            } else {
                // Check for circular dependencies or missing dependencies
                return Err("Cannot resolve task dependencies - possible circular dependency".into());
            }
        }

        println!("\n✅ Workflow execution completed successfully!");
        self.print_execution_summary();
        
        Ok(results)
    }

    fn execute_task(&mut self, task: &WorkflowTask) -> Result<WorkflowResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let mut domains_used = Vec::new();
        let mut total_cost = 0u64;
        let mut outputs = Vec::new();

        // Determine execution strategy based on task complexity
        match task.complexity {
            TaskComplexity::Simple => {
                // Single expert execution
                let primary_domain = task.required_domains.first()
                    .unwrap_or(&ExpertDomain::Reasoning);
                
                let expert = self.experts.get(primary_domain).unwrap();
                let output = expert.process(&task.description);
                
                domains_used.push(*primary_domain);
                outputs.push(output);
                total_cost = 10; // Base cost for simple tasks
            }
            TaskComplexity::Medium => {
                // Router-based execution
                let output = self.router.route(&task.description);
                domains_used.push(ExpertDomain::Reasoning); // Router determines domain
                outputs.push(output);
                total_cost = 25;
            }
            TaskComplexity::Complex => {
                // Multi-expert collaboration
                for domain in &task.required_domains {
                    let expert = self.experts.get(domain).unwrap();
                    let domain_output = expert.process(&task.description);
                    domains_used.push(*domain);
                    outputs.push(format!("{:?}: {}", domain, domain_output));
                    total_cost += 15;
                }
            }
            TaskComplexity::VeryComplex => {
                // Full consensus mode with runtime
                self.runtime.set_consensus_mode(true);
                let consensus_output = self.runtime.process(&task.description);
                
                // Use all available domains for very complex tasks
                domains_used = task.required_domains.clone();
                if domains_used.is_empty() {
                    domains_used = vec![
                        ExpertDomain::Reasoning,
                        ExpertDomain::Coding,
                        ExpertDomain::Language,
                        ExpertDomain::Mathematics,
                    ];
                }
                
                outputs.push(consensus_output);
                total_cost = 100; // Premium for consensus processing
            }
        }

        let execution_time = start_time.elapsed().as_millis() as u64;
        let combined_output = outputs.join("\n\n");
        let quality_score = self.calculate_quality_score(&combined_output, execution_time, &task);

        // Simulate potential failures for demonstration
        let success = fastrand::f64() > 0.05; // 95% success rate

        let result = WorkflowResult {
            task_id: task.id.clone(),
            success,
            output: if success { combined_output } else { "Task execution failed".to_string() },
            execution_time_ms: execution_time,
            domains_used,
            cost: total_cost,
            quality_score: if success { quality_score } else { 0.0 },
            error_message: if success { None } else { Some("Simulated execution failure".to_string()) },
        };

        println!("  ✅ Task completed: {}ms, Cost: {} ruv, Quality: {:.1}%", 
                execution_time, total_cost, quality_score * 100.0);

        Ok(result)
    }

    fn sort_tasks_by_priority(&mut self) {
        self.task_queue.sort_by(|a, b| {
            // Sort by priority first, then by complexity
            let priority_cmp = match (a.priority.clone(), b.priority.clone()) {
                (TaskPriority::Critical, TaskPriority::Critical) => std::cmp::Ordering::Equal,
                (TaskPriority::Critical, _) => std::cmp::Ordering::Less,
                (_, TaskPriority::Critical) => std::cmp::Ordering::Greater,
                (TaskPriority::High, TaskPriority::High) => std::cmp::Ordering::Equal,
                (TaskPriority::High, _) => std::cmp::Ordering::Less,
                (_, TaskPriority::High) => std::cmp::Ordering::Greater,
                (TaskPriority::Medium, TaskPriority::Medium) => std::cmp::Ordering::Equal,
                (TaskPriority::Medium, TaskPriority::Low) => std::cmp::Ordering::Less,
                (TaskPriority::Low, TaskPriority::Medium) => std::cmp::Ordering::Greater,
                (TaskPriority::Low, TaskPriority::Low) => std::cmp::Ordering::Equal,
            };

            if priority_cmp == std::cmp::Ordering::Equal {
                // If priorities are equal, prefer simpler tasks first
                match (a.complexity.clone(), b.complexity.clone()) {
                    (TaskComplexity::Simple, TaskComplexity::Simple) => std::cmp::Ordering::Equal,
                    (TaskComplexity::Simple, _) => std::cmp::Ordering::Less,
                    (_, TaskComplexity::Simple) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                }
            } else {
                priority_cmp
            }
        });
    }

    fn find_next_executable_task(&self) -> Option<usize> {
        for (idx, task) in self.task_queue.iter().enumerate() {
            // Check if all dependencies are completed
            let dependencies_met = task.dependencies.iter().all(|dep_id| {
                self.completed_tasks.iter().any(|result| &result.task_id == dep_id && result.success)
            });

            if dependencies_met {
                return Some(idx);
            }
        }
        None
    }

    fn calculate_quality_score(&self, output: &str, execution_time: u64, task: &WorkflowTask) -> f64 {
        let mut score = 0.7; // Base score

        // Longer outputs generally indicate more thorough processing
        if output.len() > 200 {
            score += 0.1;
        }

        // Reasonable execution time for task complexity
        let expected_time = match task.complexity {
            TaskComplexity::Simple => 50,
            TaskComplexity::Medium => 150,
            TaskComplexity::Complex => 300,
            TaskComplexity::VeryComplex => 500,
        };

        if execution_time <= expected_time * 2 {
            score += 0.1;
        }

        // Neural processing indicators
        if output.contains("Neural") {
            score += 0.05;
        }

        // Domain-specific quality indicators
        if output.contains("analysis") || output.contains("systematic") {
            score += 0.05;
        }

        score.min(1.0)
    }

    fn update_metrics(&self, task: &WorkflowTask, result: &WorkflowResult) {
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.total_tasks += 1;
            if result.success {
                metrics.successful_tasks += 1;
            }
            metrics.total_execution_time_ms += result.execution_time_ms;
            metrics.total_cost += result.cost;
            
            // Update average quality score
            let total_successful = metrics.successful_tasks as f64;
            metrics.average_quality_score = (metrics.average_quality_score * (total_successful - 1.0) + result.quality_score) / total_successful;

            // Update domain utilization
            for domain in &result.domains_used {
                *metrics.domain_utilization.entry(*domain).or_insert(0) += 1;
            }
        }
    }

    fn print_execution_summary(&self) {
        if let Ok(metrics) = self.performance_metrics.lock() {
            println!("\n📊 Workflow Execution Summary");
            println!("============================");
            println!("📈 Tasks Executed: {}/{}", metrics.successful_tasks, metrics.total_tasks);
            println!("⏱️  Total Execution Time: {}ms", metrics.total_execution_time_ms);
            println!("💰 Total Cost: {} ruv", metrics.total_cost);
            println!("🎯 Average Quality Score: {:.1}%", metrics.average_quality_score * 100.0);
            println!("✅ Success Rate: {:.1}%", 
                    (metrics.successful_tasks as f64 / metrics.total_tasks as f64) * 100.0);

            println!("\n🔧 Domain Utilization:");
            for (domain, count) in &metrics.domain_utilization {
                println!("  {:12} - {} tasks", format!("{:?}:", domain), count);
            }

            // Performance insights
            println!("\n💡 Performance Insights:");
            let avg_time_per_task = metrics.total_execution_time_ms as f64 / metrics.total_tasks as f64;
            if avg_time_per_task < 200.0 {
                println!("  ✅ Excellent execution speed ({:.0}ms avg per task)", avg_time_per_task);
            } else if avg_time_per_task < 500.0 {
                println!("  ✅ Good execution speed ({:.0}ms avg per task)", avg_time_per_task);
            } else {
                println!("  ⚠️  Consider optimization ({:.0}ms avg per task)", avg_time_per_task);
            }

            let avg_cost_per_task = metrics.total_cost as f64 / metrics.total_tasks as f64;
            if avg_cost_per_task < 30.0 {
                println!("  💰 Cost-efficient processing ({:.0} ruv avg per task)", avg_cost_per_task);
            } else {
                println!("  💰 Higher cost processing ({:.0} ruv avg per task)", avg_cost_per_task);
            }
        }
    }

    pub fn get_network_stats(&self) -> NetworkStats {
        if let Ok(metrics) = self.performance_metrics.lock() {
            let mut expert_utilization = HashMap::new();
            let total_domain_usage: u64 = metrics.domain_utilization.values().sum();
            
            for (domain, count) in &metrics.domain_utilization {
                let utilization = *count as f64 / total_domain_usage as f64;
                expert_utilization.insert(*domain, utilization);
            }

            NetworkStats {
                active_peers: 6, // Number of expert domains
                total_queries: metrics.total_tasks,
                average_latency_ms: metrics.total_execution_time_ms as f64 / metrics.total_tasks as f64,
                expert_utilization,
                neural_accuracy: metrics.average_quality_score,
            }
        } else {
            NetworkStats {
                active_peers: 0,
                total_queries: 0,
                average_latency_ms: 0.0,
                expert_utilization: HashMap::new(),
                neural_accuracy: 0.0,
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌟 Synaptic Mesh Complete Workflow Demo");
    println!("Version: {}", VERSION);
    println!("======================================");

    let mut engine = WorkflowEngine::new();

    // Demonstrate different workflow scenarios
    demonstrate_software_development_workflow(&mut engine)?;
    demonstrate_research_analysis_workflow(&mut engine)?;
    demonstrate_data_processing_workflow(&mut engine)?;

    println!("\n🎉 All workflow demonstrations completed successfully!");
    
    // Show final network statistics
    let network_stats = engine.get_network_stats();
    println!("\n📡 Final Network Statistics:");
    println!("  Active Experts: {}", network_stats.active_peers);
    println!("  Total Tasks: {}", network_stats.total_queries);
    println!("  Average Latency: {:.1}ms", network_stats.average_latency_ms);
    println!("  Neural Accuracy: {:.1}%", network_stats.neural_accuracy * 100.0);

    Ok(())
}

/// Demonstrate a software development workflow
fn demonstrate_software_development_workflow(engine: &mut WorkflowEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💻 Software Development Workflow Demo");
    println!("===================================");

    // Add software development tasks
    engine.add_task(WorkflowTask {
        id: "dev_001".to_string(),
        title: "Requirements Analysis".to_string(),
        description: "Analyze the requirements for a web-based task management system".to_string(),
        required_domains: vec![ExpertDomain::Reasoning, ExpertDomain::Context],
        complexity: TaskComplexity::Medium,
        priority: TaskPriority::High,
        estimated_time_ms: 200,
        dependencies: vec![],
    });

    engine.add_task(WorkflowTask {
        id: "dev_002".to_string(),
        title: "System Architecture Design".to_string(),
        description: "Design the system architecture for a scalable web application with microservices".to_string(),
        required_domains: vec![ExpertDomain::Coding, ExpertDomain::Reasoning],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::High,
        estimated_time_ms: 300,
        dependencies: vec!["dev_001".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "dev_003".to_string(),
        title: "API Implementation".to_string(),
        description: "Implement REST API endpoints for user management and task operations".to_string(),
        required_domains: vec![ExpertDomain::Coding],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::Medium,
        estimated_time_ms: 400,
        dependencies: vec!["dev_002".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "dev_004".to_string(),
        title: "Database Schema Design".to_string(),
        description: "Design normalized database schema with proper indexing strategies".to_string(),
        required_domains: vec![ExpertDomain::Coding, ExpertDomain::Mathematics],
        complexity: TaskComplexity::Medium,
        priority: TaskPriority::Medium,
        estimated_time_ms: 250,
        dependencies: vec!["dev_002".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "dev_005".to_string(),
        title: "Testing Strategy".to_string(),
        description: "Develop comprehensive testing strategy including unit, integration, and E2E tests".to_string(),
        required_domains: vec![ExpertDomain::Coding, ExpertDomain::Reasoning],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::Medium,
        estimated_time_ms: 350,
        dependencies: vec!["dev_003".to_string(), "dev_004".to_string()],
    });

    let results = engine.execute_workflow()?;
    
    println!("\n✅ Software Development Workflow Results:");
    for result in &results {
        println!("  📋 {}: {} ({}ms, {} ruv)", 
                result.task_id, 
                if result.success { "SUCCESS" } else { "FAILED" },
                result.execution_time_ms,
                result.cost);
    }

    Ok(())
}

/// Demonstrate a research analysis workflow
fn demonstrate_research_analysis_workflow(engine: &mut WorkflowEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Research Analysis Workflow Demo");
    println!("=================================");

    engine.add_task(WorkflowTask {
        id: "research_001".to_string(),
        title: "Literature Review".to_string(),
        description: "Conduct systematic literature review on neural network optimization techniques".to_string(),
        required_domains: vec![ExpertDomain::Reasoning, ExpertDomain::Language],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::High,
        estimated_time_ms: 400,
        dependencies: vec![],
    });

    engine.add_task(WorkflowTask {
        id: "research_002".to_string(),
        title: "Mathematical Analysis".to_string(),
        description: "Analyze the mathematical foundations of gradient descent optimization algorithms".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning],
        complexity: TaskComplexity::VeryComplex,
        priority: TaskPriority::High,
        estimated_time_ms: 600,
        dependencies: vec!["research_001".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "research_003".to_string(),
        title: "Experimental Design".to_string(),
        description: "Design controlled experiments to validate optimization improvements".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning, ExpertDomain::Coding],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::Medium,
        estimated_time_ms: 350,
        dependencies: vec!["research_002".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "research_004".to_string(),
        title: "Results Analysis".to_string(),
        description: "Analyze experimental results and draw statistically significant conclusions".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning],
        complexity: TaskComplexity::VeryComplex,
        priority: TaskPriority::Critical,
        estimated_time_ms: 500,
        dependencies: vec!["research_003".to_string()],
    });

    let results = engine.execute_workflow()?;
    
    println!("\n✅ Research Analysis Workflow Results:");
    for result in &results {
        println!("  🔬 {}: {} ({}ms, {} ruv)", 
                result.task_id, 
                if result.success { "SUCCESS" } else { "FAILED" },
                result.execution_time_ms,
                result.cost);
    }

    Ok(())
}

/// Demonstrate a data processing workflow
fn demonstrate_data_processing_workflow(engine: &mut WorkflowEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Data Processing Workflow Demo");
    println!("===============================");

    engine.add_task(WorkflowTask {
        id: "data_001".to_string(),
        title: "Data Validation".to_string(),
        description: "Validate data quality and identify inconsistencies in the dataset".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning],
        complexity: TaskComplexity::Medium,
        priority: TaskPriority::High,
        estimated_time_ms: 200,
        dependencies: vec![],
    });

    engine.add_task(WorkflowTask {
        id: "data_002".to_string(),
        title: "Feature Engineering".to_string(),
        description: "Extract and engineer relevant features for machine learning model training".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Coding],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::High,
        estimated_time_ms: 400,
        dependencies: vec!["data_001".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "data_003".to_string(),
        title: "Model Training".to_string(),
        description: "Train and optimize machine learning models with cross-validation".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Coding, ExpertDomain::Reasoning],
        complexity: TaskComplexity::VeryComplex,
        priority: TaskPriority::Critical,
        estimated_time_ms: 800,
        dependencies: vec!["data_002".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "data_004".to_string(),
        title: "Performance Evaluation".to_string(),
        description: "Evaluate model performance using multiple metrics and statistical tests".to_string(),
        required_domains: vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning],
        complexity: TaskComplexity::Complex,
        priority: TaskPriority::Medium,
        estimated_time_ms: 300,
        dependencies: vec!["data_003".to_string()],
    });

    engine.add_task(WorkflowTask {
        id: "data_005".to_string(),
        title: "Report Generation".to_string(),
        description: "Generate comprehensive technical report with findings and recommendations".to_string(),
        required_domains: vec![ExpertDomain::Language, ExpertDomain::Reasoning],
        complexity: TaskComplexity::Medium,
        priority: TaskPriority::Low,
        estimated_time_ms: 250,
        dependencies: vec!["data_004".to_string()],
    });

    let results = engine.execute_workflow()?;
    
    println!("\n✅ Data Processing Workflow Results:");
    for result in &results {
        println!("  📊 {}: {} ({}ms, {} ruv)", 
                result.task_id, 
                if result.success { "SUCCESS" } else { "FAILED" },
                result.execution_time_ms,
                result.cost);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_engine_creation() {
        let engine = WorkflowEngine::new();
        assert_eq!(engine.experts.len(), 6);
        assert_eq!(engine.task_queue.len(), 0);
    }

    #[test]
    fn test_task_addition() {
        let mut engine = WorkflowEngine::new();
        let task = WorkflowTask {
            id: "test_001".to_string(),
            title: "Test Task".to_string(),
            description: "Test description".to_string(),
            required_domains: vec![ExpertDomain::Reasoning],
            complexity: TaskComplexity::Simple,
            priority: TaskPriority::Medium,
            estimated_time_ms: 100,
            dependencies: vec![],
        };

        engine.add_task(task);
        assert_eq!(engine.task_queue.len(), 1);
    }

    #[test]
    fn test_task_priority_sorting() {
        let mut engine = WorkflowEngine::new();
        
        // Add tasks in reverse priority order
        engine.add_task(WorkflowTask {
            id: "low".to_string(),
            title: "Low Priority".to_string(),
            description: "Low priority task".to_string(),
            required_domains: vec![ExpertDomain::Reasoning],
            complexity: TaskComplexity::Simple,
            priority: TaskPriority::Low,
            estimated_time_ms: 100,
            dependencies: vec![],
        });

        engine.add_task(WorkflowTask {
            id: "critical".to_string(),
            title: "Critical Priority".to_string(),
            description: "Critical priority task".to_string(),
            required_domains: vec![ExpertDomain::Reasoning],
            complexity: TaskComplexity::Simple,
            priority: TaskPriority::Critical,
            estimated_time_ms: 100,
            dependencies: vec![],
        });

        engine.sort_tasks_by_priority();
        
        assert_eq!(engine.task_queue[0].priority, TaskPriority::Critical);
        assert_eq!(engine.task_queue[1].priority, TaskPriority::Low);
    }

    #[test]
    fn test_quality_score_calculation() {
        let engine = WorkflowEngine::new();
        let task = WorkflowTask {
            id: "test".to_string(),
            title: "Test".to_string(),
            description: "Test".to_string(),
            required_domains: vec![ExpertDomain::Reasoning],
            complexity: TaskComplexity::Medium,
            priority: TaskPriority::Medium,
            estimated_time_ms: 150,
            dependencies: vec![],
        };

        let score = engine.calculate_quality_score("Short output", 100, &task);
        assert!(score > 0.5 && score <= 1.0);

        let score = engine.calculate_quality_score(
            "Very long and detailed output with Neural processing indicators and systematic analysis",
            120,
            &task
        );
        assert!(score > 0.9);
    }
}