//! Unit tests for swarm orchestrator functionality

use crate::swarm::*;
use crate::agent::{MockAgent, Agent, AgentId, AgentStatus, Capability};
use crate::task::{Task, TaskPayload, TaskPriority, TaskResult, DistributionStrategy};
use crate::topology::TopologyType;
use crate::error::SwarmError;

#[tokio::test]
async fn test_swarm_creation() {
    let config = SwarmConfig::default();
    let swarm = Swarm::new(config);
    
    assert_eq!(swarm.task_queue_size(), 0);
    assert_eq!(swarm.assigned_tasks_count(), 0);
    
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 0);
    assert_eq!(metrics.active_agents, 0);
}

#[tokio::test]
async fn test_register_agent() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    let agent = Box::new(MockAgent::new("agent-1"));
    assert!(swarm.register_agent(agent).is_ok());
    
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 1);
}

#[tokio::test]
async fn test_register_agent_limit() {
    let config = SwarmConfig {
        max_agents: 2,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    // Register agents up to the limit
    assert!(swarm.register_agent(Box::new(MockAgent::new("agent-1"))).is_ok());
    assert!(swarm.register_agent(Box::new(MockAgent::new("agent-2"))).is_ok());
    
    // Try to register beyond limit
    let result = swarm.register_agent(Box::new(MockAgent::new("agent-3")));
    assert!(matches!(result, Err(SwarmError::ResourceExhausted { .. })));
}

#[tokio::test]
async fn test_unregister_agent() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    let agent_id = AgentId::new("agent-1");
    let agent = Box::new(MockAgent::new("agent-1"));
    
    swarm.register_agent(agent).unwrap();
    assert_eq!(swarm.metrics().total_agents, 1);
    
    swarm.unregister_agent(&agent_id).unwrap();
    assert_eq!(swarm.metrics().total_agents, 0);
}

#[tokio::test]
async fn test_unregister_nonexistent_agent() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    let agent_id = AgentId::new("nonexistent");
    
    let result = swarm.unregister_agent(&agent_id);
    assert!(matches!(result, Err(SwarmError::AgentNotFound { .. })));
}

#[tokio::test]
async fn test_submit_task() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    let task = Task::new("task-1", "compute");
    swarm.submit_task(task).unwrap();
    
    assert_eq!(swarm.task_queue_size(), 1);
}

#[tokio::test]
async fn test_task_distribution_no_agents() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    swarm.submit_task(Task::new("task-1", "compute")).unwrap();
    
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 0);
    assert_eq!(swarm.task_queue_size(), 1); // Task should remain in queue
}

#[tokio::test]
async fn test_task_distribution_with_agents() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Register and start agents
    let mut agent1 = Box::new(MockAgent::new("agent-1"));
    let mut agent2 = Box::new(MockAgent::new("agent-2"));
    
    agent1.start().await.unwrap();
    agent2.start().await.unwrap();
    
    swarm.register_agent(agent1).unwrap();
    swarm.register_agent(agent2).unwrap();
    
    // Submit tasks
    swarm.submit_task(Task::new("task-1", "compute")).unwrap();
    swarm.submit_task(Task::new("task-2", "analyze")).unwrap();
    
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 2);
    assert_eq!(swarm.task_queue_size(), 0);
}

#[tokio::test]
async fn test_capability_based_distribution() {
    let mut swarm = Swarm::new(SwarmConfig {
        distribution_strategy: DistributionStrategy::CapabilityBased,
        ..Default::default()
    });
    
    // Agent with specific capabilities
    let mut agent1 = Box::new(MockAgent::new("agent-1")
        .with_capabilities(vec![
            Capability::new("neural", "1.0"),
        ]));
    
    let mut agent2 = Box::new(MockAgent::new("agent-2")
        .with_capabilities(vec![
            Capability::new("compute", "1.0"),
        ]));
    
    agent1.start().await.unwrap();
    agent2.start().await.unwrap();
    
    swarm.register_agent(agent1).unwrap();
    swarm.register_agent(agent2).unwrap();
    
    // Task requiring neural capability
    let task = Task::new("task-1", "neural-processing")
        .require_capability(Capability::new("neural", "1.0"));
    
    swarm.submit_task(task).unwrap();
    
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].1, AgentId::new("agent-1"));
}

#[tokio::test]
async fn test_least_loaded_distribution() {
    let mut swarm = Swarm::new(SwarmConfig {
        distribution_strategy: DistributionStrategy::LeastLoaded,
        ..Default::default()
    });
    
    let mut agent1 = Box::new(MockAgent::new("agent-1"));
    let mut agent2 = Box::new(MockAgent::new("agent-2"));
    
    agent1.start().await.unwrap();
    agent2.start().await.unwrap();
    
    swarm.register_agent(agent1).unwrap();
    swarm.register_agent(agent2).unwrap();
    
    // Submit multiple tasks
    for i in 0..4 {
        swarm.submit_task(Task::new(format!("task-{}", i), "compute")).unwrap();
    }
    
    let assignments = swarm.distribute_tasks().await.unwrap();
    
    // Tasks should be distributed evenly
    assert_eq!(assignments.len(), 4);
    
    let agent1_tasks = assignments.iter().filter(|(_, id)| id == &AgentId::new("agent-1")).count();
    let agent2_tasks = assignments.iter().filter(|(_, id)| id == &AgentId::new("agent-2")).count();
    
    assert_eq!(agent1_tasks, 2);
    assert_eq!(agent2_tasks, 2);
}

#[tokio::test]
async fn test_start_all_agents() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    swarm.register_agent(Box::new(MockAgent::new("agent-1"))).unwrap();
    swarm.register_agent(Box::new(MockAgent::new("agent-2"))).unwrap();
    swarm.register_agent(Box::new(MockAgent::new("agent-3"))).unwrap();
    
    swarm.start_all_agents().await.unwrap();
    
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 3);
    assert_eq!(metrics.active_agents, 3);
}

#[tokio::test]
async fn test_shutdown_all_agents() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    swarm.register_agent(Box::new(MockAgent::new("agent-1"))).unwrap();
    swarm.register_agent(Box::new(MockAgent::new("agent-2"))).unwrap();
    
    swarm.start_all_agents().await.unwrap();
    assert_eq!(swarm.metrics().active_agents, 2);
    
    swarm.shutdown_all_agents().await.unwrap();
    
    let statuses = swarm.agent_statuses();
    assert!(statuses.values().all(|status| *status == AgentStatus::Stopped));
}

#[tokio::test]
async fn test_get_agent() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    let agent_id = AgentId::new("test-agent");
    swarm.register_agent(Box::new(MockAgent::new("test-agent"))).unwrap();
    
    assert!(swarm.get_agent(&agent_id).is_some());
    assert!(swarm.get_agent(&AgentId::new("nonexistent")).is_none());
}

#[tokio::test]
async fn test_swarm_metrics() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Add agents
    let mut agent1 = Box::new(MockAgent::new("agent-1"));
    let mut agent2 = Box::new(MockAgent::new("agent-2"));
    
    agent1.start().await.unwrap();
    
    swarm.register_agent(agent1).unwrap();
    swarm.register_agent(agent2).unwrap();
    
    // Add tasks
    swarm.submit_task(Task::new("task-1", "compute")).unwrap();
    swarm.submit_task(Task::new("task-2", "analyze")).unwrap();
    
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 2);
    assert_eq!(metrics.active_agents, 1); // Only agent1 is started
    assert_eq!(metrics.queued_tasks, 2);
    assert_eq!(metrics.assigned_tasks, 0);
}

#[tokio::test]
async fn test_task_priority_handling() {
    let mut swarm = Swarm::new(SwarmConfig {
        distribution_strategy: DistributionStrategy::Priority,
        ..Default::default()
    });
    
    let mut agent = Box::new(MockAgent::new("agent-1"));
    agent.start().await.unwrap();
    swarm.register_agent(agent).unwrap();
    
    // Submit tasks with different priorities
    swarm.submit_task(
        Task::new("low", "compute").with_priority(TaskPriority::Low)
    ).unwrap();
    
    swarm.submit_task(
        Task::new("critical", "compute").with_priority(TaskPriority::Critical)
    ).unwrap();
    
    swarm.submit_task(
        Task::new("normal", "compute").with_priority(TaskPriority::Normal)
    ).unwrap();
    
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 3);
}

#[tokio::test]
async fn test_topology_update_on_agent_registration() {
    let config = SwarmConfig {
        topology_type: TopologyType::Mesh,
        ..Default::default()
    };
    let mut swarm = Swarm::new(config);
    
    swarm.register_agent(Box::new(MockAgent::new("agent-1"))).unwrap();
    swarm.register_agent(Box::new(MockAgent::new("agent-2"))).unwrap();
    swarm.register_agent(Box::new(MockAgent::new("agent-3"))).unwrap();
    
    // In mesh topology, all agents should be connected
    let metrics = swarm.metrics();
    assert_eq!(metrics.total_agents, 3);
    // Each agent connects to 2 others, total 6 connections, divided by 2 for undirected = 3
    assert_eq!(metrics.total_connections, 3);
}

#[tokio::test]
async fn test_concurrent_task_submission() {
    use futures::future::join_all;
    
    let swarm = Arc::new(tokio::sync::Mutex::new(Swarm::new(SwarmConfig::default())));
    
    // Register agents
    {
        let mut swarm_guard = swarm.lock().await;
        let mut agent = Box::new(MockAgent::new("agent-1"));
        agent.start().await.unwrap();
        swarm_guard.register_agent(agent).unwrap();
    }
    
    // Submit tasks concurrently
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let swarm_clone = swarm.clone();
            tokio::spawn(async move {
                let mut swarm_guard = swarm_clone.lock().await;
                swarm_guard.submit_task(Task::new(format!("task-{}", i), "compute")).unwrap();
            })
        })
        .collect();
    
    join_all(handles).await;
    
    let swarm_guard = swarm.lock().await;
    assert_eq!(swarm_guard.task_queue_size(), 10);
}

#[tokio::test]
async fn test_failure_recovery() {
    let mut swarm = Swarm::new(SwarmConfig::default());
    
    // Agent that will fail tasks
    let mut agent = Box::new(MockAgent::new("failing-agent")
        .with_process_result(Ok(TaskResult::failure("Simulated failure"))));
    
    agent.start().await.unwrap();
    swarm.register_agent(agent).unwrap();
    
    // Submit task with retry capability
    let task = Task::new("retry-task", "compute");
    swarm.submit_task(task).unwrap();
    
    let assignments = swarm.distribute_tasks().await.unwrap();
    assert_eq!(assignments.len(), 1);
}

#[cfg(feature = "std")]
use std::sync::Arc;