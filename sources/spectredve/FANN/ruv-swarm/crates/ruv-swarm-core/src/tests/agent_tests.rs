//! Unit tests for agent trait implementations

use crate::agent::*;
use crate::task::{Task, TaskResult, TaskPayload, Capability as TaskCapability};
use crate::error::Result;

#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(feature = "std")]
use tokio::sync::Mutex;

#[tokio::test]
async fn test_agent_spawn_and_shutdown() {
    let mut agent = MockAgent::new("test-1");
    assert_eq!(agent.status(), AgentStatus::Idle);
    
    agent.start().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Running);
    
    agent.shutdown().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Stopped);
}

#[tokio::test]
async fn test_agent_pause_and_resume() {
    let mut agent = MockAgent::new("test-2");
    
    agent.start().await.unwrap();
    assert_eq!(agent.status(), AgentStatus::Running);
    
    agent.pause().unwrap();
    assert_eq!(agent.status(), AgentStatus::Paused);
    
    agent.resume().unwrap();
    assert_eq!(agent.status(), AgentStatus::Running);
}

#[tokio::test]
async fn test_agent_capabilities() {
    let agent = MockAgent::new("test-3")
        .with_capabilities(vec![
            Capability::new("neural-processing", "2.0"),
            Capability::new("data-analysis", "1.5"),
            Capability::new("pattern-recognition", "3.0"),
        ]);
    
    let capabilities = agent.capabilities();
    assert_eq!(capabilities.len(), 3);
    assert!(capabilities.iter().any(|c| c.name == "neural-processing"));
}

#[tokio::test]
async fn test_agent_can_handle_task() {
    let agent = MockAgent::new("test-4")
        .with_capabilities(vec![
            Capability::new("compute", "1.0"),
            Capability::new("analyze", "1.0"),
        ]);
    
    // Task requiring matching capabilities
    let task1 = Task::new("task-1", "analysis")
        .require_capability(Capability::new("compute", "1.0"));
    assert!(agent.can_handle(&task1));
    
    // Task requiring non-matching capability
    let task2 = Task::new("task-2", "synthesis")
        .require_capability(Capability::new("synthesize", "1.0"));
    assert!(!agent.can_handle(&task2));
}

#[tokio::test]
async fn test_agent_process_task_success() {
    let mut agent = MockAgent::new("test-5")
        .with_process_result(Ok(TaskResult::success("Processed successfully")));
    
    agent.start().await.unwrap();
    
    let task = Task::new("task-1", "compute")
        .with_payload(TaskPayload::Text("Process this data".into()));
    
    let result = agent.process(task).await.unwrap();
    assert_eq!(result.status, TaskStatus::Completed);
    assert!(result.output.is_some());
}

#[tokio::test]
async fn test_agent_process_task_failure() {
    let mut agent = MockAgent::new("test-6")
        .with_process_result(Ok(TaskResult::failure("Processing failed")));
    
    agent.start().await.unwrap();
    
    let task = Task::new("task-1", "compute");
    let result = agent.process(task).await.unwrap();
    
    assert_eq!(result.status, TaskStatus::Failed);
    assert!(result.error.is_some());
}

#[tokio::test]
async fn test_agent_metrics() {
    let agent = MockAgent::new("test-7");
    let metrics = agent.metrics();
    
    assert_eq!(metrics.tasks_completed, 0);
    assert_eq!(metrics.tasks_failed, 0);
    assert_eq!(metrics.total_processing_time_ms, 0);
}

#[tokio::test]
async fn test_multiple_agents_concurrent_operation() {
    use futures::future::join_all;
    
    let mut agents: Vec<MockAgent> = (0..5)
        .map(|i| {
            MockAgent::new(format!("agent-{}", i))
                .with_process_result(Ok(TaskResult::success(format!("Result from agent-{}", i))))
        })
        .collect();
    
    // Start all agents concurrently
    let start_futures: Vec<_> = agents.iter_mut()
        .map(|agent| agent.start())
        .collect();
    
    let results = join_all(start_futures).await;
    assert!(results.iter().all(|r| r.is_ok()));
    
    // Verify all agents are running
    assert!(agents.iter().all(|agent| agent.status() == AgentStatus::Running));
}

#[tokio::test]
async fn test_agent_id_uniqueness() {
    let agent1 = MockAgent::new("unique-1");
    let agent2 = MockAgent::new("unique-2");
    let agent3 = MockAgent::new("unique-1"); // Same ID as agent1
    
    assert_ne!(agent1.id(), agent2.id());
    assert_eq!(agent1.id(), agent3.id());
}

#[test]
fn test_capability_equality() {
    let cap1 = Capability::new("neural", "1.0");
    let cap2 = Capability::new("neural", "1.0");
    let cap3 = Capability::new("neural", "2.0");
    let cap4 = Capability::new("compute", "1.0");
    
    assert_eq!(cap1, cap2);
    assert_ne!(cap1, cap3);
    assert_ne!(cap1, cap4);
}

#[test]
fn test_agent_status_transitions() {
    let statuses = vec![
        AgentStatus::Idle,
        AgentStatus::Running,
        AgentStatus::Paused,
        AgentStatus::Stopped,
        AgentStatus::Error,
    ];
    
    // Verify all statuses are distinct
    for (i, status1) in statuses.iter().enumerate() {
        for (j, status2) in statuses.iter().enumerate() {
            if i == j {
                assert_eq!(status1, status2);
            } else {
                assert_ne!(status1, status2);
            }
        }
    }
}

#[cfg(feature = "std")]
#[tokio::test]
async fn test_agent_concurrent_task_processing() {
    use tokio::time::{sleep, Duration};
    
    struct ConcurrentAgent {
        id: AgentId,
        status: Arc<Mutex<AgentStatus>>,
        processing_count: Arc<Mutex<usize>>,
        max_concurrent: usize,
    }
    
    impl ConcurrentAgent {
        fn new(id: impl Into<String>, max_concurrent: usize) -> Self {
            ConcurrentAgent {
                id: AgentId::new(id),
                status: Arc::new(Mutex::new(AgentStatus::Idle)),
                processing_count: Arc::new(Mutex::new(0)),
                max_concurrent,
            }
        }
    }
    
    impl Agent for ConcurrentAgent {
        fn id(&self) -> &AgentId {
            &self.id
        }
        
        fn status(&self) -> AgentStatus {
            futures::executor::block_on(async {
                *self.status.lock().await
            })
        }
        
        fn capabilities(&self) -> &[Capability] {
            &[]
        }
        
        fn process<'a>(&'a mut self, _task: Task) -> Pin<Box<dyn Future<Output = Result<TaskResult>> + Send + 'a>> {
            let processing_count = self.processing_count.clone();
            let max_concurrent = self.max_concurrent;
            
            Box::pin(async move {
                let mut count = processing_count.lock().await;
                if *count >= max_concurrent {
                    return Ok(TaskResult::failure("Max concurrent tasks exceeded"));
                }
                
                *count += 1;
                drop(count);
                
                // Simulate processing
                sleep(Duration::from_millis(100)).await;
                
                let mut count = processing_count.lock().await;
                *count -= 1;
                
                Ok(TaskResult::success("Processed"))
            })
        }
        
        fn start<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            let status = self.status.clone();
            Box::pin(async move {
                *status.lock().await = AgentStatus::Running;
                Ok(())
            })
        }
        
        fn shutdown<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            let status = self.status.clone();
            Box::pin(async move {
                *status.lock().await = AgentStatus::Stopped;
                Ok(())
            })
        }
        
        fn pause(&mut self) -> Result<()> {
            futures::executor::block_on(async {
                *self.status.lock().await = AgentStatus::Paused;
            });
            Ok(())
        }
        
        fn resume(&mut self) -> Result<()> {
            futures::executor::block_on(async {
                *self.status.lock().await = AgentStatus::Running;
            });
            Ok(())
        }
    }
    
    let mut agent = ConcurrentAgent::new("concurrent-test", 3);
    agent.start().await.unwrap();
    
    // Try to process 5 tasks concurrently (should fail after 3)
    let tasks: Vec<_> = (0..5)
        .map(|i| Task::new(format!("task-{}", i), "compute"))
        .collect();
    
    let mut handles = vec![];
    for task in tasks {
        let future = agent.process(task);
        handles.push(tokio::spawn(async move {
            future.await
        }));
    }
    
    let results: Vec<_> = join_all(handles).await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    
    // First 3 should succeed, last 2 should fail
    let successes = results.iter().filter(|r| {
        matches!(r, Ok(result) if result.status == TaskStatus::Completed)
    }).count();
    
    let failures = results.iter().filter(|r| {
        matches!(r, Ok(result) if result.status == TaskStatus::Failed)
    }).count();
    
    assert!(successes <= 3);
    assert!(failures >= 2);
}