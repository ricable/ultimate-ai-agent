//! Unit tests for task distribution and management

use crate::task::*;
use crate::agent::Capability;

#[test]
fn test_task_creation() {
    let task = Task::new("task-1", "compute");
    
    assert_eq!(task.id.0, "task-1");
    assert_eq!(task.task_type, "compute");
    assert_eq!(task.priority, TaskPriority::Normal);
    assert_eq!(task.retry_count, 0);
    assert_eq!(task.max_retries, 3);
}

#[test]
fn test_task_builder_pattern() {
    let task = Task::new("task-1", "analysis")
        .with_priority(TaskPriority::High)
        .with_timeout(5000)
        .with_payload(TaskPayload::Text("Analyze this data".into()))
        .require_capability(Capability::new("analyze", "1.0"));
    
    assert_eq!(task.priority, TaskPriority::High);
    assert_eq!(task.timeout_ms, Some(5000));
    assert!(matches!(task.payload, TaskPayload::Text(_)));
    assert_eq!(task.required_capabilities.len(), 1);
}

#[test]
fn test_task_priority_ordering() {
    assert!(TaskPriority::Critical > TaskPriority::High);
    assert!(TaskPriority::High > TaskPriority::Normal);
    assert!(TaskPriority::Normal > TaskPriority::Low);
}

#[test]
fn test_task_retry_logic() {
    let mut task = Task::new("retry-task", "compute");
    
    assert!(task.can_retry());
    assert_eq!(task.retry_count, 0);
    
    // Increment retries
    task.increment_retry();
    assert_eq!(task.retry_count, 1);
    assert!(task.can_retry());
    
    // Max out retries
    task.increment_retry();
    task.increment_retry();
    assert_eq!(task.retry_count, 3);
    assert!(!task.can_retry());
}

#[test]
fn test_task_result_creation() {
    let success_result = TaskResult::success("Operation completed")
        .with_task_id(TaskId::new("task-1"))
        .with_execution_time(150);
    
    assert_eq!(success_result.status, TaskStatus::Completed);
    assert!(success_result.output.is_some());
    assert!(success_result.error.is_none());
    assert_eq!(success_result.execution_time_ms, 150);
    
    let failure_result = TaskResult::failure("Operation failed")
        .with_task_id(TaskId::new("task-2"));
    
    assert_eq!(failure_result.status, TaskStatus::Failed);
    assert!(failure_result.output.is_none());
    assert!(failure_result.error.is_some());
}

#[test]
fn test_task_payload_variants() {
    let empty_payload = TaskPayload::Empty;
    let text_payload = TaskPayload::Text("Hello".into());
    let binary_payload = TaskPayload::Binary(vec![1, 2, 3, 4]);
    let json_payload = TaskPayload::Json(r#"{"key": "value"}"#.into());
    
    // Verify we can create and match different payload types
    match empty_payload {
        TaskPayload::Empty => assert!(true),
        _ => panic!("Expected empty payload"),
    }
    
    match text_payload {
        TaskPayload::Text(s) => assert_eq!(s, "Hello"),
        _ => panic!("Expected text payload"),
    }
    
    match binary_payload {
        TaskPayload::Binary(bytes) => assert_eq!(bytes, vec![1, 2, 3, 4]),
        _ => panic!("Expected binary payload"),
    }
    
    match json_payload {
        TaskPayload::Json(s) => assert!(s.contains("key")),
        _ => panic!("Expected JSON payload"),
    }
}

#[test]
fn test_task_output_conversions() {
    let text_output: TaskOutput = "Result text".into();
    match text_output {
        TaskOutput::Text(s) => assert_eq!(s, "Result text"),
        _ => panic!("Expected text output"),
    }
    
    let string_output: TaskOutput = String::from("String result").into();
    match string_output {
        TaskOutput::Text(s) => assert_eq!(s, "String result"),
        _ => panic!("Expected text output"),
    }
    
    let binary_output: TaskOutput = vec![0u8, 1, 2, 3].into();
    match binary_output {
        TaskOutput::Binary(bytes) => assert_eq!(bytes.len(), 4),
        _ => panic!("Expected binary output"),
    }
}

#[test]
fn test_distribution_strategy_default() {
    let strategy = DistributionStrategy::default();
    assert_eq!(strategy, DistributionStrategy::LeastLoaded);
}

#[test]
fn test_task_status_transitions() {
    let statuses = vec![
        TaskStatus::Pending,
        TaskStatus::Assigned,
        TaskStatus::Running,
        TaskStatus::Completed,
        TaskStatus::Failed,
        TaskStatus::Cancelled,
        TaskStatus::TimedOut,
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

#[test]
fn test_task_id_display() {
    let task_id = TaskId::new("my-task-123");
    assert_eq!(format!("{}", task_id), "my-task-123");
}

#[test]
fn test_task_with_multiple_capabilities() {
    let task = Task::new("multi-cap-task", "complex")
        .require_capability(Capability::new("compute", "2.0"))
        .require_capability(Capability::new("analyze", "1.5"))
        .require_capability(Capability::new("synthesize", "3.0"));
    
    assert_eq!(task.required_capabilities.len(), 3);
    
    let cap_names: Vec<_> = task.required_capabilities.iter()
        .map(|c| c.name.as_str())
        .collect();
    
    assert!(cap_names.contains(&"compute"));
    assert!(cap_names.contains(&"analyze"));
    assert!(cap_names.contains(&"synthesize"));
}

#[test]
fn test_custom_task_payload() {
    #[derive(Debug, Clone)]
    struct CustomData {
        value: i32,
        name: String,
    }
    
    impl CustomPayload for CustomData {
        fn clone_box(&self) -> Box<dyn CustomPayload> {
            Box::new(self.clone())
        }
    }
    
    let custom_data = CustomData {
        value: 42,
        name: "Test".into(),
    };
    
    let task = Task::new("custom-task", "process")
        .with_payload(TaskPayload::Custom(Box::new(custom_data)));
    
    match &task.payload {
        TaskPayload::Custom(_) => assert!(true),
        _ => panic!("Expected custom payload"),
    }
}

#[test]
fn test_task_timeout_configuration() {
    let task_no_timeout = Task::new("task-1", "quick");
    assert_eq!(task_no_timeout.timeout_ms, None);
    
    let task_with_timeout = Task::new("task-2", "slow")
        .with_timeout(30000);
    assert_eq!(task_with_timeout.timeout_ms, Some(30000));
}

#[test]
fn test_task_result_with_different_statuses() {
    let completed = TaskResult {
        task_id: TaskId::new("task-1"),
        status: TaskStatus::Completed,
        output: Some(TaskOutput::Text("Done".into())),
        error: None,
        execution_time_ms: 100,
    };
    
    let failed = TaskResult {
        task_id: TaskId::new("task-2"),
        status: TaskStatus::Failed,
        output: None,
        error: Some("Error occurred".into()),
        execution_time_ms: 50,
    };
    
    let timed_out = TaskResult {
        task_id: TaskId::new("task-3"),
        status: TaskStatus::TimedOut,
        output: None,
        error: Some("Operation timed out".into()),
        execution_time_ms: 5000,
    };
    
    assert!(matches!(completed.status, TaskStatus::Completed));
    assert!(matches!(failed.status, TaskStatus::Failed));
    assert!(matches!(timed_out.status, TaskStatus::TimedOut));
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_task_creation_never_panics(
            id in "[a-zA-Z0-9-]{1,20}",
            task_type in "[a-zA-Z]{1,10}",
            priority in 0u8..4,
            timeout in 0u64..1000000,
            retry_count in 0u32..10,
            max_retries in 0u32..10,
        ) {
            let priority_enum = match priority {
                0 => TaskPriority::Low,
                1 => TaskPriority::Normal,
                2 => TaskPriority::High,
                _ => TaskPriority::Critical,
            };
            
            let mut task = Task::new(id, task_type)
                .with_priority(priority_enum)
                .with_timeout(timeout);
            
            task.retry_count = retry_count;
            task.max_retries = max_retries;
            
            // Ensure can_retry logic is consistent
            if retry_count < max_retries {
                assert!(task.can_retry());
            } else {
                assert!(!task.can_retry());
            }
        }
        
        #[test]
        fn test_task_result_consistency(
            task_id in "[a-zA-Z0-9-]{1,20}",
            success in any::<bool>(),
            execution_time in 0u64..1000000,
            output_text in "[a-zA-Z0-9 ]{0,100}",
            error_text in "[a-zA-Z0-9 ]{0,100}",
        ) {
            let result = if success {
                TaskResult::success(output_text.clone())
                    .with_task_id(TaskId::new(task_id))
                    .with_execution_time(execution_time)
            } else {
                TaskResult::failure(error_text.clone())
                    .with_task_id(TaskId::new(task_id))
                    .with_execution_time(execution_time)
            };
            
            // Verify consistency
            if success {
                assert_eq!(result.status, TaskStatus::Completed);
                assert!(result.output.is_some());
                assert!(result.error.is_none());
            } else {
                assert_eq!(result.status, TaskStatus::Failed);
                assert!(result.output.is_none());
                assert!(result.error.is_some());
            }
            
            assert_eq!(result.execution_time_ms, execution_time);
        }
    }
}