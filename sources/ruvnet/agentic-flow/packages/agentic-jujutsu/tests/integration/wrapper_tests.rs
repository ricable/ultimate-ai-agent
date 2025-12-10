//! Integration tests for JJWrapper

use agentic_jujutsu::{JJWrapper, JJConfig, JJOperation, JJOperationLog};

#[test]
fn test_wrapper_full_initialization() {
    let config = JJConfig::default()
        .with_jj_path("/usr/bin/jj".to_string())
        .with_repo_path("/tmp/repo".to_string())
        .with_timeout(45000)
        .with_verbose(true)
        .with_max_log_entries(500)
        .with_agentdb_sync(true);

    let wrapper = JJWrapper::with_config(config);

    assert_eq!(wrapper.config.jj_path, "/usr/bin/jj");
    assert_eq!(wrapper.config.repo_path, "/tmp/repo");
    assert_eq!(wrapper.config.timeout_ms, 45000);
    assert!(wrapper.config.verbose);
    assert_eq!(wrapper.operation_log.max_entries, 500);
    assert!(wrapper.config.enable_agentdb_sync);
}

#[test]
fn test_wrapper_operation_log_management() {
    let mut wrapper = JJWrapper::with_config(JJConfig::default().with_max_log_entries(3));

    // Add operations
    for i in 0..5 {
        let op = JJOperation::new(
            format!("op_{}", i),
            format!("user{}@example.com", i),
            format!("Operation {}", i),
        );
        wrapper.operation_log.add_operation(op);
    }

    // Should only keep last 3
    assert_eq!(wrapper.operation_log.len(), 3);

    // Verify oldest operations were removed
    assert!(wrapper.operation_log.get_by_id("op_0").is_none());
    assert!(wrapper.operation_log.get_by_id("op_1").is_none());

    // Verify newest operations exist
    assert!(wrapper.operation_log.get_by_id("op_2").is_some());
    assert!(wrapper.operation_log.get_by_id("op_3").is_some());
    assert!(wrapper.operation_log.get_by_id("op_4").is_some());
}

#[test]
fn test_wrapper_serialization_round_trip() {
    let original = JJWrapper::with_config(
        JJConfig::default()
            .with_timeout(55000)
            .with_max_log_entries(750)
    );

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: JJWrapper = serde_json::from_str(&json).unwrap();

    assert_eq!(original.config.timeout_ms, deserialized.config.timeout_ms);
    assert_eq!(original.config.max_log_entries, deserialized.config.max_log_entries);
}

#[test]
fn test_wrapper_config_modifications() {
    let config1 = JJConfig::default();
    let mut wrapper = JJWrapper::with_config(config1);

    // Modify config through wrapper
    wrapper.config.timeout_ms = 90000;
    wrapper.config.verbose = true;

    assert_eq!(wrapper.config.timeout_ms, 90000);
    assert!(wrapper.config.verbose);
}

#[test]
fn test_wrapper_operation_log_queries() {
    let mut wrapper = JJWrapper::with_config(JJConfig::default());

    // Add test operations
    for i in 0..10 {
        let mut op = JJOperation::new(
            format!("op_{}", i),
            "user@example.com".to_string(),
            format!("Description {}", i),
        );

        if i % 2 == 0 {
            op.op_type = agentic_jujutsu::OperationType::Commit;
        } else {
            op.op_type = agentic_jujutsu::OperationType::Rebase;
        }

        wrapper.operation_log.add_operation(op);
    }

    // Test filtering
    let commits = wrapper.operation_log.filter_by_type(agentic_jujutsu::OperationType::Commit);
    assert_eq!(commits.len(), 5);

    let rebases = wrapper.operation_log.filter_by_type(agentic_jujutsu::OperationType::Rebase);
    assert_eq!(rebases.len(), 5);

    // Test recent operations
    let recent = wrapper.operation_log.recent(3);
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0].id, "op_7");
    assert_eq!(recent[1].id, "op_8");
    assert_eq!(recent[2].id, "op_9");
}

#[cfg(feature = "native")]
#[tokio::test]
async fn test_wrapper_concurrent_safety() {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let wrapper = Arc::new(Mutex::new(
        JJWrapper::with_config(JJConfig::default())
    ));

    let mut handles = vec![];

    // Spawn multiple concurrent tasks
    for i in 0..10 {
        let wrapper_clone = Arc::clone(&wrapper);
        let handle = tokio::spawn(async move {
            let mut w = wrapper_clone.lock().await;
            let op = JJOperation::new(
                format!("concurrent_op_{}", i),
                "user@example.com".to_string(),
                format!("Concurrent operation {}", i),
            );
            w.operation_log.add_operation(op);
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    let final_wrapper = wrapper.lock().await;
    assert_eq!(final_wrapper.operation_log.len(), 10);
}

#[test]
fn test_wrapper_clear_operations() {
    let mut wrapper = JJWrapper::with_config(JJConfig::default());

    // Add operations
    for i in 0..5 {
        wrapper.operation_log.add_operation(JJOperation::new(
            format!("op_{}", i),
            "user@example.com".to_string(),
            "desc".to_string(),
        ));
    }

    assert_eq!(wrapper.operation_log.len(), 5);

    // Clear
    wrapper.operation_log.clear();

    assert_eq!(wrapper.operation_log.len(), 0);
    assert!(wrapper.operation_log.is_empty());
}
