//! Unit tests for operations module

use agentic_jujutsu::{JJOperation, JJOperationLog, OperationType};

// OperationType tests
#[test]
fn test_operation_type_equality() {
    assert_eq!(OperationType::Commit, OperationType::Commit);
    assert_eq!(OperationType::Rebase, OperationType::Rebase);
    assert_ne!(OperationType::Commit, OperationType::Rebase);
}

#[test]
fn test_operation_type_serialization() {
    let op_type = OperationType::Commit;
    let json = serde_json::to_string(&op_type).unwrap();
    let deserialized: OperationType = serde_json::from_str(&json).unwrap();

    assert_eq!(op_type, deserialized);
}

// JJOperation tests
#[test]
fn test_operation_new() {
    let op = JJOperation::new(
        "op_123".to_string(),
        "user@example.com".to_string(),
        "test operation".to_string(),
    );

    assert_eq!(op.id, "op_123");
    assert_eq!(op.user, "user@example.com");
    assert_eq!(op.description, "test operation");
}

#[test]
fn test_operation_type_string() {
    let op = JJOperation::new(
        "id".to_string(),
        "user".to_string(),
        "desc".to_string(),
    );

    let type_str = op.operation_type();
    assert!(type_str.contains("Unknown") || !type_str.is_empty());
}

#[test]
fn test_operation_timestamp_iso() {
    let op = JJOperation::new(
        "id".to_string(),
        "user".to_string(),
        "desc".to_string(),
    );

    let iso = op.timestamp_iso();
    assert!(iso.contains("T"));
    assert!(iso.len() > 10);
}

#[test]
fn test_operation_serialization() {
    let op = JJOperation::new(
        "op_456".to_string(),
        "test@test.com".to_string(),
        "operation desc".to_string(),
    );

    let json = serde_json::to_string(&op).unwrap();
    assert!(json.contains("op_456"));
    assert!(json.contains("test@test.com"));
}

#[test]
fn test_operation_deserialization() {
    let json = r#"{
        "id": "test_id",
        "op_type": "Commit",
        "user": "user@example.com",
        "timestamp": "2024-01-01T00:00:00Z",
        "description": "test desc",
        "parents": []
    }"#;

    let op: JJOperation = serde_json::from_str(json).unwrap();
    assert_eq!(op.id, "test_id");
    assert_eq!(op.user, "user@example.com");
}

#[test]
fn test_operation_parse_from_log() {
    let log_line = "@  qpvuntsm test@example.com 2024-01-01 describe commit";
    let op = JJOperation::parse_from_log(log_line);

    assert!(op.is_some());
    let op = op.unwrap();
    assert_eq!(op.id, "qpvuntsm");
    assert_eq!(op.user, "test@example.com");
}

#[test]
fn test_operation_parse_from_log_with_marker() {
    let log_line = "◉  sqpuoqvx user@test.com 2024-01-01 rebase operation";
    let op = JJOperation::parse_from_log(log_line);

    assert!(op.is_some());
    let op = op.unwrap();
    assert_eq!(op.id, "sqpuoqvx");
}

#[test]
fn test_operation_parse_invalid_line() {
    let invalid_line = "invalid";
    let op = JJOperation::parse_from_log(invalid_line);

    assert!(op.is_none());
}

#[test]
fn test_operation_parse_empty_line() {
    let empty_line = "";
    let op = JJOperation::parse_from_log(empty_line);

    assert!(op.is_none());
}

// JJOperationLog tests
#[test]
fn test_operation_log_new() {
    let log = JJOperationLog::new(100);

    assert_eq!(log.max_entries, 100);
    assert_eq!(log.len(), 0);
    assert!(log.is_empty());
}

#[test]
fn test_operation_log_add() {
    let mut log = JJOperationLog::new(10);
    let op = JJOperation::new("id1".into(), "user1".into(), "desc1".into());

    log.add_operation(op);

    assert_eq!(log.len(), 1);
    assert!(!log.is_empty());
}

#[test]
fn test_operation_log_max_entries() {
    let mut log = JJOperationLog::new(3);

    for i in 0..5 {
        let op = JJOperation::new(
            format!("id{}", i),
            format!("user{}", i),
            format!("desc{}", i),
        );
        log.add_operation(op);
    }

    assert_eq!(log.len(), 3); // Should only keep last 3

    // First two operations should be removed
    assert!(log.get_by_id("id0").is_none());
    assert!(log.get_by_id("id1").is_none());

    // Last three should exist
    assert!(log.get_by_id("id2").is_some());
    assert!(log.get_by_id("id3").is_some());
    assert!(log.get_by_id("id4").is_some());
}

#[test]
fn test_operation_log_clear() {
    let mut log = JJOperationLog::new(10);

    log.add_operation(JJOperation::new("id1".into(), "u1".into(), "d1".into()));
    log.add_operation(JJOperation::new("id2".into(), "u2".into(), "d2".into()));

    assert_eq!(log.len(), 2);

    log.clear();

    assert_eq!(log.len(), 0);
    assert!(log.is_empty());
}

#[test]
fn test_operation_log_get_by_id() {
    let mut log = JJOperationLog::new(10);

    log.add_operation(JJOperation::new("find_me".into(), "user".into(), "desc".into()));

    let op = log.get_by_id("find_me");
    assert!(op.is_some());
    assert_eq!(op.unwrap().id, "find_me");

    let missing = log.get_by_id("not_there");
    assert!(missing.is_none());
}

#[test]
fn test_operation_log_recent() {
    let mut log = JJOperationLog::new(10);

    for i in 0..5 {
        log.add_operation(JJOperation::new(
            format!("id{}", i),
            "user".into(),
            "desc".into(),
        ));
    }

    let recent = log.recent(2);
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].id, "id3");
    assert_eq!(recent[1].id, "id4");
}

#[test]
fn test_operation_log_recent_more_than_available() {
    let mut log = JJOperationLog::new(10);

    log.add_operation(JJOperation::new("id1".into(), "u".into(), "d".into()));
    log.add_operation(JJOperation::new("id2".into(), "u".into(), "d".into()));

    let recent = log.recent(10);
    assert_eq!(recent.len(), 2); // Should return all available
}

#[test]
fn test_operation_log_filter_by_type() {
    let mut log = JJOperationLog::new(10);

    let mut op1 = JJOperation::new("id1".into(), "u".into(), "d1".into());
    op1.op_type = OperationType::Commit;

    let mut op2 = JJOperation::new("id2".into(), "u".into(), "d2".into());
    op2.op_type = OperationType::Rebase;

    let mut op3 = JJOperation::new("id3".into(), "u".into(), "d3".into());
    op3.op_type = OperationType::Commit;

    log.add_operation(op1);
    log.add_operation(op2);
    log.add_operation(op3);

    let commits = log.filter_by_type(OperationType::Commit);
    assert_eq!(commits.len(), 2);

    let rebases = log.filter_by_type(OperationType::Rebase);
    assert_eq!(rebases.len(), 1);
}

#[test]
fn test_operation_log_serialization() {
    let log = JJOperationLog::new(50);

    let json = serde_json::to_string(&log).unwrap();
    assert!(json.contains("max_entries"));
    assert!(json.contains("50"));
}

#[test]
fn test_operation_clone() {
    let op1 = JJOperation::new("id".into(), "user".into(), "desc".into());
    let op2 = op1.clone();

    assert_eq!(op1.id, op2.id);
    assert_eq!(op1.user, op2.user);
}

#[test]
fn test_operation_equality() {
    let op1 = JJOperation::new("same".into(), "same".into(), "same".into());
    let op2 = JJOperation::new("same".into(), "same".into(), "same".into());

    // Timestamps will differ
    assert_eq!(op1.id, op2.id);
    assert_eq!(op1.user, op2.user);
}
