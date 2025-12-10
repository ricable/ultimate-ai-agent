//! Unit tests for types module

use agentic_jujutsu::{JJConflict, JJResult, JJCommit, JJBranch};

// JJConflict tests
#[test]
fn test_conflict_creation() {
    let conflict = JJConflict::new(
        "src/main.rs".to_string(),
        "merge conflict in main".to_string(),
        2,
    );

    assert_eq!(conflict.file_path, "src/main.rs");
    assert_eq!(conflict.description, "merge conflict in main");
    assert_eq!(conflict.sides, 2);
    assert!(!conflict.resolved);
}

#[test]
fn test_conflict_mark_resolved() {
    let mut conflict = JJConflict::new(
        "test.rs".to_string(),
        "conflict".to_string(),
        2,
    );

    assert!(!conflict.is_resolved());
    conflict.mark_resolved();
    assert!(conflict.is_resolved());
}

#[test]
fn test_conflict_serialization() {
    let conflict = JJConflict::new(
        "file.rs".to_string(),
        "conflict desc".to_string(),
        3,
    );

    let json = serde_json::to_string(&conflict).unwrap();
    assert!(json.contains("file.rs"));
    assert!(json.contains("conflict desc"));
}

#[test]
fn test_conflict_deserialization() {
    let json = r#"{
        "file_path": "test.rs",
        "description": "test conflict",
        "sides": 2,
        "resolved": false
    }"#;

    let conflict: JJConflict = serde_json::from_str(json).unwrap();
    assert_eq!(conflict.file_path, "test.rs");
    assert_eq!(conflict.sides, 2);
}

#[test]
fn test_conflict_round_trip() {
    let original = JJConflict::new(
        "original.rs".to_string(),
        "desc".to_string(),
        4,
    );

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: JJConflict = serde_json::from_str(&json).unwrap();

    assert_eq!(original, deserialized);
}

// JJResult tests
#[test]
fn test_result_ok() {
    let result = JJResult::ok("Operation succeeded".to_string());

    assert!(result.success);
    assert_eq!(result.message, "Operation succeeded");
    assert_eq!(result.exit_code, 0);
}

#[test]
fn test_result_err() {
    let result = JJResult::err("Operation failed".to_string(), 1);

    assert!(!result.success);
    assert_eq!(result.message, "Operation failed");
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_result_new() {
    let result = JJResult::new(true, "Custom result".to_string(), 0);

    assert!(result.success);
    assert_eq!(result.message, "Custom result");
}

#[test]
fn test_result_with_stdout_stderr() {
    let mut result = JJResult::ok("Success".to_string());
    result.stdout = "output data".to_string();
    result.stderr = "warning message".to_string();

    assert_eq!(result.stdout, "output data");
    assert_eq!(result.stderr, "warning message");
}

#[test]
fn test_result_serialization() {
    let result = JJResult::ok("test".to_string());
    let json = serde_json::to_string(&result).unwrap();

    assert!(json.contains("success"));
    assert!(json.contains("test"));
}

// JJCommit tests
#[test]
fn test_commit_creation() {
    let commit = JJCommit::new(
        "change_abc123".to_string(),
        "commit_def456".to_string(),
        "user@example.com".to_string(),
        "Initial commit".to_string(),
    );

    assert_eq!(commit.change_id, "change_abc123");
    assert_eq!(commit.commit_id, "commit_def456");
    assert_eq!(commit.author, "user@example.com");
    assert_eq!(commit.message, "Initial commit");
    assert!(!commit.is_working_copy);
}

#[test]
fn test_commit_timestamp_iso() {
    let commit = JJCommit::new(
        "ch1".to_string(),
        "co1".to_string(),
        "author".to_string(),
        "msg".to_string(),
    );

    let iso = commit.timestamp_iso();
    assert!(iso.contains("T")); // ISO format contains 'T'
    assert!(iso.len() > 10); // Reasonable length for ISO timestamp
}

#[test]
fn test_commit_working_copy_flag() {
    let mut commit = JJCommit::new(
        "ch1".to_string(),
        "co1".to_string(),
        "author".to_string(),
        "msg".to_string(),
    );

    assert!(!commit.is_working_copy);
    commit.is_working_copy = true;
    assert!(commit.is_working_copy);
}

#[test]
fn test_commit_serialization() {
    let commit = JJCommit::new(
        "ch1".to_string(),
        "co1".to_string(),
        "author".to_string(),
        "message".to_string(),
    );

    let json = serde_json::to_string(&commit).unwrap();
    assert!(json.contains("change_id"));
    assert!(json.contains("ch1"));
}

#[test]
fn test_commit_equality() {
    let commit1 = JJCommit::new(
        "same".to_string(),
        "same".to_string(),
        "same".to_string(),
        "same".to_string(),
    );

    let commit2 = JJCommit::new(
        "same".to_string(),
        "same".to_string(),
        "same".to_string(),
        "same".to_string(),
    );

    // Timestamps will differ, so this tests partial equality
    assert_eq!(commit1.change_id, commit2.change_id);
    assert_eq!(commit1.commit_id, commit2.commit_id);
}

// JJBranch tests
#[test]
fn test_branch_creation() {
    let branch = JJBranch::new(
        "main".to_string(),
        "commit_abc123".to_string(),
    );

    assert_eq!(branch.name, "main");
    assert_eq!(branch.target, "commit_abc123");
    assert!(!branch.is_current);
    assert!(branch.remote.is_none());
}

#[test]
fn test_branch_set_current() {
    let mut branch = JJBranch::new("feature".to_string(), "target".to_string());

    assert!(!branch.is_current);
    branch.set_current(true);
    assert!(branch.is_current);
    branch.set_current(false);
    assert!(!branch.is_current);
}

#[test]
fn test_branch_set_remote() {
    let mut branch = JJBranch::new("local".to_string(), "target".to_string());

    assert!(branch.remote.is_none());
    branch.set_remote("origin/local".to_string());
    assert_eq!(branch.remote, Some("origin/local".to_string()));
}

#[test]
fn test_branch_serialization() {
    let mut branch = JJBranch::new("test".to_string(), "target".to_string());
    branch.set_current(true);
    branch.set_remote("origin/test".to_string());

    let json = serde_json::to_string(&branch).unwrap();
    assert!(json.contains("test"));
    assert!(json.contains("origin/test"));
}

#[test]
fn test_branch_round_trip() {
    let original = JJBranch::new("branch".to_string(), "commit".to_string());

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: JJBranch = serde_json::from_str(&json).unwrap();

    assert_eq!(original, deserialized);
}

#[test]
fn test_branch_clone() {
    let branch1 = JJBranch::new("b1".to_string(), "t1".to_string());
    let branch2 = branch1.clone();

    assert_eq!(branch1.name, branch2.name);
    assert_eq!(branch1.target, branch2.target);
}

#[test]
fn test_conflict_multiple_sides() {
    let conflict = JJConflict::new(
        "file.rs".to_string(),
        "3-way conflict".to_string(),
        3,
    );

    assert_eq!(conflict.sides, 3);
}

#[test]
fn test_result_different_exit_codes() {
    let codes = vec![0, 1, 2, 127, 255];

    for code in codes {
        let result = JJResult::err("error".to_string(), code);
        assert_eq!(result.exit_code, code);
    }
}
