//! WASM-specific tests

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use agentic_jujutsu::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_wasm_config_creation() {
    let config = JJConfig::new();

    assert_eq!(config.jj_path, "jj");
    assert_eq!(config.repo_path, ".");
    assert_eq!(config.timeout_ms, 30000);
}

#[wasm_bindgen_test]
fn test_wasm_config_builder() {
    let config = JJConfig::new()
        .with_timeout(60000)
        .with_verbose(true);

    assert_eq!(config.timeout_ms, 60000);
    assert!(config.verbose);
}

#[wasm_bindgen_test]
fn test_wasm_operation_creation() {
    let op = JJOperation::new(
        "op_test".to_string(),
        "user@example.com".to_string(),
        "WASM test operation".to_string(),
    );

    assert_eq!(op.id, "op_test");
    assert_eq!(op.user, "user@example.com");
}

#[wasm_bindgen_test]
fn test_wasm_operation_log() {
    let mut log = JJOperationLog::new(10);

    for i in 0..5 {
        let op = JJOperation::new(
            format!("op_{}", i),
            "user@test.com".to_string(),
            format!("Operation {}", i),
        );
        log.add_operation(op);
    }

    assert_eq!(log.len(), 5);
}

#[wasm_bindgen_test]
fn test_wasm_conflict_management() {
    let mut conflict = JJConflict::new(
        "src/main.rs".to_string(),
        "merge conflict".to_string(),
        2,
    );

    assert!(!conflict.is_resolved());

    conflict.mark_resolved();

    assert!(conflict.is_resolved());
}

#[wasm_bindgen_test]
fn test_wasm_result_creation() {
    let success = JJResult::ok("Success!".to_string());
    let error = JJResult::err("Failed!".to_string(), 1);

    assert!(success.success);
    assert!(!error.success);
}

#[wasm_bindgen_test]
fn test_wasm_commit_creation() {
    let commit = JJCommit::new(
        "change123".to_string(),
        "commit456".to_string(),
        "wasm@test.com".to_string(),
        "WASM commit".to_string(),
    );

    assert_eq!(commit.change_id, "change123");
    assert!(!commit.is_working_copy);
}

#[wasm_bindgen_test]
fn test_wasm_branch_management() {
    let mut branch = JJBranch::new(
        "wasm-feature".to_string(),
        "target_commit".to_string(),
    );

    assert!(!branch.is_current);

    branch.set_current(true);

    assert!(branch.is_current);
}

#[wasm_bindgen_test]
fn test_wasm_wrapper_creation() {
    let result = JJWrapper::new();

    assert!(result.is_ok());

    let wrapper = result.unwrap();
    assert_eq!(wrapper.operation_log.len(), 0);
}

#[wasm_bindgen_test]
fn test_wasm_config_with_agentdb() {
    let config = JJConfig::new()
        .with_agentdb_sync(true);

    assert!(config.enable_agentdb_sync);
}

#[wasm_bindgen_test]
fn test_wasm_operation_timestamp() {
    let op = JJOperation::new(
        "id".to_string(),
        "user@test.com".to_string(),
        "desc".to_string(),
    );

    let iso = op.timestamp_iso();

    assert!(iso.contains('T'));
    assert!(!iso.is_empty());
}

#[wasm_bindgen_test]
fn test_wasm_operation_log_clear() {
    let mut log = JJOperationLog::new(10);

    log.add_operation(JJOperation::new("id".into(), "user".into(), "desc".into()));

    assert_eq!(log.len(), 1);

    log.clear();

    assert_eq!(log.len(), 0);
    assert!(log.is_empty());
}

#[wasm_bindgen_test]
fn test_wasm_error_handling() {
    let error = JJError::JJNotFound;

    let message = error.message();

    assert!(!message.is_empty());
    assert!(message.contains("jj command not found"));
}

#[wasm_bindgen_test]
fn test_wasm_serialization() {
    let config = JJConfig::new().with_verbose(true);

    let json = serde_json::to_string(&config).unwrap();

    assert!(json.contains("verbose"));
    assert!(json.contains("true"));
}

#[wasm_bindgen_test]
fn test_wasm_deserialization() {
    let json = r#"{
        "jj_path": "jj",
        "repo_path": ".",
        "timeout_ms": 30000,
        "verbose": false,
        "max_log_entries": 1000,
        "enable_agentdb_sync": false
    }"#;

    let config: JJConfig = serde_json::from_str(json).unwrap();

    assert_eq!(config.jj_path, "jj");
}
