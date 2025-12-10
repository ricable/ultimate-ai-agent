//! Unit tests for wrapper module

use agentic_jujutsu::{JJWrapper, JJConfig};

#[test]
fn test_wrapper_with_config() {
    let config = JJConfig::default();
    let wrapper = JJWrapper::with_config(config.clone());

    assert_eq!(wrapper.config.jj_path, config.jj_path);
    assert_eq!(wrapper.operation_log.len(), 0);
}

#[test]
fn test_wrapper_default_config() {
    let wrapper = JJWrapper::with_config(JJConfig::default());

    assert_eq!(wrapper.config.jj_path, "jj");
    assert_eq!(wrapper.config.timeout_ms, 30000);
}

#[test]
fn test_wrapper_custom_config() {
    let config = JJConfig::default()
        .with_max_log_entries(500)
        .with_timeout(60000);

    let wrapper = JJWrapper::with_config(config);

    assert_eq!(wrapper.operation_log.max_entries, 500);
    assert_eq!(wrapper.config.timeout_ms, 60000);
}

#[test]
fn test_wrapper_operation_log_initialized() {
    let wrapper = JJWrapper::with_config(JJConfig::default());

    assert!(wrapper.operation_log.is_empty());
    assert_eq!(wrapper.operation_log.max_entries, 1000);
}

#[test]
fn test_wrapper_serialization() {
    let wrapper = JJWrapper::with_config(JJConfig::default());

    let json = serde_json::to_string(&wrapper).unwrap();
    assert!(json.contains("config"));
    assert!(json.contains("operation_log"));
}

#[test]
fn test_wrapper_deserialization() {
    let json = r#"{
        "config": {
            "jj_path": "jj",
            "repo_path": ".",
            "timeout_ms": 30000,
            "verbose": false,
            "max_log_entries": 1000,
            "enable_agentdb_sync": false
        },
        "operation_log": {
            "operations": [],
            "max_entries": 1000
        }
    }"#;

    let wrapper: JJWrapper = serde_json::from_str(json).unwrap();
    assert_eq!(wrapper.config.jj_path, "jj");
    assert_eq!(wrapper.operation_log.max_entries, 1000);
}

#[test]
fn test_wrapper_clone() {
    let wrapper1 = JJWrapper::with_config(JJConfig::default().with_verbose(true));
    let wrapper2 = wrapper1.clone();

    assert_eq!(wrapper1.config.verbose, wrapper2.config.verbose);
}

#[test]
fn test_wrapper_debug() {
    let wrapper = JJWrapper::with_config(JJConfig::default());
    let debug_str = format!("{:?}", wrapper);

    assert!(debug_str.contains("JJWrapper"));
}
