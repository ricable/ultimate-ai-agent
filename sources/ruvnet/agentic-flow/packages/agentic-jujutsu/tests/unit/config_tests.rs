//! Unit tests for JJConfig

use agentic_jujutsu::JJConfig;

#[test]
fn test_default_configuration() {
    let config = JJConfig::default();

    assert_eq!(config.jj_path, "jj");
    assert_eq!(config.repo_path, ".");
    assert_eq!(config.timeout_ms, 30000);
    assert!(!config.verbose);
    assert_eq!(config.max_log_entries, 1000);
    assert!(!config.enable_agentdb_sync);
}

#[test]
fn test_new_constructor() {
    let config = JJConfig::new();

    assert_eq!(config.jj_path, "jj");
    assert_eq!(config.timeout_ms, 30000);
}

#[test]
fn test_builder_pattern_jj_path() {
    let config = JJConfig::default()
        .with_jj_path("/usr/local/bin/jj".to_string());

    assert_eq!(config.jj_path, "/usr/local/bin/jj");
}

#[test]
fn test_builder_pattern_repo_path() {
    let config = JJConfig::default()
        .with_repo_path("/path/to/repo".to_string());

    assert_eq!(config.repo_path, "/path/to/repo");
}

#[test]
fn test_builder_pattern_timeout() {
    let config = JJConfig::default()
        .with_timeout(60000);

    assert_eq!(config.timeout_ms, 60000);
}

#[test]
fn test_builder_pattern_verbose() {
    let config = JJConfig::default()
        .with_verbose(true);

    assert!(config.verbose);
}

#[test]
fn test_builder_pattern_max_log_entries() {
    let config = JJConfig::default()
        .with_max_log_entries(500);

    assert_eq!(config.max_log_entries, 500);
}

#[test]
fn test_builder_pattern_agentdb_sync() {
    let config = JJConfig::default()
        .with_agentdb_sync(true);

    assert!(config.enable_agentdb_sync);
}

#[test]
fn test_builder_pattern_chaining() {
    let config = JJConfig::default()
        .with_jj_path("/custom/jj".to_string())
        .with_repo_path("/custom/repo".to_string())
        .with_timeout(120000)
        .with_verbose(true)
        .with_max_log_entries(2000)
        .with_agentdb_sync(true);

    assert_eq!(config.jj_path, "/custom/jj");
    assert_eq!(config.repo_path, "/custom/repo");
    assert_eq!(config.timeout_ms, 120000);
    assert!(config.verbose);
    assert_eq!(config.max_log_entries, 2000);
    assert!(config.enable_agentdb_sync);
}

#[test]
fn test_config_serialization() {
    let config = JJConfig::default()
        .with_verbose(true)
        .with_timeout(45000);

    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("verbose"));
    assert!(json.contains("45000"));
}

#[test]
fn test_config_deserialization() {
    let json = r#"{
        "jj_path": "jj",
        "repo_path": ".",
        "timeout_ms": 30000,
        "verbose": true,
        "max_log_entries": 1000,
        "enable_agentdb_sync": false
    }"#;

    let config: JJConfig = serde_json::from_str(json).unwrap();
    assert!(config.verbose);
    assert_eq!(config.timeout_ms, 30000);
}

#[test]
fn test_config_round_trip_serialization() {
    let original = JJConfig::default()
        .with_timeout(55000)
        .with_max_log_entries(750);

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: JJConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(original.timeout_ms, deserialized.timeout_ms);
    assert_eq!(original.max_log_entries, deserialized.max_log_entries);
}

#[test]
fn test_config_clone() {
    let config1 = JJConfig::default().with_verbose(true);
    let config2 = config1.clone();

    assert_eq!(config1.verbose, config2.verbose);
    assert_eq!(config1.jj_path, config2.jj_path);
}

#[test]
fn test_config_debug() {
    let config = JJConfig::default();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("JJConfig"));
    assert!(debug_str.contains("jj_path"));
}

#[test]
fn test_zero_timeout() {
    let config = JJConfig::default().with_timeout(0);
    assert_eq!(config.timeout_ms, 0);
}

#[test]
fn test_large_max_log_entries() {
    let config = JJConfig::default().with_max_log_entries(1_000_000);
    assert_eq!(config.max_log_entries, 1_000_000);
}

#[test]
fn test_empty_paths() {
    let config = JJConfig::default()
        .with_jj_path(String::new())
        .with_repo_path(String::new());

    assert!(config.jj_path.is_empty());
    assert!(config.repo_path.is_empty());
}
