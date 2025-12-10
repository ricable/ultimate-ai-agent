//! Comprehensive wrapper tests

use agentic_jujutsu::{JJConfig, JJWrapper};

#[tokio::test]
async fn test_wrapper_creation() {
    let wrapper = JJWrapper::new().expect("Failed to create wrapper");
    let config = wrapper.get_config();

    assert_eq!(config.jj_path(), "jj");
    assert_eq!(config.timeout_ms, 30000);
}

#[tokio::test]
async fn test_wrapper_with_custom_config() {
    let config = JJConfig::default()
        .with_verbose(true)
        .with_timeout(60000)
        .with_max_log_entries(500);

    let wrapper = JJWrapper::with_config(config).expect("Failed to create wrapper");
    let retrieved_config = wrapper.get_config();

    assert!(retrieved_config.verbose);
    assert_eq!(retrieved_config.timeout_ms, 60000);
    assert_eq!(retrieved_config.max_log_entries, 500);
}

#[tokio::test]
async fn test_wrapper_stats() {
    let wrapper = JJWrapper::new().expect("Failed to create wrapper");
    let stats = wrapper.get_stats();

    // Should be valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&stats).expect("Failed to parse stats JSON");

    assert!(parsed.get("total_operations").is_some());
    assert!(parsed.get("avg_duration_ms").is_some());
    assert!(parsed.get("success_rate").is_some());
}

#[tokio::test]
async fn test_operation_logging() {
    let wrapper = JJWrapper::new().expect("Failed to create wrapper");

    // Clear any previous operations
    wrapper.clear_log();

    // Initially should have no operations
    let ops = wrapper
        .get_operations(10)
        .expect("Failed to get operations");
    assert_eq!(ops.len(), 0);
}

#[tokio::test]
async fn test_get_user_operations() {
    let wrapper = JJWrapper::new().expect("Failed to create wrapper");
    wrapper.clear_log();

    let user_ops = wrapper
        .get_user_operations(10)
        .expect("Failed to get user operations");
    assert_eq!(user_ops.len(), 0);
}

#[cfg(feature = "native")]
mod native_tests {
    use super::*;
    use agentic_jujutsu::native::execute_jj_command;
    use std::time::Duration;

    #[tokio::test]
    async fn test_command_not_found() {
        let result = execute_jj_command(
            "nonexistent_binary_xyz",
            &["--help"],
            Duration::from_secs(5),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_command_timeout() {
        // Test timeout with sleep command (if available)
        let result = execute_jj_command("sleep", &["10"], Duration::from_millis(100)).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_successful_command() {
        // Test with echo command
        let result = execute_jj_command("echo", &["test"], Duration::from_secs(5)).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().trim(), "test");
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use agentic_jujutsu::wasm::execute_jj_command;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_wasm_status() {
        let result = execute_jj_command(&["status"]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Working copy"));
    }

    #[wasm_bindgen_test]
    async fn test_wasm_version() {
        let result = execute_jj_command(&["version"]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("WASM simulation"));
    }

    #[wasm_bindgen_test]
    async fn test_wasm_unknown_command() {
        let result = execute_jj_command(&["unknown"]).await;
        assert!(result.is_err());
    }
}
