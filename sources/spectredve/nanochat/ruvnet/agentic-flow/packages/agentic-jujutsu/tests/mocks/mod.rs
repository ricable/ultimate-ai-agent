//! Mock utilities and test data

pub mod jj_output_mocks;

pub use jj_output_mocks::*;

use agentic_jujutsu::{JJOperation, JJOperationLog, JJConfig, JJWrapper};

/// Create a mock JJWrapper with test configuration
pub fn mock_wrapper() -> JJWrapper {
    let config = JJConfig::default()
        .with_timeout(5000)
        .with_max_log_entries(50);

    JJWrapper::with_config(config)
}

/// Create a mock operation log with sample data
pub fn mock_operation_log(count: usize) -> JJOperationLog {
    let mut log = JJOperationLog::new(count * 2);

    for i in 0..count {
        let op = JJOperation::new(
            format!("op_{:08}", i),
            format!("user{}@test.com", i % 3),
            format!("Test operation {}", i),
        );
        log.add_operation(op);
    }

    log
}

/// Create a list of mock operations
pub fn mock_operations(count: usize) -> Vec<JJOperation> {
    (0..count)
        .map(|i| {
            JJOperation::new(
                format!("mock_op_{}", i),
                "mock@example.com".to_string(),
                format!("Mock operation {}", i),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_wrapper() {
        let wrapper = mock_wrapper();
        assert_eq!(wrapper.config.timeout_ms, 5000);
        assert_eq!(wrapper.operation_log.max_entries, 50);
    }

    #[test]
    fn test_mock_operation_log() {
        let log = mock_operation_log(10);
        assert_eq!(log.len(), 10);
    }

    #[test]
    fn test_mock_operations() {
        let ops = mock_operations(5);
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].id, "mock_op_0");
    }
}
