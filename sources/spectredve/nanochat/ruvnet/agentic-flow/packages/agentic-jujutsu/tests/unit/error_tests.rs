//! Unit tests for error module

use agentic_jujutsu::JJError;

#[test]
fn test_jj_not_found_error() {
    let err = JJError::JJNotFound;
    let msg = err.to_string();

    assert!(msg.contains("jj command not found"));
    assert!(msg.contains("https://github.com/jj-vcs/jj"));
}

#[test]
fn test_command_failed_error() {
    let err = JJError::CommandFailed("execution failed".to_string());
    let msg = err.to_string();

    assert!(msg.contains("jj command failed"));
    assert!(msg.contains("execution failed"));
}

#[test]
fn test_parse_error() {
    let err = JJError::ParseError("invalid format".to_string());
    assert!(err.to_string().contains("invalid format"));
}

#[test]
fn test_operation_not_found() {
    let err = JJError::OperationNotFound("op_123".to_string());
    assert!(err.to_string().contains("op_123"));
}

#[test]
fn test_conflict_resolution_failed() {
    let err = JJError::ConflictResolutionFailed("merge failed".to_string());
    assert!(err.to_string().contains("merge failed"));
}

#[test]
fn test_invalid_config() {
    let err = JJError::InvalidConfig("bad config".to_string());
    assert!(err.to_string().contains("Invalid configuration"));
}

#[test]
fn test_io_error() {
    let err = JJError::IoError("file not found".to_string());
    assert!(err.to_string().contains("I/O error"));
}

#[test]
fn test_serialization_error() {
    let err = JJError::SerializationError("json error".to_string());
    assert!(err.to_string().contains("Serialization error"));
}

#[test]
fn test_unknown_error() {
    let err = JJError::Unknown("something went wrong".to_string());
    assert!(err.to_string().contains("Unknown error"));
}

#[test]
fn test_error_message_method() {
    let err = JJError::JJNotFound;
    let msg = err.message();

    assert!(!msg.is_empty());
    assert_eq!(msg, err.to_string());
}

#[test]
fn test_error_is_recoverable_command_failed() {
    let err = JJError::CommandFailed("test".into());
    assert!(err.is_recoverable());
}

#[test]
fn test_error_is_recoverable_conflict() {
    let err = JJError::ConflictResolutionFailed("test".into());
    assert!(err.is_recoverable());
}

#[test]
fn test_error_not_recoverable_jj_not_found() {
    let err = JJError::JJNotFound;
    assert!(!err.is_recoverable());
}

#[test]
fn test_error_not_recoverable_parse_error() {
    let err = JJError::ParseError("test".into());
    assert!(!err.is_recoverable());
}

#[test]
fn test_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let jj_err: JJError = io_err.into();

    match jj_err {
        JJError::IoError(msg) => assert!(msg.contains("file not found")),
        _ => panic!("Expected IoError variant"),
    }
}

#[test]
fn test_error_from_serde_error() {
    let json = "{ invalid json }";
    let result: Result<serde_json::Value, serde_json::Error> = serde_json::from_str(json);
    let serde_err = result.unwrap_err();
    let jj_err: JJError = serde_err.into();

    match jj_err {
        JJError::SerializationError(_) => {},
        _ => panic!("Expected SerializationError variant"),
    }
}

#[test]
fn test_error_debug() {
    let err = JJError::JJNotFound;
    let debug_str = format!("{:?}", err);

    assert!(debug_str.contains("JJNotFound"));
}
