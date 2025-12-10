//! Native implementation using async-process for command execution
//! This module is only compiled for native targets (not WASM)

#![cfg(not(target_arch = "wasm32"))]

use crate::error::{JJError, Result};
use std::time::Duration;
use async_process::{Command, Stdio};
use tokio::time::timeout;

/// Execute a jj command natively with timeout support
pub async fn execute_jj_command(
    jj_path: &str,
    args: &[&str],
    command_timeout: Duration,
) -> Result<String> {
    // Build the command
    let mut cmd = Command::new(jj_path);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Execute with timeout
    let output = timeout(command_timeout, cmd.output())
        .await
        .map_err(|_| JJError::CommandFailed("Command timeout exceeded".to_string()))?
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                JJError::JJNotFound
            } else {
                JJError::IoError(e.to_string())
            }
        })?;

    // Check exit status
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(JJError::CommandFailed(stderr));
    }

    // Return stdout
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jj_not_found() {
        let result = execute_jj_command(
            "nonexistent_jj_binary",
            &["--version"],
            Duration::from_secs(5),
        )
        .await;

        assert!(result.is_err());
        if let Err(JJError::JJNotFound) = result {
            // Expected
        } else {
            panic!("Expected JJNotFound error");
        }
    }

    #[tokio::test]
    async fn test_timeout() {
        // This test assumes 'sleep' command exists
        let result = execute_jj_command("sleep", &["10"], Duration::from_millis(100)).await;

        assert!(result.is_err());
        if let Err(JJError::CommandFailed(msg)) = result {
            assert!(msg.contains("timeout"));
        } else {
            panic!("Expected CommandFailed with timeout");
        }
    }

    #[tokio::test]
    async fn test_echo_command() {
        // Test with a simple command that exists on most systems
        let result = execute_jj_command("echo", &["test"], Duration::from_secs(5)).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().trim(), "test");
    }

    #[tokio::test]
    async fn test_failed_command() {
        // Test with a command that will fail
        let result =
            execute_jj_command("ls", &["nonexistent_dir_xyz"], Duration::from_secs(5)).await;

        assert!(result.is_err());
    }
}
