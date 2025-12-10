//! WASM implementation with simulated command execution
//!
//! ## Architecture Note
//!
//! This module provides **simulated** jj command execution for WASM/browser environments
//! where direct CLI execution is not possible. For real jj operations, use:
//!
//! - **Node.js/CLI**: The `bin/cli.js` executes the real `jj` binary if installed
//! - **Native Rust**: The `native.rs` module uses async-process for real command execution
//! - **Browser/WASM**: This module returns realistic mock data for demos and testing
//!
//! To use real jujutsu operations, install the jj CLI:
//! - Cargo: `cargo install --git https://github.com/martinvonz/jj jj-cli`
//! - Homebrew: `brew install jj`
//! - From source: https://github.com/martinvonz/jj#installation

#![cfg(target_arch = "wasm32")]

use crate::error::{JJError, Result};
use std::time::Duration;
use wasm_bindgen::prelude::*;

/// Execute a jj command in WASM environment (simulated for browser contexts)
///
/// **Important**: This returns simulated data for browser/WASM environments.
/// For real jj operations, use Node.js CLI or native Rust implementation.
///
/// Note: jj_path and timeout are ignored in WASM as we simulate commands
pub async fn execute_jj_command(
    _jj_path: &str,
    args: &[&str],
    _command_timeout: Duration,
) -> Result<String> {
    // Log to browser console with clear indication this is simulated
    web_sys::console::warn_1(
        &format!("WASM Simulation: jj {} (Browser environment - not real jj execution)", args.join(" ")).into()
    );

    // Return simulated responses for browser/WASM demonstrations
    let response = match args.first() {
        Some(&"version") | Some(&"--version") => {
            "jj 0.12.0 (WASM simulation)".to_string()
        }
        Some(&"status") => {
            "The working copy is clean\nWorking copy : qpvuntsm 12345678 (empty) (no description set)\nParent commit: zzzzzzzz 00000000 (empty) (no description set)".to_string()
        }
        Some(&"log") => {
            "@  qpvuntsm test@example.com 2024-01-01 12:00:00.000 12345678\n│  (empty) (no description set)\n◉  zzzzzzzz root() 00000000".to_string()
        }
        Some(&"branch") => {
            if args.len() > 1 && args[1] == "list" {
                "main: abc123def456\nfeature: def789abc012".to_string()
            } else {
                "Branch operation simulated".to_string()
            }
        }
        Some(&"diff") => {
            "+++ b/file.txt\n--- a/file.txt\n+Added line\n-Removed line".to_string()
        }
        Some(&"resolve") => {
            if args.contains(&"--list") {
                "No conflicts found".to_string()
            } else {
                "Conflicts resolved".to_string()
            }
        }
        Some(&"describe") => {
            "Working copy now at: qpvuntsm 12345678 Updated description".to_string()
        }
        Some(&"new") => {
            "Working copy now at: abcdefgh 87654321 (empty) (no description set)".to_string()
        }
        Some(&"edit") => {
            "Working copy now at: requested commit".to_string()
        }
        Some(&"abandon") => {
            "Abandoned commit".to_string()
        }
        Some(&"squash") => {
            "Squashed commits".to_string()
        }
        Some(&"rebase") => {
            "Rebased commits".to_string()
        }
        Some(&"undo") => {
            "Undid operation".to_string()
        }
        Some(&"restore") => {
            "Restored files".to_string()
        }
        Some(&"git") => {
            if args.len() > 1 && args[1] == "fetch" {
                "Fetched from remote".to_string()
            } else if args.len() > 1 && args[1] == "push" {
                "Pushed to remote".to_string()
            } else {
                "Git operation simulated".to_string()
            }
        }
        _ => {
            return Err(JJError::CommandFailed(format!(
                "Unknown command in WASM simulation: {}",
                args.join(" ")
            )));
        }
    };

    Ok(response)
}

// No stub needed - this module is only compiled for WASM targets

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"agentic-jujutsu WASM module initialized".into());
    web_sys::console::log_1(&"Note: Browser WASM environment uses simulated jj commands".into());
    web_sys::console::log_1(&"For real jj operations, use Node.js CLI or install jj binary".into());
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_wasm_version() {
        let result = execute_jj_command(&["version"]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("WASM simulation"));
    }

    #[wasm_bindgen_test]
    async fn test_wasm_status() {
        let result = execute_jj_command(&["status"]).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Working copy"));
    }

    #[wasm_bindgen_test]
    async fn test_wasm_unknown_command() {
        let result = execute_jj_command(&["unknown_command"]).await;
        assert!(result.is_err());
    }
}
