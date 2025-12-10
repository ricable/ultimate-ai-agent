//! Stdio transport for MCP (standard input/output communication)

use super::types::{MCPRequest, MCPResponse, MCPError};
use crate::{Result, JJError};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::sync::{Arc, Mutex};

/// Stdio transport for communicating with MCP servers via stdin/stdout
pub struct StdioTransport {
    process: Arc<Mutex<Option<StdioProcess>>>,
}

struct StdioProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StdioTransport {
    /// Create a new stdio transport
    pub fn new() -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
        }
    }

    /// Connect to an MCP server via stdio
    pub async fn connect(&self, command: &str, args: &[&str]) -> Result<()> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| JJError::MCPError(format!("Failed to spawn MCP server: {}", e)))?;

        let stdin = child.stdin.take().ok_or_else(|| {
            JJError::MCPError("Failed to get stdin handle".to_string())
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            JJError::MCPError("Failed to get stdout handle".to_string())
        })?;

        let process = StdioProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };

        let mut guard = self.process.lock().map_err(|e| {
            JJError::MCPError(format!("Failed to lock process: {}", e))
        })?;

        *guard = Some(process);

        Ok(())
    }

    /// Send a request and wait for response
    pub async fn send_request(&self, request: &MCPRequest) -> Result<MCPResponse> {
        let mut guard = self.process.lock().map_err(|e| {
            JJError::MCPError(format!("Failed to lock process: {}", e))
        })?;

        let process = guard.as_mut().ok_or_else(|| {
            JJError::MCPError("Not connected to MCP server. Call connect() first.".to_string())
        })?;

        // Serialize request to JSON
        let request_json = serde_json::to_string(request)
            .map_err(|e| JJError::SerializationError(format!("Failed to serialize request: {}", e)))?;

        // Write request to stdin
        writeln!(process.stdin, "{}", request_json)
            .map_err(|e| JJError::MCPError(format!("Failed to write request: {}", e)))?;

        process.stdin.flush()
            .map_err(|e| JJError::MCPError(format!("Failed to flush stdin: {}", e)))?;

        // Read response from stdout
        let mut response_line = String::new();
        process.stdout.read_line(&mut response_line)
            .map_err(|e| JJError::MCPError(format!("Failed to read response: {}", e)))?;

        // Parse response
        let response: MCPResponse = serde_json::from_str(&response_line)
            .map_err(|e| JJError::SerializationError(format!("Failed to parse response: {}", e)))?;

        Ok(response)
    }

    /// Disconnect from the MCP server
    pub async fn disconnect(&self) -> Result<()> {
        let mut guard = self.process.lock().map_err(|e| {
            JJError::MCPError(format!("Failed to lock process: {}", e))
        })?;

        if let Some(mut process) = guard.take() {
            process.child.kill()
                .map_err(|e| JJError::MCPError(format!("Failed to kill process: {}", e)))?;

            process.child.wait()
                .map_err(|e| JJError::MCPError(format!("Failed to wait for process: {}", e)))?;
        }

        Ok(())
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        // Try to cleanly disconnect
        if let Ok(mut guard) = self.process.lock() {
            if let Some(mut process) = guard.take() {
                let _ = process.child.kill();
                let _ = process.child.wait();
            }
        }
    }
}

/// Stdio-based MCP server for testing
pub struct StdioServer {
    running: Arc<Mutex<bool>>,
}

impl StdioServer {
    /// Create a new stdio server
    pub fn new() -> Self {
        Self {
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Run the server (blocks until stopped)
    pub async fn run<F>(&self, handler: F) -> Result<()>
    where
        F: Fn(MCPRequest) -> Result<MCPResponse> + Send + Sync + 'static,
    {
        {
            let mut guard = self.running.lock().map_err(|e| {
                JJError::MCPError(format!("Failed to lock running flag: {}", e))
            })?;
            *guard = true;
        }

        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = stdout.lock();

        loop {
            // Check if we should stop
            {
                let guard = self.running.lock().map_err(|e| {
                    JJError::MCPError(format!("Failed to lock running flag: {}", e))
                })?;
                if !*guard {
                    break;
                }
            }

            // Read request
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    // Parse request
                    let request: MCPRequest = match serde_json::from_str(&line) {
                        Ok(req) => req,
                        Err(e) => {
                            let error_response = MCPResponse::error(
                                "unknown".to_string(),
                                MCPError::parse_error(e.to_string()),
                            );
                            let json = serde_json::to_string(&error_response).unwrap();
                            writeln!(writer, "{}", json).ok();
                            writer.flush().ok();
                            continue;
                        }
                    };

                    // Handle request
                    let response = handler(request).unwrap_or_else(|e| {
                        MCPResponse::error(
                            "unknown".to_string(),
                            MCPError::internal_error(e.to_string()),
                        )
                    });

                    // Send response
                    let json = serde_json::to_string(&response)
                        .map_err(|e| JJError::SerializationError(e.to_string()))?;
                    writeln!(writer, "{}", json)
                        .map_err(|e| JJError::MCPError(format!("Failed to write response: {}", e)))?;
                    writer.flush()
                        .map_err(|e| JJError::MCPError(format!("Failed to flush stdout: {}", e)))?;
                }
                Err(e) => {
                    return Err(JJError::MCPError(format!("Failed to read line: {}", e)));
                }
            }
        }

        Ok(())
    }

    /// Stop the server
    pub async fn stop(&self) -> Result<()> {
        let mut guard = self.running.lock().map_err(|e| {
            JJError::MCPError(format!("Failed to lock running flag: {}", e))
        })?;
        *guard = false;
        Ok(())
    }
}

impl Default for StdioServer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdio_transport_creation() {
        let transport = StdioTransport::new();
        let guard = transport.process.lock().unwrap();
        assert!(guard.is_none());
    }

    #[test]
    fn test_stdio_server_creation() {
        let server = StdioServer::new();
        let guard = server.running.lock().unwrap();
        assert!(!*guard);
    }

    #[tokio::test]
    async fn test_server_stop() {
        let server = StdioServer::new();
        server.stop().await.unwrap();
        let guard = server.running.lock().unwrap();
        assert!(!*guard);
    }
}
