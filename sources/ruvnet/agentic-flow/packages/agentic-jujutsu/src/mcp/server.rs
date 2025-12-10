//! MCP server implementation (facade for stdio and SSE servers)

use super::{StdioServer, SSEServer, TransportType};
use super::sse::SSEServerConfig;
use super::types::{MCPRequest, MCPResponse, MCPError, MCPCapabilities};
use crate::{Result, JJError};
use std::sync::Arc;

/// MCP server configuration
#[derive(Debug, Clone)]
pub struct MCPServerConfig {
    /// Transport type
    pub transport: TransportType,
    /// SSE server config (for SSE transport)
    pub sse_config: Option<SSEServerConfig>,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for MCPServerConfig {
    fn default() -> Self {
        Self {
            transport: TransportType::Stdio,
            sse_config: None,
            verbose: false,
        }
    }
}

impl MCPServerConfig {
    /// Create config for stdio transport
    pub fn stdio() -> Self {
        Self {
            transport: TransportType::Stdio,
            ..Default::default()
        }
    }

    /// Create config for SSE transport
    pub fn sse(config: SSEServerConfig) -> Self {
        Self {
            transport: TransportType::SSE,
            sse_config: Some(config),
            ..Default::default()
        }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// MCP server that can use either stdio or SSE transport
pub struct MCPServer {
    config: MCPServerConfig,
    stdio_server: Option<Arc<StdioServer>>,
    sse_server: Option<Arc<SSEServer>>,
}

impl MCPServer {
    /// Create a new MCP server
    pub fn new(config: MCPServerConfig) -> Self {
        let (stdio_server, sse_server) = match config.transport {
            TransportType::Stdio => (Some(Arc::new(StdioServer::new())), None),
            TransportType::SSE => {
                let sse_config = config.sse_config.clone().unwrap_or_default();
                (None, Some(Arc::new(SSEServer::new(sse_config))))
            }
        };

        Self {
            config,
            stdio_server,
            sse_server,
        }
    }

    /// Register a request handler
    pub async fn register_handler<F>(&self, method: String, handler: F) -> Result<()>
    where
        F: Fn(MCPRequest) -> Result<MCPResponse> + Send + Sync + 'static + Clone,
    {
        match self.config.transport {
            TransportType::Stdio => {
                // For stdio, handlers are registered when run() is called
                Ok(())
            }
            TransportType::SSE => {
                if let Some(server) = &self.sse_server {
                    server.register_handler(method, handler).await?;
                }
                Ok(())
            }
        }
    }

    /// Start the server
    pub async fn run<F>(&self, handler: F) -> Result<()>
    where
        F: Fn(MCPRequest) -> Result<MCPResponse> + Send + Sync + 'static + Clone,
    {
        match self.config.transport {
            TransportType::Stdio => {
                if let Some(server) = &self.stdio_server {
                    if self.config.verbose {
                        eprintln!("[mcp-server] Starting stdio server");
                    }
                    server.run(handler).await?;
                }
            }
            TransportType::SSE => {
                if let Some(server) = &self.sse_server {
                    if self.config.verbose {
                        eprintln!("[mcp-server] Starting SSE server");
                    }
                    server.start().await?;
                }
            }
        }

        Ok(())
    }

    /// Stop the server
    pub async fn stop(&self) -> Result<()> {
        match self.config.transport {
            TransportType::Stdio => {
                if let Some(server) = &self.stdio_server {
                    server.stop().await?;
                }
            }
            TransportType::SSE => {
                // SSE server stop is handled by dropping the server
            }
        }

        Ok(())
    }

    /// Get server capabilities
    pub fn capabilities(&self) -> MCPCapabilities {
        MCPCapabilities::default()
    }
}

/// Default request handler that handles common MCP methods
pub fn default_handler(req: MCPRequest) -> Result<MCPResponse> {
    match req.method.as_str() {
        "initialize" => {
            let capabilities = MCPCapabilities::default();
            Ok(MCPResponse::success(
                req.id,
                serde_json::to_value(capabilities)
                    .map_err(|e| JJError::SerializationError(e.to_string()))?,
            ))
        }
        "capabilities" => {
            let capabilities = MCPCapabilities::default();
            Ok(MCPResponse::success(
                req.id,
                serde_json::to_value(capabilities)
                    .map_err(|e| JJError::SerializationError(e.to_string()))?,
            ))
        }
        _ => Ok(MCPResponse::error(
            req.id,
            MCPError::method_not_found(req.method),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config() {
        let config = MCPServerConfig::stdio();
        assert!(config.transport.is_stdio());

        let sse_config = SSEServerConfig::default();
        let config = MCPServerConfig::sse(sse_config);
        assert!(config.transport.is_sse());
    }

    #[test]
    fn test_server_creation() {
        let config = MCPServerConfig::stdio();
        let server = MCPServer::new(config);
        assert!(server.stdio_server.is_some());
        assert!(server.sse_server.is_none());

        let config = MCPServerConfig::sse(SSEServerConfig::default());
        let server = MCPServer::new(config);
        assert!(server.stdio_server.is_none());
        assert!(server.sse_server.is_some());
    }

    #[test]
    fn test_default_handler_initialize() {
        let req = MCPRequest::new(
            "1".to_string(),
            "initialize".to_string(),
            None,
        );

        let response = default_handler(req).unwrap();
        assert!(response.is_success());
        assert!(response.result.is_some());
    }

    #[test]
    fn test_default_handler_unknown_method() {
        let req = MCPRequest::new(
            "1".to_string(),
            "unknown_method".to_string(),
            None,
        );

        let response = default_handler(req).unwrap();
        assert!(response.is_error());
        assert_eq!(response.error.unwrap().code, -32601); // Method not found
    }

    #[test]
    fn test_capabilities() {
        let config = MCPServerConfig::stdio();
        let server = MCPServer::new(config);
        let caps = server.capabilities();

        assert!(caps.methods.contains(&"agentdb_pattern_store".to_string()));
        assert!(caps.methods.contains(&"agentdb_pattern_search".to_string()));
    }
}
