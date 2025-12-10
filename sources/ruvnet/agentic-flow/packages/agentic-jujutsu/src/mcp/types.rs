//! MCP protocol types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MCP protocol version
pub const MCP_VERSION: &str = "2024-11-05";

/// MCP request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID for tracking responses
    pub id: String,
    /// Method to call
    pub method: String,
    /// Method parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl MCPRequest {
    /// Create a new MCP request
    pub fn new(id: String, method: String, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method,
            params,
        }
    }

    /// Create request for agentdb_pattern_store
    pub fn pattern_store(id: String, episode: serde_json::Value) -> Self {
        Self::new(id, "agentdb_pattern_store".to_string(), Some(episode))
    }

    /// Create request for agentdb_pattern_search
    pub fn pattern_search(id: String, task: String, k: usize) -> Self {
        Self::new(
            id,
            "agentdb_pattern_search".to_string(),
            Some(serde_json::json!({
                "task": task,
                "k": k
            })),
        )
    }

    /// Create request for agentdb_pattern_stats
    pub fn pattern_stats(id: String, task: String, k: usize) -> Self {
        Self::new(
            id,
            "agentdb_pattern_stats".to_string(),
            Some(serde_json::json!({
                "task": task,
                "k": k
            })),
        )
    }
}

/// MCP response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID this response is for
    pub id: String,
    /// Response result (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPError>,
}

impl MCPResponse {
    /// Create a successful response
    pub fn success(id: String, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: String, error: MCPError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Check if this is an error response
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Check if this is a success response
    pub fn is_success(&self) -> bool {
        self.result.is_some()
    }
}

/// MCP error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl MCPError {
    /// Create a new MCP error
    pub fn new(code: i32, message: String) -> Self {
        Self {
            code,
            message,
            data: None,
        }
    }

    /// Create error with additional data
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Parse error (-32700)
    pub fn parse_error(message: String) -> Self {
        Self::new(-32700, message)
    }

    /// Invalid request (-32600)
    pub fn invalid_request(message: String) -> Self {
        Self::new(-32600, message)
    }

    /// Method not found (-32601)
    pub fn method_not_found(method: String) -> Self {
        Self::new(-32601, format!("Method not found: {}", method))
    }

    /// Invalid params (-32602)
    pub fn invalid_params(message: String) -> Self {
        Self::new(-32602, message)
    }

    /// Internal error (-32603)
    pub fn internal_error(message: String) -> Self {
        Self::new(-32603, message)
    }
}

/// MCP method names
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MCPMethod {
    /// Initialize MCP connection
    Initialize,
    /// Store pattern in AgentDB
    AgentDBPatternStore,
    /// Search for similar patterns
    AgentDBPatternSearch,
    /// Get pattern statistics
    AgentDBPatternStats,
    /// Get AgentDB statistics
    AgentDBStats,
    /// Clear AgentDB cache
    AgentDBClearCache,
    /// Custom method
    Custom(String),
}

impl MCPMethod {
    /// Convert to method string
    pub fn as_str(&self) -> &str {
        match self {
            Self::Initialize => "initialize",
            Self::AgentDBPatternStore => "agentdb_pattern_store",
            Self::AgentDBPatternSearch => "agentdb_pattern_search",
            Self::AgentDBPatternStats => "agentdb_pattern_stats",
            Self::AgentDBStats => "agentdb_stats",
            Self::AgentDBClearCache => "agentdb_clear_cache",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Parse method from string
    pub fn from_str(s: &str) -> Self {
        match s {
            "initialize" => Self::Initialize,
            "agentdb_pattern_store" => Self::AgentDBPatternStore,
            "agentdb_pattern_search" => Self::AgentDBPatternSearch,
            "agentdb_pattern_stats" => Self::AgentDBPatternStats,
            "agentdb_stats" => Self::AgentDBStats,
            "agentdb_clear_cache" => Self::AgentDBClearCache,
            s => Self::Custom(s.to_string()),
        }
    }
}

/// MCP server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPCapabilities {
    /// Supported methods
    pub methods: Vec<String>,
    /// Server version
    pub version: String,
    /// Additional capabilities
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for MCPCapabilities {
    fn default() -> Self {
        Self {
            methods: vec![
                "agentdb_pattern_store".to_string(),
                "agentdb_pattern_search".to_string(),
                "agentdb_pattern_stats".to_string(),
                "agentdb_stats".to_string(),
                "agentdb_clear_cache".to_string(),
            ],
            version: MCP_VERSION.to_string(),
            extra: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_request_creation() {
        let req = MCPRequest::new(
            "1".to_string(),
            "test_method".to_string(),
            Some(serde_json::json!({"key": "value"})),
        );

        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.id, "1");
        assert_eq!(req.method, "test_method");
        assert!(req.params.is_some());
    }

    #[test]
    fn test_mcp_response_success() {
        let resp = MCPResponse::success("1".to_string(), serde_json::json!({"result": "ok"}));

        assert!(resp.is_success());
        assert!(!resp.is_error());
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_mcp_response_error() {
        let error = MCPError::method_not_found("test".to_string());
        let resp = MCPResponse::error("1".to_string(), error);

        assert!(!resp.is_success());
        assert!(resp.is_error());
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_mcp_method_conversion() {
        assert_eq!(MCPMethod::Initialize.as_str(), "initialize");
        assert_eq!(
            MCPMethod::AgentDBPatternStore.as_str(),
            "agentdb_pattern_store"
        );
        assert_eq!(
            MCPMethod::from_str("initialize"),
            MCPMethod::Initialize
        );
    }

    #[test]
    fn test_pattern_store_request() {
        let episode = serde_json::json!({
            "sessionId": "test",
            "task": "test task",
            "success": true
        });

        let req = MCPRequest::pattern_store("1".to_string(), episode.clone());

        assert_eq!(req.method, "agentdb_pattern_store");
        assert_eq!(req.params, Some(episode));
    }

    #[test]
    fn test_pattern_search_request() {
        let req = MCPRequest::pattern_search("1".to_string(), "test task".to_string(), 5);

        assert_eq!(req.method, "agentdb_pattern_search");
        assert!(req.params.is_some());

        let params = req.params.unwrap();
        assert_eq!(params["task"], "test task");
        assert_eq!(params["k"], 5);
    }
}
