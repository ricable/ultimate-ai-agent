//! MCP server implementation

use reasoningbank_core::{EngineConfig};
use reasoningbank_learning::{AsyncLearnerV2, LearningConfig};
use reasoningbank_storage::{AsyncStorage, StorageConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug};

use crate::tools::{
    McpTool, McpResource,
    StorePatternTool, RetrievePatternsTool, AnalyzePatternsTool, OptimizeStrategyTool,
    PatternsResource, MetricsResource,
};
use crate::{Result, McpError};

/// Configuration for MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Storage configuration
    pub storage: StorageConfig,

    /// Engine configuration
    pub engine: EngineConfig,

    /// Learning configuration
    pub learning: LearningConfig,

    /// Server name
    pub server_name: String,

    /// Server version
    pub server_version: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            engine: EngineConfig::default(),
            learning: LearningConfig::default(),
            server_name: "reasoningbank-mcp".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// MCP server for reasoning operations
pub struct McpServer {
    config: McpConfig,
    tools: HashMap<String, Arc<dyn McpTool>>,
    resources: HashMap<String, Arc<dyn McpResource>>,
    storage: AsyncStorage,
    learner: AsyncLearnerV2,
}

impl McpServer {
    /// Create a new MCP server
    pub async fn new(config: McpConfig) -> Result<Self> {
        info!("Initializing MCP server: {} v{}", config.server_name, config.server_version);

        // Initialize async components
        let storage = AsyncStorage::new(config.storage.clone())?;
        let learner = AsyncLearnerV2::new(
            config.engine.clone(),
            config.storage.clone(),
            config.learning.clone(),
        ).await?;

        // Register tools
        let mut tools: HashMap<String, Arc<dyn McpTool>> = HashMap::new();

        let store_tool = Arc::new(StorePatternTool::new(learner.clone()));
        tools.insert(store_tool.name().to_string(), store_tool);

        let retrieve_tool = Arc::new(RetrievePatternsTool::new(learner.clone()));
        tools.insert(retrieve_tool.name().to_string(), retrieve_tool);

        let analyze_tool = Arc::new(AnalyzePatternsTool::new(storage.clone()));
        tools.insert(analyze_tool.name().to_string(), analyze_tool);

        let optimize_tool = Arc::new(OptimizeStrategyTool::new(storage.clone()));
        tools.insert(optimize_tool.name().to_string(), optimize_tool);

        // Register resources
        let mut resources: HashMap<String, Arc<dyn McpResource>> = HashMap::new();

        let patterns_resource = Arc::new(PatternsResource::new(storage.clone()));
        resources.insert(patterns_resource.uri_pattern().to_string(), patterns_resource);

        let metrics_resource = Arc::new(MetricsResource::new(learner.clone()));
        resources.insert(metrics_resource.uri_pattern().to_string(), metrics_resource);

        info!("MCP server initialized with {} tools and {} resources",
              tools.len(), resources.len());

        Ok(Self {
            config,
            tools,
            resources,
            storage,
            learner,
        })
    }

    /// Create with default configuration
    pub async fn default() -> Result<Self> {
        Self::new(McpConfig::default()).await
    }

    /// List available tools
    pub fn list_tools(&self) -> Vec<ToolInfo> {
        self.tools.iter()
            .map(|(name, tool)| ToolInfo {
                name: name.clone(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            })
            .collect()
    }

    /// List available resources
    pub fn list_resources(&self) -> Vec<ResourceInfo> {
        self.resources.iter()
            .map(|(pattern, resource)| ResourceInfo {
                uri_pattern: pattern.clone(),
                description: resource.description().to_string(),
            })
            .collect()
    }

    /// Execute a tool
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        debug!("Executing tool: {}", tool_name);

        let tool = self.tools.get(tool_name)
            .ok_or_else(|| McpError::ToolNotFound(tool_name.to_string()))?;

        tool.execute(params).await
    }

    /// Get a resource
    pub async fn get_resource(&self, uri: &str) -> Result<serde_json::Value> {
        debug!("Getting resource: {}", uri);

        // Find matching resource by pattern
        for (pattern, resource) in &self.resources {
            if uri.starts_with(pattern.split('{').next().unwrap()) {
                return resource.get(uri).await;
            }
        }

        Err(McpError::ResourceNotFound(uri.to_string()))
    }

    /// Get server information
    pub fn get_server_info(&self) -> ServerInfo {
        ServerInfo {
            name: self.config.server_name.clone(),
            version: self.config.server_version.clone(),
            tools_count: self.tools.len(),
            resources_count: self.resources.len(),
        }
    }
}

/// Tool information for MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Resource information for MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub uri_pattern: String,
    pub description: String,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub tools_count: usize,
    pub resources_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn test_config() -> McpConfig {
        let temp_db = std::env::temp_dir().join(format!("test_mcp_{}.db", Uuid::new_v4()));
        McpConfig {
            storage: StorageConfig {
                database_path: temp_db,
                max_connections: 1,
                enable_wal: false,
                cache_size_kb: 4096,
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_server_creation() {
        let server = McpServer::new(test_config()).await.unwrap();
        let info = server.get_server_info();

        assert_eq!(info.name, "reasoningbank-mcp");
        assert!(info.tools_count > 0);
        assert!(info.resources_count > 0);
    }

    #[tokio::test]
    async fn test_list_tools() {
        let server = McpServer::new(test_config()).await.unwrap();
        let tools = server.list_tools();

        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "reasoning_store"));
        assert!(tools.iter().any(|t| t.name == "reasoning_retrieve"));
    }

    #[tokio::test]
    async fn test_list_resources() {
        let server = McpServer::new(test_config()).await.unwrap();
        let resources = server.list_resources();

        assert!(!resources.is_empty());
    }
}
