//! AgentDB synchronization for operation logs
//!
//! This module provides integration with AgentDB for storing and querying
//! jj operation history, enabling AI agents to learn from past operations.

use crate::{JJError, JJOperation, Result};
#[cfg(not(target_arch = "wasm32"))]
use crate::mcp::{MCPClient, MCPClientConfig};
use serde::{Deserialize, Serialize};

/// Episode data structure for AgentDB storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDBEpisode {
    /// Session identifier
    pub session_id: String,
    /// Task description
    pub task: String,
    /// Agent identifier
    pub agent_id: String,
    /// Input context
    pub input: Option<String>,
    /// Output or result
    pub output: Option<String>,
    /// Self-critique or reflection
    pub critique: Option<String>,
    /// Success indicator
    pub success: bool,
    /// Reward score (0.0 to 1.0)
    pub reward: f64,
    /// Latency in milliseconds
    pub latency_ms: Option<u64>,
    /// Token usage count
    pub tokens_used: Option<u64>,
    /// Associated JJ operation
    pub operation: Option<JJOperation>,
    /// Unix timestamp
    pub timestamp: i64,
}

impl AgentDBEpisode {
    /// Create a new episode from a JJ operation
    pub fn from_operation(op: &JJOperation, session_id: String, agent_id: String) -> Self {
        Self {
            session_id,
            task: op.command.clone(),
            agent_id,
            input: None,
            output: None,
            critique: None,
            success: true,
            reward: 1.0,
            latency_ms: None,
            tokens_used: None,
            operation: Some(op.clone()),
            timestamp: chrono::DateTime::parse_from_rfc3339(&op.timestamp)
                .ok()
                .map(|dt| dt.timestamp())
                .unwrap_or_else(|| chrono::Utc::now().timestamp()),
        }
    }

    /// Set input context
    pub fn with_input(mut self, input: String) -> Self {
        self.input = Some(input);
        self
    }

    /// Set output result
    pub fn with_output(mut self, output: String) -> Self {
        self.output = Some(output);
        self
    }

    /// Set critique/reflection
    pub fn with_critique(mut self, critique: String) -> Self {
        self.critique = Some(critique);
        self
    }

    /// Set success and reward
    pub fn with_success(mut self, success: bool, reward: f64) -> Self {
        self.success = success;
        self.reward = reward.clamp(0.0, 1.0);
        self
    }

    /// Set performance metrics
    pub fn with_metrics(mut self, latency_ms: u64, tokens_used: u64) -> Self {
        self.latency_ms = Some(latency_ms);
        self.tokens_used = Some(tokens_used);
        self
    }
}

/// AgentDB synchronization manager
pub struct AgentDBSync {
    /// Whether sync is enabled
    enabled: bool,
    /// Base URL for AgentDB API (if using remote)
    api_url: Option<String>,
    /// MCP client for AgentDB communication (native only)
    #[cfg(not(target_arch = "wasm32"))]
    mcp_client: Option<MCPClient>,
}

impl AgentDBSync {
    /// Create a new AgentDB sync manager
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            api_url: None,
            #[cfg(not(target_arch = "wasm32"))]
            mcp_client: None,
        }
    }

    /// Create with MCP client for real AgentDB communication (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn with_mcp(enabled: bool, mcp_config: MCPClientConfig) -> Result<Self> {
        let mcp_client = if enabled {
            Some(MCPClient::new(mcp_config).await?)
        } else {
            None
        };

        Ok(Self {
            enabled,
            api_url: None,
            mcp_client,
        })
    }

    /// Create with custom API URL
    pub fn with_api_url(mut self, url: String) -> Self {
        self.api_url = Some(url);
        self
    }

    /// Sync a single operation to AgentDB
    pub async fn sync_operation(
        &self,
        op: &JJOperation,
        session_id: &str,
        agent_id: &str,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let episode =
            AgentDBEpisode::from_operation(op, session_id.to_string(), agent_id.to_string());
        self.store_episode(&episode).await
    }

    /// Store an episode in AgentDB
    pub async fn store_episode(&self, episode: &AgentDBEpisode) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // If MCP client is available, use it for real AgentDB communication (native only)
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(client) = &self.mcp_client {
                let episode_value = serde_json::to_value(episode)
                    .map_err(|e| JJError::SerializationError(e.to_string()))?;

                client.store_pattern(episode_value).await?;

                #[cfg(feature = "native")]
                println!("[agentdb-sync] ✅ Stored episode via MCP: {}", episode.session_id);

                return Ok(());
            }
        }

        // Fallback: Log to console/file
        let episode_json = serde_json::to_string_pretty(episode)
            .map_err(|e| JJError::SerializationError(e.to_string()))?;

        #[cfg(feature = "native")]
        {
            println!("[agentdb-sync] Would store episode:");
            println!("{}", episode_json);

            // Optionally write to file for later batch import
            if let Ok(path) = std::env::var("AGENTDB_SYNC_FILE") {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                {
                    writeln!(file, "{}", episode_json).ok();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&format!("[agentdb-sync] {}", episode_json).into());
        }

        Ok(())
    }

    /// Query similar operations from AgentDB
    pub async fn query_similar_operations(
        &self,
        task: &str,
        limit: usize,
    ) -> Result<Vec<AgentDBEpisode>> {
        if !self.enabled {
            return Ok(vec![]);
        }

        // If MCP client is available, use it for real AgentDB queries (native only)
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(client) = &self.mcp_client {
                let result = client.search_patterns(task.to_string(), limit).await?;

                // Parse response into episodes
                let episodes: Vec<AgentDBEpisode> = serde_json::from_value(result)
                    .map_err(|e| JJError::SerializationError(format!("Failed to parse episodes: {}", e)))?;

                #[cfg(feature = "native")]
                println!("[agentdb-sync] ✅ Found {} similar episodes via MCP", episodes.len());

                return Ok(episodes);
            }
        }

        // Fallback: Log and return empty
        #[cfg(feature = "native")]
        {
            println!(
                "[agentdb-sync] Would query similar operations for: {}",
                task
            );
            println!("[agentdb-sync] Limit: {}", limit);
        }

        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(
                &format!("[agentdb-sync] Would query: {} (limit: {})", task, limit).into(),
            );
        }

        Ok(vec![])
    }

    /// Get statistics for operations related to a task
    pub async fn get_task_statistics(&self, task_pattern: &str) -> Result<TaskStatistics> {
        if !self.enabled {
            return Ok(TaskStatistics::default());
        }

        // If MCP client is available, use it for real AgentDB statistics (native only)
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(client) = &self.mcp_client {
                let result = client.get_pattern_stats(task_pattern.to_string(), 10).await?;

                // Parse response into statistics
                let stats: TaskStatistics = serde_json::from_value(result)
                    .map_err(|e| JJError::SerializationError(format!("Failed to parse stats: {}", e)))?;

                #[cfg(feature = "native")]
                println!("[agentdb-sync] ✅ Retrieved statistics via MCP for: {}", task_pattern);

                return Ok(stats);
            }
        }

        // Fallback: Log and return default
        #[cfg(feature = "native")]
        {
            println!(
                "[agentdb-sync] Would get statistics for pattern: {}",
                task_pattern
            );
        }

        Ok(TaskStatistics::default())
    }

    /// Batch sync multiple operations
    pub async fn batch_sync_operations(
        &self,
        operations: &[(JJOperation, String, String)], // (operation, session_id, agent_id)
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for (op, session_id, agent_id) in operations {
            self.sync_operation(op, session_id, agent_id).await?;
        }

        Ok(())
    }

    /// Check if sync is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Statistics for task operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStatistics {
    /// Total number of operations
    pub total_operations: usize,
    /// Number of successful operations
    pub successful_operations: usize,
    /// Number of failed operations
    pub failed_operations: usize,
    /// Average reward score
    pub average_reward: f64,
    /// Average latency in milliseconds
    pub average_latency_ms: Option<f64>,
    /// Total tokens used
    pub total_tokens: Option<u64>,
    /// Common critique patterns
    pub common_critiques: Vec<String>,
}

impl TaskStatistics {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 0.0;
        }
        self.successful_operations as f64 / self.total_operations as f64
    }

    /// Calculate failure rate
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OperationType;

    #[test]
    fn test_episode_creation() {
        let op = JJOperation::builder()
            .operation_id("test-op".to_string())
            .operation_type(OperationType::Describe)
            .command("Test operation".to_string())
            .user("test-user".to_string())
            .hostname("localhost".to_string())
            .build();

        let episode =
            AgentDBEpisode::from_operation(&op, "session-001".to_string(), "agent-001".to_string());

        assert_eq!(episode.session_id, "session-001");
        assert_eq!(episode.agent_id, "agent-001");
        assert_eq!(episode.task, "Test operation");
        assert!(episode.success);
        assert_eq!(episode.reward, 1.0);
    }

    #[test]
    fn test_episode_builder() {
        let op = JJOperation::builder()
            .operation_id("test-op".to_string())
            .operation_type(OperationType::Describe)
            .command("Test operation".to_string())
            .user("test-user".to_string())
            .hostname("localhost".to_string())
            .build();

        let episode =
            AgentDBEpisode::from_operation(&op, "session-001".to_string(), "agent-001".to_string())
                .with_input("input context".to_string())
                .with_output("output result".to_string())
                .with_critique("good work".to_string())
                .with_success(true, 0.95)
                .with_metrics(1500, 250);

        assert_eq!(episode.input.unwrap(), "input context");
        assert_eq!(episode.output.unwrap(), "output result");
        assert_eq!(episode.critique.unwrap(), "good work");
        assert_eq!(episode.reward, 0.95);
        assert_eq!(episode.latency_ms.unwrap(), 1500);
        assert_eq!(episode.tokens_used.unwrap(), 250);
    }

    #[test]
    fn test_task_statistics() {
        let stats = TaskStatistics {
            total_operations: 100,
            successful_operations: 85,
            failed_operations: 15,
            average_reward: 0.85,
            average_latency_ms: Some(1200.0),
            total_tokens: Some(50000),
            common_critiques: vec!["needs optimization".to_string()],
        };

        assert!((stats.success_rate() - 0.85).abs() < 0.001);
        assert!((stats.failure_rate() - 0.15).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_agentdb_sync_creation() {
        let sync = AgentDBSync::new(true);
        assert!(sync.is_enabled());

        let sync = AgentDBSync::new(false);
        assert!(!sync.is_enabled());
    }

    #[tokio::test]
    async fn test_sync_disabled() {
        let sync = AgentDBSync::new(false);
        let op = JJOperation::builder()
            .operation_id("test-op".to_string())
            .operation_type(OperationType::Describe)
            .command("Test operation".to_string())
            .user("test-user".to_string())
            .hostname("localhost".to_string())
            .build();

        let result = sync.sync_operation(&op, "session-001", "agent-001").await;
        assert!(result.is_ok());
    }
}
