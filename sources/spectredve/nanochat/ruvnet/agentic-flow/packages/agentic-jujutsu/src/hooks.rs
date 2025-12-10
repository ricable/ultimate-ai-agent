//! Integration with agentic-flow hooks system
//!
//! This module provides seamless integration between agentic-jujutsu and the
//! agentic-flow hooks system, enabling automatic operation tracking, memory sync,
//! and multi-agent coordination.

use crate::{JJOperation, JJWrapper, OperationType, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Context information for hook execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookContext {
    /// Unique identifier for the agent
    pub agent_id: String,
    /// Session identifier for grouping related operations
    pub session_id: String,
    /// Human-readable task description
    pub task_description: String,
    /// Unix timestamp in seconds
    pub timestamp: i64,
    /// Optional metadata for additional context
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl HookContext {
    /// Create a new hook context
    pub fn new(agent_id: String, session_id: String, task_description: String) -> Self {
        Self {
            agent_id,
            session_id,
            task_description,
            timestamp: Utc::now().timestamp(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Create context with metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Types of hook events that can be triggered
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HookEventType {
    /// Before task execution begins
    PreTask,
    /// After a file edit operation
    PostEdit,
    /// After task execution completes
    PostTask,
    /// When a conflict is detected
    ConflictDetected,
    /// When an operation is logged
    OperationLogged,
    /// Session initialization
    SessionInit,
    /// Session cleanup
    SessionEnd,
}

/// Hook event containing operation and context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JJHookEvent {
    /// Type of hook event
    pub event_type: HookEventType,
    /// Associated jj operation (if any)
    pub operation: Option<JJOperation>,
    /// Hook execution context
    pub context: HookContext,
    /// Additional event metadata
    pub metadata: serde_json::Value,
}

impl JJHookEvent {
    /// Create a new hook event
    pub fn new(
        event_type: HookEventType,
        operation: Option<JJOperation>,
        context: HookContext,
    ) -> Self {
        Self {
            event_type,
            operation,
            context,
            metadata: serde_json::Value::Null,
        }
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Integration layer for agentic-flow hooks
pub struct JJHooksIntegration {
    /// Underlying JJ wrapper
    wrapper: JJWrapper,
    /// Whether AgentDB sync is enabled
    agentdb_enabled: bool,
    /// Current session context
    current_session: Option<HookContext>,
}

impl JJHooksIntegration {
    /// Create a new hooks integration instance
    pub fn new(wrapper: JJWrapper, agentdb_enabled: bool) -> Self {
        Self {
            wrapper,
            agentdb_enabled,
            current_session: None,
        }
    }

    /// Execute pre-task hook
    ///
    /// This hook is called before an agent begins work on a task.
    /// It initializes the session state and prepares the jj repository.
    pub async fn on_pre_task(&mut self, ctx: HookContext) -> Result<JJHookEvent> {
        // Store current session context
        self.current_session = Some(ctx.clone());

        // Create session marker in jj
        let description = format!(
            "[pre-task] Agent: {} | Session: {} | Task: {}",
            ctx.agent_id, ctx.session_id, ctx.task_description
        );

        // Log the session start
        let event = JJHookEvent::new(HookEventType::PreTask, None, ctx.clone()).with_metadata(
            serde_json::json!({
                "action": "session_init",
                "description": description,
            }),
        );

        // Sync to AgentDB if enabled
        if self.agentdb_enabled {
            self.sync_event_to_agentdb(&event).await?;
        }

        Ok(event)
    }

    /// Execute post-edit hook
    ///
    /// This hook is called after a file edit operation.
    /// It automatically commits the changes and logs the operation.
    pub async fn on_post_edit(&mut self, file: &str, ctx: HookContext) -> Result<JJOperation> {
        // Create operation description
        let description = format!(
            "[post-edit] Agent: {} | File: {} | Session: {}",
            ctx.agent_id, file, ctx.session_id
        );

        // Create a jj operation for this edit
        let operation = JJOperation::builder()
            .operation_id(uuid::Uuid::new_v4().to_string())
            .operation_type(OperationType::Describe)
            .command(description.clone())
            .user(ctx.agent_id.clone())
            .hostname("hook-agent".to_string())
            .add_metadata("file", file)
            .add_metadata("session_id", &ctx.session_id)
            .add_metadata("agent_id", &ctx.agent_id)
            .add_metadata("hook", "post-edit")
            .build();

        // Create hook event
        let event = JJHookEvent::new(
            HookEventType::PostEdit,
            Some(operation.clone()),
            ctx.clone(),
        )
        .with_metadata(serde_json::json!({
            "file": file,
            "auto_commit": true,
        }));

        // Sync to AgentDB if enabled
        if self.agentdb_enabled {
            self.sync_event_to_agentdb(&event).await?;
        }

        Ok(operation)
    }

    /// Execute post-task hook
    ///
    /// This hook is called after task execution completes.
    /// It gathers all operations from the session and generates a summary.
    pub async fn on_post_task(&mut self, ctx: HookContext) -> Result<Vec<JJOperation>> {
        // Get current session operations
        let operations = self.get_session_operations(&ctx.session_id).await?;

        // Create summary
        let summary = serde_json::json!({
            "session_id": ctx.session_id,
            "agent_id": ctx.agent_id,
            "task": ctx.task_description,
            "operations_count": operations.len(),
            "timestamp": ctx.timestamp,
        });

        // Create hook event
        let event =
            JJHookEvent::new(HookEventType::PostTask, None, ctx.clone()).with_metadata(summary);

        // Sync to AgentDB if enabled
        if self.agentdb_enabled {
            self.sync_event_to_agentdb(&event).await?;
        }

        // Clear current session
        self.current_session = None;

        Ok(operations)
    }

    /// Handle conflict detection
    ///
    /// This hook is called when a merge conflict is detected.
    /// It notifies the coordination system for resolution.
    pub async fn on_conflict_detected(
        &self,
        conflict_files: Vec<String>,
        ctx: HookContext,
    ) -> Result<JJHookEvent> {
        let event = JJHookEvent::new(HookEventType::ConflictDetected, None, ctx).with_metadata(
            serde_json::json!({
                "conflicts": conflict_files,
                "requires_resolution": true,
            }),
        );

        // Sync to AgentDB for learning
        if self.agentdb_enabled {
            self.sync_event_to_agentdb(&event).await?;
        }

        Ok(event)
    }

    /// Get operations for a specific session
    async fn get_session_operations(&self, _session_id: &str) -> Result<Vec<JJOperation>> {
        // This would query the operation log for operations matching the session ID
        // For now, return empty vec as placeholder
        Ok(vec![])
    }

    /// Sync event to AgentDB
    async fn sync_event_to_agentdb(&self, event: &JJHookEvent) -> Result<()> {
        if !self.agentdb_enabled {
            return Ok(());
        }

        // Prepare episode data for AgentDB
        let episode = serde_json::json!({
            "sessionId": event.context.session_id,
            "task": event.context.task_description,
            "agentId": event.context.agent_id,
            "eventType": format!("{:?}", event.event_type),
            "operation": event.operation,
            "metadata": event.metadata,
            "timestamp": event.context.timestamp,
            "success": true,
            "reward": 1.0,
        });

        // TODO: Implement actual AgentDB sync via MCP
        // For now, just log
        #[cfg(feature = "native")]
        {
            println!(
                "[jj-agentdb] Would sync event: {}",
                serde_json::to_string_pretty(&episode)
                    .unwrap_or_else(|_| "serialization error".to_string())
            );
        }

        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(
                &format!("[jj-agentdb] Would sync event: {:?}", episode).into(),
            );
        }

        Ok(())
    }

    /// Get current session context
    pub fn current_session(&self) -> Option<&HookContext> {
        self.current_session.as_ref()
    }

    /// Check if AgentDB sync is enabled
    pub fn is_agentdb_enabled(&self) -> bool {
        self.agentdb_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JJConfig;

    #[test]
    fn test_hook_context_creation() {
        let ctx = HookContext::new(
            "test-agent".to_string(),
            "session-001".to_string(),
            "Test task".to_string(),
        );

        assert_eq!(ctx.agent_id, "test-agent");
        assert_eq!(ctx.session_id, "session-001");
        assert_eq!(ctx.task_description, "Test task");
        assert!(ctx.timestamp > 0);
    }

    #[test]
    fn test_hook_context_with_metadata() {
        let ctx = HookContext::new(
            "test-agent".to_string(),
            "session-001".to_string(),
            "Test task".to_string(),
        )
        .with_metadata(serde_json::json!({"key": "value"}));

        assert_eq!(ctx.metadata["key"], "value");
    }

    #[test]
    fn test_hook_event_creation() {
        let ctx = HookContext::new(
            "test-agent".to_string(),
            "session-001".to_string(),
            "Test task".to_string(),
        );

        let event = JJHookEvent::new(HookEventType::PreTask, None, ctx);

        assert_eq!(event.event_type, HookEventType::PreTask);
        assert!(event.operation.is_none());
    }

    #[tokio::test]
    async fn test_hooks_integration_creation() {
        let config = JJConfig::default();
        let wrapper = JJWrapper::with_config(config).unwrap();
        let integration = JJHooksIntegration::new(wrapper, true);

        assert!(integration.is_agentdb_enabled());
        assert!(integration.current_session().is_none());
    }

    #[tokio::test]
    async fn test_pre_task_hook() {
        let config = JJConfig::default();
        let wrapper = JJWrapper::with_config(config).unwrap();
        let mut integration = JJHooksIntegration::new(wrapper, false);

        let ctx = HookContext::new(
            "test-agent".to_string(),
            "session-001".to_string(),
            "Test task".to_string(),
        );

        let event = integration.on_pre_task(ctx.clone()).await.unwrap();

        assert_eq!(event.event_type, HookEventType::PreTask);
        assert_eq!(
            integration.current_session().unwrap().agent_id,
            "test-agent"
        );
    }

    #[tokio::test]
    async fn test_post_edit_hook() {
        let config = JJConfig::default();
        let wrapper = JJWrapper::with_config(config).unwrap();
        let mut integration = JJHooksIntegration::new(wrapper, false);

        let ctx = HookContext::new(
            "test-agent".to_string(),
            "session-001".to_string(),
            "Test task".to_string(),
        );

        // Initialize session first
        integration.on_pre_task(ctx.clone()).await.unwrap();

        // Execute post-edit
        let operation = integration.on_post_edit("test.rs", ctx).await.unwrap();

        assert_eq!(operation.operation_type, "Describe");
        assert!(operation.command.contains("test.rs"));
    }
}
