//! Integration tests for hooks system

use agentic_jujutsu::{
    HookContext, HookEventType, JJHooksIntegration, JJWrapper, Result,
};

#[tokio::test]
async fn test_full_hooks_lifecycle() -> Result<()> {
    // Create integration
    let wrapper = JJWrapper::new()?;
    let mut integration = JJHooksIntegration::new(wrapper, false);

    // Create context
    let ctx = HookContext::new(
        "test-agent".to_string(),
        "test-session".to_string(),
        "Test task".to_string(),
    );

    // Pre-task
    let pre_event = integration.on_pre_task(ctx.clone()).await?;
    assert_eq!(pre_event.event_type, HookEventType::PreTask);
    assert_eq!(pre_event.context.agent_id, "test-agent");

    // Verify session is active
    assert!(integration.current_session().is_some());

    // Post-edit
    let op = integration.on_post_edit("test.rs", ctx.clone()).await?;
    assert!(op.command.contains("test.rs"));
    assert_eq!(op.user, "test-agent");

    // Post-task
    let operations = integration.on_post_task(ctx).await?;
    assert!(!operations.is_empty());

    // Verify session is cleared
    assert!(integration.current_session().is_none());

    Ok(())
}

#[tokio::test]
async fn test_multiple_edits() -> Result<()> {
    let wrapper = JJWrapper::new()?;
    let mut integration = JJHooksIntegration::new(wrapper, false);

    let ctx = HookContext::new(
        "multi-edit-agent".to_string(),
        "session-001".to_string(),
        "Multiple edits task".to_string(),
    );

    // Pre-task
    integration.on_pre_task(ctx.clone()).await?;

    // Multiple edits
    let files = vec!["file1.rs", "file2.rs", "file3.rs"];
    for file in &files {
        integration.on_post_edit(file, ctx.clone()).await?;
    }

    // Post-task
    let operations = integration.on_post_task(ctx).await?;

    // Should have at least as many operations as files
    assert!(operations.len() >= files.len());

    Ok(())
}

#[tokio::test]
async fn test_conflict_detection() -> Result<()> {
    let wrapper = JJWrapper::new()?;
    let integration = JJHooksIntegration::new(wrapper, false);

    let ctx = HookContext::new(
        "conflict-agent".to_string(),
        "session-001".to_string(),
        "Conflict test".to_string(),
    );

    let conflicts = vec!["file1.rs".to_string(), "file2.rs".to_string()];
    let event = integration
        .on_conflict_detected(conflicts.clone(), ctx)
        .await?;

    assert_eq!(event.event_type, HookEventType::ConflictDetected);
    assert!(event.metadata.get("conflicts").is_some());

    Ok(())
}

#[tokio::test]
async fn test_agentdb_sync_enabled() -> Result<()> {
    let wrapper = JJWrapper::new()?;
    let integration = JJHooksIntegration::new(wrapper, true);

    assert!(integration.is_agentdb_enabled());

    Ok(())
}

#[tokio::test]
async fn test_hook_context_metadata() -> Result<()> {
    let metadata = serde_json::json!({
        "priority": "high",
        "tags": ["auth", "security"],
    });

    let ctx = HookContext::new(
        "meta-agent".to_string(),
        "session-001".to_string(),
        "Metadata test".to_string(),
    )
    .with_metadata(metadata.clone());

    assert_eq!(ctx.metadata, metadata);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_sessions() -> Result<()> {
    // Create two independent integrations
    let wrapper1 = JJWrapper::new()?;
    let mut integration1 = JJHooksIntegration::new(wrapper1, false);

    let wrapper2 = JJWrapper::new()?;
    let mut integration2 = JJHooksIntegration::new(wrapper2, false);

    // Session 1
    let ctx1 = HookContext::new(
        "agent-1".to_string(),
        "session-1".to_string(),
        "Task 1".to_string(),
    );
    integration1.on_pre_task(ctx1.clone()).await?;

    // Session 2
    let ctx2 = HookContext::new(
        "agent-2".to_string(),
        "session-2".to_string(),
        "Task 2".to_string(),
    );
    integration2.on_pre_task(ctx2.clone()).await?;

    // Both should have active sessions
    assert!(integration1.current_session().is_some());
    assert!(integration2.current_session().is_some());

    // Verify session IDs are different
    assert_ne!(
        integration1.current_session().unwrap().session_id,
        integration2.current_session().unwrap().session_id
    );

    // Clean up
    integration1.on_post_task(ctx1).await?;
    integration2.on_post_task(ctx2).await?;

    Ok(())
}

#[tokio::test]
async fn test_error_handling_without_session() -> Result<()> {
    let wrapper = JJWrapper::new()?;
    let mut integration = JJHooksIntegration::new(wrapper, false);

    // Try post-task without pre-task
    let ctx = HookContext::new(
        "no-session-agent".to_string(),
        "session-001".to_string(),
        "Task".to_string(),
    );

    // This should still work but return empty operations
    let operations = integration.on_post_task(ctx).await?;
    assert!(operations.is_empty());

    Ok(())
}
