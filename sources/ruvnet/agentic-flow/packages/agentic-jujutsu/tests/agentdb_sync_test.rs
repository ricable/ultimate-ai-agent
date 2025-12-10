//! Tests for AgentDB synchronization

use agentic_jujutsu::{AgentDBEpisode, AgentDBSync, JJOperation, OperationType, Result};
use std::collections::HashMap;

#[tokio::test]
async fn test_episode_creation() -> Result<()> {
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

    Ok(())
}

#[tokio::test]
async fn test_episode_builder() -> Result<()> {
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

    Ok(())
}

#[tokio::test]
async fn test_sync_disabled() -> Result<()> {
    let sync = AgentDBSync::new(false);
    assert!(!sync.is_enabled());

    let op = JJOperation::builder()
        .operation_id("test-op".to_string())
        .operation_type(OperationType::Describe)
        .command("Test operation".to_string())
        .user("test-user".to_string())
        .hostname("localhost".to_string())
        .build();

    // Should succeed but do nothing
    let result = sync.sync_operation(&op, "session-001", "agent-001").await;
    assert!(result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_sync_enabled() -> Result<()> {
    let sync = AgentDBSync::new(true);
    assert!(sync.is_enabled());

    let op = JJOperation::builder()
        .operation_id("test-op".to_string())
        .operation_type(OperationType::Describe)
        .command("Test operation".to_string())
        .user("test-user".to_string())
        .hostname("localhost".to_string())
        .build();

    // Should log but not fail
    let result = sync.sync_operation(&op, "session-001", "agent-001").await;
    assert!(result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_batch_sync() -> Result<()> {
    let sync = AgentDBSync::new(true);

    let ops = vec![
        (
            JJOperation::builder()
                .operation_id("op-1".to_string())
                .operation_type(OperationType::Describe)
                .command("Op 1".to_string())
                .user("agent".to_string())
                .hostname("localhost".to_string())
                .build(),
            "session-001".to_string(),
            "agent-001".to_string(),
        ),
        (
            JJOperation::builder()
                .operation_id("op-2".to_string())
                .operation_type(OperationType::Describe)
                .command("Op 2".to_string())
                .user("agent".to_string())
                .hostname("localhost".to_string())
                .build(),
            "session-001".to_string(),
            "agent-001".to_string(),
        ),
    ];

    let result = sync.batch_sync_operations(&ops).await;
    assert!(result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_query_similar() -> Result<()> {
    let sync = AgentDBSync::new(true);

    let episodes = sync
        .query_similar_operations("implement authentication", 5)
        .await?;

    // Should return empty for now (mock implementation)
    assert!(episodes.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_task_statistics() -> Result<()> {
    let sync = AgentDBSync::new(true);

    let stats = sync.get_task_statistics("authentication").await?;

    // Should return default stats
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.success_rate(), 0.0);

    Ok(())
}

#[tokio::test]
async fn test_episode_serialization() -> Result<()> {
    let op = JJOperation::builder()
        .operation_id("test-op".to_string())
        .operation_type(OperationType::Describe)
        .command("Test operation".to_string())
        .user("test-user".to_string())
        .hostname("localhost".to_string())
        .build();

    let episode =
        AgentDBEpisode::from_operation(&op, "session-001".to_string(), "agent-001".to_string());

    // Test serialization
    let json = serde_json::to_string(&episode)?;
    assert!(json.contains("session-001"));
    assert!(json.contains("agent-001"));

    // Test deserialization
    let deserialized: AgentDBEpisode = serde_json::from_str(&json)?;
    assert_eq!(deserialized.session_id, episode.session_id);
    assert_eq!(deserialized.agent_id, episode.agent_id);

    Ok(())
}
