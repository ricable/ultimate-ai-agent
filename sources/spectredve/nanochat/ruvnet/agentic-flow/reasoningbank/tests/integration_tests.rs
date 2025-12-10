//! Integration tests for ReasoningBank

use reasoningbank_core::{Pattern, TaskOutcome, ReasoningEngine};
use reasoningbank_learning::{AdaptiveLearner, LearningConfig};
use reasoningbank_mcp::McpServer;
use reasoningbank_storage::SqliteStorage;
use tempfile::tempdir;

#[tokio::test]
async fn test_end_to_end_workflow() {
    // Create temporary database
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    // Initialize components
    let mut config = reasoningbank_storage::StorageConfig::default();
    config.database_path = db_path;

    let storage = SqliteStorage::new(config).unwrap();
    let engine = ReasoningEngine::default();
    let mut learner = AdaptiveLearner::new(
        engine,
        storage.clone(),
        LearningConfig::default(),
    );

    // Create a pattern
    let pattern1 = Pattern::new(
        "Implement REST API with authentication".to_string(),
        "api-development".to_string(),
        "token-based-auth".to_string(),
    ).with_outcome(TaskOutcome::partial(0.9, 15.0));

    // Learn from pattern
    let insight1 = learner.learn_from_task(&pattern1).unwrap();
    assert_eq!(insight1.similar_patterns_found, 0); // First pattern

    // Create similar pattern
    let pattern2 = Pattern::new(
        "Build API with JWT authentication".to_string(),
        "api-development".to_string(),
        "jwt-auth".to_string(),
    ).with_outcome(TaskOutcome::partial(0.85, 12.0));

    let insight2 = learner.learn_from_task(&pattern2).unwrap();
    assert!(insight2.similar_patterns_found > 0); // Should find pattern1

    // Apply learning
    let applied = learner.apply_learning(
        "Create API with secure auth",
        "api-development",
    ).unwrap();

    assert!(applied.recommended_strategy.is_some());
    assert!(applied.confidence > 0.0);
    assert!(applied.expected_success > 0.0);

    println!("✓ End-to-end workflow test passed");
    println!("  Recommended strategy: {:?}", applied.recommended_strategy);
    println!("  Confidence: {:.2}", applied.confidence);
    println!("  Expected success: {:.2}", applied.expected_success);
}

#[tokio::test]
async fn test_mcp_server_integration() {
    use serde_json::json;

    // Create MCP server with in-memory database
    let mut config = reasoningbank_mcp::McpConfig::default();
    config.storage.database_path = ":memory:".into();

    let server = McpServer::new(config).await.unwrap();

    // Test tool execution: store pattern
    let store_params = json!({
        "task_description": "Test task",
        "task_category": "testing",
        "strategy": "test-strategy",
        "success_score": 0.95,
        "duration_seconds": 1.5
    });

    let store_result = server.execute_tool("reasoning_store", store_params).await.unwrap();
    assert!(store_result.get("pattern_id").is_some());
    assert_eq!(store_result.get("stored").unwrap(), &json!(true));

    // Test tool execution: analyze
    let analyze_params = json!({
        "category": "testing"
    });

    let analyze_result = server.execute_tool("reasoning_analyze", analyze_params).await.unwrap();
    assert!(analyze_result.get("total_patterns").is_some());

    // Test resource access
    let metrics = server.get_resource("reasoning://metrics/performance").await.unwrap();
    assert!(metrics.get("total_patterns").is_some());

    println!("✓ MCP server integration test passed");
    println!("  Tools: {} available", server.list_tools().len());
    println!("  Resources: {} available", server.list_resources().len());
}

#[tokio::test]
async fn test_pattern_similarity() {
    let storage = SqliteStorage::in_memory().unwrap();
    let engine = ReasoningEngine::default();

    // Create patterns with similar descriptions
    let pattern1 = Pattern::new(
        "Build a machine learning model for classification".to_string(),
        "ml-development".to_string(),
        "supervised-learning".to_string(),
    );

    let pattern2 = Pattern::new(
        "Create ML classifier using supervised learning".to_string(),
        "ml-development".to_string(),
        "neural-network".to_string(),
    );

    let pattern3 = Pattern::new(
        "Setup database connection pooling".to_string(),
        "database".to_string(),
        "connection-pool".to_string(),
    );

    let prepared1 = engine.prepare_pattern(pattern1).unwrap();
    let prepared2 = engine.prepare_pattern(pattern2).unwrap();
    let prepared3 = engine.prepare_pattern(pattern3).unwrap();

    storage.store_pattern(&prepared1).unwrap();
    storage.store_pattern(&prepared2).unwrap();
    storage.store_pattern(&prepared3).unwrap();

    // Find similar patterns
    let candidates = storage.get_all_patterns(None).unwrap();
    let similar = engine.find_similar(&prepared1, &candidates);

    // Pattern2 should be most similar to pattern1
    assert!(similar.len() >= 2);

    let most_similar = &similar[1]; // [0] is self
    assert!(most_similar.1.overall_score > 0.5);

    println!("✓ Pattern similarity test passed");
    println!("  Found {} similar patterns", similar.len());
    println!("  Top similarity score: {:.3}", similar[1].1.overall_score);
}

#[tokio::test]
async fn test_strategy_optimization() {
    use reasoningbank_learning::StrategyOptimizer;

    let storage = SqliteStorage::in_memory().unwrap();

    // Create patterns with different strategies
    for i in 0..5 {
        let pattern = Pattern::new(
            format!("Task {}", i),
            "optimization-test".to_string(),
            "strategy-a".to_string(),
        ).with_outcome(TaskOutcome::partial(0.9, 1.0));

        storage.store_pattern(&engine.prepare_pattern(pattern).unwrap()).unwrap();
    }

    for i in 0..3 {
        let pattern = Pattern::new(
            format!("Task {}", i + 5),
            "optimization-test".to_string(),
            "strategy-b".to_string(),
        ).with_outcome(TaskOutcome::partial(0.6, 2.0));

        storage.store_pattern(&engine.prepare_pattern(pattern).unwrap()).unwrap();
    }

    // Run optimizer
    let optimizer = StrategyOptimizer::new(storage);
    let result = optimizer.optimize_for_category("optimization-test").unwrap();

    assert_eq!(result.total_patterns, 8);
    assert_eq!(result.strategy_rankings.len(), 2);
    assert_eq!(result.recommended_strategy.unwrap(), "strategy-a");

    println!("✓ Strategy optimization test passed");
    println!("  Recommended: {}", result.recommended_strategy.unwrap());
    println!("  Strategy A score: {:.3}", result.strategy_rankings[0].composite_score);
}
