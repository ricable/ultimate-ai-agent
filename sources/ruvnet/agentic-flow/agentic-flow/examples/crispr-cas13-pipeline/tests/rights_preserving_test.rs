//! Integration Tests for Rights-Preserving Countermeasure Platform

use crispr_cas13_pipeline::{
    AuditLogger, Platform, PlatformConfig, PolicyEnforcer, PrivacyConfig, PrivacyEngine,
};

#[tokio::test]
async fn test_platform_initialization() {
    let config = PlatformConfig::default();
    let platform = Platform::new(config).await.unwrap();

    let status = platform.status().await;

    assert!(status.privacy_enabled);
    assert!(status.policy_enforcement_enabled);
    assert!(status.audit_enabled);
    assert_eq!(status.api_endpoint, "0.0.0.0:8080");
}

#[tokio::test]
async fn test_privacy_engine() {
    let config = PrivacyConfig::default();
    let engine = PrivacyEngine::new(config);

    // Initialize budget
    engine
        .initialize_budget("test_user".to_string(), 10.0)
        .await
        .unwrap();

    // Test noise application
    let data = serde_json::json!({"value": 100.0});
    let result = engine.apply_noise(&data, 1.0).await.unwrap();

    assert_eq!(result.budget_consumed, 1.0);

    // Test budget consumption
    engine.consume_budget("test_user", 1.0).await.unwrap();
    let remaining = engine.check_budget("test_user").await.unwrap();
    assert_eq!(remaining, 9.0);
}

#[tokio::test]
async fn test_privacy_anonymization() {
    let config = PrivacyConfig::default();
    let engine = PrivacyEngine::new(config);

    let sensitive_data = serde_json::json!({
        "patient_id": "12345",
        "name": "John Doe",
        "age": 35,
        "gene_expression": 100
    });

    let anonymized = engine.anonymize_data(sensitive_data).await.unwrap();

    assert!(!anonymized.as_object().unwrap().contains_key("patient_id"));
    assert!(!anonymized.as_object().unwrap().contains_key("name"));
    assert!(anonymized.as_object().unwrap().contains_key("age_range"));
}

#[tokio::test]
async fn test_policy_enforcement() {
    let enforcer = PolicyEnforcer::new(crispr_cas13_pipeline::governance::policy::PolicyConfig::default());
    let policy = PolicyEnforcer::create_default_policy();

    enforcer.add_policy(policy).await.unwrap();

    // Test allow decision
    let decision = enforcer
        .evaluate("researcher", "read", "crispr:data:sample1")
        .await
        .unwrap();

    assert!(decision.allowed);

    // Test deny decision (no matching rule)
    let decision = enforcer
        .evaluate("guest", "read", "crispr:data:sample1")
        .await
        .unwrap();

    assert!(!decision.allowed);
}

#[tokio::test]
async fn test_audit_logging() {
    let logger = AuditLogger::new("test-signing-key".to_string());

    // Log multiple events
    let id1 = logger
        .log_event(
            "data_access",
            "user1",
            serde_json::json!({
                "resource": "sample1",
                "action": "read"
            }),
        )
        .await
        .unwrap();

    let id2 = logger
        .log_event(
            "data_export",
            "user1",
            serde_json::json!({
                "resource": "sample2",
                "action": "export"
            }),
        )
        .await
        .unwrap();

    assert!(!id1.is_empty());
    assert!(!id2.is_empty());

    // Verify chain integrity
    let valid = logger.verify_chain().await.unwrap();
    assert!(valid);

    // Test actor history
    let history = logger.get_actor_history("user1").await;
    assert_eq!(history.len(), 2);

    // Test report generation
    let report = logger.generate_report(Some("user1")).await;
    assert_eq!(report.total_events, 2);
    assert_eq!(report.unique_actors, 1);
}

#[tokio::test]
async fn test_audit_chain_verification() {
    let logger = AuditLogger::new("test-key".to_string());

    // Create a chain of events
    for i in 0..5 {
        logger
            .log_event(
                &format!("event_{}", i),
                "test_user",
                serde_json::json!({"index": i}),
            )
            .await
            .unwrap();
    }

    // Verify chain
    let valid = logger.verify_chain().await.unwrap();
    assert!(valid);

    // Get events by time range
    let start = chrono::Utc::now() - chrono::Duration::minutes(5);
    let end = chrono::Utc::now() + chrono::Duration::minutes(5);
    let events = logger.get_by_time_range(start, end).await;

    assert_eq!(events.len(), 5);
}

#[tokio::test]
async fn test_end_to_end_workflow() {
    // Initialize platform
    let config = PlatformConfig {
        api_port: 8081, // Use different port for test
        grpc_port: 50052,
        ..Default::default()
    };

    let platform = Platform::new(config).await.unwrap();

    // Get status
    let status = platform.status().await;
    assert!(status.privacy_enabled);

    // Shutdown
    platform.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_privacy_budget_enforcement() {
    let config = PrivacyConfig::default();
    let engine = PrivacyEngine::new(config);

    // Initialize with limited budget
    engine
        .initialize_budget("limited_user".to_string(), 2.0)
        .await
        .unwrap();

    // Consume partial budget
    engine.consume_budget("limited_user", 1.0).await.unwrap();

    // Try to exceed budget
    let result = engine.consume_budget("limited_user", 2.0).await;
    assert!(result.is_err());

    // Verify remaining budget
    let remaining = engine.check_budget("limited_user").await.unwrap();
    assert_eq!(remaining, 1.0);
}

#[tokio::test]
async fn test_comprehensive_integration() {
    // Initialize all components
    let privacy_config = PrivacyConfig::default();
    let privacy_engine = PrivacyEngine::new(privacy_config.clone());
    let policy_enforcer = PolicyEnforcer::new(crispr_cas13_pipeline::governance::policy::PolicyConfig::default());
    let audit_logger = AuditLogger::new("integration-test-key".to_string());

    // Add policy
    let policy = PolicyEnforcer::create_default_policy();
    policy_enforcer.add_policy(policy).await.unwrap();

    // Initialize privacy budget
    privacy_engine
        .initialize_budget("integration_user".to_string(), 10.0)
        .await
        .unwrap();

    // Simulate data access workflow
    let user = "integration_user";
    let resource = "crispr:data:sample1";

    // 1. Check policy
    let policy_decision = policy_enforcer
        .evaluate(user, "read", resource)
        .await
        .unwrap();

    // For this test, we expect denial since "integration_user" is not a "researcher"
    assert!(!policy_decision.allowed); // Guest users are denied

    // 2. Log audit event
    let audit_id = audit_logger
        .log_event(
            "access_denied",
            user,
            serde_json::json!({
                "resource": resource,
                "reason": "not_authorized"
            }),
        )
        .await
        .unwrap();

    assert!(!audit_id.is_empty());

    // Verify audit chain
    let valid = audit_logger.verify_chain().await.unwrap();
    assert!(valid);
}
