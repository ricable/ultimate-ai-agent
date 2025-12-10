//! Compliance validation tests for the Synaptic Market
//!
//! Validates compliance with:
//! - Anthropic Terms of Service (no shared API keys, peer-orchestrated model)
//! - Data protection regulations (GDPR, CCPA)
//! - Financial regulations (no unlicensed money transmission)
//! - Privacy requirements (user consent, data minimization)
//! - Audit trail requirements

use std::collections::HashMap;
use std::time::Duration;
use chrono::{Utc, DateTime};
use libp2p::PeerId;
use uuid::Uuid;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde_json::json;
use tokio::time::sleep;

mod compliance_utils {
    use super::*;

    pub struct ComplianceAuditor {
        pub audit_log: Vec<AuditEvent>,
    }

    #[derive(Debug, Clone)]
    pub struct AuditEvent {
        pub timestamp: DateTime<Utc>,
        pub event_type: String,
        pub user_id: PeerId,
        pub details: serde_json::Value,
        pub compliance_flags: Vec<String>,
    }

    impl ComplianceAuditor {
        pub fn new() -> Self {
            Self {
                audit_log: Vec::new(),
            }
        }

        pub fn log_event(&mut self, event_type: &str, user_id: &PeerId, details: serde_json::Value) {
            let mut compliance_flags = Vec::new();
            
            // Automatic compliance checking
            match event_type {
                "api_key_access" => compliance_flags.push("API_KEY_VIOLATION".to_string()),
                "cross_account_usage" => compliance_flags.push("ACCOUNT_SHARING_VIOLATION".to_string()),
                "data_export" => compliance_flags.push("DATA_PRIVACY_REVIEW".to_string()),
                _ => {}
            }

            self.audit_log.push(AuditEvent {
                timestamp: Utc::now(),
                event_type: event_type.to_string(),
                user_id: *user_id,
                details,
                compliance_flags,
            });
        }

        pub fn get_violations(&self) -> Vec<&AuditEvent> {
            self.audit_log.iter()
                .filter(|event| !event.compliance_flags.is_empty())
                .collect()
        }

        pub fn generate_compliance_report(&self) -> ComplianceReport {
            let mut report = ComplianceReport::default();
            
            for event in &self.audit_log {
                match event.event_type.as_str() {
                    "user_opt_in" => report.opt_in_events += 1,
                    "user_opt_out" => report.opt_out_events += 1,
                    "data_processing" => report.data_processing_events += 1,
                    "order_placement" => report.order_events += 1,
                    "task_execution" => report.task_executions += 1,
                    _ => {}
                }
                
                if !event.compliance_flags.is_empty() {
                    report.violations += 1;
                }
            }
            
            report.total_events = self.audit_log.len();
            report
        }
    }

    #[derive(Debug, Default)]
    pub struct ComplianceReport {
        pub total_events: usize,
        pub opt_in_events: usize,
        pub opt_out_events: usize,
        pub data_processing_events: usize,
        pub order_events: usize,
        pub task_executions: usize,
        pub violations: usize,
    }

    impl ComplianceReport {
        pub fn print_summary(&self) {
            println!("\n=== Compliance Report ===");
            println!("Total events: {}", self.total_events);
            println!("Opt-in events: {}", self.opt_in_events);
            println!("Opt-out events: {}", self.opt_out_events);
            println!("Data processing: {}", self.data_processing_events);
            println!("Order events: {}", self.order_events);
            println!("Task executions: {}", self.task_executions);
            println!("Violations: {}", self.violations);
            println!("Compliance rate: {:.2}%", 
                     if self.total_events > 0 {
                         ((self.total_events - self.violations) as f64 / self.total_events as f64) * 100.0
                     } else { 100.0 });
            println!("========================\n");
        }
    }

    pub fn create_test_environment() -> (tempfile::TempDir, String) {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("compliance_test.db").to_string_lossy().to_string();
        (temp_dir, db_path)
    }

    pub struct PrivacyValidator;

    impl PrivacyValidator {
        pub fn validate_data_minimization(data: &serde_json::Value) -> bool {
            // Check that only necessary data is collected
            if let Some(obj) = data.as_object() {
                // Ensure no excessive personal data collection
                let prohibited_fields = [
                    "ssn", "social_security", "credit_card", "bank_account",
                    "passport", "driver_license", "full_address"
                ];
                
                for field in &prohibited_fields {
                    if obj.contains_key(*field) {
                        return false;
                    }
                }
            }
            true
        }

        pub fn validate_consent_requirements(opt_in_status: &claude_market::OptInStatus) -> bool {
            // Verify proper consent is obtained
            opt_in_status.terms_accepted && 
            opt_in_status.privacy_policy_accepted &&
            opt_in_status.opted_in_at.is_some()
        }

        pub fn validate_data_retention(created_at: DateTime<Utc>, retention_days: i64) -> bool {
            let retention_period = chrono::Duration::days(retention_days);
            Utc::now() - created_at < retention_period
        }
    }

    pub struct TokenComplianceValidator;

    impl TokenComplianceValidator {
        pub fn validate_not_money_transmission(transaction: &claude_market::TokenTransaction) -> bool {
            // Ensure tokens are utility tokens, not money transmission
            // Check that tokens are used for computational resources, not currency exchange
            transaction.transaction_type != claude_market::TokenTransactionType::CurrencyExchange &&
            transaction.amount < 10000 && // Reasonable limits
            !transaction.is_external_payment
        }

        pub fn validate_no_investment_features(market_config: &serde_json::Value) -> bool {
            // Ensure no investment or securities features
            let prohibited_features = [
                "dividend", "profit_sharing", "voting_rights", 
                "appreciation", "speculation", "trading_fees"
            ];
            
            for feature in &prohibited_features {
                if market_config.get(feature).is_some() {
                    return false;
                }
            }
            true
        }
    }
}

#[tokio::test]
async fn test_anthropic_tos_compliance() {
    use compliance_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = claude_market::Market::new(&db_path).await.unwrap();
    market.init_schema().await.unwrap();

    let mut auditor = ComplianceAuditor::new();
    println!("Testing Anthropic Terms of Service compliance...");

    let user_a = PeerId::random();
    let user_b = PeerId::random();
    let user_c = PeerId::random();

    // Test 1: No shared API keys - each user must use their own Claude account
    println!("✓ Verifying no shared API key requirement...");
    
    // Simulate user opt-in process with individual Claude accounts
    let tos_compliance_data = json!({
        "claude_account_owner": user_a.to_string(),
        "api_key_shared": false,
        "individual_subscription": true,
        "local_execution_only": true
    });
    
    auditor.log_event("user_opt_in", &user_a, tos_compliance_data.clone());
    
    // Verify each user opts in with their own account
    market.opt_in_compute_sharing(&user_a, true).await.unwrap();
    let user_a_status = market.get_opt_in_status(&user_a).await.unwrap();
    assert!(user_a_status.opted_in);
    assert!(user_a_status.individual_claude_account);
    assert!(!user_a_status.api_key_shared);
    
    // Test violation detection: attempt to use shared API key
    let violation_data = json!({
        "claude_account_owner": user_b.to_string(),
        "api_key_shared": true,  // VIOLATION
        "shared_with": [user_a.to_string()]
    });
    
    auditor.log_event("api_key_access", &user_b, violation_data);
    
    // System should reject shared API key usage
    let shared_key_result = market.validate_claude_account_usage(&user_b, &user_a).await;
    assert!(shared_key_result.is_err(), "Shared API key usage should be rejected");
    
    println!("✓ Shared API key usage properly rejected");

    // Test 2: Peer-orchestrated model (not central brokerage)
    println!("✓ Verifying peer-orchestrated model...");
    
    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "tos_compliance_test".to_string(),
        compute_units: 50,
        max_duration_secs: 300,
        required_capabilities: vec!["claude".to_string()],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: None,
    };

    // User B opts in properly with their own account
    market.opt_in_compute_sharing(&user_b, true).await.unwrap();
    market.opt_in_compute_sharing(&user_c, true).await.unwrap();

    // User A requests compute (peer-to-peer, not through central service)
    let request = market.place_order(
        claude_market::OrderType::RequestCompute,
        user_a,
        100,
        50,
        task_spec.clone(),
        None,
        None,
        None,
    ).await.unwrap();

    auditor.log_event("order_placement", &user_a, json!({
        "order_type": "RequestCompute",
        "peer_to_peer": true,
        "central_broker": false,
        "user_controls_execution": true
    }));

    // User B offers compute using their own Claude account
    let offer = market.place_order(
        claude_market::OrderType::OfferCompute,
        user_b,
        80,
        100,
        task_spec,
        None,
        None,
        None,
    ).await.unwrap();

    auditor.log_event("order_placement", &user_b, json!({
        "order_type": "OfferCompute",
        "own_claude_account": true,
        "voluntary_participation": true,
        "user_approved": true
    }));

    sleep(Duration::from_millis(100)).await;

    // Verify peer-to-peer assignment (not central execution)
    let assignments = market.get_assignments(None, 10).await.unwrap();
    if !assignments.is_empty() {
        let assignment = &assignments[0];
        assert_eq!(assignment.requester, user_a);
        assert_eq!(assignment.provider, user_b);
        
        auditor.log_event("task_execution", &user_b, json!({
            "execution_model": "peer_to_peer",
            "provider_claude_account": user_b.to_string(),
            "requester_account_access": false,
            "local_execution": true
        }));
        
        println!("✓ Peer-to-peer assignment created correctly");
    }

    // Test 3: Verify no account access resale
    println!("✓ Verifying no account access resale...");
    
    // Test that tokens reward contribution, not access resale
    let token_usage_data = json!({
        "purpose": "contribution_reward",
        "not_access_purchase": true,
        "voluntary_compute_donation": true,
        "folding_at_home_model": true
    });
    
    auditor.log_event("token_usage", &user_b, token_usage_data);
    
    // Verify token transaction is for computational contribution, not access
    assert!(market.validate_token_usage_compliance(&user_b).await.unwrap());
    
    println!("✓ Token usage complies with contribution model");

    // Test 4: User control and transparency
    println!("✓ Verifying user control and transparency...");
    
    if !assignments.is_empty() {
        let assignment = &assignments[0];
        
        // User should be able to see what tasks are running on their account
        let user_tasks = market.get_user_task_history(&user_b).await.unwrap();
        assert!(!user_tasks.is_empty());
        
        // User should be able to set limits
        market.set_user_limits(&user_b, claude_market::UserLimits {
            max_tasks_per_day: 10,
            max_compute_units_per_task: 100,
            allowed_task_types: vec!["safe_tasks".to_string()],
            auto_approve: false,
        }).await.unwrap();
        
        auditor.log_event("user_control", &user_b, json!({
            "task_visibility": true,
            "usage_limits_set": true,
            "auto_approve_disabled": true,
            "user_in_control": true
        }));
        
        println!("✓ User control and transparency verified");
    }

    // Generate compliance report
    let compliance_report = auditor.generate_compliance_report();
    compliance_report.print_summary();
    
    let violations = auditor.get_violations();
    println!("Compliance violations detected: {}", violations.len());
    
    for violation in violations {
        println!("  - {}: {:?}", violation.event_type, violation.compliance_flags);
    }

    // Ensure compliance
    assert_eq!(violations.len(), 1, "Should only have the intentional API key sharing violation");
    assert!(violations[0].compliance_flags.contains(&"API_KEY_VIOLATION".to_string()));
}

#[tokio::test]
async fn test_data_protection_compliance() {
    use compliance_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = claude_market::Market::new(&db_path).await.unwrap();
    market.init_schema().await.unwrap();

    let mut auditor = ComplianceAuditor::new();
    println!("Testing data protection compliance (GDPR/CCPA)...");

    let user = PeerId::random();
    let validator = PrivacyValidator;

    // Test 1: Lawful basis for processing (consent)
    println!("✓ Testing consent requirements...");
    
    let consent_data = json!({
        "processing_purpose": "compute_task_coordination",
        "data_types": ["peer_id", "task_preferences", "reputation_scores"],
        "consent_explicit": true,
        "consent_freely_given": true,
        "consent_specific": true,
        "consent_informed": true,
        "withdrawal_possible": true
    });
    
    auditor.log_event("consent_collection", &user, consent_data);
    
    // User must explicitly consent to data processing
    let opt_in_result = market.opt_in_with_consent(&user, claude_market::ConsentOptions {
        terms_accepted: true,
        privacy_policy_accepted: true,
        data_processing_consent: true,
        marketing_consent: false, // Optional
        analytics_consent: false, // Optional
    }).await.unwrap();
    
    let user_status = market.get_opt_in_status(&user).await.unwrap();
    assert!(validator.validate_consent_requirements(&user_status));
    
    println!("✓ Explicit consent properly collected");

    // Test 2: Data minimization principle
    println!("✓ Testing data minimization...");
    
    let task_data = json!({
        "task_type": "data_processing",
        "required_data": ["peer_id", "compute_capacity"],
        "optional_data": [],
        "personal_data": false
    });
    
    assert!(validator.validate_data_minimization(&task_data));
    
    // Test prohibited data collection
    let excessive_data = json!({
        "task_type": "data_processing", 
        "peer_id": user.to_string(),
        "ssn": "123-45-6789",  // Prohibited
        "credit_card": "4111-1111-1111-1111"  // Prohibited
    });
    
    assert!(!validator.validate_data_minimization(&excessive_data));
    
    auditor.log_event("data_processing", &user, task_data);
    
    println!("✓ Data minimization validated");

    // Test 3: Right to access (data portability)
    println!("✓ Testing right to access...");
    
    // Create some data for the user
    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "privacy_test".to_string(),
        compute_units: 25,
        max_duration_secs: 300,
        required_capabilities: vec![],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: None,
    };

    market.place_order(
        claude_market::OrderType::OfferCompute,
        user,
        100,
        50,
        task_spec,
        None,
        None,
        None,
    ).await.unwrap();

    // User should be able to export their data
    let user_data_export = market.export_user_data(&user).await.unwrap();
    
    // Verify export contains all user data
    assert!(user_data_export.orders.len() > 0);
    assert!(user_data_export.opt_in_status.opted_in);
    assert!(user_data_export.created_at.is_some());
    
    auditor.log_event("data_export", &user, json!({
        "export_requested": true,
        "data_provided": true,
        "machine_readable": true,
        "complete_export": true
    }));
    
    println!("✓ Data export functionality verified");

    // Test 4: Right to erasure (right to be forgotten)
    println!("✓ Testing right to erasure...");
    
    // User requests data deletion
    let deletion_result = market.delete_user_data(&user, claude_market::DeletionOptions {
        delete_orders: true,
        delete_assignments: true,
        delete_reputation: true,
        anonymize_completed_transactions: true,
    }).await.unwrap();
    
    assert!(deletion_result.success);
    assert!(deletion_result.deleted_records > 0);
    
    // Verify data is actually deleted
    let post_deletion_export = market.export_user_data(&user).await.unwrap();
    assert_eq!(post_deletion_export.orders.len(), 0);
    
    auditor.log_event("data_deletion", &user, json!({
        "deletion_requested": true,
        "data_deleted": true,
        "anonymization_completed": true,
        "records_deleted": deletion_result.deleted_records
    }));
    
    println!("✓ Data erasure functionality verified");

    // Test 5: Data retention policies
    println!("✓ Testing data retention compliance...");
    
    let test_user_2 = PeerId::random();
    market.opt_in_compute_sharing(&test_user_2, true).await.unwrap();
    
    let retention_test_data = json!({
        "created_at": (Utc::now() - chrono::Duration::days(400)).to_rfc3339(), // Old data
        "retention_period_days": 365,
        "auto_deletion_enabled": true
    });
    
    auditor.log_event("retention_check", &test_user_2, retention_test_data);
    
    // Run retention policy cleanup
    let cleanup_result = market.apply_retention_policies().await.unwrap();
    assert!(cleanup_result.records_processed > 0);
    
    println!("✓ Data retention policies applied");

    // Test 6: Data breach notification readiness
    println!("✓ Testing breach notification capabilities...");
    
    let breach_response = market.test_breach_notification_system().await.unwrap();
    assert!(breach_response.notification_ready);
    assert!(breach_response.user_contacts_available);
    assert!(breach_response.authorities_contact_list.len() > 0);
    
    auditor.log_event("breach_preparedness", &user, json!({
        "notification_system_ready": true,
        "response_time_target": "72_hours",
        "authorities_identified": true
    }));
    
    println!("✓ Breach notification readiness verified");

    // Generate compliance report
    let compliance_report = auditor.generate_compliance_report();
    compliance_report.print_summary();
    
    let violations = auditor.get_violations();
    assert_eq!(violations.len(), 1); // Only the data export flag for review
    
    println!("✓ Data protection compliance verified");
}

#[tokio::test]
async fn test_financial_compliance() {
    use compliance_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = claude_market::Market::new(&db_path).await.unwrap();
    market.init_schema().await.unwrap();

    let mut auditor = ComplianceAuditor::new();
    println!("Testing financial compliance (no money transmission)...");

    let user_a = PeerId::random();
    let user_b = PeerId::random();
    let validator = TokenComplianceValidator;

    market.opt_in_compute_sharing(&user_a, true).await.unwrap();
    market.opt_in_compute_sharing(&user_b, true).await.unwrap();

    // Test 1: Utility token model (not securities)
    println!("✓ Testing utility token compliance...");
    
    let utility_config = json!({
        "token_purpose": "computational_resource_access",
        "not_investment": true,
        "no_profit_sharing": true,
        "no_voting_rights": true,
        "utility_only": true,
        "consumable": true
    });
    
    assert!(validator.validate_no_investment_features(&utility_config));
    
    auditor.log_event("token_model_validation", &user_a, utility_config);
    
    // Test invalid investment-like features
    let investment_config = json!({
        "token_purpose": "investment",
        "dividend": true,  // Prohibited
        "profit_sharing": true,  // Prohibited
        "appreciation": true  // Prohibited
    });
    
    assert!(!validator.validate_no_investment_features(&investment_config));
    
    println!("✓ Investment features properly rejected");

    // Test 2: No money transmission
    println!("✓ Testing money transmission compliance...");
    
    // Valid utility token transaction for computational resources
    let valid_transaction = claude_market::TokenTransaction {
        id: Uuid::new_v4(),
        from_user: user_a,
        to_user: user_b,
        amount: 100,
        transaction_type: claude_market::TokenTransactionType::ComputePayment,
        computational_resource: Some("code_generation_task".to_string()),
        is_external_payment: false,
        fiat_currency_involved: false,
        created_at: Utc::now(),
    };
    
    assert!(validator.validate_not_money_transmission(&valid_transaction));
    
    auditor.log_event("token_transaction", &user_a, json!({
        "transaction_type": "ComputePayment",
        "utility_purpose": true,
        "no_fiat_exchange": true,
        "computational_resource": "code_generation_task"
    }));

    // Invalid money transmission-like transaction
    let money_transmission = claude_market::TokenTransaction {
        id: Uuid::new_v4(),
        from_user: user_a,
        to_user: user_b,
        amount: 50000, // Large amount
        transaction_type: claude_market::TokenTransactionType::CurrencyExchange, // Prohibited
        computational_resource: None,
        is_external_payment: true, // Prohibited
        fiat_currency_involved: true, // Prohibited
        created_at: Utc::now(),
    };
    
    assert!(!validator.validate_not_money_transmission(&money_transmission));
    
    println!("✓ Money transmission properly rejected");

    // Test 3: Transaction limits and patterns
    println!("✓ Testing transaction limits...");
    
    // Test reasonable transaction limits
    let transaction_limits = market.get_transaction_limits().await.unwrap();
    assert!(transaction_limits.max_single_transaction < 1000);
    assert!(transaction_limits.max_daily_volume < 10000);
    assert!(transaction_limits.requires_identity_verification == false);
    
    auditor.log_event("transaction_limits", &user_a, json!({
        "max_single": transaction_limits.max_single_transaction,
        "max_daily": transaction_limits.max_daily_volume,
        "reasonable_limits": true,
        "no_kyc_required": true
    }));

    // Test 4: No currency exchange features
    println!("✓ Testing no currency exchange...");
    
    let exchange_attempt = market.attempt_currency_exchange(&user_a, 100, "USD").await;
    assert!(exchange_attempt.is_err(), "Currency exchange should not be available");
    
    let fiat_conversion_attempt = market.get_fiat_conversion_rate("RUV", "USD").await;
    assert!(fiat_conversion_attempt.is_err(), "Fiat conversion should not be available");
    
    auditor.log_event("currency_exchange_test", &user_a, json!({
        "exchange_available": false,
        "fiat_conversion_available": false,
        "utility_only": true
    }));
    
    println!("✓ Currency exchange properly unavailable");

    // Test 5: Regulatory disclosure compliance
    println!("✓ Testing regulatory disclosures...");
    
    let disclosures = market.get_regulatory_disclosures().await.unwrap();
    assert!(disclosures.contains_key("not_securities"));
    assert!(disclosures.contains_key("utility_tokens_only"));
    assert!(disclosures.contains_key("no_money_transmission"));
    assert!(disclosures.contains_key("computational_resources_only"));
    
    auditor.log_event("regulatory_disclosure", &user_a, json!({
        "disclosures_present": true,
        "utility_token_disclosure": true,
        "no_securities_disclaimer": true,
        "computational_purpose_clear": true
    }));
    
    println!("✓ Regulatory disclosures verified");

    // Generate compliance report
    let compliance_report = auditor.generate_compliance_report();
    compliance_report.print_summary();
    
    let violations = auditor.get_violations();
    assert_eq!(violations.len(), 0, "No financial compliance violations should exist");
    
    println!("✓ Financial compliance verified");
}

#[tokio::test]
async fn test_audit_trail_requirements() {
    use compliance_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = claude_market::Market::new(&db_path).await.unwrap();
    market.init_schema().await.unwrap();

    let mut auditor = ComplianceAuditor::new();
    println!("Testing audit trail requirements...");

    let user_a = PeerId::random();
    let user_b = PeerId::random();

    market.opt_in_compute_sharing(&user_a, true).await.unwrap();
    market.opt_in_compute_sharing(&user_b, true).await.unwrap();

    // Test 1: Complete transaction audit trail
    println!("✓ Testing transaction audit trail...");
    
    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "audit_test".to_string(),
        compute_units: 50,
        max_duration_secs: 300,
        required_capabilities: vec![],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: None,
    };

    // Track order placement
    let request_start = Instant::now();
    let request = market.place_order(
        claude_market::OrderType::RequestCompute,
        user_a,
        100,
        50,
        task_spec.clone(),
        None,
        None,
        None,
    ).await.unwrap();
    
    auditor.log_event("order_placement", &user_a, json!({
        "order_id": request.id,
        "order_type": "RequestCompute",
        "timestamp": Utc::now(),
        "user_ip": "masked_for_privacy",
        "user_agent": "synaptic_mesh_client",
        "processing_time_ms": request_start.elapsed().as_millis()
    }));

    let offer = market.place_order(
        claude_market::OrderType::OfferCompute,
        user_b,
        80,
        100,
        task_spec,
        None,
        None,
        None,
    ).await.unwrap();
    
    auditor.log_event("order_placement", &user_b, json!({
        "order_id": offer.id,
        "order_type": "OfferCompute",
        "timestamp": Utc::now()
    }));

    sleep(Duration::from_millis(100)).await;

    // Track assignment creation
    let assignments = market.get_assignments(None, 10).await.unwrap();
    if !assignments.is_empty() {
        let assignment = &assignments[0];
        
        auditor.log_event("assignment_created", &user_a, json!({
            "assignment_id": assignment.id,
            "requester": assignment.requester,
            "provider": assignment.provider,
            "price_agreed": assignment.price_per_unit,
            "compute_units": assignment.compute_units,
            "total_cost": assignment.total_cost,
            "timestamp": assignment.assigned_at
        }));

        // Track task execution
        market.start_task(&assignment.id, &user_b).await.unwrap();
        
        auditor.log_event("task_started", &user_b, json!({
            "assignment_id": assignment.id,
            "started_at": Utc::now(),
            "provider": user_b
        }));

        // Complete task
        let quality_scores = HashMap::from([
            ("accuracy".to_string(), 0.95),
            ("completeness".to_string(), 0.90),
        ]);
        
        market.complete_task(&assignment.id, &user_b, quality_scores.clone()).await.unwrap();
        
        auditor.log_event("task_completed", &user_b, json!({
            "assignment_id": assignment.id,
            "completed_at": Utc::now(),
            "quality_scores": quality_scores,
            "provider": user_b
        }));
    }

    // Test 2: Audit log integrity
    println!("✓ Testing audit log integrity...");
    
    let audit_integrity = market.verify_audit_log_integrity().await.unwrap();
    assert!(audit_integrity.checksum_valid);
    assert!(audit_integrity.no_gaps_detected);
    assert!(audit_integrity.timestamps_sequential);
    
    auditor.log_event("audit_verification", &user_a, json!({
        "integrity_check": "passed",
        "checksum_valid": true,
        "sequential_timestamps": true,
        "total_entries": audit_integrity.total_entries
    }));

    // Test 3: Regulatory reporting capabilities
    println!("✓ Testing regulatory reporting...");
    
    let regulatory_report = market.generate_regulatory_report(
        Utc::now() - chrono::Duration::days(30),
        Utc::now()
    ).await.unwrap();
    
    assert!(regulatory_report.total_transactions > 0);
    assert!(regulatory_report.total_volume > 0);
    assert!(regulatory_report.unique_users >= 2);
    assert!(regulatory_report.compliance_issues.is_empty());
    
    auditor.log_event("regulatory_report", &user_a, json!({
        "report_generated": true,
        "period_covered": "30_days",
        "transactions": regulatory_report.total_transactions,
        "compliance_clean": regulatory_report.compliance_issues.is_empty()
    }));

    // Test 4: Data retention and archival
    println!("✓ Testing audit data retention...");
    
    let retention_policy = market.get_audit_retention_policy().await.unwrap();
    assert!(retention_policy.retention_years >= 7); // Common regulatory requirement
    assert!(retention_policy.immutable_storage);
    assert!(retention_policy.encrypted_at_rest);
    
    auditor.log_event("retention_policy", &user_a, json!({
        "retention_years": retention_policy.retention_years,
        "immutable_storage": true,
        "encrypted": true,
        "compliant": true
    }));

    // Test 5: Access controls for audit data
    println!("✓ Testing audit access controls...");
    
    // Test that regular users cannot access others' audit data
    let unauthorized_access = market.get_user_audit_trail(&user_a, &user_b).await;
    assert!(unauthorized_access.is_err(), "Users should not access others' audit data");
    
    // Test that users can access their own audit data
    let authorized_access = market.get_user_audit_trail(&user_a, &user_a).await;
    assert!(authorized_access.is_ok(), "Users should access their own audit data");
    
    auditor.log_event("access_control_test", &user_a, json!({
        "unauthorized_access_blocked": true,
        "self_access_allowed": true,
        "privacy_maintained": true
    }));

    // Test 6: Audit log export for compliance
    println!("✓ Testing audit log export...");
    
    let audit_export = market.export_audit_logs(
        Utc::now() - chrono::Duration::days(1),
        Utc::now(),
        claude_market::AuditExportFormat::JSON
    ).await.unwrap();
    
    assert!(!audit_export.logs.is_empty());
    assert!(audit_export.signature_valid);
    assert!(audit_export.export_timestamp.is_some());
    
    auditor.log_event("audit_export", &user_a, json!({
        "export_successful": true,
        "format": "JSON",
        "entries_exported": audit_export.logs.len(),
        "signature_valid": true
    }));

    // Generate final compliance report
    let compliance_report = auditor.generate_compliance_report();
    compliance_report.print_summary();
    
    // Verify audit trail completeness
    let audit_events = [
        "order_placement", "assignment_created", "task_started", 
        "task_completed", "audit_verification", "regulatory_report"
    ];
    
    for event_type in &audit_events {
        let event_found = auditor.audit_log.iter()
            .any(|event| event.event_type == *event_type);
        assert!(event_found, "Missing audit event: {}", event_type);
    }
    
    println!("✓ Complete audit trail verified");
    println!("✓ All compliance requirements validated");
}

#[tokio::test]
async fn test_end_to_end_compliance_workflow() {
    use compliance_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = claude_market::Market::new(&db_path).await.unwrap();
    market.init_schema().await.unwrap();

    let mut auditor = ComplianceAuditor::new();
    println!("Testing end-to-end compliance workflow...");

    let requester = PeerId::random();
    let provider = PeerId::random();

    // Phase 1: Compliant user onboarding
    println!("Phase 1: User onboarding with compliance...");
    
    // Both users must consent and verify their Claude accounts
    let onboarding_data = json!({
        "claude_account_verified": true,
        "individual_subscription": true,
        "terms_accepted": true,
        "privacy_policy_accepted": true,
        "age_verified": true,
        "jurisdiction": "US"
    });
    
    auditor.log_event("user_onboarding", &requester, onboarding_data.clone());
    auditor.log_event("user_onboarding", &provider, onboarding_data);
    
    market.opt_in_with_full_compliance(&requester, claude_market::ComplianceOptions {
        verify_claude_account: true,
        accept_terms: true,
        accept_privacy_policy: true,
        enable_audit_trail: true,
        jurisdiction: "US".to_string(),
    }).await.unwrap();
    
    market.opt_in_with_full_compliance(&provider, claude_market::ComplianceOptions {
        verify_claude_account: true,
        accept_terms: true,
        accept_privacy_policy: true,
        enable_audit_trail: true,
        jurisdiction: "US".to_string(),
    }).await.unwrap();

    // Phase 2: Compliant task posting
    println!("Phase 2: Compliant task posting...");
    
    let compliant_task = claude_market::ComputeTaskSpec {
        task_type: "code_generation".to_string(),
        compute_units: 100,
        max_duration_secs: 600,
        required_capabilities: vec!["rust".to_string()],
        min_reputation: Some(70.0),
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: Some(b"encrypted_task_data".to_vec()),
    };
    
    // Validate task compliance before posting
    let task_compliance = market.validate_task_compliance(&compliant_task).await.unwrap();
    assert!(task_compliance.anthropic_tos_compliant);
    assert!(task_compliance.data_protection_compliant);
    assert!(task_compliance.no_prohibited_content);
    
    auditor.log_event("task_validation", &requester, json!({
        "tos_compliant": true,
        "data_protection_compliant": true,
        "content_appropriate": true,
        "encrypted_properly": true
    }));

    let request = market.place_order(
        claude_market::OrderType::RequestCompute,
        requester,
        150,
        100,
        compliant_task.clone(),
        None,
        None,
        None,
    ).await.unwrap();

    let offer = market.place_order(
        claude_market::OrderType::OfferCompute,
        provider,
        120,
        200,
        compliant_task,
        None,
        None,
        None,
    ).await.unwrap();

    sleep(Duration::from_millis(100)).await;

    // Phase 3: Compliant task execution
    println!("Phase 3: Compliant task execution...");
    
    let assignments = market.get_assignments(None, 10).await.unwrap();
    assert!(!assignments.is_empty());
    let assignment = &assignments[0];
    
    // Verify execution compliance
    let execution_compliance = market.validate_execution_compliance(&assignment.id).await.unwrap();
    assert!(execution_compliance.provider_owns_claude_account);
    assert!(execution_compliance.no_shared_credentials);
    assert!(execution_compliance.local_execution_only);
    
    auditor.log_event("execution_validation", &provider, json!({
        "own_claude_account": true,
        "no_shared_creds": true,
        "local_execution": true,
        "user_approved": true
    }));

    // Provider executes on their own Claude account
    market.start_task(&assignment.id, &provider).await.unwrap();
    
    // Simulate compliant execution
    sleep(Duration::from_millis(200)).await;
    
    let quality_scores = HashMap::from([
        ("accuracy".to_string(), 0.92),
        ("compliance".to_string(), 1.0),
        ("security".to_string(), 0.95),
    ]);
    
    market.complete_task(&assignment.id, &provider, quality_scores).await.unwrap();

    // Phase 4: Compliant settlement
    println!("Phase 4: Compliant token settlement...");
    
    let settlement = market.get_task_settlement(&assignment.id).await.unwrap();
    
    // Verify settlement compliance
    let settlement_compliance = market.validate_settlement_compliance(&settlement).await.unwrap();
    assert!(settlement_compliance.utility_tokens_only);
    assert!(settlement_compliance.no_money_transmission);
    assert!(settlement_compliance.computational_purpose);
    assert!(!settlement_compliance.investment_features);
    
    auditor.log_event("settlement_validation", &requester, json!({
        "utility_tokens": true,
        "no_money_transmission": true,
        "computational_purpose": true,
        "compliant_settlement": true
    }));

    // Phase 5: Compliance reporting
    println!("Phase 5: Generating compliance reports...");
    
    let final_compliance_report = market.generate_comprehensive_compliance_report().await.unwrap();
    
    assert!(final_compliance_report.anthropic_tos_compliance.score > 95.0);
    assert!(final_compliance_report.data_protection_compliance.score > 95.0);
    assert!(final_compliance_report.financial_compliance.score > 95.0);
    assert!(final_compliance_report.audit_trail_compliance.score > 95.0);
    assert!(final_compliance_report.overall_compliance_score > 95.0);
    
    auditor.log_event("compliance_report", &requester, json!({
        "overall_score": final_compliance_report.overall_compliance_score,
        "anthropic_tos": final_compliance_report.anthropic_tos_compliance.score,
        "data_protection": final_compliance_report.data_protection_compliance.score,
        "financial": final_compliance_report.financial_compliance.score,
        "audit_trail": final_compliance_report.audit_trail_compliance.score,
        "violations": final_compliance_report.violations.len()
    }));

    // Final compliance validation
    let compliance_summary = auditor.generate_compliance_report();
    compliance_summary.print_summary();
    
    let violations = auditor.get_violations();
    assert_eq!(violations.len(), 0, "No compliance violations in end-to-end workflow");
    
    println!("✓ End-to-end compliance workflow completed successfully");
    println!("✓ Overall compliance score: {:.2}%", final_compliance_report.overall_compliance_score);
    
    // Performance compliance validation
    assert!(final_compliance_report.performance_metrics.avg_response_time < Duration::from_millis(1000));
    assert!(final_compliance_report.performance_metrics.uptime_percentage > 99.0);
    
    println!("✓ Performance compliance validated");
    println!("✓ All compliance requirements met");
}