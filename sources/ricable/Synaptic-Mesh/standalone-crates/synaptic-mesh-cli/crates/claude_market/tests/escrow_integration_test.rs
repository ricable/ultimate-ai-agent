//! Integration tests for the escrow system

use claude_market::escrow::*;
use claude_market::wallet::Wallet;
use claude_market::error::Result;
use ed25519_dalek::SigningKey;
use libp2p::PeerId;
use rand::rngs::OsRng;
use std::sync::Arc;
use uuid::Uuid;
use chrono::{Utc, Duration};

async fn setup_test_env() -> (Arc<Wallet>, Escrow) {
    let wallet = Arc::new(Wallet::new(":memory:").await.unwrap());
    wallet.init_schema().await.unwrap();
    
    let escrow = Escrow::new(":memory:", wallet.clone()).await.unwrap();
    escrow.init_schema().await.unwrap();
    
    (wallet, escrow)
}

#[tokio::test]
async fn test_edge_case_zero_amount() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    
    // Should fail with zero amount
    let result = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            0, // Zero amount
            MultiSigType::Single,
            vec![],
            60,
        )
        .await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("non-zero"));
}

#[tokio::test]
async fn test_edge_case_same_party() {
    let (wallet, escrow) = setup_test_env().await;
    
    let peer = PeerId::random();
    
    // Should fail when requester and provider are the same
    let result = escrow
        .create_escrow(
            Uuid::new_v4(),
            peer,
            peer, // Same as requester
            100,
            MultiSigType::Single,
            vec![],
            60,
        )
        .await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("same party"));
}

#[tokio::test]
async fn test_insufficient_balance() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    
    // Credit less than needed
    wallet.credit(&requester, 50).await.unwrap();
    
    let agreement = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            100, // More than available
            MultiSigType::Single,
            vec![],
            60,
        )
        .await
        .unwrap();
    
    let key = SigningKey::generate(&mut OsRng);
    let result = escrow.fund_escrow(&agreement.id, &requester, &key).await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Insufficient balance"));
}

#[tokio::test]
async fn test_time_locked_escrow() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    let amount = 100;
    
    wallet.credit(&requester, amount).await.unwrap();
    
    // Create time-locked escrow
    let release_time = Utc::now() + Duration::seconds(2);
    let agreement = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::TimeLocked { release_after: release_time },
            vec![],
            60,
        )
        .await
        .unwrap();
    
    // Fund and complete
    let req_key = SigningKey::generate(&mut OsRng);
    let prov_key = SigningKey::generate(&mut OsRng);
    
    escrow.fund_escrow(&agreement.id, &requester, &req_key).await.unwrap();
    escrow.mark_completed(&agreement.id, &provider, &prov_key).await.unwrap();
    
    // Try to release before time lock
    let release = ReleaseAuth {
        escrow_id: agreement.id,
        authorizer: requester,
        decision: ReleaseDecision::ApproveRelease,
        signature: vec![0; 64],
        created_at: Utc::now(),
    };
    
    let result = escrow.release_funds(&agreement.id, release.clone()).await.unwrap();
    // Should still be completed, not released
    assert_eq!(result.state, EscrowState::Completed);
    
    // Wait for time lock to expire
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    
    // Now it should release
    let result = escrow.release_funds(&agreement.id, release).await.unwrap();
    assert_eq!(result.state, EscrowState::Released);
}

#[tokio::test]
async fn test_multiple_arbitrators() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    let arb1 = PeerId::random();
    let arb2 = PeerId::random();
    let arb3 = PeerId::random();
    let amount = 300;
    
    wallet.credit(&requester, amount).await.unwrap();
    
    // Create escrow requiring 2-of-3 arbitrators
    let agreement = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::Arbitrators { required: 2, total: 3 },
            vec![arb1, arb2, arb3],
            60,
        )
        .await
        .unwrap();
    
    // Fund and raise dispute
    let req_key = SigningKey::generate(&mut OsRng);
    escrow.fund_escrow(&agreement.id, &requester, &req_key).await.unwrap();
    
    let agreement = escrow
        .raise_dispute(
            &agreement.id,
            &requester,
            "Quality issues".to_string(),
            vec![],
            &req_key,
        )
        .await
        .unwrap();
    
    // First arbitrator decides
    let arb1_key = SigningKey::generate(&mut OsRng);
    let result = escrow
        .resolve_dispute(
            &agreement.id,
            &arb1,
            DisputeOutcome::RefundToRequester,
            "Work incomplete".to_string(),
            &arb1_key,
        )
        .await;
    
    // Should fail - need 2 arbitrators
    assert!(result.is_err());
    
    // Add second arbitrator with different decision
    let arb2_key = SigningKey::generate(&mut OsRng);
    let agreement = escrow
        .resolve_dispute(
            &agreement.id,
            &arb2,
            DisputeOutcome::RefundToRequester,
            "Agree with arb1".to_string(),
            &arb2_key,
        )
        .await
        .unwrap();
    
    // Should be resolved now
    assert_eq!(agreement.state, EscrowState::Refunded);
}

#[tokio::test]
async fn test_concurrent_releases() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    let amount = 100;
    
    wallet.credit(&requester, amount * 2).await.unwrap();
    
    // Create two escrows
    let agreement1 = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::Single,
            vec![],
            60,
        )
        .await
        .unwrap();
        
    let agreement2 = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::Single,
            vec![],
            60,
        )
        .await
        .unwrap();
    
    // Fund both
    let key = SigningKey::generate(&mut OsRng);
    escrow.fund_escrow(&agreement1.id, &requester, &key).await.unwrap();
    escrow.fund_escrow(&agreement2.id, &requester, &key).await.unwrap();
    
    // Complete both
    let prov_key = SigningKey::generate(&mut OsRng);
    escrow.mark_completed(&agreement1.id, &provider, &prov_key).await.unwrap();
    escrow.mark_completed(&agreement2.id, &provider, &prov_key).await.unwrap();
    
    // Release both
    let release1 = ReleaseAuth {
        escrow_id: agreement1.id,
        authorizer: requester,
        decision: ReleaseDecision::ApproveRelease,
        signature: vec![0; 64],
        created_at: Utc::now(),
    };
    
    let release2 = ReleaseAuth {
        escrow_id: agreement2.id,
        authorizer: provider,
        decision: ReleaseDecision::ApproveRelease,
        signature: vec![0; 64],
        created_at: Utc::now(),
    };
    
    escrow.release_funds(&agreement1.id, release1).await.unwrap();
    escrow.release_funds(&agreement2.id, release2).await.unwrap();
    
    // Check balances
    let prov_balance = wallet.get_balance(&provider).await.unwrap();
    assert_eq!(prov_balance.available, amount * 2);
}

#[tokio::test]
async fn test_dispute_evidence_tracking() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    let amount = 100;
    
    wallet.credit(&requester, amount).await.unwrap();
    
    let agreement = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::Single,
            vec![],
            60,
        )
        .await
        .unwrap();
    
    // Fund and complete
    let req_key = SigningKey::generate(&mut OsRng);
    let prov_key = SigningKey::generate(&mut OsRng);
    
    escrow.fund_escrow(&agreement.id, &requester, &req_key).await.unwrap();
    escrow.mark_completed(&agreement.id, &provider, &prov_key).await.unwrap();
    
    // Raise dispute with evidence
    let evidence = vec![
        "screenshot1.jpg".to_string(),
        "log_file.txt".to_string(),
        "correspondence.pdf".to_string(),
    ];
    
    let agreement = escrow
        .raise_dispute(
            &agreement.id,
            &requester,
            "Work not as described".to_string(),
            evidence.clone(),
            &req_key,
        )
        .await
        .unwrap();
    
    // Verify dispute info
    assert!(agreement.dispute_info.is_some());
    let dispute = agreement.dispute_info.unwrap();
    assert_eq!(dispute.raised_by, requester);
    assert_eq!(dispute.evidence, evidence);
    assert!(dispute.reason.contains("not as described"));
}

#[tokio::test]
async fn test_audit_trail_completeness() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    let arbitrator = PeerId::random();
    let amount = 100;
    
    wallet.credit(&requester, amount).await.unwrap();
    
    // Create escrow
    let agreement = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::Single,
            vec![arbitrator],
            60,
        )
        .await
        .unwrap();
    
    let escrow_id = agreement.id;
    
    // Perform various operations
    let req_key = SigningKey::generate(&mut OsRng);
    let prov_key = SigningKey::generate(&mut OsRng);
    
    escrow.fund_escrow(&escrow_id, &requester, &req_key).await.unwrap();
    escrow.mark_completed(&escrow_id, &provider, &prov_key).await.unwrap();
    escrow.raise_dispute(&escrow_id, &requester, "Test".to_string(), vec![], &req_key).await.unwrap();
    
    // Get audit log
    let audit_log = escrow.get_audit_log(&escrow_id).await.unwrap();
    
    // Verify all operations are logged
    assert!(audit_log.len() >= 4);
    
    let actions: Vec<String> = audit_log.iter().map(|e| e.action.clone()).collect();
    assert!(actions.contains(&"create_escrow".to_string()));
    assert!(actions.contains(&"fund_escrow".to_string()));
    assert!(actions.contains(&"mark_completed".to_string()));
    assert!(actions.contains(&"raise_dispute".to_string()));
    
    // Verify actors are correct
    let create_entry = audit_log.iter().find(|e| e.action == "create_escrow").unwrap();
    assert_eq!(create_entry.actor, requester);
    
    let complete_entry = audit_log.iter().find(|e| e.action == "mark_completed").unwrap();
    assert_eq!(complete_entry.actor, provider);
}

#[tokio::test]
async fn test_state_transitions() {
    let (wallet, escrow) = setup_test_env().await;
    
    let requester = PeerId::random();
    let provider = PeerId::random();
    let amount = 100;
    
    wallet.credit(&requester, amount).await.unwrap();
    
    let agreement = escrow
        .create_escrow(
            Uuid::new_v4(),
            requester,
            provider,
            amount,
            MultiSigType::Single,
            vec![],
            60,
        )
        .await
        .unwrap();
    
    let key = SigningKey::generate(&mut OsRng);
    
    // Try to complete before funding (should fail)
    let result = escrow.mark_completed(&agreement.id, &provider, &key).await;
    assert!(result.is_err());
    
    // Fund escrow
    escrow.fund_escrow(&agreement.id, &requester, &key).await.unwrap();
    
    // Try to fund again (should fail)
    let result = escrow.fund_escrow(&agreement.id, &requester, &key).await;
    assert!(result.is_err());
    
    // Complete
    escrow.mark_completed(&agreement.id, &provider, &key).await.unwrap();
    
    // Try to complete again (should fail)
    let result = escrow.mark_completed(&agreement.id, &provider, &key).await;
    assert!(result.is_err());
}