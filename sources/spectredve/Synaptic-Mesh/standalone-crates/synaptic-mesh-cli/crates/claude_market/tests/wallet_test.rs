//! Comprehensive tests for the token wallet system

use claude_market::wallet::{TokenBalance, TokenTransfer, Wallet};
use claude_market::error::MarketError;
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Verifier};
use libp2p::PeerId;
use rand::rngs::OsRng;
use tempfile::tempdir;
use tokio;

/// Helper function to create a test wallet with a temporary database
async fn create_test_wallet() -> Wallet {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_wallet.db");
    let wallet = Wallet::new(db_path.to_str().unwrap()).await.unwrap();
    wallet.init_schema().await.unwrap();
    wallet
}

#[tokio::test]
async fn test_wallet_initialization() {
    let wallet = create_test_wallet().await;
    // Should not panic - schema should be created successfully
    wallet.init_schema().await.unwrap();
}

#[tokio::test]
async fn test_balance_operations() {
    let wallet = create_test_wallet().await;
    let peer_id = PeerId::random();
    
    // Initial balance should be zero
    let balance = wallet.get_balance(&peer_id).await.unwrap();
    assert_eq!(balance.available, 0);
    assert_eq!(balance.locked, 0);
    assert_eq!(balance.total, 0);
    
    // Credit tokens
    let balance = wallet.credit(&peer_id, 1000).await.unwrap();
    assert_eq!(balance.available, 1000);
    assert_eq!(balance.locked, 0);
    assert_eq!(balance.total, 1000);
    
    // Credit more tokens
    let balance = wallet.credit(&peer_id, 500).await.unwrap();
    assert_eq!(balance.available, 1500);
    assert_eq!(balance.total, 1500);
    
    // Debit tokens
    let balance = wallet.debit(&peer_id, 300).await.unwrap();
    assert_eq!(balance.available, 1200);
    assert_eq!(balance.total, 1200);
}

#[tokio::test]
async fn test_insufficient_balance_debit() {
    let wallet = create_test_wallet().await;
    let peer_id = PeerId::random();
    
    // Credit some tokens
    wallet.credit(&peer_id, 100).await.unwrap();
    
    // Try to debit more than available
    let result = wallet.debit(&peer_id, 150).await;
    match result {
        Err(MarketError::InsufficientBalance { required, available }) => {
            assert_eq!(required, 150);
            assert_eq!(available, 100);
        }
        _ => panic!("Expected InsufficientBalance error"),
    }
}

#[tokio::test]
async fn test_lock_unlock_tokens() {
    let wallet = create_test_wallet().await;
    let peer_id = PeerId::random();
    
    // Credit tokens
    wallet.credit(&peer_id, 1000).await.unwrap();
    
    // Lock some tokens
    let balance = wallet.lock_tokens(&peer_id, 400).await.unwrap();
    assert_eq!(balance.available, 600);
    assert_eq!(balance.locked, 400);
    assert_eq!(balance.total, 1000);
    
    // Lock more tokens
    let balance = wallet.lock_tokens(&peer_id, 200).await.unwrap();
    assert_eq!(balance.available, 400);
    assert_eq!(balance.locked, 600);
    assert_eq!(balance.total, 1000);
    
    // Unlock some tokens
    let balance = wallet.unlock_tokens(&peer_id, 300).await.unwrap();
    assert_eq!(balance.available, 700);
    assert_eq!(balance.locked, 300);
    assert_eq!(balance.total, 1000);
    
    // Unlock remaining tokens
    let balance = wallet.unlock_tokens(&peer_id, 300).await.unwrap();
    assert_eq!(balance.available, 1000);
    assert_eq!(balance.locked, 0);
    assert_eq!(balance.total, 1000);
}

#[tokio::test]
async fn test_insufficient_balance_lock() {
    let wallet = create_test_wallet().await;
    let peer_id = PeerId::random();
    
    // Credit some tokens
    wallet.credit(&peer_id, 100).await.unwrap();
    
    // Try to lock more than available
    let result = wallet.lock_tokens(&peer_id, 150).await;
    match result {
        Err(MarketError::InsufficientBalance { required, available }) => {
            assert_eq!(required, 150);
            assert_eq!(available, 100);
        }
        _ => panic!("Expected InsufficientBalance error"),
    }
}

#[tokio::test]
async fn test_unlock_more_than_locked() {
    let wallet = create_test_wallet().await;
    let peer_id = PeerId::random();
    
    // Credit and lock some tokens
    wallet.credit(&peer_id, 1000).await.unwrap();
    wallet.lock_tokens(&peer_id, 300).await.unwrap();
    
    // Try to unlock more than locked
    let result = wallet.unlock_tokens(&peer_id, 400).await;
    assert!(matches!(result, Err(MarketError::Internal(_))));
}

#[tokio::test]
async fn test_peer_to_peer_transfer() {
    let wallet = create_test_wallet().await;
    let sender = PeerId::random();
    let recipient = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    
    // Credit sender
    wallet.credit(&sender, 1000).await.unwrap();
    
    // Transfer tokens
    let transfer = wallet
        .transfer(&sender, &recipient, 250, Some("Payment for compute".to_string()), &signing_key)
        .await
        .unwrap();
    
    // Verify transfer details
    assert_eq!(transfer.from, sender);
    assert_eq!(transfer.to, recipient);
    assert_eq!(transfer.amount, 250);
    assert_eq!(transfer.memo, Some("Payment for compute".to_string()));
    assert!(!transfer.signature.is_empty());
    
    // Verify balances
    let sender_balance = wallet.get_balance(&sender).await.unwrap();
    let recipient_balance = wallet.get_balance(&recipient).await.unwrap();
    assert_eq!(sender_balance.available, 750);
    assert_eq!(recipient_balance.available, 250);
}

#[tokio::test]
async fn test_transfer_signature_verification() {
    let wallet = create_test_wallet().await;
    let sender = PeerId::random();
    let recipient = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    
    // Credit sender
    wallet.credit(&sender, 1000).await.unwrap();
    
    // Transfer tokens
    let transfer = wallet
        .transfer(&sender, &recipient, 100, Some("Test transfer".to_string()), &signing_key)
        .await
        .unwrap();
    
    // Verify the signature
    let message = format!(
        "{}:{}:{}:{}:{}",
        transfer.id,
        transfer.from,
        transfer.to,
        transfer.amount,
        transfer.memo.as_ref().unwrap_or(&String::new())
    );
    
    let signature = Signature::from_bytes(&transfer.signature.try_into().unwrap());
    assert!(verifying_key.verify(message.as_bytes(), &signature).is_ok());
}

#[tokio::test]
async fn test_transfer_insufficient_balance() {
    let wallet = create_test_wallet().await;
    let sender = PeerId::random();
    let recipient = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    
    // Credit sender with less than transfer amount
    wallet.credit(&sender, 100).await.unwrap();
    
    // Try to transfer more than available
    let result = wallet
        .transfer(&sender, &recipient, 150, None, &signing_key)
        .await;
    
    match result {
        Err(MarketError::InsufficientBalance { required, available }) => {
            assert_eq!(required, 150);
            assert_eq!(available, 100);
        }
        _ => panic!("Expected InsufficientBalance error"),
    }
}

#[tokio::test]
async fn test_transfer_history() {
    let wallet = create_test_wallet().await;
    let peer1 = PeerId::random();
    let peer2 = PeerId::random();
    let peer3 = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    
    // Credit peer1
    wallet.credit(&peer1, 1000).await.unwrap();
    
    // Make several transfers
    wallet.transfer(&peer1, &peer2, 100, Some("Transfer 1".to_string()), &signing_key).await.unwrap();
    wallet.transfer(&peer1, &peer3, 150, Some("Transfer 2".to_string()), &signing_key).await.unwrap();
    wallet.transfer(&peer1, &peer2, 50, Some("Transfer 3".to_string()), &signing_key).await.unwrap();
    
    // Credit peer2 and make a transfer back
    wallet.credit(&peer2, 200).await.unwrap();
    wallet.transfer(&peer2, &peer1, 75, Some("Refund".to_string()), &signing_key).await.unwrap();
    
    // Get transfer history for peer1
    let history = wallet.get_transfers(&peer1, 10).await.unwrap();
    assert_eq!(history.len(), 4); // 3 sent + 1 received
    
    // Verify transfers are ordered by most recent first
    assert_eq!(history[0].memo, Some("Refund".to_string()));
    assert_eq!(history[1].memo, Some("Transfer 3".to_string()));
    assert_eq!(history[2].memo, Some("Transfer 2".to_string()));
    assert_eq!(history[3].memo, Some("Transfer 1".to_string()));
    
    // Test limit
    let limited_history = wallet.get_transfers(&peer1, 2).await.unwrap();
    assert_eq!(limited_history.len(), 2);
}

#[tokio::test]
async fn test_concurrent_transfers() {
    let wallet = create_test_wallet().await;
    let sender = PeerId::random();
    let recipient1 = PeerId::random();
    let recipient2 = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    
    // Credit sender
    wallet.credit(&sender, 1000).await.unwrap();
    
    // Perform multiple transfers concurrently
    let transfer1_fut = wallet.transfer(&sender, &recipient1, 200, Some("Concurrent 1".to_string()), &signing_key);
    let transfer2_fut = wallet.transfer(&sender, &recipient2, 300, Some("Concurrent 2".to_string()), &signing_key);
    
    let (result1, result2) = tokio::join!(transfer1_fut, transfer2_fut);
    
    // Both transfers should succeed
    result1.unwrap();
    result2.unwrap();
    
    // Verify final balances
    let sender_balance = wallet.get_balance(&sender).await.unwrap();
    let recipient1_balance = wallet.get_balance(&recipient1).await.unwrap();
    let recipient2_balance = wallet.get_balance(&recipient2).await.unwrap();
    
    assert_eq!(sender_balance.available, 500); // 1000 - 200 - 300
    assert_eq!(recipient1_balance.available, 200);
    assert_eq!(recipient2_balance.available, 300);
}

#[tokio::test]
async fn test_escrow_integration() {
    let wallet = create_test_wallet().await;
    let buyer = PeerId::random();
    let seller = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    
    // Credit buyer
    wallet.credit(&buyer, 1000).await.unwrap();
    
    // Lock tokens for escrow (simulating purchase)
    let balance = wallet.lock_tokens(&buyer, 400).await.unwrap();
    assert_eq!(balance.available, 600);
    assert_eq!(balance.locked, 400);
    
    // Simulate successful completion - unlock and transfer
    wallet.unlock_tokens(&buyer, 400).await.unwrap();
    let transfer = wallet
        .transfer(&buyer, &seller, 400, Some("Payment for compute task".to_string()), &signing_key)
        .await
        .unwrap();
    
    assert_eq!(transfer.amount, 400);
    
    // Verify final balances
    let buyer_balance = wallet.get_balance(&buyer).await.unwrap();
    let seller_balance = wallet.get_balance(&seller).await.unwrap();
    assert_eq!(buyer_balance.available, 600);
    assert_eq!(seller_balance.available, 400);
}

#[tokio::test]
async fn test_persistence_across_instances() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("persistent_wallet.db");
    let peer_id = PeerId::random();
    
    // Create wallet and perform operations
    {
        let wallet = Wallet::new(db_path.to_str().unwrap()).await.unwrap();
        wallet.init_schema().await.unwrap();
        wallet.credit(&peer_id, 1000).await.unwrap();
        wallet.lock_tokens(&peer_id, 300).await.unwrap();
    }
    
    // Create new wallet instance with same database
    {
        let wallet = Wallet::new(db_path.to_str().unwrap()).await.unwrap();
        let balance = wallet.get_balance(&peer_id).await.unwrap();
        
        // Verify state is persisted
        assert_eq!(balance.available, 700);
        assert_eq!(balance.locked, 300);
        assert_eq!(balance.total, 1000);
    }
}

#[tokio::test]
async fn test_zero_amount_operations() {
    let wallet = create_test_wallet().await;
    let peer_id = PeerId::random();
    let signing_key = SigningKey::generate(&mut OsRng);
    
    // Credit with zero should work (no-op)
    let balance = wallet.credit(&peer_id, 0).await.unwrap();
    assert_eq!(balance.available, 0);
    
    // Debit zero should work
    let balance = wallet.debit(&peer_id, 0).await.unwrap();
    assert_eq!(balance.available, 0);
    
    // Transfer zero should work
    let recipient = PeerId::random();
    let transfer = wallet
        .transfer(&peer_id, &recipient, 0, Some("Zero transfer".to_string()), &signing_key)
        .await
        .unwrap();
    assert_eq!(transfer.amount, 0);
}

#[tokio::test]
async fn test_large_scale_operations() {
    let wallet = create_test_wallet().await;
    let num_peers = 100;
    let mut peers = Vec::new();
    
    // Create many peers and credit them
    for _ in 0..num_peers {
        let peer = PeerId::random();
        peers.push(peer);
        wallet.credit(&peer, 1000).await.unwrap();
    }
    
    // Verify all balances
    for peer in &peers {
        let balance = wallet.get_balance(peer).await.unwrap();
        assert_eq!(balance.available, 1000);
    }
    
    // Perform many transfers
    let signing_key = SigningKey::generate(&mut OsRng);
    for i in 0..peers.len() - 1 {
        wallet
            .transfer(&peers[i], &peers[i + 1], 10, None, &signing_key)
            .await
            .unwrap();
    }
    
    // Verify first peer has less, last peer has more
    let first_balance = wallet.get_balance(&peers[0]).await.unwrap();
    let last_balance = wallet.get_balance(&peers[peers.len() - 1]).await.unwrap();
    assert_eq!(first_balance.available, 990);
    assert_eq!(last_balance.available, 1010);
}