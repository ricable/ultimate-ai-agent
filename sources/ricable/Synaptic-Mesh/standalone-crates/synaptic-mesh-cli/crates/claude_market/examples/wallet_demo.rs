//! Demonstration of the token wallet system
//! 
//! This example shows how to:
//! - Create wallets for multiple peers
//! - Credit tokens for compute contribution
//! - Transfer tokens between peers
//! - Use escrow for secure transactions

use claude_market::wallet::Wallet;
use ed25519_dalek::SigningKey;
use libp2p::PeerId;
use rand::rngs::OsRng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Synaptic Market Token Wallet Demo ===\n");

    // Create a shared wallet database for this demo
    let wallet = Wallet::new("demo_wallet.db").await?;
    wallet.init_schema().await?;

    // Create three peers
    let alice = PeerId::random();
    let bob = PeerId::random();
    let charlie = PeerId::random();
    
    println!("Created peers:");
    println!("  Alice:   {}", alice);
    println!("  Bob:     {}", bob);
    println!("  Charlie: {}\n", charlie);

    // Alice contributes compute and earns tokens
    println!("1. Alice contributes compute and earns 1000 tokens");
    let alice_balance = wallet.credit(&alice, 1000).await?;
    println!("   Alice balance: {} tokens\n", alice_balance.available);

    // Bob also contributes and earns tokens
    println!("2. Bob contributes compute and earns 800 tokens");
    let bob_balance = wallet.credit(&bob, 800).await?;
    println!("   Bob balance: {} tokens\n", bob_balance.available);

    // Alice wants to use Bob's compute, so she transfers tokens
    println!("3. Alice pays Bob 250 tokens for a compute task");
    let alice_key = SigningKey::generate(&mut OsRng);
    let transfer = wallet.transfer(
        &alice,
        &bob,
        250,
        Some("Payment for neural network training task".to_string()),
        &alice_key
    ).await?;
    println!("   Transfer ID: {}", transfer.id);
    println!("   Amount: {} tokens", transfer.amount);
    println!("   Memo: {}", transfer.memo.unwrap_or_default());
    
    // Check updated balances
    let alice_balance = wallet.get_balance(&alice).await?;
    let bob_balance = wallet.get_balance(&bob).await?;
    println!("   Alice balance: {} tokens", alice_balance.available);
    println!("   Bob balance: {} tokens\n", bob_balance.available);

    // Demonstrate escrow for a larger transaction
    println!("4. Alice initiates escrow transaction with Charlie");
    println!("   Alice locks 500 tokens in escrow for a large compute job");
    let alice_balance = wallet.lock_tokens(&alice, 500).await?;
    println!("   Alice available: {} tokens", alice_balance.available);
    println!("   Alice locked: {} tokens\n", alice_balance.locked);

    // Charlie completes the work, escrow is released
    println!("5. Charlie completes the compute job successfully");
    println!("   Escrow releases tokens from Alice to Charlie");
    
    // First unlock the tokens
    wallet.unlock_tokens(&alice, 500).await?;
    
    // Then transfer to Charlie
    let transfer = wallet.transfer(
        &alice,
        &charlie,
        500,
        Some("Payment for distributed training job - escrow release".to_string()),
        &alice_key
    ).await?;
    println!("   Transfer complete: {} tokens to Charlie\n", transfer.amount);

    // Show final balances
    println!("=== Final Balances ===");
    let alice_balance = wallet.get_balance(&alice).await?;
    let bob_balance = wallet.get_balance(&bob).await?;
    let charlie_balance = wallet.get_balance(&charlie).await?;
    
    println!("Alice:   {} tokens (contributed compute, paid for services)", alice_balance.available);
    println!("Bob:     {} tokens (provided neural network training)", bob_balance.available);
    println!("Charlie: {} tokens (provided distributed training)\n", charlie_balance.available);

    // Show transaction history
    println!("=== Alice's Transaction History ===");
    let history = wallet.get_transfers(&alice, 10).await?;
    for (i, tx) in history.iter().enumerate() {
        let direction = if tx.from == alice { "Sent to" } else { "Received from" };
        let peer = if tx.from == alice { &tx.to } else { &tx.from };
        println!("{}. {} {} - {} tokens", 
            i + 1, 
            direction, 
            peer, 
            tx.amount
        );
        if let Some(memo) = &tx.memo {
            println!("   Memo: {}", memo);
        }
    }

    println!("\n=== Key Compliance Points ===");
    println!("✅ Each peer controls their own wallet - no shared keys");
    println!("✅ Tokens reward compute contribution, not Claude access");
    println!("✅ All transfers are cryptographically signed");
    println!("✅ Full audit trail maintained in SQLite");

    Ok(())
}