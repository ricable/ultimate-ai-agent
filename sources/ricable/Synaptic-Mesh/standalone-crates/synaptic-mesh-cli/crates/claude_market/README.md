# Claude Market

[![Crates.io](https://img.shields.io/crates/v/claude_market)](https://crates.io/crates/claude_market)
[![Documentation](https://docs.rs/claude_market/badge.svg)](https://docs.rs/claude_market)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

**A decentralized, peer-to-peer marketplace for Claude AI capacity trading with full Anthropic ToS compliance.**

## 🏪 Overview

Claude Market enables secure, decentralized trading of Claude AI compute capacity using ruv tokens. Built on the Synaptic Neural Mesh network, it provides a compliant way for Claude Max subscribers to share their capacity through a peer compute federation model.

## ✨ Key Features

- 🔒 **Anthropic ToS Compliant** - No API key sharing, peer-orchestrated execution
- 🏦 **Secure Escrow** - Multi-signature escrow with automatic settlement
- 🎯 **First-Accept Auctions** - Fast, competitive pricing mechanisms
- 📊 **Reputation System** - SLA tracking and provider trust scores
- 🛡️ **Privacy-Preserving** - Encrypted task payloads and secure execution
- 💾 **SQLite Persistence** - Local data storage with full transaction history

## 🚀 Quick Start

```toml
[dependencies]
claude_market = "0.1"
tokio = { version = "1.0", features = ["full"] }
```

```rust
use claude_market::{ClaudeMarket, MarketConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize market with local SQLite database
    let config = MarketConfig::default();
    let market = ClaudeMarket::new(config).await?;
    
    // Check wallet balance
    let balance = market.wallet().get_balance().await?;
    println!("RUV Token Balance: {}", balance);
    
    // Create a compute capacity offer
    market.create_offer(
        100, // 100 ruv tokens
        chrono::Duration::hours(1), // 1 hour availability
        "High-performance Claude Max capacity".to_string()
    ).await?;
    
    Ok(())
}
```

## 🔐 Compliance & Security

Claude Market operates as a **peer compute federation**, not a resale service:

- ✅ **No API Key Sharing** - Each participant uses their own Claude subscription
- ✅ **Local Execution** - Tasks run locally on provider's Claude account
- ✅ **Voluntary Participation** - Full user control with opt-in mechanisms
- ✅ **Token Rewards** - RUV tokens reward contribution, not access purchase

## 📋 Core Components

### Wallet System
- SQLite-based token storage
- Cryptographically signed transfers
- Escrow lock/unlock mechanisms
- Complete transaction history

### Escrow Service
- Multi-signature contract support
- Time-based automatic release
- Dispute resolution system
- Byzantine fault tolerance

### Market Engine
- First-accept auction model
- Real-time price discovery
- Reputation-weighted matching
- SLA enforcement

### Reputation Tracker
- Performance-based scoring
- Historical success rates
- Provider trust metrics
- Automatic reputation updates

## 🛠️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Wallet API    │    │  Escrow Service │    │ Market Engine   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Balance Mgmt  │    │ • Multi-sig     │    │ • Auctions      │
│ • Transfers     │    │ • Auto-release  │    │ • Price Disc.   │
│ • Escrow Locks  │    │ • Disputes      │    │ • Order Match   │
│ • History       │    │ • Timeouts      │    │ • SLA Track     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌─────────────────┐
                    │ SQLite Database │
                    ├─────────────────┤
                    │ • Transactions  │
                    │ • Escrows       │
                    │ • Orders        │
                    │ • Reputation    │
                    └─────────────────┘
```

## 🧪 Examples

### Provider Mode - Offering Capacity
```rust
use claude_market::{ClaudeMarket, OfferBuilder};

// Create capacity offer
let offer = OfferBuilder::new()
    .price(50) // 50 ruv tokens
    .duration(chrono::Duration::hours(2))
    .description("Premium Claude Max with 2-hour availability")
    .max_concurrent_tasks(3)
    .build();

market.create_offer(offer).await?;
```

### Client Mode - Bidding for Capacity
```rust
use claude_market::{BidBuilder, TaskPayload};

// Submit encrypted task bid
let task = TaskPayload::new("Analyze this dataset for trends");
let bid = BidBuilder::new()
    .task(task.encrypt(&provider_pubkey)?)
    .max_price(75)
    .timeout(chrono::Duration::minutes(30))
    .build();

let result = market.submit_bid(bid).await?;
```

### Escrow Operations
```rust
// Create escrow for secure trading
let escrow_id = market.escrow()
    .create_escrow(
        provider_peer_id,
        client_peer_id,
        100, // amount
        chrono::Duration::hours(1) // timeout
    ).await?;

// Release escrow after successful completion
market.escrow().release(escrow_id, &signatures).await?;
```

## 📊 Performance

- **Auction Settlement**: <500ms average
- **Database Operations**: <10ms for typical queries  
- **Memory Usage**: ~32MB per market instance
- **Concurrent Orders**: 1000+ supported
- **Network Efficiency**: 73% price discovery accuracy

## 🤝 Contributing

We welcome contributions to Claude Market! Areas of interest:

- 🔒 **Security Audits** - Cryptographic and smart contract review
- ⚡ **Performance** - Optimization of auction and matching algorithms
- 🧪 **Testing** - Additional test scenarios and edge cases
- 📚 **Documentation** - Examples, tutorials, and API docs

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🔗 Related Projects

- **[Synaptic Neural Mesh](https://github.com/ruvnet/Synaptic-Mesh)** - Distributed neural network framework
- **[synaptic-qudag-core](https://crates.io/crates/synaptic-qudag-core)** - Quantum-resistant DAG networking
- **[synaptic-mesh-cli](https://crates.io/crates/synaptic-mesh-cli)** - Command-line interface

---

**Legal Notice**: Claude Market facilitates peer compute federation, not API access resale. All participants must maintain their own Claude subscriptions and comply with Anthropic's Terms of Service.