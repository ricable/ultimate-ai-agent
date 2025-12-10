//! # Claude Market
//! 
//! A decentralized, peer-to-peer marketplace for Claude AI capacity trading with full Anthropic ToS compliance.
//! 
//! ## Overview
//! 
//! Claude Market enables secure, decentralized trading of Claude AI compute capacity using ruv tokens. 
//! Built on the Synaptic Neural Mesh network, it provides a compliant way for Claude Max subscribers 
//! to share their capacity through a peer compute federation model.
//! 
//! ## Features
//! 
//! - 🔒 **Anthropic ToS Compliant** - No API key sharing, peer-orchestrated execution
//! - 🏦 **Secure Escrow** - Multi-signature escrow with automatic settlement
//! - 🎯 **First-Accept Auctions** - Fast, competitive pricing mechanisms
//! - 📊 **Reputation System** - SLA tracking and provider trust scores
//! - 🛡️ **Privacy-Preserving** - Encrypted task payloads and secure execution
//! - 💾 **SQLite Persistence** - Local data storage with full transaction history
//! 
//! ## Quick Start
//! 
//! ```rust
//! use claude_market::{ClaudeMarket, MarketConfig};
//! 
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize market with local SQLite database
//! let config = MarketConfig::default();
//! let market = ClaudeMarket::new(config).await?;
//! 
//! // Check wallet balance
//! let balance = market.wallet().get_balance().await?;
//! println!("RUV Token Balance: {}", balance);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use libp2p::PeerId;
use uuid::Uuid;
use chrono::{DateTime, Utc};

// Re-export error types
pub use error::{MarketError, Result};

// Module declarations
pub mod error;
pub mod wallet;
pub mod escrow;
pub mod ledger;
pub mod market;
pub mod reputation;
// pub mod p2p;  // Temporarily disabled due to libp2p compatibility issues
pub mod pricing;

/// Market configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConfig {
    /// Database file path (None for in-memory)
    pub db_path: Option<String>,
    /// Maximum concurrent orders
    pub max_orders: usize,
    /// Default order timeout in seconds
    pub default_timeout: u64,
    /// Minimum reputation score required
    pub min_reputation: f64,
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            db_path: Some("claude_market.db".to_string()),
            max_orders: 1000,
            default_timeout: 3600, // 1 hour
            min_reputation: 0.0,
        }
    }
}

/// Claude Market main interface
#[derive(Debug)]
pub struct ClaudeMarket {
    wallet: wallet::Wallet,
    config: MarketConfig,
}

impl ClaudeMarket {
    /// Create a new Claude Market instance
    pub async fn new(config: MarketConfig) -> Result<Self> {
        let db_path = config.db_path.as_ref().map(|s| s.as_str());
        
        let wallet = wallet::Wallet::new(db_path.unwrap_or(":memory:")).await?;
        
        Ok(Self {
            wallet,
            config,
        })
    }
    
    /// Get wallet interface
    pub fn wallet(&self) -> &wallet::Wallet {
        &self.wallet
    }
    
    /// Get market configuration
    pub fn config(&self) -> &MarketConfig {
        &self.config
    }
}

/// Order type in the compute contribution market
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderType {
    /// Request compute contribution (buyer of compute)
    RequestCompute,
    /// Offer compute contribution (provider of compute)
    OfferCompute,
}

/// Order status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderStatus {
    /// Order is active and can be matched
    Active,
    /// Order is partially filled
    PartiallyFilled,
    /// Order is completely filled
    Filled,
    /// Order was cancelled
    Cancelled,
    /// Order expired
    Expired,
}

/// Compute task specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeTaskSpec {
    /// Task type identifier
    pub task_type: String,
    /// Required compute resources
    pub resource_requirements: HashMap<String, u64>,
    /// Expected completion time in seconds
    pub estimated_duration: u64,
    /// Task priority (1-10, higher = more urgent)
    pub priority: u8,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// SLA (Service Level Agreement) specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLASpec {
    /// Required uptime percentage (0-100)
    pub uptime_requirement: f32,
    /// Maximum response time in seconds
    pub max_response_time: u64,
    /// Penalty for SLA violation (in tokens)
    pub violation_penalty: u64,
    /// Quality metrics requirements
    pub quality_metrics: HashMap<String, f64>,
}

/// Market order for compute contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: Uuid,
    /// Order type (request/offer compute)
    pub order_type: OrderType,
    /// Trader peer ID
    pub trader: PeerId,
    /// Price per compute unit (in smallest token unit)
    pub price_per_unit: u64,
    /// Total compute units
    pub total_units: u64,
    /// Filled units
    pub filled_units: u64,
    /// Order status
    pub status: OrderStatus,
    /// Task specification
    pub task_spec: ComputeTaskSpec,
    /// SLA requirements
    pub sla_spec: Option<SLASpec>,
    /// Reputation weight multiplier (1.0 = normal, >1.0 = preferred)
    pub reputation_weight: f64,
    /// Order expiry time (None = GTC)
    pub expires_at: Option<DateTime<Utc>>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Order signature for verification
    pub signature: Option<Vec<u8>>,
}

impl Order {
    /// Get remaining compute units
    pub fn remaining_units(&self) -> u64 {
        self.total_units.saturating_sub(self.filled_units)
    }
    
    /// Check if order is fully filled
    pub fn is_filled(&self) -> bool {
        self.filled_units >= self.total_units
    }
    
    /// Calculate total value of the order
    pub fn total_value(&self) -> u64 {
        self.price_per_unit.saturating_mul(self.total_units)
    }
}

/// Bid order (requesting compute)
pub type Bid = Order;

/// Offer order (providing compute)  
pub type Offer = Order;