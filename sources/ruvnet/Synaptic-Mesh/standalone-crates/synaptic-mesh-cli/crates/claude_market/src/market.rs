//! Market order book and matching engine with first-accept auction model
//!
//! This implementation follows a peer compute federation model where:
//! - Contributors run their own Claude Max accounts locally
//! - Tasks are routed, not account access
//! - Tokens reward successful completions, not access to Claude
//! - All participation is voluntary and contribution-based

use crate::error::{MarketError, Result};
use crate::reputation::Reputation;
use crate::pricing::{PricingStrategy, MarketConditions, DemandLevel, SupplyLevel};
use chrono::{DateTime, Duration, Utc};
use libp2p::PeerId;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use ed25519_dalek::{SigningKey, Signature, Signer, Verifier, VerifyingKey};

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
    /// Task type (e.g., "code_generation", "analysis", "testing")
    pub task_type: String,
    /// Estimated compute units required
    pub compute_units: u64,
    /// Maximum execution time in seconds
    pub max_duration_secs: u64,
    /// Required capabilities (e.g., "rust", "python", "ml")
    pub required_capabilities: Vec<String>,
    /// Minimum reputation score required
    pub min_reputation: Option<f64>,
    /// Privacy requirements
    pub privacy_level: PrivacyLevel,
    /// Encrypted task payload (only revealed to accepted provider)
    pub encrypted_payload: Option<Vec<u8>>,
}

/// Privacy level for compute tasks
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrivacyLevel {
    /// Public task, details visible to all
    Public,
    /// Private task, details only revealed after acceptance
    Private,
    /// Confidential task, requires additional verification
    Confidential,
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

/// Bid order (requesting compute)
pub type Bid = Order;

/// Offer order (providing compute)  
pub type Offer = Order;

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

    /// Check if order is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::Active | OrderStatus::PartiallyFilled)
            && self.remaining_units() > 0
            && self.expires_at.map_or(true, |exp| exp > Utc::now())
    }

    /// Update filled units
    pub fn fill(&mut self, units: u64) -> Result<()> {
        let remaining = self.remaining_units();
        if units > remaining {
            return Err(MarketError::InvalidOrder(
                format!("Cannot fill {} units, only {} remaining", units, remaining)
            ));
        }

        self.filled_units += units;
        self.updated_at = Utc::now();

        if self.filled_units == self.total_units {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }

        Ok(())
    }

    /// Calculate effective price with reputation weighting
    pub fn effective_price(&self, provider_reputation: f64) -> u64 {
        let reputation_factor = (provider_reputation / 100.0).min(2.0).max(0.5);
        (self.price_per_unit as f64 * self.reputation_weight * reputation_factor) as u64
    }

    /// Sign the order with private key
    pub fn sign(&mut self, signing_key: &SigningKey) -> Result<()> {
        let message = self.signature_message();
        let signature = signing_key.sign(message.as_bytes());
        self.signature = Some(signature.to_bytes().to_vec());
        Ok(())
    }

    /// Verify order signature
    pub fn verify_signature(&self, verifying_key: &VerifyingKey) -> Result<bool> {
        if let Some(sig_bytes) = &self.signature {
            let signature = Signature::from_bytes(sig_bytes.as_slice().try_into()
                .map_err(|_| MarketError::Crypto("Invalid signature".to_string()))?);
            let message = self.signature_message();
            Ok(verifying_key.verify(message.as_bytes(), &signature).is_ok())
        } else {
            Ok(false)
        }
    }

    /// Generate signature message
    fn signature_message(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            self.id,
            self.trader,
            self.price_per_unit,
            self.total_units,
            self.created_at.to_rfc3339(),
            serde_json::to_string(&self.task_spec).unwrap_or_default()
        )
    }
}

/// Compute request order
pub type ComputeRequest = Order;

/// Compute offer order
pub type ComputeOffer = Order;

/// First-accept auction for compute tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstAcceptAuction {
    /// Auction ID
    pub id: Uuid,
    /// Compute request that triggered the auction
    pub request_id: Uuid,
    /// Minimum required providers
    pub min_providers: u32,
    /// Maximum providers allowed
    pub max_providers: u32,
    /// Auction start time
    pub started_at: DateTime<Utc>,
    /// Auction end time
    pub ends_at: DateTime<Utc>,
    /// Accepted offers
    pub accepted_offers: Vec<Uuid>,
    /// Auction status
    pub status: AuctionStatus,
}

/// Auction status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuctionStatus {
    /// Accepting offers
    Open,
    /// Minimum providers reached, still accepting
    MinReached,
    /// Maximum providers reached or time expired
    Closed,
    /// Cancelled by requester
    Cancelled,
}

/// Compute task assignment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    /// Unique assignment ID
    pub id: Uuid,
    /// Request order ID
    pub request_id: Uuid,
    /// Offer order ID
    pub offer_id: Uuid,
    /// Requester peer ID
    pub requester: PeerId,
    /// Provider peer ID
    pub provider: PeerId,
    /// Agreed price per unit
    pub price_per_unit: u64,
    /// Assigned compute units
    pub compute_units: u64,
    /// Total cost
    pub total_cost: u64,
    /// Assignment timestamp
    pub assigned_at: DateTime<Utc>,
    /// Task started timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Task completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// SLA tracking
    pub sla_metrics: SLAMetrics,
    /// Assignment status
    pub status: AssignmentStatus,
}

/// Assignment status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssignmentStatus {
    /// Assigned but not started
    Assigned,
    /// Provider is working on task
    InProgress,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// SLA violated
    SLAViolated,
    /// Disputed by either party
    Disputed,
}

/// SLA tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SLAMetrics {
    /// Response time in seconds
    pub response_time: Option<u64>,
    /// Actual execution time
    pub execution_time: Option<u64>,
    /// Quality scores
    pub quality_scores: HashMap<String, f64>,
    /// Number of violations
    pub violations: u32,
    /// Total penalty amount
    pub total_penalty: u64,
}

/// Price discovery mechanism data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceDiscovery {
    /// Task type
    pub task_type: String,
    /// Moving average price (last 24h)
    pub avg_price_24h: f64,
    /// Volume weighted average price
    pub vwap: f64,
    /// Minimum price in period
    pub min_price: u64,
    /// Maximum price in period
    pub max_price: u64,
    /// Total volume
    pub total_volume: u64,
    /// Number of assignments
    pub assignment_count: u64,
    /// Last update
    pub last_updated: DateTime<Utc>,
}

/// Market order book and matching engine with P2P integration
pub struct Market {
    db: Mutex<Connection>,
    reputation: Reputation,
    // p2p_network: Option<Arc<Mutex<crate::p2p::P2PNetwork>>>,  // Temporarily disabled
    pricing_engine: Arc<crate::pricing::PricingEngine>,
}

impl Market {
    /// Create a new market instance
    pub async fn new(db_path: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;
        let reputation = Reputation::new(db_path).await?;
        let pricing_engine = Arc::new(crate::pricing::PricingEngine::new(db_path).await?);
        pricing_engine.init_schema().await?;
        
        Ok(Self {
            db: Mutex::new(db),
            reputation,
            // p2p_network: None,  // Temporarily disabled
            pricing_engine,
        })
    }
    
    /*
    /// Create a new market instance with P2P networking - DISABLED
    pub async fn new_with_p2p(
        db_path: &str, 
        p2p_config: crate::p2p::P2PConfig
    ) -> Result<Self> {
        let mut market = Self::new(db_path).await?;
        
        let p2p_network = crate::p2p::P2PNetwork::new(p2p_config).await?;
        market.p2p_network = Some(Arc::new(Mutex::new(p2p_network)));
        
        Ok(market)
    }
    
    /// Enable P2P networking - DISABLED
    pub async fn enable_p2p(&mut self) -> Result<()> {
        if let Some(ref p2p) = self.p2p_network {
            p2p.lock().await.start().await?;
        }
        Ok(())
    }
    */

    /// Initialize database schema
    pub async fn init_schema(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                order_type TEXT NOT NULL,
                trader TEXT NOT NULL,
                price_per_unit INTEGER NOT NULL,
                total_units INTEGER NOT NULL,
                filled_units INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                task_spec TEXT NOT NULL,
                sla_spec TEXT,
                reputation_weight REAL NOT NULL DEFAULT 1.0,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                signature BLOB
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS task_assignments (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                offer_id TEXT NOT NULL,
                requester TEXT NOT NULL,
                provider TEXT NOT NULL,
                price_per_unit INTEGER NOT NULL,
                compute_units INTEGER NOT NULL,
                total_cost INTEGER NOT NULL,
                assigned_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                sla_metrics TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES orders(id),
                FOREIGN KEY (offer_id) REFERENCES orders(id)
            )
            "#,
            [],
        )?;

        // Create auctions table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS auctions (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                min_providers INTEGER NOT NULL,
                max_providers INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ends_at TEXT NOT NULL,
                accepted_offers TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES orders(id)
            )
            "#,
            [],
        )?;

        // Create price discovery table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS price_discovery (
                task_type TEXT PRIMARY KEY,
                avg_price_24h REAL NOT NULL,
                vwap REAL NOT NULL,
                min_price INTEGER NOT NULL,
                max_price INTEGER NOT NULL,
                total_volume INTEGER NOT NULL,
                assignment_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
            "#,
            [],
        )?;

        // Create indexes
        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_orders_trader ON orders(trader);
            CREATE INDEX IF NOT EXISTS idx_orders_type ON orders(order_type);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_orders_price ON orders(price_per_unit);
            CREATE INDEX IF NOT EXISTS idx_orders_task_type ON orders(json_extract(task_spec, '$.task_type'));
            CREATE INDEX IF NOT EXISTS idx_assignments_requester ON task_assignments(requester);
            CREATE INDEX IF NOT EXISTS idx_assignments_provider ON task_assignments(provider);
            CREATE INDEX IF NOT EXISTS idx_assignments_status ON task_assignments(status);
            CREATE INDEX IF NOT EXISTS idx_assignments_assigned ON task_assignments(assigned_at);
            CREATE INDEX IF NOT EXISTS idx_auctions_status ON auctions(status);
            CREATE INDEX IF NOT EXISTS idx_auctions_ends ON auctions(ends_at);
            "#,
            [],
        )?;

        Ok(())
    }

    /// Place a new compute contribution order
    pub async fn place_order(
        &self,
        order_type: OrderType,
        trader: PeerId,
        price_per_unit: u64,
        total_units: u64,
        task_spec: ComputeTaskSpec,
        sla_spec: Option<SLASpec>,
        expires_at: Option<DateTime<Utc>>,
        signing_key: Option<&SigningKey>,
    ) -> Result<Order> {
        if total_units == 0 {
            return Err(MarketError::InvalidOrder("Units must be greater than 0".to_string()));
        }

        if price_per_unit == 0 {
            return Err(MarketError::InvalidOrder("Price per unit must be greater than 0".to_string()));
        }

        // Validate task spec
        if task_spec.task_type.is_empty() {
            return Err(MarketError::InvalidOrder("Task type must be specified".to_string()));
        }

        if task_spec.compute_units == 0 {
            return Err(MarketError::InvalidOrder("Compute units must be greater than 0".to_string()));
        }

        // Get trader reputation for weighting
        let reputation = self.reputation.get_reputation(&trader).await?;
        let reputation_weight = (reputation.score / 100.0).max(0.5).min(2.0);

        let mut order = Order {
            id: Uuid::new_v4(),
            order_type,
            trader,
            price_per_unit,
            total_units,
            filled_units: 0,
            status: OrderStatus::Active,
            task_spec,
            sla_spec,
            reputation_weight,
            expires_at,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            signature: None,
        };

        // Sign the order if key provided
        if let Some(key) = signing_key {
            order.sign(key)?;
        }

        let db = self.db.lock().await;
        db.execute(
            r#"
            INSERT INTO orders (
                id, order_type, trader, price_per_unit, total_units, filled_units,
                status, task_spec, sla_spec, reputation_weight,
                expires_at, created_at, updated_at, signature
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            "#,
            params![
                order.id.to_string(),
                format!("{:?}", order.order_type),
                order.trader.to_string(),
                order.price_per_unit as i64,
                order.total_units as i64,
                order.filled_units as i64,
                format!("{:?}", order.status),
                serde_json::to_string(&order.task_spec)?,
                order.sla_spec.as_ref().map(|s| serde_json::to_string(s).ok()).flatten(),
                order.reputation_weight,
                order.expires_at.map(|dt| dt.to_rfc3339()),
                order.created_at.to_rfc3339(),
                order.updated_at.to_rfc3339(),
                order.signature,
            ],
        )?;

        drop(db);

        // Broadcast order to P2P network if enabled - DISABLED
        /*
        if let Some(ref p2p) = self.p2p_network {
            if let Err(e) = p2p.lock().await.broadcast_order(order.clone()).await {
                tracing::warn!("Failed to broadcast order to P2P network: {}", e);
            }
        }
        */

        // For compute requests, start a first-accept auction
        if matches!(order.order_type, OrderType::RequestCompute) {
            self.start_auction(&order).await?;
        } else {
            // For compute offers, try to match with existing requests
            self.match_offer(&order).await?;
        }

        // Update price discovery and market conditions
        self.update_price_discovery(&order.task_spec.task_type).await?;
        self.update_market_conditions().await?;

        Ok(order)
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &Uuid, trader: &PeerId) -> Result<()> {
        let db = self.db.lock().await;
        
        // Verify ownership
        let owner: String = db.query_row(
            "SELECT trader FROM orders WHERE id = ?1",
            params![order_id.to_string()],
            |row| row.get(0),
        ).map_err(|_| MarketError::OrderNotFound(order_id.to_string()))?;

        if owner != trader.to_string() {
            return Err(MarketError::InvalidOrder("Not order owner".to_string()));
        }

        // Cancel the order
        db.execute(
            r#"
            UPDATE orders 
            SET status = 'Cancelled', updated_at = ?2
            WHERE id = ?1 AND status IN ('Active', 'PartiallyFilled')
            "#,
            params![order_id.to_string(), Utc::now().to_rfc3339()],
        )?;

        Ok(())
    }

    /// Get order by ID
    pub async fn get_order(&self, order_id: &Uuid) -> Result<Option<Order>> {
        let db = self.db.lock().await;
        
        let order = db
            .query_row(
                r#"
                SELECT id, order_type, trader, price_per_unit, total_units, filled_units,
                       status, task_spec, sla_spec, reputation_weight,
                       expires_at, created_at, updated_at, signature
                FROM orders
                WHERE id = ?1
                "#,
                params![order_id.to_string()],
                |row| {
                    Ok(Order {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                        order_type: match row.get::<_, String>(1)?.as_str() {
                            "RequestCompute" => OrderType::RequestCompute,
                            "OfferCompute" => OrderType::OfferCompute,
                            _ => OrderType::RequestCompute,
                        },
                        trader: row.get::<_, String>(2)?.parse().unwrap(),
                        price_per_unit: row.get::<_, i64>(3)? as u64,
                        total_units: row.get::<_, i64>(4)? as u64,
                        filled_units: row.get::<_, i64>(5)? as u64,
                        status: match row.get::<_, String>(6)?.as_str() {
                            "Active" => OrderStatus::Active,
                            "PartiallyFilled" => OrderStatus::PartiallyFilled,
                            "Filled" => OrderStatus::Filled,
                            "Cancelled" => OrderStatus::Cancelled,
                            "Expired" => OrderStatus::Expired,
                            _ => OrderStatus::Active,
                        },
                        task_spec: serde_json::from_str(&row.get::<_, String>(7)?).unwrap(),
                        sla_spec: row.get::<_, Option<String>>(8)?
                            .and_then(|s| serde_json::from_str(&s).ok()),
                        reputation_weight: row.get(9)?,
                        expires_at: row.get::<_, Option<String>>(10)?
                            .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(11)?)
                            .unwrap()
                            .with_timezone(&Utc),
                        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(12)?)
                            .unwrap()
                            .with_timezone(&Utc),
                        signature: row.get::<_, Option<Vec<u8>>>(13)?,
                    })
                },
            )
            .ok();

        Ok(order)
    }

    /// Get active orders for a trader
    pub async fn get_trader_orders(
        &self,
        trader: &PeerId,
        order_type: Option<OrderType>,
        active_only: bool,
    ) -> Result<Vec<Order>> {
        let db = self.db.lock().await;
        
        let mut query = String::from(
            r#"
            SELECT id, order_type, trader, price, quantity, filled,
                   status, expires_at, created_at, updated_at
            FROM orders
            WHERE trader = ?1
            "#
        );

        if let Some(ot) = order_type {
            query.push_str(&format!(" AND order_type = '{:?}'", ot));
        }

        if active_only {
            query.push_str(" AND status IN ('Active', 'PartiallyFilled')");
        }

        query.push_str(" ORDER BY created_at DESC");

        let mut stmt = db.prepare(&query)?;
        let orders = stmt
            .query_map(params![trader.to_string()], |row| {
                Ok(Order {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    order_type: match row.get::<_, String>(1)?.as_str() {
                        "Buy" => OrderType::RequestCompute,
                        "Sell" => OrderType::OfferCompute,
                        _ => OrderType::RequestCompute,
                    },
                    trader: row.get::<_, String>(2)?.parse().unwrap(),
                    price_per_unit: row.get::<_, i64>(3)? as u64,
                    total_units: row.get::<_, i64>(4)? as u64,
                    filled_units: row.get::<_, i64>(5)? as u64,
                    status: match row.get::<_, String>(6)?.as_str() {
                        "Active" => OrderStatus::Active,
                        "PartiallyFilled" => OrderStatus::PartiallyFilled,
                        "Filled" => OrderStatus::Filled,
                        "Cancelled" => OrderStatus::Cancelled,
                        "Expired" => OrderStatus::Expired,
                        _ => OrderStatus::Active,
                    },
                    task_spec: ComputeTaskSpec {
                        task_type: "default".to_string(),
                        compute_units: 1,
                        max_duration_secs: 3600,
                        required_capabilities: vec![],
                        min_reputation: None,
                        privacy_level: PrivacyLevel::Public,
                        encrypted_payload: None,
                    },
                    sla_spec: None,
                    reputation_weight: 1.0,
                    expires_at: row.get::<_, Option<String>>(7)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    signature: None,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(orders)
    }

    /// Get order book (active orders)
    pub async fn get_order_book(&self) -> Result<(Vec<Bid>, Vec<Offer>)> {
        let db = self.db.lock().await;
        
        // Get active buy orders sorted by price desc
        let mut buy_stmt = db.prepare(
            r#"
            SELECT id, order_type, trader, price, quantity, filled,
                   status, expires_at, created_at, updated_at
            FROM orders
            WHERE order_type = 'Buy' AND status IN ('Active', 'PartiallyFilled')
            ORDER BY price DESC, created_at ASC
            "#
        )?;

        let bids: Vec<Bid> = buy_stmt
            .query_map([], |row| {
                Ok(Order {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    order_type: OrderType::RequestCompute,
                    trader: row.get::<_, String>(2)?.parse().unwrap(),
                    price_per_unit: row.get::<_, i64>(3)? as u64,
                    total_units: row.get::<_, i64>(4)? as u64,
                    filled_units: row.get::<_, i64>(5)? as u64,
                    status: match row.get::<_, String>(6)?.as_str() {
                        "Active" => OrderStatus::Active,
                        "PartiallyFilled" => OrderStatus::PartiallyFilled,
                        _ => OrderStatus::Active,
                    },
                    task_spec: ComputeTaskSpec {
                        task_type: "default".to_string(),
                        compute_units: 1,
                        max_duration_secs: 3600,
                        required_capabilities: vec![],
                        min_reputation: None,
                        privacy_level: PrivacyLevel::Public,
                        encrypted_payload: None,
                    },
                    sla_spec: None,
                    reputation_weight: 1.0,
                    expires_at: row.get::<_, Option<String>>(7)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    signature: None,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Get active sell orders sorted by price asc
        let mut sell_stmt = db.prepare(
            r#"
            SELECT id, order_type, trader, price, quantity, filled,
                   status, expires_at, created_at, updated_at
            FROM orders
            WHERE order_type = 'Sell' AND status IN ('Active', 'PartiallyFilled')
            ORDER BY price ASC, created_at ASC
            "#
        )?;

        let offers: Vec<Offer> = sell_stmt
            .query_map([], |row| {
                Ok(Order {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    order_type: OrderType::OfferCompute,
                    trader: row.get::<_, String>(2)?.parse().unwrap(),
                    price_per_unit: row.get::<_, i64>(3)? as u64,
                    total_units: row.get::<_, i64>(4)? as u64,
                    filled_units: row.get::<_, i64>(5)? as u64,
                    status: match row.get::<_, String>(6)?.as_str() {
                        "Active" => OrderStatus::Active,
                        "PartiallyFilled" => OrderStatus::PartiallyFilled,
                        _ => OrderStatus::Active,
                    },
                    task_spec: ComputeTaskSpec {
                        task_type: "default".to_string(),
                        compute_units: 1,
                        max_duration_secs: 3600,
                        required_capabilities: vec![],
                        min_reputation: None,
                        privacy_level: PrivacyLevel::Public,
                        encrypted_payload: None,
                    },
                    sla_spec: None,
                    reputation_weight: 1.0,
                    expires_at: row.get::<_, Option<String>>(7)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    signature: None,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok((bids, offers))
    }

    /// Start a first-accept auction for a compute request
    async fn start_auction(&self, request: &Order) -> Result<FirstAcceptAuction> {
        let auction = FirstAcceptAuction {
            id: Uuid::new_v4(),
            request_id: request.id,
            min_providers: 1, // Could be configurable
            max_providers: (request.total_units / 100).max(1).min(10) as u32, // Dynamic based on size
            started_at: Utc::now(),
            ends_at: Utc::now() + Duration::minutes(15), // 15 minute auction window
            accepted_offers: Vec::new(),
            status: AuctionStatus::Open,
        };

        let db = self.db.lock().await;
        db.execute(
            r#"
            INSERT INTO auctions (
                id, request_id, min_providers, max_providers,
                started_at, ends_at, accepted_offers, status
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                auction.id.to_string(),
                auction.request_id.to_string(),
                auction.min_providers as i64,
                auction.max_providers as i64,
                auction.started_at.to_rfc3339(),
                auction.ends_at.to_rfc3339(),
                serde_json::to_string(&auction.accepted_offers)?,
                format!("{:?}", auction.status),
            ],
        )?;

        Ok(auction)
    }

    /// Match a compute offer with existing requests
    async fn match_offer(&self, offer: &Order) -> Result<Vec<TaskAssignment>> {
        let mut assignments = Vec::new();
        
        // Find active auctions for matching task types
        let db = self.db.lock().await;
        let mut stmt = db.prepare(
            r#"
            SELECT a.id, a.request_id, a.min_providers, a.max_providers,
                   a.started_at, a.ends_at, a.accepted_offers, a.status,
                   o.trader, o.price_per_unit, o.total_units, o.task_spec
            FROM auctions a
            JOIN orders o ON a.request_id = o.id
            WHERE a.status IN ('Open', 'MinReached')
              AND a.ends_at > ?1
              AND json_extract(o.task_spec, '$.task_type') = ?2
            ORDER BY o.price_per_unit DESC, a.started_at ASC
            "#,
        )?;

        let auctions: Vec<(FirstAcceptAuction, PeerId, u64, u64, ComputeTaskSpec)> = stmt
            .query_map(
                params![
                    Utc::now().to_rfc3339(),
                    offer.task_spec.task_type,
                ],
                |row| {
                    let auction = FirstAcceptAuction {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                        request_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap(),
                        min_providers: row.get::<_, i64>(2)? as u32,
                        max_providers: row.get::<_, i64>(3)? as u32,
                        started_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?).unwrap().with_timezone(&Utc),
                        ends_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?).unwrap().with_timezone(&Utc),
                        accepted_offers: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
                        status: match row.get::<_, String>(7)?.as_str() {
                            "Open" => AuctionStatus::Open,
                            "MinReached" => AuctionStatus::MinReached,
                            "Closed" => AuctionStatus::Closed,
                            "Cancelled" => AuctionStatus::Cancelled,
                            _ => AuctionStatus::Open,
                        },
                    };
                    let requester: PeerId = row.get::<_, String>(8)?.parse().unwrap();
                    let price_per_unit = row.get::<_, i64>(9)? as u64;
                    let total_units = row.get::<_, i64>(10)? as u64;
                    let task_spec: ComputeTaskSpec = serde_json::from_str(&row.get::<_, String>(11)?).unwrap();
                    Ok((auction, requester, price_per_unit, total_units, task_spec))
                },
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // drop(db);  // Removed to avoid borrowing conflicts

        // Try to match with each suitable auction
        for (mut auction, requester, request_price, request_units, request_task_spec) in auctions {
            // Check if provider meets requirements
            let provider_reputation = self.reputation.get_reputation(&offer.trader).await?;
            
            if let Some(min_rep) = request_task_spec.min_reputation {
                if provider_reputation.score < min_rep {
                    continue; // Skip if reputation too low
                }
            }

            // Check if provider has required capabilities
            let has_capabilities = request_task_spec.required_capabilities.iter()
                .all(|cap| offer.task_spec.required_capabilities.contains(cap));
            
            if !has_capabilities {
                continue;
            }

            // Check price compatibility (provider willing to accept request price)
            if offer.price_per_unit > request_price {
                continue;
            }

            // Calculate effective price with reputation weighting
            let effective_price = offer.effective_price(provider_reputation.score);

            // Check if auction still has room
            if auction.accepted_offers.len() >= auction.max_providers as usize {
                continue;
            }

            // Create assignment
            let units_to_assign = offer.remaining_units().min(request_units);
            if units_to_assign == 0 {
                continue;
            }

            let assignment = self.create_assignment(
                auction.request_id,
                offer.id,
                requester,
                offer.trader,
                effective_price,
                units_to_assign,
            ).await?;

            // Update auction
            auction.accepted_offers.push(offer.id);
            if auction.accepted_offers.len() >= auction.min_providers as usize {
                auction.status = AuctionStatus::MinReached;
            }
            if auction.accepted_offers.len() >= auction.max_providers as usize {
                auction.status = AuctionStatus::Closed;
            }

            self.update_auction(&auction).await?;
            assignments.push(assignment);

            // Break if offer fully matched
            if offer.remaining_units() == 0 {
                break;
            }
        }

        Ok(assignments)
    }

    /// Create a task assignment
    async fn create_assignment(
        &self,
        request_id: Uuid,
        offer_id: Uuid,
        requester: PeerId,
        provider: PeerId,
        price_per_unit: u64,
        compute_units: u64,
    ) -> Result<TaskAssignment> {
        let assignment = TaskAssignment {
            id: Uuid::new_v4(),
            request_id,
            offer_id,
            requester,
            provider,
            price_per_unit,
            compute_units,
            total_cost: price_per_unit * compute_units,
            assigned_at: Utc::now(),
            started_at: None,
            completed_at: None,
            sla_metrics: SLAMetrics::default(),
            status: AssignmentStatus::Assigned,
        };

        // Update orders and create assignment atomically
        let db = self.db.lock().await;
        let tx = db.unchecked_transaction()?;

        // Update request order
        tx.execute(
            r#"
            UPDATE orders 
            SET filled_units = filled_units + ?2,
                status = CASE 
                    WHEN filled_units + ?2 = total_units THEN 'Filled'
                    ELSE 'PartiallyFilled'
                END,
                updated_at = ?3
            WHERE id = ?1
            "#,
            params![
                request_id.to_string(),
                compute_units as i64,
                Utc::now().to_rfc3339()
            ],
        )?;

        // Update offer order
        tx.execute(
            r#"
            UPDATE orders 
            SET filled_units = filled_units + ?2,
                status = CASE 
                    WHEN filled_units + ?2 = total_units THEN 'Filled'
                    ELSE 'PartiallyFilled'
                END,
                updated_at = ?3
            WHERE id = ?1
            "#,
            params![
                offer_id.to_string(),
                compute_units as i64,
                Utc::now().to_rfc3339()
            ],
        )?;

        // Insert assignment
        tx.execute(
            r#"
            INSERT INTO task_assignments (
                id, request_id, offer_id, requester, provider,
                price_per_unit, compute_units, total_cost,
                assigned_at, sla_metrics, status
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
            params![
                assignment.id.to_string(),
                assignment.request_id.to_string(),
                assignment.offer_id.to_string(),
                assignment.requester.to_string(),
                assignment.provider.to_string(),
                assignment.price_per_unit as i64,
                assignment.compute_units as i64,
                assignment.total_cost as i64,
                assignment.assigned_at.to_rfc3339(),
                serde_json::to_string(&assignment.sla_metrics)?,
                format!("{:?}", assignment.status),
            ],
        )?;

        tx.commit()?;

        Ok(assignment)
    }

    /// Update auction state
    async fn update_auction(&self, auction: &FirstAcceptAuction) -> Result<()> {
        let db = self.db.lock().await;
        db.execute(
            r#"
            UPDATE auctions
            SET accepted_offers = ?2, status = ?3
            WHERE id = ?1
            "#,
            params![
                auction.id.to_string(),
                serde_json::to_string(&auction.accepted_offers)?,
                format!("{:?}", auction.status),
            ],
        )?;
        Ok(())
    }

    /// Update price discovery data
    async fn update_price_discovery(&self, task_type: &str) -> Result<()> {
        let db = self.db.lock().await;
        
        // Calculate price statistics for the last 24 hours
        let stats: Option<(f64, f64, u64, u64, u64, u64)> = db.query_row(
            r#"
            SELECT 
                AVG(price_per_unit) as avg_price,
                SUM(price_per_unit * compute_units) / CAST(SUM(compute_units) AS REAL) as vwap,
                MIN(price_per_unit) as min_price,
                MAX(price_per_unit) as max_price,
                SUM(compute_units) as total_volume,
                COUNT(*) as assignment_count
            FROM task_assignments
            WHERE assigned_at > datetime('now', '-24 hours')
              AND request_id IN (
                  SELECT id FROM orders 
                  WHERE json_extract(task_spec, '$.task_type') = ?1
              )
            "#,
            params![task_type],
            |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, i64>(3)? as u64,
                    row.get::<_, i64>(4)? as u64,
                    row.get::<_, i64>(5)? as u64,
                ))
            },
        ).ok();

        if let Some((avg_price, vwap, min_price, max_price, total_volume, assignment_count)) = stats {
            db.execute(
                r#"
                INSERT OR REPLACE INTO price_discovery (
                    task_type, avg_price_24h, vwap, min_price, max_price,
                    total_volume, assignment_count, last_updated
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                "#,
                params![
                    task_type,
                    avg_price,
                    vwap,
                    min_price as i64,
                    max_price as i64,
                    total_volume as i64,
                    assignment_count as i64,
                    Utc::now().to_rfc3339(),
                ],
            )?;
        }

        Ok(())
    }

    /// Get task assignment history
    pub async fn get_assignments(
        &self,
        trader: Option<&PeerId>,
        limit: u32,
    ) -> Result<Vec<TaskAssignment>> {
        let db = self.db.lock().await;
        
        if let Some(trader) = trader {
            let trader_str = trader.to_string();
            let query = r#"
                SELECT id, request_id, offer_id, requester, provider,
                       price_per_unit, compute_units, total_cost,
                       assigned_at, started_at, completed_at,
                       sla_metrics, status
                FROM task_assignments
                WHERE requester = ?1 OR provider = ?1
                ORDER BY assigned_at DESC
                LIMIT ?2
                "#;
            let mut stmt = db.prepare(query)?;
            let assignments = stmt.query_map(params![trader_str, limit as i64], |row| {
                Ok(TaskAssignment {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    request_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap(),
                    offer_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap(),
                    requester: row.get::<_, String>(3)?.parse().unwrap(),
                    provider: row.get::<_, String>(4)?.parse().unwrap(),
                    price_per_unit: row.get::<_, i64>(5)? as u64,
                    compute_units: row.get::<_, i64>(6)? as u64,
                    total_cost: row.get::<_, i64>(7)? as u64,
                    assigned_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    started_at: row.get::<_, Option<String>>(9)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    completed_at: row.get::<_, Option<String>>(10)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    sla_metrics: serde_json::from_str(&row.get::<_, String>(11)?).unwrap(),
                    status: match row.get::<_, String>(12)?.as_str() {
                        "Assigned" => AssignmentStatus::Assigned,
                        "InProgress" => AssignmentStatus::InProgress,
                        "Completed" => AssignmentStatus::Completed,
                        "Failed" => AssignmentStatus::Failed,
                        "SLAViolated" => AssignmentStatus::SLAViolated,
                        "Disputed" => AssignmentStatus::Disputed,
                        _ => AssignmentStatus::Assigned,
                    },
                })
            })?.collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(assignments)
        } else {
            let query = r#"
                SELECT id, request_id, offer_id, requester, provider,
                       price_per_unit, compute_units, total_cost,
                       assigned_at, started_at, completed_at,
                       sla_metrics, status
                FROM task_assignments
                ORDER BY assigned_at DESC
                LIMIT ?1
                "#;
            let mut stmt = db.prepare(query)?;
            let assignments = stmt.query_map(params![limit as i64], |row| {
                Ok(TaskAssignment {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    request_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap(),
                    offer_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap(),
                    requester: row.get::<_, String>(3)?.parse().unwrap(),
                    provider: row.get::<_, String>(4)?.parse().unwrap(),
                    price_per_unit: row.get::<_, i64>(5)? as u64,
                    compute_units: row.get::<_, i64>(6)? as u64,
                    total_cost: row.get::<_, i64>(7)? as u64,
                    assigned_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    started_at: row.get::<_, Option<String>>(9)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    completed_at: row.get::<_, Option<String>>(10)?
                        .map(|s| DateTime::parse_from_rfc3339(&s).unwrap().with_timezone(&Utc)),
                    sla_metrics: serde_json::from_str(&row.get::<_, String>(11)?).unwrap(),
                    status: match row.get::<_, String>(12)?.as_str() {
                        "Assigned" => AssignmentStatus::Assigned,
                        "InProgress" => AssignmentStatus::InProgress,
                        "Completed" => AssignmentStatus::Completed,
                        "Failed" => AssignmentStatus::Failed,
                        "SLAViolated" => AssignmentStatus::SLAViolated,
                        "Disputed" => AssignmentStatus::Disputed,
                        _ => AssignmentStatus::Assigned,
                    },
                })
            })?.collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(assignments)
        }
    }

    /// Start task execution
    pub async fn start_task(
        &self,
        assignment_id: &Uuid,
        provider: &PeerId,
    ) -> Result<()> {
        let db = self.db.lock().await;
        
        // Verify provider owns the assignment
        let owner: String = db.query_row(
            "SELECT provider FROM task_assignments WHERE id = ?1",
            params![assignment_id.to_string()],
            |row| row.get(0),
        ).map_err(|_| MarketError::InvalidOrder("Assignment not found".to_string()))?;

        if owner != provider.to_string() {
            return Err(MarketError::InvalidOrder("Not assignment provider".to_string()));
        }

        // Update assignment status
        db.execute(
            r#"
            UPDATE task_assignments
            SET status = 'InProgress', started_at = ?2
            WHERE id = ?1 AND status = 'Assigned'
            "#,
            params![assignment_id.to_string(), Utc::now().to_rfc3339()],
        )?;

        Ok(())
    }

    /// Complete task execution with SLA tracking
    pub async fn complete_task(
        &self,
        assignment_id: &Uuid,
        provider: &PeerId,
        quality_scores: HashMap<String, f64>,
    ) -> Result<()> {
        let db = self.db.lock().await;
        
        // Get assignment details
        let (started_at, sla_spec_json): (String, Option<String>) = db.query_row(
            r#"
            SELECT a.started_at, o.sla_spec
            FROM task_assignments a
            JOIN orders o ON a.request_id = o.id
            WHERE a.id = ?1 AND a.provider = ?2
            "#,
            params![assignment_id.to_string(), provider.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).map_err(|_| MarketError::InvalidOrder("Assignment not found".to_string()))?;

        let started_at = DateTime::parse_from_rfc3339(&started_at).unwrap().with_timezone(&Utc);
        let completed_at = Utc::now();
        let execution_time = (completed_at - started_at).num_seconds() as u64;

        // Calculate SLA metrics
        let mut sla_metrics = SLAMetrics {
            response_time: Some(execution_time),
            execution_time: Some(execution_time),
            quality_scores: quality_scores.clone(),
            violations: 0,
            total_penalty: 0,
        };

        // Check SLA violations if spec exists
        if let Some(spec_json) = sla_spec_json {
            if let Ok(sla_spec) = serde_json::from_str::<SLASpec>(&spec_json) {
                // Check response time violation
                if execution_time > sla_spec.max_response_time {
                    sla_metrics.violations += 1;
                    sla_metrics.total_penalty += sla_spec.violation_penalty;
                }

                // Check quality metrics
                for (metric, required_score) in &sla_spec.quality_metrics {
                    if let Some(actual_score) = quality_scores.get(metric) {
                        if actual_score < required_score {
                            sla_metrics.violations += 1;
                            sla_metrics.total_penalty += sla_spec.violation_penalty / 2; // Half penalty for quality
                        }
                    }
                }
            }
        }

        // Update assignment
        let status = if sla_metrics.violations > 0 {
            AssignmentStatus::SLAViolated
        } else {
            AssignmentStatus::Completed
        };

        db.execute(
            r#"
            UPDATE task_assignments
            SET status = ?2, completed_at = ?3, sla_metrics = ?4
            WHERE id = ?1 AND status = 'InProgress'
            "#,
            params![
                assignment_id.to_string(),
                format!("{:?}", status),
                completed_at.to_rfc3339(),
                serde_json::to_string(&sla_metrics)?,
            ],
        )?;

        // Update provider response time
        drop(db);
        self.reputation.update_response_time(provider, execution_time as f64).await?;

        Ok(())
    }

    /// Get price discovery data for a task type
    pub async fn get_price_discovery(&self, task_type: &str) -> Result<Option<PriceDiscovery>> {
        let db = self.db.lock().await;
        
        let discovery = db.query_row(
            r#"
            SELECT task_type, avg_price_24h, vwap, min_price, max_price,
                   total_volume, assignment_count, last_updated
            FROM price_discovery
            WHERE task_type = ?1
            "#,
            params![task_type],
            |row| {
                Ok(PriceDiscovery {
                    task_type: row.get(0)?,
                    avg_price_24h: row.get(1)?,
                    vwap: row.get(2)?,
                    min_price: row.get::<_, i64>(3)? as u64,
                    max_price: row.get::<_, i64>(4)? as u64,
                    total_volume: row.get::<_, i64>(5)? as u64,
                    assignment_count: row.get::<_, i64>(6)? as u64,
                    last_updated: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            },
        ).ok();

        Ok(discovery)
    }

    /// Process expired auctions
    pub async fn process_expired_auctions(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        // Close expired auctions
        db.execute(
            r#"
            UPDATE auctions
            SET status = 'Closed'
            WHERE status IN ('Open', 'MinReached')
              AND ends_at < ?1
            "#,
            params![Utc::now().to_rfc3339()],
        )?;

        // Expire old orders
        db.execute(
            r#"
            UPDATE orders
            SET status = 'Expired'
            WHERE status IN ('Active', 'PartiallyFilled')
              AND expires_at IS NOT NULL
              AND expires_at < ?1
            "#,
            params![Utc::now().to_rfc3339()],
        )?;

        Ok(())
    }

    /// Update market conditions based on current activity
    async fn update_market_conditions(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        // Count active orders and pending requests
        let active_offers: i64 = db.query_row(
            "SELECT COUNT(*) FROM orders WHERE order_type = 'OfferCompute' AND status IN ('Active', 'PartiallyFilled')",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let pending_requests: i64 = db.query_row(
            "SELECT COUNT(*) FROM orders WHERE order_type = 'RequestCompute' AND status IN ('Active', 'PartiallyFilled')",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let total_assignments: i64 = db.query_row(
            "SELECT COUNT(*) FROM task_assignments WHERE status = 'InProgress'",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let avg_response_time: f64 = db.query_row(
            r#"
            SELECT AVG(CAST((julianday(completed_at) - julianday(assigned_at)) * 86400 AS REAL))
            FROM task_assignments 
            WHERE completed_at IS NOT NULL AND assigned_at > datetime('now', '-24 hours')
            "#,
            [],
            |row| row.get(0),
        ).unwrap_or(120.0);

        drop(db);

        // Calculate utilization and demand/supply levels
        let total_capacity = active_offers.max(1);
        let utilization_rate = total_assignments as f64 / total_capacity as f64;
        
        let demand_level = match pending_requests {
            0..=5 => DemandLevel::VeryLow,
            6..=15 => DemandLevel::Low,
            16..=30 => DemandLevel::Normal,
            31..=50 => DemandLevel::High,
            _ => DemandLevel::VeryHigh,
        };

        let supply_level = match active_offers {
            0..=5 => SupplyLevel::Scarce,
            6..=15 => SupplyLevel::Limited,
            16..=50 => SupplyLevel::Normal,
            _ => SupplyLevel::Abundant,
        };

        let conditions = MarketConditions {
            demand_level,
            supply_level,
            utilization_rate: utilization_rate.min(1.0),
            active_providers: active_offers as u64,
            pending_requests: pending_requests as u64,
            avg_response_time,
            avg_quality_score: 0.85, // Could be calculated from reputation scores
            last_updated: Utc::now(),
        };

        self.pricing_engine.update_market_conditions(conditions).await?;
        Ok(())
    }

    /// Get dynamic price quote for a task
    pub async fn get_price_quote(
        &self,
        task_spec: &ComputeTaskSpec,
        provider_reputation: Option<&crate::reputation::ReputationScore>,
        urgency_level: f64,
    ) -> Result<crate::pricing::PriceQuote> {
        self.pricing_engine
            .calculate_price(task_spec, PricingStrategy::Dynamic, provider_reputation, urgency_level)
            .await
    }

    /// Get market making recommendations
    pub async fn get_market_making_recommendations(&self, peer_id: &PeerId) -> Result<Vec<MarketMakingRecommendation>> {
        let reputation = self.reputation.get_reputation(peer_id).await?;
        let conditions = self.pricing_engine.get_market_conditions().await;
        let mut recommendations = Vec::new();

        // Analyze current order book for opportunities
        let (bids, offers) = self.get_order_book().await?;
        
        // Look for spread opportunities
        if !bids.is_empty() && !offers.is_empty() {
            let best_bid = bids.iter().max_by_key(|b| b.price_per_unit).unwrap();
            let best_offer = offers.iter().min_by_key(|o| o.price_per_unit).unwrap();
            
            if best_offer.price_per_unit > best_bid.price_per_unit {
                let spread = best_offer.price_per_unit - best_bid.price_per_unit;
                let mid_price = (best_bid.price_per_unit + best_offer.price_per_unit) / 2;
                
                recommendations.push(MarketMakingRecommendation {
                    action: MarketAction::PlaceBid,
                    price: mid_price - spread / 4,
                    quantity: best_bid.remaining_units().min(best_offer.remaining_units()) / 2,
                    reasoning: "Arbitrage opportunity - place bid below mid-market".to_string(),
                    confidence: 0.8,
                });

                recommendations.push(MarketMakingRecommendation {
                    action: MarketAction::PlaceOffer,
                    price: mid_price + spread / 4,
                    quantity: best_bid.remaining_units().min(best_offer.remaining_units()) / 2,
                    reasoning: "Arbitrage opportunity - place offer above mid-market".to_string(),
                    confidence: 0.8,
                });
            }
        }

        // High demand recommendations
        if matches!(conditions.demand_level, DemandLevel::High | DemandLevel::VeryHigh) {
            recommendations.push(MarketMakingRecommendation {
                action: MarketAction::PlaceOffer,
                price: 0, // To be calculated based on current prices + premium
                quantity: reputation.score as u64 / 10, // Scale with reputation
                reasoning: "High demand detected - increase supply".to_string(),
                confidence: 0.7,
            });
        }

        Ok(recommendations)
    }

    /// Execute automated market making
    pub async fn execute_market_making(
        &self,
        peer_id: &PeerId,
        strategy: MarketMakingStrategy,
        signing_key: &SigningKey,
    ) -> Result<Vec<Order>> {
        let recommendations = self.get_market_making_recommendations(peer_id).await?;
        let mut executed_orders = Vec::new();

        for rec in recommendations {
            if rec.confidence < strategy.min_confidence {
                continue;
            }

            let task_spec = ComputeTaskSpec {
                task_type: "general".to_string(), // Could be more specific
                compute_units: rec.quantity,
                max_duration_secs: 3600,
                required_capabilities: vec!["general".to_string()],
                min_reputation: None,
                privacy_level: PrivacyLevel::Public,
                encrypted_payload: None,
            };

            let order_type = match rec.action {
                MarketAction::PlaceBid => OrderType::RequestCompute,
                MarketAction::PlaceOffer => OrderType::OfferCompute,
            };

            let order = self.place_order(
                order_type,
                *peer_id,
                rec.price,
                rec.quantity,
                task_spec,
                None,
                Some(Utc::now() + Duration::minutes(strategy.order_lifetime_minutes)),
                Some(signing_key),
            ).await?;

            executed_orders.push(order);
        }

        Ok(executed_orders)
    }

    /// Get liquidity metrics for the market
    pub async fn get_liquidity_metrics(&self) -> Result<LiquidityMetrics> {
        let (bids, offers) = self.get_order_book().await?;
        
        let bid_volume: u64 = bids.iter().map(|b| b.remaining_units()).sum();
        let offer_volume: u64 = offers.iter().map(|o| o.remaining_units()).sum();
        
        let best_bid_price = bids.iter().map(|b| b.price_per_unit).max().unwrap_or(0);
        let best_offer_price = offers.iter().map(|o| o.price_per_unit).min().unwrap_or(u64::MAX);
        
        let spread = if best_offer_price != u64::MAX && best_bid_price > 0 {
            Some(best_offer_price.saturating_sub(best_bid_price))
        } else {
            None
        };

        Ok(LiquidityMetrics {
            bid_volume,
            offer_volume,
            total_volume: bid_volume + offer_volume,
            spread,
            bid_count: bids.len() as u64,
            offer_count: offers.len() as u64,
            depth_score: (bid_volume + offer_volume) as f64 / 1000.0, // Normalized depth
        })
    }

    /// Get network-wide market statistics from P2P
    pub async fn get_network_market_stats(&self) -> Result<NetworkMarketStats> {
        /*
        if let Some(ref p2p) = self.p2p_network {
            let health = p2p.lock().await.network_health().await;
            let connected_peers = p2p.lock().await.get_connected_peers().await;
            
            let total_active_orders: u64 = connected_peers
                .values()
                .map(|p| p.market_stats.active_orders)
                .sum();
            
            let total_capacity: u64 = connected_peers
                .values()
                .map(|p| p.market_stats.available_capacity)
                .sum();
            
            let avg_response_time = if !connected_peers.is_empty() {
                connected_peers
                    .values()
                    .filter_map(|p| p.market_stats.avg_response_time)
                    .sum::<f64>() / connected_peers.len() as f64
            } else {
                0.0
            };

            Ok(NetworkMarketStats {
                total_peers: health.total_peers as u64,
                active_peers: health.active_peers as u64,
                total_active_orders,
                total_capacity,
                network_utilization: if total_capacity > 0 {
                    total_active_orders as f64 / total_capacity as f64
                } else {
                    0.0
                },
                avg_network_response_time: avg_response_time,
                network_health_score: health.avg_reputation,
            })
        } else */
        {
            // Return local stats only
            let local_metrics = self.get_liquidity_metrics().await?;
            Ok(NetworkMarketStats {
                total_peers: 1,
                active_peers: 1,
                total_active_orders: local_metrics.bid_count + local_metrics.offer_count,
                total_capacity: local_metrics.total_volume,
                network_utilization: 0.5, // Default assumption
                avg_network_response_time: 120.0,
                network_health_score: 75.0,
            })
        }
    }
}

/// Market making recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingRecommendation {
    /// Recommended action
    pub action: MarketAction,
    /// Recommended price
    pub price: u64,
    /// Recommended quantity
    pub quantity: u64,
    /// Reasoning for the recommendation
    pub reasoning: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// Market making action
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MarketAction {
    /// Place a bid order
    PlaceBid,
    /// Place an offer order
    PlaceOffer,
}

/// Market making strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingStrategy {
    /// Minimum confidence required to execute
    pub min_confidence: f64,
    /// Maximum spread to maintain
    pub max_spread: u64,
    /// Order lifetime in minutes
    pub order_lifetime_minutes: i64,
    /// Maximum position size
    pub max_position: u64,
}

/// Liquidity metrics for the market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    /// Total bid volume
    pub bid_volume: u64,
    /// Total offer volume
    pub offer_volume: u64,
    /// Total volume (bids + offers)
    pub total_volume: u64,
    /// Current spread (best offer - best bid)
    pub spread: Option<u64>,
    /// Number of bid orders
    pub bid_count: u64,
    /// Number of offer orders
    pub offer_count: u64,
    /// Market depth score
    pub depth_score: f64,
}

/// Network-wide market statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMarketStats {
    /// Total number of peers in network
    pub total_peers: u64,
    /// Number of active peers
    pub active_peers: u64,
    /// Total active orders across network
    pub total_active_orders: u64,
    /// Total compute capacity across network
    pub total_capacity: u64,
    /// Network utilization rate
    pub network_utilization: f64,
    /// Average response time across network
    pub avg_network_response_time: f64,
    /// Overall network health score
    pub network_health_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn test_compute_market_operations() {
        let market = Market::new(":memory:").await.unwrap();
        market.init_schema().await.unwrap();
        market.reputation.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider = PeerId::random();
        let signing_key = SigningKey::generate(&mut OsRng);

        // Create compute request
        let task_spec = ComputeTaskSpec {
            task_type: "code_generation".to_string(),
            compute_units: 100,
            max_duration_secs: 300,
            required_capabilities: vec!["rust".to_string(), "python".to_string()],
            min_reputation: Some(50.0),
            privacy_level: PrivacyLevel::Private,
            encrypted_payload: None,
        };

        let sla_spec = SLASpec {
            uptime_requirement: 99.0,
            max_response_time: 60,
            violation_penalty: 10,
            quality_metrics: HashMap::from([
                ("accuracy".to_string(), 0.9),
                ("completeness".to_string(), 0.95),
            ]),
        };

        // Place compute request (starts auction)
        let request = market
            .place_order(
                OrderType::RequestCompute,
                requester,
                50, // 50 tokens per unit
                100, // 100 units
                task_spec.clone(),
                Some(sla_spec),
                None,
                Some(&signing_key),
            )
            .await
            .unwrap();
        assert_eq!(request.price_per_unit, 50);
        assert_eq!(request.total_units, 100);
        assert!(request.signature.is_some());

        // Place compute offer
        let offer_task_spec = ComputeTaskSpec {
            task_type: "code_generation".to_string(),
            compute_units: 200,
            max_duration_secs: 600,
            required_capabilities: vec!["rust".to_string(), "python".to_string(), "javascript".to_string()],
            min_reputation: None,
            privacy_level: PrivacyLevel::Private,
            encrypted_payload: None,
        };

        let offer = market
            .place_order(
                OrderType::OfferCompute,
                provider,
                40, // Willing to work for 40 tokens per unit
                200, // Can handle 200 units
                offer_task_spec,
                None,
                None,
                Some(&signing_key),
            )
            .await
            .unwrap();

        // Check that assignment was created
        let assignments = market.get_assignments(None, 10).await.unwrap();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].compute_units, 100);
        assert!(assignments[0].price_per_unit <= 50); // Should use effective price

        // Start task
        market
            .start_task(&assignments[0].id, &provider)
            .await
            .unwrap();

        // Complete task with quality scores
        let quality_scores = HashMap::from([
            ("accuracy".to_string(), 0.95),
            ("completeness".to_string(), 0.98),
        ]);
        
        market
            .complete_task(&assignments[0].id, &provider, quality_scores)
            .await
            .unwrap();

        // Check assignment status
        let updated_assignments = market.get_assignments(Some(&provider), 10).await.unwrap();
        assert_eq!(updated_assignments[0].status, AssignmentStatus::Completed);
        assert_eq!(updated_assignments[0].sla_metrics.violations, 0);

        // Check price discovery
        let price_data = market
            .get_price_discovery("code_generation")
            .await
            .unwrap();
        assert!(price_data.is_some());
    }

    #[tokio::test]
    async fn test_reputation_weighted_matching() {
        let market = Market::new(":memory:").await.unwrap();
        market.init_schema().await.unwrap();
        market.reputation.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider1 = PeerId::random();
        let provider2 = PeerId::random();

        // Give provider2 better reputation
        market.reputation
            .record_event(&provider2, crate::reputation::ReputationEvent::TradeCompleted, None, None)
            .await
            .unwrap();
        market.reputation
            .record_event(&provider2, crate::reputation::ReputationEvent::TradeCompleted, None, None)
            .await
            .unwrap();

        // Create request requiring minimum reputation
        let task_spec = ComputeTaskSpec {
            task_type: "ml_training".to_string(),
            compute_units: 50,
            max_duration_secs: 3600,
            required_capabilities: vec!["cuda".to_string()],
            min_reputation: Some(60.0), // Only provider2 meets this
            privacy_level: PrivacyLevel::Confidential,
            encrypted_payload: None,
        };

        let request = market
            .place_order(
                OrderType::RequestCompute,
                requester,
                100,
                50,
                task_spec.clone(),
                None,
                None,
                None,
            )
            .await
            .unwrap();

        // Provider1 offer (should not match due to low reputation)
        let offer1 = market
            .place_order(
                OrderType::OfferCompute,
                provider1,
                80,
                100,
                task_spec.clone(),
                None,
                None,
                None,
            )
            .await
            .unwrap();

        // Provider2 offer (should match)
        let offer2 = market
            .place_order(
                OrderType::OfferCompute,
                provider2,
                90,
                100,
                task_spec.clone(),
                None,
                None,
                None,
            )
            .await
            .unwrap();

        // Check assignments
        let assignments = market.get_assignments(None, 10).await.unwrap();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].provider, provider2);
    }
}