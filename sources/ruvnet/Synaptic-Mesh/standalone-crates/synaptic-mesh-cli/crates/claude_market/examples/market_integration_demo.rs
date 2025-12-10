//! Comprehensive Synaptic Market Integration Demo
//!
//! This example demonstrates the full integration of the Synaptic Market system,
//! including P2P networking, dynamic pricing, escrow, reputation, and market making.

use claude_market::{
    error::Result,
    escrow::{Escrow, EscrowState, MultiSigType},
    market::{Market, OrderType, ComputeTaskSpec, PrivacyLevel, MarketMakingStrategy},
    p2p::{P2PNetwork, P2PConfig},
    pricing::{PricingEngine, PricingStrategy, MarketConditions, DemandLevel, SupplyLevel},
    reputation::{Reputation, ReputationEvent},
    wallet::Wallet,
};
use ed25519_dalek::SigningKey;
use libp2p::PeerId;
use rand::rngs::OsRng;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting Synaptic Market Integration Demo");
    
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Initialize the market with P2P networking
    let demo = MarketDemo::new().await?;
    
    // Run the comprehensive demo
    demo.run_full_demo().await?;
    
    println!("✅ Demo completed successfully!");
    Ok(())
}

/// Complete market demonstration
struct MarketDemo {
    market: Market,
    escrow: Escrow,
    pricing_engine: Arc<PricingEngine>,
    requester: PeerId,
    provider: PeerId,
    requester_key: SigningKey,
    provider_key: SigningKey,
    wallet: Arc<Wallet>,
}

impl MarketDemo {
    /// Initialize the demo environment
    async fn new() -> Result<Self> {
        let db_path = ":memory:"; // Use in-memory database for demo
        
        // Create P2P configuration
        let p2p_config = P2PConfig {
            local_peer_id: PeerId::random(),
            listen_addresses: vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()],
            bootstrap_addresses: vec![],
            max_connections: 50,
            connection_timeout: Duration::from_secs(30),
            record_ttl: Duration::from_secs(3600),
        };
        
        // Initialize market with P2P
        let mut market = Market::new_with_p2p(db_path, p2p_config).await?;
        market.init_schema().await?;
        market.enable_p2p().await?;
        
        // Initialize wallet
        let wallet = Arc::new(Wallet::new(db_path).await?);
        wallet.init_schema().await?;
        
        // Initialize escrow
        let escrow = Escrow::new(db_path, wallet.clone()).await?;
        escrow.init_schema().await?;
        
        // Initialize pricing engine
        let pricing_engine = Arc::new(PricingEngine::new(db_path).await?);
        pricing_engine.init_schema().await?;
        
        // Create test users
        let requester = PeerId::random();
        let provider = PeerId::random();
        let requester_key = SigningKey::generate(&mut OsRng);
        let provider_key = SigningKey::generate(&mut OsRng);
        
        // Fund the requester wallet
        wallet.credit(&requester, 10000).await?;
        wallet.credit(&provider, 1000).await?;
        
        println!("✅ Market demo environment initialized");
        println!("  - Requester: {}", requester);
        println!("  - Provider: {}", provider);
        println!("  - Requester balance: {}", wallet.get_balance(&requester).await?.available);
        println!("  - Provider balance: {}", wallet.get_balance(&provider).await?.available);
        
        Ok(Self {
            market,
            escrow,
            pricing_engine,
            requester,
            provider,
            requester_key,
            provider_key,
            wallet,
        })
    }
    
    /// Run the complete demonstration
    async fn run_full_demo(&self) -> Result<()> {
        println!("\n🎯 Running Full Market Integration Demo\n");
        
        // 1. Dynamic Pricing Demo
        self.demo_dynamic_pricing().await?;
        
        // 2. Market Operations Demo
        self.demo_market_operations().await?;
        
        // 3. Escrow Operations Demo
        self.demo_escrow_operations().await?;
        
        // 4. Reputation System Demo
        self.demo_reputation_system().await?;
        
        // 5. Market Making Demo
        self.demo_market_making().await?;
        
        // 6. Network Statistics Demo
        self.demo_network_statistics().await?;
        
        Ok(())
    }
    
    /// Demonstrate dynamic pricing features
    async fn demo_dynamic_pricing(&self) -> Result<()> {
        println!("📊 === Dynamic Pricing Demo ===");
        
        // Update market conditions to simulate different scenarios
        let high_demand_conditions = MarketConditions {
            demand_level: DemandLevel::High,
            supply_level: SupplyLevel::Limited,
            utilization_rate: 0.85,
            active_providers: 5,
            pending_requests: 25,
            avg_response_time: 180.0,
            avg_quality_score: 0.88,
            last_updated: chrono::Utc::now(),
        };
        
        self.pricing_engine.update_market_conditions(high_demand_conditions).await?;
        
        // Test different task types and pricing strategies
        let task_types = vec![
            ("code_generation", "Generate a REST API"),
            ("data_analysis", "Analyze customer data trends"),
            ("ml_training", "Train a neural network model"),
        ];
        
        for (task_type, description) in task_types {
            let task_spec = ComputeTaskSpec {
                task_type: task_type.to_string(),
                compute_units: 100,
                max_duration_secs: 3600,
                required_capabilities: vec!["rust".to_string(), "python".to_string()],
                min_reputation: Some(75.0),
                privacy_level: PrivacyLevel::Private,
                encrypted_payload: None,
            };
            
            // Test different pricing strategies
            for strategy in [PricingStrategy::Fixed, PricingStrategy::Dynamic, PricingStrategy::ReputationWeighted] {
                let quote = self.pricing_engine
                    .calculate_price(&task_spec, strategy, None, 0.8)
                    .await?;
                
                println!("  📋 Task: {} ({:?})", description, strategy);
                println!("    💰 Price: {} tokens/unit", quote.price_per_unit);
                println!("    🎯 Confidence: {:.2}", quote.confidence);
                println!("    ⏰ Valid until: {}", quote.valid_until.format("%H:%M:%S"));
                println!("    💭 Reasoning: {}", quote.reasoning);
                println!();
            }
        }
        
        Ok(())
    }
    
    /// Demonstrate market operations
    async fn demo_market_operations(&self) -> Result<()> {
        println!("🏪 === Market Operations Demo ===");
        
        // Create a compute request
        let task_spec = ComputeTaskSpec {
            task_type: "neural_inference".to_string(),
            compute_units: 50,
            max_duration_secs: 1800,
            required_capabilities: vec!["gpu".to_string(), "cuda".to_string()],
            min_reputation: Some(80.0),
            privacy_level: PrivacyLevel::Confidential,
            encrypted_payload: Some(b"encrypted_model_data".to_vec()),
        };
        
        // Place a compute request order
        let request_order = self.market.place_order(
            OrderType::RequestCompute,
            self.requester,
            75, // 75 tokens per compute unit
            50, // 50 compute units
            task_spec.clone(),
            None,
            Some(chrono::Utc::now() + chrono::Duration::hours(2)),
            Some(&self.requester_key),
        ).await?;
        
        println!("  📝 Placed compute request:");
        println!("    🆔 Order ID: {}", request_order.id);
        println!("    💰 Price: {} tokens/unit", request_order.price_per_unit);
        println!("    📊 Units: {}", request_order.total_units);
        println!("    🔐 Privacy: {:?}", request_order.task_spec.privacy_level);
        
        // Create a competing provider offer
        let offer_task_spec = ComputeTaskSpec {
            task_type: "neural_inference".to_string(),
            compute_units: 100,
            max_duration_secs: 1200,
            required_capabilities: vec!["gpu".to_string(), "cuda".to_string(), "tensorrt".to_string()],
            min_reputation: None,
            privacy_level: PrivacyLevel::Confidential,
            encrypted_payload: None,
        };
        
        let offer_order = self.market.place_order(
            OrderType::OfferCompute,
            self.provider,
            70, // Willing to work for 70 tokens per unit
            100,
            offer_task_spec,
            None,
            Some(chrono::Utc::now() + chrono::Duration::hours(4)),
            Some(&self.provider_key),
        ).await?;
        
        println!("  🎯 Placed compute offer:");
        println!("    🆔 Order ID: {}", offer_order.id);
        println!("    💰 Price: {} tokens/unit", offer_order.price_per_unit);
        println!("    📊 Units: {}", offer_order.total_units);
        println!("    🚀 Extra capabilities: {:?}", offer_order.task_spec.required_capabilities);
        
        // Check for any created assignments
        let assignments = self.market.get_assignments(None, 10).await?;
        if !assignments.is_empty() {
            println!("  ✅ Automatic matching created {} assignment(s)", assignments.len());
            for assignment in &assignments {
                println!("    📋 Assignment: {} -> {}", assignment.requester, assignment.provider);
                println!("    💰 Cost: {} tokens", assignment.total_cost);
            }
        }
        
        // Display order book
        let (bids, offers) = self.market.get_order_book().await?;
        println!("  📚 Current Order Book:");
        println!("    📈 Bids: {}", bids.len());
        println!("    📉 Offers: {}", offers.len());
        
        Ok(())
    }
    
    /// Demonstrate escrow operations
    async fn demo_escrow_operations(&self) -> Result<()> {
        println!("🔒 === Escrow Operations Demo ===");
        
        let job_id = Uuid::new_v4();
        let escrow_amount = 1000;
        
        // Create escrow agreement
        let agreement = self.escrow.create_escrow(
            job_id,
            self.requester,
            self.provider,
            escrow_amount,
            MultiSigType::Single,
            vec![], // No arbitrators for simple escrow
            60, // 60 minute timeout
        ).await?;
        
        println!("  🤝 Created escrow agreement:");
        println!("    🆔 Escrow ID: {}", agreement.id);
        println!("    💰 Amount: {} tokens", agreement.amount);
        println!("    📅 Timeout: {}", agreement.timeout_at.format("%Y-%m-%d %H:%M:%S"));
        println!("    🔄 State: {:?}", agreement.state);
        
        // Fund the escrow
        let funded_agreement = self.escrow.fund_escrow(
            &agreement.id,
            &self.requester,
            &self.requester_key,
        ).await?;
        
        println!("  💳 Funded escrow:");
        println!("    🔄 State: {:?}", funded_agreement.state);
        
        // Check wallet balances after funding
        let req_balance = self.wallet.get_balance(&self.requester).await?;
        println!("    💰 Requester balance: {} available, {} locked", 
                req_balance.available, req_balance.locked);
        
        // Simulate job completion
        println!("  ⏳ Simulating job execution...");
        sleep(Duration::from_millis(100)).await;
        
        let completed_agreement = self.escrow.mark_completed(
            &agreement.id,
            &self.provider,
            &self.provider_key,
        ).await?;
        
        println!("  ✅ Job marked as completed:");
        println!("    🔄 State: {:?}", completed_agreement.state);
        
        // Get audit log
        let audit_log = self.escrow.get_audit_log(&agreement.id).await?;
        println!("  📋 Audit log ({} entries):", audit_log.len());
        for entry in audit_log {
            println!("    📝 {}: {} ({:?} -> {:?})", 
                    entry.timestamp.format("%H:%M:%S"),
                    entry.action,
                    entry.from_state,
                    entry.to_state);
        }
        
        Ok(())
    }
    
    /// Demonstrate reputation system
    async fn demo_reputation_system(&self) -> Result<()> {
        println!("⭐ === Reputation System Demo ===");
        
        // Get initial reputation scores
        let req_reputation = self.market.reputation.get_reputation(&self.requester).await?;
        let prov_reputation = self.market.reputation.get_reputation(&self.provider).await?;
        
        println!("  📊 Initial reputation scores:");
        println!("    🙋 Requester: {:.1} ({} trades, {:.1}% success rate)", 
                req_reputation.score, req_reputation.total_trades, req_reputation.success_rate());
        println!("    🛠️  Provider: {:.1} ({} trades, {:.1}% success rate)", 
                prov_reputation.score, prov_reputation.total_trades, prov_reputation.success_rate());
        
        // Record some reputation events
        let trade_id = Uuid::new_v4();
        
        // Provider completes trade successfully
        self.market.reputation.record_event(
            &self.provider,
            ReputationEvent::TradeCompleted,
            Some(trade_id),
            None,
        ).await?;
        
        // Provider gets fast response bonus
        self.market.reputation.update_response_time(&self.provider, 45.0).await?;
        
        // Submit positive feedback
        self.market.reputation.submit_feedback(
            trade_id,
            self.requester,
            self.provider,
            5,
            Some("Excellent work, delivered ahead of schedule!".to_string()),
        ).await?;
        
        // Get updated scores
        let updated_prov_reputation = self.market.reputation.get_reputation(&self.provider).await?;
        
        println!("  📈 Updated provider reputation:");
        println!("    ⭐ Score: {:.1} (tier: {})", 
                updated_prov_reputation.score, updated_prov_reputation.tier());
        println!("    🏃 Avg response time: {:.1}s", 
                updated_prov_reputation.avg_response_time.unwrap_or(0.0));
        
        // Get feedback
        let feedback = self.market.reputation.get_feedback(&self.provider, 5).await?;
        println!("  💬 Recent feedback ({} items):", feedback.len());
        for fb in feedback {
            println!("    ⭐ {} stars: {}", fb.rating, 
                    fb.comment.as_ref().unwrap_or(&"No comment".to_string()));
        }
        
        Ok(())
    }
    
    /// Demonstrate market making functionality
    async fn demo_market_making(&self) -> Result<()> {
        println!("🎯 === Market Making Demo ===");
        
        // Get market making recommendations
        let recommendations = self.market.get_market_making_recommendations(&self.provider).await?;
        
        println!("  🎲 Market making recommendations ({} items):", recommendations.len());
        for rec in &recommendations {
            println!("    📋 Action: {:?}", rec.action);
            println!("    💰 Price: {} tokens", rec.price);
            println!("    📊 Quantity: {} units", rec.quantity);
            println!("    🎯 Confidence: {:.2}", rec.confidence);
            println!("    💭 Reasoning: {}", rec.reasoning);
            println!();
        }
        
        // Execute market making strategy
        let strategy = MarketMakingStrategy {
            min_confidence: 0.6,
            max_spread: 50,
            order_lifetime_minutes: 30,
            max_position: 500,
        };
        
        let executed_orders = self.market.execute_market_making(
            &self.provider,
            strategy,
            &self.provider_key,
        ).await?;
        
        println!("  ⚡ Executed {} market making orders", executed_orders.len());
        for order in executed_orders {
            println!("    📝 Order: {} {} units at {} tokens/unit", 
                    format!("{:?}", order.order_type), order.total_units, order.price_per_unit);
        }
        
        // Get liquidity metrics
        let liquidity = self.market.get_liquidity_metrics().await?;
        println!("  💧 Liquidity metrics:");
        println!("    📊 Total volume: {} units", liquidity.total_volume);
        println!("    📈 Bid count: {}", liquidity.bid_count);
        println!("    📉 Offer count: {}", liquidity.offer_count);
        println!("    📏 Spread: {:?} tokens", liquidity.spread);
        println!("    🌊 Depth score: {:.2}", liquidity.depth_score);
        
        Ok(())
    }
    
    /// Demonstrate network statistics and health
    async fn demo_network_statistics(&self) -> Result<()> {
        println!("📊 === Network Statistics Demo ===");
        
        // Get network market statistics
        let network_stats = self.market.get_network_market_stats().await?;
        
        println!("  🌐 Network statistics:");
        println!("    👥 Total peers: {}", network_stats.total_peers);
        println!("    ✅ Active peers: {}", network_stats.active_peers);
        println!("    📋 Total active orders: {}", network_stats.total_active_orders);
        println!("    💻 Total capacity: {} units", network_stats.total_capacity);
        println!("    📊 Network utilization: {:.1}%", network_stats.network_utilization * 100.0);
        println!("    ⏱️  Avg response time: {:.1}s", network_stats.avg_network_response_time);
        println!("    ❤️  Health score: {:.1}/100", network_stats.network_health_score);
        
        // Get current market conditions
        let conditions = self.pricing_engine.get_market_conditions().await;
        println!("  🌡️  Market conditions:");
        println!("    📈 Demand: {:?}", conditions.demand_level);
        println!("    📦 Supply: {:?}", conditions.supply_level);
        println!("    🔥 Utilization: {:.1}%", conditions.utilization_rate * 100.0);
        println!("    ⭐ Quality score: {:.2}", conditions.avg_quality_score);
        
        // Analyze pricing trends
        let trend = self.pricing_engine.analyze_trends("neural_inference", 7).await?;
        println!("  📈 Price trend analysis (7 days):");
        println!("    📊 Direction: {:?}", trend.direction);
        println!("    💪 Strength: {:.2}", trend.strength);
        println!("    💰 Avg price: {} tokens", trend.avg_price);
        println!("    📊 Price change: {:.1}%", trend.price_change);
        
        Ok(())
    }
}

/// Additional utility functions for demo
impl MarketDemo {
    /// Display a separator for better output formatting
    #[allow(dead_code)]
    fn print_separator(&self, title: &str) {
        println!("\n{:=^60}", format!(" {} ", title));
    }
    
    /// Simulate network latency for realistic demo
    #[allow(dead_code)]
    async fn simulate_network_delay(&self) {
        sleep(Duration::from_millis(50)).await;
    }
}