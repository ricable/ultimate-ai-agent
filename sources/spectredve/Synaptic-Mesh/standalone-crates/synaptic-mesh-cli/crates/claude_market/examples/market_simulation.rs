//! Market simulation demonstrating compute contribution trading
//! 
//! This example shows how the Synaptic Market enables peer compute federation
//! where participants voluntarily contribute compute resources and are rewarded
//! with tokens for successful task completions.

use claude_market::{
    Market, OrderType, ComputeTaskSpec, PrivacyLevel, SLASpec,
    Reputation, ReputationEvent,
};
use libp2p::PeerId;
use chrono::Utc;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Synaptic Market Simulation - Compute Contribution Trading\n");
    println!("This simulation demonstrates a peer compute federation where:");
    println!("- Contributors run their own Claude Max accounts locally");
    println!("- Tasks are routed, not account access");
    println!("- Tokens reward successful completions");
    println!("- Participation is voluntary and contribution-based\n");

    // Initialize market and reputation system
    let market = Market::new("market_sim.db").await?;
    market.init_schema().await?;
    let reputation = Reputation::new("market_sim.db").await?;
    reputation.init_schema().await?;

    // Create participants
    let requester1 = PeerId::random();
    let requester2 = PeerId::random();
    let provider1 = PeerId::random();
    let provider2 = PeerId::random();
    let provider3 = PeerId::random();

    println!("📋 Market Participants:");
    println!("  Requesters: {} users needing compute", 2);
    println!("  Providers: {} users offering compute\n", 3);

    // Give providers different reputation levels
    println!("📊 Setting up provider reputations...");
    
    // Provider 1: New provider
    println!("  Provider 1: New contributor (50 reputation)");
    
    // Provider 2: Experienced provider
    for _ in 0..5 {
        reputation.record_event(&provider2, ReputationEvent::TradeCompleted, None, None).await?;
    }
    reputation.record_event(&provider2, ReputationEvent::PositiveFeedback, None, None).await?;
    let p2_rep = reputation.get_reputation(&provider2).await?;
    println!("  Provider 2: Experienced ({:.0} reputation)", p2_rep.score);

    // Provider 3: Top-tier provider
    for _ in 0..10 {
        reputation.record_event(&provider3, ReputationEvent::TradeCompleted, None, None).await?;
    }
    for _ in 0..3 {
        reputation.record_event(&provider3, ReputationEvent::PositiveFeedback, None, None).await?;
    }
    reputation.record_event(&provider3, ReputationEvent::FastResponse, None, None).await?;
    let p3_rep = reputation.get_reputation(&provider3).await?;
    println!("  Provider 3: Elite contributor ({:.0} reputation)\n", p3_rep.score);

    // Scenario 1: Code generation task with moderate requirements
    println!("📝 Scenario 1: Code Generation Task");
    println!("  Requester needs Rust code generation");
    println!("  100 compute units, willing to pay 50 tokens/unit");
    println!("  Requires provider with 60+ reputation\n");

    let code_gen_spec = ComputeTaskSpec {
        task_type: "code_generation".to_string(),
        compute_units: 100,
        max_duration_secs: 300,
        required_capabilities: vec!["rust".to_string(), "algorithms".to_string()],
        min_reputation: Some(60.0),
        privacy_level: PrivacyLevel::Private,
        encrypted_payload: Some(vec![1, 2, 3, 4]), // Simulated encrypted task details
    };

    let sla_spec = SLASpec {
        uptime_requirement: 99.0,
        max_response_time: 60,
        violation_penalty: 20,
        quality_metrics: HashMap::from([
            ("accuracy".to_string(), 0.95),
            ("code_quality".to_string(), 0.90),
        ]),
    };

    let signing_key = SigningKey::generate(&mut OsRng);

    // Place request (starts auction)
    let request1 = market.place_order(
        OrderType::RequestCompute,
        requester1,
        50,
        100,
        code_gen_spec.clone(),
        Some(sla_spec),
        None,
        Some(&signing_key),
    ).await?;

    println!("  ✅ Request placed, auction started");
    println!("  ⏳ Waiting for provider offers...\n");

    sleep(Duration::from_millis(500)).await;

    // Provider 1 tries to offer (will fail due to low reputation)
    println!("  Provider 1 offers at 45 tokens/unit...");
    let offer1 = market.place_order(
        OrderType::OfferCompute,
        provider1,
        45,
        150,
        code_gen_spec.clone(),
        None,
        None,
        Some(&signing_key),
    ).await?;
    
    let assignments = market.get_assignments(None, 10).await?;
    if assignments.is_empty() {
        println!("  ❌ Provider 1 rejected - reputation too low\n");
    }

    // Provider 2 offers (should match)
    println!("  Provider 2 offers at 48 tokens/unit...");
    let offer2 = market.place_order(
        OrderType::OfferCompute,
        provider2,
        48,
        200,
        code_gen_spec.clone(),
        None,
        None,
        Some(&signing_key),
    ).await?;

    let assignments = market.get_assignments(None, 10).await?;
    if !assignments.is_empty() {
        let assignment = &assignments[0];
        println!("  ✅ Provider 2 matched!");
        println!("  💰 Effective price: {} tokens/unit", assignment.price_per_unit);
        println!("  📦 Assigned units: {}", assignment.compute_units);
        println!("  💵 Total cost: {} tokens\n", assignment.total_cost);

        // Simulate task execution
        println!("  🔄 Provider 2 starting task...");
        market.start_task(&assignment.id, &provider2).await?;
        
        sleep(Duration::from_millis(1000)).await;
        
        // Complete with good quality
        let quality_scores = HashMap::from([
            ("accuracy".to_string(), 0.98),
            ("code_quality".to_string(), 0.95),
        ]);
        
        market.complete_task(&assignment.id, &provider2, quality_scores).await?;
        println!("  ✅ Task completed successfully!");
        println!("  📊 Quality scores: accuracy=0.98, code_quality=0.95");
        println!("  🏆 No SLA violations\n");
    }

    // Scenario 2: ML training task requiring high reputation
    println!("📝 Scenario 2: ML Training Task");
    println!("  Requester needs distributed ML training");
    println!("  500 compute units, willing to pay 80 tokens/unit");
    println!("  Requires provider with 100+ reputation\n");

    let ml_spec = ComputeTaskSpec {
        task_type: "ml_training".to_string(),
        compute_units: 500,
        max_duration_secs: 7200,
        required_capabilities: vec!["cuda".to_string(), "pytorch".to_string(), "distributed".to_string()],
        min_reputation: Some(100.0),
        privacy_level: PrivacyLevel::Confidential,
        encrypted_payload: Some(vec![5, 6, 7, 8]),
    };

    let request2 = market.place_order(
        OrderType::RequestCompute,
        requester2,
        80,
        500,
        ml_spec.clone(),
        None,
        None,
        None,
    ).await?;

    println!("  ✅ Request placed, auction started");
    
    // Only provider 3 meets reputation requirement
    let offer3 = market.place_order(
        OrderType::OfferCompute,
        provider3,
        75,
        1000,
        ml_spec,
        None,
        None,
        None,
    ).await?;

    let ml_assignments = market.get_assignments(None, 10).await?;
    if ml_assignments.len() > assignments.len() {
        println!("  ✅ Provider 3 matched (only one meeting reputation requirement)");
        println!("  💰 Price: 75 tokens/unit (below request price)");
        println!("  📦 500 units assigned\n");
    }

    // Show price discovery
    println!("📈 Price Discovery Data:");
    if let Some(code_gen_price) = market.get_price_discovery("code_generation").await? {
        println!("  Code Generation:");
        println!("    Average price (24h): {:.2} tokens/unit", code_gen_price.avg_price_24h);
        println!("    Volume-weighted avg: {:.2} tokens/unit", code_gen_price.vwap);
        println!("    Price range: {} - {} tokens/unit", code_gen_price.min_price, code_gen_price.max_price);
        println!("    Total volume: {} units", code_gen_price.total_volume);
    }

    if let Some(ml_price) = market.get_price_discovery("ml_training").await? {
        println!("\n  ML Training:");
        println!("    Average price (24h): {:.2} tokens/unit", ml_price.avg_price_24h);
        println!("    Volume-weighted avg: {:.2} tokens/unit", ml_price.vwap);
        println!("    Price range: {} - {} tokens/unit", ml_price.min_price, ml_price.max_price);
    }

    println!("\n✅ Simulation complete!");
    println!("\n📝 Key Takeaways:");
    println!("  • First-accept auction model enables quick task assignment");
    println!("  • Reputation weighting ensures quality providers are preferred");
    println!("  • Price discovery helps participants make informed decisions");
    println!("  • SLA tracking enforces quality standards");
    println!("  • Privacy levels protect sensitive compute tasks");
    println!("\n🔒 Compliance:");
    println!("  • Each provider runs their own Claude Max locally");
    println!("  • No API keys are shared or transmitted");
    println!("  • Tokens reward contribution, not access");
    println!("  • All participation is voluntary");

    Ok(())
}