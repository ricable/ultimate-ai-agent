//! Economic simulation for testing market efficiency
//! 
//! This simulation tests various market conditions including:
//! - Supply/demand imbalances
//! - Price discovery convergence
//! - Reputation system effectiveness
//! - SLA enforcement impact

use claude_market::{
    Market, OrderType, ComputeTaskSpec, PrivacyLevel, SLASpec,
    Reputation, ReputationEvent,
};
use libp2p::PeerId;
use std::collections::HashMap;
use rand::{thread_rng, Rng, distributions::Uniform};
use tokio::time::{sleep, Duration};

/// Simulation parameters
struct SimulationParams {
    num_requesters: usize,
    num_providers: usize,
    num_rounds: usize,
    task_types: Vec<String>,
    price_range: (u64, u64),
    reputation_range: (f64, f64),
}

/// Market metrics for analysis
#[derive(Default)]
struct MarketMetrics {
    total_volume: u64,
    total_value: u64,
    successful_matches: u64,
    failed_matches: u64,
    avg_match_time: f64,
    price_volatility: f64,
    sla_violations: u64,
    reputation_changes: HashMap<PeerId, f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Synaptic Market Economic Simulation\n");
    
    let params = SimulationParams {
        num_requesters: 20,
        num_providers: 30,
        num_rounds: 100,
        task_types: vec![
            "code_generation".to_string(),
            "data_analysis".to_string(),
            "ml_training".to_string(),
            "document_processing".to_string(),
            "testing".to_string(),
        ],
        price_range: (10, 200),
        reputation_range: (0.0, 200.0),
    };

    println!("📊 Simulation Parameters:");
    println!("  Requesters: {}", params.num_requesters);
    println!("  Providers: {}", params.num_providers);
    println!("  Rounds: {}", params.num_rounds);
    println!("  Task Types: {}", params.task_types.len());
    println!("  Price Range: {} - {} tokens/unit", params.price_range.0, params.price_range.1);
    println!();

    // Initialize market
    let market = Market::new(":memory:").await?;
    market.init_schema().await?;
    let reputation = Reputation::new(":memory:").await?;
    reputation.init_schema().await?;

    // Create participants
    let mut requesters = Vec::new();
    let mut providers = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..params.num_requesters {
        requesters.push(PeerId::random());
    }

    for _ in 0..params.num_providers {
        let provider = PeerId::random();
        
        // Initialize with random reputation
        let initial_events = rng.gen_range(0..10);
        for _ in 0..initial_events {
            reputation.record_event(
                &provider,
                ReputationEvent::TradeCompleted,
                None,
                None
            ).await?;
        }
        
        providers.push(provider);
    }

    // Run simulation
    let mut metrics = MarketMetrics::default();
    let mut round_prices: HashMap<String, Vec<f64>> = HashMap::new();

    println!("🚀 Starting simulation...\n");

    for round in 1..=params.num_rounds {
        if round % 10 == 0 {
            println!("  Round {}/{}", round, params.num_rounds);
        }

        // Generate random requests
        let num_requests = rng.gen_range(1..5);
        for _ in 0..num_requests {
            let requester = &requesters[rng.gen_range(0..requesters.len())];
            let task_type = &params.task_types[rng.gen_range(0..params.task_types.len())];
            let price = rng.gen_range(params.price_range.0..params.price_range.1);
            let units = rng.gen_range(10..500);
            
            // Random requirements
            let min_reputation = if rng.gen_bool(0.5) {
                Some(rng.gen_range(50.0..150.0))
            } else {
                None
            };

            let task_spec = ComputeTaskSpec {
                task_type: task_type.clone(),
                compute_units: units,
                max_duration_secs: rng.gen_range(60..3600),
                required_capabilities: generate_random_capabilities(&mut rng),
                min_reputation,
                privacy_level: random_privacy_level(&mut rng),
                encrypted_payload: None,
            };

            let sla_spec = if rng.gen_bool(0.7) {
                Some(SLASpec {
                    uptime_requirement: rng.gen_range(95.0..99.9),
                    max_response_time: rng.gen_range(30..300),
                    violation_penalty: rng.gen_range(5..50),
                    quality_metrics: HashMap::new(),
                })
            } else {
                None
            };

            // Place request
            market.place_order(
                OrderType::RequestCompute,
                *requester,
                price,
                units,
                task_spec,
                sla_spec,
                None,
                None,
            ).await?;
        }

        // Generate random offers
        let num_offers = rng.gen_range(1..6);
        for _ in 0..num_offers {
            let provider = &providers[rng.gen_range(0..providers.len())];
            let task_type = &params.task_types[rng.gen_range(0..params.task_types.len())];
            let price = rng.gen_range(params.price_range.0..params.price_range.1);
            let units = rng.gen_range(50..1000);

            let task_spec = ComputeTaskSpec {
                task_type: task_type.clone(),
                compute_units: units,
                max_duration_secs: rng.gen_range(60..7200),
                required_capabilities: generate_random_capabilities(&mut rng),
                min_reputation: None,
                privacy_level: random_privacy_level(&mut rng),
                encrypted_payload: None,
            };

            // Place offer
            market.place_order(
                OrderType::OfferCompute,
                *provider,
                price,
                units,
                task_spec,
                None,
                None,
                None,
            ).await?;
        }

        // Process some assignments
        let assignments = market.get_assignments(None, 100).await?;
        for assignment in assignments.iter().filter(|a| a.status == claude_market::AssignmentStatus::Assigned) {
            metrics.successful_matches += 1;
            metrics.total_volume += assignment.compute_units;
            metrics.total_value += assignment.total_cost;

            // Record price for volatility calculation
            round_prices
                .entry(assignment.request_id.to_string())
                .or_insert_with(Vec::new)
                .push(assignment.price_per_unit as f64);

            // Simulate task execution
            if rng.gen_bool(0.9) { // 90% success rate
                market.start_task(&assignment.id, &assignment.provider).await?;
                
                // Random quality scores
                let quality_scores = HashMap::from([
                    ("quality".to_string(), rng.gen_range(0.7..1.0)),
                    ("speed".to_string(), rng.gen_range(0.6..1.0)),
                ]);

                market.complete_task(&assignment.id, &assignment.provider, quality_scores).await?;
                
                // Update reputation
                reputation.record_event(
                    &assignment.provider,
                    ReputationEvent::TradeCompleted,
                    Some(assignment.id),
                    None
                ).await?;
            } else {
                // Simulate failure
                reputation.record_event(
                    &assignment.provider,
                    ReputationEvent::TradeFailed,
                    Some(assignment.id),
                    None
                ).await?;
                metrics.sla_violations += 1;
            }
        }

        // Process expired auctions
        market.process_expired_auctions().await?;
        
        // Small delay between rounds
        sleep(Duration::from_millis(10)).await;
    }

    // Calculate final metrics
    println!("\n📊 Simulation Results:\n");

    println!("📈 Market Activity:");
    println!("  Total Volume: {} compute units", metrics.total_volume);
    println!("  Total Value: {} tokens", metrics.total_value);
    println!("  Successful Matches: {}", metrics.successful_matches);
    println!("  Match Rate: {:.1}%", 
        (metrics.successful_matches as f64 / (metrics.successful_matches + metrics.failed_matches) as f64) * 100.0);
    
    if metrics.successful_matches > 0 {
        println!("  Average Price: {:.2} tokens/unit", 
            metrics.total_value as f64 / metrics.total_volume as f64);
    }

    println!("\n💰 Price Discovery:");
    for task_type in &params.task_types {
        if let Some(price_data) = market.get_price_discovery(task_type).await? {
            println!("  {}:", task_type);
            println!("    24h Average: {:.2} tokens/unit", price_data.avg_price_24h);
            println!("    VWAP: {:.2} tokens/unit", price_data.vwap);
            println!("    Range: {} - {} tokens/unit", price_data.min_price, price_data.max_price);
            println!("    Volume: {} units", price_data.total_volume);
        }
    }

    println!("\n🏆 Reputation Impact:");
    let mut reputation_changes = Vec::new();
    for provider in &providers {
        let rep = reputation.get_reputation(provider).await?;
        if rep.total_trades > 0 {
            reputation_changes.push((provider, rep.score, rep.success_rate()));
        }
    }
    reputation_changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 5 Providers:");
    for (i, (provider, score, success_rate)) in reputation_changes.iter().take(5).enumerate() {
        println!("    {}. Score: {:.0}, Success Rate: {:.1}%", i + 1, score, success_rate);
    }

    println!("\n🔍 Market Efficiency Analysis:");
    
    // Calculate price convergence
    let price_convergence = calculate_price_convergence(&round_prices);
    println!("  Price Convergence: {:.2}%", price_convergence * 100.0);
    
    // Market liquidity
    let liquidity_ratio = metrics.successful_matches as f64 / params.num_rounds as f64;
    println!("  Liquidity Ratio: {:.2} matches/round", liquidity_ratio);
    
    // SLA effectiveness
    let sla_compliance = 1.0 - (metrics.sla_violations as f64 / metrics.successful_matches as f64);
    println!("  SLA Compliance: {:.1}%", sla_compliance * 100.0);

    println!("\n✅ Simulation Complete!");
    
    println!("\n🔑 Key Findings:");
    if price_convergence > 0.7 {
        println!("  ✓ Price discovery is working efficiently");
    } else {
        println!("  ⚠ Price discovery needs more liquidity");
    }
    
    if liquidity_ratio > 1.0 {
        println!("  ✓ Market has good liquidity");
    } else {
        println!("  ⚠ Market needs more participants");
    }
    
    if sla_compliance > 0.9 {
        println!("  ✓ SLA enforcement is effective");
    } else {
        println!("  ⚠ SLA penalties may need adjustment");
    }

    Ok(())
}

fn generate_random_capabilities(rng: &mut impl Rng) -> Vec<String> {
    let all_capabilities = vec![
        "rust", "python", "javascript", "go", "java",
        "machine_learning", "data_science", "web_scraping",
        "nlp", "computer_vision", "cuda", "distributed"
    ];
    
    let num_capabilities = rng.gen_range(1..4);
    let mut capabilities = Vec::new();
    
    for _ in 0..num_capabilities {
        let cap = all_capabilities[rng.gen_range(0..all_capabilities.len())];
        if !capabilities.contains(&cap.to_string()) {
            capabilities.push(cap.to_string());
        }
    }
    
    capabilities
}

fn random_privacy_level(rng: &mut impl Rng) -> PrivacyLevel {
    match rng.gen_range(0..3) {
        0 => PrivacyLevel::Public,
        1 => PrivacyLevel::Private,
        _ => PrivacyLevel::Confidential,
    }
}

fn calculate_price_convergence(round_prices: &HashMap<String, Vec<f64>>) -> f64 {
    if round_prices.is_empty() {
        return 0.0;
    }
    
    let mut convergence_scores = Vec::new();
    
    for prices in round_prices.values() {
        if prices.len() < 2 {
            continue;
        }
        
        // Calculate coefficient of variation (CV)
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance: f64 = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;
        
        // Convert CV to convergence score (lower CV = higher convergence)
        convergence_scores.push(1.0 / (1.0 + cv));
    }
    
    if convergence_scores.is_empty() {
        0.0
    } else {
        convergence_scores.iter().sum::<f64>() / convergence_scores.len() as f64
    }
}