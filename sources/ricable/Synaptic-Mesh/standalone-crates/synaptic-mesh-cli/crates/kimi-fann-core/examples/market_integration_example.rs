//! Market Integration Example
//!
//! This example demonstrates how to integrate the Kimi-FANN Core neural
//! processing with the Claude Market for distributed compute trading:
//! - Trading neural compute capacity
//! - Load balancing across market participants
//! - Economic incentive structures
//! - SLA enforcement and reputation

use kimi_fann_core::{
    MicroExpert, ExpertRouter, KimiRuntime, ProcessingConfig, 
    ExpertDomain, NetworkStats
};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Mock market integration types (would normally come from claude_market crate)
#[derive(Debug, Clone)]
pub struct MarketOrder {
    pub id: String,
    pub provider_id: String,
    pub compute_units: u64,
    pub price_per_unit: u64,
    pub expert_domain: ExpertDomain,
    pub sla_requirements: SLARequirements,
}

#[derive(Debug, Clone)]
pub struct SLARequirements {
    pub max_latency_ms: u64,
    pub min_accuracy: f64,
    pub uptime_requirement: f64,
}

#[derive(Debug, Clone)]
pub struct ComputeProvider {
    pub id: String,
    pub reputation_score: f64,
    pub available_experts: Vec<ExpertDomain>,
    pub pricing: HashMap<ExpertDomain, u64>,
    pub current_load: f64,
    pub total_capacity: u64,
}

impl ComputeProvider {
    pub fn new(id: String, experts: Vec<ExpertDomain>) -> Self {
        let mut pricing = HashMap::new();
        for expert in &experts {
            let base_price = match expert {
                ExpertDomain::Mathematics => 10,
                ExpertDomain::Coding => 15,
                ExpertDomain::Language => 12,
                ExpertDomain::Reasoning => 8,
                ExpertDomain::ToolUse => 20,
                ExpertDomain::Context => 6,
            };
            pricing.insert(*expert, base_price);
        }

        Self {
            id,
            reputation_score: 0.85 + fastrand::f64() * 0.1,
            available_experts: experts,
            pricing,
            current_load: 0.0,
            total_capacity: 1000,
        }
    }
}

/// Market-integrated neural processing system
pub struct MarketIntegratedProcessor {
    providers: Vec<ComputeProvider>,
    active_orders: Vec<MarketOrder>,
    processing_history: Vec<ProcessingResult>,
    total_revenue: u64,
    total_compute_units: u64,
}

#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub order_id: String,
    pub provider_id: String,
    pub domain: ExpertDomain,
    pub query: String,
    pub response: String,
    pub processing_time_ms: u64,
    pub cost: u64,
    pub accuracy_score: f64,
    pub sla_met: bool,
}

impl MarketIntegratedProcessor {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            active_orders: Vec::new(),
            processing_history: Vec::new(),
            total_revenue: 0,
            total_compute_units: 0,
        }
    }

    pub fn add_provider(&mut self, provider: ComputeProvider) {
        println!("📈 Adding compute provider: {} (reputation: {:.2})", 
                provider.id, provider.reputation_score);
        self.providers.push(provider);
    }

    pub fn create_market_order(&mut self, domain: ExpertDomain, compute_units: u64) -> Result<MarketOrder, Box<dyn std::error::Error>> {
        // Find available providers for this domain
        let available_providers: Vec<&ComputeProvider> = self.providers.iter()
            .filter(|p| p.available_experts.contains(&domain))
            .filter(|p| p.current_load < 0.8) // Only providers with <80% load
            .collect();

        if available_providers.is_empty() {
            return Err("No available providers for this domain".into());
        }

        // Select best provider based on reputation and pricing
        let best_provider = available_providers.iter()
            .min_by(|a, b| {
                let a_score = a.pricing.get(&domain).unwrap_or(&100) * 100 - (a.reputation_score * 100.0) as u64;
                let b_score = b.pricing.get(&domain).unwrap_or(&100) * 100 - (b.reputation_score * 100.0) as u64;
                a_score.cmp(&b_score)
            })
            .unwrap();

        let order = MarketOrder {
            id: format!("order_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()),
            provider_id: best_provider.id.clone(),
            compute_units,
            price_per_unit: *best_provider.pricing.get(&domain).unwrap_or(&50),
            expert_domain: domain,
            sla_requirements: SLARequirements {
                max_latency_ms: 1000,
                min_accuracy: 0.85,
                uptime_requirement: 0.95,
            },
        };

        println!("💳 Created market order: {} for {} compute units at {} ruv/unit", 
                order.id, order.compute_units, order.price_per_unit);

        self.active_orders.push(order.clone());
        Ok(order)
    }

    pub async fn process_with_market(&mut self, query: &str) -> Result<ProcessingResult, Box<dyn std::error::Error>> {
        // Determine the best domain for this query
        let domain = self.determine_domain(query);
        println!("\n🎯 Processing query with {} domain", format!("{:?}", domain));

        // Create market order for compute
        let order = self.create_market_order(domain, 1)?;

        // Find the provider
        let provider = self.providers.iter()
            .find(|p| p.id == order.provider_id)
            .ok_or("Provider not found")?;

        println!("🏭 Selected provider: {} (rep: {:.2}, price: {} ruv/unit)", 
                provider.id, provider.reputation_score, order.price_per_unit);

        // Process the query
        let start_time = Instant::now();
        let expert = MicroExpert::new(domain);
        let response = expert.process(query);
        let processing_time = start_time.elapsed().as_millis() as u64;

        // Calculate costs and accuracy
        let cost = order.compute_units * order.price_per_unit;
        let accuracy_score = 0.85 + (provider.reputation_score - 0.85) * 0.5 + fastrand::f64() * 0.1;
        
        // Check SLA compliance
        let sla_met = processing_time <= order.sla_requirements.max_latency_ms &&
                      accuracy_score >= order.sla_requirements.min_accuracy;

        let result = ProcessingResult {
            order_id: order.id.clone(),
            provider_id: provider.id.clone(),
            domain,
            query: query.to_string(),
            response: response.clone(),
            processing_time_ms: processing_time,
            cost,
            accuracy_score,
            sla_met,
        };

        // Update statistics
        self.total_revenue += cost;
        self.total_compute_units += order.compute_units;
        self.processing_history.push(result.clone());

        // Update provider load (simulate)
        if let Some(provider) = self.providers.iter_mut().find(|p| p.id == order.provider_id) {
            provider.current_load = (provider.current_load + 0.1).min(1.0);
        }

        // Remove completed order
        self.active_orders.retain(|o| o.id != order.id);

        println!("✅ Processing completed: {}ms, cost: {} ruv, SLA: {}", 
                processing_time, cost, if sla_met { "✅" } else { "❌" });

        Ok(result)
    }

    pub fn get_market_stats(&self) -> MarketStats {
        let total_providers = self.providers.len();
        let avg_reputation = if total_providers > 0 {
            self.providers.iter().map(|p| p.reputation_score).sum::<f64>() / total_providers as f64
        } else {
            0.0
        };

        let avg_processing_time = if !self.processing_history.is_empty() {
            self.processing_history.iter().map(|r| r.processing_time_ms).sum::<u64>() as f64 
                / self.processing_history.len() as f64
        } else {
            0.0
        };

        let sla_compliance_rate = if !self.processing_history.is_empty() {
            self.processing_history.iter().filter(|r| r.sla_met).count() as f64 
                / self.processing_history.len() as f64
        } else {
            0.0
        };

        // Calculate domain pricing
        let mut domain_pricing = HashMap::new();
        for provider in &self.providers {
            for (domain, price) in &provider.pricing {
                let current_avg = domain_pricing.get(domain).unwrap_or(&0u64);
                let count = self.providers.iter()
                    .filter(|p| p.pricing.contains_key(domain))
                    .count() as u64;
                domain_pricing.insert(*domain, (*current_avg * (count - 1) + price) / count);
            }
        }

        MarketStats {
            total_providers,
            active_orders: self.active_orders.len(),
            total_revenue: self.total_revenue,
            total_compute_units: self.total_compute_units,
            avg_reputation,
            avg_processing_time_ms: avg_processing_time,
            sla_compliance_rate,
            domain_pricing,
        }
    }

    fn determine_domain(&self, query: &str) -> ExpertDomain {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("code") || query_lower.contains("program") || query_lower.contains("function") {
            ExpertDomain::Coding
        } else if query_lower.contains("math") || query_lower.contains("calculate") || query_lower.contains("equation") {
            ExpertDomain::Mathematics
        } else if query_lower.contains("translate") || query_lower.contains("language") || query_lower.contains("grammar") {
            ExpertDomain::Language
        } else if query_lower.contains("tool") || query_lower.contains("execute") || query_lower.contains("run") {
            ExpertDomain::ToolUse
        } else if query_lower.contains("previous") || query_lower.contains("context") || query_lower.contains("remember") {
            ExpertDomain::Context
        } else {
            ExpertDomain::Reasoning
        }
    }
}

#[derive(Debug)]
pub struct MarketStats {
    pub total_providers: usize,
    pub active_orders: usize,
    pub total_revenue: u64,
    pub total_compute_units: u64,
    pub avg_reputation: f64,
    pub avg_processing_time_ms: f64,
    pub sla_compliance_rate: f64,
    pub domain_pricing: HashMap<ExpertDomain, u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("💰 Kimi-FANN Core Market Integration Example");
    println!("============================================");

    // Create market-integrated processor
    let mut market_processor = MarketIntegratedProcessor::new();

    // Set up the market ecosystem
    setup_market_ecosystem(&mut market_processor);

    // Demonstrate various market scenarios
    demonstrate_basic_market_processing(&mut market_processor).await?;
    demonstrate_load_balancing(&mut market_processor).await?;
    demonstrate_sla_enforcement(&mut market_processor).await?;
    demonstrate_economic_incentives(&mut market_processor).await?;
    
    // Show comprehensive market analysis
    show_market_analysis(&market_processor);

    println!("\n✅ Market integration demonstration completed!");
    Ok(())
}

/// Set up a diverse market ecosystem
fn setup_market_ecosystem(processor: &mut MarketIntegratedProcessor) {
    println!("\n🏗️ Setting up market ecosystem...");

    // High-end specialist providers
    processor.add_provider(ComputeProvider::new(
        "premium-math-co".to_string(),
        vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning],
    ));

    processor.add_provider(ComputeProvider::new(
        "code-masters-inc".to_string(),
        vec![ExpertDomain::Coding, ExpertDomain::ToolUse],
    ));

    processor.add_provider(ComputeProvider::new(
        "linguistic-ai-labs".to_string(),
        vec![ExpertDomain::Language, ExpertDomain::Context],
    ));

    // Mid-tier general providers
    processor.add_provider(ComputeProvider::new(
        "general-compute-1".to_string(),
        vec![ExpertDomain::Reasoning, ExpertDomain::Mathematics, ExpertDomain::Coding],
    ));

    processor.add_provider(ComputeProvider::new(
        "general-compute-2".to_string(),
        vec![ExpertDomain::Language, ExpertDomain::Context, ExpertDomain::ToolUse],
    ));

    // Budget providers
    processor.add_provider(ComputeProvider::new(
        "budget-ai-solutions".to_string(),
        vec![ExpertDomain::Reasoning, ExpertDomain::Context],
    ));

    // Set different pricing for budget provider
    if let Some(budget_provider) = processor.providers.iter_mut().find(|p| p.id == "budget-ai-solutions") {
        budget_provider.reputation_score = 0.75; // Lower reputation
        for (_, price) in budget_provider.pricing.iter_mut() {
            *price = *price / 2; // Half price
        }
    }

    println!("  ✅ Market ecosystem ready: {} providers active", processor.providers.len());
}

/// Demonstrate basic market processing
async fn demonstrate_basic_market_processing(processor: &mut MarketIntegratedProcessor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💼 Basic Market Processing Demonstration");
    println!("---------------------------------------");

    let test_queries = [
        "Calculate the derivative of x^3 + 2x^2 - 5x + 3",
        "Write a Python function to implement quicksort",
        "Translate 'Hello, how are you?' to Spanish and French",
        "Analyze the logical structure of this philosophical argument",
        "Execute a system command to check disk usage",
        "Based on our previous discussion about AI, what are the implications?",
    ];

    for (i, query) in test_queries.iter().enumerate() {
        println!("\n🔄 Processing Query {}: {}", i + 1, query);
        
        match processor.process_with_market(query).await {
            Ok(result) => {
                println!("  📊 Result Summary:");
                println!("    Provider: {}", result.provider_id);
                println!("    Domain: {:?}", result.domain);
                println!("    Cost: {} ruv", result.cost);
                println!("    Time: {}ms", result.processing_time_ms);
                println!("    Accuracy: {:.2}%", result.accuracy_score * 100.0);
                println!("    SLA Met: {}", if result.sla_met { "✅" } else { "❌" });
                println!("    Response: {}...", &result.response[..result.response.len().min(100)]);
            }
            Err(e) => println!("  ❌ Processing failed: {}", e),
        }
    }

    Ok(())
}

/// Demonstrate load balancing across providers
async fn demonstrate_load_balancing(processor: &mut MarketIntegratedProcessor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚖️ Load Balancing Demonstration");
    println!("------------------------------");

    // Create many concurrent orders to test load balancing
    let coding_queries = [
        "Implement a hash table in Rust",
        "Write a web scraper in Python",
        "Create a REST API endpoint",
        "Debug this JavaScript function",
        "Optimize this SQL query",
        "Write unit tests for this code",
        "Refactor this function for better performance",
    ];

    println!("\n📊 Processing {} coding queries to test load distribution...", coding_queries.len());

    let mut provider_usage = HashMap::new();

    for query in &coding_queries {
        match processor.process_with_market(query).await {
            Ok(result) => {
                *provider_usage.entry(result.provider_id.clone()).or_insert(0) += 1;
                println!("  ✅ Query processed by: {}", result.provider_id);
            }
            Err(e) => println!("  ❌ Failed: {}", e),
        }
    }

    println!("\n📈 Load Distribution Results:");
    for (provider, count) in &provider_usage {
        let percentage = (*count as f64 / coding_queries.len() as f64) * 100.0;
        println!("  {}: {} queries ({:.1}%)", provider, count, percentage);
    }

    Ok(())
}

/// Demonstrate SLA enforcement
async fn demonstrate_sla_enforcement(processor: &mut MarketIntegratedProcessor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛡️ SLA Enforcement Demonstration");
    println!("-------------------------------");

    println!("\n📋 Testing SLA compliance across different query complexities...");

    let sla_test_queries = [
        ("Simple", "What is 2 + 2?"),
        ("Medium", "Explain the concept of object-oriented programming"),
        ("Complex", "Analyze the computational complexity of merge sort and provide a detailed mathematical proof with Big O notation analysis"),
        ("Very Complex", "Design a distributed microservices architecture for a high-throughput e-commerce platform with fault tolerance, auto-scaling, and data consistency guarantees"),
    ];

    let mut sla_results = Vec::new();

    for (complexity, query) in &sla_test_queries {
        println!("\n🔍 Testing {} complexity query...", complexity);
        
        match processor.process_with_market(query).await {
            Ok(result) => {
                sla_results.push((complexity, result.sla_met, result.processing_time_ms, result.accuracy_score));
                println!("  Time: {}ms (SLA: {}ms)", result.processing_time_ms, 1000);
                println!("  Accuracy: {:.2}% (SLA: {:.0}%)", result.accuracy_score * 100.0, 85.0);
                println!("  SLA Status: {}", if result.sla_met { "✅ COMPLIANT" } else { "❌ VIOLATION" });
            }
            Err(e) => println!("  ❌ Failed: {}", e),
        }
    }

    println!("\n📊 SLA Compliance Summary:");
    let total_tests = sla_results.len();
    let compliant = sla_results.iter().filter(|(_, sla_met, _, _)| *sla_met).count();
    let compliance_rate = (compliant as f64 / total_tests as f64) * 100.0;
    
    println!("  Total Tests: {}", total_tests);
    println!("  Compliant: {}", compliant);
    println!("  Compliance Rate: {:.1}%", compliance_rate);

    Ok(())
}

/// Demonstrate economic incentives and pricing
async fn demonstrate_economic_incentives(processor: &mut MarketIntegratedProcessor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💰 Economic Incentives Demonstration");
    println!("-----------------------------------");

    println!("\n💸 Comparing costs across different providers and domains...");

    let economic_test_queries = [
        (ExpertDomain::Mathematics, "Calculate complex integrals"),
        (ExpertDomain::Coding, "Implement advanced algorithms"),
        (ExpertDomain::Language, "Professional translation services"),
        (ExpertDomain::ToolUse, "System automation tasks"),
        (ExpertDomain::Reasoning, "Complex logical analysis"),
        (ExpertDomain::Context, "Conversational AI services"),
    ];

    let mut cost_analysis = HashMap::new();

    for (domain, query) in &economic_test_queries {
        println!("\n💼 Testing {:?} domain pricing...", domain);
        
        match processor.process_with_market(query).await {
            Ok(result) => {
                cost_analysis.insert(*domain, (result.cost, result.provider_id.clone()));
                println!("  Provider: {}", result.provider_id);
                println!("  Cost: {} ruv", result.cost);
                println!("  Value Score: {:.2}", result.accuracy_score * 100.0 / result.cost as f64);
            }
            Err(e) => println!("  ❌ Failed: {}", e),
        }
    }

    println!("\n📊 Domain Pricing Analysis:");
    let mut sorted_costs: Vec<_> = cost_analysis.iter().collect();
    sorted_costs.sort_by_key(|(_, (cost, _))| *cost);

    for (domain, (cost, provider)) in sorted_costs {
        println!("  {:12} - {} ruv ({})", format!("{:?}:", domain), cost, provider);
    }

    // Show market evolution over time
    println!("\n📈 Market Evolution:");
    let stats = processor.get_market_stats();
    println!("  Total Revenue Generated: {} ruv", stats.total_revenue);
    println!("  Total Compute Units Sold: {}", stats.total_compute_units);
    println!("  Average Provider Reputation: {:.2}", stats.avg_reputation);

    Ok(())
}

/// Show comprehensive market analysis
fn show_market_analysis(processor: &MarketIntegratedProcessor) {
    println!("\n📊 Comprehensive Market Analysis");
    println!("===============================");

    let stats = processor.get_market_stats();

    println!("\n🏢 Market Overview:");
    println!("  Active Providers: {}", stats.total_providers);
    println!("  Pending Orders: {}", stats.active_orders);
    println!("  Total Revenue: {} ruv", stats.total_revenue);
    println!("  Compute Units Traded: {}", stats.total_compute_units);

    println!("\n⚡ Performance Metrics:");
    println!("  Average Processing Time: {:.1}ms", stats.avg_processing_time_ms);
    println!("  SLA Compliance Rate: {:.1}%", stats.sla_compliance_rate * 100.0);
    println!("  Average Provider Reputation: {:.2}/1.0", stats.avg_reputation);

    println!("\n💰 Domain Pricing (Average):");
    for (domain, price) in &stats.domain_pricing {
        println!("  {:12} - {} ruv/unit", format!("{:?}:", domain), price);
    }

    println!("\n📈 Market Health Indicators:");
    
    // Market liquidity
    let liquidity_score = if stats.total_providers > 5 { "High" } 
                         else if stats.total_providers > 2 { "Medium" } 
                         else { "Low" };
    println!("  Market Liquidity: {} ({} providers)", liquidity_score, stats.total_providers);
    
    // Price competitiveness
    let price_variance = stats.domain_pricing.values()
        .map(|&p| p as f64)
        .collect::<Vec<_>>();
    let avg_price = price_variance.iter().sum::<f64>() / price_variance.len() as f64;
    let competitiveness = if avg_price < 15.0 { "High" } 
                         else if avg_price < 25.0 { "Medium" } 
                         else { "Low" };
    println!("  Price Competitiveness: {} (avg: {:.1} ruv)", competitiveness, avg_price);
    
    // Service quality
    let quality_rating = if stats.sla_compliance_rate > 0.9 { "Excellent" }
                        else if stats.sla_compliance_rate > 0.8 { "Good" }
                        else if stats.sla_compliance_rate > 0.7 { "Fair" }
                        else { "Poor" };
    println!("  Service Quality: {} ({:.1}% SLA compliance)", quality_rating, stats.sla_compliance_rate * 100.0);

    println!("\n🔮 Market Insights:");
    println!("  • High demand for coding and mathematics expertise");
    println!("  • Premium providers maintain higher SLA compliance");
    println!("  • Load balancing effectively distributes work");
    println!("  • Economic incentives encourage quality service delivery");
    
    if stats.sla_compliance_rate < 0.85 {
        println!("  ⚠️  Recommendation: Consider SLA penalty mechanisms");
    }
    
    if stats.total_providers < 5 {
        println!("  ⚠️  Recommendation: Incentivize more provider participation");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_provider_creation() {
        let provider = ComputeProvider::new(
            "test-provider".to_string(),
            vec![ExpertDomain::Coding, ExpertDomain::Mathematics],
        );
        
        assert_eq!(provider.id, "test-provider");
        assert!(provider.available_experts.contains(&ExpertDomain::Coding));
        assert!(provider.pricing.contains_key(&ExpertDomain::Coding));
        assert!(provider.reputation_score > 0.8);
    }

    #[test]
    fn test_market_processor_setup() {
        let mut processor = MarketIntegratedProcessor::new();
        processor.add_provider(ComputeProvider::new(
            "test".to_string(),
            vec![ExpertDomain::Reasoning],
        ));
        
        assert_eq!(processor.providers.len(), 1);
    }

    #[test]
    fn test_domain_determination() {
        let processor = MarketIntegratedProcessor::new();
        
        assert_eq!(processor.determine_domain("write code"), ExpertDomain::Coding);
        assert_eq!(processor.determine_domain("calculate math"), ExpertDomain::Mathematics);
        assert_eq!(processor.determine_domain("translate text"), ExpertDomain::Language);
    }

    #[tokio::test]
    async fn test_market_order_creation() {
        let mut processor = MarketIntegratedProcessor::new();
        processor.add_provider(ComputeProvider::new(
            "test".to_string(),
            vec![ExpertDomain::Coding],
        ));
        
        let order = processor.create_market_order(ExpertDomain::Coding, 10);
        assert!(order.is_ok());
        
        let order = order.unwrap();
        assert_eq!(order.compute_units, 10);
        assert_eq!(order.expert_domain, ExpertDomain::Coding);
    }
}