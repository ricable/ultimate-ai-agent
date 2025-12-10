//! P2P Network Coordination Demo
//!
//! This example demonstrates P2P coordination in the Synaptic Mesh network:
//! - Setting up peer-to-peer connections
//! - Coordinated neural processing across peers
//! - Load balancing and expert distribution
//! - Fault tolerance and recovery

use kimi_fann_core::{
    MicroExpert, ExpertRouter, KimiRuntime, ProcessingConfig, 
    ExpertDomain, NetworkStats
};
use std::collections::HashMap;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Simulated P2P node in the Synaptic Mesh network
#[derive(Debug, Clone)]
pub struct P2PNode {
    pub id: String,
    pub experts: Vec<ExpertDomain>,
    pub capacity: f64,
    pub latency_ms: u64,
    pub reliability: f64,
}

impl P2PNode {
    pub fn new(id: String, experts: Vec<ExpertDomain>, capacity: f64) -> Self {
        Self {
            id,
            experts,
            capacity,
            latency_ms: fastrand::u64(10..100),
            reliability: 0.95 + fastrand::f64() * 0.05, // 95-100% reliability
        }
    }
}

/// P2P Network coordinator
pub struct P2PCoordinator {
    nodes: Vec<P2PNode>,
    query_history: Vec<(String, ExpertDomain, String)>,
    load_balancer: LoadBalancer,
}

impl P2PCoordinator {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            query_history: Vec::new(),
            load_balancer: LoadBalancer::new(),
        }
    }

    pub fn add_node(&mut self, node: P2PNode) {
        println!("🔗 Adding P2P node: {} with experts: {:?}", node.id, node.experts);
        self.nodes.push(node);
    }

    pub fn process_distributed_query(&mut self, query: &str) -> Result<String, Box<dyn std::error::Error>> {
        println!("\n🌐 Processing distributed query: {}", query);
        
        // Determine best expert domain for the query
        let best_domain = self.determine_best_domain(query);
        println!("  📍 Best domain identified: {:?}", best_domain);

        // Find available nodes with the required expert
        let available_nodes: Vec<&P2PNode> = self.nodes.iter()
            .filter(|node| node.experts.contains(&best_domain))
            .collect();

        if available_nodes.is_empty() {
            return Err("No nodes available for this expert domain".into());
        }

        // Select best node using load balancing
        let selected_node = self.load_balancer.select_best_node(&available_nodes, &best_domain)?;
        println!("  🎯 Selected node: {} (capacity: {:.2}, latency: {}ms)", 
                selected_node.id, selected_node.capacity, selected_node.latency_ms);

        // Simulate network processing
        let start_time = Instant::now();
        thread::sleep(Duration::from_millis(selected_node.latency_ms));
        
        // Process with expert
        let expert = MicroExpert::new(best_domain);
        let response = expert.process(query);
        
        let total_time = start_time.elapsed();
        
        // Record the query
        self.query_history.push((query.to_string(), best_domain, selected_node.id.clone()));

        let final_response = format!("{} [P2P: Processed by node '{}' in {:.2}ms]", 
                                   response, selected_node.id, total_time.as_millis());

        Ok(final_response)
    }

    pub fn get_network_stats(&self) -> NetworkStats {
        let mut expert_utilization = HashMap::new();
        
        // Calculate expert utilization from query history
        for (_, domain, _) in &self.query_history {
            *expert_utilization.entry(*domain).or_insert(0.0) += 1.0;
        }

        // Normalize utilization
        let total_queries = self.query_history.len() as f64;
        for utilization in expert_utilization.values_mut() {
            *utilization /= total_queries;
        }

        let average_latency = if !self.nodes.is_empty() {
            self.nodes.iter().map(|n| n.latency_ms as f64).sum::<f64>() / self.nodes.len() as f64
        } else {
            0.0
        };

        NetworkStats {
            active_peers: self.nodes.len(),
            total_queries: self.query_history.len() as u64,
            average_latency_ms: average_latency,
            expert_utilization,
            neural_accuracy: 0.92, // Simulated neural accuracy
        }
    }

    fn determine_best_domain(&self, query: &str) -> ExpertDomain {
        let query_lower = query.to_lowercase();
        
        // Simple domain detection based on keywords
        if query_lower.contains("code") || query_lower.contains("function") || query_lower.contains("program") {
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

/// Load balancer for optimal node selection
pub struct LoadBalancer {
    node_loads: HashMap<String, f64>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            node_loads: HashMap::new(),
        }
    }

    pub fn select_best_node(&mut self, nodes: &[&P2PNode], _domain: &ExpertDomain) -> Result<&P2PNode, Box<dyn std::error::Error>> {
        if nodes.is_empty() {
            return Err("No nodes available".into());
        }

        // Calculate score for each node (lower is better)
        let mut best_node = nodes[0];
        let mut best_score = f64::INFINITY;

        for &node in nodes {
            let current_load = self.node_loads.get(&node.id).copied().unwrap_or(0.0);
            
            // Score based on: current load + latency + (1 - reliability) + (1 - capacity)
            let score = current_load + 
                       node.latency_ms as f64 * 0.01 + 
                       (1.0 - node.reliability) * 100.0 + 
                       (1.0 - node.capacity) * 50.0;

            if score < best_score {
                best_score = score;
                best_node = node;
            }
        }

        // Update load for selected node
        *self.node_loads.entry(best_node.id.clone()).or_insert(0.0) += 1.0;

        // Decay all loads over time (simulate processing completion)
        for load in self.node_loads.values_mut() {
            *load *= 0.9;
        }

        Ok(best_node)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 P2P Network Coordination Demo");
    println!("================================");

    // Create a P2P network coordinator
    let mut coordinator = P2PCoordinator::new();

    // Add diverse nodes to the network
    setup_demo_network(&mut coordinator);

    // Demonstrate different coordination scenarios
    demonstrate_load_balancing(&mut coordinator)?;
    demonstrate_expert_distribution(&mut coordinator)?;
    demonstrate_fault_tolerance(&mut coordinator)?;
    demonstrate_network_monitoring(&coordinator);

    println!("\n✅ P2P coordination demonstration completed!");
    Ok(())
}

/// Set up a demo network with diverse nodes
fn setup_demo_network(coordinator: &mut P2PCoordinator) {
    println!("\n🔧 Setting up demo P2P network...");
    
    // High-performance coding specialists
    coordinator.add_node(P2PNode::new(
        "code-specialist-1".to_string(),
        vec![ExpertDomain::Coding, ExpertDomain::ToolUse],
        0.95,
    ));
    
    coordinator.add_node(P2PNode::new(
        "code-specialist-2".to_string(),
        vec![ExpertDomain::Coding, ExpertDomain::Mathematics],
        0.90,
    ));

    // Mathematical computation nodes
    coordinator.add_node(P2PNode::new(
        "math-compute-1".to_string(),
        vec![ExpertDomain::Mathematics, ExpertDomain::Reasoning],
        0.85,
    ));
    
    coordinator.add_node(P2PNode::new(
        "math-compute-2".to_string(),
        vec![ExpertDomain::Mathematics],
        0.98,
    ));

    // Language processing nodes
    coordinator.add_node(P2PNode::new(
        "language-proc-1".to_string(),
        vec![ExpertDomain::Language, ExpertDomain::Context],
        0.80,
    ));

    // General purpose nodes
    coordinator.add_node(P2PNode::new(
        "general-1".to_string(),
        vec![ExpertDomain::Reasoning, ExpertDomain::Context, ExpertDomain::ToolUse],
        0.75,
    ));
    
    coordinator.add_node(P2PNode::new(
        "general-2".to_string(),
        vec![ExpertDomain::Reasoning, ExpertDomain::Language],
        0.70,
    ));

    println!("  ✅ Network setup complete: {} nodes active", coordinator.nodes.len());
}

/// Demonstrate intelligent load balancing
fn demonstrate_load_balancing(coordinator: &mut P2PCoordinator) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚖️ Load Balancing Demonstration");
    println!("------------------------------");

    // Send multiple coding queries to test load distribution
    let coding_queries = [
        "Write a Python function to sort an array",
        "Implement a binary search algorithm in Rust",
        "Create a web API endpoint for user authentication",
        "Optimize this SQL query for better performance",
        "Debug this JavaScript function for memory leaks",
    ];

    println!("\n📊 Processing {} coding queries...", coding_queries.len());
    for (i, query) in coding_queries.iter().enumerate() {
        println!("\n🔄 Query {}: {}", i + 1, query);
        let response = coordinator.process_distributed_query(query)?;
        println!("  Response: {}", response);
    }

    // Show how load balancing works
    println!("\n📈 Load Balancing Results:");
    let stats = coordinator.get_network_stats();
    println!("  Total queries processed: {}", stats.total_queries);
    println!("  Average network latency: {:.2}ms", stats.average_latency_ms);
    
    Ok(())
}

/// Demonstrate expert distribution across domains
fn demonstrate_expert_distribution(coordinator: &mut P2PCoordinator) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Expert Distribution Demonstration");
    println!("-----------------------------------");

    // Diverse queries requiring different expert domains
    let mixed_queries = [
        ("Mathematics", "Calculate the integral of x^2 from 0 to 5"),
        ("Language", "Translate this sentence to French and explain the grammar"),
        ("Reasoning", "Analyze the logical fallacy in this argument"),
        ("ToolUse", "Execute a file system command to compress a directory"),
        ("Context", "Based on our previous discussion, what was the main point?"),
        ("Coding", "Implement a thread-safe singleton pattern"),
    ];

    println!("\n🔀 Processing diverse queries across domains...");
    for (domain_hint, query) in &mixed_queries {
        println!("\n📍 {} Query: {}", domain_hint, query);
        let response = coordinator.process_distributed_query(query)?;
        println!("  Response: {}", response);
    }

    // Show expert utilization
    println!("\n📊 Expert Domain Utilization:");
    let stats = coordinator.get_network_stats();
    for (domain, utilization) in &stats.expert_utilization {
        println!("  {:?}: {:.1}%", domain, utilization * 100.0);
    }

    Ok(())
}

/// Demonstrate fault tolerance and recovery
fn demonstrate_fault_tolerance(coordinator: &mut P2PCoordinator) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛡️ Fault Tolerance Demonstration");
    println!("--------------------------------");

    // Simulate node failures by reducing reliability
    println!("\n⚠️ Simulating network stress and node failures...");
    
    // Process queries while simulating increasing network stress
    let stress_queries = [
        "Simple calculation: 2 + 2",
        "Complex algorithm: implement quicksort with optimization",
        "Network intensive: translate and analyze multiple languages",
    ];

    for (i, query) in stress_queries.iter().enumerate() {
        println!("\n🔄 Stress Test {}: {}", i + 1, query);
        
        // Simulate increasing network stress
        for node in &mut coordinator.nodes {
            node.reliability *= 0.95; // Decrease reliability
            node.latency_ms += 10;     // Increase latency
        }
        
        let response = coordinator.process_distributed_query(query)?;
        println!("  Response: {}", response);
        
        let stats = coordinator.get_network_stats();
        println!("  Network latency: {:.2}ms", stats.average_latency_ms);
    }

    // Recovery demonstration
    println!("\n🔄 Network Recovery Simulation...");
    for node in &mut coordinator.nodes {
        node.reliability = (node.reliability + 0.1).min(1.0); // Improve reliability
        node.latency_ms = node.latency_ms.saturating_sub(15);  // Reduce latency
    }
    
    let recovery_response = coordinator.process_distributed_query("Test query after recovery")?;
    println!("  Recovery Response: {}", recovery_response);

    Ok(())
}

/// Demonstrate network monitoring and statistics
fn demonstrate_network_monitoring(coordinator: &P2PCoordinator) {
    println!("\n📊 Network Monitoring Dashboard");
    println!("------------------------------");

    let stats = coordinator.get_network_stats();
    
    println!("\n🌐 Network Overview:");
    println!("  Active Peers: {}", stats.active_peers);
    println!("  Total Queries: {}", stats.total_queries);
    println!("  Average Latency: {:.2}ms", stats.average_latency_ms);
    println!("  Neural Accuracy: {:.1}%", stats.neural_accuracy * 100.0);

    println!("\n🎯 Expert Domain Statistics:");
    for (domain, utilization) in &stats.expert_utilization {
        let bar_length = (utilization * 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  {:12} |{:<50}| {:.1}%", 
                format!("{:?}:", domain), bar, utilization * 100.0);
    }

    println!("\n🔗 Node Performance Summary:");
    for (i, node) in coordinator.nodes.iter().enumerate() {
        println!("  Node {}: {} (Cap: {:.1}%, Lat: {}ms, Rel: {:.1}%)",
                i + 1, node.id, node.capacity * 100.0, 
                node.latency_ms, node.reliability * 100.0);
    }

    println!("\n💡 Performance Insights:");
    let avg_capacity = coordinator.nodes.iter()
        .map(|n| n.capacity)
        .sum::<f64>() / coordinator.nodes.len() as f64;
    
    if avg_capacity > 0.8 {
        println!("  ✅ Network capacity is excellent ({:.1}%)", avg_capacity * 100.0);
    } else if avg_capacity > 0.6 {
        println!("  ⚠️  Network capacity is adequate ({:.1}%)", avg_capacity * 100.0);
    } else {
        println!("  ⚠️  Network capacity needs improvement ({:.1}%)", avg_capacity * 100.0);
    }

    if stats.average_latency_ms < 50.0 {
        println!("  ✅ Network latency is excellent ({:.1}ms)", stats.average_latency_ms);
    } else {
        println!("  ⚠️  Network latency could be improved ({:.1}ms)", stats.average_latency_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2p_node_creation() {
        let node = P2PNode::new(
            "test-node".to_string(),
            vec![ExpertDomain::Coding],
            0.8,
        );
        assert_eq!(node.id, "test-node");
        assert_eq!(node.capacity, 0.8);
        assert!(node.experts.contains(&ExpertDomain::Coding));
    }

    #[test]
    fn test_coordinator_basic_functionality() {
        let mut coordinator = P2PCoordinator::new();
        coordinator.add_node(P2PNode::new(
            "test".to_string(),
            vec![ExpertDomain::Reasoning],
            0.8,
        ));
        
        let result = coordinator.process_distributed_query("test query");
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_balancer() {
        let mut lb = LoadBalancer::new();
        let nodes = vec![
            P2PNode::new("node1".to_string(), vec![ExpertDomain::Coding], 0.9),
            P2PNode::new("node2".to_string(), vec![ExpertDomain::Coding], 0.8),
        ];
        let node_refs: Vec<&P2PNode> = nodes.iter().collect();
        
        let selected = lb.select_best_node(&node_refs, &ExpertDomain::Coding);
        assert!(selected.is_ok());
    }

    #[test]
    fn test_network_stats() {
        let mut coordinator = P2PCoordinator::new();
        coordinator.add_node(P2PNode::new(
            "test".to_string(),
            vec![ExpertDomain::Mathematics],
            0.8,
        ));
        
        let _ = coordinator.process_distributed_query("calculate 2+2");
        let stats = coordinator.get_network_stats();
        
        assert_eq!(stats.active_peers, 1);
        assert_eq!(stats.total_queries, 1);
    }
}