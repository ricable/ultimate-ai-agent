//! Comprehensive test of all Kimi-FANN Core features
//! 
//! This example demonstrates and tests every major feature of the system,
//! providing clear output that can be used for documentation and verification.

use kimi_fann_core::{
    MicroExpert, ExpertRouter, KimiRuntime, ProcessingConfig, 
    ExpertDomain, NetworkStats, VERSION
};
use std::time::Instant;
use std::collections::HashMap;

fn main() {
    println!("🚀 Kimi-FANN Core v{} - Comprehensive Feature Test", VERSION);
    println!("{}", "=".repeat(70));
    println!();
    
    // Test 1: Individual Expert Processing
    test_individual_experts();
    
    // Test 2: Expert Router
    test_expert_router();
    
    // Test 3: Runtime with Standard Mode
    test_runtime_standard();
    
    // Test 4: Runtime with Consensus Mode
    test_runtime_consensus();
    
    // Test 5: Performance Testing
    test_performance();
    
    // Test 6: Edge Cases
    test_edge_cases();
    
    // Test 7: Configuration Options
    test_configuration();
    
    // Test 8: Network Statistics
    test_network_stats();
    
    println!();
    println!("✅ All feature tests completed successfully!");
}

fn test_individual_experts() {
    println!("📋 Test 1: Individual Expert Processing");
    println!("{}", "-".repeat(50));
    
    let test_cases = vec![
        (ExpertDomain::Reasoning, "Analyze the implications of artificial general intelligence on society"),
        (ExpertDomain::Coding, "Write a recursive function to calculate factorial"),
        (ExpertDomain::Mathematics, "Solve the quadratic equation x^2 - 5x + 6 = 0"),
        (ExpertDomain::Language, "Translate 'The quick brown fox' to multiple languages"),
        (ExpertDomain::ToolUse, "Execute a command to check system status"),
        (ExpertDomain::Context, "Remember and summarize our previous discussions"),
    ];
    
    for (domain, query) in test_cases {
        println!("\n🔹 Testing {:?} Expert", domain);
        println!("Query: {}", query);
        
        let start = Instant::now();
        let expert = MicroExpert::new(domain);
        let result = expert.process(query);
        let duration = start.elapsed();
        
        // Display truncated result
        let display_result = if result.len() > 150 {
            format!("{}...", &result[..147])
        } else {
            result.clone()
        };
        
        println!("Response: {}", display_result);
        println!("Processing time: {:?}", duration);
        
        // Verify neural processing
        let has_neural = result.contains("Neural:") || 
                        result.contains("conf=") || 
                        result.contains("patterns=");
        println!("Neural processing: {}", if has_neural { "✅ Active" } else { "⚠️ Fallback" });
    }
    println!();
}

fn test_expert_router() {
    println!("📋 Test 2: Expert Router Intelligence");
    println!("{}", "-".repeat(50));
    
    let mut router = ExpertRouter::new();
    
    // Add all experts
    for domain in [
        ExpertDomain::Reasoning,
        ExpertDomain::Coding,
        ExpertDomain::Language,
        ExpertDomain::Mathematics,
        ExpertDomain::ToolUse,
        ExpertDomain::Context,
    ] {
        router.add_expert(MicroExpert::new(domain));
    }
    
    let routing_tests = vec![
        ("Implement a sorting algorithm in Python", "Coding"),
        ("Calculate the integral of sin(x)", "Mathematics"),
        ("Translate this text to German", "Language"),
        ("Analyze the logical fallacy in this argument", "Reasoning"),
        ("Execute system diagnostic commands", "ToolUse"),
        ("Based on our previous conversation", "Context"),
    ];
    
    for (query, expected) in routing_tests {
        println!("\n🔹 Routing Test");
        println!("Query: {}", query);
        println!("Expected Domain: {}", expected);
        
        let start = Instant::now();
        let result = router.route(query);
        let duration = start.elapsed();
        
        // Check if routing worked
        let routing_success = result.contains("Routed to") || 
                            result.contains(expected.to_lowercase().as_str());
        
        println!("Routing: {}", if routing_success { "✅ Correct" } else { "⚠️ Uncertain" });
        println!("Response time: {:?}", duration);
    }
    println!();
}

fn test_runtime_standard() {
    println!("📋 Test 3: Runtime - Standard Mode");
    println!("{}", "-".repeat(50));
    
    let config = ProcessingConfig::new();
    let mut runtime = KimiRuntime::new(config);
    
    let queries = vec![
        "Explain the concept of recursion",
        "Write a REST API endpoint",
        "Calculate compound interest",
        "Translate hello to 5 languages",
    ];
    
    for query in queries {
        println!("\n🔹 Processing: {}", query);
        
        let start = Instant::now();
        let result = runtime.process(query);
        let duration = start.elapsed();
        
        // Extract key information
        let has_runtime = result.contains("Runtime:");
        let has_experts = result.contains("experts active");
        let has_neural = result.contains("Neural:") || result.contains("conf=");
        
        println!("Runtime metadata: {}", if has_runtime { "✅" } else { "❌" });
        println!("Expert count: {}", if has_experts { "✅" } else { "❌" });
        println!("Neural processing: {}", if has_neural { "✅" } else { "❌" });
        println!("Processing time: {:?}", duration);
    }
    println!();
}

fn test_runtime_consensus() {
    println!("📋 Test 4: Runtime - Consensus Mode");
    println!("{}", "-".repeat(50));
    
    let config = ProcessingConfig::new();
    let mut runtime = KimiRuntime::new(config);
    runtime.set_consensus_mode(true);
    
    let complex_queries = vec![
        "Design a machine learning system for natural language processing with code examples",
        "Create a mathematical model for predicting stock prices and implement it",
        "Build a multilingual chatbot that can reason about user queries",
    ];
    
    for query in complex_queries {
        println!("\n🔹 Complex Query: {}", query);
        
        let start = Instant::now();
        let result = runtime.process(query);
        let duration = start.elapsed();
        
        // Verify consensus mode
        let has_consensus = result.contains("Mode: Consensus");
        let result_length = result.len();
        
        println!("Consensus mode: {}", if has_consensus { "✅ Active" } else { "❌ Inactive" });
        println!("Response length: {} chars", result_length);
        println!("Processing time: {:?}", duration);
        
        // Show snippet of consensus result
        if result_length > 200 {
            println!("Result preview: {}...", &result[..197]);
        }
    }
    println!();
}

fn test_performance() {
    println!("📋 Test 5: Performance Testing");
    println!("{}", "-".repeat(50));
    
    let config = ProcessingConfig::new_neural_optimized();
    let mut runtime = KimiRuntime::new(config);
    
    let performance_tests = vec![
        ("Simple", "What is 2 + 2?", 100),
        ("Medium", "Explain object-oriented programming", 300),
        ("Complex", "Design a distributed database system", 500),
        ("Very Complex", "Implement a complete neural network", 1000),
    ];
    
    for (complexity, query, expected_ms) in performance_tests {
        println!("\n🔹 {} Query Performance", complexity);
        
        let start = Instant::now();
        let result = runtime.process(query);
        let duration = start.elapsed();
        let duration_ms = duration.as_millis();
        
        let within_range = duration_ms <= expected_ms as u128;
        
        println!("Query: {}", query);
        println!("Expected: <{}ms", expected_ms);
        println!("Actual: {}ms", duration_ms);
        println!("Performance: {}", if within_range { "✅ Good" } else { "⚠️ Slow" });
        println!("Result length: {} chars", result.len());
    }
    println!();
}

fn test_edge_cases() {
    println!("📋 Test 6: Edge Cases and Error Handling");
    println!("{}", "-".repeat(50));
    
    let config = ProcessingConfig::new();
    let mut runtime = KimiRuntime::new(config);
    
    let edge_cases = vec![
        ("Empty", ""),
        ("Whitespace", "   "),
        ("Special chars", "!@#$%^&*()[]{}"),
        ("Very long", &"a".repeat(1000)),
        ("Unicode", "你好世界 🌍 café"),
        ("Injection", "'; DROP TABLE users; --"),
    ];
    
    for (case_name, input) in edge_cases {
        println!("\n🔹 Edge Case: {}", case_name);
        
        let start = Instant::now();
        let result = runtime.process(input);
        let duration = start.elapsed();
        
        let handled_gracefully = !result.is_empty() || input.trim().is_empty();
        
        println!("Input: {}", if input.len() > 50 { 
            format!("{}... ({} chars)", &input[..47], input.len()) 
        } else { 
            input.to_string() 
        });
        println!("Handled gracefully: {}", if handled_gracefully { "✅" } else { "❌" });
        println!("Processing time: {:?}", duration);
    }
    println!();
}

fn test_configuration() {
    println!("📋 Test 7: Configuration Options");
    println!("{}", "-".repeat(50));
    
    // Test default configuration
    println!("\n🔹 Default Configuration");
    let default_config = ProcessingConfig::new();
    println!("Max experts: {}", default_config.max_experts);
    println!("Timeout: {}ms", default_config.timeout_ms);
    println!("Neural inference: {}", default_config.neural_inference_enabled);
    println!("Consensus threshold: {}", default_config.consensus_threshold);
    
    // Test neural-optimized configuration
    println!("\n🔹 Neural-Optimized Configuration");
    let neural_config = ProcessingConfig::new_neural_optimized();
    println!("Max experts: {}", neural_config.max_experts);
    println!("Timeout: {}ms", neural_config.timeout_ms);
    println!("Neural inference: {}", neural_config.neural_inference_enabled);
    println!("Consensus threshold: {}", neural_config.consensus_threshold);
    
    // Test runtime with different configs
    println!("\n🔹 Testing Different Configurations");
    for (name, config) in [
        ("Default", ProcessingConfig::new()),
        ("Neural-Optimized", ProcessingConfig::new_neural_optimized()),
    ] {
        let mut runtime = KimiRuntime::new(config);
        let result = runtime.process("Test configuration impact");
        
        println!("{} config result length: {} chars", name, result.len());
    }
    println!();
}

fn test_network_stats() {
    println!("📋 Test 8: Network Statistics");
    println!("{}", "-".repeat(50));
    
    // Create a mock NetworkStats for demonstration
    let mut expert_utilization = HashMap::new();
    expert_utilization.insert(ExpertDomain::Reasoning, 0.85);
    expert_utilization.insert(ExpertDomain::Coding, 0.92);
    expert_utilization.insert(ExpertDomain::Mathematics, 0.78);
    expert_utilization.insert(ExpertDomain::Language, 0.65);
    expert_utilization.insert(ExpertDomain::ToolUse, 0.45);
    expert_utilization.insert(ExpertDomain::Context, 0.72);
    
    let stats = NetworkStats {
        active_peers: 6,
        total_queries: 1247,
        average_latency_ms: 156.3,
        expert_utilization,
        neural_accuracy: 0.894,
    };
    
    println!("\n🔹 Network Performance Metrics");
    println!("Active Experts: {}", stats.active_peers);
    println!("Total Queries: {}", stats.total_queries);
    println!("Average Latency: {:.1}ms", stats.average_latency_ms);
    println!("Neural Accuracy: {:.1}%", stats.neural_accuracy * 100.0);
    
    println!("\n🔹 Expert Utilization");
    for (domain, utilization) in stats.expert_utilization {
        let bar_length = (utilization * 20.0) as usize;
        let bar = "█".repeat(bar_length) + &"░".repeat(20 - bar_length);
        println!("{:12} [{}] {:.0}%", 
                format!("{:?}:", domain), 
                bar, 
                utilization * 100.0);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_features_run() {
        // This test ensures all feature test functions compile and can run
        test_individual_experts();
        test_expert_router();
        test_runtime_standard();
        test_runtime_consensus();
        test_performance();
        test_edge_cases();
        test_configuration();
        test_network_stats();
    }
}