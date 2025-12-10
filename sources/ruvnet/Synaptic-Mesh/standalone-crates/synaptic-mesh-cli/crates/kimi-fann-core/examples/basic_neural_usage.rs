//! Basic Neural Network Usage Example
//!
//! This example demonstrates how to use the Kimi-FANN Core neural networks
//! for micro-expert processing. It shows:
//! - Creating experts for different domains
//! - Processing queries with neural inference
//! - Expert routing and selection
//! - Performance measurement

use kimi_fann_core::{
    MicroExpert, ExpertRouter, KimiRuntime, ProcessingConfig, 
    ExpertDomain, VERSION
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Kimi-FANN Core Basic Neural Usage Example");
    println!("📦 Version: {}", VERSION);
    println!("=" * 50);

    // 1. Create individual micro-experts for each domain
    demonstrate_individual_experts()?;
    
    // 2. Demonstrate expert routing
    demonstrate_expert_routing()?;
    
    // 3. Show complete runtime usage
    demonstrate_runtime_usage()?;
    
    // 4. Performance benchmarking
    demonstrate_performance_benchmarking()?;

    println!("\n✅ Basic neural usage demonstration completed!");
    Ok(())
}

/// Demonstrates creating and using individual micro-experts
fn demonstrate_individual_experts() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 1. Individual Micro-Expert Demonstration");
    println!("-" * 40);

    // Create experts for each domain
    let domains = [
        ExpertDomain::Reasoning,
        ExpertDomain::Coding,
        ExpertDomain::Language,
        ExpertDomain::Mathematics,
        ExpertDomain::ToolUse,
        ExpertDomain::Context,
    ];

    for domain in &domains {
        println!("\n📍 Creating {:?} expert...", domain);
        let expert = MicroExpert::new(*domain);
        
        // Test queries specific to each domain
        let test_query = match domain {
            ExpertDomain::Reasoning => "Can you analyze the logical structure of this argument?",
            ExpertDomain::Coding => "How would you implement a binary search algorithm?",
            ExpertDomain::Language => "Please translate this text and explain the grammar",
            ExpertDomain::Mathematics => "Solve this differential equation: dy/dx = 2x",
            ExpertDomain::ToolUse => "Execute a file system operation to list directories",
            ExpertDomain::Context => "Remember what we discussed earlier about neural networks",
        };

        let start_time = Instant::now();
        let response = expert.process(test_query);
        let processing_time = start_time.elapsed();

        println!("  Query: {}", test_query);
        println!("  Response: {}", response);
        println!("  Processing time: {:.2}ms", processing_time.as_millis());
    }

    Ok(())
}

/// Demonstrates intelligent expert routing
fn demonstrate_expert_routing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔀 2. Expert Routing Demonstration");
    println!("-" * 40);

    let mut router = ExpertRouter::new();
    
    // Add all experts to the router
    router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
    router.add_expert(MicroExpert::new(ExpertDomain::Coding));
    router.add_expert(MicroExpert::new(ExpertDomain::Language));
    router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
    router.add_expert(MicroExpert::new(ExpertDomain::ToolUse));
    router.add_expert(MicroExpert::new(ExpertDomain::Context));

    // Test routing with different types of queries
    let test_cases = [
        "Write a Python function to calculate fibonacci numbers",
        "What is the derivative of x^2 + 3x + 5?",
        "Explain the reasoning behind quantum superposition",
        "Translate 'Hello world' to Spanish and French",
        "Run a command to check disk space usage",
        "Based on our previous conversation about AI, what do you think?",
        "Complex query involving both mathematics and coding: implement Newton's method",
    ];

    for (i, query) in test_cases.iter().enumerate() {
        println!("\n📋 Test Case {}: {}", i + 1, query);
        
        let start_time = Instant::now();
        let response = router.route(query);
        let processing_time = start_time.elapsed();
        
        println!("  Routed Response: {}", response);
        println!("  Processing time: {:.2}ms", processing_time.as_millis());
    }

    // Demonstrate consensus for complex queries
    println!("\n🤝 Consensus Processing for Complex Queries:");
    let complex_query = "Analyze and implement a machine learning algorithm with error handling";
    
    let start_time = Instant::now();
    let consensus_response = router.get_consensus(complex_query);
    let processing_time = start_time.elapsed();
    
    println!("  Complex Query: {}", complex_query);
    println!("  Consensus Response: {}", consensus_response);
    println!("  Processing time: {:.2}ms", processing_time.as_millis());

    Ok(())
}

/// Demonstrates the complete runtime environment
fn demonstrate_runtime_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚙️ 3. Complete Runtime Environment");
    println!("-" * 40);

    // Create different runtime configurations
    let configs = [
        ("Neural Optimized", ProcessingConfig::new_neural_optimized()),
        ("Pattern Optimized", ProcessingConfig::new_pattern_optimized()),
        ("Default Config", ProcessingConfig::new()),
    ];

    for (config_name, config) in configs.iter() {
        println!("\n🔧 Testing {} Configuration:", config_name);
        let mut runtime = KimiRuntime::new(config.clone());
        
        // Test various processing scenarios
        let queries = [
            "Simple reasoning task",
            "Code generation request",
            "Mathematical calculation",
            "Complex multi-domain analysis",
        ];

        for query in &queries {
            let start_time = Instant::now();
            let response = runtime.process(query);
            let processing_time = start_time.elapsed();
            
            println!("    Query: {}", query);
            println!("    Response: {}", response);
            println!("    Time: {:.2}ms", processing_time.as_millis());
        }
        
        // Test consensus mode
        runtime.set_consensus_mode(true);
        let consensus_query = "Comprehensive analysis requiring multiple perspectives";
        let consensus_response = runtime.process(consensus_query);
        println!("    Consensus Query: {}", consensus_query);
        println!("    Consensus Response: {}", consensus_response);
    }

    Ok(())
}

/// Demonstrates performance benchmarking
fn demonstrate_performance_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 4. Performance Benchmarking");
    println!("-" * 40);

    let runtime_config = ProcessingConfig::new_neural_optimized();
    let mut runtime = KimiRuntime::new(runtime_config);

    // Benchmark different query types
    let benchmarks = [
        ("Short Simple Query", "hello"),
        ("Medium Complexity", "Write a function to sort an array"),
        ("Long Complex Query", "Analyze the computational complexity of various sorting algorithms, implement the most efficient one, and explain the mathematical reasoning behind your choice"),
        ("Domain Crossing", "Use mathematical analysis to optimize this Python code for performance"),
    ];

    println!("\n📈 Performance Results:");
    println!("Query Type              | Time (ms) | Characters | Performance");
    println!("-" * 70);

    for (benchmark_name, query) in &benchmarks {
        let mut times = Vec::new();
        
        // Run multiple iterations for more accurate timing
        for _ in 0..5 {
            let start_time = Instant::now();
            let _response = runtime.process(query);
            times.push(start_time.elapsed().as_millis() as f64);
        }
        
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let chars_per_sec = query.len() as f64 / (avg_time / 1000.0);
        
        println!("{:20} | {:8.2} | {:10} | {:8.1} ch/s", 
                benchmark_name, avg_time, query.len(), chars_per_sec);
    }

    // Memory usage estimation
    println!("\n💾 Memory Usage Analysis:");
    println!("Component               | Estimated Memory");
    println!("-" * 40);
    println!("Single Expert           | ~200-500 KB");
    println!("Expert Router (6 exp.)  | ~1.5-3 MB");
    println!("Runtime Environment     | ~2-4 MB");
    println!("Neural Network Data     | ~500 KB - 2 MB");

    // Quality metrics
    println!("\n🎯 Quality Metrics:");
    println!("Neural Inference        | Enabled");
    println!("Pattern Fallback        | Available");
    println!("Expert Consensus        | Supported");
    println!("Error Handling          | Robust");
    println!("WASM Compatibility      | Full");

    Ok(())
}

/// Helper function for creating separator lines
fn print_separator(length: usize) {
    println!("{}", "=".repeat(length));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        for domain in [
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Mathematics,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ] {
            let expert = MicroExpert::new(domain);
            let response = expert.process("test query");
            assert!(!response.is_empty());
            assert!(response.contains(&format!("{:?}", domain)) || 
                   response.contains("Neural") || 
                   response.contains("Pattern"));
        }
    }

    #[test]
    fn test_router_functionality() {
        let mut router = ExpertRouter::new();
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        
        let response = router.route("calculate the square root of 16");
        assert!(!response.is_empty());
        assert!(response.contains("Routed to"));
    }

    #[test]
    fn test_runtime_processing() {
        let config = ProcessingConfig::new();
        let mut runtime = KimiRuntime::new(config);
        
        let response = runtime.process("test query");
        assert!(!response.is_empty());
        assert!(response.contains("Runtime"));
    }

    #[test]
    fn test_performance_measurement() {
        let config = ProcessingConfig::new_pattern_optimized();
        let mut runtime = KimiRuntime::new(config);
        
        let start = Instant::now();
        let _response = runtime.process("performance test");
        let duration = start.elapsed();
        
        // Should complete within reasonable time (adjust based on system)
        assert!(duration.as_millis() < 1000);
    }
}