use kimi_fann_core::{MicroExpert, ExpertDomain, ExpertRouter, ProcessingConfig};
use kimi_fann_core::enhanced_router::EnhancedRouter;

fn main() {
    println!("Testing Kimi-FANN Response Quality Improvements\n");
    
    // Test queries
    let test_queries = vec![
        // Machine Learning & AI
        "What is machine learning?",
        "What is deep learning?",
        "Explain neural networks",
        "What is AI?",
        
        // Programming
        "What is an array?",
        "Explain loops in programming",
        "What is recursion?",
        "What is a linked list?",
        
        // Mathematics
        "What is calculus?",
        "Explain statistics",
        "What is linear algebra?",
        "What is the Pythagorean theorem?",
        
        // Language & NLP
        "What is natural language processing?",
        "What is grammar?",
        "Hello",
        "Hi there!",
        
        // General
        "What is an algorithm?",
        "What are data structures?",
    ];
    
    // Test with enhanced router first
    println!("=== Testing with Enhanced Router ===\n");
    let config = ProcessingConfig::new();
    let mut router = EnhancedRouter::new(config);
    
    for query in &test_queries {
        println!("Query: {}", query);
        let response = router.enhanced_route(query);
        println!("Response: {}\n", response);
        println!("{}", "-".repeat(80));
        println!();
    }
    
    // Test direct expert responses
    println!("\n=== Testing Direct Expert Responses ===\n");
    
    // Test reasoning expert
    let reasoning_expert = MicroExpert::new(ExpertDomain::Reasoning);
    println!("Reasoning Expert - Query: What is machine learning?");
    let response = reasoning_expert.process("What is machine learning?");
    println!("Response: {}\n", response);
    
    // Test coding expert
    let coding_expert = MicroExpert::new(ExpertDomain::Coding);
    println!("Coding Expert - Query: What is an array?");
    let response = coding_expert.process("What is an array?");
    println!("Response: {}\n", response);
    
    // Test math expert
    let math_expert = MicroExpert::new(ExpertDomain::Mathematics);
    println!("Math Expert - Query: What is calculus?");
    let response = math_expert.process("What is calculus?");
    println!("Response: {}\n", response);
    
    // Test language expert
    let language_expert = MicroExpert::new(ExpertDomain::Language);
    println!("Language Expert - Query: Hello");
    let response = language_expert.process("Hello");
    println!("Response: {}\n", response);
    
    // Test with expert router
    println!("\n=== Testing with Expert Router ===\n");
    let mut expert_router = ExpertRouter::new();
    
    // Add all domain experts
    for domain in [
        ExpertDomain::Reasoning,
        ExpertDomain::Coding,
        ExpertDomain::Language,
        ExpertDomain::Mathematics,
        ExpertDomain::ToolUse,
        ExpertDomain::Context,
    ] {
        expert_router.add_expert(MicroExpert::new(domain));
    }
    
    // Test a few key queries
    let key_queries = vec![
        "What is machine learning?",
        "What is an array?",
        "Hello",
    ];
    
    for query in key_queries {
        println!("Query: {}", query);
        let response = expert_router.route(query);
        println!("Response: {}\n", response);
    }
    
    println!("Testing complete!");
}