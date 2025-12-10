use kimi_fann_core::{EnhancedRouter, ProcessingConfig, ExpertDomain};

fn main() {
    let config = ProcessingConfig::new();
    let mut router = EnhancedRouter::new(config);
    
    // Test queries
    let test_queries = vec![
        "What is 2+2?",
        "What is machine learning?",
        "Write a function to sort an array",
        "Calculate 10 * 5",
        "Translate hello to Spanish",
        "What is 100 divided by 4?",
        "Explain quantum computing",
        "How do I use Python decorators?",
        "5 plus 3 equals what?",
        "What is the sum of 15 and 27?",
    ];
    
    println!("Neural Routing Test Results:");
    println!("{}", "=".repeat(60));
    
    for query in test_queries {
        let classification_json = router.classify_query(query);
        let classification: serde_json::Value = serde_json::from_str(&classification_json).unwrap();
        
        let primary_domain = &classification["primary_domain"];
        let complexity = classification["complexity"].as_f64().unwrap_or(0.0);
        
        println!("Query: \"{}\"", query);
        println!("  → Domain: {:?}", primary_domain);
        println!("  → Complexity: {:.2}", complexity);
        
        // Also test routing
        let route_json = router.get_route_recommendation(query);
        let route: serde_json::Value = serde_json::from_str(&route_json).unwrap();
        let confidence = route["confidence"].as_f64().unwrap_or(0.0);
        println!("  → Confidence: {:.2}", confidence);
        println!();
    }
}