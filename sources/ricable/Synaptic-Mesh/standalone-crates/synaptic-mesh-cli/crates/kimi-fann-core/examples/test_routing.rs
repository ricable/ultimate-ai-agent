//! Test the enhanced routing system to verify arithmetic detection

use kimi_fann_core::{ProcessingConfig, enhanced_router::EnhancedRouter};

fn main() {
    println!("Testing Enhanced Neural Routing System");
    println!("{}", "=".repeat(60));
    println!();
    
    let config = ProcessingConfig::new();
    let mut router = EnhancedRouter::new(config);
    
    // Test queries
    let test_queries = vec![
        ("What is 2+2?", "Simple arithmetic"),
        ("What is machine learning?", "AI/ML concept"),
        ("Write a function to sort an array", "Programming task"),
        ("Calculate 10 * 5", "Multiplication"),
        ("Translate hello to Spanish", "Language task"),
        ("What is 100 divided by 4?", "Division"),
        ("Explain quantum computing", "Complex reasoning"),
        ("How do I use Python decorators?", "Programming concept"),
        ("5 plus 3 equals what?", "Word-based arithmetic"),
        ("What is the sum of 15 and 27?", "Addition with words"),
        ("Solve x^2 + 3x - 4 = 0", "Algebra"),
        ("What is AI?", "General AI question"),
    ];
    
    for (query, description) in test_queries {
        println!("Query: \"{}\" ({})", query, description);
        
        // Get classification
        let classification_json = router.classify_query(query);
        if let Ok(classification) = serde_json::from_str::<serde_json::Value>(&classification_json) {
            let primary_domain = &classification["primary_domain"];
            let complexity = classification["complexity"].as_f64().unwrap_or(0.0);
            
            println!("  → Primary Domain: {}", primary_domain);
            println!("  → Complexity Score: {:.2}", complexity);
            
            // Get route recommendation
            let route_json = router.get_route_recommendation(query);
            if let Ok(route) = serde_json::from_str::<serde_json::Value>(&route_json) {
                let confidence = route["confidence"].as_f64().unwrap_or(0.0);
                let reasoning = route["reasoning"].as_str().unwrap_or("No reasoning");
                
                println!("  → Routing Confidence: {:.2}", confidence);
                println!("  → Reasoning: {}", reasoning);
            }
        }
        
        // Process the query
        let result = router.enhanced_route(query);
        println!("  → Result: {}", if result.len() > 100 { 
            format!("{}...", &result[..97]) 
        } else { 
            result 
        });
        
        println!();
    }
    
    println!("\nRouting Test Complete! ✨");
}