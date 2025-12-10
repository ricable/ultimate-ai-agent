use kimi_fann_core::ProcessingConfig;
use kimi_fann_core::enhanced_router::EnhancedRouter;

fn main() {
    println!("Debugging Routing Issues\n");
    
    let config = ProcessingConfig::new();
    let router = EnhancedRouter::new(config);
    
    // Test queries that are having issues
    let test_queries = vec![
        "Explain loops in programming",
        "Explain statistics", 
        "What is the Pythagorean theorem?",
    ];
    
    for query in test_queries {
        println!("Query: {}", query);
        
        // Get classification
        let classification_json = router.classify_query(query);
        println!("Classification: {}", classification_json);
        
        // Get route recommendation
        let recommendation_json = router.get_route_recommendation(query);
        println!("Recommendation: {}", recommendation_json);
        
        println!("{}", "-".repeat(80));
        println!();
    }
}