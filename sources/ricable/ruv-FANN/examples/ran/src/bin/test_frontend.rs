//! Test binary for the frontend module

use ran_intelligence_platform::integration::frontend::{FrontendApplication, FrontendConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("Testing Frontend Module...");
    
    // Create frontend config
    let config = FrontendConfig::default();
    println!("✓ Frontend config created");
    
    // Create frontend application
    let frontend = FrontendApplication::new(config);
    println!("✓ Frontend application created");
    
    // Initialize the application
    frontend.initialize().await?;
    println!("✓ Frontend initialized");
    
    // Test session management
    let session_id = frontend.create_session("test_user".to_string()).await?;
    println!("✓ Session created: {}", session_id);
    
    let session = frontend.get_session(&session_id).await?;
    println!("✓ Session retrieved: {:?}", session.is_some());
    
    // Test module retrieval
    let modules = frontend.get_modules().await?;
    println!("✓ Modules loaded: {} modules", modules.len());
    for module in &modules {
        println!("  - {} ({})", module.display_name, module.module_id);
    }
    
    // Test HTML generation
    let html = frontend.generate_dashboard_html().await?;
    println!("✓ Dashboard HTML generated ({} chars)", html.len());
    
    // Test API info generation
    let api_info = frontend.get_api_info().await?;
    println!("✓ API info generated: {}", serde_json::to_string_pretty(&api_info)?);
    
    println!("\n🎉 All frontend tests passed!");
    
    Ok(())
}