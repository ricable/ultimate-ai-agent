//! MCP server example

use reasoningbank_mcp::{McpServer, McpConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("🚀 Starting ReasoningBank MCP Server\n");

    // Configure server
    let mut config = McpConfig::default();
    config.storage.database_path = PathBuf::from("reasoningbank_mcp.db");

    // Create server
    let server = McpServer::new(config).await?;

    // Display server info
    let info = server.get_server_info();
    println!("📋 Server Info:");
    println!("  Name: {}", info.name);
    println!("  Version: {}", info.version);
    println!("  Tools: {}", info.tools_count);
    println!("  Resources: {}", info.resources_count);
    println!();

    // List available tools
    println!("🛠️  Available Tools:");
    for tool in server.list_tools() {
        println!("  • {}: {}", tool.name, tool.description);
    }
    println!();

    // List available resources
    println!("📦 Available Resources:");
    for resource in server.list_resources() {
        println!("  • {}: {}", resource.uri_pattern, resource.description);
    }
    println!();

    // Example: Store a pattern
    println!("📝 Example: Storing a pattern...");

    let store_params = serde_json::json!({
        "task_description": "Implement caching layer with Redis",
        "task_category": "performance-optimization",
        "strategy": "redis-caching",
        "success_score": 0.92,
        "duration_seconds": 8.5,
        "context": {
            "cache_hit_rate": 0.85,
            "latency_reduction": "60%"
        }
    });

    let result = server.execute_tool("reasoning_store", store_params).await?;
    println!("✓ Pattern stored: {}", result.get("pattern_id").unwrap());
    println!("  Insight: {:?}", result.get("insight"));
    println!();

    // Example: Retrieve recommendations
    println!("🔍 Example: Getting recommendations...");

    let retrieve_params = serde_json::json!({
        "task_description": "Add caching to improve performance",
        "task_category": "performance-optimization"
    });

    let result = server.execute_tool("reasoning_retrieve", retrieve_params).await?;
    println!("✓ Recommendation:");
    println!("{}", serde_json::to_string_pretty(&result)?);
    println!();

    // Example: Analyze category
    println!("📊 Example: Analyzing category...");

    let analyze_params = serde_json::json!({
        "category": "performance-optimization"
    });

    let result = server.execute_tool("reasoning_analyze", analyze_params).await?;
    println!("✓ Analysis:");
    println!("{}", serde_json::to_string_pretty(&result)?);
    println!();

    // Example: Access resource
    println!("📈 Example: Accessing metrics resource...");

    let metrics = server.get_resource("reasoning://metrics/performance").await?;
    println!("✓ Performance Metrics:");
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    println!();

    println!("✅ MCP Server examples complete!");
    println!("💾 Database saved to: reasoningbank_mcp.db");
    println!("\n🎯 Next steps:");
    println!("  1. Integrate with Claude using MCP protocol");
    println!("  2. Use reasoning_store to save task patterns");
    println!("  3. Use reasoning_retrieve to get recommendations");
    println!("  4. Monitor performance with resources");

    Ok(())
}
