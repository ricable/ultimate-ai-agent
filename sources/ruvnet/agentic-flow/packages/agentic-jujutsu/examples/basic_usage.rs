//! Basic usage example for agentic-jujutsu

use agentic_jujutsu::{JJConfig, JJWrapper};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Agentic-Jujutsu Basic Usage Example\n");

    // Create a wrapper with default configuration
    println!("1. Creating JJWrapper with default config...");
    let wrapper = JJWrapper::new()?;
    println!("   ✓ Wrapper created successfully\n");

    // Create a wrapper with custom configuration
    println!("2. Creating JJWrapper with custom config...");
    let config = JJConfig::default()
        .with_verbose(true)
        .with_timeout(60000)
        .with_max_log_entries(500);

    let custom_wrapper = JJWrapper::with_config(config)?;
    println!("   ✓ Custom wrapper created successfully\n");

    // Get configuration
    println!("3. Retrieving configuration...");
    let retrieved_config = custom_wrapper.get_config();
    println!("   jj_path: {}", retrieved_config.jj_path());
    println!("   repo_path: {}", retrieved_config.repo_path());
    println!("   timeout_ms: {}", retrieved_config.timeout_ms);
    println!("   verbose: {}", retrieved_config.verbose);
    println!("   max_log_entries: {}\n", retrieved_config.max_log_entries);

    // Get statistics
    println!("4. Getting wrapper statistics...");
    let stats = wrapper.get_stats();
    println!("   Stats: {}\n", stats);

    // Get operations
    println!("5. Querying operation log...");
    let operations = wrapper.get_operations(10)?;
    println!("   Found {} operations in log\n", operations.len());

    // Get user operations (excluding snapshots)
    println!("6. Querying user-initiated operations...");
    let user_ops = wrapper.get_user_operations(5)?;
    println!("   Found {} user-initiated operations\n", user_ops.len());

    // Display operation details if any exist
    if !operations.is_empty() {
        println!("7. Recent operations:");
        for (i, op) in operations.iter().enumerate().take(5) {
            println!("   {}. {} - {}", i + 1, op.operation_type_str(), op.command);
            println!("      User: {} @ {}", op.user, op.hostname);
            println!(
                "      Duration: {}ms, Success: {}",
                op.duration_ms, op.success
            );
        }
    } else {
        println!("7. No operations found in log (this is expected for a new wrapper)");
    }

    println!("\n✓ Example completed successfully!");

    Ok(())
}
