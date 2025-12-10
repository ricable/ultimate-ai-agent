//! Example: Multi-agent workflow with jj tracking
//!
//! This example demonstrates how multiple AI agents can collaborate
//! on a shared codebase using jj for version control and coordination.

use agentic_jujutsu::{
    JJConfig, JJWrapper, JJHooksIntegration, HookContext, Result,
};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize environment
    env_logger::init();

    println!("🚀 Multi-Agent Workflow Example");
    println!("================================\n");

    // Shared session ID for all agents
    let session_id = "swarm-001".to_string();

    // Define agents and their tasks
    let agents = vec![
        ("coder", "Implement authentication module"),
        ("reviewer", "Review code quality and security"),
        ("tester", "Create comprehensive test suite"),
    ];

    println!("📋 Session: {}", session_id);
    println!("👥 Agents: {}\n", agents.len());

    // Simulate sequential agent workflow
    for (agent_id, task) in agents {
        println!("🤖 Agent: {} | Task: {}", agent_id, task);

        // Create agent-specific configuration
        let config = JJConfig::default()
            .with_verbose(true)
            .with_agentdb_sync(true);

        let wrapper = JJWrapper::new(config)?;
        let mut integration = JJHooksIntegration::new(wrapper, true);

        // Pre-task hook
        let ctx = HookContext::new(
            agent_id.to_string(),
            session_id.clone(),
            task.to_string(),
        );

        integration.on_pre_task(ctx.clone()).await?;
        println!("  ✅ Pre-task hook executed");

        // Simulate agent work
        sleep(Duration::from_millis(500)).await;

        // Simulate file edits
        let files = match agent_id {
            "coder" => vec!["src/auth.rs", "src/user.rs"],
            "reviewer" => vec!["REVIEW.md"],
            "tester" => vec!["tests/auth_test.rs", "tests/integration_test.rs"],
            _ => vec![],
        };

        for file in files {
            integration.on_post_edit(file, ctx.clone()).await?;
            println!("  📝 Edited: {}", file);
            sleep(Duration::from_millis(200)).await;
        }

        // Post-task hook
        let operations = integration.on_post_task(ctx.clone()).await?;
        println!("  ✅ Post-task hook executed");
        println!("  📊 Operations logged: {}\n", operations.len());

        sleep(Duration::from_millis(300)).await;
    }

    println!("✨ Multi-agent workflow completed successfully!");
    println!("\n💡 All operations tracked in jj and synced to AgentDB");
    println!("🔍 Run 'jj log' to see the operation history");

    Ok(())
}
