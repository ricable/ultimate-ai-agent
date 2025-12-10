//! CLI tool for hook integration
//!
//! This binary is called by agentic-flow hooks to integrate with jj operations.
//!
//! # Usage
//!
//! ```bash
//! # Pre-task hook
//! jj-agent-hook pre-task --agent-id coder-1 --session-id swarm-001 --description "Implement auth"
//!
//! # Post-edit hook
//! jj-agent-hook post-edit --file src/auth.rs --agent-id coder-1 --session-id swarm-001
//!
//! # Post-task hook
//! jj-agent-hook post-task --agent-id coder-1 --session-id swarm-001
//! ```

use agentic_jujutsu::{HookContext, JJConfig, JJHooksIntegration, JJWrapper, Result};
use clap::{Parser, Subcommand};
use std::process::exit;

#[derive(Parser)]
#[command(name = "jj-agent-hook")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Jujutsu hook integration for agentic-flow")]
#[command(long_about = "CLI tool for integrating Jujutsu VCS with agentic-flow hooks system")]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Path to jj executable
    #[arg(long, global = true, default_value = "jj")]
    jj_path: String,

    /// Repository path
    #[arg(long, global = true, default_value = ".")]
    repo_path: String,

    /// Enable AgentDB synchronization
    #[arg(long, global = true)]
    enable_agentdb: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute pre-task hook
    PreTask {
        /// Agent identifier
        #[arg(long)]
        agent_id: String,

        /// Session identifier
        #[arg(long)]
        session_id: String,

        /// Task description
        #[arg(long)]
        description: String,

        /// Additional metadata as JSON
        #[arg(long)]
        metadata: Option<String>,
    },

    /// Execute post-edit hook
    PostEdit {
        /// File that was edited
        #[arg(long)]
        file: String,

        /// Agent identifier
        #[arg(long)]
        agent_id: String,

        /// Session identifier
        #[arg(long)]
        session_id: String,

        /// Optional change description
        #[arg(long)]
        description: Option<String>,
    },

    /// Execute post-task hook
    PostTask {
        /// Agent identifier
        #[arg(long)]
        agent_id: String,

        /// Session identifier
        #[arg(long)]
        session_id: String,

        /// Task description
        #[arg(long)]
        description: Option<String>,
    },

    /// Detect and report conflicts
    DetectConflicts {
        /// Agent identifier
        #[arg(long)]
        agent_id: String,

        /// Session identifier
        #[arg(long)]
        session_id: String,
    },

    /// Query operation history
    QueryHistory {
        /// Session identifier to filter by
        #[arg(long)]
        session_id: Option<String>,

        /// Agent identifier to filter by
        #[arg(long)]
        agent_id: Option<String>,

        /// Number of operations to retrieve
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Configure logging
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    // Run command
    if let Err(e) = run_command(cli).await {
        eprintln!("Error: {}", e);
        exit(1);
    }
}

async fn run_command(cli: Cli) -> Result<()> {
    // Create JJ configuration
    let config = JJConfig::default()
        .with_jj_path(cli.jj_path)
        .with_repo_path(cli.repo_path)
        .with_verbose(cli.verbose)
        .with_agentdb_sync(cli.enable_agentdb);

    // Create wrapper and integration
    let wrapper = JJWrapper::with_config(config)?;
    let mut integration = JJHooksIntegration::new(wrapper, cli.enable_agentdb);

    match cli.command {
        Commands::PreTask {
            agent_id,
            session_id,
            description,
            metadata,
        } => {
            let mut ctx = HookContext::new(agent_id, session_id, description);

            // Parse metadata if provided
            if let Some(meta_str) = metadata {
                if let Ok(meta_value) = serde_json::from_str(&meta_str) {
                    ctx = ctx.with_metadata(meta_value);
                }
            }

            let event = integration.on_pre_task(ctx).await?;

            println!("✅ Pre-task hook executed successfully");
            println!("📋 Session ID: {}", event.context.session_id);
            println!("🤖 Agent ID: {}", event.context.agent_id);
            println!("📝 Task: {}", event.context.task_description);

            if cli.verbose {
                println!("\n🔍 Event details:");
                println!("{}", serde_json::to_string_pretty(&event)?);
            }
        }

        Commands::PostEdit {
            file,
            agent_id,
            session_id,
            description,
        } => {
            let task_desc = description.unwrap_or_else(|| format!("Edit {}", file));
            let ctx = HookContext::new(agent_id, session_id, task_desc);

            let operation = integration.on_post_edit(&file, ctx).await?;

            println!("✅ Post-edit hook executed successfully");
            println!("📄 File: {}", file);
            println!("🆔 Operation ID: {}", operation.id);
            println!("📝 Command: {}", operation.command);

            if cli.verbose {
                println!("\n🔍 Operation details:");
                println!("{}", serde_json::to_string_pretty(&operation)?);
            }
        }

        Commands::PostTask {
            agent_id,
            session_id,
            description,
        } => {
            let task_desc = description.unwrap_or_else(|| "Task completed".to_string());
            let ctx = HookContext::new(agent_id, session_id, task_desc);

            let operations = integration.on_post_task(ctx).await?;

            println!("✅ Post-task hook executed successfully");
            println!("📊 Operations in session: {}", operations.len());

            if !operations.is_empty() && cli.verbose {
                println!("\n🔍 Operations:");
                for op in &operations {
                    println!("  - {} | {}", op.id, op.command);
                }
            }
        }

        Commands::DetectConflicts {
            agent_id,
            session_id,
        } => {
            let ctx = HookContext::new(agent_id, session_id, "Conflict detection".to_string());

            // TODO: Implement actual conflict detection
            let conflicts: Vec<String> = vec![];

            if conflicts.is_empty() {
                println!("✅ No conflicts detected");
            } else {
                println!("⚠️  Conflicts detected:");
                for conflict in &conflicts {
                    println!("  - {}", conflict);
                }

                let event = integration.on_conflict_detected(conflicts, ctx).await?;

                if cli.verbose {
                    println!("\n🔍 Event details:");
                    println!("{}", serde_json::to_string_pretty(&event)?);
                }
            }
        }

        Commands::QueryHistory {
            session_id,
            agent_id,
            limit,
        } => {
            println!("📊 Query History");
            println!("  Session: {}", session_id.as_deref().unwrap_or("all"));
            println!("  Agent: {}", agent_id.as_deref().unwrap_or("all"));
            println!("  Limit: {}", limit);

            // TODO: Implement actual history query
            println!("\n⚠️  History query not yet implemented");
            println!("This feature will query AgentDB for operation history");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test pre-task command
        let cli = Cli::parse_from(&[
            "jj-agent-hook",
            "pre-task",
            "--agent-id",
            "test-agent",
            "--session-id",
            "session-001",
            "--description",
            "test task",
        ]);

        assert!(!cli.verbose);
        assert_eq!(cli.jj_path, "jj");

        match cli.command {
            Commands::PreTask { agent_id, .. } => {
                assert_eq!(agent_id, "test-agent");
            }
            _ => panic!("Expected PreTask command"),
        }
    }
}
