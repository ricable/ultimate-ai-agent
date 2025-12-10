//! Synaptic Mesh CLI Binary

use clap::Parser;
use synaptic_mesh_cli::{Cli, cli_to_command, execute_command, init_tracing};
use colored::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    
    let cli = Cli::parse();
    let command = cli_to_command(cli);
    
    println!("{}", "Synaptic Neural Mesh".bright_cyan().bold());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━".bright_cyan());
    
    match execute_command(command).await {
        Ok(result) => {
            println!("{} {}", "✓".green().bold(), format!("{:?}", result).green());
        }
        Err(e) => {
            eprintln!("{} {}", "✗".red().bold(), format!("{}", e).red());
            std::process::exit(1);
        }
    }
    
    Ok(())
}