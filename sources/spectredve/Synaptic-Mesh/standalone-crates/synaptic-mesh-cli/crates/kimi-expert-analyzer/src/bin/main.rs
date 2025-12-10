//! Kimi Expert Analyzer CLI

use clap::{Parser, Subcommand};
use kimi_expert_analyzer::{
    ExpertAnalyzer, AnalysisConfig, DistillationConfig, 
    ExpertDomain, OutputFormat
};
use anyhow::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze experts
    Analyze {
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: String,
        
        /// Maximum number of experts
        #[arg(short, long, default_value = "6")]
        max_experts: usize,
    },
    
    /// Distill experts for WASM
    Distill {
        /// Target size per expert
        #[arg(short, long, default_value = "50000")]
        target_size: usize,
        
        /// Quality threshold
        #[arg(short, long, default_value = "0.8")]
        quality: f64,
        
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Analyze { output, format, max_experts } => {
            analyze_experts(output, format, max_experts).await?;
        }
        Commands::Distill { target_size, quality, output } => {
            distill_experts(target_size, quality, output).await?;
        }
    }
    
    Ok(())
}

async fn analyze_experts(
    output: Option<PathBuf>, 
    format: String,
    max_experts: usize
) -> Result<()> {
    let output_format = match format.as_str() {
        "json" => OutputFormat::Json,
        "yaml" => OutputFormat::Yaml,
        "binary" => OutputFormat::Binary,
        _ => OutputFormat::Json,
    };
    
    let config = AnalysisConfig {
        max_experts,
        compression_level: 9,
        output_format,
    };
    
    let mut analyzer = ExpertAnalyzer::new(config);
    
    // Analyze all expert domains
    let domains = [
        ExpertDomain::Reasoning,
        ExpertDomain::Coding,
        ExpertDomain::Language,
        ExpertDomain::Mathematics,
        ExpertDomain::ToolUse,
        ExpertDomain::Context,
    ];
    
    for domain in domains {
        let metrics = analyzer.analyze_expert(domain)?;
        println!("Analyzed {:?}: {} parameters", metrics.domain, metrics.parameter_count);
    }
    
    let summary = analyzer.get_summary();
    println!("\nAnalysis Summary:");
    println!("  Total experts: {}", summary.total_experts);
    println!("  Total parameters: {}", summary.total_parameters);
    println!("  Average efficiency: {:.2}", summary.average_efficiency);
    println!("  Memory footprint: {} bytes", summary.memory_footprint);
    
    if let Some(output_path) = output {
        let summary_json = serde_json::to_string_pretty(&summary)?;
        std::fs::write(output_path, summary_json)?;
        println!("Results saved to file");
    }
    
    Ok(())
}

async fn distill_experts(
    target_size: usize,
    quality: f64,
    output: Option<PathBuf>
) -> Result<()> {
    let config = AnalysisConfig::default();
    let mut analyzer = ExpertAnalyzer::new(config);
    
    // Analyze all domains first
    let domains = [
        ExpertDomain::Reasoning,
        ExpertDomain::Coding,
        ExpertDomain::Language,
        ExpertDomain::Mathematics,
        ExpertDomain::ToolUse,
        ExpertDomain::Context,
    ];
    
    for domain in domains {
        analyzer.analyze_expert(domain)?;
    }
    
    let distill_config = DistillationConfig {
        target_size,
        quality_threshold: quality,
        optimization_passes: 3,
    };
    
    let distilled = analyzer.distill_experts(distill_config)?;
    
    println!("Distillation Results:");
    for expert in &distilled {
        println!("  {:?}: {} parameters (score: {:.2})", 
                expert.domain, expert.optimized_size, expert.performance_score);
    }
    
    if let Some(output_path) = output {
        let distilled_json = serde_json::to_string_pretty(&distilled)?;
        std::fs::write(output_path, distilled_json)?;
        println!("Distilled experts saved to file");
    }
    
    Ok(())
}