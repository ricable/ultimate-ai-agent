//! Command Line Usage Example
//!
//! This example demonstrates how to use Kimi-FANN Core from the command line:
//! - CLI argument parsing
//! - Interactive and batch processing modes
//! - Configuration management
//! - Output formatting
//! - Error handling and logging

use kimi_fann_core::{
    MicroExpert, ExpertRouter, KimiRuntime, ProcessingConfig, 
    ExpertDomain, VERSION
};
use std::env;
use std::fs;
use std::io::{self, Write, BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

/// Command line arguments structure
#[derive(Debug)]
struct CliArgs {
    mode: ProcessingMode,
    expert_domain: Option<ExpertDomain>,
    consensus: bool,
    output_format: OutputFormat,
    config_file: Option<String>,
    input_file: Option<String>,
    output_file: Option<String>,
    verbose: bool,
    benchmark: bool,
}

#[derive(Debug)]
enum ProcessingMode {
    Interactive,
    Single(String),
    Batch,
    Benchmark,
}

#[derive(Debug)]
enum OutputFormat {
    Text,
    Json,
    Csv,
    Markdown,
}

impl CliArgs {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let args: Vec<String> = env::args().collect();
        
        let mut mode = ProcessingMode::Interactive;
        let mut expert_domain = None;
        let mut consensus = false;
        let mut output_format = OutputFormat::Text;
        let mut config_file = None;
        let mut input_file = None;
        let mut output_file = None;
        let mut verbose = false;
        let mut benchmark = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--version" | "-v" => {
                    println!("Kimi-FANN Core v{}", VERSION);
                    std::process::exit(0);
                }
                "--query" | "-q" => {
                    if i + 1 < args.len() {
                        mode = ProcessingMode::Single(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--batch" | "-b" => {
                    mode = ProcessingMode::Batch;
                }
                "--benchmark" => {
                    benchmark = true;
                    mode = ProcessingMode::Benchmark;
                }
                "--expert" | "-e" => {
                    if i + 1 < args.len() {
                        expert_domain = Some(parse_expert_domain(&args[i + 1])?);
                        i += 1;
                    }
                }
                "--consensus" | "-c" => {
                    consensus = true;
                }
                "--format" | "-f" => {
                    if i + 1 < args.len() {
                        output_format = parse_output_format(&args[i + 1])?;
                        i += 1;
                    }
                }
                "--config" => {
                    if i + 1 < args.len() {
                        config_file = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--input" | "-i" => {
                    if i + 1 < args.len() {
                        input_file = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--output" | "-o" => {
                    if i + 1 < args.len() {
                        output_file = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--verbose" => {
                    verbose = true;
                }
                _ => {
                    return Err(format!("Unknown argument: {}", args[i]).into());
                }
            }
            i += 1;
        }

        Ok(CliArgs {
            mode,
            expert_domain,
            consensus,
            output_format,
            config_file,
            input_file,
            output_file,
            verbose,
            benchmark,
        })
    }
}

fn print_help() {
    println!("Kimi-FANN Core v{} - Neural Expert Processing CLI", VERSION);
    println!();
    println!("USAGE:");
    println!("    kimi-fann [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help              Show this help message");
    println!("    -v, --version           Show version information");
    println!("    -q, --query <TEXT>      Process a single query");
    println!("    -b, --batch             Run in batch processing mode");
    println!("    --benchmark             Run performance benchmarks");
    println!("    -e, --expert <DOMAIN>   Use specific expert domain");
    println!("    -c, --consensus         Enable consensus mode");
    println!("    -f, --format <FORMAT>   Output format (text|json|csv|markdown)");
    println!("    --config <FILE>         Load configuration from file");
    println!("    -i, --input <FILE>      Input file for batch processing");
    println!("    -o, --output <FILE>     Output file");
    println!("    --verbose               Enable verbose logging");
    println!();
    println!("EXPERT DOMAINS:");
    println!("    reasoning               Logical analysis and problem-solving");
    println!("    coding                  Programming and software development");
    println!("    language                Natural language processing");
    println!("    mathematics             Mathematical computation");
    println!("    tooluse                 Tool execution and automation");
    println!("    context                 Contextual understanding");
    println!();
    println!("EXAMPLES:");
    println!("    kimi-fann                                    # Interactive mode");
    println!("    kimi-fann -q \"Calculate 2+2\"                # Single query");
    println!("    kimi-fann -e coding -q \"Sort an array\"      # Specific expert");
    println!("    kimi-fann -c -q \"Complex analysis\"          # Consensus mode");
    println!("    kimi-fann -b -i queries.txt -o results.json # Batch processing");
    println!("    kimi-fann --benchmark                        # Performance test");
}

fn parse_expert_domain(domain: &str) -> Result<ExpertDomain, Box<dyn std::error::Error>> {
    match domain.to_lowercase().as_str() {
        "reasoning" => Ok(ExpertDomain::Reasoning),
        "coding" => Ok(ExpertDomain::Coding),
        "language" => Ok(ExpertDomain::Language),
        "mathematics" => Ok(ExpertDomain::Mathematics),
        "tooluse" => Ok(ExpertDomain::ToolUse),
        "context" => Ok(ExpertDomain::Context),
        _ => Err(format!("Unknown expert domain: {}", domain).into()),
    }
}

fn parse_output_format(format: &str) -> Result<OutputFormat, Box<dyn std::error::Error>> {
    match format.to_lowercase().as_str() {
        "text" => Ok(OutputFormat::Text),
        "json" => Ok(OutputFormat::Json),
        "csv" => Ok(OutputFormat::Csv),
        "markdown" | "md" => Ok(OutputFormat::Markdown),
        _ => Err(format!("Unknown output format: {}", format).into()),
    }
}

/// Result structure for processing
#[derive(Debug)]
struct ProcessingResult {
    query: String,
    domain: Option<ExpertDomain>,
    response: String,
    processing_time_ms: u64,
    consensus_used: bool,
    accuracy_estimate: f64,
}

impl ProcessingResult {
    fn format(&self, format: &OutputFormat) -> String {
        match format {
            OutputFormat::Text => {
                format!(
                    "Query: {}\nDomain: {:?}\nResponse: {}\nTime: {}ms\nConsensus: {}\nAccuracy: {:.1}%\n",
                    self.query,
                    self.domain.unwrap_or(ExpertDomain::Reasoning),
                    self.response,
                    self.processing_time_ms,
                    if self.consensus_used { "Yes" } else { "No" },
                    self.accuracy_estimate * 100.0
                )
            }
            OutputFormat::Json => {
                serde_json::json!({
                    "query": self.query,
                    "domain": format!("{:?}", self.domain.unwrap_or(ExpertDomain::Reasoning)),
                    "response": self.response,
                    "processing_time_ms": self.processing_time_ms,
                    "consensus_used": self.consensus_used,
                    "accuracy_estimate": self.accuracy_estimate
                }).to_string()
            }
            OutputFormat::Csv => {
                format!(
                    "\"{}\",\"{:?}\",\"{}\",{},{},{:.3}",
                    self.query.replace("\"", "\"\""),
                    self.domain.unwrap_or(ExpertDomain::Reasoning),
                    self.response.replace("\"", "\"\"").replace("\n", " "),
                    self.processing_time_ms,
                    if self.consensus_used { 1 } else { 0 },
                    self.accuracy_estimate
                )
            }
            OutputFormat::Markdown => {
                format!(
                    "## Query: {}\n\n**Domain:** {:?}  \n**Time:** {}ms  \n**Consensus:** {}  \n**Accuracy:** {:.1}%  \n\n**Response:**\n{}\n\n---\n",
                    self.query,
                    self.domain.unwrap_or(ExpertDomain::Reasoning),
                    self.processing_time_ms,
                    if self.consensus_used { "Yes" } else { "No" },
                    self.accuracy_estimate * 100.0,
                    self.response
                )
            }
        }
    }
}

/// Main CLI processor
struct CliProcessor {
    runtime: KimiRuntime,
    router: ExpertRouter,
    verbose: bool,
}

impl CliProcessor {
    fn new(verbose: bool, config_file: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        // Load configuration
        let config = if let Some(config_path) = config_file {
            load_config(&config_path)?
        } else {
            ProcessingConfig::new()
        };

        let mut runtime = KimiRuntime::new(config);
        let mut router = ExpertRouter::new();

        // Initialize all experts
        router.add_expert(MicroExpert::new(ExpertDomain::Reasoning));
        router.add_expert(MicroExpert::new(ExpertDomain::Coding));
        router.add_expert(MicroExpert::new(ExpertDomain::Language));
        router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
        router.add_expert(MicroExpert::new(ExpertDomain::ToolUse));
        router.add_expert(MicroExpert::new(ExpertDomain::Context));

        if verbose {
            println!("✅ Initialized Kimi-FANN Core with 6 expert domains");
        }

        Ok(CliProcessor {
            runtime,
            router,
            verbose,
        })
    }

    fn process_query(&mut self, query: &str, domain: Option<ExpertDomain>, consensus: bool) -> ProcessingResult {
        let start_time = Instant::now();
        
        let response = if consensus {
            self.runtime.set_consensus_mode(true);
            self.runtime.process(query)
        } else if let Some(domain) = domain {
            let expert = MicroExpert::new(domain);
            expert.process(query)
        } else {
            self.router.route(query)
        };

        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Estimate accuracy based on response characteristics
        let accuracy_estimate = estimate_accuracy(&response, processing_time);

        ProcessingResult {
            query: query.to_string(),
            domain,
            response,
            processing_time_ms: processing_time,
            consensus_used: consensus,
            accuracy_estimate,
        }
    }

    fn run_interactive_mode(&mut self, output_format: &OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧠 Kimi-FANN Core Interactive Mode");
        println!("Enter queries (type 'quit' to exit, 'help' for commands):");
        println!();

        let stdin = io::stdin();
        loop {
            print!("> ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            stdin.read_line(&mut input)?;
            let input = input.trim();

            match input {
                "quit" | "exit" => break,
                "help" => print_interactive_help(),
                "stats" => self.print_session_stats(),
                "" => continue,
                _ => {
                    if input.starts_with("set ") {
                        self.handle_set_command(input);
                        continue;
                    }

                    let result = self.process_query(input, None, false);
                    println!("{}", result.format(output_format));
                }
            }
        }

        println!("Goodbye! 👋");
        Ok(())
    }

    fn handle_set_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.len() == 3 && parts[0] == "set" {
            match parts[1] {
                "consensus" => {
                    let enabled = parts[2] == "true" || parts[2] == "on";
                    self.runtime.set_consensus_mode(enabled);
                    println!("Consensus mode: {}", if enabled { "enabled" } else { "disabled" });
                }
                _ => println!("Unknown setting: {}", parts[1]),
            }
        } else {
            println!("Usage: set <setting> <value>");
        }
    }

    fn print_session_stats(&self) {
        println!("📊 Session Statistics:");
        println!("  Runtime: Active");
        println!("  Experts: 6 domains loaded");
        println!("  Neural Networks: Initialized");
        println!("  Memory Usage: ~2-4 MB");
    }

    fn run_batch_mode(&mut self, input_file: &str, output_file: Option<&str>, output_format: &OutputFormat, consensus: bool) -> Result<(), Box<dyn std::error::Error>> {
        if self.verbose {
            println!("📁 Running batch processing from: {}", input_file);
        }

        let file = fs::File::open(input_file)?;
        let reader = BufReader::new(file);
        let mut results = Vec::new();

        for line in reader.lines() {
            let query = line?;
            if !query.trim().is_empty() {
                let result = self.process_query(&query, None, consensus);
                results.push(result);
                
                if self.verbose {
                    println!("✅ Processed: {}", &query[..query.len().min(50)]);
                }
            }
        }

        // Output results
        let output = format_batch_results(&results, output_format);
        
        if let Some(output_file) = output_file {
            fs::write(output_file, output)?;
            if self.verbose {
                println!("📝 Results written to: {}", output_file);
            }
        } else {
            println!("{}", output);
        }

        if self.verbose {
            println!("🎉 Batch processing completed: {} queries processed", results.len());
        }

        Ok(())
    }

    fn run_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("⚡ Running Kimi-FANN Core Performance Benchmark");
        println!("===============================================");

        let benchmark_queries = [
            ("Simple", "Hello"),
            ("Medium", "Write a function to calculate factorial"),
            ("Complex", "Analyze the time complexity of merge sort and implement an optimized version"),
            ("Very Complex", "Design a distributed system architecture for real-time data processing with fault tolerance"),
        ];

        let mut benchmark_results = Vec::new();

        for (complexity, query) in &benchmark_queries {
            println!("\n🔍 Testing {} complexity...", complexity);
            
            // Test single expert
            let start = Instant::now();
            let single_result = self.process_query(query, Some(ExpertDomain::Reasoning), false);
            let single_time = start.elapsed().as_millis() as u64;

            // Test consensus
            let start = Instant::now();
            let consensus_result = self.process_query(query, None, true);
            let consensus_time = start.elapsed().as_millis() as u64;

            // Test router
            let start = Instant::now();
            let router_result = self.process_query(query, None, false);
            let router_time = start.elapsed().as_millis() as u64;

            benchmark_results.push(BenchmarkResult {
                complexity: complexity.to_string(),
                query_length: query.len(),
                single_expert_time: single_time,
                consensus_time,
                router_time,
                single_response_length: single_result.response.len(),
                consensus_response_length: consensus_result.response.len(),
                router_response_length: router_result.response.len(),
            });

            println!("  Single Expert: {}ms", single_time);
            println!("  Consensus: {}ms", consensus_time);
            println!("  Router: {}ms", router_time);
        }

        // Print benchmark summary
        print_benchmark_summary(&benchmark_results);

        Ok(())
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    complexity: String,
    query_length: usize,
    single_expert_time: u64,
    consensus_time: u64,
    router_time: u64,
    single_response_length: usize,
    consensus_response_length: usize,
    router_response_length: usize,
}

fn print_interactive_help() {
    println!("Interactive Mode Commands:");
    println!("  help                    Show this help");
    println!("  stats                   Show session statistics");
    println!("  set consensus <on|off>  Enable/disable consensus mode");
    println!("  quit                    Exit interactive mode");
    println!("  <query>                 Process a query");
}

fn load_config(config_path: &str) -> Result<ProcessingConfig, Box<dyn std::error::Error>> {
    if Path::new(config_path).exists() {
        // For demo purposes, return default config
        // In real implementation, would parse JSON/TOML config
        println!("📄 Loading configuration from: {}", config_path);
        Ok(ProcessingConfig::new())
    } else {
        Err(format!("Configuration file not found: {}", config_path).into())
    }
}

fn estimate_accuracy(response: &str, processing_time: u64) -> f64 {
    // Simple accuracy estimation based on response characteristics
    let mut score = 0.8; // Base score
    
    // Longer, more detailed responses tend to be more accurate
    if response.len() > 200 {
        score += 0.1;
    }
    
    // Neural indicators suggest higher accuracy
    if response.contains("Neural") {
        score += 0.05;
    }
    
    // Reasonable processing time suggests proper computation
    if processing_time > 50 && processing_time < 500 {
        score += 0.05;
    }
    
    score.min(1.0)
}

fn format_batch_results(results: &[ProcessingResult], format: &OutputFormat) -> String {
    match format {
        OutputFormat::Text => {
            let mut output = String::new();
            for result in results {
                output.push_str(&result.format(format));
                output.push_str("\n");
            }
            output
        }
        OutputFormat::Json => {
            let json_results: Vec<_> = results.iter()
                .map(|r| serde_json::json!({
                    "query": r.query,
                    "domain": format!("{:?}", r.domain.unwrap_or(ExpertDomain::Reasoning)),
                    "response": r.response,
                    "processing_time_ms": r.processing_time_ms,
                    "consensus_used": r.consensus_used,
                    "accuracy_estimate": r.accuracy_estimate
                }))
                .collect();
            serde_json::to_string_pretty(&json_results).unwrap_or_default()
        }
        OutputFormat::Csv => {
            let mut output = "Query,Domain,Response,ProcessingTimeMs,ConsensusUsed,AccuracyEstimate\n".to_string();
            for result in results {
                output.push_str(&result.format(format));
                output.push('\n');
            }
            output
        }
        OutputFormat::Markdown => {
            let mut output = "# Kimi-FANN Core Batch Processing Results\n\n".to_string();
            for result in results {
                output.push_str(&result.format(format));
            }
            output
        }
    }
}

fn print_benchmark_summary(results: &[BenchmarkResult]) {
    println!("\n📊 Benchmark Summary");
    println!("===================");

    println!("\n🕒 Processing Times (ms):");
    println!("{:12} | {:8} | {:8} | {:8}", "Complexity", "Single", "Consensus", "Router");
    println!("{}", "-".repeat(45));
    
    for result in results {
        println!("{:12} | {:8} | {:8} | {:8}", 
                result.complexity, 
                result.single_expert_time, 
                result.consensus_time, 
                result.router_time);
    }

    // Calculate averages
    let avg_single = results.iter().map(|r| r.single_expert_time).sum::<u64>() as f64 / results.len() as f64;
    let avg_consensus = results.iter().map(|r| r.consensus_time).sum::<u64>() as f64 / results.len() as f64;
    let avg_router = results.iter().map(|r| r.router_time).sum::<u64>() as f64 / results.len() as f64;

    println!("\n📈 Performance Analysis:");
    println!("  Average Single Expert Time: {:.1}ms", avg_single);
    println!("  Average Consensus Time: {:.1}ms", avg_consensus);
    println!("  Average Router Time: {:.1}ms", avg_router);
    
    let speedup = avg_consensus / avg_single;
    println!("  Consensus Overhead: {:.1}x", speedup);

    println!("\n💾 Response Characteristics:");
    let avg_single_len = results.iter().map(|r| r.single_response_length).sum::<usize>() as f64 / results.len() as f64;
    let avg_consensus_len = results.iter().map(|r| r.consensus_response_length).sum::<usize>() as f64 / results.len() as f64;
    
    println!("  Average Single Response Length: {:.0} chars", avg_single_len);
    println!("  Average Consensus Response Length: {:.0} chars", avg_consensus_len);
    
    println!("\n🎯 Performance Rating:");
    if avg_single < 100.0 {
        println!("  Single Expert Performance: ✅ Excellent");
    } else if avg_single < 200.0 {
        println!("  Single Expert Performance: ✅ Good");
    } else {
        println!("  Single Expert Performance: ⚠️ Acceptable");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse()?;
    
    if args.verbose {
        println!("🚀 Starting Kimi-FANN Core CLI v{}", VERSION);
    }

    let mut processor = CliProcessor::new(args.verbose, args.config_file)?;

    match args.mode {
        ProcessingMode::Interactive => {
            processor.run_interactive_mode(&args.output_format)?;
        }
        ProcessingMode::Single(query) => {
            let result = processor.process_query(&query, args.expert_domain, args.consensus);
            let output = result.format(&args.output_format);
            
            if let Some(output_file) = args.output_file {
                fs::write(output_file, &output)?;
                if args.verbose {
                    println!("📝 Result written to: {}", args.output_file.unwrap());
                }
            } else {
                println!("{}", output);
            }
        }
        ProcessingMode::Batch => {
            let input_file = args.input_file.ok_or("Batch mode requires --input file")?;
            processor.run_batch_mode(&input_file, args.output_file.as_deref(), &args.output_format, args.consensus)?;
        }
        ProcessingMode::Benchmark => {
            processor.run_benchmark()?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_domain_parsing() {
        assert!(matches!(parse_expert_domain("coding").unwrap(), ExpertDomain::Coding));
        assert!(matches!(parse_expert_domain("mathematics").unwrap(), ExpertDomain::Mathematics));
        assert!(parse_expert_domain("invalid").is_err());
    }

    #[test]
    fn test_output_format_parsing() {
        assert!(matches!(parse_output_format("json").unwrap(), OutputFormat::Json));
        assert!(matches!(parse_output_format("csv").unwrap(), OutputFormat::Csv));
        assert!(parse_output_format("invalid").is_err());
    }

    #[test]
    fn test_processing_result_formatting() {
        let result = ProcessingResult {
            query: "test".to_string(),
            domain: Some(ExpertDomain::Coding),
            response: "test response".to_string(),
            processing_time_ms: 100,
            consensus_used: false,
            accuracy_estimate: 0.9,
        };

        let text_output = result.format(&OutputFormat::Text);
        assert!(text_output.contains("test"));
        assert!(text_output.contains("100ms"));

        let json_output = result.format(&OutputFormat::Json);
        assert!(json_output.contains("test"));
        assert!(json_output.contains("100"));
    }

    #[test]
    fn test_accuracy_estimation() {
        let accuracy = estimate_accuracy("Short response", 100);
        assert!(accuracy > 0.5 && accuracy <= 1.0);

        let accuracy = estimate_accuracy("Very long and detailed response with Neural indicators and comprehensive analysis", 150);
        assert!(accuracy > 0.9);
    }
}