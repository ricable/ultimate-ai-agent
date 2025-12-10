//! Kimi-FANN Core Command Line Interface
//! 
//! Ask Kimi questions directly from the command line and get neural inference responses.
//! 
//! Usage:
//!   cargo run --bin kimi "your question here"
//!   cargo run --bin kimi --expert coding "write a function"
//!   cargo run --bin kimi --consensus "complex multi-domain question"

use std::env;
use std::time::Instant;
use kimi_fann_core::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_help();
        return;
    }
    
    // Parse command line arguments
    let mut query = String::new();
    let mut expert_type: Option<ExpertDomain> = None;
    let mut use_consensus = false;
    let mut show_performance = false;
    let mut interactive = false;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--expert" | "-e" => {
                if i + 1 < args.len() {
                    expert_type = parse_expert_domain(&args[i + 1]);
                    i += 1;
                } else {
                    eprintln!("Error: --expert requires a domain name");
                    return;
                }
            }
            "--consensus" | "-c" => {
                use_consensus = true;
            }
            "--performance" | "-p" => {
                show_performance = true;
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--version" | "-v" => {
                println!("Kimi-FANN Core v{}", VERSION);
                return;
            }
            _ => {
                if !query.is_empty() {
                    query.push(' ');
                }
                query.push_str(&args[i]);
            }
        }
        i += 1;
    }
    
    if interactive {
        run_interactive_mode();
        return;
    }
    
    if query.is_empty() {
        eprintln!("Error: Please provide a question to ask Kimi");
        print_help();
        return;
    }
    
    // Process the query
    let start_time = Instant::now();
    
    println!("🤖 Kimi-FANN Core v{} - Neural Inference Engine", VERSION);
    println!("{}", "=".repeat(60));
    println!("❓ Question: {}", query);
    println!("{}", "-".repeat(60));
    
    let response = if let Some(domain) = expert_type {
        // Use specific expert
        println!("🎯 Using {} Expert", format!("{:?}", domain));
        let expert = MicroExpert::new(domain);
        expert.process(&query)
    } else if use_consensus {
        // Use multi-expert consensus
        println!("🌟 Using Multi-Expert Consensus");
        let config = ProcessingConfig::new();
        let mut runtime = KimiRuntime::new(config);
        runtime.set_consensus_mode(true);
        runtime.process(&query)
    } else {
        // Use intelligent routing
        println!("🎯 Using Intelligent Expert Routing");
        let config = ProcessingConfig::new();
        let mut runtime = KimiRuntime::new(config);
        runtime.process(&query)
    };
    
    let duration = start_time.elapsed();
    
    println!("💭 Response:");
    println!("{}", response);
    
    if show_performance {
        println!("\n{}", "-".repeat(60));
        println!("⚡ Performance Metrics:");
        println!("   Processing Time: {:?}", duration);
        println!("   Response Length: {} characters", response.len());
        println!("   Words: {} words", response.split_whitespace().count());
        
        // Check for neural indicators
        let neural_active = response.contains("Neural:") || 
                           response.contains("conf=") || 
                           response.contains("patterns=");
        println!("   Neural Processing: {}", if neural_active { "✅ Active" } else { "⚠️ Fallback" });
        
        // Check for expert routing
        let routing_active = response.contains("Routed to") || 
                            response.contains("expert") ||
                            response.contains("experts active");
        println!("   Expert Routing: {}", if routing_active { "✅ Active" } else { "⚠️ Basic" });
    }
    
    println!("{}", "=".repeat(60));
}

fn parse_expert_domain(domain_str: &str) -> Option<ExpertDomain> {
    match domain_str.to_lowercase().as_str() {
        "reasoning" | "reason" | "logic" => Some(ExpertDomain::Reasoning),
        "coding" | "code" | "programming" | "prog" => Some(ExpertDomain::Coding),
        "mathematics" | "math" | "maths" => Some(ExpertDomain::Mathematics),
        "language" | "lang" | "text" => Some(ExpertDomain::Language),
        "tooluse" | "tool" | "tools" => Some(ExpertDomain::ToolUse),
        "context" | "ctx" | "memory" => Some(ExpertDomain::Context),
        _ => {
            eprintln!("Unknown expert domain: {}. Available: reasoning, coding, mathematics, language, tooluse, context", domain_str);
            None
        }
    }
}

fn run_interactive_mode() {
    println!("🤖 Kimi-FANN Core v{} - Interactive Mode", VERSION);
    println!("{}", "=".repeat(60));
    println!("Ask Kimi questions! Type 'quit' or 'exit' to stop.");
    println!("Commands:");
    println!("  /expert <domain> - Switch to specific expert (reasoning, coding, math, language, tool, context)");
    println!("  /consensus - Toggle multi-expert consensus mode");
    println!("  /performance - Toggle performance display");
    println!("  /help - Show this help");
    println!("{}", "-".repeat(60));
    
    let config = ProcessingConfig::new();
    let mut runtime = KimiRuntime::new(config);
    let mut current_expert: Option<MicroExpert> = None;
    let mut consensus_mode = false;
    let mut show_perf = false;
    
    loop {
        print!("\n🤖 Kimi> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            eprintln!("Error reading input");
            continue;
        }
        
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input == "quit" || input == "exit" {
            println!("👋 Goodbye!");
            break;
        }
        
        // Handle commands
        if input.starts_with('/') {
            let parts: Vec<&str> = input.split_whitespace().collect();
            match parts[0] {
                "/expert" => {
                    if parts.len() > 1 {
                        if let Some(domain) = parse_expert_domain(parts[1]) {
                            current_expert = Some(MicroExpert::new(domain));
                            consensus_mode = false;
                            println!("✅ Switched to {:?} expert", domain);
                        }
                    } else {
                        current_expert = None;
                        println!("✅ Switched back to intelligent routing");
                    }
                }
                "/consensus" => {
                    consensus_mode = !consensus_mode;
                    current_expert = None;
                    println!("✅ Consensus mode: {}", if consensus_mode { "ON" } else { "OFF" });
                    runtime.set_consensus_mode(consensus_mode);
                }
                "/performance" => {
                    show_perf = !show_perf;
                    println!("✅ Performance display: {}", if show_perf { "ON" } else { "OFF" });
                }
                "/help" => {
                    println!("Commands:");
                    println!("  /expert <domain> - Switch to specific expert");
                    println!("  /consensus - Toggle multi-expert consensus mode");
                    println!("  /performance - Toggle performance display");
                    println!("  Available domains: reasoning, coding, mathematics, language, tooluse, context");
                }
                _ => {
                    println!("❌ Unknown command. Type /help for available commands.");
                }
            }
            continue;
        }
        
        // Process the query
        let start = Instant::now();
        
        let response = if let Some(ref expert) = current_expert {
            expert.process(input)
        } else {
            runtime.process(input)
        };
        
        let duration = start.elapsed();
        
        println!("\n💭 {}", response);
        
        if show_perf {
            println!("\n⚡ {:?} | {} chars | Neural: {}", 
                    duration, 
                    response.len(),
                    if response.contains("Neural:") || response.contains("conf=") { "✅" } else { "⚠️" });
        }
    }
}

fn print_help() {
    println!("🤖 Kimi-FANN Core v{} - Neural Inference CLI", VERSION);
    println!("{}", "=".repeat(60));
    println!("USAGE:");
    println!("  cargo run --bin kimi -- \"your question here\"");
    println!("  cargo run --bin kimi -- [OPTIONS] \"your question\"");
    println!();
    println!("⚠️  IMPORTANT: When using 'cargo run', always include '--' to separate");
    println!("   cargo's arguments from kimi's arguments!");
    println!();
    println!("   Alternatively, use the wrapper script: ./kimi.sh \"your question\"");
    println!();
    println!("OPTIONS:");
    println!("  -e, --expert <DOMAIN>    Use specific expert domain");
    println!("  -c, --consensus          Use multi-expert consensus");
    println!("  -p, --performance        Show performance metrics");
    println!("  -i, --interactive        Start interactive mode");
    println!("  -v, --version            Show version");
    println!("  -h, --help               Show this help");
    println!();
    println!("EXPERT DOMAINS:");
    println!("  reasoning     Logic, analysis, critical thinking");
    println!("  coding        Programming, algorithms, software");
    println!("  mathematics   Math, calculations, formulas");
    println!("  language      Translation, text, linguistics");
    println!("  tooluse       Commands, operations, procedures");
    println!("  context       Memory, conversation, references");
    println!();
    println!("EXAMPLES:");
    println!("  # Ask any question with intelligent routing");
    println!("  cargo run --bin kimi -- \"What is machine learning?\"");
    println!();
    println!("  # Use specific expert");
    println!("  cargo run --bin kimi -- --expert coding \"Write a sorting function\"");
    println!();
    println!("  # Use consensus for complex queries");
    println!("  cargo run --bin kimi -- --consensus \"Design a neural network\"");
    println!();
    println!("  # Interactive mode");
    println!("  cargo run --bin kimi -- --interactive");
    println!();
    println!("  # Show performance metrics");
    println!("  cargo run --bin kimi -- --performance \"Calculate 2+2\"");
    println!();
    println!("  # Or use the convenient wrapper script:");
    println!("  ./kimi.sh \"What is machine learning?\"");
    println!("  ./kimi.sh --expert coding \"Write a function\"");
}