//! RAN Optimizer LLM Agent
//!
//! WASI-NN based LLM inference for Ericsson RAN optimization automation.
//! Uses GGML backend for efficient model inference with streaming support.

use std::env;
use std::io::{self, Write};
use wasi_nn::{GraphBuilder, GraphEncoding, ExecutionTarget, Error, BackendError};
use serde::{Deserialize, Serialize};

/// RAN Optimization Configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct RanOptimizerConfig {
    /// Context window size for LLM
    pub ctx_size: u32,
    /// Maximum tokens to predict
    pub n_predict: u32,
    /// Number of GPU layers for acceleration
    pub n_gpu_layers: u32,
    /// Temperature for sampling (0.0-1.0)
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Repeat penalty for token diversity
    pub repeat_penalty: f32,
    /// Stream output tokens to stdout
    pub stream_stdout: bool,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for RanOptimizerConfig {
    fn default() -> Self {
        Self {
            ctx_size: 8192,
            n_predict: 2048,
            n_gpu_layers: 35,
            temperature: 0.3,  // Lower temperature for more deterministic RAN decisions
            top_p: 0.9,
            repeat_penalty: 1.1,
            stream_stdout: true,
            verbose: false,
        }
    }
}

/// RAN Optimization Categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Coverage optimization (antenna tilt, power)
    Coverage,
    /// Capacity optimization (load balancing, handover)
    Capacity,
    /// Interference mitigation
    Interference,
    /// Energy efficiency (power saving modes)
    Energy,
    /// Quality of Service optimization
    QoS,
    /// Anomaly detection and resolution
    Anomaly,
    /// Network slicing optimization
    Slicing,
    /// Handover optimization
    Handover,
}

/// RAN Metrics Input
#[derive(Debug, Serialize, Deserialize)]
pub struct RanMetrics {
    pub cell_id: String,
    pub site_name: Option<String>,
    pub timestamp: String,
    pub kpis: RanKpis,
    pub alarms: Vec<String>,
    pub optimization_type: OptimizationType,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RanKpis {
    /// Reference Signal Received Power (dBm)
    pub rsrp: Option<f32>,
    /// Reference Signal Received Quality (dB)
    pub rsrq: Option<f32>,
    /// Signal to Interference plus Noise Ratio (dB)
    pub sinr: Option<f32>,
    /// Downlink throughput (Mbps)
    pub dl_throughput: Option<f32>,
    /// Uplink throughput (Mbps)
    pub ul_throughput: Option<f32>,
    /// Physical Resource Block utilization (%)
    pub prb_utilization: Option<f32>,
    /// Connected users count
    pub connected_users: Option<u32>,
    /// Handover success rate (%)
    pub ho_success_rate: Option<f32>,
    /// Call drop rate (%)
    pub call_drop_rate: Option<f32>,
    /// Latency (ms)
    pub latency: Option<f32>,
    /// Packet loss rate (%)
    pub packet_loss: Option<f32>,
}

/// RAN Optimization Recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub cell_id: String,
    pub optimization_type: OptimizationType,
    pub severity: Severity,
    pub action: String,
    pub parameters: Vec<ParameterChange>,
    pub expected_improvement: String,
    pub reasoning: String,
    pub risk_level: RiskLevel,
    pub requires_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ParameterChange {
    pub parameter_name: String,
    pub current_value: String,
    pub recommended_value: String,
    pub unit: Option<String>,
}

/// System prompt for RAN optimization
const RAN_SYSTEM_PROMPT: &str = r#"You are an expert Ericsson RAN (Radio Access Network) optimization AI assistant. Your role is to analyze network metrics, identify issues, and provide actionable optimization recommendations.

## Your Expertise:
- 4G LTE and 5G NR network optimization
- Ericsson ENM (Ericsson Network Manager) configurations
- RAN KPI analysis and correlation
- Self-Organizing Network (SON) functions
- Mobility Load Balancing (MLB)
- Coverage and Capacity Optimization (CCO)
- Interference management
- Energy efficiency optimization

## Response Format:
Always provide responses in valid JSON format with the following structure:
{
  "analysis": "Brief analysis of the current situation",
  "root_cause": "Identified root cause if applicable",
  "recommendations": [
    {
      "priority": 1,
      "action": "Specific action to take",
      "parameter": "ENM parameter name",
      "current_value": "Current value",
      "recommended_value": "New value",
      "expected_impact": "Expected improvement",
      "risk": "low|medium|high"
    }
  ],
  "monitoring_period": "Suggested monitoring duration",
  "rollback_plan": "Steps to rollback if needed"
}

## Guidelines:
1. Always consider the impact on neighboring cells
2. Recommend conservative changes first
3. Consider time-of-day patterns in traffic
4. Account for seasonal variations
5. Prioritize service availability
6. Follow Ericsson best practices for parameter tuning
7. Consider both immediate fixes and long-term optimizations"#;

/// Build conversation prompt with RAN context
fn build_ran_prompt(metrics: &RanMetrics, user_query: &str) -> String {
    let metrics_json = serde_json::to_string_pretty(metrics).unwrap_or_default();

    format!(
        r#"<|system|>
{system_prompt}
</s>
<|user|>
## Current RAN Metrics:
```json
{metrics}
```

## User Query:
{query}

Please analyze the metrics and provide optimization recommendations.
</s>
<|assistant|>
"#,
        system_prompt = RAN_SYSTEM_PROMPT,
        metrics = metrics_json,
        query = user_query
    )
}

/// Build a Llama2-style prompt for chat
fn build_llama2_prompt(system: &str, user: &str) -> String {
    format!(
        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
        system, user
    )
}

/// Build a Llama3-style prompt for chat
fn build_llama3_prompt(system: &str, user: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        system, user
    )
}

/// Interactive chat mode for RAN optimization
fn run_interactive_chat(model_name: &str, config: &RanOptimizerConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 RAN Optimizer LLM Agent");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Model: {}", model_name);
    println!("Context: {} tokens | Max Output: {} tokens", config.ctx_size, config.n_predict);
    println!("Temperature: {} | GPU Layers: {}", config.temperature, config.n_gpu_layers);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nCommands:");
    println!("  /analyze <cell_id> - Analyze a specific cell");
    println!("  /metrics <json>    - Input custom metrics");
    println!("  /mode <type>       - Set optimization mode (coverage/capacity/interference/energy)");
    println!("  /clear             - Clear conversation history");
    println!("  /exit              - Exit the session");
    println!("\nEnter your RAN optimization query:\n");

    // Build WASI-NN graph with GGML backend
    let config_json = serde_json::json!({
        "ctx-size": config.ctx_size,
        "n-predict": config.n_predict,
        "n-gpu-layers": config.n_gpu_layers,
        "temp": config.temperature,
        "top-p": config.top_p,
        "repeat-penalty": config.repeat_penalty,
        "stream-stdout": config.stream_stdout,
    });

    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(config_json.to_string())
        .build_from_cache(model_name)?;

    let mut context = graph.init_execution_context()?;
    let mut saved_prompt = String::new();

    loop {
        print!("\n🗼 > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input.split_whitespace().next() {
                Some("/exit") => {
                    println!("👋 Exiting RAN Optimizer. Goodbye!");
                    break;
                }
                Some("/clear") => {
                    saved_prompt.clear();
                    context = graph.init_execution_context()?;
                    println!("🔄 Conversation cleared.");
                    continue;
                }
                Some("/mode") => {
                    let mode = input.strip_prefix("/mode ").unwrap_or("coverage");
                    println!("📡 Optimization mode set to: {}", mode);
                    continue;
                }
                _ => {}
            }
        }

        // Build prompt with RAN context
        let prompt = if saved_prompt.is_empty() {
            build_llama3_prompt(RAN_SYSTEM_PROMPT, input)
        } else {
            format!(
                "{}<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                saved_prompt, input
            )
        };

        // Set input tensor
        let tensor_data = prompt.as_bytes().to_vec();
        context.set_input(0, wasi_nn::TensorType::U8, &[tensor_data.len()], &tensor_data)?;

        // Stream inference
        print!("\n📊 ");
        io::stdout().flush()?;

        let mut output_tokens = String::new();

        loop {
            match context.compute_single() {
                Ok(_) => {
                    // Get output token
                    let mut output_buffer = vec![0u8; 128];
                    let output_size = context.get_output(0, &mut output_buffer)?;
                    if output_size > 0 {
                        if let Ok(token) = std::str::from_utf8(&output_buffer[..output_size]) {
                            output_tokens.push_str(token);
                            if config.stream_stdout {
                                print!("{}", token);
                                io::stdout().flush()?;
                            }
                        }
                    }
                }
                Err(Error::BackendError(BackendError::EndOfSequence)) => {
                    break;
                }
                Err(Error::BackendError(BackendError::ContextFull)) => {
                    println!("\n⚠️ Context full - clearing history and starting fresh.");
                    saved_prompt.clear();
                    context = graph.init_execution_context()?;
                    break;
                }
                Err(e) => {
                    eprintln!("\n❌ Inference error: {:?}", e);
                    break;
                }
            }
        }

        context.fini_single()?;

        // Save conversation history
        saved_prompt = format!(
            "{}<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
            prompt, output_tokens
        );

        println!();
    }

    Ok(())
}

/// Single-shot inference for RAN analysis
fn analyze_ran_metrics(model_name: &str, config: &RanOptimizerConfig, metrics: &RanMetrics) -> Result<String, Box<dyn std::error::Error>> {
    let config_json = serde_json::json!({
        "ctx-size": config.ctx_size,
        "n-predict": config.n_predict,
        "n-gpu-layers": config.n_gpu_layers,
        "temp": config.temperature,
        "top-p": config.top_p,
        "repeat-penalty": config.repeat_penalty,
        "stream-stdout": false,
    });

    let graph = GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
        .config(config_json.to_string())
        .build_from_cache(model_name)?;

    let mut context = graph.init_execution_context()?;

    let user_query = format!(
        "Analyze the following RAN metrics for cell {} and provide optimization recommendations:",
        metrics.cell_id
    );
    let prompt = build_ran_prompt(metrics, &user_query);

    let tensor_data = prompt.as_bytes().to_vec();
    context.set_input(0, wasi_nn::TensorType::U8, &[tensor_data.len()], &tensor_data)?;

    // Full compute for single-shot
    context.compute()?;

    // Get output
    let mut output_buffer = vec![0u8; config.n_predict as usize * 4];
    let output_size = context.get_output(0, &mut output_buffer)?;

    let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();
    Ok(output)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Get model name from args or use default
    let model_name = args.get(1).map(|s| s.as_str()).unwrap_or("default");

    // Check for mode
    let mode = args.get(2).map(|s| s.as_str()).unwrap_or("interactive");

    // Load config from environment or use defaults
    let config = RanOptimizerConfig {
        ctx_size: env::var("CTX_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192),
        n_predict: env::var("N_PREDICT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048),
        n_gpu_layers: env::var("N_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(35),
        temperature: env::var("TEMPERATURE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.3),
        top_p: env::var("TOP_P")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.9),
        repeat_penalty: env::var("REPEAT_PENALTY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.1),
        stream_stdout: env::var("STREAM_STDOUT")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true),
        verbose: env::var("VERBOSE")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false),
    };

    match mode {
        "interactive" | "chat" => {
            run_interactive_chat(model_name, &config)?;
        }
        "analyze" => {
            // Read metrics from stdin or file
            let metrics_json = args.get(3).cloned().unwrap_or_else(|| {
                let mut input = String::new();
                io::stdin().read_line(&mut input).expect("Failed to read metrics");
                input
            });

            let metrics: RanMetrics = serde_json::from_str(&metrics_json)?;
            let result = analyze_ran_metrics(model_name, &config, &metrics)?;
            println!("{}", result);
        }
        "batch" => {
            println!("Batch mode not yet implemented");
        }
        _ => {
            eprintln!("Unknown mode: {}. Use 'interactive', 'analyze', or 'batch'", mode);
            std::process::exit(1);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RanOptimizerConfig::default();
        assert_eq!(config.ctx_size, 8192);
        assert_eq!(config.temperature, 0.3);
    }

    #[test]
    fn test_build_llama3_prompt() {
        let prompt = build_llama3_prompt("System message", "User message");
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("System message"));
        assert!(prompt.contains("User message"));
    }

    #[test]
    fn test_ran_metrics_serialization() {
        let metrics = RanMetrics {
            cell_id: "CELL001".to_string(),
            site_name: Some("Site A".to_string()),
            timestamp: "2024-01-15T10:30:00Z".to_string(),
            kpis: RanKpis {
                rsrp: Some(-85.0),
                rsrq: Some(-10.0),
                sinr: Some(15.0),
                dl_throughput: Some(150.0),
                ul_throughput: Some(50.0),
                prb_utilization: Some(75.0),
                connected_users: Some(120),
                ho_success_rate: Some(98.5),
                call_drop_rate: Some(0.5),
                latency: Some(20.0),
                packet_loss: Some(0.1),
            },
            alarms: vec!["HIGH_PRB_UTILIZATION".to_string()],
            optimization_type: OptimizationType::Capacity,
        };

        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("CELL001"));
    }
}
