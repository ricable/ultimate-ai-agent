//! Basic usage example for Kimi-FANN Core
//!
//! This example demonstrates creating micro-experts, routing requests,
//! and managing memory efficiently.

use kimi_fann_core::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Kimi-FANN Core Basic Usage Example");
    println!("=====================================");

    // Initialize expert registry
    let mut registry = ExpertRegistry::new();

    // Create a coding expert
    let coding_expert = create_coding_expert()?;
    println!("✅ Created coding expert (ID: {})", coding_expert.id());
    registry.register_expert(coding_expert)?;

    // Create a reasoning expert
    let reasoning_expert = create_reasoning_expert()?;
    println!("✅ Created reasoning expert (ID: {})", reasoning_expert.id());
    registry.register_expert(reasoning_expert)?;

    // Create a language expert
    let language_expert = create_language_expert()?;
    println!("✅ Created language expert (ID: {})", language_expert.id());
    registry.register_expert(language_expert)?;

    // Create expert router
    let router_config = RouterConfig::default();
    let router_json = serde_json::to_string(&router_config)?;
    let mut router = ExpertRouter::new(&router_json)?;

    // Register expert profiles with the router
    register_expert_profiles(&mut router, &registry)?;

    // Create execution context
    let context = RequestContext {
        request: "Write a Python function to calculate fibonacci numbers".to_string(),
        complexity: 0.6,
        token_count: 50,
        history_length: 0,
        required_capabilities: vec![ExpertDomain::Coding, ExpertDomain::Language],
        performance_requirements: PerformanceRequirements::default(),
        metadata: HashMap::new(),
    };

    println!("\n🎯 Processing request: \"{}\"", context.request);

    // Plan execution
    let context_json = serde_json::to_string(&context)?;
    let plan_json = router.plan_execution(&context_json)?;
    let plan: ExecutionPlan = serde_json::from_str(&plan_json)?;

    println!("📋 Execution plan created:");
    println!("   - Selected {} experts", plan.experts.len());
    println!("   - Estimated time: {:.2}ms", plan.estimated_time);
    println!("   - Estimated memory: {}KB", plan.estimated_memory / 1024);

    // Create memory pool
    let memory_config = MemoryPoolConfig::default();
    let memory_json = serde_json::to_string(&memory_config)?;
    let memory_pool = WasmMemoryPool::new(&memory_json)?;

    // Create execution engine
    let exec_config = ExecutionConfig::default();
    let exec_json = serde_json::to_string(&exec_config)?;
    let mut execution_engine = ExecutionEngine::new(&exec_json, memory_pool)?;

    // Register experts with execution engine
    for (_, expert) in registry.experts.iter() {
        let expert_json = expert.to_json()?;
        execution_engine.register_expert(&expert_json)?;
    }

    // Execute the plan
    println!("\n⚡ Executing micro-experts...");
    let exec_context = ExecutionContext {
        request_id: "example_request".to_string(),
        input_data: vec![0.1, 0.2, 0.3, 0.4, 0.5], // Example input features
        strategy: ExecutionStrategy::Adaptive,
        timeout: 5000.0,
        priority: 5,
        retry_on_failure: true,
        max_retries: 3,
        metadata: HashMap::new(),
    };

    let exec_context_json = serde_json::to_string(&exec_context)?;
    let result_promise = execution_engine.execute_plan(&plan_json, &exec_context_json);

    // In a real WASM environment, you'd await this promise
    println!("🚀 Execution initiated (would await in WASM environment)");

    // Demonstrate individual expert execution
    println!("\n🔍 Testing individual expert execution:");
    for &expert_id in &plan.experts {
        let result_json = execution_engine.execute_expert(expert_id, &exec_context.input_data)?;
        let result: ExpertExecutionResult = serde_json::from_str(&result_json)?;
        
        println!("   Expert {}: {} (confidence: {:.2}, time: {:.2}ms)", 
                 expert_id, 
                 if result.success { "✅" } else { "❌" },
                 result.confidence,
                 result.execution_time);
    }

    // Show statistics
    let stats_json = execution_engine.get_stats()?;
    let stats: ExecutionStats = serde_json::from_str(&stats_json)?;
    
    println!("\n📊 Execution Statistics:");
    println!("   - Total executions: {}", stats.total_executions);
    println!("   - Success rate: {:.1}%", stats.successful_executions as f32 / stats.total_executions as f32 * 100.0);
    println!("   - Average execution time: {:.2}ms", stats.avg_execution_time);
    println!("   - Average confidence: {:.2}", stats.avg_confidence);

    // Demonstrate context management
    println!("\n📝 Testing context management:");
    let context_config = ContextConfig::default();
    let context_config_json = serde_json::to_string(&context_config)?;
    let mut context_window = ContextWindow::new(&context_config_json)?;

    context_window.add_content(&context.request, "user")?;
    context_window.add_content("I'll help you create a fibonacci function in Python.", "assistant")?;
    context_window.add_content("Here's an efficient implementation using iteration.", "assistant")?;

    println!("   - Context utilization: {:.1}%", context_window.get_utilization() * 100.0);
    
    let context_stats_json = context_window.get_stats()?;
    let context_stats: ContextStats = serde_json::from_str(&context_stats_json)?;
    println!("   - Active tokens: {}", context_stats.active_tokens);
    println!("   - Compression ratio: {:.2}", context_stats.compression_ratio);

    println!("\n✨ Example completed successfully!");
    Ok(())
}

/// Create a coding expert specialized in code generation
fn create_coding_expert() -> Result<KimiMicroExpert, Box<dyn std::error::Error>> {
    let config = ExpertConfig {
        id: 1,
        domain: ExpertDomain::Coding,
        specialization: Specialization::CodeGeneration,
        architecture: NetworkArchitecture {
            input_size: 128,
            hidden_layers: vec![64, 32],
            output_size: 16,
            activations: vec![
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::Linear,
            ],
            dropout_rates: vec![],
        },
        training_config: TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            regularization: 0.01,
        },
        performance_thresholds: PerformanceThresholds {
            min_accuracy: 0.85,
            max_inference_time: 80.0,
            max_memory_usage: 15 * 1024 * 1024, // 15MB
            min_confidence: 0.75,
        },
    };

    let config_json = serde_json::to_string(&config)?;
    let expert = KimiMicroExpert::new(&config_json)
        .map_err(|e| format!("Failed to create coding expert: {:?}", e))?;

    Ok(expert)
}

/// Create a reasoning expert specialized in logical inference
fn create_reasoning_expert() -> Result<KimiMicroExpert, Box<dyn std::error::Error>> {
    let config = ExpertConfig {
        id: 2,
        domain: ExpertDomain::Reasoning,
        specialization: Specialization::LogicalInference,
        architecture: NetworkArchitecture {
            input_size: 64,
            hidden_layers: vec![32, 16],
            output_size: 8,
            activations: vec![
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::Sigmoid,
            ],
            dropout_rates: vec![],
        },
        training_config: TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 150,
            regularization: 0.01,
        },
        performance_thresholds: PerformanceThresholds {
            min_accuracy: 0.90,
            max_inference_time: 50.0,
            max_memory_usage: 8 * 1024 * 1024, // 8MB
            min_confidence: 0.80,
        },
    };

    let config_json = serde_json::to_string(&config)?;
    let expert = KimiMicroExpert::new(&config_json)
        .map_err(|e| format!("Failed to create reasoning expert: {:?}", e))?;

    Ok(expert)
}

/// Create a language expert specialized in natural language understanding
fn create_language_expert() -> Result<KimiMicroExpert, Box<dyn std::error::Error>> {
    let config = ExpertConfig {
        id: 3,
        domain: ExpertDomain::Language,
        specialization: Specialization::LanguageUnderstanding,
        architecture: NetworkArchitecture {
            input_size: 256,
            hidden_layers: vec![128, 64, 32],
            output_size: 20,
            activations: vec![
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::ReLU,
                synaptic_neural_wasm::Activation::Linear,
            ],
            dropout_rates: vec![],
        },
        training_config: TrainingConfig {
            learning_rate: 0.0005,
            batch_size: 64,
            epochs: 200,
            regularization: 0.02,
        },
        performance_thresholds: PerformanceThresholds {
            min_accuracy: 0.88,
            max_inference_time: 120.0,
            max_memory_usage: 25 * 1024 * 1024, // 25MB
            min_confidence: 0.70,
        },
    };

    let config_json = serde_json::to_string(&config)?;
    let expert = KimiMicroExpert::new(&config_json)
        .map_err(|e| format!("Failed to create language expert: {:?}", e))?;

    Ok(expert)
}

/// Register expert profiles with the router
fn register_expert_profiles(
    router: &mut ExpertRouter,
    registry: &ExpertRegistry,
) -> Result<(), Box<dyn std::error::Error>> {
    for (expert_id, expert) in &registry.experts {
        let profile = ExpertProfile {
            id: *expert_id,
            domain: expert.domain,
            specialization: expert.specialization.clone(),
            avg_execution_time: 50.0, // Placeholder
            success_rate: 0.95,        // Placeholder
            memory_usage: expert.memory_usage(),
            capability_scores: create_capability_scores(expert.domain),
            dependencies: vec![],
            complements: find_complementary_experts(expert.domain, registry),
        };

        let profile_json = serde_json::to_string(&profile)?;
        router.register_expert(&profile_json)?;
    }

    Ok(())
}

/// Create capability scores for an expert domain
fn create_capability_scores(domain: ExpertDomain) -> HashMap<String, f32> {
    let mut scores = HashMap::new();

    match domain {
        ExpertDomain::Coding => {
            scores.insert("code_generation".to_string(), 0.95);
            scores.insert("code_analysis".to_string(), 0.90);
            scores.insert("debugging".to_string(), 0.85);
            scores.insert("optimization".to_string(), 0.80);
        }
        ExpertDomain::Reasoning => {
            scores.insert("logical_inference".to_string(), 0.95);
            scores.insert("problem_solving".to_string(), 0.90);
            scores.insert("decision_making".to_string(), 0.85);
            scores.insert("pattern_recognition".to_string(), 0.80);
        }
        ExpertDomain::Language => {
            scores.insert("text_understanding".to_string(), 0.95);
            scores.insert("text_generation".to_string(), 0.90);
            scores.insert("translation".to_string(), 0.85);
            scores.insert("summarization".to_string(), 0.80);
        }
        _ => {
            scores.insert("general".to_string(), 0.75);
        }
    }

    scores
}

/// Find complementary experts for a given domain
fn find_complementary_experts(
    domain: ExpertDomain,
    registry: &ExpertRegistry,
) -> Vec<ExpertId> {
    let mut complements = Vec::new();

    for (expert_id, expert) in &registry.experts {
        // Define complementary relationships
        let is_complement = match (domain, expert.domain) {
            (ExpertDomain::Coding, ExpertDomain::Language) => true,
            (ExpertDomain::Language, ExpertDomain::Coding) => true,
            (ExpertDomain::Reasoning, ExpertDomain::Mathematics) => true,
            (ExpertDomain::Mathematics, ExpertDomain::Reasoning) => true,
            (ExpertDomain::Coding, ExpertDomain::Reasoning) => true,
            (ExpertDomain::Reasoning, ExpertDomain::Coding) => true,
            _ => false,
        };

        if is_complement {
            complements.push(*expert_id);
        }
    }

    complements
}