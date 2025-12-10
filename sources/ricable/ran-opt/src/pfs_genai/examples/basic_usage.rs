//! Basic usage example of the PFS-GenAI service
//! 
//! This example demonstrates:
//! - Service configuration and setup
//! - Backend registration (OpenAI and local models)
//! - Basic prompt generation
//! - Streaming responses
//! - Batch processing
//! - Metrics collection

use std::sync::Arc;
use tokio;
use pfs_genai::{
    GenAIService, GenAIConfig, CacheConfig, BatchConfig,
    Prompt, utils,
};
use pfs_genai::backends::{OpenAIConfig, LocalModelConfig, BackendFactory};
use pfs_genai::compression::CompressionService;
use pfs_genai::streaming::StreamProcessor;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Starting PFS-GenAI Service Example");
    
    // Create service configuration
    let config = GenAIConfig {
        max_tokens: 2048,
        temperature: 0.7,
        cache: CacheConfig {
            enabled: true,
            vector_db_url: "http://localhost:6333".to_string(),
            similarity_threshold: 0.8,
            ttl_secs: 1800,
            max_size_mb: 512,
        },
        batch: BatchConfig {
            enabled: true,
            max_batch_size: 5,
            batch_timeout_ms: 100,
        },
        pool_size: 10,
        timeout_secs: 30,
    };
    
    // Create the GenAI service
    let service = GenAIService::new(config).await?;
    
    // Register OpenAI backend (if API key is available)
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let openai_config = OpenAIConfig {
            api_key,
            model: "gpt-3.5-turbo".to_string(),
            ..Default::default()
        };
        
        let openai_client = BackendFactory::create_openai(openai_config, 10);
        service.register_client("openai".to_string(), openai_client);
        println!("✅ OpenAI backend registered");
    }
    
    // Register local model backend (placeholder)
    let local_config = LocalModelConfig {
        model_path: "/path/to/local/model".to_string(),
        tokenizer_path: "/path/to/tokenizer".to_string(),
        ..Default::default()
    };
    
    if let Ok(local_client) = BackendFactory::create_local_model(local_config, 5) {
        service.register_client("local".to_string(), local_client);
        println!("✅ Local model backend registered");
    }
    
    // Register mock backend for testing
    let mock_client = Arc::new(pfs_genai::client::MockClient::new("mock-model".to_string(), 5));
    service.register_client("mock".to_string(), mock_client);
    println!("✅ Mock backend registered");
    
    // Example 1: Basic prompt generation
    println!("\n📝 Example 1: Basic Prompt Generation");
    let prompt = Prompt {
        system: Some("You are an expert in 5G RAN optimization and network management.".to_string()),
        user: "Explain the key factors that affect LTE/5G cell capacity and how to optimize them.".to_string(),
        context: None,
        max_tokens: Some(500),
        temperature: Some(0.7),
    };
    
    let response = service.generate("mock", &prompt).await?;
    println!("Response: {}", response.text);
    println!("Token usage: {} total ({} prompt + {} completion)", 
             response.usage.total_tokens, 
             response.usage.prompt_tokens, 
             response.usage.completion_tokens);
    println!("Latency: {}ms", response.metadata.latency_ms);
    
    // Example 2: Prompt compression
    println!("\n🗜️  Example 2: Prompt Compression");
    let compressor = CompressionService::new();
    
    let verbose_prompt = Prompt {
        system: Some("You are a helpful assistant for network management.".to_string()),
        user: "Please provide detailed information about radio access network performance optimization techniques.".to_string(),
        context: None,
        max_tokens: None,
        temperature: None,
    };
    
    let (compressed_prompt, compression_stats) = compressor.compress(&verbose_prompt, None)?;
    println!("Original prompt: {}", verbose_prompt.user);
    println!("Compressed prompt: {}", compressed_prompt.user);
    println!("Compression ratio: {:.2}% (saved {} tokens)", 
             compression_stats.compression_ratio * 100.0,
             compression_stats.original_tokens - compression_stats.compressed_tokens);
    
    // Example 3: Streaming response
    println!("\n📡 Example 3: Streaming Response");
    let streaming_prompt = Prompt {
        system: Some("You are a technical writer.".to_string()),
        user: "Write a brief overview of 5G network slicing benefits.".to_string(),
        context: None,
        max_tokens: Some(200),
        temperature: Some(0.8),
    };
    
    let mut stream = service.generate_stream("mock", &streaming_prompt).await?;
    print!("Streaming response: ");
    
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{}", text),
            Err(e) => eprintln!("Stream error: {}", e),
        }
    }
    println!();
    
    // Example 4: Batch processing
    println!("\n📦 Example 4: Batch Processing");
    let batch_prompts = vec![
        Prompt {
            system: Some("You are a network analyst.".to_string()),
            user: "What is RSRP?".to_string(),
            context: None,
            max_tokens: Some(100),
            temperature: Some(0.5),
        },
        Prompt {
            system: Some("You are a network analyst.".to_string()),
            user: "What is RSRQ?".to_string(),
            context: None,
            max_tokens: Some(100),
            temperature: Some(0.5),
        },
        Prompt {
            system: Some("You are a network analyst.".to_string()),
            user: "What is SINR?".to_string(),
            context: None,
            max_tokens: Some(100),
            temperature: Some(0.5),
        },
    ];
    
    // Process prompts individually (they may get batched internally)
    let mut batch_responses = Vec::new();
    for prompt in batch_prompts {
        let response = service.generate("mock", &prompt).await?;
        batch_responses.push(response);
    }
    
    println!("Processed {} prompts", batch_responses.len());
    for (i, response) in batch_responses.iter().enumerate() {
        println!("Response {}: {} (batched: {})", 
                 i + 1, 
                 response.text.chars().take(50).collect::<String>(),
                 response.metadata.batched);
    }
    
    // Example 5: Metrics and monitoring
    println!("\n📊 Example 5: Metrics and Monitoring");
    let metrics = service.metrics();
    let summary = metrics.get_summary().await;
    
    println!("Service uptime: {} seconds", summary.uptime_seconds);
    println!("Cache hit rate: {:.2}%", summary.cache_hit_rate * 100.0);
    println!("Total errors: {}", summary.total_errors);
    
    for (backend, stats) in &summary.backend_summaries {
        println!("Backend '{}': {} requests, avg {}ms, {} tokens", 
                 backend, 
                 stats.total_requests, 
                 stats.avg_response_time_ms, 
                 stats.total_tokens);
    }
    
    // Example 6: Advanced features
    println!("\n🎯 Example 6: Advanced Features");
    
    // Test with RAN-specific context
    let ran_prompt = Prompt {
        system: Some("You are an expert in 5G RAN optimization with deep knowledge of KPIs and network troubleshooting.".to_string()),
        user: "Analyze this scenario: Cell XYZ shows pmRrcConnEstabSucc dropping to 85% during peak hours.".to_string(),
        context: Some(vec![
            "Cell Info: gNB123-Cell-1, Band n78, 100MHz bandwidth".to_string(),
            "Recent changes: Software upgrade to 21.Q4 yesterday".to_string(),
            "Neighbor cells: Normal performance, no issues reported".to_string(),
        ]),
        max_tokens: Some(400),
        temperature: Some(0.6),
    };
    
    let ran_response = service.generate("mock", &ran_prompt).await?;
    println!("RAN Analysis: {}", ran_response.text);
    
    // Performance comparison
    let start_time = std::time::Instant::now();
    let _quick_response = service.generate("mock", &Prompt {
        system: None,
        user: "Quick test".to_string(),
        context: None,
        max_tokens: Some(10),
        temperature: None,
    }).await?;
    let duration = start_time.elapsed();
    println!("Quick response time: {}ms", duration.as_millis());
    
    println!("\n✅ PFS-GenAI Service Example completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_example_setup() {
        let config = utils::default_test_config();
        let service = GenAIService::new(config).await.unwrap();
        
        // Should be able to create service without issues
        let mock_client = Arc::new(pfs_genai::client::MockClient::new("test".to_string(), 2));
        service.register_client("test".to_string(), mock_client);
        
        // Test basic generation
        let prompt = Prompt {
            system: None,
            user: "Test".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let response = service.generate("test", &prompt).await.unwrap();
        assert!(!response.text.is_empty());
    }
}