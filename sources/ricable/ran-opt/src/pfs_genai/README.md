# PFS-GenAI: Generative AI Abstraction Service

A high-performance, production-ready abstraction layer for Large Language Model (LLM) integration with advanced optimization features.

## 🚀 Features

### Core Capabilities
- **Multi-Backend Support**: OpenAI, local models via Candle, and extensible architecture
- **Async/Await**: Fully asynchronous with tokio runtime
- **Connection Pooling**: Efficient connection management with configurable limits
- **Streaming Support**: Real-time token streaming for responsive UIs
- **Batch Processing**: Intelligent batching for improved throughput

### Advanced Optimizations
- **Semantic Caching**: Vector database integration for intelligent response caching
- **Prompt Compression**: Multiple compression algorithms to reduce token usage
- **Token Efficiency**: Smart token counting and usage optimization
- **Request Queuing**: Sophisticated queuing system for batch inference

### Production Features
- **Comprehensive Metrics**: Prometheus integration with detailed analytics
- **Health Monitoring**: Built-in health checks and service status
- **Error Handling**: Robust error handling with retry logic
- **Configuration Management**: Flexible configuration system

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pfs_genai = { path = "src/pfs_genai" }
tokio = { version = "1.0", features = ["full"] }
```

## 🛠️ Quick Start

```rust
use pfs_genai::{GenAIService, GenAIConfig, Prompt};
use pfs_genai::backends::{OpenAIConfig, BackendFactory};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create service with default configuration
    let config = pfs_genai::utils::default_dev_config();
    let service = GenAIService::new(config).await?;
    
    // Register OpenAI backend
    let openai_config = OpenAIConfig {
        api_key: "your-api-key".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        ..Default::default()
    };
    
    let openai_client = BackendFactory::create_openai(openai_config, 10);
    service.register_client("openai".to_string(), openai_client);
    
    // Create a prompt
    let prompt = Prompt {
        system: Some("You are an expert in 5G RAN optimization.".to_string()),
        user: "Explain the key factors affecting cell capacity.".to_string(),
        context: None,
        max_tokens: Some(500),
        temperature: Some(0.7),
    };
    
    // Generate response
    let response = service.generate("openai", &prompt).await?;
    println!("Response: {}", response.text);
    
    Ok(())
}
```

## 🏗️ Architecture

### Service Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        GenAIService                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Cache     │  │ Compression │  │   Batch     │  │Metrics  │ │
│  │  (Semantic) │  │  (Multi-    │  │  (Queue)    │  │ (Prom.) │ │
│  │             │  │ Algorithm)  │  │             │  │         │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   OpenAI    │  │   Local     │  │   Custom    │             │
│  │  Backend    │  │  Model      │  │  Backend    │             │
│  │             │  │ (Candle)    │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Traits

- **`LLMClient`**: Core interface for all LLM backends
- **`Cache`**: Interface for response caching systems
- **`PromptCompressor`**: Interface for prompt optimization

## 🔧 Configuration

### Basic Configuration

```rust
use pfs_genai::{GenAIConfig, CacheConfig, BatchConfig};

let config = GenAIConfig {
    max_tokens: 4096,
    temperature: 0.7,
    cache: CacheConfig {
        enabled: true,
        vector_db_url: "http://localhost:6333".to_string(),
        similarity_threshold: 0.8,
        ttl_secs: 3600,
        max_size_mb: 1024,
    },
    batch: BatchConfig {
        enabled: true,
        max_batch_size: 10,
        batch_timeout_ms: 100,
    },
    pool_size: 20,
    timeout_secs: 30,
};
```

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Vector Database (for semantic caching)
QDRANT_URL=http://localhost:6333

# Local Model Configuration
LOCAL_MODEL_PATH=/path/to/model
LOCAL_TOKENIZER_PATH=/path/to/tokenizer
```

## 📊 Monitoring & Metrics

### Prometheus Metrics

The service exposes comprehensive metrics via Prometheus:

```
# Request metrics
genai_requests_total{backend="openai",status="success"} 1234
genai_response_time_seconds{backend="openai"} 0.5

# Token usage
genai_token_usage{backend="openai",type="total"} 15000

# Cache metrics
genai_cache_total{result="hit"} 456
genai_cache_total{result="miss"} 123

# Connection metrics
genai_active_connections{backend="openai"} 5
genai_queue_size{backend="openai"} 2
```

### Health Checks

```rust
let health = service.health_check().await;
println!("Status: {}", health.status);
println!("Uptime: {} seconds", health.uptime_seconds);
```

## 🎯 Advanced Features

### Semantic Caching

```rust
// Responses are cached based on semantic similarity
let prompt1 = Prompt {
    user: "What is 5G network slicing?".to_string(),
    ..Default::default()
};

let prompt2 = Prompt {
    user: "Explain 5G network slicing concept.".to_string(),
    ..Default::default()
};

// Second request may return cached response due to semantic similarity
let response1 = service.generate("openai", &prompt1).await?;
let response2 = service.generate("openai", &prompt2).await?;
```

### Prompt Compression

```rust
use pfs_genai::compression::CompressionService;

let compressor = CompressionService::new();
let (compressed_prompt, stats) = compressor.compress(&prompt, None)?;

println!("Token reduction: {:.1}%", 
         (1.0 - stats.compression_ratio) * 100.0);
```

### Streaming Responses

```rust
use futures::StreamExt;

let mut stream = service.generate_stream("openai", &prompt).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(text) => print!("{}", text),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Batch Processing

```rust
// Requests are automatically batched for efficiency
let futures: Vec<_> = prompts.into_iter()
    .map(|prompt| service.generate("openai", &prompt))
    .collect();

let responses = futures::future::join_all(futures).await;
```

## 🔌 Backend Integration

### OpenAI Backend

```rust
use pfs_genai::backends::{OpenAIConfig, BackendFactory};

let config = OpenAIConfig {
    api_key: "sk-...".to_string(),
    model: "gpt-4".to_string(),
    base_url: "https://api.openai.com/v1".to_string(),
    max_tokens: 4096,
    temperature: 0.7,
    max_retries: 3,
    request_timeout_secs: 30,
};

let client = BackendFactory::create_openai(config, 10);
service.register_client("openai".to_string(), client);
```

### Local Model Backend

```rust
use pfs_genai::backends::{LocalModelConfig, BackendFactory};

let config = LocalModelConfig {
    model_path: "/path/to/model.safetensors".to_string(),
    tokenizer_path: "/path/to/tokenizer.json".to_string(),
    max_tokens: 2048,
    temperature: 0.7,
    top_p: 0.9,
    device: "cuda".to_string(),
};

let client = BackendFactory::create_local_model(config, 5)?;
service.register_client("local".to_string(), client);
```

## 📈 Performance Optimization

### Connection Pooling

```rust
// Configure connection pool size based on your needs
let config = GenAIConfig {
    pool_size: 20, // Max concurrent connections
    timeout_secs: 30,
    ..Default::default()
};
```

### Batch Configuration

```rust
let batch_config = BatchConfig {
    enabled: true,
    max_batch_size: 10,     // Max requests per batch
    batch_timeout_ms: 100,  // Max wait time for batching
};
```

### Cache Configuration

```rust
let cache_config = CacheConfig {
    enabled: true,
    similarity_threshold: 0.8,  // Similarity threshold for cache hits
    ttl_secs: 3600,            // Cache TTL
    max_size_mb: 1024,         // Max cache size
};
```

## 🛡️ Error Handling

```rust
use pfs_genai::GenAIError;

match service.generate("openai", &prompt).await {
    Ok(response) => println!("Success: {}", response.text),
    Err(GenAIError::Backend(msg)) => eprintln!("Backend error: {}", msg),
    Err(GenAIError::TokenLimit { current, max }) => {
        eprintln!("Token limit exceeded: {}/{}", current, max);
    }
    Err(GenAIError::Network(e)) => eprintln!("Network error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## 🧪 Testing

```rust
use pfs_genai::client::MockClient;

// Use mock client for testing
let mock_client = Arc::new(MockClient::new("test-model".to_string(), 5));
service.register_client("mock".to_string(), mock_client);

let response = service.generate("mock", &prompt).await?;
assert!(!response.text.is_empty());
```

## 📚 Examples

See the [examples](examples/) directory for comprehensive usage examples:

- [`basic_usage.rs`](examples/basic_usage.rs) - Complete service setup and usage
- More examples coming soon...

## 🤝 RAN Integration

This service is specifically designed for RAN (Radio Access Network) optimization use cases:

```rust
// RAN-specific prompt example
let ran_prompt = Prompt {
    system: Some("You are an expert in 5G RAN optimization.".to_string()),
    user: "Analyze KPI degradation: pmRrcConnEstabSucc dropped to 85%".to_string(),
    context: Some(vec![
        "Cell: gNB123-Cell-1, Band n78".to_string(),
        "Recent: Software upgrade yesterday".to_string(),
        "Neighbors: Normal performance".to_string(),
    ]),
    max_tokens: Some(400),
    temperature: Some(0.6),
};

let analysis = service.generate("openai", &ran_prompt).await?;
```

## 📝 License

Part of the RAN Optimization Platform - See main project license.

## 🔗 Dependencies

- `tokio` - Async runtime
- `candle` - Local model inference
- `reqwest` - HTTP client
- `qdrant-client` - Vector database
- `prometheus` - Metrics
- `serde` - Serialization
- `futures` - Async utilities

## 🚧 Roadmap

- [ ] Additional LLM backends (Anthropic Claude, Google PaLM)
- [ ] Advanced prompt optimization techniques
- [ ] Distributed caching with Redis
- [ ] Model fine-tuning integration
- [ ] Real-time model switching
- [ ] Cost optimization features