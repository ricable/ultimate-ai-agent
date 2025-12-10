# neuro-divergent-registry

[![Crates.io](https://img.shields.io/crates/v/neuro-divergent-registry.svg)](https://crates.io/crates/neuro-divergent-registry)
[![Documentation](https://docs.rs/neuro-divergent-registry/badge.svg)](https://docs.rs/neuro-divergent-registry)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

A comprehensive model factory and registry system for dynamic neural network model creation and management. This crate provides the foundation for discovering, creating, and managing neural network models across the neuro-divergent ecosystem with support for plugins, serialization, and automatic model discovery.

## 🚀 Technical Introduction

The neuro-divergent-registry crate implements a sophisticated registry architecture that serves as the central nervous system for neural network model management. It provides dynamic dispatch capabilities without runtime cost overhead, type-safe model creation with compile-time guarantees, and a plugin architecture for unlimited extensibility.

### Registry System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelRegistry                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Model Index    │  │   Plugin System  │  │  Discovery  │ │
│  │  - By Category  │  │  - Dynamic Load  │  │  - Auto Scan │ │
│  │  - By Capability│  │  - Security      │  │  - Filtering│ │
│  │  - By Tags      │  │  - Health Check  │  │  - Caching  │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ModelFactory                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ Dynamic Creation│  │   Configuration  │  │ Performance │ │
│  │ - By Name       │  │  - Validation    │  │ - Caching   │ │
│  │ - By Config     │  │  - Templates     │  │ - Stats     │ │
│  │ - Parallel      │  │  - Builder       │  │ - Profiling │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features Overview

### Core Registry Features
- **Global Model Registry**: Centralized management of 27+ neural network models
- **Dynamic Model Creation**: Create models by string name or configuration
- **Thread-Safe Operations**: Concurrent access with efficient locking mechanisms
- **Model Categorization**: Organized by 6 categories (Basic, Recurrent, Advanced, Transformer, Specialized, Custom)
- **Advanced Search**: Query models by capabilities, tags, performance requirements
- **Statistics Tracking**: Comprehensive usage and performance metrics

### Plugin System
- **Dynamic Loading**: Load custom models from shared libraries
- **Security Framework**: Checksum verification, digital signatures, sandboxing
- **Health Monitoring**: Plugin health checks and error recovery
- **Version Management**: Compatibility checking and dependency resolution
- **Hot Reloading**: Runtime plugin management without system restart

### Model Discovery
- **Automatic Registration**: Discover and register built-in and plugin models
- **Intelligent Filtering**: Category, capability, and pattern-based filtering
- **Performance Profiling**: Automatic benchmarking during discovery
- **Caching System**: Efficient caching with configurable expiry
- **Parallel Discovery**: Multi-threaded discovery for performance

### Factory System
- **Multiple Creation Methods**: By name, configuration, or template
- **Batch Operations**: Create multiple models in parallel
- **Validation Framework**: Comprehensive configuration validation
- **Performance Optimization**: Caching and lazy loading
- **Builder Pattern**: Fluent API for complex model creation

## 🏗️ Architecture Details

### Registry Architecture

The `ModelRegistry` serves as the central hub for all model management operations:

```rust
pub struct ModelRegistry {
    // Registered models by name
    models: RwLock<IndexMap<String, ModelDescriptor>>,
    // Models organized by category
    by_category: RwLock<AHashMap<ModelCategory, Vec<String>>>,
    // Models organized by capability
    by_capability: RwLock<AHashMap<String, Vec<String>>>,
    // Model aliases and tags
    aliases: RwLock<AHashMap<String, String>>,
    tags: RwLock<AHashMap<String, Vec<String>>>,
    // Plugin management
    plugins: RwLock<AHashMap<String, Plugin>>,
    // Performance statistics
    stats: RwLock<RegistryStats>,
}
```

**Key Design Principles:**
- **Lock Granularity**: Fine-grained locking for maximum concurrency
- **Memory Efficiency**: IndexMap for ordered storage with O(1) access
- **Search Optimization**: Multiple indices for fast lookups
- **Plugin Isolation**: Secure plugin boundaries with error containment

### Factory Architecture

The `ModelFactory` provides dynamic model creation with type safety:

```rust
pub struct ModelFactory {
    // Type-specific creators
    creators: RwLock<HashMap<String, ModelCreatorFn<f32>>>,
    creators_f64: RwLock<HashMap<String, ModelCreatorFn<f64>>>,
    // Template system
    templates: RwLock<HashMap<String, ModelTemplate>>,
    // Performance caching
    cache: RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>,
    // Creation statistics
    stats: RwLock<CreationStats>,
}
```

**Advanced Features:**
- **Type Safety**: Compile-time type checking for model creation
- **Zero-Cost Abstraction**: Dynamic dispatch without runtime overhead
- **Template System**: Reusable model configurations
- **Performance Caching**: Intelligent caching for frequently used models

## 📋 Usage Examples

### Basic Model Creation

```rust
use neuro_divergent_registry::{ModelFactory, ModelRegistry, ModelCategory};

// Create a model using the factory
let model = ModelFactory::create::<f32>("MLP")?;

// List all available models
let models = ModelFactory::list_models();
println!("Available models: {}", models.len());

// Get models by category
let transformer_models = ModelRegistry::global()
    .read()
    .list_by_category(ModelCategory::Transformer);
```

### Advanced Model Creation with Configuration

```rust
use neuro_divergent_registry::{ModelConfig, ModelFactory, ModelBuilder};
use serde_json::json;

// Create model with custom configuration
let mut config = ModelConfig::new("LSTM", ModelCategory::Recurrent);
config.set_parameter("hidden_size", json!(128));
config.set_parameter("num_layers", json!(2));
config.set_parameter("dropout", json!(0.1));
config.set_dimensions(Some(64), Some(10));

let model = ModelFactory::create_from_config::<f32>(&config)?;

// Using the builder pattern
let model = ModelBuilder::new("TFT", ModelCategory::Transformer)
    .input_size(100)
    .output_size(1)
    .parameter("d_model", json!(512))
    .parameter("n_heads", json!(8))
    .parameter("dropout", json!(0.1))
    .benchmark(true)
    .build::<f32>()?;
```

### Parallel Model Creation

```rust
use neuro_divergent_registry::{ModelFactory, CreationOptions};

// Create multiple models in parallel
let model_names = vec!["MLP", "LSTM", "TFT", "NBEATS"];
let models = ModelFactory::create_models::<f32>(model_names)?;

// With custom options
let options = CreationOptions {
    benchmark: true,
    validate: true,
    cache: true,
    timeout_ms: Some(5000),
    ..Default::default()
};

let models = ModelFactory::create_models_with_options::<f32>(
    vec!["Informer", "AutoFormer", "PatchTST"], 
    options
)?;
```

### Model Discovery and Search

```rust
use neuro_divergent_registry::{
    ModelRegistry, SearchCriteria, PerformanceRequirements,
    discover_all_models, DiscoveryConfig
};

// Discover all available models
let discovery_result = discover_all_models()?;
println!("Discovered {} models", discovery_result.models.len());

// Advanced search
let criteria = SearchCriteria {
    name_pattern: Some("*BEATS*".to_string()),
    category: Some(ModelCategory::Advanced),
    required_capabilities: vec!["batch_processing".to_string()],
    performance_requirements: Some(PerformanceRequirements {
        max_creation_time_ms: Some(1000.0),
        min_throughput_samples_per_sec: Some(100.0),
        ..Default::default()
    }),
    ..Default::default()
};

let registry = ModelRegistry::global();
let models = registry.read().search(&criteria);
```

## 📚 API Documentation

### Core Registry Interfaces

#### ModelRegistry

The global registry for model management:

```rust
impl ModelRegistry {
    // Registration
    pub fn register<T: Float>(&self, name: &str, descriptor: ModelDescriptor) -> RegistryResult<()>;
    pub fn register_info(&self, info: ModelInfo) -> RegistryResult<()>;
    pub fn unregister(&self, name: &str) -> RegistryResult<()>;
    
    // Queries
    pub fn has_model(&self, name: &str) -> bool;
    pub fn get_model_info(&self, name: &str) -> Option<ModelInfo>;
    pub fn list_all(&self) -> Vec<ModelInfo>;
    pub fn list_by_category(&self, category: ModelCategory) -> Vec<ModelInfo>;
    pub fn list_by_capability(&self, capability: &str) -> Vec<ModelInfo>;
    pub fn search(&self, criteria: &SearchCriteria) -> Vec<ModelInfo>;
    
    // Plugin management
    pub fn register_plugin(&self, plugin: Plugin) -> RegistryResult<()>;
    pub fn unregister_plugin(&self, name: &str) -> RegistryResult<()>;
    pub fn list_plugins(&self) -> Vec<&Plugin>;
    
    // Statistics
    pub fn get_stats(&self) -> RegistryStats;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

#### ModelFactory

Dynamic model creation interface:

```rust
impl ModelFactory {
    // Basic creation
    pub fn create<T: Float>(&self, name: &str) -> RegistryResult<Box<dyn BaseModel<T>>>;
    pub fn create_from_config<T: Float>(&self, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<T>>>;
    
    // Advanced creation
    pub fn create_with_options<T: Float>(&self, name: &str, options: CreationOptions) -> RegistryResult<Box<dyn BaseModel<T>>>;
    pub fn create_models<T: Float>(&self, names: Vec<&str>) -> RegistryResult<Vec<Box<dyn BaseModel<T>>>>;
    
    // Template management
    pub fn create_from_template<T: Float>(&self, template_name: &str) -> RegistryResult<Box<dyn BaseModel<T>>>;
    pub fn register_template(&self, template: ModelTemplate) -> RegistryResult<()>;
    
    // Utilities
    pub fn list_models() -> Vec<ModelInfo>;
    pub fn get_model_info(name: &str) -> Option<ModelInfo>;
    pub fn get_stats(&self) -> CreationStats;
    pub fn clear_cache(&self);
}
```

### Factory Methods

#### Dynamic Creation Methods

```rust
// Create by name
let model = ModelFactory::create::<f32>("MLP")?;

// Create from configuration
let config = ModelConfig::new("LSTM", ModelCategory::Recurrent);
let model = ModelFactory::create_from_config::<f64>(&config)?;

// Create with options
let options = CreationOptions {
    benchmark: true,
    validate: false,
    cache: true,
    ..Default::default()
};
let model = ModelFactory::create_with_options::<f32>("TFT", options)?;
```

#### Batch Operations

```rust
// Create multiple models
let models = ModelFactory::create_models::<f32>(vec!["MLP", "LSTM", "GRU"])?;

// Parallel creation with options
let models = ModelFactory::create_models_with_options::<f32>(
    vec!["Informer", "AutoFormer", "FedFormer"],
    CreationOptions::default()
)?;
```

#### Global Factory Methods

```rust
// Using global factory instance
let model = ModelFactory::create_global::<f32>("NBEATS")?;
let config = ModelConfig::new("TiDE", ModelCategory::Advanced);
let model = ModelFactory::create_from_config_global::<f32>(&config)?;
```

## 🔌 Plugin System

### Creating Custom Models

To create a custom model plugin, implement the `PluginInterface`:

```rust
use neuro_divergent_registry::{PluginInterface, PluginDescriptor, BaseModel, ModelInfo};

pub struct MyCustomPlugin {
    descriptor: PluginDescriptor,
}

impl PluginInterface for MyCustomPlugin {
    fn descriptor(&self) -> &PluginDescriptor {
        &self.descriptor
    }
    
    fn initialize(&mut self) -> RegistryResult<()> {
        log::info!("Initializing custom plugin");
        Ok(())
    }
    
    fn shutdown(&mut self) -> RegistryResult<()> {
        log::info!("Shutting down custom plugin");
        Ok(())
    }
    
    fn create_model<T: Float>(&self, name: &str, config: &ModelConfig) -> RegistryResult<Box<dyn BaseModel<T>>> {
        match name {
            "MyCustomModel" => Ok(Box::new(MyCustomModel::new(config)?)),
            _ => Err(RegistryError::ModelNotFound(name.to_string()))
        }
    }
    
    fn list_models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                name: "MyCustomModel".to_string(),
                category: ModelCategory::Custom,
                version: "1.0.0".to_string(),
                description: "My custom neural network model".to_string(),
                parameter_types: vec!["f32".to_string(), "f64".to_string()],
                input_size: None,
                output_size: None,
                capabilities: ModelCapabilities::default(),
                performance: None,
                metadata: HashMap::new(),
            }
        ]
    }
    
    fn get_model_info(&self, name: &str) -> Option<ModelInfo> {
        self.list_models().into_iter().find(|m| m.name == name)
    }
    
    fn health_check(&self) -> RegistryResult<PluginHealth> {
        Ok(PluginHealth {
            healthy: true,
            timestamp: SystemTime::now(),
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: HashMap::new(),
        })
    }
    
    fn version(&self) -> &str {
        &self.descriptor.version
    }
    
    fn is_compatible(&self, runtime_version: &str) -> bool {
        runtime_version >= &self.descriptor.min_runtime_version
    }
}

// Export the plugin creation function
#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn PluginInterface {
    let plugin = MyCustomPlugin {
        descriptor: PluginDescriptor {
            name: "my-custom-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Custom neural network models".to_string(),
            author: "Your Name".to_string(),
            license: "MIT".to_string(),
            // ... other fields
        },
    };
    
    Box::into_raw(Box::new(plugin))
}
```

### Plugin Descriptor File

Create a `plugin.toml` file for your plugin:

```toml
[plugin]
name = "my-custom-plugin"
version = "1.0.0"
description = "Custom neural network models"
author = "Your Name"
license = "MIT"
homepage = "https://github.com/yourname/my-custom-plugin"
repository = "https://github.com/yourname/my-custom-plugin"
keywords = ["neural-networks", "custom", "plugin"]
min_runtime_version = "1.0.0"

[plugin.capabilities]
model_creation = true
model_training = true
model_inference = true
model_serialization = false
gpu_acceleration = false
distributed_computing = false
thread_safe = true
memory_safe = true

[[plugin.models]]
name = "MyCustomModel"
category = "Custom"
version = "1.0.0"
description = "My custom neural network model"
parameter_types = ["f32", "f64"]

[plugin.models.capabilities]
online_learning = true
batch_processing = true
streaming = false
multi_threading = true
gpu_acceleration = false
```

### Plugin Security

The registry provides multiple security levels:

```rust
use neuro_divergent_registry::{PluginConfig, SecurityLevel};

let config = PluginConfig {
    security_level: SecurityLevel::Signature,
    trusted_authors: vec!["TrustedDeveloper".to_string()],
    allowed_capabilities: vec![
        "model_creation".to_string(),
        "model_inference".to_string(),
    ],
    ..Default::default()
};
```

**Security Levels:**
- `None`: No security checks (development only)
- `Checksum`: SHA-256 checksum verification
- `Signature`: Digital signature verification
- `Sandboxed`: Full sandboxing with resource limits

## 🚀 Advanced Features

### Model Discovery with Filtering

```rust
use neuro_divergent_registry::{DiscoveryConfig, DiscoveryFilters, ModelCategory};

let config = DiscoveryConfig {
    discover_builtin: true,
    discover_plugins: true,
    detect_capabilities: true,
    profile_performance: true,
    parallel: true,
    filters: DiscoveryFilters {
        include_categories: Some(vec![
            ModelCategory::Transformer,
            ModelCategory::Advanced,
        ]),
        exclude_names: vec!["*deprecated*".to_string()],
        required_capabilities: vec!["gpu_acceleration".to_string()],
        min_version: Some("1.0.0".to_string()),
        ..Default::default()
    },
    ..Default::default()
};

let result = discover_all_models_with_config(config)?;
```

### Performance Benchmarking

```rust
use neuro_divergent_registry::{ModelFactory, CreationOptions};

// Enable benchmarking during creation
let options = CreationOptions {
    benchmark: true,
    validate: true,
    ..Default::default()
};

let model = ModelFactory::create_with_options::<f32>("TFT", options)?;

// Get factory statistics
let factory = ModelFactory::new();
let stats = factory.get_stats();
println!("Total models created: {}", stats.total_created);
println!("Average creation time: {:.2}ms", 
         stats.total_creation_time_ms / stats.total_created as f64);
```

### Model Versioning

```rust
use neuro_divergent_registry::{ModelInfo, ModelConfig};

// Models support versioning
let model_info = ModelFactory::get_model_info("NBEATS").unwrap();
println!("Model version: {}", model_info.version);

// Version compatibility checking
let plugin = registry.get_plugin("custom-plugin").unwrap();
if plugin.is_compatible("1.2.0") {
    println!("Plugin is compatible with runtime version 1.2.0");
}
```

### Registry State Persistence

```rust
use neuro_divergent_registry::{ModelRegistry, RegistryConfig};

// Configure registry persistence
let config = RegistryConfig {
    enable_benchmarking: true,
    max_cache_size: 200,
    enable_async: true,
    ..Default::default()
};

// Initialize with custom configuration
initialize_registry_with_config(config)?;

// Get registry statistics
let registry = global_registry();
let stats = registry.read().get_stats();
println!("Registry stats: {:?}", stats);
```

## 🔧 Integration Guide

### Basic Integration

```rust
use neuro_divergent_registry::{
    initialize_registry, ModelFactory, ModelRegistry, ModelCategory
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the global registry
    initialize_registry()?;
    
    // Create models
    let mlp = ModelFactory::create::<f32>("MLP")?;
    let lstm = ModelFactory::create::<f64>("LSTM")?;
    
    // Use the models
    let input = vec![1.0, 2.0, 3.0];
    let output = mlp.forward(&input)?;
    
    println!("MLP output: {:?}", output);
    Ok(())
}
```

### Integration with Training Systems

```rust
use neuro_divergent_registry::{ModelFactory, ModelBuilder, ModelCategory};

// Create models for training pipeline
pub fn create_training_models() -> RegistryResult<Vec<Box<dyn BaseModel<f32>>>> {
    let model_configs = vec![
        ("MLP", ModelCategory::Basic),
        ("LSTM", ModelCategory::Recurrent),
        ("TFT", ModelCategory::Transformer),
    ];
    
    let mut models = Vec::new();
    
    for (name, category) in model_configs {
        let model = ModelBuilder::new(name, category)
            .input_size(64)
            .output_size(1)
            .parameter("learning_rate", serde_json::json!(0.001))
            .benchmark(true)
            .build::<f32>()?;
        
        models.push(model);
    }
    
    Ok(models)
}
```

### Integration with Inference Systems

```rust
use neuro_divergent_registry::{ModelFactory, ModelInfo, PerformanceRequirements};

// Select optimal model for inference
pub fn select_inference_model(requirements: PerformanceRequirements) -> RegistryResult<Box<dyn BaseModel<f32>>> {
    let criteria = SearchCriteria {
        required_capabilities: vec!["batch_processing".to_string()],
        performance_requirements: Some(requirements),
        ..Default::default()
    };
    
    let registry = ModelRegistry::global();
    let candidates = registry.read().search(&criteria);
    
    if let Some(best_model) = candidates.first() {
        ModelFactory::create::<f32>(&best_model.name)
    } else {
        Err(RegistryError::ModelNotFound("No suitable model found".to_string()))
    }
}
```

### Plugin Integration

```rust
use neuro_divergent_registry::{PluginManager, PluginConfig};

pub fn setup_plugin_system() -> RegistryResult<PluginManager> {
    let config = PluginConfig {
        security_level: SecurityLevel::Checksum,
        max_plugins: 10,
        loading_timeout_ms: 5000,
        ..Default::default()
    };
    
    let manager = PluginManager::new(config);
    
    // Load plugins from directory
    for entry in std::fs::read_dir("plugins")? {
        let path = entry?.path();
        if path.extension().map_or(false, |ext| ext == "so") {
            if let Ok(plugin_name) = manager.load_plugin(&path) {
                println!("Loaded plugin: {}", plugin_name);
            }
        }
    }
    
    Ok(manager)
}
```

## 🎯 Model Categories

The registry organizes models into six categories:

### Basic Models
- **MLP**: Multi-Layer Perceptron for general-purpose learning
- **DLinear**: Direct linear transformation for simple patterns
- **NLinear**: Non-linear transformation with activation functions
- **MLPMultivariate**: Multivariate version of MLP

### Recurrent Models
- **RNN**: Basic Recurrent Neural Network
- **LSTM**: Long Short-Term Memory for long sequences
- **GRU**: Gated Recurrent Unit for efficient sequence modeling

### Advanced Models
- **NBEATS**: Neural Basis Expansion Analysis for Time Series
- **NBEATSx**: Extended version of N-BEATS
- **NHITS**: Neural Hierarchical Interpolation for Time Series
- **TiDE**: Time-series Dense Encoder

### Transformer Models
- **TFT**: Temporal Fusion Transformer
- **Informer**: Beyond Efficient Transformer for long sequences
- **AutoFormer**: Decomposition Transformers with Auto-Correlation
- **FedFormer**: Frequency Enhanced Decomposed Transformer
- **PatchTST**: Patch Time Series Transformer
- **iTransformer**: Inverted Transformer architecture

### Specialized Models
- **DeepAR**: Deep Autoregressive model for probabilistic forecasting
- **DeepNPTS**: Deep Neural Point Time Series
- **TCN**: Temporal Convolutional Network
- **BiTCN**: Bidirectional Temporal Convolutional Network
- **TimesNet**: Temporal 2D-Variation modeling
- **StemGNN**: Spectral Temporal Graph Neural Network
- **TSMixer**: Time Series Mixer architecture
- **TSMixerx**: Extended Time Series Mixer
- **TimeLLM**: Time Series Large Language Model

### Custom Models
- User-defined models loaded through the plugin system
- Support for domain-specific architectures
- Dynamic registration and management

## ⚡ Performance Considerations

### Registry Performance
- **Concurrent Access**: Fine-grained locking for high concurrency
- **Memory Efficiency**: IndexMap for O(1) access with ordered storage
- **Search Optimization**: Multiple indices for fast capability-based queries
- **Lazy Loading**: Models loaded only when needed

### Factory Performance
- **Model Caching**: Intelligent caching of frequently used models
- **Parallel Creation**: Multi-threaded model instantiation
- **Type Optimization**: Zero-cost abstractions for type safety
- **Template Reuse**: Efficient configuration templates

### Plugin Performance
- **Dynamic Loading**: Efficient shared library loading
- **Security Overhead**: Minimal impact from security checks
- **Health Monitoring**: Lightweight health checking
- **Resource Management**: Automatic cleanup and memory management

### Discovery Performance
- **Parallel Discovery**: Multi-threaded model discovery
- **Intelligent Caching**: Cache discovery results with configurable expiry
- **Filter Optimization**: Early filtering to reduce processing
- **Incremental Updates**: Only discover changes since last scan

## 🛠️ Configuration

### Registry Configuration

```rust
use neuro_divergent_registry::{RegistryConfig, DiscoveryConfig};

let config = RegistryConfig {
    enable_plugins: true,
    plugin_directories: vec![
        PathBuf::from("plugins"),
        PathBuf::from("/usr/local/lib/neuro-divergent/plugins"),
    ],
    enable_benchmarking: true,
    max_cache_size: 500,
    enable_async: true,
    discovery_config: DiscoveryConfig {
        parallel: true,
        profile_performance: true,
        cache_results: true,
        timeout_ms: 60000,
        ..Default::default()
    },
};

initialize_registry_with_config(config)?;
```

### Plugin Configuration

```rust
use neuro_divergent_registry::{PluginConfig, SecurityLevel};

let plugin_config = PluginConfig {
    security_level: SecurityLevel::Signature,
    allowed_directories: vec![
        PathBuf::from("trusted_plugins"),
        PathBuf::from("/opt/neuro-divergent/plugins"),
    ],
    blocked_directories: vec![
        PathBuf::from("/tmp"),
    ],
    max_plugins: 25,
    loading_timeout_ms: 15000,
    enable_caching: true,
    trusted_authors: vec![
        "OfficialDeveloper".to_string(),
        "TrustedPartner".to_string(),
    ],
    allowed_capabilities: vec![
        "model_creation".to_string(),
        "model_inference".to_string(),
        "model_training".to_string(),
    ],
};
```

## 📊 Statistics and Monitoring

### Registry Statistics

```rust
let registry = global_registry();
let stats = registry.read().get_stats();

println!("Registry Statistics:");
println!("  Total models: {}", stats.total_models);
println!("  Active plugins: {}", stats.active_plugins);
println!("  Total accesses: {}", stats.total_accesses);
println!("  Cache efficiency: {:.2}%", 
         (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0);

// Most accessed models
for (name, count) in &stats.most_accessed {
    println!("  {}: {} accesses", name, count);
}
```

### Factory Statistics

```rust
let factory = ModelFactory::new();
let stats = factory.get_stats();

println!("Factory Statistics:");
println!("  Models created: {}", stats.total_created);
println!("  Total creation time: {:.2}ms", stats.total_creation_time_ms);
println!("  Average creation time: {:.2}ms", 
         stats.total_creation_time_ms / stats.total_created as f64);
println!("  Cache hit ratio: {:.2}%",
         (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0);
```

### Discovery Statistics

```rust
let result = discover_all_models()?;

println!("Discovery Statistics:");
println!("  Models discovered: {}", result.stats.total_discovered);
println!("  Built-in models: {}", result.stats.builtin_models);
println!("  Plugin models: {}", result.stats.plugin_models);
println!("  Discovery time: {:.2}ms", result.duration_ms);
println!("  Plugins scanned: {}", result.stats.plugins_scanned);
println!("  Directories scanned: {}", result.stats.directories_scanned);
```

## 🔍 Error Handling

The registry provides comprehensive error handling with detailed error types:

```rust
use neuro_divergent_registry::{RegistryError, ModelError};

match ModelFactory::create::<f32>("NonExistentModel") {
    Ok(model) => {
        // Use the model
    }
    Err(RegistryError::ModelNotFound(name)) => {
        eprintln!("Model '{}' not found in registry", name);
    }
    Err(RegistryError::PluginError(msg)) => {
        eprintln!("Plugin error: {}", msg);
    }
    Err(RegistryError::InvalidConfiguration(msg)) => {
        eprintln!("Configuration error: {}", msg);
    }
    Err(other) => {
        eprintln!("Registry error: {}", other);
    }
}
```

## 🧪 Testing

The crate includes comprehensive tests for all major components:

```bash
# Run all tests
cargo test

# Run specific test modules
cargo test registry
cargo test factory
cargo test plugin
cargo test discovery

# Run with features
cargo test --features plugin-system
cargo test --features benchmarks

# Run benchmarks
cargo bench --features benchmarks
```

## 📝 Examples

See the `examples/` directory for complete examples:

- `basic_usage.rs`: Basic registry and factory usage
- `plugin_development.rs`: Creating custom plugins
- `advanced_search.rs`: Advanced model search and filtering
- `performance_monitoring.rs`: Performance tracking and optimization
- `batch_operations.rs`: Batch model creation and management

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/your-org/neuro-divergent-registry
cd neuro-divergent-registry
cargo build
cargo test
```

### Adding New Models

1. Create model implementation in appropriate category
2. Add model info to discovery system
3. Register creator function in factory
4. Add comprehensive tests
5. Update documentation

### Plugin Development

1. Implement `PluginInterface` trait
2. Create plugin descriptor file
3. Handle security requirements
4. Add health monitoring
5. Test with registry system

## 📄 License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🔗 Related Crates

- `neuro-divergent-core`: Core neural network implementations
- `neuro-divergent-models`: Specific model implementations
- `neuro-divergent-training`: Training algorithms and utilities
- `neuro-divergent-inference`: Optimized inference engines

---

*Built with ❤️ for the neural forecasting community*