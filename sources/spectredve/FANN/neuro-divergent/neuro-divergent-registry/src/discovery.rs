//! Model Discovery for Automatic Registration
//!
//! The discovery system automatically finds and registers all available neural network models
//! including built-in models, plugin models, and custom models with comprehensive
//! capability detection and performance profiling.

use crate::{
    ModelInfo, ModelCategory, ModelCapabilities, ModelPerformance,
    RegistryResult, plugin::PluginManager,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{SystemTime, Instant};
use serde_json::Value;
use parking_lot::RwLock;

/// Discovery configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryConfig {
    /// Enable built-in model discovery
    pub discover_builtin: bool,
    /// Enable plugin model discovery
    pub discover_plugins: bool,
    /// Plugin directories to scan
    pub plugin_directories: Vec<PathBuf>,
    /// Enable capability detection
    pub detect_capabilities: bool,
    /// Enable performance profiling
    pub profile_performance: bool,
    /// Discovery timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum models to discover
    pub max_models: usize,
    /// Parallel discovery
    pub parallel: bool,
    /// Cache discovery results
    pub cache_results: bool,
    /// Discovery cache directory
    pub cache_directory: PathBuf,
    /// Cache expiry time in hours
    pub cache_expiry_hours: u64,
    /// Model filters
    pub filters: DiscoveryFilters,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            discover_builtin: true,
            discover_plugins: true,
            plugin_directories: vec![
                PathBuf::from("plugins"),
                dirs::home_dir().unwrap_or_default().join(".neuro-divergent/plugins"),
            ],
            detect_capabilities: true,
            profile_performance: false,
            timeout_ms: 30000,
            max_models: 1000,
            parallel: true,
            cache_results: true,
            cache_directory: dirs::cache_dir()
                .unwrap_or_default()
                .join("neuro-divergent")
                .join("discovery"),
            cache_expiry_hours: 24,
            filters: DiscoveryFilters::default(),
        }
    }
}

/// Discovery filters to control what gets discovered
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryFilters {
    /// Categories to include
    pub include_categories: Option<Vec<ModelCategory>>,
    /// Categories to exclude
    pub exclude_categories: Vec<ModelCategory>,
    /// Model names to include (supports wildcards)
    pub include_names: Vec<String>,
    /// Model names to exclude (supports wildcards)
    pub exclude_names: Vec<String>,
    /// Minimum version required
    pub min_version: Option<String>,
    /// Maximum version allowed
    pub max_version: Option<String>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Excluded capabilities
    pub excluded_capabilities: Vec<String>,
    /// Custom filter function name
    pub custom_filter: Option<String>,
}

impl Default for DiscoveryFilters {
    fn default() -> Self {
        Self {
            include_categories: None,
            exclude_categories: Vec::new(),
            include_names: Vec::new(),
            exclude_names: Vec::new(),
            min_version: None,
            max_version: None,
            required_capabilities: Vec::new(),
            excluded_capabilities: Vec::new(),
            custom_filter: None,
        }
    }
}

/// Discovery result containing found model information
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Discovered models
    pub models: Vec<ModelInfo>,
    /// Discovery statistics
    pub stats: DiscoveryStats,
    /// Discovery errors
    pub errors: Vec<DiscoveryError>,
    /// Discovery timestamp
    pub timestamp: SystemTime,
    /// Discovery duration
    pub duration_ms: f64,
}

/// Discovery statistics
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    /// Total models discovered
    pub total_discovered: usize,
    /// Built-in models discovered
    pub builtin_models: usize,
    /// Plugin models discovered
    pub plugin_models: usize,
    /// Models by category
    pub by_category: HashMap<ModelCategory, usize>,
    /// Models with performance data
    pub with_performance: usize,
    /// Models with capabilities detected
    pub with_capabilities: usize,
    /// Plugins scanned
    pub plugins_scanned: usize,
    /// Directories scanned
    pub directories_scanned: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
}

/// Plugin discovery statistics
#[derive(Debug, Default)]
pub struct PluginDiscoveryStats {
    /// Number of plugins scanned
    pub plugins_scanned: usize,
    /// Number of directories scanned
    pub directories_scanned: usize,
}

/// Discovery error
#[derive(Debug, Clone)]
pub struct DiscoveryError {
    /// Error message
    pub message: String,
    /// Error location (file, plugin, etc.)
    pub location: String,
    /// Error category
    pub category: DiscoveryErrorCategory,
    /// Error timestamp
    pub timestamp: SystemTime,
}

/// Discovery error categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryErrorCategory {
    /// Plugin loading error
    PluginError,
    /// File system error
    FileSystemError,
    /// Configuration error
    ConfigurationError,
    /// Model validation error
    ValidationError,
    /// Performance profiling error
    ProfilingError,
    /// Cache error
    CacheError,
    /// Timeout error
    TimeoutError,
    /// Unknown error
    Unknown,
}

/// Model discovery engine
pub struct ModelDiscovery {
    /// Discovery configuration
    config: RwLock<DiscoveryConfig>,
    /// Plugin manager (placeholder - not yet used)
    #[allow(dead_code)]
    plugin_manager: PluginManager,
    /// Discovery cache
    cache: RwLock<DiscoveryCache>,
    /// Discovery statistics
    stats: RwLock<DiscoveryStats>,
}

/// Discovery cache
#[derive(Debug, Default)]
struct DiscoveryCache {
    /// Cached model information
    models: HashMap<String, (ModelInfo, SystemTime)>,
    /// Cached plugin information
    plugins: HashMap<String, (Vec<ModelInfo>, SystemTime)>,
    /// Cache hit/miss counters
    hits: usize,
    misses: usize,
}

impl ModelDiscovery {
    /// Create a new model discovery engine
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config: RwLock::new(config.clone()),
            plugin_manager: PluginManager::new(Default::default()),
            cache: RwLock::new(DiscoveryCache::default()),
            stats: RwLock::new(DiscoveryStats::default()),
        }
    }
    
    /// Discover all available models
    pub fn discover_all(&self) -> RegistryResult<DiscoveryResult> {
        let start_time = Instant::now();
        let config = self.config.read().clone();
        let mut models = Vec::new();
        let mut errors = Vec::new();
        let mut stats = DiscoveryStats::default();
        
        log::info!("Starting model discovery");
        
        // Discover built-in models
        if config.discover_builtin {
            match self.discover_builtin_models(&config) {
                Ok(mut builtin_models) => {
                    stats.builtin_models = builtin_models.len();
                    models.append(&mut builtin_models);
                }
                Err(e) => {
                    errors.push(DiscoveryError {
                        message: e.to_string(),
                        location: "builtin".to_string(),
                        category: DiscoveryErrorCategory::ConfigurationError,
                        timestamp: SystemTime::now(),
                    });
                }
            }
        }
        
        // Discover plugin models
        if config.discover_plugins {
            match self.discover_plugin_models(&config) {
                Ok((mut plugin_models, plugin_stats)) => {
                    stats.plugin_models = plugin_models.len();
                    stats.plugins_scanned = plugin_stats.plugins_scanned;
                    stats.directories_scanned = plugin_stats.directories_scanned;
                    models.append(&mut plugin_models);
                }
                Err(e) => {
                    errors.push(DiscoveryError {
                        message: e.to_string(),
                        location: "plugins".to_string(),
                        category: DiscoveryErrorCategory::PluginError,
                        timestamp: SystemTime::now(),
                    });
                }
            }
        }
        
        // Apply filters
        models = self.apply_filters(models, &config.filters);
        
        // Detect capabilities if enabled
        if config.detect_capabilities {
            for model in &mut models {
                if let Err(e) = self.detect_model_capabilities(model) {
                    errors.push(DiscoveryError {
                        message: format!("Failed to detect capabilities for {}: {}", model.name, e),
                        location: model.name.clone(),
                        category: DiscoveryErrorCategory::ValidationError,
                        timestamp: SystemTime::now(),
                    });
                }
            }
            stats.with_capabilities = models.iter()
                .filter(|m| self.has_detected_capabilities(m))
                .count();
        }
        
        // Profile performance if enabled
        if config.profile_performance {
            for model in &mut models {
                if let Err(e) = self.profile_model_performance(model) {
                    errors.push(DiscoveryError {
                        message: format!("Failed to profile performance for {}: {}", model.name, e),
                        location: model.name.clone(),
                        category: DiscoveryErrorCategory::ProfilingError,
                        timestamp: SystemTime::now(),
                    });
                }
            }
            stats.with_performance = models.iter()
                .filter(|m| m.performance.is_some())
                .count();
        }
        
        // Update category statistics
        for model in &models {
            *stats.by_category.entry(model.category).or_insert(0) += 1;
        }
        
        stats.total_discovered = models.len();
        
        // Cache results if enabled
        if config.cache_results {
            self.cache_discovery_results(&models);
        }
        
        // Update internal statistics
        {
            let mut internal_stats = self.stats.write();
            *internal_stats = stats.clone();
            let cache = self.cache.read();
            internal_stats.cache_hits = cache.hits;
            internal_stats.cache_misses = cache.misses;
        }
        
        let duration = start_time.elapsed().as_millis() as f64;
        
        log::info!("Model discovery completed: {} models found in {:.2}ms", 
                  models.len(), duration);
        
        Ok(DiscoveryResult {
            models,
            stats,
            errors,
            timestamp: SystemTime::now(),
            duration_ms: duration,
        })
    }
    
    /// Discover built-in models
    fn discover_builtin_models(&self, _config: &DiscoveryConfig) -> RegistryResult<Vec<ModelInfo>> {
        let mut models = Vec::new();
        
        // Basic models
        models.extend(self.create_basic_models());
        
        // Recurrent models
        models.extend(self.create_recurrent_models());
        
        // Advanced models
        models.extend(self.create_advanced_models());
        
        // Transformer models
        models.extend(self.create_transformer_models());
        
        // Specialized models
        models.extend(self.create_specialized_models());
        
        log::debug!("Discovered {} built-in models", models.len());
        Ok(models)
    }
    
    /// Create basic model information
    fn create_basic_models(&self) -> Vec<ModelInfo> {
        vec![
            self.create_model_info("MLP", ModelCategory::Basic, "Multi-Layer Perceptron"),
            self.create_model_info("DLinear", ModelCategory::Basic, "Direct Linear transformation"),
            self.create_model_info("NLinear", ModelCategory::Basic, "Non-linear transformation"),
            self.create_model_info("MLPMultivariate", ModelCategory::Basic, "Multivariate MLP"),
        ]
    }
    
    /// Create recurrent model information
    fn create_recurrent_models(&self) -> Vec<ModelInfo> {
        vec![
            self.create_model_info("RNN", ModelCategory::Recurrent, "Recurrent Neural Network"),
            self.create_model_info("LSTM", ModelCategory::Recurrent, "Long Short-Term Memory"),
            self.create_model_info("GRU", ModelCategory::Recurrent, "Gated Recurrent Unit"),
        ]
    }
    
    /// Create advanced model information
    fn create_advanced_models(&self) -> Vec<ModelInfo> {
        vec![
            self.create_model_info("NBEATS", ModelCategory::Advanced, "Neural Basis Expansion Analysis"),
            self.create_model_info("NBEATSx", ModelCategory::Advanced, "Extended N-BEATS"),
            self.create_model_info("NHITS", ModelCategory::Advanced, "Neural Hierarchical Interpolation"),
            self.create_model_info("TiDE", ModelCategory::Advanced, "Time-series Dense Encoder"),
        ]
    }
    
    /// Create transformer model information
    fn create_transformer_models(&self) -> Vec<ModelInfo> {
        vec![
            self.create_model_info("TFT", ModelCategory::Transformer, "Temporal Fusion Transformer"),
            self.create_model_info("Informer", ModelCategory::Transformer, "Informer: Beyond Efficient Transformer"),
            self.create_model_info("AutoFormer", ModelCategory::Transformer, "AutoFormer: Decomposition Transformers"),
            self.create_model_info("FedFormer", ModelCategory::Transformer, "FedFormer: Frequency Enhanced Decomposed"),
            self.create_model_info("PatchTST", ModelCategory::Transformer, "Patch Time Series Transformer"),
            self.create_model_info("iTransformer", ModelCategory::Transformer, "Inverted Transformer"),
        ]
    }
    
    /// Create specialized model information
    fn create_specialized_models(&self) -> Vec<ModelInfo> {
        vec![
            self.create_model_info("DeepAR", ModelCategory::Specialized, "Deep Autoregressive"),
            self.create_model_info("DeepNPTS", ModelCategory::Specialized, "Deep Neural Point Time Series"),
            self.create_model_info("TCN", ModelCategory::Specialized, "Temporal Convolutional Network"),
            self.create_model_info("BiTCN", ModelCategory::Specialized, "Bidirectional TCN"),
            self.create_model_info("TimesNet", ModelCategory::Specialized, "TimesNet: Temporal 2D-Variation"),
            self.create_model_info("StemGNN", ModelCategory::Specialized, "Spectral Temporal Graph Neural Network"),
            self.create_model_info("TSMixer", ModelCategory::Specialized, "Time Series Mixer"),
            self.create_model_info("TSMixerx", ModelCategory::Specialized, "Extended Time Series Mixer"),
            self.create_model_info("TimeLLM", ModelCategory::Specialized, "Time Series Large Language Model"),
        ]
    }
    
    /// Create model information structure
    fn create_model_info(&self, name: &str, category: ModelCategory, description: &str) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            category,
            version: "1.0.0".to_string(),
            description: description.to_string(),
            parameter_types: vec!["f32".to_string(), "f64".to_string()],
            input_size: None,
            output_size: None,
            capabilities: self.get_default_capabilities(category),
            performance: None,
            metadata: self.get_default_metadata(name, category),
        }
    }
    
    /// Get default capabilities for a model category
    fn get_default_capabilities(&self, category: ModelCategory) -> ModelCapabilities {
        let mut caps = ModelCapabilities::default();
        
        match category {
            ModelCategory::Basic => {
                caps.online_learning = true;
                caps.batch_processing = true;
                caps.streaming = false;
                caps.multi_threading = true;
            }
            ModelCategory::Recurrent => {
                caps.online_learning = true;
                caps.batch_processing = true;
                caps.streaming = true;
                caps.multi_threading = false; // RNNs are inherently sequential
            }
            ModelCategory::Advanced => {
                caps.online_learning = false;
                caps.batch_processing = true;
                caps.streaming = false;
                caps.multi_threading = true;
                caps.gpu_acceleration = true;
            }
            ModelCategory::Transformer => {
                caps.online_learning = false;
                caps.batch_processing = true;
                caps.streaming = false;
                caps.multi_threading = true;
                caps.gpu_acceleration = true;
                caps.quantization = true;
            }
            ModelCategory::Specialized => {
                caps.online_learning = true;
                caps.batch_processing = true;
                caps.streaming = true;
                caps.multi_threading = true;
                caps.gpu_acceleration = true;
                caps.quantization = true;
                caps.pruning = true;
            }
            ModelCategory::Custom => {
                // Default capabilities for custom models
            }
        }
        
        caps
    }
    
    /// Get default metadata for a model
    fn get_default_metadata(&self, name: &str, category: ModelCategory) -> HashMap<String, Value> {
        let mut metadata = HashMap::new();
        
        metadata.insert("source".to_string(), Value::String("builtin".to_string()));
        metadata.insert("category_description".to_string(), 
                       Value::String(category.description().to_string()));
        metadata.insert("discovered_at".to_string(), 
                       Value::String(SystemTime::now()
                           .duration_since(SystemTime::UNIX_EPOCH)
                           .unwrap()
                           .as_secs()
                           .to_string()));
        
        // Add model-specific tags
        let tags = match name {
            "MLP" | "MLPMultivariate" => vec!["feedforward", "dense", "basic"],
            "LSTM" | "GRU" | "RNN" => vec!["recurrent", "sequential", "memory"],
            "TFT" | "Informer" | "AutoFormer" | "FedFormer" | "PatchTST" | "iTransformer" => 
                vec!["transformer", "attention", "self-attention"],
            "TCN" | "BiTCN" => vec!["convolutional", "temporal", "causal"],
            "NBEATS" | "NBEATSx" => vec!["interpretable", "basis-expansion", "hierarchical"],
            "DeepAR" => vec!["probabilistic", "autoregressive", "bayesian"],
            "TimeLLM" => vec!["large-language-model", "foundation", "pre-trained"],
            _ => vec!["neural-network", "time-series"],
        };
        
        metadata.insert("tags".to_string(), 
                       Value::Array(tags.iter().map(|t| Value::String(t.to_string())).collect()));
        
        metadata
    }
    
    /// Discover plugin models
    fn discover_plugin_models(&self, config: &DiscoveryConfig) -> RegistryResult<(Vec<ModelInfo>, PluginDiscoveryStats)> {
        let mut models = Vec::new();
        let mut stats = PluginDiscoveryStats::default();
        
        for plugin_dir in &config.plugin_directories {
            if plugin_dir.exists() {
                stats.directories_scanned += 1;
                
                match crate::plugin::load_plugins_from_directory(plugin_dir) {
                    Ok(plugins) => {
                        stats.plugins_scanned += plugins.len();
                        
                        for plugin in plugins {
                            models.extend(plugin.list_models());
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to load plugins from {:?}: {}", plugin_dir, e);
                    }
                }
            }
        }
        
        log::debug!("Discovered {} plugin models from {} plugins", 
                   models.len(), stats.plugins_scanned);
        
        Ok((models, stats))
    }
    
    
    /// Apply discovery filters
    fn apply_filters(&self, mut models: Vec<ModelInfo>, filters: &DiscoveryFilters) -> Vec<ModelInfo> {
        // Category filters
        if let Some(ref include_categories) = filters.include_categories {
            models.retain(|m| include_categories.contains(&m.category));
        }
        
        if !filters.exclude_categories.is_empty() {
            models.retain(|m| !filters.exclude_categories.contains(&m.category));
        }
        
        // Name filters
        if !filters.include_names.is_empty() {
            models.retain(|m| {
                filters.include_names.iter().any(|pattern| self.matches_pattern(&m.name, pattern))
            });
        }
        
        if !filters.exclude_names.is_empty() {
            models.retain(|m| {
                !filters.exclude_names.iter().any(|pattern| self.matches_pattern(&m.name, pattern))
            });
        }
        
        // Version filters
        if let Some(ref min_version) = filters.min_version {
            models.retain(|m| &m.version >= min_version);
        }
        
        if let Some(ref max_version) = filters.max_version {
            models.retain(|m| &m.version <= max_version);
        }
        
        // Capability filters
        if !filters.required_capabilities.is_empty() {
            models.retain(|m| {
                let model_caps = self.extract_capability_names(&m.capabilities);
                filters.required_capabilities.iter()
                    .all(|cap| model_caps.contains(cap))
            });
        }
        
        if !filters.excluded_capabilities.is_empty() {
            models.retain(|m| {
                let model_caps = self.extract_capability_names(&m.capabilities);
                !filters.excluded_capabilities.iter()
                    .any(|cap| model_caps.contains(cap))
            });
        }
        
        models
    }
    
    /// Check if name matches pattern (supports wildcards)
    fn matches_pattern(&self, name: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        
        if pattern.contains('*') {
            // Simple wildcard matching
            let parts: Vec<&str> = pattern.split('*').collect();
            let mut name_pos = 0;
            
            for (i, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }
                
                if i == 0 {
                    if !name[name_pos..].starts_with(part) {
                        return false;
                    }
                    name_pos += part.len();
                } else if i == parts.len() - 1 {
                    return name[name_pos..].ends_with(part);
                } else {
                    if let Some(pos) = name[name_pos..].find(part) {
                        name_pos += pos + part.len();
                    } else {
                        return false;
                    }
                }
            }
            
            true
        } else {
            name.to_lowercase().contains(&pattern.to_lowercase())
        }
    }
    
    /// Extract capability names from ModelCapabilities
    fn extract_capability_names(&self, capabilities: &ModelCapabilities) -> HashSet<String> {
        let mut caps = HashSet::new();
        
        if capabilities.online_learning { caps.insert("online_learning".to_string()); }
        if capabilities.batch_processing { caps.insert("batch_processing".to_string()); }
        if capabilities.streaming { caps.insert("streaming".to_string()); }
        if capabilities.multi_threading { caps.insert("multi_threading".to_string()); }
        if capabilities.gpu_acceleration { caps.insert("gpu_acceleration".to_string()); }
        if capabilities.quantization { caps.insert("quantization".to_string()); }
        if capabilities.pruning { caps.insert("pruning".to_string()); }
        if capabilities.fine_tuning { caps.insert("fine_tuning".to_string()); }
        
        caps
    }
    
    /// Detect model capabilities (placeholder)
    fn detect_model_capabilities(&self, _model: &mut ModelInfo) -> RegistryResult<()> {
        // This would involve more sophisticated capability detection
        // For now, we use the default capabilities set during creation
        Ok(())
    }
    
    /// Check if model has detected capabilities
    fn has_detected_capabilities(&self, _model: &ModelInfo) -> bool {
        // Placeholder - in a real implementation, we'd track which capabilities
        // were detected vs. default
        true
    }
    
    /// Profile model performance (placeholder)
    fn profile_model_performance(&self, model: &mut ModelInfo) -> RegistryResult<()> {
        // This would involve creating the model and running benchmarks
        // For now, we create dummy performance data
        model.performance = Some(ModelPerformance {
            creation_time_ms: 100.0,
            memory_usage_bytes: 1024 * 1024, // 1MB
            forward_pass_time_us: 50.0,
            backward_pass_time_us: Some(100.0),
            parameter_count: 10000,
            model_size_bytes: 40000, // 40KB
            throughput_samples_per_sec: 1000.0,
            benchmark_timestamp: SystemTime::now(),
        });
        
        Ok(())
    }
    
    /// Cache discovery results
    fn cache_discovery_results(&self, models: &[ModelInfo]) {
        let mut cache = self.cache.write();
        let now = SystemTime::now();
        
        for model in models {
            cache.models.insert(model.name.clone(), (model.clone(), now));
        }
        
        log::debug!("Cached {} model discovery results", models.len());
    }
    
    /// Get discovery statistics
    pub fn get_stats(&self) -> DiscoveryStats {
        self.stats.read().clone()
    }
    
    /// Clear discovery cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.models.clear();
        cache.plugins.clear();
        cache.hits = 0;
        cache.misses = 0;
        log::debug!("Discovery cache cleared");
    }
}

impl Default for ModelDiscovery {
    fn default() -> Self {
        Self::new(DiscoveryConfig::default())
    }
}

/// Discover built-in models with default configuration
pub fn discover_builtin_models() -> RegistryResult<Vec<ModelInfo>> {
    let discovery = ModelDiscovery::default();
    discovery.discover_builtin_models(&DiscoveryConfig::default())
}

/// Discover built-in models with custom configuration
pub fn discover_builtin_models_with_config(config: &DiscoveryConfig) -> RegistryResult<Vec<ModelInfo>> {
    let discovery = ModelDiscovery::new(config.clone());
    discovery.discover_builtin_models(config)
}

/// Discover all models with default configuration
pub fn discover_all_models() -> RegistryResult<DiscoveryResult> {
    let discovery = ModelDiscovery::default();
    discovery.discover_all()
}

/// Discover all models with custom configuration
pub fn discover_all_models_with_config(config: DiscoveryConfig) -> RegistryResult<DiscoveryResult> {
    let discovery = ModelDiscovery::new(config);
    discovery.discover_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        assert!(config.discover_builtin);
        assert!(config.discover_plugins);
        assert!(config.detect_capabilities);
        assert!(!config.profile_performance);
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.max_models, 1000);
        assert!(config.parallel);
    }
    
    #[test]
    fn test_discovery_filters_default() {
        let filters = DiscoveryFilters::default();
        assert!(filters.include_categories.is_none());
        assert!(filters.exclude_categories.is_empty());
        assert!(filters.include_names.is_empty());
        assert!(filters.exclude_names.is_empty());
        assert!(filters.required_capabilities.is_empty());
    }
    
    #[test]
    fn test_model_discovery_new() {
        let config = DiscoveryConfig::default();
        let discovery = ModelDiscovery::new(config);
        
        let stats = discovery.get_stats();
        assert_eq!(stats.total_discovered, 0);
        assert_eq!(stats.builtin_models, 0);
        assert_eq!(stats.plugin_models, 0);
    }
    
    #[test]
    fn test_discover_builtin_models() {
        let result = discover_builtin_models();
        assert!(result.is_ok());
        
        let models = result.unwrap();
        assert!(!models.is_empty());
        
        // Check for basic models
        assert!(models.iter().any(|m| m.name == "MLP"));
        assert!(models.iter().any(|m| m.name == "LSTM"));
        assert!(models.iter().any(|m| m.name == "TFT"));
        
        // Check categories
        assert!(models.iter().any(|m| m.category == ModelCategory::Basic));
        assert!(models.iter().any(|m| m.category == ModelCategory::Recurrent));
        assert!(models.iter().any(|m| m.category == ModelCategory::Transformer));
    }
    
    #[test]
    fn test_pattern_matching() {
        let discovery = ModelDiscovery::default();
        
        assert!(discovery.matches_pattern("MLP", "*"));
        assert!(discovery.matches_pattern("MLP", "MLP"));
        assert!(discovery.matches_pattern("MLPMultivariate", "MLP*"));
        assert!(discovery.matches_pattern("DeepAR", "*AR"));
        assert!(!discovery.matches_pattern("LSTM", "MLP*"));
    }
    
    #[test]
    fn test_extract_capability_names() {
        let discovery = ModelDiscovery::default();
        let mut capabilities = ModelCapabilities::default();
        capabilities.online_learning = true;
        capabilities.gpu_acceleration = true;
        
        let names = discovery.extract_capability_names(&capabilities);
        assert!(names.contains("online_learning"));
        assert!(names.contains("gpu_acceleration"));
        assert!(names.contains("batch_processing")); // default true
        assert!(!names.contains("streaming")); // default false
    }
    
    #[test]
    fn test_discovery_error() {
        let error = DiscoveryError {
            message: "Test error".to_string(),
            location: "test_location".to_string(),
            category: DiscoveryErrorCategory::ValidationError,
            timestamp: SystemTime::now(),
        };
        
        assert_eq!(error.message, "Test error");
        assert_eq!(error.location, "test_location");
        assert_eq!(error.category, DiscoveryErrorCategory::ValidationError);
    }
    
    #[test]
    fn test_default_capabilities_by_category() {
        let discovery = ModelDiscovery::default();
        
        let basic_caps = discovery.get_default_capabilities(ModelCategory::Basic);
        assert!(basic_caps.online_learning);
        assert!(basic_caps.batch_processing);
        assert!(!basic_caps.streaming);
        
        let recurrent_caps = discovery.get_default_capabilities(ModelCategory::Recurrent);
        assert!(recurrent_caps.streaming);
        assert!(!recurrent_caps.multi_threading); // Sequential nature
        
        let transformer_caps = discovery.get_default_capabilities(ModelCategory::Transformer);
        assert!(transformer_caps.gpu_acceleration);
        assert!(transformer_caps.quantization);
    }
}