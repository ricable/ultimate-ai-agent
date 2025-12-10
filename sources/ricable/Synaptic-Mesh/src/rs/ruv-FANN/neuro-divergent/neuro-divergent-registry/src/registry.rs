//! Model Registry for Global Model Management
//!
//! The ModelRegistry provides centralized management of all available neural network models.
//! It supports registration, discovery, categorization, and efficient lookup of models
//! with thread-safe operations and comprehensive metadata management.

use crate::{
    BaseModel, Float, ModelConfig, ModelInfo, ModelCategory, ModelCapabilities, 
    ModelPerformance, RegistryError, RegistryResult, RegistryConfig,
    plugin::{Plugin, PluginDescriptor},
};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use indexmap::IndexMap;
use serde_json::Value;
use ahash::AHashMap;

/// Model descriptor containing creation information
pub struct ModelDescriptor {
    /// Model information
    pub info: ModelInfo,
    /// Model creation function for f32
    pub creator_f32: Option<Arc<dyn Fn(&ModelConfig) -> RegistryResult<Box<dyn BaseModel<f32>>> + Send + Sync>>,
    /// Model creation function for f64
    pub creator_f64: Option<Arc<dyn Fn(&ModelConfig) -> RegistryResult<Box<dyn BaseModel<f64>>> + Send + Sync>>,
    /// Model plugin (if from plugin)
    pub plugin: Option<PluginDescriptor>,
    /// Registration timestamp
    pub registered_at: std::time::SystemTime,
    /// Last accessed timestamp
    pub last_accessed: RwLock<std::time::SystemTime>,
    /// Access count
    pub access_count: RwLock<u64>,
    /// Model tags for search
    pub tags: Vec<String>,
    /// Model aliases
    pub aliases: Vec<String>,
}

impl std::fmt::Debug for ModelDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelDescriptor")
            .field("info", &self.info)
            .field("creator_f32", &self.creator_f32.as_ref().map(|_| "<closure>"))
            .field("creator_f64", &self.creator_f64.as_ref().map(|_| "<closure>"))
            .field("plugin", &self.plugin)
            .field("registered_at", &self.registered_at)
            .field("last_accessed", &self.last_accessed)
            .field("access_count", &self.access_count)
            .field("tags", &self.tags)
            .field("aliases", &self.aliases)
            .finish()
    }
}

impl Clone for ModelDescriptor {
    fn clone(&self) -> Self {
        Self {
            info: self.info.clone(),
            creator_f32: self.creator_f32.clone(),
            creator_f64: self.creator_f64.clone(),
            plugin: self.plugin.clone(),
            registered_at: self.registered_at,
            last_accessed: RwLock::new(*self.last_accessed.read()),
            access_count: RwLock::new(*self.access_count.read()),
            tags: self.tags.clone(),
            aliases: self.aliases.clone(),
        }
    }
}

/// Registry search criteria
#[derive(Debug, Clone, Default)]
pub struct SearchCriteria {
    /// Model name pattern (supports wildcards)
    pub name_pattern: Option<String>,
    /// Model category filter
    pub category: Option<ModelCategory>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Tag filters
    pub tags: Vec<String>,
    /// Parameter type filter
    pub parameter_type: Option<String>,
    /// Performance requirements
    pub performance_requirements: Option<PerformanceRequirements>,
    /// Plugin filter
    pub from_plugin: Option<String>,
}

/// Performance requirements for model search
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum creation time in milliseconds
    pub max_creation_time_ms: Option<f64>,
    /// Maximum memory usage in bytes
    pub max_memory_usage_bytes: Option<usize>,
    /// Minimum throughput (samples per second)
    pub min_throughput_samples_per_sec: Option<f64>,
    /// Maximum model size in bytes
    pub max_model_size_bytes: Option<usize>,
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total registered models
    pub total_models: usize,
    /// Models by category
    pub by_category: HashMap<ModelCategory, usize>,
    /// Models by parameter type
    pub by_parameter_type: HashMap<String, usize>,
    /// Total plugins loaded
    pub total_plugins: usize,
    /// Active plugins
    pub active_plugins: usize,
    /// Total model accesses
    pub total_accesses: u64,
    /// Most accessed models
    pub most_accessed: Vec<(String, u64)>,
    /// Registry size in bytes
    pub registry_size_bytes: usize,
    /// Average model creation time
    pub average_creation_time_ms: f64,
    /// Last updated timestamp
    pub last_updated: std::time::SystemTime,
}

impl Default for RegistryStats {
    fn default() -> Self {
        Self {
            total_models: 0,
            by_category: HashMap::new(),
            by_parameter_type: HashMap::new(),
            total_plugins: 0,
            active_plugins: 0,
            total_accesses: 0,
            most_accessed: Vec::new(),
            registry_size_bytes: 0,
            average_creation_time_ms: 0.0,
            last_updated: std::time::SystemTime::UNIX_EPOCH,
        }
    }
}

/// Model Registry for centralized model management
pub struct ModelRegistry {
    /// Registered models by name
    models: RwLock<IndexMap<String, ModelDescriptor>>,
    /// Models by category
    by_category: RwLock<AHashMap<ModelCategory, Vec<String>>>,
    /// Models by capability
    by_capability: RwLock<AHashMap<String, Vec<String>>>,
    /// Model aliases mapping
    aliases: RwLock<AHashMap<String, String>>,
    /// Model tags mapping
    tags: RwLock<AHashMap<String, Vec<String>>>,
    /// Loaded plugins
    plugins: RwLock<AHashMap<String, Plugin>>,
    /// Registry configuration
    config: RwLock<Option<RegistryConfig>>,
    /// Registry statistics
    stats: RwLock<RegistryStats>,
    /// Search index for fast lookups
    search_index: RwLock<AHashMap<String, Vec<String>>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(IndexMap::new()),
            by_category: RwLock::new(AHashMap::new()),
            by_capability: RwLock::new(AHashMap::new()),
            aliases: RwLock::new(AHashMap::new()),
            tags: RwLock::new(AHashMap::new()),
            plugins: RwLock::new(AHashMap::new()),
            config: RwLock::new(None),
            stats: RwLock::new(RegistryStats::default()),
            search_index: RwLock::new(AHashMap::new()),
        }
    }
    
    /// Register a model with creation functions
    pub fn register<T: Float + 'static>(
        &self,
        name: &str,
        descriptor: ModelDescriptor,
    ) -> RegistryResult<()> {
        let name = name.to_string();
        
        // Check if model already exists
        if self.models.read().contains_key(&name) {
            return Err(RegistryError::ModelAlreadyExists(name));
        }
        
        // Register model
        {
            let mut models = self.models.write();
            models.insert(name.clone(), descriptor.clone());
        }
        
        // Update category index
        {
            let mut by_category = self.by_category.write();
            by_category
                .entry(descriptor.info.category)
                .or_insert_with(Vec::new)
                .push(name.clone());
        }
        
        // Update capability index
        {
            let mut by_capability = self.by_capability.write();
            for capability in self.extract_capabilities(&descriptor.info.capabilities) {
                by_capability
                    .entry(capability)
                    .or_insert_with(Vec::new)
                    .push(name.clone());
            }
        }
        
        // Register aliases
        {
            let mut aliases = self.aliases.write();
            for alias in &descriptor.aliases {
                aliases.insert(alias.clone(), name.clone());
            }
        }
        
        // Register tags
        {
            let mut tags = self.tags.write();
            for tag in &descriptor.tags {
                tags.entry(tag.clone())
                    .or_insert_with(Vec::new)
                    .push(name.clone());
            }
        }
        
        // Update search index
        self.update_search_index(&name, &descriptor);
        
        // Update statistics
        self.update_stats();
        
        log::info!("Registered model '{}' in category {:?}", name, descriptor.info.category);
        Ok(())
    }
    
    /// Register model information only (without creators)
    pub fn register_info(&self, info: ModelInfo) -> RegistryResult<()> {
        let descriptor = ModelDescriptor {
            info: info.clone(),
            creator_f32: None,
            creator_f64: None,
            plugin: None,
            registered_at: std::time::SystemTime::now(),
            last_accessed: RwLock::new(std::time::SystemTime::now()),
            access_count: RwLock::new(0),
            tags: self.extract_tags_from_metadata(&info.metadata),
            aliases: Vec::new(),
        };
        
        self.register::<f32>(&info.name, descriptor)
    }
    
    /// Unregister a model
    pub fn unregister(&self, name: &str) -> RegistryResult<()> {
        let descriptor = {
            let mut models = self.models.write();
            models.shift_remove(name)
                .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))?
        };
        
        // Remove from category index
        {
            let mut by_category = self.by_category.write();
            if let Some(models) = by_category.get_mut(&descriptor.info.category) {
                models.retain(|n| n != name);
                if models.is_empty() {
                    by_category.remove(&descriptor.info.category);
                }
            }
        }
        
        // Remove from capability index
        {
            let mut by_capability = self.by_capability.write();
            for capability in self.extract_capabilities(&descriptor.info.capabilities) {
                if let Some(models) = by_capability.get_mut(&capability) {
                    models.retain(|n| n != name);
                    if models.is_empty() {
                        by_capability.remove(&capability);
                    }
                }
            }
        }
        
        // Remove aliases
        {
            let mut aliases = self.aliases.write();
            for alias in &descriptor.aliases {
                aliases.remove(alias);
            }
        }
        
        // Remove from tags
        {
            let mut tags = self.tags.write();
            for tag in &descriptor.tags {
                if let Some(models) = tags.get_mut(tag) {
                    models.retain(|n| n != name);
                    if models.is_empty() {
                        tags.remove(tag);
                    }
                }
            }
        }
        
        // Remove from search index
        {
            let mut search_index = self.search_index.write();
            search_index.retain(|_, models| {
                models.retain(|n| n != name);
                !models.is_empty()
            });
        }
        
        // Update statistics
        self.update_stats();
        
        log::info!("Unregistered model '{}'", name);
        Ok(())
    }
    
    /// Check if a model exists
    pub fn has_model(&self, name: &str) -> bool {
        self.models.read().contains_key(name) || self.aliases.read().contains_key(name)
    }
    
    /// Get model information
    pub fn get_model_info(&self, name: &str) -> Option<ModelInfo> {
        let models = self.models.read();
        
        // Try direct lookup first
        if let Some(descriptor) = models.get(name) {
            self.update_access_stats(name);
            return Some(descriptor.info.clone());
        }
        
        // Try alias lookup
        let aliases = self.aliases.read();
        if let Some(real_name) = aliases.get(name) {
            if let Some(descriptor) = models.get(real_name) {
                self.update_access_stats(real_name);
                return Some(descriptor.info.clone());
            }
        }
        
        None
    }
    
    /// Get model descriptor
    pub fn get_model_descriptor(&self, name: &str) -> Option<ModelDescriptor> {
        let models = self.models.read();
        
        // Try direct lookup first
        if let Some(descriptor) = models.get(name) {
            self.update_access_stats(name);
            return Some(descriptor.clone());
        }
        
        // Try alias lookup
        let aliases = self.aliases.read();
        if let Some(real_name) = aliases.get(name) {
            if let Some(descriptor) = models.get(real_name) {
                self.update_access_stats(real_name);
                return Some(descriptor.clone());
            }
        }
        
        None
    }
    
    /// List all registered models
    pub fn list_all(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().map(|d| d.info.clone()).collect()
    }
    
    /// List models by category
    pub fn list_by_category(&self, category: ModelCategory) -> Vec<ModelInfo> {
        let by_category = self.by_category.read();
        let models = self.models.read();
        
        if let Some(model_names) = by_category.get(&category) {
            model_names
                .iter()
                .filter_map(|name| models.get(name))
                .map(|d| d.info.clone())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// List models by capability
    pub fn list_by_capability(&self, capability: &str) -> Vec<ModelInfo> {
        let by_capability = self.by_capability.read();
        let models = self.models.read();
        
        if let Some(model_names) = by_capability.get(capability) {
            model_names
                .iter()
                .filter_map(|name| models.get(name))
                .map(|d| d.info.clone())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Search models by criteria
    pub fn search(&self, criteria: &SearchCriteria) -> Vec<ModelInfo> {
        let models = self.models.read();
        let mut results = Vec::new();
        
        for (name, descriptor) in models.iter() {
            if self.matches_criteria(name, descriptor, criteria) {
                results.push(descriptor.info.clone());
            }
        }
        
        // Sort by relevance (access count for now)
        results.sort_by(|a, b| {
            let count_a = models.get(&a.name).map(|d| *d.access_count.read()).unwrap_or(0);
            let count_b = models.get(&b.name).map(|d| *d.access_count.read()).unwrap_or(0);
            count_b.cmp(&count_a)
        });
        
        results
    }
    
    /// Get models by tags
    pub fn get_by_tags(&self, tags: &[String]) -> Vec<ModelInfo> {
        let tag_index = self.tags.read();
        let models = self.models.read();
        let mut model_names = std::collections::HashSet::new();
        
        for tag in tags {
            if let Some(names) = tag_index.get(tag) {
                for name in names {
                    model_names.insert(name.clone());
                }
            }
        }
        
        model_names
            .into_iter()
            .filter_map(|name| models.get(&name))
            .map(|d| d.info.clone())
            .collect()
    }
    
    /// Register a plugin
    pub fn register_plugin(&self, plugin: Plugin) -> RegistryResult<()> {
        let name = plugin.descriptor.name.clone();
        
        // Register plugin models
        for model_info in &plugin.descriptor.models {
            self.register_info(model_info.clone())?;
        }
        
        // Store plugin
        self.plugins.write().insert(name.clone(), plugin);
        
        // Update statistics
        self.update_stats();
        
        log::info!("Registered plugin '{}'", name);
        Ok(())
    }
    
    /// Unregister a plugin
    pub fn unregister_plugin(&self, name: &str) -> RegistryResult<()> {
        let plugin = {
            let mut plugins = self.plugins.write();
            plugins.remove(name)
                .ok_or_else(|| RegistryError::PluginError(format!("Plugin '{}' not found", name)))?
        };
        
        // Unregister plugin models
        for model_info in &plugin.descriptor.models {
            if let Err(e) = self.unregister(&model_info.name) {
                log::warn!("Failed to unregister model '{}' from plugin '{}': {}", 
                          model_info.name, name, e);
            }
        }
        
        // Update statistics
        self.update_stats();
        
        log::info!("Unregistered plugin '{}'", name);
        Ok(())
    }
    
    /// List all plugins
    pub fn list_plugins(&self) -> Vec<Plugin> {
        let plugins = self.plugins.read();
        plugins.values().cloned().collect()
    }
    
    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Plugin> {
        let plugins = self.plugins.read();
        plugins.get(name).cloned()
    }
    
    /// Set registry configuration
    pub fn set_config(&self, config: RegistryConfig) {
        *self.config.write() = Some(config);
        log::debug!("Registry configuration updated");
    }
    
    /// Get registry configuration
    pub fn get_config(&self) -> Option<RegistryConfig> {
        self.config.read().clone()
    }
    
    /// Get registry statistics
    pub fn get_stats(&self) -> RegistryStats {
        self.stats.read().clone()
    }
    
    /// Clear all models
    pub fn clear(&self) {
        self.models.write().clear();
        self.by_category.write().clear();
        self.by_capability.write().clear();
        self.aliases.write().clear();
        self.tags.write().clear();
        self.search_index.write().clear();
        self.update_stats();
        log::info!("Registry cleared");
    }
    
    /// Get registry size (number of models)
    pub fn len(&self) -> usize {
        self.models.read().len()
    }
    
    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.models.read().is_empty()
    }
    
    /// Clear cache (placeholder for future caching implementation)
    pub fn clear_cache(&self) {
        // Placeholder for model instance caching
        log::debug!("Model cache cleared");
    }
    
    /// Cleanup plugins
    pub fn cleanup_plugins(&self) -> RegistryResult<()> {
        let plugin_names: Vec<String> = {
            let plugins = self.plugins.read();
            plugins.keys().cloned().collect()
        };
        
        for plugin_name in plugin_names {
            self.unregister_plugin(&plugin_name)?;
        }
        
        log::info!("All plugins cleaned up");
        Ok(())
    }
    
    /// Update access statistics for a model
    fn update_access_stats(&self, name: &str) {
        if let Some(descriptor) = self.models.read().get(name) {
            *descriptor.last_accessed.write() = std::time::SystemTime::now();
            *descriptor.access_count.write() += 1;
        }
    }
    
    /// Extract capabilities from model capabilities struct
    fn extract_capabilities(&self, capabilities: &ModelCapabilities) -> Vec<String> {
        let mut caps = Vec::new();
        
        if capabilities.online_learning { caps.push("online_learning".to_string()); }
        if capabilities.batch_processing { caps.push("batch_processing".to_string()); }
        if capabilities.streaming { caps.push("streaming".to_string()); }
        if capabilities.multi_threading { caps.push("multi_threading".to_string()); }
        if capabilities.gpu_acceleration { caps.push("gpu_acceleration".to_string()); }
        if capabilities.quantization { caps.push("quantization".to_string()); }
        if capabilities.pruning { caps.push("pruning".to_string()); }
        if capabilities.fine_tuning { caps.push("fine_tuning".to_string()); }
        
        caps
    }
    
    /// Extract tags from model metadata
    fn extract_tags_from_metadata(&self, metadata: &HashMap<String, Value>) -> Vec<String> {
        if let Some(Value::Array(tags)) = metadata.get("tags") {
            tags.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Update search index for a model
    fn update_search_index(&self, name: &str, descriptor: &ModelDescriptor) {
        let mut search_index = self.search_index.write();
        
        // Index by name
        search_index
            .entry(name.to_lowercase())
            .or_insert_with(Vec::new)
            .push(name.to_string());
        
        // Index by category
        let category_key = format!("category:{:?}", descriptor.info.category).to_lowercase();
        search_index
            .entry(category_key)
            .or_insert_with(Vec::new)
            .push(name.to_string());
        
        // Index by tags
        for tag in &descriptor.tags {
            let tag_key = format!("tag:{}", tag).to_lowercase();
            search_index
                .entry(tag_key)
                .or_insert_with(Vec::new)
                .push(name.to_string());
        }
        
        // Index by capabilities
        for capability in self.extract_capabilities(&descriptor.info.capabilities) {
            let cap_key = format!("capability:{}", capability).to_lowercase();
            search_index
                .entry(cap_key)
                .or_insert_with(Vec::new)
                .push(name.to_string());
        }
    }
    
    /// Check if model matches search criteria
    fn matches_criteria(&self, name: &str, descriptor: &ModelDescriptor, criteria: &SearchCriteria) -> bool {
        // Name pattern matching
        if let Some(pattern) = &criteria.name_pattern {
            if !self.matches_pattern(name, pattern) {
                return false;
            }
        }
        
        // Category filter
        if let Some(category) = criteria.category {
            if descriptor.info.category != category {
                return false;
            }
        }
        
        // Required capabilities
        let model_capabilities = self.extract_capabilities(&descriptor.info.capabilities);
        for required_cap in &criteria.required_capabilities {
            if !model_capabilities.contains(required_cap) {
                return false;
            }
        }
        
        // Tag filters
        for required_tag in &criteria.tags {
            if !descriptor.tags.contains(required_tag) {
                return false;
            }
        }
        
        // Parameter type filter
        if let Some(param_type) = &criteria.parameter_type {
            if !descriptor.info.parameter_types.contains(param_type) {
                return false;
            }
        }
        
        // Performance requirements
        if let Some(perf_req) = &criteria.performance_requirements {
            if let Some(performance) = &descriptor.info.performance {
                if !self.meets_performance_requirements(performance, perf_req) {
                    return false;
                }
            }
        }
        
        // Plugin filter
        if let Some(plugin_name) = &criteria.from_plugin {
            if let Some(plugin) = &descriptor.plugin {
                if plugin.name != *plugin_name {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
    
    /// Check if name matches pattern (supports * wildcard)
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
                    // Must start with this part
                    if !name[name_pos..].starts_with(part) {
                        return false;
                    }
                    name_pos += part.len();
                } else if i == parts.len() - 1 {
                    // Must end with this part
                    return name[name_pos..].ends_with(part);
                } else {
                    // Must contain this part
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
    
    /// Check if performance meets requirements
    fn meets_performance_requirements(&self, performance: &ModelPerformance, requirements: &PerformanceRequirements) -> bool {
        if let Some(max_time) = requirements.max_creation_time_ms {
            if performance.creation_time_ms > max_time {
                return false;
            }
        }
        
        if let Some(max_memory) = requirements.max_memory_usage_bytes {
            if performance.memory_usage_bytes > max_memory {
                return false;
            }
        }
        
        if let Some(min_throughput) = requirements.min_throughput_samples_per_sec {
            if performance.throughput_samples_per_sec < min_throughput {
                return false;
            }
        }
        
        if let Some(max_size) = requirements.max_model_size_bytes {
            if performance.model_size_bytes > max_size {
                return false;
            }
        }
        
        true
    }
    
    /// Update registry statistics
    fn update_stats(&self) {
        let mut stats = self.stats.write();
        let models = self.models.read();
        let plugins = self.plugins.read();
        
        stats.total_models = models.len();
        stats.total_plugins = plugins.len();
        stats.active_plugins = plugins.values().filter(|p| p.is_active()).count();
        
        // Update category counts
        stats.by_category.clear();
        for descriptor in models.values() {
            *stats.by_category.entry(descriptor.info.category).or_insert(0) += 1;
        }
        
        // Update parameter type counts
        stats.by_parameter_type.clear();
        for descriptor in models.values() {
            for param_type in &descriptor.info.parameter_types {
                *stats.by_parameter_type.entry(param_type.clone()).or_insert(0) += 1;
            }
        }
        
        // Calculate total accesses and most accessed models
        stats.total_accesses = 0;
        let mut access_counts: Vec<(String, u64)> = Vec::new();
        
        for (name, descriptor) in models.iter() {
            let count = *descriptor.access_count.read();
            stats.total_accesses += count;
            access_counts.push((name.clone(), count));
        }
        
        // Sort by access count and take top 10
        access_counts.sort_by(|a, b| b.1.cmp(&a.1));
        stats.most_accessed = access_counts.into_iter().take(10).collect();
        
        // Calculate average creation time
        let mut total_time = 0.0;
        let mut count = 0;
        for descriptor in models.values() {
            if let Some(performance) = &descriptor.info.performance {
                total_time += performance.creation_time_ms;
                count += 1;
            }
        }
        stats.average_creation_time_ms = if count > 0 { total_time / count as f64 } else { 0.0 };
        
        // Estimate registry size (rough approximation)
        stats.registry_size_bytes = models.len() * 1024; // Rough estimate
        
        stats.last_updated = std::time::SystemTime::now();
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_model_info() -> ModelInfo {
        ModelInfo {
            name: "test_model".to_string(),
            category: ModelCategory::Basic,
            version: "1.0.0".to_string(),
            description: "Test model".to_string(),
            parameter_types: vec!["f32".to_string()],
            input_size: Some(10),
            output_size: Some(1),
            capabilities: ModelCapabilities::default(),
            performance: None,
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_registry_new() {
        let registry = ModelRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }
    
    #[test]
    fn test_register_model_info() {
        let registry = ModelRegistry::new();
        let model_info = create_test_model_info();
        
        assert!(registry.register_info(model_info.clone()).is_ok());
        assert_eq!(registry.len(), 1);
        assert!(registry.has_model("test_model"));
        
        let retrieved = registry.get_model_info("test_model");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test_model");
    }
    
    #[test]
    fn test_unregister_model() {
        let registry = ModelRegistry::new();
        let model_info = create_test_model_info();
        
        registry.register_info(model_info).unwrap();
        assert_eq!(registry.len(), 1);
        
        assert!(registry.unregister("test_model").is_ok());
        assert_eq!(registry.len(), 0);
        assert!(!registry.has_model("test_model"));
    }
    
    #[test]
    fn test_list_by_category() {
        let registry = ModelRegistry::new();
        let mut model_info = create_test_model_info();
        model_info.category = ModelCategory::Basic;
        
        registry.register_info(model_info).unwrap();
        
        let basic_models = registry.list_by_category(ModelCategory::Basic);
        assert_eq!(basic_models.len(), 1);
        
        let advanced_models = registry.list_by_category(ModelCategory::Advanced);
        assert_eq!(advanced_models.len(), 0);
    }
    
    #[test]
    fn test_search_criteria() {
        let criteria = SearchCriteria {
            name_pattern: Some("test*".to_string()),
            category: Some(ModelCategory::Basic),
            ..Default::default()
        };
        
        assert_eq!(criteria.name_pattern, Some("test*".to_string()));
        assert_eq!(criteria.category, Some(ModelCategory::Basic));
        assert!(criteria.required_capabilities.is_empty());
    }
    
    #[test]
    fn test_pattern_matching() {
        let registry = ModelRegistry::new();
        
        assert!(registry.matches_pattern("test_model", "*"));
        assert!(registry.matches_pattern("test_model", "test*"));
        assert!(registry.matches_pattern("test_model", "*model"));
        assert!(registry.matches_pattern("test_model", "test_model"));
        assert!(!registry.matches_pattern("test_model", "other*"));
    }
    
    #[test]
    fn test_extract_capabilities() {
        let registry = ModelRegistry::new();
        let mut capabilities = ModelCapabilities::default();
        capabilities.online_learning = true;
        capabilities.gpu_acceleration = true;
        
        let caps = registry.extract_capabilities(&capabilities);
        assert!(caps.contains(&"online_learning".to_string()));
        assert!(caps.contains(&"gpu_acceleration".to_string()));
        assert!(caps.contains(&"batch_processing".to_string())); // default true
        assert!(!caps.contains(&"streaming".to_string())); // default false
    }
    
    #[test]
    fn test_registry_stats() {
        let registry = ModelRegistry::new();
        let stats = registry.get_stats();
        
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.total_plugins, 0);
        assert_eq!(stats.total_accesses, 0);
        assert!(stats.by_category.is_empty());
    }
}