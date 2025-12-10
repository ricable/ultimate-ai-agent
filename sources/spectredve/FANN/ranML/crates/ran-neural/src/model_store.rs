//! Model storage and management for RAN neural networks

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{NeuralError, NeuralResult, ModelType, RanNeuralNetwork};

/// Model store for managing neural network models
#[derive(Debug)]
pub struct ModelStore {
    /// Base directory for model storage
    pub base_path: PathBuf,
    /// Model registry
    pub registry: HashMap<Uuid, ModelMetadata>,
    /// Model type index
    pub type_index: HashMap<ModelType, Vec<Uuid>>,
}

impl ModelStore {
    /// Create a new model store
    pub fn new<P: AsRef<Path>>(base_path: P) -> NeuralResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .map_err(|e| NeuralError::Io(e))?;
        
        let mut store = Self {
            base_path,
            registry: HashMap::new(),
            type_index: HashMap::new(),
        };
        
        // Load existing models
        store.load_registry()?;
        
        Ok(store)
    }

    /// Save a model to the store
    pub fn save_model(&mut self, model: &RanNeuralNetwork, name: Option<String>) -> NeuralResult<Uuid> {
        let model_id = Uuid::new_v4();
        let model_name = name.unwrap_or_else(|| format!("model_{}", model_id));
        
        // Create model directory
        let model_dir = self.base_path.join(model_id.to_string());
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| NeuralError::Io(e))?;
        
        // Save network file
        let network_path = model_dir.join("network.fann");
        model.save_model(&network_path)?;
        
        // Create metadata
        let metadata = ModelMetadata {
            id: model_id,
            name: model_name,
            model_type: model.model_type,
            version: "1.0.0".to_string(),
            description: format!("RAN {} model", model.model_type.description()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_path: network_path.to_string_lossy().to_string(),
            file_size: std::fs::metadata(&network_path)
                .map(|m| m.len())
                .unwrap_or(0),
            accuracy: model.metadata.accuracy,
            performance_metrics: model.metadata.performance_metrics.clone(),
            tags: vec![format!("{:?}", model.model_type)],
            author: "ranML".to_string(),
            checksum: None, // TODO: Calculate checksum
        };
        
        // Save metadata
        self.save_metadata(&metadata)?;
        
        // Update registry
        self.registry.insert(model_id, metadata);
        self.type_index.entry(model.model_type)
            .or_insert_with(Vec::new)
            .push(model_id);
        
        tracing::info!("Saved model {} to store", model_id);
        Ok(model_id)
    }

    /// Load a model from the store
    pub fn load_model(&self, model_id: Uuid) -> NeuralResult<RanNeuralNetwork> {
        let metadata = self.registry.get(&model_id)
            .ok_or_else(|| NeuralError::ModelNotFound(model_id.to_string()))?;
        
        let mut model = RanNeuralNetwork::new(metadata.model_type)?;
        model.load_model(&metadata.file_path)?;
        model.metadata = metadata.clone();
        
        tracing::info!("Loaded model {} from store", model_id);
        Ok(model)
    }

    /// List all models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.registry.values().collect()
    }

    /// List models by type
    pub fn list_models_by_type(&self, model_type: ModelType) -> Vec<&ModelMetadata> {
        self.type_index.get(&model_type)
            .map(|ids| ids.iter()
                .filter_map(|id| self.registry.get(id))
                .collect())
            .unwrap_or_default()
    }

    /// Delete a model from the store
    pub fn delete_model(&mut self, model_id: Uuid) -> NeuralResult<()> {
        let metadata = self.registry.remove(&model_id)
            .ok_or_else(|| NeuralError::ModelNotFound(model_id.to_string()))?;
        
        // Remove from type index
        if let Some(ids) = self.type_index.get_mut(&metadata.model_type) {
            ids.retain(|&id| id != model_id);
            if ids.is_empty() {
                self.type_index.remove(&metadata.model_type);
            }
        }
        
        // Delete model directory
        let model_dir = self.base_path.join(model_id.to_string());
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)
                .map_err(|e| NeuralError::Io(e))?;
        }
        
        tracing::info!("Deleted model {} from store", model_id);
        Ok(())
    }

    /// Update model metadata
    pub fn update_metadata(&mut self, model_id: Uuid, metadata: ModelMetadata) -> NeuralResult<()> {
        if !self.registry.contains_key(&model_id) {
            return Err(NeuralError::ModelNotFound(model_id.to_string()));
        }
        
        self.save_metadata(&metadata)?;
        self.registry.insert(model_id, metadata);
        
        tracing::info!("Updated metadata for model {}", model_id);
        Ok(())
    }

    /// Search models by name or tags
    pub fn search_models(&self, query: &str) -> Vec<&ModelMetadata> {
        let query_lower = query.to_lowercase();
        
        self.registry.values()
            .filter(|metadata| {
                metadata.name.to_lowercase().contains(&query_lower) ||
                metadata.description.to_lowercase().contains(&query_lower) ||
                metadata.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Get model metadata
    pub fn get_metadata(&self, model_id: Uuid) -> Option<&ModelMetadata> {
        self.registry.get(&model_id)
    }

    /// Check if model exists
    pub fn model_exists(&self, model_id: Uuid) -> bool {
        self.registry.contains_key(&model_id)
    }

    /// Get store statistics
    pub fn get_statistics(&self) -> StoreStatistics {
        let total_models = self.registry.len();
        let total_size: u64 = self.registry.values().map(|m| m.file_size).sum();
        
        let mut models_by_type = HashMap::new();
        for metadata in self.registry.values() {
            *models_by_type.entry(metadata.model_type).or_insert(0) += 1;
        }
        
        StoreStatistics {
            total_models,
            total_size_bytes: total_size,
            models_by_type,
            oldest_model: self.registry.values()
                .map(|m| m.created_at)
                .min(),
            newest_model: self.registry.values()
                .map(|m| m.created_at)
                .max(),
        }
    }

    /// Load registry from disk
    fn load_registry(&mut self) -> NeuralResult<()> {
        if !self.base_path.exists() {
            return Ok(());
        }
        
        for entry in std::fs::read_dir(&self.base_path)
            .map_err(|e| NeuralError::Io(e))? {
            let entry = entry.map_err(|e| NeuralError::Io(e))?;
            let path = entry.path();
            
            if path.is_dir() {
                if let Some(dir_name) = path.file_name() {
                    if let Ok(model_id) = Uuid::parse_str(&dir_name.to_string_lossy()) {
                        let metadata_path = path.join("metadata.json");
                        if metadata_path.exists() {
                            match self.load_metadata(&metadata_path) {
                                Ok(metadata) => {
                                    self.type_index.entry(metadata.model_type)
                                        .or_insert_with(Vec::new)
                                        .push(model_id);
                                    self.registry.insert(model_id, metadata);
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to load metadata for {}: {}", model_id, e);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        tracing::info!("Loaded {} models from registry", self.registry.len());
        Ok(())
    }

    /// Save metadata to disk
    fn save_metadata(&self, metadata: &ModelMetadata) -> NeuralResult<()> {
        let model_dir = self.base_path.join(metadata.id.to_string());
        let metadata_path = model_dir.join("metadata.json");
        
        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        
        std::fs::write(&metadata_path, json)
            .map_err(|e| NeuralError::Io(e))?;
        
        Ok(())
    }

    /// Load metadata from disk
    fn load_metadata(&self, path: &Path) -> NeuralResult<ModelMetadata> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| NeuralError::Io(e))?;
        
        let metadata: ModelMetadata = serde_json::from_str(&json)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        
        Ok(metadata)
    }
}

/// Model metadata for the store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub id: Uuid,
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// File path
    pub file_path: String,
    /// File size in bytes
    pub file_size: u64,
    /// Model accuracy (if available)
    pub accuracy: Option<f64>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Author/creator
    pub author: String,
    /// File checksum for integrity
    pub checksum: Option<String>,
}

impl ModelMetadata {
    /// Create new metadata for a model type
    pub fn new(model_type: ModelType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: format!("{:?}_model", model_type),
            model_type,
            version: "1.0.0".to_string(),
            description: model_type.description().to_string(),
            created_at: now,
            updated_at: now,
            file_path: String::new(),
            file_size: 0,
            accuracy: None,
            performance_metrics: HashMap::new(),
            tags: vec![format!("{:?}", model_type)],
            author: "ranML".to_string(),
            checksum: None,
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Set performance metric
    pub fn set_performance_metric(&mut self, name: String, value: f64) {
        self.performance_metrics.insert(name, value);
        self.updated_at = Utc::now();
    }

    /// Get performance metric
    pub fn get_performance_metric(&self, name: &str) -> Option<f64> {
        self.performance_metrics.get(name).copied()
    }

    /// Update timestamp
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

/// Store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStatistics {
    /// Total number of models
    pub total_models: usize,
    /// Total storage size in bytes
    pub total_size_bytes: u64,
    /// Models by type
    pub models_by_type: HashMap<ModelType, usize>,
    /// Oldest model creation time
    pub oldest_model: Option<DateTime<Utc>>,
    /// Newest model creation time
    pub newest_model: Option<DateTime<Utc>>,
}

impl StoreStatistics {
    /// Get average model size in bytes
    pub fn average_model_size(&self) -> f64 {
        if self.total_models > 0 {
            self.total_size_bytes as f64 / self.total_models as f64
        } else {
            0.0
        }
    }

    /// Get storage size in human-readable format
    pub fn human_readable_size(&self) -> String {
        let size = self.total_size_bytes as f64;
        
        if size < 1024.0 {
            format!("{:.1} B", size)
        } else if size < 1024.0 * 1024.0 {
            format!("{:.1} KB", size / 1024.0)
        } else if size < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.1} MB", size / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", size / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_store_creation() {
        let temp_dir = TempDir::new().unwrap();
        let store = ModelStore::new(temp_dir.path()).unwrap();
        
        assert_eq!(store.registry.len(), 0);
        assert!(store.base_path.exists());
    }

    #[test]
    fn test_model_metadata() {
        let mut metadata = ModelMetadata::new(ModelType::ThroughputPredictor);
        assert_eq!(metadata.model_type, ModelType::ThroughputPredictor);
        assert!(!metadata.name.is_empty());
        
        metadata.add_tag("test".to_string());
        assert!(metadata.tags.contains(&"test".to_string()));
        
        metadata.set_performance_metric("accuracy".to_string(), 0.95);
        assert_eq!(metadata.get_performance_metric("accuracy"), Some(0.95));
    }

    #[test]
    fn test_store_statistics() {
        let stats = StoreStatistics {
            total_models: 5,
            total_size_bytes: 1024 * 1024, // 1 MB
            models_by_type: HashMap::new(),
            oldest_model: None,
            newest_model: None,
        };
        
        assert_eq!(stats.average_model_size(), 1024.0 * 1024.0 / 5.0);
        assert_eq!(stats.human_readable_size(), "1.0 MB");
    }
}